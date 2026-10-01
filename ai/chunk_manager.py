"""
ai/chunk_manager.py

Manages change-aware chunking for large PR contexts.
Extracts diff hunks, groups nearby changed regions with surrounding context,
and orchestrates parallel chunk review calls with controlled concurrency and memory aggregation.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ai.context_config import (
    SAFE_INPUT_TOKEN_LIMIT,
    MAX_CONCURRENCY,
    CONTEXT_WINDOW_LINES,
    CHUNK_OVERLAP_LINES,
    SAFE_PROMPT_CHAR_LIMIT,
)
from ai.token_counter import count_tokens
from ai.prompt_builder import build_review_prompt, build_chunk_prompt
from ai.result_aggregator import aggregate_reviews
from ai.reviewer import review_code


@dataclass
class DiffHunk:
    """Represents a single diff hunk in a file."""
    new_start: int
    new_end: int
    hunk_text: str
    modified_lines: List[int] = field(default_factory=list)


@dataclass
class ReviewChunk:
    """Represents an individual chunk sent to the AI reviewer."""
    chunk_id: int
    total_chunks: int
    file_path: str
    changed_files: List[str]
    code_snippet: str
    diff_snippet: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


def _normalize_path(path: str) -> str:
    """Normalize file path for consistent matching."""
    return (path or "").strip().replace("\\", "/").lstrip("./")


def parse_diff_by_file(diff_text: str) -> Dict[str, Dict]:
    """
    Parse a raw git diff into a dictionary keyed by normalized file path:
    {
        "file/path.py": {
            "full_diff": str,
            "hunks": List[DiffHunk]
        }
    }
    """
    if not diff_text or not diff_text.strip():
        return {}

    blocks = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
    file_diffs: Dict[str, Dict] = {}

    hunk_header_pattern = re.compile(
        r"^@@ -\d+(?:,\d+)? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
    )

    for block in blocks:
        if not block.strip():
            continue

        first_line = block.splitlines()[0]
        if not first_line.startswith("diff --git"):
            continue

        try:
            rel_path = first_line.split(" b/")[1].strip()
        except Exception:
            continue

        norm_path = _normalize_path(rel_path)
        hunks: List[DiffHunk] = []

        # Split block into hunks
        hunk_chunks = re.split(r"(?=^@@ )", block, flags=re.MULTILINE)
        for h in hunk_chunks:
            if not h.startswith("@@"):
                continue

            lines = h.splitlines()
            hunk_match = hunk_header_pattern.match(lines[0])
            if not hunk_match:
                continue

            new_start = int(hunk_match.group("new_start"))
            new_count_str = hunk_match.group("new_count")
            new_count = int(new_count_str) if new_count_str is not None else 1

            new_end = new_start if new_count == 0 else (new_start + new_count - 1)

            modified_lines: List[int] = []
            curr_line = new_start
            for line in lines[1:]:
                if line.startswith("+"):
                    modified_lines.append(curr_line)
                    curr_line += 1
                elif line.startswith(" "):
                    curr_line += 1
                elif line.startswith("-"):
                    pass

            hunks.append(
                DiffHunk(
                    new_start=new_start,
                    new_end=new_end,
                    hunk_text=h,
                    modified_lines=modified_lines,
                )
            )

        file_diffs[norm_path] = {
            "full_diff": block.strip(),
            "hunks": hunks,
        }

    return file_diffs


def parse_full_code_files(full_code: str) -> Dict[str, Dict]:
    """
    Parse the line-numbered full_code string produced by extract_full_code:
    FILE: <rel_path>
    ```<lang>
    L1: ...
    L2: ...
    ```
    Returns a dict mapping normalized path to {lang, lines_dict, raw_lines, total_lines}.
    """
    if not full_code or not full_code.strip():
        return {}

    blocks = re.split(r"(?=^FILE:\s*)", full_code, flags=re.MULTILINE)
    parsed: Dict[str, Dict] = {}

    header_pattern = re.compile(r"^FILE:\s*([^\n\r]+)")
    code_block_start = re.compile(r"^```(\w*)")

    for block in blocks:
        block_clean = block.strip()
        if not block_clean:
            continue

        match = header_pattern.match(block_clean)
        if not match:
            continue

        rel_path = _normalize_path(match.group(1))
        lines = block_clean.splitlines()

        lang = "text"
        code_lines: List[Tuple[int, str]] = []

        inside_code = False
        for line in lines[1:]:
            if not inside_code:
                code_match = code_block_start.match(line)
                if code_match:
                    inside_code = True
                    lang = code_match.group(1) or "text"
                continue

            if line.strip() == "```":
                inside_code = False
                continue

            l_match = re.match(r"^L(\d+):\s?(.*)$", line)
            if l_match:
                line_no = int(l_match.group(1))
                code_lines.append((line_no, line))
            else:
                code_lines.append((len(code_lines) + 1, line))

        total_lines = code_lines[-1][0] if code_lines else len(code_lines)
        lines_dict = {ln: l_text for ln, l_text in code_lines}

        parsed[rel_path] = {
            "lang": lang,
            "lines_dict": lines_dict,
            "code_lines": code_lines,
            "total_lines": total_lines,
        }

    return parsed


def _format_code_block(file_path: str, lang: str, selected_lines: List[str]) -> str:
    """Format selected line-numbered lines as a FILE code block."""
    inner = "\n".join(selected_lines)
    return f"FILE: {file_path}\n```{lang}\n{inner}\n```"


def _subdivide_large_region(
    file_path: str,
    lang: str,
    region_lines: List[str],
    region_diff: str,
    is_prompt_safe: Callable[[str], bool],
    overlap_lines: int,
) -> List[Tuple[str, str, int, int]]:
    """
    Subdivide a region that alone exceeds the safe budget into sub-chunks with overlap.
    Returns list of (code_snippet, diff_snippet, start_line, end_line).
    """
    sub_chunks = []
    total = len(region_lines)
    if total == 0:
        return []

    lines_per_chunk = max(100, total // 2)

    idx = 0
    while idx < total:
        end_idx = min(total, idx + lines_per_chunk)
        slice_lines = region_lines[idx:end_idx]
        code_snippet = _format_code_block(file_path, lang, slice_lines)

        # Enforce that build_review_prompt strictly fits the safe budget
        while not is_prompt_safe(build_review_prompt(code_snippet, [file_path], region_diff)) and len(slice_lines) > 1:
            new_len = max(1, int(len(slice_lines) * 0.75))
            slice_lines = slice_lines[:new_len]
            end_idx = idx + len(slice_lines)
            code_snippet = _format_code_block(file_path, lang, slice_lines)

        start_line_no = None
        end_line_no = None
        if slice_lines:
            m_start = re.match(r"^L(\d+):", slice_lines[0])
            m_end = re.match(r"^L(\d+):", slice_lines[-1])
            if m_start:
                start_line_no = int(m_start.group(1))
            if m_end:
                end_line_no = int(m_end.group(1))

        sub_chunks.append((code_snippet, region_diff, start_line_no, end_line_no))

        if end_idx >= total:
            break
        actual_overlap = min(overlap_lines, max(1, len(slice_lines) // 4))
        idx = max(idx + 1, end_idx - actual_overlap)

    return sub_chunks


def create_review_chunks(
    full_code: str,
    changed_files: List[str],
    diff: str,
    safe_token_limit: int = SAFE_INPUT_TOKEN_LIMIT,
    context_lines: int = CONTEXT_WINDOW_LINES,
    safe_char_limit: Optional[int] = None,
) -> List[ReviewChunk]:
    """
    Break down changed files and diff into manageable ReviewChunk objects.
    - Groups nearby hunks with context_lines of surrounding code.
    - Keeps each chunk prompt strictly within the safe token budget (default 120,000 tokens).
    - Preserves file and line context for accurate review.
    """
    parsed_files = parse_full_code_files(full_code)
    parsed_diff = parse_diff_by_file(diff)

    def is_prompt_safe(prompt_text: str) -> bool:
        tokens_ok = count_tokens(prompt_text) <= safe_token_limit
        if safe_char_limit is not None:
            return tokens_ok and (len(prompt_text) <= safe_char_limit)
        return tokens_ok

    raw_chunk_specs: List[Dict] = []

    # Process each changed file
    for rel_path in changed_files:
        norm_path = _normalize_path(rel_path)

        file_info = parsed_files.get(norm_path)
        diff_info = parsed_diff.get(norm_path)

        # Fallback path lookup if exact normalized path differed slightly
        if not file_info:
            for p, info in parsed_files.items():
                if p.endswith(norm_path) or norm_path.endswith(p):
                    file_info = info
                    norm_path = p
                    break

        if not diff_info:
            for p, dinfo in parsed_diff.items():
                if p.endswith(norm_path) or norm_path.endswith(p):
                    diff_info = dinfo
                    break

        if not file_info:
            # File content not found in full_code; use diff if present
            if diff_info:
                raw_chunk_specs.append({
                    "file_path": rel_path,
                    "code_snippet": f"FILE: {rel_path}\n(Full content unavailable)",
                    "diff_snippet": diff_info["full_diff"],
                    "start_line": None,
                    "end_line": None,
                })
            continue

        lang = file_info["lang"]
        lines_dict = file_info["lines_dict"]
        code_lines = file_info["code_lines"]
        total_lines = file_info["total_lines"]
        file_diff = diff_info["full_diff"] if diff_info else ""
        hunks = diff_info["hunks"] if diff_info else []

        if hunks:
            # Group nearby hunks by expanding with context_lines
            hunk_regions: List[Tuple[int, int, List[DiffHunk]]] = []
            for h in hunks:
                h_start = max(1, h.new_start - context_lines)
                h_end = min(total_lines, h.new_end + context_lines)

                if not hunk_regions:
                    hunk_regions.append((h_start, h_end, [h]))
                else:
                    last_start, last_end, last_hunks = hunk_regions[-1]
                    # If overlapping or close (within context_lines), merge regions
                    if h_start <= last_end + context_lines:
                        merged_end = max(last_end, h_end)
                        hunk_regions[-1] = (last_start, merged_end, last_hunks + [h])
                    else:
                        hunk_regions.append((h_start, h_end, [h]))

            # Convert merged regions to chunk candidates
            for r_start, r_end, r_hunks in hunk_regions:
                region_selected_lines = [
                    lines_dict[ln]
                    for ln in range(r_start, r_end + 1)
                    if ln in lines_dict
                ]
                region_diff_text = "\n\n".join(h.hunk_text for h in r_hunks)
                code_snippet = _format_code_block(rel_path, lang, region_selected_lines)

                candidate_prompt = build_review_prompt(
                    code=code_snippet,
                    changed_files=[rel_path],
                    diff=region_diff_text,
                )

                if is_prompt_safe(candidate_prompt):
                    raw_chunk_specs.append({
                        "file_path": rel_path,
                        "code_snippet": code_snippet,
                        "diff_snippet": region_diff_text,
                        "start_line": r_start,
                        "end_line": r_end,
                    })
                else:
                    # Region itself is oversized (e.g. massive continuous edit)
                    subdivided = _subdivide_large_region(
                        file_path=rel_path,
                        lang=lang,
                        region_lines=region_selected_lines,
                        region_diff=region_diff_text,
                        is_prompt_safe=is_prompt_safe,
                        overlap_lines=CHUNK_OVERLAP_LINES,
                    )
                    for sub_code, sub_diff, s_line, e_line in subdivided:
                        raw_chunk_specs.append({
                            "file_path": rel_path,
                            "code_snippet": sub_code,
                            "diff_snippet": sub_diff,
                            "start_line": s_line,
                            "end_line": e_line,
                        })

        else:
            # File has no hunks (or pure file addition/deletion without hunk headers)
            file_code_lines = [item[1] for item in code_lines]
            code_snippet = _format_code_block(rel_path, lang, file_code_lines)
            candidate_prompt = build_review_prompt(
                code=code_snippet,
                changed_files=[rel_path],
                diff=file_diff,
            )

            if is_prompt_safe(candidate_prompt):
                raw_chunk_specs.append({
                    "file_path": rel_path,
                    "code_snippet": code_snippet,
                    "diff_snippet": file_diff,
                    "start_line": 1 if total_lines > 0 else None,
                    "end_line": total_lines if total_lines > 0 else None,
                })
            else:
                subdivided = _subdivide_large_region(
                    file_path=rel_path,
                    lang=lang,
                    region_lines=file_code_lines,
                    region_diff=file_diff,
                    is_prompt_safe=is_prompt_safe,
                    overlap_lines=CHUNK_OVERLAP_LINES,
                )
                for sub_code, sub_diff, s_line, e_line in subdivided:
                    raw_chunk_specs.append({
                        "file_path": rel_path,
                        "code_snippet": sub_code,
                        "diff_snippet": sub_diff,
                        "start_line": s_line,
                        "end_line": e_line,
                    })

    total_chunks = len(raw_chunk_specs)
    chunks: List[ReviewChunk] = []

    for idx, spec in enumerate(raw_chunk_specs, start=1):
        chunks.append(
            ReviewChunk(
                chunk_id=idx,
                total_chunks=total_chunks,
                file_path=spec["file_path"],
                changed_files=[spec["file_path"]],
                code_snippet=spec["code_snippet"],
                diff_snippet=spec["diff_snippet"],
                start_line=spec.get("start_line"),
                end_line=spec.get("end_line"),
            )
        )

    return chunks


def _review_single_chunk(chunk: ReviewChunk) -> str:
    """
    Process an individual chunk request using review_code with live streaming.
    Thread-safe and isolated per worker.
    """
    chunk_prompt = build_chunk_prompt(chunk)
    token_count = count_tokens(chunk_prompt)
    print(
        f"\n[CHUNK {chunk.chunk_id}/{chunk.total_chunks}] Starting AI review "
        f"for {chunk.file_path} (Lines {chunk.start_line or 'all'}..{chunk.end_line or 'all'}, "
        f"{token_count} input tokens, {len(chunk_prompt)} chars)..."
    )
    return review_code(chunk_prompt)


def execute_chunked_review(
    full_code: str,
    changed_files: List[str],
    diff: str,
    safe_token_limit: int = SAFE_INPUT_TOKEN_LIMIT,
    context_lines: int = CONTEXT_WINDOW_LINES,
    max_concurrency: int = MAX_CONCURRENCY,
    safe_char_limit: Optional[int] = None,
) -> str:
    """
    Orchestrate parallel chunked AI review:
    1. Divide oversized context into change-aware review chunks.
    2. Process chunks concurrently with controlled concurrency (max 3 at a time).
    3. Stream each chunk's tokens and collect complete response.
    4. Store chunk responses in an in-memory list, preserving deterministic chunk order.
    5. Aggregate and deduplicate findings into a single final review.
    """
    chunks = create_review_chunks(
        full_code=full_code,
        changed_files=changed_files,
        diff=diff,
        safe_token_limit=safe_token_limit,
        context_lines=context_lines,
        safe_char_limit=safe_char_limit,
    )

    if not chunks:
        print("\n[CHUNK MANAGER] No review chunks could be generated.")
        return "No review generated."

    total_chunks = len(chunks)
    print(
        f"\n[CHUNK MANAGER] Large context detected. Created {total_chunks} "
        f"chunk(s) for AI review (max concurrency: {max_concurrency}).\n"
    )

    # In-memory storage for completed chunk responses, indexed to preserve deterministic order
    review_results: List[str] = [""] * total_chunks

    # Process in batches of up to max_concurrency (max 3 concurrent requests)
    for batch_start in range(0, total_chunks, max_concurrency):
        batch_chunks = chunks[batch_start : batch_start + max_concurrency]
        chunk_indices = [batch_start + i for i in range(len(batch_chunks))]
        chunk_ids = [c.chunk_id for c in batch_chunks]

        print(
            f"\n[CHUNK MANAGER] Running batch: Chunks {chunk_ids} concurrently "
            f"(concurrency: {len(batch_chunks)}/{max_concurrency})..."
        )

        with ThreadPoolExecutor(max_workers=len(batch_chunks)) as executor:
            future_to_idx = {
                executor.submit(_review_single_chunk, chunk): chunk_idx
                for chunk, chunk_idx in zip(batch_chunks, chunk_indices)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk = chunks[idx]
                try:
                    result = future.result()
                    review_results[idx] = result if result else ""
                except Exception as chunk_err:
                    print(f"\n[CHUNK ERROR] Chunk {chunk.chunk_id} failed with error: {chunk_err}")
                    # Continue processing remaining chunks; do not fabricate results
                    review_results[idx] = ""

    # Aggregate all chunk responses in memory and deduplicate overlapping issues
    aggregated_review = aggregate_reviews(review_results)

    print("\n[CHUNK MANAGER] Completed aggregation of all chunk reviews.\n")
    return aggregated_review
