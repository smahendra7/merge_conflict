"""
ai/result_aggregator.py

Aggregates multiple chunk review responses and deduplicates findings based
on normalized (File, Line) pairs while preserving existing formatting.
"""

import re
from typing import List, Tuple


def _bold_labels(text: str) -> str:
    """
    Ensure target labels are bolded outside code blocks:
    - **Issue:**
    - **File:**
    - **Line:**
    - **Code:**
    - **Reason:**
    - **Suggestion:**
    """
    if not text:
        return ""
    parts = re.split(r"(```[\s\S]*?```)", text)
    pattern = re.compile(
        r"(?m)^(\s*(?:💡\s*|\*\s*)?)(?!\*\*)(Issue|File|Line|Comment|Code|Reason|Suggestion):"
    )
    for i in range(0, len(parts), 2):
        parts[i] = pattern.sub(r"\1**\2:**", parts[i])
    return "".join(parts)


def normalize_file_path(file_str: str) -> str:
    """
    Normalize a file path extracted from **File:** for deduplication.
    Strips backticks, quotes, whitespace, leading ./ and trailing parentheticals.
    """
    cleaned = file_str.strip("`'\" \t").replace("\\", "/")
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned).strip()
    cleaned = cleaned.lstrip("./")
    return cleaned.lower()


def normalize_line_spec(line_str: str) -> str:
    """
    Normalize a line specification extracted from **Line:** for deduplication.
    e.g. 'L50' -> '50', 'L50-L55' -> '50-55', '50 to 55' -> '50-55'.
    """
    cleaned = line_str.strip("`'\" \t")
    nums = re.findall(r"\d+", cleaned)
    if len(nums) >= 2:
        return f"{nums[0]}-{nums[1]}"
    if len(nums) == 1:
        return nums[0]
    return cleaned.lower()


def extract_dedup_key(issue_block: str) -> Tuple[str, str]:
    """
    Extract a (normalized_file, normalized_line) key from an **Issue:** block.
    If File or Line is not found, falls back to block content for uniqueness.
    """
    file_match = re.search(r"^\*\*File:\*\*\s*(.+)$", issue_block, re.MULTILINE)
    line_match = re.search(r"^\*\*Line:\*\*\s*(.+)$", issue_block, re.MULTILINE)

    if file_match and line_match:
        file_key = normalize_file_path(file_match.group(1))
        line_key = normalize_line_spec(line_match.group(1))
        return (file_key, line_key)

    if file_match:
        file_key = normalize_file_path(file_match.group(1))
        issue_title = issue_block.splitlines()[0].strip()
        return (file_key, issue_title.lower())

    # Fallback to the first line or entire text
    first_line = issue_block.splitlines()[0].strip()
    return ("unknown_file", first_line.lower())


def aggregate_reviews(review_results: List[str]) -> str:
    """
    Combine a list of raw chunk review responses into a single coherent review:
    1. Extract all **Issue:** blocks across all chunk responses.
    2. Deduplicate blocks sharing the same (normalized_file, normalized_line).
    3. Preserve order of discovery.
    4. Separate unique issue blocks with '---'.
    5. Handle empty responses, all-error states, and clean non-issue text gracefully.
    """
    if not review_results:
        return "No review generated."

    # Filter out empty or whitespace-only results
    valid_results = [r.strip() for r in review_results if r and r.strip()]
    if not valid_results:
        return "No review generated."

    # Check if all chunks resulted in errors
    all_errors = all(
        "Error occurred during AI code review" in r
        or "inference engine failure" in r
        for r in valid_results
    )
    if all_errors:
        return "Error occurred during AI code review."

    unique_issue_blocks: List[str] = []
    seen_keys = set()
    non_issue_texts: List[str] = []

    issue_split_pattern = re.compile(r"(?m)(?=^\*\*Issue:\*\*)")

    for raw_result in valid_results:
        # Ignore individual chunk error messages if other chunks might have succeeded
        if (
            "Error occurred during AI code review" in raw_result
            or "inference engine failure" in raw_result
        ):
            continue

        bolded = _bold_labels(raw_result)
        raw_blocks = [b.strip() for b in issue_split_pattern.split(bolded) if b.strip()]

        found_issues_in_chunk = False
        for block in raw_blocks:
            if not block.startswith("**Issue:**"):
                # Non-issue introductory or concluding commentary
                if len(block) > 5 and not block.startswith("---"):
                    non_issue_texts.append(block)
                continue

            found_issues_in_chunk = True
            # Clean leading and trailing markdown horizontal rules
            cleaned = re.sub(r"^\s*---\s*", "", block)
            cleaned = re.sub(r"\s*---\s*$", "", cleaned).strip()
            if not cleaned:
                continue

            dedup_key = extract_dedup_key(cleaned)
            if dedup_key in seen_keys:
                # Duplicate finding from overlapping chunk context - skip
                continue

            seen_keys.add(dedup_key)
            unique_issue_blocks.append(cleaned)

        if not found_issues_in_chunk and raw_result:
            non_issue_texts.append(raw_result)

    if unique_issue_blocks:
        return "\n\n---\n\n".join(unique_issue_blocks)

    # If no structured issue blocks were found across all chunks
    clean_non_issues = [
        t for t in non_issue_texts
        if not t.startswith("---")
        and "Error occurred" not in t
        and "No review generated" not in t
    ]

    if clean_non_issues:
        # If there is a meaningful summary message like "No issues found"
        for text in clean_non_issues:
            if any(term in text.lower() for term in ["no issue", "no bug", "looks good", "lgtm"]):
                return text
        return clean_non_issues[0]

    return "No issues found."
