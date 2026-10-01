import sys
import os
import threading
from groq import Groq
from openai import OpenAI

_stdout_lock = threading.Lock()


def review_code(prompt: str) -> str:

    print("\n===== PROMPT LENGTH =====")
    print(len(prompt))
    print("=========================\n")

    # NVIDIA OpenAI-compatible client (active provider)
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"]
    )

    # Groq client kept for reference (inactive — switch back by swapping the client above)
    # groq_client = Groq(
    #     api_key=os.environ.get("GROQ_API_KEY")
    # )

    print("\n===== AI REVIEW (STREAMING LIVE) =====")

    review_buffer = []

    try:
        # Replace characters that cannot be encoded as UTF-8 before sending the prompt to NVIDIA.
        safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")

        completion = client.chat.completions.create(
            model="google/gemma-4-31b-it",
            messages=[
                {
                    "role": "user",
                    "content": safe_prompt
                }
            ],
            temperature=1,
            max_tokens=4096,
            top_p=1,
            stream=True,
        )

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if content:
                token = content

                # Append to our local aggregator buffer
                review_buffer.append(token)

                # Instantly write the token out to the console terminal
                with _stdout_lock:
                    sys.stdout.write(token)
                    sys.stdout.flush()

    except Exception as e:
        print(f"\nNVIDIA inference engine failure: {e}")
        return "Error occurred during AI code review."

    print("\n======================================")

    # Reassemble individual tokens into one complete string response body
    full_review = "".join(review_buffer).strip()
    return full_review if full_review else "No review generated."
