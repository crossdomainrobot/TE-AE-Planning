# @File : initialexample .py
# author: Hou Chenfei
# Time：2026-1-25


import os
import sys
from openai import OpenAI


# -----------------------------
# Configuration
# -----------------------------
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_A = "qwen-plus"   # 示例：你可换成你实际要用的模型名
MODEL_B = "qwen-plus"   # 示例：可不同模型

API_KEY_A = ""   # ← 你的真实 Key
API_KEY_B = ""   # ← 可同一个或不同

# 如果你只想用一个 Key，也可以这样：
# API_KEY_B = API_KEY_A


if not API_KEY_A:
    raise RuntimeError("Missing env var: DASHSCOPE_API_KEY_A")
if not API_KEY_B:

    API_KEY_B = API_KEY_A

client_a = OpenAI(api_key=API_KEY_A, base_url=BASE_URL)
client_b = OpenAI(api_key=API_KEY_B, base_url=BASE_URL)


SYSTEM_PROMPT_A = (
    "You are Model A. Your job is to read the user's request and produce a structured draft.\n"
    "Output should be detailed and include any necessary intermediate reasoning as plain text.\n"
    "This is a placeholder prompt—replace it with your own."
)

SYSTEM_PROMPT_B = (
    "You are Model B. Your job is to take Model A's output and produce the final answer.\n"
    "Be concise, well-structured, and output only the final result.\n"
    "This is a placeholder prompt—replace it with your own."
)

USER_INPUT = (
    "示例需求：请给出一个项目计划的大纲，包括里程碑、风险、资源。\n"
    "（这是示例输入，你可以替换成真实输入）"
)

def iter_stream_text(stream):
    """
    Robustly extract text chunks from OpenAI ChatCompletions stream responses.
    Works with the common pattern: event.choices[0].delta.content
    """
    for event in stream:
        try:
            if hasattr(event, "choices") and event.choices:
                delta = getattr(event.choices[0], "delta", None)
                if delta is not None:
                    chunk = getattr(delta, "content", None)
                    if chunk:
                        yield chunk
        except Exception:
            # If a provider returns a slightly different schema, don't crash the whole stream.
            continue


def stream_chat_completion(client: OpenAI, model: str, messages: list, print_prefix: str = "") -> str:
    """
    Stream a chat completion to stdout, return the full accumulated text.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    full_text_parts = []
    for chunk in iter_stream_text(stream):
        full_text_parts.append(chunk)
        if print_prefix:
            sys.stdout.write(print_prefix)
            print_prefix = ""  # only once at the beginning
        sys.stdout.write(chunk)
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full_text_parts)


# -----------------------------
# Main: A -> B
# -----------------------------
def main():
    # ---- Step 1: Call Model A (streaming) ----
    messages_a = [
        {"role": "system", "content": SYSTEM_PROMPT_A},
        {"role": "user", "content": USER_INPUT},
    ]

    print("========== Model A (stream) ==========")
    a_output = stream_chat_completion(
        client=client_a,
        model=MODEL_A,
        messages=messages_a,
        print_prefix="A> ",
    )

    # ---- Step 2: Feed A output into Model B, call B (streaming) ----
    # You can choose how to pass A output to B:
    # - as a single user message
    # - or as a tool-like wrapper text
    b_user_content = (
        "Below is the output from Model A. Use it as input and produce the final result.\n\n"
        "=== Model A Output Start ===\n"
        f"{a_output}\n"
        "=== Model A Output End ==="
    )

    messages_b = [
        {"role": "system", "content": SYSTEM_PROMPT_B},
        {"role": "user", "content": b_user_content},
    ]

    print("\n========== Model B (stream) ==========")
    _ = stream_chat_completion(
        client=client_b,
        model=MODEL_B,
        messages=messages_b,
        print_prefix="B> ",
    )


if __name__ == "__main__":
    main()
