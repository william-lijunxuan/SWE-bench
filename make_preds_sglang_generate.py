import json
import requests
from datasets import load_dataset

SGLANG_URL = "http://127.0.0.1:30000/generate"
OUT = "qwen3_sglang_preds_1.jsonl"

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
CACHE_DIR = "/home/seagl/dataset"
SPLIT = "test[:1]"

MAX_NEW_TOKENS = 8192


def extract_patch(text: str) -> str:
    text = text.strip()

    text = text.replace("```diff", "")
    text = text.replace("```patch", "")
    text = text.replace("```python", "")
    text = text.replace("```", "")

    start = text.find("diff --git")
    if start == -1:
        start = text.find("--- a/")

    if start == -1:
        return ""

    patch = text[start:].strip()

    cleaned_lines = []
    for line in patch.splitlines():
        if line.startswith("index 1234567..abcdefg"):
            continue
        cleaned_lines.append(line)

    patch = "\n".join(cleaned_lines).strip()

    if patch.startswith("--- a/"):
        first_file = patch.splitlines()[0].replace("--- a/", "").strip()
        patch = f"diff --git a/{first_file} b/{first_file}\n" + patch

    return patch


def looks_truncated(patch: str) -> bool:
    if not patch:
        return True

    last = patch.splitlines()[-1].rstrip()

    bad_endings = (
        ":",
        "\\",
        "(",
        "[",
        "{",
        ",",
    )

    if last.startswith("@@"):
        return True

    if last.endswith(bad_endings):
        return True

    return False


def call_sglang(prompt: str) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.01,
            "top_p": 1.0,
        },
    }

    r = requests.post(SGLANG_URL, json=payload, timeout=1200)

    if r.status_code != 200:
        print("SGLang request failed")
        print("status:", r.status_code)
        print("response:", r.text)
        raise SystemExit(1)

    data = r.json()
    return data.get("text", "")


def build_prompt(ex) -> str:
    return f"""You are solving a SWE-bench task.

Repository: {ex["repo"]}
Base commit: {ex["base_commit"]}
Instance ID: {ex["instance_id"]}

Issue:
{ex["problem_statement"]}

Return ONLY a valid git unified diff patch.

Rules:
- The first line must start with: diff --git
- Do not explain.
- Do not output analysis.
- Do not use markdown.
- Do not use code fences.
- Do not include fake examples.
- Do not invent unrelated helper functions.
- Keep the patch minimal.
- The patch must be complete and valid for git apply.
"""


def main():
    ds = load_dataset(
        DATASET_NAME,
        split=SPLIT,
        cache_dir=CACHE_DIR,
    )

    with open(OUT, "w", encoding="utf-8") as f:
        for ex in ds:
            prompt = build_prompt(ex)
            raw_text = call_sglang(prompt)
            patch = extract_patch(raw_text)

            print("instance:", ex["instance_id"])
            print("raw length:", len(raw_text))
            print("patch length:", len(patch))
            print("patch lines:", len(patch.splitlines()))

            if not patch:
                print("WARNING: no valid diff found")
                print(raw_text[:2000])

            if looks_truncated(patch):
                print("WARNING: patch may be truncated")
                print("patch tail:")
                print("\n".join(patch.splitlines()[-30:]))

            row = {
                "instance_id": ex["instance_id"],
                "model_name_or_path": "Qwen3-Coder-Next-FP8-SGLang",
                "model_patch": patch,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print("patch head:")
            print(patch[:1000])
            print("=" * 80)


if __name__ == "__main__":
    main()
