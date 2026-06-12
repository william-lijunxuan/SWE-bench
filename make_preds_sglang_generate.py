import json
import requests
from datasets import load_dataset

SGLANG_URL = "http://127.0.0.1:30000/generate"
OUT = "qwen3_sglang_preds_1.jsonl"

ds = load_dataset(
    "princeton-nlp/SWE-bench_Lite",
    split="test[:1]",
    cache_dir="/home/seagl/dataset",
)


def extract_patch(text: str) -> str:
    text = text.strip()


    text = text.replace("```diff", "")
    text = text.replace("```patch", "")
    text = text.replace("```python", "")
    text = text.replace("```", "")


    idx = text.find("diff --git")
    if idx == -1:
        return ""

    text = text[idx:].strip()

    return text


with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        prompt = f"""You are solving a SWE-bench task.

Repository: {ex["repo"]}
Base commit: {ex["base_commit"]}
Instance ID: {ex["instance_id"]}

Issue:
{ex["problem_statement"]}

Output ONLY a valid git unified diff patch.
The first line of your response must be exactly a git diff header starting with: diff --git
Do not explain.
Do not output analysis.
Do not use markdown.
Do not use code fences.
Do not output any text before diff --git.
"""

        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 8192,
                "temperature": 0.01,
                "top_p": 1.0,
            },
        }

        r = requests.post(SGLANG_URL, json=payload, timeout=600)

        if r.status_code != 200:
            print("SGLang request failed")
            print("status:", r.status_code)
            print("response:", r.text)
            raise SystemExit(1)

        data = r.json()
        raw_text = data.get("text", "")
        patch = extract_patch(raw_text)

        if not patch:
            print("WARNING: model did not return a valid diff patch")
            print("raw output head:")
            print(raw_text[:2000])

        row = {
            "instance_id": ex["instance_id"],
            "model_name_or_path": "Qwen3-Coder-Next-FP8-SGLang",
            "model_patch": patch,
        }

        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("saved:", ex["instance_id"])
        print("patch head:")
        print(patch[:1000])
