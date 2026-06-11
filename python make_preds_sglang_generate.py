import json
import requests
from datasets import load_from_disk, load_dataset

SGLANG_URL = "http://127.0.0.1:30000/generate"
OUT = "qwen3_sglang_preds_1.jsonl"

ds = load_dataset(
    "princeton-nlp/SWE-bench_Lite",
    split="test[:1]",
    cache_dir="/home/seagl/dataset",
)

with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        prompt = f"""You are solving a SWE-bench task.

Repository: {ex["repo"]}
Base commit: {ex["base_commit"]}
Instance ID: {ex["instance_id"]}

Issue:
{ex["problem_statement"]}

Return ONLY a valid unified git diff patch. Do not include markdown fences.
"""

        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 2048,
                "temperature": 0,
            },
        }

        r = requests.post(SGLANG_URL, json=payload, timeout=600)
        r.raise_for_status()

        data = r.json()
        patch = data.get("text", "").strip()

        row = {
            "instance_id": ex["instance_id"],
            "model_name_or_path": "Qwen3-Coder-Next-FP8-SGLang",
            "model_patch": patch,
        }

        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("saved:", ex["instance_id"])
        print(patch[:500])
