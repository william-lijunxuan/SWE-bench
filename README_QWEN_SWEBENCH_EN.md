# Qwen3-Coder-Next-FP8 Evaluation on SWE-bench Lite

This document explains how to evaluate **Qwen3-Coder-Next-FP8** on **SWE-bench Lite** using:

* **SGLang** to serve the model
* **mini-swe-agent** to generate patches
* **SWE-bench harness** to validate whether the generated patch resolves the issue

The current test instance is:

```text
sympy__sympy-20590
```

Dataset:

```text
princeton-nlp/SWE-bench_Lite
```

---

## 1. Overall Workflow

The full evaluation pipeline contains three main steps:

```text
1. Start Qwen3-Coder-Next-FP8 with SGLang
        ↓
2. Use mini-swe-agent to generate a patch
        ↓
3. Use SWE-bench harness to validate the patch
```

| Component         | Purpose                                                                               |
| ----------------- | ------------------------------------------------------------------------------------- |
| SGLang            | Serves Qwen3-Coder-Next-FP8 through an OpenAI-compatible API                          |
| mini-swe-agent    | Lets the model inspect the repository, run commands, edit code, and generate a patch  |
| SWE-bench harness | Applies the generated patch and runs tests to determine whether the issue is resolved |

---

## 2. Model Configuration

Go to the SWE-bench working directory:

```bash
cd /home/seagl/model/SWE-bench
```

Create or edit the configuration file:

```bash
nano qwen_next_sglang.yaml
```

Example configuration:

```yaml
model:
  model_name: "openai//home/seagl/model/Qwen3-Coder-Next-FP8"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://127.0.0.1:30000/v1"
    api_key: "EMPTY"
    temperature: 1.0
    top_p: 0.95
    max_tokens: 2048
    drop_params: true
    parallel_tool_calls: false
```

### Parameter Explanation

| Parameter             | Meaning                                                                                |
| --------------------- | -------------------------------------------------------------------------------------- |
| `model_name`          | Model name used by mini-swe-agent through LiteLLM                                      |
| `openai//home/...`    | Uses an OpenAI-compatible endpoint; the actual model ID is the local SGLang model path |
| `api_base`            | SGLang API endpoint                                                                    |
| `api_key`             | Local SGLang service can use `EMPTY`                                                   |
| `temperature`         | Sampling temperature                                                                   |
| `top_p`               | Nucleus sampling parameter                                                             |
| `max_tokens`          | Maximum tokens generated per model response                                            |
| `drop_params`         | Drops unsupported API parameters                                                       |
| `parallel_tool_calls` | Disables parallel tool calls for better local model compatibility                      |

> Note: `max_tokens` should not be too large.
> If you encounter context length errors, reduce it to `2048` or `1024`.

---

## 3. Full Startup Script

Save the following script as:

```bash
run_qwen_swebench_one.sh
```

```bash
#!/usr/bin/env bash
set -e

# =========================
# Basic paths
# =========================

SWE_BENCH_DIR="/home/seagl/model/SWE-bench"
MODEL_PATH="/home/seagl/model/Qwen3-Coder-Next-FP8"

QWEN_PYTHON="/home/seagl/miniconda3/envs/qwen/bin/python"
MINI_EXTRA="/home/seagl/miniconda3/envs/qwen/bin/mini-extra"

RUN_ID="qwen-next-lite-one"
OUTPUT_DIR="runs/qwen_next_lite_one"
INSTANCE_ID="sympy__sympy-20590"

SGLANG_PORT="30000"
SGLANG_LOG="sglang_qwen_coder.log"

cd "$SWE_BENCH_DIR"

echo "======================================"
echo "Step 1: Stop old SGLang process"
echo "======================================"

pkill -f sglang || true
sleep 3

echo "======================================"
echo "Step 2: Start SGLang server"
echo "======================================"

nohup python -m sglang.launch_server \
  --model "$MODEL_PATH" \
  --port "$SGLANG_PORT" \
  --tp-size 1 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --mem-fraction-static 0.90 \
  --context-length 262144 \
  --allow-auto-truncate \
  --tool-call-parser qwen3_coder \
  > "$SGLANG_LOG" 2>&1 &

echo "SGLang is starting. Log file: $SGLANG_LOG"

echo "Waiting for SGLang API..."

for i in {1..120}; do
  if curl -s "http://127.0.0.1:${SGLANG_PORT}/v1/models" > /tmp/sglang_models.json; then
    echo "SGLang API is ready."
    cat /tmp/sglang_models.json | python -m json.tool
    break
  fi

  echo "Waiting... $i"
  sleep 5
done

echo "======================================"
echo "Step 3: Test Qwen API"
echo "======================================"

$QWEN_PYTHON - <<'PY'
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="/home/seagl/model/Qwen3-Coder-Next-FP8",
    messages=[{"role": "user", "content": "Only answer OK"}],
    max_tokens=16,
)

print(resp.choices[0].message.content)
PY

echo "======================================"
echo "Step 4: Clean old mini-swe-agent results"
echo "======================================"

sudo rm -rf "$OUTPUT_DIR"

echo "======================================"
echo "Step 5: Generate patch with mini-swe-agent"
echo "======================================"

sudo "$MINI_EXTRA" swebench \
  --output "$OUTPUT_DIR" \
  --subset lite \
  --split test \
  --filter "^(${INSTANCE_ID})$" \
  --workers 1 \
  -c swebench.yaml \
  -c qwen_next_sglang.yaml \
  --environment-class docker

echo "======================================"
echo "Step 6: Show generated prediction"
echo "======================================"

sudo cat "$OUTPUT_DIR/preds.json"

echo "======================================"
echo "Step 7: Evaluate patch with SWE-bench harness"
echo "======================================"

sudo "$QWEN_PYTHON" -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path "$OUTPUT_DIR/preds.json" \
  --max_workers 1 \
  --instance_ids "$INSTANCE_ID" \
  --run_id "$RUN_ID"

echo "======================================"
echo "Done"
echo "======================================"
```

Make the script executable:

```bash
chmod +x run_qwen_swebench_one.sh
```

Run it:

```bash
./run_qwen_swebench_one.sh
```

---

## 4. SGLang Startup Command

The SGLang server is started with:

```bash
python -m sglang.launch_server \
  --model /home/seagl/model/Qwen3-Coder-Next-FP8 \
  --port 30000 \
  --tp-size 1 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --mem-fraction-static 0.90 \
  --context-length 262144 \
  --allow-auto-truncate \
  --tool-call-parser qwen3_coder
```

### Parameter Explanation

| Parameter                        | Meaning                                                                  |
| -------------------------------- | ------------------------------------------------------------------------ |
| `python -m sglang.launch_server` | Starts the SGLang inference server                                       |
| `--model`                        | Local model path                                                         |
| `--port 30000`                   | API service port                                                         |
| `--tp-size 1`                    | Tensor parallel size; use `1` for a single GPU                           |
| `--attention-backend triton`     | Uses the Triton attention backend                                        |
| `--sampling-backend pytorch`     | Uses PyTorch sampling to avoid FlashInfer compilation issues             |
| `--disable-cuda-graph`           | Disables CUDA graph for better compatibility                             |
| `--mem-fraction-static 0.90`     | Allocates 90% of GPU memory for model and KV cache                       |
| `--context-length 262144`        | Sets the maximum model context length to 262,144 tokens                  |
| `--allow-auto-truncate`          | Automatically truncates overly long input instead of failing immediately |
| `--tool-call-parser qwen3_coder` | Enables Qwen3-Coder tool call parsing                                    |

---

## 5. Patch Generation with mini-swe-agent

Command:

```bash
sudo /home/seagl/miniconda3/envs/qwen/bin/mini-extra swebench \
  --output runs/qwen_next_lite_one \
  --subset lite \
  --split test \
  --filter '^(sympy__sympy-20590)$' \
  --workers 1 \
  -c swebench.yaml \
  -c qwen_next_sglang.yaml \
  --environment-class docker
```

### Purpose

This step uses mini-swe-agent to call Qwen3-Coder-Next-FP8 through SGLang. The agent enters the SWE-bench Docker environment, inspects the repository, runs commands, edits files, and generates a patch.

The output is saved to:

```text
runs/qwen_next_lite_one/preds.json
```

### Parameter Explanation

| Parameter                           | Meaning                                     |
| ----------------------------------- | ------------------------------------------- |
| `sudo`                              | Required if Docker needs root permission    |
| `mini-extra swebench`               | Runs mini-swe-agent in SWE-bench mode       |
| `--output runs/qwen_next_lite_one`  | Output directory                            |
| `--subset lite`                     | Uses SWE-bench Lite                         |
| `--split test`                      | Uses the test split                         |
| `--filter '^(sympy__sympy-20590)$'` | Runs only the specified instance            |
| `--workers 1`                       | Runs one task at a time                     |
| `-c swebench.yaml`                  | Loads the SWE-bench agent configuration     |
| `-c qwen_next_sglang.yaml`          | Loads the Qwen + SGLang model configuration |
| `--environment-class docker`        | Runs the task inside a Docker environment   |

---

## 6. Check the Generated Patch

After patch generation, check:

```bash
sudo cat runs/qwen_next_lite_one/preds.json
```

A successful patch should look like:

```json
{
  "sympy__sympy-20590": {
    "model_name_or_path": "openai//home/seagl/model/Qwen3-Coder-Next-FP8",
    "instance_id": "sympy__sympy-20590",
    "model_patch": "diff --git a/..."
  }
}
```

The key field is:

```json
"model_patch": "diff --git ..."
```

If it is empty:

```json
"model_patch": ""
```

then the model did not generate a valid patch, and SWE-bench evaluation cannot proceed.

---

## 7. Patch Validation with SWE-bench Harness

Command:

```bash
sudo /home/seagl/miniconda3/envs/qwen/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path runs/qwen_next_lite_one/preds.json \
  --max_workers 1 \
  --instance_ids sympy__sympy-20590 \
  --run_id qwen-next-lite-one
```

### Purpose

This step uses the official SWE-bench harness to validate the patch generated by Qwen.

The harness will:

```text
Read preds.json
        ↓
Apply model_patch
        ↓
Run tests inside Docker
        ↓
Determine whether the issue is resolved
```

### Parameter Explanation

| Parameter                                     | Meaning                                          |
| --------------------------------------------- | ------------------------------------------------ |
| `python -m swebench.harness.run_evaluation`   | Starts the SWE-bench official evaluation harness |
| `--dataset_name princeton-nlp/SWE-bench_Lite` | Uses the SWE-bench Lite dataset                  |
| `--predictions_path`                          | Path to the model prediction file                |
| `--max_workers 1`                             | Runs one evaluation worker                       |
| `--instance_ids sympy__sympy-20590`           | Evaluates only the specified instance            |
| `--run_id qwen-next-lite-one`                 | Name of this evaluation run                      |

---

## 8. How to Read the Evaluation Result

A successful result looks like:

```text
Total instances: 1
Instances submitted: 1
Instances completed: 1
Instances resolved: 1
Instances unresolved: 0
Instances with empty patches: 0
Instances with errors: 0
```

### Field Explanation

| Field                          | Meaning                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| `Total instances`              | Total number of evaluated instances                                  |
| `Instances submitted`          | Number of submitted predictions                                      |
| `Instances completed`          | Number of instances that completed the evaluation process            |
| `Instances resolved`           | Number of instances successfully fixed                               |
| `Instances unresolved`         | Number of instances not fixed                                        |
| `Instances with empty patches` | Number of instances with empty patches                               |
| `Instances with errors`        | Number of instances that failed due to runtime or environment errors |

If you see:

```text
Instances resolved: 1
```

then the patch successfully fixed the issue.

If you see:

```text
Instances unresolved: 1
```

then the patch was generated but failed the tests.

If you see:

```text
Instances with empty patches: 1
```

then no valid patch was generated.

---

## 9. Example Successful Result

For `sympy__sympy-20590`, the successful report was:

```json
{
  "total_instances": 1,
  "submitted_instances": 1,
  "completed_instances": 1,
  "resolved_instances": 1,
  "unresolved_instances": 0,
  "empty_patch_instances": 0,
  "error_instances": 0,
  "completed_ids": [
    "sympy__sympy-20590"
  ],
  "resolved_ids": [
    "sympy__sympy-20590"
  ],
  "schema_version": 2
}
```

This means:

```text
Qwen3-Coder-Next-FP8 + SGLang + mini-swe-agent
resolved 1 out of 1 SWE-bench Lite instance.
```

Result:

```text
resolved = 1/1
```

---

## 10. Common Issues

### 10.1 Empty `model_patch`

Symptom:

```json
"model_patch": ""
```

Possible causes:

* The model did not produce a valid `git diff`
* The agent stopped before generating a patch
* Context length exceeded the limit
* Docker or tool execution failed

Check the log:

```bash
sudo cat runs/qwen_next_lite_one/minisweagent.log
```

---

### 10.2 Context Length Error

Symptom:

```text
Requested token count exceeds the model's maximum context length
```

Solutions:

Reduce `max_tokens` in `qwen_next_sglang.yaml`:

```yaml
max_tokens: 2048
```

Or start SGLang with:

```bash
--allow-auto-truncate
```

---

### 10.3 FlashInfer `-lcuda` Error

Symptom:

```text
ld: cannot find -lcuda
```

Solution:

Start SGLang with:

```bash
--attention-backend triton
--sampling-backend pytorch
```

This avoids FlashInfer sampling compilation issues.

---

### 10.4 Docker Permission Error

Symptom:

```text
permission denied while trying to connect to the Docker daemon socket
```

Solution:

Use `sudo`:

```bash
sudo ...
```

Or add the user to the Docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 11. Evaluate More Instances

The current run evaluates only one instance. To evaluate the first 5 instances:

```bash
sudo rm -rf runs/qwen_next_lite_5

sudo /home/seagl/miniconda3/envs/qwen/bin/mini-extra swebench \
  --output runs/qwen_next_lite_5 \
  --subset lite \
  --split test \
  --slice 0:5 \
  --workers 1 \
  -c swebench.yaml \
  -c qwen_next_sglang.yaml \
  --environment-class docker
```

Then validate:

```bash
sudo /home/seagl/miniconda3/envs/qwen/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path runs/qwen_next_lite_5/preds.json \
  --max_workers 1 \
  --run_id qwen-next-lite-5
```

If the first 5 instances run successfully, you can gradually scale to 10, 50, or the full SWE-bench Lite set.
