# Qwen3-Coder-Next-FP8 在 SWE-bench Lite 上的评估流程说明

本文档介绍如何使用：

* **SGLang** 启动 `Qwen3-Coder-Next-FP8`
* **mini-swe-agent** 生成 SWE-bench 补丁 `preds.json`
* **SWE-bench harness** 验证补丁是否修复成功

当前测试样本：

```text
sympy__sympy-20590
```

该样本来自：

```text
princeton-nlp/SWE-bench_Lite
```

---

## 一、整体流程

完整评估分为三步：

```text
1. 启动 SGLang，加载 Qwen3-Coder-Next-FP8
        ↓
2. 使用 mini-swe-agent 调用 Qwen 生成补丁
        ↓
3. 使用 SWE-bench harness 验证补丁是否通过测试
```

其中：

| 阶段                | 作用                                           |
| ----------------- | -------------------------------------------- |
| SGLang            | 提供 Qwen3-Coder-Next-FP8 的 OpenAI API 服务      |
| mini-swe-agent    | 让模型进入 SWE-bench Docker 环境，读取代码、修改代码、生成 patch |
| SWE-bench harness | 应用模型生成的 patch，运行测试，判断是否 resolved             |

---

## 二、准备配置文件

在 SWE-bench 项目目录下：

```bash
cd /home/seagl/model/SWE-bench
```

创建或修改 `qwen_next_sglang.yaml`：

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

### 参数说明

| 参数                    | 说明                                             |
| --------------------- | ---------------------------------------------- |
| `model_name`          | mini-swe-agent 通过 LiteLLM 调用模型的名称              |
| `openai//home/...`    | 表示使用 OpenAI-compatible API，后面是 SGLang 暴露的模型 ID |
| `api_base`            | SGLang 服务地址                                    |
| `api_key`             | 本地服务可填 `EMPTY`                                 |
| `temperature`         | 采样温度，代码任务可以用 `1.0`                             |
| `top_p`               | nucleus sampling 参数                            |
| `max_tokens`          | 每次模型最多生成多少 token                               |
| `drop_params`         | 丢弃不兼容的 API 参数                                  |
| `parallel_tool_calls` | 关闭并行工具调用，避免部分本地模型兼容问题                          |

> 注意：`max_tokens` 不宜过大。
> 如果上下文报错，可以改成 `1024` 或 `2048`。

---

## 三、启动脚本

可以保存为：

```bash
run_qwen_swebench_one.sh
```

内容如下：

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
    messages=[{"role": "user", "content": "只回答 OK"}],
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

赋予执行权限：

```bash
chmod +x run_qwen_swebench_one.sh
```

运行：

```bash
./run_qwen_swebench_one.sh
```

---

## 四、SGLang 启动命令解释

脚本中的 SGLang 启动命令是：

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

### 参数说明

| 参数                               | 说明                                     |
| -------------------------------- | -------------------------------------- |
| `python -m sglang.launch_server` | 启动 SGLang 推理服务                         |
| `--model`                        | 模型路径                                   |
| `--port 30000`                   | API 服务端口                               |
| `--tp-size 1`                    | Tensor Parallel 数量，单卡用 `1`             |
| `--attention-backend triton`     | 使用 Triton attention 后端                 |
| `--sampling-backend pytorch`     | 使用 PyTorch sampling，避免 FlashInfer 编译问题 |
| `--disable-cuda-graph`           | 关闭 CUDA graph，降低兼容问题                   |
| `--mem-fraction-static 0.90`     | 分配 90% 显存给模型和 KV cache                 |
| `--context-length 262144`        | 设置最大上下文长度为 262144 tokens               |
| `--allow-auto-truncate`          | 输入过长时自动截断，避免直接报错                       |
| `--tool-call-parser qwen3_coder` | 启用 Qwen3-Coder 工具调用解析器                 |

---

## 五、生成补丁命令解释

生成补丁使用：

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

### 作用

这一步会让 mini-swe-agent 调用 Qwen3-Coder-Next-FP8，进入 SWE-bench 的 Docker 环境，尝试修复 `sympy__sympy-20590` 这个问题，并生成补丁。

生成结果保存在：

```text
runs/qwen_next_lite_one/preds.json
```

### 参数说明

| 参数                                  | 说明                               |
| ----------------------------------- | -------------------------------- |
| `sudo`                              | 因为 Docker 需要权限                   |
| `mini-extra swebench`               | 使用 mini-swe-agent 的 SWE-bench 模式 |
| `--output runs/qwen_next_lite_one`  | 输出目录                             |
| `--subset lite`                     | 使用 SWE-bench Lite                |
| `--split test`                      | 使用 test split                    |
| `--filter '^(sympy__sympy-20590)$'` | 只运行一个指定样本                        |
| `--workers 1`                       | 并发数为 1                           |
| `-c swebench.yaml`                  | 加载 SWE-bench 默认 agent 配置         |
| `-c qwen_next_sglang.yaml`          | 加载 Qwen + SGLang 模型配置            |
| `--environment-class docker`        | 使用 Docker 环境执行任务                 |

---

## 六、检查生成的补丁

生成完成后查看：

```bash
sudo cat runs/qwen_next_lite_one/preds.json
```

如果成功，会看到：

```json
{
  "sympy__sympy-20590": {
    "model_name_or_path": "openai//home/seagl/model/Qwen3-Coder-Next-FP8",
    "instance_id": "sympy__sympy-20590",
    "model_patch": "diff --git a/..."
  }
}
```

关键字段是：

```json
"model_patch": "diff --git ..."
```

如果是：

```json
"model_patch": ""
```

说明模型没有生成有效 patch，不能进入正式评测。

---

## 七、验证补丁命令解释

验证补丁使用：

```bash
sudo /home/seagl/miniconda3/envs/qwen/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path runs/qwen_next_lite_one/preds.json \
  --max_workers 1 \
  --instance_ids sympy__sympy-20590 \
  --run_id qwen-next-lite-one
```

### 作用

这一步会使用 SWE-bench harness：

```text
读取 preds.json
        ↓
应用 model_patch
        ↓
在 Docker 容器中运行测试
        ↓
判断 patch 是否修复成功
```

### 参数说明

| 参数                                            | 说明                    |
| --------------------------------------------- | --------------------- |
| `python -m swebench.harness.run_evaluation`   | 启动 SWE-bench 官方评测器    |
| `--dataset_name princeton-nlp/SWE-bench_Lite` | 指定数据集为 SWE-bench Lite |
| `--predictions_path`                          | 指定模型生成的预测文件           |
| `--max_workers 1`                             | 同时评测 1 个任务            |
| `--instance_ids sympy__sympy-20590`           | 只评测指定样本               |
| `--run_id qwen-next-lite-one`                 | 本次评测的运行 ID            |

---

## 八、评测结果怎么看

成功时会看到类似：

```text
Total instances: 1
Instances submitted: 1
Instances completed: 1
Instances resolved: 1
Instances unresolved: 0
Instances with empty patches: 0
Instances with errors: 0
```

### 字段说明

| 字段                             | 含义           |
| ------------------------------ | ------------ |
| `Total instances`              | 总样本数         |
| `Instances submitted`          | 提交评测的样本数     |
| `Instances completed`          | 成功完成测试流程的样本数 |
| `Instances resolved`           | 成功修复的样本数     |
| `Instances unresolved`         | 没有修复成功的样本数   |
| `Instances with empty patches` | patch 为空的样本数 |
| `Instances with errors`        | 评测过程出错的样本数   |

如果出现：

```text
Instances resolved: 1
```

说明模型成功修复该 SWE-bench 问题。

如果出现：

```text
Instances unresolved: 1
```

说明模型生成了 patch，但没有通过测试。

如果出现：

```text
Instances with empty patches: 1
```

说明模型没有生成有效 patch。

---

## 九、本次单样本成功结果示例

本次 `sympy__sympy-20590` 的结果为：

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

说明：

```text
Qwen3-Coder-Next-FP8 + SGLang + mini-swe-agent
在 SWE-bench Lite 的 sympy__sympy-20590 单样本上 resolved = 1/1
```

也就是该样本修复成功。

---

## 十、常见问题

### 1. `model_patch` 是空的

表现：

```json
"model_patch": ""
```

原因可能是：

* 模型没有生成 `git diff`
* agent 中途失败
* 上下文超限
* Docker 环境异常

解决：

```bash
sudo cat runs/qwen_next_lite_one/minisweagent.log
```

查看日志。

---

### 2. context length 超限

表现：

```text
Requested token count exceeds the model's maximum context length
```

解决方式：

降低 `qwen_next_sglang.yaml` 中的：

```yaml
max_tokens: 2048
```

或者 SGLang 启动时加：

```bash
--allow-auto-truncate
```

---

### 3. FlashInfer 报 `cannot find -lcuda`

表现：

```text
ld: cannot find -lcuda
```

解决方式：

SGLang 启动时使用：

```bash
--attention-backend triton
--sampling-backend pytorch
```

避免使用 FlashInfer sampling。

---

### 4. Docker 权限问题

表现：

```text
permission denied while trying to connect to the Docker daemon socket
```

解决：

```bash
sudo ...
```

或者把当前用户加入 docker 组：

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 十一、下一步：评测更多样本

当前只评测了 1 个样本。可以继续跑前 5 个样本：

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

然后验证：

```bash
sudo /home/seagl/miniconda3/envs/qwen/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path runs/qwen_next_lite_5/preds.json \
  --max_workers 1 \
  --run_id qwen-next-lite-5
```

如果前 5 个样本稳定，再逐步扩大到 10 个、50 个，最后跑完整 SWE-bench Lite。
