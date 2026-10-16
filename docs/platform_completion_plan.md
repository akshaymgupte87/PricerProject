# PricerProject Platform Completion Plan

This checklist completes the remaining AI/LLM platform work without rebuilding the existing `src/pricer` architecture. Record only measurements that are actually run.

## AI model gateway evaluation (implemented)

`pricer gateway-eval` sends the same small JSONL prompt set to direct Ollama and LiteLLM, checks responses, and reports failures, invalid outputs, p50/p95 latency, throughput, token usage, and cost when known. Reports exclude prompts, responses, and credentials. Optional MAE/WAPE gates support independently labelled pricing cases.

The 10-request local smoke passed 10/10 checks for both targets after two warmups. Direct Ollama recorded p50 293 ms, p95 628 ms, and 2.438 requests/sec; LiteLLM recorded p50 269 ms, p95 651 ms, and 2.338 requests/sec. These serial results verify connectivity and basic response checks; there are no independent price labels, and local token costs are unknown.

Repeat the run at concurrency 1, 4, and 8 before setting thresholds:

```powershell
uv run pricer gateway-eval --targets ollama,litellm --requests 50 --concurrency 1 --warmups 3
uv run pricer gateway-eval --targets ollama,litellm --requests 50 --concurrency 4 --warmups 3
uv run pricer gateway-eval --targets ollama,litellm --requests 50 --concurrency 8 --warmups 3
```

To compare vLLM, add a `[targets.vllm]` entry to `benchmarks/gateway/targets.toml` and run with `--targets vllm`. The direct Ollama target uses its native chat API because the OpenAI-compatible route ignored `think=false` in this local Qwen setup.

## 1. Prepare WSL2 and vLLM

From PowerShell:

```powershell
wsl --status
wsl -l -v
```

Inside Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git curl build-essential
nvidia-smi
python3.12 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
python -m pip install --upgrade pip
pip install vllm
vllm --version
```

Do not continue until `nvidia-smi` reports the intended GPU.

## 2. Start and verify Qwen vLLM

```bash
source ~/vllm-env/bin/activate
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_VERSION=v0

vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --api-key local-dev-key \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

In another WSL terminal:

```bash
curl http://localhost:8001/v1/models -H "Authorization: Bearer local-dev-key"
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"Return only 100."}],"max_tokens":8,"temperature":0}'
```

## 3. Connect the application

In the project `.env`:

```dotenv
PRICER_LLM_PROVIDER=vllm
VLLM_API_BASE=http://localhost:8001/v1
PRICER_QWEN_MODEL=Qwen/Qwen2.5-3B-Instruct
OPENAI_API_KEY=local-dev-key
```

If WSL is not reachable through `localhost`, run `hostname -I` in WSL and use that address in `VLLM_API_BASE`.

Start the Windows application:

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m pricer dashboard
```

Open `http://127.0.0.1:7860`.

## 4. Serve the fine-tuned LoRA adapter

Use an adapter directory containing `adapter_config.json` and `adapter_model.safetensors`:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --lora-modules price-specialist=/absolute/path/to/adapter \
  --host 0.0.0.0 --port 8001 \
  --api-key local-dev-key \
  --gpu-memory-utilization 0.85 --max-model-len 8192
```

Configure `PRICER_LLM_PROVIDER=vllm`, `PRICER_SPECIALIST_BACKEND=vllm`, `PRICER_SPECIALIST_MODEL=price-specialist`, `VLLM_API_BASE=http://localhost:8001/v1`, and `OPENAI_API_KEY=local-dev-key`. The direct Transformers/PEFT path remains available with `PRICER_SPECIALIST_BACKEND=finetuned`.

## 5. Run Ollama and vLLM benchmarks

Use `pricer gateway-eval` from the section above. The default target set compares direct Ollama with LiteLLM; for vLLM, add a target to `benchmarks/gateway/targets.toml` and pass `--targets vllm`. Repeat at concurrency `1`, `4`, and `8`, retaining each JSON report. Record model revision, quantization, context length, request count, failures, invalid responses, p50/p95 latency, token usage, and requests/sec. Do not add numbers until commands have actually run.

## 6. Add GPU measurements

Verify NVIDIA telemetry in WSL:

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv
```

The runner records the available GPU model. It does not yet sample GPU utilization or memory; capture those alongside the real provider run with `nvidia-smi` and note the sampling interval.

## 7. Build fixed evaluation data

Create `benchmarks/evaluation/pricing_cases.jsonl` with independently verified labels:

```json
{"id":"laptop-001","description":"Lenovo laptop with Ryzen 5, 16GB RAM, and 512GB SSD","actual_price":599.0,"category":"Electronics"}
```

For retrieval cases, include `relevant_product_ids`. Never use model output as ground truth.

Run measured artifacts through the existing threshold checker:

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m pricer eval --metrics benchmarks\results\vllm-evaluation.json --max mae=100 --max p95_latency_seconds=10 --min recall_at_5=0.8 --min valid_output_rate=0.95
```

Choose thresholds from a measured baseline; do not invent them silently.

## 8. Enable LangSmith

```powershell
uv sync --extra platform
```

Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT=pricerproject`, and `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com` in `.env`. Run a workflow and verify traces for LangGraph execution, retrieval, model calls, tool calls, failures, and latency. Set `LANGCHAIN_TRACING_V2=false` to disable tracing. Never commit credentials.

## 9. Verify LangChain's limited boundary

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -c "from pricer.agents.langchain_components import pricing_prompt; print(pricing_prompt('A laptop'))"
```

Keep the division: LangChain for reusable components, LangGraph for stateful orchestration, LiteLLM for provider abstraction, MCP for external capabilities, and LangSmith for tracing.

## 10. Start Prometheus and Grafana

```bash
docker compose --profile monitoring up --build
```

Open Prometheus at `http://localhost:9090` and Grafana at `http://localhost:3000`. The datasource is provisioned from `monitoring/grafana/datasources/pricer.yml`. Useful queries are `pricer_http_requests_total`, `pricer_http_failures_total`, `pricer_http_latency_seconds_p95`, and `pricer_metric_average`.

## 11. Reorganize notebooks carefully

Search references before moving files:

```powershell
rg -n "Qlora_intro|Qlora_Training|notebooks/|notebooks\\"
New-Item -ItemType Directory -Force notebooks\course
New-Item -ItemType Directory -Force notebooks\experiments
Move-Item Qlora_intro.ipynb notebooks\course\qlora_intro.ipynb
Move-Item Qlora_Training.ipynb notebooks\course\qlora_training.ipynb
```

Update README, attribution, and notebook references after moving files.

## 12. Final checks

Run the full test suite, `uv build`, both `docker compose config` commands, and `git diff --check`. Push only after reviewing `git status` and verifying GitHub Actions completes for tests, package build, CLI smoke test, and Docker build.

The repository now contains the provider abstraction, a compact gateway evaluation command with output checks and release thresholds, DPO holdout evaluation, Prometheus metrics, optional tracing, and optional LangChain instrumentation. Higher-concurrency runs, independently labelled pricing cases, GPU utilization samples, verified deployment, and production thresholds remain evidence-gathering tasks; record results only after they actually run.
