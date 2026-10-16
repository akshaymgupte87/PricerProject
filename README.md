# PricerProject

An AI application that estimates product prices, compares them with listing prices, and presents potential deals for human review.

## How it works

Fetch listings → summarize descriptions → retrieve comparable products → combine price estimates → review results.

The project demonstrates:

- **Local LLMs:** Qwen through Ollama for summarization and pricing.
- **Hybrid RAG:** Chroma semantic search and BM25 keyword search for comparable products.
- **Ensemble prediction:** combines specialist, RAG, and neural-network estimates.
- **Fine-tuning and evaluation:** notebooks explore QLoRA and compare pricing models.
- **Human review:** a dashboard records approvals, rejections, and price corrections.

The specialist uses Ollama by default; the fine-tuned adapter is optional.

## Run locally

Requirements: Python 3.14+, uv, and either Ollama or a vLLM server with a compatible Qwen model.

```sh
uv sync --locked
ollama serve
```

Create a `.env` file using [.env.example](.env.example) as a reference. For local execution, set:

```dotenv
OLLAMA_API_BASE=http://localhost:11434
```

Pricing requires a populated `products_vectorstore/` and `artifacts/deep_neural_network.pth`. See the [technical guide](AGENTIFY_PRICER_GUIDE.md) for the vector store and [training guide](BUILD_DEEP_NEURAL_NETWORK.md) for the neural checkpoint. Generated assets are excluded from Git.

The LLM provider is configurable. Ollama is the default:

```dotenv
PRICER_LLM_PROVIDER=ollama
OLLAMA_API_BASE=http://localhost:11434
```

To use a vLLM OpenAI-compatible server instead, set `PRICER_LLM_PROVIDER=vllm` and point `VLLM_API_BASE` (or `PRICER_LLM_API_BASE`) at its `/v1` endpoint. The same model variables are used; vLLM model names are sent through LiteLLM's OpenAI-compatible provider.

For local vLLM, start Qwen from Linux or WSL2 (vLLM does not support native Windows). In the WSL terminal, set the compatibility overrides before launching:

```sh
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_VERSION=v0

vllm serve Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8001
```

Add those `export` lines to `~/.bashrc` if they should apply to every WSL session. These are WSL shell settings; they do not belong in the project's `.env` file.

Then configure:

```dotenv
PRICER_LLM_PROVIDER=vllm
VLLM_API_BASE=http://localhost:8001/v1
PRICER_QWEN_MODEL=Qwen/Qwen2.5-3B-Instruct
```

If you configure an API key for the local vLLM server, set it through your local environment and keep it out of the repository. If vLLM is started without `--api-key`, `OPENAI_API_KEY` is unnecessary.

### LiteLLM gateway

Docker Compose includes a LiteLLM Proxy and Postgres. The API container routes
its agent calls through the gateway; host-run commands continue to use the
provider selected in `.env` unless `PRICER_LLM_PROVIDER=gateway` is set.
The default gateway route points to Ollama on the host. To use vLLM instead,
set these values in `.env`:

```dotenv
PRICER_GATEWAY_UPSTREAM_MODEL=openai/Qwen/Qwen2.5-3B-Instruct
PRICER_GATEWAY_UPSTREAM_API_BASE=http://host.docker.internal:8001/v1
PRICER_GATEWAY_UPSTREAM_API_KEY=not-required
```

Copy `.env.example` to `.env`, then replace `LITELLM_MASTER_KEY`,
`LITELLM_SALT_KEY`, `LITELLM_DB_PASSWORD`, and `PRICER_API_KEY` with generated
secrets. Generate the first three with `openssl rand -hex 32`; prefix the first
two values with `sk-`. Keep the salt key stable because LiteLLM uses it to
protect credentials stored in Postgres.

Start the gateway and app with:

```sh
docker compose up -d
```

The proxy listens on `http://127.0.0.1:4000`; the API container reaches it on
the internal Docker network. LiteLLM stores usage and key records in Postgres;
request bodies are not stored in spend logs by default.
For a dedicated app credential, create a virtual key from the master key:

```sh
curl http://127.0.0.1:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"models":["pricer-local"],"rpm":60,"tpm":12000,"metadata":{"team":"pricer-app"}}'
```

Put the returned token in `.env` as `PRICER_LLM_API_KEY`, then recreate the
API container with `docker compose up -d --force-recreate api`. Until that
optional step, Compose uses the master key locally.
The model alias `pricer-local` routes every agent to the configured upstream.
The current Ollama/vLLM route is local, so token limits are useful; dollar
budgets only help when LiteLLM has a nonzero cost for the selected model.
The current 10-request warm sequential smoke result is recorded in
[`benchmarks/results/gateway-smoke-2026-10-14.json`](benchmarks/results/gateway-smoke-2026-10-14.json);
it validates nonempty responses and basic latency only, not load capacity.
Keep the gateway bound to localhost and do not expose the master key publicly.

#### Repeatable gateway evaluation

The provider-neutral runner compares the same request cases through direct
Ollama and LiteLLM. It records request failures, invalid outputs, latency,
throughput, token usage, and cost only when pricing is known. Reports omit
prompts, responses, and credentials. Run a baseline with:

```sh
uv run pricer gateway-eval --targets ollama,litellm \
  --requests 50 --concurrency 1 --warmups 3
```

Repeat at concurrency 4 and 8 and compare JSON reports. The evaluator exits
nonzero when request-failure, invalid-output, or configured pricing thresholds
are exceeded. Pricing gates (`--max-mae`, `--max-wape`, and
`--max-paired-mae-regression`) require independent holdout labels; set them only
after measuring a baseline. The local smoke run is not a capacity or pricing
accuracy claim.

For a host-run dashboard or command, set `PRICER_LLM_PROVIDER=gateway`,
`PRICER_LLM_GATEWAY_API_BASE=http://localhost:4000/v1`, and
`PRICER_LLM_API_KEY` to the generated virtual key.

```sh
uv run pricer dashboard
```

Open **http://127.0.0.1:7860**. The dashboard scans listings and displays estimates and comparable products for review.

Optional commands:

```sh
uv run pricer api
uv run pricer build-index
uv run pricer --help
```

### Train from reviewed prices (DPO)

When submitting feedback, enter a corrected fair price to create a preference
example. Approve/reject without a correction is retained for review metrics but
does not teach the pricing model a target. The exporter stores the reviewed
product and comparable-price context, and pairs the corrected response as
preferred with the original estimate as rejected. It keeps only the latest
corrected review per product URL.

This is a small human-feedback DPO experiment, not a complete RLHF pipeline. A
corrected price is treated as preferred to the original estimate; that
preference may be noisy and does not prove improved pricing quality. The
exporter uses the same prompt format as the existing Qwen price adapter.

Export and inspect the generated file before training:

```sh
uv run pricer dpo export --dataset artifacts/price_preferences.jsonl
uv run pricer dpo train --dataset artifacts/price_preferences.jsonl \
  --output artifacts/qwen-price-dpo
```

QLoRA DPO requires a CUDA-enabled Linux or WSL2 environment. The default base
is `Qwen/Qwen2.5-3B-Instruct`; training needs at least three corrected products
to form train and validation splits. Three is only a technical minimum, not a
statistically meaningful sample size. The saved adapter is separate from the
active inference adapter. Before using it, compare it with the base model on a
separately curated JSONL holdout whose product IDs/URLs and related products
were excluded from training. Each line needs `id`, `url`, `product_description`,
and `actual_price`; labels should be independently checked and never derived
from either model. The evaluator rejects ID/URL overlap and writes predictions,
MAE/RMSE/WAPE, valid-output rate, latency, paired MAE change, and a bootstrap
confidence interval:

```sh
uv run pricer dpo evaluate \
  --dataset benchmarks/evaluation/pricing_holdout.jsonl \
  --training-data artifacts/price_preferences.jsonl \
  --adapter artifacts/qwen-price-dpo \\
  --report artifacts/dpo-evaluation.json
```

Do not claim an improvement unless the independent holdout results support it.
This is direct preference optimization from human corrections, not a complete
reward-model/PPO RLHF pipeline; uncorrected approvals and rejections are
deliberately excluded from price supervision.

## Project layout

- `src/pricer/` — application, pricing models, retrieval, data processing, and storage.
- `notebooks/` — research and experiments.
- `tests/` — automated tests.
- `artifacts/` — generated models, logs, and local state.

See the [code map](docs/architecture.md) for package details.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

Price estimates are experimental predictions; verify the product and listing before approving a deal.

## Attribution

The QLoRA notebooks were based on Ed Donner's course material. See [docs/origins-and-credits.md](docs/origins-and-credits.md) for the distinction between course-derived research and the original or extended application engineering.
