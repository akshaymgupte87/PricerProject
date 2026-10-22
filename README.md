# PricerProject

**Product pricing and deal intelligence · Python, PyTorch, Chroma, SQLite BM25, FastAPI, Gradio, Ollama/vLLM**

I built a local-first application that compares deal listings with product evidence, combines three pricing signals, and routes uncertain opportunities through a human review workflow. The main engineering challenge is making model estimates inspectable: each opportunity carries comparable-product evidence, model disagreement, confidence, and a saved review decision. This is a portfolio project and research system; it has not been validated for production pricing decisions.

[Run the offline walkthrough](#quick-demo) · [Architecture](docs/guides/architecture.md) · [Source](src/pricer/) · [Attribution](docs/guides/origins-and-credits.md)

**Resume-ready summary:** Built a local-first product deal intelligence application in Python using hybrid Chroma/BM25 retrieval, a three-signal pricing ensemble, FastAPI/Gradio interfaces, and an auditable human-review workflow; added provider routing, evaluation tooling, and reproducible model/index jobs.

## What I built

- Packaged the application as `src/pricer`, with lazy CLI startup, configuration, API/dashboard interfaces, and persistent review history ([architecture](docs/guides/architecture.md)).
- Implemented hybrid semantic and BM25 retrieval with reranking and cited comparables, then combined retrieved-price, local-LLM, and neural estimates ([retrieval](src/pricer/agents/retrieval.py), [ensemble](src/pricer/agents/ensemble_agent.py)).
- Added guardrails, confidence and disagreement signals, and a human approval flow that records decisions and corrected prices ([dashboard](src/pricer/interfaces/dashboard.py), [persistence](src/pricer/persistence/intelligence_store.py)).
- Added interchangeable Ollama/vLLM/LiteLLM routing, CI, and benchmark/evaluation commands, plus a reproducible streaming index builder ([prepare_index.py](src/pricer/jobs/prepare_index.py)). Course-derived QLoRA notebooks are credited separately in [origins and credits](docs/guides/origins-and-credits.md).

## Evaluation evidence

There is currently **no independently labelled pricing holdout result checked into this repository**, so no pricing MAE/WAPE or accuracy claim is reported. The tracked [gateway smoke measurement](benchmarks/results/gateway-smoke-2026-10-14.json) covers 10 warm, sequential LiteLLM requests: 10 succeeded, mean latency was 0.207 s, p95 was 0.253 s, and throughput was 4.822 requests/s. It measures nonempty responses and latency for one local setup; it says nothing about price accuracy, concurrency capacity, cold starts, or other machines.

For a pricing-quality claim, prepare independent product labels that do not overlap training products, run `pricer dpo evaluate` (or the shared evaluation tooling), and commit the dataset description, baseline, sample count, metrics, and report under `benchmarks/`. The DPO evaluator rejects ID/URL overlap, but the repository does not yet contain that independent holdout or a completed result. The small fixture below is synthetic and must never be counted as evaluation data.

## Quick demo

The **AI deal-pricing walkthrough** replays a real local dashboard run captured on October 1, 2026. It shows DealNews RSS intake, Qwen's structured selection, description preprocessing, Chroma/BM25 retrieval, reciprocal-rank fusion, cross-encoder reranking, all three price signals, the 80/10/10 ensemble, and the planner's threshold decision. Its saved outputs and five comparables are historical model results, not current market prices or proof of pricing accuracy. Replay mode does not contact feeds or run models; reviewer choices update only the browser session. For a fresh scan and live inference, use the dashboard setup below. Pricing accuracy has not been established on an independent labeled holdout.

```sh
uv sync --locked
uv run pricer demo
```

Open **http://127.0.0.1:7860** to start with one guided example: a DJI camera listing that the run did not flag because its estimated fair price was below asking. The deal status shows each step from feed scan through the automatic screen to pending human review. Inspect its actual model outputs and retrieved products, then choose approve or reject, give a reason of at least 10 characters, and optionally enter an independently checked fair price. Applying the review visibly changes the status and adds the reason and any correction to the details. The corrected label is shown separately; the original estimate is preserved. These changes last only for the current browser session and do not alter the saved run or train a model. The other four captured deals and engineering stages are available to explore. The replay records that no deal met the planner's alert threshold. Fine-tuning is a separate feedback-training path, and the capture identifies the Ollama backend that actually ran. For a fresh RSS scan and live inference, follow [Live pricing setup](#live-pricing-setup) and launch the dashboard. The walkthrough binds to localhost and does not share publicly.

## Live pricing setup

Live pricing needs a Qwen-compatible Ollama or vLLM endpoint, a Chroma product collection, a BM25 sidecar, and `artifacts/deep_neural_network.pth`. For a fresh checkout, prepare a bounded local index from the public `ed-donner/items_full` dataset, then train the neural model from that dataset:

```sh
uv sync --locked
ollama pull qwen3.6:latest
ollama serve
uv run pricer prepare-index --max-items 100000
uv run pricer train --profile improved --output artifacts/deep_neural_network.pth
uv run pricer dashboard
```

Before starting, copy `.env.example` to `.env` and set `OLLAMA_API_BASE=http://localhost:11434` for a host-run Ollama service. For a vLLM service use the configuration further down. The index job downloads the dataset and the `sentence-transformers/all-MiniLM-L6-v2` encoder on first use. The 100,000-item setting is a setup-sized index, not an evaluation sample; increase `--max-items` for a larger retrieval corpus. Training uses the dataset's training split and validation split and may take a long time, especially on CPU. `prepare-index` adds/updates collection records and rebuilds the BM25 sidecar. To rebuild BM25 after changing an existing collection, run `uv run pricer build-index`.

The full retrieval/training details live in [the technical guide](docs/guides/AGENTIFY_PRICER_GUIDE.md), [the neural training guide](docs/guides/BUILD_DEEP_NEURAL_NETWORK.md), and [the local alert guide](docs/guides/LOCAL_DEAL_ALERTS.md). Generated indexes and checkpoints are intentionally excluded from Git.

## How it works

Fetch listings → summarize descriptions → retrieve comparable products → combine price estimates → review results.

The project demonstrates:

- **Local LLMs:** Qwen through Ollama for summarization and pricing.
- **Hybrid RAG:** Chroma semantic search and BM25 keyword search for comparable products.
- **Ensemble prediction:** combines specialist, RAG, and neural-network estimates.
- **Fine-tuning and evaluation:** notebooks explore QLoRA and compare pricing models.
- **Human review:** a dashboard records approvals, rejections, and price corrections.

The specialist uses Ollama by default; the fine-tuned adapter is optional.

<details>
<summary>Advanced provider and gateway setup</summary>

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

The live application uses the index and checkpoint provisioned above. Demo mode needs neither.

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

</details>

<details>
<summary>Human feedback and DPO experiment</summary>

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

</details>

## Project layout

- `src/pricer/` — application, pricing models, retrieval, data processing, and storage.
- `docs/guides/` — all project Markdown and Jupyter notebooks, including the runnable package tour, historical research, QLoRA experiments, and operator guides. Start with [the notebook guide](docs/guides/notebooks.md).
- `tests/` — automated tests.
- `benchmarks/demo/` — explicitly synthetic walkthrough fixtures, not evaluation data.
- `artifacts/` — generated models, logs, and local state (not committed).

See the [code map](docs/guides/architecture.md) for package details.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

Price estimates are experimental predictions; verify the product and listing before approving a deal.

## Attribution

The QLoRA notebooks were based on Ed Donner's course material. See [the attribution notes](docs/guides/origins-and-credits.md) for the distinction between course-derived research and the original or extended application engineering.
