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

For local vLLM, start Qwen from Linux or WSL2 (vLLM does not support native Windows):

```sh
vllm serve Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8001 --api-key local-dev-key
```

Then configure:

```dotenv
PRICER_LLM_PROVIDER=vllm
VLLM_API_BASE=http://localhost:8001/v1
PRICER_QWEN_MODEL=Qwen/Qwen2.5-3B-Instruct
OPENAI_API_KEY=local-dev-key
```

The key is only a local server credential; it is not an OpenAI billing key. If vLLM is started without `--api-key`, `OPENAI_API_KEY` is unnecessary.

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
