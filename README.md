# The Price is Right

A local-first autonomous deal intelligence system. It scans current RSS deals,
uses Qwen through Ollama to select and price products, augments estimates with
similar products from Chroma, and publishes qualified opportunities to a local
Gradio dashboard.

## Architecture

```text
Deal RSS feeds
    -> ScannerAgent (Chroma seen-deal filter + local Qwen structured output)
    -> PlanningAgent (per-deal progress and failure isolation)
    -> EnsembleAgent
         |- FrontierAgent (Chroma dense + SQLite BM25 + cross-encoder + local Qwen)
         |- SpecialistAgent (local Qwen; fine-tuned Qwen2.5 optional)
         `- NeuralNetworkAgent (local .pth checkpoint)
    -> atomic memory.json persistence
    -> persistent Chroma seen_deals identity store
    -> Gradio dashboard + local HTML alerts
```

No OpenAI, Claude, Pushover, or Modal credential is required by the active
dashboard workflow.

## Prerequisites

- Python 3.14 and the project virtual environment
- Ollama running locally
- The `qwen3.6:latest` Ollama model
- An NVIDIA CUDA GPU when using the optional fine-tuned specialist (about 2 GB VRAM)
- A populated `products_vectorstore` Chroma collection
- `artifacts/deep_neural_network.pth`

Start Ollama and confirm the model is installed:

```powershell
ollama serve
ollama list
```

Build the neural-network checkpoint locally if it is missing:

```powershell
.\.venv\Scripts\python.exe .\train_deep_neural_network.py --epochs 5
```

Build the persistent BM25 sidecar after creating or replacing the Chroma product
collection. The application uses dense-only retrieval until this one-time build
has completed:

```powershell
.\.venv\Scripts\python.exe .\build_bm25_index.py
```

## Run the dashboard

Stop any older dashboard process first so it does not continue serving cached
Python modules, then run:

```powershell
.\.venv\Scripts\python.exe .\price_is_right.py
```

The application opens on `http://127.0.0.1:7860`. Saved opportunities appear
immediately. Newly evaluated deals stream into the table one at a time, and a
failure is displayed in the status and log panels instead of leaving the UI
waiting indefinitely.

## Configuration

All settings are optional:

| Variable | Default | Purpose |
|---|---|---|
| `PRICER_QWEN_MODEL` | `qwen3.6:latest` | Shared local Qwen model |
| `PRICER_SPECIALIST_BACKEND` | `ollama` | Use `finetuned` to enable the Qwen2.5 adapter |
| `PRICER_FINETUNED_BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Fine-tuned specialist base model |
| `PRICER_FINETUNED_ADAPTER` | Local August 5 adapter, then Hub fallback | PEFT adapter path or Hub ID |
| `PRICER_SPECIALIST_MODEL` | `PRICER_QWEN_MODEL` | Model used only by the Ollama fallback |
| `PRICER_SCANNER_MODEL` | `qwen3.6:latest` | Scanner override |
| `PRICER_FRONTIER_MODEL` | `qwen3.6:latest` | RAG estimator override |
| `PRICER_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model; set to `off` to disable |
| `PRICER_BM25_PATH` | `products_bm25.sqlite3` | Persistent lexical index path |
| `PRICER_MAX_MODEL_SPREAD_RATIO` | `2.0` | Reject ensemble estimates with greater relative disagreement |
| `PRICER_JUDGE_MODEL` | `qwen3.6:latest` | Optional offline RAG judge model |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Local Ollama endpoint |
| `PRICER_ALERT_DIR` | `artifacts` | Local alert output directory |

## Reliability behavior

- Each successfully priced deal is streamed to the dashboard immediately.
- A failure pricing one deal is logged and the remaining deals continue.
- Worker-level failures always emit a terminal UI event.
- Duplicate timer/button runs are rejected by a non-blocking run lock.
- Feed-local duplicates and tracking-parameter URL variants are collapsed.
- Every successfully scanned candidate is persisted in Chroma as selected or rejected.
- Failed scanner calls remain retryable because their candidates are not marked seen.
- Winning opportunity history is written atomically to `memory.json`.
- Alerts require both a discount greater than `$50` and a discount of at least 20%.
- Logs are HTML-escaped before rendering.
- Phone and cloud notification calls are disabled.

## AI and RAG concepts

The project currently implements:

- **Local LLM inference:** Qwen through Ollama, with an optional fine-tuned
  Qwen2.5 specialist adapter.
- **Embedding and dense retrieval:** MiniLM query embeddings search an
  800,000-product Chroma HNSW collection using its current L2 distance setting.
- **BM25 hybrid search:** SQLite FTS5 supplies lexical candidates, and Reciprocal
  Rank Fusion combines them with Chroma results.
- **Top-K retrieval and reranking:** dense and BM25 retrieval each return 20
  candidates; a cross-encoder reranks the fused set and selects five comparables.
- **RAG context injection:** selected product descriptions and prices are passed
  to local Qwen as evidence for its estimate.
- **RAG evaluation:** Precision@K, Recall@K, reciprocal rank, and nDCG utilities
  support labelled retrieval experiments.
- **LLM-as-judge:** an optional offline structured judge scores comparable-product
  relevance; actual prices remain the source of truth for price evaluation.
- **Guardrails:** structured output, URL and price validation, untrusted-input
  delimiters, ensemble-disagreement checks, and two-part alert thresholds.
- **Multi-agent orchestration:** scanner, preprocessing, retrieval, specialist,
  neural-network, planning, and messaging agents form the end-to-end workflow.

Partially covered or future concepts include human approval, an LLM gateway,
deployment authentication and authorization, metadata filters, context
compression, query expansion, multi-query or recursive retrieval, cosine-distance
evaluation, Self-RAG, Graph RAG, and multimodal RAG. Chunking and chunk overlap
are intentionally absent because each indexed product is currently a short,
atomic record rather than a long document.

## Tests

```powershell
$env:LITELLM_LOCAL_MODEL_COST_MAP='True'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The 26-test regression suite covers hybrid retrieval, BM25 indexing, rank fusion,
reranking, retrieval metrics, LLM judging, price and alert guardrails, local model
routing and parsing, partial pricing failures, progress streaming, durable memory,
table conversion, and worker error propagation.
