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
         |- FrontierAgent (Chroma comparables + local Qwen)
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
- Logs are HTML-escaped before rendering.
- Phone and cloud notification calls are disabled.

## Tests

```powershell
$env:LITELLM_LOCAL_MODEL_COST_MAP='True'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The regression suite covers local model routing and parsing, partial pricing
failures, progress streaming, durable memory, table conversion, and worker error
propagation.
