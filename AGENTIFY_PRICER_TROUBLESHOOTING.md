# Agentify Pricer notebook troubleshooting

This guide records the recurring fixes used by `notebooks/Agentify_Pricer.ipynb`.

## Local Qwen through LiteLLM

Ollama displays model names without a provider prefix:

```text
qwen3.6:latest
```

LiteLLM requires the provider in front of that tag:

```python
local_qwen_tag = os.getenv("PRICER_QWEN_MODEL", "qwen3.6:latest").strip()
LOCAL_QWEN_MODEL = (
    local_qwen_tag
    if local_qwen_tag.startswith("ollama/")
    else f"ollama/{local_qwen_tag}"
)
OLLAMA_API_BASE = "http://localhost:11434"
```

`agents/frontier_agent.py` also uses this local Ollama route. Its optional model override is:

```dotenv
PRICER_FRONTIER_MODEL=qwen3.6:latest
```

Do not instantiate `OpenAI()` or select a `gpt-*` model in `FrontierAgent` when running fully locally; doing so requires `OPENAI_API_KEY`. The local implementation uses `litellm.completion`, the `ollama/` provider prefix, and `localhost:11434` instead.

Use this complete call pattern:

```python
from litellm import completion

response = completion(
    model=LOCAL_QWEN_MODEL,
    api_base=OLLAMA_API_BASE,
    messages=messages,
    temperature=0,
    max_tokens=50,
    num_ctx=4096,
    think=False,
    keep_alive="30m",
    seed=42,
)

reply = response.choices[0].message.content
```

Do not pass the bare `qwen3.6:latest` tag to LiteLLM. That produces:

```text
LLM Provider NOT provided
```

Do not copy OpenAI-only arguments such as `reasoning_effort="none"` into this Ollama call. `think=False` is the relevant Qwen/Ollama option for returning the short answer without a reasoning trace.

To switch local models without editing the notebook, add these optional values to `.env`:

```dotenv
PRICER_QWEN_MODEL=qwen3.6:latest
OLLAMA_API_BASE=http://localhost:11434
```

Confirm the exact installed Ollama tag before changing `PRICER_QWEN_MODEL`:

```powershell
ollama list
```

The normalization code accepts either `qwen3.6:latest` or `ollama/qwen3.6:latest` from `.env` and always gives LiteLLM the required provider prefix.

LiteLLM may warn that it could not download its remote model-cost map and is using a local backup. That warning does not mean the Qwen request went to the cloud or failed; Ollama inference still runs locally through `localhost:11434`.

For evaluation, start with a small workload and one worker:

```python
evaluate(qwen_rag, test, size=20, workers=1)
```

After it works, increase `size` gradually. Keep `workers=1` for a large local model unless the machine has enough RAM/VRAM for concurrent inference.

## Chroma: `too many SQL variables`

Do not retrieve all 800,000 records with one `collection.get()` call. Chroma uses SQLite internally, and a very large request can exceed SQLite's bound-variable limit.

Use pagination:

```python
BATCH_SIZE = 10_000
total = collection.count()

for offset in range(0, total, BATCH_SIZE):
    page = collection.get(
        include=["embeddings", "metadatas"],
        limit=min(BATCH_SIZE, total - offset),
        offset=offset,
    )
    # Process this page before loading the next one.
```

The 10,000-record batch size was tested against this project's local Chroma database.

## Chroma metadata dictionary-key warning

Chroma declares metadata values as a broad union that includes unhashable values such as lists. A type checker therefore rejects this:

```python
category_to_code.get(metadata.get("category"), -1)
```

Narrow the value to `str` first:

```python
def get_category_code(metadata):
    category = metadata.get("category")
    return category_to_code.get(category, -1) if isinstance(category, str) else -1
```

## Full-dataset 2D and 3D visualization

Do not run t-SNE over all 800,000 embeddings. Instead, use the notebook's batched `IncrementalPCA` cell.

The PCA cell must use three components:

```python
ipca = IncrementalPCA(n_components=3, batch_size=10_000)
projected = np.empty((total_records, 3), dtype=np.float32)
```

Run cells in this order whenever the PCA configuration changes:

1. Run the full-dataset PCA cell and wait for both progress bars.
2. Run the following 2D/3D scatter cell.

If the scatter cell reports that `projected` has fewer than three columns, the kernel still contains the old two-component array. Rerun the PCA cell; editing its source does not update variables already held in memory.

Static rasterized Matplotlib plots are intentional. An interactive Plotly figure containing 800,000 individual 3D markers can consume excessive browser memory or become unresponsive.

## Quick diagnostic checklist

1. Run `ollama list` and copy the exact model tag.
2. Add `ollama/` before that tag when calling LiteLLM.
3. Set `api_base="http://localhost:11434"`.
4. Test `qwen_rag(test[0])` before running `evaluate(...)`.
5. Use `workers=1` for local evaluation.
6. Page large Chroma reads with `limit` and `offset`.
7. After changing PCA dimensions, rerun the PCA cell before the plotting cell.
