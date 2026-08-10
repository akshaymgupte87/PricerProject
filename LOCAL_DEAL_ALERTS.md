# Local deal alerts (no phone required)

The project's `MessagingAgent` now keeps notifications entirely on this computer.
It does not call Pushover, Claude, OpenAI, or another remote notification/model API.

## End-to-end application flow

The active application starts in `price_is_right.py`. The final notebook cell runs
that Python file; the older Gradio cells in the notebook are demonstrations and are
not part of the active dashboard flow.

```text
Browser / Gradio dashboard
        |
        | page load, Run button, or five-minute timer
        v
DealAgentFramework
        |
        v
PlanningAgent
        |
        +--> ScannerAgent
        |      1. Read DealNews RSS feeds
        |      2. Follow deal pages and extract product details
        |      3. Canonicalize URLs and exclude records in Chroma seen_deals
        |      4. Ask local Qwen to select five clear, detailed deals
        |      5. Persist every processed candidate as selected or rejected
        |
        +--> EnsembleAgent, once for each selected deal
        |      +--> Preprocessor: normalize the description with local Qwen
        |      +--> SpecialistAgent: estimate value with local Qwen
        |      +--> FrontierAgent: hybrid retrieval, reranking, and local Qwen
        |      `--> NeuralNetworkAgent: run the local .pth regression model
        |             |
        |             `--> 80% frontier + 10% specialist + 10% neural network
        |
        +--> Stream each priced opportunity to the dashboard
        |
        `--> Select the largest estimated discount
               |
               +--> require discount > $50 and at least 20% of estimated value
               `--> save the winner and publish a local alert
```

### 1. Dashboard startup

`App.run()` creates `DealAgentFramework`. The framework opens the persistent
`products_vectorstore` Chroma database, loads saved opportunities from
`memory.json`, and exposes them to the table immediately. The 3D plot is built
from a sample of the product embeddings.

Opening the page starts a background planning run. A run can also be requested
with **Run deal scan now**, and a Gradio timer requests another run every five
minutes. A non-blocking lock prevents two planning runs from executing at the
same time.

### 2. Scan and selection

`ScannerAgent` reads up to ten entries from each configured DealNews RSS feed.
It visits the linked pages, cleans and truncates their product text, and removes
URLs found in the saved opportunity memory. It also canonicalizes URLs, removes
known tracking parameters such as `utm_*` and `iref`, collapses duplicates within
the feed response, and excludes IDs stored in the separate Chroma `seen_deals`
collection. It then sends only unseen descriptions to `qwen3.6:latest` through
local Ollama. Qwen returns a structured `DealSelection` containing up to five
products with a description, advertised price, and URL.

After that structured response has been validated, all candidates sent to Qwen
are stored in `seen_deals` with selected/rejected status and their first-seen
time. If the scanner-model call or response validation fails, nothing is marked
as seen, so a later run can retry the batch. The seen collection uses constant
one-dimensional embeddings because it is an identity store, not a similarity
index; the existing `products` collection remains dedicated to RAG retrieval.

### 3. Price estimation

`PlanningAgent` evaluates the five selected deals sequentially. For each deal,
`EnsembleAgent` first normalizes the description and obtains three estimates:

- `SpecialistAgent` asks local Qwen for a direct fair-retail estimate.
- `FrontierAgent` embeds the description with `all-MiniLM-L6-v2`. It retrieves 20
  dense candidates from Chroma and 20 lexical candidates from the persistent
  SQLite BM25 index, combines them with Reciprocal Rank Fusion, reranks the fused
  candidates with a cross-encoder, and gives the best five comparables to Qwen.
- `NeuralNetworkAgent` runs the local `artifacts/deep_neural_network.pth` model.

The final estimate is calculated as:

```text
estimate = 0.80 * frontier + 0.10 * specialist + 0.10 * neural_network
discount = estimate - advertised_price
```

A failure on one deal is logged without stopping the other deals. Invalid prices
and excessive disagreement between the three pricing models are rejected. Every
successful result is sent to the UI callback, so rows should appear one at a time
while the run is still in progress.

### 4. Winner, memory, and alerts

After pricing finishes, `PlanningAgent` sorts the opportunities by estimated
discount. The best deal must save more than `$50` and at least 20% of its estimated
value before it is published as a local alert and returned to the framework. The
framework appends the winner to `memory.json` only when its URL is not already
present and writes the file atomically.

The table combines saved winners with all successfully priced opportunities from
the current run. Consequently, five rows may be visible immediately after a run,
while only the qualifying winner survives an application restart. Feed
deduplication is independent of this display memory: both winners and non-winners
are remembered in Chroma and will not be sent through the scanner model again.

### 5. Dashboard and manual alert behavior

The dashboard receives logs, table rows, completion state, and errors from queues
owned by the background worker. Clicking a table row republishes that opportunity
through `MessagingAgent`. This manual action does not reprice the product.

## What happens when a deal is found

When `PlanningAgent` finds a deal that passes its discount threshold, the alert is:

1. Printed directly in the PyCharm Run console or notebook output.
2. Appended to `artifacts/deal_alerts.jsonl` for programmatic use.
3. Displayed in `artifacts/deal_alerts.html` as a local dashboard.

## View alerts in a browser

After the agent has initialized, open `artifacts/deal_alerts.html` from PyCharm's
Project pane and choose **Open in Browser**. The page is local and refreshes every
30 seconds.

If the browser does not refresh a `file://` page reliably, start a local-only web
server from PyCharm's terminal:

```powershell
python -m http.server 8000 --directory artifacts --bind 127.0.0.1
```

Then visit <http://127.0.0.1:8000/deal_alerts.html>. Binding to `127.0.0.1` keeps
the page accessible only from this computer. Stop the server with `Ctrl+C`.

## Use it directly in the notebook

```python
from agents.messaging_agent import MessagingAgent

messenger = MessagingAgent()
messenger.notify(
    description="Example item",
    deal_price=79.99,
    estimated_true_value=129.99,
    url="https://example.com/item",
)
```

No API key or phone credential is needed. The existing `PlanningAgent` already
constructs `MessagingAgent`, so normal planning runs need no further changes.

## AI and RAG concepts covered

The project currently demonstrates these concepts:

- **Local LLM inference:** Qwen runs through Ollama, with an optional fine-tuned
  Qwen2.5 specialist adapter.
- **Vector embeddings and dense retrieval:** MiniLM query embeddings search the
  800,000-product Chroma HNSW collection. The current collection uses L2 distance.
- **BM25 and hybrid search:** a persistent SQLite FTS5 index provides lexical
  retrieval, which is combined with Chroma results using Reciprocal Rank Fusion.
- **Top-K retrieval and reranking:** each retriever returns 20 candidates; a
  cross-encoder reranks the fused set and selects five items for context.
- **Context injection:** the selected product descriptions and known prices are
  inserted into the local Qwen pricing prompt.
- **RAG evaluation:** Precision@K, Recall@K, reciprocal rank, and nDCG utilities
  are available for labelled retrieval examples.
- **LLM-as-judge:** an optional offline structured judge rates whether a retrieved
  item is a useful pricing comparable. It is not used as price ground truth.
- **Guardrails and hallucination reduction:** structured outputs, URL and price
  validation, untrusted-content delimiters, model-disagreement checks, and alert
  thresholds reduce unsafe or unsupported results.
- **Multi-agent workflow:** scanner, preprocessing, retrieval, specialist,
  neural-network, planning, and messaging agents cooperate in a fixed pipeline.

Some concepts have been discussed but are not yet implemented: human approval,
a deployed LLM gateway, metadata-filtered retrieval, context compression, query
expansion, multi-query and recursive retrieval, Self-RAG, Graph RAG, multimodal
RAG, and explicit cosine similarity. Chunking and chunk overlap are not currently
needed because each product is stored as a short, atomic document. Deployment
security is partial: the system is local-first and validates untrusted data, but
it does not yet provide authentication, TLS, RBAC, rate limiting, or audit logs.

Rebuild the BM25 sidecar after replacing or substantially updating the Chroma
product collection:

```powershell
.\.venv\Scripts\python.exe .\build_bm25_index.py
```

## Change the output folder (optional)

Set `PRICER_ALERT_DIR` before starting Python, or pass a directory explicitly:

```python
messenger = MessagingAgent(output_dir="my_local_alerts")
```
