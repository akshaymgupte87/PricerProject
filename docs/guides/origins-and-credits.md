# Origins and credits

## Course-derived material

The QLoRA learning notebooks were based on Ed Donner's course material and are retained as research artifacts:

- `Qlora_intro.ipynb`
- `Qlora_Training.ipynb`
- related notebook experiments following the same course workflow

These notebooks are not presented as wholly original work. Model, dataset, and library licenses remain governed by their respective owners.

## Original and extended engineering

The packaged application and its extensions include the `src/pricer` architecture, hybrid Chroma plus BM25 retrieval, agent workflows, LangGraph-compatible state handling, MCP interfaces, FastAPI and Gradio interfaces, persistence, human feedback, interchangeable Ollama/vLLM configuration, evaluation utilities, benchmark tooling, telemetry, deployment files, and CI.

Concrete extensions beyond the retained course notebook workflows are linked below. These describe repository implementation scope; they do not establish sole authorship of underlying algorithms, libraries, or course-derived pricing concepts.

| Extension | Implementation evidence | Outcome |
| --- | --- | --- |
| Installable application and configurable inference | [CLI](../../src/pricer/cli.py), [provider routing](../../src/pricer/agents/llm_provider.py), [package configuration](../../pyproject.toml) | Runs the pricing workflow through packaged commands with selectable providers. |
| Hybrid comparable-product retrieval | [Retrieval implementation](../../src/pricer/agents/retrieval.py) | Combines semantic and keyword retrieval; relevance improvement remains unmeasured. |
| Persistent review and recoverable workflow state | [Review store](../../src/pricer/persistence/intelligence_store.py), [state graph](../../src/pricer/agents/state_graph.py) | Records review decisions and workflow state for inspection and recovery. |
| Reproducible offline pricing evidence | [Benchmark implementation](../../src/pricer/jobs/pricing_evidence.py), [benchmark tests](../../tests/test_pricing_evidence.py), [method and limitations](../../benchmarks/evaluation/PRICING_RESULTS.md) | Fixed 5,000-product snapshot; 4,001 training and 999 test rows; $95.65 baseline MAE versus $66.71 for TF-IDF + Ridge. |

The new benchmark adds deterministic ID-based splitting, train-only feature fitting,
duplicate rejection, dataset hashing, environment/configuration capture, and saved
per-product predictions. These are project extensions; the measured improvement
belongs to this text-regression comparison and does not validate the course-derived
QLoRA workflow or the application's live ensemble.

The repository does not claim production scale, commercial deployment, or benchmark improvements that have not been measured.
