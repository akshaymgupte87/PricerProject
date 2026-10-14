# Project architecture

The installable application lives in src/pricer. Install with uv sync --locked.

| Package | Responsibility |
|---|---|
| interfaces | FastAPI, Gradio, MCP and presentation helpers |
| application | Deal workflow orchestration and review |
| agents | Scanning, pricing, guardrails, retrieval and state graph |
| inference | Lazy local Qwen adapter loading and price inference |
| persistence | SQLite intelligence/history/feedback storage |
| telemetry | Runtime measurements and drift analysis |
| data | Dataset parsing, loading and summary batches |
| evaluation | Shared prediction and retrieval evaluation |
| experiments | Reusable research experiments |
| jobs | Explicit BM25, neural, and preference-tuning commands |
| integrations/modal | Historical, opt-in Modal examples |
| benchmarks | Provider-agnostic inference benchmark tooling |
| LiteLLM Proxy | Central chat-model routing, keys, usage records, and request limits |

Interfaces call application services; services compose agents and persistence.
Research and training remain explicit jobs. Importing the top-level package or
asking for CLI help does not initialize models, databases, or network clients.
Keep future implementations inside these packages, not root scripts.

## Commands

    uv sync --locked
    uv run pricer api --host 127.0.0.1 --port 8000
    uv run pricer dashboard
    uv run pricer mcp
    uv run pricer scan
    uv run pricer build-index
    uv run pricer train --epochs 5
    uv run pricer dpo export --dataset artifacts/price_preferences.jsonl
    uv run pricer dpo train --dataset artifacts/price_preferences.jsonl
    uv run python -m unittest discover -s tests -v
    uv build

Training/index arguments are forwarded to the existing job parsers.
The module entry point python -m pricer is equivalent to the installed command.

Docker Compose routes application LLM requests through LiteLLM Proxy at
`litellm_config.yaml`, with Postgres for virtual keys and usage records.
Host-run commands can opt in with `PRICER_LLM_PROVIDER=gateway`; direct Ollama
and vLLM modes remain available. Compose defaults to one local upstream model
alias. Virtual-key RPM/TPM limits can be configured in the proxy; local model
costs may be zero unless a cost map is configured.

DPO examples come only from reviewed opportunities with a corrected fair price.
The feedback snapshot stores the product description and comparable-price
evidence; export uses the correction as the preferred answer and the saved model
estimate as the dispreferred answer. Only the latest corrected review per URL is
exported. Training is an explicit CUDA QLoRA job and writes a separate adapter;
it does not switch inference automatically.

## Configuration and writable state

PRICER_HOME selects the asset/state root. A source checkout defaults to the
repository root, preserving existing vector stores, models, and memory.
An installed wheel defaults to ~/.pricer. See .env.example for path overrides.
Relative overrides resolve under that root. Existing model and retrieval
environment variables remain supported. Environment variables take precedence
over .env; dotenv files must not override deployment configuration.

Containers set PRICER_HOME=/app. Compose persists memory and SQLite history under
the artifacts mount. Chroma must be writable because it also records seen deals.
Provision assets before live pricing.

## Installation and migration

There is one installable package: src/pricer. Root Python launchers, the agents
package, old flat pricer modules, and evaluator adapters have been removed.
All application code, tests, and notebook imports use the canonical packages.

Run uv sync --locked before execution. Use uv run pricer for CLI commands, or
python -m pricer inside the installed project environment. Select that same
environment as the notebook kernel. Do not run package files by filesystem path.

The single product Item model lives in pricer.data.items and accepts text or
integer identifiers. Evaluation lives in pricer.evaluation.evaluator; pass
workers=1 when a local model requires sequential execution.

Existing notebook outputs, datasets, model artifacts, and runtime state remain
in place. Old pickles referring to removed module names must be regenerated
from their source data; JSON/JSONL and state-dict checkpoints are unaffected.

## Scope

This migration changes organization, installation, configuration, and launching.
It does not claim model accuracy or introduce a distributed job queue.
Run one API worker while the process-local scan lock is in use.
Authentication retains the optional PRICER_API_KEY behavior; configure a key
before exposing the service. Dependency versions are preserved; separating
CPU/GPU and research dependencies is a separate migration.

The API exports create_app() and uses FastAPI dependency injection, allowing
isolated service overrides in tests. Health checks do not load ML models.

Optional platform integrations are intentionally lightweight. `GET
/metrics/prometheus` exposes operational metrics, `LANGCHAIN_TRACING_V2=true`
enables LangSmith tracing when the optional package and credentials are present,
and `pricer eval` checks measured metric artifacts against explicit thresholds.
