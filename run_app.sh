#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure Git for Windows utilities are available even when Bash is launched
# directly from PowerShell instead of through the Git Bash shortcut.
export PATH="/usr/bin:/bin:$PATH"

# Windows launcher for Git Bash. Run from the repository with: ./run_app.sh
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

APP_URL="${PRICER_APP_URL:-http://127.0.0.1:7860}"
OLLAMA_URL="${OLLAMA_API_BASE:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${PRICER_QWEN_MODEL:-qwen3.6:latest}"
OPEN_BROWSER=true

if [[ "${1:-}" == "--no-browser" ]]; then
  OPEN_BROWSER=false
elif [[ -n "${1:-}" ]]; then
  echo "Usage: ./run_app.sh [--no-browser]" >&2
  exit 2
fi

open_app() {
  if [[ "$OPEN_BROWSER" == true ]]; then
    powershell.exe -NoProfile -Command "Start-Process '$APP_URL'" >/dev/null 2>&1 || true
  fi
}

echo "[1/7] Checking required Windows tools..."
for tool in uv.exe ollama.exe curl.exe powershell.exe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required command: $tool" >&2
    echo "Install uv, Ollama, and Git for Windows, then run this script again." >&2
    exit 1
  fi
done

if curl.exe --silent --fail --max-time 2 "$APP_URL" >/dev/null 2>&1; then
  echo "The Pricer dashboard is already running at $APP_URL"
  open_app
  exit 0
fi

echo "[2/7] Synchronizing Python dependencies from uv.lock..."
uv.exe sync --frozen

PYTHON=".venv/Scripts/python.exe"
if [[ ! -x "$PYTHON" ]]; then
  echo "uv did not create $PYTHON" >&2
  exit 1
fi

mkdir -p artifacts

echo "[3/7] Checking Ollama..."
if ! curl.exe --silent --fail --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama in the background..."
  ollama.exe serve >artifacts/ollama.stdout.log 2>artifacts/ollama.stderr.log &
  OLLAMA_PID=$!
  OLLAMA_READY=false
  for _ in {1..60}; do
    if curl.exe --silent --fail --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
      OLLAMA_READY=true
      break
    fi
    sleep 1
  done
  if [[ "$OLLAMA_READY" != true ]]; then
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    echo "Ollama did not become ready. Check artifacts/ollama.stderr.log." >&2
    exit 1
  fi
fi

echo "[4/7] Checking Ollama model $OLLAMA_MODEL..."
if ! ollama.exe list | awk 'NR > 1 {print $1}' | grep -Fxq "$OLLAMA_MODEL"; then
  echo "Pulling $OLLAMA_MODEL. This can be a large download..."
  ollama.exe pull "$OLLAMA_MODEL"
fi

echo "[5/7] Checking the product vector store..."
if [[ ! -d "products_vectorstore" ]]; then
  cat >&2 <<'EOF'
Missing products_vectorstore/.
This generated third-party dataset is intentionally not stored in Git.
Follow AGENTIFY_PRICER_GUIDE.md to create or import it, then rerun this script.
EOF
  exit 1
fi

echo "[6/7] Checking the BM25 index and neural checkpoint..."
if [[ ! -f "products_bm25.sqlite3" ]]; then
  echo "Building the BM25 index..."
  "$PYTHON" build_bm25_index.py
fi

if [[ ! -f "artifacts/deep_neural_network.pth" ]]; then
  echo "Building the neural-network checkpoint. This may take a long time..."
  "$PYTHON" train_deep_neural_network.py --epochs 5
fi

echo "[7/7] Starting the complete Gradio application..."
echo "Dashboard: $APP_URL"
echo "The RSS scan starts when the page loads. Select a result to approve or reject it."

export PRICER_OPEN_BROWSER="$OPEN_BROWSER"
export LITELLM_LOCAL_MODEL_COST_MAP="True"
exec "$PYTHON" price_is_right.py
