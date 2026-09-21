"""Local Ollama price predictor used by offline evaluation."""

import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOCAL_QWEN_MODEL = "qwen3.6:latest"
LOCAL_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not LOCAL_OLLAMA_HOST.startswith(("http://", "https://")):
    LOCAL_OLLAMA_HOST = f"http://{LOCAL_OLLAMA_HOST}"

def build_price_estimation_messages(item):
    prompt = f"Estimate the price of this product. Respond with the price only, no explanation.\n\n{item.summary}"
    return [{"role": "user", "content": prompt}]

def call_local_qwen(messages):
    payload = {
        "model": LOCAL_QWEN_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 20, "seed": 42},
    }
    request = Request(
        f"{LOCAL_OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {LOCAL_OLLAMA_HOST}. Start Ollama first.") from exc
    return result["message"]["content"].strip()

def local_qwen_3_6_pricer(item):
    return call_local_qwen(build_price_estimation_messages(item))
