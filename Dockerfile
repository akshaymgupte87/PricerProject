FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-install-project
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 10001 pricer && chown -R pricer:pricer /app
USER pricer

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
