FROM python:3.14-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PRICER_HOME=/app
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home --uid 10001 pricer && mkdir -p /app/artifacts /app/products_vectorstore && chown -R pricer:pricer /app
USER pricer
EXPOSE 8000
CMD ["pricer", "api", "--host", "0.0.0.0", "--port", "8000"]
