# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

FROM base AS build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[all]"

FROM base AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
# embedding and reranker models are cached here; mount a volume to keep them between runs
ENV FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed CBRAG_DATA_DIR=/data
USER app
VOLUME ["/data", "/home/app/.cache"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["cbrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
