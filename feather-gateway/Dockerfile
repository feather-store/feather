# Feather gateway: routes /v1/{namespace}/... to the shard that owns it.
# Build context: feather-gateway/   (no engine build needed — it only proxies)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway.py .
ENV FEATHER_SHARDS=""
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
