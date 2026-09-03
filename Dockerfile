# Hugging Face Space (SDK docker) — assistente-master-iag
# Deploy: ver deploy/README.md (bash deploy/deploy.sh)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# opencv/onnxruntime (rapidocr) e libs de audio precisam de libs de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --upgrade pip && pip install -r deploy/requirements.txt

# codigo + corpus (data/raw) + indices processados (sem caches de modelo, baixados em runtime)
COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.projeto_final.main:app", "--host", "0.0.0.0", "--port", "7860"]
