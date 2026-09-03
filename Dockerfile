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

# codigo + entrypoint (o corpus vem do dataset privado no boot — nao esta no repo)
COPY . .
RUN chmod +x /app/deploy/entrypoint.sh

ENV HF_DATA_REPO=rafaelang/assistente-master-iag-dados

EXPOSE 7860

CMD ["bash", "/app/deploy/entrypoint.sh"]
