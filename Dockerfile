# Hugging Face Space (SDK docker) — assistente-master-iag
# Deploy: ver deploy/README.md (bash deploy/deploy.sh)
# v0.4: Python 3.14 + toolchain p/ compilar o llama-cpp-python (sdist, CPU).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# opencv/onnxruntime (rapidocr) e libs de audio precisam de libs de sistema;
# build-essential/cmake/git compilam o llama-cpp-python (Qwen2.5-1.5B local).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt deploy/requirements.txt
# --ignore-requires-python: rapidocr-onnxruntime==1.4.4 declara <3.13 no metadata,
# mas roda em 3.14 (validado pela suite pytest); os demais pins aceitam 3.14.
# CMAKE_BUILD_PARALLEL_LEVEL/MAX_JOBS=-j2: compilar o llama.cpp com paralelismo total
# estoura a memoria do build do Space (OOMKilled, exit 137).
# CMAKE_ARGS=-DGGML_NATIVE=OFF: sem -march=native do host (SIGILL/exit 132 no CPU do Space).
RUN pip install --upgrade pip \
 && CMAKE_ARGS="-DGGML_NATIVE=OFF" CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2 MAKEFLAGS=-j2 \
    pip install --ignore-requires-python -r deploy/requirements.txt

# codigo + entrypoint (o corpus vem do dataset privado no boot — nao esta no repo)
COPY . .
RUN chmod +x /app/deploy/entrypoint.sh

ENV HF_DATA_REPO=rafaelang/assistente-master-iag-dados

EXPOSE 7860

CMD ["bash", "/app/deploy/entrypoint.sh"]
