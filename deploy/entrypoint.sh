#!/usr/bin/env bash
#
# Entrypoint do container do Space: baixa o corpus/indices do DATASET PRIVADO
# (assistente-master-iag-dados) no primeiro boot e so entao sobe o uvicorn.
set -euo pipefail

DATA_REPO="${HF_DATA_REPO:-}"
if [[ -n "$DATA_REPO" && -n "${HF_TOKEN_READ:-}" && ! -f /app/data/raw/curso_indice.md ]]; then
  echo "[entrypoint] Baixando corpus/indices do dataset privado: ${DATA_REPO}"
  python - "$DATA_REPO" <<'PYEOF'
import os
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    token=os.environ["HF_TOKEN_READ"],
    local_dir="/app/data",
    allow_patterns=[
        "raw/*",
        "processed/rag/*",
        "processed/rag_v3/*",
        "processed/slm_models/*",
    ],
)
PYEOF
  echo "[entrypoint] dados prontos em /app/data:"
  ls /app/data/raw | head -5
else
  echo "[entrypoint] corpus ja presente (ou HF_DATA_REPO/HF_TOKEN_READ ausentes) — seguindo direto."
fi

exec uvicorn src.projeto_final.main:app --host 0.0.0.0 --port 7860
