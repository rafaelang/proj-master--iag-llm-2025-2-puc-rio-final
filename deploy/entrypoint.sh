#!/usr/bin/env bash
#
# Entrypoint do container do Space: baixa o corpus/indices do DATASET PRIVADO
# (assistente-master-iag-dados) no primeiro boot e so entao sobe o uvicorn.
#
# v1.0: o snapshot_download baixa ~240 arquivos (~2,5 GB) e o tqdm de progresso
# escrevia no stderr do container; em ambiente headless o HF fecha esse pipe e o
# processo morria com SIGPIPE (exit 141). Desabilita a barra de progresso e faz
# retry com backoff para o boot ser deterministico.
set -euo pipefail

# sem barra de progresso do huggingface_hub (evita SIGPIPE no container headless)
export HF_HUB_DISABLE_PROGRESS_BARS=1

DATA_REPO="${HF_DATA_REPO:-}"
if [[ -n "$DATA_REPO" && -n "${HF_TOKEN_READ:-}" && ! -f /app/data/raw/curso_indice.md ]]; then
  echo "[entrypoint] Baixando corpus/indices do dataset privado: ${DATA_REPO}"
  python - "$DATA_REPO" <<'PYEOF'
import os
import sys
import time

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

repo_id = sys.argv[1]
padroes = ["raw/*", "processed/rag/*", "processed/rag_v3/*", "processed/slm_models/*"]

ultimo_erro = None
for tentativa in range(1, 4):  # 3 tentativas
    try:
        with disable_progress_bars():
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                token=os.environ["HF_TOKEN_READ"],
                local_dir="/app/data",
                allow_patterns=padroes,
            )
        ultimo_erro = None
        break
    except Exception as e:  # noqa: BLE001 - qualquer falha justifica retry
        ultimo_erro = e
        print(f"[entrypoint] tentativa {tentativa}/3 falhou: {e}", flush=True)
        time.sleep(10 * tentativa)

if ultimo_erro is not None:
    raise SystemExit(f"[entrypoint] download do dataset falhou apos 3 tentativas: {ultimo_erro}")

print("[entrypoint] dados prontos em /app/data.", flush=True)
PYEOF
  echo "[entrypoint] dados prontos em /app/data:"
  ls /app/data/raw | wc -l
else
  echo "[entrypoint] corpus ja presente (ou HF_DATA_REPO/HF_TOKEN_READ ausentes) — seguindo direto."
fi

exec uvicorn src.projeto_final.main:app --host 0.0.0.0 --port 7860
