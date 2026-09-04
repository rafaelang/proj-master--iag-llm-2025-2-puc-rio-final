#!/usr/bin/env bash
#
# Deploy do projeto para um Space na Hugging Face (API PUBLICO + corpus PRIVADO).
#
#   Space:   <usuario>/assistente-master-iag      -> PUBLICO (codigo + app)
#   Dataset: <usuario>/assistente-master-iag-dados -> PRIVADO (corpus data/raw
#            + indices data/processed/rag*), baixado em runtime no boot.
#   SDK: docker · Hardware: cpu-upgrade · Sleep: 1 h (3600 s)
#
# O repositorio publico do Space NAO contem o corpus: o deploy.sh sobe os dados
# para o dataset privado e o entrypoint do container faz snapshot_download no
# boot usando o secret HF_TOKEN_READ (token de leitura do dataset).
#
# Uso:
#   cp deploy/.env.example deploy/.env   # preencha HF_TOKEN e DEEPSEEK_API_KEY
#   bash deploy/deploy.sh                # deploy real
#   bash deploy/deploy.sh --dry-run      # so monta o bundle (sem publicar)
#
# Segredos ficam em deploy/.env (gitignored) e/ou como secrets do Space.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$RAIZ/deploy/.env"
BUILD_DIR="$RAIZ/deploy/build"
SPACE_NAME="assistente-master-iag"
DATASET_NAME="assistente-master-iag-dados"
HARDWARE="cpu-upgrade"
SLEEP_TIME="3600"
DRY_RUN="${1:-}"

# ---------------------------------------------------------------- segredos/env
if [[ -f "$ENV_FILE" ]]; then
  set -a; # shellcheck disable=SC1090
  source "$ENV_FILE"; set +a
fi
# fallback: chave DeepSeek costuma estar no .env da raiz do projeto
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "$RAIZ/.env" ]]; then
  set -a; # shellcheck disable=SC1090
  source "$RAIZ/.env"; set +a
fi
# token de leitura do dataset (se vazio, usa o proprio HF_TOKEN; preferir um read-only)
HF_TOKEN_READ="${HF_TOKEN_READ:-${HF_TOKEN:-}}"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  HF_TOKEN="${HF_TOKEN:-dry-run}"   # nao precisa do token real p/ montar o bundle
  DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
else
  : "${HF_TOKEN:?Defina HF_TOKEN no ambiente ou em deploy/.env}"
  : "${DEEPSEEK_API_KEY:?Defina DEEPSEEK_API_KEY no ambiente, em deploy/.env ou no .env da raiz}"
fi

if ! command -v python >/dev/null 2>&1 && [[ -x "$RAIZ/.venv/bin/python" ]]; then
  export PATH="$RAIZ/.venv/bin:$PATH"
fi
if ! command -v python >/dev/null 2>&1; then
  echo "ERRO: python nao encontrado (use o .venv do projeto ou ative o ambiente)." >&2
  exit 1
fi
PY="$(command -v python)"

# ---------------------------------------------------------------- 1. identidade
mask() { echo "${1//$HF_TOKEN/***}"; }
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  HF_USER="$("$PY" -c "
from huggingface_hub import HfApi
import sys
print(HfApi().whoami(token=sys.argv[1])['name'])
" "$HF_TOKEN")"
  SPACE_ID="$HF_USER/$SPACE_NAME"
  DATASET_ID="$HF_USER/$DATASET_NAME"
  SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"
  echo ">> Space PUBLICO: $SPACE_ID (hardware=$HARDWARE, sleep=${SLEEP_TIME}s)"
  echo ">> Dataset PRIVADO: $DATASET_ID"
else
  HF_USER="seu-usuario"
  SPACE_ID="$HF_USER/$SPACE_NAME"
  DATASET_ID="$HF_USER/$DATASET_NAME"
  SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"
  echo ">> DRY-RUN: so monta o bundle em $BUILD_DIR (nao publica)"
fi

# ---------------------------------------------------------------- 2. dataset privado
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  :
elif ! "$PY" - "$HF_TOKEN" "$DATASET_ID" <<'PYEOF'
from huggingface_hub import HfApi
import sys

api = HfApi(token=sys.argv[1])
api.create_repo(repo_id=sys.argv[2], repo_type="dataset", private=True, exist_ok=True)
print("   dataset privado pronto:", sys.argv[2])
PYEOF
then
  exit 1
fi

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  :
else
  echo ">> Enviando corpus e indices para o dataset privado (sem caches de modelo)..."
  "$PY" - "$HF_TOKEN" "$DATASET_ID" "$RAIZ" <<'PYEOF'
from huggingface_hub import HfApi
import sys

api = HfApi(token=sys.argv[1])
dataset_id = sys.argv[2]
raiz = sys.argv[3]
ignora = ["*_models/*", "*_models", "*.log"]
api.upload_folder(
    repo_id=dataset_id, repo_type="dataset",
    folder_path=f"{raiz}/data/raw", path_in_repo="raw",
)
api.upload_folder(
    repo_id=dataset_id, repo_type="dataset",
    folder_path=f"{raiz}/data/processed/rag", path_in_repo="processed/rag",
    ignore_patterns=ignora,
)
api.upload_folder(
    repo_id=dataset_id, repo_type="dataset",
    folder_path=f"{raiz}/data/processed/rag_v3", path_in_repo="processed/rag_v3",
    ignore_patterns=ignora,
)
# v0.4 - SLM local (GGUF ~1,1 GB) vai para o dataset privado (nao para o Space)
api.upload_folder(
    repo_id=dataset_id, repo_type="dataset",
    folder_path=f"{raiz}/data/processed/slm_models", path_in_repo="processed/slm_models",
    ignore_patterns=["*.log"],
)
print("   dataset atualizado com corpus (raw) + indices (processed/rag*, rag_v3, slm_models)")
PYEOF
fi

# ---------------------------------------------------------------- 3. cria/atualiza space + publico + secrets
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  echo ">> Criando/atualizando o Space (docker) e tornando PUBLICO..."
  "$PY" - "$HF_TOKEN" "$SPACE_ID" "$HARDWARE" "$SLEEP_TIME" "$HF_TOKEN_READ" "$DATASET_ID" <<'PYEOF'
from huggingface_hub import HfApi
import os
import sys

token = sys.argv[1]
repo_id = sys.argv[2]
hardware, sleep_time = sys.argv[3], int(sys.argv[4])
hf_read, dataset_id = sys.argv[5], sys.argv[6]
api = HfApi(token=token)
api.create_repo(
    repo_id=repo_id, repo_type="space", private=True,
    space_sdk="docker", space_hardware=hardware, space_sleep_time=sleep_time, exist_ok=True,
)
api.update_repo_settings(repo_id=repo_id, repo_type="space", private=False, token=token)
print("   space publico pronto:", repo_id)

for key, value in (("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
                   ("HF_TOKEN_READ", hf_read),
                   ("HF_DATA_REPO", dataset_id)):
    if not value:
        continue
    try:
        api.add_space_secret(repo_id=repo_id, key=key, value=value, token=token)
    except Exception:
        api.delete_space_secret(repo_id=repo_id, key=key, token=token)
        api.add_space_secret(repo_id=repo_id, key=key, value=value, token=token)
print("   secrets: DEEPSEEK_API_KEY / HF_TOKEN_READ / HF_DATA_REPO definidos")
PYEOF
fi

# ---------------------------------------------------------------- 4. bundle (sem corpus)
echo ">> Montando bundle em $BUILD_DIR (sem data/ — corpus fica no dataset privado)..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude '.env' \
  --exclude '*.ses' \
  --exclude 'deploy/build' \
  --exclude 'deploy/.env' \
  --exclude 'data' \
  --exclude '*.log' \
  "$RAIZ/" "$BUILD_DIR/"

echo ">> Tamanho do bundle:"
du -sh "$BUILD_DIR"

# Repo card do Space: metadata YAML exigida pelo HF no README.md do bundle
# (nao altera o README do repositorio principal; so a copia do bundle).
"$PY" - "$BUILD_DIR" <<'PYEOF'
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "README.md"
texto = p.read_text(encoding="utf-8")
if not texto.startswith("---"):
    card = (
        "---\n"
        "title: assistente-master-iag\n"
        "emoji: 🎓\n"
        "colorFrom: indigo\n"
        "colorTo: blue\n"
        "sdk: docker\n"
        "app_port: 7860\n"
        "pinned: false\n"
        "---\n\n"
    )
    p.write_text(card + texto, encoding="utf-8")
    print("   metadata YAML adicionada ao README.md do bundle")
else:
    print("   metadata YAML ja presente no README.md do bundle")
PYEOF

# ---------------------------------------------------------------- 5. git + push
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo ">> DRY-RUN concluido. Para publicar: bash deploy/deploy.sh"
  exit 0
fi

cd "$BUILD_DIR"
git init -b main >/dev/null
git config user.name "deploy"
git config user.email "deploy@users.noreply.huggingface.co"
git add -A
git commit -q -m "deploy v0.4 - Space publico (API + agentes) + dataset privado (corpus + SLM)"
git remote add origin "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}"
git fetch --quiet origin main || true
git push --force origin main

echo ""
echo ">> Deploy enviado com sucesso!"
echo ">> Space (publico): $(mask "$SPACE_URL")"
echo ">> Dataset (privado): https://huggingface.co/datasets/$(mask "$DATASET_ID")"
echo ">> Acompanhe o build:  $(mask "$SPACE_URL")/settings"
echo ">> Quando 'Running', abra: $(mask "$SPACE_URL")"
