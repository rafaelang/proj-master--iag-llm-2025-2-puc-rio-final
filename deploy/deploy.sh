#!/usr/bin/env bash
#
# Deploy do projeto para um Space PRIVADO na Hugging Face.
#
#   Space: <usuario>/assistente-master-iag
#   SDK:   docker · Hardware: cpu-upgrade · Sleep: 1 h (3600 s)
#
# Opcao 1 (recomendada): Space privado + dados do corpus (data/raw + indices)
# embarcados no repositorio do Space via bundle + push de git. Modelos pesados
# (whisper/fastembed/piper/rerank) sao baixados em runtime no container.
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
api = HfApi()
print(api.whoami(token=sys.argv[1])['name'])
" "$HF_TOKEN")"
  SPACE_ID="$HF_USER/$SPACE_NAME"
  SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"
  echo ">> Space alvo: $SPACE_ID (hardware=$HARDWARE, sleep=${SLEEP_TIME}s, privado)"
else
  HF_USER="seu-usuario"
  SPACE_ID="$HF_USER/$SPACE_NAME"
  SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"
  echo ">> DRY-RUN: so monta o bundle em $BUILD_DIR (nao publica)"
fi

# ---------------------------------------------------------------- 2. cria space
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  echo ">> Criando/atualizando o Space (privado, docker)..."
  "$PY" - "$HF_TOKEN" "$SPACE_ID" "$HARDWARE" "$SLEEP_TIME" <<'PYEOF'
from huggingface_hub import HfApi
import sys

token, repo_id, hardware, sleep_time = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
api = HfApi()
api.create_repo(
    repo_id=repo_id,
    token=token,
    repo_type="space",
    private=True,
    space_sdk="docker",
    space_hardware=hardware,
    space_sleep_time=sleep_time,
    exist_ok=True,
)
print("   space pronto/atualizado:", repo_id)
PYEOF

  echo ">> Definindo secret DEEPSEEK_API_KEY no Space..."
  "$PY" - "$HF_TOKEN" "$SPACE_ID" "$DEEPSEEK_API_KEY" <<'PYEOF'
from huggingface_hub import HfApi
import sys

token, repo_id, value = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
try:
    api.add_space_secret(repo_id=repo_id, key="DEEPSEEK_API_KEY", value=value, token=token)
    print("   secret DEEPSEEK_API_KEY definido.")
except Exception as e:
    if "already exists" in str(e).lower():
        api.delete_space_secret(repo_id=repo_id, key="DEEPSEEK_API_KEY", token=token)
        api.add_space_secret(repo_id=repo_id, key="DEEPSEEK_API_KEY", value=value, token=token)
        print("   secret DEEPSEEK_API_KEY atualizado.")
    else:
        raise
PYEOF
fi

# ---------------------------------------------------------------- 3. bundle
echo ">> Montando bundle em $BUILD_DIR ..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# copia o repositorio (sem git/venv/caches), incluindo data/raw e os indices processados.
# Caches de modelos (*_models) ficam de fora — sao baixados em runtime no Space.
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
  --exclude '*_models' \
  --exclude '*.log' \
  "$RAIZ/" "$BUILD_DIR/"

echo ">> Tamanho do bundle:"
du -sh "$BUILD_DIR"

# Repo card do Space: metadata YAML exigida pelo HF no README.md do bundle
# (nao altera o README do repositorio principal; so a copia do bundle).
"$PY" - <<'PYEOF'
from pathlib import Path

p = Path("README.md")
texto = p.read_text(encoding="utf-8")
if not texto.startswith("---"):
    card = (
        "---\n"
        "title: assistente-master-iag\n"
        "emoji: book\n"
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

# ---------------------------------------------------------------- 4. git + push
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo ">> DRY-RUN concluido. Para publicar: bash deploy/deploy.sh"
  exit 0
fi

cd "$BUILD_DIR"
git init -b main >/dev/null
git config user.name "deploy"
git config user.email "deploy@users.noreply.huggingface.co"
git add -A
git commit -q -m "deploy v0.3 - Space privado (cpu-upgrade, sleep 1h)"
git remote add origin "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}"
# O HF inicializa o repo do Space com arquivos gerados (README etc.); como o
# bundle e a fonte unica da aplicacao, o push substitui o conteudo remoto.
git fetch --quiet origin main || true
git push --force origin main
echo ""
echo ">> Deploy enviado com sucesso!"
echo ">> Acompanhe o build:  $(mask "$SPACE_URL")/settings"
echo ">> Quando 'Running', abra: $(mask "$SPACE_URL")"
