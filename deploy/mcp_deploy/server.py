"""Servidor MCP para o ``deploy.sh`` do assistente (publicação na Hugging Face).

Expõe o script ``deploy/deploy.sh`` como tools do Model Context Protocol,
usando FastMCP sobre **stdio** (transporte padrão do FastMCP —
``mcp.run()``). Nenhum servidor HTTP é aberto.

Uso (a partir deste diretório)::

    uv sync                      # instala fastmcp
    uv run python server.py      # sobe o servidor MCP em stdio

Pré-requisito: ``deploy/.env`` com ``HF_TOKEN`` (e opcionalmente
``DEEPSEEK_API_KEY`` / ``HF_TOKEN_READ``) — ver ``deploy/README.md``.
O token de escrita NUNCA é aceito como argumento de tool: o script lê o
arquivo ``deploy/.env`` e mascara o token na saída (função ``mask``).

Segurança de ferramenta: ``deploy_publish`` é a ÚNICA tool com efeito
externo (publica Space público + dataset privado) e exige confirmação
explícita via argumento. ``deploy_dry_run`` e ``deploy_status`` são
read-only — usáveis por qualquer agente (aluno ou orientador) sem
risco de publicar.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("deploy")

# --------------------------------------------------------------------------- #
# caminhos do projeto (deploy/mcp_deploy/ -> raiz do projeto_final)
# --------------------------------------------------------------------------- #
_RAIZ = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _RAIZ / "deploy" / "deploy.sh"
_ENV_FILE = _RAIZ / "deploy" / ".env"
_ENV_RAIZ = _RAIZ / ".env"
_BUILD_DIR = _RAIZ / "deploy" / "build"

#: Tempo limite (s): upload do corpus + SLM (~1 GB) ao dataset e push do Space
#: podem demorar vários minutos.
_PROC_TIMEOUT = 1800


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run_deploy(*args: str) -> str:
    """Executa ``bash deploy/deploy.sh <args>`` e devolve a saída como texto.

    Levanta ``RuntimeError`` com o stderr quando o script falha, e uma
    mensagem clara quando o script não existe.
    """
    if not _SCRIPT.exists():
        raise RuntimeError(
            f"Script de deploy não encontrado: {_SCRIPT}. Rode a partir do "
            "repositório projeto_final (deploy/mcp_deploy é uma tool de dev)."
        )
    try:
        proc = subprocess.run(
            ["bash", str(_SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=_PROC_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - difícil testar
        raise TimeoutError(
            f"`bash {_SCRIPT.name} {' '.join(args)}` excedeu o limite de {_PROC_TIMEOUT}s."
        ) from exc
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{_SCRIPT.name} {' '.join(args)}` falhou (rc={proc.returncode}):\n"
            f"{stderr or stdout or 'sem saída do script'}"
        )
    return stdout or stderr


def _chaves_env() -> list[str]:
    """Nomes (sem valores!) das variáveis definidas em deploy/.env."""
    if not _ENV_FILE.exists():
        return []
    chaves: list[str] = []
    for linha in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#") and "=" in linha:
            chaves.append(linha.split("=", 1)[0])
    return sorted(chaves)


def _info_build() -> str:
    """Estado do bundle local (último dry-run/publicação)."""
    if not _BUILD_DIR.exists():
        return "ausente (rode deploy_dry_run para montar)"
    n_arquivos = sum(1 for _ in _BUILD_DIR.rglob("*") if _.is_file())
    mtime = datetime.fromtimestamp(_BUILD_DIR.stat().st_mtime)
    return f"{n_arquivos} arquivos · modificado em {mtime:%Y-%m-%d %H:%M}"


# --------------------------------------------------------------------------- #
# tools (leitura)
# --------------------------------------------------------------------------- #
@mcp.tool
def deploy_status() -> str:
    """Inspeção local read-only do estado do deploy (sem chamar a Hugging Face).

    Reporta: raiz do projeto, existência do script de deploy, presença de
    deploy/.env (apenas os NOMES das variáveis definidas — nunca os valores)
    e estado do bundle deploy/build/.
    """
    estado = [
        f"raiz: {_RAIZ}",
        f"script: {'OK' if _SCRIPT.exists() else 'AUSENTE'} ({_SCRIPT})",
    ]
    if _ENV_FILE.exists():
        chaves = _chaves_env()
        estado.append(f"deploy/.env: presente ({', '.join(chaves) or 'sem variáveis'})")
    else:
        estado.append("deploy/.env: AUSENTE (deploy_publish exige HF_TOKEN — ver deploy/README.md)")
    estado.append(f"bundle build/: {_info_build()}")
    return "\n".join(estado)


@mcp.tool
def deploy_dry_run() -> str:
    """Monta o bundle em deploy/build/ SEM publicar (bash deploy.sh --dry-run).

    Read-only: não toca a Hugging Face nem exige token real. Use antes de
    qualquer publicação para revisar o conteúdo do bundle (código sem data/).
    """
    return _run_deploy("--dry-run")


# --------------------------------------------------------------------------- #
# tools (escrita — com confirmação explícita)
# --------------------------------------------------------------------------- #
@mcp.tool
def deploy_publish(confirm: str = "") -> str:
    """PUBLICA o assistente na Hugging Face (Space público + dataset privado).

    Única tool com efeito externo: envia data/raw + índices para o dataset
    privado, cria/atualiza o Space público (docker, cpu-upgrade) e faz push
    do bundle. Exige confirmação explícita e deploy/.env com HF_TOKEN.

    Args:
        confirm: deve ser exatamente "publicar".
    """
    if confirm != "publicar":
        raise ValueError(
            'Confirmação inválida — passe confirm="publicar" para publicar de '
            "fato (ou use deploy_dry_run para apenas montar o bundle)."
        )
    if not (_ENV_FILE.exists() or _ENV_RAIZ.exists()):
        raise RuntimeError(
            "deploy/.env (ou .env da raiz) não encontrado. Defina HF_TOKEN e "
            "DEEPSEEK_API_KEY antes de publicar (ver deploy/README.md)."
        )
    return _run_deploy()


# --------------------------------------------------------------------------- #
# entrada
# --------------------------------------------------------------------------- #
def main() -> None:
    """Sobe o servidor MCP no transporte padrão (stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()