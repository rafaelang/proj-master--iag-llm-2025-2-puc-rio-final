"""Servidor MCP para o ``google-colab-cli``.

Expõe os comandos *não interativos* do CLI ``colab`` (Google Colab) como tools
do Model Context Protocol, usando FastMCP sobre **stdio** (transporte padrão
do FastMCP — ``mcp.run()``).

Uso (a partir deste diretório)::

    uv sync                      # instala fastmcp + google-colab-cli
    uv run python server.py      # sobe o servidor MCP em stdio

Pré-requisito: autenticar uma vez no Google com ``uv run colab auth`` (o fluxo
é interativo e não faz sentido como tool MCP). Confira com ``uv run colab whoami``.

Comandos interativos/TTY do CLI (``repl``, ``console``, ``ssh``, ``edit``) não
são expostos: exigem terminal local.
"""

from __future__ import annotations

import shutil
import subprocess

from fastmcp import FastMCP

mcp = FastMCP("colab")

#: Tempo limite (s) do subprocess: provisionar VM/GPU + rodar código remoto demora.
_PROC_TIMEOUT = 1800

#: Aceleradores aceitos pelo CLI (ver `colab new --help` / `colab run --help`).
_GPUS = ("T4", "L4", "G4", "H100", "A100")
_TPUS = ("v5e1", "v6e1")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run_colab(*args: str, input_text: str | None = None) -> str:
    """Executa ``colab <args>`` e devolve a saída (stdout/stderr) como texto.

    Levanta ``RuntimeError`` com o stderr quando o CLI falha, e uma mensagem
    clara quando o binário ``colab`` não está no PATH do servidor.
    """
    if shutil.which("colab") is None:
        raise RuntimeError(
            "Binário 'colab' não encontrado no PATH. Execute `uv sync` no diretório "
            "deploy/mcp_colab (ou `uv tool install google-colab-cli`) e reinicie o servidor."
        )
    try:
        proc = subprocess.run(
            ["colab", *args],
            capture_output=True,
            text=True,
            input=input_text,
            timeout=_PROC_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - difícil testar
        raise TimeoutError(
            f"Comando `colab {' '.join(args)}` excedeu o limite de {_PROC_TIMEOUT}s."
        ) from exc
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"`colab {' '.join(args)}` falhou (rc={proc.returncode}):\n"
            f"{stderr or stdout or 'sem saída do CLI'}"
        )
    return stdout or stderr


def _check_gpu(gpu: str | None) -> None:
    if gpu and gpu.upper() not in _GPUS:
        raise ValueError(f"GPU inválida: {gpu!r}. Opções: {', '.join(_GPUS)}.")


def _check_tpu(tpu: str | None) -> None:
    if tpu and tpu not in _TPUS:
        raise ValueError(f"TPU inválida: {tpu!r}. Opções: {', '.join(_TPUS)}.")


# --------------------------------------------------------------------------- #
# sessão
# --------------------------------------------------------------------------- #
@mcp.tool
def colab_new_session(
    session: str | None = None,
    gpu: str | None = None,
    tpu: str | None = None,
) -> str:
    """Provisiona uma nova VM/sessão no Google Colab (CPU por padrão).

    Args:
        session: Nome da sessão (reutilize depois com `-s`/`session=`).
        gpu: Acelerador GPU — T4, L4, G4, H100 ou A100 (disponibilidade varia
            com o plano Colab).
        tpu: Acelerador TPU — v5e1 ou v6e1.
    """
    _check_gpu(gpu)
    _check_tpu(tpu)
    cmd = ["new"]
    if session:
        cmd += ["-s", session]
    if gpu:
        cmd += ["--gpu", gpu.upper()]
    if tpu:
        cmd += ["--tpu", tpu]
    return _run_colab(*cmd)


@mcp.tool
def colab_list_sessions() -> str:
    """Lista as sessões ativas no backend do Google Colab."""
    return _run_colab("sessions")


@mcp.tool
def colab_session_status(session: str | None = None) -> str:
    """Mostra hardware, tipo de máquina e estado de uma sessão ativa."""
    cmd = ["status"]
    if session:
        cmd += ["-s", session]
    return _run_colab(*cmd)


@mcp.tool
def colab_restart_kernel(session: str | None = None) -> str:
    """Reinicia o kernel Jupyter da sessão ativa."""
    cmd = ["restart-kernel"]
    if session:
        cmd += ["-s", session]
    return _run_colab(*cmd)


@mcp.tool
def colab_stop_session(session: str | None = None) -> str:
    """Encerra a VM/sessão e libera o recurso (GPU/TPU/CPU)."""
    cmd = ["stop"]
    if session:
        cmd += ["-s", session]
    return _run_colab(*cmd)


@mcp.tool
def colab_session_url(session: str | None = None, open_url: bool = False) -> str:
    """Imprime a URL do navegador que conecta à sessão ativa.

    Args:
        session: Nome da sessão.
        open_url: Se True, abre a URL no navegador local (evite em headless).
    """
    cmd = ["url"]
    if session:
        cmd += ["-s", session]
    if open_url:
        cmd += ["--open"]
    return _run_colab(*cmd)


# --------------------------------------------------------------------------- #
# execução de código
# --------------------------------------------------------------------------- #
@mcp.tool
def colab_exec_code(
    code: str,
    session: str | None = None,
    output_image: str | None = None,
    timeout: float | None = None,
) -> str:
    """Executa código Python remoto na VM do Colab (enviado via stdin).

    Use para snippets. Para scripts/notebooks locais use `colab_exec_file`.

    Args:
        code: Código Python a executar na VM (multilinha suportado).
        session: Nome da sessão (omitir quando houver só uma).
        output_image: Caminho remoto p/ salvar a última figura (matplotlib etc).
        timeout: Tempo limite (s) da execução remota; padrão do CLI é 30 s.
    """
    cmd = ["exec"]
    if session:
        cmd += ["-s", session]
    if output_image:
        cmd += ["--output-image", output_image]
    if timeout:
        cmd += ["--timeout", str(timeout)]
    return _run_colab(*cmd, input_text=code if code.endswith("\n") else f"{code}\n")


@mcp.tool
def colab_exec_file(
    file: str,
    session: str | None = None,
    output_image: str | None = None,
    timeout: float | None = None,
) -> str:
    """Executa um arquivo local (.py ou .ipynb) na VM do Colab.

    O CLI lê o arquivo local e transmite o conteúdo ao kernel remoto — não é
    preciso fazer upload antes.

    Args:
        file: Caminho local do script .py ou notebook .ipynb.
        session: Nome da sessão.
        output_image: Caminho remoto p/ salvar a última figura.
        timeout: Tempo limite (s) da execução remota; padrão do CLI é 30 s.
    """
    cmd = ["exec"]
    if session:
        cmd += ["-s", session]
    if output_image:
        cmd += ["--output-image", output_image]
    if timeout:
        cmd += ["--timeout", str(timeout)]
    cmd += ["-f", file]
    return _run_colab(*cmd)


@mcp.tool
def colab_run_script(
    script: str,
    script_args: list[str] | None = None,
    session: str | None = None,
    gpu: str | None = None,
    tpu: str | None = None,
    keep: bool = False,
    timeout: float | None = None,
) -> str:
    """Roda um script local numa VM nova e efêmera e a libera ao terminar.

    O CLI provisiona a VM, executa o script (com argumentos) e derruba a VM.
    Use `keep=True` + `session=` para preservar e anexar depois.

    Args:
        script: Caminho local do script Python a executar.
        script_args: Argumentos repassados ao script (sys.argv).
        session: Nome da sessão efêmera (útil com keep=True).
        gpu: Acelerador GPU — T4, L4, G4, H100 ou A100.
        tpu: Acelerador TPU — v5e1 ou v6e1.
        keep: Se True, não derruba a VM ao terminar.
        timeout: Tempo limite (s) da execução remota; padrão do CLI é 30 s.
    """
    _check_gpu(gpu)
    _check_tpu(tpu)
    cmd = ["run"]
    if session:
        cmd += ["-s", session]
    if gpu:
        cmd += ["--gpu", gpu.upper()]
    if tpu:
        cmd += ["--tpu", tpu]
    if keep:
        cmd += ["--keep"]
    if timeout:
        cmd += ["--timeout", str(timeout)]
    cmd += [script, *(script_args or [])]
    return _run_colab(*cmd)


# --------------------------------------------------------------------------- #
# arquivos remotos
# --------------------------------------------------------------------------- #
@mcp.tool
def colab_list_files(path: str | None = None, session: str | None = None) -> str:
    """Lista arquivos/diretórios do sistema de arquivos da VM remota.

    Args:
        path: Caminho remoto a listar (padrão do CLI: /content).
        session: Nome da sessão.
    """
    cmd = ["ls"]
    if session:
        cmd += ["-s", session]
    if path:
        cmd += [path]
    return _run_colab(*cmd)


@mcp.tool
def colab_upload_file(
    local_path: str, remote_path: str, session: str | None = None
) -> str:
    """Envia um arquivo local para a VM remota (via Jupyter Contents API).

    Args:
        local_path: Caminho local do arquivo a enviar.
        remote_path: Caminho remoto de destino (ex.: /content/dados.csv).
        session: Nome da sessão.
    """
    cmd = ["upload"]
    if session:
        cmd += ["-s", session]
    cmd += [local_path, remote_path]
    return _run_colab(*cmd)


@mcp.tool
def colab_download_file(
    remote_path: str, local_path: str, session: str | None = None
) -> str:
    """Baixa um arquivo da VM remota para o disco local.

    Args:
        remote_path: Caminho remoto do arquivo (ex.: /content/modelo.bin).
        local_path: Caminho local onde salvar.
        session: Nome da sessão.
    """
    cmd = ["download"]
    if session:
        cmd += ["-s", session]
    cmd += [remote_path, local_path]
    return _run_colab(*cmd)


@mcp.tool
def colab_remove_file(path: str, session: str | None = None) -> str:
    """Remove um arquivo remoto na VM do Colab.

    Args:
        path: Caminho remoto a remover.
        session: Nome da sessão.
    """
    cmd = ["rm"]
    if session:
        cmd += ["-s", session]
    cmd += [path]
    return _run_colab(*cmd)


# --------------------------------------------------------------------------- #
# autenticação, drive, pacotes e logs
# --------------------------------------------------------------------------- #
@mcp.tool
def colab_whoami() -> str:
    """Mostra o e-mail/conta Google autenticada (e os escopos vigentes).

    Útil para conferir se a autenticação do CLI já foi feita (`colab auth` é
    interativo e deve ser executado uma vez fora do MCP).
    """
    return _run_colab("whoami")


@mcp.tool
def colab_mount_drive(session: str | None = None, path: str | None = None) -> str:
    """Monta o Google Drive na VM remota (padrão: /content/drive).

    Pode exigir autorização interativa na primeira vez.

    Args:
        session: Nome da sessão.
        path: Caminho remoto de montagem (padrão do CLI: /content/drive).
    """
    cmd = ["drivemount"]
    if session:
        cmd += ["-s", session]
    if path:
        cmd += [path]
    return _run_colab(*cmd)


@mcp.tool
def colab_install_packages(
    packages: list[str],
    session: str | None = None,
    requirements: str | None = None,
) -> str:
    """Instala pacotes Python na VM remota (gerenciador `uv` do CLI).

    Args:
        packages: Nomes dos pacotes (ex.: ["torch", "transformers"]).
        session: Nome da sessão.
        requirements: Caminho remoto de um requirements.txt (usa -r).
    """
    if not packages and not requirements:
        raise ValueError("Informe `packages` e/ou `requirements`.")
    cmd = ["install"]
    if session:
        cmd += ["-s", session]
    if requirements:
        cmd += ["-r", requirements]
    cmd += list(packages)
    return _run_colab(*cmd)


@mcp.tool
def colab_export_log(
    output: str | None = None,
    session: str | None = None,
    lines: int | None = None,
    event_type: str | None = None,
) -> str:
    """Exporta/mostra o histórico de execução da sessão.

    Args:
        output: Caminho local do arquivo de saída; o sufixo define o formato
            (.ipynb, .md, .txt ou .jsonl). Sem output, lista/printa o log.
        session: Nome da sessão (omitir lista sessões com log).
        lines: Número de linhas a mostrar/exportar (padrão: todas).
        event_type: Filtro por tipo de evento (ex.: execution, file_operation).
    """
    cmd = ["log"]
    if session:
        cmd += ["-s", session]
    if lines is not None:
        cmd += ["-n", str(lines)]
    if event_type:
        cmd += ["-t", event_type]
    if output:
        cmd += ["-o", output]
    return _run_colab(*cmd)


@mcp.tool
def colab_version() -> str:
    """Mostra a versão instalada do google-colab-cli."""
    return _run_colab("version")


# --------------------------------------------------------------------------- #
# entrada
# --------------------------------------------------------------------------- #
def main() -> None:
    """Sobe o servidor MCP no transporte padrão (stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()



