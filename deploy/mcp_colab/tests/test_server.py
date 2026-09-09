"""Testes do servidor MCP do google-colab-cli (deploy/mcp_colab).

Nenhum comando `colab` real é executado: o `subprocess.run` é mockado.
"""

import asyncio
import subprocess

import pytest

import server


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def colab_mock(monkeypatch):
    """Substitui `subprocess.run`/`shutil.which` e registra as chamadas."""
    holder: dict = {"calls": [], "result": _FakeProc(stdout="ok")}

    def fake_run(cmd, **kwargs):
        holder["calls"].append({"cmd": list(cmd), **kwargs})
        return holder["result"]

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.shutil, "which", lambda _name: "/usr/bin/colab")
    return holder


def _last_call(holder) -> tuple[list[str], dict]:
    call = holder["calls"][-1]
    return call["cmd"], {k: v for k, v in call.items() if k != "cmd"}


# --------------------------------------------------------------------------- #
# _run_colab
# --------------------------------------------------------------------------- #
def test_binary_ausente_levanta_erro_claro(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="colab"):
        server.colab_version()


def test_erro_do_cli_propaga_stderr(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _name: "/usr/bin/colab")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *_a, **_k: _FakeProc(returncode=1, stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        server.colab_version()


def test_saida_limpa_em_sucesso(colab_mock):
    colab_mock["result"] = _FakeProc(stdout="0.9.0\n")
    assert server.colab_version() == "0.9.0"


# --------------------------------------------------------------------------- #
# sessão
# --------------------------------------------------------------------------- #
def test_new_session_monta_comandos(colab_mock):
    server.colab_new_session(session="trainer", gpu="t4")
    cmd, kwargs = _last_call(colab_mock)
    assert cmd == ["colab", "new", "-s", "trainer", "--gpu", "T4"]
    assert kwargs.get("input") is None


def test_new_session_rejeita_gpu_invalida(colab_mock):
    with pytest.raises(ValueError, match="GPU"):
        server.colab_new_session(gpu="RTX-5090")
    assert colab_mock["calls"] == []


def test_sessions_sem_flag_session(colab_mock):
    server.colab_list_sessions()
    cmd, _ = _last_call(colab_mock)
    assert cmd == ["colab", "sessions"]


def test_stop_sem_nome(colab_mock):
    server.colab_stop_session()
    cmd, _ = _last_call(colab_mock)
    assert cmd == ["colab", "stop"]


# --------------------------------------------------------------------------- #
# execução
# --------------------------------------------------------------------------- #
def test_exec_code_usa_stdin(colab_mock):
    server.colab_exec_code(code="print(1)", session="s1")
    cmd, kwargs = _last_call(colab_mock)
    assert cmd == ["colab", "exec", "-s", "s1"]
    assert kwargs.get("input") == "print(1)\n"


def test_exec_file_usa_flag_f(colab_mock):
    server.colab_exec_file(file="train.py", timeout=120.0)
    cmd, kwargs = _last_call(colab_mock)
    assert cmd == ["colab", "exec", "--timeout", "120.0", "-f", "train.py"]
    assert kwargs.get("input") is None


def test_run_script_fluxo_efemero(colab_mock):
    server.colab_run_script("train.py", script_args=["--epochs", "3"], gpu="A100")
    cmd, _ = _last_call(colab_mock)
    assert cmd == [
        "colab", "run", "--gpu", "A100", "train.py", "--epochs", "3",
    ]


# --------------------------------------------------------------------------- #
# arquivos
# --------------------------------------------------------------------------- #
def test_upload_ordem_flags_e_posicionais(colab_mock):
    server.colab_upload_file("a.csv", "/content/a.csv", session="s1")
    cmd, _ = _last_call(colab_mock)
    assert cmd == ["colab", "upload", "-s", "s1", "a.csv", "/content/a.csv"]


def test_download_ordem_remoto_local(colab_mock):
    server.colab_download_file("/content/m.bin", "./m.bin")
    cmd, _ = _last_call(colab_mock)
    assert cmd == ["colab", "download", "/content/m.bin", "./m.bin"]


# --------------------------------------------------------------------------- #
# utilitários
# --------------------------------------------------------------------------- #
def test_install_requer_pacotes_ou_requirements(colab_mock):
    with pytest.raises(ValueError, match="packages"):
        server.colab_install_packages(packages=[])
    assert colab_mock["calls"] == []


def test_install_pacotes(colab_mock):
    server.colab_install_packages(packages=["torch", "transformers"], session="s1")
    cmd, _ = _last_call(colab_mock)
    assert cmd == ["colab", "install", "-s", "s1", "torch", "transformers"]


# --------------------------------------------------------------------------- #
# registro das tools no FastMCP
# --------------------------------------------------------------------------- #
def test_tools_registradas_no_mcp():
    tools = asyncio.run(server.mcp.list_tools())
    nomes = {t.name for t in tools}
    esperadas = {
        "colab_new_session",
        "colab_list_sessions",
        "colab_session_status",
        "colab_restart_kernel",
        "colab_stop_session",
        "colab_session_url",
        "colab_exec_code",
        "colab_exec_file",
        "colab_run_script",
        "colab_list_files",
        "colab_upload_file",
        "colab_download_file",
        "colab_remove_file",
        "colab_whoami",
        "colab_mount_drive",
        "colab_install_packages",
        "colab_export_log",
        "colab_version",
    }
    assert esperadas <= nomes


def test_modulo_importa_e_tem_main():
    assert callable(server.main)
    assert callable(server.mcp.run)

