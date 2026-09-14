"""Testes do servidor MCP do deploy.sh (deploy/mcp_deploy).

Nenhum `deploy.sh` real é executado: o `subprocess.run` é mockado e o caminho
do script aponta para arquivos temporários (tmp_path).
"""

import asyncio
from pathlib import Path

import pytest

import server


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def script_fake(tmp_path) -> str:
    """Cria um deploy.sh fake apontado pelo servidor sob teste."""
    script = tmp_path / "deploy.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return str(script)


@pytest.fixture
def deploy_mock(monkeypatch, script_fake):
    """Substitui `subprocess.run` e aponta o script/env/build para tmp."""
    holder: dict = {"calls": [], "result": _FakeProc(stdout="ok")}

    def fake_run(cmd, **kwargs):
        holder["calls"].append({"cmd": list(cmd), **kwargs})
        return holder["result"]

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server, "_SCRIPT", Path(script_fake))
    monkeypatch.setattr(server, "_ENV_FILE", Path(script_fake).parent / "deploy.env")
    monkeypatch.setattr(server, "_ENV_RAIZ", Path(script_fake).parent / "raiz.env")
    monkeypatch.setattr(server, "_BUILD_DIR", Path(script_fake).parent / "build")
    return holder


def _last_call(holder) -> tuple[list[str], dict]:
    call = holder["calls"][-1]
    return call["cmd"], {k: v for k, v in call.items() if k != "cmd"}


# --------------------------------------------------------------------------- #
# _run_deploy
# --------------------------------------------------------------------------- #
def test_script_ausente_levanta_erro_claro(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_SCRIPT", tmp_path / "nao_existe.sh")
    with pytest.raises(RuntimeError, match="Script de deploy não encontrado"):
        server.deploy_dry_run()


def test_erro_do_script_propaga_stderr(monkeypatch, script_fake):
    monkeypatch.setattr(server, "_SCRIPT", Path(script_fake))
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *_a, **_k: _FakeProc(returncode=1, stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        server.deploy_dry_run()


def test_saida_limpa_em_sucesso(deploy_mock):
    deploy_mock["result"] = _FakeProc(stdout=">> bundle ok\n")
    assert server.deploy_dry_run() == ">> bundle ok"


# --------------------------------------------------------------------------- #
# deploy_dry_run
# --------------------------------------------------------------------------- #
def test_dry_run_monta_comando_com_flag(deploy_mock):
    server.deploy_dry_run()
    cmd, _ = _last_call(deploy_mock)
    assert cmd == ["bash", str(server._SCRIPT), "--dry-run"]


# --------------------------------------------------------------------------- #
# deploy_publish
# --------------------------------------------------------------------------- #
def test_publish_sem_confirmacao_recusa(deploy_mock):
    with pytest.raises(ValueError, match="publicar"):
        server.deploy_publish()
    assert deploy_mock["calls"] == []


def test_publish_confirmacao_invalida_recusa(deploy_mock):
    with pytest.raises(ValueError, match="publicar"):
        server.deploy_publish(confirm="deploy agora")
    assert deploy_mock["calls"] == []


def test_publish_sem_env_recusa(deploy_mock):
    with pytest.raises(RuntimeError, match="deploy/.env"):
        server.deploy_publish(confirm="publicar")
    assert deploy_mock["calls"] == []


def test_publish_com_env_e_confirmacao_chama_script(deploy_mock):
    server._ENV_FILE.write_text("HF_TOKEN=abc\n", encoding="utf-8")
    deploy_mock["result"] = _FakeProc(stdout=">> Deploy enviado com sucesso!")
    saida = server.deploy_publish(confirm="publicar")
    assert "sucesso" in saida
    cmd, _ = _last_call(deploy_mock)
    assert cmd == ["bash", str(server._SCRIPT)]  # sem flag --dry-run


# --------------------------------------------------------------------------- #
# deploy_status
# --------------------------------------------------------------------------- #
def test_status_env_ausente_e_build_ausente(deploy_mock, tmp_path):
    # deploy_mock já aponta env/build para tmp inexistentes
    saida = server.deploy_status()
    assert "deploy/.env: AUSENTE" in saida
    assert "build/: ausente" in saida


def test_status_reporta_nomes_das_variaveis_sem_valores(deploy_mock, tmp_path):
    server._ENV_FILE.write_text(
        "# comentario\nHF_TOKEN=segredo123\nDEEPSEEK_API_KEY=x\n", encoding="utf-8"
    )
    saida = server.deploy_status()
    assert "HF_TOKEN" in saida and "DEEPSEEK_API_KEY" in saida
    assert "segredo123" not in saida  # valores nunca aparecem


def test_status_build_montado(deploy_mock, tmp_path):
    build = server._BUILD_DIR / "app"
    build.mkdir(parents=True)
    (build / "main.py").write_text("x", encoding="utf-8")
    saida = server.deploy_status()
    assert "build/" in saida and "arquivos" in saida


# --------------------------------------------------------------------------- #
# registro das tools no FastMCP
# --------------------------------------------------------------------------- #
def test_tools_registradas_no_mcp():
    tools = asyncio.run(server.mcp.list_tools())
    nomes = {t.name for t in tools}
    assert {"deploy_status", "deploy_dry_run", "deploy_publish"} <= nomes


def test_modulo_importa_e_tem_main():
    assert callable(server.main)
    assert callable(server.mcp.run)