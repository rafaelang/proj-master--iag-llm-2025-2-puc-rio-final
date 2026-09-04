"""Testes da v0.4 — fluxo multiagente (roteador + geradores + fallback).

Todos os cenarios usam mock (sem API, sem LLM, sem GGUF). A unica excecao e o
smoke test do SLM, que roda apenas se o GGUF local ja foi baixado (skipif).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from projeto_final import agentes, config
from projeto_final.main import app

CHUNK_FAKE = [{"id": 1, "doc_id": "nlp_aula06_rag_avancado_ocr.pdf", "pagina": 1,
               "tipo": "texto", "texto": "RAG recupera trechos e o modelo responde."}]


def _gerado(resposta: str = "RAG recupera trechos relevantes. [1]") -> dict:
    return {
        "resposta": resposta, "chunks": CHUNK_FAKE, "contexto": "ctx",
        "uso": None, "latencia_s": 0.1, "modelo": "fake", "motivo": None,
    }


# ---------------------------------------------------------------- roteador

def test_roteador_simples():
    with patch("projeto_final.agentes.slm.chat", return_value="SIMPLES") as chat:
        assert agentes.classificar_rota("O que é DDL?", roteador="slm") == "SIMPLES"
        chat.assert_called_once()


def test_roteador_complexa():
    with patch("projeto_final.agentes.slm.chat", return_value="COMPLEXA"):
        assert agentes.classificar_rota("Qual a diferença entre X e Y?", roteador="slm") == "COMPLEXA"


def test_roteador_slm_falha_assume_simples():
    with patch("projeto_final.agentes.slm.chat", side_effect=RuntimeError("SLM indisponivel")):
        assert agentes.classificar_rota("qualquer coisa", roteador="slm") == "SIMPLES"


def test_roteador_flash_usa_modelo_mapeado():
    with patch("projeto_final.agentes.llm_mod.completar",
               return_value=("SIMPLES", {"modelo": config.AGENTE_MODELO_FLASH})) as comp:
        assert agentes.classificar_rota("O que é OCR?", roteador="flash") == "SIMPLES"
        # a funcao nova (roteador flash) chama o modelo flash mapeado explicitamente
        _, kwargs = comp.call_args
        assert kwargs["modelo"] == config.AGENTE_MODELO_FLASH


def test_roteador_flash_falha_assume_simples():
    with patch("projeto_final.agentes.llm_mod.completar", side_effect=RuntimeError("API fora")):
        assert agentes.classificar_rota("pergunta", roteador="flash") == "SIMPLES"


# ---------------------------------------------------------------- orquestracao

def test_rota_simples_usa_slm():
    with patch("projeto_final.agentes.classificar_rota", return_value="SIMPLES"), \
         patch("projeto_final.agentes._gerar_slm", return_value=_gerado()) as g_slm, \
         patch("projeto_final.agentes._gerar_api") as g_api:
        out = agentes.resposta_multiagente("pergunta", chunks=CHUNK_FAKE)
    assert out["agente"] == "gerador-slm"
    assert out["rota"] == "simples"
    assert out["fallback"] is False
    g_slm.assert_called_once()
    g_api.assert_not_called()


def test_fallback_slm_para_pro():
    with patch("projeto_final.agentes.classificar_rota", return_value="SIMPLES"), \
         patch("projeto_final.agentes._gerar_slm", side_effect=RuntimeError("SLM quebrou")), \
         patch("projeto_final.agentes._gerar_api", return_value=_gerado()) as g_api:
        out = agentes.resposta_multiagente("pergunta", chunks=CHUNK_FAKE)
    assert out["agente"] == "gerador-pro"
    assert out["fallback"] is True
    assert out["erros"] and "gerador-slm" in out["erros"][0]
    g_api.assert_called_once()


def test_fallback_pro_para_slm():
    with patch("projeto_final.agentes.classificar_rota", return_value="COMPLEXA"), \
         patch("projeto_final.agentes._gerar_api", side_effect=RuntimeError("API fora do ar")), \
         patch("projeto_final.agentes._gerar_slm", return_value=_gerado()) as g_slm:
        out = agentes.resposta_multiagente("pergunta", chunks=CHUNK_FAKE)
    assert out["agente"] == "gerador-slm"
    assert out["fallback"] is True
    assert out["rota"] == "complexa"
    g_slm.assert_called_once()


def test_resposta_vazia_fallback():
    vazio = {**_gerado(), "resposta": ""}
    with patch("projeto_final.agentes.classificar_rota", return_value="SIMPLES"), \
         patch("projeto_final.agentes._gerar_slm", return_value=vazio), \
         patch("projeto_final.agentes._gerar_api", return_value=_gerado()) as g_api:
        out = agentes.resposta_multiagente("pergunta", chunks=CHUNK_FAKE)
    assert out["agente"] == "gerador-pro"
    assert out["fallback"] is True
    assert any("resposta vazia" in e for e in out["erros"])
    g_api.assert_called_once()


def test_ambas_rotas_falham_abstem():
    with patch("projeto_final.agentes.classificar_rota", return_value="SIMPLES"), \
         patch("projeto_final.agentes._gerar_slm", side_effect=RuntimeError("x")), \
         patch("projeto_final.agentes._gerar_api", side_effect=RuntimeError("y")):
        out = agentes.resposta_multiagente("pergunta", chunks=CHUNK_FAKE)
    assert out["abstencao"] is True
    assert out["resposta"] == "NAO_SEI"
    assert out["motivo"] == "falha em ambas as rotas"
    assert out["agente"] is None
    assert len(out["erros"]) == 2


def test_gerador_desconhecido_erro():
    with pytest.raises(ValueError):
        agentes._fabricar_gerador("super")


# ---------------------------------------------------------------- API

def test_perguntar_agentes_true():
    fake = {
        "pergunta": "O que e RAG?", "resposta": "RAG ... [1]", "resposta_tts": "RAG ...",
        "referencias": [{"n": 1, "doc_id": "nlp_aula06_rag_avancado_ocr.pdf", "pagina": 1}],
        "abstencao": False, "chunks": [], "contexto": "", "latencia_s": 1.0,
        "modelo_llm": "Qwen2.5-1.5B-Instruct (GGUF Q4_K_M, local)", "uso": None,
        "usar_imagens": True, "rota": "simples", "agente": "gerador-slm",
        "roteador": "slm", "config": {"simples": "slm", "complexa": "pro"},
        "fallback": False, "erros": [], "motivo": None,
    }
    with patch("projeto_final.agentes.responder_agentes", return_value=fake) as ra:
        client = TestClient(app)
        resp = client.post("/rag/perguntar", params={"pergunta": "O que e RAG?", "agentes": "true"})
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["rota"] == "simples"
    assert corpo["agente"] == "gerador-slm"
    assert corpo["fallback"] is False
    ra.assert_called_once()
    assert ra.call_args.args[0] == "O que e RAG?"


# ---------------------------------------------------------------- SLM (local)

@pytest.mark.skipif(not agentes.slm.disponivel(), reason="GGUF do SLM nao baixado")
def test_slm_chat_quando_gguf_presente():
    texto = agentes.slm.chat(
        [{"role": "user", "content": "Responda apenas: OK"}], max_tokens=8, temperature=0.0
    )
    assert texto.strip()


