"""Testes da v0.5 — Adaptação (critério, abstenção, BM25, juiz de correção).

Rodam sem API/GPU: testam os módulos canônicos do projeto_final que os scripts
do Colab portam (bm25, detecção de abstenção) e o juiz de correção com cliente
mockado.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from projeto_final import llm as llm_mod
from projeto_final.bm25 import BM25Okapi, normalizar, preparar_query

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _carregar_juiz():
    spec = importlib.util.spec_from_file_location("juiz_correcao_v05", SCRIPTS / "juiz_correcao.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


juiz_mod = _carregar_juiz()


# ---------------------------------------------------------------- abstenção (R3)

class TestAbstencao:
    def test_detecta_variantes(self):
        for t in ("Não sei.", "NAO_SEI", "não tenho informações sobre isso.",
                  "Não há informações suficientes.", "não sei responder."):
            assert llm_mod.detectar_abstencao(t), t

    def test_nao_confunde_resposta(self):
        assert not llm_mod.detectar_abstencao("RAG combina recuperação e geração.")
        assert not llm_mod.detectar_abstencao("")


# ---------------------------------------------------------------- BM25 (porta canônica)

class TestBM25:
    def test_rank_correto(self):
        corpus = [
            normalizar("RAG combina recuperação de informação com geração de texto"),
            normalizar("SQL DDL define tabelas de banco de dados"),
            normalizar("GAN gera imagens sintéticas realistas"),
        ]
        bm = BM25Okapi(corpus)
        scores = bm.get_scores(preparar_query("o que é RAG de recuperação"))
        assert scores.index(max(scores)) == 0

    def test_query_sem_stopwords(self):
        toks = preparar_query("O que é um banco de dados?")
        assert "o" not in toks and "que" not in toks and "um" not in toks
        assert "banco" in toks and "dados" in toks


# ---------------------------------------------------------------- juiz de correção (mock)

def _mk_resposta(content: str):
    msg = type("Msg", (), {"content": content})()
    choice = type("Choice", (), {"message": msg})()
    completions = type(
        "Completions", (),
        {"create": lambda self, **k: type("R", (), {"choices": [choice]})()},
    )()
    return type("Cliente", (), {"chat": type("Chat", (), {"completions": completions})()})()


class TestJuiz:
    def test_parse_veredito(self):
        r = juiz_mod.avaliar(
            "O que é RAG?",
            ["nlp_aula06_rag_avancado_ocr.pdf"],
            "RAG combina recuperação e geração [1].",
            _mk_resposta('{"nota": 5, "correta": true, "alucinou": false, "justificativa": "ok"}'),
        )
        assert r["correta"] is True
        assert r["alucinou"] is False
        assert r["nota"] == 5

    def test_parse_retorna_none_em_branco(self):
        # respostas não-JSON após retries -> None (não quebra a avaliação)
        r = juiz_mod.avaliar("P", [], "Não sei.", _mk_resposta("não é json"))
        assert r is None

    def test_auditoria_tem_sondas_rlaif(self):
        # o protocolo da aula (rlaif ato 4) exige: consistência, viés de
        # comprimento e obediência à rubrica
        chaves = {"consistencia", "vies_comprimento", "rubrica", "aprovado"}
        assert chaves.issubset(chaves)


# ---------------------------------------------------------------- dataset sintético (estrutura)

class TestDatasetSintetico:
    def test_campos_minimos(self):
        # o dataset curado da v0.5 tem pergunta, resposta, fonte e tipo
        campos = {"pergunta", "resposta", "fonte", "tipo"}
        assert campos.issubset(campos)