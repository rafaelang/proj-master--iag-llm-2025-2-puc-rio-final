"""Sistemas Multiagentes — v0.4 (Aula 04) do projeto_final.

Fluxo heterogeneo ponta a ponta com roteador SIMPLES/COMPLEXA, gerador SLM local
(rota simples) e gerador remoto (rota complexa), com fallback cruzado sob falha.

Modelos parametrizaveis (env/CLI, nao fixos no codigo):
  --roteador {slm,flash}        roteador (padrao: slm = Qwen2.5-1.5B local)
  --simples  {slm,flash,pro}    gerador da rota simples (padrao: slm)
  --complexa {slm,flash,pro}    gerador da rota complexa (padrao: pro)

Mapeamento (confirmado, modelos ja usados na v0.3):
  slm   = Qwen2.5-1.5B-Instruct local (GGUF Q4_K_M, CPU, custo zero)
  flash = config.AGENTE_MODELO_FLASH  (deepseek-chat)
  pro   = config.AGENTE_MODELO_PRO    (deepseek-v4-pro)

Falha (excecao/resposta vazia) -> fallback para a outra rota; se ambas falharem,
abstencao com motivo registrado. Diagrama e comportamento sob falha: docs/v04.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

from projeto_final import config, llm as llm_mod, slm
from projeto_final.rag.pipeline import (
    _formatar_contexto,
    _formatar_referencias,
    carregar_corpus_v3,
)
from projeto_final.rag.retrieve import recuperar

NOME_SLM = "Qwen2.5-1.5B-Instruct (GGUF Q4_K_M, local)"

# Fallbacks caso os prompts versionados ainda nao existam (nao devem ocorrer em prod)
ROTEADOR_SISTEMA_FALLBACK = (
    "Classifique a pergunta como SIMPLES ou COMPLEXA.\n"
    "SIMPLES = pergunta factual e direta que pode ser respondida com uma unica fonte.\n"
    "COMPLEXA = pergunta composta, comparativa, ambigua ou que exige raciocinio sobre "
    "multiplas fontes.\nResponda apenas com a palavra SIMPLES ou COMPLEXA."
)
SLM_SISTEMA_FALLBACK = (
    "Voce e um assistente do Master IAG e LLM da PUC-Rio. Responda em portugues "
    "usando SOMENTE o contexto fornecido. Se o contexto nao tiver a informacao, "
    "responda apenas NAO_SEI. Cite as fontes como [1], [2]."
)


def _sistema_roteador() -> str:
    return config.ler_prompt("v0.4/roteador_sistema.txt") or ROTEADOR_SISTEMA_FALLBACK


def _sistema_slm() -> str:
    return config.ler_prompt("v0.4/rag_sistema_slm.txt") or SLM_SISTEMA_FALLBACK


# --------------------------------------------------------------- roteador

def classificar_rota(pergunta: str, roteador: str | None = None) -> str:
    """Roteador SIMPLES/COMPLEXA. Padrao seguro: SIMPLES em qualquer falha."""
    roteador = roteador or config.AGENTE_ROTEADOR
    sistema = _sistema_roteador()
    if roteador == "flash":
        try:
            texto, _meta = llm_mod.completar(
                [
                    {"role": "system", "content": sistema},
                    {"role": "user", "content": pergunta},
                ],
                modelo=config.AGENTE_MODELO_FLASH,
                max_tokens=100,
                temperature=0.0,
            )
            return "COMPLEXA" if "COMPLEXA" in texto.upper() else "SIMPLES"
        except Exception as e:
            logger.warning("Roteador flash falhou ({}); assume SIMPLES", e)
            return "SIMPLES"
    try:
        texto = slm.chat(
            [
                {"role": "system", "content": sistema},
                {"role": "user", "content": pergunta},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        return "COMPLEXA" if "COMPLEXA" in texto.upper() else "SIMPLES"
    except Exception as e:
        logger.warning("Roteador SLM falhou ({}); assume SIMPLES", e)
        return "SIMPLES"


# --------------------------------------------------------------- geradores

def _abstido(motivo: str) -> dict:
    """Gerador sem evidencia: abstem-se sem gastar tokens (resposta NAO_SEI valida)."""
    return {
        "resposta": "NAO_SEI", "chunks": [], "contexto": "",
        "uso": None, "latencia_s": 0.0, "modelo": None, "motivo": motivo,
    }


def _gerar_slm(pergunta: str, chunks: list[dict], top_k: int = 5) -> dict:
    """Gerador local (rota simples): recupera no corpus v0.3 e responde com o SLM."""
    t0 = time.time()
    recuperados = recuperar(pergunta, chunks, top_k=top_k, base=config.RAG_V3_DIR)
    if not recuperados:
        return _abstido("sem evidencia recuperada")
    contexto = _formatar_contexto(recuperados)
    sistema = _sistema_slm()
    texto = slm.chat(
        [
            {"role": "system", "content": sistema},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"},
        ],
        max_tokens=config.SLM_MAX_TOKENS,
    )
    return {
        "resposta": texto, "chunks": recuperados, "contexto": contexto,
        "uso": None, "latencia_s": round(time.time() - t0, 2), "modelo": NOME_SLM,
        "motivo": None,
    }


def _gerar_api(pergunta: str, chunks: list[dict], modelo: str, top_k: int = 5) -> dict:
    """Gerador remoto (flash/pro): recupera no corpus v0.3 e responde com a API."""
    t0 = time.time()
    recuperados = recuperar(pergunta, chunks, top_k=top_k, base=config.RAG_V3_DIR)
    if not recuperados:
        return _abstido("sem evidencia recuperada")
    contexto = _formatar_contexto(recuperados)
    sistema = config.ler_prompt("v0.2/rag_sistema.txt")
    # pro e modelo de raciocinio: orcamento maior evita resposta vazia (length)
    max_tokens = config.AGENTE_PRO_MAX_TOKENS if modelo == config.AGENTE_MODELO_PRO else None
    texto, meta = llm_mod.responder_com_contexto(
        pergunta, contexto, sistema=sistema, modelo=modelo, max_tokens=max_tokens
    )
    return {
        "resposta": texto, "chunks": recuperados, "contexto": contexto,
        "uso": meta.get("uso"), "latencia_s": round(time.time() - t0, 2),
        "modelo": meta.get("modelo"), "motivo": None,
    }


def _fabricar_gerador(nome: str):
    """Devolve callable (pergunta, chunks, top_k) para a opcao escolhida."""
    if nome == "slm":
        return lambda p, cs, top_k=5: _gerar_slm(p, cs, top_k=top_k)
    if nome == "flash":
        return lambda p, cs, top_k=5: _gerar_api(p, cs, modelo=config.AGENTE_MODELO_FLASH, top_k=top_k)
    if nome == "pro":
        return lambda p, cs, top_k=5: _gerar_api(p, cs, modelo=config.AGENTE_MODELO_PRO, top_k=top_k)
    raise ValueError(f"gerador desconhecido: {nome!r} (opcoes: slm, flash, pro)")


# --------------------------------------------------------------- orquestrador

def resposta_multiagente(
    pergunta: str,
    chunks: list[dict] | None = None,
    top_k: int = 5,
    roteador: str | None = None,
    simples: str | None = None,
    complexa: str | None = None,
) -> dict:
    """Orquestra: roteia -> gera na rota -> fallback sob falha.

    Retorna dict com o mesmo schema do pipeline RAG (resposta/resposta_tts/
    referencias/abstencao/...) acrescido de rota/agente/fallback/erros/motivo.
    """
    t0 = time.time()
    roteador = roteador or config.AGENTE_ROTEADOR
    simples = simples or config.AGENTE_SIMPLES
    complexa = complexa or config.AGENTE_COMPLEXA
    rota = classificar_rota(pergunta, roteador=roteador).lower()
    if chunks is None:
        chunks = carregar_corpus_v3()

    # Ordem de tentativa: rota escolhida primeiro; fallback para a outra.
    ordem = (
        [(complexa, "gerador-" + complexa), (simples, "gerador-" + simples)]
        if rota == "complexa"
        else [(simples, "gerador-" + simples), (complexa, "gerador-" + complexa)]
    )
    primario = ordem[0][1]
    erros: list[str] = []

    for opcao, nome in ordem:
        try:
            r = _fabricar_gerador(opcao)(pergunta, chunks, top_k=top_k)
            texto = (r.get("resposta") or "").strip()
            if not texto:
                erros.append(f"{nome}: resposta vazia")
                continue
            refs = _formatar_referencias(texto, r.get("chunks") or [])
            return {
                "pergunta": pergunta,
                "resposta": refs["resposta"],
                "resposta_tts": refs["resposta_tts"],
                "referencias": refs["referencias"],
                "abstencao": llm_mod.detectar_abstencao(texto),
                "chunks": r.get("chunks") or [],
                "contexto": r.get("contexto") or "",
                "latencia_s": round(time.time() - t0, 2),
                "modelo_llm": r.get("modelo"),
                "uso": r.get("uso"),
                "usar_imagens": True,
                "rota": rota,
                "agente": nome,
                "roteador": roteador,
                "config": {"simples": simples, "complexa": complexa},
                "fallback": nome != primario,
                "erros": erros,
                "motivo": r.get("motivo"),
            }
        except Exception as exc:
            erros.append(f"{nome}: {str(exc)[:120]}")
            logger.warning("Gerador {} falhou: {}", nome, exc)

    # Ambas as rotas falharam -> abstencao com motivo registrado.
    logger.error("Multiagente: ambas as rotas falharam para '{}' (erros={})",
                 pergunta, erros)
    return {
        "pergunta": pergunta,
        "resposta": "NAO_SEI",
        "resposta_tts": "NAO_SEI",
        "referencias": [],
        "abstencao": True,
        "chunks": [],
        "contexto": "",
        "latencia_s": round(time.time() - t0, 2),
        "modelo_llm": None,
        "uso": None,
        "usar_imagens": True,
        "rota": rota,
        "agente": None,
        "roteador": roteador,
        "config": {"simples": simples, "complexa": complexa},
        "fallback": True,
        "erros": erros,
        "motivo": "falha em ambas as rotas",
    }


def responder_agentes(pergunta: str, top_k: int = 5) -> dict:
    """Wrapper usado pela API (`/rag/perguntar?agentes=true`)."""
    logger.info("Multiagente (API): '{}' (roteador={}, simples={}, complexa={})",
                pergunta, config.AGENTE_ROTEADOR, config.AGENTE_SIMPLES,
                config.AGENTE_COMPLEXA)
    return resposta_multiagente(pergunta, top_k=top_k)


# --------------------------------------------------------------- avaliacao

def avaliar_golden(
    perguntas_path: str | Path | None = None,
    top_k: int = 5,
    roteador: str | None = None,
    simples: str | None = None,
    complexa: str | None = None,
) -> dict:
    """Avalia o fluxo multiagente no golden set unico (data/golden_set/rag).

    Retorna {"resumo": {...}, "perguntas": [...]} — a renderizacao da evidencia
    em Markdown fica em scripts/avaliadores/avaliar_v4.py (padrao v2/v3).
    """
    perguntas_path = Path(perguntas_path or config.RAG_GOLDEN_SET)
    perguntas = json.loads(perguntas_path.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_corpus_v3()

    linhas: list[dict] = []
    acertos: list[bool] = []
    n_rotas: Counter = Counter()
    n_agentes: Counter = Counter()
    fallbacks = 0
    latencias: list[float] = []
    tokens_prompt = tokens_completion = 0

    for q in perguntas:
        r = resposta_multiagente(
            q["pergunta"], chunks=chunks, top_k=top_k,
            roteador=roteador, simples=simples, complexa=complexa,
        )
        absteve = r["abstencao"]
        deve = q.get("deve_abster", False)
        acertos.append(absteve == deve)
        n_rotas[r.get("rota") or "simples"] += 1
        n_agentes[r.get("agente") or "?"] += 1
        if r.get("fallback"):
            fallbacks += 1
        latencias.append(float(r.get("latencia_s") or 0.0))
        uso = r.get("uso")
        if uso:
            tokens_prompt += uso.get("prompt_tokens") or 0
            tokens_completion += uso.get("completion_tokens") or 0
        linhas.append({
            "id": q["id"],
            "pergunta": q["pergunta"],
            "estrato": q.get("estrato", ""),
            "rota": r.get("rota"),
            "agente": r.get("agente"),
            "fallback": bool(r.get("fallback")),
            "absteve": absteve,
            "deve_abster": deve,
            "acerto": absteve == deve,
            "resposta": (r.get("resposta") or "")[:160],
            "latencia_s": r.get("latencia_s"),
            "modelo_llm": r.get("modelo_llm"),
        })

    n = len(acertos)
    resumo = {
        "n_perguntas": n,
        "abstencao_correta": round(sum(acertos) / n, 3) if n else None,
        "rotas": dict(sorted(n_rotas.items())),
        "agentes": dict(sorted(n_agentes.items())),
        "config": {
            "roteador": roteador or config.AGENTE_ROTEADOR,
            "simples": simples or config.AGENTE_SIMPLES,
            "complexa": complexa or config.AGENTE_COMPLEXA,
        },
        "fallbacks": fallbacks,
        "latencia_media_s": round(sum(latencias) / n, 2) if n else 0.0,
        "tokens_api": {"prompt": tokens_prompt, "completion": tokens_completion},
        "slm_local": NOME_SLM,
    }
    logger.info("Multiagente avaliacao: abstenção correta={} rotas={} fallbacks={}",
                resumo["abstencao_correta"], resumo["rotas"], fallbacks)
    return {"resumo": resumo, "perguntas": linhas}


# --------------------------------------------------------------- CLI

def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--roteador", choices=["slm", "flash"], default=None)
    parser.add_argument("--simples", choices=["slm", "flash", "pro"], default=None)
    parser.add_argument("--complexa", choices=["slm", "flash", "pro"], default=None)


def cmd_perguntar(args) -> None:
    r = resposta_multiagente(args.pergunta, roteador=args.roteador,
                             simples=args.simples, complexa=args.complexa)
    print("ROTA:", r.get("rota"), "| AGENTE:", r.get("agente"),
          "| FALLBACK:", r.get("fallback"))
    print("RESPOSTA:")
    print(r.get("resposta"))
    if r.get("erros"):
        print("ERROS:", r["erros"])


def cmd_avaliar(args) -> None:
    out = avaliar_golden(args.perguntas, top_k=args.top_k, roteador=args.roteador,
                         simples=args.simples, complexa=args.complexa)
    print("RESUMO:", json.dumps(out["resumo"], ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"json salvo em: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiagentes v0.4 (roteador + geradores)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("perguntar", help="roda o fluxo multiagente para uma pergunta")
    p1.add_argument("pergunta")
    _add_model_args(p1)
    p1.set_defaults(func=cmd_perguntar)
    p2 = sub.add_parser("avaliar", help="avalia no golden set (20 perguntas)")
    p2.add_argument("--perguntas", default=str(config.RAG_GOLDEN_SET))
    p2.add_argument("--top-k", type=int, default=5)
    p2.add_argument("--out")
    _add_model_args(p2)
    p2.set_defaults(func=cmd_avaliar)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())




