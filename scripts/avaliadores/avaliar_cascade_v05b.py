"""P2 · Avaliador do fluxo COMPLETO multiagente (roteador cascade) no golden v0.5b.

Mede a cascata de producao (SIMPLES->SLM local, COMPLEXA->frontier pro, fallback)
nas 44 perguntas do golden v0.5b CORRIGIDO, com o RAG de P1 (RAG_DIR, pool 60):

- Rota/agente/fallback por pergunta (trafico SLM x frontier);
- Acuracia via juiz (LLM-as-judge, mesmo prompt do avaliar_v5), abstencao, citacao;
- Custo/tokens por rota (SLM = US$ 0 local; pro = tokens da API).

Leitura-chave (R8 nao medida): o SLM e julgado APENAS nas perguntas que a cascata
realmente envia para ele (rota simples), e o frontier nas complexas — eliminando o
pessimismo artificial da v0.5b (que julgou o SLM nas 44Q, inclusive as complexas).

Uso (Colab/local):
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_cascade_v05b.py
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_cascade_v05b.py --pools ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

from projeto_final import agentes, config
from projeto_final.rag.pipeline import carregar_chunks

GOLDEN = config.RAG_GOLDEN_SET
# Nome de saída derivado da versão do golden (cascade_v05b_p2.jsonl vs
# cascade_v06_p2.jsonl) — preserva as medições de releases fechadas.
_VERSAO = re.search(r"v\d[\w]*", GOLDEN.name)
_VERSAO = _VERSAO.group() if _VERSAO else "v05b"
SAIDA_JSONL = config.PROCESSED_DIR / "v05b_ab" / f"cascade_{_VERSAO}_p2.jsonl"

# Preco por milhao de tokens (USD, DeepSeek) para estimativa de custo.
PRECO_IN = 0.27      # $/M token de entrada (deepseek-v4-pro)
PRECO_OUT = 1.10     # $/M token de saida (deepseek-v4-pro)


def _juiz(pergunta: str, fontes: list[str], resposta: str, cliente):
    sys.path.insert(0, str(Path(__file__).parent))
    from avaliar_v5 import juiz  # type: ignore

    return juiz(pergunta, fontes, resposta, cliente)


def avaliar(roteador: str = "cascade", simples: str = "slm", complexa: str = "pro",
            incremental: bool = True, inicio: int = 0, fim: int | None = None) -> list[dict]:
    from openai import OpenAI

    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    if fim is not None:
        perguntas = perguntas[inicio:fim]
    chunks = carregar_chunks()
    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    done_ids = set()
    if incremental and SAIDA_JSONL.exists():
        for l in SAIDA_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(l)["id"])
            except Exception:
                pass

    linhas = []
    log = open(SAIDA_JSONL, "a") if incremental else None
    for q in perguntas:
        if q["id"] in done_ids:
            continue
        t0 = time.time()
        try:
            r = agentes.resposta_multiagente(
                q["pergunta"], chunks=chunks, top_k=5,
                roteador=roteador, simples=simples, complexa=complexa,
            )
        except Exception as e:
            logger.error("multiagente falhou #{}: {}", q["id"], e)
            r = {"resposta": "", "abstencao": True, "rota": "?",
                 "agente": None, "fallback": False, "erros": [str(e)]}
        esperados = sorted(q.get("docs_esperados", []))
        j = _juiz(q["pergunta"], esperados, r.get("resposta", ""), cliente)
        linha = {
            "id": q["id"], "estrato": q["estrato"], "pergunta": q["pergunta"],
            "deve_abster": q.get("deve_abster", False),
            "docs_esperados": esperados,
            "rota": r.get("rota"), "agente": r.get("agente"),
            "fallback": bool(r.get("fallback")), "erros": r.get("erros") or [],
            "absteve": bool(r.get("abstencao")),
            "resposta": (r.get("resposta") or "")[:400],
            "latencia_s": r.get("latencia_s"),
            "modelo": r.get("modelo_llm"),
            "uso": r.get("uso"),
            "juiz": j,
        }
        linhas.append(linha)
        if log:
            log.write(json.dumps(linha, ensure_ascii=False) + "\n")
            log.flush()
        print(f"#{q['id']:02d} rota={linha['rota']} agente={linha['agente']} "
              f"fallback={linha['fallback']} absteve={linha['absteve']} "
              f"juiz={j and j['correta']} t={time.time()-t0:.0f}s", flush=True)
    if log:
        log.close()
    return linhas


def resumir(linhas: list[dict]) -> dict:
    if not linhas:
        return {"erro": "sem linhas"}
    por_rota: dict[str, list[dict]] = {}
    for l in linhas:
        por_rota.setdefault(l["rota"] or "?", []).append(l)

    def _met(rows: list[dict]) -> dict:
        n = len(rows)
        julg = [r for r in rows if r["juiz"]]
        acerto = sum(1 for r in julg if r["juiz"]["correta"])
        aluc = sum(1 for r in julg if r["juiz"]["alucinou"])
        nota = round(sum(r["juiz"]["nota"] for r in julg) / max(1, len(julg)), 2)
        abs_ok = sum(1 for r in rows if r["absteve"] == r["deve_abster"])
        cit = sum(1 for r in rows if not r["absteve"] and r["resposta"])
        tokens = {"prompt": 0, "completion": 0}
        for r in rows:
            u = r.get("uso") or {}
            tokens["prompt"] += u.get("prompt_tokens") or 0
            tokens["completion"] += u.get("completion_tokens") or 0
        custo = tokens["prompt"] * PRECO_IN / 1e6 + tokens["completion"] * PRECO_OUT / 1e6
        lat = [r.get("latencia_s") or 0 for r in rows]
        return {
            "n": n,
            "acuracia": round(acerto / len(julg), 3) if julg else None,
            "acertos": acerto, "julgadas": len(julg),
            "abstencao_correta": round(abs_ok / n, 3),
            "alucinacoes": aluc, "nota_media": nota,
            "citacao": round(cit / max(1, sum(1 for r in rows if not r["absteve"])), 3),
            "tokens_api": tokens,
            "custo_usd": round(custo, 5),
            "latencia_media_s": round(sum(lat) / n, 2),
            "fallbacks": sum(1 for r in rows if r["fallback"]),
        }

    resumo = {"n": len(linhas), "por_rota": {r: _met(rs) for r, rs in sorted(por_rota.items())}}
    resumo["total"] = _met(linhas)
    resumo["rotas"] = dict(sorted(Counter(l["rota"] or "?" for l in linhas).items()))
    resumo["agentes"] = dict(sorted(Counter(l["agente"] or "?" for l in linhas).items()))
    # P4: corte por estrato em toda medição (relatório por estrato)
    por_estrato: dict[str, list[dict]] = {}
    for l in linhas:
        por_estrato.setdefault(l.get("estrato") or "?", []).append(l)
    resumo["por_estrato"] = {e: _met(rs) for e, rs in sorted(por_estrato.items())}
    return resumo


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliador P2 — fluxo cascade nas 44Q")
    ap.add_argument("--inicio", type=int, default=0)
    ap.add_argument("--fim", type=int, default=None)
    ap.add_argument("--sem-incremental", action="store_true",
                    help="nao roda: apenas consolida o jsonl existente em resumo")
    a = ap.parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if a.sem_incremental:
        linhas = [json.loads(l) for l in SAIDA_JSONL.read_text(encoding="utf-8").splitlines()]
    else:
        avaliar(incremental=True, inicio=a.inicio, fim=a.fim)
        linhas = [json.loads(l) for l in SAIDA_JSONL.read_text(encoding="utf-8").splitlines()]
    resumo = resumir(linhas)
    print("\n=== P2 RESUMO ===")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    (config.PROCESSED_DIR / "v05b_ab" / f"cascade_{_VERSAO}_p2_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()