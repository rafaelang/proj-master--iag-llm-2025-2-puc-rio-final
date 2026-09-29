"""v0.7 · Avaliador frontier-RAG no golden expandido (pro em todas as perguntas).

Mede a linha de base "frontier" da v0.5/v0.5b (RAG com deepseek-v4-pro em TODAS
as perguntas, sem roteamento) no golden v0.7 — o contraste com a cascade. Mesmo
juiz (avaliar_v5.juiz) e mesma geracao (_gerar_api) das releases anteriores.

Uso:
  python scripts/avaliadores/avaliar_v07.py            # roda incremental (132+)
  python scripts/avaliadores/avaliar_v07.py --resumo   # consolida o jsonl e sai

Saida (fora do Git): data/processed/v07_ab/frontier_v07.jsonl (+ resumo json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from openai import OpenAI

from projeto_final import agentes, config, llm as llm_mod
from projeto_final.rag.pipeline import carregar_chunks

RAIZ = Path(__file__).resolve().parent.parent.parent
GOLDEN = config.GOLDEN_SET_DIR / "rag" / "perguntas_v07.json"
SAIDA = RAIZ / "data/processed/v07_ab/frontier_v07.jsonl"
RESUMO = RAIZ / "data/processed/v07_ab/frontier_v07_resumo.json"

PRECO_IN = 0.27
PRECO_OUT = 1.10


def _juiz(pergunta: str, fontes: list[str], resposta: str, cliente):
    sys.path.insert(0, str(Path(__file__).parent))
    from avaliar_v5 import juiz  # type: ignore

    return juiz(pergunta, fontes, resposta, cliente)


def rodar(inicio: int = 0, fim: int | None = None) -> None:
    from projeto_final.agentes import _gerar_api

    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    if fim is not None:
        perguntas = perguntas[inicio:fim]
    chunks = carregar_chunks()
    cliente = llm_mod.cliente()

    done = set()
    if SAIDA.exists():
        for l in SAIDA.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(l)
                # v0.7: respostas VAZIAS (falha transitoria do gateway) sao
                # re-geradas na proxima rodada — so marcam done as com conteudo.
                if (d.get("resposta") or "").strip():
                    done.add(d["id"])
            except Exception:
                pass

    log = open(SAIDA, "a")
    for q in perguntas:
        if q["id"] in done:
            continue
        t0 = time.time()
        try:
            r = _gerar_api(q["pergunta"], chunks, modelo=config.AGENTE_MODELO_PRO, top_k=5)
            texto = (r.get("resposta") or "").strip()
        except Exception as e:
            logger.error("pro falhou #{}: {}", q["id"], e)
            r, texto = {}, ""
        j = _juiz(q["pergunta"], sorted(q.get("docs_esperados", [])), texto, cliente)
        linha = {
            "id": q["id"], "estrato": q["estrato"], "pergunta": q["pergunta"],
            "deve_abster": q.get("deve_abster", False),
            "docs_esperados": sorted(q.get("docs_esperados", [])),
            "resposta": texto[:400], "absteve": bool(texto) and "NAO_SEI" in texto.upper(),
            "uso": r.get("uso"), "latencia_s": r.get("latencia_s"),
            "modelo": r.get("modelo"), "juiz": j,
        }
        log.write(json.dumps(linha, ensure_ascii=False) + "\n")
        log.flush()
        print(f"#{q['id']:03d} [{q['estrato']:11s}] absteve={linha['absteve']} "
              f"juiz={j and j['correta']} t={time.time()-t0:.0f}s", flush=True)
    log.close()


def resumir() -> dict:
    linhas = [json.loads(l) for l in SAIDA.read_text(encoding="utf-8").splitlines()]
    linhas = [l for l in linhas if "juiz" in l]

    def _met(rows: list[dict]) -> dict:
        n = len(rows)
        julg = [r for r in rows if r["juiz"]]
        acerto = sum(1 for r in julg if r["juiz"]["correta"])
        aluc = sum(1 for r in julg if r["juiz"]["alucinou"])
        nota = round(sum(r["juiz"]["nota"] for r in julg) / max(1, len(julg)), 2)
        abs_ok = sum(1 for r in rows if r["absteve"] == r["deve_abster"])
        nao_abst = sum(1 for r in rows if not r["absteve"])
        cit = sum(1 for r in rows if not r["absteve"] and r["resposta"])
        tok = {"prompt": 0, "completion": 0}
        for r in rows:
            u = r.get("uso") or {}
            tok["prompt"] += u.get("prompt_tokens") or 0
            tok["completion"] += u.get("completion_tokens") or 0
        custo = tok["prompt"] * PRECO_IN / 1e6 + tok["completion"] * PRECO_OUT / 1e6
        lat = [r.get("latencia_s") or 0 for r in rows]
        return {
            "n": n,
            "acuracia": round(acerto / len(julg), 3) if julg else None,
            "acertos": acerto, "julgadas": len(julg),
            "abstencao_correta": round(abs_ok / n, 3),
            "alucinacoes": aluc, "nota_media": nota,
            "citacao": round(cit / max(1, nao_abst), 3),
            "tokens_api": tok, "custo_usd": round(custo, 5),
            "latencia_media_s": round(sum(lat) / n, 2) if n else None,
        }

    por_estrato: dict[str, list[dict]] = {}
    for l in linhas:
        por_estrato.setdefault(l["estrato"], []).append(l)
    out = {"n": len(linhas), "total": _met(linhas),
           "por_estrato": {e: _met(rs) for e, rs in sorted(por_estrato.items())},
           "dataset": GOLDEN.name}
    RESUMO.parent.mkdir(parents=True, exist_ok=True)
    RESUMO.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.7 — frontier-RAG no golden expandido")
    ap.add_argument("--resumo", action="store_true")
    ap.add_argument("--inicio", type=int, default=0)
    ap.add_argument("--fim", type=int, default=None)
    a = ap.parse_args()
    if a.resumo:
        r = resumir()
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    rodar(inicio=a.inicio, fim=a.fim)
    print(json.dumps(resumir(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()