"""v0.6 · P2 (plano do orientador) — perda de roteamento: pro forçado nas compostas que o SLM errou.

Mede, intra-sessão, quantas compostas roteadas à rota SIMPLES o pro teria acertado
(counterfactual). Decide com número se o limiar da cascade muda ou se a correção é
forçar COMPLEXA (a alavanca real — subir limiar cego só empurra INDETERMINADO ao R1,
que também é SLM).

Entrada:
- data/processed/v05b_ab/cascade_v06_p2.jsonl (compostas da rota simples com juiz
  correta=false: #37 #113 #118 #120 #122 #124 #125 #126 #127 #130 #131)

Ação (API pro + juiz):
- para cada uma, gera via _gerar_api(modelo=AGENTE_MODELO_PRO) com o RAG real
  (mesmo índice/pool da cascade v0.6) e julga com o mesmo juiz.

Saída:
- data/processed/v05b_ab/cascade_v06_p2_proforcado.jsonl (incremental)
- tabela rota simples (SLM) × rota complexa (pro forçado) por pergunta

Uso:
  python scripts/avaliadores/avaliar_rotas_pro_forcado.py --id 37    # incremental por id
  python scripts/avaliadores/avaliar_rotas_pro_forcado.py --resumo   # consolida jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger
from openai import OpenAI

from projeto_final import agentes, config
from projeto_final.rag.pipeline import carregar_chunks

RAIZ = Path(__file__).resolve().parent.parent.parent
CASCADE = RAIZ / "data/processed/v05b_ab/cascade_v06_p2.jsonl"
SAIDA = RAIZ / "data/processed/v05b_ab/cascade_v06_p2_proforcado.jsonl"

IDS_TARGET = [37, 113, 118, 120, 122, 124, 125, 126, 127, 130, 131]


def _juiz(pergunta: str, fontes: list[str], resposta: str, cliente):
    sys.path.insert(0, str(Path(__file__).parent))
    from avaliar_v5 import juiz  # type: ignore

    return juiz(pergunta, fontes, resposta, cliente)


def _original() -> dict[int, dict]:
    linhas = [json.loads(l) for l in CASCADE.read_text(encoding="utf-8").splitlines()]
    return {l["id"]: l for l in linhas if l["id"] in IDS_TARGET}


def rodar(ids: list[int]) -> None:
    from projeto_final.agentes import _gerar_api

    orig = _original()
    if not orig:
        raise SystemExit("cascade_v06_p2.jsonl não encontrado ou sem os IDs alvo")
    chunks = carregar_chunks()
    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    done = set()
    if SAIDA.exists():
        for l in SAIDA.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass

    log = open(SAIDA, "a")
    for qid in ids:
        if qid in done:
            print(f"#{qid} já feito", flush=True)
            continue
        o = orig[qid]
        t0 = time.time()
        try:
            r = _gerar_api(o["pergunta"], chunks, modelo=config.AGENTE_MODELO_PRO, top_k=5)
            texto = (r.get("resposta") or "").strip()
            absteve = "NAO_SEI" in texto.upper() or not texto
            linha = {"id": qid, "pergunta": o["pergunta"], "estrato": o["estrato"],
                     "docs_esperados": o["docs_esperados"],
                     "resposta_slm": o.get("resposta"),
                     "juiz_slm": o.get("juiz"),
                     "rota_original": o.get("rota"),
                     "rota_forcada": "complexa",
                     "resposta_pro": texto[:400], "absteve_pro": absteve,
                     "uso": r.get("uso"),
                     "latencia_s": round(time.time() - t0, 2),
                     "modelo": r.get("modelo")}
        except Exception as e:
            logger.error("pro falhou #{}: {}", qid, e)
            linha = {"id": qid, "erro": str(e)[:200], "rota_original": o.get("rota")}
        j = None
        if "resposta_pro" in linha:
            j = _juiz(o["pergunta"], o["docs_esperados"], linha["resposta_pro"], cliente)
            linha["juiz_pro"] = j
        log.write(json.dumps(linha, ensure_ascii=False) + "\n")
        log.flush()
        print(f"#{qid} pro: {j and j['correta']} t={time.time()-t0:.0f}s "
              f"| SLM era: {o.get('juiz') and o['juiz']['correta']}", flush=True)
    log.close()


def resumir() -> None:
    if not SAIDA.exists():
        raise SystemExit("sem jsonl pro-forçado")
    linhas = [json.loads(l) for l in SAIDA.read_text(encoding="utf-8").splitlines()]
    linhas = [l for l in linhas if "juiz_pro" in l]
    n = len(linhas)
    acertos_pro = sum(1 for l in linhas if l["juiz_pro"] and l["juiz_pro"]["correta"])
    acertos_slm = sum(1 for l in linhas if l["juiz_slm"] and l["juiz_slm"]["correta"])
    aluc_pro = sum(1 for l in linhas if l["juiz_pro"] and l["juiz_pro"]["alucinou"])
    aluc_slm = sum(1 for l in linhas if l["juiz_slm"] and l["juiz_slm"]["alucinou"])
    print("=== ROTEAMENTO: compostas simples ===")
    print(f"| id | SLM | pro forçado | pro corrige? |")
    print("|---|---|---|---|")
    for l in sorted(linhas, key=lambda x: x["id"]):
        js, jp = l["juiz_slm"], l["juiz_pro"]
        cs, cp = js and js["correta"], jp and jp["correta"]
        print(f"| #{l['id']} | {cs} | {cp} | {'SIM' if (not cs and cp) else '-'} |")
    print(f"\nSLM: {acertos_slm}/{n} · pro forçado: {acertos_pro}/{n} · "
          f"pro resgataria {acertos_pro - acertos_slm} · aluc SLM {aluc_slm} → pro {aluc_pro}")
    custo_in = sum((l.get("uso") or {}).get("prompt_tokens") or 0 for l in linhas)
    custo_out = sum((l.get("uso") or {}).get("completion_tokens") or 0 for l in linhas)
    custo = custo_in * 0.27 / 1e6 + custo_out * 1.10 / 1e6
    print(f"custo pro forçado nas {n}Q: US$ {custo:.5f} (in {custo_in} / out {custo_out})")


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 — pro forçado nas compostas da rota simples")
    ap.add_argument("--id", type=int, action="append", default=None,
                    help="ids a rodar (default: todos os alvo)")
    ap.add_argument("--resumo", action="store_true", help="consolida o jsonl e sai")
    a = ap.parse_args()
    if a.resumo:
        resumir()
        return
    rodar(a.id or IDS_TARGET)


if __name__ == "__main__":
    main()