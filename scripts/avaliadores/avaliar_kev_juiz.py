"""Estudo: Kev como juiz (LLM-as-judge) — concordancia com o juiz pro (v0.7).

Mede, num sample de respostas da fronteira v0.7, se o Kev (modelo de decisao,
noul/score calibrado) concorda com o juiz `deepseek-v4-pro` (rubrica v5):

  - `correta`: Kev noul "a resposta esta correta?" (prob >= 0.5 -> correta) x
    juiz pro correta (concordancia, matriz de confusao, por estrato);
  - `nota`: Kev score 1-5 x nota do juiz pro (MAE/correlacao).

NAO substitui o juiz: e uma validacao/cross-check (Kev nao gera justificativa e
nao conhece o dominio PT-BR em zero-shot). Se a concordancia for alta nos casos
de alta confianca do Kev, ele poderia triar/flaggar casos para revisao.

Uso (endpoint Kev no ar):
  python scripts/avaliadores/avaliar_kev_juiz.py --sample <amostra.json> --saida <out.json>

Saida (fora do Git): data/processed/v07_ab/kev_juiz_concordancia.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import urllib.request
from pathlib import Path

KEV_BASE_URL = os.getenv("KEV_BASE_URL", "http://127.0.0.1:8009")
KEV_MODEL = os.getenv("KEV_MODEL", "kev-latest")
KEV_API_KEY = os.getenv("KEV_API_KEY", "")
KEV_TIMEOUT_S = int(os.getenv("KEV_TIMEOUT_S", "120"))

NOUL_INSTRUCAO = (
    "A resposta do assistente esta CORRETA? Regras do juiz: "
    "se FONTES_ESPERADAS e vazio e a resposta se abstem (NAO_SEI ou equivalente), "
    "a resposta e CORRETA. Se FONTES_ESPERADAS e vazio e a resposta responde de "
    "fato, e INCORRETA (alucinou). Se ha fontes esperadas e a resposta esta "
    "correta e completa sobre o tema, e CORRETA. Se ha fontes esperadas e a "
    "resposta se abstem, e INCORRETA. Responda sim se CORRETA, nao se INCORRETA."
)
SCORE_CRITERIOS = ["1 - muito ruim", "2 - ruim", "3 - regular", "4 - bom", "5 - muito bom"]


def _chamar(state: str) -> dict:
    payload = {
        "state": state,
        "model": KEV_MODEL,
        "questions": {
            "correta": {"type": "noul", "instructions": NOUL_INSTRUCAO},
            "nota": {"type": "score", "instructions": "Qual a nota (1-5) da resposta?",
                     "criteria": SCORE_CRITERIOS},
        },
    }
    req = urllib.request.Request(
        f"{KEV_BASE_URL.rstrip('/')}/v1/systemone",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {KEV_API_KEY}"} if KEV_API_KEY else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=KEV_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _montar_state(item: dict) -> str:
    fontes = ", ".join(item.get("docs_esperados") or [])
    if not fontes:
        fontes = "<fora do corpus>"
    resp = (item.get("resposta") or "").strip()[:400]
    return (f"PERGUNTA: {item['pergunta']}\n"
            f"FONTES_ESPERADAS: {fontes}\n"
            f"RESPOSTA DO ASSISTENTE: {resp}")


def avaliar(itens: list[dict]) -> dict:
    linhas = []
    for it in itens:
        try:
            data = _chamar(_montar_state(it))
            ans = data["answers"]
            noul = float(ans["correta"]["noul"])
            score = float(ans["nota"]["score"])  # nivel 0-based -> +1 = nota
            kev_correta = noul >= 0.5
            kev_nota = round(score + 1.0, 2)
            conf = max(noul, 1 - noul)
        except Exception as e:
            linhas.append({"id": it["id"], "erro": str(e)[:150]})
            continue
        pro = it.get("juiz_pro") or {}
        linhas.append({
            "id": it["id"], "estrato": it.get("estrato"),
            "pro_correta": pro.get("correta"), "pro_nota": pro.get("nota"),
            "kev_correta": kev_correta, "kev_noul": round(noul, 3),
            "kev_nota": kev_nota, "confianca": round(conf, 3),
        })
    return linhas


def _resumir(linhas: list[dict]) -> dict:
    ok = [l for l in linhas if "kev_correta" in l and l.get("pro_correta") is not None]
    n = len(ok)
    conc = sum(1 for l in ok if l["kev_correta"] == l["pro_correta"])
    conf = {}
    for l in ok:
        conf.setdefault(l["estrato"], [0, 0])
        conf[l["estrato"]][0] += int(l["kev_correta"] == l["pro_correta"])
        conf[l["estrato"]][1] += 1
    # matriz de confusao (linha=pro, col=kev)
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for l in ok:
        p, k = l["pro_correta"], l["kev_correta"]
        if p and k: cm["TP"] += 1
        elif not p and not k: cm["TN"] += 1
        elif not p and k: cm["FP"] += 1
        else: cm["FN"] += 1
    notas = [(l["pro_nota"], l["kev_nota"]) for l in ok
             if l.get("pro_nota") is not None and l.get("kev_nota") is not None]
    mae = round(sum(abs(p - k) for p, k in notas) / len(notas), 2) if notas else None
    # concordancia apenas nos casos de alta confianca do Kev (conf >= 0.7)
    altos = [l for l in ok if l["confianca"] >= 0.7]
    conc_alto = sum(1 for l in altos if l["kev_correta"] == l["pro_correta"]) if altos else None
    return {
        "n": n,
        "concordancia_geral": round(conc / n, 3) if n else None,
        "concordancia_por_estrato": {e: round(a / t, 3) for e, (a, t) in sorted(conf.items())},
        "matriz_confusao_pro_vs_kev": cm,
        "n_altac_confianca": len(altos),
        "concordancia_altac_confianca": round(conc_alto / len(altos), 3) if altos else None,
        "mae_nota": mae,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Kev como juiz — concordancia com o juiz pro (v0.7)")
    ap.add_argument("--sample", required=True, help="json com itens {id,pergunta,estrato,docs_esperados,resposta,juiz_pro}")
    ap.add_argument("--saida", default=str(Path(__file__).resolve().parent.parent.parent /
                                           "data/processed/v07_ab/kev_juiz_concordancia.json"))
    a = ap.parse_args()

    itens = json.loads(Path(a.sample).read_text(encoding="utf-8"))
    print(f"sample: {len(itens)} itens · modelo={KEV_MODEL}", flush=True)
    linhas = avaliar(itens)
    erros = [l for l in linhas if "erro" in l]
    print(f"avaliados: {len(linhas)} (erros: {len(erros)})", flush=True)
    resumo = _resumir(linhas)
    out = {"modelo": KEV_MODEL, "resumo": resumo, "itens": linhas}
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== RESUMO Kev-juiz ===")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"salvo em {a.saida}")


if __name__ == "__main__":
    main()