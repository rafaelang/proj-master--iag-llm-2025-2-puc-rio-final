"""Estudo: Laya como juiz (LLM-as-judge) — concordancia com o juiz pro (v0.7).

Repete a MESMA bateria de `avaliar_kev_juiz.py` (Kev -> Laya) no MESMO sample de
80 respostas da fronteira v0.7 (`kev_amostra.json`), para comparar no mesmo
protocolo: concordancia `correta` (noul), `nota` (score 1-5), por estrato, matriz
de confusao e concordancia nos casos de alta confianca.

Contrato Laya (mesmo endpoint do Kev, `POST /v1/systemone`):
  - `score` e o nivel ordinal ESPERADO, base 0 (ex.: levels [a,b,c] -> 1.84 ~ b),
    entao `nota = score + 1`, como no Kev;
  - a confianca calibrada e `answer_confidence` (= P da resposta reportada).

NAO substitui o juiz: e um cross-check. O Laya nao gera justificativa e nao
conhece o dominio PT-BR em zero-shot.

Uso (endpoint laya no ar):
  LAYA_BASE_URL=http://127.0.0.1:8010 python scripts/avaliadores/avaliar_laya_juiz.py \
      --sample data/processed/v07_ab/kev_amostra.json --saida <out.json>

Saida (fora do Git): data/processed/v07_ab/laya_juiz_concordancia.json
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

LAYA_BASE_URL = os.getenv("LAYA_BASE_URL", "http://127.0.0.1:8010")
LAYA_MODELO = os.getenv("LAYA_MODELO", "multilingual")
LAYA_API_KEY = os.getenv("LAYA_API_KEY", "")
LAYA_TIMEOUT_S = int(os.getenv("LAYA_TIMEOUT_S", "120"))

# Mesmas rubricas/instrucoes do estudo do Kev (variavel so o motor).
NOUL_INSTRUCAO = (
    "A resposta do assistente esta CORRETA? Regras do juiz: "
    "se FONTES_ESPERADAS e vazio e a resposta se abstem (NAO_SEI ou equivalente), "
    "a resposta e CORRETA. Se FONTES_ESPERADAS e vazio e a resposta responde de "
    "fato, e INCORRETA (alucinou). Se ha fontes esperadas e a resposta esta "
    "correta e completa sobre o tema, e CORRETA. Se ha fontes esperadas e a "
    "resposta se abstem, e INCORRETA. Responda sim se CORRETA, nao se INCORRETA."
)
SCORE_CRITERIOS = ["1 - muito ruim", "2 - ruim", "3 - regular", "4 - bom", "5 - muito bom"]


def _chamar(state: str, modelo: str) -> dict:
    payload = {
        "state": state,
        "model": modelo,
        "questions": {
            "correta": {"type": "noul", "instructions": NOUL_INSTRUCAO},
            "nota": {"type": "score", "instructions": "Qual a nota (1-5) da resposta?",
                     "criteria": SCORE_CRITERIOS},
        },
    }
    req = urllib.request.Request(
        f"{LAYA_BASE_URL.rstrip('/')}/v1/systemone",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {LAYA_API_KEY}"} if LAYA_API_KEY else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=LAYA_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _montar_state(item: dict) -> str:
    fontes = ", ".join(item.get("docs_esperados") or [])
    if not fontes:
        fontes = "<fora do corpus>"
    resp = (item.get("resposta") or "").strip()[:400]
    return (f"PERGUNTA: {item['pergunta']}\n"
            f"FONTES_ESPERADAS: {fontes}\n"
            f"RESPOSTA DO ASSISTENTE: {resp}")


def _nota_esperada(ans_score: dict) -> float | None:
    """nota 1-5 a partir da distribuicao de `probabilities` (auditoria do score).

    As chaves de `probabilities` podem ser os proprios criterios ou indices/rotulos
    proprios do checkpoint; por isso a posicao do nivel e resolvida por
    `SCORE_CRITERIOS` quando a chave bate, e cai para a ordem de insercao do dict
    caso contrario. Devolve None se nao houver distribuicao.
    """
    probs = ans_score.get("probabilities") or {}
    if not probs:
        return None
    ordem = {c: i for i, c in enumerate(SCORE_CRITERIOS)}
    chaves = list(probs)
    pos = [ordem[c] if c in ordem else i for i, c in enumerate(chaves)]
    total = sum(float(v) for v in probs.values()) or 1.0
    return round(sum((p + 1) * float(probs[c]) for p, c in zip(pos, chaves)) / total, 2)


def avaliar(itens: list[dict], modelo: str) -> list[dict]:
    linhas = []
    for it in itens:
        try:
            data = _chamar(_montar_state(it), modelo)
            ans = data["answers"]
            noul = float(ans["correta"]["noul"])
            score = float(ans["nota"]["score"])  # nivel 0-based -> +1 = nota
            laya_correta = noul >= 0.5
            laya_nota = round(score + 1.0, 2)
            conf = ans["correta"].get("answer_confidence")
            conf = max(noul, 1.0 - noul) if conf is None else float(conf)
        except Exception as e:
            linhas.append({"id": it["id"], "erro": str(e)[:150]})
            continue
        pro = it.get("juiz_pro") or {}
        linhas.append({
            "id": it["id"], "estrato": it.get("estrato"),
            "pro_correta": pro.get("correta"), "pro_nota": pro.get("nota"),
            "laya_correta": laya_correta, "laya_noul": round(noul, 3),
            "laya_nota": laya_nota, "confianca": round(conf, 3),
            "nota_pelas_probs": _nota_esperada(ans["nota"]),
            "checkpoint": (data.get("routing") or {}).get("model"),
        })
        print(f"#{it['id']:>4} [{str(it.get('estrato'))[:9]:9s}] pro_correta={pro.get('correta')} "
              f"laya={laya_correta} conf={conf:.2f} nota pro={pro.get('nota')} "
              f"laya={laya_nota}", flush=True)
    return linhas


def _resumir(linhas: list[dict]) -> dict:
    ok = [l for l in linhas if "laya_correta" in l and l.get("pro_correta") is not None]
    n = len(ok)
    conc = sum(1 for l in ok if l["laya_correta"] == l["pro_correta"])
    conf = {}
    for l in ok:
        conf.setdefault(l["estrato"], [0, 0])
        conf[l["estrato"]][0] += int(l["laya_correta"] == l["pro_correta"])
        conf[l["estrato"]][1] += 1
    # matriz de confusao (linha=pro, col=laya)
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for l in ok:
        p, k = l["pro_correta"], l["laya_correta"]
        if p and k: cm["TP"] += 1
        elif not p and not k: cm["TN"] += 1
        elif not p and k: cm["FP"] += 1
        else: cm["FN"] += 1
    notas = [(l["pro_nota"], l["laya_nota"]) for l in ok
             if l.get("pro_nota") is not None and l.get("laya_nota") is not None]
    mae = round(sum(abs(p - k) for p, k in notas) / len(notas), 2) if notas else None
    # concordancia apenas nos casos de alta confianca (conf >= 0.7) — igual ao Kev
    altos = [l for l in ok if l["confianca"] >= 0.7]
    n_altos = len(altos)
    conc_alto = (sum(1 for l in altos if l["laya_correta"] == l["pro_correta"]) / n_altos
                 if n_altos else None)
    # ECE da noul (a promessa do Laya e probabilidade calibrada)
    pares = [(l["laya_noul"], float(l["laya_correta"])) for l in ok]
    ece = 0.0
    bins: dict[int, list[tuple[float, float]]] = {}
    for c, y in pares:
        bins.setdefault(min(int(c * 10), 9), []).append((c, y))
    for itens in bins.values():
        if n:
            ece += (len(itens) / n) * abs(sum(c for c, _ in itens) / len(itens)
                                          - sum(y for _, y in itens) / len(itens))
    return {
        "n": n,
        "concordancia_geral": round(conc / n, 3) if n else None,
        "concordancia_por_estrato": {e: round(a / t, 3) for e, (a, t) in sorted(conf.items())},
        "matriz_confusao_pro_vs_laya": cm,
        "n_altac_confianca": n_altos,
        "concordancia_altac_confianca": round(conc_alto, 3) if conc_alto is not None else None,
        "mae_nota": mae,
        "ece_noul_10bins": round(ece, 4) if n else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Laya como juiz — concordancia com o juiz pro (v0.7)")
    ap.add_argument("--sample", required=True, help="json com itens {id,pergunta,estrato,docs_esperados,resposta,juiz_pro}")
    ap.add_argument("--modelo", default=LAYA_MODELO)
    ap.add_argument("--saida", default=str(Path(__file__).resolve().parent.parent.parent /
                                           "data/processed/v07_ab/laya_juiz_concordancia.json"))
    a = ap.parse_args()

    itens = json.loads(Path(a.sample).read_text(encoding="utf-8"))
    print(f"sample: {len(itens)} itens · checkpoint={a.modelo}", flush=True)
    linhas = avaliar(itens, a.modelo)
    erros = [l for l in linhas if "erro" in l]
    print(f"avaliados: {len(linhas)} (erros: {len(erros)})", flush=True)
    for e in erros[:5]:
        print(f"  erro #{e['id']}: {e['erro']}")
    resumo = _resumir(linhas)
    out = {"modelo": a.modelo, "resumo": resumo, "itens": linhas}
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== RESUMO Laya-juiz ===")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"salvo em {a.saida}")


if __name__ == "__main__":
    main()