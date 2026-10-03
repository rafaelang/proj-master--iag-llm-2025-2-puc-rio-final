"""Cenario de teste — Laya (convaiinnovations/laya) como roteador SIMPLES/COMPLEXA.

Repete a MESMA bateria de `avaliar_roteador_kev.py` (Kev -> Laya) sobre o
dataset do estudo do roteador (`roteador_rotulos.json`: 20 canonicas + 14 typos),
para comparar no mesmo protocolo: choice vs noul, metricas identicas, sweep de
limiar (conceito cascade) e comparativo com todos os backends ja medidos
(tfidf/slm/flash em `avaliacao_roteador.json`, mais as linhas do Kev).

Diferencas de contrato em relacao ao Kev (mesmo endpoint `POST /v1/systemone`):
  - `model` nomeia o checkpoint (`english` | `multilingual` | `typed-decisions`);
    o trafego aqui e PT-BR, entao o default e `multilingual`.
  - a confianca calibrada e `answer_confidence` (a probabilidade da resposta
    reportada); `confidence` a parte e 1 - entropia normalizada e NAO transfere
    threshold. Usamos `answer_confidence` como p_max (equivale ao p_max do Kev).
  - extra: ECE (expected calibration error) e Brier, porque a promessa do Laya e
    probabilidade calibrada (RLCD) — e o sweep de limiar so interpreta se ela
    for confiavel.

Uso (endpoint laya no ar):
  LAYA_BASE_URL=http://127.0.0.1:8010 python scripts/avaliadores/avaliar_roteador_laya.py
  LAYA_BASE_URL=http://127.0.0.1:8010 python scripts/avaliadores/avaliar_roteador_laya.py \
      --questao noul --modelo typed-decisions

Saida (fora do Git): data/processed/rag_v4/avaliacao_roteador_laya_<modelo>_<questao>.{json,jsonl}
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]  # projeto_final/
DEFAULT_DATASET = RAIZ / "data" / "processed" / "rag_v4" / "roteador_rotulos.json"
DEFAULT_COMPARATIVO = RAIZ / "data" / "processed" / "rag_v4" / "avaliacao_roteador.json"
SAIDA_DIR = RAIZ / "data" / "processed" / "rag_v4"

LAYA_BASE_URL = os.getenv("LAYA_BASE_URL", "http://127.0.0.1:8010")
LAYA_MODELO = os.getenv("LAYA_MODELO", "multilingual")
LAYA_API_KEY = os.getenv("LAYA_API_KEY", "")
LAYA_TIMEOUT_S = int(os.getenv("LAYA_TIMEOUT_S", "120"))

# Mesma definicao de criterio do Kev (SIMPLES=1 fonte factual;
# COMPLEXA=composta/comparativa/ambigua/multifonte).
CRITERIOS = {
    "simples": "Pergunta factual e direta que pode ser respondida com uma unica fonte.",
    "complexa": "Pergunta composta, comparativa, ambigua ou que exige raciocinio sobre multiplas fontes.",
}
INSTRUCAO = (
    "Classifique a pergunta do assistente do Master IAG e LLM da PUC-Rio "
    "como SIMPLES ou COMPLEXA."
)
INSTRUCAO_NOUL = (
    "A pergunta e COMPLEXA? COMPLEXA = composta, comparativa, ambigua ou que "
    "exige raciocinio sobre multiplas fontes. Responda sim se for COMPLEXA, "
    "nao se for SIMPLES."
)


def _metricas(rows: list[dict]) -> dict:
    """Idêntica a avaliar_roteador._metricas e avaliar_roteador_kev._metricas."""
    por_tipo = {"canonica": [0, 0], "typo": [0, 0]}
    conf = {"VP_S": 0, "VP_C": 0, "falso_S": 0, "falso_C": 0}
    for r in rows:
        certa = r["pred"] == r["esperado"]
        por_tipo[r["tipo"]][0] += int(certa)
        por_tipo[r["tipo"]][1] += 1
        conf["VP_S" if r["esperado"] == "SIMPLES" else "VP_C"] += int(certa)
        if not certa:
            if r["esperado"] == "COMPLEXA" and r["pred"] == "SIMPLES":
                conf["falso_S"] += 1
            elif r["esperado"] == "SIMPLES" and r["pred"] == "COMPLEXA":
                conf["falso_C"] += 1

    def _acc(campo):
        a, n = por_tipo[campo]
        return round(a / n, 3) if n else None

    n_total = len(rows)
    acertos = conf["VP_S"] + conf["VP_C"]
    return {
        "acuracia_canonicas": _acc("canonica"),
        "acuracia_typos": _acc("typo"),
        "acuracia_total": round(acertos / n_total, 3) if n_total else None,
        "falso_simples": conf["falso_S"],
        "falso_complexa": conf["falso_C"],
        "acertos": {"simples": conf["VP_S"], "complexa": conf["VP_C"]},
        "n": n_total,
    }


def _calibracao(rows: list[dict]) -> dict:
    """ECE (10 bins) + Brier sobre a confianca da classe predita."""
    pares = [(r["conf"], float(r["pred"] == r["esperado"])) for r in rows]
    n = len(pares)
    brier = round(sum((c - y) ** 2 for c, y in pares) / n, 4) if n else None
    bins: dict[int, list[tuple[float, float]]] = {}
    for c, y in pares:
        bins.setdefault(min(int(c * 10), 9), []).append((c, y))
    ece, detalhe = 0.0, {}
    for b, itens in sorted(bins.items()):
        conf_b = sum(c for c, _ in itens) / len(itens)
        acc_b = sum(y for _, y in itens) / len(itens)
        ece += (len(itens) / n) * abs(conf_b - acc_b)
        detalhe[f"{b / 10:.1f}-{(b + 1) / 10:.1f}"] = {
            "n": len(itens), "conf_media": round(conf_b, 3), "acuracia": round(acc_b, 3)}
    return {"ece_10bins": round(ece, 4), "brier": brier, "bins": detalhe}


def _questoes(questao: str) -> dict:
    if questao == "choice":
        return {"rota": {"type": "choice", "instructions": INSTRUCAO, "criteria": CRITERIOS}}
    return {"complexa": {"type": "noul", "instructions": INSTRUCAO_NOUL}}


def _confianca(ans: dict) -> float:
    """answer_confidence (calibrada) > max(probabilities) > 0.5."""
    ac = ans.get("answer_confidence")
    if ac is not None:
        return float(ac)
    probs = ans.get("probabilities") or {}
    if probs:
        return max(float(v) for v in probs.values())
    return 0.5


def _classificar(texto: str, questao: str, modelo: str) -> tuple[str, float, dict]:
    """Chama o laya-serve e retorna (pred, p_max, resposta_bruta). Falha -> SIMPLES."""
    payload = {"state": texto, "model": modelo, "questions": _questoes(questao)}
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
        data = json.loads(resp.read().decode("utf-8"))
    ans = data["answers"]
    if questao == "choice":
        a = ans["rota"]
        pred = "COMPLEXA" if a["choice"] == "complexa" else "SIMPLES"
        p_max = _confianca(a)
    else:
        noul = float(ans["complexa"]["noul"])
        pred = "COMPLEXA" if noul >= 0.5 else "SIMPLES"
        p_max = _confianca(ans["complexa"]) if ans["complexa"].get("answer_confidence") \
            is not None else max(noul, 1.0 - noul)
    return pred, p_max, data


def _sweep_limiar(rows: list[dict]) -> list[dict]:
    """Idêntico ao Kev: decide sozinho se p_max >= limiar; senao escala."""
    out = []
    for lim in [0.50, 0.60, 0.70, 0.80, 0.90]:
        auto = [r for r in rows if r["conf"] >= lim]
        escala = len(rows) - len(auto)
        certos = sum(1 for r in auto if r["pred"] == r["esperado"])
        out.append({
            "limiar": lim,
            "n_auto": len(auto),
            "taxa_auto": round(len(auto) / max(1, len(rows)), 3),
            "acuracia_auto": round(certos / max(1, len(auto)), 3) if auto else None,
            "n_escala": escala,
        })
    return out


def _comparativo() -> dict | None:
    """Backends de avaliar_roteador.py + linhas ja medidas do Kev (mesmo dataset)."""
    out: dict[str, dict] = {}
    p = DEFAULT_COMPARATIVO
    if p.exists():
        dados = json.loads(p.read_text(encoding="utf-8"))
        for b, r in dados.get("resultados", {}).items():
            out[b] = {
                "acuracia_canonicas": r.get("acuracia_canonicas"),
                "acuracia_typos": r.get("acuracia_typos"),
                "acuracia_total": r.get("acuracia_total"),
                "falso_simples": r.get("falso_simples"),
                "falso_complexa": r.get("falso_complexa"),
                "latencia_mediana_s": r.get("latencias", {}).get("mediana_s"),
            }
    for q in ("choice", "noul"):
        kp = SAIDA_DIR / f"avaliacao_roteador_kev_{q}.json"
        if kp.exists():
            k = json.loads(kp.read_text(encoding="utf-8"))
            m, lat = k["metricas"], k["latencias"]["mediana_s"]
            out[f"kev-{q}"] = {
                "acuracia_canonicas": m["acuracia_canonicas"],
                "acuracia_typos": m["acuracia_typos"],
                "acuracia_total": m["acuracia_total"],
                "falso_simples": m["falso_simples"],
                "falso_complexa": m["falso_complexa"],
                "latencia_mediana_s": lat,
            }
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliacao isolada do roteador Laya (SIMPLES/COMPLEXA)")
    ap.add_argument("--questao", choices=["choice", "noul"], default="choice")
    ap.add_argument("--modelo", default=LAYA_MODELO,
                    help="checkpoint: english | multilingual | typed-decisions")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--saida-json", default=None)
    ap.add_argument("--saida-jsonl", default=None)
    a = ap.parse_args()

    dados = json.loads(Path(a.dataset).read_text(encoding="utf-8"))
    itens = dados["itens"]
    print(f"Dataset: {len(itens)} itens (canonicas={dados['meta']['n_canonicas']} "
          f"+ typos={dados['meta']['n_typos']}) · questao={a.questao} · "
          f"modelo={a.modelo}", flush=True)

    rows: list[dict] = []
    latencias: list[float] = []
    erros = 0
    roteados: dict[str, int] = {}
    for item in itens:
        t0 = time.perf_counter()
        try:
            pred, p_max, bruto = _classificar(item["texto"], a.questao, a.modelo)
            ck = ((bruto.get("routing") or {}).get("model")) or "?"
            roteados[ck] = roteados.get(ck, 0) + 1
        except Exception as e:
            pred, p_max, bruto, erros = "SIMPLES", 0.0, {"erro": str(e)}, erros + 1
        dt = time.perf_counter() - t0
        latencias.append(dt)
        rows.append({
            "id": item["id"], "tipo": item["tipo"], "texto": item["texto"],
            "esperado": item["rotulo"], "pred": pred, "conf": round(p_max, 3),
            "latencia_s": round(dt, 3),
        })
        print(f"#{item['id']:02d} [{item['tipo']:8s}] esperado={item['rotulo']:8s} "
              f"pred={pred:8s} conf={p_max:.2f} ({dt:.2f}s)", flush=True)

    met = _metricas(rows)
    confs = [r["conf"] for r in rows]
    resumo = {
        "meta": {
            "descricao": "Laya como roteador SIMPLES/COMPLEXA (cenario isolado)",
            "questao": a.questao,
            "checkpoint": a.modelo,
            "base_url": LAYA_BASE_URL,
            "dataset": str(Path(a.dataset)),
            "criterio": dados["meta"]["criterio"],
            "checkpoint_servido": roteados,
        },
        "metricas": met,
        "calibracao": _calibracao(rows),
        "latencias": {
            "mediana_s": round(statistics.median(latencias), 4),
            "media_s": round(sum(latencias) / len(latencias), 4),
        },
        "erros_excecao": erros,
        "confianca": {
            "media": round(sum(confs) / len(confs), 3),
            "mediana": round(statistics.median(confs), 3),
            "min": round(min(confs), 3),
            "max": round(max(confs), 3),
        },
        "sweep_limiar": _sweep_limiar(rows),
        "comparativo": _comparativo(),
    }

    tag = f"avaliacao_roteador_laya_{a.modelo}_{a.questao}"
    saida_json = a.saida_json or SAIDA_DIR / f"{tag}.json"
    saida_jsonl = a.saida_jsonl or SAIDA_DIR / f"{tag}.jsonl"
    Path(saida_json).parent.mkdir(parents=True, exist_ok=True)
    Path(saida_json).write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(saida_jsonl).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== RESUMO (Laya {a.modelo}/{a.questao}) ===")
    print(json.dumps({k: v for k, v in resumo.items() if k != "comparativo"},
                     ensure_ascii=False, indent=2))
    print("\n=== COMPARATIVO (acuracia total · falso-S/falso-C · lat mediana) ===")
    print("| backend | acc can | acc typos | acc tot | falso-S | falso-C | lat (s) |")
    print("|---|---|---|---|---|---|---|")
    cmp = resumo["comparativo"] or {}
    for b, r in sorted(cmp.items(), key=lambda kv: -(kv[1]["acuracia_total"] or 0)):
        print(f"| {b} | {r['acuracia_canonicas']} | {r['acuracia_typos']} | {r['acuracia_total']} "
              f"| {r['falso_simples']} | {r['falso_complexa']} | {r['latencia_mediana_s']} |")
    print(f"| laya-{a.modelo}-{a.questao} | {met['acuracia_canonicas']} | {met['acuracia_typos']} "
          f"| {met['acuracia_total']} | {met['falso_simples']} | {met['falso_complexa']} "
          f"| {resumo['latencias']['mediana_s']} |")
    print(f"\njson salvo em: {saida_json}")


if __name__ == "__main__":
    main()