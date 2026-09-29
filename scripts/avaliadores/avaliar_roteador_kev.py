"""Cenario de teste — Kev (jaredpalmer/kev) como roteador SIMPLES/COMPLEXA.

Avaliacao ISOLADA do roteador Kev (modelo de decisao estilo Jev/TypeSafe,
servido via POST /v1/systemone) no dataset do estudo do roteador
(roteador_rotulos.json: 20 canonicas + 14 typos), comparado aos backends ja
medidos em avaliar_roteador.py (slm / tfidf / flash — avaliacao_roteador.json).

Duas variantes de questao (--questao choice|noul):
  choice -> criterios simples/complexa (rotulo direto + probabilities/confidence)
  noul   -> "A pergunta e COMPLEXA?" (probabilidade calibrada; >= 0.5 = COMPLEXA)

Metricas identicas a avaliar_roteador.py (_metricas, copiada aqui para o script
ficar auto-contido e rodar no Colab sem importar o pacote) + extras Kev:
confianca media (p_max) e varredura de limiar de confianca (conceito cascade:
decide sozinho se conf >= limiar; senao escala — nao conta na acuracia auto).

Uso (endpoint Kev no ar):
  KEV_BASE_URL=http://127.0.0.1:8009 python scripts/avaliadores/avaliar_roteador_kev.py
  python scripts/avaliadores/avaliar_roteador_kev.py --questao noul

Saida (fora do Git): data/processed/rag_v4/avaliacao_roteador_kev_{choice,noul}.json
(+ .jsonl por item, com probabilidades).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]  # projeto_final/
DEFAULT_DATASET = RAIZ / "data" / "processed" / "rag_v4" / "roteador_rotulos.json"
DEFAULT_COMPARATIVO = RAIZ / "data" / "processed" / "rag_v4" / "avaliacao_roteador.json"
SAIDA_DIR = RAIZ / "data" / "processed" / "rag_v4"

KEV_BASE_URL = os.getenv("KEV_BASE_URL", "http://127.0.0.1:8009")
KEV_MODEL = os.getenv("KEV_MODEL", "kev-latest")
KEV_API_KEY = os.getenv("KEV_API_KEY", "")
KEV_TIMEOUT_S = int(os.getenv("KEV_TIMEOUT_S", "120"))

# Mesma definicao de criterio dos roteadores atuais (SIMPLES=1 fonte factual;
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
    """Idêntica a avaliar_roteador._metricas (copiada p/ autoconterencia)."""
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


def _questoes(questao: str) -> dict:
    if questao == "choice":
        return {"rota": {"type": "choice", "instructions": INSTRUCAO, "criteria": CRITERIOS}}
    return {"complexa": {"type": "noul", "instructions": INSTRUCAO_NOUL}}


def _classificar(texto: str, questao: str) -> tuple[str, float, dict]:
    """Chama o Kev e retorna (pred, p_max, resposta_bruta). Falha -> SIMPLES."""
    payload = {"state": texto, "model": KEV_MODEL, "questions": _questoes(questao)}
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
        data = json.loads(resp.read().decode("utf-8"))
    ans = data["answers"]
    if questao == "choice":
        a = ans["rota"]
        pred = "COMPLEXA" if a["choice"] == "complexa" else "SIMPLES"
        probs = a.get("probabilities") or {"simples": 0.0, "complexa": 0.0}
        p_max = max(float(v) for v in probs.values())
    else:
        noul = float(ans["complexa"]["noul"])
        pred = "COMPLEXA" if noul >= 0.5 else "SIMPLES"
        probs = {"simples": 1.0 - noul, "complexa": noul}
        p_max = max(probs["simples"], probs["complexa"])
    return pred, p_max, data


def _sweep_limiar(rows: list[dict]) -> list[dict]:
    """Decide sozinho se p_max >= limiar; senao escala (fora da acuracia auto)."""
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
    p = DEFAULT_COMPARATIVO
    if not p.exists():
        return None
    dados = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for b, r in dados.get("resultados", {}).items():
        out[b] = {
            "acuracia_canonicas": r.get("acuracia_canonicas"),
            "acuracia_typos": r.get("acuracia_typos"),
            "acuracia_total": r.get("acuracia_total"),
            "falso_simples": r.get("falso_simples"),
            "falso_complexa": r.get("falso_complexa"),
            "latencia_mediana_s": r.get("latencias", {}).get("mediana_s"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliacao isolada do roteador Kev (SIMPLES/COMPLEXA)")
    ap.add_argument("--questao", choices=["choice", "noul"], default="choice")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--saida-json", default=None)
    ap.add_argument("--saida-jsonl", default=None)
    a = ap.parse_args()

    dados = json.loads(Path(a.dataset).read_text(encoding="utf-8"))
    itens = dados["itens"]
    print(f"Dataset: {len(itens)} itens (canonicas={dados['meta']['n_canonicas']} "
          f"+ typos={dados['meta']['n_typos']}) · questao={a.questao} · modelo={KEV_MODEL}", flush=True)

    rows: list[dict] = []
    latencias: list[float] = []
    erros = 0
    for item in itens:
        t0 = time.perf_counter()
        try:
            pred, p_max, bruto = _classificar(item["texto"], a.questao)
        except Exception as e:
            pred, p_max, bruto, erros = "SIMPLES", 0.0, {"erro": str(e)}, erros + 1
            _ = bruto
        latencias.append(time.perf_counter() - t0)
        rows.append({
            "id": item["id"], "tipo": item["tipo"], "texto": item["texto"],
            "esperado": item["rotulo"], "pred": pred, "conf": round(p_max, 3),
            "latencia_s": round(time.perf_counter() - t0, 3),
        })
        print(f"#{item['id']:02d} [{item['tipo']:8s}] esperado={item['rotulo']:8s} "
              f"pred={pred:8s} conf={p_max:.2f}", flush=True)

    met = _metricas(rows)
    confs = [r["conf"] for r in rows]
    resumo = {
        "meta": {
            "descricao": "Kev como roteador SIMPLES/COMPLEXA (cenario isolado)",
            "questao": a.questao,
            "modelo": KEV_MODEL,
            "base_url": KEV_BASE_URL,
            "dataset": str(Path(a.dataset)),
            "criterio": dados["meta"]["criterio"],
        },
        "metricas": met,
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

    saida_json = a.saida_json or SAIDA_DIR / f"avaliacao_roteador_kev_{a.questao}.json"
    saida_jsonl = a.saida_jsonl or SAIDA_DIR / f"avaliacao_roteador_kev_{a.questao}.jsonl"
    Path(saida_json).parent.mkdir(parents=True, exist_ok=True)
    Path(saida_json).write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(saida_jsonl).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== RESUMO (Kev) ===")
    print(json.dumps({k: v for k, v in resumo.items() if k not in ("comparativo",)},
                     ensure_ascii=False, indent=2))
    print("\n=== COMPARATIVO (acuracia total · falso-S/falso-C · lat mediana) ===")
    print("| backend | acc can | acc typos | acc tot | falso-S | falso-C | lat (s) |")
    print("|---|---|---|---|---|---|---|")
    cmp = resumo["comparativo"] or {}
    for b, r in sorted(cmp.items()):
        print(f"| {b} | {r['acuracia_canonicas']} | {r['acuracia_typos']} | {r['acuracia_total']} "
              f"| {r['falso_simples']} | {r['falso_complexa']} | {r['latencia_mediana_s']} |")
    print(f"| kev-{a.questao} | {met['acuracia_canonicas']} | {met['acuracia_typos']} | {met['acuracia_total']} "
          f"| {met['falso_simples']} | {met['falso_complexa']} | {resumo['latencias']['mediana_s']} |")
    print(f"\njson salvo em: {saida_json}")


if __name__ == "__main__":
    main()