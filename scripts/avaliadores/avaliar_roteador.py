"""Avaliacao ISOLADA do roteador — R1 (slm) x R2 (tfidf) x flash x R3 (cascade).

Usa o dataset do estudo (gerar_dataset_roteador.py -> roteador_rotulos.json):
20 canonicas + 14 typos. NAO e golden set da release.

R3 (cascade) e SIMULADA a partir das medicoes reais: R2 decide pela margem
(predict_proba); abaixo do limiar (INDETERMINADO) usa a predicao REAL do R1
(slm) ja medida no mesmo run — varre o limiar sem novas chamadas ao SLM/API.

Saida (fora do Git): data/processed/rag_v4/avaliacao_roteador.json
Uso:
  python scripts/avaliadores/avaliar_roteador.py [--backends slm tfidf flash]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from projeto_final import agentes, config, roteador_tfidf

ROTULOS_JSON = config.PROCESSED_DIR / "rag_v4" / "roteador_rotulos.json"
SAIDA_JSON = config.PROCESSED_DIR / "rag_v4" / "avaliacao_roteador.json"
LAT_SLM_REF = None  # preenchido apos medir o slm
LAT_TFIDF_REF = None
MEMO = {
    "tfidf": lambda: roteador_tfidf.tamanho_artefato_bytes() or 0,
    "slm": lambda: Path(config.SLM_MODELO_PATH).stat().st_size
    if Path(config.SLM_MODELO_PATH).exists() else 0,
    "flash": lambda: 0,
}


def _metricas(rows: list[dict]) -> dict:
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


def _avaliar(backend: str, itens: list[dict]) -> dict:
    rows: list[dict] = []
    latencias: list[float] = []
    erros = 0
    for item in itens:
        t0 = time.perf_counter()
        try:
            pred = agentes.classificar_rota(item["texto"], roteador=backend)
        except Exception as exc:
            pred = "SIMPLES"
            erros += 1
            _ = exc
        latencias.append(time.perf_counter() - t0)
        rows.append({"tipo": item["tipo"], "esperado": item["rotulo"], "pred": pred})
    met = _metricas(rows)
    met.update({
        "backend": backend,
        "latencias": {
            "mediana_s": round(statistics.median(latencias), 4),
            "media_s": round(sum(latencias) / len(latencias), 4),
        },
        "erros_excecao": erros,
        "memoria_artefato_bytes": MEMO[backend](),
        "memoria_proxy": ("pickle(modelo)" if backend == "tfidf"
                          else ("arquivo GGUF" if backend == "slm" else "0 (API)")),
        "predicoes": rows,
    })
    return met


def _cv_estratificada_tfidf() -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    from projeto_final import roteador_tfidf as rt

    pares = [(rt._normalizar(p), r) for p, r in rt.ROTULOS.items()]
    textos = [p for p, _ in pares]
    num = {"SIMPLES": 0, "COMPLEXA": 1}
    y = [num[r] for _, r in pares]
    acertos = total = 0
    falso_s = 0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for treino_idx, teste_idx in skf.split(textos, y):
        vetor = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=1000)
        clf = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1,
                            subsample=1.0, colsample_bytree=1.0,
                            random_state=0, n_jobs=1, eval_metric="logloss")
        Xtr = vetor.fit_transform([textos[i] for i in treino_idx])
        clf.fit(Xtr, [y[i] for i in treino_idx])
        pred = clf.predict(vetor.transform([textos[i] for i in teste_idx]))
        for j, p in zip(teste_idx, pred):
            total += 1
            certa = int(p) == y[j]
            acertos += int(certa)
            if not certa and y[j] == 1 and int(p) == 0:
                falso_s += 1
    return {"folds": skf.get_n_splits(),
            "acuracia_media": round(acertos / total, 3) if total else None,
            "falso_simples_total": falso_s}


# ------------------------------------------------------------- cascata simulada (R3)

def _simular_cascade(itens: list[dict], pred_slm: list[str],
                     limiar: float, lat_slm_s: float, lat_tfidf_s: float) -> dict:
    """R2 decide; INDETERMINADO (margem < limiar) usa a predicao real do R1."""
    escaladas = 0
    rows: list[dict] = []
    for i, item in enumerate(itens):
        pred = roteador_tfidf.classificar_limiar(item["texto"], limiar)
        if pred == "INDETERMINADO":
            pred = pred_slm[i]
            escaladas += 1
        rows.append({"tipo": item["tipo"], "esperado": item["rotulo"], "pred": pred})
    met = _metricas(rows)
    n = max(len(rows), 1)
    lat_mix = (escaladas * lat_slm_s + (n - escaladas) * lat_tfidf_s) / n
    met.update({
        "limiar": limiar,
        "escaladas_r1": escaladas,
        "taxa_escalada": round(escaladas / n, 3),
        "latencia_mix_media_s": round(lat_mix, 4),
    })
    return met


def main() -> None:
    global LAT_SLM_REF, LAT_TFIDF_REF
    ap = argparse.ArgumentParser(description="Avaliacao isolada do roteador (R1 x R2 x R3)")
    ap.add_argument("--backends", nargs="*", default=["slm", "tfidf", "flash"],
                    choices=["slm", "tfidf", "flash", "cascade"])
    ap.add_argument("--limiares", nargs="*", type=float,
                    default=[0.60, 0.65, 0.70, 0.75, 0.80])
    args = ap.parse_args()

    dados = json.loads(ROTULOS_JSON.read_text(encoding="utf-8"))
    itens = dados["itens"]
    print(f"Dataset do estudo: {len(itens)} itens "
          f"(canonicas={dados['meta']['n_canonicas']} + typos={dados['meta']['n_typos']})")

    resultados = {}
    for backend in args.backends:
        if backend == "cascade":
            continue  # R3 e simulada abaixo (evita novas chamadas ao SLM)
        print(f"\n== avaliando backend: {backend} ==", flush=True)
        r = _avaliar(backend, itens)
        resultados[backend] = r
        print(json.dumps({k: v for k, v in r.items() if k != "predicoes"},
                         ensure_ascii=False, indent=2))
        if backend == "flash":
            time.sleep(2)

    if "slm" in resultados:
        LAT_SLM_REF = resultados["slm"]["latencias"]["mediana_s"]
    if "tfidf" in resultados:
        LAT_TFIDF_REF = resultados["tfidf"]["latencias"]["mediana_s"]

    cascatas = []
    if "slm" in resultados and "tfidf" in resultados:
        pred_slm = [r["pred"] for r in resultados["slm"]["predicoes"]]
        print("\n== R3 (cascata simulada: R2 decide; INDETERMINADO -> R1 slm) ==")
        for lim in args.limiares:
            c = _simular_cascade(itens, pred_slm, lim, LAT_SLM_REF, LAT_TFIDF_REF)
            cascatas.append(c)
            print(json.dumps({k: v for k, v in c.items() if k not in ("predicoes",)},
                             ensure_ascii=False))

    payload = {
        "meta": dados["meta"],
        "resultados": resultados,
        "cascade_simulada": cascatas,
    }
    if "tfidf" in resultados:
        payload["cv_estratificada_tfidf"] = _cv_estratificada_tfidf()
    SAIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== RESUMO (isolado) ==")
    print("| backend | acc can | acc typos | acc tot | falso-S | lat med (s) | memoria (B) |")
    print("|---|---|---|---|---|---|---|")
    for b, r in sorted(resultados.items()):
        print(f"| {b} | {r['acuracia_canonicas']} | {r['acuracia_typos']} "
              f"| {r['acuracia_total']} | {r['falso_simples']} "
              f"| {r['latencias']['mediana_s']} | {r['memoria_artefato_bytes']} |")
    print("\n== RESUMO (R3 cascata por limiar) ==")
    print("| limiar | acc can | acc typos | acc tot | falso-S | escalada % | lat mix (s) |")
    print("|---|---|---|---|---|---|---|")
    for c in cascatas:
        print(f"| {c['limiar']} | {c['acuracia_canonicas']} | {c['acuracia_typos']} "
              f"| {c['acuracia_total']} | {c['falso_simples']} "
              f"| {c['taxa_escalada']} | {c['latencia_mix_media_s']} |")
    print(f"\njson salvo em: {SAIDA_JSON}")


if __name__ == "__main__":
    main()

