"""P3 · Avaliador — AB de prompts do SLM na rota simples (34Q que a cascata envia).

Compara variantes do prompt de sistema do gerador SLM nas 34 perguntas que a
cascata direciona para a rota simples (split CONGELADO do P2), com a mesma RAG
de P1 (RAG_DIR, pool 60) e o mesmo juiz R3 do avaliar_v5.

Variantes (prompts/v0.4/):
  v04 = rag_sistema_slm.txt      (producao — baseline historico P2: 0.606/8 aluc)
  p3a = rag_sistema_slm_p3a.txt  (grounding + criterio de abstencao escrito)
  p3b = rag_sistema_slm_p3b.txt  (p3a + few-shot de decisao responde/abstencao)

O v04 e re-medido na MESMA execucao/sessao p/ controlar a variancia do juiz:
a comparacao honesta e p3a/p3b vs v04-re-medido, com o 0.606/8 do P2 como
referencia historica.

Uso (Colab/local), um --variante por execucao (incremental por pergunta):
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_slm_p3.py --variante v04
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_slm_p3.py --variante p3a
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_slm_p3.py --variante p3b
  ... --variante p3a --fim 10   (parcela: numera as 34Q de 0..33)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from projeto_final import agentes, config, llm as llm_mod
from projeto_final.rag.pipeline import carregar_chunks

GOLDEN = config.RAG_GOLDEN_SET
SAIDA_DIR = config.PROCESSED_DIR / "v05b_ab"

# Split congelado do P2: perguntas que a cascata (limiar 0.60) enviou ao SLM.
# Fonte: data/processed/v05b_ab/cascade_v05b_p2.jsonl (rota == "simples").
FROZEN_SIMPLES_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 20, 21, 22, 23, 24,
                      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44]

PROMPTS = {
    "v04": "v0.4/rag_sistema_slm.txt",
    "p3a": "v0.4/rag_sistema_slm_p3a.txt",
    "p3b": "v0.4/rag_sistema_slm_p3b.txt",
}


def _juiz(pergunta: str, fontes: list[str], resposta: str, cliente):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from avaliar_v5 import juiz  # type: ignore
    return juiz(pergunta, fontes, resposta, cliente)


def avaliar(variante: str, incremental: bool = True,
            inicio: int = 0, fim: int | None = None) -> list[dict]:
    from projeto_final import slm  # noqa: F401  (carrega/valida SLM disponivel)

    rel_prompt = PROMPTS[variante]
    sistema = config.ler_prompt(rel_prompt) or agentes.SLM_SISTEMA_FALLBACK
    prompt_src = rel_prompt if config.ler_prompt(rel_prompt) else "FALLBACK"
    if not config.ler_prompt(rel_prompt):
        raise SystemExit(f"prompt nao encontrado: {rel_prompt}")

    perguntas = [
        q for q in json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
        if int(q["id"]) in FROZEN_SIMPLES_IDS
    ]
    perguntas.sort(key=lambda q: int(q["id"]))
    if fim is not None:
        perguntas = perguntas[inicio:fim]
    else:
        perguntas = perguntas[inicio:]

    chunks = carregar_chunks()
    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    saida = SAIDA_DIR / f"slm_p3_{variante}.jsonl"
    done_ids = set()
    if incremental and saida.exists():
        for l in saida.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(l)["id"])
            except Exception:
                pass

    linhas = []
    log = open(saida, "a") if incremental else None
    for q in perguntas:
        if q["id"] in done_ids:
            continue
        r = agentes._gerar_slm(q["pergunta"], chunks, top_k=5, sistema=sistema)
        resposta = (r.get("resposta") or "").strip()
        absteve = llm_mod.detectar_abstencao(resposta)
        esperados = sorted(q.get("docs_esperados", []))
        j = _juiz(q["pergunta"], esperados, resposta, cliente)
        linha = {
            "id": q["id"], "estrato": q["estrato"], "pergunta": q["pergunta"],
            "deve_abster": q.get("deve_abster", False),
            "docs_esperados": esperados,
            "absteve": absteve, "resposta": resposta[:400],
            "latencia_s": r.get("latencia_s"), "motivo": r.get("motivo"),
            "prompt_src": prompt_src, "juiz": j,
        }
        linhas.append(linha)
        if log:
            log.write(json.dumps(linha, ensure_ascii=False) + "\n")
            log.flush()
        print(f"#{q['id']:02d} absteve={absteve} deve={linha['deve_abster']} "
              f"juiz={j and j['correta']} aluc={(j or {}).get('alucinou')} "
              f"t={linha['latencia_s']}s", flush=True)
    if log:
        log.close()
    return linhas


def resumir(linhas: list[dict]) -> dict:
    julg = [r for r in linhas if r.get("juiz")]
    n = len(linhas)
    acerto = sum(1 for r in julg if r["juiz"]["correta"])
    aluc = sum(1 for r in julg if r["juiz"]["alucinou"])
    abs_ok = sum(1 for r in linhas if r["absteve"] == r["deve_abster"])
    cit = sum(1 for r in linhas if not r["absteve"] and r["resposta"])
    return {
        "n": n, "acuracia": round(acerto / max(1, len(julg)), 3),
        "acertos": acerto, "julgadas": len(julg),
        "abstencao_correta": round(abs_ok / n, 3),
        "alucinacoes": aluc,
        "nota_media": round(sum(r["juiz"]["nota"] for r in julg) / max(1, len(julg)), 2),
        "citacao": round(cit / max(1, sum(1 for r in linhas if not r["absteve"])), 3),
        "respostas_abstidas": sum(1 for r in linhas if r["absteve"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 — AB de prompts do SLM na rota simples")
    ap.add_argument("--variante", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--inicio", type=int, default=0)
    ap.add_argument("--fim", type=int, default=None)
    ap.add_argument("--sem-incremental", action="store_true",
                    help="nao roda: apenas consolida o jsonl existente em resumo")
    a = ap.parse_args()

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    saida = SAIDA_DIR / f"slm_p3_{a.variante}.jsonl"
    if a.sem_incremental:
        linhas = [json.loads(l) for l in saida.read_text(encoding="utf-8").splitlines()]
    else:
        avaliar(a.variante, inicio=a.inicio, fim=a.fim)
        linhas = [json.loads(l) for l in saida.read_text(encoding="utf-8").splitlines()]
    resumo = resumir(linhas)
    print(f"\n=== P3 [{a.variante}] RESUMO ===")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    (SAIDA_DIR / f"slm_p3_{a.variante}_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()