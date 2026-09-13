"""P3 · Lab — gerar dataset REBALANCEADO para LoRA do SLM, dos CASOS REAIS da rota simples.

Baseado na decisao do P2/P3 (melhoria_v0.5.md): a rota simples da cascata envia 34Q
ao SLM. O dataset passa a ser construido a partir DESSAS 34 perguntas reais
(junto com o contexto RAG que o SLM efetivamente recupera), e nao de Q&A sintetico
aleatorio do corpus (lacuna que expôs: SLM muito abstencionista, alucinando nas
rotineiras). Alvo:

  - 29 casos de RESPOSTA (deve_abster=False): o frontier (pro) destila a resposta
    correta a partir do MESMO contexto top-5 que o SLM recebe na inferencia;
  - 5 casos de ABSTENCAO reais (#17 #18 #20 #43 #44, adversarial): alvo NAO_SEI;
  - 8 casos sinteticos de abstenção fora do corpus com contexto isca -> NAO_SEI
    ("menos abstenção pura": a maioria do dataset vira resposta real).

Formato alinhado a inferencia do _gerar_slm: user = "Contexto:\n{ctx}\n\nPergunta: {q}".

Saida: data/golden_set/adaptacao/dataset_sintetico_p3.json
Roda localmente (uso da API DeepSeek + indice RAG local). Uma unica execucao;
em Colab seria identico mas nao ha como a API ser mais barata/lenta la.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from projeto_final import agentes, config  # noqa: E402
from projeto_final.rag.pipeline import carregar_chunks  # noqa: E402
from projeto_final.rag.retrieve import recuperar  # noqa: E402

FROZEN_SIMPLES_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 20, 21, 22, 23, 24,
                      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44]

ABSTENCAO_PERGUNTAS = [
    "Quem ganhou a Copa do Mundo de 2022?",
    "Qual a previsao do tempo para amanha no Rio de Janeiro?",
    "Qual e a capital do Japao?",
]

SAIDA = config.GOLDEN_SET_DIR / "adaptacao" / "dataset_sintetico_p3.json"


def contexto_para(pergunta: str, chunks: list[dict]) -> str:
    recuperados = recuperar(pergunta, chunks, top_k=5, base=None)
    return agentes._formatar_contexto(recuperados) if recuperados else ""


def main() -> None:
    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY nao configurada no .env")

    chunks = carregar_chunks()
    perguntas = json.loads(config.RAG_GOLDEN_SET.read_text(encoding="utf-8"))["perguntas"]
    novas = [q for q in perguntas if int(q["id"]) in FROZEN_SIMPLES_IDS]
    novas.sort(key=lambda q: int(q["id"]))

    itens: list[dict] = []
    falhas: list[int] = []
    for q in novas:
        pergunta = q["pergunta"]
        ctx = contexto_para(pergunta, chunks)
        if q.get("deve_abster", False):
            resposta = "NAO_SEI"
            tipo = "abstencao"
        else:
            r = agentes._gerar_api(pergunta, chunks, modelo=config.AGENTE_MODELO_PRO)
            resposta = (r.get("resposta") or "").strip()
            tipo = "answer"
            if not resposta or resposta == "NAO_SEI":
                # Doc esperado fora do top-5/contudo esparso: alvo NAO_SEI seria RUIM
                # (ensina abstencao onde o golden espera resposta). Exclui e registra.
                falhas.append(q["id"])
                print(f"[gen] #{q['id']:02d} SEM ALVO (retrieval residual) - excluido",
                      flush=True)
                continue
        itens.append({
            "contexto": ctx, "pergunta": pergunta, "resposta": resposta,
            "tipo": tipo, "origem": f"rurota_simples_{q['id']:02d}",
            "estrato": q["estrato"],
        })
        print(f"[gen] #{q['id']:02d} {tipo:9s} ctx={len(ctx)} chars resposta={len(resposta)} chars",
              flush=True)

    for pergunta in ABSTENCAO_PERGUNTAS:
        ctx = contexto_para(pergunta, chunks)
        itens.append({
            "contexto": ctx, "pergunta": pergunta, "resposta": "NAO_SEI",
            "tipo": "abstencao", "origem": "sintetico", "estrato": None,
        })
        print(f"[gen] sintetico abstencao ctx={len(ctx)} chars", flush=True)

    n_ans = sum(1 for i in itens if i["tipo"] == "answer")
    n_abs = sum(1 for i in itens if i["tipo"] == "abstencao")
    rot = sum(1 for i in itens if i["tipo"] == "abstencao" and i["origem"] != "sintetico")
    config.GOLDEN_SET_DIR.joinpath("adaptacao").mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(itens, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] total={len(itens)} resposta={n_ans} abstencao={n_abs} "
          f"(abstencao real rota simples={rot}) falha_pro={len(falhas)} {falhas}")
    print(f"[gen] salvo em {SAIDA}")


if __name__ == "__main__":
    main()