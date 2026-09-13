# v0.6 · Re-medição P1/P2 no golden expandido (132Q)

**Por que:** o golden v0.6 (`perguntas_v06.json`, 132 casos — P5) tem CI bem menor
que o v0.5b (44Q). Re-medir P1 (retrieval isolado) e P2 (fluxo cascade completo)
neste golden **sem re-editar as 44Q congeladas** (critério de rótulo congelado em
`data/golden_set/adaptacao/criterio_rotulo.md`) é o pré-requisito para fechar a v0.6.
Golden v0.6: **132Q = 44 congeladas (IDs 1–44) + 88 novas (IDs 45+)**, estratos:
rotineira 69 / composta 29 / negativa 16 / adversarial 18.

---

## 1. P1 — Retrieval isolado (sem LLM) · `scripts/avaliadores/avaliar_retrieval_v05b.py`

`RAG_GOLDEN_SET_FILE=perguntas_v06.json` → `data/processed/rag/retrieval_v06.json`

| Métrica | v0.5b (44Q) | v0.6 (132Q) | Δ |
|---|---|---|---|
| **Recall@5** | 0.737 | **0.763** (114 com docs) | +0.026 |
| **MRR** | — | **0.592** | — |

Por estrato (v0.6, recall@5):

| Estrato | n | recall@5 | MRR |
|---|---|---|---|
| rotineira | 69 | 0.797 | 0.595 |
| composta | 29 | 0.759 | 0.636 |
| negativa | 16 | 0.625 | 0.500 |
| adversarial | 18 | — (docs_esperados vazio) | — |

**Leitura:** com 3× mais casos, o recall@5 **sobe** (0.737→0.763) — o retriever P1
mantém o perfil; o **alvo 0.90 segue aberto** (ceiling documentado: pesos planos,
rerank contraproducente, boilerplate de slides). Nenhuma suspeita de golden "fácil":
as 88 novas são alcançáveis (filtro top-60 do P5) mas apenas ~0.77 no top-5 — mesma
dificuldade real do corpus.

---

## 2. P2 — Cascade completo (roteador + SLM + pro + juiz) · `avaliar_cascade_v05b.py`

Rodado no **Colab T4** (`llama-cpp-python` CUDA, `SLM_N_GPU_LAYERS=99`), retomada
incremental a partir das 52Q julgadas localmente — **132/132** concluídas
(`data/processed/v05b_ab/cascade_v06_p2.jsonl` + `_resumo.json`) e sessão encerrada.

| Métrica | v0.5b (44Q) | v0.6 (132Q) |
|---|---|---|
| **Acurácia total** | **0.651** | **0.575** (73/127 julgadas) |
| Abstenção correta | 0.886 | **0.750** |
| Alucinações | 8 | **20** |
| Nota média | ~3.4 | 3.30 |
| Custo US$ (API pro + juiz) | 0.01206 | **0.01637** |
| Custo por pergunta | 0.00027 | **0.00012** |
| Fallbacks | — | 4 |

**Trafego de rotas (v0.6):** simples (SLM local) = 116 · complexa (frontier pro) = 16.

Por estrato (v0.6):

| Estrato | n | acuracia | abstenção ok | aluc. | custo US$ |
|---|---|---|---|---|---|
| rotineira | 69 | 0.609 | 0.812 | 10 | 0.00197 |
| composta | 29 | 0.586 | 0.897 | 7 | 0.01125 |
| **negativa** | 16 | **0.125** (2/16) | 0.125 | 0 | 0.00268 |
| adversarial | 18 | **0.833** (15/18) | 0.833 | 3 | 0.00047 |

**Leitura central (CI menor agora):** a acurácia total **cai** (0.651→0.575) — o
0.651 do v05b era **superestimado pela amostra de 44Q**. O sistema real:
rotineira/composta ~0.59–0.61; **adversarial robusta (0.833)**; **negativa segue o
elo mais fraco** (0.125 com 16Q — confirma o 0.000/4 do v05b). Custo/Q caiu
(0.00027→0.00012): rota simples dominante (116/132) com custo API zero e T4 mais
rápido.

---

## 3. Decisões que os números sustentam

1. **Golden v0.6 é a régua oficial da v0.6** (amostra 3× maior, CI menor).
2. **Elo fraco = negativa (0.125)** — não é mais uma suspeita de 4 casos; é o
   primeiro alvo de melhoria da v0.6 (roteador/SLM no estrato "qual destes NÃO é").
3. **P1 recall@5 0.763** confirma desempenho estável do retriever; o gap para 0.90
   é de formulação/boilerplate do corpus, não de regressão.
4. **Custo cascade mais que confirmado como produção**: US$ 0.00012/Q (132Q) — o
   hedge do P6 (US$ 0.00027) é conservador.

---

## 4. Reprodução

```bash
# P1 (local, sem LLM; ~2-3 min)
RAG_GOLDEN_SET_FILE=perguntas_v06.json python scripts/avaliadores/avaliar_retrieval_v05b.py

# P2 (Colab T4 — ver docs/colab_experimentos.md; upload corpus+src+.env+jsonl parcial)
RAG_GOLDEN_SET_FILE=perguntas_v06.json SLM_N_GPU_LAYERS=99 \
  python scripts/avaliadores/avaliar_cascade_v05b.py   # incremental: pula IDs julgados
```

**Artefatos:** `data/processed/rag/retrieval_v06.json` (fora do Git) ·
`data/processed/v05b_ab/cascade_v06_p2.{jsonl,resumo.json}` (fora do Git) ·
golden `data/golden_set/rag/perguntas_v06.json` (Git, commit `0e96db7`).