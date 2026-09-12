# v0.5b · P1 — Retrieval corrigido no golden expandido (recall@5 + end-to-end)

> Resultado FINAL do P1 (melhoria_v0.5.md) após: auditoria de golden
> (`docs/v05b_golden_audit.md`, correções #13/#30), fallback de OCR na ingestão
> (índice 3.401 → **3.517 chunks**; docs escaneados/letra-a-letra recuperados) e
> decisão de pool `POOL_RERANK=60` como over-fetch padrão (sem rerank em produção).
>
> Dataset: `data/golden_set/rag/perguntas_v05b.json` (44 perguntas, 38 com `docs_esperados`; corrigido).
> Retriever: BM25 + fastembed + RRF (BM25 1.0 × denso 1.5, k=60, imagem 1.15) + dedup · top_k 5.

## 1. Recall@5 — variação de pool (RRF direto, sem cross-encoder)

| pool | recall@5 | MRR | rotineira | composta | negativa |
|---|---|---|---|---|---|
| 10 (produção antiga) | **0.632** (24/38) | 0.501 | 0.720 | 0.556 | 0.250 |
| **60 (produção P1)** | **0.737** (28/38) | 0.486 | **0.880** | 0.556 | 0.250 |
| 100 | 0.711 (27/38) | 0.479 | 0.880 | 0.444 | 0.250 |
| 200 | 0.711 (27/38) | 0.479 | 0.880 | 0.444 | 0.250 |
| 300 | 0.711 (27/38) | 0.479 | 0.880 | 0.444 | 0.250 |
| 400 | 0.711 (27/38) | 0.479 | 0.880 | 0.444 | 0.250 |

**Leitura:** pool 60 é o ponto ótimo. Acima disso o pool maior afoga o ranking em
ruído (composta cai de 0.556 → 0.444). Evidência por pergunta:
`data/processed/rag/recall_pool_{10,60,200}_norerank.json`.

**Ganho acumulado do P1:** baseline v0.5b (pool 10, golden original, índice antigo)
0.605 → **0.737** (pool 60 + golden corrigido + OCR). +0.132.

## 2. Grid de pesos RRF (onda extra — sem ganho)

| pesos (BM25 × denso) | recall@5 |
|---|---|
| 1.0 × 1.0 / 1.25 / 1.5 | 0.737 |
| 1.5 × 1.25 / 1.5 / 2.0 | 0.737 |
| 1.5 × 1.0 · 0.5 × 1.0 · 1.0 × 2.0 · 1.5 × 3.0 | 0.658–0.684 |

O recall@5 é **plano** numa faixa larga de pesos: o teto é definido por **quais docs
entram no pool**, não pela ordenação relativa BM25×denso. Ajuste de pesos não
aproxima o alvo 0.90. Evidência: `data/processed/rag/rrf_grid.json`.

## 3. Rerank (jina cross-encoder)

- Re-medida em pool 60 abortada por instabilidade de sessão Colab (rerank ~160 s/pergunta
  em CPU/ONNX; 3 sessões perdidas). Números de referência do pool 30 (originais,
  `melhoria_v0.5.md`): resgata #03/#07/#16/#31/#33 mas **perde** #08/#13/#40 e **piora
  composta** (0.667→0.444); observação nova: rerank em pool 60 **derrubou #01** (r5 1.0→0.0).
- **Decisão de produção:** `RAG_RERANK=false` + pool 60. O rerank custa ~160 s/pergunta,
  derruba #01 e degrada composta — não compensa o ganho marginal.

## 4. Misses restantes em pool 60 (10/38)

| grupo | IDs | causa |
|---|---|---|
| doc inglês fora do pool | #13 (Attention Is All You Need, pos 84), #07/#12 (prompt-eng paper) | matching léxico pt-BR × EN; golden correto mas doc difícil |
| no pool mas fora do top-5 | #35 (pos 37), #38 (pos 13), #31 | superáveis por rerank (não adotado) |
| negativas difíceis | #16 (WER), #41, #42 | acrônimos/matching fraco |
| recuperado pós-OCR | #28 ✅ · #30 ✅ · #34 ✅ (pool 60) | — |

## 5. End-to-end — acurácia RAG frontier (critério secundário de P1)

Re-medição do RAG frontier no golden **corrigido** + índice pós-OCR + pool 60
(`data/processed/v05b_ab/e2e_v05b_p1.jsonl`):

| Métrica | Prompt-only | RAG (P1) |
|---|---|---|
| **Acurácia end-to-end** (juiz) | 0.907 (39/43) | **0.886 (39/44)** |
| Abstenção correta | 0.909 | 0.955 |
| Alucinações | 2 | 2 |
| Nota média | 4.63 | 4.57 |
| Citação presente | 0.000 | 1.000 |

Por estrato (RAG): **rotineira 0.840** (21/25) · composta 0.889 · negativa 1.000 ·
adversarial 1.000.

**Comparativo:** v0.5b RAG = 0.705 (golden original, pool 10, índice antigo) → **0.886**
(+0.18). **Critério P1 "recuperar ≥ 0.737, ideal > 0.80": ATINGIDO (0.886).**
Erros RAG restantes: #13/#31 (retrieval hard — abstém/alucina), #22/#23/#28
(geração/juiz).

## 6. Critérios de aceite de P1

| Item | Alvo | Resultado | Status |
|---|---|---|---|
| recall@5 (v0.5b) | ≥ 0.90 | **0.737** (pool 60) | ❌ ceiling documentado (pesos planos, rerank contraproducente) |
| acurácia RAG frontier | ≥ 0.737, ideal > 0.80 | **0.886** | ✅✅ |

O alvo recall@5 ≥ 0.90 herdado do corpus rarefeito da v0.2 (272 chunks) não é
alcançável no corpus denso (3.517 chunks): o teto real com o combinado (OCR +
golden + pool 60) é 0.737. O objetivo prático — **recuperar a acurácia do frontier
(0.705 → 0.886)** — foi superado. A diferença é documentada, não mascarada por
ajuste de golden.