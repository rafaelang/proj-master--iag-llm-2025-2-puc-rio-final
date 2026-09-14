# v0.6 — Evidência medida (release de avaliação; golden 132Q congelado)

> Todos os números abaixo foram medidos **intra-sessão** sobre o índice de produção v0.6
> (3517 chunks / 192 docs) com o **mesmo golden** (`perguntas_v06.json`, 132Q). SLM = Qwen
> 1.5B GGUF local; pro = DeepSeek; juiz = DeepSeek (rotulagem congelada). JSONs em
> `data/processed/` são dado intermediário; a evidência oficial é este documento.

## 1. P1 isolado — retrieval (132Q)

`data/processed/rag/retrieval_v06.json` · RRF BM25 1.0 × denso 1.5, sem rerank, ice-index padrão.

| estrato | n | recall@5 | MRR |
|---|---|---|---|
| rotineira | 69 | 0.797 (55/69) | 0.595 |
| composta | 29 | 0.759 (22/29) | 0.548 |
| negativa | 16 | 0.625 (10/16) | 0.511 |
| adversarial | 18 | — (sem docs_esperados) | — |
| **geral** | 132 | **0.659 DOC (87/132)** | 0.592 |

Régua histórica v05b (114 com docs_esperados): recall@5 **0.763** (87/114) · MRR 0.592.

## 2. P2 fim-a-fim — cascade roteador→gerador→juiz (132Q; 127 com veredito)

`cascade_v06_p2.jsonl` · baseline congelado da v0.6 (antes do plano). 5 sem veredito do
juiz (rotineira) → acurácia sobre as **julgadas**.

| estrato | n | julgadas | acertos | acurácia | alucinações |
|---|---|---|---|---|---|
| rotineira | 69 | 64 | 39 | 0.609 | 10 |
| composta | 29 | 29 | 17 | 0.586 | 7 |
| negativa | 16 | 16 | 2 | 0.125 | 0 |
| adversarial | 18 | 18 | 15 | 0.833 | 3 |
| **total** | **132** | **127** | **73** | **0.575** | **20** |

Rotas: simples (SLM) 60/111 = 0.541 (aluc 19); complexa (pro) 13/16 = 0.812 (aluc 1).
Abstenção correta por estrato: rotineira 0.812 · composta 0.897 · negativa 0.125 ·
adversarial 0.833 (o 0.125 da negativa é o "abster demais" — substrato do P1).

## 3. P0 — auditoria do juiz nas 16 negativas (rlaif reduzido)

- Tipo A (14Q): veredito **0.000** (juiz não dá leniência) → as "2 corretas" da negativa
  eram tipo B (2/2; ancoradas em outros docs do top-5 — WER→voz, TensorFlow→imagens).
- Sondas: S1 consistência (3 estáveis) ✓ · S2 viés de comprimento (5×5) ✓ · S3 rubrica
  (abstenção com fonte esperada = incorreta; 4/4) ✓ · S4 grounding documentado.
- **Decisão: aceitar o juiz como está** (caveat: alinhamento do golden documentado).
- Perfil #41 (recall@5=0): pro listou conteúdo mas não completou o silogismo → forte
  argumento para a regra determinística (P1).

## 4. P2 — rota×estrato (crosstab aditivo) e decisão de limiar

| rota | rotineira | composta | negativa | adversarial | total |
|---|---|---|---|---|---|
| simples (SLM) | 36/61 · 0.590 | 8/19 · 0.421 | 2/14 · 0.143 | 14/17 · 0.824 | 60/111 · 0.541 |
| complexa (pro) | 3/3 · 1.0 | 9/10 · 0.900 | 0/2 · 0.0 | 1/1 · 1.0 | 13/16 · 0.812 |

- **Pro forçado nas 11 compostas que o SLM errou (37, 113, 118, 120, 122, 124, 125, 126,
  127, 130, 131)**: resgata **6/11**, alucinações **6 → 0**, custo **US$ 0.0162**
  (≈ US$ 0.0015/Q).
- **Modelagem de limiar** (R2 determinístico em 0.50–0.90): nenhum limiar captura as 11
  (p_sim 0.57–0.98; subir o limiar só sobe INDETERMINADO → R1, que **também é SLM**).
- **Decisão: manter `AGENTE_CASCADE_LIMIAR=0.60`**; a correção da composta é combinada
  (regra V5 + retrieval), não de roteamento.

## 5. P1 — regra determinística da negativa tipo A (14Q; avaliador isolado)

| variante | contexto | hit | miss | abst | projeção 127Q |
|---|---|---|---|---|---|
| baseline cascade | — | 0/14 | 14 abst | — | 73/127 · 0.575 |
| V1 | top-5 chunks | 5/14 | 0 | 9 | 78/127 · 0.614 |
| V2 | pool top-60 | 6/14 | 0 | 8 | 79/127 · 0.622 |
| V4 | docs completos (top-5 doc) | 7/14 | 0 | 7 | 80/127 · 0.630 |
| **V5** | **docs completos + matcher robusto** | **11/14** | **0** | **3** | **84/127 · 0.661** |
| V5_top5 | top-5 + matcher robusto | 5/14 | 0 | 9 | 80/127 · 0.630 |
| V3 | V1 + menor-ocorrência | 5/14 | 0 | 9 | 78/127 · 0.614 |

- **V5 (vencedora)**: checagem **literal ∪ janela (14 tokens, lema-lite) ∪ mapa pt→SQL
  (DDL)**; responde somente com **ausente única**; custo **US$ 0,00**; vias por pergunta:
  janela em #96 ("fabricantes de torres" — "…às **torres**, há 12 **fabricantes**…"),
  #97 ("algoritmo de damas de Arthur Samuel" — "…**Arthur Samuel** publicou um
  **algoritmo** … **damas**…"), #100 ("classificação de sentimento" e "score (Pos/Neg)" —
  "score: +3 (**pos**: 3, **neg**: 0)"); SQL em #99 ("create table"/"alter column"/"drop
  column").
- **Acertos V5**: #15 GAN, #89 Copa, #90/#92/#94 teoria da relatividade, #95 Capital do
  Japão, #96 receita de bolo, #97 Copa do Mundo, #98 fundo de investimento, #99 teoria da
  relatividade, #100 cotação de ação. **Abstenções (3)**: #41/#91/#93 — **recall@5=0**
  (doc esperado fora do top-5 → alvo do P3/retrieval), sem erro introduzido.
- **Colisão adversarial**: template dispara em **0/18** adversarial real; **3/3** sondas
  sintéticas abstêm (2+ ausentes ou todas ausentes → nunca responde). Tipo B intocado.
- Efeito no estrato: negativa 2/16 → **13/16 (0.812)** projetado; total 73/127 → **84/127
  (0.661)**, acima do teto planejado 0.63–0.65.

## 6. P3 — experimentos de índice + régua doc vs página (132Q)

| experimento | índice | recall@5 DOC (132) | recall@5 PÁGINA (76 âncoras) |
|---|---|---|---|
| **E0 base** (produção) | 3517 chunks | **0.659 (87/132)** | **0.605 (46/76)** |
| E1 +título (1 chunk/doc) | +192 | 0.659 (87/132) | 0.605 (46/76) |
| E2 −boilerplate (18) | 3499 | **0.652 (86/132)** | 0.605 (46/76) |
| E3 combinado | 3709 | 0.659 (87/132) | 0.605 (46/76) |

- E0 sanidade: 0 divergências vs `retrieval_v06.json` por pergunta.
- **Lacuna doc×página** (76 âncoras): ambos ✓ = 46 · doc✓/página✗ = 13 · doc✗ = 17.
  Dado doc recuperado, página certa em 46/59 (0.78) — **a página é o limitante real**
  (substrato das abstenções V5 #41/#91/#93 e de parte das compostas).
- **Decisão: índice de produção mantido** (título ganho 0; boilerplate −1); dedup Jaccard
  0.85 mantido (lacuna não é de cluttering — docs já estavam no top-5 com 1–2 slots).

## 7. Fechamento (P4)

- Golden 132Q auditado (P4.1): íntegro, sem re-edições; único dataset da release.
- Testes: **`pytest tests/` = 86 passed** (reparos: test_v02 não-destrutivo + fixtures,
  test_v03 semântica atual, test_v05b allowlist #13).
- Alvos v1.0 e régua doc/página/fim-a-fim/TCO: tabela em `docs/v06.md` §4.

## 8. Reprodução (um comando por passo)

```bash
cd projeto_final && source .venv/bin/activate
RAG_GOLDEN_SET_FILE=perguntas_v06.json python scripts/avaliadores/avaliar_retrieval_v05b.py   # §1 (retrieval_v06.json)
python scripts/avaliadores/avaliar_cascade_v05b.py                                            # §2,§4 (cascade_v06_p2.jsonl)
python scripts/avaliadores/auditar_juiz_negativa.py                                           # §3
python scripts/avaliadores/avaliar_regra_negativa.py                                          # §5 (regra_negativa_ev.json)
python scripts/avaliadores/avaliar_retrieval_v06_experimentos.py                              # §6 (retrieval_v06_E*.jsonl)
```

Artefatos intermediários (não versionados): `data/processed/rag/retrieval_v06.json`,
`data/processed/v05b_ab/cascade_v06_p2.{jsonl,resumo.json}`,
`data/processed/v05b_ab/regra_negativa_ev.json`,
`data/processed/v05b_ab/retrieval_v06_E*.jsonl`.