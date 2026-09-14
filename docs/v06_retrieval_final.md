# v0.6 · P3 (plano do orientador) — título no índice + boilerplate/dedup + régua doc vs página

> Data: 2026-09-13 · Golden congelado 132Q · Índices experimentais isolados
> (`data/processed/rag/exp_*`, íntegros do índice de produção) · Mesma execução de
> retrieval da v0.6 (`recuperar`, RRF BM25 1.0 × denso 1.5, sem rerank, `POOL_RERANK=60`,
> dedup Jaccard 0.85) · avaliador `scripts/avaliadores/avaliar_retrieval_v06_experimentos.py`.

## 1. R´egua: RECALL@5 de DOC e de PÁGINA (132Q)

Régua DOC = doc esperado nos 5 documentos recuperados. Régua PÁGINA = o trecho-âncora de
`_auditoria.fonte` (76 das 88 novas têm âncora normalizável) aparece no texto de algum
chunk do top-5 — **a régua que a resposta real precisa**, pois a geração cita página.

| experimento | índice | recall@5 DOC (132) | recall@5 PÁGINA (76) |
|---|---|---|---|
| **E0 base** (produção) | 3.517 chunks | **87/132 · 0.659** | **46/76 · 0.605** |
| E1 título | +1 chunk sintético por doc (192) | 87/132 · 0.659 | 46/76 · 0.605 |
| E2 sem boilerplate | −18 chunks "Transformando Dados em Percepção" | 86/132 · 0.652 | 46/76 · 0.605 |
| E3 combinado | E1 + E2 (3.709) | 87/132 · 0.659 | 46/76 · 0.605 |

Sanidade: E0 pergunta-a-pergunta = `retrieval_v06.json` anterior (0 divergências em 132).
Nota: o "recall@5 0.763" registrado na v0.6 é sobre as **114 com docs_esperados** (87/114);
sobre as 132Q a régua nova é 0.659.

## 2. Por estrato (E0)

| estrato | DOC | PÁGINA |
|---|---|---|
| rotineira | 55/69 · 0.797 | 27/44 · 0.614 |
| composta | 22/29 · 0.759 | 13/20 · 0.650 |
| negativa | 10/16 · 0.625 | 6/12 · 0.500 |
| adversarial | 0/18 (sem doc esperado) | — |

## 3. Decisões (hipóteses do orientador testadas e refutadas no golden atual)

1. **Título no índice — NÃO implementar.** E1 = E0 exato (87/132 e 46/76): os chunks
   sintéticos de título (BM25 com poucos tokens, IDF baixo) não alteraram nenhum top-5
   do golden. Ganho medido = 0.
2. **Penalizar/remover boilerplate — NÃO implementar (e não remover).** E2 piora em 1
   (86/132; perdeu uma composta) e PÁGINA não muda (46/76). Os ~18–30 chunks do rodapé
   não poluem o top-5 deste golden (RRF já os dilui); remover custa um doc recuperado.
3. **Dedup intra-doc — manter Jaccard 0.85 atual.** A lacuna PÁGINA não é de
   cluttering: nos 13 casos doc✓/página✗ o doc esperado já estava no top-5 com 1–2
   slots (ex.: #124 com um único chunk do doc recuperado; #92/#95 com 1 slot) — o
   problema é **quais páginas** do doc o RRF traz, não quantas.
4. **Régua doc vs página documentada**: recall de PÁGINA (0.605) é o limitante real —
   o recall de DOC (0.659) superestima a resposta. Entre as 76 ancoradas: doc-hit
   59/76 (0.776); dado doc recuperado, página certa em 46/59 (0.78). A lacuna 13
   divide-se em 3 negativas (#92 #95 #100 — substrato do P1), 3 compostas (#119 #123
   #124 #131) e 6 rotineiras (#45 #62 #73 #79 #82 #85) — nenhuma recuperável por
   título/boilerplate/dedup.

**Consequência para a v0.6:** o índice de produção **permanece como está** (retrieval não
muda nesta release — ganho nulo medido em 4 experimentos). A correção de página
(pool/rerank seletivo, OCR, segmentação) fica registrada como melhoria futura — fora do
escopo da v0.6, coerente com a decisão v0.5b de manter o rerank desativado. As negativas
residuais do P1 já são tratadas pela regra V4 (que consulta o **doc completo**
recuperado — exatamente para contornar a lacuna de página).

**Aceite:** título no índice e boilerplate decidiram-se por evidência (0 ganho / −1);
dedup mantido; régua doc vs página estabelecida e quantificada (DOC 0.659 × PÁGINA 0.605).

(dados por pergunta em `data/processed/v05b_ab/retrieval_v06_*.jsonl`; reprodução:
`python scripts/avaliadores/avaliar_retrieval_v06_experimentos.py`.)