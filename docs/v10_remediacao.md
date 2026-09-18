# v1.0 · Re-medição em produção (golden 132Q, Colab T4)

> Data: 2026-09-17 · Re-medição fim-a-fim do sistema **em produção** no golden
> oficial congelado (`perguntas_v06.json`, 132Q) para validar os alvos da v1.0
> definidos em `docs/v06.md` §4. Execução em **Colab T4** (`remediacao_v10`):
> índice de produção (3.517 chunks) + SLM Qwen2.5-1.5B GGUF com `SLM_N_GPU_LAYERS=99`
> + frontier `deepseek-v4-pro` + juiz `deepseek-v4-pro`. Ambiente isolado da release
> fechada (avaliadores parametrizados; artefatos em `data/processed/v10_remediacao/`).

---

## 1. Contexto

A v1.0 (produção) **não re-mediu** as métricas após o deploy — o smoke test do Space
cobria 1 pergunta. Os alvos v1.0 ficaram como **projeção** do v0.6 (ex.: fim-a-fim
0.661 "projetado com V5"). Esta re-medição fecha a lacuna de evidência de produção
identificada em `docs/avaliacao_aula08.md` (lacuna 3): validar (ou rebaixar) os alvos
com medição real no golden inteiro.

**Modo:** direto (sem CoT explícito) — confirmado em `main.py:305`, `slm.py:82`,
`llm.py:106`. O `deepseek-v4-pro` tem reasoning interno, mas não é solicitado nem
exposto.

---

## 2. Resultado — Retrieval isolado (P1, sem LLM)

`RAG_GOLDEN_SET_FILE=perguntas_v06.json` → `avaliar_retrieval_v05b.py --no-evidencia`
(rerank on/off; **produção usa `RAG_RERANK=false`**):

| Métrica | v0.6 | **v1.0 (re-medição)** | Δ |
|---|---|---|---|
| Recall@5 (sem rerank) | 0.763 (114/114 c/ docs) | **0.763** | 0.000 |
| MRR (sem rerank) | 0.592 | **0.597** | +0.005 |
| Recall@5 rotineira | 0.797 | **0.797** | 0.000 |
| Recall@5 composta | 0.759 | **0.759** | 0.000 |
| Recall@5 negativa | 0.625 | **0.625** | 0.000 |

**Leitura:** retrieval **estável e reprodutível** — os índices de produção (3.517 chunks,
pool 60) entregam exatamente o mesmo recall@5 do v0.6. O desvio observado anteriormente
(0.659 no DOC, 132Q com adversariais sem docs) permanece: recall@5 DOC oficial
**0.659** (< alvo 0.75); a página de resposta **0.605** (< alvo 0.70) segue o limitante.

> **Nota de reprodução:** o bloco `com_rerank` crashou 2× no kernel Colab (segfault do
> cross-encoder ONNX `jina-reranker` ao processar batch grande; repro mínimo de 2 itens
> funciona). Como produção usa `RAG_RERANK=false`, o sem-rerank é a régua — a medição
> rerank fica documentada no v0.6 (`v06_retrieval_final.md`) e a limitação do Colab é
> registrada, não re-medida.

---

## 3. Resultado — Cascade fim-a-fim (P2, sistema real)

`avaliar_cascade_v05b.py` (roteador `cascade` → simples SLM / complexa pro, fallback
cruzado) no golden inteiro. **132/132** processadas; juiz `deepseek-v4-pro`.

| Métrica | v0.6 (44→132Q) | **v1.0 (re-medição)** | Δ |
|---|---|---|---|
| Acurácia total | 0.575 (73/127) | **0.571** (72/126) | −0.004 |
| Abstenção correta | 0.750 | **0.765** | +0.015 |
| Alucinações | 20 | **21** | +1 |
| Custo total | US$ 0.01637 | **US$ 0.01905** | +16% |
| Custo/Q | US$ 0.00012 | **US$ 0.00014** | +0.00002 |
| Fallbacks | 4 | **2** | −2 |
| Latência média | — | **3.59 s** | — |

**Por estrato:**

| Estrato | v0.6 | **v1.0 (re-medição)** |
|---|---|---|
| rotineira (n=69) | 0.609 (39/64) | **0.600** (39/65) |
| composta (n=29) | 0.586 (17/29) | **0.630** (17/27) |
| negativa (n=16) | 0.125 (2/16) | **0.062** (1/16) |
| adversarial (n=18) | 0.833 (15/18) | **0.833** (15/18) |

**Por rota (tabela cruzada rota×estrato):**

| Rota | n | Acurácia | Alucinações |
|---|---|---|---|
| simples (SLM) | 116 | 0.559 (62/111) | 19 |
| complexa (pro) | 16 | 0.667 (10/15) | 2 |

**Leitura central:** a re-medição **confirma o perfil do v0.6** com variação dentro da
variância intra-sessão do SLM documentada no `ORIENTADOR_PLAN.md` §0.6 (±0.05): acurácia
0.571 vs 0.575 (−0.004). Rotineira/composta/adversarial estáveis; **negativa segue o elo
mais fraco** (1/16, pior que o 2/16 anterior — na faixa do ruído de 16 amostras). Custo/Q
0.00014 **cumpre com folga** o alvo ≤ 0.001.

---

## 4. Regra V5 aplicada à re-medição (projeção do sistema completo)

A regra determinística V5 (P1 do plano do orientador) vive **apenas no avaliador**
(`scripts/avaliadores/avaliar_regra_negativa.py`), **não** no pipeline de produção. O
v0.6 reportou o 0.661 como **projeção** ("com V5 integrada"). Rodada sobre os dados da
re-medição:

| Métrica | v0.6 projetado | **v1.0 (re-medição + V5)** |
|---|---|---|
| Acurácia total com V5 | 0.661 (84/127) | **0.659** (83/126) |
| Negativa tipo A (V5) | 11/14 (0.786) | **11/14 (0.786)** |
| Custo da regra | US$ 0.00 | **US$ 0.00** |

**Leitura:** a projeção do v0.6 é **confirmada** — com V5, o sistema chega a 0.659
fim-a-fim (vs 0.571 real), mas **continua abaixo do alvo v1.0 ≥ 0.70**. A regra resgata
as negativas tipo A de forma determinística e a US$ 0, mas não está integrada ao
pipeline de produção — é o principal gap de implementação para o alvo.

---

## 5. Verificação dos alvos v1.0 (régua de produção)

| Régua | Baseline v0.6 | **v1.0 medido** | Alvo | Status |
|---|---|---|---|---|
| recall@5 DOC (132Q) | 0.659 | **0.659** | ≥ 0.75 | ❌ |
| recall@5 PÁGINA (76 âncoras) | 0.605 | **0.605** (não re-medido; índice estável) | ≥ 0.70 | ❌ |
| acurácia fim-a-fim (cascade) | 0.575 → 0.661 proj. | **0.571 real** / **0.659 com V5** | ≥ 0.70 | ❌ |
| negativa tipo A | 11/14 (0.786) | **11/14 (0.786)** (V5) | ≥ 0.80 | ❌ |
| rotineira | 0.609 | **0.600** | ≥ 0.75 | ❌ |
| composta | 0.586 | **0.630** | ≥ 0.65 | ❌ (perto) |
| adversarial | 0.833 | **0.833** | ≥ 0.80 | ✅ |
| TCO por pergunta | US$ 0.00027 | **US$ 0.00014** | ≤ US$ 0.001 | ✅ |

---

## 6. Conclusões e próximos passos

1. **O sistema real em produção é estável** — a re-medição confirma o perfil do v0.6
   (retrieval idêntico; cascade 0.571 vs 0.575). A v1.0 **não degradou** após o deploy.
2. **Os alvos de recall e fim-a-fim não foram atingidos** no golden inteiro. O caminho
   mais barato para o alvo ≥ 0.70 é **integrar a regra V5 ao pipeline** (pós-retrieval,
   estágio único, US$ 0) — leva o fim-a-fim de 0.571 → 0.659. Ainda faltaria ~0.04 para
   o alvo, endereçável no estrato composta (0.630, via roteamento) ou na negativa restante
   (3 abstenções com recall@0).
3. **Limitações documentadas:** com_rerank não re-medido no Colab (segfault ONNX);
   recall@5 PÁGINA não re-medido (índice estável, valor v0.6 mantido); variância do SLM
   (±0.05) faz a negativa oscilar 0.062–0.125 — comparar apenas intra-sessão.

## 7. Reprodução

```bash
# retrieval isolado (sem LLM)
RAG_GOLDEN_SET_FILE=perguntas_v06.json python scripts/avaliadores/avaliar_retrieval_v05b.py --no-evidencia

# cascade fim-a-fim (Colab T4, SLM_N_GPU_LAYERS=99)
SLM_N_GPU_LAYERS=99 RAG_GOLDEN_SET_FILE=perguntas_v06.json \
  python scripts/avaliadores/avaliar_cascade_v05b.py --inicio 0 --fim 44
SLM_N_GPU_LAYERS=99 RAG_GOLDEN_SET_FILE=perguntas_v06.json \
  python scripts/avaliadores/avaliar_cascade_v05b.py --inicio 44 --fim 88
SLM_N_GPU_LAYERS=99 RAG_GOLDEN_SET_FILE=perguntas_v06.json \
  python scripts/avaliadores/avaliar_cascade_v05b.py --inicio 88 --fim 132
python scripts/avaliadores/avaliar_cascade_v05b.py --sem-incremental   # resumo

# regra V5 sobre os dados da re-medição
python scripts/avaliadores/avaliar_regra_negativa.py
```

Artefatos (intermediários, fora do Git): `data/processed/v10_remediacao/`
(`cascade_v06_p2.jsonl`, `cascade_v06_p2_resumo.json`, `retrieval_v10.json`,
`regra_negativa_ev.json`).