# v0.6 · P2 — Tabela cruzada rota×estrato e decisão do limiar da cascade (plano do orientador)

> Data: 2026-09-13 · Roteador R2 TF-IDF+XGB (`limiar=0.60`, zone de rejeição→R1/SLM) ·
> Geradores: SLM local Qwen 1.5B (rota simples) / `deepseek-v4-pro` (rota complexa) ·
> Golden congelado `perguntas_v06.json` (132Q) · Juiz `deepseek-v4-pro` (aceito no P0).

## 1. Tabela cruzada rota×estrato (acertos/julgadas · acurácia)

Gerada de forma **aditiva** no `resumir()` de `scripts/avaliadores/avaliar_cascade_v05b.py`
(novo campo `cruzada_rota_estrato`; schema existente intocado).

| rota | rotineira | composta | negativa | adversarial | total |
|---|---|---|---|---|---|
| **simples (SLM)** | 36/61 · 0.590 | 8/19 · 0.421 | 2/14 · 0.143 | 14/17 · 0.824 | 60/111 · 0.541 |
| **complexa (pro)** | 3/3 · 1.000 | 9/10 · 0.900 | 0/2 · 0.000 | 1/1 · 1.000 | 13/16 · 0.812 |
| **total** | 39/64 · 0.609 | 17/29 · 0.586 | 2/16 · 0.125 | 15/18 · 0.833 | 73/127 · 0.575 |

Perfil de falhas da rota simples (julgadas): **rotineira** 36/61 — 13 abstenções
indevidas (doc esperado existia) + 10 alucinações; **composta** 8/19 — 6 alucinações +
2 abstenções indevidas; **adversarial** 14/17 — saudável (todas as 14 falhas são
abstenções corretas); **negativa** 2/14 — 12 abstenções estruturais (P1).

## 2. Perda de roteamento: pro forçado nas 11 compostas que o SLM errou

Re-medição intra-sessão (mesmo índice RAG v0.6, mesmo juiz; `_gerar_api` com
`AGENTE_MODELO_PRO`; ids #37 #113 #118 #120 #122 #124 #125 #126 #127 #130 #131):

| id | SLM | pro forçado | pro corrige? | id | SLM | pro forçado | pro corrige? |
|---|---|---|---|---|---|---|---|
| #37 | False | False | – | #125 | False | False | – |
| #113 | False | True | SIM | #126 | False | True | SIM |
| #118 | False | True | SIM | #127 | False | True | SIM |
| #120 | False | False | – | #130 | False | True | SIM |
| #122 | False | True | SIM | #131 | False | False | – |
| #124 | False | False | – | | | | |

**Resultado:** SLM 0/11 · pro forçado **6/11** · alucinações 6 → 0 ·
custo das 11Q: **US$ 0,0162** (in 20,9k / out 9,6k tok ≈ US$ 0,0015/Q).

Se as 19 compostas da rota simples fossem pro: estimativa **8/19 → 14/19 (0.737)**
(8 acertos do SLM mantidos + 6 resgatados). Composta total: 0.586 → ~0.793.

## 3. Modelagem do limiar (R2 determinístico, 132Q — sem API)

`classificar_limiar`: SIMPLES/COMPLEXA se max(p) ≥ limiar; senão INDETERMINADO→R1(SLM).

| limiar | simples | complexa | INDETERMINADO→R1 | compostas→complexa |
|---|---|---|---|---|
| 0.50 | 117 | 15 | 0 | 10 |
| **0.60 (atual)** | 110 | 14 | 8 | **9** |
| 0.70 | 107 | 10 | 15 | 7 |
| 0.80 | 102 | 6 | 24 | 3 |
| 0.90 | 61 | 2 | 69 | 1 |

**Achado-chave:** subir o limiar **piora** o roteamento de compostas (complexa 9→1) e
só empurra 8→69 INDETERMINADO para o **R1, que também é SLM** (mesmo gerador fraco). As
11 compostas erradas têm `p_simples` alto do R2 (0.57–0.98; ex.: #118/#126/#127/#130/#131
≥ 0.94) — **NENHUM valor de limiar as captura**, porque o R2 não detecta a complexidade
*comportamental* (o SLM falha em raciocínio relacional) por wording TF-IDF. A alavanca de
roteamento é **arquiteturalmente indisponível** para esse padrão de erro.

## 4. Decisão (fundamentada na tabela cruzada)

**MANTER o limiar `AGENTE_CASCADE_LIMIAR=0.60` e o roteador cascade como está.**

1. **Subir o limiar não resgata nada** na célula doente (composta/simples: 0.421):
   todas as 11 erradas permanecem simples ou viram INDETERMINADO→R1/SLM em qualquer
   regime testado (0.60–0.90). Ganho = 0, custo extra = latência de R1 (8→69 chamadas).
2. **Forçar COMPLEXA** resgataria ~6/11 nas compostas, mas custa ~US$ 0,0015/Q (≈ US$ 0,20
   por ciclo de 132Q) + latência 3–8 s vinda de API — e as células rotineira/adversarial/
   negativa (105/132Q) o SLM resolve a **US$ 0,00 local**. Pela régua do P6 (custo×acurácia
   em produção contínua), forçar pro nas 132Q **não se paga**.
3. A correção do 0.541 da rota simples **não é de roteamento** — é de geração/pipeline:
   - **P1** (próximo passo): regra determinística de template negativa → ataca 2/14→~14/14.
   - **P3**: títulos no índice + penalizar boilerplate + dedup → ataca o recall de docs
     não recuperados (abstenções indevidas das rotineiras — 13 dos 61 julgados) e as 6
     alucinações das compostas sem grounding recuperável.
4. Registro do custo pro-forçado para referência futura: **11Q = US$ 0,0162**
   (régua P6 aplicada apenas às células onde o ganho é mensurável, ex.: composta).

**Aceite:** decisão fundamentada na tabela cruzada §§1–3, com a célula doente
identificada (composta/simples) e o mecanismo da correção delegado aos passos P1/P3 —
não ao limiar.

(dados em `data/processed/v05b_ab/cascade_v06_p2_proforcado.jsonl`; re-medição completa
regenerável com `RAG_GOLDEN_SET_FILE=perguntas_v06.json python scripts/avaliadores/avaliar_cascade_v05b.py --sem-incremental`.)