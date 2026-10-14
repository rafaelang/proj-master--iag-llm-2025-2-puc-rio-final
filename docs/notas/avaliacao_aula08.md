# Avaliação do projeto_final contra as orientações da Aula 08

> Aula 08 (PROJ) — **"SLM na prática: Usar · Treinar · Alinhar"** (`AULAS/PROJ/aula08/`).
> Fonte das orientações: deck `slm_usar_treinar_alinhar.pptx` (46 slides, 3 blocos +
> defesa) e labs da aula. Projeto avaliado: `projeto_final` (v0.1 → v1.0, fechada).
> Evidências do projeto consultadas: `docs/v05.md`, `v05b*.md`, `v06.md`,
> `v06_evidencia.md`, `v10.md`, `v10_evidencia.md`, `resumo_orientacao.md`.

---

## 1. Síntese da aula 08 (orientações usadas na avaliação)

| Bloco | Orientação central |
|---|---|
| **1 · Como usar** | Oito medições antes de fechar o modelo: qualidade na tarefa real (gold), memória na precisão alvo, TTFT/tokens/s **sob carga**, raciocínio on/off, tool calling, multilíngue, contexto longo, licença/custo. Arquitetura que ganha: **roteador SLM → executores → verificador → escalada com critério**, com a **taxa de escalada** como ponto de medição. |
| **2 · Como treinar** | Escada prompt → RAG → LoRA → treino completo (suba um degrau e **escreva por que o anterior não bastou**). Diagnóstico **forma × fato** (teste de 30 min: colar a informação no prompt). Anti-gatilhos: volume <1M/mês, escopo instável, falha de fato, sem gold, sem MLOps. Medir treino: gold + retenção geral + formato + adversarial + custo/latência **depois**. |
| **3 · Alinhar** | Escolha pelo sinal que você tem: resposta escrevível → SFT e pare; preferência real → DPO; preferência ruidosa → RLHF; verificador → GRPO/RLVR; rubrica → RLAIF; **nada disso → não treinar preferência**. Auditar o juiz (viés de posição, comprimento, rubrica). Par de preferência como unidade do projeto. Reward hacking: duas curvas (proxy × régua-ouro). |
| **Defesa** | Seis evidências: 1 régua · 2 comparação · 3 conta de recursos · 4 decisão de hospedagem · 5 modos de falha · 6 caminho de volta. *"Nenhuma se defende com adjetivo."* |

---

## 2. Bloco 1 — Como usar (as oito medições)

| # | Medição (slide 18) | Situação | Evidência / lacuna |
|---|---|---|---|
| 1 | Qualidade na tarefa real | **✓✓** | golden 132Q congelado, critério escrito (`criterio_rotulo.md`), auditado (P4.1) e **nunca re-editado para melhorar métrica** |
| 2 | Memória na precisão alvo | **◦** | GGUF **Q4_K_M 1,1 GB** documentado; **sem** a conta de VRAM = pesos + cache KV sob concorrência (slide 9) |
| 3 | TTFT e tokens/s **sob carga** | **✗** | não medido. v0.4 só tem latência de requisição única (simples ~40–140 s CPU · complexa ~9–28 s · média 64,96 s) |
| 4 | Raciocínio on/off | **✗** | não avaliado |
| 5 | Tool calling / formato de saída | **◦** | sem tool calling (inócuo); o smoke revelou `flash` devolvendo **vazio** (fallback→pro), mas isso não virou métrica de contrato |
| 6 | Multilíngue / pt-BR | **✓** | golden 100% pt-BR no formato de produção |
| 7 | Retenção em contexto longo | **N/A** | RAG top-5, não contexto longo — janela não é a régua do projeto |
| 8 | Licença e custo total | **✓◦** | Qwen2.5 Apache-2.0; custo US$/1M **por token** medido (P6/v1.0); **utilização de GPU real não entra na conta** (slide 13) |

**Arquitetura (slide 16) — acerto central do projeto:** o desenho implementado é o
vencedor do deck — roteador hermético (`cascade` R2 TF-IDF+XGB local, θ=0.60) →
executor simples (SLM Qwen 1.5B local / `flash`) → executor complexo (`pro`) →
escalada com critério. ~85–87% das julgadas ficam na rota operacional a **US$ 0** e a
**taxa de escalada é medida** (P2/P6) — exatamente o "1 ponto de medição obrigatório".

---

## 3. Bloco 2 — Como treinar

| Orientação | Situação | Leitura |
|---|---|---|
| Escada de intervenção (slide 20) | **✓✓** | seguiu prompt→RAG→LoRA e **parou com justificativa escrita**: pós-aula07 mostrou que 7/13 erros eram de **retrieval** (fato), não do gerador |
| Diagnóstico forma × fato (slide 21) | **✓✓** | o teste de 30 min (colar a infomação no prompt) embasou a v0.5: falha de **grounding**, não de comportamento; correções do golden #13/#30 partiram dele |
| Anti-gatilhos (slide 22) | **✓✓** | volume acadêmico <1M tok/mês + falha de fato + sem dono de MLOps → **"não treinar" é a decisão correta mais comum** — e aqui **quantificada**: LoRA 0.529/12 aluc regride vs baseline 0.758 |
| LoRA executado/medido (slides 23/25/28) | **✓◦** | r=8, base 4-bit, perda só na resposta, comparação **intra-sessão** baseline→+RAG→+LoRA (protocolo do slide 28). Dataset pequeno (32 itens < 200–500 do slide 23) e **retenção geral fora do treino não medida** (limitações documentadas pelo próprio projeto) |
| Destilação (slide 27) | **✓◦** | receita cumprida (coletar do pro → curadoria com juiz de ancoragem → LoRA); SLM servido com frontier como escalada; o adaptador não passou na régua e foi **descartado com número** |

---

## 4. Bloco 3 — Alinhar

| Orientação | Situação | Leitura |
|---|---|---|
| Escolha pelo sinal (slide 38) | **✓✓** | sem preferência real validada → "não treinar preferência" é o que o slide manda; o guardrail DPO do `melhoria_v0.5.md` coincide com o deck |
| Auditar o juiz (slide 41) | **✓✓** | consistência, viés de comprimento, rubrica e grounding rodados de verdade (rlaif ato 4 + P0 nas 16 negativas); detecção e correção do critério (`correta` estável × nota 4↔5) |
| Reward hacking (slide 40) | **◦** | a régua-ouro externa existe (golden) e o P4 detectou a divergência proxy×ouro, mas as **duas curvas** não foram traçadas explicitamente |
| Quatro camadas (slide 43) | **◦✓** | dados (golden) e juiz (auditado) e operação (smoke, `/saude`, timeout, degradação) com artefato versionado + modo de falha; "política" = decisão versionada **"não treinar"** (sem adaptador) |

---

## 5. Defesa — as seis evidências (slide 45)

| # | Evidência | Situação | Comentário |
|---|---|---|---|
| 1 | **Régua** (gold congelado, quem revisou, por que representa produção) | **✓✓** | 132Q congelado + auditoria + critério escrito |
| 2 | **Comparação** base / +RAG / +treino com delta por etapa | **✓✓** | tabelas v0.5 / v0.5b / P3 no mesmo juiz, intra-sessão |
| 3 | **Conta de recursos** (VRAM, TTFT sob carga, US$/1M com utilização) | **◦//** | **lacuna principal**: TTFT/carga e utilização não medidos |
| 4 | **Decisão de hospedagem** (volume, preço API, operação, ponto de virada) | **✓◦** | TCO 3 cenários + US$ 0,27/mil × 1,50 frontier; **break-even do slide 14 não calculado explicitamente** |
| 5 | **Modos de falha** (o que quebra, detecção, rollback) | **✓✓** | fallback cruzado, timeout 90 s, abstenção com motivo, adversarial 0.833, 3 defeitos reais de produção achados/corrigidos |
| 6 | **Caminho de volta** (versões de modelo/adaptador/rubrica + reversão) | **◦✓** | prompts/golden/tags versionados; sem adaptador/rubrica; rollback = redeploy (ok, pouco detalhado) |

---

## 6. Análise dos resultados do projeto_final

### 6.1 Forças (o que a régua da aula 08 confirma)

- **Custo** ✓✓ — cascade **US$ 0,00027/Q** vs frontier-only **0,0015/Q** (~5,5× mais
  barato); **US$ 0,27 / mil consultas** na v1.0. Alvo v1.0 `≤ 0,001/Q` cumprido.
- **Adversarial robusto** ✓✓ — 0.833 (15/18): a abstenção correta fora do corpus
  é o comportamento que a aula 08 exigiria ("recusa correta em vez de invenção confiante").
- **Citação** ✓✓ — 100% nas não-abstidas; auditoria de citação (fiel/fora/fantasma).
- **Negativa resgatada por regra determinística** ✓ — 0 → **11/14** no tipo A,
  0 erros, **US$ 0**, sem disparo em adversarial (0/18): degrau 1 da escada
  (regra/prompt), não pesos.
- **Juiz auditado** ✓✓ — base da confiança de todas as demais métricas (rlaif ato 4).
- **Decisão "não treinar" protegida por medição** ✓✓ — o que a aula 08 mais cobra.

### 6.2 Exposições (o que a v1.0 não fechou sob a régua da aula 08)

1. **A "semana das 8 medições" (slide 18) não foi rodada.** TTFT, tokens/s sob
   carga/concorrência, raciocínio on/off e contrato de saída nunca foram medidos —
   o risco exato do slide: *"o projeto que pula essa semana descobre o mesmo em três
   meses, em produção"*.
2. **Os 5 alvos v1.0 ficaram como projeção, não medidos em produção.** recall@5 DOC
   **0.659** (`< 0.75`), PÁGINA **0.605** (`< 0.70`), fim-a-fim **0.661 projetado**
   (`< 0.70`), negativa tipo A **0.786** (`< 0.80`). A v1.0 **não re-mede** — o smoke
   do Space é 1 pergunta.
3. **Retenção geral do LoRA (slide 28)** não documentada — só acurácia/aluc/abstenção.
4. **Aula 08 pede "provar" (dias 61–90)**: re-medição fim-a-fim em produção com a
   régua oficial continua **em aberto**.

---

## 7. Veredito

O projeto está **acima da média em decisões** — repetindo a régua central da aula 08
("*o melhor modelo é o menor que resolve a tarefa dentro das restrições reais*"):
bloco 2 inteiro (escada + forma×fato + não treinar com número), auditoria do juiz
(que poucos fazem) e a arquitetura do slide 16 (roteador + executores + escalada +
taxa de escalada medida) desenham o sistema heterogêneo que o deck defende.

As **três lacunas objetivas** para a defesa desta aula:

1. **Medição operacional (slide 18)** — TTFT e tokens/s sob carga, raciocínio
   on/off, e a conta de VRAM/KV + utilização no custo (slides 9/13).
2. **Conta de hospedagem completa (slide 14)** — break-even explícito com os quatro
   números (volume mensal, utilização, preço da API, custo de operação).
3. **Fechamento da régua v1.0 (dias 61–90 do slide 44)** — re-medição fim-a-fim do
   sistema em produção para validar (ou rebaixar) os 5 alvos, em vez de projetá-los.

> Nenhuma delas é um "recomeço": são as medições que transformam a evidência de
> *decisão* em evidência de *produção* — a fronteira entre "projeto avaliado" e
> "projeto defendido" na régua da aula 08.