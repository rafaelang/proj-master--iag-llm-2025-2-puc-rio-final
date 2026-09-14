# Análise — v0.5 · Adaptação (projeto2)

> Síntese da documentação da release **v0.5 · Aula 05 — Adaptação** do projeto2
> (protótipo), a partir de `docs/v05_adaptacao.md` e `docs/v05_adaptacao_evidencia.json`.
> Requisito da spec: tabela comparativa das alternativas (prompt/RAG/fine-tuning/pré-treino)
> e decisão justificada — inclusive "não vale a pena treinar". ✅ FECHADO.

---

## 1. Contexto do caso

Assistente sobre **10 documentos** do próprio curso (corpus pequeno e controlado), com
requisito de **citação da fonte e abstenção** fora do corpus. Baseline já medido em
releases anteriores: **WER 0.043** (v0.1) · **recall@5 1.0** (v0.3) · **abstenção correta 0.95** (v0.4).

## 2. Comparativo das 4 abordagens

| Critério | Prompt (zero-shot) | RAG (busca + geração) | Fine-tuning / LoRA | Pré-treino |
|---|---|---|---|---|
| **Abstenção correta** (amostra) | **0.75** — alucina fora do corpus | **1.00** ✅ | depende dos dados de treino | — |
| **Citação da fonte** | ✗ (não tem fonte) | ✅ [n] | parcial (memoriza sem rastrear) | ✗ |
| **Atualização do corpus** | trocar prompt | **re-indexar (minutos)** | **re-treinar** (horas/GPU) | inviável |
| **Dados necessários** | nenhum | nenhum (só o corpus) | **centenas de pares Q&A rotulados** (não há) | TB de texto |
| **Custo de infra** | ~0 | ~0 (embeddings locais) | GPU (LoRA horas, ~US$ 5–30) | altíssimo |
| **Risco** | alucinação | baixo (grounding) | esquecimento catastrófico / overfit nos 10 docs | — |
| **Esforço de manutenção** | baixo | baixo | alto (re-treinar a cada mudança) | — |

**Leitura:** as únicas abordagens que atendem **citação + abstenção com custo próximo de
zero** são prompt e RAG — e o prompt sozinho falha justamente na abstenção fora do corpus.

## 3. Medição (mini-experimento, 8 perguntas do golden set)

| | Prompt-only (sem evidência) | RAG (com evidência) |
|---|---|---|
| **Abstenção correta** | 6/8 = **0.75** | 8/8 = **1.00** |
| **Respostas com citação [n]** | 0/8 | 4/8 (as 4 rotineiras; adversariais abstêm) |

Casos ilustrativos: *"Quem ganhou a Copa do Mundo de 2022?"* e *"Explique a teoria da
relatividade geral"* — **prompt-only respondeu (alucinação)**, o **RAG absteve corretamente**
(`na evidência: "Copa 2022" → prompt-only "Argentina." [errado], RAG → "NAO_SEI"`).
É exatamente o requisito da spec, que **só o RAG cumpre**.

## 4. Decisão (justificada)

> **Manter prompt engineering + RAG. NÃO vale a pena treinar (nem LoRA).**

1. **O requisito exige citação e abstenção** — propriedade de **grounding** (RAG), não de
   parâmetros: fine-tuning memoriza o corpus, mas **não ensina a citar nem a se abster**
   com garantia (comportamento instruído, não treinado).
2. **Corpus pequeno e estável (10 docs)** — treinar para memorizar 10 PDFs é desperdício;
   RAG indexa em minutos, sem GPU e sem dados rotulados.
3. **Custo-benefício** — LoRA exigiria centenas de pares Q&A (não existem) + GPU; o ganho
   sobre o RAG atual (abstenção 0.95) não se justifica.
4. **Manutenção** — mudança no material do curso → re-indexar é trivial; re-treinar não.

**Conclusão para a banca:** a escolha não é "a técnica mais avançada", e sim a que **atende
o requisito mensurável com menor custo** — o experimento mostra que o prompt-only falha
justamente no que a spec pede.

## 5. Lab executado no Colab (T4) — destilação via LoRA

A spec pede o lab: *"o modelo grande gera o dataset da tarefa; treina-se um SLM"*.
Executado na **T4 (GPU)** via google-colab-cli (infra pesada separada do projeto local):

1. **Dataset sintético (50 Q&A)** — `data/golden_set/adaptacao/dataset_sintetico.json`,
   gerado pelo **DeepSeek pro** a partir de 8 docs do corpus (10 trechos × 5 Q&A).
2. **LoRA no Qwen2.5-1.5B-Instruct** — 4-bit (bitsandbytes), r=8, α=16, q/k/v/o_proj,
   3 épocas, batch 2, lr 2e-4:
   - **2,18M params treináveis (0,14%)** · 39 steps · **32,6 s na T4**
   - loss: 2.272 → 2.239 (train_loss final 2.405)
3. **Adapter salvo** em `data/processed/slm_adapter/` (8,7 MB, fora do Git).
4. **Inferência pós-treino teste:** *"O que é DDL?"* → responde do domínio ✅ — **o lab
   funciona** (o SLM aprendeu o domínio).

**Leitura para a decisão:** destilar é **viável e barato (32 s de GPU)**, mas o SLM destilado
**memoriza conhecimento sem grounding** (sem citação [n], sem abstenção garantida) — o
requisito da spec. Logo: **RAG continua sendo a escolha de produção**; o adapter fica como
artefato/lab e base para experimentos futuros.

## 6. A/B: RAG com SLM base × SLM destilado (o valor da destilação, medido)

Mesmo pipeline RAG (BM25, golden set de 20 perguntas), trocando apenas o gerador:

| Gerador | Sem RAG (memória) | Com RAG (grounding) |
|---|---|---|
| **SLM base** (Qwen2.5-1.5B) | **0.70** (14/20) | **0.75** (15/20) |
| **SLM destilado** (LoRA) | **0.65** (13/20) | **0.95** (19/20) |

**Matriz 2×2 — leitura em 3 eixos:**
1. **Valor da destilação** (mesmo RAG): base 0.75 → destilado **0.95** — o LoRA no dataset
   sintético ensinou o SLM a responder do domínio e abster melhor.
2. **Valor do grounding** (mesmo SLM): base sem→com RAG **+0.05**; destilado **+0.30** — o
   destilado **exploita muito mais a evidência**; o base abstém errado em #1/#15 mesmo com RAG.
3. **Sinergia**: destilação + RAG são complementares — juntos (0.95) **empatam com o pipeline
   multiagente (router + pro) da v0.4, só com SLM local (custo zero de API)**.

Exemplos: na previsão do tempo (#17), o **base alucinou**; o **destilado abstém**. A única
falha restante (#19, relatividade) é limitação genuína de um 1.5B — candidata à
"1 falha encontrada" da v0.6.

> **Detalhe de medição:** o SLM abstém em linguagem natural ("Não sei."/"Não_sei"), não no
> token exato "NAO_SEI" — a métrica usa **detecção semântica**
> (`scripts/colab/avaliar_slm_rag.py`). Resultados crus:
> `data/processed/v05_ab/resultado_{base,base_no_rag,destilado,destilado_no_rag}.json` (fora do Git).

## 7. Extra: DPO (preferência) — pipeline ok, efeito marginal (achado honesto)

Fora da spec (bônus), DPO com estratégia **preferido = resposta COM RAG** (ancorada) ×
**rejeitado = resposta SEM RAG** (memória), objetivo de **tom de tutor**.

- **Dataset:** 24 pares (20 do golden set + 4 gerados com tom de tutor e filtrados por
  contraste) — `data/golden_set/adaptacao/dataset_dpo.json`.
- **Treino:** `trl DPOTrainer` + LoRA sobre o SFT mergeado, na **L4** (18 steps, 41,7 s,
  `rewards/accuracies` pico 0.55) → `data/processed/slm_dpo_adapter/`.
- **Avaliação (golden set, RAG BM25):** SFT 0.95 → **SFT+DPO 0.90** (18/20; falhas #15 e #19).

**Leitura honesta:**
1. **Efeito na métrica neutro/levemente negativo** — mas o caso #15: SFT respondeu **errado**
   (GAN "relacionado a banco de dados") e a métrica contou como OK; **SFT+DPO abstém**
   (mais honesto) e foi penalizado.
2. **Tom de tutor fraco — agora MEDIDO:** LLM-as-judge (DeepSeek pro) 1–5: **SFT 1.89 ×
   SFT+DPO 1.89 (EMPATE)**; vitórias SFT 8 × DPO 6 × 5 empates
   (`data/processed/v05_ab/tom_judge.json`, `scripts/juiz_tom.py`). Apenas **4/24 pares**
   tinham tom de tutor — para "pegar", precisaria de centenas de pares (geração era o
   gargalo: ~3–5 min/par no Colab).
3. **Conclusão:** o **pipeline DPO está validado** (roda ponta a ponta), mas com dataset
   pequeno o ganho não se sustenta — **experimento demonstrativo**, não melhoria comprovada.
   Scripts: `scripts/colab/{gen_pares_dpo,train_dpo,monta_dataset_dpo,avaliar_sft_dpo}.py`.

## 8. ⚠️ Falha metodológica identificada (decisão delegada à v0.6)

Ao investigar o caso **#15**, identificou-se uma falha na própria métrica: a **"abstenção
correta" não mede acurácia** — só verifica *"absteve quando deveria?"*. Consequência: uma
resposta **errada/alucinada** numa pergunta que deveria ser respondida conta como **OK**
(a métrica **recompensou a alucinação**: SFT errou mas passou; SFT+DPO abstém e foi punido).
Isso afeta a leitura dos números de v0.4/v0.5: "abstenção 0.95" é **honestidade de
abstenção**, não acurácia. **Decisão:** correção/ancoragem/alucinação (LLM-as-judge +
auditoria de citação) será feita na **v0.6** — que é o que a spec da Aula 06 pede
("desempenho por estrato", "métrica por elo", "auditoria de citações"). Essa falha vira a
**"1 falha encontrada pelo próprio aluno"** da v0.6.

### Desdobramento na v0.6 (contexto do fechamento)

A v0.6 executou o juiz de correção e **fechou o veredito do DPO** pendente:

| sujeito | acurácia | nota média | alucinações | abstenção correta |
|---|---|---|---|---|
| base (Qwen2.5-1.5B) | **0.45** | 2.95 | 9 | 0.85 |
| sft (destilado) | **0.21** | 2.16 | **14** | 0.84 |
| sft_dpo | **0.40** | 2.90 | 10 | 0.85 |
| pro (frontier) | **0.90** | 4.55 | 0 | 0.90 |

Conclusões v0.6: a destilação **piorou** a acurácia real (0.45 → 0.21; 14 alucinações) — o
ganho de abstenção da v0.5 não resistiu à re-geração com respostas completas; o **DPO
recuperou parte** (0.21 → 0.40; alucinações 14 → 10), placar SFT 1 × SFT+DPO 4 (14 empates):
**DPO ajuda, mas não salva a destilação. Base > SFT > SFT+DPO em acurácia.** (Fonte:
`docs/v06_avaliacao.md`.)

## 9. Síntese das decisões da v0.5

- **Produção = prompt + RAG. Não treinar.** (LoRA/pré-treino descartados por custo,
  necessidade de dados rotulados e ausência de garantia de citação/abstenção.)
- **Destilação (LoRA) é viável e barata** (32 s na T4) e **agrega quando combinada ao RAG**
  (0.75 → 0.95, empata com o multiagente da v0.4 a custo zero de API) — mas **sozinha não
  atende o requisito** (memoriza sem grounding).
- **DPO: pipeline validado, efeito marginal** com dataset pequeno → documentado como
  experimento demonstrativo; veredito definitivo (acurácia) delegado à v0.6.
- **Métrica de abstenção tinha falha** (não mede acurácia) → correção na v0.6;
  o número 0.95 das releases anteriores é "honestidade de abstenção".

## 10. Artefatos da release (projeto2)

| Artefato | Local |
|---|---|
| Documento oficial | `docs/v05_adaptacao.md` |
| Evidência mini-experimento (8 perguntas, prompt-only × RAG) | `docs/v05_adaptacao_evidencia.json` |
| Dataset sintético (50 Q&A) | `data/golden_set/adaptacao/dataset_sintetico.json` (Git) |
| Dataset DPO (24 pares) | `data/golden_set/adaptacao/dataset_dpo.json` (Git) |
| Adapter LoRA SFT | `data/processed/slm_adapter/` (fora do Git) |
| Adapter LoRA DPO | `data/processed/slm_dpo_adapter/` (fora do Git) |
| Resultados crus do A/B | `data/processed/v05_ab/resultado_*.json` (fora do Git) |
| Juiz de tom | `data/processed/v05_ab/tom_judge.json` |
| Scripts | `scripts/colab/{avaliar_slm_rag,gen_pares_dpo,train_dpo,monta_dataset_dpo,avaliar_sft_dpo}.py`, `scripts/juiz_tom.py` |
| Registro de status | `README.md` (matriz 2×2 + conclusão) |