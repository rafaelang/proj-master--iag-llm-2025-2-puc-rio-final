# P3 — SLM na rota simples: AB de prompts + LoRA rebalanceado (resultado NEGATIVO)

**Plano:** `melhoria_v0.5.md` · **Dependências satisfeitas:** P1 (`fa99c8b`), P2 (`8574d47`) ·
**Escopo:** as **34Q** que a cascata (limiar 0.60) envia ao SLM local (split congelado do P2) ·
**Objetivo:** mitigar **super-cautela** (abstém quando deveria responder) e **alucinação**
(responde inventando) com mudanças no prompt (ações a/b) e, se não bastasse, LoRA com
dataset **rebalanceado** (ação c).

## Método

- **Mesma-sessão:** as 3 variantes de prompt **e** o baseline foram medidos na **mesma sessão
  Colab (T4)**, com o mesmo RAG (índice P1, pool 60, top_k 5), mesmo juiz R3 e mesmo corpus.
  Ainda assim o baseline v04 re-medido deu **0.758** vs **0.606** medido no P2 (outra sessão) —
  **variância do juiz entre sessões é grande**; a comparação honesta é intra-sessão.
- **Variantes de prompt** (`prompts/v0.4/`):
  - `v04` = `rag_sistema_slm.txt` (produção) — baseline re-medido na mesma sessão;
  - `p3a` = `rag_sistema_slm_p3a.txt` — grounding + critério de abstenção escrito
    ("responda quando o CONTEXTO menciona o tema; paráfrase apoiada é válida e preferível a NAO_SEI;
     NAO_SEI só quando o tema está ausente");
  - `p3b` = `rag_sistema_slm_p3b.txt` — p3a + few-shot de decisão responde/abstém;
  - `lora` = **LoRA** treinado em dataset rebalanceado dos **casos reais** da rota simples,
    avaliado com o prompt de produção (v04) fixo.
- **Avaliador:** `scripts/avaliadores/avaliar_slm_p3.py` (juiz R3 do `avaliar_v5`, incremental).

## Resultados (34Q, mesma sessão)

| Variante | Acurácia | Aluc | Abstenção correta | Nota média |
|---|---|---|---|---|
| **v04** (baseline produção) | **0.758** (25/33) | **5** | **0.882** | 3.94 |
| p3a (criterio escrito) | 0.735 (25/34) | 7 | 0.912 | 3.85 |
| p3b (+ few-shot) | 0.636 (21/33) | 6 | 0.912 | 3.70 |
| **lora** (LoRA rebalanceado) | **0.529** (18/34) | **12** | **0.824** | 3.21 |

**Leitura:** **nenhuma variante vence o baseline intrasessão.** p3a reduz super-cautela
(#16/#31 passam a responder) mas **aumenta alucinação** (#22/#28/#35 pioram; aluc 5→7) —
custo/benefício nulo em acurácia. p3b é claramente pior. O LoRA **regride forte** (0.529,
12 aluc): abaixo de qualquer prompt.

## Falhas persistentes (o prompt e o LoRA NÃO resolvem)

| Falha | IDs | Causa |
|---|---|---|
| alucinação | **#24, #42, #43** | persistem nas 4 variantes (0 de 4; #43 não absteve em adversarial) |
| super-cautela | **#15** (negativa) | abstém indevidamente nas 4 variantes |
| **retrieval residual** | **#22 #23 #24 #31 #40** | o **frontier pro** também NÃO consegue responder com o top-5 (doc esperado fora do top-5 ou conteúdo esparso) — é falha de **FATO** (RAG), não do SLM |

Os 5 casos de retrieval residual somam 5/34 da rota simples: mesmo o gerador forte ancorado
não encontra apoio no top-5 (ex. #31 CNN: doc `pai_aula05_redes_convolucionais.pdf` **não está
no top-5**; #40: doc esperado fora do top-5). Isso confirma a escada da aula07: para esses,
o remédio continua sendo **RAG** (P1 residual), não pesos.

## Dataset LoRA (ação c) — como foi montado

`data/golden_set/adaptacao/dataset_sintetico_p3.json` (`scripts/colab/gen_dataset_p3.py`):
- **24 casos reais de resposta**: pergunta das 34Q (deve_abster=False) + **contexto top-5 real**
  (mesma recuperação da inferência) + **alvo destilado do frontier pro** com esse contexto;
- **5 abstenções reais** (#17 #18 #20 #43 #44, adversarial) → alvo `NAO_SEI`;
- **3 abstenções sintéticas** fora do corpus (contexto isca) → `NAO_SEI`;
- **5 excluídos** (#22 #23 #24 #31 #40) por falta de alvo confiável (retrieval residual —
  ensinar `NAO_SEI` ali reforçaria a super-cautela que se quer eliminar).

**Treino** (`scripts/colab/train_slm_p3.py`, T4): Qwen2.5-1.5B-Instruct 4-bit (NF4) + LoRA r=8,
4 épocas, perda **só na resposta** (labels -100 no contexto/pergunta), max_length 2048,
formato alinhado à inferência (`Contexto:\n…\n\nPergunta: …\nResposta: …`). Loss 2.05 → 1.88.
**Avaliação** (`scripts/colab/avaliar_slm_p3_adapter.py`): transformers+peft, prompt de produção fixo.

**Modos de falha do LoRA (diagnóstico):** (1) **deriva de estilo** — os alvos do frontier
trazem markdown/listas e o LoRA reproduz isso, violando a regra "sem markdown" do prompt de
produção (#21 #28 #37 #43) e degradando o juiz; (2) **super-cautela induzida** — o peso de
abstenção do treino (8/32) fez o adapter abstém em rotineiras que respondiam antes (#2 #4 #24
#31); (3) **instabilidade com dataset pequeno** (32 itens). O LoRA **não entra em produção**.

## Decisão

- **Produção fica com o prompt v04 (inalterado)** e **sem LoRA** no SLM — nada a mudar no
  `rag_sistema_slm.txt` atual; a decisão "não treinar para volume de startup" da v0.5 **se
  mantém fortalecida** (a tentativa de LoRA rebalanceado nas 34Q piorou mensurável).
- P3 é **resultado negativo honesto**: prompt e LoRA **não** batem o baseline intrasessão;
  a super-cautela/alucinação remanescente do SLM 1.5B é **divisor de retrieval** para 5Q
  (mesmo o frontier não responde) → o investimento de maior alavanca para v0.6 é **P1
  residual (retrieval)**, não o SLM.
- Guardrails atendidos: LoRA só após P1+P2 (`fa99c8b`→`8574d47`→este); rota sem-RAG não é
  gerador (cascade usa RAG).

## Reprodução

```bash
# AB (roda local sem GPU; SLM GGUF local + API): 3 execucoes, uma por variante
RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_slm_p3.py --variante {v04,p3a,p3b}

# Dataset (use o pro como destilador) + treino + avaliação do LoRA (GPU Colab T4)
SLM_N_GPU_LAYERS=-1 uv run python scripts/colab/gen_dataset_p3.py
uv run --project deploy/mcp_colab colab exec -s <sessao> --file run_train_p3c.py   # train_slm_p3.py
uv run --project deploy/mcp_colab colab exec -s <sessao> --file run_eval_p3c.py    # avaliar_slm_p3_adapter.py

# Consolidação local
uv run python - <<'PY'  # resumir() do avaliar_slm_p3 para cada slm_p3_{v04,p3a,p3b,lora}.jsonl
PY
```

**Dados:** `data/processed/v05b_ab/slm_p3_{v04,p3a,p3b,lora}.jsonl` (dado intermediário, fora do Git) ·
**Dataset de treino:** `data/golden_set/adaptacao/dataset_sintetico_p3.json` (entra no Git).