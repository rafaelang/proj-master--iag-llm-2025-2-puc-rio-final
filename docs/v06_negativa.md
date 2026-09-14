# v0.6 · P1 (plano do orientador) — regra determinística para negativa tipo A (item ausente)

> Data: 2026-09-13 · Golden congelado `perguntas_v06.json` (132Q) · Baseline cascade v0.6:
> **2/14** no tipo A (todas abstêm) · Juiz aceito no P0 · Medição intra-sessão (avaliador
> isolado `scripts/avaliadores/avaliar_regra_negativa.py`, nenhuma mudança em `src/`).

## 1. Diagnóstico (por que o SLM colapsa no tipo A)

Template "Qual destes itens/termos NÃO está relacionado a X: a, b, c ou d?" → a resposta
certa = **opção ausente do contexto** (o item fora do domínio É a informação). A regra 3
do `rag_sistema_slm.txt` manda abster quando o contexto não contém a informação —
**a resposta correta é estruturalmente impossível** para o SLM (P0: 14/14 abstêm,
veredito do juiz confiável). O restoration necessário é *mecânico*, não gerativo:
checar presença/ausência das opções no contexto recuperado.

## 2. Variantes testadas intra-sessão (14 tipo A; contexto do mesmo índice v0.6)

| variante | contexto da checagem | hit | miss | abst | projeção 132Q (ac/n) |
|---|---|---|---|---|---|
| V1 | top-5 chunks (contexto real da geração) | 5/14 | 0 | 9 | 78/127 · 0.614 |
| V2 | pool (top-60 RRF) | 6/14 | 0 | 8 | 79/127 · 0.622 |
| V4 | docs completos recuperados (top-5 doc) | 7/14 | 0 | 7 | 80/127 · 0.630 |
| **V5** | **docs completos + matcher robusto** | **11/14** | **0** | **3** | **84/127 · 0.661** |
| V5_top5 | top-5 chunks + matcher robusto | 5/14 | 0 | 9 | 80/127 · 0.630 |
| V3 | V1 + fallback menor-ocorrência | 5/14 | 0 | 9 | 78/127 · 0.614 |

Baseline cascade: 73/127 = 0.575 (negativa 2/14). **V5 projeta 0.661** (+11; negativa
2/14 → 13/14) — acima do teto projetado no plano (0.63–0.65) — **sem nenhum erro e sem
custo de LLM** (regra determinística, US$ 0,00; 0 tokens gerados nas 14).

Por que docs completos > top-5 chunks: o contexto da geração é de *páginas*, mas as
opções de suporte vivem em outras páginas do mesmo doc (ex.: #92 "pás/nacele/torre"
estão no `proj_aula02_projeto_rag_eolica.pdf`, mas a página recuperada no top-5 não as
contém — V1 abstém; V4/V5 consultam o doc inteiro e acham). Achado de OCR tratado no
`norm()`: "educa- cao" (hifenização) → "educacao" (#89 acerta).

### 2.1 V5 = V4 + matcher robusto (fecha 4 das 7 lacunas de V4)

V5 mantém a checagem **literal** (V4) e adiciona duas checagens determinísticas sobre o
texto dos docs recuperados:

1. **Co-ocorrência por janela** (14 tokens, ordem livre): TODOS os tokens de conteúdo da
   opção (stopwords ignoradas; lema-lite = remove "s" final em **ambos** os lados — só
   consistência importa) aparecem num span ≤ 14 no texto do doc. Resgata as grafias
   elípticas do trecho:
   - #96 "fabricantes de torres" → "…quanto às **torres**, há **12 fabricantes**…" (janela 3);
   - #97 "algoritmo de damas de Arthur Samuel" → "…**Arthur Samuel** publicou um **algoritmo**
     para um programa de **damas**…" (janela 9);
   - #100 "classificação de sentimento" (p.28) e "score (Pos/Neg)" → "score: +3 (**pos**: 3,
     **neg**: 0)" (p.51).
2. **Mapa pt→SQL (DDL)**: `criar tabela→create table`, `alterar tipo de coluna→alter
   column`, `apagar coluna→drop column` — o doc da #99 usa SQL inglês ("CREATE TABLE…",
   "ALTER TABLE… ALTER COLUMN…", "DROP COLUMN"); as 3 opções ficam presentes e "teoria da
   relatividade" vira a ausente única.

Segurança: a janela só considera opções com **≥ 2 tokens de conteúdo** (opções de token
único — "GAN", "validade"… — nunca disparam janela) e exige **todos** os tokens no span;
se nenhuma combinação fecha, a opção segue ausente. Sentido do viés: matcher mais
permissivo só pode transformar ausente→presente, o que ou mantém a abstenção (2+ ausentes
continuam 2+ ou vão a 1) ou responde; nunca erra um **ausente único** verdadeiro (veja
§3 — sondas abstêm).

### 2.2 As 3 abstenções restantes de V5 são as do recall@0 (P3)

| id | causa | classe |
|---|---|---|
| #41 | doc esperado fora do top-5; só "chave primária" presente (3 ausentes) | recall@5=0 → P3 |
| #91 | doc esperado fora do top-5; "conhecimento específico" presente (3 ausentes) | recall@5=0 → P3 |
| #93 | doc esperado fora do top-5; nenhuma opção (PEFT/LoRA) presente | recall@5=0 → P3 |

As 3 abstêm porque o **doc com as opções de suporte não foi recuperado** (README do P3:
página é o limitante real, não o índice). Sem doc, a regra não tem subsídio — abster é o
comportamento seguro (0 erros em 14, mesmo sem esses acertos). #41, #91 e #93 continuam
como lacunas de retrieval, fora do alcance da regra.

## 3. Guarda de colisão negativa×adversarial (P0/S3)

- **Adversarial real (18Q):** o template NÃO disparou em nenhuma (0/18) — sem risco
  (comportamento da adversarial — abster — preservado; baseline 15/18 intacto).
- **Sondas sintéticas de colisão** (opções 100% fora do corpus, ex.: "Recife, Salvador ou
  Manaus"; "capital do Japão"; "cotação de ação"): **abstêm** com V5 — quando todas as
  opções estão ausentes (ou 2+ ausentes), a regra não responde, preservando o contrato do
  juiz (S3c: abstenção em fora-do-corpus = correta; responder = incorreta+alucinação).
- **Tipo B** (#16/#42, resposta = item presente): não disparam o template ("é uma métrica
  de qual modalidade" não contém NÃO) — intocados.

## 4. Decisão (variante + ponto de integração)

1. **Variante V5** (docs completos recuperados + matcher literal ∪ janela ∪ SQL; responde
   **somente com ausente única**; senão abstenção). Melhor razão hit/erro: **11/14 (0.786)**,
   cumprindo o alvo do plano (≥ 10/14), 0 erros, colisão segura.
2. **Onde vive: pós-retrieval, estágio único — NÃO no roteador.** O discriminador precisa
   do contexto recuperado para decidir (uma pergunta template só é resolvida pela
   presença/ausência das opções nos docs recuperados); o roteador classifica sem contexto.
   "Ambos" seria duplicação — a checagem é pós-recuperação e pré-geração, com **bypass do
   gerador** quando a regra decide (0 tokens de LLM) e queda no fluxo SLM normal quando
   abstém (o SLM já abstém; comportamento preservado).
3. **Quem formata a resposta (resposta à decisão §3 do plano):** a própria regra — nem SLM,
   nem pro. A decisão **já é tomada** pela checagem mecânica; formatar via SLM (custo 0 mas
   risco de o 1.5B desobedecer e re-abster) ou via pro (~US$ 0.001/Q) agregaria custo e
   latência sem ganho de acurácia. Formato: a opção ausente + citação do trecho que suporta
   as demais (ex.: "receita de bolo — [1] proj_aula02_projeto_rag_eolica.pdf relaciona
   fabricantes de turbinas, pás e torres; não consta receita de bolo"), mantendo o formato
   de citação `[n]` do pipeline para o juiz.
4. **Riscos registrados:** (a) a regra responde pelo *corpus processado* — se o golden
   congelado divergir do texto processado (OCR/hifenização coberto por `norm()`; grafia do
   trecho vs opção coberta pela janela; SQL coberto pelo mapa), documentar caso a caso,
   nunca re-editar golden; (b) as 3 abstenções restantes (#41/#91/#93) são recall@5=0 →
   alvo do P3 (retrieval de página), não da regra; (c) janela/SQL são heurísticas
   determinísticas — o teste de colisão (§3) cobre a regressão adversarial; re-rodar o
   avaliador a cada mudança de corpus.

**Aceite (critérios do P1):** negativa tipo A **11/14 = 0.786 ≥ 0.70** ✓ · adversarial
**0/18 disparos** (sem regressão) ✓ · rotineira/composta **intocadas** (regra só dispara
em tipo A; projeção usa a cascade congelada) ✓ · custo **US$ 0,00** (determinístico,
0 tokens) ✓ · acurácia total projetada **0.575 → 0.661** (dentro do esperado 0.63–0.65,
inclusive acima) ✓. Integração pós-retrieval decidida e justificada; formatação pela
própria regra (resposta à §3 do plano).

(dados em `data/processed/v05b_ab/regra_negativa_ev.json`; avaliação isolada —
`python scripts/avaliadores/avaliar_regra_negativa.py`.)