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
| **V4** | **docs completos recuperados (top-5 doc)** | **7/14** | **0** | **7** | **80/127 · 0.630** |
| V3 | V1 + fallback menor-ocorrência | 5/14 | 0 | 9 | 78/127 · 0.614 |

Baseline cascade: 73/127 = 0.575 (negativa 2/14). **V4 projeta 0.630** (+7, negativa
2/14 → 9/14) **sem nenhum erro e sem custo de LLM** (regra determinística, US$ 0,00).

Por que V4 > V1 (top-5 chunks): o contexto da geração é de *páginas*, mas as opções de
suporte vivem em outras páginas do mesmo doc (ex.: #92 "pás/nacele/torre" estão no
`proj_aula02_projeto_rag_eolica.pdf`, mas a página recuperada no top-5 não as contém —
V1 abstém; V4 consulta o doc inteiro e acha). Achado de OCR tratado no `norm()`:
"educa- cao" (hifenização) → "educacao" (#89 acerta).

### 2.1 As 7 abstenções de V4 são seguras e mapeadas

| id | causa | classe |
|---|---|---|
| #41 | doc esperado fora do top-5; só "chave primária" presente (3 ausentes) | recall@5=0 → P3 |
| #91 | doc esperado fora do top-5; "conhecimento específico" presente (3 ausentes) | recall@5=0 → P3 |
| #93 | doc esperado fora do top-5; nenhuma opção (PEFT/LoRA) presente | recall@5=0 → P3 |
| #96 | doc recuperado; "fabricantes de torres"/"receita de bolo" ausentes (registro do trecho usa outra grafia) | registro de palavras |
| #97 | doc recuperado; "algoritmo de damas de Arthur Samuel" em outra grafia | registro de palavras |
| #99 | doc recuperado; doc usa SQL ("CREATE TABLE") e o trecho "criar tabela" não aparece | registro pt × SQL |
| #100 | doc recuperado; "classificação de sentimento"/"score (Pos/Neg)" em outra grafia | registro de palavras |

Nenhuma das 7 responde errado: a regra **só responde com ausente única** (estrita) e
abstém no resto — sem alucinação (diferente das 6 alucinações que o SLM produz nas
compostas). As 4 de registro de palavras poderiam ganhar com stemming/lematização ou
mapeamento pt→SQL, mas isso é além do escopo P1; a abstenção é o comportamento seguro.

## 3. Guarda de colisão negativa×adversarial (P0/S3)

- **Adversarial real (18Q):** o template NÃO disparou em nenhuma (0/18) — sem risco.
- **Sondas sintéticas de colisão** (opções 100% fora do corpus, ex.: "Recife, Salvador ou
  Manaus"): as 3 variantes **abstêm** — quando todas as opções estão ausentes, a regra não
  responde, preservando o contrato do juiz (S3c: abstenção em fora-do-corpus = correta;
  responder = incorreta+alucinação).
- **Tipo B** (#16/#42, resposta = item presente): não disparam o template ("é uma métrica
  de qual modalidade" não contém NÃO) — intocados.

## 4. Decisão (variante + ponto de integração)

1. **Variante V4** (docs completos recuperados; responde **somente com ausente única**;
   senão abstenção). Melhor razão hit/erro: 7/14, 0 erros, colisão segura.
2. **Onde vive: pós-retrieval, estágio único — NÃO no roteador.** O discriminador
   precisa do contexto recuperado para decidir (uma pergunta template só é resolvida
   pela presença/ausência das opções nos docs recuperados); o roteador classifica sem
   contexto. "Ambos" seria duplicação — a checagem é pós-recuperação e pré-geração, com
   **bypass do gerador** quando a regra decide (0 tokens de LLM) e queda no fluxo SLM
   normal quando abstém (o SLM já abstém; comportamento preservado).
3. **Formato da resposta** (integração futura em `src/`): opção + citação do trecho que
   suporta as demais opções (ex.: "GAN — segundo [1], tdp_aula03_sql_ddl.pdf, bancos de
   dados trata de DDL/DQL; GAN não consta no material recuperado"), mantendo o formato de
   citação `[n]` do pipeline para o juiz.
4. **Riscos registrados:** (a) a regra responde pelo *corpus processado* — se o golden
   congelado divergir do texto processado (OCR/hifenização já coberto por `norm()`;
   grafia do trecho vs opção), a métrica congelada pode marcar erro onde o sistema está
   certo — documentar caso a caso, nunca re-editar golden; (b) as 4 lacunas de registro
   e as 3 de recall são alvo de P3 (títulos/boilerplate/dedup + régua doc vs página).

**Aceite:** regra determinística resgata 7/14 do tipo A (0.125 → 0.563 no estrato) e
projeta 0.575 → 0.630 no geral, sem custo, sem erro e sem abrir colisão adversarial;
integração pós-retrieval decidida e justificada.

(dados em `data/processed/v05b_ab/regra_negativa_ev.json`; avaliação isolada —
`python scripts/avaliadores/avaliar_regra_negativa.py`.)