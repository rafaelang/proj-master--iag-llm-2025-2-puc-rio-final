# v0.4 · Agentes — evidência (roteador SIMPLES/COMPLEXA + SLM local + LLM remoto)

> Gerado por `scripts/avaliadores/avaliar_v4.py` em 2026-09-04T13:30:00.
> Dataset: `data/golden_set/rag/perguntas.json` (20 perguntas, inalterado).
> Config: roteador=`slm` · simples=`slm` · complexa=`pro` · SLM local: `Qwen2.5-1.5B-Instruct (GGUF Q4_K_M, local)`.

## Resumo honesto

| Métrica | Valor |
|---|---|
| Abstenção correta (20) | **0.950** |
| Rotas | simples 13 · complexa 7 |
| Agentes | {'gerador-pro': 7, 'gerador-slm': 13} |
| Fallbacks | **0** |
| Latência média | **64.96 s** |
| Tokens API (rotas pagas) | prompt 10003 · completion 3650 |
| Custo do SLM (rotas simples locais) | US$ 0.00 |

## Por pergunta

| # | Estrato | Rota | Agente | Fallback | Absteve | Deve abster | Acerto | Latência (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | rotineira | simples | gerador-slm | nao | False | False | sim | 139.92 |
| 2 | rotineira | simples | gerador-slm | nao | False | False | sim | 71.56 |
| 3 | rotineira | simples | gerador-slm | nao | False | False | sim | 60.34 |
| 4 | rotineira | simples | gerador-slm | nao | False | False | sim | 103.1 |
| 5 | rotineira | simples | gerador-slm | nao | False | False | sim | 72.82 |
| 6 | rotineira | simples | gerador-slm | nao | False | False | sim | 110.48 |
| 7 | rotineira | simples | gerador-slm | nao | False | False | sim | 97.9 |
| 8 | rotineira | simples | gerador-slm | nao | False | False | sim | 123.42 |
| 9 | rotineira | complexa | gerador-pro | nao | False | False | sim | 21.73 |
| 10 | rotineira | complexa | gerador-pro | nao | False | False | sim | 15.86 |
| 11 | composta | complexa | gerador-pro | nao | False | False | sim | 27.61 |
| 12 | composta | complexa | gerador-pro | nao | True | False | nao | 9.31 |
| 13 | composta | simples | gerador-slm | nao | False | False | sim | 100.91 |
| 14 | composta | complexa | gerador-pro | nao | False | False | sim | 21.53 |
| 15 | negativa | complexa | gerador-pro | nao | False | False | sim | 8.83 |
| 16 | negativa | simples | gerador-slm | nao | False | False | sim | 62.16 |
| 17 | adversarial | simples | gerador-slm | nao | True | True | sim | 38.03 |
| 18 | adversarial | simples | gerador-slm | nao | True | True | sim | 132.12 |
| 19 | adversarial | complexa | gerador-pro | nao | True | True | sim | 15.52 |
| 20 | adversarial | simples | gerador-slm | nao | True | True | sim | 66.06 |

## Respostas (resumo)

### #1 [rotineira] O que é RAG?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** O RAG é um sistema de raciocínio iterativo que usa agências autônomas para planejar, resolver complexas perguntas, usar ferramentas e decidir quando buscar mais

### #2 [rotineira] O que significa DDL e para que serve?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** DDL é uma linguagem de criação de objetos em um banco de dados, como tabelas, índices, visões, esquemas, etc. Serve para criar, alterar e deletar objetos no ban

### #3 [rotineira] O que é DQL?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** DQL é a Linguagem de Consultas de Dados (Data Query Language).

### #4 [rotineira] Como o FastAPI é usado para servir um modelo?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** O FastAPI é usado para servir um modelo como parte de uma API local. A documentação e a utilização através do OpenAPI (Swagger) são elementos importantes para c

### #5 [rotineira] O que são GAN e VAE?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** GAN (Generative Adversarial Network) é um modelo que gera imagens a partir de ruído aleatório, com o objetivo de imitar realistas. VAE (Variational Autoencoder)

### #6 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** OCR é uma tecnologia que converte imagens de texto em dados de texto editáveis e pesquisáveis, transformando pixels em caracteres. Quando é essencial, PDFs esca

### #7 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** Fine-tuning de um modelo de linguagem é a ação de ajustar os parâmetros de um modelo pre-trained para que ele seja mais eficaz em um novo contexto ou tarefa.

### #8 [rotineira] O que é transfer learning?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** Transfer Learning é uma técnica que utiliza arquiteturas de redes neurais pré-entregues, como VGG16 e VGG19, para extração de características visuais básicas e 

### #9 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **Rota:** complexa · **Agente:** gerador-pro · **Fallback:** False
- **Resposta:** O trecho [1] indica que WER é uma métrica medida no protótipo de voz funcional, em 10 amostras próprias, antes e depois do vocabulário customizado, para avaliar

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **Rota:** complexa · **Agente:** gerador-pro · **Fallback:** False
- **Resposta:** Sistemas multiagentes com small language models são uma equipe de modelos que operam em um fluxo controlado, com coordenação, eficiência e controle para agentes

### #11 [composta] O que são embeddings e como são usados no RAG?
- **Rota:** complexa · **Agente:** gerador-pro · **Fallback:** False
- **Resposta:** Os trechos não definem formalmente embeddings, mas indicam que são gerados por um extrator de embeddings a partir dos dados e da query [1][2]. No RAG, eles são 

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **Rota:** complexa · **Agente:** gerador-pro · **Fallback:** False
- **Resposta:** NAO_SEI

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** A atenção em modelos de linguagem é um conceito crucial que permite que os modelos processem e sintetizem informações de várias partes de um texto ao mesmo temp

### #14 [composta] Como Whisper e WER se relacionam com o protótipo de voz do projeto?
- **Rota:** complexa · **Agente:** gerador-pro · **Fallback:** False
- **Resposta:** Whisper é uma das tecnologias de ASR citadas para o pipeline de fala do protótipo de voz, responsável pela transcrição do áudio [1]. O WER é a métrica usada na 

### #15 [negativa] Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?
- **Rota:** complexa · **Agente:** gerador-pro · **Fallback:** False
- **Resposta:** O termo que não está relacionado a bancos de dados é GAN. No contexto, DDL e DQL aparecem como categorias de comandos da SQL [1], enquanto GAN não é mencionado.

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** WER é uma métrica de qualidade de voz.

### #17 [adversarial] Qual a previsão do tempo para amanhã no Rio de Janeiro?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** NAO_SEI

### #18 [adversarial] Quem ganhou a Copa do Mundo de 2022?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** NAO_SEI

### #19 [adversarial] Explique a teoria da relatividade geral de Einstein.
- **Rota:** complexa · **Agente:** gerador-pro · **Fallback:** False
- **Resposta:** NAO_SEI

### #20 [adversarial] Qual o preço atual da ação da Petrobras?
- **Rota:** simples · **Agente:** gerador-slm · **Fallback:** False
- **Resposta:** NAO_SEI

<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->


## Seções manuais — dataset e processo (v0.4)

### Dataset

`data/golden_set/rag/perguntas.json` — as **mesmas 20 perguntas** das releases
RAG (dataset único por release; v0.4 reavalia este, não cria outro):

- 16 perguntas com `docs_esperados` (rotineiras 1–10, compostas 11–14, negativas
  15–16) devem responder; 4 adversariais (17–20) devem **abster** (`deve_abster=true`);
- métrica principal: **abstenção correta** = `(absteve == deve_abster)` sobre 20,
  complementada por distribuição de rotas, agente usado, fallbacks, latência e
  tokens de API consumidos (SLM local = US$ 0.00).

### Processo de avaliação

1. `projeto_final.agentes::resposta_multiagente` — roteador (SLM local Qwen
   2.5-1.5B com few-shot) → gerador da rota (SLM local p/ simples; `deepseek-v4-pro`
   p/ complexa) recuperando top-5 no **corpus v0.3** (294 chunks);
2. `avaliar_golden` percorre as 20 perguntas e registra rota/agente/fallback/
   absteve/deve_abster/latência/tokens por pergunta;
3. `scripts/avaliadores/avaliar_v4.py` persiste o JSON intermediário em
   `data/processed/rag_v4/avaliacao_v4.json` (fora do Git), mantém histórico de
   medições e gera este Markdown (seções manuais preservadas pelo marcador).

### Rodadas e ajustes honestos

- **1ª rodada (config inicial):** abstenção correta 0.900; **4 fallbacks** — todas
  causadas por chamadas do `deepseek-v4-pro` retornando **vazias**
  (`finish_reason='length'`: o modelo de raciocínio consumia os 400 tokens
  de `MAX_TOKENS_RAG`). 
- **Correção:** `AGENTE_PRO_MAX_TOKENS=1500` para o gerador pro (mesma lição do
  retry da visão na v0.3) → **2ª rodada: abstenção correta 0.950 (19/20),
  0 fallbacks**.
- Único erro final: **#12** ("Qual a diferença entre prompt engineering e
  fine-tuning?") — o `pro` abstém-se (`NAO_SEI`) mesmo havendo fontes esperadas
  (abstenção indevida 0.05). As 4 adversariais abstiveram 4/4; as demais 15 com
  fonte responderam 15/15.

### Reproduzir

```bash
python scripts/avaliadores/avaliar_v4.py                  # config padrao (slm/slm/pro)
python scripts/avaliadores/avaliar_v4.py --roteador flash # roteador remoto
pytest                                                    # suíte completa (45 testes)
```

### Custo / ambiente

- SLM (Qwen2.5-1.5B GGUF Q4_K_M) e retrieval: **100% locais (CPU)** — US$ 0.00;
- API DeepSeek: só as 7 rotas complexas da rodada final (prompt 10 003 tokens +
  completion 3 650) + 1 pergunta de validação do ajuste;
- Python **3.14** (uv) + `llama-cpp-python` 0.3.35 compilado do sdist (CPU);
  GGUF em `data/processed/slm_models/` (fora do Git).




