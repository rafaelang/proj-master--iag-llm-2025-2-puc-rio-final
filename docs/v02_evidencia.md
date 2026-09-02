# v0.2 - Evidencia - RAG (dataset unico do projeto2)

> Pipeline alvo: `projeto_final (BM25 + fastembed + RRF, DeepSeek deepseek-v4-flash)`
> Dataset: `data/golden_set/rag/perguntas.json`.
> Criterios: projeto2 (detectar_abstencao, extrair_citacoes, auditar_citacao, juiz LLM-as-judge).
> Juiz: LLM-as-judge `deepseek-v4-pro` (mesmo prompt do 'juiz_correcao.py' v0.6).
> Prompt de geração: `rag_sistema.txt#16b8a4cb`.

## Metricas por estrato

| Estrato | n | recall@5 | acerto | abstencao correta | citacao presente |
|---|---|---|---|---|---|
| rotineira | 10 | 1.000 | 0.300 | 0.900 | 0.667 |
| composta | 4 | 0.750 | 0.500 | 0.500 | 1.000 |
| negativa | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| adversarial | 4 | - | 1.000 | 1.000 | 0.000 |

## Resumo honesto

- **recall@5** (sobre as 16 com `docs_esperados`): **0.938**
- **acurácia end-to-end** (sobre 20, juiz LLM-as-judge): **0.550**
- **abstenção correta** (sobre 20): **0.850**
- **abstenção indevida** (sobre as 16 com `deve_abster`: false): **0.188** (3)
- **citação presente** (respostas não-abstidas com `[n]`): **0.769**
- **auditoria de citação** — fiel: 10 · fora: 0 · fantasma: 0
- **alucinações** (juiz flag "alucinou"): 0
- juiz não julgou (erro de parse): 0

## Falhas detectadas

### #01 [rotineira] O que é RAG?
- **causa:** resposta incorreta (juiz): A resposta está vazia e não define RAG, apesar de existir fonte esperada no material do curso.
- **resposta:** 

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **causa:** resposta incorreta (juiz): A resposta se abstém de explicar como usar FastAPI para servir um modelo, apesar de o material da aula cobrir esse tema; apenas menciona asp
- **resposta:** De acordo com o trecho, o FastAPI é usado para criar uma API rodando localmente, com documentação e utilização através do OpenAPI/Swagger, e futuramente será feito deploy no Heroku. Não há informação 

### #05 [rotineira] O que são GAN e VAE?
- **causa:** resposta incorreta (juiz): A resposta está vazia e não aborda GAN e VAE, mesmo havendo fonte esperada no curso.
- **resposta:** 

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **causa:** resposta incorreta (juiz): A resposta está vazia, não fornecendo qualquer informação sobre OCR, apesar de haver fonte esperada no corpus.
- **resposta:** 

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): Há fontes esperadas sobre o tema, mas o assistente respondeu NAO_SEI em vez de aproveitar o material do curso.
- **resposta:** NAO_SEI

### #08 [rotineira] O que é transfer learning?
- **causa:** resposta incorreta (juiz): A resposta se absteve de definir transfer learning, apenas citando arquiteturas mencionadas no PDF, mesmo havendo fonte esperada no corpus.
- **resposta:** Nao foi encontrada nos trechos fornecidos uma definicao de transfer learning. O trecho [1] apenas menciona arquiteturas populares usadas para transfer learning, como VGG, ResNet e Inception, sem expli

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **causa:** resposta incorreta (juiz): A resposta é vaga e não explica o que caracteriza os small language models em sistemas multiagentes, limitando-se a uma descrição genérica d
- **resposta:** Com base nos trechos fornecidos, sistemas multiagentes com small language models são descritos como uma equipe de modelos trabalhando em um fluxo controlado, com coordenação, eficiência e controle vol

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve com 'NAO_SEI' mesmo havendo fonte esperada no corpus (nlp_aula05_prompt_engineering.pdf), deixando de responder à perg
- **resposta:** NAO_SEI

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve com 'NAO_SEI' apesar de existir fonte esperada, o que é considerado incorreto segundo as regras.
- **resposta:** NAO_SEI

## Detalhes por pergunta

### #01 [rotineira] O que é RAG?
- **Resposta:** 
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta está vazia e não define RAG, apesar de existir fonte esperada no material do curso.
- **Citações:** -

### #02 [rotineira] O que significa DDL e para que serve?
- **Resposta:** DDL é a linguagem SQL para criação de estruturas, servindo para criar objetos em um banco de dados, como tabelas, índices, visões e esquemas. [2] tdp_aula03_sql_ddl.pdf [5] tdp_aula03_sql_ddl.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 4
- **Justificativa do juiz:** A resposta descreve corretamente DDL como linguagem SQL para criar estruturas e cita objetos como tabelas, índices, visões e esquemas, alinhada à fonte, embora não expanda explicitamente a sigla 'Data Definition Language'.
- **Citações:** [2] tdp_aula03_sql_ddl.pdf; [5] tdp_aula03_sql_ddl.pdf

### #03 [rotineira] O que é DQL?
- **Resposta:** DQL é a Linguagem de Consultas de Dados, uma das categorias de comandos da SQL. [1] tdp_aula03_sql_ddl.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente DQL como Linguagem de Consultas de Dados, categoria de comandos SQL, conforme esperado do material.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **Resposta:** De acordo com o trecho, o FastAPI é usado para criar uma API rodando localmente, com documentação e utilização através do OpenAPI/Swagger, e futuramente será feito deploy no Heroku. Não há informação específica sobre servir um modelo. [1] pai_aula07_deploy_fastapi.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se abstém de explicar como usar FastAPI para servir um modelo, apesar de o material da aula cobrir esse tema; apenas menciona aspectos gerais e nega haver informação específica.
- **Citações:** [1] pai_aula07_deploy_fastapi.pdf

### #05 [rotineira] O que são GAN e VAE?
- **Resposta:** 
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta está vazia e não aborda GAN e VAE, mesmo havendo fonte esperada no curso.
- **Citações:** -

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **Resposta:** 
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta está vazia, não fornecendo qualquer informação sobre OCR, apesar de haver fonte esperada no corpus.
- **Citações:** -

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** Há fontes esperadas sobre o tema, mas o assistente respondeu NAO_SEI em vez de aproveitar o material do curso.
- **Citações:** -

### #08 [rotineira] O que é transfer learning?
- **Resposta:** Nao foi encontrada nos trechos fornecidos uma definicao de transfer learning. O trecho [1] apenas menciona arquiteturas populares usadas para transfer learning, como VGG, ResNet e Inception, sem explicar o conceito. [1] pai_aula07_deploy_fastapi.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 2
- **Justificativa do juiz:** A resposta se absteve de definir transfer learning, apenas citando arquiteturas mencionadas no PDF, mesmo havendo fonte esperada no corpus.
- **Citações:** [1] pai_aula07_deploy_fastapi.pdf; [1] pai_aula07_deploy_fastapi.pdf

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **Resposta:** O contexto menciona WER apenas como uma métrica medida sobre 10 amostras próprias, comparando o resultado antes e depois da aplicação de um vocabulário customizado, dentro da entrega de um protótipo de voz funcional. Não há definição do significado da sigla nem explicação detalhada de como o WER é usado. Portanto, o que encontrei é somente essa referência. [1] proj_spec_tcc.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 4
- **Justificativa do juiz:** A resposta relata fielmente a única menção a WER na fonte, sem inventar definição ou uso além do material; é limitada pela ausência de detalhes no documento.
- **Citações:** [1] proj_spec_tcc.pdf

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **Resposta:** Com base nos trechos fornecidos, sistemas multiagentes com small language models são descritos como uma equipe de modelos trabalhando em um fluxo controlado, com coordenação, eficiência e controle voltados para agentes especializados. [1] proj_a
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 2
- **Justificativa do juiz:** A resposta é vaga e não explica o que caracteriza os small language models em sistemas multiagentes, limitando-se a uma descrição genérica de equipe de modelos.
- **Citações:** [1] proj_aula04_multiagentes_slm.pdf

### #11 [composta] O que são embeddings e como são usados no RAG?
- **Resposta:** Os trechos fornecidos não apresentam uma definição formal do que são embeddings. Sobre o uso no RAG, eles indicam que um extrator de embeddings processa os dados e também a query, e que a recuperação da informação utiliza a vizinhança de embeddings para encontrar os parágrafos mais similares à query. [1] n
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 4
- **Justificativa do juiz:** A resposta reflete corretamente o conteúdo da fonte: não há definição formal de embeddings, mas descreve o uso no RAG (extrator processa dados e query, recuperação por vizinhança de embeddings para similaridade). Não há invenção de fatos.
- **Citações:** [1] nlp_aula06_rag_avancado_ocr.pdf

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve com 'NAO_SEI' mesmo havendo fonte esperada no corpus (nlp_aula05_prompt_engineering.pdf), deixando de responder à pergunta.
- **Citações:** -

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve com 'NAO_SEI' apesar de existir fonte esperada, o que é considerado incorreto segundo as regras.
- **Citações:** -

### #14 [composta] Como Whisper e WER se relacionam com o protótipo de voz do projeto?
- **Resposta:** No trecho fornecido, o protótipo de voz da aula 01 deve ser entregue como v0.1, com WER medido em 10 amostras próprias, antes e depois do vocabulário customizado. Whisper aparece listado entre as tecnologias de fala, junto a GPT-4o Audio, ElevenLabs e Azure Speech, dentro do pipeline de ASR, diarização e TTS. A relação é que o Whisper é uma das tecnologias possíveis para o reconhecimento de fala, e o WER é a métrica usada para avaliar a qualidade desse protótipo de voz. Não há informação adicional sobre como eles se relacionam em detalhe. [1] proj_spec_tcc.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente Whisper como uma tecnologia possível para ASR e WER como métrica de avaliação do protótipo de voz, em linha com o documento do projeto.
- **Citações:** [1] proj_spec_tcc.pdf

### #15 [negativa] Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?
- **Resposta:** Entre os termos, GAN não está relacionado a bancos de dados, pois apenas DDL e DQL são citados como categorias da linguagem SQL [1] tdp_aula03_sql_ddl.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente GAN como o termo não relacionado a bancos de dados, enquanto DDL e DQL são categorias da linguagem SQL.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **Resposta:** O WER é uma métrica da modalidade voz. [1] proj_spec_tcc.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente o WER como métrica da modalidade voz, conforme a fonte indicada.
- **Citações:** [1] proj_spec_tcc.pdf

### #17 [adversarial] Qual a previsão do tempo para amanhã no Rio de Janeiro?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** Pergunta fora do corpus do curso e o assistente se absteve corretamente.
- **Citações:** -

### #18 [adversarial] Quem ganhou a Copa do Mundo de 2022?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus e a resposta se absteve corretamente com NAO_SEI.
- **Citações:** -

### #19 [adversarial] Explique a teoria da relatividade geral de Einstein.
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus e a resposta se absteve corretamente com NAO_SEI, sem alucinar.
- **Citações:** -

### #20 [adversarial] Qual o preço atual da ação da Petrobras?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus e o assistente se absteve corretamente com NAO_SEI.
- **Citações:** -

## Histórico de medições

| Data | Geração | Juiz | Prompt | recall@5 | acerto | abstenção correta | abstenção indevida | citação presente | auditoria fiel/fora/fantasma |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-31T17:41:00 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt (anterior) | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |
| 2026-08-31T20:06:50 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#b3a31998 | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |
| 2026-09-01T11:22:24 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#68e5f51f | 0.938 | 0.550 | 0.650 | 0.438 | 1.000 | 9/0/0 |
| 2026-09-02T19:40:16 | deepseek-v4-flash | deepseek-v4-pro | rag_sistema.txt#16b8a4cb | 0.938 | 0.550 | 0.850 | 0.188 | 0.769 | 10/0/0 |

<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->

## Terceira Rodada: Otimização Isolada do Retrieval

### Embasamento

O gargalo da v0.2 não é a recuperação em si, e sim a **qualidade do contexto
entregue ao LLM**: na medição end-to-end anterior, o Recall@5 estava em **0.812**
(os documentos certos eram encontrados), mas a **Abstenção Indevida** cravou
**0.438** — o LLM recebia o documento certo no Top-5, mas o *chunk* específico podia
estar cortado no meio da ideia (divisão estrita por frases) ou fora das primeiras
posições (ranqueamento). Esta rodada **isola o retrieval** (BM25 + fastembed + RRF
+ rerank) e mede **Recall@5** e **MRR** sobre o top-5 definitivo, sem LLM.

### Resultados — otimizações aplicadas

#### 1. Chunking: frases → parágrafos (300–1200 chars)

| Métrica | Chunking por frases (anterior) | Chunking por parágrafos | Δ |
|---|---|---|---|
| Recall@5 | 0.812 | 0.875 | **+0.063** |
| MRR | 0.790 | 0.819 | **+0.029** |

#### 2. Limpeza de mobília de slides (ingestão)

| Métrica | Sem limpeza (anterior) | Com limpeza | Δ |
|---|---|---|---|
| Recall@5 | 0.875 | 0.875 | 0.000 |
| MRR | 0.819 | 0.812 | −0.007 |

A limpeza reduziu o corpus (356 → 347 chunks), mas **não melhorou as métricas**;
foi mantida por higiene de corpus, mas não é o gargalo do retrieval.

#### 3. BM25 Okapi próprio (Python puro, stopwords pt-BR na query)

| Métrica | rank_bm25 (anterior) | BM25 próprio | Δ |
|---|---|---|---|
| Recall@5 | 0.875 | 0.875 | 0.000 |
| MRR | 0.812 | **0.884** | **+0.072** |

O BM25 próprio (`k1=1.5, b=0.75`, stopwords pt-BR removidas da query, tokens > 1,
IDF suavizado) foi um grande ganho de ranking: #11 (embeddings no RAG) subiu o doc
para o **top-1**.

#### 4. RRF ponderado (BM25 1.0 × denso 1.5)

| Métrica | RRF igual (anterior) | RRF ponderado | Δ |
|---|---|---|---|
| Recall@5 | 0.875 | **0.938** | **+0.063** |
| MRR | 0.884 | **0.906** | **+0.022** |

O peso maior no denso (`PESO_DENSO=1.5`) resolveu **#07** (fine-tuning), que passou
a entrar no **top-5** (rank 2) — a rotineira chegou a recall@5 **1.000**.

#### 5. Rerank (cross-encoder jina-reranker-v2-base-multilingual, over-fetch 30)

| Métrica | RRF ponderado (anterior) | RRF + Rerank | Δ |
|---|---|---|---|
| Recall@5 | 0.938 | 0.938 | +0.000 |
| MRR | 0.906 | 0.865 | −0.041 |

O rerank foi **testado e desativado por padrão no pipeline** (`rerank=False`): MRR caiu 0.906 → 0.865 (rotineira ficou perfeita, mas #13/#15 regrediram) e o custo é alto (~40 s/query em CPU). O estágio continua disponível em `rag/rerank.py` — basta usar `rerank=True`.

#### 6. Overlay (100 chars) + filtro de micro-chunks (>= 100 chars / > 2 palavras)

| Métrica | RRF ponderado (anterior) | + Overlay/filtro micro | Δ |
|---|---|---|---|
| Recall@5 | 0.938 | 0.938 | +0.000 |
| MRR | 0.906 | 0.875 | −0.031 |

O overlay manteve o recall, mas o MRR caiu levemente (#06 OCR rank 1→2). O filtro
rigoroso removeu os micro-chunks sem densidade semântica (corpus 347 → 279).

#### 7. Filtros anti-garbage de PDF na ingestão (ratio < 70%, controle, espaçamento)

| Métrica | Overlay/filtro micro (anterior) | + Anti-garbage | Δ |
|---|---|---|---|
| Recall@5 | 0.938 | **1.000** | **+0.063** |
| MRR | 0.875 | **0.969** | **+0.094** |

A ingestão passou a descartar linhas com razão alfanumérica < 70%, bytes de controle
e espaçamento anômalo — removeu chunks corrompidos do `nlp_aula05` (doc das #07/#12),
**desbloqueando recall@5 = 1.000** (corpus 279 → 272).

### Baseline atual — parágrafos + limpeza + BM25 próprio + RRF ponderado (sem rerank)

- **Recall@5** (sobre as 16 com `docs_esperados`): **1.000**
- **MRR** (idem): **0.969**

Por estrato:

| Estrato | n | Recall@5 | MRR |
|---|---|---|---|
| rotineira | 10 | 1.000 | 1.000 |
| composta | 4 | 1.000 | 0.875 |
| negativa | 2 | 1.000 | 1.000 |

Detalhes por pergunta em `data/processed/rag/retrieval.json`.

### Próximos ajustes (pesos do RRF BM25×embeddings e/ou tamanho dos chunks)

| Ajuste | Recall@5 | MRR | Δ MRR vs baseline |
|---|---|---|---|
| Baseline (RRF k=60, denso 1.5x, BM25 próprio, parág. + limpeza) | 1.000 | 0.969 | — |
| _a definir: ex. RRF k=30 / peso BM25 2x_ | | | |
| _a definir: ex. max_chars=800 / min_chars=200_ | | | |

## Diagnóstico — troca do modelo de geração (deepseek-chat → deepseek-v4-flash)

Experimento diagnóstico (retrieval e juiz idênticos; só a geração mudou):

| Métrica | `deepseek-chat` | `deepseek-v4-flash` |
|---|---|---|
| acurácia end-to-end | 0.550 (11/20) | 0.550 (11/20) |
| abstenção indevida | 0.438 (7/16) | 0.188 (3/16) |
| abstenção correta | 0.650 | 0.850 |
| citação presente | 1.000 (9/9) | 0.769 (10/13) |

O flash **responde mais, mas erra mais** — a acurácia não mudou. **Conclusão:** o
gargalo é a **qualidade/cobertura do contexto (chunk)**, não o modelo de geração
(`docs/v02.md` §5.4). O resultado **final da v0.2** permanece com `deepseek-chat`
(linha anterior do "Histórico de medições").
