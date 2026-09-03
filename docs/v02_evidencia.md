# v0.2 - Evidencia - RAG (dataset unico do projeto2)

> Pipeline alvo: `projeto_final (BM25 + fastembed + RRF, DeepSeek deepseek-chat)`
> Dataset: `data/golden_set/rag/perguntas.json`.
> Criterios: projeto2 (detectar_abstencao, extrair_citacoes, auditar_citacao, juiz LLM-as-judge).
> Juiz: LLM-as-judge `deepseek-v4-pro` (mesmo prompt do 'juiz_correcao.py' v0.6).
> Prompt de geração: `rag_sistema.txt#16b8a4cb`.

## Metricas por estrato

| Estrato | n | recall@5 | acerto | abstencao correta | citacao presente |
|---|---|---|---|---|---|
| rotineira | 10 | 1.000 | 0.700 | 0.800 | 1.000 |
| composta | 4 | 1.000 | 0.250 | 0.500 | 1.000 |
| negativa | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| adversarial | 4 | - | 1.000 | 1.000 | 0.000 |

## Resumo honesto

- **recall@5** (sobre as 16 com `docs_esperados`): **1.000**
- **acurácia end-to-end** (sobre 20, juiz LLM-as-judge): **0.700**
- **abstenção correta** (sobre 20): **0.800**
- **abstenção indevida** (sobre as 16 com `deve_abster`: false): **0.250** (4)
- **citação presente** (respostas não-abstidas com `[n]`): **1.000**
- **auditoria de citação** — fiel: 10 · fora: 2 · fantasma: 0
- **alucinações** (juiz flag "alucinou"): 0
- juiz não julgou (erro de parse): 0

## Falhas detectadas

### #01 [rotineira] O que é RAG?
- **causa:** citacao fora
- **resposta:** RAG é um mecanismo que envolve recuperação de informação e geração de respostas, seguindo etapas como extração de dados, geração de embeddings, busca por parágrafos similares à consulta e uso de promp

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **causa:** resposta incorreta (juiz): A resposta não explica como o FastAPI serve modelos (endpoints, carregamento, predição), limitando-se a mencionar API local, Swagger e deplo
- **resposta:** Com base nos trechos fornecidos, o FastAPI é mencionado como um framework para criar uma API rodando localmente, com documentação e utilização através do OpenAPI (Swagger), e que mais adiante seria fe

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **causa:** citacao fora
- **resposta:** OCR é uma tecnologia que converte imagens de texto em dados de texto editáveis e pesquisáveis, sendo essencial para PDFs escaneados e fotografias de documentos físicos, quando a extração direta não re

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se abstém mesmo havendo fontes esperadas, portanto é considerada incorreta.
- **resposta:** NAO_SEI

### #08 [rotineira] O que é transfer learning?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): O assistente respondeu 'NAO_SEI' mesmo havendo fonte esperada indicada, portanto a abstenção é incorreta para a avaliação.
- **resposta:** NAO_SEI

### #11 [composta] O que são embeddings e como são usados no RAG?
- **causa:** resposta incorreta (juiz): A resposta não define embeddings nem explica completamente seu uso no RAG; apenas menciona superficialmente o papel na recuperação e se abst
- **resposta:** Com base nos trechos fornecidos, posso informar que no fluxo do RAG, os dados passam por um extrator de embeddings, e a recuperação da informação depende da vizinhança de embeddings para encontrar os 

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve (NAO_SEI) apesar de existirem fontes do curso sobre o tema, portanto não respondeu ao que foi solicitado.
- **resposta:** NAO_SEI

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve com 'NAO_SEI', embora houvesse fonte esperada no corpus que poderia embasar o tema; portanto está incorreta por omissã
- **resposta:** NAO_SEI

## Detalhes por pergunta

### #01 [rotineira] O que é RAG?
- **Resposta:** RAG é um mecanismo que envolve recuperação de informação e geração de respostas, seguindo etapas como extração de dados, geração de embeddings, busca por parágrafos similares à consulta e uso de prompt engineering para gerar a resposta [3] nlp_aula06_rag_avancado_ocr.pdf. Além disso, no futuro, o RAG deve evoluir para se tornar um sistema de raciocínio iterativo sobre dados privados, com capacidades como agentes autônomos, auto-correção e suporte multimodal [1] nlp_aula06_rag_avancado_ocr.pdf. Também é citado no contexto de sociedades de agentes e workflows RAG, onde há recuperação, grounding e loops de geração sintética verificável [2] proj_aula04_multiagentes_slm.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define RAG corretamente, citando recuperação de informação, embeddings, similaridade e prompt engineering, e acrescenta informações pertinentes do curso sem contradizer a fonte esperada.
- **Citações:** [3] nlp_aula06_rag_avancado_ocr.pdf; [1] nlp_aula06_rag_avancado_ocr.pdf; [2] proj_aula04_multiagentes_slm.pdf

### #02 [rotineira] O que significa DDL e para que serve?
- **Resposta:** DDL significa Linguagem de Definição de Dados, e é utilizada para criar objetos em um banco de dados, como tabelas, índices, visões e esquemas. O comando CREATE é a base para a criação desses objetos, como mostrado na sintaxe básica para criar tabelas. Além disso, o comando ALTER é usado para modificar estruturas existentes, como adicionar, alterar ou apagar colunas, ou definir valores padrão. Essas operações são fundamentais para gerenciar a estrutura do banco de dados, conforme exemplificado nos exercícios de implementação e modificação de tabelas. [1] tdp_aula03_sql_ddl.pdf, [4] tdp_aula03_sql_ddl.pdf, [5] tdp_aula03_sql_ddl.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente DDL (Data Definition Language) e descreve seus principais comandos CREATE e ALTER, em conformidade com o material da aula sobre SQL DDL.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf; [4] tdp_aula03_sql_ddl.pdf; [5] tdp_aula03_sql_ddl.pdf

### #03 [rotineira] O que é DQL?
- **Resposta:** DQL é a sigla para Linguagem de Consultas de Dados, e faz parte das cinco categorias em que os comandos SQL podem ser agrupados, sendo a SQL a linguagem padrão de consultas aos bancos de dados relacionais [1] tdp_aula03_sql_ddl.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente DQL como Linguagem de Consultas de Dados e a situa entre as categorias de comandos SQL, em conformidade com o material do curso.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **Resposta:** Com base nos trechos fornecidos, o FastAPI é mencionado como um framework para criar uma API rodando localmente, com documentação e utilização através do OpenAPI (Swagger), e que mais adiante seria feito deploy no Heroku. O texto não detalha etapas específicas de como servir um modelo, como carregamento ou predição, então essa parte específica não está coberta. [1] pai_aula07_deploy_fastapi.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 2
- **Justificativa do juiz:** A resposta não explica como o FastAPI serve modelos (endpoints, carregamento, predição), limitando-se a mencionar API local, Swagger e deploy no Heroku, deixando de atender à pergunta apesar da fonte esperada.
- **Citações:** [1] pai_aula07_deploy_fastapi.pdf

### #05 [rotineira] O que são GAN e VAE?
- **Resposta:** Os GANs (Redes Adversariais Generativas) são modelos generativos que aprendem a imitar dados com grande realismo, mas sem estrutura interna. Já os VAEs (Autoencoders Variacionais) são modelos que aprendem a organizar e representar dados, combinando compressão com uma estrutura probabilística, funcionando como um funil que comprime a informação em um espaço latente para depois reconstruí-la. Enquanto os GANs focam no realismo, os VAEs aprendem a estrutura e a organização dos dados, sendo uma base para entender modelos de difusão, que combinam o realismo dos GANs com a estrutura dos VAEs, representando a fronteira atual do estado da arte. [1] dgl_aula01_gan_vae.pdf, [3] dgl_aula01_gan_vae.pdf, [4] dgl_aula01_gan_vae.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta explica corretamente GANs e VAEs, suas diferenças e a relação com modelos de difusão, em conformidade com o material da aula.
- **Citações:** [1] dgl_aula01_gan_vae.pdf; [3] dgl_aula01_gan_vae.pdf; [4] dgl_aula01_gan_vae.pdf

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **Resposta:** OCR é uma tecnologia que converte imagens de texto em dados de texto editáveis e pesquisáveis, sendo essencial para PDFs escaneados e fotografias de documentos físicos, quando a extração direta não retorna conteúdo [1] nlp_aula06_rag_avancado_ocr.pdf. Ele ajuda na recuperação de documentos ao transformar dados visuais em texto que pode ser processado por sistemas de busca e análise. Isso permite que documentos sem camada textual, como aqueles capturados por imagem, se tornem acessíveis para consultas e extração de informações, viabilizando etapas como fragmentação e recuperação baseada em consultas em pipelines de RAG [2] proj_aula04_multiagentes_slm.pdf. A pergunta sobre como o OCR integra especificamente a recuperação em RAG não é totalmente respondida pelo contexto, que foca no que é OCR e em sua necessidade, deixando lacuna sobre a integração prática no fluxo de recuperação.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 4
- **Justificativa do juiz:** A resposta define corretamente OCR e explica seu papel em tornar documentos escaneados pesquisáveis, embora cite fonte extra e ressalve lacunas de integração que não invalida o acerto.
- **Citações:** [1] nlp_aula06_rag_avancado_ocr.pdf; [2] proj_aula04_multiagentes_slm.pdf

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se abstém mesmo havendo fontes esperadas, portanto é considerada incorreta.
- **Citações:** -

### #08 [rotineira] O que é transfer learning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** O assistente respondeu 'NAO_SEI' mesmo havendo fonte esperada indicada, portanto a abstenção é incorreta para a avaliação.
- **Citações:** -

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **Resposta:** O WER é uma métrica usada na avaliação de voz que mede a taxa de erro em transcrições de áudio. No projeto, ele é utilizado no protótipo de voz funcional, onde deve ser medido em 10 amostras próprias, antes e depois da aplicação de um vocabulário customizado, para quantificar o ganho de precisão na transcrição. [1] proj_spec_tcc.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente o WER e descreve seu uso no protótipo de voz conforme o documento, sem invenções.
- **Citações:** [1] proj_spec_tcc.pdf

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **Resposta:** Sistemas multiagentes com small language models envolvem o uso de uma equipe de modelos de linguagem de pequeno porte, coordenados de forma controlada e eficiente para executar tarefas especializadas [1] proj_aula04_multiagentes_slm.pdf. Essa abordagem se baseia na ideia de que esses modelos menores são o futuro da inteligência artificial agêntica, e a literatura destaca a importância de coordenar múltiplos agentes, exigindo estado, papéis, protocolos e integrações explícitas, com frameworks como AutoGen, CAMEL-AI e MetaGPT servindo como exemplos [2] proj_aula04_multiagentes_slm.pdf. [3] proj_aula04_multiagentes_slm.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente sistemas multiagentes com small language models e menciona coordenação, estado, papéis, protocolos e frameworks, alinhando-se ao material indicado.
- **Citações:** [1] proj_aula04_multiagentes_slm.pdf; [2] proj_aula04_multiagentes_slm.pdf; [3] proj_aula04_multiagentes_slm.pdf

### #11 [composta] O que são embeddings e como são usados no RAG?
- **Resposta:** Com base nos trechos fornecidos, posso informar que no fluxo do RAG, os dados passam por um extrator de embeddings, e a recuperação da informação depende da vizinhança de embeddings para encontrar os parágrafos mais similares à query. Contudo, os trechos não definem o que são embeddings nem explicam detalhadamente o processo de extração, além de mencionarem que há um problema na extração. Não encontrei uma explicação conceitual sobre o que são embeddings no contexto fornecido. [1] nlp_aula06_rag_avancado_ocr.pdf e [2] nlp_aula06_rag_avancado_ocr.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 2
- **Justificativa do juiz:** A resposta não define embeddings nem explica completamente seu uso no RAG; apenas menciona superficialmente o papel na recuperação e se abstém da definição, apesar de haver fonte esperada no curso.
- **Citações:** [1] nlp_aula06_rag_avancado_ocr.pdf; [2] nlp_aula06_rag_avancado_ocr.pdf

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve (NAO_SEI) apesar de existirem fontes do curso sobre o tema, portanto não respondeu ao que foi solicitado.
- **Citações:** -

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve com 'NAO_SEI', embora houvesse fonte esperada no corpus que poderia embasar o tema; portanto está incorreta por omissão.
- **Citações:** -

### #14 [composta] Como Whisper e WER se relacionam com o protótipo de voz do projeto?
- **Resposta:** O protótipo de voz v0.1 do projeto deve ser funcional e ter a métrica WER (Word Error Rate) medida em 10 amostras próprias, antes e depois da aplicação de um vocabulário customizado. O Whisper é uma das tecnologias citadas no pipeline de fala, que inclui ASR (reconhecimento automático de fala), diarização e TTS (texto para fala). A relação entre eles está no uso do Whisper como ferramenta de transcrição, enquanto o WER é a métrica usada para avaliar a qualidade dessa transcrição, especialmente para medir o ganho obtido com a correção por vocabulário de domínio. O pipeline completo também considera a latência percebida e a resposta em streaming, e o áudio pode quebrar em situações de ruído, sotaque, jargão técnico e nomes próprios. [1] proj_spec_tcc.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta descreve corretamente a relação entre Whisper como ferramenta de transcrição e WER como métrica de avaliação no protótipo de voz, conforme o documento do projeto.
- **Citações:** [1] proj_spec_tcc.pdf

### #15 [negativa] Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?
- **Resposta:** O termo GAN não está relacionado a bancos de dados, enquanto DDL e DQL são categorias de comandos da linguagem SQL usada para consultas a bancos de dados relacionais. A SQL possui comandos agrupados em cinco categorias, incluindo DDL, que é a Linguagem de Definição de Dados, e DQL, que é a Linguagem de Consultas de Dados. Já GAN não aparece em nenhum dos materiais fornecidos sobre bancos de dados. [1] tdp_aula03_sql_ddl.pdf
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente GAN como termo não relacionado a bancos de dados e contextualiza DDL e DQL como categorias da SQL, em coerência com a fonte do curso.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **Resposta:** O WER é uma métrica da modalidade de voz, pois é utilizado para medir a qualidade de um protótipo de voz funcional, avaliando a taxa de erro antes e depois da aplicação de um vocabulário customizado em amostras de áudio [1] proj_spec_tcc.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente o WER como métrica da modalidade de voz, com justificativa alinhada ao documento do curso.
- **Citações:** [1] proj_spec_tcc.pdf

### #17 [adversarial] Qual a previsão do tempo para amanhã no Rio de Janeiro?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus e o assistente se absteve adequadamente, sem afirmar fatos.
- **Citações:** -

### #18 [adversarial] Quem ganhou a Copa do Mundo de 2022?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus e o assistente se absteve corretamente com 'NAO_SEI'.
- **Citações:** -

### #19 [adversarial] Explique a teoria da relatividade geral de Einstein.
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus do curso e o assistente se absteve corretamente com NAO_SEI.
- **Citações:** -

### #20 [adversarial] Qual o preço atual da ação da Petrobras?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus do curso e o assistente se absteve corretamente, sem alucinar.
- **Citações:** -

## Histórico de medições

| Data | Geração | Juiz | Prompt | recall@5 | acerto | abstenção correta | abstenção indevida | citação presente | auditoria fiel/fora/fantasma |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-31T17:41:00 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt (anterior) | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |
| 2026-08-31T20:06:50 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#b3a31998 | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |
| 2026-09-01T11:22:24 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#68e5f51f | 0.938 | 0.550 | 0.650 | 0.438 | 1.000 | 9/0/0 |
| 2026-09-02T19:40:16 | deepseek-v4-flash | deepseek-v4-pro | rag_sistema.txt#16b8a4cb | 0.938 | 0.550 | 0.850 | 0.188 | 0.769 | 10/0/0 |
| 2026-09-02T21:02:10 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#16b8a4cb | 1.000 | 0.579 | 0.800 | 0.250 | 1.000 | 10/2/0 |
| 2026-09-02T21:11:34 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#16b8a4cb | 1.000 | 0.700 | 0.800 | 0.250 | 1.000 | 10/2/0 |

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
