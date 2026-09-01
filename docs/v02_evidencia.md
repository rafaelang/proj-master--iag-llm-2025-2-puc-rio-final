# v0.2 - Evidencia - RAG (dataset unico do projeto2)

> Pipeline alvo: `projeto_final (BM25 + fastembed + RRF, DeepSeek deepseek-chat)`
> Dataset: `data/golden_set/rag/perguntas.json`.
> Criterios: projeto2 (detectar_abstencao, extrair_citacoes, auditar_citacao, juiz LLM-as-judge).
> Juiz: LLM-as-judge `deepseek-v4-pro` (mesmo prompt do 'juiz_correcao.py' v0.6).
> Prompt de geração: `rag_sistema.txt#68e5f51f`.

## Metricas por estrato

| Estrato | n | recall@5 | acerto | abstencao correta | citacao presente |
|---|---|---|---|---|---|
| rotineira | 10 | 1.000 | 0.400 | 0.500 | 1.000 |
| composta | 4 | 0.750 | 0.500 | 0.500 | 1.000 |
| negativa | 2 | 1.000 | 0.500 | 1.000 | 1.000 |
| adversarial | 4 | - | 1.000 | 1.000 | 0.000 |

## Resumo honesto

- **recall@5** (sobre as 16 com `docs_esperados`): **0.938**
- **acurácia end-to-end** (sobre 20, juiz LLM-as-judge): **0.550**
- **abstenção correta** (sobre 20): **0.650**
- **abstenção indevida** (sobre as 16 com `deve_abster`: false): **0.438** (7)
- **citação presente** (respostas não-abstidas com `[n]`): **1.000**
- **auditoria de citação** — fiel: 9 · fora: 0 · fantasma: 0
- **alucinações** (juiz flag "alucinou"): 0
- juiz não julgou (erro de parse): 0

## Falhas detectadas

### #01 [rotineira] O que é RAG?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): Havia fonte esperada sobre RAG, mas o assistente se absteve com NAO_SEI, deixando de responder ao tema do curso.
- **resposta:** NAO_SEI

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **causa:** resposta incorreta (juiz): A resposta se abstém de explicar como o FastAPI serve um modelo, apesar de a fonte esperada tratar exatamente desse tema. Deveria detalhar a
- **resposta:** O trecho [1] menciona o desejo de criar uma API usando o framework FastAPI, com documentação e utilização através do OpenAPI (Swagger), e que mais adiante será feito o deploy no Heroku. No entanto, os

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): Havia fonte esperada no material do curso, mas o assistente se absteve em vez de responder sobre fine-tuning.
- **resposta:** NAO_SEI

### #08 [rotineira] O que é transfer learning?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): Havia fonte esperada no curso sobre o tema, mas o assistente se absteve com NAO_SEI, deixando de responder.
- **resposta:** NAO_SEI

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve com NAO_SEI mesmo havendo fonte esperada (proj_spec_tcc.pdf), portanto está incorreta e incompleta.
- **resposta:** NAO_SEI

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se abstém de definir o conceito mesmo havendo fonte esperada sobre o tema e apenas menciona referências e frameworks, sem atender
- **resposta:** Com base nos trechos fornecidos, não há informações suficientes para responder diretamente o que são sistemas multiagentes com small language models, pois o material contém apenas títulos e referência

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve com NAO_SEI, mas havia fonte esperada sobre o tema, logo não respondeu ao conteúdo do curso.
- **resposta:** NAO_SEI

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve com 'NAO_SEI' mesmo havendo fonte esperada no corpus, portanto incorreta.
- **resposta:** NAO_SEI

### #15 [negativa] Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?
- **causa:** resposta incorreta (juiz): A resposta explicou corretamente DDL e DQL, mas se absteve em relação ao GAN, deixando de responder qual termo não está relacionado a bancos
- **resposta:** DQL é uma das categorias da linguagem SQL, que é a linguagem padrão de consultas a bancos de dados relacionais [1] tdp_aula03_sql_ddl.pdf. DDL também é uma categoria da SQL, usada para definição de da

## Detalhes por pergunta

### #01 [rotineira] O que é RAG?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** Havia fonte esperada sobre RAG, mas o assistente se absteve com NAO_SEI, deixando de responder ao tema do curso.
- **Citações:** -

### #02 [rotineira] O que significa DDL e para que serve?
- **Resposta:** DDL significa Linguagem de Definição de Dados, e serve para criar objetos em um banco de dados, como tabelas, índices, visões e esquemas [5] tdp_aula03_sql_ddl.pdf. Não há informações no material sobre outros possíveis significados de DDL.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 4
- **Justificativa do juiz:** A resposta define corretamente DDL e cita criação de objetos como tabelas, índices, visões e esquemas, mas não menciona que DDL também abrange alteração e exclusão dessas estruturas, tornando a explicação parcialmente incompleta frente ao tópico do curso.
- **Citações:** [5] tdp_aula03_sql_ddl.pdf

### #03 [rotineira] O que é DQL?
- **Resposta:** DQL é a Linguagem de Consultas de Dados, uma das cinco categorias de comandos da SQL, conforme [1] tdp_aula03_sql_ddl.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define DQL corretamente como Linguagem de Consultas de Dados, alinhada ao material do curso.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **Resposta:** O trecho [1] menciona o desejo de criar uma API usando o framework FastAPI, com documentação e utilização através do OpenAPI (Swagger), e que mais adiante será feito o deploy no Heroku. No entanto, os trechos fornecidos não contêm informações específicas sobre como o FastAPI é usado para servir um modelo de aprendizado de máquina ou IA. Portanto, sobre esse aspecto específico, não há informações no material.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se abstém de explicar como o FastAPI serve um modelo, apesar de a fonte esperada tratar exatamente desse tema. Deveria detalhar a criação de endpoints e a integração do modelo, mas apenas menciona superficialmente o framework e afirma incorretamente que não há informações específicas.
- **Citações:** [1] pai_aula07_deploy_fastapi.pdf

### #05 [rotineira] O que são GAN e VAE?
- **Resposta:** GAN e VAE são modelos generativos que resolvem problemas diferentes: GAN aprende a imitar dados com realismo, mas sem estrutura, enquanto VAE aprende a organizar e representar dados, sendo a base perfeita para entender diffusion models [1] dgl_aula01_gan_vae.pdf. O VAE é um modelo que combina compressão com estrutura probabilística, e sua arquitetura funciona como um funil inteligente que comprime a imagem em um espaço latente e depois reconstrói a partir dessa representação [5] dgl_aula01_gan_vae.pdf, [4] dgl_aula01_gan_vae.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 4
- **Justificativa do juiz:** A resposta define corretamente GAN e VAE como modelos generativos com enfoques distintos e indica a relação com diffusion models, em linha com o material da aula, embora seja um pouco sucinta sobre GAN.
- **Citações:** [1] dgl_aula01_gan_vae.pdf; [5] dgl_aula01_gan_vae.pdf; [4] dgl_aula01_gan_vae.pdf

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **Resposta:** OCR é a sigla para Optical Character Recognition, uma tecnologia que converte imagens de texto, ou pixels, em dados de texto editáveis e pesquisáveis, ou caracteres [3] nlp_aula06_rag_avancado_ocr.pdf. Ele é essencial para extrair informações de PDFs escaneados, fotografias de documentos físicos e em casos onde a extração direta retorna vazio [3] nlp_aula06_rag_avancado_ocr.pdf. No contexto de recuperação de documentos, o OCR é importante porque a extração de dados é um dos primeiros passos em sistemas RAG, e é nessa etapa que um problema pode ocorrer, conforme destacado [9] nlp_aula06_rag_avancado_ocr.pdf. A ausência de OCR pode impedir que o texto de documentos escaneados seja recuperado e processado corretamente.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente OCR e explica seu papel na extração de texto de documentos escaneados para sistemas RAG, conforme o material citado.
- **Citações:** [3] nlp_aula06_rag_avancado_ocr.pdf; [3] nlp_aula06_rag_avancado_ocr.pdf; [9] nlp_aula06_rag_avancado_ocr.pdf

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** Havia fonte esperada no material do curso, mas o assistente se absteve em vez de responder sobre fine-tuning.
- **Citações:** -

### #08 [rotineira] O que é transfer learning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** Havia fonte esperada no curso sobre o tema, mas o assistente se absteve com NAO_SEI, deixando de responder.
- **Citações:** -

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve com NAO_SEI mesmo havendo fonte esperada (proj_spec_tcc.pdf), portanto está incorreta e incompleta.
- **Citações:** -

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **Resposta:** Com base nos trechos fornecidos, não há informações suficientes para responder diretamente o que são sistemas multiagentes com small language models, pois o material contém apenas títulos e referências bibliográficas sobre o tema, sem uma definição conceitual ou explicação detalhada [1] proj_aula04_multiagentes_slm.pdf; [2] proj_aula04_multiagentes_slm.pdf. Portanto, sobre o conceito em si, não há informações no material. No entanto, posso afirmar que a bibliografia principal menciona um survey sobre mecanismos de colaboração multiagente e uma posição de que small language models são o futuro da IA agêntica [2] proj_aula04_multiagentes_slm.pdf. Além disso, os trechos indicam que frameworks como AutoGen, CAMEL-AI e MetaGPT são usados para construir tais sistemas, com ênfase em coordenação, papéis e protocolos [3] proj_aula04_multiagentes_slm.pdf; [9] proj_aula04_multiagentes_slm.pdf.
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 2
- **Justificativa do juiz:** A resposta se abstém de definir o conceito mesmo havendo fonte esperada sobre o tema e apenas menciona referências e frameworks, sem atender de fato à pergunta.
- **Citações:** [1] proj_aula04_multiagentes_slm.pdf; [2] proj_aula04_multiagentes_slm.pdf; [2] proj_aula04_multiagentes_slm.pdf; [3] proj_aula04_multiagentes_slm.pdf; [9] proj_aula04_multiagentes_slm.pdf

### #11 [composta] O que são embeddings e como são usados no RAG?
- **Resposta:** Com base nos trechos fornecidos, embeddings são usados no RAG para extrair e representar dados de fontes como PDFs e crawlers, permitindo a recuperação de parágrafos mais similares à query através da vizinhança de embeddings [1] nlp_aula06_rag_avancado_ocr.pdf. No LangChain, a memória baseada em vetor (vectorstore) usa embeddings para busca semântica por similaridade, onde o vectorstore decide o que recuperar [2] nlp_aula06_rag_avancado_ocr.pdf. O processo de extração de embeddings é destacado como um ponto de atenção no fluxo do RAG [3] nlp_aula06_rag_avancado_ocr.pdf. Não há informações adicionais nos trechos sobre a definição formal de embeddings ou outros usos.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 4
- **Justificativa do juiz:** A resposta descreve corretamente o uso de embeddings no RAG conforme os materiais, citando recuperação por similaridade e vectorstore do LangChain, embora não apresente definição formal de embeddings.
- **Citações:** [1] nlp_aula06_rag_avancado_ocr.pdf; [2] nlp_aula06_rag_avancado_ocr.pdf; [3] nlp_aula06_rag_avancado_ocr.pdf

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve com NAO_SEI, mas havia fonte esperada sobre o tema, logo não respondeu ao conteúdo do curso.
- **Citações:** -

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve com 'NAO_SEI' mesmo havendo fonte esperada no corpus, portanto incorreta.
- **Citações:** -

### #14 [composta] Como Whisper e WER se relacionam com o protótipo de voz do projeto?
- **Resposta:** O Whisper é uma das tecnologias citadas no contexto do pipeline de fala, que inclui ASR, diarização e TTS [1] proj_spec_tcc.pdf. O WER é a métrica que deve ser medida no protótipo de voz v0.1, em 10 amostras próprias, antes e depois da aplicação do vocabulário customizado [1] proj_spec_tcc.pdf. Não há informações adicionais nos trechos sobre a relação específica entre Whisper e WER no projeto.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente o Whisper no pipeline de fala e o WER como métrica do protótipo de voz v0.1, conforme a fonte indicada.
- **Citações:** [1] proj_spec_tcc.pdf; [1] proj_spec_tcc.pdf

### #15 [negativa] Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?
- **Resposta:** DQL é uma das categorias da linguagem SQL, que é a linguagem padrão de consultas a bancos de dados relacionais [1] tdp_aula03_sql_ddl.pdf. DDL também é uma categoria da SQL, usada para definição de dados [1] tdp_aula03_sql_ddl.pdf. Não há informações sobre o termo GAN nos trechos, portanto não posso afirmar se está ou não relacionado a bancos de dados.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 2
- **Justificativa do juiz:** A resposta explicou corretamente DDL e DQL, mas se absteve em relação ao GAN, deixando de responder qual termo não está relacionado a bancos de dados.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf; [1] tdp_aula03_sql_ddl.pdf

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **Resposta:** O WER é uma métrica da modalidade de voz, pois está relacionado à medição do protótipo de voz funcional [1] proj_spec_tcc.pdf.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** WER (Word Error Rate) é métrica de avaliação de transcrição de voz, portanto a resposta associa corretamente à modalidade de voz conforme o material do curso.
- **Citações:** [1] proj_spec_tcc.pdf

### #17 [adversarial] Qual a previsão do tempo para amanhã no Rio de Janeiro?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** Pergunta fora do corpus e o assistente se absteve corretamente com NAO_SEI.
- **Citações:** -

### #18 [adversarial] Quem ganhou a Copa do Mundo de 2022?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus do curso e o assistente se absteve corretamente com NAO_SEI.
- **Citações:** -

### #19 [adversarial] Explique a teoria da relatividade geral de Einstein.
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus do curso, e o assistente se absteve corretamente com 'NAO_SEI'.
- **Citações:** -

### #20 [adversarial] Qual o preço atual da ação da Petrobras?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus e o assistente se absteve corretamente, sem alucinar.
- **Citações:** -

## Histórico de medições

| Data | Geração | Juiz | Prompt | recall@5 | acerto | abstenção correta | abstenção indevida | citação presente | auditoria fiel/fora/fantasma |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-31T17:41:00 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt (anterior) | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |
| 2026-08-31T20:06:50 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#b3a31998 | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |
| 2026-09-01T11:22:24 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#68e5f51f | 0.938 | 0.550 | 0.650 | 0.438 | 1.000 | 9/0/0 |

<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->



## Terceira Rodada: Otimização Isolada do Retrieval

### Embasamento

O gargalo da v0.2 não é a recuperação em si, e sim a **qualidade do contexto
entregue ao LLM**: na medição end-to-end anterior, o Recall@5 estava em **0.812**
(os documentos certos eram encontrados), mas a **Abstenção Indevida** cravou
**0.438** — o LLM recebia o documento certo no Top-5, mas o *chunk* específico podia
estar cortado no meio da ideia (divisão estrita por frases) ou fora das primeiras
posições (ranqueamento). Esta rodada **isola o retrieval** (BM25 + fastembed + RRF),
sem LLM, para medir a dispersão (Recall@5/@10) e a ordem (MRR) dos documentos.

### Resultados — otimizações aplicadas

#### 1. Chunking: frases → parágrafos (300–1200 chars)

| Métrica | Chunking por frases (anterior) | Chunking por parágrafos | Δ |
|---|---|---|---|
| Recall@5 | 0.812 | 0.875 | **+0.063** |
| Recall@10 | 0.875 | 0.938 | **+0.063** |
| MRR | 0.790 | 0.819 | **+0.029** |

#### 2. Limpeza de mobília de slides (ingestão)

| Métrica | Sem limpeza (anterior) | Com limpeza | Δ |
|---|---|---|---|
| Recall@5 | 0.875 | 0.875 | 0.000 |
| Recall@10 | 0.938 | **0.875** | **−0.063** |
| MRR | 0.819 | 0.812 | −0.007 |

A limpeza reduziu o corpus (356 → 347 chunks), mas **não melhorou as métricas**:
a pergunta #07 (fine-tuning) perdeu o doc do top-10 e o MRR caiu levemente — o
ruído de slides não era o gargalo do retrieval.

#### 3. BM25 Okapi próprio (Python puro, stopwords pt-BR na query)

| Métrica | rank_bm25 (anterior) | BM25 próprio | Δ |
|---|---|---|---|
| Recall@5 | 0.875 | 0.875 | 0.000 |
| Recall@10 | 0.875 | **0.938** | **+0.063** |
| MRR | 0.812 | **0.884** | **+0.072** |

O BM25 próprio (`k1=1.5, b=0.75`, stopwords pt-BR removidas da query, tokens > 1,
IDF suavizado) foi o **maior ganho da rodada**: #11 (embeddings no RAG) subiu o
doc esperado para o **top-1** e #12 (prompt eng vs fine-tuning) passou a entrar
no **top-10** (rank 7).

#### 4. RRF ponderado (BM25 1.0 × denso 1.5) + over-fetch 2x

| Métrica | RRF igual (anterior) | RRF ponderado | Δ |
|---|---|---|---|
| Recall@5 | 0.875 | **0.938** | **+0.063** |
| Recall@10 | 0.938 | 0.938 | 0.000 |
| MRR | 0.884 | **0.906** | **+0.022** |

O peso maior no denso (`PESO_DENSO=1.5`, over-fetch `top_k*2`) resolveu **#07**
(fine-tuning), que passou a entrar no **top-5** (rank 2) — recall@5 da rotineira
chegou a **1.000**. Custo: **#12** (prompt eng vs fine-tuning) saiu do top-10,
recuando a composta (recall@10 1.000 → 0.750).

### Baseline atual — parágrafos + limpeza + BM25 próprio + RRF ponderado (k=60, denso 1.5x)

- **Recall@5** (sobre as 16 com `docs_esperados`): **0.938**
- **Recall@10** (idem): **0.938**
- **MRR** (idem): **0.906**

Por estrato:

| Estrato | n | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| rotineira | 10 | 1.000 | 1.000 | 0.950 |
| composta | 4 | 0.750 | 0.750 | 0.750 |
| negativa | 2 | 1.000 | 1.000 | 1.000 |

Detalhes por pergunta em `data/processed/rag/retrieval.json`.

### Próximos ajustes (pesos do RRF BM25×embeddings e/ou tamanho dos chunks)

| Ajuste | Recall@5 | Recall@10 | MRR | Δ MRR vs baseline |
|---|---|---|---|---|
| Baseline (RRF k=60, denso 1.5x, BM25 próprio, parág. + limpeza) | 0.938 | 0.938 | 0.906 | — |
| _a definir: ex. RRF k=30 / peso BM25 2x_ | | | | |
| _a definir: ex. max_chars=800 / min_chars=200_ | | | | |
