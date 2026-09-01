# v0.2 - Evidencia - RAG (dataset unico do projeto2)

> Pipeline alvo: `projeto_final (BM25 + fastembed + RRF, DeepSeek deepseek-chat)`
> Dataset: `data/golden_set/rag/perguntas.json`.
> Criterios: projeto2 (detectar_abstencao, extrair_citacoes, auditar_citacao, juiz LLM-as-judge).
> Juiz: LLM-as-judge `deepseek-v4-pro` (mesmo prompt do 'juiz_correcao.py' v0.6).
> Prompt de geração: `rag_sistema.txt#b3a31998`.

## Metricas por estrato

| Estrato | n | recall@5 | acerto | abstencao correta | citacao presente |
|---|---|---|---|---|---|
| rotineira | 10 | 0.900 | 0.500 | 0.600 | 1.000 |
| composta | 4 | 0.750 | 0.250 | 0.250 | 1.000 |
| negativa | 2 | 0.500 | 1.000 | 1.000 | 1.000 |
| adversarial | 4 | - | 1.000 | 1.000 | 0.000 |

## Resumo honesto

- **recall@5** (sobre as 16 com `docs_esperados`): **0.812**
- **acurácia end-to-end** (sobre 20, juiz LLM-as-judge): **0.600**
- **abstenção correta** (sobre 20): **0.650**
- **abstenção indevida** (sobre as 16 com `deve_abster`: false): **0.438** (7)
- **citação presente** (respostas não-abstidas com `[n]`): **1.000**
- **auditoria de citação** — fiel: 7 · fora: 2 · fantasma: 0
- **alucinações** (juiz flag "alucinou"): 0
- juiz não julgou (erro de parse): 0

## Falhas detectadas

### #01 [rotineira] O que é RAG?
- **causa:** citacao fora
- **resposta:** RAG, ou Retrieval-Augmented Generation, é uma técnica que conecta a resposta de um modelo de linguagem a documentos recuperados no momento da tarefa, com o objetivo de substituir a memória implícita d

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **causa:** resposta incorreta (juiz): A resposta se abstém mesmo havendo fonte esperada (pai_aula07_deploy_fastapi.pdf) que deveria explicar como o FastAPI serve modelos, caracte
- **resposta:** Com base nos trechos fornecidos, o FastAPI é mencionado apenas como um framework para criar uma API rodando localmente, com documentação e utilização através do OpenAPI (Swagger), e que posteriormente

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se abstém (NAO_SEI) mesmo havendo fonte esperada no corpus, portanto é incorreta.
- **resposta:** NAO_SEI

### #08 [rotineira] O que é transfer learning?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve ('NAO_SEI') mesmo havendo fonte esperada no corpus, portanto não respondeu à pergunta sobre transfer learning.
- **resposta:** NAO_SEI

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A pergunta está dentro do corpus (proj_spec_tcc.pdf), mas o assistente se absteve com NAO_SEI, deixando de fornecer a resposta esperada.
- **resposta:** NAO_SEI

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A pergunta está dentro do corpus (proj_aula04_multiagentes_slm.pdf), mas o assistente respondeu NAO_SEI e não forneceu a definição solicitad
- **resposta:** NAO_SEI

### #11 [composta] O que são embeddings e como são usados no RAG?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A pergunta possui fonte esperada no corpus, mas o assistente se absteve com NAO_SEI, deixando de fornecer a resposta.
- **resposta:** NAO_SEI

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve mesmo havendo fonte esperada sobre o tema, deixando de responder à pergunta.
- **resposta:** NAO_SEI

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **causa:** abstencao indevida
- **causa:** resposta incorreta (juiz): A resposta se absteve (NAO_SEI) mesmo havendo fonte esperada no corpus, deixando de responder a pergunta.
- **resposta:** NAO_SEI

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **causa:** recuperacao falhou (nenhum doc esperado no top-5)
- **causa:** citacao fora
- **resposta:** O WER é uma métrica da modalidade de voz, pois é utilizado para medir o desempenho do protótipo de voz funcional, conforme descrito na entrega v0.1 do projeto, que menciona "WER medido em 10 amostras 

## Detalhes por pergunta

### #01 [rotineira] O que é RAG?
- **Resposta:** RAG, ou Retrieval-Augmented Generation, é uma técnica que conecta a resposta de um modelo de linguagem a documentos recuperados no momento da tarefa, com o objetivo de substituir a memória implícita do modelo por evidências rastreáveis e relevantes [4] proj_aula04_multiagentes_slm. O processo envolve extrair dados de fontes como PDFs e crawlers, gerar embeddings, recuperar os parágrafos mais similares à consulta e então usar essa informação na geração da resposta [7] nlp_aula06_rag_avancado_ocr. Uma limitação importante é que apenas recuperar documentos não garante uma conclusão correta, sendo necessários filtros de qualidade, atribuição de fonte e checagem de suporte entre evidência e resposta [4] proj_aula04_multiagentes_slm.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define RAG corretamente, descreve o fluxo de extração, embeddings, recuperação e geração, e menciona limitações relevantes, alinhando-se ao conteúdo do curso.
- **Citações:** [4] proj_aula04_multiagentes_slm.pdf; [7] nlp_aula06_rag_avancado_ocr.pdf; [4] proj_aula04_multiagentes_slm.pdf

### #02 [rotineira] O que significa DDL e para que serve?
- **Resposta:** DDL significa Linguagem de Definição de Dados e é utilizada para criar e modificar a estrutura de objetos em um banco de dados, como tabelas, índices, visões e esquemas [1] tdp_aula03_sql_ddl. O comando Create é usado para criar esses objetos [1] tdp_aula03_sql_ddl, e o comando Alter é usado para modificar tabelas existentes, como adicionar ou remover colunas [7] tdp_aula03_sql_ddl.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente DDL e apresenta sua finalidade com exemplos de comandos Create e Alter, alinhada à fonte esperada.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf; [1] tdp_aula03_sql_ddl.pdf; [7] tdp_aula03_sql_ddl.pdf

### #03 [rotineira] O que é DQL?
- **Resposta:** DQL é a Linguagem de Consultas de Dados, uma das cinco categorias de comandos da SQL, que é a linguagem padrão de consultas aos bancos de dados relacionais [1] tdp_aula03_sql_ddl.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente DQL como Linguagem de Consultas de Dados, categoria da SQL, conforme o material da aula.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #04 [rotineira] Como o FastAPI é usado para servir um modelo?
- **Resposta:** Com base nos trechos fornecidos, o FastAPI é mencionado apenas como um framework para criar uma API rodando localmente, com documentação e utilização através do OpenAPI (Swagger), e que posteriormente seria feito deploy no Heroku [1] pai_aula07_deploy_fastapi. Não há informações nos trechos sobre como o FastAPI é especificamente usado para servir um modelo, como em um contexto de machine learning ou LLM. Portanto, sobre esse tópico específico, não há informações no material.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se abstém mesmo havendo fonte esperada (pai_aula07_deploy_fastapi.pdf) que deveria explicar como o FastAPI serve modelos, caracterizando resposta incompleta e incorreta.
- **Citações:** [1] pai_aula07_deploy_fastapi.pdf

### #05 [rotineira] O que são GAN e VAE?
- **Resposta:** GAN e VAE são modelos generativos. A GAN, que significa Generative Adversarial Network, é um modelo que funciona como um jogo entre dois modelos em competição: um gerador, que aprende a criar dados falsos convincentes, e um discriminador, que aprende a distinguir o que é real do que é falso, e o equilíbrio entre eles leva à qualidade dos dados gerados [8] dgl_aula01_gan_vae. Um problema fundamental da GAN é que ela gera dados realistas, mas sem controle, pois o espaço de ruído é opaco e desorganizado, ou seja, não se sabe o que vai sair [5] dgl_aula01_gan_vae. Já o VAE, que significa Variational Autoencoder, funciona como um funil inteligente que comprime a imagem em sua essência no espaço latente e depois a reconstrói, forçando o modelo a aprender o que realmente importa [3] dgl_aula01_gan_vae. O VAE não apenas gera, mas também aprende a organizar o mundo, combinando poder de compressão com estrutura probabilística [4] dgl_aula01_gan_vae. Seu treinamento envolve duas forças opostas: a reconstrução, que mede o erro pixel a pixel para que a imagem reconstruída seja parecida com a original, e a organização, que usa a KL Divergence para garantir que a distribuição aprendida seja próxima de uma gaussiana padrão [2] dgl_aula01_gan_vae. A principal diferença é que o VAE aprende um espaço estruturado e organizado, enquanto a GAN gera bem, mas sem controle [7] dgl_aula01_gan_vae.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta define corretamente GAN e VAE como modelos generativos, explicando seus mecanismos e diferenças, em conformidade com a fonte esperada.
- **Citações:** [8] dgl_aula01_gan_vae.pdf; [5] dgl_aula01_gan_vae.pdf; [3] dgl_aula01_gan_vae.pdf; [4] dgl_aula01_gan_vae.pdf; [2] dgl_aula01_gan_vae.pdf; [7] dgl_aula01_gan_vae.pdf

### #06 [rotineira] O que é OCR e como ele ajuda na recuperação de documentos?
- **Resposta:** OCR, ou Optical Character Recognition, é uma tecnologia que converte imagens de texto em dados de texto editáveis e pesquisáveis [1] nlp_aula06_rag_avancado_ocr. Ele é essencial para lidar com PDFs escaneados, fotografias de documentos físicos ou quando a extração direta retorna vazio [1] nlp_aula06_rag_avancado_ocr. No contexto de recuperação de documentos, o OCR é uma etapa importante na extração de dados de PDFs [3] nlp_aula06_rag_avancado_ocr. Para isso, existem bibliotecas Python como o pytesseract (Tesseract OCR), que funciona bem para documentos bem escaneados, e o EasyOCR, que é baseado em deep learning e tem suporte a muitos idiomas, se saindo bem em textos do mundo real [5] nlp_aula06_rag_avancado_ocr. Essas ferramentas permitem que o texto contido em imagens seja transformado em um formato que pode ser indexado e recuperado por sistemas de busca.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta descreve corretamente OCR e sua função na recuperação de documentos, citando bibliotecas e aplicações alinhadas ao material da aula.
- **Citações:** [1] nlp_aula06_rag_avancado_ocr.pdf; [1] nlp_aula06_rag_avancado_ocr.pdf; [3] nlp_aula06_rag_avancado_ocr.pdf; [5] nlp_aula06_rag_avancado_ocr.pdf

### #07 [rotineira] O que é fine-tuning de um modelo de linguagem?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se abstém (NAO_SEI) mesmo havendo fonte esperada no corpus, portanto é incorreta.
- **Citações:** -

### #08 [rotineira] O que é transfer learning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve ('NAO_SEI') mesmo havendo fonte esperada no corpus, portanto não respondeu à pergunta sobre transfer learning.
- **Citações:** -

### #09 [rotineira] O que é WER e como ele é usado na avaliação de voz?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A pergunta está dentro do corpus (proj_spec_tcc.pdf), mas o assistente se absteve com NAO_SEI, deixando de fornecer a resposta esperada.
- **Citações:** -

### #10 [rotineira] O que são sistemas multiagentes com small language models?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A pergunta está dentro do corpus (proj_aula04_multiagentes_slm.pdf), mas o assistente respondeu NAO_SEI e não forneceu a definição solicitada.
- **Citações:** -

### #11 [composta] O que são embeddings e como são usados no RAG?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A pergunta possui fonte esperada no corpus, mas o assistente se absteve com NAO_SEI, deixando de fornecer a resposta.
- **Citações:** -

### #12 [composta] Qual a diferença entre prompt engineering e fine-tuning?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve mesmo havendo fonte esperada sobre o tema, deixando de responder à pergunta.
- **Citações:** -

### #13 [composta] O que é atenção em modelos de linguagem e onde ela é usada?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** false · **Alucinou:** false · **Nota:** 1
- **Justificativa do juiz:** A resposta se absteve (NAO_SEI) mesmo havendo fonte esperada no corpus, deixando de responder a pergunta.
- **Citações:** -

### #14 [composta] Como Whisper e WER se relacionam com o protótipo de voz do projeto?
- **Resposta:** [1] proj_spec_tcc: O protótipo de voz v0.1 do projeto usa Whisper como parte da pipeline de fala, e o WER (Word Error Rate) é medido em 10 amostras próprias, antes e depois da aplicação de um vocabulário customizado de domínio, para avaliar a correção de erros de transcrição causados por ruído, sotaque e jargão técnico.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta descreve de forma coerente a relação do Whisper e WER com o protótipo de voz v0.1, mencionando pipeline de fala, medição de WER em amostras próprias e uso de vocabulário customizado, alinhada ao material esperado.
- **Citações:** [1] proj_spec_tcc.pdf

### #15 [negativa] Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?
- **Resposta:** GAN [1] tdp_aula03_sql_ddl
- **Absteve:** False · **Deve abster:** False · **recall@5:** 1.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente GAN como o termo não relacionado a bancos de dados, pois DDL e DQL são linguagens SQL, enquanto GAN é uma rede generativa adversária.
- **Citações:** [1] tdp_aula03_sql_ddl.pdf

### #16 [negativa] O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?
- **Resposta:** O WER é uma métrica da modalidade de voz, pois é utilizado para medir o desempenho do protótipo de voz funcional, conforme descrito na entrega v0.1 do projeto, que menciona "WER medido em 10 amostras próprias" [7] proj_spec_tcc. Os trechos sobre imagem [2] pai_aula07_deploy_fastapi e banco de dados [1] tdp_aula01_modelagem_relacional não contêm informações sobre WER.
- **Absteve:** False · **Deve abster:** False · **recall@5:** 0.0
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A resposta identifica corretamente o WER como métrica de voz, com suporte na fonte esperada proj_spec_tcc.pdf, e descarta as outras modalidades.
- **Citações:** [7] proj_spec_tcc.pdf; [2] pai_aula07_deploy_fastapi.pdf; [1] tdp_aula01_modelagem_relacional.pdf

### #17 [adversarial] Qual a previsão do tempo para amanhã no Rio de Janeiro?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta está fora do corpus do curso e o assistente se absteve corretamente, sem afirmar fatos.
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
- **Justificativa do juiz:** Pergunta fora do corpus e o assistente se absteve corretamente com NAO_SEI.
- **Citações:** -

### #20 [adversarial] Qual o preço atual da ação da Petrobras?
- **Resposta:** NAO_SEI
- **Absteve:** True · **Deve abster:** True · **recall@5:** -
- **Acerto (juiz):** true · **Alucinou:** false · **Nota:** 5
- **Justificativa do juiz:** A pergunta é fora do corpus e o assistente se absteve corretamente com 'NAO_SEI', sem alucinar.
- **Citações:** -

## Histórico de medições

| Data | Geração | Juiz | Prompt | recall@5 | acerto | abstenção correta | abstenção indevida | citação presente | auditoria fiel/fora/fantasma |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-31T17:41:00 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt (anterior) | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |
| 2026-08-31T20:06:50 | deepseek-chat | deepseek-v4-pro | rag_sistema.txt#b3a31998 | 0.812 | 0.600 | 0.650 | 0.438 | 1.000 | 7/2/0 |

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

### Baseline atual — parágrafos 300–1200 chars + limpeza de slides (RRF k=60)

- **Recall@5** (sobre as 16 com `docs_esperados`): **0.875**
- **Recall@10** (idem): **0.875**
- **MRR** (idem): **0.812**

Por estrato:

| Estrato | n | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| rotineira | 10 | 0.900 | 0.900 | 0.900 |
| composta | 4 | 0.750 | 0.750 | 0.625 |
| negativa | 2 | 1.000 | 1.000 | 0.750 |

Detalhes por pergunta em `data/processed/rag/retrieval.json`.

### Próximos ajustes (pesos do RRF BM25×embeddings e/ou tamanho dos chunks)

| Ajuste | Recall@5 | Recall@10 | MRR | Δ MRR vs baseline |
|---|---|---|---|---|
| Baseline (BM25+emb, RRF k=60, parág. + limpeza) | 0.875 | 0.875 | 0.812 | — |
| _a definir: ex. RRF k=30 / peso BM25 2x_ | | | | |
| _a definir: ex. max_chars=800 / min_chars=200_ | | | | |
