# Assistente Master IAG & LLM — domínio

Assistente generativo que responde dúvidas sobre os materiais didáticos do Master
IAG & LLM (PUC-Rio), citando a fonte e abstendo-se fora do corpus. Este glossário
fixa o vocabulário **de avaliação e arquitetura** usado no projeto — para que
código, docs e conversa usem os mesmos termos.

## Linguagem

### Avaliação

**Release**:
Uma entrega do roadmap (v0.1 → v1.0), aprovada por commit, com evidência medida.
_Avoid_: versão, entrega

**Estrato**:
Cada classe de pergunta do golden set: **rotineira** (fato direto), **composta**
(raciocínio sobre 2+ fontes), **negativa** (resposta não está no corpus) e
**adversarial** (fora do corpus, escrita à mão para sondar abstenção).
_Avoid_: categoria, tipo

**Golden set**:
Conjunto **congelado** de perguntas com resposta/fonte esperada, usado como régua
de avaliação de uma release. Nunca é re-editado para melhorar métrica.
_Avoid_: dataset de teste, benchmark

**Critério de rótulo**:
Documento que define, por escrito, o que conta como resposta correta — de modo que
duas pessoas cheguem ao mesmo veredito. Congelado junto com o golden.
_Avoid_: rubrica do juiz, gabarito

**Régua (doc × página)**:
Métrica dupla de retrieval: acerto no nível do **documento** e da **página**.
A página é mais fina e é o limitante real do sistema.
_Avoid_: recall, acurácia

**Evidência**:
O número medido de uma release, registrado em **Markdown** (`docs/v0X_evidencia.md`).
JSON é dado intermediário, não evidência.
_Avoid_: resultado, métrica

**Abstenção (`NAO_SEI`)**:
Resposta correta do sistema quando a pergunta está fora do corpus, acompanhada de
motivo. Distinta de erro.
_Avoid_: recusa, fallback

**Alucinação**:
Resposta afirmativa sem ancoragem no corpus recuperado (veredito do juiz).
_Avoid_: erro, ruído

### Arquitetura e roteamento

**Rota simples / complexa**:
As duas rotas do fluxo multiagente. **Simples** → SLM local (custo US$ 0);
**complexa** → LLM frontier remoto.
_Avoid_: caminho, modo

**Cascade (`R1`/`R2`)**:
Roteador de produção que decide a rota. **R2** (TF-IDF + XGBoost) decide ~85% a
<2 ms/US$ 0; `INDETERMINADO` cai no **R1** (SLM local few-shot). Limiar θ=0.60.
_Avoid_: classificador, roteador SLM

**Frontier**:
O LLM remoto grande (`deepseek-v4-pro`) usado na rota complexa e como baseline
RAG (pro em todas as perguntas).
_Avoid_: LLM grande, API

**Regra V5**:
Regra **determinística** pós-retrieval que responde o estrato negativa tipo A sem
LLM (matcher literal ∪ janela de 14 tokens ∪ mapa pt→SQL); bypass do gerador, US$ 0.
_Avoid_: heurística, filtro

### Operação

**TCO**:
Custo total por consulta, somando inferência + infraestrutura. Régua oficial da v1.0
(≤ US$ 0,001/pergunta).
_Avoid_: custo, preço

**Space público / dataset privado**:
A separação do deploy: o **Space** (público) contém só o código; o **dataset
privado** guarda corpus e índices, baixados no boot (LGPD/direitos autorais).
_Avoid_: deploy, produção
