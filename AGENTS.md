# AGENTS.md - Contrato permanente do projeto

> Este arquivo é o contrato de trabalho para agentes de IA (Claude Code, Codex, etc.)
> e para qualquer pessoa que continue este projeto. Leia antes de qualquer mudança.

## Domínio e proposta

- **Domínio:** materiais didáticos do Master IAG & LLM (PUC-Rio) - ementas, PDFs, slides e notebooks.
- **Proposta:** assistente generativo que responde dúvidas sobre o curso, citando a
  fonte e abstendo-se quando a pergunta está fora do corpus.
- **Corpus:** `data/raw/` (10 documentos do próprio curso, fora do Git).

## Regras do curso (avaliação)

1. Uma **release por aula**, avaliada por **commit** - nunca quebre uma release já aprovada.
2. Toda release fecha com **evidência medida** (WER, recall@k, taxa de abstenção, etc.)
   registrada no `README.md` ou em `docs/`.
3. Sequência obrigatória: v0.1 voz → v0.2 RAG → v0.3 imagem → v0.4 agentes → v0.5
   adaptação → v0.6 avaliação → v1.0 produção.
4. Cada release deve ser **testada manualmente e aprovada** pelo aluno antes de avançar.
5. O protótipo `projeto2/` é fonte de consulta para decisões técnicas, mas o código
   do `projeto_final/` deve ser reimplementado do zero.

## Stack e ambiente

- Python >=3.12 - ambiente gerenciado por `uv` - venv em `.venv` (fora do Git).
- Dependências em `pyproject.toml` - adicionar **por release**, sem bloat.
- APIs via variáveis de ambiente (nunca chaves no código): `.env` (fora do Git).

## Convenções

- `data/raw/` e `data/processed/` **não entram no Git** (conteúdo bruto + índices).
- `data/golden_set/` **entra no Git** (avaliação reprodutível).
- Prompts de sistema/etapa vivem em `prompts/` como arquivos versionados.
- Testes em `tests/` com `pytest`; rodar `pytest` antes de fechar cada release.
- Commits por release com mensagem `v0.X - <descrição>`; tags `v0.X` quando aprovado.
- Scripts de avaliação padronizados em `scripts/avaliadores/avaliar_v{n}.py`
  (ex.: `avaliar_v1.py` para voz, `avaliar_v2.py` para RAG) — avaliadores não ficam em `src/`.
- **Dataset único por release:** a v0.2 avalia apenas `data/golden_set/rag/perguntas.json`
  (cópia da tag `v0.2` do projeto2). Um novo dataset substitui o anterior — não manter
  dois datasets avaliados para a mesma release.
- **Evidência em Markdown:** JSONs podem servir como dado de entrada/intermediário, mas a
  evidência oficial de uma release é o resultado em Markdown em `docs/`
  (ex.: `docs/v02_evidencia.md`).
- **Documentação consolidada:** cada release tem, no máximo, dois arquivos —
  `docs/v0X.md` (todos os detalhes) e `docs/v0X_evidencia.md` (evidência medida).
  A documentação deve explicar o **dataset** (origem, estrutura, estratos) e o
  **processo de avaliação** (métricas, critérios, como reproduzir).

## Comandos úteis

```bash
cd projeto_final
source .venv/bin/activate   # ativar ambiente
uv sync                     # sincronizar dependências
pytest                      # rodar testes
python -m src.projeto_final # executar módulo principal
python scripts/avaliadores/avaliar_v1.py  # avaliar v0.1 (WER voz)
python scripts/avaliadores/avaliar_v2.py  # avaliar v0.2 (RAG)
python scripts/avaliadores/avaliar_v3.py  # avaliar v0.3 (imagem/OCR; --e2e p/ LLM)

# Deploy (Hugging Face Space privado 'assistente-master-iag')
bash deploy/deploy.sh --dry-run   # monta o bundle sem publicar (revisar)
bash deploy/deploy.sh             # cria/atualiza o Space e publica
```

## Deploy (Hugging Face Space)

- **API publica:** Space `assistente-master-iag` (SDK `docker`, hardware
  `cpu-upgrade`, sleep 1 h) — **sem corpus no repo** (nao expor PDFs/LGPD).
- **Corpus privado:** dataset `assistente-master-iag-dados` carrega `data/raw`
  + indices em runtime no boot (entrypoint → `snapshot_download`).
- **Segredos:** `HF_TOKEN` e `DEEPSEEK_API_KEY` em `deploy/.env` (fora do Git);
  secrets do Space: `DEEPSEEK_API_KEY`, `HF_TOKEN_READ` (leitura do dataset) e
  `HF_DATA_REPO`. Nunca commitar tokens.
- **Bundle:** `deploy/deploy.sh` monta `deploy/build/` com o codigo (sem `data/`),
  sobe o dataset privado e faz push no Space publico (ver `deploy/README.md`).

## Não fazer

- Não subir PDFs brutos do curso em repositório público (direitos autorais/LGPD).
- Não commitar `.env`, `.venv/`, caches, dados brutos.
- Não pular releases nem misturar escopo de duas aulas num commit só.
- Não copiar código diretamente de `projeto2/` sem reimplementar e entender cada parte.
- Não fragmentar a documentação/evidência de uma release (máx.: `docs/v0X.md` +
  `docs/v0X_evidencia.md`, com evidência em Markdown).
- Não manter mais de um dataset avaliado por release nem usar evidência em JSON
  sem o Markdown correspondente.
