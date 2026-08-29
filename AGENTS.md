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

## Comandos úteis

```bash
cd projeto_final
source .venv/bin/activate   # ativar ambiente
uv sync                     # sincronizar dependências
pytest                      # rodar testes
python -m src.projeto_final # executar módulo principal
```

## Não fazer

- Não subir PDFs brutos do curso em repositório público (direitos autorais/LGPD).
- Não commitar `.env`, `.venv/`, caches, dados brutos.
- Não pular releases nem misturar escopo de duas aulas num commit só.
- Não copiar código diretamente de `projeto2/` sem reimplementar e entender cada parte.
