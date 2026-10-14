# Índice da documentação — `projeto_final`

> **Como ler.** Cada release tem **dois** arquivos canônicos: `v0X.md` (todos os
> detalhes: dataset, processo de avaliação, decisões) e `v0X_evidencia.md` (os números
> medidos). Comece pelo [`RESUMO_ORIENTADOR.md`](../RESUMO_ORIENTADOR.md) (executivo) e
> pelo [`README.md`](../README.md) (técnico). Notas de processo e cenários ficam em
> [`notas/`](notas/); decisões de arquitetura em [`adr/`](adr/).

## Releases (ordem oficial)

| Release | Detalhes | Evidência |
|---|---|---|
| v0.1 · Voz | [`v01.md`](v01.md) | [`v01_evidencia.md`](v01_evidencia.md) |
| v0.2 · RAG | [`v02.md`](v02.md) | [`v02_evidencia.md`](v02_evidencia.md) |
| v0.3 · Imagem | [`v03.md`](v03.md) | [`v03_evidencia.md`](v03_evidencia.md) |
| v0.4 · Agentes | [`v04.md`](v04.md) | [`v04_evidencia.md`](v04_evidencia.md) |
| v0.5 · Adaptação | [`v05.md`](v05.md) | [`v05_evidencia.md`](v05_evidencia.md) |
| v0.5b · Corpus/golden | [`v05b.md`](v05b.md) | [`v05b_evidencia.md`](v05b_evidencia.md) |
| v0.6 · Avaliação | [`v06.md`](v06.md) | [`v06_evidencia.md`](v06_evidencia.md) |
| v0.7 · Pós-v1.0 (apêndice) | [`v07.md`](v07.md) | [`v07_evidencia.md`](v07_evidencia.md) |
| v1.0 · Produção | [`v10.md`](v10.md) | [`v10_evidencia.md`](v10_evidencia.md) |

> **v0.7** é um apêndice **pós-v1.0** (exploração); o roadmap oficial da disciplina
> termina em **v1.0**. Os cenários de roteador (Kev, Laya) e o TCO detalhado da v0.7
> estão em [`notas/`](notas/).

## Notas de processo e cenários (`notas/`)

Fragmentos por passo das releases (P0–P6), cenários de roteador, análises e guias de
processo — material de apoio, **não** é a evidência oficial. Ex.: `notas/v05b_p3_slm_ab.md`,
`notas/v06_roteador_kev.md`, `notas/v07_roteador_laya_ft.md`, `notas/v07_tco.md`,
`notas/v10_remediacao.md`, `notas/avaliacao_aula08.md`, `notas/colab_experimentos.md`.

## Decisões de arquitetura (`adr/`)

- [`0001-nao-treinar.md`](adr/0001-nao-treinar.md) — RAG + frontier é a produção; fine-tune refutado.
- [`0002-space-publico-dataset-privado.md`](adr/0002-space-publico-dataset-privado.md) — corpus fora do repo (LGPD).
- [`0003-avaliacao-por-estrato-e-regua-pagina.md`](adr/0003-avaliacao-por-estrato-e-regua-pagina.md) — medir por estrato; a página é o limitante.

## Convenções

- Evidência oficial de release = **Markdown** (JSON é dado intermediário).
- Golden set e critério de rótulo são **congelados** (`data/golden_set/`).
- Avaliadores ficam em `scripts/avaliadores/` (fora de `src/`).
