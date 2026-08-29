"""Script de avaliacao WER da v0.1."""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from projeto_final import config
from projeto_final.voz import avaliar, ler_referencias


def atualizar_index_html(resultados: list[dict], media: dict, path: Path) -> None:
    """Substitui as linhas da tabela de WER no index.html pelos valores medidos."""
    html = path.read_text(encoding="utf-8")

    # Localiza o tbody da primeira result-table
    tbody_match = re.search(r"(<tbody>)(.*?)(</tbody>)", html, re.DOTALL)
    if not tbody_match:
        logger.warning("Nao encontrado tbody em index.html")
        return

    linhas = []
    for r in resultados:
        wer_sem = r["sem_vocab"]["wer"]
        wer_com = r["com_vocab"]["wer"]
        acc_sem = 1 - wer_sem
        acc_com = 1 - wer_com
        linhas.append(
            f"<tr><td>{r['id']:02d}.ogg</td>"
            f"<td>{wer_sem:.4f}</td><td>{wer_com:.4f}</td>"
            f"<td>{acc_sem:.4f}</td><td>{acc_com:.4f}</td></tr>"
        )

    linhas.append(
        f"<tr><td>MEDIA</td><td>{media['wer_medio_sem_vocab']:.4f}</td>"
        f"<td>{media['wer_medio_com_vocab']:.4f}</td>"
        f"<td>{1 - media['wer_medio_sem_vocab']:.4f}</td>"
        f"<td>{1 - media['wer_medio_com_vocab']:.4f}</td></tr>"
    )

    novo_tbody = "<tbody>\n" + "\n".join("".join(l) for l in linhas) + "\n</tbody>"
    html = html[:tbody_match.start(1)] + novo_tbody + html[tbody_match.end(3):]

    path.write_text(html, encoding="utf-8")
    logger.info("index.html atualizado com resultados WER")


def gerar_evidencia(out: dict, path: Path) -> None:
    """Gera o markdown de evidencia da v0.1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# v0.1 - Evidencia - WER antes/depois do vocabulario",
        "",
        f"- **Frases avaliadas:** {out['resumo']['n_frases']}",
        f"- **Backend:** {out['resumo']['backend']} (modelo '{out['resumo']['modelo']}')",
        "- **Custo:** US$ 0.00 (100% local)",
        f"- **WER medio sem vocabulario:** {out['resumo']['wer_medio_sem_vocab']:.4f}",
        f"- **WER medio com vocabulario:** {out['resumo']['wer_medio_com_vocab']:.4f}",
        f"- **Ganho absoluto:** {out['resumo']['ganho_absoluto']:+.4f}",
        "",
        "| frase | audio | WER sem | WER com |",
        "|---|---|---|---|",
    ]
    for r in out["frases"]:
        md.append(
            f"| {r['id']} | {Path(r['audio']).name} | {r['sem_vocab']['wer']:.4f} | {r['com_vocab']['wer']:.4f} |"
        )
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Evidencia salva em: {}", path)


def main() -> None:
    log_dir = config.PROCESSED_DIR / "voz"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / "avaliacao.log"), rotation="1 MB", level="DEBUG")
    prompt = config.ler_prompt_vocabulario()
    modelo = config.ASR_MODELO

    refs_path = str(config.GOLDEN_SET_DIR / "voz" / "referencias.json")
    audios_dir = str(config.GOLDEN_SET_DIR / "voz")

    logger.info("Iniciando avaliacao WER: refs={}, audios={}, modelo={}", refs_path, audios_dir, modelo)
    out = avaliar(refs_path, audios_dir, prompt=prompt, modelo=modelo)

    proc = config.PROCESSED_DIR / "voz"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "resultados.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Resultados JSON salvos em: {}", proc / "resultados.json")

    gerar_evidencia(out, config.DOCS_DIR / "v01_voz_evidencia.md")
    atualizar_index_html(out["frases"], out["resumo"], config.STATIC_DIR / "index.html")


if __name__ == "__main__":
    main()
