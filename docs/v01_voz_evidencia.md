# v0.1 - Evidencia - WER antes/depois do vocabulario

- **Frases avaliadas:** 10
- **Backend:** faster-whisper local (CPU) (modelo 'small')
- **Custo:** US$ 0.00 (100% local)
- **WER medio sem vocabulario:** 0.2290
- **WER medio com vocabulario:** 0.0863
- **Ganho absoluto:** +0.1427

| frase | audio | WER sem | WER com |
|---|---|---|---|
| 1 | 01.ogg | 0.2222 | 0.0000 |
| 2 | 02.ogg | 0.3333 | 0.1667 |
| 3 | 03.ogg | 0.1667 | 0.0000 |
| 4 | 04.ogg | 0.0000 | 0.0000 |
| 5 | 05.ogg | 0.3750 | 0.1250 |
| 6 | 06.ogg | 0.1000 | 0.0000 |
| 7 | 07.ogg | 0.8571 | 0.5714 |
| 8 | 08.ogg | 0.1250 | 0.0000 |
| 9 | 09.ogg | 0.0000 | 0.0000 |
| 10 | 10.ogg | 0.1111 | 0.0000 |

## Pipeline v0.1

```mermaid
flowchart LR
    A[Navegador - microfone] -->|audio/webm| B[FastAPI POST /chat]
    B --> C[faster-whisper small CPU]
    C -->|transcricao| D[DeepSeek deepseek-chat]
    D -->|resposta texto| E[piper-tts pt-BR]
    E -->|audio/wav| F[Navegador - player]
```

## Conclusoes

- O vocabulario de dominio reduziu o WER medio de 0.2290 para 0.0863 (ganho absoluto de 0.1427).
- A acuracia media subiu de 77.10% para 91.37% (+14.27 pontos percentuais).
- 7 das 10 frases atingiram WER 0.0000 com o vocabulario.
- A frase 7 ("Agentes multimodais combinam texto, imagem e audio") permaneceu como o caso mais desafiador, com WER 0.5714 mesmo com vocabulario.
- Todo o pipeline ASR/TTS e local (custo zero); apenas o LLM DeepSeek consome API.
