# Avaliação por estrato e régua doc × página

## Status

accepted

## Contexto e decisão

A acurácia agregada escondia onde o sistema falhava. A partir da v0.6 passamos a medir
**por estrato** (rotineira / composta / negativa / adversarial) e a usar uma **régua
dupla** de retrieval: acerto no **documento** e na **página**.

## Consequences

- A falha real apareceu: o estrato **negativa** era o elo fraco (0.125) e a **página**
  (0.605) rende menos que o **documento** (0.659) — o limitante é a **granularidade da
  fonte**, não o modelo.
- A métrica dupla virou régua oficial dos alvos da v1.0.
- Toda medição nova reporta `por_estrato`; não se compara só o total.
