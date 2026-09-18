# candidate_win_model_v10 — asesor por escalón anclado en la RECETA de planta (iteración 15, 2026-09-17)

Código: `asesor_receta_v10.py`. Experimentos: `experimentos/v13/` (E15-01 Fable, E15-02 auditoría adversarial Sonnet, E15-03 replay). Fichas previas:
`candidate_win_model_v9_1.md` (piloto, protocolo, potencia), `candidate_win_model_v9.md`. Hallazgos §26.

## 0. Qué cambió y por qué importa

La hoja **"Alimentación"** del Excel (nunca usada por el pipeline) trae la **receta por escalón de Reducción**: GN, O₂, aire y carbón para R01-R04 de los
362 batches (niveles discretos, revisados por tramos de calendario; la auditoría la considera ex-ante con alta confianza). Con ella la exposición deja de
depender de un modelo de "práctica habitual" (la fuente del ruido que debilitaba el replay estricto de v9.1):

    d_jt = (ejecutado_jt − receta_jt) / sd_pasado(j, orden)        x_bj = media_{t ∈ R0..R3} d_jt        (exacta, conocida al inicio del escalón)

| | v9.1 (práctica habitual por HGB) | **v10 (receta)** |
|---|---|---|
| GN → KPI, DEV conjunta | −1.50 pp (t −3.2) | **−1.58 pp (t −3.9)**; dross **+1.23 (t +4.1)**; total −1.17 (t −3.1) / dross +1.21 (t +4.1) |
| Prueba prospectiva (β de los últimos 100 batches → 10 siguientes) | t 2.3; p corregida 0.010-0.014 | t 2.6-2.7; **p corregida por 12 combinaciones ventana × contracción: 0.002 (bloques 20), 0.004 (bloques 40), 0.006 (circular), 0.002 (signo), 0.012 (bloques 10)** |
| Especificaciones razonables con p < 0.01 | 43 % de 81 | 61 % de 162; **93 % (100 % < 0.05) con GN + carbón juntos**; carbón solo 0 % |
| Fragilidad (quitar 5 / 10 / 20 batches: % con t > 1.645) | 95 / 68 / 59 % | **100 / 92 / 97 %**; leave-one-block-out t 1.89-3.04 |
| Replay ESTRICTO del asesor (todo con el pasado) | pendiente 0.58, t 1.4-1.9, perm. 0.09-0.16 | pendiente **0.64** (JS) / 0.51, t 1.8-2.2, Spearman parcial **0.20 (p 0.003)**, DEV 0.27 (p 0.0006), terciles −1.04 / +0.30 / +0.74 pp |
| Ganancia calibrada de seguir la recomendación | +0.57..+0.80 pp, IC que toca 0 | **+0.58 pp/batch, IC95 [0.13, 1.13], P(> 0) 0.999** (sin JS +0.69 [0.25, 1.23]; en unidades físicas +0.42..+0.66, IC siempre > 0) |
| Mecanismo GN → dross en modo estricto | nulo (t −0.7) | **t +4.1** (la exposición es exacta) |

## 1. Definición

- **STATE**: receta del escalón (GN, carbón, O₂, aire), orden, T horno previa (resguardo). **CONTROL**: GN y carbón ejecutados; O₂ acompaña al GN
  (ΔO₂ = 2·ΔGN, piso 0); aire intacto; Fusión = receta.
- **Valor**: KPI_b = controles + β_G·x_bG + β_C·x_bC, OLS HC3 sobre los últimos 100 batches, re-anclado cada 10; contracción James-Stein β̃ = β̂·max(0, 1 − 1/t²).
- **Recomendación**: a* = receta + min(1, |t|/2)·signo(β̂)·sd_desvío si |t| ≥ 1 (si no, "ejecutar la receta"); caja [P5, P95] de lo ejecutado en el orden;
  carbón R0 ≤ P90; sin recorte de GN con T_prev < P10. La sd del desvío, las cajas y β se estiman sólo con batches anteriores.
- **Agregación**: x_b = media de los 4 escalones (fase); valor del batch = Σ_palanca β̃·x; los efectos físicos por escalón (θ de v8) siguen disponibles para explicar.

## 2. Política en el replay (`experimentos/v13/e15_03_resumen_politica.csv`)

| Período | GN recomendado − receta (Nm³/min; % de escalones) | Carbón recomendado − receta (kg/min; %) |
|---|---|---|
| batches 140-199 | R0 −2.6 (97 %), R1 −2.5 (93 %), R2 −0.9 (70 %), R3 −0.9 (55 %) | ≈ 0 (33 %) |
| 200-299 | −1.1 / −1.1 / −0.9 / −0.6 (55-31 %) | +0.9 / +0.8 / +0.5 / +0.2 (20 %) |
| 300-361 (lockbox) | −0.8 / −1.0 / −0.7 / −0.6 (47-36 %; el resto "horno frío" o β ≈ 0) | +2.7 / +0.9 / +1.3 / +1.4 (52 %) |

Dato operativo: los operadores ejecutan el GN +0.1..+1.0 Nm³/min **por encima** de la receta y el carbón **+4.5 kg/min sobre la receta en R0-R1 y −4.5..−5.7
por debajo en R2-R3**. Sólo cumplir la receta (desvío 0) vale +0.20 pp/batch [0.02, 0.53] en el replay (la auditoría, con otra métrica, lo da nulo: no se
afirma); la ganancia está en ejecutar el GN por debajo de la receta cuando β lo respalda.

## 3. Estado frente al criterio de suficiencia para planta (fijado en la iteración 12)

| Criterio | Estado |
|---|---|
| 1. Relación prospectiva > 0, p corregida < 0.01 estable | **Cumple en la especificación principal bajo 4 de 5 nulas (0.002-0.006; 0.012 con bloques de 10) y en el 93 % de las variantes con GN + carbón; no es uniforme**: con estandarización estricta centrada 0.013-0.027; replay estricto del asesor p 0.013-0.034 (Spearman 0.003). Rango honesto del auditor: 0.01-0.03 |
| 2. Calibración en [0.6, 1.4] | **Cumple en el punto** (0.63-0.64 con JS; IC95 [0.17-0.25, 1.0-1.2]); 0.49-0.51 sin contracción |
| 3. No negativa en lockbox | Cumple débilmente (≈ 0): el β estático de DEV se invierte en fin de campaña (+1.36, t 1.8) y el asesor adaptativo lleva β_GN a t ≈ 0 y recomienda "ejecutar la receta" |
| 4. Ganancia calibrada con IC que excluya 0 | **Cumple**: +0.58 [0.13, 1.13] (todas las variantes con límite inferior 0.04-0.25) |
| 5. Mecanismo por palanca | **Cumple para GN** (dross t +4.1 con exposición exacta; escalón: GN → −retención de FeO p 1e-4). Carbón: no significativo solo; entra sólo con \|t\| ≥ 1 |
| 6. Ejecutable al inicio del escalón | **Cumple exactamente** (la receta existe antes del batch; no hace falta ensayo ni modelo) |

Salvedades que quedan: (a) placebo lead 1 no nulo (t 1.83) y el desvío de GN correlaciona con el KPI del batch anterior (r −0.14, p 0.009) — controlando esa
persistencia el resultado se mantiene (t 2.47, p 0.002); (b) **inestabilidad de régimen en fin de campaña**: el asesor se abstiene, no gana; (c) observacional:
la confirmación causal sigue siendo el piloto aleatorizado de v9.1 (§4 de esa ficha), ahora más simple: brazos = receta ± 1 sd del desvío.

## 4. Reproducir

    .venv/Scripts/python.exe experimentos/v13/e15_01_receta.py        # exposiciones por receta, OLS conjunta, prueba prospectiva
    .venv/Scripts/python.exe asesor_receta_v10.py                      # replay estricto + evidencia + ganancia calibrada (≈ 1 min)
    experimentos/v13/e15_02_resultados.md                              # auditoría adversarial completa
