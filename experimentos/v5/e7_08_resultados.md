# E7-08 — Variante del prescriptor v5 con la posición de lanza como palanca (lineal + cuadrática, signo libre)

Pregunta del usuario: la posición de lanza es controlable; la convención de la medida (mm hacia arriba o hacia abajo) no se conoce, así
que la forma debe aprender dirección y óptimo. Implementación: `modelo_predictivo_v5.activar_variante_lanza()` añade
`posicion_vertical_lanza_mm` (ACTION_primitiva del escalón) y su cuadrático centrado (2 650 mm) como palancas de signo libre en los 5
modelos, como acción optimizable (rango P1-P99 por fase: 2 027-5 230 mm en Reducción, 342-4 614 en Fusión) y en el soporte. La
posición previa sigue en el estado. Script `e7_08_variante_lanza.py` (misma cadena que E7-05: validación, 5 folds + lockbox,
evidencia) y `e7_06_evidencia_extra.py _lanza`. Salidas con sufijo `_lanza`.

## 1. Qué aprende el modelo sobre la lanza (θ por +1 sd, IC95 %, DEV)

| Modelo | lineal | cuadrático | Forma aprendida | Vértice |
|---|---|---|---|---|
| R / agotamiento Sn | +0.003 [−0.046, 0.047] | **−0.059 [−0.104, −0.019]** | cóncava: **máximo interior** | ≈ 2 715 mm (mediana histórica 2 944) |
| R / retención FeO | −0.008 [−0.020, 0.004] | **+0.014 [0.003, 0.031]** | convexa: mínimo interior (extremos retienen más FeO) | ≈ 3 254 mm |
| R / ΔT | +0.95 [−0.66, 2.29] | −1.97 [−3.43, 0.48] | cóncava (n.s.) | ≈ 3 252 mm |
| F / agotamiento Sn | −0.014 [−0.055, 0.026] | +0.018 [−0.006, 0.044] | plana | — |
| F / ΔT | −1.04 [−2.93, 0.61] | **−1.43 [−2.59, −0.30]** | cóncava | ≈ 2 251 mm |

Ningún término lineal es significativo: la lanza no tiene una "dirección" monótona sino curvatura, coherente con la teoría de inmersión
(demasiado profunda perturba la interfaz y salpica; demasiado alta pierde transferencia de masa y calor). Los θ de las otras palancas
no cambian (Cx_sn 0.183 → 0.164, GN sobre FeO −0.021 → −0.020). R² OOF/lockbox idénticos (±0.02).

Lectura de la convención: dentro del batch la medida sube durante la Fusión (546 mm en F1 → 3 843 en F6, mientras crece el baño) y
baja en Reducción (3 781 → 2 842). Es compatible con "mm = altura de la punta sobre un datum fijo, que sigue al nivel del baño"; la
recomendación de Reducción (bajar) equivaldría a mayor inmersión. Hipótesis a confirmar con planta.

## 2. Recomendación de la variante (política cross-fitted, 362 batches)

| Fase / estado | lanza histórica | recomendada | Δ mediana | % escalones baja / sube |
|---|---|---|---|---|
| R, avance < 0.84 | 3 781 | 3 675 | −162 | 57 / 24 |
| R, 0.84-0.96 | 3 295 | 2 951 | −179 | 69 / 13 |
| R, 0.96-0.975 | 2 864 | 2 558 | −175 | 73 / 12 |
| R, > 0.975 | 2 842 | 2 633 | −120 | 59 / 15 |
| F1-F2 | 546-1 156 | 694-1 309 | +32 a +65 | 6-12 / 48-52 |
| F5-F6 | 3 268-3 843 | 3 204-3 742 | 0 | 38-42 / 19-21 |

Las demás palancas cambian igual que en la base (carbón temprano, GN abajo, menos gas en Fusión).

## 3. Comparación base v5 vs variante lanza

| | base v5 | variante lanza |
|---|---|---|
| Uplift físico J (media / mediana kg Sn por batch) | 182 / 157 | **252 / 214** (Reducción 115 → 198) |
| Cadena θ→agregado→KPI, DEV (pp de KPI) | +0.20 [0.13, 0.30] | **+0.36 [0.25, 0.53]** (≈ 186 kg) |
| Cadena, lockbox | +0.28 [0.18, 0.43] | +0.32 [0.21, 0.48] |
| Adherencia Fusión `dist_F` → KPI, DEV | −0.70 pp/sd (p 0.025) | −0.56 (p 0.089) |
| Adherencia Fusión → polvo, DEV | +0.008 (p < 0.001) | +0.007 (p 0.001) |
| Adherencia Reducción `dist_R` | nula | nula |
| Dirección de la lanza (`dir_*_lanza`) → KPI | — | n.s. (Fusión −0.41 pp/sd, p 0.13; polvo −0.008 en lockbox p 0.025) |
| Emparejamiento, media batch actual+siguiente (TOTAL) | dist_total +1.8 [1.1, 2.5] | dist_total +1.2 [0.4, 1.9]; dist_F +1.8 [1.0, 2.6] |

## 4. Veredicto

La forma flexible **detecta** la lanza: curvatura significativa en 3 de 5 modelos con un máximo interior del agotamiento de Sn en
≈ 2 700 mm en Reducción, y el prescriptor la usa (baja la lanza 120-180 mm en Reducción, la sube en F1-F2). El uplift estimado sube
(+0.36 vs +0.20 pp por batch, +70 kg de J), pero esa ganancia adicional es **extrapolación del modelo**: la evidencia off-policy no
mejora (la señal de Fusión se debilita, la adherencia específica de la lanza no es significativa) y la convención física sigue sin
confirmar. Se mantiene la base v5 como ganadora y la variante como candidata para el piloto una vez que planta confirme el sentido de
la medida; el vértice (≈ 2 700 mm en Reducción) es la hipótesis a probar.
