# E10-04a — Anclajes del objetivo de Reducción: K, w con IC, corrección EIV, objetivo lineal en kg, legado y barrido de w

Script `e10_04a_anclajes.py` (log `e10_04a_log.txt`; CSV `e10_04a_anclaje_base.csv`, `e10_04a_eiv.csv`, `e10_04a_eiv_corregido.csv`,
`e10_04a_legado.csv`, `e10_04a_lineal_kg.csv`, `e10_04a_barrido_w.csv`). Datos `v8_lib.cargar_df()`, tabla de batch `v8_lib.construir_batch`
(KPI = `recuperacion_refinada_pct`, controles `CONTROLES_BATCH`), OLS HC3, bootstrap por batch (1000). DEV n 297 / lockbox 63 / total 360.

## 1. Anclaje base (KPI del batch actual sobre Σ_R ln_sn_dep y Σ_R ln_feo_ret)

| muestra | agregación | K_Sn [IC] | K_FeO [IC] | w = K_FeO/K_Sn [IC boot] |
|---|---|---|---|---|
| DEV | suma por escalón | 2.85 [1.67, 4.04] | 17.26 [11.04, 23.48] | **6.05 [4.02, 10.21]** |
| DEV | cociente de inventarios (ini/fin) | 2.82 [1.61, 4.02] | 17.39 [11.14, 23.65] | 6.18 [4.10, 10.50] |
| lockbox | suma por escalón | 4.05 [1.42, 6.68] | 2.82 [−12.7, 18.4] (p 0.72) | 0.70 [−8.6, 4.2] |
| total | suma por escalón | 3.22 [2.18, 4.25] | 15.71 [10.00, 21.41] | 4.88 [3.42, 6.88] |
| DEV, sólo batches con ensayo en R3 (n 203) | suma | 2.35 [0.85, 3.84] | 16.25 [8.23, 24.27] | 6.93 [4.06, 17.0] |

Lecturas: (i) la agregación por suma y por cociente de inventarios dan lo mismo (identidad telescópica confirmada, Δ < 0.15 en w);
(ii) el anclaje NO depende de los 105 batches sin ensayo en R3 (w 6.9 vs 6.05, IC solapados; corrige la cifra "3 % sin masa" del
dictamen §4.6: el anclaje es robusto a ese 29 %); (iii) **w ≈ 0.7 en lockbox** (n 63, K_FeO no significativo) confirma la duda del
comité: en fin de campaña el KPI premia agotar Sn y no distingue la retención de FeO. Con 63 batches el IC de w en lockbox es [−8.6, 4.2]:
no contradice DEV con potencia, pero exige que la política conserve el signo bajo w bajo.

## 2. Corrección por errores-en-variables (EIV)

Fiabilidad λ = 1 − var(ruido)/var(total) de las sumas por batch. Vía (a) ruido de ensayo propagado (2 % Sn, 2.8 % FeO en los dos
extremos del cociente): λ_FeO 0.82, λ_Sn 0.997 → K_FeO corregido 21.1, w 7.37. Vía (b) residuo OOF de un HGB por escalón como
"ruido": λ negativo (el residuo del modelo NO es ruido de medida, es señal no explicada) → **descartada** (produce w 0.4-120, absurdo).
Decisión (criterio 5 del diseño: corregir sólo si λ ∈ [0.6, 0.95]): la corrección por la vía (a) es admisible (λ 0.82) pero mueve w
de 6.05 a 7.4, dentro del IC bootstrap [4.0, 10.2]. **Se mantiene w = 6.09 / K = 2.893 (CONFIG v6/v7 sin legado) como base** y w 7.4
entra en el rango de sensibilidad; no se adopta como punto porque la fiabilidad depende de un ruido de ensayo supuesto (2.8 %), no medido.

## 3. Legado (KPI del batch siguiente)

| muestra | Σ ln_feo_ret → KPI_next (controlando KPI_actual) | placebo (KPI_prev) |
|---|---|---|
| DEV | +9.37 [3.59, 15.16], p 0.0015 | +2.09, p 0.56 |
| lockbox | **+19.75 [4.28, 35.22], p 0.012** | −13.2, p 0.14 |
| total | +10.42 [5.04, 15.80], p 0.0001 | +0.73, p 0.83 |

El legado replica en lockbox (único hallazgo del proyecto con réplica fuera de muestra). Anclaje con KPI_actual + KPI_next: DEV K 3.52 /
w 7.86 [5.11, 13.26]; lockbox K 2.96 (n.s.) / w 7.5 [−46, 36] (sin potencia). Decisión (dictamen §4.3): **base sin legado** (K 2.893,
w 6.09) y el legado como sensibilidad (K 3.52, w 7.86) hasta que planta confirme el talón de escoria; la política v8 debe conservar
el signo entre ambas.

## 4. Objetivo lineal en kg (contraste, auditoría A §4)

KPI ~ ΔSn_kg_R (Sn agotado en Reducción, t) + ΔFeO_kg_R (FeO reducido, t) + controles:

| muestra | pp por t Sn agotado | pp por t FeO reducido | λ_kg = −coef_FeO/coef_Sn [IC] |
|---|---|---|---|
| DEV | +0.170 (p 0.04) | −0.797 (p 0.007) | **4.7 [0.9, 20.7]** |
| total | +0.194 (p 0.003) | −0.547 (p 0.04) | 2.8 [0.4, 8.0] |
| lockbox | +0.184 (p 0.27) | +1.19 (p 0.06) | signo contrario (n.s.) |

Lectura pirometalúrgica: 1 t de Sn agotado vale 0.17-0.19 pp (≈ 90-100 kg de Sn a metal por t: sólo ~10 % del Sn agotado en Reducción
se convierte en recuperación marginal, el resto ya iba a salir o va a dross/polvo), y 1 t de FeO reducido cuesta 0.55-0.80 pp (≈ 280-410 kg
de Sn a metal por t de FeO, es decir, 0.35-0.5 kg Sn por kg FeO: coincide con el mecanismo hardhead 0.5-0.65 kg Sn/kg Fe de la auditoría A).
El λ en kg (2.8-4.7 kg Sn agotado por kg FeO, IC amplio) es más alto que el 0.5-1.5 que el comité propuso a priori: **el dato dice que
reducir FeO es más caro de lo que la estequiometría del dross sugiere** (polvo + legado). Para v8: `lambda_kg` = 3.0 (mediana entre DEV y
total, dentro de ambos IC) como contraste; sensibilidad 1.0-4.7.

## 5. Barrido de w (Spearman J_R(w) con el KPI)

| w | DEV | lockbox | total | tercio 0 / 1 / 2 (total) |
|---|---|---|---|---|
| 1 | 0.14 | **0.35** | 0.19 | 0.03 / 0.18 / 0.39 |
| 2 | 0.20 | 0.30 | 0.23 | 0.09 / 0.23 / 0.40 |
| 4 | 0.28 | 0.24 | 0.29 | 0.18 / 0.29 / 0.39 |
| 6.09 | **0.32** | 0.20 | **0.30** | 0.24 / 0.32 / 0.36 |
| 7.86 | 0.32 | 0.13 | 0.29 | 0.25 / 0.31 / 0.30 |
| 11 | 0.32 | 0.07 | 0.27 | 0.27 / 0.28 / 0.23 |

La meseta de DEV está en w ≥ 4; el lockbox prefiere w ≤ 2; el total es máximo en w 4-6. **w 6.09 es un compromiso defendible** (máximo
total y de DEV) pero la política debe ser robusta en w ∈ [2, 11]: en E10-04b se mide el % de escalones que conservan el signo.

## Veredicto E10-04a

- Base v8: **K = 2.893 pp/unidad, w = 6.09 [4.05, 9.99]**, sin legado; sensibilidades obligatorias: legado (3.52 / 7.86), EIV (w 7.4),
  lockbox (w ≈ 1-2), objetivo lineal en kg con λ_kg = 3.0 [1, 4.7].
- Muestras bootstrap de (K_Sn, K_FeO) de DEV para propagar el IC del uplift (E10-05): `e10_04a_anclaje_base.csv` (IC) — se regeneran en E10-05.
- El anclaje es invariante a la agregación (suma vs cociente) y a los batches sin ensayo en R3.
