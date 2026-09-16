# E9-00 (Fable) — Estructura de agregación del objetivo y anclaje por grupos de escalones

Script `e9_00_agregacion_fable.py`; salidas `e9_00_telescopica.csv`, `e9_00_anclaje_grupos.csv`, `e9_00_varianza_J.csv`,
`e9_00_consistencia_J_kpi.csv`. Datos v6 (`df_v6.pkl`), KPI = `recuperacion_refinada_pct`, controles E7-01 (`CONTROLES_BATCH`).

## 1. Identidad telescópica (escalón → fase) — exacta

Con los 4 escalones de Reducción válidos (254 batches): Σ_t `m6_ln_sn_dep` − ln((Sn_ini + Sn alimentado)/Sn_fin) = +0.005 ± 0.018
(el Sn alimentado en Reducción, mediana 476 kg, entra escalón a escalón y no al inicio, de ahí el residuo); Σ_t `m6_ln_feo_ret` −
ln(FeO_fin/FeO_ini) = 0.0000 ± 0.0000. Con escalones inválidos (Sn_inv < 20 kg) la suma pierde el tramo final; la agregación por
fase debe hacerse siempre con el cociente de inventarios (ini/fin), y la suma por escalón sólo como descomposición.

## 2. Anclaje por grupo de escalones (¿un único K y w por fase?)

OLS HC3 del KPI (pp) sobre las sumas por grupo, DEV n = 298 (total n = 361 entre paréntesis):

| componente | R0 | R1 | R2-R3 | Wald homogeneidad p |
|---|---|---|---|---|
| ln_sn_dep (pp/unidad) | 2.97 [1.1, 4.8] (3.33) | 4.21 [2.4, 6.1] (3.99) | 1.44 [−0.3, 3.1] (2.16) | 0.025 (0.18) |
| ln_feo_ret (pp/unidad) | 24.6 [13.5, 35.7] (23.2) | 24.1 [14.8, 33.3] (20.9) | 11.9 [3.7, 20.1] (10.4) | 0.073 (0.06) |
| w = coef FeO / coef Sn | 8.3 | 5.7 | 8.3 | — |

Lectura: (a) el cociente w es estable entre grupos (5.7-8.3, en el IC de w = 6.09 [4.05, 9.99] de E8-03) → un único w por fase es
defendible; (b) los coeficientes del grupo tardío (R2-R3) son ~la mitad en ambas componentes. Como el KPI depende sólo del estado
terminal (identidad telescópica), la física no distingue de qué escalón viene un ln; la atenuación del grupo tardío es la firma del
error de medida en los targets tardíos (SNR del cierre físico 3.0 en R1 frente a 1.2 en R2-R3, E8-01): errores-en-variables sesgan
el coeficiente hacia 0 en proporción a var(ruido)/var(total). Conclusión: **K único por fase, w único; el objetivo aditivo por
escalón es exacto en la física y la heterogeneidad observada es atenuación estadística, no un peso pirometalúrgico distinto**. El
prescriptor usa predicciones (no targets ruidosos), así que no hereda la atenuación.

## 3. Qué grupo decide el batch (descomposición de la varianza de J_R con K = 2.893, w = 6.09)

| componente | media (pp) | fracción de var(J_R) |
|---|---|---|
| Sn R0 / R1 / R2-R3 | +4.92 / +1.68 / +1.75 | 0.18 / −0.01 / 0.20 |
| FeO R0 / R1 / R2-R3 | −2.71 / −1.20 / −1.87 | 0.12 / 0.23 / 0.27 |

FeO explica el 62 % de la varianza de J_R y Sn el 38 %; el grupo tardío (R2-R3) el 48 %, R0 el 30 %, R1 el 22 %. En media, R0 gana
+4.9 pp por agotamiento y pierde −2.7 pp por FeO (selectivo); R2-R3 gana +1.75 y pierde −1.87 (ya no selectivo): consistente con
la selectividad decreciente con el avance (E7-03) y con la política v5/v6 (carbón temprano, menos tarde).

## 4. Consistencia J_batch (targets reales) ↔ KPI

Spearman: J_R 0.319 DEV (p < 1e-8) / 0.198 lockbox (p 0.12) / 0.298 total; J_F ≈ 0 (0.04 / 0.14 / 0.09); J_batch 0.31 / 0.24 / 0.30.
En lockbox el agotamiento de Sn domina (0.38, p 0.002) y la retención de FeO no correlaciona (−0.14): fin de campaña (horno frío)
premia agotar Sn; el w anclado en DEV no es el óptimo del lockbox (ya visto en E8-03). Fusión: Σ_F ln_sn_dep −0.46 pp/unidad (p 0.29
DEV; −0.71 p 0.05 total con heel); por grupos F1-F3 −0.92 (p 0.23) y F4-F6 +0.31 (p 0.65): si hay conservación que valga, es en la
primera mitad de la Fusión; nada sostenible aún.

## Implicación para v7

- Objetivo por escalón `J_t = K·(ln_sn_dep_t + w·ln_feo_ret_t) − costos_t − …`; por grupo `J_G = Σ_{t∈G} J_t`; por fase `J_R =
  K·[ln(Sn_ini/Sn_fin) + w·ln(FeO_fin/FeO_ini)] − Σ costos` (exacto); batch `J = J_R + J_F`, con J_F = −β_F·ln(Sn_disp_F/Sn_fin_F)
  (débil). La demostración de "sentido pirometalúrgico de la eficiencia" en 1 escalón, un grupo o el batch es la misma fórmula.
- El heel de matriz (`F_lnirf_F0`, +7.3 pp/unidad, p < 1e-10) sigue siendo el mayor determinante del KPI y no es palanca del batch
  actual: es el estado con el que se entrega la escoria al batch siguiente (candidato a término de "legado" de la Reducción, ver E9-04).

## 5. Término de legado (E9-00b, `e9_00b_legado_fable.py`, `e9_00b_legado.csv`)

La retención de FeO de este batch también sube el KPI del **siguiente** batch: KPI_next ~ Σ_R ln_feo_ret +10.4 pp/unidad
[5.0, 15.9] (p 2e-4, DEV; +11.4 total), Σ_R ln_sn_dep +0.7 n.s. Placebo temporal (KPI del batch ANTERIOR sobre las mismas sumas):
+2.1 [−4.9, 9.0] p 0.56 → no es confusión de campaña. Mediación: Σ_R ln_feo_ret → ln IRF al cierre de F0 del batch siguiente +0.50
[0.22, 0.79] (p 5e-4) → KPI_next +7.6 pp/unidad; controlando el IRF final de Reducción el coeficiente del FeO cae a 4.8 n.s. y el IRF
final toma 5.4 (p 0.04): el legado viaja por el estado terminal de la matriz (talón químico), no por el Sn.

Anclaje sobre KPI_actual + KPI_siguiente (OLS HC3, controles, bootstrap 1000 por batch):

| muestra | y | K_Sn (pp/unidad) | K_FeO | w = K_FeO/K_Sn [IC95 %] |
|---|---|---|---|---|
| DEV n 297 | KPI actual | 2.853 | 17.26 | 6.05 [4.11, 9.63] |
| DEV n 297 | KPI actual + siguiente | **3.522** | **27.68** | **7.86 [5.14, 13.04]** |
| total n 360 | KPI actual | 3.218 | 15.71 | 4.88 [3.32, 7.06] |
| total n 359 | KPI actual + siguiente | 3.823 | 26.94 | 7.05 [4.81, 10.42] |

Decisión para v7: el objetivo de Reducción contabiliza el legado: `J_R = K·(ln_sn_dep + w·ln_feo_ret)` con K = 3.52 pp (dos batches)
y w = 7.86 (DEV). La estructura aditiva/telescópica no cambia (el legado depende sólo del estado terminal). Sensibilidad: con w = 6.09
(v6) las recomendaciones cambian poco de dirección (mismo signo), sólo de intensidad; se reporta como variante.

## 6. Anclaje de la conservación de Sn en Fusión como nivel terminal (E9-00c, `e9_00c_anclaje_fusion_fable.py`)

E9-04 encontró `m6_sn_inv_kg_prev` al cierre de F6 con +8.7 pp/sd (p 0.038) en una OLS con 12 variables de estado; aislado y en
log, el nivel terminal es débil: KPI ~ ln(Sn conservado/Sn alimentado en F) +0.83 pp/unidad (p 0.33 DEV), +1.22 con heel (p 0.14),
+1.70 total (p 0.013); ln %Sn al cierre de F6 +1.65 (p 0.10 DEV) / +2.28 (p 0.007 total). El anclaje por suma de escalón (v6)
da −0.58 (p 0.17 DEV) / −0.91 (p 0.013 total). Canales (DEV): conservar Sn baja el dross (−0.019/unidad, p 0.005) y sube el Sn en
escoria final (+0.005, p 5e-4); el neto es positivo pero pequeño. Pesos de dilución de cada escalón sobre el nivel terminal
(∂ln Sn_F6/∂ln_sn_dep_t): F1 0.02, F2 0.06, F3 0.14, F4 0.26, F5 0.49, F6 1.00 — el nivel terminal lo deciden F5-F6.

Decisión v7: se mantiene la forma de v6 (`J_F = −β_F·ln_sn_dep_t`, homogénea por escalón, β_F = 0.58 pp/unidad, evidencia débil
en DEV y significativa en total) y se documenta la variante de nivel terminal con pesos de dilución (β_F·w_t, β_F 1.22) como
equivalente en fuerza. Ninguna de las dos justifica por sí sola mover carbón/gas en Fusión: la recomendación de Fusión sigue
apoyada en costos, ventana térmica y la evidencia off-policy del canal polvo (E7-05/E8-07).
