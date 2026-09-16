# E9-03: politica con horizonte en Reduccion y agregacion escalon -> grupo -> batch

Hipotesis H3 (ITERACION_9_diseno.md): la secuencia de acciones dentro de Reduccion importa porque la
selectividad del carbon (FeO metalizado por Sn reducido) crece con el avance; una politica con horizonte
deberia batir a la miope v6 moviendo carbon hacia R0-R1. Script: `experimentos/v7/e9_03_horizonte.py`.
Parametros (reducidos por costo computacional, documentado): miope `n_candidatos=100`,
`n_refinamiento=25`; horizonte `n_secuencias=80` en **lazo abierto** (se planifica
una sola vez desde R0 mediante busqueda coordinada vectorizada -- perturbaciones gaussianas de la miope y
de la historica -- y NO se re-planifica tipo MPC escalon a escalon; ver docstring de `politica_horizonte`).
Cross-fitting: 5 folds de batches DEV (sistema entrenado sin el fold) + lockbox (sistema entrenado en DEV).

## 1) Validacion del simulador (encadenamiento)

Sistema entrenado en todo DEV; se aplican las acciones HISTORICAS de R0..R3 desde el estado real de R0 y
se compara la trayectoria simulada con la real, por `orden_escalon_fase` (0=primer escalon, sin
acumulacion de error; 3=cuarto, con 4 predicciones encadenadas). 294/299 batches DEV usables (5
descartados: escalones incompletos o `tasa_gn_nm3_min=0`, que indefine `exceso_o2_combustion_pct`).

| variable | orden_escalon_fase | n | r2 | mae | sesgo | sesgo_pct |
|---|---|---|---|---|---|---|
| Sn_inv_kg | 0 | 293 | 0.687 | 482.608 | -171.800 | -7.041 |
| Sn_inv_kg | 1 | 291 | 0.319 | 237.984 | -35.725 | -2.812 |
| Sn_inv_kg | 2 | 290 | 0.217 | 236.136 | 72.752 | 8.179 |
| Sn_inv_kg | 3 | 199 | 0.116 | 238.579 | 46.712 | 6.006 |
| FeO_inv_kg | 0 | 293 | 0.893 | 403.232 | 1.950 | 0.017 |
| FeO_inv_kg | 1 | 291 | 0.831 | 501.870 | -157.497 | -1.436 |
| FeO_inv_kg | 2 | 290 | 0.729 | 619.433 | -218.224 | -2.123 |
| FeO_inv_kg | 3 | 199 | 0.660 | 755.738 | -463.962 | -4.664 |
| T_celsius | 0 | 294 | 0.973 | 7.893 | -0.122 | -0.011 |
| T_celsius | 1 | 294 | 0.949 | 11.934 | 1.388 | 0.131 |
| T_celsius | 2 | 294 | 0.910 | 16.891 | 3.889 | 0.370 |
| T_celsius | 3 | 294 | 0.898 | 18.057 | 4.924 | 0.468 |
| masa_kg | 0 | 293 | 0.911 | 982.438 | -331.163 | -0.542 |
| masa_kg | 1 | 291 | 0.949 | 773.824 | -322.958 | -0.553 |
| masa_kg | 2 | 290 | 0.932 | 861.262 | -261.977 | -0.462 |
| masa_kg | 3 | 199 | 0.928 | 962.162 | -546.406 | -0.977 |

**Degradacion por acumulacion de error**: R2 de Sn_inv_kg cae de 0.69 (orden 0) a 0.12 (orden 3) -- el
target `ln_sn_dep` es el mas dificil de encadenar porque el error de cada escalon se compone
multiplicativamente sobre un inventario que decrece. FeO_inv_kg degrada menos (0.89 -> 0.66) y T_celsius
(0.97 -> 0.90) y masa_kg (0.91 -> 0.93, estable) casi no degradan -- son variables mas "inerciales" o con
error aditivo en vez de multiplicativo. El sesgo no crece de forma monotona ni es enorme (< 8% del valor
medio en todos los casos), consistente con un simulador sin sesgo estructural grande pero con varianza
creciente. Ver `e9_03_validacion_simulador.csv` (fila a fila) y `e9_03_validacion_resumen.csv`.

## 2) J_batch simulado por politica (DEV cross-fitted, misma vara para las 3 politicas)

| conjunto | policy | n | J_batch_mediana | ci_lo | ci_hi | J_batch_media |
|---|---|---|---|---|---|---|
| DEV | historica | 294 | -2072.800 | -2160.500 | -1962.000 | -2060.700 |
| DEV | miope | 294 | -1967.300 | -2050.800 | -1862.300 | -1953.800 |
| DEV | horizonte | 294 | -1903.500 | -2021.100 | -1824.000 | -1911.400 |
| lockbox | historica | 63 | -1948.700 | -2079.700 | -1754.200 | -1929.500 |
| lockbox | miope | 63 | -1780.900 | -1873.900 | -1693.900 | -1735.700 |
| lockbox | horizonte | 63 | -1781.800 | -1862.600 | -1706.400 | -1740.500 |

Mediana del % de mejora de horizonte sobre miope en J_batch (DEV, cross-fitted, IC bootstrap 95% de la
mediana): **0.0% [0.0, 0.0]** (media 2.5%); lockbox:
**0.0% [0.0, 0.0]**. Distribucion por batch (DEV): horizonte mejora
J_batch en el 40% de los batches (uplift > 1 kg Sn-eq), empata (dentro de +-1 kg,
la busqueda no encontro nada mejor que el ancla miope) en el 59%, y empeora en el
1% (ruido de la busqueda aleatoria, uplift pequeno y negativo). El horizonte SI dobla
a la miope en la cola derecha (max +82% en un batch DEV) pero la mediana queda muy por debajo del umbral
de adopcion (10%) porque en la mayoria de los batches la busqueda de secuencias no logra separarse del
ancla miope.

Descomposicion temprano (R0-R1) / tardio (R2-R3) del uplift horizonte-miope (DEV, mediana kg Sn-eq):

| grupo | uplift_mediana_kg | ci_lo | ci_hi |
|---|---|---|---|
| temprano | 0.000 | 0.000 | 0.000 |
| tardio | 0.000 | 0.000 | 0.000 |

Ver `e9_03_politicas_batch.csv` (J_batch, J_temprano, J_tardio, uplifts por batch/politica) y
`e9_03_politicas_escalon.csv` (detalle por escalon).

## 3) Robustez

**Bootstrap de theta** (20 replicas, sistema DEV completo, batches lockbox n=63): fraccion de
batches/replica donde horizonte > miope:

| replica_theta | frac_horizonte_gana |
|---|---|
| 0 | 0.111 |
| 1 | 0.143 |
| 10 | 0.063 |
| 11 | 0.127 |
| 12 | 0.111 |
| 13 | 0.111 |
| 14 | 0.143 |
| 15 | 0.095 |
| 16 | 0.127 |
| 17 | 0.143 |
| 18 | 0.127 |
| 19 | 0.127 |
| 2 | 0.111 |
| 3 | 0.143 |
| 4 | 0.095 |
| 5 | 0.127 |
| 6 | 0.111 |
| 7 | 0.111 |
| 8 | 0.111 |
| 9 | 0.127 |

Fraccion global (todas las replicas x batches): **0.118** -- muy por debajo de 0.5: bajo
perturbaciones plausibles de theta (IC95% bootstrap de cada palanca), el horizonte deja de superar a la
miope en la gran mayoria de los casos, seal de que la ventaja observada en el punto 2 no es robusta.

**Escenario pesimista** (prediccion - 0.5*ancho IC en sn_dep/feo_ret, mismo sistema lockbox): horizonte >
miope en **0.095** de los batches (mediana J: horizonte -4926.9, miope -4925.4 kg Sn-eq) --
tampoco sobrevive. Ver `e9_03_robustez.csv`.

## 4) Agregacion escalon -> grupo -> batch

Batches con 4 escalones de Reduccion presentes: 362/362. De esos, 254 con los 4 targets validos (m6_valido=True). Error absoluto |suma_log - directo| (identidad telescopica sobre inventarios reales): ln_sn_dep mediana 4.81e-04 (max 2.22e-01, no exactamente 0 por el umbral de feed_Sn_kgf residual en algun escalon intermedio); ln_feo_ret mediana 5.6e-17 (exacta a precision de punto flotante, sin feed de FeO intermedio).

Correlacion (Spearman, IC bootstrap 95%) de J_batch (con los TARGETS REALES, no simulados) y de sus dos
mitades temporales con el KPI refinado:

| variable | conjunto | n | rho | p | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| J_batch_real | DEV | 298 | 0.319 | 0.000 | 0.213 | 0.416 |
| J_batch_real | lockbox | 63 | 0.198 | 0.119 | -0.051 | 0.441 |
| J_batch_real | total | 361 | 0.298 | 0.000 | 0.196 | 0.390 |
| J_temprano | DEV | 299 | 0.347 | 0.000 | 0.243 | 0.444 |
| J_temprano | lockbox | 63 | 0.307 | 0.014 | 0.064 | 0.527 |
| J_temprano | total | 362 | 0.329 | 0.000 | 0.238 | 0.415 |
| J_tardio | DEV | 299 | 0.089 | 0.126 | -0.017 | 0.199 |
| J_tardio | lockbox | 63 | -0.065 | 0.611 | -0.339 | 0.217 |
| J_tardio | total | 362 | 0.070 | 0.185 | -0.036 | 0.173 |

**El grupo temprano (R0-R1) explica mas del KPI que el tardio (R2-R3)**: rho(J_temprano, KPI) ~0.33-0.35
(DEV y lockbox, p<0.05 en ambos) frente a rho(J_tardio, KPI) ~0.07-0.09 (DEV, no significativo) y NEGATIVO
en lockbox (no significativo, n=63). Esto es consistente con la teoria de selectividad (H3): lo que pasa
en R0-R1, cuando el Sn todavia esta abundante y la reduccion es mas selectiva, pesa mas sobre el resultado
del batch que lo que pasa en R2-R3 -- pero esta asimetria NO se traduce en una ganancia de horizonte
robusta en el punto 2/3 (la miope ya captura la mayor parte del valor disponible escalon a escalon, y el
margen de re-secuenciar dentro de un horizonte de solo 4 pasos fijos es pequeno frente al ruido).

## 5) Perfil de accion por politica y orden (adelanta el carbon el horizonte?)

| orden_escalon_fase | policy | n | tasa_feed_Carbon_kg_min | tasa_gn_nm3_min | tasa_o2_nm3_min | tasa_aire_nm3_min |
|---|---|---|---|---|---|---|
| 0 | historica | 357 | 61.120 | 27.980 | 28.430 | 119.750 |
| 0 | horizonte | 357 | 61.380 | 27.460 | 27.800 | 121.050 |
| 0 | miope | 357 | 62.790 | 27.500 | 28.040 | 120.910 |
| 1 | historica | 357 | 31.450 | 24.410 | 23.180 | 118.550 |
| 1 | horizonte | 357 | 25.470 | 24.100 | 22.020 | 121.000 |
| 1 | miope | 357 | 25.000 | 24.060 | 22.070 | 121.360 |
| 2 | historica | 357 | 12.080 | 18.810 | 12.130 | 119.750 |
| 2 | horizonte | 357 | 8.220 | 18.160 | 11.230 | 123.560 |
| 2 | miope | 357 | 7.810 | 18.010 | 10.950 | 123.870 |
| 3 | historica | 357 | 6.930 | 16.680 | 7.740 | 119.770 |
| 3 | horizonte | 357 | 6.050 | 16.240 | 7.240 | 121.530 |
| 3 | miope | 357 | 6.320 | 16.050 | 7.230 | 122.120 |

Carbon medio (kg/min): horizonte R0-R1 43.43 vs R2-R3 7.13; miope R0-R1
43.90 vs R2-R3 7.06; historica R0-R1 46.29 vs R2-R3
9.51. **No hay una senal clara de "adelantar" carbon**: horizonte y miope mueven el
carbon en la MISMA direccion que la historica (menos que la historica en R1-R2, practicamente igual en R0
y R3) y la diferencia horizonte-miope es marginal (R0 +0.26, R1 +0.47, R2 +0.41, R3 -0.27 kg/min); el
horizonte no concentra mas carbon en los primeros escalones de forma sistematica. GN medio: horizonte
21.491, miope 21.407, historica 21.970 nm3/min -- diferencias de <0.1 nm3/min, sin patron.
La optimizacion (miope u horizonte) esta dominada por REDUCIR el carbon total frente a la historica
(sobre todo en R1-R2), no por resecuenciarlo.

## Veredicto

Criterio de diseno (ITERACION_9_diseno.md, punto 4): se adopta la politica con horizonte solo si mejora
J_batch simulado >= 10% sobre la miope en DEV cross-fitted (mediana por batch, IC bootstrap) Y la mejora
sobrevive a perturbaciones de theta y al modelo pesimista.

Resultado: mediana DEV 0.0% (IC [0.0, 0.0]) < 10%;
robustez a theta 0.118 < 0.5; pesimista 0.095
< 0.5.

**NO SE ADOPTA: la miope v6 queda como politica vigente, con justificacion.** La busqueda de horizonte (lazo abierto, perturbaciones gaussianas) encuentra mejoras
puntuales grandes en un ~40% de los batches DEV, pero en la mayoria (~58%) no logra separarse de la
politica miope, y la ventaja mediana no sobrevive ni al bootstrap de theta ni al escenario pesimista. La
asimetria real entre temprano/tardio (punto 4) confirma que la SECUENCIA importa para el KPI, pero la
miope v6 (que ya condiciona cada decision en el estado simulado que hereda de las decisiones anteriores)
capta la mayor parte de ese valor; el margen adicional de planificar explicitamente el horizonte de 4
pasos fijos de Reduccion es pequeno y fragil frente al ruido de los targets. La miope v6 queda como
politica vigente. Lineas para revisitar el horizonte: (a) MPC real (re-planificar en cada escalon en vez
de lazo abierto), (b) mas secuencias / busqueda dirigida por gradiente en vez de perturbacion aleatoria,
(c) un horizonte que tambien cubra Fusion (donde H3 anticipa mayor palanca por selectividad temprana).
