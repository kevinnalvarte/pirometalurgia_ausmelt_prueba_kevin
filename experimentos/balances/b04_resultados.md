# B-04 — Balance global de masa por batch, KPI refinado con Sn en escoria y modelo multi-oxido de la escoria

Script: `b04_balance_global.py` (semilla 42). Log completo: `b04_log.txt`. Datos: `bal.cargar_df()` (3982 escalones,
362 batches) via `bal.construir_batch_balance(df)`. DEV = 299 batches no lockbox, lockbox = 63 batches (17.5% cronologico final).
CSVs de soporte listados en cada seccion.

## 0. Verificacion de reproducibilidad

Las medianas del balance global reproducen exactamente el smoke test declarado en el enunciado:
`masa_escoria_balance_kg` 57917 kg (IQR 53539-62583, esperado 57.9 t IQR 53.5-62.6), `masa_escoria_trazador_kg`
53495 kg (esperado 53.5 t), `factor_escala_trazador` 1.083 (esperado 1.09), `sn_escoria_final_balance_kg` 689 kg
(esperado 690 kg), `cierre_sn_frac` mediana -1.3%, sd 6.6% (exacto), `recuperacion_refinada_pct` 68.40 vs proxy
69.59 (esperado 68.4 vs 69.6). `bal_features.py` no fue modificado.

---

## 1. Balance global de masa

### 1.1 Distribuciones (n=362 batches)

| columna | mediana | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|
| carga_solida_kg | 141 754 | 124 860 | 137 852 | 144 663 | 148 369 |
| feed_sn_kg | 51 247 | 45 126 | 49 655 | 53 128 | 55 388 |
| carbon_kg | 16 521 | 15 405 | 16 071 | 17 162 | 18 254 |
| cal_kg | 12 298 | 10 230 | 11 659 | 12 835 | 13 538 |
| dross_fe_in_kg | 19 614 | 16 355 | 18 435 | 22 702 | 25 602 |
| mineral_fe_kg | 14 422 | 7 | 12 405 | 16 892 | 21 552 |
| pellets_kg | 30 203 | 23 409 | 25 491 | 32 946 | 37 149 |
| sn_metal_kg (M) | 35 458 | 30 471 | 33 350 | 37 398 | 40 471 |
| sn_dross_kg (D) | 6 302 | 3 426 | 5 007 | 7 892 | 10 022 |
| sn_polvo_kg (P) | 9 049 | 6 550 | 7 942 | 10 362 | 13 067 |
| **masa_escoria_balance_kg** | **57 917** | 46 339 | 53 539 | 62 583 | 67 728 |
| masa_escoria_trazador_kg (v3) | 53 495 | 44 167 | 51 567 | 56 020 | 60 681 |
| factor_escala_trazador | 1.083 | 0.852 | 1.005 | 1.149 | 1.273 |
| ley_sn_escoria_final_pct | 1.189 | 0.639 | 0.949 | 1.513 | 2.219 |
| ley_feo_escoria_final_pct | 17.653 | 15.322 | 16.414 | 18.682 | 20.221 |
| **sn_escoria_final_balance_kg** | **689** | 356 | 536 | 885 | 1 381 |
| sn_escoria_final_trazador_kg | 638 | 274 | 505 | 812 | 1 251 |
| cierre_sn_kg | -646 | -6 433 | -3 039 | 1 267 | 3 761 |
| cierre_sn_frac | -1.3% | -13.3% | -5.6% | 2.5% | 7.8% |
| rendimiento_proxy_batch (%) | 69.59 | 61.29 | 66.71 | 72.34 | 76.92 |
| recuperacion_real_pct (%) | 69.60 | 60.80 | 66.57 | 72.66 | 77.56 |
| **recuperacion_refinada_pct (%)** | **68.40** | 60.21 | 65.71 | 71.31 | 75.58 |
| sn_perdido_escoria_frac | 1.34% | 0.69% | 1.03% | 1.73% | 2.77% |
| duracion_reduccion_min | 86 | 85 | 85 | 90 | 98 |

Tabla completa: `b04_distribuciones.csv`; tabla de batch completa: `b04_batch_balance.csv` (362 x 122).

### 1.2 Factor de escala del trazador: es el artefacto contable de la cal, no (solo) deriva de campana

| variable | rho Spearman | IC95% | OLS coef | p | R2 |
|---|---:|---:|---:|---:|---:|
| cal_kg / carga_solida_kg | **-0.354** | [-0.44, -0.26] | -7.23 | <1e-4 | 0.193 |
| frac_carga_secundaria_F | -0.010 | [-0.12, 0.11] | 0.23 | 0.029 | 0.018 |
| pellets_kg | 0.051 | [-0.05, 0.16] | ~0 | <1e-3 | 0.033 |
| idx_cronologico | **0.191** | [0.09, 0.29] | 0.0002/batch | 0.001 | 0.025 |

El factor depende fuertemente de la fraccion de cal en la carga solida (rho=-0.35, R2=0.19): a mayor proporcion de
cal, MENOR el factor de escala (masa_balance/masa_trazador se acerca mas a 1, o incluso cae bajo 1). Esto es
exactamente el artefacto contable descrito en el diseno (exp 18): el trazador v3 asume que TODO el CaO de la
escoria viene de la tolva 6 a una pureza fija; cuando la cal pesa mas en la carga, el trazador (que solo ve % CaO
en la escoria) sobreestima relativamente la masa total porque atribuye a la cal el CaO que en realidad proviene
tambien de la ganga de otras corrientes, y el error de esa atribucion se diluye. `frac_carga_secundaria_F` y
`pellets_kg` no tienen relacion practica (rho<0.06; el R2 marginal de OLS es ruido de un coeficiente ~0 con n=350).
Si existe ademas una deriva cronologica leve y significativa (rho=0.19, mas debil que el efecto de la cal):
el factor tiende a subir ligeramente batch tras batch, compatible con una acumulacion gradual de talon o con un
cambio lento en la composicion de la ganga, pero es un efecto secundario frente al de la cal.

**Implicacion:** un factor de escala unico (mediana 1.08) NO es una correccion multiplicativa estable porque esta
confundido con la mezcla de carga de cada batch; aplicarlo igual a todos los batches sesgaria sistematicamente
los que tienen mas o menos cal que el tipico. El balance global (`masa_escoria_balance_kg`), que ya parte de las
tolvas y no de un solo trazador, evita este confusor de raiz.

### 1.3 Cierre de Sn (`cierre_sn_frac`)

Media = -1.94%, sd = 6.60% (n=362), t=-5.60, p=4.2e-8 (rechaza H0: media=0). El balance global tiene un sesgo
pequeno pero estadisticamente significativo: en promedio "falta" ~2% del Sn cargado respecto a metal+dross+polvo+
escoria estimada (compatible con perdidas no modeladas: volatilizacion fina, salpicadura fuera de las corrientes
contempladas, o un ligero sesgo conjunto de los supuestos de leyes). La dispersion (sd 6.6%) es mucho mayor que
el sesgo medio, y **no depende de la ley del concentrado** (rho=0.048, p=0.36) **ni del dross de Fe cargado**
(rho=0.015, p=0.77): el cierre no delata un canal fisico especifico, es ruido de balance repartido de forma
pareja entre batches (tablas: `b04_cierre_sn.csv`, `b04_cierre_sn_ttest.csv`).

### 1.4 Tornado de sensibilidad a los supuestos

Rango (max-min de la mediana sobre los 362 batches) al barrer cada parametro en su grilla, con los demas en su
valor base:

| parametro | rango masa_escoria (kg) | rango Sn escoria final (kg) | rango recuperacion_refinada (pp) |
|---|---:|---:|---:|
| HUMEDAD_CARGA {0.03,0.06,0.09} | **7 873** | **87** | 0.15 |
| LEY_SN_POLVO {0.45,0.55,0.65} | **6 280** | **73** | 0.14 |
| CENIZA_CARBON {0.05,0.10,0.15} | 1 691 | 18 | 0.03 |
| FE_MET_DROSS {0.2,0.3,0.4} | 1 657 | 14 | 0.03 |
| LEY_SN_METAL {0.94,0.96,0.98} | 1 640 | 17 | 0.03 |
| LEY_SN_DROSS {0.55,0.65,0.75} | 1 271 | 14 | 0.02 |

La masa absoluta de escoria es mas sensible a la humedad de carga y a la ley de Sn del polvo (7-8 t de rango
sobre un valor tipico de 58 t, ~13%) que a las leyes de metal/dross/Fe metalico del dross/ceniza del carbon
(1.3-1.7 t, ~2-3%). **Pero la recuperacion refinada en si (la metrica que se usaria para comparar batches) es
casi insensible a todos estos supuestos: el rango maximo es 0.15 puntos porcentuales**, dos ordenes de magnitud
menor que la dispersion batch a batch (sd=4.5 pp). Esto es porque M (numerador) no depende de ningun supuesto de
`SUPUESTOS`, y la correccion de escoria es una fraccion pequena (~1.3%) del denominador: variar sus supuestos
mueve la correccion, pero apenas mueve el cociente final. Tablas: `b04_tornado.csv`, `b04_tornado_rango.csv`.

---

## 2. KPI refinado

### 2.1 Correlaciones entre KPIs de batch

| par | rho | r |
|---|---:|---:|
| recuperacion_refinada_pct vs rendimiento_proxy_batch | 0.993 | 0.994 |
| recuperacion_refinada_pct vs f_metal | 0.993 | 0.994 |
| recuperacion_refinada_pct vs recuperacion_real_pct | 0.575 | 0.588 |
| rendimiento_proxy_batch vs recuperacion_real_pct | 0.552 | 0.566 |
| rendimiento_proxy_batch vs f_metal | 1.000 | 1.000 |

sd: refinada 4.50 pp, proxy 4.58 pp, real 5.19 pp (medias 68.3/69.3/69.5). `rendimiento_proxy_batch` y `f_metal`
son la misma cantidad en otra escala (tautologico). `recuperacion_real_pct` sigue siendo una senal distinta y mas
ruidosa (rho~0.55-0.57 con las otras dos), consistente con la baja fiabilidad ya documentada en ST-08
(r=0.57 con el proxy).

### 2.2 Sn en escoria final: magnitud y consistencia con el trazador

`sn_perdido_escoria_frac` (Sn en escoria final / Sn de salida total): mediana 1.34% (IQR 1.03-1.73%, P5-P95
0.69-2.77%). Es una correccion modesta y consistente con la diferencia mediana observada entre proxy (69.59) y
refinada (68.40): ~1.2 pp.

`sn_escoria_final_balance_kg` vs `sn_escoria_final_trazador_kg` (v3, CaO solo): n=362, r=0.912, rho=0.888,
razon balance/trazador mediana 1.087 (IQR 1.01-1.16) — el balance global estima consistentemente ~9% mas Sn en
escoria que el trazador CaO puro, coherente con que el trazador v3 subestima la masa absoluta de escoria (seccion
1.2).

### 2.3 Que explica el Sn en escoria final (kg, balance)

| variable | rho | IC95% |
|---|---:|---:|
| ley_sn_escoria_final_pct | 0.942 | [0.93, 0.95] |
| st_R23_lnirf_R | 0.214 | [0.12, 0.31] |
| duracion_reduccion_min | 0.107 | [0.00, 0.21] |
| st_R22_sdi | 0.047 (n.s.) | [-0.04, 0.16] |

Domina, como es de esperar, la ley final de Sn en escoria (r casi tautologico dado que Sn_esc = masa x ley/100 y
la ley varia mucho mas que la masa: sd relativa 56% vs 13%). El indice lnIRF de Reduccion (matriz mas oxidante =
mas FeO relativo a los otros tres inertes durante toda la fase) tiene una asociacion debil pero significativa con
mas Sn residual; la duracion de Reduccion, marginal. El SDI (indice de agotamiento SELECTIVO, una cantidad
relativa al avance) no predice el residuo ABSOLUTO de Sn — tiene sentido, ya que el SDI mide la forma de la
trayectoria de agotamiento, no el tamano del batch ni su punto de partida.

### 2.4 Cambio de ranking proxy -> refinado

Spearman(proxy, refinado) = **0.993** (p<1e-300, n=362): son casi el mismo ranking. Sin embargo, **34 de 362
batches (9.4%) cambian de cuartil** al pasar de proxy a refinado. Es decir: para el analisis agregado (tendencias,
promedios) la correccion es irrelevante; para decisiones a nivel de batch individual (por ejemplo, identificar los
peores 25% para revision) el 1 de cada 10 batches que cambia de cuartil si importa. Tabla: `b04_ranking_cambio.csv`.

---

## 3. Targets ganadores (ST-01) vs KPI refinado

### 3.1 Spearman en DEV, BH dentro de KPI (n=10 targets)

vs `recuperacion_refinada_pct` (DEV):

| target | rho_dev | IC95% | p_bh | sig BH | rho_lockbox | p_lockbox |
|---|---:|---:|---:|:--:|---:|---:|
| st_R22_sdi | 0.317 | [0.21, 0.43] | <1e-4 | si | 0.087 | 0.50 |
| st_R23_lnirf_R | 0.297 | [0.18, 0.41] | <1e-4 | si | **0.378** | 0.002 |
| st_R21_ln_feo_ret | 0.286 | [0.17, 0.39] | <1e-4 | si | -0.088 | 0.49 |
| st_R02_feo_ret | 0.256 | [0.15, 0.35] | <1e-4 | si | -0.162 | 0.20 |
| b6_sdi_multi_batch | 0.230 | [0.12, 0.33] | 1e-4 | si | -0.058 | 0.65 |
| st_F10_irf_next | 0.207 | [0.10, 0.31] | 5e-4 | si | **0.341** | 0.006 |
| st_F05_ley_feo_next | 0.144 | [0.03, 0.25] | 0.018 | si | 0.195 | 0.13 |
| b6_feo_extraido_multi_neg | 0.130 | [0.02, 0.23] | 0.030 | si | -0.232 | 0.07 |
| st_F01_sn_ext | 0.039 | [-0.08, 0.17] | 0.56 | no | 0.180 | 0.16 |
| st_F16_sn_ret_frac | 0.027 | [-0.09, 0.14] | 0.64 | no | 0.165 | 0.20 |

**Hallazgo relevante:** con el KPI refinado, el *ranking* de targets en DEV es casi identico al de ST-01 con el
proxy (SDI primero, luego lnIRF_R, ln_feo_ret, feo_ret...), pero la **replicacion en lockbox cambia de forma
notable**: `st_R22_sdi`, el ganador teorico de Reduccion contra el proxy, **pierde significancia en lockbox contra
el refinado** (rho 0.087, p=0.50), mientras que **los dos targets basados en IRF (`st_R23_lnirf_R`, `st_F10_irf_next`)
replican con mas fuerza en lockbox que en DEV** (0.378 y 0.341, ambos p<0.01). Esto no estaba visible al mirar solo
el proxy y sugiere que, si el objetivo de negocio es minimizar Sn perdido en escoria (no solo maximizar metal),
la matriz de la escoria (IRF) es una senal mas robusta que la selectividad de agotamiento (SDI) fuera de muestra.

vs `sn_perdido_escoria_frac` (DEV; recordar: direccion "menor=mejor", por lo que rho>0 aqui = MAS FeO retenido /
mas Sn extraido en Fusion -> MAS perdida de Sn en escoria final, contraintuitivo con la teoria de que esas
acciones deberian ayudar; se reporta el signo crudo, ver interpretacion):

| target | rho_dev | IC95% |
|---|---:|---:|
| st_R21_ln_feo_ret | 0.337 | [0.23, 0.44] |
| st_R02_feo_ret | 0.266 | [0.17, 0.38] |
| st_F16_sn_ret_frac | 0.261 | [0.16, 0.37] |
| st_F01_sn_ext | 0.211 | [0.09, 0.32] |
| st_R23_lnirf_R | 0.188 | [0.08, 0.29] |
| b6_feo_extraido_multi_neg | 0.197 | [0.08, 0.31] |
| st_F05_ley_feo_next | -0.119 | [-0.24, 0.00] |
| st_F10_irf_next | -0.010 (n.s.) | |
| b6_sdi_multi_batch | 0.071 (n.s.) | |
| st_R22_sdi | 0.069 (n.s.) | |

Retener mas FeO (`R21_ln_feo_ret`, `R02_feo_ret`) esta asociado a **mas** Sn perdido en escoria final, no menos.
Esto es consistente con la teoria si se interpreta como una senal de escoria "dificil": los batches donde el FeO
no se metaliza tampoco liberan bien el Sn (misma matriz viscosa/fayalitica que retiene ambos), en vez de que el
FeO retenido "proteja" activamente al Sn de perderse. `F16_sn_ret_frac` (Sn retenido en escoria al fin de Fusion)
tambien predice mas Sn perdido al final: lo que no se extrae en Fusion tampoco termina saliendo en Reduccion.
Tabla completa (120 filas, 10 targets x 4 KPIs x 3 muestras): `b04_targets_vs_kpi.csv`.

### 3.2 Es el KPI refinado mas o menos explicable que el proxy

| target | rho_dev (refinado) | rho_dev (proxy) | delta |
|---|---:|---:|---:|
| st_R22_sdi | 0.317 | 0.319 | -0.002 |
| st_R23_lnirf_R | 0.297 | 0.304 | -0.007 |
| st_F10_irf_next | 0.207 | 0.195 | +0.012 |
| st_F01_sn_ext | 0.039 | 0.063 | -0.024 |
| st_R21_ln_feo_ret | 0.286 | 0.315 | -0.030 |
| st_R02_feo_ret | 0.256 | 0.274 | -0.018 |
| st_F16_sn_ret_frac | 0.027 | 0.063 | -0.035 |
| st_F05_ley_feo_next | 0.144 | 0.121 | +0.024 |
| b6_sdi_multi_batch | 0.230 | 0.234 | -0.005 |
| b6_feo_extraido_multi_neg | 0.130 | 0.151 | -0.021 |

Media(delta) = -0.011. **El KPI refinado no es ni claramente mas ni claramente menos explicable que el proxy**:
todas las diferencias son pequenas (|delta|<=0.035), dentro del ruido de los IC (~0.1 de ancho), y el orden de
los targets es identico. La correccion por Sn en escoria no cambia la conclusion de la iteracion 5 sobre cuales
son los mejores targets teoricos; solo (seccion 3.1) cambia cual de ellos replica mejor en lockbox.
OLS HC3 con controles + tendencia: `b04_targets_ols.csv`; quintiles DEV: `b04_targets_quintiles.csv` (SDI, lnIRF_R,
F10_irf_next y R02_feo_ret tienen monotonia 4/4 o 3/4 en sus quintiles contra recuperacion_refinada_pct).

---

## 4. Techo de predictibilidad del KPI refinado

HGB (max_depth=3, 200 iter, min_leaf=10) y RidgeCV sobre 44 features (39 `st_*` + 5 controles;
`factor_escala_trazador` excluido por usar salidas del propio balance del batch). KFold 5x3 repeticiones, DEV;
lockbox = entrenar en DEV completo, predecir lockbox.

| KPI | modelo | R2 OOF (DEV) | rho OOF (DEV) | R2 lockbox | rho lockbox |
|---|---|---:|---:|---:|---:|
| recuperacion_refinada_pct | HGB | 0.087 (+-0.032) | 0.355 | 0.029 | 0.227 |
| recuperacion_refinada_pct | Ridge | **0.158** (+-0.013) | 0.405 | **0.089** | 0.248 |
| rendimiento_proxy_batch (mismo pipeline) | HGB | 0.103 | 0.374 | -0.024 | 0.297 |
| rendimiento_proxy_batch (mismo pipeline) | Ridge | 0.152 | 0.403 | 0.002 | 0.222 |
| recuperacion_real_pct | HGB | 0.098 | 0.397 | -0.018 | 0.422 |
| recuperacion_real_pct | Ridge | 0.145 | 0.392 | -0.128 | 0.479 |
| sn_perdido_escoria_frac | HGB | 0.791 | 0.888 | 0.533 | 0.857 |
| sn_perdido_escoria_frac | Ridge | 0.726 | 0.869 | 0.946 | 0.886 |

Referencia externa ST-15 (proxy, HGB, pipeline con 46 features incl. heel): R2 OOF 0.148 / rho 0.43 DEV; lockbox
R2 0.122 / rho 0.48.

**Comparacion honesta (mismo pipeline, misma corrida):** el techo del KPI refinado (Ridge: R2 OOF 0.158, lockbox
R2 0.089, rho lockbox 0.248) es muy similar al del proxy calculado aqui (Ridge: R2 OOF 0.152, lockbox R2 0.002,
rho lockbox 0.222) — el refinado no pierde predictibilidad, y en lockbox incluso se comporta un poco mejor en
R2 (0.089 vs 0.002). `recuperacion_real_pct` confirma su fiabilidad mas baja (R2 lockbox negativo con ambos
modelos, pese a rho lockbox positivo: la ordenacion se conserva pero la calibracion no).

**Alerta de fuga parcial en `sn_perdido_escoria_frac`:** su R2 (0.73-0.79 DEV, hasta 0.95 en lockbox) es
sospechosamente alto frente a los demas KPIs. La razon: `sn_perdido_escoria_frac` se construye con
`ley_sn_escoria_final_pct`, la ley del ULTIMO escalon ensayado de Reduccion; 4 de los 39 `st_*` (`R13_sn_residual`,
`R14_log_sn_feo_next`, `F05_ley_feo_next`, `F11_log_sn_feo_next`, todos con `agg='last'`) usan las MISMAS columnas
crudas (`ley_sn_escoria_pct`, `ley_feo_escoria_pct`) en ese mismo escalon terminal — son TARGET validos segun la
clasificacion anti-fuga de `st_targets.py` (conocidos solo al cierre de fase, igual que el propio KPI), pero
comparten literalmente el dato que construye el KPI. Repitiendo el techo SIN esas 4 columnas:

| KPI | modelo | R2 OOF (DEV) | R2 lockbox |
|---|---|---:|---:|
| recuperacion_refinada_pct | HGB | 0.079 | -0.093 |
| recuperacion_refinada_pct | Ridge | 0.150 | 0.080 |
| sn_perdido_escoria_frac | HGB | 0.538 | 0.314 |
| sn_perdido_escoria_frac | Ridge | 0.125 (+-0.155, inestable) | 0.416 |

`recuperacion_refinada_pct` apenas cambia (como se esperaba: no usa esas 4 columnas de forma directa). El techo de
`sn_perdido_escoria_frac` baja pero sigue siendo alto (R2 DEV 0.54 HGB) frente a los demas KPIs (~0.08-0.16):
hay una componente real, no solo el artefacto de comparticion de columna — probablemente via `st_R23_lnirf_R` y
otras features de matriz de escoria. **El numero de cabecera para `sn_perdido_escoria_frac` debe reportarse con
esta salvedad**: 0.79/0.73 (HGB/Ridge, DEV, con las 4 columnas terminales) sobreestima la predictibilidad
genuina; 0.54/0.13 (sin ellas) es una lectura mas honesta pero aun con senal real. Tablas: `b04_techo.csv`,
`b04_techo_sin_terminales.csv`.

---

## 5. Modelo multi-oxido de la escoria (talon + ganga por tolva)

### 5.1 Especificacion

`M_X[t] = H_X + sum_tolva a_(X,tolva) * cum_feed_tolva_incl[t]` para X en {CaO, SiO2, Al2O3, MgO}, ajustado con
`scipy.optimize.least_squares` (loss soft_l1, f_scale=0.3) sobre los log-cocientes observables
ln(%X[t]/%CaO[t]) = ln(M_X[t]/M_CaO[t]) para (SiO2, Al2O3, MgO) contra CaO, en los 3850/3982 escalones con ensayo
simultaneo de los 4 oxidos. Supuesto fisico impuesto: la cal (t6) solo aporta CaO (`a_cao_cal` en [0.7,1.0], resto
de sus coeficientes fijos en 0); el carbon (t_carbon, via su ceniza) solo aporta SiO2/Al2O3. Los demas 5 x 4 = 20
coeficientes quedan libres en [0,1] (kg oxido / kg tolva); H_X en [0,10000] kg. Total: 23 coeficientes libres + 4
talones = 27 parametros. Detalle: `b04_multioxido_coeficientes.csv`, `b04_multioxido_talon.csv`,
`b04_multioxido_estabilidad.csv`.

### 5.2 Ajuste y estabilidad

El ajuste completo converge (cost=146.9, 25 evaluaciones, 6.5 s). **Multi-start con 5 semillas distintas converge
exactamente al mismo punto en las 27 dimensiones (sd=0.0000 en todos los parametros)**: no es un problema de
optimizacion local, es el optimo global de este objetivo con estas cotas — y ese optimo es fisicamente
inadmisible:

| oxido | tolva | a (kg/kg) | en cota |
|---|---|---:|:--:|
| SiO2 | t1_conc, t3_reciclo, t5_dross, t7_pellets | **1.000** | **superior (4 de 5)** |
| SiO2 | t4_fe (mineral) | 0.572 | no |
| SiO2 | H_SiO2 | **10 000 kg** | **superior** |
| CaO | t6_cal (pureza) | **0.700** | **inferior** (se esperaba ~0.85-0.95) |
| CaO | t1_conc / t3_reciclo / t5_dross / t7_pellets | 0.47 / 0.67 / 0.66 / 0.39 | no |
| Al2O3 | t1_conc..t7_pellets | 0.23-0.44 | no |
| MgO | t1_conc..t7_pellets | 0.02-0.10 | no |
| SiO2, Al2O3 | carbon (ceniza) | 0.000 | inferior (ambos) |

**Interpretacion:** el ajuste "resuelve" el SiO2 poniendo 4 de 5 corrientes al 100% SiO2 (fisicamente absurdo:
implica que el concentrado, el reciclo, el dross de Fe y los pellets serian solidos de silice pura) y aun asi
necesita un talon adicional de 10 toneladas de SiO2 pura (el techo permitido) para explicar la razon SiO2/CaO
observada. Esto no es ruido de optimizacion sino evidencia de que **el modelo, tal como esta planteado, no tiene
suficiente informacion para separar el aporte de SiO2 por tolva**: las curvas de alimentacion acumulada de las
distintas tolvas dentro de un batch son demasiado colineales (todas crecen durante la Fusion y se aplanan en la
Reduccion), y con solo 3 ecuaciones de log-cociente por escalon el optimizador prefiere maximizar el coeficiente
de SiO2 en todas las corrientes en vez de distribuir el efecto de forma identificable. El hecho de que
`a_cao_t6_cal` se vaya a su cota INFERIOR (en vez de acercarse al ~90% de pureza esperado) sugiere ademas que el
CaO predicho por el modelo ya es insuficiente en escala para las razones SiO2/CaO observadas, agravando el
problema del lado del denominador.

### 5.3 Validacion GroupKFold(5) por batch

R2 de los log-cocientes predichos en hold-out (promedio de 5 folds), contra el modelo trivial (media del log-
cociente en el fold de entrenamiento):

| par | R2 modelo (hold-out) | R2 trivial | rho (hold-out) |
|---|---:|---:|---:|
| SiO2/CaO | **-0.045** | 0.000 | 0.02 |
| Al2O3/CaO | **-0.013** | 0.000 | 0.14 |
| MgO/CaO | +0.005 | 0.000 | 0.31 |

El modelo **no generaliza**: en los 3 pares el R2 fuera de muestra es <=0 (peor o igual que predecir la media de
entrenamiento) pese a que en el ajuste completo (dentro de muestra) el costo baja considerablemente. Solo
MgO/CaO conserva algo de orden (rho=0.31) sin llegar a mejorar el R2 de calibracion. Tabla completa por fold:
`b04_multioxido_cv.csv`.

### 5.4 Masa de escoria resultante: no es utilizable

Con los parametros del ajuste completo, la masa de escoria estimada por oxido y su promedio ponderado
(`b6_masa_escoria_multioxido_kg`, 3851/3982 escalones, csv `b04_masa_multioxido_escalon.csv`) da una masa final de
batch con **mediana 314 179 kg** — **5.4 veces** la mediana del balance global (57 917 kg) y 5.8 veces la del
trazador v3, con correlacion debil (rho=0.33 vs balance, 0.22 vs trazador). Es la consecuencia directa del talon
de SiO2 saturado en 10 t y de los coeficientes de SiO2 al 100%: la masa "explicada" por SiO2 infla el promedio
ponderado muy por encima de lo fisicamente razonable. Comparacion completa: `b04_multioxido_batch.csv`.

### 5.5 Talon

Talon de los 4 oxidos inertes (suma H_X): CaO 5524 kg + SiO2 10 000 kg (en cota) + Al2O3 3304 kg + MgO 457 kg =
**19 285 kg**, equivalente a 6.1% de la mediana de masa final del propio modelo multi-oxido (314 t) — un numero
sin sentido pirometalurgico dado que la mitad de ese talon (SiO2) es un artefacto de saturacion, no una medicion.
La referencia del modelo Rust (~15 t de talon fisico total a 600 mm de espesor y 2.32 t/m3, sobre solo ~10.8 m2
de area) no puede contrastarse de forma honesta con este resultado: H_X aqui es solo la fraccion de 4 oxidos
inertes del talon (no incluye FeO, SnO2 residual, metal atrapado, etc.), y el H_SiO2 esta en su cota superior, no
es una estimacion libre.

### 5.6 Coeficientes con sentido fisico parcial (para RONDA 2)

A pesar de que el ajuste global es inadmisible, algunos coeficientes individuales SI tienen el signo y orden de
magnitud esperado y podrian rescatarse en una re-especificacion: Al2O3 y MgO no saturan en ninguna cota y siguen
el orden fisico esperado (mineral de Fe y concentrado con mas Al2O3/MgO relativo que dross o pellets, magnitudes
2-10% del peso de tolva, plausibles para ganga de mineral); MgO/CaO es el unico par con senal de rango (rho
hold-out 0.31) medianamente util. El problema esta concentrado en la ecuacion de SiO2 (y, por arrastre, en la
pureza de la cal).

---

## 6. Limitaciones

- Los supuestos de leyes de metal/dross/polvo y humedad (`SUPUESTOS` de `bal_features.py`) no estan calibrados
  contra un ensayo independiente; la seccion 1.4 muestra que afectan la masa absoluta de escoria en 2-14% segun el
  parametro, pero el KPI refinado en si es robusto (<0.2 pp) a esa incertidumbre.
- `cierre_sn_frac` tiene un sesgo pequeno pero real (-1.9%, p<1e-7) no explicado por las dos covariables probadas;
  puede reflejar perdidas sistematicas no modeladas (volatilizacion, arrastre) o un sesgo conjunto de supuestos
  que este experimento no puede separar.
- El techo de predictibilidad de `sn_perdido_escoria_frac` esta inflado por comparticion de columna cruda con 4 de
  los 39 features `st_*` (seccion 4); se reporta la version corregida como lectura principal.
- El modelo multi-oxido (seccion 5) usa `cum_feed_tolva_incl[t]` sin ponderar por la correlacion temporal entre
  tolvas; no se probo regularizacion L2/ridge sobre los coeficientes `a` ni una re-especificacion con menos
  parametros libres (agrupar tolvas, fijar mas coeficientes a 0 por prior fisico) por el limite de tiempo
  asignado a esta parte (40% del esfuerzo). El resultado reportado es la mejor version obtenida, no una version
  optimizada.
- La comparacion de `factor_escala_trazador` contra `idx_cronologico` (deriva) y contra `ratio_cal_carga`
  (artefacto) es observacional; no se descompuso el efecto conjunto con un modelo multivariado (ambas variables
  podrian estar correlacionadas entre si a lo largo de la campana).

---

## 7. Recomendaciones

**(i) KPI refinado con Sn en escoria (`recuperacion_refinada_pct`) -> ADOPTAR.**
Es una generalizacion fisicamente mas completa del proxy (incluye el Sn que efectivamente queda en la escoria
final, ~1.3% mediana del Sn de salida) sin costo aparente: correlaciona 0.99 con el proxy (no cambia las
conclusiones agregadas), es casi insensible a los supuestos de leyes/humedad (rango <0.2 pp), su techo de
predictibilidad es equivalente o mejor que el del proxy en la comparacion homogenea (R2 lockbox 0.089 vs 0.002),
y su ranking de targets teoricos ganadores es el mismo que con el proxy. Su valor agregado real es que
**reclasifica el 9.4% de los batches a otro cuartil**, relevante para auditorias o rankings de batch individual, y
que en lockbox favorece a los targets de matriz de escoria (IRF) sobre el SDI de forma mas clara que el proxy —
una senal a seguir en la sintesis (B-05).

**(ii) Factor de escala del trazador (`factor_escala_trazador`) -> RECHAZAR como corrector multiplicativo unico.**
Esta dominado por la fraccion de cal en la carga (rho=-0.35, R2=0.19), que es el artefacto contable ya identificado
en el experimento 18 (el trazador CaO-solo no separa el CaO de la cal del CaO de la ganga), mas una deriva
cronologica menor y significativa (rho=0.19). Aplicar un factor mediana fijo (1.08) a todos los batches
introduciria un sesgo sistematico correlacionado con la mezcla de carga de cada batch. **Usar directamente
`masa_escoria_balance_kg`** (que ya calcula la masa desde las tolvas sin pasar por el trazador CaO) en lugar de
"trazador x factor" para cualquier estimacion de masa absoluta de escoria.

**(iii) Modelo multi-oxido con talon (`b6_masa_escoria_multioxido_kg`, coeficientes de ganga por tolva) -> RONDA 2.**
El ajuste es matematicamente estable (multi-start identico) pero fisicamente inadmisible: 4 de 5 coeficientes de
SiO2 saturan en su cota superior (100% SiO2 por tolva), el talon de SiO2 satura en 10 t, la pureza de la cal cae a
su cota inferior (0.70 en vez de ~0.85-0.95), la validacion GroupKFold da R2<=0 en los 3 pares de log-cociente
(el modelo no generaliza), y la masa de escoria resultante es 5.4x la del balance global. **No usar en su forma
actual** (ni para masa de escoria ni para talon ni para atribucion de ganga por tolva). Para una segunda ronda:
(a) fijar la pureza de la cal cerca de un valor externo conocido (0.85-0.90) en vez de dejarla libre en [0.7,1.0],
dado que el ajuste la empuja sistematicamente a la cota inferior; (b) anadir una ecuacion de escala absoluta
(por ejemplo, anclar la masa total contra `masa_escoria_balance_kg` en el ultimo escalon de cada batch) ademas de
los log-cocientes, que por construccion son invariantes de escala; (c) reducir parametros libres agrupando tolvas
con perfiles de alimentacion muy colineales o imponiendo mas ceros por prior fisico (el reciclo t3, si es escoria
reprocesada, probablemente no deberia tratarse como una corriente de ganga fresca independiente); (d) considerar
una penalizacion ridge sobre los coeficientes `a` para desalentar la saturacion en cotas. Los coeficientes de
Al2O3 y MgO (que no saturan y tienen sentido fisico razonable) son el punto de partida mas prometedor para esa
segunda ronda.
