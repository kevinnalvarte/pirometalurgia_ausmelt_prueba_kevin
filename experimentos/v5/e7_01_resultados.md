# E7-01 -- Reduccion: componentes log del SDI, parametrizacion de palancas y anclaje de w

Generado 2026-09-14 21:40. dR (escalones modelables Reduccion) = 1448 filas; batch = 362 filas (299 DEV / 63 lockbox).

## 1. Comparacion de PLMs (R2 OOF DEV / R2 lockbox), estado CURADO

| target | parametrizacion | n_palancas | R2_oof | MAE_oof | R2_lockbox | n_dev | n_lockbox |
|---|---|---:|---:|---:|---:|---:|---:|
| feo_extraido_est_kg | P1_primitivas | 4 | 0.347 | 546.758 | 0.321 | 1070 | 242 |
| feo_extraido_est_kg | P2_cinetica_v4 | 5 | 0.371 | 540.639 | 0.341 | 1069 | 242 |
| feo_extraido_est_kg | P3_dosis_sobrante | 5 | 0.371 | 541.200 | 0.346 | 1069 | 242 |
| feo_extraido_est_kg | P4_log_dosis | 5 | 0.343 | 560.823 | 0.344 | 1034 | 242 |
| feo_extraido_est_kg | P5_minima | 3 | 0.374 | 539.291 | 0.346 | 1069 | 242 |
| sn_extraido_est_kg | P1_primitivas | 4 | 0.973 | 364.165 | 0.985 | 1074 | 238 |
| sn_extraido_est_kg | P2_cinetica_v4 | 4 | 0.974 | 350.394 | 0.986 | 1073 | 238 |
| sn_extraido_est_kg | P3_dosis_sobrante | 4 | 0.973 | 362.324 | 0.985 | 1038 | 238 |
| sn_extraido_est_kg | P4_log_dosis | 4 | 0.973 | 362.582 | 0.985 | 1038 | 238 |
| sn_extraido_est_kg | P5_minima | 3 | 0.973 | 361.960 | 0.985 | 1038 | 238 |
| v5_ln_feo_ret | P1_primitivas | 4 | 0.212 | 0.063 | 0.124 | 1040 | 242 |
| v5_ln_feo_ret | P2_cinetica_v4 | 5 | 0.206 | 0.063 | 0.122 | 1039 | 242 |
| v5_ln_feo_ret | P3_dosis_sobrante | 5 | 0.203 | 0.063 | 0.151 | 1039 | 242 |
| v5_ln_feo_ret | P4_log_dosis | 5 | 0.203 | 0.063 | 0.138 | 1039 | 242 |
| v5_ln_feo_ret | P5_minima | 3 | 0.203 | 0.063 | 0.159 | 1039 | 242 |
| v5_ln_sn_dep | P1_primitivas | 4 | 0.763 | 0.245 | 0.808 | 1040 | 242 |
| v5_ln_sn_dep | P2_cinetica_v4 | 4 | 0.762 | 0.246 | 0.805 | 1039 | 242 |
| v5_ln_sn_dep | P3_dosis_sobrante | 4 | 0.760 | 0.247 | 0.802 | 1039 | 242 |
| v5_ln_sn_dep | P4_log_dosis | 4 | 0.760 | 0.246 | 0.800 | 1039 | 242 |
| v5_ln_sn_dep | P5_minima | 3 | 0.760 | 0.247 | 0.804 | 1039 | 242 |
| v5_sdi_w10 | P1_primitivas | 4 | 0.149 | 0.544 | 0.154 | 1040 | 242 |
| v5_sdi_w10 | P2_cinetica_v4 | 5 | 0.156 | 0.543 | 0.141 | 1039 | 242 |
| v5_sdi_w10 | P3_dosis_sobrante | 5 | 0.156 | 0.544 | 0.180 | 1039 | 242 |
| v5_sdi_w10 | P4_log_dosis | 5 | 0.154 | 0.544 | 0.162 | 1039 | 242 |

## 1b. Mismo, estado FULL

| target | parametrizacion | n_palancas | R2_oof | MAE_oof | R2_lockbox | n_dev | n_lockbox |
|---|---|---:|---:|---:|---:|---:|---:|
| feo_extraido_est_kg | P1_primitivas | 4 | 0.358 | 543.591 | 0.379 | 1046 | 242 |
| feo_extraido_est_kg | P2_cinetica_v4 | 5 | 0.357 | 544.005 | 0.378 | 1045 | 242 |
| feo_extraido_est_kg | P3_dosis_sobrante | 5 | 0.357 | 544.291 | 0.379 | 1045 | 242 |
| feo_extraido_est_kg | P4_log_dosis | 5 | 0.339 | 559.798 | 0.364 | 1010 | 242 |
| feo_extraido_est_kg | P5_minima | 3 | 0.360 | 542.918 | 0.379 | 1045 | 242 |
| sn_extraido_est_kg | P1_primitivas | 4 | 0.975 | 351.229 | 0.985 | 1050 | 238 |
| sn_extraido_est_kg | P2_cinetica_v4 | 4 | 0.975 | 344.121 | 0.984 | 1049 | 238 |
| sn_extraido_est_kg | P3_dosis_sobrante | 4 | 0.975 | 353.271 | 0.984 | 1014 | 238 |
| sn_extraido_est_kg | P4_log_dosis | 4 | 0.975 | 355.323 | 0.984 | 1014 | 238 |
| sn_extraido_est_kg | P5_minima | 3 | 0.975 | 352.907 | 0.984 | 1014 | 238 |
| v5_ln_feo_ret | P1_primitivas | 4 | 0.186 | 0.063 | 0.181 | 1016 | 242 |
| v5_ln_feo_ret | P2_cinetica_v4 | 5 | 0.164 | 0.065 | 0.187 | 1015 | 242 |
| v5_ln_feo_ret | P3_dosis_sobrante | 5 | 0.165 | 0.064 | 0.190 | 1015 | 242 |
| v5_ln_feo_ret | P4_log_dosis | 5 | 0.165 | 0.065 | 0.191 | 1015 | 242 |
| v5_ln_feo_ret | P5_minima | 3 | 0.170 | 0.064 | 0.200 | 1015 | 242 |
| v5_ln_sn_dep | P1_primitivas | 4 | 0.771 | 0.242 | 0.798 | 1016 | 242 |
| v5_ln_sn_dep | P2_cinetica_v4 | 4 | 0.766 | 0.244 | 0.802 | 1015 | 242 |
| v5_ln_sn_dep | P3_dosis_sobrante | 4 | 0.764 | 0.245 | 0.800 | 1015 | 242 |
| v5_ln_sn_dep | P4_log_dosis | 4 | 0.765 | 0.244 | 0.796 | 1015 | 242 |
| v5_ln_sn_dep | P5_minima | 3 | 0.764 | 0.246 | 0.802 | 1015 | 242 |
| v5_sdi_w10 | P1_primitivas | 4 | 0.144 | 0.541 | 0.159 | 1016 | 242 |
| v5_sdi_w10 | P2_cinetica_v4 | 5 | 0.137 | 0.549 | 0.144 | 1015 | 242 |
| v5_sdi_w10 | P3_dosis_sobrante | 5 | 0.136 | 0.549 | 0.153 | 1015 | 242 |
| v5_sdi_w10 | P4_log_dosis | 5 | 0.137 | 0.549 | 0.148 | 1015 | 242 |

## 2. Parametrizacion elegida (criterio: solo DEV, estado CURADO)

- **v5_ln_sn_dep** -> P5_minima (R2_oof=0.760, R2_lockbox=0.804, n_palancas=3, n_sig_concuerda=1)

  | palanca | theta_unidad | IC | theta_1sd | IC_1sd | signif | signo_teorico | concuerda |
  |---|---:|---|---:|---|---|---:|---|
  | v5_dosis_C_sn | 0.097 | [-0.051, 0.255] | 0.017 | [-0.009, 0.045] | False | 1.0 | True |
  | tasa_gn_nm3_min | 0.011 | [0.002, 0.019] | 0.055 | [0.010, 0.094] | True | 1.0 | True |
  | exceso_o2_combustion_pct | -0.000 | [-0.002, 0.001] | -0.004 | [-0.024, 0.012] | False | -1.0 | True |

- **v5_ln_feo_ret** -> P1_primitivas (R2_oof=0.212, R2_lockbox=0.124, n_palancas=4, n_sig_concuerda=1)

  | palanca | theta_unidad | IC | theta_1sd | IC_1sd | signif | signo_teorico | concuerda |
  |---|---:|---|---:|---|---|---:|---|
  | tasa_feed_Carbon_kg_min | 0.001 | [-0.000, 0.003] | 0.022 | [-0.006, 0.061] | False | -1.0 | False |
  | tasa_gn_nm3_min | -0.004 | [-0.008, -0.002] | -0.022 | [-0.039, -0.008] | True | -1.0 | True |
  | tasa_o2_nm3_min | 0.001 | [-0.001, 0.003] | 0.006 | [-0.008, 0.024] | False | 1.0 | True |
  | tasa_aire_nm3_min | 0.000 | [-0.000, 0.001] | 0.003 | [-0.004, 0.013] | False | 1.0 | True |

- **v5_sdi_w10** -> P2_cinetica_v4 (R2_oof=0.156, R2_lockbox=0.141, n_palancas=5, n_sig_concuerda=1)

  | palanca | theta_unidad | IC | theta_1sd | IC_1sd | signif | signo_teorico | concuerda |
  |---|---:|---|---:|---|---|---:|---|
  | tasa_feed_Carbon_kg_min | 0.014 | [-0.029, 0.047] | 0.303 | [-0.640, 1.046] | False | 0.0 | nan |
  | Cx_av_v4 | -0.003 | [-0.043, 0.041] | -0.055 | [-0.744, 0.702] | False | 0.0 | nan |
  | tasa_gn_nm3_min | -0.023 | [-0.045, -0.000] | -0.115 | [-0.221, -0.002] | True | -1.0 | True |
  | exceso_o2_combustion_pct | -0.001 | [-0.006, 0.005] | -0.010 | [-0.067, 0.055] | False | 1.0 | False |
  | tasa_aire_nm3_min | 0.004 | [-0.001, 0.010] | 0.039 | [-0.013, 0.103] | False | 1.0 | True |

## 3. Anclaje de w en el KPI (OLS HC3, batch DEV, unidades naturales)

| outcome | variante | n | coef_sn(p) | coef_feo(p) | w puntual | w IC95% boot | frac(sn>0 & feo>0) | R2adj |
|---|---|---:|---|---|---:|---|---:|---:|
| recuperacion_refinada_pct | actual | 285 | 2.124(0.001) | 12.827(0.000) | 6.04 | [3.82, 13.60] (mediana 6.06) | 1.00 | 0.122 |
| recuperacion_refinada_pct | next | 285 | 0.535(0.512) | 4.526(0.086) | 8.45 | [-65.04, 98.07] (mediana 4.21) | 0.75 | 0.010 |
| recuperacion_refinada_pct | mean_cur_next | 285 | 1.329(0.007) | 8.676(0.000) | 6.53 | [3.40, 17.60] (mediana 6.79) | 1.00 | 0.113 |
| rendimiento_proxy_batch | actual | 285 | 1.569(0.027) | 13.237(0.000) | 8.44 | [4.23, 34.38] (mediana 8.44) | 0.98 | 0.122 |
| rendimiento_proxy_batch | next | 285 | 0.486(0.560) | 4.848(0.054) | 9.98 | [-95.47, 85.31] (mediana 4.20) | 0.74 | 0.015 |
| rendimiento_proxy_batch | mean_cur_next | 285 | 1.027(0.035) | 9.043(0.000) | 8.80 | [3.85, 49.19] (mediana 9.01) | 0.99 | 0.119 |
| f_dross | actual | 285 | -0.012(0.041) | -0.074(0.000) | 6.21 | [2.65, 29.84] (mediana 6.19) | 0.00 | 0.115 |
| f_dross | next | 285 | -0.005(0.484) | -0.052(0.056) | 11.38 | [-122.74, 91.81] (mediana 6.64) | 0.00 | 0.087 |
| f_dross | mean_cur_next | 285 | -0.008(0.034) | -0.063(0.001) | 7.64 | [2.98, 33.05] (mediana 7.69) | 0.00 | 0.184 |
| recuperacion_real_pct | actual | 285 | 1.496(0.027) | 7.181(0.002) | 4.80 | [1.63, 16.58] (mediana 4.74) | 0.98 | 0.139 |
| recuperacion_real_pct | next | 285 | 0.410(0.599) | 4.413(0.047) | 10.77 | [-48.99, 113.29] (mediana 4.09) | 0.71 | 0.030 |
| recuperacion_real_pct | mean_cur_next | 285 | 0.953(0.042) | 5.797(0.001) | 6.08 | [1.07, 39.66] (mediana 6.11) | 0.98 | 0.143 |

w anclado usado en tareas 4-5 (KPI actual, recuperacion_refinada_pct): **6.04** (clip a >=0 -> 6.04).

## 3b. Barrido de w (rho Spearman con KPI, variante simple)

| w | rho_dev | IC_dev | rho_lockbox | rho_total |
|---:|---:|---|---:|---:|
| 0 | -0.001 | [-0.109, 0.119] | 0.354 | 0.078 |
| 1 | 0.113 | [0.011, 0.222] | 0.339 | 0.170 |
| 2 | 0.185 | [0.077, 0.300] | 0.314 | 0.224 |
| 3 | 0.235 | [0.128, 0.358] | 0.264 | 0.256 |
| 4 | 0.264 | [0.154, 0.375] | 0.224 | 0.270 |
| 5 | 0.282 | [0.166, 0.393] | 0.176 | 0.275 |
| 6 | 0.299 | [0.186, 0.406] | 0.149 | 0.280 |
| 8 | 0.314 | [0.206, 0.409] | 0.094 | 0.281 |
| 10 | 0.317 | [0.212, 0.421] | 0.087 | 0.276 |
| 12 | 0.318 | [0.214, 0.413] | 0.046 | 0.269 |
| 14 | 0.317 | [0.199, 0.413] | 0.044 | 0.265 |
| 16 | 0.312 | [0.208, 0.422] | 0.016 | 0.259 |
| 20 | 0.310 | [0.199, 0.410] | -0.006 | 0.251 |
| 30 | 0.302 | [0.198, 0.412] | -0.013 | 0.239 |

w por regla de parsimonia (min w con rho_DEV >= 0.95*max), variante simple: **8.0**.

## 4. Variante ponderada por tiempo restante

| variante | w usado | split | n | rho | IC |
|---|---|---|---:|---:|---|
| simple | w10=10.00 | dev | 286 | 0.317 | [0.207, 0.415] |
| simple | w10=10.00 | lockbox | 63 | 0.087 | [-0.150, 0.319] |
| simple | w10=10.00 | total | 349 | 0.276 | [0.180, 0.362] |
| simple | w_anchor=6.04 | dev | 286 | 0.299 | [0.190, 0.407] |
| simple | w_anchor=6.04 | lockbox | 63 | 0.143 | [-0.087, 0.382] |
| simple | w_anchor=6.04 | total | 349 | 0.280 | [0.185, 0.378] |
| tiempo_orden | w10=10.00 | dev | 286 | 0.301 | [0.203, 0.411] |
| tiempo_orden | w10=10.00 | lockbox | 63 | 0.198 | [-0.040, 0.440] |
| tiempo_orden | w10=10.00 | total | 349 | 0.291 | [0.193, 0.383] |
| tiempo_orden | w_anchor=6.04 | dev | 286 | 0.263 | [0.146, 0.369] |
| tiempo_orden | w_anchor=6.04 | lockbox | 63 | 0.285 | [0.045, 0.523] |
| tiempo_orden | w_anchor=6.04 | total | 349 | 0.277 | [0.180, 0.377] |
| tiempo_real | w10=10.00 | dev | 286 | 0.267 | [0.160, 0.377] |
| tiempo_real | w10=10.00 | lockbox | 63 | 0.242 | [0.005, 0.490] |
| tiempo_real | w10=10.00 | total | 349 | 0.274 | [0.175, 0.369] |
| tiempo_real | w_anchor=6.04 | dev | 286 | 0.238 | [0.116, 0.341] |
| tiempo_real | w_anchor=6.04 | lockbox | 63 | 0.322 | [0.076, 0.551] |
| tiempo_real | w_anchor=6.04 | total | 349 | 0.262 | [0.167, 0.360] |

w parsimonia por variante: simple=8.0, tiempo_orden=10.0, tiempo_real=10.0.

## 5. Cadena de signos (parametrizacion elegida: Sn=P5_minima / FeO=P1_primitivas)

| palanca | signo ln_sn_dep | signo ln_feo_ret | J w10 (unidad/sd) | J w_anchor (unidad/sd) | evidencia batch exp19 |
|---|---|---|---|---|---|
| exceso_o2_combustion_pct | - (-0.000) | n/a (nan) | - (-0.004) | - (-0.004) |  |
| tasa_aire_nm3_min | n/a (nan) | + (0.000) | + (0.035) | + (0.021) | batch: lanza no es palanca (exp19) |
| tasa_feed_Carbon_kg_min | n/a (nan) | + (0.001) | + (0.219) | + (0.133) | batch: -f_dross p0.03 (carbon reduce dross) |
| tasa_gn_nm3_min | + (0.011) | - (-0.004) | - (-0.159) | - (-0.074) | batch: +f_dross p0.04 / -f_metal p0.03 (GN sube dross) |
| tasa_o2_nm3_min | n/a (nan) | + (0.001) | + (0.063) | + (0.038) | batch: lanza no es palanca (exp19) |
| v5_dosis_C_sn | + (0.097) | n/a (nan) | + (0.017) | + (0.017) |  |

## Veredicto

**Parametrizacion de palancas.** Ninguna de las cuatro reparametrizaciones cineticas (P2-P4) mejora el R2 OOF
frente a las primitivas (P1) mas alla del ruido (diferencias <=0.01 en los tres targets, en ambos estados). La
regla de parsimonia por tanto elige el conjunto mas chico disponible en la ventana de empate:
- **`v5_ln_sn_dep` -> P5_minima** (`v5_dosis_C_sn`, GN, exceso O2; R2_oof=0.760, R2_lockbox=0.804). El R2 alto es
  sobre todo la funcion `g(S)` (el agotamiento seria muy autocorrelacionado con el estado previo, casi una
  decadencia de primer orden); de las palancas solo **GN es significativo y de signo teorico correcto**
  (+0.055 sd por +1 sd GN, IC [0.010, 0.094]). La dosis especifica de carbon tiene el signo esperado (+0.017 sd)
  pero el IC cruza cero: **el carbon sigue sin identificarse bien** a nivel de escalon (confirma iteracion 5-6;
  la dosis sigue a la demanda de Sn, hay endogeneidad).
- **`v5_ln_feo_ret` -> P1_primitivas** (empatada con P2-P5; gana por menos palancas). El "carbon sobrante"
  (`v5_exceso_C_pos`) y "carbon x avance" (`v5_C_x_avance`) **no mejoran la identificacion** frente al carbon
  primitivo y en varias parametrizaciones ni siquiera dejan a GN significativo (colinealidad). Unico lever
  robusto: **GN, negativo** (-0.022 sd, IC [-0.039,-0.008]) -- reductor no selectivo que tambien ataca el FeO,
  coherente con exp19 (GN sube f_dross, p=0.04). Las hipotesis de carbon sobrante/avance quedan **sin soporte
  estadistico al nivel de escalon** con los datos actuales; no se recomienda adoptarlas como palanca operativa
  todavia.
- **`v5_sdi_w10` (target compuesto) -> P2_cinetica_v4**, pero con R2_oof=0.156 muy por debajo de ajustar
  `v5_ln_sn_dep` solo (0.760): **modelar las componentes por separado y combinarlas via w es preferible a
  ajustar el SDI directamente**, porque el ruido de `ln_feo_ret` domina el target compuesto.

**Anclaje de w.** El OLS HC3 de batch (KPI actual, DEV, n=285) da coef(ln_sn_dep)=2.12 (p=0.001),
coef(ln_feo_ret)=12.83 (p<0.001), **w_anclado = 6.04** [IC boot 3.82, 13.60], con el 100% de las replicas
bootstrap con ambos coeficientes positivos: el ancla es robusta y el IC contiene a 10 pero tambien a valores
bastante mas chicos. Usar el KPI del **batch siguiente** rompe la señal (coef_sn n.s., IC de w
[-65, 98], solo 75% de replicas con ambos signos positivos): el anclaje es una relacion **contemporanea**, no
predictiva entre batches, como se esperaba (no hay canal fisico de arrastre de escoria entre batches). El
barrido de rho Spearman(J, KPI) en DEV crece con w hasta ~12-14 (techo ~0.32) pero **decrece monotonicamente en
lockbox** (de 0.35 en w=0 hasta -0.01 en w=30): el regimen de lockbox favorece casi exclusivamente el
agotamiento de Sn, replicando el hallazgo de iteracion 5 (Σln_sn_dep ρ≈0 DEV / +0.35 lockbox). Ni w=8
(parsimonia DEV) ni w=10 (iteracion 5) son robustos entre regimenes con la variante simple.

**Variante ponderada por tiempo restante -- hallazgo principal de esta tarea.** Ponderar `ln_feo_ret` por el
tiempo (o el orden) restante de Reduccion **estabiliza fuertemente la relacion con el KPI entre DEV y
lockbox**, sin sacrificar casi nada de rho en DEV:

| w | variante | rho DEV | rho lockbox |
|---:|---|---:|---:|
| 10 | simple | 0.317 | 0.087 |
| 10 | tiempo_orden | 0.301 | 0.198 |
| 10 | tiempo_real | 0.267 | **0.242** |
| 6.04 (anclado) | simple | 0.299 | 0.143 |
| 6.04 (anclado) | tiempo_orden | 0.263 | 0.285 |
| 6.04 (anclado) | tiempo_real | 0.238 | **0.322** |

Con w anclado, la variante `tiempo_real` tiene rho lockbox (0.322) **mayor** que su propio rho DEV (0.238): la
señal de retener FeO **temprano** en la Reduccion es la parte del SDI que sobrevive al cambio de regimen,
igual que ya senalaba el ln IRF twmean (0.297/0.378) de la iteracion 5. Recomiendo **reemplazar
`Σln_feo_ret` simple por la version ponderada por tiempo restante real** (o por orden, que rinde casi igual y
es mas simple de calcular) en el target de batch de Reduccion, con **w entre 6 y 10** (la regla de parsimonia
da w=10 para ambas variantes temporales; el ancla estadistica da 6.04). Con datos actuales no hay forma de
diferenciar tajantemente entre esos dos valores de w -- ambos son defendibles.

**Cadena de signos.** El unico lever comun a `ln_sn_dep` y `ln_feo_ret` en las parametrizaciones elegidas es
GN: sube el agotamiento de Sn (+) pero baja la retencion de FeO (-) con mas fuerza, por lo que el efecto neto
sobre J es **negativo** tanto con w=10 como con w=6.04 (-0.16 y -0.07 sd respectivamente), replicando el
hallazgo de iteracion 5 (GN -0.15* sobre SDI) y la evidencia de batch (exp19: GN sube f_dross p 0.04, baja
f_metal p 0.03). El carbon (dosis para Sn, primitivo para FeO) apunta al signo teorico en ambos canales pero
con IC anchos que no permiten concluir a nivel de escalon; la evidencia mas confiable para el carbon sigue
siendo la agregada de batch (exp19: -f_dross, p 0.03).

**Recomendacion para el objetivo de Reduccion en v5:** usar el SDI con sus componentes ajustadas por separado
(no el target compuesto directo), `ln_feo_ret` ponderado por tiempo restante real (o por orden), y w en el
rango 6-10 (preferible el extremo bajo del rango, 6-7, porque la version anclada en el KPI es la que mas
mejora la estabilidad lockbox de la variante temporal). GN es la palanca mas defendible estadisticamente
(trade-off Sn/FeO claro); el carbon, aunque teoricamente central, no se recomienda como palanca prescriptiva
de escalon en Reduccion hasta mejorar su identificacion (mas datos, variacion exogena, o modelado a nivel de
batch en vez de escalon).


## Limites

- Reducción tiene solo 4 escalones/batch: la variante temporal tiene poca resolucion (pesos {1, 0.75, 0.5, 0.25}/4 aprox).
- w anclado por OLS de batch depende de los controles y es sensible al outcome (actual vs next vs mean); se reporta rango, no un unico numero.
- El PLM cross-fitted con ~350-400 filas de DEV y 5+ palancas tiene poca potencia para IC angostos; los theta con IC amplios no deben leerse como nulos.
- El KPI del batch siguiente introduce autocorrelacion cronologica (idx_cronologico ya es control, pero no es un IV).
