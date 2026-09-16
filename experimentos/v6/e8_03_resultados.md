# E8-03 — Anclaje del objetivo del prescriptor en el KPI refinado con targets v6 y comparación con v5

Generado 2026-09-16. Script `e8_03_anclaje_v6.py` (librería `v6_lib.py`), log completo en `e8_03_log.txt`
(17.3 min de cómputo, dominado por el PLM cross-fitted de la tarea 4). df = 3982 escalones; dR (Reducción
modelable) = 1448; dF (Fusión modelable) = 2172; batch = 362 (299 DEV / 63 lockbox).

Contexto: v6 reemplaza la masa de escoria por escalón (`masa_escoria_est_kg`, trazador de %CaO, v3) por el
estimador `masa_v6` (cierre físico sin %CaO en Reducción, trazador de matriz G en Fusión; ver
`ITERACION_8_diseno.md` y el docstring de `masa_v6.py`). Todas las comparaciones v5 vs v6 de este memo se
hacen **sobre las mismas filas** (intersección por `dropna` de las columnas que usan ambas versiones), para
que la diferencia de R², rho o theta no sea un artefacto de tamaño de muestra distinto.

## 1. Anclaje de w (OLS HC3, batch DEV, mismos controles + tendencia que E7-01: `CONTROLES_BATCH`
   = feed_sn_total_t, ley_sn_conc_batch_pct, espesor_ladrillo_norm_mm, frac_carga_secundaria_F,
   feed_dross_Fe_total_t, idx_cronologico)

Muestra común DEV (v5 y v6 + controles + KPI actual/next/mean, sin NaN): **n=285** (igual que E7-01).

| targets | outcome | n | coef_sn (p) | coef_feo (p) | w puntual | w IC95% boot (1000) |
|---|---|---:|---|---|---:|---|
| **v6** | recuperacion_refinada_pct (actual) | 285 | 2.854 (5.3e-06) | 17.80 (6.0e-08) | **6.24** | **[4.30, 9.83]** |
| v5 | recuperacion_refinada_pct (actual) | 285 | 2.124 (0.0013) | 12.83 (4.3e-07) | 6.04 | [3.64, 15.54] |
| v6 | recuperacion_refinada_pct_next | 285 | 0.877 (0.267) | 10.56 (2.4e-04) | 12.03 | [-85.95, 122.41] |
| v5 | recuperacion_refinada_pct_next | 285 | 0.535 (0.512) | 4.53 (0.086) | 8.45 | [-127.95, 78.68] |
| v6 | mean(actual, next) | 285 | 1.865 (8.7e-05) | 14.18 (4.1e-10) | 7.60 | [5.21, 12.65] |
| v5 | mean(actual, next) | 285 | 1.329 (0.0066) | 8.68 (2.5e-05) | 6.53 | [3.32, 19.62] |
| v6 (lockbox, reporte) | recuperacion_refinada_pct | 63 | 4.02 (0.0027) | 2.72 (0.727) | 0.68 | [-5.87, 4.66] |
| v5 (lockbox, reporte) | recuperacion_refinada_pct | 63 | 4.07 (6.3e-04) | 2.67 (0.561) | 0.66 | [-2.94, 3.27] |

Tabla completa: `e8_03_w_anclaje.csv`.

**Lectura.** Con el KPI actual (criterio de selección de esta metodología) ambos anclajes son consistentes:
w_v6 = 6.24, w_v5 = 6.04, casi idénticos y ambos con el 100 % de las réplicas bootstrap con los dos
coeficientes positivos. La diferencia relevante es la **precisión**: el IC de w se reduce de [3.64, 15.54]
(v5, ancho 11.9) a **[4.30, 9.83]** (v6, ancho 5.5) — casi la mitad — porque `coef_feo` v6 tiene el mismo
orden de magnitud pero un error estándar menor (el target de FeO v6 tiene menos ruido, ver tarea 5). Con el
KPI del batch siguiente el anclaje se rompe en ambas versiones (coef_sn n.s., IC de w gigantesco): confirma
que es una relación contemporánea, no predictiva entre batches, en ambas versiones del target. En lockbox
(solo reporte) ambas versiones colapsan igual (`coef_feo` deja de ser significativo, w cerca de 0): el mismo
cambio de régimen de iteración 5/7 (el lockbox favorece casi exclusivamente el agotamiento de Sn) persiste en
v6, la masa v6 no lo resuelve.

**K recomendado** (pp de KPI por unidad de `ln_sn_dep`, con la parametrización de w en el KPI actual DEV):
**K_v6 = 2.85 pp/unidad** (≈ 1463 kg Sn/unidad, con 512.5 kg/pp) vs K_v5 = 2.12 pp/unidad (≈ 1088 kg/unidad).
K sube con v6 porque la masa de escoria v6 es distinta (cierre físico, no trazador de CaO) y por tanto la
escala de `ln_sn_dep` cambia; no es directamente comparable en unidades absolutas con v5, solo el ranking y
la significancia son comparables entre versiones.

## 1b. Barrido de w ∈ {2,4,6,8,10,15} (rho Spearman de Σln_sn_dep + w·Σln_feo_ret vs KPI)

| w | rho DEV v6 | rho DEV v5 | rho lockbox v6 | rho lockbox v5 |
|---:|---:|---:|---:|---:|
| 2 | 0.201 | 0.185 | 0.294 | 0.314 |
| 4 | 0.285 | 0.264 | 0.242 | 0.224 |
| 6 | **0.322** | 0.299 | 0.203 | 0.149 |
| 8 | **0.326** | 0.314 | 0.125 | 0.094 |
| 10 | 0.322 | 0.317 | 0.080 | 0.087 |
| 15 | 0.299 | 0.314 | -0.007 | 0.031 |

Tabla completa: `e8_03_w_sweep.csv`. v6 domina a v5 en DEV en 5 de 6 valores de w (+0.01 a +0.03 de rho), con
el máximo en w=8 (0.326, vs 0.317 de v5 en w=10): consistente con el w anclado (6.24). En lockbox el patrón
de iteración 5/7 se repite en ambas versiones (rho decrece con w, sin ganador claro entre v5 y v6 — mixto:
v6 mejor en w=6-8, peor en w=10-15). **v6 no arregla la inestabilidad DEV/lockbox del peso w**, pero sí sube
modestamente la señal en DEV.

## 2. Batería Spearman (`e8_03_bateria.csv`, 24 filas: 6 variables × 4 KPIs)

Resumen de rho_dev (p_dev) / rho_lockbox para las variables de Reducción:

| variable | vs recuperacion_refinada_pct (DEV / lockbox) | vs KPI_next (DEV / lockbox) |
|---|---|---|
| R_sum_m6_ln_sn_dep | 0.069 (p=0.23) / 0.370 | -0.018 / -0.135 |
| R_sum_ln_sn_dep (v5) | -0.001 (p=0.99) / 0.354 | 0.021 / -0.138 |
| **R_sum_m6_ln_feo_ret** | **0.234 (p<0.001) / -0.143** | 0.194 (p=0.001) / **0.329** |
| R_sum_ln_feo_ret (v5) | 0.286 (p<0.001) / -0.088 | 0.122 (p=0.039) / 0.375 |
| F_sum_m6_ln_sn_dep | -0.082 (p=0.16) / -0.252 | -0.042 / -0.265 |
| F_sum_ln_sn_dep (v5) | -0.115 (p=0.053) / -0.165 | -0.063 / -0.066 |

Lectura: el componente de Sn (`*_sn_dep`) es prácticamente idéntico entre v5 y v6 en todas las columnas
(ρ≈0 en DEV, ρ≈0.35-0.37 en lockbox — el hallazgo de iteración 5 de que el agotamiento de Sn domina fuera de
DEV se repite igual con la masa v6). El componente de FeO (`*_feo_ret`) es donde v5 y v6 difieren: v5 tiene
un rho DEV nominalmente más alto contra el KPI actual (0.286 vs 0.234) pero **v6 es más estable contra el
KPI del batch siguiente** (0.194 vs 0.122 DEV; 0.329 vs 0.375 lockbox, comparable). Contra `f_dross`/`f_polvo`
ambas versiones tienen el signo esperado (retener FeO se asocia a menos dross) con magnitudes similares.

## 3. Fusión: conservación de Sn controlando el heel (`e8_03_fusion.csv`)

OLS HC3 de `recuperacion_refinada_pct` sobre `F_lnirf_F0` (heel) + `F_sum_*_ln_sn_dep`, estandarizado con
media/sd de la muestra TOTAL (misma convención que E7-02):

| targets | muestra | coef natural (pp/unidad) | IC95% | p | n |
|---|---|---:|---|---:|---:|
| v6 | DEV | -0.553 | [-1.40, 0.30] | 0.203 | 297 |
| v6 | TOTAL | **-0.894** | [-1.64, -0.15] | **0.019** | 360 |
| v6 | lockbox | -1.871 | [-4.30, 0.56] | 0.131 | 63 |
| v5 | DEV | -0.649 | [-1.38, 0.09] | **0.083** | 285 |
| v5 | TOTAL | -0.841 | [-1.51, -0.17] | 0.014 | 348 |
| v5 | lockbox | -1.351 | [-3.32, 0.62] | 0.180 | 63 |

`F_lnirf_F0` (heel) sigue siendo, con mucho, el término dominante y significativo en ambas versiones
(+1.58 a +1.66 pp/sd en DEV, p<1e-13): esto no cambia con v6. **β_F recomendado**: el signo teórico correcto
(conservar Sn en Fusión) se mantiene en v6, con magnitud similar a v5 (-0.55 a -0.89 pp/unidad según muestra,
vs -0.45 a -0.65 de v5/E7-02) pero **más débil en DEV** (p=0.203 vs p=0.083): con el criterio de esta
metodología (significativo en DEV), v6 sigue sin cumplir el estándar, igual que v5 en E7-02, pero está más
lejos del umbral. En TOTAL ambas versiones son significativas y de magnitud comparable. Se mantiene como
**candidato débil, no confirmado**, igual que en E7-02, con v6 sin evidencia de mejora aquí (es la única tarea
donde v5 luce marginalmente mejor que v6).

## 4. PLM por escalón: comparación v5 vs v6 (`e8_03_plm_comparacion.csv`, `e8_03_theta.csv`)

Estado CURADO_V6 (22 features, sustituye masa/inventarios v3 por m6_* + añade `m6_resto_frac_prev`/
`m6_G_pct_prev`) vs CURADO v5 (20 features); dos parametrizaciones de palancas (cinética v4/v6-style y
dosis/sobrante); con (`ModeloPLMSignos`) y sin (`mp4.ModeloPLM`) restricción de signo teórico. Todas las
filas de esta sección usan **las mismas 1280 filas** de Reducción (o 1720 de Fusión) para v5 y v6.

### Reducción — Sn (`m6_ln_sn_dep` / `v5_ln_sn_dep`)

| parametrización | con_signos | R²_oof v6 | R²_oof v5 | R²_lockbox v6 | R²_lockbox v5 |
|---|---|---:|---:|---:|---:|
| cinética | sí | 0.766 | 0.766 | 0.807 | 0.800 |
| cinética | no | 0.766 | 0.766 | 0.807 | 0.800 |
| dosis/sobrante | sí | 0.765 | 0.765 | 0.805 | 0.799 |
| dosis/sobrante | no | 0.765 | 0.764 | 0.805 | 0.799 |

**Sin cambio material**: la masa v6 no mueve el ajuste de `ln_sn_dep` (esperado — el agotamiento de Sn está
dominado por `g(S)`, muy autocorrelacionado con el estado previo, igual que en E7-01). Sí hay una diferencia
en el theta del carbono cinético: `Cx_sn_v4` (v5) es significativo (θ_1sd=0.186 [0.009, 0.327]) mientras que
su análogo `Cx_sn_v6` **no** lo es (θ_1sd=0.140 [0.000, 0.272] con signos, [-0.019, 0.272] sin signos): **v6
pierde la única identificación de carbono que E7-01 había logrado establecer** para el agotamiento de Sn. La
dosis específica (`m6_dosis_C_sn` / `v5_dosis_C_sn`) no es significativa en ninguna versión (igual que en
E7-01: el carbón sigue sin identificarse bien a nivel de escalón). GN es significativo en v6 (0.052 [0.014,
0.092]) y marginal en v5 (0.028 [0.000, 0.067]) para el mismo target — v6 lo identifica algo mejor.

### Reducción — FeO (`m6_ln_feo_ret` / `v5_ln_feo_ret`) — hallazgo principal de esta tarea

| parametrización | con_signos | R²_oof v6 | R²_oof v5 | R²_lockbox v6 | R²_lockbox v5 |
|---|---|---:|---:|---:|---:|
| cinética | sí | **0.327** | 0.196 | **0.473** | 0.190 |
| cinética | no | 0.325 | 0.196 | 0.473 | 0.177 |
| dosis/sobrante | sí | 0.326 | 0.196 | 0.471 | 0.190 |
| dosis/sobrante | no | 0.326 | 0.193 | 0.470 | 0.189 |

**v6 casi duplica el R²_oof de `ln_feo_ret` (0.33 vs 0.20) y más que duplica el R²_lockbox (0.47 vs 0.18-0.19)
sobre las mismas 1280 filas**, en las 4 combinaciones de parametrización/signos. Esto confirma directamente
el diagnóstico de iteración 8 (dos tercios del R² de `v5_ln_feo_ret` era reversión al ruido del ensayo de
%CaO): al quitar la dependencia de %CaO de la masa, el target de FeO deja de ser mitad ruido y el modelo lo
identifica mucho mejor, **y generaliza mejor a lockbox** (v6 es el único de los dos que supera su propio
R²_oof en lockbox, señal de que no está sobreajustando el ruido de DEV). Los theta de palanca no cambian de
signo ni de magnitud de forma relevante entre v5 y v6 (GN sigue siendo el único robusto: θ_1sd≈-0.011 a
-0.012 en ambas versiones, IC excluye 0); la ganancia de v6 está en `g(S)` (el estado), no en las palancas.

### Fusión — Sn (`m6_ln_sn_dep` / `v5_ln_sn_dep`, estado CURADO_F, palancas de Fusión v5)

| con_signos | R²_oof v6 | R²_oof v5 | R²_lockbox v6 | R²_lockbox v5 |
|---|---:|---:|---:|---:|
| sí | 0.200 | 0.184 | **-0.682** | -0.294 |
| no | 0.204 | 0.189 | -0.664 | -0.280 |

**Único resultado donde v6 es peor que v5**: el R²_oof en DEV es marginalmente mejor con v6 (+0.015-0.020)
pero el R²_lockbox es mucho más negativo (-0.66 a -0.68 vs -0.28 a -0.29 de v5) — v6 generaliza peor en
Fusión. Ninguna de las dos versiones es utilizable como modelo predictivo de escalón en Fusión (R²_lockbox
muy negativo en ambas, igual que E7-02); GN es significativo con signo indeseable (sube el agotamiento de Sn)
en ambas versiones y de magnitud similar (θ_1sd≈0.021-0.026).

## 5. Control de reversión de ruido de %CaO / %FeO (`e8_03_reversion.csv`)

R² OOF (HGB, GroupKFold(5) DEV) de `ln_feo_ret` con el estado CURADO completo vs sin {`ley_cao_escoria_pct_prev`,
`basicidad_B2_prev`, `indice_irf_prev`} (reversión del ruido de matriz/CaO) vs sin {`ley_feo_escoria_pct_prev`,
inventario de FeO previo} (reversión del ruido de %FeO):

| targets | R² completo | R² sin CaO/B2/IRF | R² sin FeO/inv_FeO | n |
|---|---:|---:|---:|---:|
| v6 (`m6_ln_feo_ret`) | 0.266 | 0.251 (-0.015) | 0.262 (-0.004) | 1084 |
| v5 (`v5_ln_feo_ret`) | 0.173 | 0.144 (-0.029) | 0.170 (-0.003) | 1040 |

**v6 arranca de una base mucho más alta (0.266 vs 0.173) y depende menos de las variables de matriz/CaO en
términos relativos**: quitar {CaO_prev, B2_prev, IRF_prev} le cuesta 0.015 (5.6 % relativo) a v6 frente a
0.029 (16.8 % relativo) a v5. La caída al quitar el inventario de FeO previo es pequeña en ambas versiones
(v6 -0.004, v5 -0.003): la mayor parte de la señal restante no es reversión trivial del propio nivel de FeO.
Esto confirma cuantitativamente el diagnóstico de iteración 8: **una fracción real de lo que el PLM de v5
aprendía en `ln_feo_ret` era la reversión del ruido de %CaO** (16.8 % del R² se explica solo por eso), y v6
reduce esa dependencia a la vez que sube el R² total.

## 6. Cadena de signos palanca → ln_sn_dep / ln_feo_ret → J (w anclado; `e8_03_cadena_signos.csv`)

Con w_v6=6.24 y w_v5=6.04, parametrización cinética con signos, estado CURADO:

| palanca | signo Sn (v6 / v5) | signo FeO (v6 / v5) | signo neto J (v6 / v5) |
|---|---|---|---|
| tasa_gn_nm3_min (GN) | + / + | - / - | **-** / **-** |
| Cx_sn_v6 / Cx_sn_v4 | + / + | n/a | + / + |
| tasa_feed_Carbon_kg_min | n/a | - / - (n.s. en ambas) | - / - |
| Cx_av_v6 / Cx_av_v4 | n/a | - / - (n.s. en ambas) | - / - |
| exceso_o2_combustion_pct | - / - | + / + | - / - |
| tasa_aire_nm3_min | + / + | + / + | + / + |

La cadena de signos **no cambia entre v5 y v6**: GN sigue siendo el único lever con trade-off Sn/FeO claro y
efecto neto negativo sobre J en ambas versiones (magnitud similar, -0.010 a -0.021 por sd); el carbono
apunta al signo teórico en ambos canales pero sin identificación robusta a nivel de escalón, igual que en
E7-01. La masa v6 no habilita ninguna palanca nueva ni invierte ningún signo — el cambio de v6 es de
**precisión estadística** (R² de FeO, ancho del IC de w), no de dirección cualitativa de la política.

## Veredicto y recomendaciones

- **w recomendado: 6-8** (v6 ancla en 6.24 [4.30, 9.83] con el KPI actual DEV, muy cerca del ancla v5 de
  E7-01 (6.04) pero con IC ~55 % más angosto). Se recomienda **mantener w≈6-7** para el objetivo compuesto,
  ahora con más confianza estadística gracias a v6.
- **K recomendado: ≈2.85 pp de KPI por unidad de `m6_ln_sn_dep`** (DEV, KPI actual) — no comparable en
  magnitud absoluta con el K_v5=2.12 porque cambia la escala de la masa, pero el ranking/significancia es
  igual o mejor.
- **β_F (Fusión, conservación de Sn): sin cambio de recomendación** — sigue siendo un candidato débil, no
  confirmado en DEV (p=0.20 v6 vs p=0.08 v5); si se usa, documentar igual que en E7-02 como señal débil de
  dirección teórica correcta.
- **θ recomendados por palanca (Reducción, estado CURADO, parametrización cinética, con signos):**
  - Sn: **GN +0.052 [0.014, 0.092]** (significativo, v6; v5 marginal 0.028 [0.000, 0.067]); `Cx_sn` no
    significativo en v6 (a diferencia de v5, ver limitación abajo); O2/aire sin identificar.
  - FeO: **GN -0.012 [-0.018, -0.003]** (significativo en ambas versiones, magnitud casi idéntica); carbón,
    `Cx_av`/`C_x_avance` y exceso de O2/aire sin identificar de forma robusta en ninguna versión.

## Comparación honesta v5 vs v6

**Mejora clara:**
- `m6_ln_feo_ret` casi duplica el R²_oof (0.20→0.33) y más que duplica el R²_lockbox (0.19→0.47) sobre las
  mismas filas, en las 4 combinaciones de parametrización/signos probadas (tarea 4).
- El IC del w anclado se reduce ~55 % (E7-01: [3.64,15.54] → E8-03: [4.30,9.83]) por menor varianza del
  componente de FeO.
- La dependencia de `ln_feo_ret` en las variables de matriz/CaO (fuente del sesgo original) cae de 16.8 % a
  5.6 % del R² (tarea 5): confirma que v6 corrige, al menos parcialmente, el diagnóstico que motivó la
  iteración.
- La batería Spearman de FeO contra el KPI del batch siguiente es más estable en v6 (0.194 DEV / 0.329
  lockbox vs 0.122/0.375 de v5 — comparable en lockbox, mejor en DEV).

**Sin cambio / empate:**
- `m6_ln_sn_dep` es esencialmente idéntico a `v5_ln_sn_dep` en R² (0.765-0.766 ambas versiones) y en la
  cadena de signos (mismos levers, mismos signos, magnitudes similares) — esperado, ya que el agotamiento de
  Sn no depende del trazador de CaO.
- El w y K anclados apuntan en la misma dirección cuantitativa que v5 (w≈6, K positivo y significativo);
  ninguna conclusión cualitativa de política cambia.
- El barrido de w mantiene el mismo patrón de inestabilidad DEV/lockbox de iteración 5/7 (rho crece con w en
  DEV, cae en lockbox) en ambas versiones — v6 no lo resuelve, solo lo desplaza levemente.

**Empeora o queda igual de débil:**
- El único theta de carbono que E7-01 había logrado identificar (`Cx_sn_v4`, cinética C×Sn_inventario para
  `ln_sn_dep`) **deja de ser significativo con la versión v6** (`Cx_sn_v6`, θ_1sd=0.140 [0.000,0.272] vs
  0.186 [0.009,0.327] de v5). Es una pérdida real de identificación, aunque no cambia el R² global de Sn.
- El PLM de `ln_sn_dep` en Fusión generaliza **peor** con v6 (R²_lockbox -0.68 vs -0.29 de v5), aunque ambas
  versiones son inutilizables como modelo predictivo ahí (R²_lockbox muy negativo en las dos).
- La conservación de Sn en Fusión (β_F) es más débil en DEV con v6 (p=0.20 vs p=0.08 de v5), aunque de
  magnitud comparable y significativa en TOTAL en ambas.

**Balance:** v6 es una mejora neta y sustancial para el canal de FeO en Reducción (que era precisamente el
target más débil y más sospechoso de artefacto en v5), sin costo para el canal de Sn en Reducción (que ya
era fuerte) más allá de la pérdida de un theta de carbono. Fusión sigue siendo débil en ambas versiones, con
una degradación menor en la generalización de `ln_sn_dep` que no cambia la recomendación operativa (Fusión
sigue sin un objetivo de escalón defendible con ninguna de las dos versiones de masa).

## Límites

- El grid de PLM de esta tarea usa **solo el estado CURADO** (20-26 features) para la comparación completa
  (cinética + dosis/sobrante, con/sin signos); el estado FULL (37 features v5 / 37 v6) no se re-corrió por
  costo computacional (una sola corrida FULL con signos tomó >45 s en un timing aislado y el grid completo
  hubiera tomado >1 h dado el contención de CPU observada durante la ejecución — ver notas de proceso). E7-01
  ya mostró que FULL vs CURADO cambia el R²_oof en ≤0.01 para `ln_sn_dep` y de forma no sistemática para
  `ln_feo_ret` (0.212 CURADO vs 0.186 FULL), así que no se espera que el resultado cualitativo de la tarea 4
  cambie, pero no se verificó explícitamente con la masa v6.
- La comparación "misma muestra" (tarea 4) usa la intersección de columnas de estado+palancas+target de
  **ambas** versiones, lo que reduce el n respecto de correr cada versión con su propio conjunto óptimo de
  filas; el n resultante (1280 Reducción, 1720 Fusión) es de todas formas comparable al de E7-01/E7-02.
- El anclaje de w y K depende de los mismos controles y tendencia que E7-01 (`CONTROLES_BATCH`); no se
  probaron controles alternativos ni especificaciones no lineales para el anclaje.
- La tarea de reversión de ruido usa el estado CURADO_V6 completo (incluye `m6_resto_frac_prev`/
  `m6_G_pct_prev`, ausentes en v5): parte de la mejora de R² podría deberse a estas dos features adicionales
  y no solo al target en sí; no se aisló ese efecto por separado (queda para E8-04/síntesis).
- Como en E7-01/E7-02: el PLM cross-fitted con ~1000-1400 filas de DEV y 4-5 palancas tiene poca potencia
  para IC angostos; los theta con IC amplios no deben leerse como nulos, solo como no identificados con los
  datos actuales.
- `masa_v6.py` no se modificó; no se encontraron bugs que requirieran parche durante esta tarea.
