# E7-02 — Fusión: qué objetivo por escalón es defendible con el KPI refinado

Script: `e7_02_fusion.py`. Datos: `v5_lib.cargar_df()` + `v5_lib.construir_batch_v5(df)` + `19_batch_dataset.csv`
(decisiones de Fusión). DEV = 299 batches, LOCKBOX = 63 (últimos cronológicos), TOTAL = 362 (menos NaN por
modelo). Salidas: `e7_02_batch_regresiones.csv` (465 filas), `e7_02_matriz_fusion.csv` (18), `e7_02_vertices.csv`
(0 filas — ningún cuadrático significativo), `e7_02_plm_fusion.csv` (13), `e7_02_theta_fusion.csv` (49),
`e7_02_cadena_signos.csv` (28).

## Veredicto en una frase

**Re-anclado en el KPI refinado, Fusión sigue sin tener un objetivo de escalón robusto y accionable.** Los
anclajes de v4 (beta_C, beta_T) replican sobre los canales `f_dross`/`f_polvo` pero se disuelven sobre el KPI
compuesto; el único determinante fuerte y estable es el estado heredado (heel de escoria del batch anterior,
`F_lnirf_F0`), no una decisión del escalón. El único candidato de palanca con signo teórico correcto
(conservar Sn en Fusión) es débil y no pasa el criterio de selección en DEV solo.

## 1. Regresiones de batch re-ancladas (paso 2)

M0 = una decisión a la vez + controles; M1 = las 7 conjuntas + controles; M2 = M1 + cuadráticos centrados de
`C_por_Sn_F` y `T_media_F`; M3 = M1 + `F_lnirf_F0`. Todas OLS HC3, variables estandarizadas con media/sd de
la muestra TOTAL (convención exp 19).

### KPI refinado (`recuperacion_refinada_pct`), M1 conjunta, DEV (n=298)

| variable | coef (pp/sd) | IC95% | p | kg Sn/escalon |
|---|---|---|---|---|
| C_por_Sn_F | +0.198 | [-0.63, 1.03] | 0.64 | +14 [-46, +75] |
| gn_por_t_carga_F | +0.003 | [-0.62, 0.63] | 0.99 | +0.3 [-45, +46] |
| exceso_o2_F | -0.189 | [-0.84, 0.46] | 0.57 | -14 [-62, +34] |
| T_media_F | -0.264 | [-1.25, 0.73] | 0.60 | -19 [-92, +53] |
| **cal_por_carga_F** (PLAN) | **-0.866** | **[-1.68, -0.05]** | **0.038** | **-63 [-123, -4]** |
| lanza_media_F (no palanca) | +0.499 | [-0.18, 1.17] | 0.15 | +37 [-13, +86] |
| feed_rate_F | -1.067 | [-2.17, 0.04] | 0.058 | -78 [-159, +3] |

**Ninguna de las 4 palancas del prescriptor (C/Sn, GN, O2, T) es significativa, ni sola (M0) ni conjunta (M1),
en DEV ni en TOTAL**, sobre el KPI refinado. Único término significativo en M1: `cal_por_carga_F`, que es PLAN
(cal por tolva), no palanca. En M3 (+`F_lnirf_F0`) el patrón se repite: nada de las 7 decisiones es
significativo; `F_lnirf_F0` domina con +1.68 pp/sd [1.19, 2.17], p=1.8e-11.

### Los anclajes v4 SI replican sobre los canales, no sobre el KPI compuesto (M0, DEV)

| kpi | variable | coef/sd | IC95% | p | kg Sn/escalon |
|---|---|---|---|---|---|
| f_dross | C_por_Sn_F | -0.0081 | [-0.0142, -0.0020] | 0.0097 | -59 [-104, -14] |
| f_polvo | T_media_F | +0.0083 | [0.0010, 0.0156] | 0.027 | +61 [+7, +114] |
| f_dross | T_media_F | -0.0078 | [-0.0152, -0.0005] | 0.037 | -57 [-111, -4] |
| f_polvo | C_por_Sn_F | +0.0063 | [0.0011, 0.0115] | 0.017 | +46 [+8, +84] |

Coeficientes casi idénticos a `BETA_KPI_FUSION` de v4 (exp 19: -0.0087 f_dross, +0.0129 f_polvo) -- la
regresión de sub-canal replica bien. Pero **más C/Sn baja `f_dross` (bueno) y sube `f_polvo` (malo) en
magnitud comparable, y más T baja `f_dross` pero sube `f_polvo`**: los canales se compensan y el efecto neto
sobre `recuperacion_refinada_pct` (que ya descuenta dross+polvo con sus lambda) queda dentro del ruido (M0/M1
arriba). Esto es el hallazgo central de este experimento: **v4 optimizaba una señal de sub-canal que no se
traduce, con este tamaño de muestra, en una señal sobre el KPI compuesto**.

### Términos cuadráticos (M2): sin evidencia de óptimo interior

18 combinaciones (5 KPIs x 2 variables x ~2 muestras relevantes) -- **0 significativas** (todas con IC que
cruza cero; ver `e7_02_batch_regresiones.csv`, filas `_sq`). `e7_02_vertices.csv` queda vacío porque no hay
ningún cuadrático significativo del cual calcular vértice. **No hay evidencia de óptimo interior en C/Sn ni
en T media a nivel de batch.**

## 2. ¿Aporta el cambio de la matriz durante la Fusión? (paso 3)

OLS HC3 de `recuperacion_refinada_pct` sobre `F_lnirf_F0` (heel, cierre de F0) + un segundo término,
controles estándar:

| spec | variable | DEV coef/sd [IC] p | TOTAL coef/sd [IC] p | LOCKBOX coef/sd [IC] p |
|---|---|---|---|---|
| F0 + delta IRF acumulado | **F_lnirf_F0** | +1.49 [0.41,2.56] 0.007 | +1.59 [0.67,2.52] 0.0008 | +3.64 [1.11,6.18] 0.005 |
| | F_sum_d_lnirf | -0.134 [-1.29,1.02] 0.82 | -0.117 [-1.08,0.85] 0.81 | +0.271 [-1.77,2.31] 0.79 |
| F0 + IRF final | **F_lnirf_F0** | +1.68 [0.94,2.42] 7.7e-6 | +1.74 [1.12,2.36] 4.4e-8 | +3.18 [1.80,4.57] 6.7e-6 |
| | F_lnirf_fin | -0.119 [-0.99,0.75] 0.79 | -0.066 [-0.79,0.66] 0.86 | +0.391 [-1.11,1.89] 0.61 |
| F0 + agotar Sn en F | **F_lnirf_F0** | +1.66 [1.24,2.08] 1.2e-14 | +1.75 [1.37,2.14] 5.5e-19 | +3.11 [1.75,4.47] 7.4e-6 |
| | **F_sum_ln_sn_dep** | -0.454 [-0.97,0.06] **0.083** | **-0.589 [-1.06,-0.12] 0.014** | -0.947 [-2.33,0.44] 0.18 |

Conclusiones:
- **`F_lnirf_F0` (heel, estado, no palanca) es el predictor más fuerte y robusto de todo el experimento**:
  significativo al 1% en las 3 muestras y las 3 especificaciones (+1.5 a +3.6 pp KPI/sd). Confirma iteración
  5/6: lo que importa de la matriz es la condición de partida heredada del batch anterior, no lo que se hace
  con ella dentro de la Fusión.
- `F_sum_d_lnirf` y `F_lnirf_fin` **no aportan nada** una vez controlado F0 (coef pequeño, signo inestable
  entre muestras, p > 0.6 siempre): confirma que el cambio de IRF observado es regresión a la media del
  estado inicial, no señal explotable.
- `F_sum_ln_sn_dep` (conservar Sn durante la Fusión) tiene **signo teórico correcto y estable en las 3
  muestras** (más agotamiento de Sn en Fusión -> peor KPI) pero **no cruza el umbral de significancia en DEV
  solo** (p=0.083, que es el criterio de selección de esta metodología); sí es significativo en TOTAL
  (p=0.014, n=348) y tiene la magnitud correcta en lockbox aunque sin significancia (n=63). Se reporta como
  **candidato débil, no confirmado**.

## 3. PLM por escalón en Fusión (paso 4)

Estado `v5.ESTADO_F_CURADO` (incluye plan de carga); 3 juegos de palancas: (a) primitivas, (b) derivadas,
(c) intensidad térmica; para ΔT también (c)+`b6_q_neto_por_min_MJ`+`b6_dT_teorico_sin_reaccion`.

| target | palancas | R2oof DEV | R2lockbox | n_dev | Lectura |
|---|---|---|---|---|---|
| v5_d_lnirf (delta IRF) | a/b/c | 0.28-0.31 | 0.04-0.08 | ~1440 | dominado por g(S) (heel), palancas casi sin aporte |
| v5_ln_sn_dep | a/b/c | 0.18-0.19 | **-0.37 a -0.24** | ~1407 | no generaliza (peor que la media fuera de DEV) |
| sn_extraido_est_kg | a/b/c | 0.68-0.69 | 0.62-0.64 | ~1430 | predecible (masa), sí generaliza |
| d_temperatura_horno_celsius | a/b/c/c+b6 | 0.48-0.50 | **0.045-0.08** | ~1450 | ajusta bien en DEV, colapsa fuera |

theta significativos (IC excluye 0) más relevantes (`e7_02_theta_fusion.csv`):

| target | palanca (set) | theta (1 sd) [IC] | signo teórico | concuerda |
|---|---|---|---|---|
| v5_d_lnirf | tasa_feed_Carbon_kg_min (a) | +0.0043 [0.0001, 0.0094] | -1 | **NO** (magnitud minúscula) |
| v5_ln_sn_dep | tasa_o2_nm3_min (a) | +0.051 [0.008, 0.094] | 0 | -- |
| v5_ln_sn_dep | tasa_gn_nm3_min (b) | +0.053 [0.032, 0.075] | +1 | sí |
| v5_ln_sn_dep | exceso_o2_combustion_pct (b) | +0.027 [0.006, 0.050] | -1 | **NO** |
| v5_ln_sn_dep | v5_gn_esp_nm3_t (c) | +0.079 [0.044, 0.183] | +1 | sí |
| sn_extraido_est_kg | tasa_o2_nm3_min (a) | +276 kg [67, 476] | 0 | -- |
| sn_extraido_est_kg | tasa_gn_nm3_min (b) | +268 kg [106, 437] | +1 | sí |
| sn_extraido_est_kg | v5_gn_esp_nm3_t (c) | +545 kg [241, 1044] | 0 | -- |
| d_temperatura | tasa_gn_nm3_min (b) | **-1.88 C [-3.11, -0.82]** | +1 | **NO** |
| d_temperatura | exceso_o2_combustion_pct (b) | -1.09 C [-1.95, -0.35] | 0 | -- |
| d_temperatura | tasa_aire_nm3_min (b) | +1.69 C [0.14, 3.08] | -1 | **NO** |
| d_temperatura | b6_dT_teorico_sin_reaccion (c+b6) | -3.09 C/sd [-6.78, -1.22] | 0 | -- |

Lectura: GN y O2 **elevan** la extracción de Sn de la escoria durante la Fusión (`v5_ln_sn_dep`,
`sn_extraido_est_kg`), coherente con volatilización/fuming a más intensidad térmica-oxidante -- esto es
indeseable (el Sn que sale de la escoria en Fusión no va limpiamente a metal). Pero el modelo de
`v5_ln_sn_dep` no generaliza (R2lockbox negativo): **no se puede confiar en esta cuantificación fuera de
DEV**. El modelo de delta-T ajusta bien en DEV pero **colapsa en lockbox y sus theta significativos tienen
signo contrario a la teoría** (GN enfría, aire calienta) -- es el mismo patrón de inestabilidad temporal ya
documentado en "Lecciones iteración 4" (theta de carbón/GN cambian de signo entre períodos); no se recomienda
usarlo como lever de temperatura. `v5_d_lnirf` confirma iteración 5: prácticamente ninguna palanca la mueve
de forma fiable dentro del escalón.

## 4. Cadena de signos (paso 5)

Las **28 combinaciones palanca x target quedan "indeterminadas"** (`e7_02_cadena_signos.csv`) -- no por un
error del script, sino porque en el paso 2 (M1, DEV) **ninguna de las 3 decisiones mapeables (C/Sn, GN, O2)
resultó individualmente significativa sobre el KPI refinado**, así que no existe un ancla batch-level válida
con la cual comparar el signo del theta de escalón. Es en sí mismo un resultado: **la cadena palanca -> target
de escalón -> KPI de batch no se puede cerrar con la evidencia actual para ninguna palanca de Fusión.**

## 5. Propuesta de J_F v5 -- honesta

Con el KPI refinado, **no hay términos de Fusión que cumplan el estándar de v4** (significativos en DEV,
kg de Sn con IC que excluye 0). Se proponen dos niveles:

**Términos NO defendibles (se retiran de J_F o se degradan a "orientativos"):**
- beta_C · z(C/Sn cargado) -- n.s. sobre el KPI refinado en M0/M1/M3 (todas las muestras); solo significativo
  sobre el sub-canal `f_dross`, que no domina el KPI compuesto.
- beta_T · z(T media) -- mismo caso, solo significativo sobre `f_polvo`.
- Cualquier término cuadrático de C/Sn o T (0/18 significativos: sin evidencia de óptimo interior).
- Delta ln IRF de Fusión (cualquier variante): no aporta controlando el heel F0, y ninguna palanca la mueve
  de forma fiable en el escalón (theta≈0, R2lockbox bajo).
- Posición de lanza: marginal solo en lockbox/borderline en total, inestable entre muestras -- además no es
  palanca por convención del proyecto (sentido físico sin confirmar).

**Único candidato con algo de apoyo (marcar explícitamente como NO confirmado):**
- **-beta_S · (agotamiento de Sn en el escalón de Fusión, `v5_ln_sn_dep` residualizado por el estado)**,
  ancla: beta_S(batch) = 0.454 pp KPI/sd [-0.06, 0.97] p=0.083 en DEV (n=285); 0.589 [0.12, 1.06] p=0.014 en
  TOTAL (n=348); ~33-43 kg Sn/escalón de magnitud, pero **la IC en DEV cruza cero** (criterio de selección de
  esta metodología no se cumple). El signo es el correcto según la teoría (lambda_F, "fundir oxidando,
  reducir después") y es consistente en las 3 muestras, pero no alcanza el estándar de confianza. Si se
  decide incluirlo en v5 pese a esto, debe documentarse como "señal débil, dirección teórica correcta, no
  significativa en DEV solo".

**Recomendación operativa:** en Fusión, salvo la restricción de soporte/ventana térmica ya vigente en v4
(sin evidencia para reemplazarla), no hay una función objetivo de escalón que se pueda anclar con confianza
al KPI refinado. La palanca de mayor apalancamiento observable sobre el KPI de batch (`F_lnirf_F0`, heel de
escoria) **no es una decisión de este escalón**: es una consecuencia de cómo terminó el batch anterior
(cuánta escoria/heel quedó, con qué IRF). Esto sugiere que la optimización de mayor impacto en Fusión podría
estar en el CIERRE del batch anterior (o en el manejo del heel), no en las palancas C/GN/O2/T del escalón
actual -- una hipótesis para investigar en E7-03/E7-04, fuera del alcance de E7-02.

## Límites

- Todas las regresiones de batch tienen R2 marginal pequeño (delta R2adj de las 7 decisiones sobre
  controles ~0.01-0.02 en M1): el ruido batch-level domina, n=298 DEV es la principal restricción de potencia.
- El PLM de `v5_ln_sn_dep` y de `d_temperatura_horno_celsius` no generaliza a lockbox (R2 negativo o casi
  nulo): los theta reportados para esas palancas no deben usarse para prescribir sin más validación
  (E7-04, estabilidad temporal).
- Colinealidad esperada entre GN/O2/aire (todos co-varían con el estado térmico y el plan de carga) puede
  estar generando signos theta contrarios a la teoría (GN enfriando, aire calentando) que son artefactos de
  identificación, no efectos causales reales -- coherente con "cal Fusión artefacto" y "theta cambia de
  signo entre períodos" de iteración 4.
- El heel de escoria (`F_lnirf_F0`) es, con mucho, la variable más predictiva del KPI de Fusión, pero al ser
  estado heredado del batch anterior queda fuera del alcance de "palanca del escalón" que pide esta
  iteración; no se convierte en término de J_F pero se señala como hallazgo de mayor prioridad.
