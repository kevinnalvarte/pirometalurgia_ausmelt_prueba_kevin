# E9-04 — Fusión: reformulación (predictibilidad robusta a la campaña y objetivo defendible)

Script: `e9_04_fusion.py` (etapas `diagnostico|reentrenamiento|targets|estado_final|polvo|resumen`). Datos:
`v7_lib.cargar_df()` (df v6, 3982×281), `L.base_fase(df,'Fusión')` (2172 escalones, F1..F6), `L.construir_batch(df)`
(362 batches). DEV = 299 batches (abril-julio 2026), LOCKBOX = 63 últimos cronológicos (agosto 2026, fin de
campaña de refractario, horno ~160°C más frío en F6). OOF GroupKFold(5) por batch en DEV = criterio; lockbox y
walk-forward = reporte. Salidas: `e9_04_diag_*.csv`, `e9_04_reentrenamiento.csv`, `e9_04_targets.csv`,
`e9_04_valor_estado_final*.csv`, `e9_04_polvo_*.csv`, `e9_04_theta_{sn_dep,dT}.csv`, `e9_04_resumen.csv`.

## Veredicto en una frase

**H4 se confirma sólo a medias.** El fallo de lockbox de Fusión es en parte deriva de campaña (el reentrenamiento
progresivo y el walk-forward SÍ mejoran mucho sobre el modelo congelado, y para `dT` cruzan el umbral R²>0.1), pero
para `m6_ln_sn_dep` (conservación de Sn) **ningún esquema de reentrenamiento cruza R²>0.1 en lockbox** — el techo
de la deriva no es sólo de intercepto, es de escala/varianza del target. Sí se confirma con fuerza la segunda mitad
de H4: **el estado al cierre de Fusión (fila R0 de Reducción) predice el resultado de Reducción y el KPI**, y dentro
de ese estado el candidato más defendible para "conservar Sn en Fusión" no es el target de escalón (`Σ ln_sn_dep`,
p=0.17 DEV, iteración 6-8) sino el **nivel del inventario de Sn no reducido al cierre** (`m6_sn_inv_kg_prev`,
+8.7 pp de KPI/sd, p=0.038 DEV) — mismo signo teórico, mucho más identificado. El canal polvo no aporta un término
de objetivo nuevo: replica exactamente el hallazgo v4/v5 (sólo `T_media_F` sobre `f_polvo`, +0.008 pp/sd, p=0.027)
y el gas de lanza normalizado por tonelada de carga no tiene señal de batch, aunque sí hay una pista de escalón
(GN sube la temperatura de gas pre-BHF, un precursor plausible de arrastre de polvo).

---

## 1. Diagnóstico: reproducción y descomposición del fallo de lockbox

`e9_04_diag_metricas.csv`, `e9_04_diag_por_orden.csv`, `e9_04_diag_recalibracion.csv`, `e9_04_theta_{sn_dep,dT}.csv`.

Reproducción exacta de v6 (misma librería, mismo estado ESTADO_F_CURADO_V6, 25 features + `m6_resto_frac_prev`):

| target | R² OOF DEV | R² lockbox | n DEV | n lockbox |
|---|---|---|---|---|
| `m6_ln_sn_dep` | 0.161 | **−0.466** | 1469 | 313 |
| `d_temperatura_horno_celsius` (ΔT) | 0.497 | 0.037 | 1482 | 314 |

θ (1 sd, IC95%): GN +0.026 [0.006, 0.047]* sobre `sn_dep` (extrae Sn, coherente con v6: +0.030); relación C/Sn
+0.013 [0, 0.037] no significativo (en la cota); exceso O₂ y aire sin efecto identificado. Para ΔT: O₂ −1.16
[−2.45, −0.24]* (única palanca significativa; GN y aire colapsan en la cota) — replica "modelo térmico de Fusión
sin generalización" de la iteración 7.

**Hallazgo nuevo, no documentado antes**: el estado `ESTADO_F_CURADO_V6` incluye `grad_temperatura_horno_celsius_prev`
(gradiente de temperatura del escalón anterior), que es **estructuralmente NaN en el 100% de las filas de F1**
(el gradiente no está definido para el segundo escalón del batch, ya que requiere un escalón anterior *a* F0). Esto
significa que **F1 nunca se modela** con este estado (0 filas tras `dropna`), no por falta de datos sino por diseño
del set de features. F2-F6 sí se modelan (353-359 filas/orden en DEV). No se corrige aquí (es la arquitectura base
v5/v6, fuera del alcance de E9-04) pero se documenta como limitación a resolver en una reformulación futura
(imputar el gradiente de F1 con 0 o excluirlo del estado).

### R² y sesgo por orden de escalón (F2..F6), `sn_dep`

| conjunto | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|
| OOF DEV, R² | 0.072 | 0.035 | 0.105 | 0.029 | 0.041 |
| lockbox, R² | **−0.637** | −0.084 | −0.380 | −0.290 | **−1.174** |
| lockbox, sesgo (pred−real) | +0.072 | +0.040 | +0.102 | +0.050 | **+0.216** |
| lockbox, sd(real) | 0.092 | 0.139 | 0.148 | 0.150 | 0.225 |

Confirma `e7_09`: el sesgo positivo (el modelo sobre-predice agotamiento en lockbox) crece con el orden y es máximo
en F6 (+0.216, la mitad de la sd del target real en lockbox); el R² OOF DEV ya era bajo (0.03-0.11) en todos los
órdenes, no sólo al final — **la deriva de campaña empeora un modelo que ya era débil, no rompe uno bueno**.

### Recalibración de intercepto con los primeros 20 batches del lockbox ("ajuste mínimo de campaña")

Simulación: calibrar con los primeros 20 batches cronológicos del lockbox (n=100-101 filas), evaluar sobre los 43
restantes (n=213-214 filas).

| target | política | R² | MAE | pearson |
|---|---|---|---|---|
| sn_dep | sin recalibrar | −0.449 | 0.162 | 0.349 |
| sn_dep | intercepto global | −0.104 | 0.133 | 0.349 |
| sn_dep | **intercepto por orden** | **+0.012** | 0.127 | 0.310 |
| dT | sin recalibrar | −0.051 | 6.98 | 0.356 |
| dT | intercepto global | −0.046 | 6.96 | 0.356 |
| dT | intercepto por orden | −0.109 | 7.30 | 0.365 |

Para `sn_dep`, corregir sólo el sesgo (sin retocar θ ni g(S)) con 20 batches recupera **la mitad del R² perdido**
(de −0.45 a −0.10 con un intercepto único; a +0.01 con intercepto por orden) — el sesgo por orden documentado
arriba es una parte real y corregible del fallo. Para `dT` la recalibración de intercepto no ayuda (el fallo de
`dT` no es de sesgo sistemático sino de correlación baja, pearson 0.36-0.37 sin cambios).

---

## 2. Reentrenamiento: walk-forward sobre TODO y progresivo en el lockbox

`e9_04_reentrenamiento.csv` (362 batches, bloques de 30; ventana móvil 150; progresivo cada 10 dentro del lockbox).

| clave | política | R² agregado | n test | pearson |
|---|---|---|---|---|
| sn_dep | expandido (todo el historial) | −0.007 | 1634 | 0.291 |
| sn_dep | ventana móvil 150 | −0.045 | 1634 | 0.278 |
| sn_dep | progresivo cada 10, sólo lockbox | −0.146 | 313 | 0.285 |
| sn_dep | fijo DEV → lockbox (referencia v6) | **−0.466** | 313 | 0.338 |
| dT | expandido (todo el historial) | **+0.322** | 1646 | 0.586 |
| dT | ventana móvil 150 | **+0.312** | 1646 | 0.581 |
| dT | progresivo cada 10, sólo lockbox | **+0.157** | 314 | 0.433 |
| dT | fijo DEV → lockbox (referencia v6) | +0.037 | 314 | 0.386 |

**Respuesta a "¿recupera Fusión R²>0.1 en lockbox con reentrenamiento?": depende del target.**
- **`dT` sí**: el reentrenamiento (cualquier política) sube el R² de 0.037 (congelado) a 0.16-0.32. El walk-forward
  agregado sobre TODO el historial (que mezcla muchos bloques de régimen "normal", no sólo el lockbox de fin de
  campaña) ronda 0.31-0.32; incluso restringido al lockbox, el progresivo cruza el umbral (0.157). θ_GN oscila
  bloque a bloque (0 a 1.04, nunca negativo) — el signo nunca se invierte pero la magnitud no es estable, coherente
  con "modelo térmico de Fusión sin generalización" pero mejorable con reentrenamiento.
- **`sn_dep` no**: ninguna política cruza 0. El reentrenamiento SÍ ayuda muchísimo en términos relativos (de −0.47
  fijo a −0.01/−0.05 con walk-forward sobre todo el historial; de −0.47 a −0.15 progresivo dentro del lockbox) pero
  nunca llega a ser útil (R²≤0). θ_GN es positivo y estable en magnitud en todos los bloques (0.011-0.05, 1 sd) —
  la palanca identificada no cambia de signo con el reentrenamiento, sólo el ajuste global del modelo mejora sin
  volverse predictivo. **Conclusión: el fallo de `sn_dep` no es únicamente un problema de "modelo viejo", es un
  techo de señal/ruido bajo cualquier ventana de entrenamiento** (ver también sección 3, R² OOF nunca supera 0.26
  con ninguna de las 8 formulaciones de target).

---

## 3. Formas de target robustas a la deriva

`e9_04_targets.csv` (mismo estado ESTADO_F_CURADO_V6 + mismas 4 palancas C/Sn, GN, O₂, aire en todos los casos).

| id | descripción | R² OOF | R² lockbox | ρ(agregado batch, KPI) DEV | p | ρ lockbox | GN sig. | C/Sn sig. |
|---|---|---|---|---|---|---|---|---|
| a_ln_sn_dep | agotamiento log (ref. v6) | 0.161 | −0.466 | −0.037 | 0.53 | −0.141 | sí (+0.026) | no |
| b_frac_retenida | fracción retenida (nivel) | 0.167 | −0.294 | +0.037 | 0.53 | +0.141 | sí (−0.013) | no |
| c_sn_ext_norm | Sn extraído/Sn alimentado | 0.219 | −0.147 | −0.051 | 0.38 | −0.188 | no | no |
| d_dley_sn | Δ%Sn escoria (target v1) | 0.259 | −0.266 | −0.077 | 0.18 | −0.045 | sí (−0.44) | no |
| e1_lnirf_nivel | ln IRF al cierre (nivel) | **0.773** | **0.838** | +0.207 | <0.001 | +0.341 | no | no |
| e2_d_lnirf | Δ ln IRF del escalón | 0.278 | **0.079** | −0.321 | <0.001 | −0.470 | no | no |
| f_dley_feo | Δ%FeO escoria | **0.177** | **0.174** | −0.297 | <0.001 | −0.331 | no | **sí (+0.097)** |
| g_tasa_sn_dep | agotamiento/min (tasa) | 0.168 | −0.294 | −0.041 | 0.48 | −0.168 | sí (+0.0005) | no |

Lectura:
- **`e1_lnirf_nivel`** tiene el R² más alto con mucha diferencia (0.77/0.84) porque es casi puramente autorregresivo
  (el estado ya trae `indice_irf_prev`/`basicidad_B2_prev`, que casi determinan el nivel siguiente); no hay ninguna
  palanca significativa. **No es un target útil para optimizar el escalón** — replica el hallazgo de e7_02 de que
  la matriz que importa es heredada, no decidida en Fusión. Se descarta como target de escalón (aunque su ρ con el
  KPI, 0.21-0.34, es el más alto de la tabla — es señal de estado, no de acción).
- **`f_dley_feo` es el candidato más robusto a la deriva de los 8**: R² OOF 0.177 casi idéntico a lockbox 0.174
  (la brecha DEV−lockbox más pequeña de toda la tabla, incluida `e2_d_lnirf`), y **es la única formulación en la
  que la relación C/Sn cargado (`relacion_C_Sn_carga`) resulta significativa** (+0.097 [IC excluye 0]) en las 8
  evaluadas: más carbón relativo al Sn cargado en Fusión empuja el %FeO hacia arriba dentro del mismo escalón. La
  correlación con el KPI es fuerte y estable en las tres muestras (−0.28 a −0.33) pero de signo que exige controles
  para interpretarse causalmente (ver limitación abajo).
- **`e2_d_lnirf`** es el único, junto con `f_dley_feo`, con R² lockbox positivo aparte de `e1`; su correlación con
  el KPI es la más fuerte de las formas "de acción" (−0.32 DEV, −0.47 lockbox), coherente con h6/e7-02 en dirección
  pero en tensión con el hallazgo previo de que "Δ ln IRF no aporta controlando el heel" — aquí no se controla el
  heel (`F_lnirf_F0`), así que la correlación cruda puede estar inflada por él.
- **La forma en tasa (`g_tasa_sn_dep`) no ayuda**: mismo patrón que `a` (no es la escala temporal el problema).
- **Ningún target recupera la identificación del carbón de Fusión como palanca dominante**: sólo `f_dley_feo` la
  identifica, y con un signo que necesita controles (sección de límites) antes de usarse en un objetivo.

**Límite metodológico**: las correlaciones de esta tabla son Spearman/OLS crudos (vía `bateria_batch`, con los
controles estándar de `CONTROLES_BATCH` pero sin el heel `F_lnirf_F0` de e7-02); no reemplazan el anclaje causal
con heel + tendencia de `e7_02`/`e8_03`. Se reportan como evidencia de primer orden sobre qué targets vale la pena
re-anclar con ese protocolo completo en una siguiente iteración.

---

## 4. Valor del estado al final de Fusión (fila R0 de Reducción)

`e9_04_valor_estado_final*.csv`. Predictores: `ESTADO_FINAL_F` (12 variables `_prev` de la fila R0: leyes de Sn/FeO,
B2, IRF, T, masa/inventarios m6, avance, resto_frac, carbón acumulado y C/Sn acumulado) + `CONTROLES_BATCH`.
**Nota técnica**: `m6_resto_frac_prev = 1 − 1.270·(ley_sn/100) − (ley_feo/100)` (fórmula de cierre físico v6) es
una combinación lineal EXACTA de las otras dos leyes (VIF≈1e15 si se incluyen las tres en la misma OLS) — se
excluyó de la regresión OLS (queda en HGB/importancia por permutación, donde no rompe el ajuste).

### R² (OLS en muestra vs. HGB fuera de muestra)

| dependiente | R²adj OLS DEV | R² OOF HGB | R² lockbox HGB |
|---|---|---|---|
| `R_sum_m6_ln_feo_ret` | 0.171 | 0.066 | 0.006 |
| `R_sum_m6_ln_sn_dep` | 0.211 | 0.119 | 0.121 |
| `J_R_batch` | 0.190 | 0.025 | 0.070 |
| `recuperacion_refinada_pct` (KPI) | 0.078 | **−0.057** | 0.128 |

El estado de cierre de Fusión sí tiene señal sobre Reducción y el KPI (R²adj OLS 0.08-0.21, honesto: es ajuste en
muestra, no OOF); la versión HGB fuera de muestra es más modesta y hasta negativa en OOF-DEV para el KPI (18
regresores con 236 filas de entrenamiento por fold — al límite de la potencia), aunque positiva en lockbox
(0.128) — no hay evidencia de que HGB mejore a la OLS lineal aquí, es más bien lo opuesto.

### Gradiente V(S_F): pp de KPI por sd de estado (OLS HC3 estandarizada, DEV, `e9_04_valor_estado_final.csv`)

| variable | pp KPI/sd | p | ¿movible por palanca de Fusión? |
|---|---|---|---|
| **`m6_sn_inv_kg_prev`** (Sn no reducido al cierre) | **+8.72** | **0.038** | sí — vía agotamiento (`sn_dep`), pero ese target R² OOF 0.16/lockbox −0.47 (sec. 1-3): se sabe QUÉ mover, no se puede predecir bien CUÁNTO mueve una acción dada |
| `basicidad_B2_prev` | +0.58 | 0.055 | no directo — vía cal (PLAN), no palanca de escalón |
| `m6_masa_kg_prev` | −6.11 | 0.129 | parcialmente — covaría con feed_total (PLAN) |
| `m6_feo_inv_kg_prev` | +3.75 | 0.282 | vía retención de FeO (no modelada en Fusión, sólo Reducción) |
| `temperatura_horno_celsius_prev` | +0.41 | 0.349 | sí — vía ΔT, pero ΔT casi sin θ significativo (sec. 1) |
| `cum_feed_Carbon_kg_prev` | +0.80 | 0.426 | sí — directamente vía carbón acumulado |
| `m6_avance_prev` | +2.88 | 0.534 | sí — resultado del agotamiento acumulado |
| `ley_sn_escoria_pct_prev` | −2.01 | 0.542 | sí — mismo canal que `m6_sn_inv_kg_prev` |
| `relacion_C_cum_Sn_cum_prev` | −0.21 | 0.873 | sí — directamente vía carbón acumulado |
| `ley_feo_escoria_pct_prev` | −0.47 | 0.838 | no directo (PLAN/matriz) |
| `indice_irf_prev` | +0.01 | 0.995 | no directo — vía cal (PLAN) |

**Hallazgo central de esta sección**: `m6_sn_inv_kg_prev` (cuánto Sn queda sin reducir en la escoria al cerrar
Fusión, es decir, cuánto se CONSERVÓ) es el único predictor significativo del KPI en esta regresión (p=0.038,
n=295), con el signo que la teoría de "fundir oxidando, reducir después" predice (más Sn conservado → mejor KPI
final). Esto **confirma y ancla con más fuerza la conservación de Sn como objetivo de Fusión** que el término de
escalón `Σ F_sum_m6_ln_sn_dep` de v6/v8 (p=0.17 DEV, candidate_win_model_v6.md): medido como NIVEL final en vez de
como SUMA de log-cambios, el efecto es significativo. En kg: +8.72 pp/sd × 512 kg/pp ≈ **4 465 kg de Sn a metal
equivalente por sd de `m6_sn_inv_kg_prev`** (sd ≈ ver `e9_04_valor_estado_final_ols.csv`/estado del batch), aunque
la dirección causal es ambigua: podría ser conservación real (Fusión suave) o simplemente escala de carga
(batches con más concentrado tienen más Sn_inv y más Sn total, aunque el KPI ya es una fracción y los controles
incluyen `feed_sn_total_t`).

Para el par Reducción, `indice_irf_prev` (heredado, IRF al cierre de Fusión = IRF de entrada a Reducción) es el
predictor más fuerte de `R_sum_m6_ln_feo_ret` (+0.062, p=0.0002) y de `J_R_batch` (+0.92, p=0.0003) — confirma
iteración 6-8: la matriz que importa para la Reducción se decide con la cal (PLAN), no con las 4 palancas de
Fusión. `ley_sn_escoria_pct_prev` es el predictor más fuerte de `R_sum_m6_ln_sn_dep` (+0.89, p=0.004, mecánico:
más %Sn de entrada → más log-agotamiento posible en Reducción). `cum_feed_Carbon_kg_prev` tiene efecto negativo
significativo sobre `R_sum_m6_ln_feo_ret` (−0.072, p=0.014): el carbón ya alimentado durante Fusión (antes de
Reducción) reduce la capacidad de retener FeO en Reducción — primera identificación cuantitativa de un "carry-over"
del carbón de Fusión sobre el resultado de Reducción.

**Importancia por permutación** (HGB, hold-out 75/25): para el KPI, `ley_feo_escoria_pct_prev` (0.078),
`basicidad_B2_prev` (0.056) y `feed_sn_total_t` (0.051, control) dominan — consistente con que la matriz de cierre
de Fusión (gobernada por la cal, PLAN) es más importante que los inventarios de masa. Para `J_R_batch`,
`basicidad_B2_prev` (0.063) y `cum_feed_Carbon_kg_prev` (0.050) dominan.

---

## 5. Canal polvo

`e9_04_polvo_batch.csv`, `e9_04_polvo_escalon.csv`.

### Batch: gas de lanza de Fusión por tonelada de carga y T media, OLS HC3 con controles (DEV, n=299)

| KPI | variable | ρ Spearman | OLS pp(o frac)/sd | p |
|---|---|---|---|---|
| `f_polvo` | GN total/t | −0.005 | −0.0019 | 0.30 |
| `f_polvo` | O₂ total/t | +0.013 | −0.0010 | 0.60 |
| `f_polvo` | aire total/t | +0.050 | −0.0048 | 0.15 |
| `f_polvo` | gas total (GN+O₂+aire)/t | +0.032 | −0.0029 | 0.22 |
| `f_polvo` | **T media Fusión** | +0.159 | **+0.0081** | **0.027** |
| KPI refinado | cualquiera de los 5 | ≤0.06 | n.s. | ≥0.28 |

**Ningún gas de lanza normalizado por tonelada de carga tiene señal de batch** sobre `f_polvo` ni sobre el KPI (p
≥0.15 en todos los casos). El único término que replica es **T media de Fusión → `f_polvo` +0.0081 pp/sd (p=0.027)**,
prácticamente idéntico al ya reportado en v4/e7_02 (+0.0083, p=0.027) y a BETA_KPI_FUSION de v4 (+0.0129) — no hay
nada nuevo aquí, sólo confirmación con la normalización por tonelada en vez de por escalón.

### Escalón: PLM de proxies de polvo (sin restricción de signo) sobre estado + palancas de Fusión

| proxy | R² OOF | R² lockbox | GN (1sd) | O₂ (1sd) | aire (1sd) | carbón (1sd) |
|---|---|---|---|---|---|---|
| `d_temperatura_gas_pre_bhf_celsius` | 0.407 | 0.577 | **+1.53 [0.41, 2.98]*** | −0.67 [−1.87, 0.12] | +0.95 [−0.67, 2.33] | +0.30 [0, 0.66] |
| `d_tiro_horno_pct` | 0.508 | 0.665 | +0.23 [−0.26, 0.58] | −0.02 [−0.83, 0.68] | **+2.06 [1.08, 2.99]*** | −0.02 [−0.42, 0.42] |

Ambos proxies (temperatura de gas antes del baghouse, tiro del horno) tienen R² sorprendentemente altos y estables
DEV→lockbox (mejor que cualquier target de Sn/FeO de Fusión) y **sí identifican palancas con signo plausible**: el
GN sube la temperatura del gas pre-BHF (+1.5°C/sd, IC excluye 0) y el aire sube el tiro (+2.1 pp/sd, IC excluye 0)
— ambos son precursores físicamente razonables de arrastre de finos (gas más caliente/más caudal de aire arrastra
más polvo), pero **no hay una medición directa de polvo por escalón** para cerrar la cadena palanca→proxy→polvo
real; el batch-level (arriba) no detecta un efecto neto de esos mismos gases sobre `f_polvo`.

**Veredicto del canal polvo**: no hay un término de objetivo de Fusión nuevo o mejor defendible que el ya
existente en v4/v5 (T media de Fusión, +0.008-0.013 pp de KPI/sd sobre `f_polvo`, vía escalón repartido). El hallazgo
de escalón (GN↑ temperatura de gas, aire↑ tiro) es sugerente pero no se traduce en una señal de batch sobre polvo
real ni sobre el KPI — no se recomienda añadirlo como término cuantificado, sólo como hallazgo de vigilancia.

---

## 6. Resumen y veredicto integrador

`e9_04_resumen.csv` (28 filas, todas las métricas anteriores en un solo lugar).

### Lo que SÍ se sostiene con estos datos

1. **El fallo de lockbox de `sn_dep` es en parte (no del todo) deriva de campaña corregible**: recalibrar sólo el
   intercepto por orden con 20 batches recupera la mitad del R² perdido (−0.45→+0.01); el walk-forward sobre todo
   el historial también mejora mucho la referencia congelada (−0.47→−0.01/−0.05). Pero **nunca cruza R²>0.1**: el
   régimen de fin de campaña no es sólo un sesgo de intercepto, la correlación real-predicho de fondo (pearson
   ~0.28-0.35 en todas las variantes) es demasiado baja para un target útil de prescripción.
2. **`dT` sí recupera R²>0.1 con reentrenamiento** (0.16 progresivo dentro del lockbox, 0.31-0.32 walk-forward
   sobre todo el historial, contra 0.04 congelado) — de los dos targets de Fusión, es el que más se beneficia de
   una política de reentrenamiento por campaña, aunque su θ de palancas sigue sin ser estable bloque a bloque.
3. **El estado al cierre de Fusión predice Reducción y el KPI** (R²adj OLS 0.08-0.21 DEV): confirma H4. El canal
   más fuerte y mejor identificado es matriz heredada (`indice_irf_prev`, `basicidad_B2_prev` → `R_sum_m6_ln_feo_ret`
   y `J_R_batch`, p<0.001-0.03) y leyes heredadas (`ley_sn_escoria_pct_prev` → `R_sum_m6_ln_sn_dep`, p=0.004) — casi
   todos gobernados por el PLAN (mezcla de carga, cal), no por las 4 palancas de control de Fusión.
4. **La conservación de Sn en Fusión tiene un anclaje más fuerte como NIVEL de estado final que como SUMA de
   escalón**: `m6_sn_inv_kg_prev` (Sn no reducido al cerrar F6) +8.7 pp de KPI/sd, p=0.038 DEV — mismo signo
   teórico que v6/v8 pero con significancia que el target de escalón (p=0.17) no alcanzaba.
5. **`f_dley_feo` (Δ%FeO por escalón) es la formulación de target más robusta a la deriva** de las 8 evaluadas
   (R² OOF≈lockbox≈0.17-0.18) y la única que identifica la relación C/Sn cargado como palanca significativa
   (+0.097, IC excluye 0) — candidato a reemplazar o complementar `m6_ln_sn_dep` en una reformulación de Fusión,
   sujeto a validar su relación con el KPI con el protocolo completo de controles + heel de e7_02/e8_03 (aquí sólo
   se reporta la correlación cruda, fuertemente negativa y estable: −0.28 a −0.33 en las tres muestras).
6. **El canal polvo no tiene una señal de objetivo nueva**: se reconfirma exactamente el hallazgo v4/v5 (T media
   de Fusión → `f_polvo`, +0.008-0.013 pp/sd) y nada de gas normalizado por tonelada aporta a nivel de batch.

### Lo que NO se puede sostener con estos datos

- No se puede predecir el agotamiento de Sn de Fusión (`m6_ln_sn_dep`) fuera de muestra con ninguna combinación de
  target/forma/política de reentrenamiento probada aquí (R² lockbox ≤0 siempre, 10 variantes en total entre
  secciones 2 y 3). El objetivo de "conservar Sn en Fusión" sigue sin una palanca de escalón identificable y
  predecible con confianza — sólo el ESTADO resultante (sección 4) tiene una relación estadística defendible con
  el KPI, no la ACCIÓN que lo produce.
- No se puede separar, con la evidencia disponible, si `m6_sn_inv_kg_prev` → KPI es causal (conservación real) o
  reflejo de escala de carga/campaña; los controles de batch estándar no bastan para descartarlo (ver limitación
  de la sección 4).
- No se puede cerrar la cadena palanca→proxy de escalón→polvo real→KPI: hay identificación de escalón (GN, aire)
  pero ninguna medición directa de polvo por escalón para validar que esos proxies efectivamente predicen más
  polvo, y el batch-level no detecta el efecto neto de esos mismos gases.
- El hallazgo de F1 nunca modelado (gradiente de T indefinido) no se resuelve aquí; cualquier reformulación futura
  de Fusión debe decidir explícitamente qué hacer con ese escalón (imputar, excluir el gradiente del estado, o
  aceptar que Fusión se modela y prescribe sólo para F2-F6).

### Propuesta de reformulación (para decidir en la síntesis v7, no adoptada aquí)

1. Mantener `m6_ln_sn_dep` como target de escalón para θ/palancas (sigue siendo el más consistente con la teoría:
   GN identificado, signo estable) pero **dejar de usar su R² de escalón como criterio de si Fusión "funciona"**:
   el objetivo defendible de Fusión es el estado final (`m6_sn_inv_kg_prev`, `ley_sn_escoria_pct_prev`), no la
   predicción del cambio marginal.
2. Añadir reentrenamiento progresivo (cada 10 batches) para `dT` en producción (recupera R²>0.1); no hacerlo para
   `sn_dep` no cambiaría la conclusión de "no accionable con confianza fuera de DEV".
3. Evaluar `f_dley_feo` con el protocolo completo de anclaje (heel + controles + tendencia, como e7_02/e8_03) antes
   de adoptarlo — aquí sólo pasó el filtro de robustez de forma, no el de causalidad.
4. El término de polvo de v4/v5 (T media de Fusión) se mantiene sin cambios; no se encontró nada mejor.
