# E8-02 — Balance de energía por escalón con la masa de escoria v6 (2026-09-16)

Script: `experimentos/v6/e8_02_energia_v6.py`. Datos: `masa_v6.cargar_df_base()` → `mp4.agregar_columnas_v4`
→ `mp5.agregar_columnas_v5` → `masa_v6.agregar_masa_v6(df, None, bal)` (parámetros por defecto de
`masa_v6.PARAMS_DEFAULT`: Fusión trazador G, Reducción cierre físico). Split DEV/lockbox
`modelo_prescriptivo.split_dev_lockbox` (lockbox = últimos 17.5% cronológico). Modelos: PLM v5
(`modelo_predictivo_v5.ModeloPLMSignos`), validación OOF `GroupKFold(5)` por Batch en DEV, lockbox solo
reporte, igual que `tabla_validacion_v5`. `masa_v6.py` es un artefacto compartido con E8-01 (calibración
de la masa v6); los números de este memo usan la versión de `masa_v6.py` vigente al momento de correr el
script (2026-09-16, madrugada) y no fueron editados por este experimento.

## 0. Veredicto corto

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿Sustituir masa v3→v6 en el estado del modelo de ΔT ayuda? | **NO** — pequeña ganancia OOF, pero degrada el lockbox (Reducción −0.11, Fusión −0.02 en R²) | tabla §2, config b vs a |
| ¿Agregar las features térmicas del balance (q_neto, dT_teórico, margen) como palanca ayuda? | **NO** — θ colapsa a 0 (en cota), R² OOF/lockbox idénticos a la config sin ellas | tabla §2 config c = config b; tabla θ §2 |
| ¿La energía específica (GN o q_neto por tonelada / por kg Sn disponible) como palanca, en vez de la tasa absoluta, ayuda? | **SÍ, para ΔT** (lockbox +0.05–0.06 R², sin costo en OOF); **marginal/nulo** para `m6_ln_sn_dep`/`m6_ln_feo_ret` (Δ ≤ 0.01) | tabla §3 |
| ¿La masa v6 cierra mejor el balance de energía (calor aparente vs trazador) que la v3? | **NO** — mismo orden de sesgo y correlación en Reducción (v6 levemente peor: sesgo −5 742 vs −3 482 MJ, sd 14 608 vs 12 034); en Fusión v6 atenúa algo la correlación espuria de signo incorrecto pero no la elimina | tabla §4 |
| Feature que SÍ entra al prescriptor v6 | `e6_gn_nm3_por_t_escoria` (= `masa_v6.m6_gn_esp_nm3_t`) como palanca del modelo de ΔT de Reducción, en vez de `tasa_gn_nm3_min` | §3, §5 |

## 1. Features térmicas: `b6_*` (masa v3) vs `e6_*` (masa v6)

Se copió (sin editar `bal_features.py`) una versión parametrizada de `agregar_balance_energia` que acepta
la columna de masa/inventario previo como argumento, para generar el mismo cálculo con dos fuentes de masa:

- `b6_*`: `masa_escoria_est_kg_prev` / `sn_inventario_escoria_est_kg_prev` (trazador CaO re-anclado, v3).
- `e6_*`: `m6_masa_kg_prev` / `m6_sn_inv_kg_prev` (trazador de matriz G en Fusión + cierre físico en
  Reducción, v6).

Fórmulas (idénticas a `bal_features.agregar_balance_energia`; solo cambia la columna de masa marcada):

```
q_combustion   = f(GN, O2, aire; PCI_GN)                                    (no depende de la masa)
q_carbon_lanza = f(carbon, O2 libre)                                        (no depende de la masa)
q_carga        = H_CARGA·feed_total + H_CARBON·carbon                       (no depende de la masa)
q_gases        = f(v_gas, T_horno_prev)                                     (no depende de la masa)
q_perdidas     = PERDIDAS_MJ_MIN[fase]·delta_tiempo                         (no depende de la masa)
q_neto         = q_combustion + q_carbon_lanza − q_carga − q_gases − q_perdidas
cap_baño       = MASA_PREV · CP_ESCORIA_KJ_KG_K / 1000                      [MJ/K]   ← difiere b6/e6
dT_teórico     = q_neto / cap_baño                                          [K]      ← difiere b6/e6
q_reduccion_max = Q_SN·(SN_INV_PREV + feed_Sn) + Q_FE_HEMATITA·feed_Fe·FE_MINERAL   ← difiere b6/e6
margen_térmico = q_neto − q_reduccion_max                                           ← difiere b6/e6
```

### 1.1 Descriptiva comparada (DEV+lockbox, `df_base` sin primer escalón de batch)

| Fase | Feature | v3 (b6) media | v6 (e6) media | Δ% | ρ Spearman b6~e6 | r Pearson b6~e6 |
|---|---|---:|---:|---:|---:|---:|
| Fusión | capacidad térmica (MJ/K) | 33.83 | 39.49 | +17% | 0.938 | 0.846 |
| Fusión | ΔT teórico sin reacción (K) | −159.7 | −150.1 | −6% (menos negativo) | 0.989 | 0.982 |
| Fusión | margen térmico (MJ) | −45 528 | −48 496 | +6% (más negativo) | 0.976 | 0.972 |
| Reducción | capacidad térmica (MJ/K) | 67.13 | 76.01 | +13% | 0.690 | 0.562 |
| Reducción | ΔT teórico sin reacción (K) | 72.9 | 64.7 | −11% | 0.987 | 0.982 |
| Reducción | margen térmico (MJ) | −6 305 | −8 270 | +31% (más negativo) | 0.948 | 0.969 |

La masa v6 es sistemáticamente mayor que la v3 (+13–17%), consistente con el diagnóstico de la
iteración 8 (el trazador CaO v3 subestima la masa porque asume tolva 6 = CaO puro). `dT_teórico` y
`margen_térmico` heredan una correlación muy alta entre versiones (ρ ≥ 0.95, salvo Fusión ρ=0.976) porque
`q_neto` —que domina la escala— no depende de la masa; solo la capacidad térmica del baño (que sí depende
de la masa) diverge de forma notoria, y lo hace más en Reducción (ρ Spearman 0.69, Pearson 0.56) que en
Fusión (0.94/0.85), reflejando que el estimador v6 diverge más del v3 justamente donde reemplaza la
lógica de trazador por un cierre físico encadenado. Tabla completa: `e8_02_features_termicas.csv`.

## 2. Modelos de ΔT (PLM v5, 4 configuraciones)

Config (a) v5 tal cual (estado `_S_DT`/`_S_DT`+tasas de carga en Fusión, palancas `PALANCAS_V5[(fase,"dT")]`,
signos `SIGNO_TEORICO_V5[(fase,"dT")]`); (b) igual que (a) pero con `masa_escoria_est_kg_prev` sustituida
por `m6_masa_kg_prev` en el estado; (c) igual que (b) más `e6_q_neto_por_min_MJ`, `e6_dT_teorico_sin_reaccion`,
`e6_margen_termico_MJ` **como palanca lineal adicional** (no como estado — ver nota de diseño abajo);
(d) referencia HGB con estado+palancas de (c) juntos como features planas.

| Fase | Config | n palancas | R² OOF | MAE OOF | R² lockbox | MAE lockbox |
|---|---|---:|---:|---:|---:|---:|
| Fusión | a_v5_tal_cual | 4 | 0.483 | 9.06 | 0.059 | 6.93 |
| Fusión | b_masa_v6_en_estado | 4 | 0.497 | 8.94 | 0.037 | 6.97 |
| Fusión | c_b_mas_termicas_e6 | 7 | 0.496 | 8.95 | 0.037 | 6.97 |
| Fusión | d_HGB_estado_palancas_c | 7 | 0.481 | 9.05 | 0.063 | 7.13 |
| Reducción | a_v5_tal_cual | 4 | 0.466 | 9.22 | 0.465 | 6.40 |
| Reducción | b_masa_v6_en_estado | 4 | 0.464 | 9.12 | 0.351 | 7.07 |
| Reducción | c_b_mas_termicas_e6 | 7 | 0.464 | 9.12 | 0.351 | 7.07 |
| Reducción | d_HGB_estado_palancas_c | 7 | 0.480 | 9.02 | 0.489 | 6.16 |

Lectura:

- **(a)→(b), sustituir la masa en el estado**: gana ~0.01–0.02 de R² OOF (ruido de muestreo, dentro de lo
  esperable) pero **pierde** fuerte en lockbox: Fusión 0.059→0.037, Reducción **0.465→0.351** (−0.11). La
  masa v6 en Reducción se construye por cierre físico encadenado con coeficientes poblacionales fijos
  (`pG`,`gG`,`aC`, calibrados en DEV); en el tramo de lockbox (fin de campaña, régimen distinto, ver §17.4
  de hallazgos) ese encadenamiento diverge más del comportamiento real que el trazador CaO re-anclado de
  v3, y esa divergencia se cuela en el estado `g(S)` del modelo de ΔT.
- **(b)→(c), agregar las térmicas e6 como palanca**: **no cambia nada** (R² OOF/lockbox idénticos a 3
  decimales en Reducción, y a 3 decimales en Fusión). La tabla de θ (`e8_02_theta_dT.csv`) confirma la
  razón: `e6_q_neto_por_min_MJ`, `e6_dT_teorico_sin_reaccion` y `e6_margen_termico_MJ` colapsan a
  θ ≈ 0 (marcadas `en_cota=True`) en las 4 combinaciones fase×config donde se prueban. Esto es
  consistente con que estas tres columnas son recombinaciones lineales/algebraicas de las mismas acciones
  primitivas (GN, O2, aire, carbón) ya presentes como palanca en (a)/(b): el residualizado de Double ML no
  les deja nada por explicar.
- **(d) HGB de referencia**: en Reducción el HGB (0.480/0.489) supera ligeramente al PLM en OOF y lockbox,
  sugiriendo alguna no linealidad residual, pero el patrón "masa v6 no ayuda / térmicas no ayudan" se
  sostiene también ahí (el HGB con el mismo set de columnas de (c) no mejora sustancialmente sobre el de
  (a)/(b) salvo por la flexibilidad no paramétrica).

**Nota de diseño (anti-fuga).** `e6_q_neto_por_min_MJ`/`e6_dT_teorico_sin_reaccion`/`e6_margen_termico_MJ`
combinan la acción del escalón t (GN, O2, aire, carbón, feed) con el estado_prev (masa, T, Sn disponible) —
son `DERIVED (A_t x S_prev)`, igual que `Cx_sn_v6` o `b6_q_neto_por_min_MJ` en `bal_features.py`. Por eso
NO se agregaron a la lista `estado` del PLM (eso las metería dentro de g(S), compitiendo con las palancas
primitivas de (a)/(b) dentro de la misma nuisance model y sesgando el Double ML de esas palancas); se
usaron únicamente como palanca lineal adicional del propio modelo de ΔT, cuyo target (`d_temperatura_horno_celsius`)
es del propio escalón t. θ completo con IC en `e8_02_theta_dT.csv`.

## 3. Energía específica como palanca (en vez de tasas absolutas)

Features nuevas (fundamento: normalizar la dosis por el tamaño real de la carga térmica, en vez de la
tasa absoluta, para que el coeficiente sea comparable entre escalones con distinto inventario de escoria
o de Sn disponible — la misma lógica que ya usa `masa_v6.m6_gn_esp_nm3_t` / `Cx_sn_v6`):

```
e6_gn_nm3_por_t_escoria     = GN_nm3(t) / (m6_masa_kg_prev(t)/1000)              [Nm3 GN / t escoria]
                              (idéntica a masa_v6.m6_gn_esp_nm3_t; se re-expone con prefijo e6_)
e6_q_neto_MJ_por_t          = e6_q_neto_disponible_MJ(t) / (m6_masa_kg_prev(t)/1000)   [MJ / t escoria]
e6_q_neto_MJ_por_kg_sn_disp = e6_q_neto_disponible_MJ(t) / m6_sn_disp(t)               [MJ / kg Sn disponible]
```

Todas son `DERIVED (A_t x S_prev)` (acción_t / estado_prev), verificadas seguras con `es_segura_e802` /
`masa_v6.es_segura_v6` sobre sus columnas base. Se probó sustituir **solo** `tasa_gn_nm3_min` (única
tasa absoluta de GN en las palancas existentes) por cada una de las tres, dejando el resto de la palanca
(O2, aire, carbón) igual, en tres modelos de Reducción: ΔT (estado `_S_DT` con masa v6), `m6_ln_sn_dep`
(estado `ESTADO_R_CURADO` con inventarios v3→v6) y `m6_ln_feo_ret` (mismo estado). Signo teórico de las
tres nuevas features: **+1 para ΔT** (más calor neto específico → más subida de temperatura, relación
directa); **libre (0) para `m6_ln_sn_dep`/`m6_ln_feo_ret`** porque `q_neto` mezcla algebraicamente GN + O2
+ aire + carbón (reductor y oxidante a la vez) y no tiene signo teórico unívoco frente al agotamiento de
Sn o la retención de FeO — se documenta como exploratorio.

| Target | Variante | R² OOF | R² lockbox | Δ OOF vs base | Δ lockbox vs base |
|---|---|---:|---:|---:|---:|
| ΔT (R) | base (`tasa_gn_nm3_min`) | 0.4645 | 0.3506 | — | — |
| ΔT (R) | `e6_gn_nm3_por_t_escoria` | 0.4636 | **0.4069** | −0.001 | **+0.056** |
| ΔT (R) | `e6_q_neto_MJ_por_t` | 0.4638 | **0.4030** | −0.001 | **+0.052** |
| ΔT (R) | `e6_q_neto_MJ_por_kg_sn_disp` | 0.4631 | **0.4056** | −0.001 | **+0.055** |
| `m6_ln_sn_dep` | base | 0.7548 | 0.8110 | — | — |
| `m6_ln_sn_dep` | `e6_gn_nm3_por_t_escoria` | 0.7557 | 0.8108 | +0.001 | −0.0002 |
| `m6_ln_sn_dep` | `e6_q_neto_MJ_por_t` | 0.7571 | 0.8103 | +0.002 | −0.001 |
| `m6_ln_sn_dep` | `e6_q_neto_MJ_por_kg_sn_disp` | 0.7548 | 0.8096 | 0.000 | −0.001 |
| `m6_ln_feo_ret` | base | 0.3299 | 0.4643 | — | — |
| `m6_ln_feo_ret` | `e6_gn_nm3_por_t_escoria` | 0.3256 | 0.4670 | −0.004 | +0.003 |
| `m6_ln_feo_ret` | `e6_q_neto_MJ_por_t` | 0.3380 | 0.4736 | +0.008 | +0.009 |
| `m6_ln_feo_ret` | `e6_q_neto_MJ_por_kg_sn_disp` | 0.3297 | 0.4664 | 0.000 | +0.002 |

Lectura: para **ΔT**, cualquiera de las tres normalizaciones recupera **0.05–0.06 de R² en lockbox** sin
costo en OOF (que se mantiene esencialmente plano) — la dosis absoluta de GN generaliza mal entre régimen
DEV y lockbox (batches de distinto tamaño de carga/masa de escoria), y normalizar por masa o por Sn
disponible corrige buena parte de esa falta de transferencia. Para **`m6_ln_sn_dep`** el efecto es nulo
(R² ya es 0.75–0.81, dominado por el término cinético `Cx_sn_v6`, y GN no aporta mucho en ninguna forma).
Para **`m6_ln_feo_ret`** el efecto es pequeño pero consistentemente positivo con `e6_q_neto_MJ_por_t`
(+0.008 OOF / +0.009 lockbox); las otras dos variantes son prácticamente neutras. Tabla completa:
`e8_02_energia_especifica.csv`.

## 4. Calor de reacción aparente vs. calor de reacción por inventarios (v3 vs v6, diagnóstico post-hoc)

```
q_reaccion_aparente  = q_neto − cap_baño·ΔT_medido
q_reaccion_trazador  = Q_SN·SN_EXTRAÍDO + Q_FEO·FEO_EXTRAÍDO + Q_FE_HEMATITA·feed_Fe·FE_MINERAL
```

Si el balance de energía "viera" la extracción real, ambos deberían correlacionar positivamente. Se
comparó usando SN/FEO_EXTRAÍDO de v3 (`sn_extraido_est_kg`/`feo_extraido_est_kg`) y de v6
(`m6_sn_extraido_kg`/`m6_feo_extraido_kg`), y se estimó, por regresión a través del origen sobre
`delta_tiempo`, el ajuste de pérdidas (MJ/min) que anularía el sesgo medio.

| Versión | Fase | ρ Spearman | r Pearson | Sesgo medio (MJ) | Sesgo sd (MJ) | Pérdidas actuales (MJ/min) | Pérdidas ajustadas (MJ/min) |
|---|---|---:|---:|---:|---:|---:|---:|
| masa v3 | Fusión | −0.257 (p≈3e-28) | −0.177 | −23 883 | 11 007 | 117.0 | **−391.6** (unfísico) |
| masa v3 | Reducción | +0.315 (p≈1e-28) | +0.151 | −3 483 | 12 034 | 158.0 | **1.8** (plausible) |
| masa v6 | Fusión | −0.203 (p≈5e-18) | −0.067 | −23 319 | 11 074 | 117.0 | **−379.6** (unfísico) |
| masa v6 | Reducción | +0.317 (p≈5e-29) | +0.157 | −5 742 | 14 608 | 158.0 | **−99.5** (unfísico) |

Lectura: en **Fusión** ambas versiones dan correlación de **signo incorrecto** (negativa; se espera
positiva) — consistente con el hallazgo previo (§17/B-03) de que el balance sin calibrar omite el carbón
quemado con aire de post-combustión y la oxidación de Fe metálico/sulfuros, que no están en el dataset.
v6 **atenúa** la magnitud de esa correlación espuria (Pearson −0.067 vs −0.177) pero no cambia el signo ni
resuelve el sesgo (de hecho el sesgo medio absoluto es prácticamente el mismo, −23 319 vs −23 883 MJ) — el
ajuste de pérdidas necesario para anular el sesgo sigue siendo una pérdida **negativa** (heat gain, no
loss) de ~380–510 MJ/min, físicamente inadmisible con las pérdidas del modelo Rust (117 MJ/min): la
brecha es de fuentes de calor no medidas, no de la masa. En **Reducción**, v3 y v6 dan una correlación casi
idéntica (ρ≈0.32, r≈0.15) — de signo correcto pero débil — y v6 **empeora** el cierre: sesgo medio más
negativo (−5 742 vs −3 483 MJ) y más disperso (sd 14 608 vs 12 034), y el ajuste de pérdidas que anula el
sesgo con v3 es un valor plausible (≈1.8 MJ/min, cercano a 0, sugiriendo que las 158 MJ/min asumidas están
sobrestimadas en ese margen) mientras que con v6 exige de nuevo una pérdida negativa (−99.5 MJ/min),
inadmisible. **Conclusión de la tarea 4: el balance de energía por escalón NO cierra mejor con la masa
v6** — el techo de esta pieza del análisis sigue siendo la falta de variables físicas (aire de
post-combustión, temperatura de baño real vs. temperatura de cara interior medida), tal como ya se había
concluido con v3 en la iteración 6 (hallazgos §17). Tabla completa: `e8_02_cierre_energia.csv`.

## 5. Conclusión y recomendación para el prescriptor v6

1. **Ninguna de las features térmicas "de balance" (`e6_q_neto_por_min_MJ`, `e6_dT_teorico_sin_reaccion`,
   `e6_margen_termico_MJ`) entra al prescriptor v6**, ni como estado ni como palanca: no mejoran R² del
   modelo de ΔT (θ colapsa a 0) y el balance físico subyacente sigue sin cerrar (§4), igual que con v3
   (hallazgos §17). Esto reconfirma, con la masa v6, el veredicto de la iteración 6.
2. **No sustituir `masa_escoria_est_kg_prev` por `m6_masa_kg_prev` en el estado del modelo de ΔT**: la
   ganancia OOF es marginal y el costo en lockbox es significativo (Reducción −0.11 en R², Fusión −0.02),
   probablemente porque el cierre físico encadenado de v6 en Reducción diverge más del comportamiento del
   tramo final de campaña (lockbox) que el trazador CaO re-anclado de v3.
3. **Sí entra al prescriptor v6 una única feature nueva**: `e6_gn_nm3_por_t_escoria` (idéntica a
   `masa_v6.m6_gn_esp_nm3_t`, dosis de GN por tonelada de escoria) **como palanca del modelo de ΔT de
   Reducción, sustituyendo a `tasa_gn_nm3_min`** — mejora el R² de lockbox en +0.05–0.06 sin costo en OOF,
   la única ganancia neta y reproducible de todo el experimento. `e6_q_neto_MJ_por_t` y
   `e6_q_neto_MJ_por_kg_sn_disp` dan una ganancia casi idéntica (+0.05) pero no superior, así que se
   recomienda la más simple e interpretable de las tres (dosis de GN, no una mezcla algebraica de las
   cuatro acciones de la lanza).
4. **Para los targets cinéticos de Reducción** (`m6_ln_sn_dep`, `m6_ln_feo_ret`) **no hay evidencia
   suficiente para cambiar las palancas actuales** por versiones de energía específica: el efecto es nulo
   en `sn_dep` (R² ya dominado por `Cx_sn_v6`) y pequeño (+0.01) en `feo_ret` con `e6_q_neto_MJ_por_t` —
   se documenta como señal débil, no se adopta.
5. **La masa v6 no hace que el balance de energía por escalón sea más coherente que con v3**: la
   correlación calor-aparente vs. calor-trazador es del mismo orden (Reducción) o incluso más sesgada
   (Fusión sigue con signo incorrecto; Reducción con sesgo mayor) — el problema es la falta de variables
   físicas del proceso (aire de post-combustión, temperatura real de baño), no la calidad del estimador de
   masa de escoria.

### Límites

- Los resultados usan la calibración de `masa_v6.py` (`PARAMS_DEFAULT`) vigente al momento de correr este
  script; ese módulo es un artefacto compartido con el experimento paralelo E8-01 (calibración/robustez del
  estimador) y puede recalibrarse en esa línea de trabajo — no se modificó desde aquí.
  Descriptivas de una corrida parcial anterior (antes de que E8-01 actualizara `masa_v6.py`) mostraron
  magnitudes ligeramente distintas para las mismas columnas (p. ej. capacidad térmica ~35 vs ~39 MJ/K en
  Fusión); los números reportados aquí corresponden a UNA sola corrida completa y coherente de principio a
  fin (verificado con los timestamps `[Ns] sección X lista` del log).
- Todos los supuestos físicos (`PCI_GN`, `CARBONO_FIJO`, `H_CARGA`, `CP_ESCORIA`, `PERDIDAS_MJ_MIN`, etc.)
  son los mismos de `bal_features.SUPUESTOS`, con las mismas advertencias de la iteración 6: faltan aire de
  post-combustión, temperatura de baño real y análisis del carbón en el dataset, por lo que el balance de
  energía por escalón no puede cerrarse de forma independiente con ninguna de las dos masas.
- La ablación de energía específica (§3) usa el estado `ESTADO_R_CURADO` con los cuatro inventarios v3
  sustituidos por sus equivalentes v6 (`m6_masa_kg_prev`, `m6_sn_inv_kg_prev`, `m6_feo_inv_kg_prev`,
  `m6_avance_prev`), no el estado FULL de producción (`ESTADO_R_FULL`) que usa v5 para estos targets — es
  una simplificación deliberada para aislar el efecto de la palanca de energía; la comparación con la
  política de producción completa es tarea de E8-03.
