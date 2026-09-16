# E9-02 — La cal (CaO, tolva 6) y la basicidad como palanca bajo la masa v6

Script: `e9_02_cal.py` (etapas `kpi`, `batch`, `escalon_F`, `escalon_R`, `basicidad`, `resumen`). Datos:
`v7_lib.cargar_df()` / `v7_lib.construir_batch()`. DEV = 299 batches, LOCKBOX = 63 (reporte), TOTAL = 362.
Salidas: `e9_02_kpi_comparacion.csv`, `e9_02_kpi_decision.json`, `e9_02_batch_regresiones.csv`,
`e9_02_cuadratico.csv`, `e9_02_dosis_respuesta.csv`, `e9_02_boot_coef.csv`, `e9_02_plm_fusion.csv`,
`e9_02_theta_fusion.csv`, `e9_02_estabilidad_fusion.csv`, `e9_02_plm_reduccion.csv`,
`e9_02_theta_reduccion.csv`, `e9_02_distribucion_cal_R.csv`, `e9_02_basicidad_cuadratico.csv`,
`e9_02_basicidad_deciles.csv`, `e9_02_resumen.csv`, `e9_02_veredicto.json`.

## Veredicto en una frase

**Bajo la masa v6 (cierre físico, sin dividir por %CaO) el efecto de batch de la cal DESAPARECE**: el
hallazgo de E7-02 (`cal_por_carga_F` −0.87 pp/sd, p = 0.038 con el trazador v3) no replica (p = 0.145 en
DEV). Hay un canal físico real y coherente con la teoría — la **basicidad** (no la dosis cruda de cal)
reduce el polvo y el Sn perdido en escoria final y sube la retención de FeO, todo en DEV con p < 0.01 —
pero, igual que el carbón/temperatura de Fusión en E7-02, ese canal no sobrevive agregado al KPI compuesto.
**No se adopta la cal como palanca de PLAN.**

## 1. Etapa `kpi`: ¿depende `recuperacion_refinada_pct` del trazador CaO antiguo?

`fe.construir_resumen_batch_balance_global` usa una masa de escoria de **balance físico de batch**
(entradas − salidas, con `feed_CaO_kgh` como masa húmeda, no dividida por %CaO) — no es el trazador v3
escalón a escalón. La única dependencia posible es indirecta: al elegir la fila de Reducción con el
"último ensayo" de `ley_sn_escoria_pct`, el código históricamente filtraba por disponibilidad de
`masa_escoria_est_kg` (v3).

| Chequeo | Resultado |
|---|---|
| Filas donde v3 elige un escalón "final" distinto del que elige v6 | 1/362 (0.3 %) |
| Diferencia de `ley_fin` entre criterios | media 0.005 pp, sd 0.093 pp |
| `kpi_v6` (sustituye la masa de balance por `R_m6_masa_fin_kg`, ley del mismo criterio v6) vs original | ρ = 0.9993, Pearson = 0.9994, MAE = 0.095 pp, máx. \|dif\| = 1.67 pp, 1.7 % de batches cambian de cuartil |

**Conclusión**: la dependencia del trazador v3 en `recuperacion_refinada_pct` es marginal (afecta la
selección de fila en 0.3 % de los batches). La diferencia numérica entre el KPI original y `kpi_v6` viene
sobre todo de usar dos masas de escoria distintas (balance de batch vs. cierre físico v6 por escalón), no
del trazador. Se adopta **`kpi_v6`** para el resto de este experimento por consistencia metodológica (toda
la evidencia de Reducción ya usa la masa v6); la diferencia con el KPI original es < 0.1 pp en promedio y
no cambia ninguna conclusión cualitativa.

## 2. Etapa `batch`: cal/carga, cal/Sn, basicidad vs KPI y canales (DEV, OLS HC3 + controles)

Rango histórico de `cal_por_carga_F` (Σ cal / Σ carga en Fusión): P25 = 0.0908, mediana = 0.0972,
P75 = 0.1015 (media 0.0940, sd 0.0202); `cal_por_sn_F`: P25 = 227, mediana = 245, P75 = 258 kg cal / t Sn
alimentado (media 236, sd 50.5).

### KPI (`kpi_v6`)

| variable | DEV coef/sd | IC95% | p (DEV) | TOTAL coef/sd | p (TOTAL) |
|---|---|---|---|---|---|
| `cal_por_carga_F` | −0.519 | [−1.237, 0.172] | 0.145 | −0.531 | 0.071 |
| `cal_por_sn_F` | −0.587 | [−1.283, 0.109] | 0.099 | −0.601 | **0.039** |
| `b2_media_F` | +0.129 | [−0.435, 0.693] | 0.653 | +0.051 | 0.840 |
| `b2_fin_F` | +0.409 | [−0.093, 0.911] | 0.110 | +0.357 | 0.131 |

Ninguna variable de cal/basicidad es significativa sobre el KPI en DEV (criterio de selección); sólo
`cal_por_sn_F` cruza p < 0.05 en TOTAL (n = 362, impulsado por el lockbox). Bootstrap por batch (500
réplicas, DEV) del coeficiente de `cal_por_carga_F`: −0.519 [−1.237, 0.172] — coincide con HC3, confirma
que no es un artefacto del error estándar analítico.

### Cuadrático (óptimo interior de `cal_por_carga_F`)

| target | muestra | coef_sq | IC95% | p_sq | n |
|---|---|---|---|---|---|
| `kpi_v6` | DEV | +0.352 | [−0.051, 0.754] | 0.087 | 298 |
| `R_sum_m6_ln_feo_ret` | DEV | +0.0065 | [−0.0007, 0.0136] | 0.076 | 297 |
| `R_sum_m6_ln_sn_dep` | DEV | −0.0016 | [−0.041, 0.038] | 0.938 | 297 |

**0/3 cuadráticos significativos en DEV** (sólo `R_sum_m6_ln_feo_ret` roza p < 0.05 en TOTAL, n = 360, no
en DEV): sin evidencia de óptimo interior en `cal_por_carga_F` dentro del rango observado. Dosis-respuesta
por quintiles de `cal_por_carga_F` (DEV): el KPI es plano (67.7 a 68.6 pp, IC bootstrap solapados en los 5
quintiles, sin monotonía) y lo mismo `f_dross`, `f_polvo`, `sn_perdido_escoria_frac`, `R_sum_m6_ln_feo_ret`
y `R_sum_m6_ln_sn_dep` — la dosis cruda de cal no mueve ningún canal de forma visible.

### Los canales SÍ responden a la BASICIDAD (no a la dosis cruda), signo teórico correcto (DEV)

| canal | variable | coef/sd | IC95% | p | Lectura |
|---|---|---|---|---|---|
| `f_polvo` | `b2_fin_F` | −0.00593 | [−0.0094, −0.0025] | **0.0008** | más B2 → menos polvo (bueno) |
| `f_polvo` | `b2_media_F` | −0.00432 | [−0.0083, −0.0003] | **0.034** | ídem |
| `sn_perdido_escoria_frac` | `b2_fin_F` | −0.00112 | [−0.0019, −0.0003] | **0.005** | más B2 → menos Sn en escoria final (bueno) |
| `sn_perdido_escoria_frac` | `b2_media_F` | −0.00090 | [−0.0017, −0.0001] | **0.033** | ídem |
| `R_sum_m6_ln_feo_ret` | `b2_fin_F` | +0.0179 | [0.0065, 0.0293] | **0.002** | más B2 → más retención de FeO = mejor selectividad (bueno) |
| `R_sum_m6_ln_feo_ret` | `b2_media_F` | +0.0138 | [0.0011, 0.0266] | **0.034** | ídem |

Los tres canales se mueven en DEV con p < 0.05, en el signo que predice la teoría (más CaO despolimeriza
la red silicatada → menor viscosidad → mejor selectividad SnO/FeO y menos arrastre a polvo/escoria). **Este
es exactamente el patrón de E7-02** ("v4 optimizaba una señal de sub-canal que no se traduce en el KPI
compuesto"): los tres canales mejoran, pero el efecto neto sobre `kpi_v6` (que ya pondera dross+polvo+Sn en
escoria) queda dentro del ruido (tabla de arriba, p = 0.110–0.145). Nótese además que el canal que
"debería" mover la basicidad es `b2_fin_F`/`b2_media_F` (la razón CaO/SiO2 realizada), no la dosis de cal
por sí sola — la sílice de la carga varía batch a batch y diluye la relación cal → B2 → canal.

### Heel / batch siguiente

`cal_por_carga_F → kpi_v6_next` (DEV): coef −0.267 pp/sd, p = 0.437 (n.s.); `b2_media_F → kpi_v6_next`:
coef +0.509, p = 0.088 (borderline). Sin evidencia de artefacto de heel que infle o enmascare el efecto de
la cal del batch actual.

## 3. Etapa `escalon_F`: PLM con signos, cal como palanca adicional en Fusión

Palancas: `tasa_feed_CaO_kg_min` + `relacion_C_Sn_carga`, `tasa_gn_nm3_min`, `exceso_o2_combustion_pct`,
`tasa_aire_nm3_min`. Estado = `L.ESTADO[('Fusión', clave)]` quitando `tasa_feed_CaO_kg_min` en cada caso.

| target | R²oof DEV | R²lockbox | θ cal (por sd) | IC95% | signo esperado | significativo |
|---|---|---|---|---|---|---|
| `m6_ln_sn_dep` | 0.156 | −0.384 | +0.0052 | [−0.043, 0.052] | libre (0) | No |
| `d_temperatura_horno_celsius` | 0.498 | 0.041 | ≈0 (en cota) | [−0.618, ≈0] | −1 (consume calor) | **No** (colapsa a 0: el dato apuntaba al signo contrario) |
| `d_basicidad_B2` | 0.220 | −0.020 | +0.00299 | [≈0, 0.0134] | +1 (control mecánico) | **No** (IC roza cero) |
| `v5_d_lnirf` | 0.272 | 0.087 | −0.00616 | [−0.0144, −0.0004] | −1 | **Sí** |

La prueba mecánica más exigente del diseño — cal sobre ΔB2, que "debía" ser fuerte y significativa porque
es casi un control directo — **no la alcanza** con el PLM de doble ML (el residuo de cal tras remover lo
predecible por el estado, que ya incluye `basicidad_B2_prev`/`ley_cao_escoria_pct_prev`, casi no varía).
Sólo `v5_d_lnirf` (que suma CaO al SiO2+Al2O3 en el denominador) cruza el umbral, con el signo correcto.

**Estabilidad por tercios cronológicos** (`e9_02_estabilidad_fusion.csv`): ninguno de los 4 efectos de cal
es estable en ≥ 2/3 de los tercios —
`d_basicidad_B2`: no significativo en los 3 tercios;
`v5_d_lnirf`: significativo sólo en el tercio 1 (medio);
`m6_ln_sn_dep`: no significativo en tercios 0-1, significativo (signo negativo, fuera de la teoría "libre"
pero sin contradicción) sólo en el tercio 2;
`d_temperatura_horno_celsius`: nunca significativo.
Esto reproduce el patrón ya documentado ("theta cambia de signo/significancia entre períodos", Lecciones
iteración 4): **no hay una relación cal→matriz de Fusión estable en el tiempo**, incluso con la masa v6.

## 4. Etapa `escalon_R`: la cal en Reducción

`feed_CaO_kgh` > 0 en 1446/1448 escalones (99.9 %) pero la distribución es degenerada: mediana = 0,
P75 = 0.092 kg/min, media 4.96 arrastrada por un máximo de 653 (unos pocos escalones con carga real de
cal; el resto es ruido de redondeo alrededor de 0). **No hay dosificación real de cal en Reducción en la
práctica.** El PLM (cal añadida como palanca extra a `ESTADO_R_FULL_V6`/`PALANCAS_V6`) confirma la
ausencia de señal:

| target | R²oof DEV | R²lockbox | θ cal (por sd) | IC95% | significativo |
|---|---|---|---|---|---|
| `m6_ln_sn_dep` | 0.765 (igual que baseline v6) | 0.799 | −0.0043 | [−0.0385, 0.0061] | No |
| `m6_ln_feo_ret` | 0.329 (igual que baseline v6) | 0.486 | +0.00017 | [−0.0014, 0.0065] | No |

Los R² OOF/lockbox son idénticos a los del PLM v6 sin cal (hallazgos §19.4): añadir la cal no aporta ni
resta información. Confirma la premisa del diseño ("la cal en Reducción es pequeña").

## 5. Etapa `basicidad`: ¿óptimo interior al cierre de Fusión?

Cuadrático centrado de `b2_fin_F` e `irf_fin_F` sobre `kpi_v6`, `R_sum_m6_ln_feo_ret`,
`R_sum_m6_ln_sn_dep` (controles + `T_media_R` + heel `F_lnirf_F0`):

| variable | target | p_sq (DEV) | p_sq (TOTAL) |
|---|---|---|---|
| `b2_fin_F` | `kpi_v6` | 0.159 | 0.373 |
| `b2_fin_F` | `R_sum_m6_ln_feo_ret` | 0.938 | 0.621 |
| `b2_fin_F` | `R_sum_m6_ln_sn_dep` | 0.198 | 0.328 |
| `irf_fin_F` | `kpi_v6` | 0.503 | 0.801 |
| `irf_fin_F` | `R_sum_m6_ln_feo_ret` | 0.476 | 0.411 |
| `irf_fin_F` | `R_sum_m6_ln_sn_dep` | 0.732 | 0.547 |

**0/6 cuadráticos significativos** (ver también `e9_02_basicidad_deciles.csv`, deciles sin forma de U
visible). No hay evidencia de óptimo interior de basicidad/IRF al cierre de Fusión en el rango histórico
observado: la relación (donde existe, vía `b2_fin_F` sobre los canales de la sección 2) es del signo
esperado pero **monótona**, no en forma de U.

## 6. Veredicto (criterio §5 del diseño)

Criterio de adopción: **p < 0.05 en DEV con controles** + **signo explicable por un canal físico** +
**recomendación dentro de P25–P75 histórico**.

| condición | resultado |
|---|---|
| p < 0.05 en DEV (batch, con controles) | **No cumple** — `cal_por_carga_F` p = 0.145, `cal_por_sn_F` p = 0.099, `b2_fin_F` sobre KPI p = 0.110 |
| Signo explicable por canal físico | **Parcial** — la basicidad (no la dosis de cal) mueve `f_polvo`, `sn_perdido_escoria_frac` y `R_sum_m6_ln_feo_ret` con p < 0.05 y signo correcto, pero no se traduce en el KPI compuesto (mismo patrón que C/Sn y T de Fusión en E7-02) |
| Prueba mecánica a nivel de escalón (cal → ΔB2) | **No confirmada** (IC roza cero); sólo Δln IRF es significativa, y no es estable por tercios |
| Cal en Reducción | Dosis prácticamente nula; sin efecto identificable |
| Óptimo interior de basicidad | **Sin evidencia** (0/6 cuadráticos significativos) |

### VEREDICTO: **NO SE ADOPTA la cal como palanca de PLAN.**

El hallazgo de E7-02 (`cal_por_carga_F` −0.87 pp/sd, p = 0.038 con el trazador CaO v3) **no replica** bajo
la masa v6 (p = 0.145 en DEV): esto **confirma** el diagnóstico de hallazgos §19.1 — ese efecto era el
artefacto contable del trazador (`masa = cal/%CaO`, correlacionado con la dosis de cal por construcción),
no un efecto metalúrgico real. H2 de la iteración 9 queda **refutada** para la cal como dosis controlable
por PLAN.

Hallazgo secundario que **sí** se conserva y es potencialmente útil (aunque no accionable como PLAN por
dosis): la **basicidad realizada al cierre de Fusión** (`b2_fin_F`, ya presente como `basicidad_B2_prev`/
`indice_irf_prev` en los estados curados de Reducción) predice, con significancia DEV y signo teórico
correcto, menos polvo, menos Sn perdido en escoria final y mejor retención de FeO — es decir, sigue
siendo un buen candidato de **STATE** (ya usado), pero no se identificó un canal de acción (dosis de cal
por tolva) que la mueva con la fuerza y estabilidad necesarias para recomendarla como decisión de PLAN.

No se reporta rango recomendado de cal/carga ni de cal/Sn: dado que no hay efecto significativo y estable,
cualquier "óptimo" estimado sería ruido. El rango histórico observado (`cal_por_carga_F` P25–P75: 9.08–
10.15 %; `cal_por_sn_F` P25–P75: 227–258 kg cal/t Sn) queda documentado sólo como referencia operativa
actual, no como recomendación.

## Límites

- `kpi_v6` y `recuperacion_refinada_pct` difieren en la masa de escoria usada (balance de batch vs. cierre
  físico v6 por escalón); la diferencia es pequeña (MAE 0.095 pp) pero no nula — no se recalibró el
  objetivo `CONFIG_V6` (w, K) con `kpi_v6`, se usó sólo para este experimento.
- El PLM de Fusión (todos los targets) no generaliza bien a lockbox (R² 0.04–0.09, uno negativo): los θ
  reportados en la sección 3 son válidos sólo como evidencia DEV, no para prescribir sin más validación.
- La ausencia de dosificación real de cal en Reducción (mediana 0) impide cualquier conclusión sobre un
  posible uso de la cal como palanca en esa fase; sería necesario un piloto con dosis deliberadas.
- El bootstrap por batch (sección 2) es un bootstrap simple por fila (una fila = un batch), no cluster;
  es equivalente en este caso porque la unidad de análisis ya es el batch.
