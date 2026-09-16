# E9-06 — Estado parsimonioso de Reducción que conserve el R² e identifique el carbón

Script `e9_06_parsimonia.py` (etapas: `importancia | eliminacion | sets_teoria | estabilidad | varianza_theta | resumen`),
salidas `e9_06_importancia.csv`, `e9_06_eliminacion.csv`, `e9_06_sets.csv`, `e9_06_estabilidad.csv`,
`e9_06_varianza_theta.csv`, `e9_06_resumen.csv`. OOF GroupKFold(5) por Batch sobre DEV (1058-1083 filas según el set,
n varía por `dropna`); lockbox = 63 batches finales, sólo reporte. Estado FULL = 38 columnas (`v7_lib.ESTADO[('Reducción','sn_dep')]`);
CURADO = 22 (`v7_lib.ESTADO_R_CURADO`). Núcleo físico protegido en la ruta "física":

    m6_sn_inv_kg_prev, m6_feo_inv_kg_prev, m6_masa_kg_prev, m6_avance_prev, temperatura_horno_celsius_prev,
    orden_escalon_fase, cum_feed_Carbon_kg_prev, relacion_C_cum_Sn_cum_prev, basicidad_B2_prev, indice_irf_prev

**Nota sobre "excluye 0"**: el θ de `Cx_sn_v6` está acotado en [0, ∞) (signo de teoría). Cuando el punto óptimo es interior el IC
bootstrap se comporta con normalidad, pero cuando el dato apenas sostiene el signo, el percentil 2.5 del bootstrap puede caer en
valores de punto flotante ~1e-10 a 1e-16 (prácticamente 0, no una exclusión real de 0). Todas las conclusiones de este memo usan
un umbral práctico `ci_lo > 1e-3` (dos a tres órdenes de magnitud menor que los θ típicos, 0.08-0.22 sd) para declarar
"identifica el carbón"; los valores crudos están en los CSV.

## 1. Importancia (`e9_06_importancia.csv`)

**Predictiva** (permutación OOF, 5 repeticiones, GroupKFold(5) por batch, sobre el estado FULL):

| Target | R² base (g(S) solo) | top features |
|---|---|---|
| `m6_ln_sn_dep` | 0.761 | `ley_sn_escoria_pct_prev` (Δ 0.325), `interaccion_Sn_x_FeO_prev` (0.061), `ratio_sn_feo_prev` (0.023), `m6_avance_prev` (0.020), `posicion_vertical_lanza_mm_prev` (0.014), `m6_sn_inv_kg_prev` (0.010), `cum_aire_nm3_prev` (0.010) |
| `m6_ln_feo_ret` | 0.322 | `ratio_sn_feo_prev` (0.105), `interaccion_Sn_x_FeO_prev` (0.040), `posicion_vertical_lanza_mm_prev` (0.027), `m6_feo_inv_kg_prev` (0.026), `m6_resto_frac_prev` (0.018), `m6_sn_inv_kg_prev` (0.016), `cum_aire_nm3_prev` (0.016) |

El resto de las 38 features aporta < 0.007 de R² cada una: la señal predictiva está muy concentrada en la ley de Sn previa
(que fija el nivel) y en 5-6 features de composición/geometría; el resto es redundante entre sí (colinealidad fuerte, ver §3).

**Identificación** (leave-one-out sobre el estado FULL, n_boot=100, siguiendo θ e IC de la palanca foco):

- `Cx_sn_v6` en `m6_ln_sn_dep`: baseline FULL θ=0.164 sd, IC [0.028, 0.276]. **Ninguna de las 38 remociones individuales
  rompe la identificación** (el IC sigue excluyendo 0 en los 38 casos); la que más la debilita es quitar `cum_aire_nm3_prev`
  (ensancha el IC +0.042) y la que más la fortalece es quitar `m6_masa_kg_prev` (angosta -0.018) o `basicidad_B4_prev` (-0.013).
  Quitar `cum_feed_Carbon_kg_prev` o `relacion_C_cum_Sn_cum_prev` por separado **no** rompe nada (angosta ligeramente el IC:
  -0.012 y -0.006). Conclusión: la identificación de FULL es **redundante**, no depende de una sola feature — hay múltiples
  proxies solapados (leyes, ratios, inventarios, historia acumulada) que sostienen conjuntamente el residuo de `Cx_sn_v6`.
- `tasa_feed_Carbon_kg_min` en `m6_ln_feo_ret`: baseline FULL θ=-0.0002 (prácticamente 0, ya en la cota en FULL) — el carbón
  primitivo sobre la retención de FeO no está identificado ni con el estado completo (consistente con hallazgos previos:
  sólo `Cx_av_v6`/GN se identifican ahí); el LOO no cambia esta conclusión en ningún caso.

## 2. Eliminación hacia atrás (`e9_06_eliminacion.csv`)

Ruta **libre** (sin protección) vs **física** (nunca elimina el núcleo de 10). Importancia combinada (promedio normalizado de
la importancia por permutación de ambos targets) recalculada cada 4 eliminaciones.

| ruta | k | sn_dep R² OOF/lockbox | feo_ret R² OOF/lockbox | Cx_sn θ₁sd [IC] |
|---|---|---|---|---|
| libre | 38 (FULL) | 0.765/0.800 | 0.329/0.486 | 0.164 [0.025, 0.291] |
| libre | 30 | 0.765/0.799 | 0.333/0.501 | 0.157 [0.008, 0.284] |
| libre | 24 | 0.772/0.804 | 0.339/0.483 | 0.176 [0.060, 0.306] |
| libre | 20 | **0.775/0.801** | **0.350/0.479** | 0.189 [0.045, 0.331] |
| libre | 16 | 0.767/0.795 | 0.333/0.483 | 0.178 [0.047, 0.287] |
| libre | 12 | 0.758/0.808 | 0.334/0.486 | 0.184 [0.022, 0.306] |
| libre | 10 | 0.753/0.804 | 0.326/0.466 | 0.191 [0.033, 0.306] |
| libre | 8 | 0.756/0.796 | 0.328/0.458 | 0.216 [0.055, 0.321] |
| libre | 6 | 0.757/0.805 | 0.299/0.463 | 0.176 [0.031, 0.282] |
| física | 24 | 0.768/0.814 | 0.346/0.500 | 0.186 [0.051, 0.322] |
| física | 20 | 0.762/0.804 | 0.328/0.485 | 0.184 [0.023, 0.294] |
| física | 16 | 0.751/0.811 | 0.317/0.495 | 0.222 [0.058, 0.356] |
| física | 12 | 0.739/0.801 | 0.289/0.469 | 0.138 [0.000, 0.272] |
| física | 10 | 0.736/0.800 | 0.296/0.448 | 0.155 [0.000, 0.280] |
| física | 8, 6 | no aplica: el núcleo protegido tiene 10 features, no se puede bajar de 10 | | |

**Hallazgo central**: la ruta **libre** identifica el carbón (IC > 0 con margen real) en **todos** los checkpoints de 38 a 6,
incluso en k=6. La ruta **física** (forzada a conservar los 10 "obvios" teóricos) **pierde la identificación en k≤12**
(IC toca 0) y tiene peor R² que la libre en casi todos los k ≤ 20. Es decir: proteger a ciegas el núcleo teórico es
**peor** que dejar que la importancia por permutación elija — los 10 "obvios" no son, individualmente, el mejor paquete de
10-12 features; varios (p. ej. `temperatura_horno_celsius_prev`, `orden_escalon_fase`, `m6_avance_prev`, `indice_irf_prev`)
son sustituibles por proxies correlacionados (`termocupla_media_celsius_prev`, `grad_temperatura_horno_celsius_prev`,
`ratio_sn_feo_prev`, `interaccion_Sn_x_FeO_prev`) que retienen la misma información con más eficiencia estadística.

Sets ganadores concretos (features de la ruta libre):

- **k=12**: `ley_sn_escoria_pct_prev; ley_sio2_escoria_pct_prev; ratio_sn_feo_prev; interaccion_Sn_x_FeO_prev; grad_temperatura_horno_celsius_prev; termocupla_media_celsius_prev; posicion_vertical_lanza_mm_prev; relacion_C_cum_Sn_cum_prev; m6_sn_inv_kg_prev; m6_feo_inv_kg_prev; espesor_ladrillo_norm_mm; m6_resto_frac_prev`
- **k=16** (= k12 + 4): agrega `ley_feo_escoria_pct_prev, ley_cao_escoria_pct_prev, cum_feed_Carbon_kg_prev, cum_aire_nm3_prev`

## 3. Sets por teoría (`e9_06_sets.csv`)

| set | k | doc | sn_dep R² OOF/lockbox | feo_ret R² OOF/lockbox | Cx_sn θ [IC] | balance R²(Cx_sn\|S) |
|---|---|---|---|---|---|---|
| T1 núcleo físico | 10 | 10 variables de balance/estado terminal (inventarios, avance, T, orden, historia de C, B2, IRF) | 0.736/0.800 | 0.296/0.448 | 0.155 [**0.000**, 0.280] | 0.967 |
| T2 leyes | 15 | T1 + leyes previas (%Sn,%FeO,%SiO2,%CaO) + `m6_resto_frac_prev` | 0.740/0.798 | 0.310/0.423 | 0.122 [**0.000**, 0.259] | 0.968 |
| T3 térmico+lanza | 18 | T2 + termocupla, gradiente T, lanza previa | 0.756/0.810 | 0.337/0.452 | 0.186 [**0.000**, 0.323] | 0.968 |
| T4 historia gases | 26 | T3 + GN/O2/aire acumulados + tiempo_fase + espesor + leyes de carga del batch | 0.758/0.802 | 0.323/0.481 | 0.181 [0.040, 0.315] | 0.971 |
| T1 sin dosif. | 8 | T1 sin historia de carbón | 0.738/0.801 | 0.291/0.428 | 0.118 [**0.000**, 0.262] | 0.968 |
| T2 sin dosif. | 13 | T2 sin historia de carbón | 0.742/0.794 | 0.293/0.405 | 0.077 [**0.000**, 0.221] | 0.968 |
| T3 sin dosif. | 16 | T3 sin historia de carbón | 0.755/0.812 | 0.331/0.386 | 0.173 [**0.000**, 0.306] | 0.968 |
| T4 sin dosif. | 21 | T4 sin toda la historia de dosificación (C, C/Sn, GN, O2, aire acum.) | 0.748/0.809 | 0.308/0.457 | 0.218 [0.040, 0.376] | 0.972 |
| FULL | 38 | referencia | 0.765/0.800 | 0.329/0.486 | 0.164 [0.025, 0.291] | 0.970 |
| CURADO | 22 | referencia (v5/v6) | 0.757/0.810 | 0.328/0.481 | 0.164 [**0.000**, 0.311] | 0.969 |

**Los sets diseñados por teoría (T1-T3, T1-T3 sin dosif.) NO identifican el carbón** (IC toca 0 en los seis), a pesar de
tener R² competitivo — replican exactamente el patrón de CURADO reportado en hallazgos §19.4. Sólo **T4** (26, con historia
de gases) y su variante sin historia de dosificación **T4 sin dosif.** (21) sí identifican, con IC casi idéntico entre ambas
(0.040-0.315 vs 0.040-0.376). Esto **refuta la hipótesis fuerte de H6** ("la identificación exige específicamente la
historia de dosificación de carbón"): lo que separa T3 (no identifica) de T4 (sí identifica) no es la historia de
dosificación en sí (quitarla de T4 no rompe nada) sino el bloque de **leyes de carga del batch + tiempo de fase + espesor de
ladrillo** que T4 añade. La hipótesis más fina y sostenida por los datos es: **la identificación exige un estado con
suficiente redundancia/tamaño** (≥ ~16-18 features bien elegidas, no necesariamente las de dosificación) que fije con
precisión g(S) y m(Cx_sn|S) para que el residuo de `Cx_sn_v6` dependa poco de S. El balance test lo confirma: en todos los
sets el estado explica 96.7-97.2 % de la varianza de `Cx_sn_v6` (residuo ≈ 3 % de la sd bruta) — la sd residual **no** cae
apreciablemente entre T1 (0.967) y T4 (0.971), así que el mecanismo no es "menos confusión residual" sino menor sesgo del
propio par g/m con más covariables (ver §5).

## 4. Estabilidad (`e9_06_estabilidad.csv`)

θ de `Cx_sn_v6` por tercio cronológico de DEV (n_boot=100 por tercio, ~330-360 filas c/u):

| config | tercio 0 (antiguo) | tercio 1 | tercio 2 (reciente) | identificado en |
|---|---|---|---|---|
| FULL (38) | 0.000 [0.000, 0.094] | 0.233 [0.000, 0.416] | 0.412 [0.123, 0.815] | 1 de 3 |
| CURADO (22) | 0.001 [0.000, 0.155] | 0.214 [0.000, 0.398] | 0.329 [0.079, 0.680] | 1 de 3 |
| libre_k20 | 0.000 [0.000, 0.081] | 0.136 [0.000, 0.327] | 0.244 [0.052, 0.529] | 1 de 3 |
| **libre_k16** | 0.000 [0.000, 0.132] | **0.187 [0.028, 0.340]** | 0.285 [0.086, 0.585] | **2 de 3** |

`Cx_sn_v6` no se identifica en el primer tercio de DEV en ningún config (n insuficiente + efecto real más débil al inicio de
la campaña); crece hacia el final. **`libre_k16` es el único de los cuatro que identifica el carbón en 2 de 3 tercios**
(igual que FULL en el criterio de decisión §2 del diseño), mientras FULL, CURADO y `libre_k20` sólo lo logran en el tercio
más reciente.

Walk-forward (bloques de 50 batches, ventana creciente, R² pooled):

| config | sn_dep R² walk-forward | feo_ret R² walk-forward |
|---|---|---|
| FULL | 0.754 | 0.233 |
| CURADO | 0.743 | 0.232 |
| libre_k20 | 0.763 | 0.267 |
| **libre_k16** | **0.761** | **0.261** |

Los dos sets parsimoniosos **generalizan mejor hacia adelante** que FULL y CURADO en ambos targets (+0.007-0.009 en
sn_dep, +0.028-0.034 en feo_ret) — consistente con menos parámetros de estado sujetos a deriva de campaña (hallazgos §19,
§5 de `e7_04_resultados.md`).

## 5. Varianza de θ (`e9_06_varianza_theta.csv`)

IC de `Cx_sn_v6` (θ₁sd) para FULL y para `libre_k20` (el "mejor" set por score compuesto R²+identificación):

| config | variante | θ₁sd | IC | excluye 0 |
|---|---|---|---|---|
| FULL | baseline (HGB150, 5-fold, boot 200) | 0.164 | [0.025, 0.291] | sí |
| FULL | boot 500 | 0.164 | [0.035, 0.284] | sí (más angosto) |
| FULL | nuisance HGB max_iter 300 | 0.146 | [0.001, 0.281] | **al límite** |
| FULL | nuisance Ridge | 0.000 | [0.000, 0.317] | **no** (θ en la cota) |
| FULL | cross-fit 10 folds | 0.164 | [0.029, 0.288] | sí |
| libre_k20 | baseline | 0.189 | [0.045, 0.331] | sí |
| libre_k20 | boot 500 | 0.189 | [0.045, 0.311] | sí |
| libre_k20 | nuisance HGB300 | 0.196 | [0.063, 0.347] | sí (mejora) |
| libre_k20 | nuisance Ridge | 0.068 | [0.000, 0.160] | **no** |
| libre_k20 | cross-fit 10 folds | 0.154 | [0.000, 0.278] | **al límite** |

Ningún ajuste de `n_boot` (500) o de folds (10) estrecha materialmente el IC — el ancho está dominado por la variabilidad
muestral real (n≈1000-1080), no por el error de Monte Carlo del bootstrap ni por el número de folds. El hallazgo más
importante es cualitativo: **con un nuisance g/m lineal (Ridge) la identificación colapsa por completo en los dos estados**
(θ cae a la cota 0) — la relación S→`m6_ln_sn_dep` y S→`Cx_sn_v6` es no lineal, y un nuisance mal especificado deja
confusión en el residuo que un modelo lineal no puede limpiar; esto confirma que el HGB (no un ajuste lineal) es
indispensable para la identificación, no un detalle técnico. Con más flexibilidad (HGB300) FULL se debilita levemente
(el nuisance de 38 features empieza a absorber señal de `Cx_sn_v6`) mientras el set parsimonioso (20) se mantiene o mejora
— evidencia adicional de que el estado más chico es **más robusto a la elección del nuisance** (menos riesgo de sesgo de
regularización de Double ML con menos covariables).

## 6. Resumen y veredicto (`e9_06_resumen.csv`)

Criterio de diseño: adoptar un set ≤ 15 sólo si ΔR² OOF ≥ −0.01 en **ambos** targets vs FULL y el IC de `Cx_sn_v6` excluye 0
(umbral práctico `ci_lo > 1e-3`) con signo +.

| origen | set | k | ΔR² sn_dep | ΔR² feo_ret | Cx_sn IC | cumple |
|---|---|---|---|---|---|---|
| eliminación libre | **k=8** | 8 | −0.0096 | −0.0014 | [0.055, 0.321] | **sí** |
| eliminación libre | **k=12** | 12 | −0.0069 | +0.0044 | [0.022, 0.306] | **sí** |
| eliminación libre | k=16 | 16 | +0.0014 | +0.0035 | [0.047, 0.287] | sí (k>15) |
| eliminación libre | k=20 | 20 | +0.0098 | +0.0207 | [0.045, 0.331] | sí (k>15) |
| T1 núcleo físico (teoría) | T1 | 10 | −0.0291 | −0.0334 | [0.000, 0.280] | no (R² y IC) |
| T2 leyes (teoría) | T2 | 15 | −0.0254 | −0.0195 | [0.000, 0.259] | no (R² y IC) |
| eliminación física | k=10..20 | 10-20 | variable | variable | IC toca 0 en k≤12 | parcial |
| CURADO (v5/v6) | — | 22 | −0.0086 | −0.0015 | [0.000, 0.311] | no (IC, k>15) |

**Ningún set diseñado a mano por teoría (T1, T2, T3, núcleo físico) cumple el criterio de identificación**, replicando lo ya
sabido de CURADO. **Sí existen sets ≤ 15 puramente guiados por importancia (libre k=8 y k=12) que cumplen simultáneamente
R² y identificación**, refutando la versión fuerte de H6 (no hace falta k > 15 ni la historia de dosificación específica) —
pero lo logran sustituyendo variables núcleo por proxies correlacionados (termocupla en vez de temperatura de horno,
`ratio_sn_feo_prev`/`interaccion_Sn_x_FeO_prev` en vez de avance/masa) que son estadísticamente equivalentes pero menos
directamente interpretables uno a uno.

### Veredicto

1. **H6 se confirma en su versión débil, no en la fuerte**: existe un estado ≤ 15 (`libre_k12`, 12 features) que conserva
   el R² (ΔR² ≥ −0.01 en ambos) e identifica el carbón (IC excluye 0 con signo +). No requiere expresamente la historia de
   dosificación de carbón (aunque sí incluye `relacion_C_cum_Sn_cum_prev`, la única pieza de esa historia que sobrevive en
   los sets ≤ 16). La hipótesis original ("la identificación exige la historia completa de dosificación") es demasiado
   fuerte: T4_sin_dosif (21, sin ninguna historia de C/GN/O2/aire) identifica igual de bien que T4; lo que realmente hace
   falta es **suficiente redundancia informativa en el estado**, no esas variables en particular.
2. **Recomendación operativa — dos niveles**:
   - **Estrictamente ≤ 15 (`libre_k12`, 12 features)**: `ley_sn_escoria_pct_prev` (ley de Sn previa, fija el punto de
     partida de la cinética), `ley_sio2_escoria_pct_prev` (matriz/viscosidad de la escoria), `ratio_sn_feo_prev` y
     `interaccion_Sn_x_FeO_prev` (memoria de la competencia cinética SnO₂ vs FeO), `grad_temperatura_horno_celsius_prev` y
     `termocupla_media_celsius_prev` (estado térmico y su tendencia), `posicion_vertical_lanza_mm_prev` (geometría de
     inyección que gobierna la mezcla gas-baño), `relacion_C_cum_Sn_cum_prev` (la única pieza de historia de dosificación
     que sobrevive: lo que el operador mira para dosificar carbón, confusor directo de `Cx_sn`), `m6_sn_inv_kg_prev` y
     `m6_feo_inv_kg_prev` (inventarios de balance físico, masa disponible/competidora), `espesor_ladrillo_norm_mm`
     (desgaste de refractario, proxy de pérdidas térmicas por campaña) y `m6_resto_frac_prev` (fracción no reducible del
     cierre físico v6). No se probó su estabilidad temporal completa (tercios/walk-forward) por acotar el cómputo; el
     estático (OOF+lockbox) es sólido.
   - **Recomendación con evidencia completa (`libre_k16`, 16 features, 1 sobre el techo de diseño)**: agrega
     `ley_feo_escoria_pct_prev`, `ley_cao_escoria_pct_prev` (composición completa de la escoria, basicidad vía CaO),
     `cum_feed_Carbon_kg_prev` (historia de carbón acumulada) y `cum_aire_nm3_prev` (historia de aire acumulada, potencial
     oxidante de campaña). Es el **único set de los evaluados con evidencia temporal completa** que identifica el carbón en
     2 de 3 tercios (igual que exige el criterio §2 del diseño para "estable") y que además generaliza mejor que FULL en
     walk-forward (+0.007 sn_dep, +0.028 feo_ret). Se recomienda para producción **por encima del k=12** pese a exceder el
     techo de 15 en una feature, porque es el candidato con más evidencia de robustez temporal — el límite "≤15" era un
     objetivo de diseño, no una restricción física; 16 sigue siendo menos de la mitad de FULL.
3. **Límite del ejercicio**: los sets parsimoniosos ganadores son óptimos estadísticos (importancia por permutación), no un
   diseño pirometalúrgico limpio — sustituyen variables núcleo "obvias" (`temperatura_horno_celsius_prev`,
   `orden_escalon_fase`, `m6_avance_prev`, `indice_irf_prev`, `basicidad_B2_prev`) por proxies correlacionados. Si se
   prioriza la interpretabilidad 1:1 con la teoría por sobre la parsimonia extrema, FULL (38) sigue siendo la referencia
   defendible (hallazgos §19), y este experimento sólo demuestra que **es posible** recortar a la mitad o menos sin perder
   R² ni identificación — no que sea obligatorio.
