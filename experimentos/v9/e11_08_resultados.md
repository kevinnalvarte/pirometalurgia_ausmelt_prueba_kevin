# E11-08 — ¿La moderación térmica se ve en el ESCALÓN? (2026-09-16)

Scripts: `e11_08_01_escalon_moderacion.py` (tareas 1-3, n≈1448 escalones R0-R3, 724 Rt + 724 Rl),
`e11_08_02_batch_moderacion.py` (tarea 4, n=362 batches, tabla `e11_06b_tabla_batch.csv`).
CSV: `e11_08_tabla1_terciles.csv`, `_cuartiles.csv`, `_continua.csv`, `_diferencia_regimenes.csv`,
`e11_08_tabla2_idx_vs_T.csv`, `_dentro_tercios.csv`, `e11_08_tabla3_proxies.csv`,
`e11_08_tabla4_batch_T_vs_idx.csv`, `_resumen_ganador.csv`.

## Veredicto corto

**La moderación térmica que domina a nivel BATCH (ventana móvil de 80) NO se reproduce con solidez a
nivel de ESCALÓN individual.** El único canal que sobrevive DEV+TOTAL con el mismo signo y p<0.05 en
ambos es **GN→dross moderado por T, a nivel BATCH** (tarea 4) — y ahí T gana claramente sobre el tiempo
de campaña (idx). A nivel de escalón (tareas 1-2), ninguna interacción continua A_res×zT pasa el
criterio de réplica DEV/TOTAL; sólo un contraste terciles frío-vs-caliente (GN→FeO_ret) es significativo
en DEV (p=.044) pero no en TOTAL (p=.224) — frágil. No hay proxy de polvo/arrastre defendible: la única
interacción robusta (GN_res×zT sobre `tiro_horno_pct`) probablemente refleja mecánica de tiro/flujo de
gas, no arrastre de polvo. No hay umbral T con IC razonable que soporte una regla operativa dura.

## 1-2. Escalón: terciles/cuartiles de T, continuo, y T dentro de tercios cronológicos

Specs (residuos cross-fitted `theta_dual`, `S`=`ESTADO_R_V8`+T+orden): (a) `Cx_sn_v6` y GN sobre
`m6_ln_sn_dep`; (b) carbón crudo (`tasa_feed_Carbon_kg_min`) y GN sobre el mismo target; (c) GN sobre
`m8_ln_feo_ret_dross50`. Terciles de T (TOTAL): frío ~970 °C, medio ~1049 °C, caliente ~1135 °C (n≈442
c/u); DEV similar (frío 986/medio 1072/caliente 1144).

- **Interacción continua A_res×zT** (`e11_08_tabla1_continua.csv`): ningún término significativo en
  DEV **y** TOTAL a la vez (p entre .18 y .98). El signo ni siquiera es estable entre muestras (p.ej.
  GN sobre Sn: +0.0002 TOTAL vs −0.0020 DEV, ambos n.s.).
- **Terciles** (`e11_08_tabla1_terciles.csv`): GN→FeO_ret es negativo y significativo en frío y medio
  (TOTAL: frío −0.003 p=.027, medio −0.003 p=.014; DEV: frío −0.003 p=.023, medio −0.004 p=.003) pero
  ≈0 y n.s. en caliente (TOTAL 0.000 p=.58; DEV −0.001 p=.89) — dirección coherente con "GN vale menos/
  daña menos cuando el horno está caliente", pero la diferencia frío−caliente sólo es significativa en
  DEV (−0.0036, IC [−0.0071,−0.0003], p=.044) y **no en TOTAL** (−0.0018, IC [−0.0047,0.0014], p=.224):
  no replica, no pasa el criterio de la iteración 11 (mismo signo Y significancia en ambos regímenes).
  GN y carbón sobre Sn: sin gradiente consistente por tercil ni diferencia frío-caliente significativa
  en ninguna muestra (p=.32-.82).
- **T vs. tiempo de campaña en el mismo modelo** (`e11_08_tabla2_idx_vs_T.csv`): al añadir A_res×z(idx)
  junto a A_res×zT, ninguno de los dos gana de forma consistente; los t más altos (DEV, Cx_sn×zidx t=2.13
  p=.033; carbón crudo×zidx t=1.89 p=.059) apuntan a **tiempo**, no a T, y no replican en TOTAL. Esto es
  la comprobación pedida: al separar T de tiempo dentro del escalón, ninguno de los dos domina con
  solidez — la señal es débil en ambos ejes a esta escala.
- **T dentro de tercios cronológicos** (`e11_08_tabla2_dentro_tercios.csv`): sólo un caso pasa p<0.05
  (carbón crudo, tercio tardío de campaña, TOTAL: coef 0.0086, t=2.87, p=.004) pero **no replica en DEV**
  (mismo tercio: coef 0.0047, t=1.17, p=.24). El resto de combinaciones (9 palanca×tercio×spec) son n.s.
  Conclusión: dentro de un mismo tramo cronológico (controlando el "reloj"), la T residual no modula de
  forma reproducible ninguno de los tres vínculos de escalón estudiados.

## 3. Proxies de polvo/arrastre en línea (`e11_08_tabla3_proxies.csv`)

Columnas candidatas en `feature_engineering.py`: `tiro_horno_pct`, `temperatura_gas_pre_bhf_celsius`,
`temperatura_gas_pre_cuchilla_celsius`, `delta_T_gas_pre`, `presion_punta_lanza_kpa`,
`presion_suministro_gn/o2_lanza_kpa`, `contrapresion_gn/aire_lanza_kpa`, `margen_presion_gn` (estas 3
últimas muy colineales con GN mismo, se excluyen de la prueba). Se probó `GN_res × zT` (mismo `S`,
A=GN+exo2+aire+carbón) sobre cada proxy:

- `tiro_horno_pct`: efecto principal de GN robusto y fuerte (TOTAL t=3.58 p=.0003; DEV t=3.46 p=.0005) —
  esperado mecánicamente (más GN → más caudal de gas → más tiro). La interacción GN×zT es significativa
  en **ambas** muestras (TOTAL coef .150 t=2.36 p=.019; DEV coef .144 t=2.44 p=.015): en caliente el GN
  empuja más el tiro. Es el único proxy que replica, pero es dudoso como proxy de "arrastre de polvo":
  puede ser sólo mecánica de flotabilidad del penacho de gases, no evidencia de fuming/arrastre.
- `presion_punta_lanza_kpa` y su delta: interacción con signo negativo pero **no replica**
  (TOTAL p=.070/.158; DEV p=.145/.090 — significancias marginales que cambian de caso).
- `temperatura_gas_pre_bhf/cuchilla_celsius`, `delta_T_gas_pre`: sin interacción significativa en
  ninguna muestra.
- **Veredicto de la tarea 3**: no hay proxy de polvo/arrastre defendible con moderación térmica robusta;
  `tiro_horno_pct` es el candidato más fuerte pero su interpretación causal es ambigua (mecánico vs.
  arrastre) y no se recomienda usarlo para una regla operativa.

## 4. Nivel batch: xG/xC × zT vs. × z(idx) (`e11_08_tabla4_batch_T_vs_idx.csv`, resumen en `_resumen_ganador.csv`)

Modelo por outcome (KPI, f_dross, f_polvo, f_metal, Sn perdido en escoria, en pp): `y ~ xG + xC +
xG:zT + xC:zT + xG:zidx + xC:zidx + controles` (sin espesor, colineal con T/idx). HC3 + bootstrap (500,
no cluster).

| Outcome | Interacción que gana | t (DEV) / t (TOTAL) | p (DEV) / p (TOTAL) | Replica |
|---|---|---|---|---|
| f_dross | **xG × zT** | −2.01 / −2.21 | .044 / .027 | **Sí**, mismo signo, T > idx en ambas muestras |
| f_polvo | xG × zT (no sig.) | 1.86 / 1.27 | .063 / .204 | Parcial (sólo marginal en DEV) |
| f_metal | ninguna | ≤0.68 en ambas | >.49 | No |
| KPI | ninguna | ≤1.0 (xC×zT) | >.16 | No |
| Sn perdido en escoria | xC × zidx (DEV sólo) | −2.50 (DEV) / −0.46 (TOTAL) | .012 / .696 | No |

`xG × zT` sobre `f_dross` es la **única interacción de la tarea 4 que replica con el mismo signo y
p<0.05 en DEV y en TOTAL**, y gana claramente sobre `xG × zidx` (t=−1.02/−1.81, p=.31/.070) en ambas
muestras: el efecto conocido "GN → dross ↑" (coef principal xG: DEV +1.12 p=.0004, TOTAL +0.86 p=.0045)
se **atenúa cuando el horno está más caliente** (zT alto) — coherente con la deriva de la ventana móvil
del enunciado (β_GN→dross positivo siempre, pero de magnitud decreciente cuando el horno estuvo más
caliente al inicio de campaña). El resto de outcomes no muestra un moderador ganador robusto.

## 5. Síntesis física: palanca × régimen térmico

| Palanca (nivel) | Caliente (>~1070 °C) | Medio (~1000-1070) | Frío (<~1000 °C) | Coherente con teoría | Solidez |
|---|---|---|---|---|---|
| GN → dross (BATCH) | efecto positivo pequeño/atenuado | efecto positivo pleno | efecto positivo pleno/mayor | Sí (fuming/no-selectividad más dañina en frío-medio) | **Alta** (replica DEV+TOTAL) |
| GN → FeO_ret (ESCALÓN) | ≈0, n.s. | negativo, sig. | negativo, sig. pero dif. vs. caliente sólo sig. en DEV | Parcial | Baja-media |
| GN → Sn_dep (ESCALÓN) | n.s. | mejor punto puntual (n.s. gradiente limpio) | n.s. | No concluyente | Baja |
| Carbón (Cx_sn/crudo) → Sn_dep (ESCALÓN) | n.s. | n.s. | n.s. (un hallazgo aislado no replica) | No concluyente | Baja |
| GN → tiro (proxy polvo, ESCALÓN) | tiro más sensible al GN | — | tiro menos sensible | Ambiguo (¿mecánico o fuming?) | Media (replica número, no interpretación) |

**Reglas dependientes de T que se sostienen con significancia**: ninguna con IC de umbral razonable.
Lo único defendible cualitativamente: *"el valor negativo de GN sobre el canal dross se diluye cuando el
horno arranca el tramo más caliente (T_prev alto); no se diluye el efecto sobre FeO/Sn a nivel de
escalón individual, donde el ruido domina."* No se recomienda codificar un umbral T_prev fijo en el
prescriptor a partir de esta iteración — la señal que sí sostiene el comité (batch, ventana móvil) es de
naturaleza agregada/de tendencia (posible confusión residual con el reloj de campaña vía composición
mineral u otra variable no observada), y el escalón no aporta la prueba independiente que se buscaba.

## Nota de honestidad

Se probaron 2 muestras (DEV, TOTAL) × 3 specs × 2 palancas × {terciles, cuartiles, continuo, T+idx,
dentro-de-tercios} = ~72 pruebas a nivel escalón, más 7 proxies × 2 términos × 2 muestras = 28 a nivel
proxy, más 5 outcomes × 6 términos × 2 muestras = 60 a nivel batch. Con ese número de pruebas, los 2-3
hallazgos puntuales con p≈.02-.04 que NO replican (terciles GN→FeO en TOTAL, C crudo tercio tardío en
DEV, presión de punta) son consistentes con falsos positivos esperados por azar y se descartan. Sólo se
reporta como sólido lo que replica en DEV y TOTAL con el mismo signo: GN×zT sobre f_dross a nivel batch.
