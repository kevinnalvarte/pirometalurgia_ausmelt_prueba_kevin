# E12-01 — Duración de R3 y F6 como palanca (criterio de corte)

Diseño: `ITERACION_12_diseno.md` H1. R3 y F6 ocurren exactamente una vez por batch (n=362), así que la exposición
zD = residuo estandarizado de la duración es directamente una exposición de batch (sin agregación). Scripts:
`e12_01_construir.py` (residuos), `e12_01_descriptivo.py` (tarea 1), `e12_01_escalon.py` (tarea 2, mecanismo),
`e12_01_batch.py` (tarea 3), `e12_01_placebos.py` (tarea 4), `e12_01_prospectivo.py` (tarea 5).

## 1. Descriptivo

Duración real DEV: R3 media 14.1 min (sd 8.6; mediana 12; P95 24.1; máx 125 — versus plan fijo 13). F6 media 73.7 min
(sd 15.9; mediana 70; P95 103; máx 188, plan 58-100 típico). Outliers (desvío > 30 min sobre plan): R3 4/362 (1.1 %),
F6 32/362 (8.8 %); winsorizando a P1-P99 la sd de R3 baja de 8.1 a 5.7 min y la de F6 de 15.2 a 13.6 (filtrando >30 min
de desvío: sd 4.5 y 9.9). Sin relación con la hora del día ni el turno (Spearman duración~hora: R3 ρ=0.02 p=0.68,
F6 ρ=-0.05 p=0.38; medias por turno 00-08/08-16/16-24 prácticamente iguales) — consistente con que el corte lo decide
el laboratorio y no un patrón operativo por turno. R3 sí cae con `idx_cronologico` (ρ=-0.32, p<0.0001): el escalón se
acorta a lo largo de la campaña (se controla en todas las regresiones batch vía `idx_cronologico`).

**HGB cross-fitted dur ~ estado al inicio del escalón**: R3 (`L.ESTADO_R`) R² OOF DEV = **0.150**, R² lockbox = -0.36
(sd del residuo 8.0 min) → la mayor parte de la variación de R3 es exógena al estado conocido, favorable para
identificación. F6 (`L.ESTADO_F`) R² OOF DEV = **0.394**, R² lockbox = 0.19 (sd residuo 11.9 min) → F6 es bastante más
predecible desde el estado (proceso, no sólo laboratorio) — advertencia ya visible aquí de que F6 es un cuasi-experimento
más débil que R3.

## 2. Mecanismo de escalón (R3)

Residuo-sobre-residuo con `ESTADO_R_V8` (HGB cross-fitted, un solo orden ⇒ señal escasa: R² OOF de los targets sobre
el estado es NEGATIVO para m6_ln_sn_dep (-0.31), m8_ln_feo_ret_dross50 (-0.12) y los deltas en kg — el nuisance model
casi no explica nada dentro de un único orden, así que el residuo ≈ el target crudo menos el estado).

- `resid(m6_ln_sn_dep) ~ zD_R3` (DEV): coef +0.036 (t=2.29, p=0.022) — más minutos exógenos → más Sn agotado (signo
  correcto). `resid(m8_ln_feo_ret_dross50) ~ zD_R3`: coef -0.0006 (t=-0.16, n.s.) — **sin efecto detectable sobre el
  FeO reducido**, contra lo esperado por H1 (se esperaba más FeO reducido también). ΔT: n.s.
- En kg: Sn extraído (`resid(d_sn_kg) ~ res_dur_min`) +5.14 kg/min (t=2.33, p=0.020); FeO reducido +0.14 kg/min
  (t=0.03, n.s.). El ratio Sn/FeO "por minuto extra" (≈37) **no es interpretable**: el denominador no es distinto de 0.
- Al agregar minuto×tasa de carbón / minuto×tasa de GN como regresores adicionales, todo pierde significancia
  (colinealidad fuerte con `res_dur`: esas variables son `res_dur × tasa`) — no aportan sobre el efecto simple.
- Dosis-respuesta por quintiles de zD_R3 (DEV): `resid_dSn_kg` es mayormente monótono (Q1→Q5: -55, -20, -6, -21,
  +70 kg) pero con un quiebre en Q4; `resid_dFeO_kg` NO es monótono (-47, -30, -72, +28, +34 kg) — ruido.
- Interacción con `ley_sn_escoria_pct_prev` (%Sn al inicio de R3): `res_dur × z(%Sn)` no identificado ni en Sn
  (t=-0.51) ni en FeO (t=0.19). Por tercil de %Sn: el coeficiente Sn_kg/min es positivo pero con t<1 salvo en el
  tercil alto (t=2.12); el de FeO_kg/min nunca es significativo salvo un signo positivo aislado en el tercil medio
  (t=1.70). **No hay un patrón de moderación limpio** — el canal FeO simplemente no se mueve con la duración de R3.

Conclusión del mecanismo: hay indicio (débil, un solo predictor significativo con p≈0.02, n=203-286) de que más
minutos agotan algo más de Sn; no hay evidencia de que agoten más FeO dentro de R3. Eso contradice la premisa del
"óptimo interior" de H1 (se necesitaría que ambos canales se movieran).

## 3. Batch (KPI y canales)

OLS HC3, `KPI ~ xG + xC + zD_R3 + zD_F6 + controles` (sin espesor):

| muestra | zD_R3 coef (t, p) | zD_F6 coef (t, p) |
|---|---|---|
| DEV (n=286) | +0.369 (1.50, 0.133) | +0.226 (0.85, 0.398) |
| LOCKBOX (n=62) | +0.348 (0.29, 0.769) | **-1.611 (-2.12, 0.034)** |
| TOTAL (n=348) | +0.399 (1.21, 0.227) | +0.033 (0.13, 0.896) |

zD_R3: signo positivo estable en DEV/TOTAL/lockbox y en los 3 tercios cronológicos (+0.49/+0.73/+0.22) pero **nunca
p<0.05**; bootstrap por batch (1000): P(coef>0)=0.91 DEV / 0.85 TOTAL, IC95% [-0.33, 0.90] (DEV) — incluye 0.
zD_F6: **signo inestable** (positivo en DEV, fuertemente negativo y significativo en lockbox, ~0 en TOTAL; tercios
-0.07/-0.21/+0.75) — no pasa el criterio "mismo signo en DEV y total". Canales (DEV): f_dross -0.0032 (t=-1.66,
p=0.097) y f_metal +0.0037 (t=1.69, p=0.092) con zD_R3 — dirección coherente (más tiempo → algo menos dross, algo
más metal) pero sólo marginal.

Forma cuadrática + interacción `zD_R3² ` y `zD_R3×z(%Sn_R3)`: ningún término es significativo (todos \|t\|<1) en DEV
ni en TOTAL. El "corte" de %Sn implícito (`-b1/b3`) da un punto ≈2.8 % Sn (dentro del rango histórico 0.45-4.08 %)
pero el IC bootstrap es **[-4.8, 13.3] en DEV** ([-9.9, 11.5] en TOTAL) — no identificado; ni el signo del término de
interacción es estable entre bootstraps (85 %/71 % de acuerdo). **No hay evidencia de óptimo interior en la duración
de R3.**

## 4. Placebos

Leads (zD_{t+1}, zD_{t+3} → KPI_t): todo n.s. (\|t\|≤1.47) para zD_R3 y zD_F6 — limpio. zD_R3 ~ pre-tratamiento
(ley_sn_conc, feed_sn_total, F_lnirf_F0, KPI_prev): todos n.s., F conjunto p=0.69 (DEV) — **zD_R3 se comporta como
exógeno**. **zD_F6 ~ pre-tratamiento FALLA**: `feed_sn_total_t` predice zD_F6 con t=5.94 (p<1e-4, DEV) y F conjunto
p<1e-4 — F6 dura más cuando el batch trae más Sn alimentado (tamaño de carga), algo que `ESTADO_F` no captura del
todo. **zD_F6 no es exógeno** en el mismo sentido que zD_R3. Confusión inversa: corr(zD_R3, %Sn al cierre de R2) =
0.05 (p=0.32, Pearson), -0.05 (p=0.31, Spearman) — el residuo no arrastra el estado de cierre de R2, la
residualización funciona.

## 5. Prueba prospectiva (W=100, paso 10, permutación en bloques de 20, n=1000)

| spec | t (TODO) | p_perm(t) | spearman | p_perm(rho) | t (DEV) | t (lockbox) |
|---|---|---|---|---|---|---|
| P0 {xG,xC} | 2.325 | 0.006 | 0.154 | 0.008 | 2.201 | 0.779 |
| P1 {xG,xC,zD_R3} | 1.979 | 0.013 | 0.154 | 0.008 | 2.473 | **-0.542** |
| P2 {+zD_F6} | 1.468 | 0.048 | 0.099 | 0.045 | 1.707 | **-0.416** |

Añadir zD_R3 **no mejora** la prueba prospectiva (t baja de 2.33 a 1.98 en el pool, spearman igual) y el t de lockbox
se vuelve negativo (-0.54); añadir zD_F6 la empeora más (t 1.47, spearman 0.10) y el lockbox sigue negativo. Ambas
extensiones fallan el criterio "mejora la prueba prospectiva sin empeorar lockbox" del método común.

## Veredicto

**No se adopta la duración de R3 ni la de F6 como palanca del asesor.** Resumen frente al diseño:

- R3 es efectivamente en gran parte exógeno (R² OOF de la duración desde el estado = 0.15; sin patrón por hora/turno;
  limpio en placebos de pre-tratamiento y confusión inversa) — el cuasi-experimento es razonable para R3.
- F6 NO lo es de la misma forma: R² OOF = 0.39 y falla el placebo de pre-tratamiento (correlaciona con
  `feed_sn_total_t`) — su variación de duración mezcla "cuánto tarda el laboratorio" con "cuánto hay que fundir",
  y además cambia de signo entre DEV y lockbox en la regresión de batch. **F6 se descarta.**
- Para R3: el mecanismo de escalón sólo sostiene, débilmente (p≈0.02, sin corrección por multiplicidad), el canal Sn
  (más minutos → más Sn agotado, ~5 kg Sn/min); el canal FeO nunca se mueve, así que **no hay evidencia del
  trade-off** que motivaba la hipótesis del óptimo interior/corte por %Sn. A nivel batch el signo de zD_R3 sobre el
  KPI es siempre positivo pero nunca significativo (p 0.13-0.77) y el bootstrap no excluye 0. No se identifica un
  %Sn de corte (IC del corte va de -5 a +13 %, fuera del rango observado). La prueba prospectiva, que es el
  requisito obligatorio del método común, **no mejora** al añadir zD_R3 (t 2.33→1.98; lockbox se vuelve negativo).
- Magnitud: aun con el signo más favorable (bootstrap DEV), la pendiente puntual es ≈+0.35-0.40 pp de KPI por sd de
  duración de R3 (sd ≈ 8 min) — del mismo orden que xG/xC, pero sin significancia ni mejora prospectiva no hay base
  para convertirlo en pp/min.
- Ejecutabilidad: aunque fuera identificado, cortar R3 "cuando ya no conviene" requeriría conocer el %Sn en línea, que
  es precisamente el dato que el laboratorio tarda en entregar (la razón de la variación de duración) — la regla de
  corte dependiente del %Sn que se buscaba no sería ejecutable sin ese mismo dato, y aquí ni siquiera se identifica.

En síntesis: H1 se **rechaza** para F6 (no exógeno) y **no se adopta** para R3 (exógeno pero sin señal batch/prospectiva
suficiente); el mecanismo apunta a un efecto pequeño y unilateral (sólo Sn) que no alcanza el umbral del método común.
