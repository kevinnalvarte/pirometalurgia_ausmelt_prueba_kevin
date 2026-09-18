# E15-02 — Auditoría adversarial de E15-01 (receta ex-ante de "Alimentación")

Objetivo: romper la afirmación de E15-01 (d_gn −1.58 pp t−3.9 en DEV; prospectivo JS/OLS W=100
pendiente 0.56–0.63, t 2.6–2.7, p_perm 0.001–0.003, p_circ 0.006, lockbox≈0). Scripts
`e15_02_0{1..8}_*.py`, lib `e15_02_lib.py`. Se reutiliza `e11_07_lib`/`e13_03_lib` (mismo motor JS y
nulas que la auditoría de E13-03).

## 1. Fuga en la estandarización (`e15_02_01_*.csv`)

E15-01 usa media/sd de **todo** DEV (constante) para estandarizar el desvío — incluye el futuro de cada
batch temprano. Reestandarizando ESTRICTO (expansivo, solo pasado, ≥40 batches previos):

| Variante | n | JS: t (p_perm/p_circ) | OLS: t (p_perm/p_circ) |
|---|---|---|---|
| Original E15-01 (DEV cte) | 261 | 2.585 (.002/.006) | 2.740 (.003/.006) |
| Expansivo centrado (≥40) | 222 | 2.042 (**.007/.017**) | 2.044 (**.013/.027**) |
| Expansivo sin centrar | 222 | 2.535 (.007/.010) | 2.381 (.006/.021) |
| Crudo físico (Nm3/min, sin estandarizar) | 261 | 2.442 (.004/.006) | 2.675 (.004/.009) |
**Grieta:** la versión estricta y centrada pierde 39 batches (calentamiento) y el t cae de ~2.6 a ~2.0;
con OLS sin contracción la p corregida cruza 0.01 (.013/.027). Las unidades físicas crudas sobreviven
casi intactas — la fuga afecta más al centrado que a la escala. Coeficiente DEV: original b_gn=−1.58
(t−3.90); crudo b_gn=−0.57 pp/(Nm3/min) (t−3.64). **En lockbox el signo se invierte** en las tres
variantes (b_gn > 0, t 1.4–1.8) — ver §7.

## 2. ¿Receta ex-ante o ex-post? (`e15_02_02_*.csv`)

**(a-b) Niveles discretos y corridas:** aire tiene solo 2 valores distintos en TODA la campaña (corridas
de 181 batches); carbón-R02 solo 3 valores (corrida media 90.5 batches); GN 6–13 valores/orden (corridas
9.5–27.8 batches). Es una **tabla de planta revisada por tramos calendario**, no un valor ajustado
batch-a-batch — descarta con alta confianza el llenado ex-post individual.
**(c) La receta SÍ responde al estado pre-Reducción** (legítimo, no fuga): `p_carbon ~ feed_sn_total_t +
ley_sn_conc + frac_carga_secundaria_F + feed_dross_Fe_total_t + F_lnirf_F0`: R²=0.238, p_perm=0.001;
`p_gn` R²=0.088, p_perm=0.001. El planificador dosifica carbón según carga/ley — esperable.
**(d) El DESVÍO correlaciona con el KPI del batch anterior:** d_gn vs `kpi_prev` corr=−0.137, **p=0.0094**.
Tras un batch malo el operador tiende a subir GN por encima de receta. Riesgo de confusión inversa
(reversión a la media + compensación) con el mismo signo que el efecto reclamado. El resto de variables
pre-tratamiento no correlaciona con d_gn/d_carbon de forma significativa (p≥0.046, marginal en un caso).

## 3. Multiplicidad de nulas (`e15_02_03_*.csv`)

Grid W∈{60,80,100,120,150,expansiva}×{JS,OLS} = 12 combos, p corregida (máximo t) por tipo de nula:

| Nula | p_W100(JS) | p corregida (12 combos) |
|---|---|---|
| Bloques de 10 | 0.012 | **0.012** |
| Bloques de 20 (primaria, E15-01) | 0.002 | 0.002 |
| Bloques de 40 | 0.004 | 0.004 |
| Circular exhaustivo | 0.006 | 0.006 |
| Signo por bloque de 20 | 0.006 | 0.002 |
Igual patrón que E13-03: bloques de 10 es la nula más exigente y única que cruza 0.01.

## 4. Grados de libertad del analista (`e15_02_04_gl_analista.csv`)

162 combos (paso∈{5,10,20}×start∈{100,120,150}×controles∈{+idx,sin idx,espesor}×6 juegos de exposición),
JS W=100, nula bloques20 (300 rep):
- **dG+dC (27):** 93% p<.01, 100% p<.05 — robusto. **Solo dG (27):** 78% / 100%.
- **Solo dC (27):** **0% / 0%** — el carbón solo nunca sostiene la prueba (igual que xC en E13-03/E11-07).
- **dG solo R0-R1 (27):** 59% / 100%. **dG solo R2-R3 (27):** 41% / 96% — al partir por mitad de
  escalones el resultado se debilita marcadamente; no está limpiamente localizado en un sub-tramo.
- **dG+dC+dO2+dAire (27):** 96% / 100% — agregar canales sube el % pero es más grados de libertad.
- **Global: 61% de las 162 combos con p<.01, 83% con p<.05.**
## 5. Placebos (`e15_02_05_*.csv`)

| Placebo | pendiente | t | p_perm |
|---|---|---|---|
| lead1 (exposición del batch siguiente) | 0.855 | **1.829** | **0.022** |
| lead3 | 0.321 | 0.768 | 0.232 |
| lag1 (exposición del batch anterior) | 0.162 | 0.202 | 0.309 |
| kpi_prev (outcome = KPI del batch anterior) | 0.218 | 0.878 | 0.212 |
| **Limpio de persistencia** (+kpi_prev, d_gn_prev, d_carbon_prev) | 0.608 | 2.466 | 0.002 |
A diferencia de E13-03 (lead1 con JS colapsaba a t≈0.05), aquí **lead1 no colapsa** (t=1.83, p=0.022):
hay eco de corto alcance no trivial. Pese a ello, el contraste que controla directamente por persistencia
sostiene casi intacto el resultado principal (t 2.47 vs 2.59) — la correlación dev↔kpi_prev de §2(d) no
parece explicar por sí sola el hallazgo.

## 6. Fragilidad (`e15_02_06_*.csv`)

Jackknife (300 rep): quitando 5 → 100% t>1.645; quitando 10 → 92%; quitando 20 → 97%. Leave-one-block-out
(19 bloques de 20): t entre 1.887 y 3.043, **ningún bloque** cae bajo 1.645. Mucho más robusto a nivel de
batch/bloque que el xG,xC de E13-03 (que caía a 59–68%) — la fragilidad de esta afirmación NO está aquí.

## 7. Por qué lockbox ≈ 0 (`e15_02_07_*.csv`)

Trayectoria de beta_gn (OLS crudo, ventana dura W=100): débil al inicio (t −0.8 a −1.8), se fortalece en
el tramo medio de DEV (t hasta **−3.53**), se debilita al final de DEV/lockbox y llega a **t≈−0.03 en
las dos últimas ventanas** (las que puntúan los ~20 batches finales). Por régimen: DEV_t1 b_gn=−0.53
(t−0.84, ns); DEV_t2 b_gn=−3.38 (t−3.67); DEV_t3 b_gn=−1.07 (t−1.23, ns); **LOCKBOX b_gn=+1.36 (t+1.82) —
el signo se invierte.** De 7 ventanas que puntúan lockbox, 4 tienen |t_gn|<1 (el asesor se habría
abstenido, igual que v9). El lockbox≈0 no es solo ruido: es un coeficiente que decae a cero justo cuando
debe extrapolarse, con relación estática de signo contrario a la de DEV — no hay estabilidad de régimen.

## 8. Magnitud honesta (`e15_02_08_*.csv`)

Calibración (pendiente del score JS W=100), bootstrap bloques de 10 (2000 rep): observado 0.630, **IC95
[0.247, 0.997]**, P(>0)=1.0 — el punto cae en [0.6,1.4] pero el IC baja muy por debajo de 0.6.
Ganancia calibrada (coef. DEV bootstrap bloques10): ejecutar GN 1 sd por debajo de lo ejecutado →
**+1.47 pp KPI, IC95 [0.68, 2.23], P(>0)=1.0**. Ejecutar EXACTAMENTE la receta (dev=0) frente al
comportamiento DEV real promedio → **−0.006 pp, IC95 [−0.20, 0.18], P(>0)=0.475 (nulo)**: el desvío medio
real ya es ≈0 (mean d_gn=−0.004), así que "cumplir la receta en promedio" no rinde nada — la ganancia
reclamada exige operar por DEBAJO de receta, no simplemente seguirla.

## 9. Veredicto del auditor

**Criterio "p corregida<0.01 estable": NO se sostiene de forma uniforme.** Con la nula y estandarización
exactas de E15-01 da 0.002–0.006 según la nula (§3), pero: (a) la estandarización estricta (solo pasado)
empuja la variante centrada a 0.013–0.027 (§1); (b) sobre 162 combinaciones razonables del analista solo
61% da p<.01 (§4); (c) el carbón solo nunca es significativo (0%) — todo el peso recae en GN; (d)
partiendo por sub-tramo de escalones (R0-R1 vs R2-R3) el % cae a 41–59%. La p honesta a reportar es
**≈0.01–0.03** (no <0.01 de forma estable), mejor que el 0.01–0.02 de E13-03 en la nula primaria pero
peor en la sensibilidad a la estandarización y a la partición por orden.
**Criterio "calibración en [0.6,1.4]": se sostiene en el punto (0.63) pero no en el intervalo** — el IC95
bootstrap baja a 0.25, compatible con sub-calibración severa.
**Lo nuevo respecto de E13-03:** (1) la fragilidad batch-a-batch es sólida (§6, no es el punto débil);
(2) el punto débil real es la NO estabilidad temporal — el coeficiente decae a cero y se invierte de
signo en lockbox (§7), y la exposición GN es endógena de corto plazo a la performance previa (§2d, aunque
sobrevive el control directo); (3) el placebo lead1 no colapsa como en E13-03 (t=1.83, p=.022), señal de
autocorrelación residual sin depurar del todo.
**Recomendación:** no declarar cumplido el criterio 1. Usar p≈0.01–0.03, calibración 0.63 con IC ancho
[0.25,1.0], y exigir estabilidad de régimen (§7) antes de considerar esto más sólido que v9/E13-03 — son
auditorías con grietas distintas pero de severidad comparable.
