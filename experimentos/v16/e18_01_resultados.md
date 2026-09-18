# E18-01 — Identificación cuasi-experimental del efecto del GN (y carbón) con las revisiones de receta

Código: `experimentos/v16/e18_01_lib.py` (tabla batch + OLS-HC3/HAC + 2SLS manual, sin `linearmodels`) y
`e18_01_0{1..5}_*.py`. Datos: `asesor_receta_v10.escalones_con_receta()`. Motivación: el desvío
ejecutado−receta de GN se asocia a −1.58 pp/sd de KPI (t −3.9, DEV) y +1.23 pp/sd de dross (t +4.1) —
E15-01/E17. Objeción pendiente: el desvío lo decide el operador. Aquí se usa la RECETA (decisión del
planificador, no del operador ni del batch) como instrumento.

## 1. Las revisiones no responden a estado observable pre-revisión (`e18_01_01_*.csv`)

49 versiones de receta de GN (18 con ≥8 batches), 38 de carbón (17 con ≥8). Regresión del cambio de
receta (Δ media R0-R3, estandarizado) sobre el estado de los 10 batches previos a cada revisión (n=34
eventos GN, n=25 carbón): **ningún coeficiente significativo** (KPI previo, T horno, ley, dross previo;
|t| ≤ 0.9 en todos los casos, R² 0.056/0.118). El planificador no parece reaccionar a nada observable
antes de revisar — soporta (con poca potencia, n pequeño) el uso de la receta como variación exógena.

## 2. Primera etapa: el instrumento es débil, sobre todo para GN (`e18_01_02_primera_etapa.csv`)

F del instrumento propio (ejecutado ~ receta + controles + tendencia), por tendencia:

| Palanca | Régimen | Lineal | Cúbica | Bloques 40 |
|---|---|---|---|---|
| GN | TOTAL | 6.7 | 7.7 | 4.0 |
| GN | DEV | 6.9 | **10.4** | 4.8 |
| GN | LB (últimos 63) | 0.03 | 0.13 | 0.26 |
| Carbón | TOTAL | **242** | 5.4 | 11.9 |
| Carbón | DEV | **84** | 3.1 | 10.7 |
| Carbón | LB | 0.7 | 1.1 | 0.2 |

GN nunca supera cómodamente F=10 (criterio de la tarea); carbón lo supera ampliamente pero **solo con
tendencia lineal** — con tendencia flexible (cúbica o FE de bloques) la F cae a 3-12: casi toda la
capacidad del instrumento de carbón es la propia tendencia calendario, no variación independiente de
ella. En el lockbox (últimos 63 batches) el instrumento **no tiene poder para ninguna palanca** (F≈0):
la receta es casi constante en esa ventana (una versión domina), no hay variación que instrumentar.

## 3. 2SLS: el punto estimado invierte el signo y nunca es significativo (`e18_01_03_iv_resultados.csv`)

Referencia de conversión: −1.58 pp/sd ÷ sd_desvío_GN (2.18-3.01 Nm³/min por orden) = **−0.52 a −0.73
pp por Nm³/min** (coherente con el −0.57 crudo de E15-02).

KPI ~ GN ejecutado (instrumentado por receta), HC1, n≈298-361:

| Régimen | Tendencia | OLS b (t) | 2SLS b (t) | IC95 2SLS | DWH t (p) |
|---|---|---|---|---|---|
| DEV | lineal | −0.557 (−3.72) | +0.983 (0.80) | [−1.43, 3.40] | −1.40 (0.16) |
| DEV | cúbica | −0.566 (−3.58) | +0.356 (0.43) | [−1.28, 1.99] | −1.12 (0.26) |
| DEV | bloque40 | −0.560 (−3.20) | +1.252 (0.82) | [−1.75, 4.25] | −1.31 (0.19) |
| TOTAL | lineal | −0.377 (−2.72) | +1.279 (0.97) | [−1.32, 3.87] | −1.45 (0.15) |
| LB | (las 3) | +0.48..+0.65 (1.6-2.5) | −2.51..−0.14 (t<0.4) | anchísimo (±10-25) | n.s. |

El signo del 2SLS es **opuesto** al del OLS/desvío en TODAS las especificaciones DEV/TOTAL (+0.04 a
+1.77 pp/Nm³/min en vez de −0.52/−0.73) y el IC95 cruza ampliamente cero y también el valor OLS-implícito
— el instrumento es demasiado débil para arbitrar entre ambos. HAC(5) da SE 10-20 % mayores que HC1, sin
cambiar la conclusión. El test de Durbin-Wu-Hausman (función de control) **no rechaza exogeneidad** de
GN en ningún caso (|t| ≤ 1.5, p ≥ 0.14) — pero con esta potencia tampoco lo confirma.

Canales (DEV, GN): f_dross OLS +0.48 pp (t 4.25) → 2SLS +0.53 a +1.07 pp (t 0.5-1.6, **ya no
significativo**); f_polvo, f_metal, Sn-escoria: ningún 2SLS con |t| > 1.8.

Carbón: 2SLS positivo y significativo con tendencia lineal (TOTAL +0.52 t 3.3; DEV +0.74 t 2.6) y con
bloques (DEV +2.03 t 2.1), pero colapsa a t < 0.9 con tendencia cúbica — inestable con la forma de la
tendencia, tal como advierte la amenaza principal. DWH marginal solo en DEV/bloque40 (t −2.54, p 0.011).

## 4. Estudio de eventos: mismo signo contrario, mismo resultado nulo (`e18_01_04_eventos.csv`)

29 revisiones con |Δ receta GN media| ≥ 0.4 Nm³/min; 15 cumplen ≥6 batches sin otra revisión grande a
cada lado. Pendiente ΔKPI/Δreceta (residualizado por controles, bootstrap por evento):

| k | pendiente (pp/Nm³/min) | t | IC95 bootstrap | pendiente dDROSS |
|---|---|---|---|---|
| 6 | +0.552 | 0.77 | [−0.77, 1.85] | −0.778 (t −2.28) |
| 8 | +0.378 | 0.63 | [−0.64, 1.59] | −0.505 (t −1.10) |
| 10 | +0.120 | 0.26 | [−0.68, 1.12] | −0.178 (t −0.37) |

Signo contrario al reclamado (positivo en KPI, negativo en dross) y nunca significativo — segunda
identificación independiente que no confirma el hallazgo observacional.

## 5. Falsificación (`e18_01_05*_.csv`)

- **(a) Placebo por ubicación aleatoria**: la pendiente real (+0.552, k=6) cae dentro de la nula de 3000
  revisiones placebo (media −0.02, sd 0.71); p = 0.405 — indistinguible de ruido.
- **(b) Receta futura → KPI actual**: **FALLA**. La media de la receta de GN de la *siguiente* versión
  predice el KPI de hoy, controlando por la receta vigente: coef −0.944, t −2.05, p = 0.040 (n=355). Es
  la señal más preocupante: receta y campaña derivan juntas, tal como advertía la amenaza principal.
- **(c) Aire como instrumento placebo** (solo 2 valores en toda la campaña): primera etapa también débil
  (F=6.5); 2SLS KPI~aire da b=+0.62, t=1.53 (no significativo) — no genera un falso positivo, pero la
  prueba tiene poca potencia (mismo problema del instrumento débil).

## 6. Régimen (Task 3, tabla completa en el CSV) y fin de campaña

DEV y TOTAL cualitativamente iguales (signo positivo no significativo del 2SLS de GN). En LB (últimos
63 batches, donde planta subió R0 de 25→30.8) el instrumento no tiene poder (F≈0 para ambas palancas):
**no se puede decir nada** sobre el signo en fin de campaña con este método — ni confirma ni contradice
la inversión de signo del OLS estático reportada en E15-02 (§7), porque no hay variación que explotar.

## 7. Veredicto

La premisa del diseño (revisiones no reactivas al estado observable) se sostiene, con poca potencia. Pero
el experimento natural **no confirma causalmente** que más GN en Reducción baje el KPI o suba el dross:
(1) el instrumento de GN es débil (F 2.8-10.4, casi nunca > 10) y el de carbón depende casi enteramente
de la tendencia calendario; (2) dos identificaciones independientes (2SLS y estudio de eventos) dan el
**punto estimado con signo opuesto** al del OLS/desvío y nunca significativo (IC95 siempre cruza cero);
(3) la prueba de falsificación (b) falla: la receta futura predice el KPI de hoy, evidencia directa de
que receta y campaña derivan juntas — la amenaza principal es real, no hipotética. El efecto en pp por
Nm³/min queda sin identificar con precisión útil: el IC95 de las especificaciones DEV/TOTAL es
aproximadamente [−1.9, +4.3], demasiado ancho para distinguir el −0.52/−0.73 observacional de cero o de
un efecto positivo. **No se debe elevar el hallazgo del desvío a "causal confirmado"**; sigue siendo una
asociación observacional fuerte y replicada (E15-01/E17), pero la única vía de identificación fuerte
pendiente es el piloto aleatorizado (receta ± 1 sd) descrito en `candidate_win_model_v9_1.md` §4.
