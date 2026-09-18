# E13-03 — Auditoría adversarial de E13-01 (contracción James-Stein)

Objetivo: romper la afirmación de E13-01 (calibración 0.70, p_perm 0.0055 en W=100, p corregida 0.006
sobre 6 ventanas, jackknife-5 97% t>1.645). Scripts `e13_03_0{1..6}_*.py`, lib `e13_03_lib.py`.

## 1. Reproducción (`e13_03_01_repro.csv`)

Exacto: n=261, pendiente 0.700, t 2.320. Con 1000 réplicas (vs 2000) y 4 semillas de permutación:
p_perm = 0.003 / 0.005 / 0.005 / 0.005. Estable — no fue un golpe de suerte de la RNG.

## 2. Nulas alternativas (`e13_03_02_nulas_alt*.csv`)

| Nula (JS, W=100) | p_W100 | p corregida (6 ventanas) |
|---|---|---|
| Bloques de 10 | 0.010 | **0.012** |
| Bloques de 20 (primaria, E13-01) | 0.004 | 0.004 |
| Bloques de 40 | 0.004 | 0.010 |
| Circular, TODOS los k∈[15,346] (exhaustiva) | 0.012 | **0.012** |
| Signo por bloque de 20 | 0.002 | 0.002 |

**Grieta:** con bloques de 10 o circular exhaustivo la p corregida es 0.010–0.012, sobre el umbral 0.01.
Bloques de 20 —la nula elegida— es la más favorable de las cinco. Con 500 réplicas el error de muestreo
de p es ≈0.004–0.005: 0.006 y 0.012 no son la misma cifra con ruido, son nulas con distinta sensibilidad
a la autocorrelación de xG/xC.

## 3. Familia de contracciones (`e13_03_03_familia.csv`)

8 modos × 6 ventanas, 500 rep pareadas. JS no es favorable por casualidad: `hard1.5` (umbral duro
|t|≥1.5, W=60) da t=2.654, **más fuerte que JS** (2.592). `lineal`/`hard1` rinden similar a JS. `eb`
(Bayes empírico online) y `js_joint` (factor único) son claramente más débiles (t hasta 0.26, celdas
p>0.28); `hard2` es errático. **p corregida sobre TODA la familia (contracciones×ventanas) = 0.014** —
pasa <0.05, **no pasa <0.01**. Tratar la contracción como grado de libertad (igual que la ventana)
incumple el criterio 1.

## 4. Grados de libertad del analista (`e13_03_04_gl_analista.csv`)

81 combos (paso∈{5,10,20}×start∈{100,120,150}×controles∈{+idx,sin idx,espesor en vez de idx}×
exposiciones∈{xG+xC, sólo xG, sólo xC}), JS, W=100, misma nula de bloques de 20 (500 rep).

- **xG+xC (27):** robusto — 25/27 p<0.01, 27/27 p<0.05.
- **Sólo xG (27):** al límite — t 1.76–2.19; ~15/27 (56%) p<0.01, 27/27 p<0.05.
- **Sólo xC (27):** **no significativo** (t 0.45–1.51); 0/27 p<0.01, 8/27 p<0.05 marginal. Confirma
  E11-07: carbón solo no sostiene la prueba.
- **Global:** p<0.01 en 43% de las 81 combos, p<0.05 en 74%. La robustez depende de usar xG **y** xC
  juntos, no de cada palanca por separado.

## 5. Placebos con JS (`e13_03_05_*.csv`)

| Placebo | pendiente | t | p (perm) |
|---|---|---|---|
| lead1 (exposición del batch siguiente) | 0.028 | 0.05 | 0.36 |
| lead3 | −0.79 | −1.08 | 0.81 |
| lag1 (exposición del batch anterior) | −0.49 | −0.66 | 0.59 |
| kpi_prev (outcome = KPI del batch anterior) | 0.21 | 0.68 | 0.19 |
| **Limpio de persistencia** (+y_prev, xG_prev, xC_prev como controles) | 0.693 | 2.299 | 0.004 |

E11-07 (OLS) tenía lead1 atenuado a la mitad (t=1.29, p≈0.10) — persistencia de corto alcance. Con JS
lead1 colapsa a t≈0.05: la contracción, al recortar β cuando t_j es bajo, parece absorber justo la parte
que en OLS era ruido correlacionado a 1 batch. El resultado principal sobrevive casi intacto (t 2.30 vs
2.32) controlando directamente por KPI_prev/xG_prev/xC_prev — no depende de esa autocorrelación residual.

## 6. Fragilidad (`e13_03_06_*.csv`)

| Quitando (300 rep) | t P50 | % t>1.645 |
|---|---|---|
| 5 al azar | 1.94 | 95% |
| 10 al azar | 1.81 | **68%** |
| 20 al azar | 1.72 | **59%** |

El 97% (jackknife-5) de E13-01 se reproduce (95% aquí), pero cae a 68% al quitar 10 y 59% al quitar 20
batches (5.5% de n=361) — más de un tercio de réplicas ya no cruza t>1.645. Sensible a un puñado de
batches. Leave-one-block-out (19 bloques de 20): t entre 1.957 y 2.493, **ningún bloque sostiene todo el
resultado** — la fragilidad viene de combinaciones dispersas de batches, no de un tramo cronológico.

## 7. Replay estricto del asesor de planta (dato del coordinador, `e13_02_evidencia_{js,ols}.csv`)

Con el estado EJECUTABLE (E[a|S] entrenado sólo con el pasado, recomendación al inicio del escalón —
criterio 6), JS da pendiente 0.576 pero **t=1.409, p_perm bloques=0.162 (no significativo)**; Spearman
parcial 0.189 sí lo es (p=0.0021). En **lockbox** el Spearman parcial es ≈0 (p=0.9997) y t=0.38. Esto es
más débil que el análisis de E13-01/E13-03 (que usa xG,xC "ideales", no la recomendación ejecutable): el
salto de la exposición retrospectiva al asesor ejecutable pierde casi toda la significancia lineal, y el
lockbox del asesor no muestra señal de rango.

## 8. Veredicto del auditor

**Criterio 1 (p<0.01, corregida): NO se sostiene de forma robusta.** Solo bajo la nula y contracción
exactas de Fable (bloques20, JS) da 0.006. Con nulas igual de razonables: 0.010–0.012. Corrigiendo por
familia de contracciones: 0.014. Y el replay ejecutable del asesor (no solo el análisis de exposiciones)
da p=0.162 — muy por encima. **La p honesta a reportar es ≈0.01–0.02 en el mejor caso analítico, y no
significativa (p≈0.16) en el asesor ejecutable real.**

**Criterio 2 (calibración en [0.6,1.4]): se sostiene mejor.** 0.70 en la especificación primaria; en el
43% de combos con p<0.01 la calibración ronda 0.6–1.0 (riesgo de sub-calibración, no de sobre-ajuste).
El replay ejecutable da 0.576 (JS) — dentro del rango pero con t bajo.

**Salvedades:** (1) depende de xG **y** xC juntos, no de una palanca sola; (2) frágil a nivel de batches
individuales (jackknife-10/20 falla 32-41% de las veces) aunque no a nivel de bloque cronológico; (3) JS
no es la contracción más fuerte de su familia (`hard1.5` la supera) pero tampoco es neutral (`eb`/
`js_joint` son claramente peores) — el resultado sí depende de la familia funcional; (4) el placebo lead1
limpio con JS podría ser en parte un artefacto de que la contracción encoge el score hacia 0 en general,
aunque el control directo por persistencia (t 2.30) lo sostiene.

**Recomendación:** no declarar cumplido el criterio 1. Usar el rango p≈0.006–0.02 (análisis) y exigir que
el replay ejecutable (sección 7) mejore antes de piloto — hoy ese es el eslabón más débil, no la elección
de contracción ni la nula.
