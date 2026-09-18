# E10-03 — Estado ejecutable / latencia del ensayo (H3, iteración 10, 2026-09-16)

Script: `experimentos/v8/e10_03_estado_ejecutable.py` (`global`, `orden`, `simulador_r0`). Logs: `e10_03_log_global.txt`,
`e10_03_log_orden.txt`, `e10_03_log_simulador_r0.txt`. Salidas: `e10_03_estados.csv`, `e10_03_theta.csv`,
`e10_03_por_orden.csv`, `e10_03_simulador_r0.csv`. Todo sobre `L.base_fase(df, "Reducción")`, DEV = GroupKFold(5) por
Batch = criterio; lockbox (63 batches finales) sólo reporte (declarado "gastado" por el comité). Nota de proceso: a
mitad de la corrida, `experimentos/v8/v8_lib.py` fue actualizada en paralelo para incorporar de forma nativa el estado
PLAN (`L.ESTADO_R_PLAN`, columnas `*_F6`, `Cx_sn_F6`, `Cx_av_F6`) con la misma definición que este script construía a
mano; se adoptó la versión de la librería (sin recalcular `global`/`orden`, que ya habían corrido con la construcción
propia y son numéricamente idénticos) y sólo se mantuvo aquí la construcción de las columnas `*_online` (`Cx_sn_online`,
`Cx_av_online`, `avance_prev2`), que la librería no trae.

## 0. Tres estados, mismas filas

- **A = k15** (`L.ESTADO_R_SIN_ESPESOR`, 15 features): usa el ensayo de laboratorio de t−1 (referencia v7/E9-06). No
  ejecutable en tiempo real en R1-R3 (auditoría C §1).
- **B = online** (`L.ESTADO_R_ONLINE`, 22 features): leyes/inventarios de **t−2** (`*_prev2`) + variables en línea de
  t−1. Es lo que de verdad se conoce en tiempo real: en R0 t−2 = F5, en R1 t−2 = F6, en R2 t−2 = R0, en R3 t−2 = R1.
- **C = plan** (`L.ESTADO_R_PLAN`, 20 features): leyes/inventarios **congelados al cierre de F6** (la fila `_prev` de
  R0, difundida a R0-R3 del batch) + `orden_escalon_fase` + variables en línea de t−1. F6 se conoce antes de que
  empiece R1 en cualquier escenario de latencia razonable (F6 dura 58-101 min; un ensayo de 30-60 min ya está listo).

Palancas cinéticas reconstruidas por estado: `Cx_sn_v6`/`Cx_av_v6` (A, con `m6_sn_inv_kg_prev`), `Cx_sn_online`/
`Cx_av_online` (B, con `m6_sn_inv_kg_prev2` y `avance_prev2 = 1 − Sn_inv,t−2/Sn_cargado`, `fe.division_segura` umbral
500 kg) y `Cx_sn_F6`/`Cx_av_F6` (C, con el inventario de F6). Targets: `m6_ln_sn_dep`, `m6_ln_feo_ret`. Filas comunes a
A, B y C (intersección de NaN): **n = 1138** (dev = 898, 252 batches DEV; lockbox aparte). Muestra algo menor que el
k15 "puro" de E9-06/E10-01 (~1083 dev) por la intersección con B y C.

## 1. R² por conjuntos (etapa `global`)

| target | estado | n_dev | R² OOF | R² cronológico | R² walk-fwd | R² lockbox | R² HGB directo | % de A (OOF) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sn_dep | A_k15   | 898 | **0.786** | 0.780 | 0.775 | 0.784 | 0.784 | 100 % |
| sn_dep | B_online| 898 | 0.697 | 0.697 | 0.695 | 0.756 | 0.692 | **88.7 %** |
| sn_dep | C_plan  | 898 | 0.724 | 0.720 | 0.719 | 0.735 | 0.720 | **92.1 %** |
| feo_ret| A_k15   | 898 | **0.309** | 0.272 | 0.241 | 0.473 | 0.302 | 100 % |
| feo_ret| B_online| 898 | 0.241 | 0.171 | 0.182 | 0.380 | 0.235 | 77.9 % |
| feo_ret| C_plan  | 898 | 0.242 | 0.187 | 0.181 | 0.392 | 0.259 | 78.3 % |

El PLM y el HGB directo (mismas features) coinciden dentro de 0.01-0.02 en todos los casos: la forma PLM no está
perdiendo información frente a un modelo no paramétrico con el mismo estado.

## 2. Identificación del carbón (θ libre, `theta_dual`, n_boot = 500)

| target | estado | palanca de carbón | θ libre /sd | IC95 % | identificado libre | GN θ libre /sd | IC95 % GN | identificado |
|---|---|---|---:|---|:---:|---:|---|:---:|
| sn_dep | A_k15    | `Cx_sn_v6`     | 0.145 | [0.044, 0.248] | **sí** | 0.068 | [0.023, 0.117] | sí |
| sn_dep | B_online | `Cx_sn_online` | 0.083 | [−0.065, 0.224] | **no** | 0.091 | [0.039, 0.152] | sí |
| sn_dep | C_plan   | `Cx_sn_F6`     | 0.118 | [0.028, 0.209] | **sí** | 0.064 | [0.007, 0.121] | sí |
| feo_ret| A_k15    | `Cx_av_v6`     | −0.036| [−0.091, 0.021] | no | −0.0096 | [−0.0189, 0.0005] | no |
| feo_ret| B_online | `Cx_av_online` | −0.057| [−0.113, 0.005] | no | −0.0165 | [−0.0264, −0.0068] | **sí** |
| feo_ret| C_plan   | `Cx_av_F6`     | −0.038| [−0.112, 0.067] | no | −0.0118 | [−0.0208, −0.0005] | **sí** |

Hallazgo central: para `sn_dep`, **el estado "plan" (C) conserva la identificación del carbón con IC libre y el estado
"online" (B) la pierde** (aunque ambos retienen > 80 % del R² pooled). La razón física: `Cx_sn_F6` usa un único punto
de referencia (inventario al cierre de F6) para toda la fase, mientras `Cx_sn_online` mezcla lags de distinta
antigüedad según el orden (F5 en R0, F6 en R1, R0 en R2, R1 en R3) — más heterogéneo, más ruidoso, IC más ancho
(bootstrap "en cota" 10.8 % vs 1.0 % de C_plan). Para `feo_ret`, ni A ni B ni C identifican el término de carbón
(`Cx_av_*`); es una debilidad **preexistente del target** (ya lo señalaba el comité, IC libres cruzan 0 incluso con
estado completo), no algo que la latencia empeore. El GN sí se identifica con el signo de teoría en los tres estados
para `sn_dep`, y **mejora** su identificación sobre `feo_ret` en B y C respecto de A (posiblemente porque, al quitar
las leyes ruidosas/colineales del estado completo, el efecto lineal del GN queda más limpio en el residuo).

## 3. Cuánto del R² pooled es sólo "saber en qué orden estás" (decomposición)

Antes de leer la tabla por orden hay que aislar un efecto mecánico: `ln_sn_dep` decae con el orden (R0 ≈ 1.70, R1 ≈
0.58, R2 ≈ 0.38, R3 ≈ 0.32 — el agotamiento absoluto es enorme en R0 y casi nulo en R3 porque ya no queda Sn que
agotar). Un modelo que sólo conociera `orden_escalon_fase` (media por orden, in-sample) ya explica:

| target | R² "sólo orden" | R² pooled A_k15 | incremento de A sobre "sólo orden" | R² pooled B_online | incremento de B | R² pooled C_plan | incremento de C |
|---|---:|---:|---:|---:|---:|---:|---:|
| sn_dep | 0.699 | 0.786 | **+0.087** | 0.697 | **≈ 0.000** (ligeramente negativo) | 0.724 | **+0.025** |
| feo_ret| 0.246 | 0.309 | **+0.063** | 0.241 | **≈ −0.005** | 0.242 | **≈ −0.004** |

Es decir: del 0.786 de A, 0.699 (89 %) es el patrón mecánico "qué tan avanzada está la reducción" y sólo 0.087 (11 %)
es información de estado genuina más allá del orden. **El estado online (B) no aporta ninguna información
incremental sobre saber el orden** (su R² pooled 0.697 es prácticamente igual al 0.699 de la línea base "sólo
orden" — de hecho ligeramente inferior); toda su aparente "retención del 88.7 %" viene de que el orden sigue siendo
observable en tiempo real, no de que B contenga señal útil sobre el estado físico. El estado plan (C) sí aporta algo
incremental (+0.025) — impulsado por R0, donde C = A. Para `feo_ret` el patrón es igual de crudo: B y C no aportan
nada sobre la media por orden; sólo A (con el ensayo real) agrega +0.063. **Esta es la lectura correcta del "92 % de
R²" de C_plan: no es 92 % de capacidad predictiva sobre el estado físico, es 92 % de un número dominado por un patrón
de calendario que cualquier reloj de escalón ya te da gratis.**

## 4. R² OOF aleatorio por orden (pooled fit, evaluado dentro de cada orden) — `e10_03_por_orden.csv`

| target | orden | n | R² A_k15 (ref) | R² B_online | % de A | R² C_plan | % de A | sesgo medio (C_plan) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sn_dep | R0 | 240 | 0.210 | −0.080 | **−38 %** | 0.167 | **80 %** | +0.007 |
| sn_dep | R1 | 246 | 0.554 | 0.038 | **7 %** | 0.028 | **5 %** | +0.002 |
| sn_dep | R2 | 243 | 0.166 | 0.055 | 33 % | 0.081 | 49 % | −0.010 |
| sn_dep | R3 | 169 | −0.013 | −0.024 | (n.d.: ref ≈ 0) | 0.0003 | (n.d.: ref ≈ 0) | −0.018 |
| feo_ret| R0 | 240 | 0.026 | −0.089 | (n.d.: ref ≈ 0) | −0.032 | (n.d.: ref ≈ 0) | −0.002 |
| feo_ret| R1 | 246 | 0.262 | 0.130 | 50 % | 0.046 | 18 % | +0.003 |
| feo_ret| R2 | 243 | 0.020 | −0.045 | (n.d.: ref ≈ 0) | −0.038 | (n.d.: ref ≈ 0) | 0.000 |
| feo_ret| R3 | 169 | −0.137 | −0.119 | (n.d.: ref ≈ 0) | −0.013 | (n.d.: ref ≈ 0) | +0.002 |

Lectura por orden (sn_dep, la única con señal utilizable):

- **R0**: C_plan conserva 80 % del R² de A (esperable: a nivel de features, C ≡ A en R0, ambos usan el cierre de F6;
  la diferencia residual viene de que el modelo se ajusta *pooled* sobre las 4 órdenes con features distintas para
  R1-R3). B_online es **negativo** (peor que predecir la media) — confirmado y cuantificado en la sección 5.
- **R1 es el escalón que más pierde**: tanto B como C caen a 5-7 % del R² de A. La razón es estructural, no un
  artefacto: en R1 tanto `*_prev2` (B) como el estado congelado (C) equivalen **al mismo dato, F6** — ninguno de los
  dos tiene la ley de R0 (que en A sí está disponible como `_prev`, pero que en la realidad tarda ≥ 15-60 min y no
  llega a tiempo para R1, de 25 min). Es decir, la referencia A está usando en R1 un dato que en producción **no
  existe todavía**: R1 es el escalón donde la brecha entre "lo que el modelo evaluado en OOF ve" y "lo que un
  operador real vería" es más grande.
- **R2**: recuperación parcial (33-49 % de A) porque a esta altura `*_prev2` de B ya alcanza la ley real de R0 (dos
  escalones atrás, ~55 min transcurridos — plausible que el ensayo ya esté listo), mientras C sigue anclado a F6. La
  comparación B vs C en R2 es ruidosa (n=243, GroupKFold aleatorio) y no debe leerse como que C > B de forma robusta.
- **R3**: el R² de referencia (A) ya es ≈ 0 en esta submuestra (n=169, la más pequeña y con duración más variable,
  10-23 min) — los cocientes % de A no son interpretables aquí; no hay evidencia de que ningún estado prediga
  `sn_dep` de forma útil en R3 dentro de este esquema pooled.
- `feo_ret` por orden: sólo R1 tiene R² de referencia por encima de ruido (0.262); ahí B retiene 50 % y C 18 %. En
  R0/R2/R3 el R² de referencia ya es ≈ 0 o negativo — no hay base para comparar estados.

## 5. Simulador de un paso en R0: ley de F6 (A) vs ley de F5 (B, `*_prev2` en R0)

Modelo ajustado **sólo con filas de R0** (no pooled con el resto de Reducción): `e10_03_simulador_r0.csv`.

| target | estado | n_dev | R² OOF | R² cronológico | R² lockbox (n=62) |
|---|---|---:|---:|---:|---:|
| sn_dep | A (F6 real)   | 240 | **0.124** | 0.152 | −0.506 |
| sn_dep | B (F5, prev2) | 240 | **−0.221** | −0.222 | 0.075 |
| feo_ret| A (F6 real)   | 240 | 0.001 | −0.076 | 0.174 |
| feo_ret| B (F5, prev2) | 240 | −0.292 | −0.385 | 0.276 |

No tener F6 en R0 no sólo reduce el R²: **lo vuelve negativo** (peor que predecir la media histórica). El lockbox
(n=62, un solo bloque cronológico final) es errático en ambas direcciones y no se usa para decidir (convención del
proyecto). Conclusión operativa: **R0 exige la ley de F6; no se puede sustituir por la ley de F5 con un modelo de un
paso**. Esto es consistente con el hallazgo previo (auditoría C §1) de que el simulador F5→F6 tiene R² 0.69 (Sn) /
0.89 (FeO) en niveles pero cae a 0.12 en órdenes tardíos — aquí se cuantifica directamente sobre el target operativo
(`ln_sn_dep`, no el nivel de inventario) y el resultado es más severo: **usar F5 en vez de F6 en R0 destruye, no sólo
reduce, la capacidad predictiva.**

## 6. Veredicto frente a H3

Criterio fijado (`ITERACION_10_diseno.md`): *"con leyes de t−2 + variables en línea de t−1 el agotamiento de Sn
conserva ≥ 80 % del R² OOF y el carbón sigue identificado; si no, sólo R0 (con F6 conocida) es recomendable en tiempo
real."*

- **Estado online (B, leyes de t−2) tal como estaba definido en el diseño: NO cumple.** Retiene 88.7 % del R² OOF de
  `sn_dep` (pasa el umbral numérico) pero **pierde la identificación del carbón** (`Cx_sn_online` IC [−0.065, 0.224],
  cruza 0) — y la sección 3 muestra que ese 88.7 % es casi enteramente el patrón "sólo orden": el estado online no
  aporta información física incremental medible sobre el estado del baño.
- **Estado plan (C, cierre de F6 congelado), la alternativa construida en este experimento: SÍ cumple para `sn_dep`**
  — retiene 92.1 % del R² OOF y el carbón (`Cx_sn_F6`) sigue identificado con IC libre [0.028, 0.209]. Es el único de
  los dos candidatos que satisface ambas condiciones de H3.
- **Ni B ni C cumplen para `feo_ret`** (77.9 % / 78.3 %, bajo el umbral de 80 %, y el término de carbón no se
  identifica en ninguno de los tres estados — tampoco en la referencia con ensayo completo). La retención de FeO ya
  era frágil con información completa; la latencia no la empeora cualitativamente, simplemente no hay ahí un target
  utilizable para recomendar carbón en tiempo real bajo ningún esquema de estado.
- **Matiz decisivo que el criterio agregado esconde**: el 92 % de C_plan es un promedio pooled dominado por R0 (donde
  C ≡ A) y por el patrón mecánico de orden. **Desglosado por escalón, R1 pierde 93-95 % del R² de A** bajo cualquiera
  de los dos estados ejecutables — el criterio "≥ 80 % agregado" se cumple aritméticamente pero no describe lo que
  pasa escalón a escalón.

**Veredicto: H3 EN DUDA → confirma y refina la arquitectura de dos niveles de la auditoría C, con una pieza nueva
(el estado "plan")**:

1. **R0 es recomendable en tiempo real** con el estado "plan" (= estado "F6 real" de la referencia): R² 0.124-0.21
   según el corte, carbón identificado libre en el ajuste pooled (`Cx_sn_F6` [0.028, 0.209]). Requiere que el ensayo
   de F6 (tomado al cierre de Fusión, que dura 58-101 min) esté procesado antes de fijar la dosis de R0 — plausible
   para latencias de 30-60 min. **No usar la ley de F5 como sustituto** (sección 5: R² negativo).
2. **R1 no es recomendable con un estado propio en tiempo real**: ni "online" ni "plan" aportan más información que
   saber que es R1 (5-7 % del R² de A). Recomendación operativa: **R1 se planifica junto con R0**, al inicio de
   Reducción, con el mismo estado "plan" (F6) — no se re-optimiza con datos "frescos" que en la práctica no existen
   todavía. Es decir: un plan estático R0+R1 fijado con la información de F6.
3. **R2 admite un re-plan** una vez que la ley de R0 esté disponible (`*_prev2` de B en R2 = ley de R0): la mejora
   sobre "plan" es real pero modesta y ruidosa (33-49 % de A, n=243); tratarla como una actualización de baja
   confianza, no como una recomendación por escalón con la misma seguridad que R0.
4. **R3 queda sin evidencia utilizable** (referencia A ya ≈ 0 en esta submuestra): no se recomienda ningún ajuste
   específico de R3 basado en este modelo; mantener el plan de R2 o la receta histórica.
5. **FeO (retención) no debe usarse para justificar cambios de carbón/GN en tiempo real en ningún escalón**: ni con
   estado completo ni con estado ejecutable se identifica el término de carbón, y el R² incremental sobre "conocer
   el orden" es ≈ 0 para B y C. El único uso defendible de FeO sigue siendo fuera de línea (auditoría/target de
   diseño de receta), nunca como señal de recomendación en vivo.

Esto es consistente con — y afina cuantitativamente — la arquitectura de 2 niveles ya propuesta en
`experimentos/v7/auditoria_C_operacion.md` §1: Nivel 0 (receta de Fusión, sin cambios), **Nivel 1 = plan R0+R1
con el estado "plan" (F6) al inicio de Reducción** (nuevo: R1 se pega al plan de R0, no se trata como escalón propio),
**Nivel 2 = re-plan de R2 (y, sin evidencia adicional, R3) tras el reporte de R0**. Ningún nivel usa `feo_ret` como
señal de control en tiempo real.
