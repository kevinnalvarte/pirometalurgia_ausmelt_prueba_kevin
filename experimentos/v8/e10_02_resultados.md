# E10-02 — Target de FeO con descuento del Fe alimentado (dross) y estado de Reducción (k15/k16/k17)

Script `experimentos/v8/e10_02_target_estado.py` (etapas `targets [estado] [familia] [targets_csv]`, `corr`, `reversion`,
`tercios`). Salidas: `e10_02_targets_estados.csv` (18 celdas), `e10_02_theta.csv` (84 filas θ libre/acotado), `e10_02_por_orden.csv`
(72 filas, R² OOF pooled por orden de escalón), `e10_02_corr_targets.csv` (sensibilidad de ranking), `e10_02_reversion.csv`,
`e10_02_tercios.csv`. Logs `e10_02_log_*.txt`. DEV/lockbox = split cronológico estándar (299/63 batches); SEED 42; θ con
`n_boot=500`; OOF GroupKFold(5) por Batch + OOF cronológico (5 bloques contiguos) + walk-forward (bloques de 50, ventana
creciente) + lockbox (sólo reporte).

**Nota de ejecución**: cada celda (familia × estado × target) del grid principal cuesta 45-435 s (mediana ~90 s; la variabilidad
es del propio proceso, no del código — se verificó con una prueba de temporización). Para mantener cada invocación del CLI
< 10 min se corrió la grilla de 18 celdas en 15 invocaciones (una por estado×familia con como máximo 2 targets, y para
`k17`/parte de `k15` una por target), todas en primer plano con `timeout=600000`. **Corrección aplicada a medio camino**: la
primera versión de `procesar_familia` calculaba la intersección de filas válidas ("mismas filas") sólo sobre los targets
incluidos en cada invocación del CLI, no sobre el total de la familia — para FeO no tuvo efecto (los 4 targets comparten
exactamente el mismo patrón de NaN, `n_dev=1083` en las 12 celdas), pero para Sn sí (`m8_ln_sn_dep_cstar` exige
`num>20 kg` y `den>20 kg`, cae a n=1045 de 1083). Se corrigió el código (la intersección siempre usa los targets completos
de la familia, aunque la invocación sólo calcule uno) y se re-corrieron las 2 celdas afectadas
(`k15_sin_espesor`/`m6_ln_sn_dep`, `k17`/`m6_ln_sn_dep`); la tabla final ya tiene `n_dev=1045` consistente en las 6 celdas
de Sn. Se documenta para que quede trazable.

## 1. R² por target × estado (`e10_02_targets_estados.csv`)

### FeO (retención), n_dev = 1083 en las 12 celdas

| estado (k) | target | R² OOF | R² cronológico | R² walk-forward | R² lockbox |
|---|---|---|---|---|---|
| k15 (15) | `m6_ln_feo_ret` (ref.) | 0.325 | 0.304 | 0.255 | 0.469 |
| k16 (16) | `m6_ln_feo_ret` (ref.) | **0.333** | **0.305** | 0.246 | 0.483 |
| k17 (17) | `m6_ln_feo_ret` (ref.) | 0.326 | 0.299 | 0.228 | 0.462 |
| k15 | `m8_ln_feo_ret_dross40` | 0.424 | 0.395 | 0.325 | 0.556 |
| k16 | `m8_ln_feo_ret_dross40` | 0.429 | 0.399 | 0.317 | 0.540 |
| k17 | `m8_ln_feo_ret_dross40` | 0.427 | 0.396 | 0.313 | 0.550 |
| k15 | `m8_ln_feo_ret_dross50` | 0.438 | 0.416 | 0.333 | 0.566 |
| k16 | `m8_ln_feo_ret_dross50` | 0.447 | 0.415 | 0.334 | 0.554 |
| k17 | `m8_ln_feo_ret_dross50` | 0.441 | 0.414 | 0.330 | 0.562 |
| k15 | `m8_ln_feo_ret_dross60` | 0.460 | 0.430 | 0.351 | 0.574 |
| k16 | `m8_ln_feo_ret_dross60` | **0.464** | **0.434** | **0.349** | 0.561 |
| k17 | `m8_ln_feo_ret_dross60` | 0.463 | 0.427 | 0.335 | 0.571 |

**El descuento del Fe alimentado mejora el R² de FeO en TODOS los estados y en las tres variantes de fracción de Fe, muy por
encima del umbral del criterio 4 (+0.01 en cronológico o walk-forward):** Δ OOF +0.09 a +0.13, Δ cronológico +0.09 a +0.13,
Δ walk-forward +0.07 a +0.10, según fFe. No hay pérdida de OOF aleatorio en ningún caso (la condición "> −0.01" del criterio 4
se cumple con enorme margen porque en realidad *mejora*, no empeora). El orden fFe 0.6 > 0.5 > 0.4 es monótono y consistente
en los tres estados y en las tres métricas OOF — la mejora no depende del estado elegido.

### Sn (agotamiento), n_dev = 1045 en las 6 celdas (fijado por el piso C* de `m8_ln_sn_dep_cstar`)

| estado (k) | target | R² OOF | R² cronológico | R² walk-forward | R² lockbox |
|---|---|---|---|---|---|
| k15 (15) | `m6_ln_sn_dep` (ref.) | 0.803 | 0.800 | 0.802 | 0.822 |
| k16 (16) | `m6_ln_sn_dep` (ref.) | 0.804 | 0.799 | 0.801 | 0.821 |
| k17 (17) | `m6_ln_sn_dep` (ref.) | **0.805** | **0.803** | **0.805** | 0.818 |
| k15 | `m8_ln_sn_dep_cstar` | 0.550 | 0.544 | 0.538 | 0.590 |
| k16 | `m8_ln_sn_dep_cstar` | 0.550 | 0.544 | 0.537 | 0.593 |
| k17 | `m8_ln_sn_dep_cstar` | 0.547 | 0.549 | 0.539 | 0.591 |

**(b) El piso de equilibrio C* SÍ cambia algo, y en la dirección contraria a lo esperado: el R² se desploma ≈ 0.80 → 0.55
(−0.25) en las tres métricas OOF, en los tres estados, sobre la MISMA submuestra de 1045 filas.** No es un problema del
estado; es el propio target: cerca de C* (0.7 % Sn) el logaritmo `ln((Sn_disp−C*M)/(Sn_inv−C*M))` amplifica el ruido de
ambos términos (la resta cerca de un piso pequeño es numéricamente inestable), y en R2-R3 el Sn de escoria se acerca a
niveles de 1-2 % donde `Sn−C*` es una fracción pequeña y ruidosa del total. **Veredicto (b): NO adoptar `m8_ln_sn_dep_cstar`
como reemplazo de `m6_ln_sn_dep`** — la corrección física es razonable en el papel pero destruye la relación señal/ruido del
target. Si se quiere el piso C* como sensibilidad teórica, mantenerlo fuera del prescriptor operativo.

## 2. Desglose por orden de escalón (`e10_02_por_orden.csv`) — ¿el descuento del dross importa sólo en R0?

R² OOF pooled por orden, target FeO (estado k16 de ejemplo; k15 y k17 muestran el mismo patrón, ver CSV):

| orden | n | `m6_ln_feo_ret` | `dross40` | `dross50` | `dross60` |
|---|---|---|---|---|---|
| R0 | 296 | 0.053 | 0.118 | 0.115 | 0.128 |
| R1 | 294 | 0.279 | 0.280 | 0.286 | 0.280 |
| R2 | 290 | 0.057 | 0.039 | 0.037 | 0.048 |
| R3 | 203 | −0.111 | −0.092 | −0.089 | −0.103 |

**Confirmado**: la ganancia real está concentrada en **R0** (+0.06 a +0.08 de R² pooled, más del doble del R² base) —
coherente con la física (el dross de Fe entra sobre todo en R0, mediana ~340-600 kg/escalón, hasta 3.8 t). En R1 la
mejora es marginal (+0.00 a +0.01); en R2 el descuento **no ayuda o empeora levemente** (el dross ya se agotó y restar un
término que ya no aplica sólo añade ruido); en R3 los cuatro targets son negativos en pool (poco FeO residual, ruido
domina, ningún target predice bien ahí). El descuento del dross es una corrección de R0, no un cambio genérico de forma.

## 3. Identificación de palancas (`e10_02_theta.csv`, θ libre con IC bootstrap por batch, n_boot=500)

### FeO: GN es la única palanca identificable; carbón y Cx_av NUNCA se identifican

| target | GN θ libre₁ₛd [IC] | identificado (k15/k16/k17) | carbón / Cx_av identificado |
|---|---|---|---|
| `m6_ln_feo_ret` | −0.008 a −0.010 | k15 sí, k16 **no** (IC [−0.0178,+0.0001]), k17 sí | no, en ningún estado |
| `m8_ln_feo_ret_dross40` | −0.010 a −0.011 | **sí en los 3 estados** | no, en ningún estado |
| `m8_ln_feo_ret_dross50` | −0.010 a −0.012 | **sí en los 3 estados** | no, en ningún estado |
| `m8_ln_feo_ret_dross60` | −0.010 a −0.012 | **sí en los 3 estados** | no, en ningún estado |

**(a) segunda mitad de la pregunta — el descuento del dross SÍ mejora la identificación de GN**: con el target original
(`m6_ln_feo_ret`) GN sólo se identifica en 2 de 3 estados (falla justo en k16, el estado de referencia v7); con cualquiera
de las tres variantes dross se identifica en los 3 estados, con IC más angostos y más lejos de 0 (p.ej. k16: libre
[−0.0178,+0.0001] con m6 → [−0.0201,−0.0019] con dross40). **Carbón y Cx_av (aire/O₂ ya se sabía que no) siguen sin
identificarse bajo ningún target ni estado** — la corrección del dross no resuelve el problema de fondo señalado por la
auditoría B §1: la colinealidad carbón↔Cx_av (r=0.97) y la falta de variación exógena del carbón en el estado hacen que su
efecto sobre FeO siga sin poder aislarse. Esto responde la parte de (c) sobre "conservar la identificación del carbón":
**no hay identificación que conservar** — carbón nunca estuvo identificado sobre FeO, ni en k16 (v7) ni en k15 ni en k17.

### Sn: Cx_sn_v6 y GN se identifican con ambos targets, pero más débilmente con el piso C*

| target | Cx_sn_v6 identificado | GN identificado |
|---|---|---|
| `m6_ln_sn_dep` | sí, k15/k16/k17 | sí, k15/k16/k17 |
| `m8_ln_sn_dep_cstar` | sí, k15/k16/k17 (IC más ancho) | sí sólo en k15; **no** en k16 ni k17 (IC roza 0) |

Coherente con la caída de R² de (b): el piso C* no sólo predice peor, sino que debilita la identificación de GN sobre Sn en
2 de 3 estados — otro argumento en contra de adoptarlo.

## 4. Sensibilidad de ranking entre variantes (`e10_02_corr_targets.csv`)

| comparación | Spearman global | Pearson global |
|---|---|---|
| `m6_ln_feo_ret` vs `dross40` | 0.982 | 0.980 |
| `m6_ln_feo_ret` vs `dross50` | 0.976 | 0.970 |
| `m6_ln_feo_ret` vs `dross60` | 0.970 | 0.959 |
| `dross40` vs `dross50` | 0.9995 | 0.9990 |
| `dross40` vs `dross60` | 0.9982 | 0.9964 |
| `dross50` vs `dross60` | 0.9996 | 0.9992 |

El criterio 4 pide "sin sensibilidad de ranking (corr ≥ 0.99 entre variantes)" — leído como *entre las variantes de fFe
candidatas* (no frente al target antiguo, que es exactamente lo que se busca cambiar): se cumple con holgura, las tres
variantes dross correlacionan ≥ 0.996 (Pearson) / ≥ 0.998 (Spearman) entre sí — **la elección de fFe dentro de 0.4-0.6 no
cambia el ranking de escalones**, así que la decisión de qué fFe usar es de robustez del R² (fFe=0.6 es marginalmente mejor
en las tres métricas OOF, ver §1), no de qué escalones se favorecen. La correlación con el target antiguo (0.96-0.98,
decreciente con fFe) es la firma esperada del cambio: se aleja más del original cuanto mayor es la corrección aplicada,
sin que eso implique inestabilidad entre las variantes candidatas.

## 5. Reversión de ruido del ensayo de t-1 (`e10_02_reversion.csv`) — pregunta (d)

Estado k15 completo vs sin `ley_feo_escoria_pct_prev`, `m6_feo_inv_kg_prev`, `ratio_sn_feo_prev`,
`interaccion_Sn_x_FeO_prev` (las 4 features que comparten el ensayo de FeO de t-1 con el target):

| target | R² OOF completo | R² OOF sin las 4 | frac. reversión OOF | R² cron. completo | R² cron. sin las 4 | frac. reversión cron. |
|---|---|---|---|---|---|---|
| `m6_ln_feo_ret` | 0.325 | 0.309 | **5.0 %** | 0.304 | 0.268 | **11.8 %** |
| `m8_ln_feo_ret_dross60` | 0.460 | 0.450 | **2.2 %** | 0.430 | 0.416 | **3.2 %** |

**(d)**: entre el 2 % y el 12 % del R² de FeO es atribuible a la reversión del ruido del ensayo de t-1 (mayor en el
cronológico que en el aleatorio, como predice la auditoría B §5: el aleatorio mezcla batches de todos los regímenes y
diluye la reversión, el cronológico la deja más expuesta). Es una fracción **modesta pero no nula** — el R² de FeO no es
"puro artefacto de reversión" (queda 88-98 % explicado sin esas 4 features), pero tampoco es cero. Dato adicional a favor
del target dross: su fracción de reversión es **menor** (2-3 % vs 5-12 %) que la del target original, pese a tener un R²
absoluto mucho mayor — la ganancia de dross40/50/60 no proviene de amplificar la reversión de ruido, proviene de una señal
física adicional real (el Fe del dross).

## 6. Estabilidad por tercios cronológicos (`e10_02_tercios.csv`, n_boot=500) — pregunta (c), segunda mitad

θ libre de `Cx_sn_v6` sobre `m6_ln_sn_dep` (signo teórico +) y de `tasa_gn_nm3_min` sobre `m6_ln_feo_ret` (signo teórico −),
por tercio cronológico de DEV, para k15 y k17:

| estado | palanca | tercio 0 (antiguo) | tercio 1 | tercio 2 (reciente) | identificado en |
|---|---|---|---|---|---|
| k15 | Cx_sn_v6 (sn_dep) | no [−0.224,+0.154] | no [−0.045,+0.334] | **sí** [+0.013,+0.554] | 1 de 3 |
| k17 | Cx_sn_v6 (sn_dep) | no [−0.232,+0.119] | no [−0.088,+0.311] | **sí** [−0.003,+0.543]* | 1 de 3 (*borde, IC roza 0 por 0.0025) |
| k15 | tasa_gn_nm3_min (feo_ret) | **sí** [−0.031,−0.005] | no [−0.004,+0.030] | **sí** [−0.032,−0.003] | 2 de 3 |
| k17 | tasa_gn_nm3_min (feo_ret) | **sí** [−0.028,−0.002] | no [−0.008,+0.024] | no [−0.026,+0.005] | 1 de 3 |

**GN sobre FeO es razonablemente estable** (identificado en 2/3 tercios con k15, 1/3 con k17 — la diferencia es de grado, no
de signo: en k17 el tercio 2 tiene el mismo signo pero el IC roza 0). **Cx_sn_v6 sobre Sn es MENOS estable de lo reportado
en `e9_06_resultados.md` para k16** (que daba 2 de 3 tercios con n_boot=100): con n_boot=500 (menos ruido Monte Carlo del
propio bootstrap) y los estados k15/k17 (no k16), sólo el tercio más reciente identifica el efecto, en ambos estados. No
puedo separar con estos datos si la caída de "2/3" a "1/3" se debe a quitar `espesor_ladrillo_norm_mm` (k15) / agregar T
horno y Al₂O₃ (k17), o al mayor n_boot (la propia auditoría B §2 ya advertía que la diferencia entre 1/3 y 2/3 tercios con
n_boot=100 no se distingue del ruido Monte Carlo, ±0.02) — no se volvió a correr k16 con n_boot=500 aquí porque no estaba
en el diseño de este experimento. **Lectura conservadora: la estabilidad de `Cx_sn_v6` por tercios NO debe usarse como
criterio para preferir k15/k16/k17 entre sí** — es ruidosa a esta escala de n_boot y de n por tercio (~360 filas), y el
efecto interior/exterior de cota (`frac_replicas_en_cota` 65-82 % en tercios 0-1) confirma que en los dos tercios más
antiguos el efecto simplemente no está identificado con estos datos, independientemente del estado.

## 7. Veredicto contra los criterios de decisión del diseño

**Criterio 1 (predictivo, Δ≥−0.01 OOF; cronológico/WF no empeorar >0.02)**: k15 vs k16 en la referencia (`m6_ln_feo_ret`,
`m6_ln_sn_dep`) cumple con margen amplio (ΔR² entre −0.008 y +0.005 en todas las métricas). k17 vs k16 también cumple
(peor caso Δ walk-forward FeO = −0.018, dentro del margen de 0.02; el resto ≤ −0.007). **Ambos estados pasan el criterio 1.**

**Criterio 2 (palancas con IC libre que excluya 0 y signo de teoría)**: con el target dross, GN se identifica en **los 3
estados** para FeO (mejora sobre el target original, que fallaba en k16); Cx_sn_v6 y GN se identifican en los 3 estados
para Sn con el target original; con `m8_ln_sn_dep_cstar` GN pierde identificación en 2 de 3 estados. Carbón/Cx_av sobre
FeO y O₂/aire sobre ambos targets **siguen sin identificarse en ningún caso** — deben declararse θ=0 (no estimados), como
exige el criterio, no usarse en el objetivo.

**Criterio 4 (target de FeO: mejora cronológico/WF ≥+0.01 sin perder OOF aleatorio, fFe en 0.4-0.6, corr≥0.99 entre
variantes)**: **se cumple con holgura amplia** en las tres variantes y en los tres estados (Δ cronológico +0.09 a +0.13,
Δ walk-forward +0.07 a +0.10, sin pérdida de OOF aleatorio — de hecho también mejora; corr entre variantes de fFe ≥ 0.996).

## 8. Veredicto final y recomendación

1. **Adoptar el target de FeO con descuento del dross** (`m8_ln_feo_ret_dross{40,50,60}`) en reemplazo de `m6_ln_feo_ret`:
   mejora el R² en las tres métricas OOF por un margen muy superior al umbral, concentrado físicamente en R0 (donde entra
   el dross), mejora la identificación de GN a los 3 estados, y reduce (no aumenta) la fracción de reversión de ruido.
   Entre fFe 0.4/0.5/0.6, el ranking es insensible (corr ≥ 0.996) y **fFe = 0.6 da el mejor R² en las tres métricas y en
   los tres estados**, pero la ventaja sobre 0.5 es pequeña (≤ 0.02 de R²); dado que la auditoría A documenta 0.4-0.6 como
   el rango físico sin evidencia de un valor central preciso (la planta no reporta %Fe del dross), **se recomienda
   `dross50` como opción central y defendible** salvo que se obtenga un ensayo de %Fe del dross que fije el valor real; si
   se prioriza estrictamente el R², usar `dross60`.
2. **NO adoptar el piso C* para Sn** (`m8_ln_sn_dep_cstar`): degrada el R² en ≈0.25 en las tres métricas OOF y debilita la
   identificación de GN — la corrección física es razonable en el papel pero destruye la relación señal/ruido cerca del
   equilibrio con los datos actuales. Mantener `m6_ln_sn_dep`.
3. **Retirar `espesor_ladrillo_norm_mm` (pasar de k16 a k15) no cuesta R²** (Δ ≤ 0.008 en las tres métricas, en ambos
   targets) — confirma la auditoría A §5 (es un proxy de campaña, no de estado físico) sin pérdida práctica.
4. **k17 (+T horno, +Al₂O₃) no aporta ni perjudica de forma decisiva**: mejora levemente Sn (+0.001 a +0.004), empeora
   levemente FeO (−0.006 a −0.018, dentro de tolerancia). No hay identificación de carbón que "conservar" en ningún estado
   — carbón nunca se identifica sobre FeO con estos datos, independientemente de k15/k16/k17. Recomendación: usar **k15**
   por parsimonia (mismo desempeño que k16/k17, una feature menos con problemas de interpretación) salvo que se necesite
   Al₂O₃/T horno por otra razón (p.ej. seguimiento de refractario).
5. **Fracción de reversión de ruido en FeO**: 5.0 % (OOF) / 11.8 % (cronológico) del R² de `m6_ln_feo_ret`, y 2.2 % / 3.2 %
   del R² de `m8_ln_feo_ret_dross60` — modesta, mayor en el cronológico que en el aleatorio (como predice la teoría de la
   auditoría B §5), y **menor** para el target corregido pese a su R² más alto: la ganancia de dross es señal física, no
   reversión amplificada.
6. **Pendiente honesto**: la estabilidad por tercios de `Cx_sn_v6` (sn_dep) es más débil de lo reportado en v7 para k16
   (1 de 3 tercios identificados aquí con n_boot=500, vs 2 de 3 con n_boot=100 en `e9_06_resultados.md`); no se puede
   atribuir con estos datos a k15/k17 vs k16 o al mayor n_boot — no usar la estabilidad por tercios como criterio de
   selección de estado sin volver a correr k16 con el mismo n_boot para tener una comparación limpia.
