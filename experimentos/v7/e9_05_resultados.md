# E9-05 (iteración 9, 2026-09-16): techo de ruido, curva de aprendizaje y bake-off de algoritmos

Pregunta (H5, `ITERACION_9_diseno.md`): ¿el R² OOF de los targets v6 está a 0.05-0.10 del techo impuesto por el
ruido de ensayo, la curva de aprendizaje está saturada, y ninguna familia de algoritmos supera al PLM/HGB? Se
prueban tres targets v6: Reducción `m6_ln_sn_dep` (R_sn_dep), Reducción `m6_ln_feo_ret` (R_feo_ret) y Fusión
`m6_ln_sn_dep` (F_sn_dep). Todo con `experimentos/v7/e9_05_techo.py <etapa>` (`techo|curva|bakeoff|desglose|resumen`);
CSVs y logs en `experimentos/v7/e9_05_*`.

Nota metodológica importante: `masa_v6.agregar_masa_v6(df, params)` **congela** las columnas `m6_*` si el `df` de
entrada ya las trae (hace `pd.concat` + `drop_duplicates(keep='first')`, que conserva las viejas). Para el Monte
Carlo del ruido hubo que **eliminar** las columnas `m6_*`/`Cx_*_v6` del df antes de perturbar y recalcular; si no,
la perturbación de leyes no tiene ningún efecto (se verificó empíricamente con una perturbación de +50% en
`ley_sn_escoria_pct` que no cambiaba `m6_masa_kg` en absoluto). Vale la pena documentarlo como gotcha del módulo.

## 1. Techo de ruido (Monte Carlo, 200 réplicas, ruido log-normal multiplicativo independiente por fila)

`e9_05_techo.csv`. R²_max = 1 − var_ruido/var_total, en DEV, con %Sn sd=2%, %FeO sd=2.8%, %CaO/%SiO2/%Al2O3/%MgO
sd=2.8% (niveles base):

| target | var_total | var_ruido | **R²_max** | R² OOF PLM v6 | **R² relativa** |
|---|---|---|---|---|---|
| R_sn_dep | 0.458 | 0.0010 | **0.998** | 0.765 | 76.7% |
| R_feo_ret | 0.0057 | 0.0024 | **0.571** | 0.329 | 57.6% |
| F_sn_dep | 0.0482 | 0.0011 | **0.978** | 0.161 | 16.5% |

Hallazgo central, no anticipado: **el ruido de ensayo casi no ata a `ln_sn_dep`** (techo ≈0.98-1.00 en ambas
fases). La razón física: `ln_sn_dep = ln((Sn_inv_prev + Sn_alimentado)/Sn_inv_t)`; el numerador es prácticamente
determinista (viene de la alimentación acumulada) y sólo el denominador depende de `ley_sn_escoria_pct` (una ley
pequeña, 3-15%), así que un 2% de ruido relativo en la ley se traduce en <2% de ruido en `Sn_inv_t`, minúsculo
frente al rango real del target (avance de reducción de ~0.5 a ~0.95, sd 0.68 en R). El techo de ruido **no es el
factor limitante** de `ln_sn_dep`; la brecha 0.765→0.998 (R) o 0.161→0.978 (F) es una brecha de **modelo/estado**,
no de ensayo. Distinto es el caso de `ln_feo_ret = ln(FeO_inv_t/FeO_inv_t-1)`: es un cociente de dos inventarios
consecutivos, ambos ruidosos y con retención de FeO de por sí de bajo rango (sd 0.075), así que el ruido no
cancela y el techo cae a 0.57 — consistente con el hallazgo previo (`e8_01_resultados.md`, SNR FeO 1.08-1.47).

**Sensibilidad** (`e9_05_techo.csv`, niveles 1/2/3/4% para %Sn y 2/2.8/4% para %FeO, ver tabla completa en el CSV):
para `ln_sn_dep` el techo es prácticamente insensible al nivel de ruido asumido (0.992-0.999 en todo el rango
probado): confirma que no es el cuello de botella. Para `ln_feo_ret` el techo es **muy sensible** al ruido de
%FeO: sd_feo=2% → techo 0.78; sd_feo=2.8% → 0.57; sd_feo=4% → **0.127** (por debajo del R² OOF observado, 0.329:
imposible, ya que el modelo no puede superar el techo real). Esto acota por arriba el ruido real de %FeO: **no
puede ser tan alto como 4%**, es coherente con el 2-2.8% supuesto (o incluso algo menor), un chequeo de
consistencia útil sobre el propio supuesto de ruido.

**Autocorrelación lag-1 de residuos OOF** (evidencia empírica de si el residuo es ruido de ensayo puro, que
predeciría autocorr ≈ −0.5 en una cadena telescópica): R_sn_dep −0.050, R_feo_ret −0.114, F_sn_dep −0.039 (n≈768-
1171). Todas muy por encima (en valor absoluto muy por debajo) de −0.5: **el residuo del PLM no es
predominantemente ruido de ensayo propagado** — es coherente con la conclusión de que la brecha en `ln_sn_dep`
es de modelo, y matiza a `ln_feo_ret` (algo más negativo, −0.114, consistente con que ahí el ruido sí pesa más,
pero lejos de ser el único componente).

**Por orden de escalón** (nivel base, `e9_05_techo.csv`): el techo de `ln_sn_dep` es uniformemente alto (0.99-
0.995 en R0-R3); el de `ln_feo_ret` es más bajo y desigual: R0 0.42, R1 0.54, R2 0.42, R3 **0.15** (el ruido dado
que FeO_inv es pequeño en R3 domina casi todo). Para Fusión, el techo sube de F1 (0.975) a F6 (0.99), sin patrón
fuerte.

## 2. Curva de aprendizaje (Reducción, n∈{50,100,150,200,250,290}, 3 repeticiones; `e9_05_curva.csv`)

n_max real = 290 (no 299: se pierden ~9 batches por NaN en estado/palancas de `ln_sn_dep`/`ln_feo_ret`).

| target | modelo | R²(n=290) | asíntota (a) ajustada R²=a−b/n | gap a la asíntota |
|---|---|---|---|---|
| R_sn_dep | PLM_v6 | 0.765 | 0.778 | **+0.013** |
| R_sn_dep | HGB estado+palancas | 0.759 | 0.771 | +0.012 |
| R_feo_ret | PLM_v6 | 0.329 | 0.325 | **−0.005** (ya en/sobre la asíntota) |
| R_feo_ret | HGB estado+palancas | 0.305 | 0.303 | −0.002 |

R² promedio por tamaño (PLM_v6): 0.711(n=50)→0.731(100)→0.759(150)→0.756(200)→0.772(250)→0.765(290) para
sn_dep, y 0.217→0.224→0.269→0.312→0.291→0.329 para feo_ret: sube rápido hasta ~150 batches y luego **se aplana**
(ruido de muestreo de ±0.02-0.04 entre repeticiones domina el resto de la curva). **La curva está saturada**: más
datos (dentro del rango observable, 290→∞) aportarían a lo sumo +0.01 a +0.013 de R² en `sn_dep` y básicamente
nada en `feo_ret`. Más BATCHES no es la vía para cerrar la brecha.

## 3. Bake-off de algoritmos (mismas filas/features por target; `e9_05_bakeoff.csv`, grid HGB en
`e9_05_bakeoff_grid_hgb.csv`)

Top-3 por target (R² OOF medio; PLM_v6 y HGB_ref promediados sobre 5 semillas, sd≈0 → estables):

| target | mejor OOF | R² OOF | R² lockbox | PLM_v6 | R² lockbox PLM |
|---|---|---|---|---|---|
| R_sn_dep | Stacking_promedio(HGB+XGB+Ridge) | 0.771 | 0.807 | 0.765 | 0.800 |
| R_feo_ret | XGBoost_cfg1 (honesto) / HGB_grid (optimista) | 0.326 / 0.332 | 0.411 / 0.485 | **0.329** | **0.486** |
| F_sn_dep | Stacking_promedio(HGB+XGB+Ridge) | 0.191 | **−0.765** | 0.161 | **−0.466** |

Notas:
- **HGB grid (81 configs: depth {3,4,6} × lr {.03,.05,.1} × max_iter {150,300,600} × min_leaf {10,20,40})**:
  seleccionado por el mismo OOF que reporta → **sesgo optimista explícito**. Rango de R² OOF en la grilla: R_sn_dep
  [0.737, 0.771], R_feo_ret [0.240, 0.332], F_sn_dep [0.060, 0.192]. El HGB por defecto (`L.hgb_nuisance`, depth 4,
  lr .05, max_iter 300, min_leaf 20) ya está cerca de la mediana/óptimo de la grilla en R_sn_dep y R_feo_ret: la
  búsqueda de hiperparámetros **no vale mucho** ahí. En F_sn_dep el rango es más ancho (0.06-0.19): el modelo de
  Fusión es más sensible a la varianza de selección, señal de que el estado ahí es débil.
- **kNN** (k=10..30, seleccionado por OOF, mismo sesgo): mejor k=15-25 según target; nunca compite (R_sn_dep 0.730,
  R_feo_ret 0.289, F_sn_dep 0.155), consistente con "mucho estado, poca estructura de vecindad útil".
- **MLP (64×64)**: el peor algoritmo en los tres targets, y en R_feo_ret **diverge** (R² OOF −19.4): red
  sobreajustada/mal condicionada con ~1000 filas y 38-42 features; no se recomienda para este tamaño de dato.
- **Ridge/Lasso + cuadráticos de las 8 features más importantes** (permutation importance sobre HGB): no superan
  al PLM/HGB en ningún target; en R_feo_ret Ridge cae a 0.097 (los cuadráticos con tan poca señal sobreajustan).
- **Stacking simple (promedio HGB+XGB+Ridge)** es el mejor u honesto-top en R_sn_dep y F_sn_dep, y el meta-Ridge
  sobre OOF no mejora al promedio simple en ningún caso (la correlación entre bases es alta, poco que ganar
  combinando linealmente).

**Comparación formal contra PLM v6** (bootstrap por batch del ΔR² OOF, IC95%, `e9_05_resumen.csv`, excluyendo del
"mejor algoritmo" las selecciones con sesgo de grid/kNN):

| target | mejor algoritmo (honesto) | ΔR² OOF vs PLM | IC95% bootstrap | ¿supera umbral +0.01 con IC>0? |
|---|---|---|---|---|
| R_sn_dep | Stacking_promedio | +0.005 | [−0.003, +0.014] | **No** |
| R_feo_ret | XGBoost_cfg1 | −0.004 (PLM gana) | [−0.030, +0.021] | **No** |
| F_sn_dep | Stacking_promedio | **+0.031** | **[+0.007, +0.055]** | **Sí (IC>0, aunque <0.01 no aplica: el umbral se cumple)** |

En Fusión el stacking gana en OOF con IC que excluye 0, pero su **lockbox es peor** que el del PLM (−0.765 vs
−0.466): es sobreajuste al régimen de DEV que no generaliza al período de fin de campaña (deriva de refractario,
hallazgo ya conocido de v5/v6). No es una mejora utilizable tal cual; corrobora H4 (el problema de Fusión es de
deriva/target, no de algoritmo) más que dar una alternativa viable.

## 4. Desglose del PLM v6 (`e9_05_desglose.csv`)

**Por orden de escalón** (OOF/lockbox): el modelo es fuerte en el escalón intermedio y débil en los extremos.
`R_sn_dep`: R0 0.15/−0.02, **R1 0.57/0.65**, R2 0.12/0.02, R3 −0.10/−0.23. `R_feo_ret`: R0 0.12/0.25, **R1
0.27/0.23**, R2 0.01/−0.28, R3 −0.09/−0.13. `F_sn_dep`: OOF 0.03-0.11 en todos los escalones pero **lockbox
negativo en todos** (−0.08 a −1.17, peor al final de Fusión) — deriva de campaña, no un escalón puntual.

**Por tercio cronológico de DEV**: R_sn_dep estable (0.72/0.79/0.78); R_feo_ret **decae** con el tiempo (0.40 →
0.30 → 0.28): el modelo de FeO pierde algo de poder según el batch es más reciente (posible deriva de
composición de carga, más leve que en Fusión). F_sn_dep **mejora** con el tiempo dentro de DEV (0.07→0.14→0.24)
pero igual falla fuera de DEV (lockbox), lo que sugiere que el problema no es tendencia lineal sino un quiebre en
el tramo de lockbox (fin de campaña de refractario, ya documentado).

**Por cuartil de `m6_avance_prev`**: en ambos targets de Reducción el modelo funciona mucho mejor en el cuartil
Q1 (avance medio-bajo: 0.60/0.24) que en los extremos Q0 (avance bajo: 0.09/0.11) y Q3 (avance alto: 0.01/−0.06).
Interpretación física: cerca de avance 0 o 1 el inventario relevante (Sn o FeO restante) es pequeño, así que el
mismo ruido relativo de ensayo se amplifica más en el log-ratio — coherente con el hallazgo de techo por orden
(R3 tiene el techo más bajo de FeO).

**Residuos OOF-dev**: no normales en los tres targets (Shapiro p≈0, colas pesadas: kurtosis 1.09 / 2.33 / 5.87);
F_sn_dep con sesgo derecho fuerte (skew +1.17, outliers de sobreajuste/campaña). Heterocedasticidad
(Spearman |res| vs predicho): significativa en R_sn_dep (ρ=0.099, p≈0.001) y **fuerte en F_sn_dep** (ρ=0.215,
p≈8e-17: más predicción, más error, otra huella de la deriva); no significativa en R_feo_ret (ρ=−0.011, p=0.71).
**Residuos vs palancas**: correlaciones pequeñas (|ρ|≤0.10) en las 4-5 palancas de cada target, sin ningún patrón
consistente de estructura no lineal no capturada — el efecto lineal-en-palanca del PLM no deja una señal grande
sin explicar (aunque `Cx_sn_v6` en R_sn_dep, ρ=0.07, es el más alto y sería el primer candidato si se quisiera
explorar una forma no lineal específica, tarea de E9-01).

## 5. Resumen y veredicto (`e9_05_resumen.csv`)

| target | R²_max (techo) | R² OOF PLM v6 | R² relativa | mejor algoritmo (Δ, IC95%) | curva saturada | **Veredicto** |
|---|---|---|---|---|---|---|
| **R_sn_dep** | 0.998 | 0.765 | 77% | Stacking +0.005 [−0.003,+0.014] (no significativo) | sí (gap +0.013) | **No hay modelo mejor con estos datos y este estado.** El techo de ruido no ata (≈1.0): la brecha 0.765→~1 es de estado/forma, no de ensayo ni de algoritmo ni de tamaño de muestra — ninguno de los 14 algoritmos, el grid de HGB, ni más batches la cierran. Margen real, pero solo vía mejor estado (E9-06) o forma (E9-01), no vía esta iteración. |
| **R_feo_ret** | 0.571 | 0.329 | 58% | Ningún algoritmo supera al PLM (mejor Δ −0.004) | sí (ya en la asíntota) | **No hay modelo mejor con estos datos.** Es el target más cerca de su propio techo (58% de un techo bastante más bajo que 1): aquí sí una parte sustancial de la brecha (0.329 vs 0.571) es ruido de ensayo de %FeO propagado (confirmado por la sensibilidad del techo a sd_FeO y la autocorrelación −0.114, la más negativa de los tres). El margen remanente (~0.24 R²) exige **menos ruido de ensayo** (mejor precisión analítica de %FeO) más que otro algoritmo. |
| **F_sn_dep** | 0.978 | 0.161 | 16% | Stacking +0.031 [+0.007,+0.055] pero lockbox −0.765 (peor que PLM) | sin datos de curva (fuera de alcance de esta etapa) | **Queda margen grande (techo ≈0.98, PLM sólo 0.16), pero no es explotable con más flexibilidad algorítmica**: el único algoritmo que gana en OOF (stacking) generaliza peor al lockbox, reproduciendo el patrón de deriva de campaña ya diagnosticado (H4/E9-04). El camino no es bake-off sino reformular el target/estado de Fusión para que sea robusto a la deriva. |

**Conclusión de la iteración (H5)**: parcialmente confirmada. Para **`ln_sn_dep`** (ambas fases) el techo de ruido
NO es la restricción activa —contrario a la hipótesis original— pero la conclusión práctica coincide: ni el
bake-off (14 algoritmos + grid de HGB de 81 configs + stacking, 5 semillas) ni la curva de aprendizaje (saturada
desde n≈150-200) encuentran una mejora sólida sobre el PLM v6 con el estado actual. Para **`ln_feo_ret`** el techo
de ruido SÍ es la restricción dominante (H5 tal como se planteó) y el PLM ya captura ~58% de la señal
identificable. Para **Fusión** el margen es grande y real pero no cerrable con el enfoque de esta iteración
(algoritmo/datos): el cuello de botella es la deriva de campaña, ya señalado como tarea aparte (E9-04).
