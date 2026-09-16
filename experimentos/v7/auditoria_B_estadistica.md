# Auditoría B — rigor estadístico e identificación del enfoque ganador v7 (2026-09-16)

Auditor B (inferencia causal en datos observacionales de proceso). Objeto: `candidate_win_model_v7.md`, hallazgos §18-20,
`ITERACION_9_diseno.md`, memos E9-00/05/06, `e9_07_evidencia_politica_v7.csv`, `e9_08_potencia.csv`, `modelo_predictivo_v7.py`
(sobre `ModeloPLMSignos` v5 / `ModeloPLM` v4, `split_dev_lockbox`). No se modificó nada del repo. Comprobaciones ejecutadas con
`.venv/Scripts/python.exe` desde la raíz sobre `df_v6.pkl` (scripts en el scratchpad de la sesión: `chk1_theta.py`,
`chk2_optimismo.py`, `chk3_anclaje.py`, `chk4_uplift.py`, `chk6_techo.py`, `chk7_next.py`; sus salidas se citan abajo).

## Tabla de veredictos

| # | Afirmación auditada | Veredicto | Motivo (número clave) |
|---|---|---|---|
| 1a | θ(Cx_sn→Sn) +0.178 [0.041, 0.285], IC bootstrap con cota válido | RESCATAR | óptimo interior: acotado = libre (+0.1776), 1 % de réplicas en la cota; IC libre [0.041, 0.286] |
| 1b | θ(GN→FeO) −0.008 [−0.015, 0] "significativo"; O₂/aire sobre FeO | EN DUDA | significancia creada por la cota: IC libre [−0.0174, +0.0002]; O₂ [−0.0009, +0.0068], aire [−0.0033, +0.0047]; 20-30 % de réplicas en la cota |
| 1c | "carbón selectivo" (θ ≈ 0 sobre FeO) | DESCARTAR como hallazgo | no identificado: IC libre ±0.045/sd (5× el efecto de GN); residuos carbón/Cx_av r 0.97 |
| 1d | Sin confusión residual; hay variación exógena (R² balance 0.97) | EN DUDA | R² 0.964 es mecánico (Sn_inv_prev ∈ S); carbón primitivo R² 0.90, sd residual 7 kg/min de 22 (32 %): hay variación, pero es la desviación del operador respecto del estado, no aleatoria |
| 2 | Estado k16 conserva R² y "identifica mejor" | RESCATAR (R²) / EN DUDA (identificación) | 4 semillas de folds: k16 0.767±0.003 / 0.326±0.007 vs FULL 0.765 / 0.305; optimismo ≤ 0.007. "2/3 tercios vs 1/3" está dentro del error MC de n_boot 100 |
| 3a | Legado: KPI_next ~ Σln_feo_ret +10.4 (p 2e-4) | RESCATAR | sobrevive a KPI_actual como control (+9.4, p 0.0015), HAC(3), FeO propio del batch siguiente (+9.1, p 7e-4), placebo inverso (p 0.70) y **lockbox +19.8 (p 0.011)** |
| 3b | K = 3.52 / w = 7.86 sobre dos batches | EN DUDA | IC K_Sn [1.8, 5.5] (±50 %); K_FeO atenuado por EIV ×1.2-1.4; **w en lockbox 0.7** (FeO p 0.72): pesos dependientes de régimen |
| 3c | Doble conteo heel / legado | RESCATAR | K anclado sin heel; con heel propio K_FeO 25.4 (w 6.9); IRF_fin es mediador (no debe controlarse) |
| 4 | Lockbox "sólo reporte"; OOF ≈ desempeño futuro | EN DUDA | ≥ 3 decisiones tocaron el lockbox (E9-05 stacking F, E9-04 recalibración, E7-01); walk-forward FeO 0.261 vs OOF 0.333 |
| 5 | Techo FeO 0.571; PLM al 58 %; "ruido real ≤ 2.8 %"; autocorr −0.11 como evidencia | DESCARTAR (cálculo) | el ruido de t−1 está en el estado y es revertible: techo corregido **0.79** (PLM al 42 %); la inferencia sobre el ruido y la autocorrelación no son diagnósticas |
| 6a | Uplift +0.27 pp [0.25, 0.29], > 99 % batches positivos | DESCARTAR (IC) / EN DUDA (magnitud) | IC sólo remuestrea batches; con K/w [0.20, 0.35]; con θ ≈ [0.12, 0.63]; "> 99 % positivo" es tautológico (min 0.000); 82 % del uplift viene del canal FeO de 1b |
| 6b | Off-policy: polvo p 0.013, dist_F −0.49 | DESCARTAR como evidencia | BH q 0.24 (18 pruebas DEV); dist_R +0.13 (signo contrario); placebo por permutación es no informativo |
| 6c | "Las recomendaciones mejoran el batch" | DESCARTAR; "no lo empeoran" EN DUDA | potencia 32 %; A/B por batch para +0.3 pp: 3 500 batches/brazo (≈ 8 años) |
| 7 | Protección frente a pruebas múltiples | mixto | legado sí (p 2e-4 y réplica en lockbox); polvo no; Cx_sn robusto a sets/semillas pero seleccionado entre iteraciones |

---

## 1. Identificación de θ en el PLM (`chk1_theta.py`)

**Qué se afirma.** `y = g(S) + Σθ_j(a_j − E[a_j|S])`, g y m por HGB cross-fitted GroupKFold(5) por batch; θ por mínimos
cuadrados acotados por signo; IC95 % por bootstrap clúster (300) re-resolviendo el problema acotado en cada réplica.

**Riesgos.** (i) El bootstrap percentil es inconsistente cuando el parámetro está en la cota (Andrews 2000): la masa en 0 comprime
el IC y "excluir 0" pasa a ser artefacto de la restricción. (ii) Colapsar a 0 los θ de signo contrario introduce sesgo de
selección: E[max(θ̂,0)] > 0 aun con θ = 0, y un θ "en cota" se lee como "sin efecto" cuando en realidad es "no identificado".
(iii) El IC re-muestrea residuos con nuisances fijos: ignora la variabilidad de g/m (aceptable asintóticamente por ortogonalidad
de Neyman, dudoso con n ≈ 1 000 y R²(a|S) 0.96; E9-06 §5 lo muestra: HGB300 mueve Cx_sn 0.164 → 0.146, 10 folds en k20 →
[0.000, 0.278]). (iv) Confusión residual por decisiones del operador basadas en información no registrada.

**Verificación (DEV, Reducción, estado k16, n 1 083).** Re-estimé θ acotado y libre con el mismo cross-fitting y 300 réplicas:

| target | palanca | θ₁sd acotado [IC] | θ₁sd libre [IC] | réplicas en cota |
|---|---|---|---|---|
| Sn_dep | Cx_sn_v6 | +0.178 [+0.041, +0.285] | +0.178 [+0.041, +0.286] | 0.01 |
| Sn_dep | GN | +0.047 [+0.011, +0.086] | +0.047 [+0.011, +0.091] | 0.00 |
| Sn_dep | exceso O₂ | −0.008 [−0.025, −0.000] | −0.008 [−0.025, +0.010] | 0.20 |
| FeO_ret | carbón | −0.0001 [−0.010, 0] | −0.0004 [−0.045, +0.044] | 0.73 |
| FeO_ret | Cx_av | 0 [−0.010, 0] | +0.0003 [−0.039, +0.037] | 0.67 |
| FeO_ret | GN | −0.0081 [−0.0151, −0.0000] | −0.0081 [−0.0174, +0.0002] | 0.03 |
| FeO_ret | exceso O₂ | +0.0031 [0, +0.0067] | +0.0031 [−0.0009, +0.0068] | 0.05 |
| FeO_ret | aire | +0.0008 [0, +0.0047] | +0.0008 [−0.0033, +0.0047] | 0.30 |

Balance: R² OOF de E[a|S]: Cx_sn 0.964 (sd residual 19 % de la sd), carbón primitivo 0.897 (sd residual 7.1 de 22 kg/min), GN
0.68, O₂ y aire ≈ 0 (exógenos respecto del estado). Correlación de residuos carbón–Cx_av 0.97 (colinealidad severa en el modelo
de FeO); GN–O₂ −0.54.

**Lectura.** Los dos efectos que sostienen el agotamiento de Sn (Cx_sn, GN) son interiores: el IC acotado coincide con el libre y
es válido. Todo lo que sostiene la retención de FeO es de cota: GN sólo "excluye 0" porque la cota corta el extremo superior
(+0.0002 libre); O₂ y aire tienen IC libres centrados en 0. La "selectividad del carbón" (θ ≈ 0 sobre FeO) no es un hallazgo sino
ausencia de identificación (IC libre ±0.045/sd, cinco veces mayor que el efecto de GN que sí se usa). El R² 0.964 del balance es
mecánico (Cx_sn = carbón × Sn_inv_prev y Sn_inv_prev ∈ S); la variación identificadora real es la desviación del operador respecto
de la dosis que el estado predice (32 % de la sd del carbón). Esa desviación es exógena sólo si el operador no reacciona a señales no
registradas (aspecto de la escoria, muestras intermedias, instrucciones de turno); si dosifica más carbón cuando "ve" más Sn, θ está
sesgado hacia arriba y ninguna prueba con estos datos lo descarta. El colapso con nuisance Ridge (E9-06 §5) se presenta como "el
HGB es necesario", pero es igualmente compatible con "la identificación depende de la flexibilidad del nuisance" (sesgo de
regularización de DML): la robustez a HGB300/10 folds es parcial.

**Cómo resolverlo.** Reportar siempre θ libre junto al acotado y declarar "no identificado" (no "≈ 0") a los θ en cota; para los IC de
efectos acotados usar bootstrap con corrección (m-out-of-n o intervalos de proyección) o simplemente el IC libre; incluir el
re-ajuste de g/m en cada réplica (al menos en una submuestra) para medir cuánto ensancha; prueba de resultado de control negativo
(θ de Cx_sn sobre un target que la física dice inafectado, p. ej. Δ%Al₂O₃) para detectar confusión del operador.

## 2. Selección del estado parsimonioso E9-06 (`chk2_optimismo.py`)

**Qué se afirma.** `libre_k16` conserva el R² de FULL (0.767/0.333 vs 0.765/0.329), identifica el carbón en 2/3 tercios (FULL
1/3) y generaliza mejor en walk-forward. **Riesgo.** La ruta de eliminación (importancia por permutación recalculada cada 4 pasos)
y la elección entre 15 checkpoints/sets usan el mismo OOF que se reporta; los tercios (n ≈ 350, n_boot 100) son ruidosos y el
criterio "2 de 3" es una selección sobre el propio resultado de identificación.

**Verificación.** OOF con 4 asignaciones aleatorias de batches a folds (distintas de la GroupKFold determinista usada para elegir),
mismas filas para los tres sets:

| set | k | Sn_dep media ± sd | FeO_ret media ± sd |
|---|---|---|---|
| FULL | 37 | 0.7645 ± 0.006 | 0.3049 ± 0.007 |
| libre_k12 | 12 | 0.7571 ± 0.003 | 0.3183 ± 0.006 |
| libre_k16 | 16 | 0.7667 ± 0.003 | 0.3261 ± 0.007 |

k16 ≥ FULL en las 4 semillas y en ambos targets; el optimismo de selección respecto del OOF reportado es ≤ 0.007 (FeO) y ≈ 0
(Sn). El R² por tercios de k16 no se re-evaluó; con n_boot 100 el error Monte Carlo del percentil 2.5 es del orden de ±0.02, por lo
que "0.028 > 1e-3" en el tercio 1 (la única diferencia con FULL, cuyo tercio 1 da 0.000) no distingue configuraciones.

**Veredicto.** RESCATAR el estado k16 como predictor (la ganancia sobre FULL es real aunque pequeña); EN DUDA que "identifique
mejor": la ventaja es de una réplica ruidosa. Resolver con n_boot ≥ 1 000 por tercio y reportar el IC libre; no usar la
estabilidad por tercios como criterio de elección entre sets (o pre-registrar el set antes de mirarla).

## 3. Anclaje del objetivo y legado E9-00/E9-00b (`chk3_anclaje.py`, `chk7_next.py`)

**Qué se afirma.** OLS HC3 de KPI (y KPI_actual + KPI_siguiente) sobre Σ_R ln_sn_dep y Σ_R ln_feo_ret con 6 controles: K_Sn
3.52, K_FeO 27.68, w 7.86 [5.14, 13.04]; legado +10.4 pp/unidad (p 2e-4), placebo KPI_prev +2.1 n.s.

**Riesgos y verificación.**
- *Autocorrelación de campaña.* KPI: autocorr lag-1 0.12, lag-2 0.04; residuos del anclaje 0.04. Con KPI_actual como control el
  legado queda +9.4 [3.6, 15.2] (p 0.0015; KPI_actual coef 0.06 p 0.26); con HAC(3) p < 1e-4; con KPI_prev añadido +9.4; con el
  Σln_feo_ret *propio* del batch siguiente en la regresión +9.1 (p 7e-4); placebo inverso (KPI_actual ~ FeO del batch siguiente)
  +1.3 (p 0.70). En el **lockbox** (n 62) el legado replica: +19.8 (p 0.011). Es el hallazgo mejor protegido del proyecto.
- *Errores-en-variables.* var(Σ_R ln_feo_ret) DEV = 0.0086; el ruido de la suma telescópica es el de los dos extremos (2·0.028² =
  0.0016) hasta el ruido por escalón de E9-05 (0.0024): fiabilidad λ 0.72-0.82 → K_FeO atenuado ×1.2-1.4 (≈ 34-38 pp/unidad; w ≈
  9.5-11). Para Sn λ ≈ 1.00. La afirmación de E9-00 §2 "el prescriptor no hereda la atenuación porque usa predicciones" es
  incorrecta: K se estima sobre targets ruidosos y es el K atenuado el que multiplica las predicciones.
- *Endogeneidad.* Σtargets son el estado terminal (identidad contable con el KPI) y no decisiones; el sesgo de "otras vías" es
  pequeño, pero los controles explican 0.4 % del KPI (R²adj 0.004) y el modelo completo 10.7 %: casi todo el KPI es varianza no
  modelada, de ahí el IC de K_Sn [1.8, 5.5] (bootstrap iid y por bloques de 5 coinciden).
- *Régimen.* En el lockbox KPI ~ Σln_feo_ret +2.8 (p 0.72) y Σln_sn_dep +4.05 (p 0.003): **w = 0.7 frente a 7.86 en DEV**. Los
  pesos del objetivo no son estables entre regímenes de campaña (E9-00 §4 ya lo señala); una política optimizada con w 7.86 en un
  régimen con w ≈ 1 recorta carbón/GN donde el KPI premiaría agotar Sn.
- *Doble conteo.* K se ancló sin heel (correcto); añadir el heel propio F_lnirf_F0 (pre-determinado, control legítimo) da 3.69 /
  25.4 (w 6.9); añadir R_lnirf_fin es controlar un mediador (K_FeO 10.8) y no debe hacerse. No hay doble conteo en J: cada batch
  acredita sólo su retención; el batch siguiente recibe el heel como estado, no como acción.

**Veredicto.** RESCATAR el legado; EN DUDA la magnitud de (K, w): reportar K corregido por EIV, IC de K_Sn, y sensibilidad de
la política a w ∈ [1, 11] (no sólo 6.09 vs 7.86). "K = 3.52 sobre dos batches" es defendible como efecto total; su precisión no.

## 4. Validación, lockbox y deriva

- OOF GroupKFold por batch como criterio: adecuado y consistentemente aplicado. Pero el lockbox ya no es virgen: (a) E9-05 rechazó el
  stacking de Fusión (ΔR² OOF +0.031 [0.007, 0.055], que **cumplía** el criterio 1 del diseño) porque "pierde en lockbox"; (b) E9-04
  calibró interceptos con los primeros 20 batches del lockbox; (c) E7-01 y E8-02 documentan variantes descartadas tras ver que
  ganaban en lockbox. Además el lockbox se ha reportado para decenas de variantes en las iteraciones 4-9. Con 63 batches, cada
  mirada informa la siguiente decisión; los R² lockbox (0.795 / 0.483, superiores al OOF) no pueden leerse como confirmación
  independiente.
- Deriva: para Reducción el lockbox no muestra deriva (mejor que OOF), pero el walk-forward es la estimación honesta del futuro:
  Sn 0.761 (≈ OOF) y **FeO 0.261 vs 0.333** OOF (bloque 4: 0.05). Fusión: OOF 0.19 vs lockbox −0.70 — no generaliza; el memo lo
  reconoce. GN sobre FeO sólo se identifica en el último tercio (hallazgos §20.4): la palanca que sostiene el 82 % del uplift es la
  menos estable en el tiempo.

**Veredicto.** RESCATAR el criterio OOF; EN DUDA la pureza del lockbox. Declararlo "gastado", reportar walk-forward como métrica
principal de generalización y pre-registrar la próxima campaña (o los primeros N batches del piloto) como nuevo holdout.

## 5. Techo de ruido E9-05 (`chk6_techo.py`)

**Qué se afirma.** R²_max = 1 − var_ruido/var_total con ruido log-normal independiente por fila (2 % Sn, 2.8 % FeO): FeO_ret 0.571,
PLM al 58 %; "el R² OOF 0.329 supera el techo con 4 % → el ruido real es ≤ 2.8 %"; autocorrelación de residuos −0.05/−0.11 "muy
lejos de −0.5" prueba que el residuo no es ruido de ensayo.

**Riesgo.** El target `ln_feo_ret[t] = ln FeO_inv[t] − ln FeO_inv[t−1]` comparte el ensayo de t−1 con el estado (`m6_feo_inv_kg_prev`,
`ley_feo_escoria_pct_prev`, ratios). El modelo puede revertir esa mitad del ruido (§19.1 lo documentó para v5: "aprendía la reversión
del ruido de laboratorio"); sólo el ruido contemporáneo de t es irreducible.

**Verificación.** 30 réplicas Monte Carlo con las mismas funciones de E9-05 (`perturbar_leyes`, `agregar_masa_v6` tras eliminar
m6_*), descomponiendo Δy en la parte explicada por Δln FeO_inv[t−1] (visible en el estado) y el resto:

| target | var_total | var_ruido total | R²_max E9-05 | corr(Δy, Δestado_prev) | var ruido no revertible | **techo corregido** |
|---|---|---|---|---|---|---|
| FeO_ret | 0.00570 | 0.00246 | 0.568 | −0.71 (coef −1.00) | 0.00121 | **0.788** |
| Sn_dep | 0.4586 | 0.00102 | 0.998 | +0.72 | 0.00049 | 0.999 |

Consecuencias: el PLM de FeO está al 42 % del techo alcanzable, no al 58 %; la inferencia "ruido ≤ 2.8 %" cae (con 4 % el techo
corregido sigue > 0.33); y la autocorrelación de residuos no es diagnóstica: si el modelo absorbe ε_{t−1} vía el estado, el residuo
es ≈ −ε_t y su autocorrelación es ≈ 0 aun cuando fuera ruido puro (la predicción "−0.5" supone un modelo ciego al estado).
Corolario incómodo: parte del R² de FeO_ret (y de la señal de `%FeO_prev` en el estado) es reversión de ruido, no física; E8-03
midió 5.6 % del R² con esa firma.

**Veredicto.** DESCARTAR el cálculo y las dos inferencias derivadas; RESCATAR la conclusión del bake-off (ningún algoritmo supera al
PLM) y la curva saturada. Resolver: techo = 1 − var(ε_t no revertible)/var_total; medir la fracción de R² que es reversión de ruido
comparando OOF con y sin las features de t−1 que comparten ensayo con el target.

## 6. Evidencia de valor y diseño de piloto (`chk4_uplift.py`, `e9_08_potencia.csv`)

**Cadena θ → J → KPI.** `uplift_batch_pp = K_Sn·ΔSn_pred + K_FeO·ΔFeO_pred + β_F·ΔF_pred` (reconstruido, error 3e-4). Mediana
0.268, media 0.346, mínimo 0.000, 1 % ≤ 0. Descomposición media: Sn 0.025 pp, **FeO 0.284 pp (82 %)**, Fusión 0.038 pp. El IC
[0.25, 0.29] sólo remuestrea batches con K, w y θ fijos. Propagando (K_Sn, K_FeO) por bootstrap conjunto del anclaje: [0.20, 0.35];
añadiendo un factor de escala de θ con IC aproximado [0.25, 1.6] (el de Cx_sn; el de GN sobre FeO es aún más ancho): ≈ [0.12,
0.63]. "> 99 % de batches positivos" es tautológico: el optimizador maximiza la misma J con los mismos θ (uplift ≥ 0 por
construcción salvo penalizaciones), y no se corrige la maldición del optimizador (elige la acción donde el error del modelo es más
favorable; el cross-fitting protege contra ver el batch, no contra explorar el espacio de acciones con un modelo ruidoso). El 82 %
del valor descansa en θ(GN, O₂, aire → FeO) cuyos IC libres cruzan 0 (§1) y cuya identificación es del último tercio (§4).

**Off-policy.** `dist_*` = |hist − rec|/sd promediado: distancia simétrica que no distingue "más carbón que la política" de "menos".
dist_R +0.13 pp/sd (signo contrario al esperado, p 0.54), dist_F −0.49 (p 0.13; perm 0.06), polvo +0.0059 (p 0.013). Entre las 18
pruebas dist × KPI de DEV, q-BH del polvo = 0.24 (Holm 0.24); en TOTAL q 0.07. Con 0.9 falsos positivos esperados y 2 observados, la
familia es compatible con el nulo. El placebo (permutar dist entre batches) destruye toda estructura, incluida la dependencia
dist–estado, así que sólo demuestra que la regresión no inventa efectos con ruido puro; no descarta confusión. La distancia es
endógena: depende del estado (vía la recomendación) y de la desviación del operador (§1d).

**Potencia.** E9-08 es correcto en su aritmética (se 0.21 → MDE 0.59; potencia 32 % para 0.32). Para un piloto: KPI sd 4.51, sd
residual tras controles 4.50 (los controles no ayudan), KPI_sum2 sd 6.70. Con α 0.05 y potencia 80 %, A/B por batch:

| endpoint | δ | n por brazo | duración a ≈ 72 batches/mes |
|---|---|---|---|
| KPI refinado | +0.3 pp | 3 547 | ≈ 8 años |
| KPI refinado | +1.0 pp | 319 | ≈ 9 meses |
| Σ_R ln_feo_ret (sustituto) | +0.0103 (efecto predicho) | 1 280 | ≈ 3 años |
| Σ_R ln_sn_dep (sustituto) | +0.0071 | 69 000 | — |

Ni alternancia ni stepped-wedge cambian el orden de magnitud (reducen sesgo de tendencia, no varianza). El efecto predicho es
inverificable a nivel de batch en cualquier plazo razonable. Lo verificable es θ a nivel de escalón con perturbaciones deliberadas
de ±1 sd: para Cx_sn (θ₁sd 0.178, RMSE OOF ≈ 0.33) bastan ≈ 55 escalones por brazo (≈ 15 batches por brazo); para GN sobre FeO
(θ₁sd 0.008, RMSE ≈ 0.061) ≈ 900 escalones por brazo (≈ 230 batches). Diseño recomendado: aleatorización por escalón (R0-R1) de
la dosis de carbón a ±1 sd alrededor de la recomendación, bloqueada por batch; después GN. El A/B de KPI queda como seguimiento de
seguridad ("no empeora"), no como prueba de mejora.

**Veredicto.** DESCARTAR "las recomendaciones mejoran el batch" como afirmación demostrada; "no lo empeoran" queda EN DUDA (la
regresión de adherencia no tiene potencia ni para eso: MDE 0.59 pp). La cadena es un cálculo del modelo, no evidencia.

## 7. Pruebas múltiples

En nueve iteraciones se han evaluado ≥ 39 targets (it. 5), 66 variantes de masa (E8-01), 14 algoritmos + 81 configuraciones (E9-05),
≥ 15 sets de estado (E9-06), 8 targets de Fusión (E9-04), y hallazgos.md reporta ~40 valores p < 0.05. Sin pre-registro de la familia,
un p nominal 0.01 no está protegido. Con esa vara: **legado** p 2e-4 (sobrevive Bonferroni sobre ~250 pruebas) y replica en lockbox
(p 0.011) → protegido; **polvo** p 0.013 → no (q 0.24 dentro de su propia familia; no replica en lockbox: −0.004); **Cx_sn** IC
[0.041, 0.285] → robusto a 4 semillas, 3 sets y a la cota, pero fue "encontrado" en la iteración 7 sólo con el estado FULL y elegido
entre configuraciones; su cota inferior (0.04, tercio 0 = 0.000) es la parte frágil. Las señales de Fusión (dist_F, β_F −0.58 p 0.17,
GN +0.021 [0.001, 0.038] con lockbox −0.70) no están protegidas.

## 8. Recomendaciones priorizadas

1. Reportar θ libre y acotado; reclasificar "carbón selectivo" y "GN/O₂/aire sobre FeO" como no identificados; recomputar el uplift
   sólo con los θ interiores (Cx_sn, GN sobre Sn) y con IC que propague K, w y θ.
2. Corregir el techo de FeO (0.79) y retirar las inferencias sobre el ruido; medir la fracción de R² que es reversión de ruido.
3. Corregir K_FeO por EIV (×1.2-1.4) y estudiar la política bajo w ∈ [1, 11]; el lockbox dice w ≈ 1.
4. Declarar el lockbox gastado; walk-forward como métrica de generalización; nuevo holdout pre-registrado.
5. Piloto: aleatorización por escalón de carbón (R0-R1, ±1 sd) con target de escalón como endpoint (≈ 15 batches/brazo); el KPI
   como seguimiento. Abandonar la regresión de adherencia como evidencia.
6. Pre-registrar familias de pruebas y reportar q-BH en cada memo de evidencia.
