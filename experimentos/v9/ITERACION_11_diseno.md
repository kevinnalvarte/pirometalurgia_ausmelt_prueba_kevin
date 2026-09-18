# Iteración 11 (2026-09-16/17) — prescriptor v9: anclar el valor de cada CONTROL en el resultado del batch por el canal de la acción

Diseño (Fable 5.1). Punto de partida: v8 (`modelo_predictivo_v8.py`, `candidate_win_model_v8.md`, hallazgos §22) y todo lo anterior
(§13-§21). Librería: `experimentos/v9/v9_lib.py` (residuos de la acción cross-fitted, exposiciones por batch, prueba anidada OOF).

## 1. Diagnóstico: por qué v8 no cumple el requisito obligatorio y qué pista no se explotó

| Hecho (iteración 10) | Lectura |
|---|---|
| `uplift_batch_pp` y adherencia de v8 vs KPI: nulos; un HGB directo no reconoce la ganancia; uplift +0.07 pp/batch | El objetivo J de v8 (K 2.78, w 5.93) está anclado regresando el KPI sobre los **targets agregados**, cuya varianza es sobre todo de **estado** (mineral, talón, campaña), no de acción. Ese anclaje no tiene por qué valorar bien lo que la acción mueve |
| Residuo de la acción (a − E[a\|S]) agregado por batch → KPI: GN −0.46 pp/sd (p 0.04), carbón tardío +0.81 pp/sd (p 0.015, Spearman p 0.0015) | Es la única evidencia **directa** control → rendimiento, y contradice en magnitud a J (que da al GN un neto ≈ 0 y al carbón tardío ≈ 0). Se usó sólo como "salvaguarda", nunca como ancla |
| w ≈ 1 en lockbox (fin de campaña), GN cambia de dirección con w | Un peso constante en escala log no puede representar que el valor de reducir depende de **cuánto Sn queda** |

Piloto de diseño (Fable, `_piloto1.py`, `_piloto2.py`; DEV n 297, residuo z medio por grupo, OLS HC3 conjunta + controles), canales del KPI:
- GN en Reducción → **dross ↑** (+0.4..+0.8 pp/sd) y metal ↓: coherente con el escalón (GN → −retención de FeO, p 1e-4): reductor gaseoso no selectivo → Fe metálico → hardhead.
- Carbón en Reducción → **Sn perdido en escoria ↓** (R0-R1 −0.27 pp/sd, p 0.01; R2-R3 −0.16, p 0.03) y KPI ↑: coherente con el escalón (C × Sn disponible → +agotamiento).
- Carbón en Fusión → **polvo ↑** (+1.0 pp/sd, p 0.01) y dross ↓: neto ≈ 0 (confirma E10-06).
- Lanza más oxidante en Fusión (`exo2`) → KPI −0.9..−1.4 pp/sd (p 0.09 DEV / 0.003 total), dross ↑: **nuevo**, a auditar.
- Moderación por el Sn que queda: GN con %Sn previo bajo → −1.98 pp/sd (p 0.002 DEV), con %Sn alto → −0.58 (n.s.); en lockbox GN con Sn alto → **+2.8** (p 0.003) y con Sn bajo −1.6 (n.s.). El signo del valor del GN depende del Sn disponible en los dos regímenes (desplazado de nivel en fin de campaña, horno ~100 °C más frío). El carbón es positivo en los dos regímenes.
- Prueba anidada con 6 β libres: pendiente 0.33 (p 0.14, GroupKFold) / 0.53 (p 0.033, cronológica): hay señal pero 6 parámetros libres la diluyen → hace falta estructura física (pocos parámetros).

Nota de honestidad: el lockbox ya estaba "gastado" (dictamen) y el piloto lo miró para la moderación; la inferencia confirmatoria de esta
iteración descansa en (a) CV **anidada** sobre batches no vistos (aleatoria y cronológica) con permutación, (b) especificaciones fijadas
por teoría antes de estimar, (c) réplica de signo por régimen (DEV / lockbox) y (d) placebos y análisis de sensibilidad a confusión.

## 2. Hipótesis (cada una refutable)

- **H1 (vínculo directo robusto)**: los vínculos GN_R → KPI (−, vía dross) y carbón_R → KPI (+, vía Sn en escoria) sobreviven a: regresión conjunta,
  estado de la política de comportamiento más rico/más pobre, ponderación por duración, controles de batch, BH en la familia, placebos
  (exposición del batch SIGUIENTE → KPI actual = 0; exposición → variables pre-tratamiento = 0) y valor de robustez (confusor no observado).
  → E11-01.
- **H2 (valor del reductor dependiente del Sn disponible)**: β_GN(s) cruza 0 en un umbral de Sn disponible (positivo con Sn alto, negativo con Sn
  agotado) y β_C(s) ≥ 0 creciente con el Sn disponible; el patrón replica en DEV y lockbox y en CV cronológica. Teoría: selectividad
  a_SnO/a_FeO; criterio de fin de reducción. → E11-02.
- **H3 (objetivo coherente con la cadena)**: con los θ de escalón de v8 (GN → Sn y FeO; C×Sn → Sn), el objetivo **lineal en kg**
  `J = λ_Sn·ΔSn_kg(a) − λ_Fe·ΔFeO_kg(a)` anclado por el canal de la acción (KPI ~ Σ_t Δ̂Sn_kg(res_t), Σ_t Δ̂FeO_kg(res_t)) (i) predice el KPI de batches
  no vistos mejor que el anclaje log de v8 y que los β libres, (ii) reproduce la moderación de H2 sin parámetros extra, (iii) pasa la prueba de
  sobre-identificación (los residuos crudos no añaden nada a la cadena) o, si no, identifica el canal directo que falta (polvo/temperatura). → E11-03.
- **H4 (KPI más preciso)**: existe una medida de rendimiento con mejor cociente señal/ruido para la acción que `recuperacion_refinada_pct`
  (p. ej. sin polvo —cuya atribución por batch es periódica y no responde a nada— o con polvo suavizado/por campaña, o media de 2 batches por
  el legado), sin perder sentido metalúrgico (el polvo SÍ responde al carbón de Fusión: no se puede ignorar allí). → E11-04.
- **H5 (Fusión)**: el efecto directo de `exo2_F` (−) y el par carbón_F (polvo ↑ / dross ↓) son robustos (mismos filtros que H1) y permiten una
  recomendación de Fusión distinta de "receta"; si no, Fusión sigue en receta. → E11-05.

## 3. Experimentos (ola 1 en paralelo, Sonnet 5; Fable diseña, revisa y decide)

Todos: `sys.path.insert(0, "experimentos/v9"); import v9_lib as L`; `res = L.residuos_escalon()` (caché), `t = L.tabla_batch()`,
DEV = `t[t.es_dev]`, lockbox = `t[~t.es_dev]`; `L.prueba_anidada(...)`; KPI `L.KPI`; controles `L.CONTROLES` (en lockbox quitar `espesor`
si es colineal). `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=3`, sin joblib, en primer plano. Salidas
`experimentos/v9/e11_0N_*.csv`, memo `e11_0N_resultados.md`, script `e11_0N_*.py`.

## 4. Criterios de decisión (fijados antes de ver los resultados de la ola 1)

1. Un vínculo control → KPI se acepta si: signo de teoría; p < 0.05 (HC3) en DEV en la regresión conjunta **y** q-BH < 0.10 en su familia; mismo
   signo en ≥ 2/3 tercios cronológicos; placebo (batch siguiente) no significativo; canal (metal/dross/polvo/escoria) coherente con el mecanismo
   de escalón; valor de robustez RV_q=1 ≥ 5 % (un confusor debería explicar ≥ 5 % de la varianza residual de acción y de KPI para anularlo).
2. Moderación (H2): interacción continua con p < 0.05 en DEV o total, mismo signo de la pendiente en DEV y lockbox, umbral con IC bootstrap
   dentro del rango observado.
3. Objetivo (H3): se adopta la forma (log v8 / lineal kg / β directos moderados) con mayor pendiente-t en la prueba anidada (GroupKFold ×20 y
   cronológica) con ≤ 3 parámetros libres; empate → la más física (cadena). Calibración: pendiente del KPI sobre el puntaje en [0.5, 1.5].
4. **Requisito obligatorio (demostración)**: puntaje de concordancia OOF (valor que la política atribuye a lo que el operador hizo, calculado
   con un sistema entrenado sin ese batch) con pendiente > 0, p unilateral < 0.01 por permutación en DEV-anidada **y** < 0.05 en la CV
   cronológica sobre los 362 batches, mismo signo en lockbox, monotonía por terciles, y canal coherente (dross/escoria). Magnitud reportada con
   IC bootstrap y expresada en pp de KPI y kg de Sn por batch para un desplazamiento realista (≤ 1 sd por escalón).
5. KPI (H4): se cambia de KPI principal sólo si el alternativo tiene mayor fiabilidad (autocorrelación/ruido) **y** mayor t de los vínculos
   aceptados, sin excluir un canal por donde actúe una palanca recomendada.
6. Parsimonia/estabilidad: nada entra al prescriptor sin IC libre que excluya 0 o sin réplica de signo por régimen.

## 5. Ola 2 (Fable)

`modelo_predictivo_v9.py`: v8 (modelos de escalón, cajas, soporte, ventana térmica) + objetivo v9 (forma ganadora de H3/H2) + recomendación
relativa a la práctica habitual E[a|S] (no a la acción histórica, para que la dirección recomendada no dependa de lo que el operador hizo).
E11-06: simulación batch-no-visto escalón a escalón (5 folds + lockbox), tablas de política, prueba obligatoria (criterio 4), sensibilidad.
