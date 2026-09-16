# Iteración 9 (2026-09-16) — el mejor enfoque predictivo/prescriptivo para Fusión y Reducción (v7)

Diseño (Fable 5.1). Punto de partida: prescriptor v6 (`modelo_predictivo_v6.py`, `candidate_win_model_v6.md`, hallazgos §19),
masa de escoria por cierre físico (`experimentos/v6/masa_v6.py`), targets log telescópicos y objetivo anclado en el KPI refinado.
Librería compartida: `experimentos/v7/v7_lib.py` (caché `experimentos/cache/df_v6.pkl`).

## Meta (/goal del usuario)

Encontrar el mejor enfoque analítico para un modelo predictivo/prescriptivo por escalón de Fusión y Reducción que:
(1) prediga bien (OOF + lockbox), (2) identifique con rigor el efecto de las variables de control (carbón, GN, aire, O₂, cal),
(3) sea pirometalúrgicamente coherente y parsimonioso, (4) tenga una función objetivo por escalón con agregación (escalón → grupo
de escalones → fase → batch) fundamentada, y (5) demuestre que seguir las recomendaciones mejora el rendimiento del batch.
Parar sólo cuando las métricas sean muy buenas o se demuestre que no hay enfoque mejor con los datos disponibles.

## Diagnóstico de partida (qué está resuelto y qué no)

| Componente | Estado v6 | Brecha |
|---|---|---|
| Masa de escoria / inventarios | Cierre físico, cierre de Sn 0.99, sin artefacto de trazador | pG/gG por campaña (no resoluble sin datos nuevos) |
| Target Reducción agotamiento Sn (`m6_ln_sn_dep`) | R² 0.765 / 0.800 | ¿está en el techo de ruido? ¿forma más parsimoniosa? |
| Target Reducción retención FeO (`m6_ln_feo_ret`) | R² 0.329 / 0.486 | techo por ruido de ensayo desconocido con v6; carbón sobre FeO no identificado |
| Fusión (`m6_ln_sn_dep`, ΔT) | R² 0.16 / −0.47 ; ΔT 0.50 / 0.04 | sin generalización al lockbox (deriva de campaña); objetivo débil (β_F p 0.17) |
| Palancas identificadas | Cx_sn +0.16*, GN +0.05* (Sn) / −0.012* (FeO); O₂/aire ≈ 0 | sólo con estado FULL (38); cal nunca re-evaluada sin el artefacto del trazador |
| Objetivo / agregación | J por escalón, Σ telescópica por fase, w y K anclados en el KPI | la política es **miope** (optimiza cada escalón por separado); no está demostrado que la secuencia óptima por escalón sea óptima para el batch |
| Evidencia | cadena θ→agregado→KPI +0.26/+0.37 pp; adherencia Fusión −0.42 pp/sd (p 0.17); Reducción nula | observacional, potencia baja; v5 y v6 discrepan en el carbón de Reducción |

## Hipótesis de la iteración (cada una con un experimento que la pueda refutar)

- **H1 (forma estructural)**: un modelo cinético/estequiométrico parsimonioso (≤ 10 parámetros, reducción competitiva SnO₂/FeO por el
  reductor disponible) alcanza un R² OOF comparable al PLM (Δ ≤ 0.03) con mejor identificación del carbón y más estabilidad temporal.
  Si no, el PLM con estado FULL sigue siendo la forma ganadora y el híbrido física+residuo es la alternativa. → E9-01.
- **H2 (cal)**: sin el trazador de CaO, la cal (tolva 6) tiene un efecto identificable a nivel de escalón (Fusión: ΔT, matriz/B2,
  agotamiento) y/o de batch (KPI refinado) que justifica una recomendación de PLAN (cal/carga) con rango y signo teórico. → E9-02.
- **H3 (consistencia dinámica)**: como los targets telescopan, el objetivo del batch depende sólo del estado terminal de Reducción;
  una política con horizonte (rollout sobre los escalones restantes) mejora el J de batch simulado frente a la miope, sobre todo
  moviendo carbón hacia los escalones con Sn abundante (selectividad alta). → E9-03.
- **H4 (Fusión)**: el fallo de lockbox en Fusión es deriva de campaña y no falta de señal; con reentrenamiento progresivo y/o un target
  robusto a la deriva, Fusión recupera R² > 0 en lockbox; y el estado al final de Fusión (T, %FeO, IRF, inventarios) predice el
  resultado de Reducción, lo que da a Fusión un objetivo de "preparar la Reducción". → E9-04.
- **H5 (techo)**: el R² OOF de los targets v6 está dentro de 0.05-0.10 del techo impuesto por el ruido de ensayo (%Sn, %FeO), la curva
  de aprendizaje está saturada y ninguna familia de algoritmos supera al PLM/HGB: no hay modelo mejor con estos datos. → E9-05.
- **H6 (parsimonia)**: existe un estado de ≤ 15 features (con la historia de dosificación) que conserva el R² (Δ ≤ 0.01) e identifica el
  carbón (IC de Cx_sn excluye 0). → E9-06.

## Experimentos (paralelos, Sonnet 5.1; Fable diseña, revisa y da veredicto)

Todos: datos `v7_lib.cargar_df()`; DEV/lockbox con `v7_lib.split`; OOF GroupKFold(5) por Batch en DEV = criterio; lockbox sólo
reporte; anti-fuga con `v7_lib.asegurar_seguras`; salidas `experimentos/v7/e9_0N_*.csv`, memo `e9_0N_resultados.md`, log
`e9_0N_log.txt`. Python `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=3`, sin joblib.

| Exp | Pregunta | Entregable clave |
|---|---|---|
| E9-01 | Formas del modelo de Reducción: PLM v6 (ref) vs forma en tasa (target/Δt) vs cinético estructural (NLS, reducción competitiva) vs log-lineal cinético vs híbrido estructural+residuo HGB | tabla OOF/lockbox × forma × target; parámetros con IC; identificación del carbón; estabilidad por tercios; veredicto |
| E9-02 | La cal como palanca bajo v6: escalón (Fusión: ΔT, ΔB2, Δln IRF, agotamiento; Reducción) y batch (KPI refinado, canales, Sn final en escoria) con controles y heel; dosis-respuesta; óptimo de basicidad | veredicto cal = PLAN con rango recomendado o no palanca; signo vs teoría |
| E9-03 | Política con horizonte en Reducción: simulador de estado con los PLM v6, rollout sobre escalones restantes vs miope vs histórica; descomposición J escalón → grupo (R0-R1 / R2-R3) → batch (telescópica) | J de batch simulado por política (DEV cross-fitted), perfil de carbón/GN por escalón, robustez a perturbación de θ |
| E9-04 | Fusión: reentrenamiento progresivo, targets robustos a deriva, valor del estado al final de Fusión para la Reducción y el KPI, canal polvo | formulación de Fusión con métricas honestas y términos de objetivo defendibles |
| E9-05 | Techo de ruido (Monte Carlo del ensayo), curva de aprendizaje, bake-off de algoritmos, R² por orden/tercio, semillas | tabla "techo vs alcanzado"; veredicto "no hay modelo mejor" o dónde queda margen |
| E9-06 | Estado parsimonioso de Reducción que conserve R² e identifique el carbón (k ∈ {6, 8, 10, 12, 16, 20, 38}) | set recomendado, θ con IC, estabilidad por tercios y walk-forward |

Después (Fable): síntesis → `modelo_predictivo_v7.py` (forma ganadora, cal si procede, política con horizonte si procede,
Fusión reformulada), E9-07 validación + política cross-fitted + evidencia off-policy (362 batches), `candidate_win_model_v7.md`,
hallazgos §20, README, artefacto.

## Criterios de decisión (fijados antes de ver resultados)

1. Predictivo: R² OOF DEV (selección); lockbox y walk-forward (robustez). Se acepta un cambio de forma sólo si ΔR² OOF ≥ +0.01 o si
   con ΔR² ≥ −0.03 gana en parsimonia (≤ 1/3 de parámetros) **y** en identificación/estabilidad del efecto de las palancas.
2. Palancas: efecto con IC95 % bootstrap por batch que excluya 0 y signo de teoría; estable en ≥ 2 de 3 tercios.
3. Objetivo: w, K y β re-anclados con el KPI refinado si cambian los targets; la agregación escalón → batch debe ser exacta (telescópica)
   o explícitamente aproximada.
4. Política: se adopta la de horizonte sólo si el J de batch simulado mejora ≥ 10 % sobre la miope en DEV cross-fitted y la mejora
   sobrevive a perturbaciones de θ (bootstrap); si no, la miope queda con justificación.
5. Cal: se adopta como PLAN sólo si el efecto de batch tiene p < 0.05 en DEV con controles y el signo es explicable; la recomendación
   es de rango histórico (P25-P75), nunca de extrapolación.
