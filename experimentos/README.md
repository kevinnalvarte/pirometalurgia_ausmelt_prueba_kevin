# experimentos/

Scripts exploratorios de la sesión "modelo predictivo v2" (2026-09-13 tarde) que llevaron a los
resultados consolidados en `modelo_predictivo_v2.py`, `hallazgos.md` sección 13 y el notebook
`modelo_v2_causalidad_y_prescripcion.ipynb`. Se conservan tal cual se usaron durante el desarrollo
(narrativa de investigación, no código de producción) para trazabilidad del proceso; el código
reutilizable y mantenido vive en los módulos de la raíz del proyecto.

Se pueden ejecutar directamente desde la raíz del proyecto o desde esta carpeta (las rutas al Excel
y a los módulos se resuelven solas, ej. `python experimentos/01_causal_dml.py`).

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `01_causal_dml.py` | Double ML con cross-fitting + refutation tests (placebo/confusor/subset) | `modelo_predictivo_v2.dml_partialling_out` / `tabla_efectos_causales` |
| `02_feature_selection_avanzada.py` | Mutual information + permutation importance estable + Boruta-shadow sobre las 79 candidatas seguras | Motivó Config C (`modelo_predictivo_v2.FEATURES_STATE_ACTION_V2`) |
| `03_modelo_batch.py` | Primera versión del modelo de batch con features de trayectoria (38 features, sobreajuste con OOT negativo) | Diagnóstico que llevó a la versión parsimoniosa (03b) |
| `03b_modelo_batch_parsimonioso.py` | Versión reducida (top-10/top-5) que recupera estabilidad OOT | `modelo_predictivo_v2.entrenar_modelo_batch` |
| `04_bakeoff_y_conformal.py` | Bake-off de algoritmos con restricciones monotónicas locales + CV+ conformal prediction | `modelo_predictivo_v2.tabla_validacion_v2` / `validar_cobertura_lockbox` |
| `05_reduccion_featureset_v2.py` | Validación de Config C (decorrelación + comparación 3 niveles) | `modelo_predictivo_v2.FEATURES_STATE_ACTION_V2["Reducción"]` |
| `build_notebook.py` | Genera `modelo_v2_causalidad_y_prescripcion.ipynb` desde cero (nbformat) | El notebook mismo |

### Sesión "v3: carga por tolva, masa de escoria y prescriptor en kg de Sn" (2026-09-13 noche)

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `06_exploracion_v3_targets_features.py` | Targets (Δ%Sn, T3, kg de Sn extraído, fracción extraída, Δ%FeO, kg FeO) × feature sets (v2 actual / v3 carga / v3 carga+inventario / pool seguro) × algoritmos (HistGB monotónico, XGBoost), OOF GroupKFold + lockbox por fase; permutation importance del pool | `modelo_predictivo_v3.TARGETS_V3`, hallazgos.md 14.3 |
| `07_causalidad_palancas_v3.py` | Double ML cross-fitted (nuisance HistGB) + placebo/subset para las palancas nuevas (mezcla de carga, cal/carga, carbón, GN, O2, duración planificada, lanza) sobre kg Sn extraído, Δ%Sn y Δ%FeO, con el estado v3 como control | hallazgos.md 14.4, `LAMBDA_FE`/pesos de `modelo_predictivo_v3` |
| `08_modelo_batch_canales_perdida_v3.py` | Modelo de batch por canal de pérdida (Sn a dross Fe, Sn a polvo, rendimiento HA) con features de trayectoria v3 incluyendo rollups en masa por trazador CaO; Spearman + XGB/ElasticNet OOF/walk-forward/lockbox | hallazgos.md 14.6 |
| `09_seleccion_features_v3.py` | Permutation importance estable + Boruta-shadow + decorrelación (\|ρ\|>0.85) + curva de tamaño de set, por (fase, target). Se detuvo tras 5 de 8 pares (log parcial) por costo de CPU; sus sets sirvieron de guía, no de configuración final | Insumo de `11_finalizar_config_v3.py` |
| `10_validacion_final_v3.py` | Validación final del sistema v3 congelado: tabla OOF/lockbox, cobertura CV+, DML conjunto de palancas, política recomendada sobre 8 batches del lockbox, reporte de batch y sensibilidad local | `modelo_predictivo_v3`, hallazgos.md 14.5/14.7 |
| `12_diag_deriva_fusion_log.txt` | (script inline, solo log) Diagnóstico de deriva de campaña: qué familias de features v3 degradan el lockbox de Δ%Sn en Fusión y distribución dev vs lockbox de refractario/termocuplas/leyes | hallazgos.md 14.3 punto 6 |
| `11_finalizar_config_v3.py` | Sets CURADOS interpretables por (fase, target) vs. baselines triviales (solo Sn cargado / solo inventario), FeO en kg vs. puntos, diagnóstico de la fracción extraída dev vs lockbox, signos SHAP | `modelo_predictivo_v3.FEATURES_V3_POR_DEFECTO`, hallazgos.md 14.5 |
| `build_notebook_v3.py` | Genera `modelo_v3_masa_escoria_y_prescriptor.ipynb` desde cero (nbformat) | El notebook mismo |

Los `06_*.csv`, `07_*.csv`, `08_*.csv`, `09_*.csv`, `11_*.csv` y `*_log.txt` son las salidas ya calculadas de cada script.

Los `.csv` son las salidas ya calculadas de cada script (tablas de MI/permutation/Boruta, consenso,
importancias del modelo de batch, tabla de bake-off, tabla causal) — evitan tener que re-ejecutar
todo para inspeccionar un resultado puntual.

**Nota:** estos scripts usan menos disciplina que los módulos de producción (sin la aserción
anti-fuga de `feature_engineering.filtrar_features_seguras`, sin las mismas convenciones de
docstring) — son notebooks de investigación en formato `.py`, no código a mantener. Cualquier
cambio real de comportamiento del sistema prescriptivo debe hacerse en `modelo_predictivo_v2.py`.

### Iteración 4: "modelo prescriptor ganador" (2026-09-14, `/goal`; diseño en `ITERACION_4_diseno.md`, ficha en `../candidate_win_model.md`)

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `_cache_dataset_v3.py` | Construye una vez el dataset v3 y lo guarda en `cache/df_v3.pkl` (5 s; no es fuente de verdad) | uso interno de 13-21 |
| `13_evidencia_politica_batch.py` | Recomendaciones v3 cross-fitted para los 362 batches (5 folds + lockbox, una etapa por proceso) y adherencia histórica vs rendimiento (OLS HC3, placebo, permutación) → evidencia NULA para v3 | hallazgos.md §15.1 |
| `14_formas_modelo_palancas.py` | HGB v3 vs monotonía teórica vs modelo parcialmente lineal (PLM) vs cinético log-lineal vs fracción; signos de palancas; parsimonia | `modelo_predictivo_v4.ModeloPLM` |
| `15_senal_sobre_reduccion_feo.py` | Techo de ruido del trazador, targets alternativos de FeO, agregado por fase, λ_Fe re-estimado, selectividad marginal FeO/Sn por avance | hallazgos.md §15.3, `LAMBDA_FE_RANGO` |
| `16_formas_pd_teoria.py` | PD/ALE/SHAP de los modelos v3 vs tabla teórica; interacciones; forma de J por palanca | hallazgos.md §15.2, notebook win §5 |
| `17_estabilidad_temporal.py` | Walk-forward cronológico, reentrenamiento por ventana/progresivo, sets sin contexto de campaña, efectos DML por período, deriva de features | hallazgos.md §15.5 |
| `18_artefacto_trazador_cal.py` | ¿La cal en Fusión es efecto físico o artefacto del trazador? Balance de CaO, DML con rezago, batch-level, reversión de ruido | retiro de la cal como palanca (v4) |
| `19_palancas_fase_vs_kpi_batch.py` | Decisiones agregadas por fase vs canales de pérdida del batch (OLS HC3 con controles, VIF, HGB, PD) | `BETA_KPI_FUSION`, `LAMBDA_FE = 0.60` |
| `20_plm_parsimonioso_v4.py` | PLM con estado curado por fase/target, variantes del carbón, curva de parsimonia, curvas de respuesta y forma de J | `modelo_predictivo_v4.ESTADO_V4/PALANCAS_V4` |
| `21_smoke_v4.py`, `21_validacion_v4.py` | Validación del sistema v4 (OOF/lockbox vs HGB v3, θ con IC, predicciones para gráficas, curvas de respuesta) y recomendaciones cross-fitted + evidencia off-policy (etapas `validacion`, `0..4`, `lockbox`, `evidencia [w_soporte]`) | `candidate_win_model.md`, `win_analytics.ipynb` |
| `build_notebook_win.py` | Genera `win_analytics.ipynb` (nbformat) | El notebook mismo |

Lección operativa de esta iteración: `joblib`/loky se colgaba en Windows con los modelos de sklearn (3 h sin progreso); los
cross-fittings se lanzan como procesos independientes por fold (`python 21_validacion_v4.py <fold>`), con `OMP_NUM_THREADS` bajo.

### Iteración 5: `search_targets/` — qué target por escalón debe maximizar el recomendador (2026-09-14 tarde, `/goal`; README propio en la carpeta, hallazgos.md §16)

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `search_targets/st_targets.py` | Librería: 36 targets candidatos por escalón (+2 ganadores) con fórmula, dirección, agregación coherente escalón→batch y referencia teórica; `st_batch.csv` | `search_targets/README.md` |
| `search_targets/st_01_correlacion_batch.py` | Spearman/Pearson vs KPIs en DEV/lockbox/total, IC bootstrap, BH por fase, OLS HC3 con controles, dosis-respuesta, barrido de λ, combinación F+R, redundancia | README §2 |
| `search_targets/st_02_predictibilidad_escalon.py` | HGB OOF por batch con estado / estado+control / control; techo de ruido del trazador (Monte Carlo); importancias | README §4 |
| `search_targets/st_03_identificacion_control.py` | PLM v4 (θ carbón/GN/O₂/aire con IC) para los 36 candidatos; cadena de signos vs exp 19 | README §4 |
| `search_targets/st_04_robustez.py` | Agregaciones alternativas, estabilidad por tercios cronológicos, Monte Carlo del ruido de ensayo, bootstrap del ranking | README §5 |
| `search_targets/st_05_hipotesis_ronda2.py` | Hipótesis Fable: índice compuesto log (SDI_w), origen del IRF (plan vs estado inicial), heel químico de escoria, efecto sobre el batch siguiente | hallazgos §16.3 |
| `search_targets/st_06_ganador.py` | Selección de w sólo con DEV, métricas finales de los ganadores vs referencias v4, efecto sobre el batch siguiente, targets por escalón (`st_06_targets_escalon.csv`) | hallazgos §16.1 |
| `search_targets/st_07_plm_ganadores.py` | PLM v4 y predictibilidad OOF de los ganadores (SDI, componentes, IRF, ΔIRF) | hallazgos §16.4 |
| `search_targets/st_08_techo_kpi.py` | Ronda 2: fiabilidad del KPI (proxy vs recuperación real, cierre del balance) y techo empírico con toda la información de escalón (2000 agregados; Ridge/ElasticNet/HGB, CV y walk-forward) | hallazgos §16.6 |
| `search_targets/st_09_familia2.py` | Ronda 2: 30 candidatos teóricos nuevos (balances Fe/Sn, matriz NBO/T-IRF-fluidez, sobrecalentamiento, polvo, compuestos) con la batería de ST-01 + Monte Carlo | README §R2.3 |
| `search_targets/st_10_compuesto_supervisado.py` | Ronda 2: compuesto supervisado (Ridge/Lasso/EN/HGB, CV 5×5, walk-forward) y compuesto parsimonioso de ≤ 6 términos | README §R2.1/R2.3 |
| `search_targets/st_11_kpi_y_polvo.py` | Ronda 2: estructura del KPI (identidad 1 − f_dross − f_polvo, autocorrelación lag 3), KPIs alternativos, candidatos de polvo y dross | README §R2.1 |
| `search_targets/st_12_matriz_reduccion.py` | Ronda 2: formalización de `R23_lnirf_R` (ln IRF en Reducción, twmean): bloques, IC, OLS, batch siguiente, Monte Carlo, PLM; comparación con SDI | README §R2.2, hallazgos §16.6 |
| `search_targets/st_13_atribucion_dross_polvo.py` | Ronda 3: correlación cruzada de dross/polvo/metal con señales de proceso a lags −6..+6 (¿atribución retardada?), KPIs alternativos (recuperación real, ventanas, desplazamientos, canal dross) vs targets; ranking de los 39 candidatos contra la recuperación real | README search_targets §R3 |
| `search_targets/st_15_techo_kpis_alternativos.py` | Ronda 3: techo de predictibilidad (HGB/Ridge, CV) de los KPIs alternativos y correlación de los targets teóricos con cada uno | README search_targets §R3.3 |

### Iteración 6: `balances/` — balances de masa y energía inspirados en `func-teorethical-model` (2026-09-14 noche, `/goal`; README propio en la carpeta, hallazgos.md §17)

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `balances/bal_features.py` | Librería bloque v6 (`b6_*`): cierre multi-trazador, balance reductor/oxidante en kg C-eq, balance de energía por escalón, balance global por batch; clasificación anti-fuga y supuestos parametrizados | `feature_engineering.construir_resumen_batch_balance_global` (solo la parte adoptada) |
| `balances/b01_multitrazador.py`, `b01b_multitrazador_ronda2.py` | Consistencia entre inertes, ruido (Monte Carlo y lag-1), targets multi vs CaO (batería ST-01/02/03), residuo de masa; consensos alternativos | RECHAZADO (hallazgos §17.2) |
| `balances/b02_balance_reductor.py` | PLM v4 con reductor neto / capacidad / acumulados; utilización del reductor; sensibilidad; curvas de respuesta | RECHAZADO como sustituto; utilización ≈ 1 (feedback al modelo Rust) |
| `balances/b03_balance_energia.py` | Descomposición térmica, calibración NNLS, modelo de ΔT con features térmicas, calor de reacción aparente, sensibilidad | térmicas en Reducción (estado opcional); físico y estimador rechazados |
| `balances/b04_balance_global.py` | Balance global por batch, factor de escala del trazador, KPI refinado, targets vs KPI refinado, techo, modelo multi-óxido | `recuperacion_refinada_pct` ADOPTADO; multi-óxido RONDA 2 |
| `balances/b05_hipotesis_fable.py`, `b05b_avance_feo_estado.py` | Cementación por dross de Fe (H1), monitor de Fe (H2), avance de FeO como estado | H1 no se observa; H2 cubierto por el SDI; estado RONDA 2 |

### Iteración 7: `v5/` — prescriptor v5 con objetivo en escala log anclado en el KPI refinado (2026-09-14/15, `/goal`; diseño `v5/ITERACION_7_diseno.md`, hallazgos.md §18)

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `v5/v5_lib.py` | Librería: targets v5 (`v5_ln_sn_dep`, `v5_ln_feo_ret`, `v5_lnirf`, `v5_d_lnirf`), palancas v5 (dosis específica de C, carbón sobrante, C×avance, GN específico), tabla de batch con KPI refinado, batería Spearman/OLS, `evaluar_plm` | `modelo_predictivo_v5.py` (targets/palancas) |
| `v5/e7_01_reduccion_componentes.py` | Reducción: PLM de las componentes log con 5 parametrizaciones de palancas × 2 estados; anclaje de w en el KPI refinado (OLS + bootstrap), barrido de w, variante ponderada por tiempo, cadena de signos | `CONFIG_V5` (w = 6.04, K = 2.12 pp/unidad), `e7_01_resultados.md` |
| `v5/e7_02_fusion.py` | Fusión: regresiones de batch re-ancladas (KPI refinado, canales, cuadráticos, heel), matriz vs heel, PLM por escalón, cadena de signos | J_F = conservación del Sn + ventana térmica, `e7_02_resultados.md` |
| `v5/e7_03_formas.py` | Formas funcionales: cuadráticos residualizados (Robinson), θ por tramo de avance, PD/ALE/SHAP vs tabla teórica, curva de J | forma lineal por palanca con signos de teoría (`ModeloPLMSignos`), `e7_03_resultados.md` |
| `v5/e7_04_estabilidad.py` | Parsimonia del estado, θ por tercios cronológicos y lockbox, walk-forward, reentrenamiento por campaña, deriva | `e7_04_*.csv`, `e7_04_resultados.md` |
| `v5/e7_05_validacion_v5.py` | Validación del sistema v5 (OOF/lockbox vs HGB, θ con IC, predicciones, curvas de respuesta), recomendaciones cross-fitted por fold y evidencia off-policy con KPI refinado (+placebo) | `candidate_win_model.md`, `win_analytics_v1.ipynb` |
| `v5/e7_06_evidencia_extra.py` | Emparejamiento por estado, dosis-respuesta por terciles, cadena θ→agregado→KPI con bootstrap | `e7_06_resultados.md` |
| `v5/e7_07_parsimonia_v5.py` | Re-evalúa los sets parsimoniosos de E7-04 con las palancas y signos definitivos de v5 (R² y θ) | `candidate_win_model.md` §4, `e7_04_resultados.md` |
| `v5/e7_08_variante_lanza.py` | Variante: posición de lanza como palanca (lineal + cuadrática, signo libre): validación, folds, evidencia con sufijo `_lanza` | `e7_08_resultados.md`, hallazgos §18.5 |
| `v5/e7_09_diag_fusion_lockbox.py` | Diagnóstico del R² lockbox negativo del agotamiento de Sn en Fusión (sesgo, correlación, θ sólo lockbox, reentrenamiento progresivo) | hallazgos §18.6 |
| `build_notebook_win_v1.py` | Genera `win_analytics_v1.ipynb` (nbformat) | El notebook mismo |

Lección operativa: el término absoluto de soporte `−w_S·(1−s)` de v4 dominaba la física y empujaba las recomendaciones hacia la moda
histórica; v5 penaliza sólo la pérdida de soporte respecto de la acción histórica (`soporte_ref`). Los θ de las palancas se estiman con
mínimos cuadrados acotados por el signo de la teoría (un reductor no puede aumentar la retención de FeO): los efectos de signo imposible
colapsan a 0 en vez de entrar al objetivo.
