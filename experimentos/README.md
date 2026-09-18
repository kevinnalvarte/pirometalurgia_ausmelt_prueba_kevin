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


### Iteración 8: `v6/` — masa de escoria por cierre físico y prescriptor v6 (2026-09-16; diseño `v6/ITERACION_8_diseno.md`, hallazgos.md §19)

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `v6/masa_v6.py` | Iteración 8: estimador de masa de escoria v6 (cierre físico del resto no reducible, coeficientes efectivos calibrados en DEV contra el balance global), inventarios/targets/palancas m6_*, batería de evaluación común | `modelo_predictivo_v6.py`, hallazgos §19 |
| `v6/e8_01_masa_variantes.py` | Grilla de 66 variantes del estimador (fusión × reducción × δ_R × w_obs × aC), calibración por tercios/folds, sensibilidad ±30 %, propagación de ruido de ensayo, casos borde, diagnóstico de `m6_k_anclaje`, QC | PARAMS_DEFAULT (cierre/cierre, δ 0.01), `e8_01_resultados.md` |
| `v6/e8_02_energia_v6.py` | Balance de energía por escalón con la masa v6: features térmicas e6_*, modelos de ΔT (4 configuraciones), energía específica como palanca, cierre del calor de reacción | Balance de energía = diagnóstico, no feature; `e8_02_resultados.md` |
| `v6/e8_03_anclaje_v6.py` (+ `v6_lib.py`) | Anclaje de w/K/β_F con los targets v6, batería Spearman, PLM v5 vs v6 sobre las mismas filas, control de reversión de ruido, cadena de signos | CONFIG_V6 (w 6.09, K 2.89, β_F −0.58), `e8_03_resultados.md` |
| `v6/e8_05_validacion_v6.py` | Validación del sistema v6 (OOF/lockbox vs HGB y vs PLM v5, θ con IC, predicciones, curvas de respuesta), política cross-fitted por fold y evidencia off-policy (+placebo) | hallazgos §19.4-19.5, artefacto |
| `v6/e8_06_datos_figuras.py` | Exporta `figs/datos_v6.js` (trayectorias, grilla, calibración, SHAP/PDP, predicciones, curvas, evidencia) para el artefacto | `figs/masa_escoria_v6.html` |
| `v6/e8_07_evidencia_extra_v6.py` | Emparejamiento por estado, dosis-respuesta, cadena θ→agregado→KPI con bootstrap, comparación v5 vs v6 por batch, política por tramo | `e8_07_resultados.md`, hallazgos §19.5 |


### Iteración 9: `v7/` — el mejor enfoque predictivo/prescriptivo (2026-09-16, `/goal`; diseño `v7/ITERACION_9_diseno.md`, hallazgos.md §20)

Datos: caché `cache/df_v6.pkl` (`v6/_cache_df_v6.py`); librería `v7/v7_lib.py`. Seis hipótesis con experimento refutable cada una (Sonnet 5.1 en
paralelo), análisis de agregación/anclaje por Fable (E9-00) y síntesis en `modelo_predictivo_v7.py` + `candidate_win_model_v7.md`.

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `v7/e9_00_agregacion_fable.py`, `e9_00b_legado_fable.py`, `e9_00c_anclaje_fusion_fable.py` | Identidad telescópica escalón→fase, anclaje del KPI por grupo de escalones (Wald de homogeneidad), varianza de J por grupo, término de legado (KPI del batch siguiente, placebo temporal, mediación por el IRF terminal), anclaje de Fusión como nivel terminal | `CONFIG_V7` (K 3.52, w 7.86), `e9_00_resultados.md`, hallazgos §20.1-20.2 |
| `v7/e9_01_formas.py` | Formas de Reducción: PLM (ref) vs tasa vs cinético estructural NLS (7 par.) vs log-lineal vs híbridos | PLM se mantiene (estructural 0.34 vs 0.77), `e9_01_resultados.md` |
| `v7/e9_02_cal.py` | Cal/basicidad como palanca bajo la masa v6: batch (KPI, canales, dosis-respuesta), escalón F/R, óptimo de basicidad | Cal NO es palanca (p 0.145; el efecto de E7-02 era artefacto), `e9_02_resultados.md` |
| `v7/e9_03_horizonte.py` | Simulador de estado con los PLM, política con horizonte (rollout) vs miope vs histórica cross-fitted, robustez a θ y pesimista, agregación temprano/tardío | Horizonte NO adoptado (mediana 0 %, robusto 12 %); J temprano ρ 0.35 vs tardío 0.09, `e9_03_resultados.md` |
| `v7/e9_04_fusion.py` | Fusión: diagnóstico por orden, reentrenamiento walk-forward/progresivo, 8 targets, valor del estado final, canal polvo | F1 modelable (gradiente imputado), ΔT reentrenable (0.16-0.32), agotamiento no generaliza, `e9_04_resultados.md` |
| `v7/e9_05_techo.py` | Techo de ruido de ensayo (Monte Carlo), curva de aprendizaje, bake-off de 14 algoritmos + grid HGB + stacking, desglose | R²_max FeO 0.57 (PLM 0.33/0.49), Sn ≈ 1.0; curva saturada; ningún algoritmo supera al PLM, `e9_05_resultados.md` |
| `v7/e9_06_parsimonia.py` | Importancia por permutación y para la identificación, eliminación hacia atrás (libre/física), sets por teoría, estabilidad, varianza de θ | `ESTADO_R_V7` = libre_k16 (16 features, carbón identificado, mejor walk-forward), `e9_06_resultados.md` |
| `v7/e9_07_validacion_v7.py` | Validación del sistema v7 (OOF/lockbox, θ, predicciones, curvas), política cross-fitted por fold + lockbox, evidencia off-policy, agregación escalón→grupo→batch en pp de KPI | `candidate_win_model_v7.md`, hallazgos §20.4-20.5 |
| `v7/e9_08_potencia_fable.py` | Potencia del diseño off-policy: MDE al 80 % vs efecto esperado por la cadena (pendiente del uplift predicho sobre la distancia), n de batches requerido | dist_R: potencia 32 %, ~1 000 batches; hallazgos §20.5 |
| `v7/e9_09_datos_artefacto.py` | Exporta `figs/datos_v7.js` (validación, θ, predicciones, techo, curva, bake-off, formas, parsimonia, anclajes, política, curvas, evidencia, potencia) para el artefacto | `figs/enfoque_v7.html`, https://claude.ai/artifact/L6GFrWjVcddg2tTa1AkL9F |
| `v7/auditoria_[A-D]_*.md`, `v7/test_antifuga_v7.py` | Auditoría de comité (Fable 5.1 × 4: pirometalurgia, estadística, operación, datos) del enfoque v7 y test anti-fuga sobre las 281 columnas | `dictamen_comite_v7.md`, hallazgos §21 |


### Iteración 10: `v8/` — prescriptor v8: fase 0 del dictamen del comité + mejoras refutables (2026-09-16 tarde/noche; diseño `v8/ITERACION_10_diseno.md`, hallazgos.md §22)

Datos: caché `cache/df_v8.pkl` (df v6 + columnas `m8_*`, `*_prev2`, `*_F6`; `v8/v8_lib.py`). Cinco experimentos de ola 1 en paralelo (Sonnet 5;
E10-01 y E10-06 relanzados como procesos tras la caída de los agentes por expiración del token) y ola 2 (E10-05) sobre `modelo_predictivo_v8.py`.

| Script | Qué hace | Dónde quedó consolidado |
|---|---|---|
| `v8/v8_lib.py` | Librería: target de FeO neto del Fe del dross (`m8_ln_feo_ret_dross{40,50,60}`), piso C*, formas del carbón (dosis, ln1p, cuadrática, heterogénea), leyes de t−2 (`*_prev2`), estado plan (`*_F6`, `Cx_sn_F6`), estados candidatos, `theta_dual` (θ libre y acotado), `oof_cronologico`, `walk_forward`, `r2_por_conjuntos` | `modelo_predictivo_v8.py` |
| `v8/e10_01_formas_carbon.py` | 8 formas del carbón sobre Sn y 5 sobre FeO: R² por conjuntos, θ libre/acotado, tercios, dosis-respuesta residual, PD del HGB por orden, óptimo interior de J en 12 estados | `Cx_sn_v6` se mantiene (única identificada); carbón→FeO no identificado; 0 % óptimos interiores; `e10_01_resultados.md` |
| `v8/e10_02_target_estado.py` | Targets de FeO (v6 vs neto del Fe del dross 40/50/60) y Sn (v6 vs C*) × estados k16/k15/k17: R² por conjuntos, θ dual, por orden, correlación entre targets, reversión de ruido, tercios | `m8_ln_feo_ret_dross50` + estado k15 (sin espesor); `e10_02_resultados.md` |
| `v8/e10_03_estado_ejecutable.py` | Estado k15 vs online (leyes de t−2) vs plan (cierre de F6 para todo R): R² global y por orden, θ dual, simulador de R0 con F5 | modo `plan` de `modelo_predictivo_v8` (`activar_modo_plan`); `e10_03_resultados.md` |
| `v8/e10_04a_anclajes.py`, `e10_04b_reanclaje.py` | Anclaje K/w con IC, EIV, legado (KPI_next), objetivo lineal en kg (λ), barrido de w; re-anclaje con el target de FeO corregido | `CONFIG_V8` (K 2.781, w 5.93 [3.96, 9.62], λ_kg 3.0); `e10_04a_resultados.md` |
| `v8/e10_05_validacion_v8.py` | Validación del sistema v8 (OOF aleatorio/cronológico/walk-forward/lockbox, θ dual, predicciones, curvas, óptimo interior), política cross-fitted con salvaguardas (0..4, lockbox), evidencia (magnitud de cambios, uplift con IC propagado, adherencia/dirección como monitoreo), evaluación cruzada anti-maldición, sensibilidad del objetivo (w, legado, lineal en kg), dosis-respuesta residual y residuo de la acción → KPI | `candidate_win_model_v8.md`, `win_analytics_v2.ipynb` |
| `v8/e10_06_fusion.py` | Fusión con β_F = 0: ΔT con/sin F1, θ dual, familia de batch (21 pruebas, q-BH), recorte con λ de lanza constante, envolvente por orden | ΔT sin F1; Fusión = receta; `e10_06_resultados.md` |
| `build_notebook_win_v2.py` | Genera `win_analytics_v2.ipynb` (nbformat) | El notebook mismo |

Lecciones operativas: un proceso sin `OMP_NUM_THREADS` sobre-suscribe los 20 núcleos y ralentiza ×50 al resto (matar con `Stop-Process`);
los subagentes cierran el turno al lanzar procesos en background (vigilar el PID con `until ! tasklist //FI "PID eq N"` y reanudar con SendMessage).

## Iteración 11 — `v9/` (asesor v9: valor del control anclado en el KPI por el canal de la acción, re-anclaje adaptativo)

Diseño `v9/ITERACION_11_diseno.md`; librería `v9/v9_lib.py` (residuos a − E[a|S] cross-fitted, exposiciones por batch, `prueba_anidada`); módulo
`modelo_predictivo_v9.py`; ficha `candidate_win_model_v9.md`; hallazgos §23.

| Script / memo | Qué hace | Resultado |
|---|---|---|
| `v9/e11_01_*` | Robustez del vínculo directo control → KPI (conjunta, tercios, estado de E[a\|S], placebos, RV, dosis-respuesta) | GN_R aceptado (−1.5 pp, q 0.008, RV 17 %); carbón sólo por canal escoria |
| `v9/e11_02_*` | Moderación por Sn disponible | Rechazada como rampa; patrón por tramos |
| `v9/e11_03_*` | Objetivo anclado por la cadena θ×residuo (log/kg) | Rechazado (signo inestable); efecto directo del carbón |
| `v9/e11_04_*` | KPI más preciso, anatomía del ruido | KPI principal se mantiene; KPI secundarios por canal |
| `v9/e11_05_*` | Fusión: exo2, carbón, subgrupos, mecanismo | Receta |
| `v9/e11_06_valor_termico.py`, `e11_06b/c/d` (Fable) | Specs con compuerta térmica, β DEV → lockbox, walk-forward, ventanas móviles | β estático falla en lockbox; re-anclaje W 60-120 prospectivo positivo |
| `v9/e11_07_*` | Validez de la prueba prospectiva (permutación por bloques, multiplicidad, placebos, canales, magnitud) | p 0.019-0.027 corregida; +1.45 pp entre terciles |
| `v9/e11_08_*` | Moderación térmica en el escalón | No identificada |
| `modelo_predictivo_v9.py` (`e11_09_*`), `v9/e11_09b_resumen_politica.py` | Replay prospectivo escalón a escalón + evidencia + resumen de política | pendiente +0.44 (p 0.023), Spearman 0.165 (p 0.008) |

## Iteración 12 — `v10/` (mejoras de v9, límite de información, asesor de planta v9.1)

Diseño `v10/ITERACION_12_diseno.md`; módulo `asesor_planta_v9_1.py`; ficha `candidate_win_model_v9_1.md`; hallazgos §24.

| Script / memo | Qué hace | Resultado |
|---|---|---|
| `v10/e12_01_*` | Duración de R3/F6 como palanca | No adoptada |
| `v10/e12_02_*` | Valoración por canal de pérdida | ≡ KPI directo; no adoptada |
| `v10/e12_03_*` | β variable por filtro de Kalman | Peor que W=100 |
| `v10/e12_04_*` (`e12_04_lib.py`: estado S3') | Estado ejecutable al inicio del escalón | **Adoptado** (87 % de la evidencia) |
| `v10/e12_05_*` | Lanza, O₂, enriquecimiento, aire, caudal, presión | Rechazadas |
| `v10/e12_06_covariables.py` | Covariables pre-tratamiento | Sin mejora |
| `v10/e12_07_potencia_piloto.py` | Potencia del piloto con ruido real | ±1 sd: 80 batches → 0.80 |
| `v10/e12_08_objetivo_fisico.py` | Objetivo ΔSn_kg − λ·ΔFeO_kg con λ fijo | Regímenes opuestos; rechazado |
| `v10/e12_09_lazo_cerrado.py` | Simulación de lazo cerrado de estrategias | +0.3..+0.5 pp/batch; 0 bajo el nulo |
| `v10/e12_10_robusto.py` | Huber / winsor / rangos | Sin mejora; evidencia estable (p 0.004-0.011) |

## Iteración 13 — `v11/` (calibración, auditoría adversarial, decisión)

| Script / memo | Qué hace | Resultado |
|---|---|---|
| `v11/e13_01_contraccion_js.py` | Contracción James-Stein de β (a priori, sin hiperparámetros) | Calibración 0.53 → 0.70; adoptada |
| `v11/e13_02_replay_planta_js.py` | Replay estricto del asesor de planta con/sin JS | t 1.4-1.7; Spearman 0.19 (p 0.002) |
| `v11/e13_03_*` (Sonnet, auditor) | Nulas alternativas, familia de contracciones, 81 especificaciones, placebos, jackknife | Calibración sí; p < 0.01 no (0.010-0.014) |
| `v11/e13_04_sesgo_reciente.py` | Corrección de deriva de receta en Ê[a\|S] (k 20 / 40) | k 20: t 1.89, pendiente 0.74; k 40 sin mejora |
| `v11/e13_05_decision.py` | Ganancia calibrada y P(> 0) por bootstrap | +0.57..+0.80 pp/batch; P(> 0) 0.96-0.98 |
| `v12/e14_01_canales_estricto.py` | Evidencia prospectiva estricta por canal (GN → dross, C → Sn en escoria) con exposiciones entrenadas sólo con el pasado | Negativa: GN → dross nulo; KPI compuesto p 0.019 |


## Iteración 15 — `v13/` (receta de planta → asesor v10)

| Script / memo | Qué hace | Resultado |
|---|---|---|
| `v13/e15_01_receta.py` | Exposición exacta ejecutado − receta (hoja "Alimentación"), OLS conjunta y prueba prospectiva | GN −1.58 pp (t −3.9), dross t +4.1; prospectiva t 2.6-2.7, p 0.001-0.006 |
| `v13/e15_02_*` (Sonnet, auditor) | Fuga de estandarización, ex-ante, multiplicidad, 162 especificaciones, placebos, jackknife, régimen | p corregida 0.002-0.012; 93 % con GN + C; robusto a jackknife; débil en fin de campaña |
| `asesor_receta_v10.py` (`v13/e15_03_*`) | Replay estricto del asesor v10 + ganancia calibrada + política | +0.58 pp/batch [0.13, 1.13]; Spearman 0.20 (p 0.003) |
| v14/e16_01_receta_fusion.py | Desvíos respecto de la receta de Fusión (carbón, carga) | Carbón F → polvo t 2.6; neto KPI nulo; Fusión = receta |
| v15/e17_01_regimen_termico.py | Moderación térmica con exposición exacta y holdout DEV → lockbox | Canales con signo de teoría (polvo +, dross −), KPI nulo; holdout estático falla (−0.6); persistencia controlada t −3.5 |


## Iteración 18 — `v16/` (alternativas bajo restricciones: una campaña, sin piloto)

| Script / memo | Qué hace | Resultado |
|---|---|---|
| `v16/e18_01_*` | Revisiones de receta como instrumento / estudio de eventos | Instrumento débil; placebo de receta futura falla; no identifica |
| `v16/e18_02_*`, `e18_05_balance_por_orden.py` | Efectos fijos de batch con desvío exacto; selectividad marginal del GN por orden | GN preciso y estable entre regímenes; R3 0.49 kg Sn/kg FeO |
| `v16/e18_04_lambda_fisico.py`, `e18_06_estado_final.py` | Conversión FeO → dross; efecto sobre el estado final de la Reducción | +0.43 kg Sn a dross por kg FeO; Sn final no cambia; GN tardío +102 kg FeO (t 3.8) |
| `v16/e18_03_*`, `e18_07_regla_final.py` | Alcance ex-ante, regla "GN ≤ receta", expediente de seguridad | Regla: p 0.003, +0.66 pp/batch [0.10, 1.19]; sin señales de riesgo |

## Iteración 19 — `v17/` (asesor físico v11)

| Script | Qué hace | Resultado |
|---|---|---|
| `v17/e19_01_modelo_fisico_prospectivo.py` | θ intra-batch por orden estimado sólo con el pasado → kg de FeO extra reducido por batch; contraste con FeO reducido (ensayo), dross, polvo, KPI | pendiente 0.76 [0.45, 1.07], t 4.8, p 0.0002; dross t 3.0; KPI normal t −3.7 |
| `asesor_fisico_v11.py` | θ por orden con IC, recomendación por escalón con kg de FeO evitado, valor de la regla | k=0: +0.41 pp [0.19, 0.63]; 281/362 batches en régimen |
| v17/e19_02_kpi_con_conmutador.py | Asesor v11 con conmutador de régimen vs KPI de toda la campaña | KPI t −1.42 (p 0.15) con conmutador; exceso modelado t −2.46; dross t 2.93 |
| v18/e20_01_kpi_cierre.py | Cierre del balance de Sn como covariable de error de medida del KPI | −15 % de varianza; sin ganancia de potencia; prospectiva p 0.002 se mantiene |

## Cierre — `v19/` (verificación final)

| Script | Qué hace | Resultado |
|---|---|---|
| `v19/e21_01_verificacion_final.py` | Adherencia a 'GN ≤ receta' vs KPI: OLS, cuartiles, prospectiva en 6 ventanas con corrección por multiplicidad, jackknife, placebos | DEV +1.00 pp (t 3.8); p corregida 0.0035; jackknife 97-100 % |
| `v19/e21_02_persistencia.py` | Separa la parte propia del batch de la persistencia | innovación +0.91 (t 3.1); EF por bloques +1.06 (t 3.6) |

| v20/e22_01_fuego_y_potencial_o2.py | Potencia de fuego y potencial de oxígeno de lanza como desvío respecto de la receta (batch, intra-batch, prospectivo) | Fuego → KPI −0.60 (t −3.5), dross t 3.4-4.0; potencial de O₂ sin valor neto (compensa FeO vs Sn final); GN sigue siendo el mejor mando |
