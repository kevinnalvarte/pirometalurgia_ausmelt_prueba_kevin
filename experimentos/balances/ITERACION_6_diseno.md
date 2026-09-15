# Iteración 6 (2026-09-14, noche) — balances de masa y energía inspirados en `func-teorethical-model`

Diseño (Fable 5.1). Punto de partida: v4 (`candidate_win_model.md`), targets ganadores de la iteración 5 (`search_targets/README.md`:
SDI en Reducción, IRF/lnIRF en Fusión y Reducción) y el análisis del modelo teórico Rust de los metalurgistas
(artefacto "Modelo teórico del batch Ausmelt", https://claude.ai/artifact/1urzPypbqPJ7SRNN1Mo6cH).

## Qué aporta el modelo teórico que la analítica todavía no usa

| Idea del modelo teórico (Rust) | Idea del equipo metalúrgico | Lo que existe en v3-v5 | Brecha que ataca esta iteración |
|---|---|---|---|
| La escoria es un balance por óxidos: SiO₂, Al₂O₃, CaO, MgO son inertes; sólo SnO y FeO salen (reducción) | "que la ley suba o baje no significa que la masa suba o baje" | trazador CaO (un solo inerte, cal de tolva 6) | **cierre multi-inerte** en Reducción (4 trazadores → menos ruido en Sn/FeO en kg y en el SDI); residuo de masa no explicado (splash / arrastre) |
| Carbón = demanda estequiométrica (SnO₂→SnO→Sn, Fe₂O₃→FeO, FeO→Fe) + carbón quemado por el O₂ libre; O₂ libre = exceso sobre el estequiométrico del gas natural (negativo en Reducción: CO reductor) | reacciones redox: conservar masa | `exceso_o2_combustion_pct`, `relacion_C_Sn_carga`, `Cx_sn_v4` | **balance reductor/oxidante** en kg C-equivalente: carbón + CO de lanza + Fe metálico del dross − demanda del Sn disponible − demanda del mineral de Fe = *reductor neto* (capacidad de sobre-reducir FeO) |
| Balance térmico: combustión (con corrección sub-estequiométrica), sensible de carga, gases, pérdidas fijas, reacciones endo/exotérmicas; la temperatura de baño cierra el balance | "incluir la variación de temperatura y/o el balance de energía estequiométrico" | ΔT como target (R² 0.47 R / 0.06 lockbox F) sin estructura física | **balance de energía por escalón** (features a tiempo de decisión) y **calor de reacción aparente** (estimador de la extracción independiente del ensayo de escoria) |
| Balance global del batch: carga por corriente → metal, dross, humos, escoria, gases | rendimiento refinado con Sn en escoria; pérdidas por volatilización, splash, finos | KPI proxy = M/(M+D+P); trazador subestima la masa absoluta (CaO de ganga, talón) | **balance global por batch**: masa final de escoria, factor de escala del trazador, Sn en escoria final, KPI refinado; talón de escoria y ganga por tolva por regresión multi-óxido |
| FeO→Fe (dist_feo_fe) es la incógnita que cuadra el Fe a metal/dross | "el Fe en escoria da pistas de sobre-reducción respecto al Sn oxidado" | SDI, `feo_extraido_est_kg`, selectividad por avance (exp 15) | FeO en kg con menos ruido (multi-trazador) + *reductor neto* como palanca estructurada del FeO |

## Librería compartida: `experimentos/balances/bal_features.py`

`bal.cargar_df()` devuelve el `df_v3` cacheado con el bloque v6 (todas las columnas nuevas llevan prefijo `b6_`); `bal.CLASIFICACION_V6`
etiqueta cada columna (STATE / DERIVED-ACTION / DERIVED (A_t x S_prev) / TARGET / LEAKAGE / DIAG). `bal.construir_batch_balance(df)`
devuelve una fila por batch con el balance global. Constantes y supuestos (poder calorífico, carbono fijo, Fe del mineral y del dross,
pérdidas) están en `bal.SUPUESTOS` y son parámetros: cada experimento hace sensibilidad sobre los que usa.

## Experimentos (paralelos, Sonnet 5.1) y síntesis (Fable 5.1)

- **B-01 — Cierre multi-trazador y targets en masa (Reducción)** (`b01_multitrazador.py`): consistencia entre los 4 inertes
  (dispersión de ln-ratios por escalón, ¿hay un inerte sesgado?), ruido del trazador único vs multi (Monte Carlo del ensayo), targets
  `b6_sn_extraido_multi_kg`, `b6_feo_extraido_multi_kg`, `b6_sdi_multi` vs los de trazador CaO: correlación con KPI de batch
  (batería ST-01: Spearman con IC, BH, OLS con controles, tercios, lockbox), predictibilidad OOF/lockbox (HGB con STATE+CONTEXT v3;
  PLM v4 con las palancas), techo de ruido. Residuo de masa no explicado `b6_residuo_masa_kg` (¿correlaciona con tiro, gas total,
  polvo del batch?).
- **B-02 — Balance reductor/oxidante como feature estructurada** (`b02_balance_reductor.py`): `b6_reductor_neto_kg_ceq`,
  `b6_ratio_reductor_demanda`, `b6_ceq_lanza_kg`, `b6_capacidad_sobre_reduccion_feo_kg` y acumulados `_prev` como (i) features en
  g(S) y (ii) palanca lineal del PLM (sustituyendo/añadiendo a `Cx_av_v4`, `tasa_feed_Carbon_kg_min`) para targets `feo_extraido`,
  `sn_extraido`, SDI y `b6_sdi_multi`; ¿mejora R² OOF/lockbox? ¿θ del carbón sale del IC que cruza 0? Cadena de signos vs exp 19.
  Utilización del reductor por batch (`b6_utilizacion_reductor`) vs el 48 % "perdido al tiro" del modelo teórico. Sensibilidad a
  `CARBONO_FIJO`, `FE_MINERAL`, `FE_MET_DROSS`.
- **B-03 — Balance de energía por escalón** (`b03_balance_energia.py`): features `b6_q_*` y `b6_dT_teorico_sin_reaccion` en el PLM de
  ΔT (R² OOF/lockbox vs v4 0.474/0.427 R y 0.478/0.059 F); calor de reacción aparente `b6_q_reaccion_aparente_MJ` vs calor de
  reacción del trazador `b6_q_reaccion_trazador_MJ` (correlación por fase; ¿el balance de energía "ve" la extracción?); estimador
  combinado trazador+energía; sensibilidad a PCI, pérdidas, Cp.
- **B-04 — Balance global de masa por batch y KPI refinado** (`b04_balance_global.py`): `construir_batch_balance` → masa final de
  escoria por balance vs trazador (factor de escala y su dispersión), Sn en escoria final (kg y % del Sn cargado), KPI refinado
  `recuperacion_refinada_pct` = M/(M+D+P+Sn_esc) y `sn_perdido_escoria_frac`; correlación de los targets ganadores (SDI, lnIRF_R, IRF F,
  −Sn_ext F) con el KPI refinado (ST-01) y techo de predictibilidad (ST-15). Parte 2: modelo multi-óxido de la escoria
  (talón + ganga por tolva por mínimos cuadrados no negativos sobre los 362 batches, validado por predicción de leyes en hold-out).
- **B-05 — Síntesis (Fable)**: consolidar, hipótesis de ronda 2, decidir qué entra a `feature_engineering.py` (bloque v6) y a los
  targets del recomendador; `hallazgos.md` §17, README de experimentos, `candidate_win_model.md` (addendum), memoria.

## Criterios de aceptación (mismos que iteraciones 4-5)

Sentido pirometalúrgico explícito (fórmula, unidades, supuesto declarado); sin fuga (clasificación de cada columna); OOF GroupKFold por
batch como criterio, lockbox sólo reporte; una feature/target nueva se adopta sólo si mejora al mejor anterior en OOF y no empeora en
lockbox, o si aporta una señal física nueva (independiente del ensayo de escoria) con correlación demostrable.
