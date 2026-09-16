# Iteración 8 (2026-09-16) — inferencia de masa de escoria v6 y prescriptor v6

Diseño (Fable 5.1). Punto de partida: prescriptor v5 (`candidate_win_model.md`, hallazgos §18) y la **auditoría del trazador CaO**
(2026-09-15/16, scripts `audit_cao*.py` / `mejora_masa*.py` en el scratchpad; resumen en el docstring de `experimentos/v6/masa_v6.py`).

## Diagnóstico que motiva la iteración

| Hecho verificado | Consecuencia |
|---|---|
| `masa = cum cal / %CaO` supone tolva 6 = CaO puro y única fuente de CaO. Regresión con el balance global: CaO en escoria final ≈ 0.26-0.50 × cal + 0.055-0.079 × carga; %CaO en Fusión 17-20 % frente a 9 % si la cal fuera pura | El trazador acierta por compensación (la cal se dosifica al ~10 % de la carga en todos los escalones). Factor de escala entre batches correlacionado con cal/carga (ρ −0.35): sesgo ±7 % en los STATE en kg; "efecto de la cal" de exp 18 / E7-02 es artefacto |
| Ruido del ensayo de %CaO ≈ 2.8 % relativo vs señal de masa por escalón en Reducción −1.3 %: 31 % de los escalones sin carga "ganan masa"; el cambio de %CaO comparte r 0.12 con Al₂O₃ y 0.24 con SiO₂ | `v5_ln_feo_ret` es mitad %FeO y mitad ruido del trazador. Su R² OOF 0.15 cae a 0.05 sin `%CaO_prev`/B2/IRF_prev en el estado: el PLM aprendía la **reversión a la media del ruido de CaO** (corr término CaO vs %CaO_prev 0.68) |
| Cierre físico en Reducción (sin %CaO): el resto no reducible `1 − 1.27·s − f` se conserva | Escalones que ganan masa 31 % → 10 %, sd 0.040 → 0.025, R² OOF ln_feo_ret 0.15 → 0.26-0.29 y **no** depende de %CaO_prev (0.25-0.27); ln_sn_dep igual (~0.70) |
| Talón: F0 parece 1/3 de escoria final previa por composición, pero F1 correlaciona MÁS con el final previo que F0 y el rezago 2 casi igual | Autocorrelación temporal, no talón físico; corrección sin impacto (r 0.97 con H = 770 kg) |

## Estimador v6 (implementado en `experimentos/v6/masa_v6.py`, `agregar_masa_v6`)

- Entradas en base húmeda (kgh) con leyes desconocidas → coeficientes efectivos poblacionales calibrados en DEV contra la masa final
  del balance global (`fe.construir_resumen_batch_balance_global`, usa metal/dross/polvo). Constantes de campaña, no del batch.
- Fusión: trazador de matriz G = CaO+SiO₂+Al₂O₃+MgO, `M_t = (pG·Σcal + gG·Σcarga + aC·Σcarbón)/G_t` (pG 0.70, gG 0.30, aC 0).
- Reducción: cierre físico `M_t·(1 − K·s_t − f_t) = M_{t−1}·(1 − K·s_{t−1} − f_{t−1})·e^{−δ} + resto_in_t`, K = M_SnO₂/M_Sn = 1.270.
- Estado `_prev` causal; targets `m6_ln_sn_dep`, `m6_ln_feo_ret`; palancas `Cx_sn_v6`, `Cx_av_v6`, `m6_dosis_C_sn`, `m6_exceso_C_pos`.
- Diagnóstico offline (LEAKAGE): `m6_k_anclaje = M_balance / M_fin` por batch.

Batería común: `masa_v6.evaluar_masa(df6, col, bal, dev, lockbox)` (física, escala vs balance, cierre de Sn contra M+D+P, suavidad en
Fusión, R² OOF de los targets con y sin %CaO_prev).

## Experimentos paralelos (Sonnet) y síntesis (Fable)

- **E8-01 — Variantes, calibración y robustez del estimador** (`e8_01_masa_variantes.py`): grilla fusion × reduccion × (δ, w_obs, aC),
  calibración por tercios cronológicos y sensibilidad ±30 % de pG/gG, propagación de error a lo largo de Reducción, comportamiento en
  primer escalón / ensayos faltantes / batch sin cal, diagnóstico de `m6_k_anclaje` (con qué correlaciona), bandera QC de residuo de
  masa. Salidas `e8_01_*.csv`, `e8_01_resultados.md` con recomendación de parámetros.
- **E8-02 — Balance de energía con la masa v6** (`e8_02_energia_v6.py`): recomputar las features térmicas de `bal_features.
  agregar_balance_energia` con `m6_masa_kg_prev`/`m6_sn_inv_kg_prev` (capacidad térmica, ΔT teórico, margen), modelos de ΔT por fase
  (PLM v5: estado `_S_DT` + palancas) v3 vs v6, features de energía específica (MJ/t escoria, GN Nm³/t), calor de reacción aparente vs
  trazador. Salidas `e8_02_*.csv`, `e8_02_resultados.md`.
- **E8-03 — Anclaje del objetivo y PLM con targets v6** (`e8_03_anclaje_v6.py` + `v6_lib.py`): `v6_lib.cargar_df()` (v5_lib + m6),
  `construir_batch_v6` (agregados `R_sum_m6_*`, `F_sum_m6_*`), OLS HC3 del KPI refinado sobre Σm6_ln_sn_dep y Σm6_ln_feo_ret con
  controles + tendencia (DEV; w = b_FeO/b_Sn con IC bootstrap), batería Spearman DEV/lockbox, Fusión (conservación de Sn con control de
  heel), PLM (`ModeloPLMSignos`) de los targets v6 con estado FULL/CURADO donde los inventarios v3 se sustituyen por los v6 (θ con IC,
  comparación con v5). Salidas `e8_03_*.csv`, `e8_03_resultados.md`.
- **E8-04 — Síntesis (Fable)**: `modelo_predictivo_v6.py` (extiende v5), validación/θ/predicciones/curvas, política cross-fitted y
  evidencia off-policy, figuras (SHAP, PDP, real vs pred, evolutivo), artefacto, `candidate_win_model.md`, hallazgos §19.

## Convenciones

- Python `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4`; sin joblib/loky; scripts largos en background con
  log `experimentos/v6/e8_0N_log.txt`; en Git Bash escribir scripts a archivo (no heredocs largos).
- Datos: `masa_v6.cargar_df_base()` (cache v3) → `mp4.agregar_columnas_v4` → `masa_v6.agregar_masa_v6(df, params, bal)`; batch
  outputs: `sn_en_metal_crudo_batch_t`, `sn_en_dross_fe_batch_t`, `sn_en_polvo_fundicion_batch_t` (t), `recuperacion_refinada_pct`.
- Validación: OOF GroupKFold(5) por Batch sobre DEV = criterio; lockbox (`mp.split_dev_lockbox`, últimos 17.5 %) sólo reporte.
- Anti-fuga: features de entrada sólo con `masa_v6.es_segura_v6(nombre)`; `m6_masa_kg`, `m6_*_inv_kg`, `m6_resto_frac`, `m6_G_pct`
  contemporáneos y `m6_masa_kg_anclada` son LEAKAGE.
- No se modifican `feature_engineering.py`, los módulos de la raíz, `masa_v6.py` (salvo bugs, avisando) ni `hallazgos.md`.
