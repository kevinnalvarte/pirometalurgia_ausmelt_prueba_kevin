# Iteración 7 (2026-09-14/15) — prescriptor v5: objetivo por escalón anclado en los targets ganadores y en el KPI refinado

Diseño (Fable 5.1). Punto de partida: v4 (`candidate_win_model.md`), targets ganadores de la iteración 5
(`experimentos/search_targets/README.md`), KPI refinado y features térmicas de la iteración 6 (`experimentos/balances/README.md`,
`hallazgos.md` §17). Librería compartida: `experimentos/v5/v5_lib.py`.

## Estado de partida y brechas frente a las condiciones de aceptación

| Condición | Estado (v4 + iteraciones 5-6) | Brecha que ataca esta iteración |
|---|---|---|
| Target por escalón con agregación a batch y relación demostrada con el rendimiento | v4 optimiza `Sn − 0.6·FeO` en kg, cuyo agregado **no** correlaciona con el KPI (ρ 0.035). Iteración 5: SDI (ρ 0.32 DEV) y ln IRF en Reducción (0.30/0.38 lockbox); iteración 6: KPI refinado con Sn en escoria | **Sustituir el objetivo de Reducción por las componentes logarítmicas del SDI** (agotamiento de Sn + retención de FeO), con el peso w **anclado en el KPI refinado** (no elegido a mano) y verificar la variante ponderada por tiempo restante (lo que premia el ln IRF twmean) |
| Efecto de las CONTROL robusto y significativo | SDI: GN −0.15*, carbón +0.08 (IC cruza 0); componentes: carbón +0.09* y Cx_sn +0.17* sobre Sn, ≈0 sobre FeO; O₂ +0.19* sobre ln IRF | Parametrización de las palancas con estructura cinética en escala log: **dosis específica de carbón** (kg C / kg Sn disponible) para el agotamiento; **carbón sobrante** sobre la demanda del Sn (y carbón × avance) para la retención de FeO; exceso de O₂ para la lanza |
| No asumir monotonía; óptimos interiores por teoría | Palancas lineales en el target; el óptimo interior sólo por soporte/ventana térmica | Test cross-fitted de términos cuadráticos y de θ por tramo de avance; PD/ALE del HGB sobre las componentes; tabla teórica de formas (abajo) |
| Parsimonia y estabilidad | Estado de 37 columnas en Reducción; θ de carbón/GN en Fusión cambian de signo entre períodos | Curva de parsimonia del estado para las componentes; θ por tercios cronológicos y lockbox; reentrenamiento progresivo |
| Fusión: target con palancas identificadas | IRF de Fusión no responde a palancas (θ ≈ 0); objetivo v4 anclado en exp 19 (proxy) | Re-anclar β_C, β_T con el **KPI refinado**; probar GN/exceso O₂ de Fusión sobre matriz y KPI; per-escalón: Δln IRF y agotamiento de Sn en Fusión (ρ −0.14 con KPI refinado, p 0.01) |
| Evidencia de que seguir la política sube el KPI | v4: sólo lockbox (dist_R −0.84 pp/sd), DEV nula | Política v5 cross-fitted para 362 batches; adherencia vs KPI refinado (DEV/lockbox/total/batch siguiente), placebo, emparejamiento por estado, y cadena θ→agregado→KPI con bootstrap |

## Hechos nuevos verificados al construir la librería (`v5_lib.py __main__`)

- Reproduce iteraciones 5-6 con el KPI refinado: SDI ρ 0.317 DEV / 0.087 lockbox; ln IRF twmean 0.297 / 0.378; IRF fin F 0.207 / 0.341;
  ln IRF al cierre de F0 (heel) 0.326 / 0.579.
- Componentes por separado (DEV): Σln_feo_ret ρ +0.286 (OLS +1.27 pp/sd, p < 0.001); Σln_sn_dep ρ −0.001 (OLS +0.39, n.s.) pero
  **+0.354 en lockbox (p 0.004)**: en el régimen de fin de campaña la señal está en agotar el Sn.
- Fusión: Σln_sn_dep (extraer Sn en Fusión) ρ −0.115 DEV (p 0.05), −0.165 lockbox, −0.137 total (p 0.01): conservar el Sn en
  Fusión tiene señal débil pero de signo teórico (λ_F) con el KPI refinado. Δln IRF acumulado en Fusión ρ −0.32: artefacto de
  regresión a la media respecto del IRF inicial (hay que controlar F0).

## Experimentos paralelos (Sonnet 5) y síntesis (Fable 5.1)

- **E7-01 — Reducción: componentes log, parametrización de palancas y anclaje de w** (`e7_01_reduccion_componentes.py`).
  PLM (`mp4.ModeloPLM`) para `v5_ln_sn_dep`, `v5_ln_feo_ret`, `v5_sdi_w10` (+ referencia `sn_extraido_est_kg`,
  `feo_extraido_est_kg`) con estado FULL vs CURADO y cuatro parametrizaciones de palancas (primitivas; cinética v4; dosis/sobrante;
  log-dosis). Métricas OOF/lockbox, θ con IC, concordancia de signo, ancho de IC. Anclaje de w: OLS HC3 del KPI refinado sobre
  Σln_sn_dep y Σln_feo_ret con controles (w = b_FeO/b_Sn, IC bootstrap), barrido de w en DEV (lockbox sólo reporte) y variante
  ponderada por tiempo restante.
- **E7-02 — Fusión: qué objetivo es defendible con el KPI refinado** (`e7_02_fusion.py`). Regresiones de batch (estilo exp 19)
  del KPI refinado, f_dross, f_polvo sobre decisiones de Fusión (C/Sn, GN por t de carga, exceso O₂, T media, lanza, cal/carga), con
  términos cuadráticos para C/Sn y T; PLM por escalón de `v5_d_lnirf`, `v5_ln_sn_dep`, `sn_extraido_est_kg`, `d_temperatura`;
  ¿el cambio de IRF durante la Fusión aporta una vez controlado el IRF de F0? Decisión sobre los términos de J_F.
- **E7-03 — Formas funcionales y óptimos interiores** (`e7_03_formas.py`). Términos cuadráticos residualizados en el PLM, θ por
  tramo de avance (selectividad), PD/ALE del HGB estado+palancas sobre las componentes; contraste con la tabla teórica; curva de J
  (SDI) vs carbón/GN por nivel de avance.
- **E7-04 — Parsimonia, estabilidad temporal y reentrenamiento** (`e7_04_estabilidad.py`). Curva de parsimonia del estado
  (eliminación hacia atrás por importancia de permutación), θ por tercios cronológicos + lockbox, walk-forward, reentrenamiento
  progresivo, deriva de features.
- **E7-05 — Síntesis y v5 (Fable)**: `modelo_predictivo_v5.py` (extiende v4), recomendaciones cross-fitted (`e7_05_validacion_v5.py`
  por folds), evidencia off-policy con KPI refinado (`e7_06_evidencia_v5.py`), `candidate_win_model.md`, `hallazgos.md` §18,
  `win_analytics_v1.ipynb`.

## Tabla teórica de formas esperadas (para E7-03; fuentes: Guia_Ausmelt_Sn §3.3-3.4, 5.2-5.5, 6.1, 7.2-7.4, 11.2-11.3; consideraciones §7.7, 8, 13, 16, 29)

| Palanca / estado | Sobre `v5_ln_sn_dep` (agotar Sn) | Sobre `v5_ln_feo_ret` (retener FeO) | Forma esperada | Origen del óptimo interior |
|---|---|---|---|---|
| carbón (dosis C/Sn disponible) | + (SnO₂ + 2C; 1er orden en C mientras hay SnOₓ) | ≈0 con SnOₓ abundante; − creciente con el avance (selectividad 0.11→1.5 al pasar avance 0.96, exp 15) | Sn: cóncava/saturante; FeO: − con pendiente que crece con el avance | trade-off Sn vs FeO por estado (avance), no monotonía impuesta |
| carbón sobrante (C − 0.2024·Sn disp) | ≈0 | − (reductor disponible para FeO + C) | lineal − | idem |
| GN | + (calor + CO/H₂ reductor) | − (reductor no selectivo; exp 19: GN total de Reducción sube dross) | monótonas; el óptimo de J es interior por la ventana térmica (fume a T alta; frío = viscosidad) | ventana térmica + w |
| exceso de O₂ (O₂ + 0.21·aire − 2·GN) | − (atmósfera oxidante frena la reducción / reoxida Sn) | + (retiene FeO; lanza oxidante) | monótonas de signo opuesto | trade-off por w |
| aire | ≈0 / − (enfría, diluye) | + débil (aporta O₂) | monótona débil | costo térmico |
| T previa (estado) | + (cinética, viscosidad) | − (más reducción de FeO a T alta) | + con meseta | ventana (volatilización, refractario) |
| basicidad B2 previa (estado) | no monótona (óptimo ~1.4) | ≈0 | valle | viscosidad/liquidus |
| posición de lanza previa (estado) | no monótona (valle 3300-4200 mm) | no monótona | valle | sin confirmar con planta: no es palanca |
| avance previo (estado) | + en escala log (queda poco Sn, cada kg pesa más) | − (menos SnOₓ que compita) | monótonas | — |

## Convenciones comunes

- Python: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4`; sin joblib/multiprocessing (loky se cuelga en
  Windows); scripts largos en background con log `experimentos/v5/e7_0N_log.txt`.
- Datos: `v5_lib.cargar_df()` (cache v3 + v4 + b6 + v5, 15 s) y `v5_lib.construir_batch_v5(df)`; escalones modelables por fase con
  `v5_lib.base_fase(df, fase)`.
- Validación: OOF GroupKFold(5) por Batch sobre DEV = criterio; lockbox sólo reporte; ningún criterio de selección usa el lockbox.
- Anti-fuga: sólo columnas con `v5_lib.es_segura(nombre) == True` como features; las `b6_*` DERIVED contienen la acción (GN) y **no**
  pueden entrar en g(S) de un PLM (sólo como palanca lineal o en el modelo de ΔT).
- Salidas: `experimentos/v5/e7_0N_*.csv`, memo `e7_0N_resultados.md` (tablas + veredicto + límites), figuras en `experimentos/v5/figs/`.
- No se modifican los módulos de la raíz ni `hallazgos.md` (lo hace la síntesis).
