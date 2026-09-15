# Iteración 4 (2026-09-14) — diseño de experimentos hacia el modelo prescriptor ganador

Diseño (Fable 5.1) a partir del estado v3 (`hallazgos.md` §14, `modelo_predictivo_v3.py`). Ejecución paralela
por subagentes (Sonnet 5), revisión y decisión centralizada.

## Estado de partida (v3) y brechas frente a las condiciones de aceptación

| Condición | Estado v3 | Brecha |
|---|---|---|
| Features/target con sentido pirometalúrgico, parsimonia | targets en kg vía trazador CaO; sets curados 10-13 features | Fusión: R² lo da casi todo `feed_Sn_kgf`; set de 13 podría ser más parsimonioso |
| STATE/CONTROL separados; recomendación óptima por escalón | roles v3 + optimizador con soporte duro | ok, pero el efecto del carbón en Fusión NO se identifica separado de la carga (DML conjunto) |
| Efecto de las CONTROL robusto y significativo | DML: Reducción carbón (borde), GN (+Sn, +FeO), O₂; Fusión cal (−), GN (+) | carbón en Fusión no significativo; FeO en kg R² 0.27 (penalización ruidosa) |
| No asumir monotonía; óptimos interiores por teoría | monotonía sólo en carbón; lanza excluida | falta auditoría sistemática forma aprendida vs. teoría (PD/ALE) por palanca y estado |
| Sin fuga; OOF/lockbox | GroupKFold + lockbox 17.5% | deriva de campaña en lockbox (Fusión) sin tratamiento |
| Recomendaciones ⇒ mayor rendimiento proxy batch, defendible | uplift ESTIMADO por el modelo en 8 batches | **no demostrado con datos**: falta evidencia observacional off-policy |

## Experimentos de esta iteración (paralelos)

- **13 — Evidencia off-policy** (`13_evidencia_politica_batch.py`): recomendaciones cross-fitted (el sistema que recomienda
  a un batch nunca lo vio) para los 362 batches; adherencia histórica a la política ⇒ regresiones sobre
  `rendimiento_proxy_batch` con controles de contexto, dosis-respuesta por quintiles, placebo con política aleatoria,
  permutación, dev vs lockbox.
- **14 — Formas de modelo para las palancas** (`14_formas_modelo_palancas.py`): HGB v3 vs HGB con monotonía teórica
  extendida vs modelo parcialmente lineal (estado no lineal + palancas lineales con término cuadrático en cal) vs
  cinético log-lineal (k aparente en masa). Misma escala kg, OOF/lockbox; consistencia de signos de las palancas.
- **15 — Señal de sobre-reducción** (`15_senal_sobre_reduccion_feo.py`): techo de ruido del trazador, targets
  alternativos de FeO (fracción, k aparente, suma por fase), trazador suavizado, pérdida robusta; re-estimación de λ_Fe.
- **16 — Formas aprendidas vs teoría** (`16_formas_pd_teoria.py`): PD/ALE/SHAP-dependence de cada feature de los modelos
  v3, clasificación monótona/óptimo interior/plana y contraste con la tabla teórica de abajo.
- **17 — Estabilidad temporal** (`17_estabilidad_temporal.py`): walk-forward por bloques cronológicos, efectos DML por
  mitades de campaña, reentrenamiento por ventana vs. historia completa sobre el lockbox.

## Tabla teórica de formas esperadas (base para 16 y para las restricciones del modelo ganador)

Fuente: `Guia_Ausmelt_Sn.md` (§3.3 ventana selectiva Sn/Fe, §3.4 temperatura y selectividad, §5.3 basicidad,
§6.1 lanza, §11.2 compromisos) y `consideracioness_pirometalurgia_ausmelt.md` (§7.7, §8, §13, §16, §29).

| Variable | Rol | Efecto sobre Sn extraído (kg) | Efecto sobre FeO extraído (kg, Reducción) | Forma esperada |
|---|---|---|---|---|
| `tasa_feed_Carbon_kg_min` | CONTROL | + (SnO₂ + 2C → Sn) | + pero débil mientras hay SnOₓ; crece al agotarse el Sn | monótona + (saturante), interacción con inventario de Sn |
| `exceso_o2_combustion_pct` | CONTROL (derivada de GN/O₂/aire) | − en Reducción (atmósfera oxidante frena la reducción); ≈0 en Fusión | − (retiene FeO) | monótona −; el ÓPTIMO del objetivo es interior (selectividad) |
| `tasa_gn_nm3_min` | CONTROL | + (calor + gas reductor) | + (menos selectivo que el carbón) | monótona + con meseta térmica; óptimo interior en J |
| `tasa_feed_CaO_kg_min` / `relacion_CaO_carga` | CONTROL (Fusión) | − (dilución de la escoria, menor actividad relativa, menor T) | n/a | monótona − dentro del rango; el ÓPTIMO de basicidad es interior (viscosidad/liquidus) |
| `duracion_plan_min` | CONTROL | + con rendimientos decrecientes (cinética de 1er orden) | + | cóncava; óptimo interior en J por costo de tiempo |
| `basicidad_B2_prev` | STATE | Fusión: −/no monótona; Reducción: ≈0 | ≈0 | óptimo interior (~1.4 en Fusión) |
| `temperatura_horno_celsius_prev` | STATE | + (cinética, viscosidad) | + | monótona + dentro de ventana; óptimo interior en J (volatilización, refractario) |
| `posicion_vertical_lanza_mm_prev` | STATE (no palanca) | no monótona | no monótona | valle ~3300-4200 mm (Pasos 46-47) |
| `sn_inventario_escoria_est_kg_prev`, `ley_sn_escoria_pct_prev` | STATE | + (fuerza impulsora) | − (compite con la reducción de FeO) | monótona + |
| `ley_feo_escoria_pct_prev` | STATE | ambiguo (FeO baja la actividad de SnO; también es matriz fluida) | + | sin restricción |
| `feed_Sn_kgf` / `tasa_feed_total_kg_min` | ACTION de plan (no optimizable) | + | n/a | monótona + |

## Convenciones comunes a todos los experimentos

- Python: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4` (5 agentes en paralelo sobre 20 núcleos).
- Dataset: `pd.read_pickle("experimentos/cache/df_v3.pkl")` (equivale a `mp3.construir_dataset_modelo_v3`, 5 s).
- Validación: OOF GroupKFold(5) por Batch sobre DEV (`mp.split_dev_lockbox`), lockbox sólo reporte. Ningún criterio de
  selección usa el lockbox.
- Anti-fuga: sólo features con `fe.es_feature_segura_para_prescripcion(nombre) == True`.
- Salidas: `experimentos/NN_*.csv`, `experimentos/NN_log.txt`; figuras en `experimentos/figs_NN/`.
