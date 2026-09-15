# search_targets — familia de experimentos: ¿qué target por escalón debe maximizar el recomendador? (2026-09-14)

Diseño (Fable 5.1) a partir del estado v4 (`candidate_win_model.md`, `win_analytics.ipynb`). Ejecución paralela por
subagentes (Sonnet 5); merge, hipótesis y decisión centralizados (Fable 5.1).

## Pregunta

El prescriptor v4 maximiza por escalón `sn_extraido_est_kg` (Reducción, con `−λ_Fe·feo_extraido`) y, en Fusión, un objetivo
anclado en coeficientes de batch. Aquí se busca, de forma sistemática, la variable objetivo **por escalón** para cada fase
que, (i) sea calculable desde la teoría pirometalúrgica, (ii) sea coherente con "mejor eficiencia" de la fase, (iii) tenga una
agregación escalón→batch coherente matemática y pirometalúrgicamente, y (iv) cuyo agregado correlacione positiva y
significativamente con `rendimiento_proxy_batch` — de modo que maximizarla escalón a escalón apunte a maximizar el KPI.

## Librería común: `st_targets.py`

36 candidatos (20 Reducción, 16 Fusión), cada uno con fórmula, unidad, dirección teórica (+1 maximizar / −1 minimizar /
0 exploratorio), regla de agregación y referencia teórica (`st_registro.csv`). Columnas `st_<id>` orientadas "mayor = mejor".

| Tipo de magnitud | Agregación coherente | Ejemplos |
|---|---|---|
| Masa extensiva por escalón | suma (telescópica cuando es Δ de un inventario) | Sn extraído, FeO reducido, Sn − λ·FeO, FeO formado |
| Fracción de 1er orden del disponible | supervivencia `1 − Π(1 − f_t)`; retención `Π r_t = fin/ini` | fracción de Sn extraída, fracción de FeO retenida |
| Constante cinética | `−ln(inv_fin/inv_ini)/Σdt` | k aparente de Sn en masa |
| Cociente de masas | cociente de sumas (nunca media de cocientes, Guía 10.8) | selectividad Sn/FeO, utilización del carbón, FeO/Sn disponible |
| Intensiva (T, T gases) | media ponderada por tiempo | ventana térmica, T gases pre-BHF |
| Estado terminal de la fase | último valor disponible | Sn residual, %FeO fin Fusión, log(Sn/FeO), IRF, basicidad |

Referencia teórica: `Guia_Ausmelt_Sn.md` §3.3, 5.2-5.3, 7.4, 8, 10.7-10.8, 11.3, 14, 19, 22; `consideracioness_...md`
§26.1, 32, 35-38, 44; `hallazgos.md` §14.6; experimentos 15 y 19.

## Experimentos (paralelos, Sonnet 5)

- **ST-01 correlación batch** (`st_01_correlacion_batch.py`): para cada candidato, Spearman/Pearson vs KPIs
  (`rendimiento_proxy_batch`, `f_metal`, `f_dross`, `f_polvo`, `recuperacion_real_pct`) en DEV (selección), lockbox
  (confirmación) y total; IC bootstrap; OLS HC3 por sd con controles (`CONTROLES_BATCH`) y con tendencia cronológica;
  corrección BH (FDR 5 %) dentro de cada fase en DEV; dosis-respuesta por quintiles y monotonía; barrido de λ para la
  familia metal-equivalente; combinaciones F+R (regresión conjunta) y redundancia entre candidatos (|ρ|>0.8).
- **ST-02 predictibilidad por escalón** (`st_02_predictibilidad_escalon.py`): HGB OOF GroupKFold(5) por batch en DEV y
  lockbox para cada candidato con (a) STATE+CONTEXT v3, (b) STATE+CONTEXT+CONTROL (4 palancas + Cx v4); R², MAE, techo de
  ruido; ganancia de R² por CONTROL (= cuánto del target depende de lo que el operador decide).
- **ST-03 identificación de las palancas** (`st_03_identificacion_control.py`): PLM v4 (θ con IC bootstrap por batch) de
  las 4 palancas de CONTROL sobre cada candidato; signo teórico; **cadena de signos**: sign(θ_palanca→target) ×
  sign(target→KPI) vs sign(palanca→KPI del exp 19) — si maximizar el target mueve las palancas en la dirección que sube
  el KPI, la cadena es coherente.
- **ST-04 robustez** (`st_04_robustez.py`): agregaciones alternativas (suma / media / último / producto / media
  ponderada) para ver si la coherente es también la más informativa; estabilidad temporal (tercios cronológicos de DEV +
  lockbox); sensibilidad al ruido del trazador (perturbar %CaO ±2 % y %Sn/%FeO ±ruido de ensayo, 200 réplicas: ρ y rango);
  estabilidad del ranking.

## Criterio de decisión (Fable, tras el merge)

1. **Primario**: ρ de Spearman con `rendimiento_proxy_batch` en DEV positivo, significativo tras BH, confirmado en signo
   en lockbox, y robusto a controles (OLS HC3 por sd).
2. **Secundario**: predictible por escalón (R² OOF) y con palancas de CONTROL identificadas con signo teórico; cadena de
   signos coherente (ST-03).
3. **Terciario**: parsimonia y legibilidad operativa (una fórmula, unidades físicas), estabilidad temporal y al ruido.

El lockbox nunca se usa para elegir. Techo de referencia: el KPI de batch tiene R² OOF ≤ 0.18 con cualquier set de
agregados (hallazgos 14.6) → ρ ≈ 0.4 es el máximo plausible; ρ ≈ 0.2-0.3 con p < 0.001 ya es "fuerte" relativo a ese techo.

## Convenciones

`.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4`; datos: `cache_df_st.pkl` (escalones, 280 col.) y
`st_batch.csv` (362 batches) generados por `st_targets.py`; salidas `st_0N_*.csv`, `st_0N_log.txt`, figuras en `figs/`.
