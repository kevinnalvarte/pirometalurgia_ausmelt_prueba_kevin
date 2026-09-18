# Iteración 10 (2026-09-16) — prescriptor v8: fase 0 del dictamen (v7.1 conservador) + mejoras refutables con los datos actuales

Diseño (Fable 5.1). Punto de partida: v7 (`modelo_predictivo_v7.py`, `candidate_win_model_v7.md`, hallazgos §20) y el **dictamen del
comité** (`dictamen_comite_v7.md`, auditorías A-D en `experimentos/v7/`, hallazgos §21). Librería: `experimentos/v8/v8_lib.py`
(caché `experimentos/cache/df_v8.pkl` = df v6 + columnas m8_*).

## Meta (condiciones de aceptación del usuario)

Mejor prescriptor operacional por escalón: (1) features/targets con sentido pirometalúrgico, parsimonia y estabilidad; (2) STATE/CONTROL
separados y recomendación robusta, coherente y significativa de las CONTROL; (3) efectos de CONTROL identificados **sin cotas de signo que
fabriquen la significancia**; (4) monotonía/óptimo interior según teoría, no supuestos; (5) sin fuga, métricas OOF + lockbox (y, por el
dictamen, OOF cronológico + walk-forward); (6) SHAP/PD coherentes; (7) evidencia de que las decisiones derivadas mejoran el rendimiento del
batch; (8) target agregable escalón → grupo → fase → batch.

## Diagnóstico de partida (qué deja el comité)

| Componente v7 | Estado | Brecha que ataca la iteración 10 |
|---|---|---|
| Masa v6, targets log telescópicos, PLM cross-fitted, k16 como predictor | RESCATADO | reetiquetar FeO → Fe total; medir reversión de ruido; `espesor` es un reloj → retirar |
| Cx_sn +0.178 y GN +0.047 sobre Sn | RESCATADO (interiores) | forma `Cx_sn` = orden 2 en Sn, no justificada: probar dosis saturante (`ln(1+C/Sn_disp)`) |
| θ sobre retención de FeO (GN, O₂, aire, carbón) | EN DUDA (IC libres cruzan 0; 82 % del uplift) | reportar θ libre; target corregido por Fe alimentado; θ heterogéneo; declarar "no identificado" lo que lo sea |
| K, w (3.52 / 7.86) | EN DUDA (w ≈ 1 en lockbox; EIV) | base sin legado (2.89 / 6.09); K corregido por EIV; objetivo lineal en kg como contraste; política robusta en w ∈ [1, 11] |
| Optimizador (cota P99 en 52 % de R0; aire sin θ; IC constante) | DESCARTADO tal cual | límites P5-P95 por orden, |Δ| ≤ 1 sd, carbón R0 ≤ P90, aire/O₂ congelados, IC dependiente de la acción, soporte < P10 → sin recomendación |
| Fusión: β_F −0.58 | DESCARTADO | β_F = 0; Fusión = costos + ventana térmica dura + polvo; F1 sin recomendación; ΔT con indicador F1 |
| Latencia del ensayo (R1-R3 no ejecutables) | EN DUDA | estado ejecutable (leyes de t−2 + variables en línea): ¿cuánto R² e identificación se pierde? |
| Evidencia off-policy (KPI) | DESCARTADA como prueba (potencia 32 %) | evidencia por escalón (dosis-respuesta residual, n ≈ 1 000), adherencia con signo, uplift con IC propagado y evaluado con un modelo distinto |

## Hipótesis (cada una con experimento que la puede refutar)

- **H1 (forma del carbón, óptimo interior)**: la dosis específica saturante `ln(1 + C/Sn_disp)` (o `dosis` con cuadrático ≤ 0) conserva el
  R² OOF de `Cx_sn` (Δ ≥ −0.01), sigue identificada con IC LIBRE, y junto con `m8_exceso_C_pos` (≤ 0 sobre FeO) produce un óptimo
  **interior** del carbón en J (no de esquina). → E10-01.
- **H2 (target FeO y estado)**: descontar el FeO equivalente del Fe metálico alimentado (dross de Fe en R0) mejora el R² OOF cronológico /
  walk-forward de la retención de FeO y la identificación de GN/carbón sobre FeO; retirar `espesor` no cuesta R²; k17 (+T horno, +Al₂O₃)
  conserva la identificación del carbón. → E10-02.
- **H3 (estado ejecutable)**: con leyes de t−2 + variables en línea de t−1 el agotamiento de Sn conserva ≥ 80 % del R² OOF y el carbón
  sigue identificado; si no, sólo R0 (con F6 conocida) es recomendable en tiempo real. → E10-03.
- **H4 (objetivo robusto)**: K corregido por errores-en-variables, IC de (K, w) por bootstrap, y un objetivo lineal en kg
  (ΔSn_kg − λ·ΔFeO_kg) anclado en el KPI; el signo de la política debe conservarse en w ∈ [1, 11] y λ ∈ [0.5, 1.5]. → E10-04 (a: anclajes;
  b: sensibilidad de la política, tras construir v8).
- **H5 (evidencia por escalón)**: la dosis-respuesta no paramétrica del residuo de la acción (a − E[a|S]) sobre el residuo del target es
  monótona/saturante y significativa (n ≈ 1 000), replica en OOF cronológico; la adherencia CON SIGNO (moverse en la dirección
  recomendada) se asocia a mejor KPI; el uplift evaluado con un modelo distinto del que optimiza (anti-maldición) sigue > 0 con IC
  propagado (K, w, θ). → E10-05 (tras v8).
- **H6 (Fusión)**: con β_F = 0, la única palanca defendible de Fusión es el recorte acotado de carbón (costo + polvo) manteniendo λ de
  lanza (−O₂ proporcional) y ΔT dentro de ventana dura; ΔT con indicador F1 mejora el OOF; el término polvo (f_polvo ~ C_F) sobrevive
  q-BH en la familia de Fusión. → E10-06.

## Experimentos (ola 1 en paralelo, Sonnet 5; Fable diseña, revisa y decide)

Todos: `import v8_lib as L` (desde `experimentos/v8/`), `df = L.cargar_df()`, `r = L.base_fase(df, "Reducción")`, DEV/lockbox con
`L.split`; OOF GroupKFold(5) por Batch en DEV = criterio; **OOF cronológico y walk-forward obligatorios** (`L.r2_por_conjuntos`);
θ LIBRE y ACOTADO siempre (`L.theta_dual`, n_boot ≥ 500); anti-fuga `L.asegurar_seguras`; salidas `experimentos/v8/e10_0N_*.csv`, memo
`e10_0N_resultados.md`, log `e10_0N_log.txt`. `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=3`, sin joblib.

| Exp | Pregunta | Entregable |
|---|---|---|
| E10-01 | Forma del carbón en Reducción: `Cx_sn` (ref) vs `m8_dosis` vs `m8_ln1p_dosis` vs `dosis + dosis²` vs `ln1p_dosis + ln1p_dosis×avance`; FeO: carbón + Cx_av (ref) vs `m8_exceso_C_pos` (+ Cx_av); θ heterogéneo (C×zT, C×zB2); GN cuadrático. Óptimo interior: curva de J_R (K 2.89, w 6.09) sobre la grilla de carbón para 12 estados representativos por orden | tabla R² (OOF, cronológico, WF, lockbox) × forma × target; θ libre/acotado; % de estados con óptimo interior; PD/ALE del carbón; veredicto |
| E10-02 | Target de FeO con descuento de Fe alimentado (fFe 0.4/0.5/0.6) y piso C*; estados k16/k15/k17; fracción del R² que es reversión de ruido (sin `%FeO_prev`/`m6_feo_inv_kg_prev`/`ratio`/`interacción`); estabilidad por tercios con n_boot 500 | tabla R² × target × estado; θ libre de Cx_sn/GN; veredicto de target y estado |
| E10-03 | Estado ejecutable (`L.ESTADO_R_ONLINE`) vs k15: R² e identificación por orden R0-R3; simulador un paso (F5→F6) para R0 | tabla por orden; qué escalones son recomendables con latencia ≥ 15 min |
| E10-04a | Anclajes: K, w con IC bootstrap y corrección EIV (fiabilidad del Σln_feo_ret); objetivo lineal en kg (KPI ~ ΔSn_kg_R, ΔFeO_kg_R) → λ; legado como sensibilidad; anclaje sólo con batches con ensayo R3 | tabla de anclajes con IC; λ recomendado; K/w base v8 |
| E10-06 | Fusión: ΔT con `es_F1` vs sin F1; polvo/dross ~ carbón de Fusión, T media, GN (familia con q-BH, DEV/lockbox/total); recorte de carbón con λ constante (Δ O₂ = 2·ΔC·(O₂ teórico por kg C)) y su efecto térmico; recorte acotado (≤ 1 sd del orden, ≥ P10) | términos de J_F defendibles; regla de recorte; veredicto |

Ola 2 (Fable construye `modelo_predictivo_v8.py` con los veredictos): E10-05 validación completa + política cross-fitted con salvaguardas +
sensibilidad w/λ (E10-04b) + evidencia por escalón y batch + uplift con IC propagado y evaluación cruzada.

## Criterios de decisión (fijados antes de ver resultados)

1. Predictivo: R² OOF DEV (selección) con tolerancia Δ ≥ −0.01; OOF cronológico y walk-forward no deben empeorar más de 0.02; lockbox sólo
   reporte (gastado).
2. Palancas: efecto con IC95 % bootstrap **libre** que excluya 0 y signo de teoría; "en cota" = no identificado → no entra al objetivo
   (θ = 0 declarado, no estimado).
3. Forma del carbón: se adopta la saturante si cumple 1 y 2 y el óptimo de J es interior en ≥ 70 % de los estados de R0-R1.
4. Target de FeO: se adopta el corregido si mejora OOF cronológico o walk-forward ≥ +0.01 sin perder OOF aleatorio (> −0.01) y con la
   fracción de Fe del dross en el rango físico (0.4-0.6) sin sensibilidad de ranking (corr ≥ 0.99 entre variantes).
5. Objetivo: base sin legado; K corregido por EIV sólo si la fiabilidad estimada está en [0.6, 0.95]; la política final debe conservar el
   signo (≥ 80 % de escalones) bajo w ∈ [1, 11] y bajo el objetivo lineal en kg; donde no, la recomendación se recorta a "mantener".
6. Fusión: sólo términos con p < 0.05 y q-BH < 0.10 en DEV y mismo signo en total; si ninguno, J_F = −costos − ventana dura.
7. Evidencia: dosis-respuesta residual monótona en el signo de teoría (Spearman p < 0.01, DEV y cronológico); adherencia con signo p < 0.05
   con controles en DEV o total; uplift cruzado > 0 con IC propagado.
