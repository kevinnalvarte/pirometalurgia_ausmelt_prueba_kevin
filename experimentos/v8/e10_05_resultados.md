# E10-05 — Validación del sistema v8, política cross-fitted con salvaguardas, sensibilidad, evidencia y evaluación cruzada

Script `e10_05_validacion_v8.py` (etapas `validacion | 0..4 | lockbox | evidencia | cruzada | sensibilidad | dosis_respuesta`; logs
`e10_05_log_*.txt`). Sistema `modelo_predictivo_v8.py` con la configuración final de la iteración 10 (estado k15, targets `m6_ln_sn_dep` y
`m8_ln_feo_ret_dross50`, palancas `Cx_sn_v6`/GN/exceso O₂/aire y `m8_exceso_C_pos`/`Cx_av_v6`/GN/exceso O₂/aire, K 2.781 / w 5.93, caja por
orden, aire y O₂ congelados, Fusión receta, salvaguarda de ganancia física).

## 1. Validación (`e10_05_tabla_validacion_v8.csv`, `e10_05_efectos_control_v8.csv`, `e10_05_r2_por_orden.csv`)

| target | R² OOF | cron. | WF | lockbox | v7 (OOF / cron / WF / lockbox) |
|---|---|---|---|---|---|
| Reducción agotamiento Sn | 0.767 | 0.775 | 0.771 | 0.800 | 0.767 / 0.775 / 0.761 / 0.795 |
| Reducción retención FeO neta | **0.438** | **0.416** | **0.333** | **0.566** | 0.333 / 0.296 / 0.261 / 0.483 |
| Reducción ΔT | 0.440 | 0.264 | −0.016 | 0.423 | 0.464 / — / — / 0.351 |
| Fusión ΔT (F2-F6) | 0.480 | 0.359 | 0.166 | 0.050 | 0.403 / 0.258 / 0.092 / 0.092 (con F1) |

El HGB directo (mismas features) no supera al PLM en ningún target. R² dentro de cada orden (OOF DEV / lockbox): Sn R0 0.17/0.01, R1 0.58/0.65,
R2 0.05/−0.15, R3 −0.10/−0.14; FeO R0 0.10/0.14, R1 0.27/0.14, R2 0.02/−0.14, R3 −0.08/−0.13 (el orden explica 0.70 y 0.25 del pooled, E10-03).

θ (1 sd, n_boot 500, DEV): agotamiento — `Cx_sn_v6` libre +0.179 [+0.033, +0.292] (identificado, 0.8 % en cota), GN +0.056 [+0.020, +0.097];
exceso O₂ −0.004 [−0.022, +0.014] y aire +0.006 [−0.012, +0.022] no identificados. Retención de FeO neta — GN **−0.012 [−0.023, −0.004]**
(identificado con IC libre), `m8_exceso_C_pos` +0.003 [−0.004, +0.011] y `Cx_av_v6` −0.003 [−0.018, +0.010] no identificados (θ de uso 0),
exceso O₂ +0.002 [−0.002, +0.006], aire +0.003 [−0.001, +0.008] no identificados. ΔT de Reducción: GN +3.5 °C/sd, O₂ −4.9, aire −0.8,
carbón +3.2 (todos con IC libre que excluye 0; el signo del O₂ es control reactivo, auditoría A: la ventana térmica es relativa, no un
objetivo). ΔT de Fusión: sólo O₂ −2.1 [−3.5, −0.6] (mismo artefacto) → sin palanca.

Óptimo interior de J (curvas P5-P95 del orden en 12 estados, `e10_05_optimo_interior.csv`): GN 100 % interior en los cuatro órdenes; carbón
67 % en R0 y 100 % en R1-R3 (soporte relativo + IC dependiente de la acción; en v7 el 52 % de los R0 caía en la cota P99).

## 2. Política cross-fitted (`e10_05_recomendaciones_v8_escalones.csv`, `e10_05_magnitud_cambios.csv`)

3 619 escalones, 362 batches. Reducción: recomendar 58 % (R0) / 41 % / 41 % / 39 %; mantener 10-24 %; **mantener: sólo costo 9-25 %**
(la búsqueda mejoraba J sólo por el costo con precios placeholder → se mantiene la acción histórica); sin recomendación por soporte < P10
22 % (R0-R1) → 11 % (R3); sin estado 0-2 %. Fusión: F1 sin recomendación, F2-F6 receta (85 %) o sin soporte (14-25 %).

Δ carbón (kg/min): R0 P50 +0.7 (P5 −5.4, P95 +7.0; 56 % sube, 16 % baja; > P90 sólo si el histórico ya lo superaba); R1 P50 0.0 (±6.8; 26/26);
R2 0.0 (−5.7/+4.4; 21/27); R3 0.0 (−3.5/+3.1; 18/25). Δ GN: P50 0.0, P5-P95 −2.1..+2.9 Nm³/min. 0 % con |Δ| > 1 sd; 2-4 % en el borde de la
caja; 0 % fuera de P5-P95 salvo históricos ya fuera. Frente a v7: desaparece el recorte tardío de carbón (−1.3..−2.6) y el "+3 en R0"; GN
deja de ser "−0.3..−0.6" uniforme y pasa a ±2 según el estado; aire y O₂ no se tocan.

## 3. Uplift con IC propagado (`e10_05_uplift_propagado.csv`, `e10_05_por_batch_v8.csv`)

Por batch (pp de KPI, K_Sn y K_FeO por bootstrap del anclaje, θ × U(0.6, 1.4)): DEV **+0.072 [0.038, 0.121]**, lockbox +0.060 [0.032, 0.104],
total +0.070 [0.037, 0.117]; 85 % de batches > 0 (no tautológico: el 15 % restante son batches con "mantener" o penalizaciones). ≈ 30-37 kg
de Sn a metal por batch (v7: +0.27 [0.25, 0.29] con IC que no propagaba nada). Por grupo: R temprano +0.04 pp, R tardío +0.03 pp.

## 4. Evaluación cruzada anti-maldición del optimizador (`e10_05_evaluacion_cruzada*.csv`)

555 escalones "recomendar" de DEV. ΔJ predicho por escalón (pp): optimizador 0.037 (100 % > 0, por construcción); **PLM de otro fold 0.010
(27 %, 70 % > 0; por batch +0.023 [0.017, 0.030])**; **HGB directo −0.043 (41 % > 0; por batch −0.10 [−0.14, −0.06])**; corr con el
optimizador 0.63 (PLM) y −0.03 (HGB). Por orden el patrón se repite (R1: 0.029 / 0.003 / −0.125). Lectura: dentro de la familia PLM el signo
del uplift se conserva con un tamaño ×3-4 menor (maldición del optimizador); un evaluador no paramétrico no ve ganancia (su PD del carbón es
plana dentro del orden y no contiene el término lineal pequeño que el PLM identifica). El valor de la política no puede afirmarse desde los
datos observacionales: sólo el piloto aleatorizado por escalón.

## 5. Sensibilidad del objetivo (`e10_05_sensibilidad_resumen.csv`)

(fold 0, 60-70 batches, mismos estados; recomendaciones re-optimizadas bajo cada
objetivo y filtradas por ganancia física): % de escalones con recomendación de **signo estrictamente opuesto** a la base —

| palanca | orden | legado (K 3.52, w 7.86) | EIV w 7.4 | w 11 | w 4 | w 2 | lineal kg λ 1 | λ 3 | λ 4.7 |
|---|---|---|---|---|---|---|---|---|---|
| carbón | R0 / R1 / R2 / R3 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 2 / 0 / 0 | 2 / 2 / 0 / 0 | 3 / 2 / 0 / 0 | 0 / 0 / 0 / 0 | 7 / 3 / 2 / 2 | 7 / 5 / 3 / 5 |
| GN | R0 / R1 / R2 / R3 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 / 5 / 2 / 5 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 2 / 2 / 5 | 12 / 10 / 12 / 7 | 13 / 12 / 15 / 13 |

El **carbón es robusto** (≤ 7 % de signos opuestos bajo cualquier objetivo; Δ medio −0.3..−2.2 kg/min en todas las variantes). El **GN es
sensible al peso del FeO**: su Δ medio pasa de +0.1..+0.4 Nm³/min (base, w 5.9) a −0.4..−0.7 con legado/EIV/w 11 y a −0.9..−1.7 con el
objetivo lineal en kg (que valora más el FeO), y a +0.3..+0.9 con w 2-4. Criterio 5 del diseño (conservar el signo en ≥ 80 %): se cumple
para el carbón y para el GN en w ∈ [4, 11] y bajo λ ≤ 1, y **no** para el GN bajo λ 3-4.7 en R0-R2 (85-88 %). Consecuencia adoptada en el
módulo (`robustez_w = (3.96, 9.62)`): el GN sólo se mueve si su dirección se conserva en los dos extremos del IC de w; si no, GN = histórico
(las tablas cross-fitted de §5 se calcularon sin esta regla: son un superconjunto de las recomendaciones de GN).

## 6. Evidencia por escalón y por batch (`e10_05_dosis_respuesta*.csv`)

- Dosis-respuesta residual (quintiles de a − E[a|S] vs residuo del target, bootstrap por batch): `Cx_sn_v6` → agotamiento Spearman +0.080
  (p 0.008), quintiles −0.054 [−0.084, −0.023], −0.013, −0.011, +0.048 [+0.017, +0.078], +0.015: creciente y saturante; por tercios
  cronológicos +0.03 (n.s.), +0.11 (p 0.04), +0.11 (p 0.04). GN → agotamiento +0.112 (p 2e-4; −0.047 → +0.035; 3/3 tercios). GN → retención
  de FeO −0.120 (p 1e-4; +0.014 → −0.010; tercios 0 y 2 p < 0.01). Exceso O₂ → FeO +0.069 (p 0.02) pero θ no identificado; aire nulo;
  carbón sobre FeO nulo (Spearman 0.00-0.04).
- Residuo de la acción agregado por batch → KPI refinado (OLS HC3 + controles): GN (R todo) **−0.46 pp/sd, p 0.043**; carbón × Sn: R temprano
  −0.28 (p 0.24), **R tardío +0.81 (p 0.015; Spearman +0.19, p 0.0015)**; `Cx_av` R tardío +0.79 (p 0.002); exceso O₂ y aire nulos. El GN
  respalda la política; el carbón tardío la contradice en la dirección de v7 (recortar) y respalda la salvaguarda "sólo costo" de v8.

## 7. Monitoreo de adherencia (`e10_05_evidencia_politica_v8.csv`; sin potencia, dictamen §3)

DEV y total: `dist_R`, dirección por palanca y `uplift_batch_pp` vs KPI nulos (|coef| ≤ 0.37 pp/sd, p ≥ 0.16). Lockbox (n 63): `dist_R`
+1.3 pp/sd (p 0.03): en fin de campaña (w ≈ 1) los batches más alejados de la política anclada en DEV rindieron mejor → re-anclar y
reentrenar por campaña antes de operar. Placebo (permutación de dist/dir entre batches): nulo.

## Veredicto E10-05

v8 cumple la fase 0 del dictamen (θ honesto, caja por orden, aire/O₂ congelados, IC dependiente de la acción, sin legado, Fusión receta) y
mejora el núcleo predictivo (retención de FeO +0.10 de R² en OOF, cronológico y walk-forward). La política resultante es conservadora
(pasos ≤ 1 sd, mediana 0) con un uplift modelado pequeño (+0.07 pp/batch, IC propagado que excluye 0) cuyo tamaño no sobrevive a un
evaluador no paramétrico. Lo que falta no es más modelado: es el piloto aleatorizado por escalón (carbón R0-R1 y R2-R3 ±1 sd; GN R1-R3),
que identifica el canal carbón → FeO, la curvatura del carbón y el valor real.
