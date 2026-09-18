# candidate_win_model_v8 — prescriptor operacional por escalón v8 (iteración 10, 2026-09-16)

Ficha del modelo candidato ganador tras la iteración 10 (`experimentos/v8/`, diseño `ITERACION_10_diseno.md`, memos `e10_0N_resultados.md`,
hallazgos §22). Código: `modelo_predictivo_v8.py` (sobre `modelo_predictivo_v6/v5/v4`, `feature_engineering.py`, `experimentos/v6/masa_v6.py`,
`experimentos/v8/v8_lib.py`). Notebook: `win_analytics_v2.ipynb` (`experimentos/build_notebook_win_v2.py`). Fichas anteriores:
`candidate_win_model_v7.md` (+ `dictamen_comite_v7.md`), `candidate_win_model_v6.md`, `candidate_win_model.md` (v5), `candidate_win_model_v4.md`.

Validación: OOF GroupKFold(5) por batch sobre DEV (299 batches, abril-julio 2026) = criterio de selección; **OOF cronológico** (5 bloques
contiguos) y **walk-forward** (bloques de 50 batches, ventana creciente ≥ 100) = generalización entre regímenes (exigidos por el dictamen);
lockbox = 63 últimos batches (agosto 2026, fin de campaña) = sólo reporte, declarado "gastado". KPI de batch: `recuperacion_refinada_pct`.

---

## 0. Resumen ejecutivo

| | Reducción (`WIN-R v8`) | Fusión (`WIN-F v8`) |
|---|---|---|
| Forma | PLM dual: `y = g(S) + Σ θ_j (a_j − E[a_j|S])`, g y E[a|S] por HGB cross-fitted por batch; **θ libre y acotado por signo sobre los mismos residuos**; θ de uso = acotado sólo si el IC95 % libre excluye 0 (`ModeloPLMDual`) | idem (sólo ΔT) |
| Targets del escalón | `m6_ln_sn_dep` = ln(Sn_disp/Sn_inv[t]) (agotar Sn); **`m8_ln_feo_ret_dross50`** = ln(FeO_inv[t] / (FeO_inv[t−1] + 1.2865·0.5·dross_Fe[t])) (retener Fe neto del Fe alimentado); ΔT (restricción) | ΔT (F2-F6; restricción de estado) |
| Masa de escoria | cierre físico v6 (sin trazador CaO) | idem |
| STATE en g(S) | **15** (`ESTADO_R_V8` = k16 de v7 sin `espesor_ladrillo_norm_mm`, reloj de campaña); modo **plan** alternativo (cierre de F6 + variables en línea, 20) | 7 (térmico + carga + cal) |
| CONTROL (θ lineal) | `Cx_sn_v6` = C·Sn_inv_prev/1000, GN, exceso O₂, aire (Sn); `m8_exceso_C_pos`, `Cx_av_v6`, GN, exceso O₂, aire (FeO) | GN, O₂, aire, carbón (ΔT) |
| Identificadas (IC libre excluye 0) | **carbón × Sn disponible → +Sn**; **GN → +Sn y −FeO**; (ΔT: GN +, O₂ −, aire −, carbón +) | ninguna |
| NO identificadas (θ de uso 0) | carbón → FeO (los dos canales), exceso O₂ y aire en todo | GN, O₂, aire, carbón sobre ΔT |
| Optimizables / congeladas | carbón y GN / O₂ y aire (setpoint de protección de lanza) | ninguna (receta) / todas |
| Objetivo | `J = K·(ln_sn_dep + w·ln_feo_ret) − costos − 0.5·sd_J(acción) − soporte relativo`, **K = 2.781 pp/unidad [1.60, 3.96], w = 5.93 [3.96, 9.62]**, sin legado; contraste `lineal_kg`: ΔSn_kg − 3.0·ΔFeO_kg | `J_F = −costos` (β_F = 0; polvo y dross se cancelan) |
| Salvaguardas (dictamen fase 0) | caja por orden = [P5, P95] ∩ [hist − 1 sd, hist + 1 sd]; carbón R0 ≤ P90; soporte del estado < P10 → sin recomendación; ventana térmica dura relativa (P10-P90 del orden con margen MAE, nunca peor que la histórica); IC dependiente de la acción (réplicas bootstrap de θ); **sin ganancia física → "mantener: sólo costo"**; **GN sólo si su dirección se conserva en w ∈ [3.96, 9.62]** | F1 sin recomendación; recorte económico sólo con precios reales, ≤ 1 sd, ≥ P10, con −O₂ = 0.933·ΔC; prohibido en fin de campaña o con T previa < P10 |
| R² OOF / cronológico / walk-forward / lockbox | ver §4 (E10-05) | ΔT 0.50 / 0.34 / 0.21 / 0.04 (sin F1) |
| Política (362 batches cross-fitted) | ver §5 (E10-05) | receta |
| Evidencia | θ con IC libre + dosis-respuesta residual (n ≈ 1 000) + residuo de la acción → KPI + anclaje target → KPI (replica en lockbox vía legado) + uplift con IC propagado (K, w, θ) + evaluación cruzada anti-maldición | — |

---

## 1. Qué se probó en la iteración 10 y qué sobrevivió

| Hipótesis | Experimento | Veredicto (criterios fijados a priori en `ITERACION_10_diseno.md`) |
|---|---|---|
| H1 forma saturante del carbón con óptimo interior | E10-01 | **Refutada**: 8 formas dan el mismo R² (0.763-0.767); sólo `Cx_sn_v6` se identifica con IC libre (+0.179 [+0.033, +0.292]); dosis saturante +0.023 [−0.006, +0.056]; curvatura −0.034 [−0.130, +0.006] (signo teórico, no identificada); θ heterogéneo (C×T, C×B2) nulo; 0 % de estados con óptimo interior de J. Carbón sobre FeO no identificado con ninguna forma; GN sobre FeO sí (−0.010 [−0.020, −0.0025]) |
| H2 target de FeO neto del Fe alimentado; estado sin espesor / k17 | E10-02 | **Adoptado el target**: R² OOF 0.33 → 0.43-0.45 (R0 0.05 → 0.12; R1-R3 idénticos, corr 1.0), OOF cronológico 0.30 → 0.40, walk-forward 0.25 → 0.32; GN identificado; ranking robusto a fFe 0.4-0.6 (corr ≥ 0.81 en R0); piso C* rechazado (R² 0.55). Estado: k15 sin espesor (el comité; R² igual, carbón identificado) — detalle en `e10_02_resultados.md` |
| H3 estado ejecutable (latencia del ensayo) | E10-03 | **A medias**: con leyes de t−2 (online) se conserva el 89 % del R² pooled de Sn pero el carbón deja de identificarse; con el cierre de F6 para todo R (plan) 92 % pooled y carbón (+0.118 [0.028, 0.209]) y GN identificados, **pero dentro de cada orden** el R² cae (R1 0.03 vs 0.55): el plan sirve para R0 y para la dirección de R1-R3; el re-plan con el ensayo de t−1 (modo k15) es el que fija la magnitud |
| H4 objetivo robusto | E10-04a/b | Base sin legado K 2.78 / w 5.93 [3.96, 9.62]; legado replica en lockbox (+19.8 pp/unidad, p 0.012) → sensibilidad K 3.52 / w 7.86; EIV (λ_FeO 0.82) → w 7.4 (sensibilidad); lockbox w ≈ 0.8; objetivo lineal en kg λ 4.7 DEV / 2.8 total → 3.0 como contraste; anclaje invariante al target de FeO y a los 105 batches sin ensayo R3. Robustez de la política en E10-05 (§5) |
| H5 evidencia por escalón | E10-05 | ver §6 |
| H6 Fusión | E10-06 | ΔT **sin F1** (OOF 0.40 → 0.50; walk-forward 0.09 → 0.21); `es_F1` no aporta; ninguna palanca térmica identificada; carbón de Fusión sube polvo (+0.006/sd, p 0.02) y baja dross (−0.008/sd, p 0.01), neto KPI nulo (p 0.7) → **sin término de polvo; Fusión = receta** |

## 2. Fórmulas

```
r_t = 1 − 1.270·s_t − f_t ;  M_t·r_t = M_{t−1}·r_{t−1}·e^{−0.01·[R]} + 0.703·cal_t + 0.301·carga_t          (masa v6, cierre físico)
m6_ln_sn_dep[t]        = ln( (Sn_inv[t−1] + Sn alimentado[t]) / Sn_inv[t] )
m8_ln_feo_ret_dross50[t] = ln( FeO_inv[t] / (FeO_inv[t−1] + (71.844/55.845)·0.50·feed_dross_Fe_kgh[t]) )     (sólo Reducción; fFe 0.50, sens. 0.40/0.60)
Cx_sn_v6 = C [kg/min]·Sn_inv_prev/1000 ;  Cx_av_v6 = C·avance_prev ;  m8_exceso_C_pos = max(0, C_kg − 0.2024·Sn_disp)
exceso_o2 = 100·(O2 + 0.21·aire − 2·GN)/(2·GN)                                                              (índice de lanza, congelado)
θ_uso_j = θ_acot_j · 1[ IC95 % libre_j excluye 0 ]
J_t(R)  = K·(ln_sn_dep_t + w·ln_feo_ret_t) − 0.05·C·dt − 0.03·GN·dt − 0.01·O2·dt − 0.5·sd_J(acción) − 2000·max(0, s_hist − s)
J_t(F)  = −costos                                                                                            (K en pp de KPI; × 512 kg Sn/pp)
lineal_kg: J = (ΔSn_kg − 3.0·ΔFeO_kg)/512·512, ΔSn_kg = Sn_disp·(1 − e^{−ln_sn_dep}), ΔFeO_kg = FeO_prev·(1 − e^{ln_feo_ret})
Caja por orden o: a ∈ [max(P5_o, a_hist − sd_o), min(P95_o, a_hist + sd_o)] ; carbón R0 ≤ P90_0 ; O2, aire = plan
```
Agregación (E9-00, invariante en v8): Σ_R ln_sn_dep = ln(Sn_ini/Sn_fin) (+0.005 ± 0.018), Σ_R ln_feo_ret neto = ln(FeO_fin/(FeO_ini + Fe alimentado))
(exacto); `J_R = Σ_t J_t` por grupo (R0-R1 temprano, R2-R3 tardío) y por fase; `J_batch = J_R + J_F`.

## 3. Estado de Reducción (`ESTADO_R_V8`, 15) y modo plan

Leyes previas %Sn, %FeO, %SiO₂, %CaO; `ratio_sn_feo_prev`, `interaccion_Sn_x_FeO_prev`; `termocupla_media_celsius_prev`,
`grad_temperatura_horno_celsius_prev`; `posicion_vertical_lanza_mm_prev`; `cum_feed_Carbon_kg_prev`, `cum_aire_nm3_prev`,
`relacion_C_cum_Sn_cum_prev`; `m6_sn_inv_kg_prev`, `m6_feo_inv_kg_prev`, `m6_resto_frac_prev`. Se retira `espesor_ladrillo_norm_mm`
(ρ −0.9996 con la fecha; rango del lockbox disjunto; no cuesta R²). Modo **plan** (`activar_modo_plan()`): leyes/inventarios del cierre de F6
(`*_F6`) + orden + variables en línea de t−1 + acumulados; palancas `Cx_sn_F6`, `Cx_av_F6`.

## 4. Métricas (E10-05, `e10_05_tabla_validacion_v8.csv`)

| Fase | Target | Forma | k estado | n DEV / lockbox | R² OOF | MAE OOF | R² OOF cronológico | R² walk-forward | R² lockbox | MAE lockbox |
|---|---|---|---|---|---|---|---|---|---|---|
| Reducción | agotamiento Sn `m6_ln_sn_dep` | **PLM v8** | 15 | 1083 / 242 | **0.767** | 0.247 | **0.775** | **0.771** | 0.800 | 0.242 |
| Reducción | agotamiento Sn | HGB directo (15 + palancas) | 15 | | 0.760 | 0.251 | 0.765 | — | 0.805 | 0.239 |
| Reducción | agotamiento Sn | v7 (k16, referencia) | 16 | | 0.767 | 0.247 | 0.775 | 0.761 | 0.795 | 0.245 |
| Reducción | retención FeO neta `m8_ln_feo_ret_dross50` | **PLM v8** | 15 | 1083 / 242 | **0.438** | 0.047 | **0.416** | **0.333** | 0.566 | 0.043 |
| Reducción | retención FeO neta | HGB directo | 15 | | 0.442 | 0.047 | 0.398 | — | 0.536 | 0.044 |
| Reducción | retención FeO `m6_ln_feo_ret` (v7, referencia) | PLM | 16 | | 0.333 | 0.047 | 0.296 | 0.261 | 0.483 | 0.042 |
| Reducción | ΔT [°C] | PLM v8 | 5 | 1182 / 252 | 0.440 | 9.5 | 0.264 | −0.016 | 0.423 | 6.5 |
| Fusión (F2-F6) | ΔT [°C] | PLM v8 | 7 | 1482 / 314 | 0.480 | 9.1 | 0.359 | 0.166 | 0.050 | 7.0 |
| Fusión (F1-F6) | ΔT (v7, con F1 imputado) | PLM | 8 | 1778 / 376 | 0.403 | 9.9 | 0.258 | 0.092 | 0.092 | 7.5 |

Lectura honesta del R² pooled (E10-03 §3): el orden del escalón por sí solo explica 0.70 del agotamiento y 0.25 de la retención; el estado
añade +0.09 y +0.06. La métrica dentro de cada orden (`e10_05_r2_por_orden.csv`, OOF DEV / lockbox): agotamiento R0 0.17 / 0.01, **R1 0.58 /
0.65**, R2 0.05 / −0.15, R3 −0.10 / −0.14; retención neta R0 0.10 / 0.14, **R1 0.27 / 0.14**, R2 0.02 / −0.14, R3 −0.08 / −0.13. Es decir, el
modelo predice el nivel del escalón y tiene poder discriminante dentro del orden sólo en R0-R1 (donde están el Sn disponible y las
decisiones que importan); en R2-R3 el target es ruido de ensayo (SNR 1.2, E8-01) y lo que el prescriptor usa allí es θ, no g(S). ΔT de
Reducción no generaliza en walk-forward (−0.02): sólo se usa como ventana térmica relativa a la acción histórica, nunca como objetivo.
El HGB directo no supera al PLM en ningún target (E9-05 sigue vigente).

## 5. Capa prescriptiva (E10-05, `e10_05_recomendaciones_v8_escalones.csv`, `e10_05_magnitud_cambios.csv`, `e10_05_sensibilidad_resumen.csv`)

Política cross-fitted sobre los 362 batches (cada batch recomendado por un sistema entrenado sin él; folds 0-4 de DEV + lockbox con todo DEV;
`n_boot` 100 para el IC dependiente de la acción). Estados de recomendación en Reducción (fila = orden):

| orden | recomendar | mantener | mantener: sólo costo | sin rec.: soporte < P10 | sin rec.: estado |
|---|---|---|---|---|---|
| R0 | 58 % | 10 % | 9 % | 22 % | 0 % |
| R1 | 41 % | 17 % | 19 % | 22 % | 1 % |
| R2 | 41 % | 20 % | 22 % | 16 % | 2 % |
| R3 | 39 % | 24 % | 25 % | 11 % | 2 % |

Fusión: F1 sin recomendación (orden), F2-F6 "receta" (85 %) o sin soporte (14-25 %). "Sólo costo" = la búsqueda mejoraba J únicamente por el
término de costo (precios placeholder) sin ganancia física K·Δ(ln_sn_dep + w·ln_feo_ret) > 0 → se mantiene la acción histórica (dictamen §2).

Magnitud de los cambios (escalones recomendar/mantener; Δ = recomendado − histórico; sd = sd del orden en DEV):

| orden | palanca | hist P50 | Δ P5 | Δ P50 | Δ P95 | Δ medio | % sube / % baja | % \|Δ\| > 1 sd | % rec > P95 / < P5 del orden | % en borde de la caja |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | carbón kg/min | 62.6 | −5.4 | **+0.7** | +7.0 | +1.4 | 56 / 16 | 0 | 2.9 / 0.4 (hist ya fuera) | 3.6 (R0 > P90: 18 %, todos con hist > P90) |
| R1 | carbón | 29.7 | −6.8 | 0.0 | +6.8 | −0.2 | 26 / 26 | 0 | 3.9 / 1.8 | 1.8 |
| R2 | carbón | 11.2 | −5.7 | 0.0 | +4.4 | −0.5 | 21 / 27 | 0 | 2.3 / 1.0 | 2.0 |
| R3 | carbón | 5.6 | −3.5 | 0.0 | +3.1 | −0.2 | 18 / 25 | 0 | 5.1 / 5.1 | 2.5 |
| R0-R3 | GN Nm³/min | 28.3 / 25.0 / 18.8 / 16.7 | −2.1..−1.7 | 0.0 | +1.8..+2.9 | +0.0..+0.1 | 22-40 / 17-34 | 0 | ≤ 11 / ≤ 1 | ≤ 2.5 |

Lectura: las recomendaciones son **pasos pequeños y dependientes del estado** (mediana 0; P5-P95 ≈ ±1 sd del orden), nunca fuera de la caja
(0 % con |Δ| > 1 sd; sólo 2-4 % en el borde: el término de soporte relativo y el IC dependiente de la acción producen óptimos interiores), y
el carbón de R0 sólo supera el P90 cuando el histórico ya lo superaba. En R0 el sesgo es "algo más de carbón" (56 % sube) — con techo P90 —
y en R1-R3 la dirección se reparte (el recorte tardío de v7, −1.3..−2.6 kg/min, **desaparece**: sin el canal carbón → FeO identificado y sin
precios reales, no hay base para recortar). GN: ajustes de ±2 Nm³/min según el balance +Sn / −FeO del estado.

Robustez de la política al objetivo (E10-05 `sensibilidad`, fold 0, 60-70 batches, mismos estados; recomendaciones re-optimizadas bajo cada
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

## 6. Evidencia (E10-05, `e10_05_dosis_respuesta*.csv`, `e10_05_uplift_propagado.csv`, `e10_05_evaluacion_cruzada*.csv`, `e10_05_evidencia_politica_v8.csv`)

1. **Palanca → target por escalón, con potencia (n 1083, DEV)** — θ con IC libre (§0) y dosis-respuesta no paramétrica del residuo de la
   acción (a − E[a|S]) sobre el residuo del target (`e10_05_dosis_respuesta*.csv`): carbón × Sn disponible → agotamiento Spearman +0.080
   (p 0.008), quintil inferior −0.054 [−0.084, −0.023] y superior +0.015 (creciente y **saturante**; tercios 1 y 2 p < 0.05); GN → agotamiento
   +0.112 (p 2e-4; quintiles −0.047 → +0.035; los 3 tercios p ≤ 0.05); GN → retención de FeO −0.120 (p 1e-4; quintiles +0.014 → −0.010;
   tercios 0 y 2 p < 0.01). Exceso de O₂ y aire: nulos. Carbón sobre FeO (sobrante, C×avance): nulos.
2. **Residuo de la acción → KPI del batch, sin pasar por el modelo** (OLS HC3 con controles, DEV n 292-297): más GN del que el estado predice
   → **menor KPI** (−0.46 pp/sd, p 0.043; R tardío −0.42, p 0.08), coherente con la política (el efecto neto del GN en J es ≤ 0 con w 5.9).
   Carbón: en R temprano nulo (−0.28 pp/sd, p 0.24; Spearman −0.10, p 0.08), en **R tardío positivo** (+0.81 pp/sd, p 0.015; Spearman +0.19,
   p 0.0015): operar con más carbón del esperado en R2-R3 se asocia a mejor KPI. Es observacional (36 pruebas; sobrevive ~BH 0.05) pero
   contradice el recorte tardío de v7 y **motiva la salvaguarda "sólo costo"** (§5) y un brazo del piloto (carbón R2-R3 +1 sd).
3. **Target → KPI** (E10-04): K_Sn 2.78 [1.60, 3.96], K_FeO 16.5 [10.3, 22.7] (p < 1e-4); legado del FeO retenido sobre el batch siguiente
   +9.4 [3.6, 15.2] DEV y **+19.8 [4.3, 35.2] en lockbox** (réplica fuera de muestra).
4. **Uplift con IC propagado** (`e10_05_uplift_propagado.csv`; K y K_FeO por bootstrap del anclaje, factor de θ U(0.6, 1.4)): por batch
   **+0.072 pp [0.038, 0.121] DEV, +0.060 [0.032, 0.104] lockbox** (≈ 30-37 kg de Sn a metal por batch), 85 % de batches > 0; por grupo
   R temprano +0.04 pp, R tardío +0.03 pp. Es un cuarto del +0.27 de v7 (que descansaba en θ no identificados sobre FeO y en cajas globales).
5. **Evaluación cruzada anti-maldición** (`e10_05_evaluacion_cruzada*.csv`, 555 escalones "recomendar" de DEV): el ΔJ que el optimizador se
   atribuye (0.037 pp/escalón, 100 % > 0) se reduce a **0.010 pp/escalón (27 %) evaluado con el PLM de otro fold** (70 % > 0; por batch
   +0.023 [0.017, 0.030], IC que excluye 0; corr 0.63 con el optimizador) y a **−0.043 pp con un HGB directo** (41 % > 0; corr −0.03): un
   evaluador no paramétrico no reconoce la ganancia. Conclusión honesta: el signo del uplift es robusto dentro de la familia PLM y el
   tamaño está inflado ×3-4 por la maldición del optimizador; fuera de la familia PLM no hay evidencia de ganancia → sólo el piloto aleatorizado
   puede fijar el valor.
6. **Monitoreo (sin potencia, dictamen §3)**: adherencia `dist_R` y dirección por palanca vs KPI: nulas en DEV y total (|coef| ≤ 0.37 pp/sd,
   p ≥ 0.16); en lockbox `dist_R` +1.3 pp/sd (p 0.03, n 63): en fin de campaña (w ≈ 1) los batches más lejos de la política de DEV rindieron
   mejor → reentrenar y re-anclar por campaña es obligatorio antes de operar.

## 7. Cómo reproducir

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v8/v8_lib.py                # caché df_v8.pkl
    .venv/Scripts/python.exe experimentos/v8/e10_05_validacion_v8.py validacion                                # tablas, θ, predicciones, curvas
    .venv/Scripts/python.exe experimentos/v8/e10_05_validacion_v8.py 0..4|lockbox                              # política cross-fitted (Start-Process, ~20 min c/u)
    .venv/Scripts/python.exe experimentos/v8/e10_05_validacion_v8.py evidencia|cruzada|sensibilidad|dosis_respuesta
    .venv/Scripts/python.exe experimentos/build_notebook_win_v2.py && jupyter nbconvert --to notebook --execute --inplace win_analytics_v2.ipynb

## 8. Positivo / negativo (qué mejorar en la siguiente iteración)

**Positivo (lo que v8 deja resuelto).** Identificación honesta (θ libre y acotado; palancas declaradas no identificadas con θ de uso 0) sin
perder ninguna palanca real: carbón × Sn disponible y GN, con dosis-respuesta no paramétrica que confirma el signo con potencia. Target de FeO
neto del Fe alimentado (+0.10 de R² en OOF, cronológico y walk-forward; GN identificado sobre FeO; anclaje invariante). Estado sin relojes de
campaña. Objetivo anclado sin legado con IC y contraste lineal en kg; política robusta en el carbón bajo cualquier objetivo. Optimizador
conservador que cumple las siete correcciones de la fase 0 del dictamen (caja por orden, R0 ≤ P90, aire/O₂ congelados, soporte P10, ventana
térmica dura, IC dependiente de la acción, F1/Fusión receta) y dos salvaguardas nuevas (ganancia física, robustez del GN a w). Modo `plan`
para planificar R0-R1 sin latencia de laboratorio. Uplift con IC propagado y evaluación cruzada anti-maldición.

**Negativo (qué queda abierto y sólo el piloto o datos nuevos cierran).** (1) Carbón → retención de FeO no identificado con ninguna forma: el
prescriptor no puede fijar el óptimo del carbón (recomienda pasos acotados) y el recorte tardío de v7 pierde su base; el residuo de carbón
tardío → KPI positivo (+0.8 pp/sd) pide un brazo de piloto "carbón R2-R3 +1 sd". (2) Uplift pequeño (+0.07 pp/batch) e inflado ×3-4 por la
maldición del optimizador; un HGB directo no lo reconoce. (3) El GN cambia de dirección con w (lockbox w ≈ 1): reentrenar y re-anclar por
campaña es obligatorio. (4) R² dentro del orden sólo en R0-R1; R2-R3 son ruido de ensayo (%FeO 2.8 %). (5) Precios placeholder: el costo no
puede ser motor de ninguna recomendación hasta tener precios de planta. (6) Fusión sin palanca identificada: receta. (7) Modo plan pierde el
R² dentro de R1 (0.03 vs 0.55): el re-plan con el ensayo de t−1 sigue siendo necesario para la magnitud. Siguiente iteración = fase 1-3 del
dictamen (datos de planta, modo sombra, piloto aleatorizado por escalón), no más modelado con estos datos.
