# candidate_win_model — Modelo prescriptor operacional por escalón (iteración 7 → v5, 2026-09-15)

Ficha del modelo candidato ganador tras la iteración 7 (`experimentos/v5/`, diseño en `ITERACION_7_diseno.md`). Código:
`modelo_predictivo_v5.py` (extiende v4: reutiliza `dataset_lingo_smelter.py`, `feature_engineering.py`, `modelo_prescriptivo.py`,
`modelo_predictivo_v3.py`, `modelo_predictivo_v4.py`). Notebook: `win_analytics_v1.ipynb` (`experimentos/build_notebook_win_v1.py`).
La ficha de v4 se conserva en `candidate_win_model_v4.md`.

Validación: OOF GroupKFold(5) por batch sobre DEV (299 batches, abril-julio 2026) = criterio de selección; lockbox = últimos 63
batches cronológicos (agosto 2026, fin de campaña de refractario), sólo reporte; walk-forward cronológico y reentrenamiento por
campaña (`e7_04`) como backtest. KPI de batch: `recuperacion_refinada_pct` = 100·M/(M+D+P+Sn escoria final) (iteración 6).

---

## 0. Resumen ejecutivo

| | Reducción (`WIN-R v5`) | Fusión (`WIN-F v5`) |
|---|---|---|
| Forma | PLM con signos de teoría: `y = g(S) + Σ θ_j (a_j − E[a_j|S])`, θ por mínimos cuadrados acotados (reductor no sube la retención de FeO, etc.) | idem |
| Targets del escalón (log, adimensionales) | `v5_ln_sn_dep` = ln(Sn_disp/Sn_inv[t]) (agotar Sn), `v5_ln_feo_ret` = ln(FeO_inv[t]/FeO_inv[t−1]) (retener FeO), ΔT | `v5_ln_sn_dep` (conservar Sn en la escoria), ΔT |
| STATE en `g(S)` | 31 estado + 6 contexto v3 (el estado completo es el único que identifica el carbón) | 24 curado (composición, matriz, térmico, lanza, inventarios, plan de carga) |
| CONTROL (θ lineal) | carbón × Sn disponible (`Cx_sn_v4`), GN, exceso O₂, aire (Sn); carbón, carbón × avance, GN, exceso O₂, aire (FeO) | C/Sn cargado, GN, exceso O₂, aire |
| R² OOF / lockbox | Sn 0.766 / 0.802 · FeO 0.164 / 0.187 · ΔT 0.466 / 0.465 | Sn 0.187 / −0.39 (sólo θ) · ΔT 0.481 / 0.066 |
| Objetivo | `J_R = K·(ln_sn_dep + w·ln_feo_ret) − costos − ventana T − IC − w_S·max(0, s_hist − s)` con **w = 6.04 [3.8, 13.6]** y **K = 2.12 pp de KPI/unidad ≈ 1 090 kg Sn** anclados en la regresión del KPI refinado | `J_F = −K_S·ln_sn_dep − costos − ventana T − IC − soporte`, **K_S = 0.45 pp/unidad ≈ 230 kg** (conservación) |
| Efectos CONTROL significativos (IC95 %, por +1 sd) | **Cx_sn +0.183 sd [0.008, 0.288]** sobre agotamiento; **GN −0.021 [−0.031, −0.005]** sobre retención de FeO (no selectivo); GN +2.3 °C, O₂ −3.7 °C | **GN +0.030 [0.011, 0.050]** sobre agotamiento (extrae Sn); O₂ −1.25 °C |
| Recomendación típica (política cross-fitted, 362 batches) | carbón +3.4 kg/min con avance < 0.84 y −1.4 con 0.96-0.975; GN −0.7 a −1.7 Nm³/min; O₂ −0.2 a −1.5; aire +1 a +2 | carbón −3 a −6 kg/min; GN −0.2 a −0.3; O₂ −0.3 a −0.5 (menos gas de lanza dentro de la ventana térmica) |
| Uplift físico estimado (J, mediana por batch) | +85 kg Sn (media 115) | +53 kg (media 67) |
| Uplift por la cadena θ → agregado → KPI | **+0.20 pp de KPI [0.13, 0.30] en DEV; +0.28 [0.18, 0.43] en lockbox** (≈ +100-140 kg Sn/batch; > 97 % de batches positivos) | — |
| Evidencia off-policy (adherencia vs KPI refinado, OLS HC3 con controles) | `dist_R` nula (recomendaciones pequeñas, sin potencia) | **`dist_F` −0.70 pp/sd (p 0.025, permutación 0.004) en DEV; polvo +0.008/sd (p < 0.001 DEV y total)**; placebo nulo |

---

## 1. Qué se modela (fórmulas)

Masa de escoria por trazador CaO (la cal, tolva 6, es inerte; ver `feature_engineering.agregar_inventario_masa_escoria`):
```
masa_escoria_est_kg[t]          = Σ_{k≤t} feed_CaO_kgh[k] / (ley_cao_escoria_pct[t] / 100)
sn_inventario_escoria_est_kg[t] = masa · %Sn/100 ;  feo_inventario_escoria_est_kg[t] = masa · %FeO/100
avance_reduccion_sn_prev[t]     = 1 − sn_inventario[t−1] / Σ_{k<t} feed_Sn_kgf[k]
```
Targets v5 por escalón (`modelo_predictivo_v5.agregar_targets_v5`; nunca features; válidos si Sn_inv > 20 kg y FeO_inv > 100 kg):
```
v5_ln_sn_dep[t]  = ln( (sn_inv[t−1] + feed_Sn_kgf[t]) / sn_inv[t] )   agotamiento del Sn de la escoria (1er orden: = k·Δt)
v5_ln_feo_ret[t] = ln( feo_inv[t] / feo_inv[t−1] )                    retención de FeO (≤ 0 si se metaliza Fe); sólo Reducción
v5_lnirf[t]      = ln( %FeO / (%SiO2 + %Al2O3 + %CaO) )               matriz (índice IRF de planta); v5_d_lnirf = Δ por escalón
```
La masa del trazador se cancela en cada cociente (quedan leyes y el cociente de %CaO); la suma por batch es telescópica:
`Σ ln_sn_dep = ln(Sn_ini/Sn_fin)`, `Σ ln_feo_ret = ln(FeO_fin/FeO_ini)`. SDI (iteración 5) = `Σ ln_sn_dep + w·Σ ln_feo_ret`.

Palancas con estructura cinética (acción del escalón × estado previo, seguras):
```
Cx_sn_v4 = tasa_feed_Carbon_kg_min · sn_inventario_escoria_est_kg_prev / 1000    (SnO2 + 2C: velocidad ∝ [C]·[SnOx])
Cx_av_v4 = tasa_feed_Carbon_kg_min · avance_reduccion_sn_prev                   (FeO + C crece al agotarse el SnOx)
exceso_o2_combustion_pct = 100·(O2 + 0.21·aire − 2·GN) / (2·GN)                   (potencial redox de la lanza)
relacion_C_Sn_carga = feed_Carbon_kgh / feed_Sn_kgf                               (Fusión; estequiométrico 0.20)
```
Modelo parcialmente lineal con signos de teoría (`ModeloPLMSignos`; Robinson cross-fitted GroupKFold(5) por batch; HGB max_depth 4,
150 it., min_leaf 20; θ por `scipy.optimize.lsq_linear` con cotas de signo; IC95 % bootstrap cluster por batch, 300):
```
y = g(S) + Σ_j θ_j · (a_j − E[a_j|S]) + ε ;   θ_j ≥ 0 si la teoría dice +, ≤ 0 si dice −, libre si no hay expectativa
```
Un θ que el dato empuja contra su signo colapsa a 0 ("sin efecto identificado") en vez de entrar al objetivo con un signo físicamente
imposible (p. ej. carbón +0.0004/kg·min sobre la retención de FeO en la versión sin cotas).

## 2. Roles STATE / CONTEXT / CONTROL / PLAN

- **STATE** (cierre de t−1): leyes (%Sn, %FeO, %SiO₂, %CaO), basicidades, IRF, log-ratio Sn/FeO, temperatura y gradiente, termocuplas,
  tiro, posición de lanza previa, acumulados por corriente y de gases, C/Sn acumulado, masa de escoria e inventarios de Sn/FeO, avance.
- **CONTEXT**: posición del escalón, tiempo de fase, espesor de refractario, leyes de Sn por corriente del batch.
- **CONTROL**: `tasa_feed_Carbon_kg_min`, `tasa_gn_nm3_min`, `tasa_o2_nm3_min`, `tasa_aire_nm3_min`.
- **PLAN** (no se optimiza): mezcla de carga por tolva, cal, `duracion_plan_min` (fija por protocolo en Reducción).
  Hallazgo de batch (E7-02): más cal por t de carga → −0.87 pp de KPI/sd (p 0.038 DEV, 0.006 con heel): recomendación de plan,
  no de escalón.

## 3. Features por modelo y efectos θ (DEV, `e7_05_efectos_control_v5.csv`)

| Fase / target | Estado `g(S)` | Palanca | Signo teórico | θ por +1 sd [IC95 %] | Lectura |
|---|---|---|---|---|---|
| R / `v5_ln_sn_dep` (sd 0.68) | STATE+CONTEXT v3 (37) | `Cx_sn_v4` | + | **+0.183 [0.008, 0.288]** | carbón selectivo: agota Sn ∝ Sn disponible |
| | | `tasa_gn_nm3_min` | + | +0.046 [0.000, 0.091] | calor + gas reductor |
| | | `exceso_o2_combustion_pct` | − | −0.004 [−0.025, 0] | lanza oxidante frena (débil) |
| | | `tasa_aire_nm3_min` | · | +0.009 [−0.016, 0.026] | |
| R / `v5_ln_feo_ret` (sd 0.11) | idem | `tasa_feed_Carbon_kg_min` | − | 0 (cota) [−0.024, 0] | el carbón no metaliza Fe de forma identificable |
| | | `Cx_av_v4` | − | 0 (cota) [−0.020, 0] | |
| | | `tasa_gn_nm3_min` | − | **−0.021 [−0.031, −0.005]** | reductor no selectivo |
| | | `exceso_o2_combustion_pct` | + | +0.001 [0, 0.008] | |
| | | `tasa_aire_nm3_min` | + | +0.002 [0, 0.010] | |
| R / ΔT | T_prev, termocupla, masa, orden, grad T, espesor | GN, O₂, aire, carbón | +, ·, −, · | **GN +2.3 [0.2, 4.2]**, **O₂ −3.7 [−6.5, −1.0]**, aire −0.4, C +1.9 | |
| F / `v5_ln_sn_dep` (sd 0.24) | 24 curado + plan de carga | `relacion_C_Sn_carga` | + | 0 (cota) [0, 0.017] | el carbón no mueve el Sn de la escoria en Fusión |
| | | `tasa_gn_nm3_min` | + | **+0.030 [0.011, 0.050]** | el gas de lanza extrae Sn |
| | | `exceso_o2_combustion_pct` | − | 0 (cota) | |
| | | `tasa_aire_nm3_min` | · | +0.018 [−0.005, 0.042] | |
| F / ΔT | idem R + carga + cal | GN, O₂, aire, carbón | | O₂ −1.25 [−2.65, −0.22]; resto n.s. | modelo térmico de Fusión débil (lockbox 0.07) |

Estabilidad (`e7_04_theta_periodos_resumen.csv`, tercios cronológicos de DEV + lockbox): `Cx_sn` mismo signo en los 4 períodos y
significativo en 2; GN sobre agotamiento positivo en los 4; GN sobre retención de FeO negativo en tercios 2-3 y DEV completo pero
positivo en el tercio 1 y en el lockbox (n = 63; frágil); carbón sobre FeO cambia de signo (por eso la cota de teoría).

## 4. Métricas (OOF DEV / lockbox), `e7_05_tabla_validacion_v5.csv`

| Fase | Target | Forma | k estado (+θ) | R² OOF | MAE OOF | R² lockbox | MAE lockbox | sd DEV |
|---|---|---|---|---|---|---|---|---|
| Reducción | agotamiento Sn [ln] | **PLM v5** | 37 (+4) | **0.766** | 0.244 | **0.802** | 0.228 | 0.678 |
| Reducción | agotamiento Sn [ln] | HGB estado+palancas (ref.) | 37 | 0.761 | 0.249 | 0.801 | 0.229 | |
| Reducción | retención FeO [ln] | **PLM v5** | 37 (+5) | **0.164** | 0.065 | **0.187** | 0.054 | 0.109 |
| Reducción | retención FeO [ln] | HGB (ref.) | 37 | 0.133 | 0.066 | 0.186 | 0.054 | |
| Reducción | ΔT [°C] | **PLM v5** | 6 (+4) | 0.466 | 9.2 | **0.465** | 6.4 | 17.5 |
| Reducción | ΔT [°C] | HGB (ref.) | 6 | 0.461 | 9.3 | 0.350 | 7.0 | |
| Fusión | agotamiento Sn [ln] | PLM v5 | 24 (+4) | 0.187 | 0.147 | −0.386 | 0.149 | 0.238 |
| Fusión | ΔT [°C] | PLM v5 | 8 (+4) | 0.481 | 9.1 | 0.066 | 6.9 | 17.2 |

Estado curado (20) en Reducción: agotamiento 0.760 / 0.804, retención de FeO 0.203-0.212 / 0.12-0.16 (`e7_01`), pero el carbón deja de
identificarse (IC cruza 0): se adopta el estado completo porque contiene la historia de dosificación (acumulados de carbón, C/Sn
acumulado) que confunde la dosis del escalón. Parsimonia (`e7_04_parsimonia.csv`, eliminación hacia atrás por importancia de permutación; `e7_07_parsimonia_v5.csv` re-evalúa los
sets con las palancas y signos de v5): agotamiento con 10 features R² 0.775 / 0.807 pero `Cx_sn` baja a +0.12 [0, 0.23] (queda en la
cota) y el GN pasa a +0.064 [0.018, 0.097]*; retención de FeO con 16 features R² 0.223 / 0.263 pero el GN colapsa a 0 [−0.011, 0] y el
exceso de O₂ pasa a +0.007 [0.001, 0.012]*. La parsimonia mejora el ajuste ~0.01-0.05 y cambia qué palanca se identifica; se adopta el
estado completo (identifica el carbón selectivo y el GN no selectivo, coherente con exp 19 y ST-12) y se deja el set de 10/16 como
variante de producción cuando prime la simplicidad.

Backtest (`e7_04_walkforward.csv`, bloques de 50 batches): agotamiento 0.744 walk-forward vs 0.762 OOF; retención de FeO 0.144 vs
0.206. Reentrenamiento progresivo cada 10 batches sobre el lockbox (`e7_04_reentrenamiento.csv`): agotamiento 0.805 → 0.812,
**retención de FeO 0.122 → 0.286**, SDI 0.141 → 0.216. Techo de ruido del trazador para la retención de FeO ≈ 0.8 (ST-02).

## 5. Objetivo y capa prescriptiva

```
Reducción: J_R = K·[ ln_sn_dep_pred + w·ln_feo_ret_pred ] − 0.05·C·dt − 0.03·GN·dt − 0.01·O2·dt − 2·(T_pred fuera de P5-P95)
                 − 0.25·K·ancho_IC − 2000·max(0, s_hist − s_cand)
Fusión:    J_F = −K_S·ln_sn_dep_pred − costos − 2·(T fuera de P5-P95) − 0.25·K_S·ancho_IC − 2000·max(0, s_hist − s_cand)
```
- **Anclaje en el KPI** (`e7_01_w_anclaje.csv`, OLS HC3 del KPI refinado sobre Σln_sn_dep y Σln_feo_ret con controles + tendencia,
  DEV n = 285): +2.12 pp por unidad de agotamiento (p 0.001) y +12.83 pp por unidad de retención de FeO (p < 0.001) → w = 6.04,
  IC bootstrap [3.8, 13.6], 100 % de réplicas con ambos coeficientes positivos. K = 2.12 pp × 512 kg/pp. Con el proxy w = 8.4; con el
  batch siguiente w ≈ 8-10 pero n.s. Barrido de w (`e7_01_w_sweep.csv`): ρ_DEV sube hasta w ≈ 12-14 (0.32) y ρ_lockbox baja con w;
  w = 6 es el anclaje causal, no el máximo de correlación.
- **Fusión** (`e7_02`): KPI refinado ~ −0.45 pp por unidad de Σln_sn_dep de Fusión (p 0.083 DEV; −0.59, p 0.014 total; −0.95 lockbox
  n.s.), controlando el heel de escoria. Los términos v4 (β_C sobre dross, β_T sobre polvo) replican por canal pero se cancelan sobre
  el KPI (C/Sn: dross −0.008/sd p 0.014, polvo +0.006 p 0.04; T: polvo +0.008 p 0.05, dross −0.006) → retirados.
- **Soporte relativo**: en v4 el término absoluto `−w_S(1−s)` dominaba (diferencias de 900 kg entre candidatos frente a ~200 kg de
  física) y empujaba hacia la moda histórica; v5 penaliza sólo perder densidad respecto de la acción histórica y mantiene el mínimo
  duro (P10 DEV). Búsqueda aleatoria (300) en P1-P99 + refinamiento local (60); la acción histórica siempre es candidata.
- **Formas**: lineales en las palancas con signo de teoría; `e7_03_cuadraticos.csv`: ningún cuadrático mejora el OOF (agotamiento
  0.760 → 0.759, retención 0.203 → 0.191); sólo el aire sobre el agotamiento sale cóncavo (vértice 117 Nm³/min, IC [88, 260]).
  Los óptimos interiores de J vienen del trade-off Sn/FeO con w, de la ventana térmica y del soporte. Selectividad del carbón por
  tramo de avance (`e7_03_selectividad_por_tramo.csv`, −θ_FeO/θ_Sn): 0.10 con avance < 0.84, 0.34 en 0.84-0.96 y no identificable por
  encima de 0.96 (queda poco Sn): es lo que `Cx_sn` reproduce por construcción. PD/ALE (`e7_03_formas_vs_teoria.csv`): carbón, GN y
  exceso de O₂ monótonos con el signo teórico en las dos componentes; basicidad con óptimo interior; T y lanza previas (estado) con
  formas no monótonas, libres en g(S).

Recomendación media de la política cross-fitted (`e7_05_recomendaciones_v5_escalones.csv`, 362 batches):

| Reducción, avance previo | Δ carbón [kg/min] | Δ GN [Nm³/min] | Δ O₂ | Δ aire | uplift J [kg/escalón] | carbón hist. |
|---|---|---|---|---|---|---|
| < 0.84 | **+3.4** | −0.7 | −0.2 | +1.0 | 31 | 60.8 |
| 0.84-0.96 | +0.9 | −1.4 | −0.9 | +2.2 | 28 | 40.6 |
| 0.96-0.975 | **−1.4** | −1.7 | −1.5 | +2.1 | 30 | 22.2 |
| > 0.975 | −0.6 | −1.1 | −0.7 | +1.7 | 23 | 11.9 |

| Fusión, escalón | Δ carbón | Δ GN | Δ O₂ | Δ aire | uplift J |
|---|---|---|---|---|---|
| F1 | −3.0 | −0.3 | −0.5 | −0.3 | 7 |
| F2-F6 | −3.9 a −5.6 | −0.2 a −0.3 | −0.3 a −0.5 | −0.1 a −0.4 | 11-14 |

## 6. Evidencia de que las recomendaciones conducen a mayor recuperación del batch

1. **Target → KPI** (iteraciones 5-7, `v5_lib.bateria_batch`): Σln_feo_ret ρ +0.29 DEV (OLS +1.27 pp/sd, p < 0.001); Σln_sn_dep ≈ 0
   en DEV pero +0.35 en lockbox (p 0.004); SDI (w 6) ρ 0.30 DEV / 0.14 lockbox / 0.28 total; ln IRF twmean 0.30 / 0.38 / 0.33;
   Fusión Σln_sn_dep −0.14 total (p 0.01). Monótono en quintiles (SDI +4.4 pp Q5−Q1).
2. **Palanca → target**: θ con IC (sección 3); cadena de signos coherente con exp 19 (GN total de Reducción sube dross y baja metal;
   carbón total baja dross).
3. **Cadena θ → agregado → KPI** (`e7_06_cadena.csv`): cambio predicho por la política (Σ pred_rec − pred_hist): agotamiento −0.03,
   retención de FeO +0.023 por batch → **+0.20 pp de KPI [0.13, 0.30] en DEV, +0.28 [0.18, 0.43] en lockbox** (pendientes de batch
   con bootstrap), ≈ +100-140 kg Sn/batch, positivo en > 97 % de los batches. Coincide con el uplift físico de J (mediana 157 kg/batch).
4. **Off-policy cross-fitted** (`e7_05_evidencia_politica_v5.csv`, `e7_06_*`; cada batch recomendado por un sistema que no lo vio):
   - Fusión: `dist_F` −0.70 pp de KPI refinado por sd (IC HC3, p 0.025; permutación 0.004) en DEV; sobre la media del batch y el
     siguiente −0.61 (p 0.006 DEV; −0.60 lockbox n.s.; −0.32 total p 0.11); sobre f_polvo **+0.008/sd (p < 0.001 DEV y total)**: alejarse
     de la política de Fusión (más gas y carbón) = más polvo. Por palanca: carbón de Fusión −0.79 pp/sd (p 0.024 DEV; media2 −0.76,
     p 0.002 DEV / 0.044 lockbox / 0.031 total). Placebo con política aleatoria ≈ 0.
   - Reducción: `dist_R` nula en OLS, emparejamiento y dosis-respuesta (cerca 69.2 vs lejos 67.5 pp en DEV, n.s.): los cambios
     recomendados son pequeños (GN −0.24 sd, carbón ≈ 0 en media) y el diseño observacional no tiene potencia.
   - Emparejamiento por estado (`e7_06_matching.csv`): nulo en DEV y lockbox; en TOTAL sobre la media del batch actual y el siguiente
     +1.8 pp [1.1, 2.5] (`dist_total`) y +1.8 [1.0, 2.6] (`dist_F`), no replicado dentro de cada régimen.

## 7. Hallazgos relevantes de la iteración (por qué v5 y no v4)

- El objetivo v4 en kg (Sn − 0.6·FeO) no correlaciona con el KPI (ρ 0.035); las componentes log sí, y su peso relativo se estima del
  KPI en vez de fijarse (w 6 frente a un λ_Fe de 0.6 en masa).
- El carbón en Reducción sólo se identifica con el estado completo (`Cx_sn` +0.18*); reparametrizarlo (dosis específica, sobrante,
  log) no mejora ni el R² ni la identificación (`e7_01`). Sobre la retención de FeO el carbón es ≈ 0 (selectivo) y el GN negativo.
- Fusión: con el KPI refinado ningún control tiene efecto neto de batch (dross y polvo se compensan); el heel de escoria domina
  (+1.7 pp/sd); Δln IRF durante la Fusión es nulo controlando el heel; conservar el Sn en la escoria es el único término de signo
  teórico (−0.45 pp/unidad).
- Cuadráticos residualizados no significativos donde la teoría no los espera; θ con cotas de signo evitan efectos espurios.
- El soporte absoluto de v4 dominaba J; con soporte relativo las recomendaciones son modestas y físicamente dirigidas.
- Reentrenar progresivamente por campaña mejora el lockbox de la retención de FeO (0.12 → 0.29).

## 8. Positivo / negativo (a mejorar en la siguiente iteración)

**Positivo**
- Targets adimensionales con agregación telescópica y relación demostrada con el KPI; objetivo con w y K anclados en el KPI refinado;
  una sola familia de modelos por fase con 4-5 palancas, θ con IC y signos de teoría; roles STATE/CONTROL/PLAN nítidos.
- Recomendaciones coherentes: carbón temprano y menos tardío, menos GN (no selectivo), lanza algo más oxidante, menos gas y carbón
  en Fusión dentro de la ventana térmica; nunca fuera del envolvente histórico.
- Evidencia en cuatro capas; la de Fusión con placebo nulo y canal físico identificado (polvo); la de Reducción por la cadena
  θ → agregado → KPI con IC positivo en DEV y lockbox.

**Negativo / abierto**
- Retención de FeO: R² OOF 0.16-0.21 (techo 0.8); GN sobre FeO cambia de signo en el primer tercio y en el lockbox; el carbón
  sobre FeO no se identifica (queda en la cota 0): la selectividad decreciente con el avance (exp 15) entra sólo vía `Cx_sn`.
- La evidencia off-policy de Reducción es nula (baja potencia); la de Fusión es DEV-only y la recomendación de carbón en Fusión es
  de costo/conservación (θ = 0), respaldada por el canal polvo pero con efecto neto de batch n.s. → validar en piloto A/B.
- R² lockbox negativo del agotamiento de Sn en Fusión (`e7_09_diag_fusion_lockbox.py`): correlación real-predicho 0.37 (0.43 en DEV),
  sesgo +0.08 concentrado en F6 (+0.23; horno ~160 °C más frío al fin de campaña) y varianza del target 25 % menor; ajustando sólo
  con el lockbox θ_GN cae a 0 [0, 0.004] y con reentrenamiento progresivo a +0.017 (DEV +0.030): el término de conservación de
  Fusión es de identificación débil y debe operarse con reentrenamiento por campaña o retirarse (dejando sólo ventana térmica y costos).
- Modelo térmico de Fusión sin generalización (lockbox 0.07); estado de 37 columnas en Reducción (los sets de 10/16 features igualan
  o superan el R² pero cambian la palanca identificada: carbón/GN → GN/exceso O₂).
- Lockbox = fin de campaña: reentrenamiento progresivo cada 10 batches recomendado en producción.
- Variante con lanza como palanca (`e7_08_resultados.md`, `activar_variante_lanza`): curvatura significativa (máximo del agotamiento
  ≈ 2 700 mm en Reducción), uplift estimado +0.36 pp vs +0.20, pero sin mejora de la evidencia off-policy y con la convención de la
  medida sin confirmar → no adoptada; candidata para el piloto.
- Pendientes de planta: sentido de la posición de lanza, CaO de ganga, precios de reactivos, asignación del polvo por batch.
