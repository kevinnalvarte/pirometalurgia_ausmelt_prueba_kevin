# candidate_win_model — Modelo prescriptor operacional por escalón (iteración 4, 2026-09-14)

Ficha del/los modelos candidatos tras la iteración 4 (`experimentos/13-21`, diseño en `experimentos/ITERACION_4_diseno.md`).
Código: `modelo_predictivo_v4.py` (reutiliza `dataset_lingo_smelter.py`, `feature_engineering.py`, `modelo_prescriptivo.py`,
`modelo_predictivo_v3.py`). Notebook: `win_analytics.ipynb` (`experimentos/build_notebook_win.py`).

Validación: OOF GroupKFold(5) por batch sobre DEV (299 batches, abril-julio 2026) = criterio de selección; lockbox = últimos
63 batches cronológicos (agosto 2026), sólo reporte; walk-forward cronológico (`experimentos/17`) como backtest de estabilidad.

---

## 0. Resumen ejecutivo

| | Reducción (`WIN-R`) | Fusión (`WIN-F`) |
|---|---|---|
| Forma | PLM: `y = g(S) + Σ θ_j (a_j − E[a_j|S])` | PLM (Sn extraído, ΔT) + objetivo anclado en KPI de batch |
| Targets del escalón | `sn_extraido_est_kg`, `feo_extraido_est_kg`, `d_temperatura_horno_celsius` | `sn_extraido_est_kg` (what-if), `d_temperatura_horno_celsius` |
| STATE en `g(S)` | 31 estado + 6 contexto v3 (no lineal) | 7 estado curado + 2 controles de plan (`feed_Sn_kgf`, tasa de carga) |
| CONTROL (θ lineal) | carbón×Sn disponible, GN, exceso O₂ (Sn); carbón, carbón×avance, GN, exceso O₂, aire (FeO) | C/Sn cargado, cal/carga (sólo lectura), GN, exceso O₂ |
| Palancas que mueve el optimizador | carbón, GN, O₂, aire | carbón, GN, O₂, aire |
| R² OOF / lockbox | Sn 0.975 / 0.984 · FeO 0.357 / 0.378 · ΔT 0.474 / 0.427 | Sn 0.727 / 0.726 · ΔT 0.478 / 0.059 |
| Objetivo | `Sn − 0.60·FeO − costos − ventana T − IC − soporte` | `β_C·z(C/Sn) − β_T·z(T_pred) − costos − IC − soporte` |
| Efectos CONTROL significativos (IC95%) | GN +81 kg Sn/sd; exceso O₂ −50 kg FeO/sd; GN +2.3 °C/sd; O₂ −3.7 °C/sd | GN +221 kg Sn/sd; (cal −440 kg/sd: contaminada por trazador, no palanca) |
| Recomendación típica | más carbón con avance < 0.84 (+2.6 kg/min), menos carbón con avance > 0.96 (−1.1 a −1.5), GN ≈ igual | carbón +4 a +10 kg/min para llevar C/Sn de 0.21-0.29 a 0.27-0.35 |
| Uplift físico estimado (mediana por batch) | +34 kg Sn (DEV) / +21 (lockbox) | +732 kg (DEV) / +368 (lockbox) |
| Evidencia off-policy (lockbox, n=63) | `dist_R` −0.84 pp de rendimiento por sd (p=0.008); `dist_total` −0.91 (p=0.055) | `dist_F` −0.63 (n.s.) |

---

## 1. Qué se modela (fórmulas)

Masa de escoria por trazador CaO (la cal, tolva 6, es inerte y es el único aporte de CaO cuantificado):

```
masa_escoria_est_kg[t]          = Σ_{k≤t} feed_CaO_kgh[k] / (ley_cao_escoria_pct[t] / 100)
sn_inventario_escoria_est_kg[t] = masa_escoria_est_kg[t] · ley_sn_escoria_pct[t] / 100
feo_inventario_escoria_est_kg[t]= masa_escoria_est_kg[t] · ley_feo_escoria_pct[t] / 100
avance_reduccion_sn_prev[t]     = 1 − sn_inventario[t−1] / Σ_{k<t} feed_Sn_kgf[k]
```
Targets (nunca features):
```
sn_extraido_est_kg[t]  = feed_Sn_kgf[t] − (sn_inventario[t] − sn_inventario[t−1])   # Sn que dejó la escoria (metal o polvo)
feo_extraido_est_kg[t] = −(feo_inventario[t] − feo_inventario[t−1])                   # FeO reducido (Reducción)
d_temperatura_horno_celsius[t] = T[t] − T[t−1]
```
Estructura cinética v4 (acción × estado previo, seguras):
```
Cx_sn_v4 = tasa_feed_Carbon_kg_min · sn_inventario_escoria_est_kg_prev / 1000   # SnO2 + 2C: velocidad ∝ [C]·[SnOx]
Cx_av_v4 = tasa_feed_Carbon_kg_min · avance_reduccion_sn_prev                   # FeO + C: crece al agotarse el SnOx
relacion_C_Sn_carga = feed_Carbon_kgh / feed_Sn_kgf                               # kg C por kg Sn cargado (estequiométrico 0.20)
exceso_o2_combustion_pct = 100·(O2 + 0.21·aire − 2·GN) / (2·GN)                   # potencial redox de la lanza
```
Modelo parcialmente lineal (Robinson, cross-fitted GroupKFold(5) por batch; HGB max_depth 4, 150 it., min_leaf 20):
```
y = g(S) + Σ_j θ_j · (a_j − E[a_j | S]) + ε ;  θ por OLS sobre residuos out-of-fold; IC95% bootstrap cluster por batch (300)
```

## 2. Roles STATE / CONTEXT / CONTROL / PLAN

- **STATE** (conocido al cierre de t−1): leyes previas (%Sn, %FeO, %SiO₂, %CaO), basicidades, IRF, ratios, temperatura y gradiente, termocuplas de carcasa, tiro, posición de lanza previa, acumulados por corriente, masa de escoria e inventario de Sn/FeO, avance de reducción.
- **CONTEXT**: posición del escalón, tiempo de fase, espesor de refractario, ley de Sn por corriente del batch.
- **CONTROL** (palancas del escalón): `tasa_feed_Carbon_kg_min`, `tasa_gn_nm3_min`, `tasa_o2_nm3_min`, `tasa_aire_nm3_min`.
- **PLAN** (no se optimiza): mezcla de carga por tolva, cal (`tasa_feed_CaO_kg_min`), `duracion_plan_min` (en Reducción fija por protocolo: 30/25/20/13 min, std 0 en 362 batches).

## 3. Features por modelo

| Fase / target | Estado `g(S)` | Palancas θ | Signo teórico | θ por +1 sd [IC95%] |
|---|---|---|---|---|
| R / Sn extraído | STATE_V3 (31) + CONTEXT_V3 (6) | `Cx_sn_v4` | + | +461 [−44, +904] kg |
| | | `tasa_gn_nm3_min` | + | **+81 [+23, +144]** kg |
| | | `exceso_o2_combustion_pct` | − | −0.4 [−22, +18] kg |
| R / FeO reducido | idem | `tasa_feed_Carbon_kg_min` | + | +414 [−340, +1298] kg |
| | | `Cx_av_v4` | + | −302 [−1054, +387] kg |
| | | `tasa_gn_nm3_min` | + | +57 [−51, +166] kg |
| | | `exceso_o2_combustion_pct` | − | **−50 [−101, −3]** kg |
| | | `tasa_aire_nm3_min` | − | +8 [−46, +56] kg |
| R / ΔT | T_prev, termocupla_prev, masa_prev, orden, grad_T_prev, espesor | GN, O₂, aire, carbón | +, ·, −, · | **GN +2.3 [0.0, 4.2]**, **O₂ −3.7 [−6.5, −0.7]**, aire −0.4, C +1.9 °C |
| F / Sn extraído | sn_inv_prev, masa_prev, %Sn_prev, %FeO_prev, B2_prev, T_prev, orden, `feed_Sn_kgf`, tasa de carga | `relacion_C_Sn_carga` | + | +51 [−77, +195] kg |
| | | `relacion_CaO_carga` | (artefacto) | −440 [−720, −182] kg |
| | | `tasa_gn_nm3_min` | + | **+221 [+126, +339]** kg |
| | | `exceso_o2_combustion_pct` | · | +81 [−19, +164] kg |
| F / ΔT | idem R + tasa de carga + cal | GN, O₂, aire, carbón | | ninguno significativo |

## 4. Métricas (OOF DEV / lockbox), `experimentos/21_tabla_validacion_v4.csv`

| Fase | Target | Forma | k estado | R² OOF | MAE OOF | R² lockbox | MAE lockbox | sd lockbox |
|---|---|---|---|---|---|---|---|---|
| Fusión | Sn extraído [kg] | **PLM v4** | 9 (+4 θ) | 0.727 | 1003 | **0.726** | 1050 | 2941 |
| Fusión | Sn extraído [kg] | HGB v3 (ref.) | 13 | 0.724 | 1023 | 0.665 | 1145 | |
| Fusión | ΔT [°C] | PLM v4 | 8 (+4) | 0.478 | 9.1 | 0.059 | 7.0 | 9.2 |
| Reducción | Sn extraído [kg] | **PLM v4** | 37 (+3) | 0.975 | 344 | **0.984** | 333 | 4386 |
| Reducción | Sn extraído [kg] | HGB v3 (ref.) | 10 | 0.971 | 372 | 0.983 | 359 | |
| Reducción | FeO reducido [kg] | **PLM v4** | 37 (+5) | 0.357 | 544 | **0.378** | 490 | 814 |
| Reducción | FeO reducido [kg] | HGB v3 (ref.) | 11 | 0.288 | 566 | 0.321 | 496 | |
| Reducción | ΔT [°C] | **PLM v4** | 6 (+4) | 0.474 | 9.2 | **0.427** | 6.6 | 11.3 |

Backtest de estabilidad (walk-forward cronológico en DEV, `experimentos/17`, sets v3): Sn Fusión 0.688 (vs OOF 0.722), Sn Reducción
0.970 (vs 0.971), FeO 0.227 (vs 0.286), ΔT Reducción 0.09 (vs 0.44; el modelo térmico depende de la campaña). Reentrenamiento
progresivo por campaña (cada ~10 batches) mejora el lockbox +0.04 a +0.13.

Techo de ruido del trazador (`experimentos/15`): R²_max ≈ 0.68 para FeO reducido (sd de ruido ≈ 500 kg), ≈ 1.0 para Sn.

## 5. Objetivo y capa prescriptiva

```
Reducción: J_R = Sn_pred − λ_Fe·max(FeO_pred, 0) − 0.05·C·dt − 0.03·GN·dt − 0.01·O2·dt − 2·(T_pred fuera de P5-P95) − 0.25·ancho_IC(Sn) − w_S·(1 − soporte)
Fusión:    J_F = β_C·z(C/Sn) − β_T·z(T_pred) − costos − 0.25·ancho_IC(ΔT)·β_T/sd_T − w_S·(1 − soporte)
```
- `λ_Fe = 0.60` kg de Sn a metal por kg de FeO reducido (canales dross + polvo; `f_metal ~ −0.0139/sd de FeO_R`, p=0.013,
  `experimentos/19`; sólo dross 0.30-0.32 [0.09, 0.54], `experimentos/15`). Con 0.30 el GN parecía neutro y el optimizador lo subía,
  contra la evidencia de batch.
- `β_C = 0.0087·f_dross` por sd de C/Sn de Fusión (IC [0.0019, 0.0154], p=0.011) y `β_T = 0.0129·f_polvo` por sd de T media de
  Fusión (IC [0.0015, 0.0243], p=0.027), OLS HC3 con controles (`experimentos/19` M1); en kg por escalón: 0.0087·51 250/7 ≈ 64 kg
  por sd de C/Sn (sd 0.018) y ≈ 94 kg por sd de T (sd 75 °C).
- Soporte histórico kNN sobre estado curado + palancas; mínimo duro = P10 de DEV; `w_S` = 2000 kg (v4) — variante 500 evaluada
  en `experimentos/21_*_ws500.csv`. Búsqueda aleatoria (300) en P1-P99 + refinamiento local (60); la acción histórica siempre
  es candidata.
- Formas: las palancas son monótonas en el target (por construcción y por teoría); el óptimo interior de `J` surge de la
  selectividad (`−λ·FeO`), la ventana térmica y el soporte. Curvas de respuesta en `experimentos/21_curvas_respuesta_v4.csv`.

Recomendación media (cross-fitted, 362 batches, `experimentos/21_recomendaciones_v4_escalones.csv`):

| Reducción, avance previo | Δ carbón [kg/min] | Δ GN | Δ O₂ | Δ aire | uplift J [kg/escalón] |
|---|---|---|---|---|---|
| < 0.84 (R0) | **+2.6** | −0.05 | +0.1 | +0.4 | 96 |
| 0.84-0.96 | 0.0 | +0.4 | +0.3 | +1.9 | 110 |
| 0.96-0.975 | **−1.5** | +0.2 | −0.7 | +0.8 | 144 |
| > 0.975 | **−1.1** | +0.2 | −0.3 | +1.2 | 125 |

| Fusión, escalón | Δ carbón [kg/min] | C/Sn histórico → recomendado |
|---|---|---|
| F1 | +10.6 | 0.21 → 0.27 |
| F2-F5 | +3.7 a +5.3 | 0.29-0.32 → 0.32-0.35 |
| F6 | +6.6 | 0.27 → 0.31 |

## 6. Evidencia de que las recomendaciones conducen a mayor rendimiento proxy

1. **Mecanismo escalón → batch** (`experimentos/15`, `19`): FeO reducido en Reducción baja `f_metal` (−0.0139/sd, p=0.013) y sube
   dross (λ_Fe 0.30-0.32, p=0.003) y polvo (+0.0074/sd, p=0.04); selectividad marginal FeO/Sn 0.11 → 1.55 al pasar avance 0.96.
2. **Decisiones por fase vs. KPI** (`experimentos/19`, OLS HC3 con controles, DEV): C/Sn en Fusión −0.0087 f_dross/sd (p=0.011);
   carbón total en Reducción −0.0072 f_dross/sd (p=0.03); GN total en Reducción +0.0076 f_dross/sd (p=0.04) y −0.0104 f_metal/sd
   (p=0.03); T media de Fusión +0.0129 f_polvo/sd (p=0.027). Las direcciones del prescriptor son exactamente éstas.
3. **Off-policy cross-fitted** (`experimentos/21_evidencia_politica_v4.csv`; cada batch recomendado por un sistema que no lo vio):
   en el **lockbox** (n=63) la distancia de la operación real a la política de Reducción se asocia a menor rendimiento
   (`dist_R` −0.84 pp por sd, IC [−1.47, −0.22], p=0.008; `dist_total` −0.91, p=0.055; Spearman −0.21/−0.23); placebo con
   política aleatoria: −0.27 (p=0.66). En **DEV** la asociación es nula (+0.26, p=0.26) y en TOTAL +0.07 (n.s.): la evidencia
   off-policy es **parcial** (sólo el hold-out cronológico) y observacional.
4. **Uplift estimado por el modelo**: mediana +784 kg Sn-metal eq./batch en DEV (+399 en lockbox), ~93% por el término de
   Fusión (β_C) y ~5% por Reducción. Es una estimación, no una observación.

## 7. Hallazgos relevantes de la iteración (por qué v4 y no v3)

- La política v3 no tiene evidencia off-policy (adherencia vs rendimiento ρ≈0.02, placebo y permutación nulos, `13`).
- La cal en Fusión: su efecto sobre el Sn extraído (kg) contiene un artefacto contable del trazador (+2.2 t/sd de masa de escoria
  no explicada) y a nivel batch más cal/carga no mejora `f_metal` (dross +, p=0.06) → deja de ser palanca (`18`).
- La duración planificada de Reducción es fija por protocolo → no es palanca (`19`, `20`).
- El PLM iguala/supera al HGB y expone θ con IC (`14`, `20`, `21`); el R² 0.97 de Reducción es una regla física (inventario ×
  fracción por escalón): el valor está en los θ, no en el R² (`18`).
- Selectividad FeO/Sn por avance (`15`) y λ_Fe total sobre metal 0.6 (`19`) → el optimizador reduce el carbón tardío y lo sube
  temprano, y deja de subir el GN.
- Estabilidad: GN y carbón en Fusión cambian de signo entre períodos; cal/carga y GN en Reducción son estables (`17`).
  Reentrenar progresivamente por campaña.

## 8. Positivo / negativo (a mejorar en la siguiente iteración)

**Positivo**
- Un solo tipo de modelo por fase, parsimonioso en la parte interpretable (3-5 palancas), con efectos explícitos e IC; mejor
  lockbox que v3 en todos los targets; roles STATE/CONTROL/PLAN nítidos; objetivo en kg de Sn a metal anclado en evidencia de
  batch; recomendaciones coherentes con la pirometalurgia (carbón selectivo temprano, menos carbón al agotarse el SnOx, GN no
  selectivo, lanza oxidante retiene FeO, carbón proporcional al Sn cargado en Fusión, temperatura acotada).
- Evidencia de valor en tres capas, con la parte off-policy en el hold-out cronológico en la dirección correcta.

**Negativo / abierto**
- El efecto del carbón en Reducción y de todas las palancas químicas de Fusión tiene IC que cruza 0 en el escalón (la dosis sigue
  al estado; identificación débil): el respaldo del carbón viene del nivel de batch.
- La evidencia off-policy es nula en DEV; sólo el lockbox la apoya. Hace falta un piloto A/B por batches.
- El objetivo de Fusión depende de dos coeficientes de batch (p≈0.01-0.03) y de un modelo térmico débil (ΔT lockbox 0.06).
- `w_S` (soporte) domina el tamaño de las recomendaciones; con 2000 kg el prescriptor de Reducción es conservador (uplift
  ≈30 kg/batch). Sensibilidad con `w_S` = 500 (`21_*_ws500.csv`): recomendaciones más agresivas (carbón en Fusión +1.3 sd,
  uplift estimado mediana +1.6 t) pero el soporte de la recomendación cae por debajo del histórico (0.37 vs 0.42: extrapola) y
  la evidencia off-policy no mejora de forma consistente (`dist_F` −0.51 pp/sd p=0.04 en TOTAL; `dist_R` Spearman −0.35 p=0.005
  en lockbox pero OLS n.s.; DEV nulo). Se mantiene `w_S` = 2000 (la recomendación nunca tiene menos soporte que la operación real).
- Pendientes de planta: sentido físico de la posición de lanza; CaO de la ganga (calibra la masa absoluta); precios de reactivos;
  causa del 22.5% de escalones que no cumplen la duración planificada.
- Próximos caminos: (i) piloto controlado; (ii) reentrenamiento progresivo por campaña; (iii) incorporar la selectividad
  marginal empírica `s(avance)` directamente en `J_R`; (iv) modelo térmico de Fusión con balance de energía explícito.

---

## Addendum iteración 6 (2026-09-14, noche): balances inspirados en el modelo teórico (`experimentos/balances/`, hallazgos §17)

- **KPI de batch para evaluar políticas**: `recuperacion_refinada_pct` (M/(M+D+P+Sn en escoria final), función
  `construir_resumen_batch_balance_global` de `feature_engineering.py`) sustituye al proxy en la evidencia off-policy y en la elección de
  targets. Con él, en lockbox replican lnIRF_R (ρ 0.38) e IRF fin de Fusión (0.34), no el SDI (0.09): el objetivo de Reducción debe ser
  lnIRF_R (o z(SDI)+z(lnIRF_R)).
- **Modelo térmico de Reducción**: añadir `b6_q_neto_por_min_MJ`, `b6_dT_teorico_sin_reaccion`, `b6_q_gases_MJ`, `b6_margen_termico_MJ`
  (fórmulas en `experimentos/balances/bal_features.py`) al estado: R² OOF 0.466→0.474, lockbox 0.465→0.501.
- **No cambia**: palancas v4 (el balance reductor en kg C-eq no mejora R² ni identifica el carbón; estrecha el IC del FeO sin cruzar 0),
  trazador CaO (el cierre multi-inerte es más ruidoso), objetivo de Fusión.
- **Feedback al modelo teórico Rust**: cementación total por Fe metálico y 48 % de carbón perdido al tiro no se observan; utilización del
  reductor ≈ 1; SiO₂/Al₂O₃/MgO no sirven como trazadores por escalón.
