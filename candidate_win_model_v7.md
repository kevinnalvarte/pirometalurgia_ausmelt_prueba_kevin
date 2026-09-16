# candidate_win_model_v7 — el mejor enfoque predictivo/prescriptivo por escalón (iteración 9, 2026-09-16)

Artefacto: https://claude.ai/artifact/L6GFrWjVcddg2tTa1AkL9F. Ficha del modelo ganador tras la iteración 9 (`experimentos/v7/`, diseño `ITERACION_9_diseno.md`, hallazgos §20). Código:
`modelo_predictivo_v7.py` (capa fina sobre `modelo_predictivo_v6.py`, que a su vez reutiliza v5/v4/v3, `feature_engineering.py`,
`experimentos/v6/masa_v6.py`). Fichas anteriores: `candidate_win_model_v6.md`, `candidate_win_model.md` (v5), `candidate_win_model_v4.md`.

Validación: OOF GroupKFold(5) por batch sobre DEV (299 batches, abril-julio 2026) = criterio de selección; lockbox = últimos 63 batches
cronológicos (agosto 2026, fin de campaña de refractario), sólo reporte; walk-forward y tercios cronológicos como robustez. KPI de batch:
`recuperacion_refinada_pct` = 100·M/(M+D+P+Sn escoria final).

> **Dictamen del comité (2026-09-16, `dictamen_comite_v7.md`)**: el núcleo predictivo de Reducción se rescata; los efectos sobre la retención de FeO
> no están identificados (IC libres cruzan 0), los pesos K/w dependen del régimen, el optimizador produce soluciones de esquina y la evidencia
> off-policy no tiene potencia. No desplegar en línea: aplicar las correcciones de la fase 0 y seguir la hoja de ruta (modo sombra, piloto
> aleatorizado por escalón, piloto de política restringida). Cifras corregidas por la auditoría: techo de ruido de FeO 0.79 (no 0.57, el cálculo
> ignoraba la reversión del ruido de t−1 vía el estado); escalones de Reducción sin ensayo 8 % (R3 29 %), no 3 %; OOF cronológico de FeO 0.296.

---

## 0. Resumen ejecutivo

| | Reducción (`WIN-R v7`) | Fusión (`WIN-F v7`) |
|---|---|---|
| Forma | PLM con signos de teoría: `y = g(S) + Σ θ_j (a_j − E[a_j|S])`, g y E[a|S] por HGB cross-fitted, θ por mínimos cuadrados acotados | idem |
| Targets del escalón (log, adimensionales, telescópicos) | `m6_ln_sn_dep` = ln(Sn_disp/Sn_inv[t]) (agotar Sn), `m6_ln_feo_ret` = ln(FeO_inv[t]/FeO_inv[t−1]) (retener FeO), ΔT | `m6_ln_sn_dep` (conservar Sn), ΔT |
| Masa de escoria | cierre físico del resto no reducible (v6), sin trazador de CaO | idem (F1 ahora modelable) |
| STATE en `g(S)` | **16 features** (`ESTADO_R_V7`, E9-06; v6 usaba 38) | 25 curado v6 (+ F1 con gradiente imputado) |
| CONTROL (θ lineal) | carbón × Sn disponible (`Cx_sn_v6`), GN, exceso O₂, aire (Sn); carbón, carbón × avance, GN, exceso O₂, aire (FeO) | C/Sn cargado, GN, exceso O₂, aire; (ΔT: GN, O₂, aire, carbón) |
| NO son palancas | cal (E9-02: el efecto de E7-02 era artefacto del trazador), duración (protocolo), carga (plan), lanza (convención sin confirmar) | idem |
| R² OOF / lockbox | Sn 0.767 / 0.795 · FeO 0.333 / 0.483 · ΔT 0.464 / 0.351 | Sn 0.190 / −0.70 (sin generalización, E9-04) · ΔT 0.403 / 0.092 (0.49 OOF sin F1) |
| Techo de ruido de ensayo (E9-05) | Sn 0.998 (PLM al 77 %) · FeO **0.571** (PLM al 58 % OOF, 85 % lockbox) | Sn 0.978 (deriva de campaña, no ruido) |
| Objetivo por escalón | `J_R = K·(ln_sn_dep + w·ln_feo_ret) − costos − ventana T − IC − soporte`, **K = 3.52 pp (batch actual + siguiente)**, **w = 7.86 [5.14, 13.04]** | `J_F = −0.58 pp·ln_sn_dep − costos − ventana T − IC − soporte` (débil: p 0.17 DEV / 0.014 total) |
| Agregación | `J_grupo = Σ J_t`; `J_R = K·[ln(Sn_ini/Sn_fin) + w·ln(FeO_fin/FeO_ini)] − Σ costos` (exacto, telescópico); `J_batch = J_R + J_F`; el legado al batch siguiente depende sólo del estado terminal | `agregar_objetivo()` en el módulo |
| Efectos CONTROL (IC95 %, +1 sd, DEV) | **Cx_sn +0.178 [0.041, 0.285]** (agota Sn, selectivo); **GN +0.047 [0.011, 0.086]** sobre Sn y −0.008 [−0.015, 0] sobre FeO (no selectivo); GN +2.9 °C, O₂ −4.2 °C | GN +0.021 [0.001, 0.038] (extrae Sn); O₂ −1.3 °C |
| Política (362 batches cross-fitted) | carbón **+3.0 kg/min** con avance < 0.84 y −1.3 / −2.6 / −1.4 después; GN −0.3..−0.6 Nm³/min; aire +2..+4; O₂ ≈ +0.3 | carbón −4.4 kg/min, GN −0.2, O₂ −0.3, aire −0.1 (menos gas y carbón dentro de la ventana térmica) |
| Uplift modelado (cadena θ → J → KPI) | **+0.27 pp [0.25, 0.29] DEV, +0.35 [0.32, 0.39] lockbox** por batch (mediana; KPI de dos batches), ≈ 137 / 179 kg Sn; > 99 % de batches positivos | incluido (J_F +0.04 pp) |
| Evidencia off-policy (adherencia vs KPI, OLS HC3 + controles) | `dist_R` nula: **potencia 32 %** (MDE 0.59 pp/sd vs efecto esperado 0.32; harían falta ~1 000 batches, E9-08) | `dist_F` −0.49 pp/sd (p 0.13; perm 0.06); **polvo +0.0059/sd (p 0.013; perm 0.001)**; dross total −0.0045 (p 0.038); placebo nulo |

---

## 1. Qué se probó en la iteración 9 y qué sobrevivió

| Hipótesis | Experimento | Veredicto (criterios fijados a priori) |
|---|---|---|
| H1 forma estructural/cinética | E9-01 | Refutada: NLS competitivo 0.34 / 0.13 vs PLM 0.77 / 0.33; parámetros saturan la cota; log-lineal 0.41; híbridos no mejoran |
| H2 cal como palanca | E9-02 | Refutada: cal/carga → KPI p 0.145 con la masa v6; sin dosis-respuesta ni óptimo de basicidad; cal → ΔB2 n.s. |
| H3 política con horizonte | E9-03 | No adoptada: mejora mediana 0 % sobre la miope, robusta en 12 % de réplicas; la secuencia importa (J temprano ρ 0.35 vs tardío 0.09) pero la miope ya la captura |
| H4 Fusión robusta a deriva | E9-04 | A medias: ΔT reentrenable (0.04 → 0.16-0.32); agotamiento no generaliza con ninguna de 8 formulaciones; F1 corregido; conservación de Sn como nivel terminal +1.2 pp/unidad (p 0.14) |
| H5 techo de ruido / bake-off | E9-05 | Confirmada: 14 algoritmos + grid + stacking no superan al PLM (Δ ≤ +0.005, IC cruza 0); curva de aprendizaje saturada desde n ≈ 150; techo FeO 0.57 |
| H6 estado parsimonioso | E9-06 | Confirmada: 16 features conservan el R², identifican el carbón (2/3 tercios; FULL 1/3) y generalizan mejor en walk-forward (+0.007 / +0.028) |
| Legado (Fable, E9-00b) | — | Adoptado: FeO retenido → KPI del batch siguiente +10.4 pp/unidad (p 2e-4), placebo nulo, mediado por el talón de matriz → K 3.52, w 7.86 |

## 2. Fórmulas

```
r_t = 1 − 1.270·s_t − f_t ;  M_t·r_t = M_{t−1}·r_{t−1}·e^{−0.01·[R]} + 0.703·cal_t + 0.301·carga_t     (masa v6, cierre físico)
m6_ln_sn_dep[t]  = ln( (Sn_inv[t−1] + Sn alimentado[t]) / Sn_inv[t] )
m6_ln_feo_ret[t] = ln( FeO_inv[t] / FeO_inv[t−1] )                                   (sólo Reducción)
Cx_sn_v6 = carbón [kg/min] · Sn_inv_prev/1000 ;  Cx_av_v6 = carbón · avance_prev ;  exceso_o2 = 100·(O2 + 0.21·aire − 2·GN)/(2·GN)
J_t(R) = K·(ln_sn_dep_t + w·ln_feo_ret_t) − 0.05·C·dt − 0.03·GN·dt − 0.01·O2·dt − 2·(T fuera de P5-P95) − 0.25·K·ancho_IC − 2000·max(0, s_hist − s)
J_t(F) = −0.58·ln_sn_dep_t − costos − ventana T − IC − soporte           (K en pp de KPI; × 512 kg Sn/pp)
```
Agregación (E9-00): Σ_{t∈R} ln_sn_dep = ln((Sn_ini + Sn alimentado)/Sn_fin) (+0.005 ± 0.018), Σ ln_feo_ret = ln(FeO_fin/FeO_ini) (exacto).
Anclaje por grupo (DEV): ln_sn_dep R0 2.97 / R1 4.21 / R2-R3 1.44 pp/unidad; ln_feo_ret 24.6 / 24.1 / 11.9; w por grupo 8.3 / 5.7 / 8.3 →
un único (K, w): la atenuación tardía es errores-en-variables (SNR 1.2 en R2-R3), no física. Legado: KPI_next ~ Σ ln_feo_ret +10.4
(p 2e-4), placebo KPI_prev +2.1 (p 0.56), mediación por IRF terminal (0.50 × 7.6 pp).

## 3. Estado parsimonioso de Reducción (`ESTADO_R_V7`, 16)

Leyes previas %Sn (fuerza motriz; importancia 0.32), %FeO (competidor), %SiO₂ y %CaO (matriz); `ratio_sn_feo_prev` y `interaccion_Sn_x_FeO_prev`
(selectividad y cinética bimolecular); `termocupla_media_celsius_prev` y `grad_temperatura_horno_celsius_prev` (térmico); `posicion_vertical_lanza_mm_prev`
(agitación); `cum_feed_Carbon_kg_prev`, `cum_aire_nm3_prev`, `relacion_C_cum_Sn_cum_prev` (historia de dosificación); `m6_sn_inv_kg_prev`,
`m6_feo_inv_kg_prev`, `m6_resto_frac_prev` (inventarios del cierre físico); `espesor_ladrillo_norm_mm` (campaña). Sustituye T de horno, orden,
avance, B2 e IRF por proxies correlacionados; la ruta que protegía el núcleo teórico perdía la identificación del carbón en k ≤ 12 (E9-06).
Nuisance lineal (Ridge) colapsa θ → 0: el HGB no es un detalle.

## 4. Métricas (E9-07, `e9_07_tabla_validacion_v7.csv`)

| Fase | Target | Forma | k estado | n DEV / lockbox | R² OOF | MAE OOF | R² lockbox | MAE lockbox |
|---|---|---|---|---|---|---|---|---|
| Reducción | agotamiento Sn [ln] | **PLM v7** | 16 | 1083 / 242 | **0.767** | 0.247 | **0.795** | 0.245 |
| Reducción | agotamiento Sn | HGB ref (16) | 16 | | 0.761 | 0.251 | 0.806 | 0.237 |
| Reducción | agotamiento Sn | PLM v5 ref (37, trazador CaO) | 37 | 1015 / 242 | 0.766 | 0.244 | 0.802 | 0.228 |
| Reducción | retención FeO [ln] | **PLM v7** | 16 | 1083 / 242 | **0.333** | 0.047 | **0.483** | 0.042 |
| Reducción | retención FeO | HGB ref (16) | 16 | | 0.322 | 0.047 | 0.433 | 0.044 |
| Reducción | retención FeO | PLM v5 ref | 37 | | 0.169 | 0.064 | 0.194 | 0.054 |
| Reducción | ΔT [°C] | PLM v7 | 6 | 1182 / 252 | 0.464 | 9.1 | 0.351 | 7.1 |
| Fusión | agotamiento Sn [ln] | PLM v7 (con F1) | 25 | 1764 / 375 | 0.190 | 0.136 | −0.695 | 0.158 |
| Fusión | ΔT [°C] | PLM v7 (con F1) | 8 | 1778 / 376 | 0.403 (0.49 sin F1) | 9.9 | 0.092 | 7.5 |

Techo de ruido (E9-05, ruido de ensayo 2 % %Sn / 2.8 % %FeO): Sn_dep 0.998, FeO_ret 0.571 (0.78 con 2 %, 0.13 con 4 % → el ruido real de
%FeO es ≤ 2.8 %), Fusión 0.978. Curva de aprendizaje saturada (asíntota +0.013 / −0.005). Bake-off: stacking +0.005 [−0.003, +0.014] en Sn;
nadie supera al PLM en FeO; en Fusión el stacking gana en OOF (+0.031) y pierde en lockbox (−0.77). Estabilidad: R² por tercios 0.72/0.79/0.78
(Sn) y 0.40/0.30/0.28 (FeO); walk-forward 0.761 / 0.267 (E9-06). Reentrenamiento progresivo recomendado en producción (ΔT de Fusión 0.04 → 0.16;
FeO de Reducción 0.12 → 0.29 en v5).

## 5. Capa prescriptiva y evidencia

Política cross-fitted (cada batch recomendado por un sistema que no lo vio; `e9_07_recomendaciones_v7_escalones.csv`):

| Reducción, avance previo | Δ carbón [kg/min] | Δ GN | Δ O₂ | Δ aire | uplift J [kg/escalón] |
|---|---|---|---|---|---|
| < 0.84 | **+3.0** | −0.3 | +0.3 | +1.9 | 37 |
| 0.84-0.96 | −1.3 | −0.4 | +0.3 | +4.1 | 40 |
| 0.96-0.975 | −2.6 | −0.6 | +0.1 | +3.1 | 40 |
| > 0.975 | −1.4 | −0.5 | +0.6 | +2.5 | 34 |

Fusión (F1-F6): carbón −4.4 kg/min, GN −0.2, O₂ −0.3, aire −0.1; uplift J 14 kg/escalón. v7 coincide con v6 en Fusión (r 0.99) y en la
dirección de Reducción (77-83 % mismo signo), pero recorta menos carbón tardío (−1.3..−2.6 vs −4.9..−5.6) por el estado parsimonioso y el
mayor w.

Evidencia (E9-07, `e9_07_evidencia_politica_v7.csv`, `e9_07_por_batch_v7.csv`):
1. **Cadena θ → J → KPI**: uplift por batch (mediana, pp de KPI de dos batches) +0.27 [0.25, 0.29] DEV y +0.35 [0.32, 0.39] lockbox;
   media +0.35 / +0.38; > 99 % de batches positivos; por grupo, R temprano +0.15 pp, R tardío +0.16 pp, Fusión +0.04 pp.
2. **Off-policy**: `dist_F` −0.49 pp/sd (p 0.13, permutación 0.06) sobre el KPI, **f_polvo +0.0059/sd (p 0.013, perm 0.001 DEV; p 0.004
   total)**, f_dross −0.0045/sd (p 0.038 total); `dist_R` nula; placebo con política permutada nulo.
3. **Potencia (E9-08)**: para `dist_R` la se HC3 es 0.21 pp/sd → MDE al 80 % 0.59 pp/sd frente a un efecto esperado de 0.32 (pendiente del
   uplift predicho sobre la distancia): potencia 32 %, se necesitarían ~1 000 batches. El nulo de Reducción es el resultado esperado del
   diseño observacional, no evidencia de ausencia de efecto → piloto A/B.

Sensibilidad al legado (variante `activar_v7(con_legado=False)`, w 6.09 / K 2.89, misma política cross-fitted): Fusión idéntica (100 % mismo
signo); Reducción 88-90 % mismo signo, con un recorte de GN menor (−0.08 vs −0.45 Nm³/min); uplift modelado +0.17 / +0.22 pp (un batch) frente a
+0.27 / +0.35 (dos batches); evidencia off-policy idéntica en Fusión (dist_F −0.49, polvo +0.0059).

## 6. Por qué no hay un enfoque mejor con estos datos (síntesis)

- Forma: PLM ≥ 14 algoritmos, grid, stacking y forma estructural (E9-01, E9-05). Estado: 16 features al mismo R² que 38, mejor
  identificación y walk-forward (E9-06). Target: el agotamiento de Sn está al 77 % de un techo ≈ 1 con curva saturada; la retención de FeO al
  58-85 % de un techo de 0.57 impuesto por el ruido de %FeO (E9-05). Objetivo: aditivo y exacto por identidad telescópica, anclado en el KPI
  con legado (E9-00). Política: la miope captura lo que el horizonte podría (E9-03). Palancas: carbón (selectivo), GN (no selectivo), aire/O₂
  térmicos; cal descartada con la masa corregida (E9-02).
- Lo que queda fuera del alcance de los datos: predecir el agotamiento de Fusión fuera de campaña (deriva), medir el efecto de Reducción
  sobre el KPI sin un piloto (potencia), y bajar el ruido de %FeO (laboratorio).

## 7. Cómo reproducir

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v6/_cache_df_v6.py          # caché df_v6.pkl
    .venv/Scripts/python.exe experimentos/v7/e9_07_validacion_v7.py validacion                                 # tablas, θ, predicciones, curvas
    .venv/Scripts/python.exe experimentos/v7/e9_07_validacion_v7.py 0..4|lockbox                              # política cross-fitted (Start-Process, ~4 min c/u)
    .venv/Scripts/python.exe experimentos/v7/e9_07_validacion_v7.py evidencia                                  # adherencia, placebo, agregación
    V7_SIN_LEGADO=1 ... (mismas etapas)                                                                         # variante w 6.09 / K 2.89
    .venv/Scripts/python.exe experimentos/v7/e9_08_potencia_fable.py

## 8. Pendientes (fuera del alcance de los datos actuales)

Piloto A/B por batches (GN de Reducción y carbón de Fusión/Reducción temprana); reentrenamiento progresivo cada 10 batches; análisis de
CaO/humedad de la cal y ganga (pG, gG); precisión analítica de %FeO; pesaje de polvo por batch; convención de la posición de lanza.
