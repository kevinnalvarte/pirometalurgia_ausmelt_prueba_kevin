# candidate_win_model_v9 — asesor prescriptivo por escalón v9 (iteración 11, 2026-09-17)

Ficha del candidato ganador tras la iteración 11 (`experimentos/v9/`: diseño `ITERACION_11_diseno.md`, memos `e11_0N_resultados.md`, librería
`v9_lib.py`; hallazgos §23). Código: `modelo_predictivo_v9.py` (sobre `modelo_predictivo_v8.py`, que se conserva para los modelos de escalón).
Fichas anteriores: `candidate_win_model_v8.md` … `candidate_win_model_v4.md`, `dictamen_comite_v7.md`.

## 0. Resumen ejecutivo

| | Reducción (`WIN-R v9`) | Fusión (`WIN-F v9`) |
|---|---|---|
| Qué es | **Asesor adaptativo**: valora cada CONTROL por su vínculo DIRECTO y fuera de muestra con el KPI del batch, re-anclado con los últimos 100 batches; recomienda un setpoint relativo a la **práctica habitual dado el estado** E[a\|S] | receta (sin palanca que pase los criterios: E10-06, E11-05) |
| STATE (17) | k15 de v8 (leyes previas Sn/FeO/SiO₂/CaO, Sn/FeO, Sn×FeO, termocupla, gradiente T, lanza, acumulados C/aire/C:Sn, inventarios Sn/FeO, resto) + orden del escalón + T horno previa | — |
| CONTROL | GN (Nm³/min) y carbón (kg/min); O₂ y aire siguen al GN con la λ de lanza habitual (congelados como índice) | — |
| Exposición | z_jt = (a_jt − Ê[a_j\|S_t]) / sd_res(j, orden); x_bj = media_t z_jt sobre R0-R3 (agregación escalón → fase) | — |
| Valor | KPI_b = controles + β_G·x_bG + β_C·x_bC + ε, OLS HC3 sobre los últimos **W = 100** batches; β en pp de KPI por unidad de z medio | — |
| Recomendación | a*_jt = Ê[a_j\|S_t] + δ·signo(β̂_j)·sd_res, δ = min(1, \|t\|/2) sd; sólo si \|t\| ≥ 1; caja [P5, P95] del orden, carbón R0 ≤ P90; **no recortar GN con T_prev < P10**; si no → "mantener la práctica habitual" | mantener receta |
| Explicación física por escalón | θ de escalón (PLM v8): Δln_sn_dep y Δln_feo_ret esperados de la recomendación; suman telescópicamente a fase y batch | — |
| Target de escalón (agregable) | se conservan `m6_ln_sn_dep` y `m8_ln_feo_ret_dross50` (Σ_R exacta) como targets predictivos/explicativos; el **valor** se mide en pp de KPI del batch | ΔT (restricción) |
| KPI | `recuperacion_refinada_pct` (principal; E11-04 no encontró uno mejor) + **KPI secundarios por canal**: `f_dross` para GN (+53 % de potencia), Sn perdido en escoria para carbón (+220 %) | idem |

## 1. Qué se probó en la iteración 11

| Hipótesis | Exp. | Veredicto |
|---|---|---|
| H1 vínculo directo control → KPI | E11-01 | **GN_R → KPI aceptado**: −1.50 pp por unidad de z medio (p 0.0014, q-BH 0.008), concentrado en R2-R3 (−1.10, p 0.004); canal dross +1.13 pp (p 0.004), metal −1.06; 3/3 tercios; bootstrap [−2.40, −0.65]; robusto al estado de E[a\|S] (pobre/rico p ≤ 0.006); placebos nulos; permutación por bloques p 0.016; **RV_q1 17 %** (control observado más fuerte 0.95 %); dosis-respuesta monótona (Q5−Q1 −1.98 pp). **Carbón_R → KPI: no robusto** (p 0.15); su canal sí: Sn perdido en escoria −0.27 pp (R0-R1, p 0.011) / −0.16 (R2-R3, p 0.032) |
| H2 valor del reductor moderado por el Sn disponible | E11-02 | **Rechazada como rampa continua** (interacción p ≥ 0.14, umbral no identificado, no replica en lockbox); sobrevive un patrón por tramos (GN con poco Sn −2.05, p 0.04; con mucho Sn −0.50 n.s.), colineal con el orden (r −0.91) |
| H3 objetivo anclado por la cadena θ×residuo (log / kg, tipo 2SLS) | E11-03 | **Rechazada**: K_Sn y K_FeO sin signo estable (cambian entre DEV y total); sobre-identificación: el carbón tiene un efecto directo que los dos targets de escoria no agotan (F 6.57, p 0.038 total); sólo los β directos sobre residuos crudos conservan el signo |
| H4 KPI más preciso | E11-04 | **No se cambia**: sin polvo es ciego al carbón de Fusión; suavizados fallan el placebo/HAC; media de 2 batches destruye la señal. La autocorrelación a lag 3 afecta a metal, dross, polvo y al cierre de Sn (sistema de pesaje), sin compensación entre batches vecinos. Se adoptan KPI secundarios por canal |
| H5 Fusión | E11-05 | **Receta**: `exo2_F` → KPI −1.0 pp (p 0.067, q-BH 0.25 en DEV; IC bootstrap incluye 0; sin mecanismo de escalón; no aporta OOF); carbón F: F1-F3 → polvo ↑ (p 0.013), F4-F6 → dross ↓ (p 0.0004), neto 0 |
| Deriva de régimen | E11-06b/c | El β de DEV aplicado al lockbox **falla** (pendiente −0.9..−4.8): β_GN deriva de ≈0 (1130 °C) → −3 (1060-1100 °C, vía polvo ↑/metal ↓) → −1 (1000-1050 °C, vía dross ↑) → +1 (985 °C, fin de campaña); β_C de −1 → +3..+4; GN → dross es + en 14/15 ventanas |
| Moderación térmica en el escalón | E11-08 | **No identificada** (ninguna interacción θ×T replica DEV+total; ~160 pruebas); a nivel batch sólo xG×T sobre dross (p 0.044 / 0.027, T gana a tiempo). Sin umbral térmico duro |
| Re-anclaje adaptativo prospectivo | E11-06d, E11-07, E11-09 | **Adoptado** (§3) |

## 2. Fórmulas

```
Ê[a_j|S]       = HGB(S) entrenado con todo el pasado (residuos del pasado por GroupKFold por batch)
z_jt           = (a_jt − Ê[a_j|S_t]) / sd_res(j, orden)              x_bj = media_{t∈R0..R3} z_jt
β̂ (ventana)    : KPI_b ~ 1 + x_bG + x_bC + feed_sn_total_t + ley_sn_conc + frac_carga_secundaria_F + feed_dross_Fe_total_t + idx   (últimos 100 batches, HC3)
a*_jt          = clip( Ê[a_j|S_t] + min(1, |t_j|/2)·signo(β̂_j)·sd_res(j, orden)·1[|t_j| ≥ 1] , caja_orden )   ;  GN: sin recorte si T_prev < P10
valor_b(op)    = Σ_j β̂_j · x_bj        (pp de KPI que el asesor atribuye a lo que el operador hizo; se calcula con β̂ del PASADO)
valor_b(rec)   = Σ_j β̂_j · media_t (a*_jt − Ê)/sd_res
efecto físico  : Δln_sn_dep_t = θ_Sn·(a* − Ê),  Δln_feo_ret_t = θ_FeO·(a* − Ê)  →  Σ_t = efecto de fase (telescópico, E9-00)
```

## 3. Evidencia: aplicar la recomendación ↔ rendimiento del batch (requisito obligatorio)

Simulación **prospectiva** (`replay_prospectivo`): cada bloque de 10 batches se recomienda escalón a escalón con un asesor entrenado sólo con
los batches anteriores (≥ 100); 262 batches puntuados (199 DEV + 63 lockbox), 1 366 escalones.

| Prueba (KPI ~ valor_b(op) + controles) | Resultado |
|---|---|
| Replay completo v9 (`e11_09_evidencia_prospectiva.csv`) | pendiente **+0.44** (t 1.99, p 0.023); Spearman parcial **0.165 (p 0.008)**; permutación en bloques de 20: p 0.043; terciles del puntaje: KPI residual **−1.26 / +0.38 / +0.89 pp** (monótono) |
| Sólo DEV prospectivo | +0.48 (p 0.020); Spearman 0.202 (p 0.004); perm. 0.026; terciles −1.70 / +0.72 / +1.00 |
| Lockbox (fin de campaña) | −0.26 (n.s.): el asesor ya había llevado β_GN a ≈ 0 (0 % de recomendaciones de GN) → se abstiene; **no** repite el fallo del β estático (−0.9) |
| Nulas que respetan adaptatividad (E11-07; W 100) | permutación por bloques p 0.0065; **corregida por elegir entre 6 ventanas p 0.019-0.027**; desplazamiento circular p 0.015-0.018 |
| Magnitud (bootstrap por bloques) | pendiente 0.53 [0.21, 0.96]; tercio alto − bajo **+1.45 pp [0.56, 2.61] ≈ 741 kg Sn/batch [284, 1 336]** |
| Canales | Sn perdido en escoria t 2.68 (p 0.004), dross t 2.39 (p 0.009), polvo p 0.10; KPI reconstruido desde canales t 2.38 |
| Robustez del re-anclaje | 6/7 variantes (semivida 40-80, ridge, contracción, sólo GN, + Fusión) p < 0.05 |
| CV anidada dentro del régimen (DEV, β estático, 2 parámetros) | pendiente 0.79-0.87, p perm. **0.005-0.008**; terciles −0.85 / +0.77 |
| Placebos | lead 3: nulo; lead 1 y KPI anterior: atenuados a la mitad (p ≈ 0.10): hay persistencia de corto alcance (≤ 2 batches) que los controles no absorben |

Coherencia pirometalúrgica de la cadena: GN por encima de lo habitual → menor retención de FeO en el escalón (θ −0.012, p 1e-4; E10-05) →
más dross en el batch (+1.1 pp, p 0.004) → menos KPI; carbón por encima de lo habitual → más agotamiento de Sn en el escalón (C×Sn +0.18,
IC libre) → menos Sn perdido en la escoria final (p 0.01-0.03). La deriva con el régimen térmico (fuming a alta T; cinética/viscosidad a baja
T) es físicamente plausible pero no identificable en una sola campaña → se trata con re-anclaje, no con un umbral.

## 4. Política resultante en el replay (`e11_09b_resumen_politica.csv`)

| Período | GN (Δ medio vs habitual, Nm³/min; % escalones con recomendación) | Carbón (Δ medio kg/min; %) |
|---|---|---|
| batches 100-199 (1 060-1 100 °C) | R0 −2.1 (78 %), R1 −1.8 (73 %), R2 −1.2 (54 %), R3 −0.7 (40 %) | ≈ 0 (18 %) |
| batches 200-299 (≈ 1 000 °C) | −0.6 … −0.3 (39-21 %) | **+1.9 / +2.0 / +1.9 / +1.2** (40 %) |
| 300-361 (lockbox, ≈ 980 °C) | 0 (0 %: β≈0 + horno frío) | +1.8 / +1.5 / +1.3 / +0.9 (42 %) |

Pasos ≤ 1 sd del residuo (GN ≤ 3.3 Nm³/min; carbón ≤ 7.3 kg/min), nunca fuera de [P5, P95] del orden. Motivos de "mantener" del GN: β no
distinguible de 0 (31 %), horno frío (32 %). Efecto físico esperado por batch: Σ Δln_feo_ret +0.008, Σ Δln_sn_dep −0.041 (el recorte de GN
cede algo de agotamiento a cambio de retener Fe: el trade-off queda explícito). Ganancia modelada +1.08 pp/batch; **con la calibración
observada (pendiente 0.44-0.53) la expectativa honesta es ≈ +0.5 pp/batch (≈ 250 kg Sn)**, a confirmar en piloto.

## 5. Métricas predictivas de los modelos de escalón (sin cambios respecto de v8, §4 de su ficha)

Reducción: agotamiento Sn R² OOF 0.767 / cron. 0.775 / WF 0.771 / lockbox 0.800; retención FeO neta 0.438 / 0.416 / 0.333 / 0.566; ΔT
0.44 / 0.26 / −0.02 / 0.42. Fusión ΔT (F2-F6) 0.48 / 0.36 / 0.17 / 0.05. Política de comportamiento Ê[a\|S] (R² OOF): carbón 0.84 (R0-R1) /
0.29 (R2-R3); GN 0.32 / 0.15.

## 6. Positivo / negativo

**Positivo.** Primera evidencia directa, prospectiva y estadísticamente significativa (p ≈ 0.02 corregida; Spearman p 0.008) de que operar en
la dirección recomendada se asocia a mejor KPI en batches no vistos, con canal físico coherente (escoria, dross) y magnitud relevante
(+1.45 pp entre terciles). Sólo 2 parámetros de valor, re-estimables cada 10 batches; se abstiene cuando el régimen cambia (lockbox). La
recomendación ya no depende de lo que el operador hizo (setpoint relativo a la práctica habitual). Vínculo GN → dross con RV 17 %.

**Negativo / abierto.** (1) Sigue siendo observacional: placebos de corto alcance no limpios (persistencia ≤ 2 batches); pendiente 0.44-0.53 < 1
(sobre-atribución ×2). (2) En fin de campaña el asesor sólo se abstiene; no demuestra ganancia allí. (3) La causa de la deriva (T vs desgaste vs
mineral) no es identificable en una campaña. (4) El carbón sólo entra cuando su β de ventana supera |t| 1: es la palanca de menor confianza
(canal escoria sí, KPI neto no). (5) δ ≤ 1 sd es una cota de soporte, no un óptimo: la curvatura no se identifica. (6) Fusión sin palanca.
Siguiente paso: modo sombra con el asesor re-anclado cada 10 batches y piloto aleatorizado por escalón (GN R2-R3 ±1 sd; carbón R0-R1 +1 sd),
con `f_dross` y Sn en escoria como endpoints de mayor potencia.

## 7. Reproducir

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v9/v9_lib.py            # residuos y piloto
    .venv/Scripts/python.exe experimentos/v9/e11_06b_specs_politica.py ; e11_06c_walkforward.py ; e11_06d_ventana_movil.py
    .venv/Scripts/python.exe modelo_predictivo_v9.py                                                       # replay prospectivo + evidencia (~2 min)
    .venv/Scripts/python.exe experimentos/v9/e11_09b_resumen_politica.py
