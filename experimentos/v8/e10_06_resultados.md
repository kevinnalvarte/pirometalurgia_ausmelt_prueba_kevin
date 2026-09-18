# E10-06 — Fusión con β_F = 0: qué términos del objetivo son defendibles y cómo acotar el recorte de carbón

Script `e10_06_fusion.py` (etapas `dT | familia | recorte | envolvente`; log `e10_06_log.txt`; CSV `e10_06_dT.csv`, `e10_06_dT_theta.csv`,
`e10_06_dT_por_orden.csv`, `e10_06_familia_batch.csv`, `e10_06_recorte.csv`, `e10_06_envolvente.csv`). Datos `v8_lib.cargar_df()`, Fusión F1-F6,
DEV 299 / lockbox 63 batches; OOF GroupKFold(5) por batch, OOF cronológico (5 bloques), walk-forward (bloques de 50), θ libre y acotado
(`theta_dual`, n_boot 500). Familia de batch pre-registrada: 7 términos × 3 KPI (KPI refinado, f_polvo, f_dross).

## 1. Modelo térmico de Fusión (ΔT del horno por escalón)

| variante | n DEV | R² OOF | R² OOF cronológico | R² walk-forward | R² lockbox |
|---|---|---|---|---|---|
| (a) sin F1 (el gradiente previo es NaN en F1) | 1482 | **0.497** | **0.338** | **0.209** | 0.037 |
| (b) F1 imputado a 0 (v7) | 1778 | 0.403 | 0.258 | 0.092 | 0.092 |
| (c) b + indicador `es_F1` | 1778 | 0.403 | 0.258 | 0.092 | 0.092 |
| (d) b sin T de horno previa | 1778 | 0.370 | 0.112 | 0.138 | 0.067 |

Por orden (variante b): F1 R² 0.05 (MAE 14.6 °C), F2-F5 0.42-0.54, F6 0.33. **F1 no es predecible** y diluye el modelo; el indicador
`es_F1` no cambia nada (el HGB ya lo separa por el gradiente imputado). Decisión: **modelar ΔT de Fusión sin F1** (F1 hereda la receta y no
recibe recomendación, dictamen §3), con lo que el OOF sube de 0.40 a 0.50 y el walk-forward de 0.09 a 0.21.

θ libre (1 sd, variante a): GN −0.26 [−1.78, +1.15], O₂ −1.01 [−2.46, +0.52], aire +0.58 [−1.01, +2.09], carbón +0.11 [−1.08, +1.28]:
**ninguna palanca térmica de Fusión está identificada** (el acotado de O₂ "significativo" −1.16 [−2.63, −0.25] es la cota más el control
reactivo: corr(O₂, T_prev) −0.53, corr(GN, T_prev) −0.43; sin T previa en el estado el carbón pasa a +1.65 [+0.02, +2.96], es decir, el
efecto térmico del carbón sólo aparece cuando se deja entrar la confusión). Consecuencia: la ventana térmica de Fusión sólo puede actuar
como restricción de estado (no hay efecto de la acción sobre T que el modelo pueda usar) y el recorte de carbón no tiene un costo térmico
estimable con estos datos: se acota por envolvente (≤ 1 sd del orden, ≥ P10) y con λ de lanza constante.

## 2. Familia de batch (OLS HC3 por +1 sd con `CONTROLES_BATCH`, DEV n 299; réplica en total n 360)

| término de Fusión | KPI refinado (DEV) | f_polvo (DEV / total) | f_dross (DEV / total) |
|---|---|---|---|
| carbón total F [kg] | +0.11 pp, p 0.72 | +0.0053, p 0.010 / ρ 0.04 n.s. | −0.0062, p 0.013 / ρ −0.15, p 0.005 |
| C/Sn cargado | +0.17, p 0.65 | **+0.0064, p 0.017 / ρ 0.17, p 0.001** | **−0.0082, p 0.010 / ρ −0.24, p < 1e-4** |
| T media F | −0.13, p 0.80 | +0.0081, p 0.027 / ρ 0.13, p 0.01 | −0.0077, p 0.037 / n.s. |
| GN total F | −0.17, p 0.64 | n.s. | ρ +0.17, p 0.003 (OLS n.s.) |
| exceso O₂ medio F | −0.20, p 0.56 (lockbox ρ −0.27, p 0.03) | n.s. | n.s. |
| T gas pre-BHF media | −0.24, p 0.44 | n.s. | ρ +0.15, p 0.009 (OLS n.s.) |
| Σ ln_sn_dep F (extraer Sn en F) | −0.27, p 0.34 | −0.0034, p 0.09 | **+0.0073, p 0.003 / ρ 0.23, p < 1e-4** |

Lectura pirometalúrgica: más carbón (y más C/Sn) en Fusión **sube el polvo y baja el dross** con magnitudes casi iguales (+0.006 vs −0.008
por sd) → efecto neto sobre el KPI refinado nulo (p 0.65-0.72). Es el mismo resultado de v5 (E7-02: "dross y polvo se cancelan") y del
dictamen (recorte defendible por costo y polvo, no por conservar Sn). Extraer Sn en Fusión sube el dross (coherente con Fe metalizado
acompañando al Sn cuando la Fusión reduce), pero tampoco mueve el KPI. Criterio 6 del diseño (p < 0.05 y q-BH < 0.10 en DEV **sobre el
KPI** y mismo signo en total): **ningún término lo cumple** → `J_F = −costos − ventana térmica (estado) − soporte`, sin término de polvo
(`pp_kpi_por_kg_carbon_F = 0`).

## 3. Recorte de carbón con λ de lanza constante

λ extendida (O₂ disponible / O₂ demandado por GN + carbón a CO, 0.933 Nm³ O₂/kg C): F1 0.77, F2-F5 0.66-0.69, F6 0.72 (P5-P95 0.61-0.83).
Con el carbón contado, la lanza de Fusión es sub-estequiométrica: el carbón de Fusión es en parte combustible y el exceso de O₂ "nominal"
(que ignora el carbón) sobreestima el carácter oxidante. Recortar 1 sd de carbón (5-8 kg/min) sin tocar O₂ sube λ ext. en ~0.03; con
ΔO₂ = 0.933·ΔC se mantiene. Efecto térmico estimado del recorte de 1 sd: carbón +0.11 °C/(kg/min) [−0.05, +0.24] → −0.6 a −0.9 °C
(n.s.); sobre T de gas pre-BHF +0.04 °C/(kg/min) [−0.01, +0.10] → −0.2 a −0.3 °C (n.s.). Envolvente (DEV): carbón F1 35 ± 5.7, F2 48 ± 7.4,
F3 51 ± 8.0, F4-F6 ~45-52 ± 5-7 kg/min; 10 % de escalones en fin de campaña (espesor < P10 = 238 mm).

## Veredicto E10-06

- ΔT de Fusión: **sin F1** (OOF 0.50 / cron 0.34 / WF 0.21); F1 sin recomendación. Sin palanca térmica identificada → la ventana térmica
  de Fusión es un filtro de estado (no recortar con T previa < P10 del orden ni en fin de campaña), no un término de la acción.
- Objetivo de Fusión: β_F = 0 y sin término de polvo (los canales polvo y dross se cancelan sobre el KPI). Con precios placeholder el único
  motor de recomendación sería el costo → **la política de Fusión v8 es "mantener la receta"** (nivel 0 del dictamen) y el recorte de
  carbón queda como opción económica que sólo se activa con precios reales (`PESOS_V8` de Fusión = 0 por defecto), siempre ≤ 1 sd del
  orden, ≥ P10 y con −O₂ proporcional.
- Lo que sí aporta Fusión al prescriptor: el **estado terminal** (leyes, inventarios y T al cierre de F6) que alimenta el plan de Reducción.
