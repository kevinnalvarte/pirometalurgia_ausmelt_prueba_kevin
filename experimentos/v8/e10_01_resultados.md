# E10-01 — Forma funcional del carbón en Reducción y óptimo interior

Script `e10_01_formas_carbon.py` (etapas 1-4; log `e10_01_log.txt`; CSV `e10_01_formas.csv`, `e10_01_theta.csv`, `e10_01_tercios.csv`,
`e10_01_dosis_respuesta.csv`, `e10_01_pd_carbon.csv`, `e10_01_optimo_interior.csv`, `e10_01_curvas_J.csv`). Reducción, estado k15 (v7 sin
`espesor`), DEV n 1083 / lockbox 242 (mismas filas en todas las formas); OOF GroupKFold(5), OOF cronológico, walk-forward, lockbox; θ libre y
acotado con IC bootstrap por batch (n_boot 500). Las palancas de lanza (GN, exceso O₂, aire) fijas en su signo de teoría en todas las formas.

## 1. Formas del carbón sobre el agotamiento de Sn (`m6_ln_sn_dep`)

| forma | término de carbón | R² OOF | cron. | WF | lockbox | θ libre 1 sd [IC95 %] | identificado (libre) |
|---|---|---|---|---|---|---|---|
| F0 (v7) | `Cx_sn_v6` = C · Sn_inv_prev (bimolecular) | **0.767** | 0.775 | 0.771 | 0.800 | **+0.179 [+0.033, +0.292]** | **sí** |
| F1 | `m8_dosis` = C_kg / Sn_disp | 0.765 | 0.773 | 0.769 | 0.800 | +0.020 [−0.009, +0.051] | no |
| F2 | `ln(1 + dosis)` (saturante) | 0.765 | 0.773 | 0.769 | 0.799 | +0.023 [−0.006, +0.056] | no |
| F3 | dosis + dosis² (cóncava, θ₂ ≤ 0) | 0.763 | 0.773 | 0.771 | 0.798 | +0.057 [+0.013, +0.129]; dosis² −0.034 [−0.130, +0.006] | lineal sí, curvatura no |
| F4 | ln1p_dosis + ln1p_dosis × avance | 0.765 | 0.771 | 0.769 | 0.800 | +0.142 [−0.073, +0.354]; ×avance −0.123 [−0.340, +0.092] | no (colineales) |
| F5 | carbón primitivo (kg/min) | 0.766 | 0.773 | 0.770 | 0.800 | +0.057 [−0.005, +0.116] | no (borde) |
| F6 | ln1p_dosis + C × zT + C × zB2 (θ heterogéneo) | 0.765 | 0.767 | 0.763 | 0.799 | C×zT −0.003 [−0.033, +0.028]; C×zB2 −0.009 [−0.116, +0.022] | no |
| F7 | ln1p_dosis + GN² | 0.765 | 0.773 | 0.769 | 0.799 | GN² −0.001 [−0.025, +0.023] | no |

GN queda identificado en todas las formas (+0.048 a +0.056, IC libre [+0.008, +0.097]); exceso de O₂ y aire nunca.

Lectura: **la predicción no distingue la forma** (ΔR² ≤ 0.004; la señal está en g(S)); la **identificación sí**: el único término de carbón cuyo
IC libre excluye 0 es `Cx_sn_v6`, es decir, el efecto marginal del carbón es proporcional al Sn disponible (a más Sn en la escoria, más
rinde cada kg de carbón: cinética bimolecular SnO₂ + C limitada por el Sn en R2-R3). La curvatura de F3 tiene el signo de la teoría
(rendimientos decrecientes) pero no excluye 0; la heterogeneidad con T y basicidad es nula; GN no tiene óptimo interior detectable.
Por tercios cronológicos (`e10_01_tercios.csv`) `Cx_sn` sólo se identifica con IC libre en el tercio reciente (+0.28 [+0.02, +0.57]; tercio
medio +0.18 [−0.05, +0.34]; tercio antiguo ≈ 0): el efecto crece con la campaña (o la varianza identificadora), como ya vio E9-06.

## 2. Formas del carbón sobre la retención de FeO (`m6_ln_feo_ret`)

| forma | términos de carbón | R² OOF | cron. | WF | lockbox | θ libre carbón | GN libre |
|---|---|---|---|---|---|---|---|
| G0 (v7) | carbón, Cx_av | 0.325 | 0.304 | 0.255 | 0.469 | −0.015 [−0.062, +0.033]; +0.013 [−0.028, +0.050] | **−0.010 [−0.020, −0.0025]** |
| G1 | `m8_exceso_C_pos` (sobrante sobre el estequiométrico de Sn) | 0.326 | 0.303 | 0.255 | 0.469 | +0.0006 [−0.005, +0.007] | −0.010 [−0.020, −0.002] |
| G2 | exceso_C_pos + Cx_av | 0.325 | 0.304 | 0.255 | 0.469 | ≈ 0 ambos | −0.010 [−0.020, −0.003] |
| G3 | ln1p_dosis | 0.326 | 0.303 | 0.255 | 0.469 | −0.001 [−0.008, +0.004] | −0.009 [−0.020, −0.001] |
| G4 | exceso_C_pos + C × zT + C × zB2 | 0.310 | 0.290 | 0.234 | 0.471 | ≈ 0 | −0.010 |

**Ninguna forma del carbón sobre la retención de FeO está identificada** (IC libres centrados en 0, 54-85 % de réplicas en la cota); el
efecto del **GN sí** (−0.010/sd, IC libre que excluye 0 en todas las formas — con el estado k15 sin `espesor`; con k16 el comité lo vio en
[−0.0174, +0.0002]). Exceso de O₂ +0.0025 [−0.0015, +0.006] y aire +0.002 [−0.002, +0.007]: no identificados.

## 3. Dosis-respuesta no paramétrica y PD

Residuo del carbón (a − E[a|S]) por deciles contra el residuo del target (`e10_01_dosis_respuesta.csv`): Spearman +0.087 (p 0.004) para el
carbón primitivo y +0.101 (p 0.0008) para ln1p_dosis; el decil más bajo (−13.7 kg/min respecto de lo esperado por el estado) tiene
−0.072 [−0.132, −0.008] de agotamiento y el más alto (+13 kg/min) +0.036 [−0.010, +0.077]: **monótona creciente, débil y asimétrica**
(quitar carbón cuesta más de lo que añadirlo aporta: saturación). PD del HGB directo por orden (`e10_01_pd_carbon.csv`): en R0 el
agotamiento sube de 1.57 a 1.73 entre 5 y 65 kg/min (+10 %, casi todo por encima de 55); en R1-R3 sube hasta ~15-20 kg/min y luego es
plano. Coherente con la teoría (reductor limitante sólo a dosis bajas; luego limita el Sn) y con el θ pequeño.

## 4. Óptimo interior de J_R(C) (K 2.89, w 6.09; 12 estados representativos por orden y avance)

**0 % de estados con óptimo interior en todas las combinaciones de formas** (`e10_01_optimo_interior.csv`): con F0 el óptimo está en la cota
superior de carbón en R0-R1 y con las formas de dosis no identificadas (θ ≈ 0) el óptimo está en la cota inferior (sólo costo). Sin un
efecto identificado del carbón sobre FeO no existe el intercambio que produciría el óptimo interior; la curvatura de F3 no está identificada.

## Veredicto (criterios 1-3 del diseño)

- Forma del carbón sobre Sn: **se mantiene `Cx_sn_v6`** (única identificada con IC libre; R² idéntico); la dosis saturante se **rechaza** (no
  identificada) y la curvatura queda como "signo de teoría, no identificada" (H1 refutada en su parte de identificación).
- Forma sobre FeO: se conserva la parametrización teórica (exceso_C_pos + Cx_av) para reportarla, **declarada NO identificada** → θ de uso 0;
  el GN es la única palanca identificada sobre FeO (con k15).
- Consecuencia para el prescriptor: J es lineal en el carbón dentro del rango histórico → la recomendación de carbón es una **dirección con
  paso acotado** (|Δ| ≤ 1 sd del orden, techo P90 en R0), no un óptimo: +carbón donde el Sn disponible lo paga (R0-R1) y −carbón donde ya no
  (R2-R3, por costo). Esto es exactamente la política restringida de la fase 0 del dictamen y el piloto por escalón es lo que puede
  identificar la curvatura y el canal FeO del carbón.
