# E11-01 — robustez del vínculo directo control → KPI del batch (H1)

DEV n=298-299 (1 batch se pierde por NaN en algún control), lockbox n=63. `x_R_*` = Reducción agrupada R0-R3 (spec
parsimonioso); `x_Rt_*`/`x_Rl_*` = R0-R1 / R2-R3 (spec completo). Familia BH = 6 exposiciones (spec6) o 9 (spec9) × KPI,
dentro de (spec, conjunto). Scripts: `e11_01_common.py`, `e11_01_principal.py` (tareas 1/2/4), `e11_01_estado.py` (3),
`e11_01_placebos.py` (5), `e11_01_sensemakr.py` (6), `e11_01_dosis.py` (7).

## 1-2. Regresión conjunta + estabilidad (KPI ~ x_F/x_R_{carbon,gn,exo2} + controles, DEV, spec6)

| exposición | coef (pp/sd) | p | q_BH | signo 3 tercios | bootstrap CI95 (batch) |
|---|---|---|---|---|---|
| **x_R_gn** | **-1.50** | **0.0014** | **0.008** | −/−/− (3/3) | [-2.40, -0.65] excluye 0 |
| x_R_carbon | +1.09 | 0.147 | 0.293 | −/+/+ (2/3) | [-0.15, 2.63] incluye 0 |
| x_F_carbon | -0.54 | 0.325 | 0.487 | −/−/− (3/3, débil) | [-1.64, 0.43] incluye 0 |
| x_F_exo2 | -0.91 | 0.099 | 0.293 | −/−/− (3/3, débil) | [-2.10, 0.18] incluye 0 |
| x_F_gn, x_R_exo2 | n.s. | >0.4 | >0.4 | — | — |

spec9 (Rt=R0-R1, Rl=R2-R3): el vínculo de GN se concentra en **Rl** (coef -1.10, p=0.004, q=0.035); Rt es nulo
(-0.29, p=0.60). Carbón: Rt -0.27pp (p=.011) y Rl -0.16pp (p=.032) sobre `sn_perdido_escoria_frac` (canal, ↓ = bien),
ambos significativos, pero el neto sobre KPI no lo es en ninguno de los dos (p=.39 / p=.081).

Canales (DEV, spec6): GN_R → f_metal -1.06pp (p=.0065), f_dross **+1.13pp (p=.0036)** — coherente con el mecanismo
(reductor no selectivo → hardhead). Carbón_R → sn_perdido_escoria_frac **-0.36pp (p=.020)** — coherente, pero no
sobrevive el paso a KPI neto. Carbón_F → f_polvo **+1.03pp (p=.007)**, f_dross -0.52 (n.s.): neto ≈0 (confirma E10-06).

Lockbox (spec6, KPI): todo n.s.; **GN_R cambia de signo** (+1.52, p=.16) — coherente con H2 (moderación por Sn
disponible), no contradice DEV. Total (pool): x_F_exo2 se vuelve "significativo" (-1.31, p=.002, q=.015) sólo al
agrupar con la tendencia de campaña — no replica en DEV aislado (ver abajo, sensibilidad al estado).

## 3. Sensibilidad al estado de la política (KPI~spec6, DEV)

| variante estado | n | x_R_gn (p) | x_R_carbon (p) | x_F_carbon (p) | x_F_exo2 (p) |
|---|---|---|---|---|---|
| (a) librería (17R/16F) | 298 | -1.50 (.0014) | +1.09 (.147) | -0.54 (.325) | -0.91 (.099) |
| (b) pobre (4/4: orden+leyes+T) | 298 | -1.45 (.0023) | +1.25 (.066) | -0.70 (.273) | -0.32 (.515) |
| (c) rica (a +5: cum GN/O2, tiempo_fase, B2, avance) | 298 | -1.33 (.0055) | +0.85 (.278) | -0.60 (.296) | -0.94 (.098) |

GN_R es robusto a un estado más pobre o más rico (siempre p≤0.006, mismo signo, magnitud -1.3..-1.5). Carbón_R nunca
cruza p<0.05 y su magnitud oscila 0.85-1.25. exo2_F se debilita con estado pobre (p=.51) — su "significancia" en el
pool `total` no es robusta al condicionamiento.

## 4. Ponderación por duración vs exposición física acumulada (DEV, |t| aprox.)

| forma | x_R_gn | x_R_carbon | x_F_carbon | x_F_exo2 |
|---|---|---|---|---|
| z media (base) | t=-3.20 (p=.0014) | t=1.45 (p=.147) | t=-0.98 (p=.325) | t=-1.65 (p=.099) |
| z ponderada por duración | t=-2.68 (p=.007) | t=0.90 (p=.369) | t=-0.59 (p=.557) | t=-1.81 (p=.071) |
| física acumulada (kg/Nm³ de más) | Rl: t=-2.58 (p=.010) | Rt/Rl n.s. (p=.59/.20) | t=-0.66 (p=.510) | t=-0.96 (p=.335, sobre xf_F_exo2) |

La forma con más señal para GN_R es **z media simple** (spec base); ponderar por duración y pasar a unidades físicas
no mejora el t, pero SÍ confirma que la señal física vive en Rl (kg Nm³ de GN de más en R2-R3). Para carbón_R ninguna
forma alcanza significancia — no es un problema de escala/unidades, es falta de señal neta sobre KPI.

## 5. Placebos

(i) Exposición del batch **siguiente** → KPI actual (controlando la actual, DEV): x_R_gn_next coef -0.69 (p=.075,
no significativo, algo de eco); x_R_carbon_next +1.12 (p=.131, n.s.); x_F_carbon_next +0.27 (p=.71); x_F_exo2_next
-0.59 (p=.28). Todos **no significativos** → pasa, con la salvedad de que x_R_gn queda cerca del umbral (vigilar).

(ii) Exposición actual → **pre-tratamiento** (ley_sn_conc_batch_pct, feed_sn_total_t, F_lnirf_F0, KPI_prev), DEV:
x_R_gn y x_R_carbon: todos los p > 0.11 (rango .12-.94) → **pasa limpio**. Alerta: **x_F_carbon → ley_sn_conc_batch_pct
coef 0.50 (p=.0020)** — el carbón de Fusión SÍ se correlaciona con la ley de la carga (posible ajuste del operador
a la concentración, o confusión); no afecta el veredicto de H1 (x_F_carbon no es uno de los dos vínculos centrales)
pero se marca como caveat para H5.

(iii) Permutación en bloques cronológicos de 30 batches (500 reps), t de GN_R y carbón_R en la regresión conjunta:
t_obs(GN_R) = **-3.20, p_perm bilateral = 0.016** (nula centrada en -0.80±0.96) → **pasa**. t_obs(carbón_R) = 1.45,
p_perm = 0.198 (nula centrada en 0.65±0.90) → **no pasa** (dentro de lo esperable por azar/estructura de bloque).

## 6. Valor de robustez a confusión no observada (Cinelli-Hazlett; dof=285, DEV spec6)

| exposición | t | R² parcial | RV_q=1 | RV_q=1,α=0.05 | benchmark (control/exp. más fuerte) |
|---|---|---|---|---|---|
| **x_R_gn** | -3.20 | 3.46% | **17.2%** | **7.0%** | x_F_exo2 R²=0.95% (control puro más fuerte: feed_dross_Fe 0.58%) |
| x_R_carbon | 1.45 | 0.73% | 8.2% | **0.0%** | — |

GN_R: un confusor no observado necesitaría explicar ~17-18× la varianza que explica el control/exposición más fuerte
del modelo para anular el efecto — **cumple el umbral RV_q1≥5% con holgura**. Carbón_R: RV_q1,α05=0% porque el
punto de partida ya no es significativo (t=1.45); el "margen" de 8.2% es un artefacto de la fórmula sin significancia
que anular — **no cumple**.

## 7. Dosis-respuesta (KPI residualizado por controles, DEV, quintiles)

| exposición | patrón | pasos (signo) | R² lineal (5 ptos) | ΔQ5-Q1 (pp) |
|---|---|---|---|---|
| **x_R_gn** | **decreciente monótono** | ----  | **0.99** | **-1.98** |
| x_R_carbon | creciente monótono | ++++ | 0.90 | +1.18 |
| x_F_carbon | no monótono | +--+ | 0.05 | +0.17 |
| x_F_exo2 | no monótono | -++- | 0.34 | -0.97 |

Los IC bootstrap de cada quintil individual son anchos y cruzan 0 en casi todos los casos (n≈60/quintil); la
dosis-respuesta se sostiene por la forma agregada (monotonía + linealidad), no por un quintil aislado.

## Veredictos (criterio 1)

- **GN_R → KPI (vía f_dross↑ / f_metal↓): ACEPTADO.** p=.0014 y q_BH=.008 en DEV; signo estable en 3/3 tercios;
  robusto a estado pobre/rico (p≤.006) y a ponderación/unidades físicas; placebos nulos (siguiente p=.075, pre-trat.
  p>.11); permutación p=.016; RV_q1=17.2%≥5% (RV_q1,α05=7.0%); dosis-respuesta monótona y casi lineal (R²=.99).
  Único matiz: el efecto vive en R2-R3 (Rl), no en R0-R1; y se invierte de signo en lockbox (fin de campaña, H2) —
  no contradice el vínculo dentro de régimen, sí acota su alcance.
- **Carbón_R → KPI directo: RECHAZADO** (no cumple p<.05 en DEV, ni en spec6 ni en spec9; bootstrap incluye 0;
  sensibilidad de estado nunca <.05; placebo ok pero permutación p=.198; RV_q1,α05=0%). **Carbón_R → escoria
  (sn_perdido_escoria_frac ↓): ACEPTADO como canal** (Rt p=.011, Rl p=.032, signo teórico correcto, coherente con
  agotamiento de Sn), pero no se traduce en un neto de KPI robusto — posible cancelación con dross/polvo o falta de
  potencia. No entra al prescriptor como palanca directa sobre KPI; sí como evidencia de mecanismo.
- **Carbón_F → KPI: neutral, confirma E10-06** (canal polvo↑ p=.007 sí robusto; canal dross↓ n.s.; neto n.s.; dosis
  no monótona). No es una reclamación de H1; fuera de alcance de esta ola salvo como confirmación.
- **exo2_F → KPI: EN DUDA.** DEV p=.099 no cruza el umbral; se debilita más con estado pobre (p=.51); sólo "significativo"
  en el pool `total` (p=.0025, probablemente tendencia de campaña, no señal de acción); dosis-respuesta no monótona.
  Pertenece a H5 (Fusión), no se acepta bajo el criterio de H1.

**Conclusión honesta**: de los dos vínculos que H1 se propuso defender, **sólo GN_R sobrevive integro** el paquete de
pruebas (conjunta+BH, estabilidad, sensibilidad de estado, ponderación, 3 placebos, RV, dosis-respuesta). Carbón_R
sobrevive **sólo como mecanismo de canal** (escoria), no como vínculo directo a KPI — la pista original (Spearman
univariante p=.0015) no replica en la regresión conjunta controlando GN/exo2 simultáneamente; se recomienda no
anclar carbón_R al KPI en v9 sin evidencia adicional (p. ej. un objetivo que sume canales en vez de mirar sólo KPI).
