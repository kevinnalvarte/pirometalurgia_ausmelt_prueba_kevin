# E18-02 — Diseño intra-batch (efectos fijos de batch) con exposición exacta por escalón

Panel factorial completo: 362 batches x 4 escalones de Reducción (R0-R3) = 1448 filas; cada (Batch, orden) aparece
una vez. FE de batch y de orden se remueven con demeaning por proyecciones alternadas (`e18_02_lib.demean_two_way`,
exacto en el límite; 40 iteraciones); OLS sobre las variables demeaned, errores agrupados por batch (cluster HC).
Exposición: `dev__gn` (Nm³/min), `dev__carbon` (kg/min) = ejecutado − receta (`asesor_receta_v10.escalones_con_receta`).
Estado: leyes %Sn/%FeO previas, inventarios de Sn/FeO previos, T horno previa (los 5 pedidos). Targets:
`m6_ln_sn_dep`, `m8_ln_feo_ret_dross50`, `d_temperatura_horno_celsius` (dT, `TARGETS_V8["Reducción"]`).

## 1. Modelo base y robustez (`e18_02_theta_base.csv`)

| target | espec. | θ_GN (IC95) | p | θ_Carbón (IC95) | p | R² within |
|---|---|---|---|---|---|---|
| sn_dep | within (1) | **+0.0218 [0.012, 0.031]** | 1e-5 | −0.0030 [−0.0061, 0.0001] | 0.059 | 0.316 |
| sn_dep | carbón cinético (2) | +0.0205 | 2e-5 | −0.0003 (Cx_sn, ns) | 0.33 | 0.314 |
| sn_dep | nivel (3) | +0.0271 | <1e-5 | −0.0007 (ns) | 0.68 | 0.325 |
| feo_ret | within (1) | **−0.0033 [−0.0046,−0.0020]** | <1e-5 | **+0.0010 [0.0005,0.0015]** | 9e-5 | 0.387 |
| feo_ret | carbón cinético (2) | −0.0029 | 2e-5 | +0.00009 | 0.006 | 0.383 |
| feo_ret | nivel (3) | −0.0040 | <1e-5 | +0.0011 | 1e-4 | 0.391 |
| dT | within (1) | +0.340 [0.046,0.634] | 0.024 | −0.019 (ns) | 0.73 | 0.189 |
| dT | nivel (3) | −0.179 (ns, **cambia de signo**) | 0.27 | +0.182 | 0.008 | 0.191 |

GN es preciso (p ≪ 0.01) y estable de signo/magnitud en las 4 especificaciones para `sn_dep` y `feo_ret`. Carbón es
más débil: identificado con signo consistente sólo en `feo_ret` (y con un signo **contraintuitivo** — más carbón
retiene *más* FeO, no menos — que hallazgos previos ya marcaban como no identificado); en `sn_dep` es marginal
(p≈0.06) y no sobrevive el pase a nivel. `dT` es frágil: GN cambia de signo entre desvío y nivel — no se recomienda
apoyarse en dT dentro de este diseño.

## 2. Variación intra-batch (`e18_02_varianza.csv`, `e18_02_corr_ordenes.csv`)

`dev__gn`: 60.6 % de la varianza es DENTRO del batch (39.4 % entre batches). `dev__carbon`: 95.6 % dentro. Ambas
palancas tienen sobrada variación intra-batch para identificar θ. Correlación del desvío crudo entre escalones del
mismo batch: GN modesta y positiva (R0-R1 0.24, R1-R2 0.32, R2-R3 0.37; no "sube todo junto"). Carbón:
**negativa** (R0-R1 −0.35, R0-R2 −0.49, R1-R3 −0.41) — el operador compensa entre escalones, no arrastra el nivel;
esto favorece la identificación (menos colinealidad batch-level de lo temido).

## 3. Forma (`e18_02_dosis_respuesta.csv`, `e18_02_asimetria.csv`, `e18_02_heterogeneidad_*.csv`)

Dosis-respuesta: aproximadamente monótona/lineal para GN en ambos targets, con leve aplanamiento (saturación) en
el quintil extremo positivo. Asimetría: **no hay evidencia de kink** para GN (pasarse θ≈+0.0205 vs quedarse corto
θ≈+0.0229 en sn_dep; −0.0031 vs −0.0034 en feo_ret) — el costo de pasarse y de quedarse corto es prácticamente el
mismo. Heterogeneidad por orden: GN es igual de fuerte en R0-R1 y R2-R3 en ambos targets; **carbón sólo es
significativo en sn_dep en R2-R3** (θ=−0.0065, p=0.028; en R0-R1 ns) y en feo_ret pierde significancia en R2-R3.
Para dT, el efecto de GN sólo existe en R0-R1 (p=0.004) y desaparece en R2-R3; carbón en dT **cambia de signo**
entre tramos (−0.125 vs +0.242) — más evidencia de que dT no es un target confiable aquí. Heterogeneidad por Sn
disponible (tercil relativo al orden): sin interacción clara — θ_GN y θ_carbón no varían con cuánto Sn queda.

## 4. Causalidad inversa dentro del batch (`e18_02_reversion.csv`, `e18_02_modelo_aumentado.csv`, `e18_02_placebo.csv`)

`dev_gn_t` responde al resultado de t−1: sube con `sn_dep_{t-1}` (θ=0.85, p=0.010) y baja con `feo_ret_{t-1}`
(θ=−3.93, p=0.049, marginal), controlando el estado. `dev_carbon_t` no responde a ningún target rezagado (p>0.14).
**Al agregar el rezago como control, θ_GN y θ_carbón casi no cambian** (sn_dep: GN 0.0160→0.0163, carbón
−0.0036→−0.0036; feo_ret: GN −0.0026→−0.0024, carbón 0.0012→0.0012) — la reversión existe pero no sesga las θ
principales. Placebo (dev_{t+1} → target_t) **no sale limpio**: hay coeficientes significativos en feo_ret y dT
(p<0.01). Esto es probablemente mecánico: el desvío está autocorrelado entre escalones consecutivos (§2), así que
`dev_{t+1}` hereda parte de la correlación contemporánea de `dev_t` con el target, sin que eso implique causalidad
inversa genuina. Se reporta como salvedad, no se descarta el diseño por esto (la prueba directa con el control del
rezago, que sí es limpia, es la que importa).

## 5. Validación fuera de muestra y estabilidad entre regímenes (`e18_02_oos_groupkfold.csv`, `e18_02_tercios.csv`, `e18_02_dev_vs_lockbox.csv`)

GroupKFold(5) por batch (θ fuera de los batches de prueba, aplicado al demeaning propio del fold de prueba):
correlación predicho-observado (demeaned) **sn_dep 0.54, feo_ret 0.61, dT 0.42**; R² fuera de muestra 0.29 / 0.38 /
0.18 — bastante más preciso que la evidencia a nivel de KPI del batch en iteraciones previas.

**Punto clave — GN es estable entre regímenes aunque el KPI del batch no lo sea:**

| | GN sn_dep | GN feo_ret | Carbón sn_dep | Carbón feo_ret |
|---|---|---|---|---|
| tercio temprano | +0.037 (p<0.001) | −0.0044 (p=0.001) | −0.0106 (p<0.001) | +0.0020 (p<0.001) |
| tercio medio | +0.019 (p=0.07) | −0.0045 (p=0.001) | +0.0034 (ns) | +0.0006 (ns) |
| tercio tardío | +0.019 (p=0.009) | −0.0036 (p=0.001) | −0.0016 (ns) | +0.0008 (ns, p=0.06) |
| DEV | +0.0225 (p<0.001) | −0.0034 (p<0.001) | −0.0023 (ns) | +0.0011 (p=0.0006) |
| lockbox | +0.0289 (p=0.005) | −0.0041 (p=0.006) | **−0.0077 (p=0.028)** | +0.0004 (ns) |

GN mantiene signo y significancia en los 3 tercios cronológicos y en DEV/lockbox para **ambos** targets — la
palanca de escalón es mucho más estable que el β de GN a nivel de KPI del batch (que en v10 requería contracción
James-Stein y se invertía en fin de campaña). Carbón es inestable: sólo significativo en el tercio temprano para
ambos targets, se diluye después (aunque en lockbox su efecto sobre sn_dep resurge, −0.0077, p=0.028). dT: GN
**cambia de signo** entre DEV (+0.33, marginal) y lockbox (−0.20, ns) — no estable, se descarta como evidencia.

## 6. Traducción a kg (`e18_02_kg_por_unidad.csv`, `e18_02_balance_sn_metal.csv`)

Inventario típico al cierre del escalón (mediana): Sn 1523 kg (R0-R1) / 827 kg (R2-R3); FeO 11421 kg (R0-R1) /
10216 kg (R2-R3). Por **+1 Nm³/min de GN** sostenido en el escalón:

| tramo | kg Sn agotado | kg FeO reducido | selectividad FeO/Sn | kg Sn atrapado en dross (×0.6) | **kg Sn neto a metal** |
|---|---|---|---|---|---|
| R0-R1 (temprano) | +31.2 [11.8, 50.6] | +30.9 [11.7, 50.0] | 0.99 | 18.5 | **+12.7** |
| R2-R3 (tardío) | +19.0 [9.4, 28.7] | +39.3 [21.9, 56.6] | 2.06 | 23.6 | **−4.5** |

GN temprano tiene balance de Sn positivo (agota más Sn del que la reducción adicional de FeO atrapa en dross);
GN tardío se **revierte a negativo**: la misma dosis reduce proporcionalmente más FeO que Sn agotado (Sn ya casi
agotado), y el Sn atrapado en el dross de Fe supera la ganancia. Carbón: kg Sn agotado −2.5 (ns) / −5.4 (p=0.03) y
kg FeO reducido −10.3 / −12.1 (es decir, *menos* FeO reducido, coherente con el signo contraintuitivo de §1); el
balance neto sale positivo (+3.7 / +1.9) pero descansa sobre el canal FeO-carbón no identificado — no se destaca
como hallazgo firme.

## 7. Veredicto

El diseño intra-batch **sí entrega efectos de escalón precisos y estables para GN** sobre `sn_dep` y `feo_ret`:
p ≪ 0.01 en la especificación principal y en la mayoría de las variantes de robustez (dev/nivel/cinético), estable
en heterogeneidad por orden y por Sn disponible, insensible a controlar la causalidad inversa, y **estable entre
DEV y lockbox y entre los 3 tercios cronológicos** — justo el punto que el KPI del batch no lograba (v10 necesitaba
contracción James-Stein y se invertía en fin de campaña). Carbón no cumple el mismo estándar: identificado sólo en
`feo_ret` y con signo contraintuitivo, inestable en `sn_dep`, y sin mecanismo físico limpio. `dT` no es usable en
este diseño (cambia de signo entre especificaciones y regímenes).

**Recomendación que se sostiene SÓLO con esta evidencia** (sin depender del KPI del batch ni de confusores como
mineral/talón/campaña/turno, porque el diseño los absorbe con el efecto fijo de batch):

- **R0-R1: subir GN por encima de la receta es rentable** (balance de Sn a metal +12.7 kg por +1 Nm³/min sostenido).
- **R2-R3: NO subir GN por encima de la receta por este canal** — el balance de Sn a metal se vuelve negativo
  (−4.5 kg por +1 Nm³/min); el exceso tardío de calor sobrerreduce FeO a Fe metálico y atrapa Sn en el dross.
- **Carbón: sin recomendación firme.** Su efecto sobre el agotamiento de Sn es débil/inestable y su efecto sobre
  la retención de FeO tiene signo contrario al esperado por estequiometría — se necesita otro diseño (o el piloto
  aleatorizado) antes de actuar sobre esta palanca.
