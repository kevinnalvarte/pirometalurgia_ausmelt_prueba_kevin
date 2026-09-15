# B-03 -- Balance de energia por escalon (features termicas y calor de reaccion aparente)

Script: `experimentos/balances/b03_balance_energia.py` (semilla 42). Log completo: `experimentos/balances/b03_log.txt`
(tiempo total 1646 s = 27.4 min). CSVs: `b03_descomposicion*.csv`, `b03_correlacion_qneto_dT.csv`,
`b03_calibracion_coeficientes.csv`, `b03_calibracion_r2.csv`, `b03_dT_variantes.csv`, `b03_theta_dT.csv`,
`b03_correlacion_reaccion.csv`, `b03_sensibilidad.csv`. Dataset: `bal.cargar_df()` (3982 escalones), base sin
primer escalon de batch (`modelo_prescriptivo.dataset_base_modelo`, 3620 filas). DEV = 299 batches / lockbox =
63 batches ultimos cronologicos (17.5 %). Criterio: OOF GroupKFold(5) por Batch en DEV; lockbox solo reporte.

## Resumen ejecutivo

| Pregunta | Resultado | Recomendacion |
|---|---|---|
| (i) Features termicas (`b6_dT_teorico_sin_reaccion`, `b6_q_neto_por_min_MJ`, `b6_q_gases_MJ`, `b6_margen_termico_MJ`) como palancas adicionales del PLM de dT | Reduccion: R2 OOF/lockbox 0.466/0.465 -> **0.474/0.501** (mejora en ambos). Fusion: 0.481/0.066 -> 0.466/0.052 (empeora en ambos) | **ADOPTAR en Reduccion, RECHAZAR en Fusion** |
| (ii) Balance de energia calibrado por regresion fisica (NNLS/OLS) como modelo de dT | R2 OOF de la calibracion 0.03-0.09 (coef. de `q_combustion` calibrado ~0.03, muy lejos de ~1 teorico); como predictor fisico puro (dT = balance/(m*Cp)) es **numericamente inestable** (R2 OOF diverge, R2 lockbox -1.35 Fusion / -0.04 Reduccion) | **RECHAZAR** |
| (iii) Calor de reaccion aparente como estimador independiente de la extraccion | max\|Spearman\| en Reduccion = 0.248 (< 0.30); con el balance calibrado (sin el termino Sn/FeO) tampoco supera 0.30 en Reduccion; en Fusion el signo es incluso negativo (el balance esta sesgado, no "ve" la extraccion) | **RECHAZAR** (no se construye el estimador combinado; se documenta el techo) |

## 0) Datos y metodo

`bal.cargar_df()` -> 3982 escalones (362 batches, 7 Fusion + 4 Reduccion). Excluyendo el primer escalon de cada
batch (sin estado `_prev`): 3620 filas, de las cuales 2172 de Fusion y 1448 de Reduccion. Todas las features
usadas en modelos son `bal.SEGURAS_V6` (STATE / DERIVED-ACTION / DERIVED (A_t x S_prev)); `b6_q_reaccion_aparente_MJ`,
`b6_q_reaccion_trazador_MJ`, `b6_sn_extraido_energia_kg` son LEAKAGE y solo se usan para contrastar (secciones 2 y 4).

## 1) Descomposicion del balance

Medianas globales por fase (MJ, salvo dT en K):

| fase | n | q_combustion | q_carbon_lanza | q_carga | q_gases | q_perdidas | **q_neto** | **dT_teorico_sin_reaccion** | dT medido |
|---|---|---|---|---|---|---|---|---|---|
| Fusion | 2172 | 47 838 | 4 665 | 34 197 | 18 127 | 5 265 | **-4 020** | **-130.3** | 0.3 |
| Reduccion | 1448 | 16 444 | 0 | 758 | 6 490 | 3 950 | **+4 543** | **+66.5** | -4.3 |

Coincide con el smoke test del enunciado. Por `orden_escalon_fase` (medianas, MJ / K):

| fase | orden | q_combustion | q_carga | q_gases | q_perdidas | q_neto | dT_teorico | dT medido |
|---|---|---|---|---|---|---|---|---|
| Fusion | 1 | 34 892 | 24 888 | 11 563 | 3 744 | -2 129 | -304.8 | -1.8 |
| Fusion | 2 | 40 458 | 29 586 | 14 607 | 4 329 | -4 059 | -222.2 | 8.3 |
| Fusion | 3 | 46 031 | 33 393 | 17 030 | 4 914 | -4 588 | -162.5 | 1.8 |
| Fusion | 4 | 51 854 | 36 829 | 19 191 | 5 499 | -4 443 | -111.0 | -1.6 |
| Fusion | 5 | 57 494 | 40 457 | 20 993 | 6 084 | -4 401 | -82.3 | -2.7 |
| Fusion | 6 | 76 851 | 53 344 | 27 542 | 8 190 | -4 812 | -72.2 | -6.2 |
| Reduccion | 0 | 27 888 | 4 832 | 12 024 | 4 740 | 5 698 | 72.6 | -9.0 |
| Reduccion | 1 | 20 826 | 1 161 | 7 867 | 3 950 | 7 758 | 117.2 | -13.0 |
| Reduccion | 2 | 12 436 | 338 | 4 980 | 3 160 | 4 085 | 62.5 | -3.4 |
| Reduccion | 3 | 6 423 | 111 | 2 605 | 1 738 | 1 809 | 28.1 | 2.5 |

**Sesgo de Fusion (consistente en los 6 escalones):** `dT_teorico_sin_reaccion` es fuertemente negativo (-305 a
-72 K) mientras el dT medido oscila cerca de cero (-6 a +8 K). El balance, tal como esta parametrizado, predice
un enfriamiento del bano en cada escalon de Fusion que NO ocurre. Causa mas probable (documentada en el
docstring de `bal_features.py` y confirmada aqui): el balance **omite fuentes de calor** que si existen en el
modelo Rust original pero no estan en este dataset a nivel de escalon -- el aire de post-combustion (600-6500
Nm3/h) que quema el CO/carbon remanente a CO2 dentro del horno, y la oxidacion exotermica del Fe metalico y de
sulfuros del dross. `H_CARGA_MJ_KG` (1.7) tambien puede estar sobreestimado. El error crece con `orden_escalon_fase`
en magnitud absoluta de MJ (mas carga acumulada, mas gases) pero el dT_teorico se ACHICA relativamente porque la
masa del bano (denominador) tambien crece -- el sesgo persiste en las 6 etapas, no es un artefacto de un solo
escalon.

**Reduccion:** `dT_teorico_sin_reaccion` es positivo en las 4 etapas (28-117 K) mientras el dT medido es negativo
en R0-R1 y apenas positivo en R3 -- consistente con reacciones endotermicas (SnO2+2C, FeO+C) no capturadas por
`dT_teorico_sin_reaccion` (por diseno: esa columna es el balance SIN reaccion). El calor de reaccion aparente
(`q_neto - m*Cp*dT`) tiene mediana +5 024 MJ en Reduccion vs +3 097 MJ del trazador -- mismo orden de magnitud,
consistente con el enunciado.

Correlacion de `q_neto` / `dT_teorico_sin_reaccion` vs dT medido, DEV (GroupKFold no aplica aqui, es descriptivo):

| fase | columna | n | Spearman | p | Pearson |
|---|---|---|---|---|---|
| Fusion | q_neto_disponible_MJ | 1794 | 0.031 | 0.19 | 0.014 |
| Fusion | q_neto_por_min_MJ | 1794 | -0.039 | 0.10 | -0.045 |
| Fusion | dT_teorico_sin_reaccion | 1704 | **-0.122** | <0.001 | -0.095 |
| Reduccion | q_neto_disponible_MJ | 1196 | **-0.328** | <0.001 | -0.293 |
| Reduccion | q_neto_por_min_MJ | 1196 | -0.255 | <0.001 | -0.239 |
| Reduccion | dT_teorico_sin_reaccion | 1136 | **-0.301** | <0.001 | -0.269 |

Notable: en Fusion la correlacion de `q_neto` con dT es practicamente nula (0.03) y la de `dT_teorico_sin_reaccion`
es **negativa** (-0.12) -- el balance no solo no explica el dT medido, apunta al signo contrario, reforzando que
esta dominado por el sesgo, no por señal fisica real. En Reduccion la correlacion es negativa y mas fuerte
(-0.30/-0.33): mas "margen termico teorico" coincide con MENOR dT medido, lo cual es coherente si ese margen se
consume en reacciones endotermicas que no estan en la formula (a mas Sn/FeO disponible para reducir, mas q_neto
teorico pero tambien mas absorcion real) -- una señal fisica real, aunque de signo contra-intuitivo para un
predictor lineal directo de dT.

## 2) Calibracion del balance por regresion fisica

`y = masa_escoria_est_kg_prev * 1.2 * dT_medido / 1000` [MJ absorbidos por el bano]. Regresores orientados por
signo (NNLS fuerza el signo teorico; OLS libre sin restriccion), GroupKFold(5) por batch para R2 OOF (DEV),
lockbox con coeficientes de DEV.

**R2 OOF / lockbox de la calibracion (variante estandar, con "Sn disponible" como proxy endotermico):**

| fase | metodo | R2 OOF | R2 lockbox | n dev |
|---|---|---|---|---|
| Fusion | NNLS | 0.065 | -0.63 | 1727 |
| Fusion | OLS | 0.090 | -0.83 | 1727 |
| Reduccion | NNLS | 0.033 | 0.03 | 1056 |
| Reduccion | OLS | 0.077 | -0.07 | 1056 |

**Variante LEAKAGE** (r7 reemplazado por la extraccion REAL: `sn_extraido_est_kg` + `b6_feo_extraido_multi_kg`
en Reduccion, o `sn_extraido_est_kg` en Fusion -- misma n en Fusion, n menor en Reduccion por disponibilidad del
multi-trazador):

| fase | metodo | R2 OOF | R2 lockbox | n dev |
|---|---|---|---|---|
| Fusion | NNLS | 0.051 | -0.77 | 1727 |
| Fusion | OLS | 0.070 | -0.98 | 1727 |
| Reduccion | NNLS | 0.013 | -0.02 | 759 |
| Reduccion | OLS | 0.071 | -0.14 | 759 |

**Hallazgo clave:** conocer la extraccion REAL (en vez del proxy "Sn disponible") **no mejora** el ajuste del
balance -- en Fusion (misma muestra, comparacion limpia) el R2 OOF de NNLS BAJA de 0.065 a 0.051. Esto indica que
el problema del balance no es la especificacion del termino endotermico, sino que **los demas terminos
(combustion, carga, gases, perdidas) no explican el dT medido de forma consistente entre escalones** -- el
balance esta dominado por ruido/sesgo estructural, no por una mala aproximacion de la extraccion.

**Teoria vs calibrado (NNLS, DEV, IC95% bootstrap cluster por batch, n_boot=150):**

| fase | regresor | teoria | coef. NNLS | IC95% | concuerda signo |
|---|---|---|---|---|---|
| Fusion | q_combustion | ~1 | 0.027 | [0.018, 0.036] | si (pero muy lejos de 1) |
| Fusion | c_fijo (carbon) | 9.2-32.8 MJ/kg | 0.740 | [0.51, 1.01] | si (pero 40x menor que el rango teorico) |
| Fusion | dross_fe (Fe met.) | 0-1.5 MJ/kg | 0.000 | [0,0] | pin en 0 |
| Fusion | feed_total (H_CARGA) | -1.3 a -2.2 MJ/kg | -0.000 | [0,0] | pin en 0 |
| Fusion | q_gas_sensible (Cp_gas) | -1.35 a -2.10 | -0.159 | [-0.196,-0.126] | si (10x menor) |
| Fusion | delta_tiempo (perdidas) | -117 MJ/min | -0.000 | [0,0] | pin en 0 |
| Fusion | sn_disponible (endo.) | -(peq., ~3.0 x frac. extraida) | -0.039 | [-0.067,-0.019] | si |
| Fusion | feed_fe_mineral | -0.84 MJ/kg | -0.000 | [0,0] | pin en 0 |
| Reduccion | q_combustion | ~1 | 0.000 | [0,0] | pin en 0 |
| Reduccion | c_fijo (carbon) | 9.2-32.8 MJ/kg | 0.858 | [0.47,1.23] | si (11-38x menor) |
| Reduccion | dross_fe (Fe met.) | 0-1.5 MJ/kg | 0.260 | [0.07,0.46] | si (dentro de rango) |
| Reduccion | feed_total (H_CARGA) | -1.3 a -2.2 MJ/kg | -0.000 | [0,0] | pin en 0 |
| Reduccion | q_gas_sensible (Cp_gas) | -1.35 a -2.10 | -0.194 | [-0.238,-0.126] | si (7-10x menor) |
| Reduccion | delta_tiempo (perdidas) | -158 MJ/min | -0.000 | [-7.1,0] | pin en 0 |
| Reduccion | sn_disponible (endo.) | -(peq.) | -0.041 | [-0.071,-0.013] | si |
| Reduccion | feed_fe_mineral | -0.84 MJ/kg | -0.000 | [0,0] | pin en 0 |

La NNLS colapsa la mitad de los regresores a 0 (`dross_fe` y `feed_fe_mineral` en Fusion, `feed_total`,
`delta_tiempo`, `q_combustion` en varios casos): sintoma de **colinealidad severa** -- `q_combustion`,
`q_carga`, `q_gases` y `delta_tiempo` se mueven juntos con el plan de la fase (mas GN -> mas combustion, mas
carga, mas gases, en la misma tasa), asi que el optimizador reparte el "credito" de forma arbitraria entre ellos
sujeto al signo. El coeficiente de `q_combustion` (que por construccion fisica deberia ser ~1, un simple paso de
unidades MJ->MJ) sale en 0.00-0.03: la regresion NO recupera ni siquiera la relacion mas basica y menos
discutible del balance. El OLS libre (sin restriccion de signo, ver `b03_calibracion_coeficientes.csv`) confirma
la inestabilidad: `q_combustion` sale NEGATIVO en Reduccion (-0.10) y `delta_tiempo` (perdidas) sale POSITIVO en
ambas fases (+20 a +42 MJ/min), signos fisicamente imposibles. Con R2 OOF de 0.03-0.09 y coeficientes que no
recuperan ni el orden de magnitud ni siempre el signo esperado, **la calibracion no es identificable con estos
datos**: el balance de escalon, aun ajustado, no cierra.

## 3) Modelo de dT: variantes (a-d)

(a) PLM v4 base (estado/palancas de `modelo_predictivo_v4`), (b) PLM v4 + 4 features termicas b6 como palancas
adicionales, (c) predictor fisico puro `dT = balance_calibrado(NNLS,DEV) / (masa_prev*Cp)`, (d) HGB (sin
estructura PLM) con el mismo set de (b).

| fase | variante | R2 OOF | R2 lockbox | n dev | n lockbox |
|---|---|---|---|---|---|
| Fusion | a) PLM v4 | 0.481 | 0.066 | 1482 | 314 |
| Fusion | b) PLM v4 + termicas | 0.466 | 0.052 | 1420 | 314 |
| Fusion | c) fisico calibrado | numericamente inestable (OOF diverge); lockbox -1.35 | -- | 1727 | 374 |
| Fusion | d) HGB set (b) | 0.453 | 0.041 | 1420 | 314 |
| Reduccion | a) PLM v4 | 0.466 | 0.465 | 1183 | 252 |
| Reduccion | b) PLM v4 + termicas | **0.474** | **0.501** | 1135 | 252 |
| Reduccion | c) fisico calibrado | numericamente inestable (OOF diverge); lockbox -0.04 | -- | 1056 | 238 |
| Reduccion | d) HGB set (b) | 0.485 | 0.504 | 1135 | 252 |

**Variante (c):** el R2 OOF calculado por fold da valores absurdos (~-7e47 en Fusion, ~-5e46 en Reduccion): la
NNLS, ajustada dentro de un fold de entrenamiento (4/5 de DEV agrupado por batch), a veces produce coeficientes
enormes para un regresor casi colineal con otro dentro de ese subconjunto (sin regularizacion ni intercepto, NNLS
no acota la magnitud, solo el signo) -- al aplicar esos coeficientes al fold de test la prediccion explota. El
numero de lockbox (coeficientes ajustados con TODO DEV, mas estable) es mas representativo: -1.35 (Fusion) y
-0.037 (Reduccion), ambos peores que predecir la media. **La inestabilidad numerica es en si misma el hallazgo**:
un balance con R2 interno de 0.03-0.09 y coeficientes que colapsan a cero o cambian de signo entre subconjuntos
de datos no es un predictor fisico utilizable, ni siquiera como referencia.

**Variante (b) vs (a):** en Reduccion, agregar las 4 features termicas mejora el PLM en OOF (+0.008) y sobre todo
en lockbox (+0.036, de 0.465 a 0.501) -- unica mejora limpia y consistente de toda la seccion 3. En Fusion,
agregarlas EMPEORA tanto OOF (-0.015) como lockbox (-0.014): coherente con el sesgo sistematico de Fusion
documentado en la seccion 1 (el balance de Fusion "miente" en la direccion del dT, asi que darle esa señal al
modelo lo confunde). La variante (d) HGB confirma el patron: en Reduccion el HGB con el set (b) es el mejor de
todos (0.485/0.504), en Fusion es el peor (0.453/0.041) -- no es un artefacto de la estructura PLM, es la calidad
de las features termicas por fase.

**Theta de las palancas (IC95% bootstrap, n_boot=150), `b03_theta_dT.csv`:**

Reduccion, variante (a): GN significativo y con signo teorico correcto: theta=0.456 [0.058, 0.870] por Nm3/min
(2.28 kg-eq/sd [0.29, 4.35]); O2 significativo pero NEGATIVO (-0.419 [-0.732,-0.091], sin signo teorico asignado
en v4). Variante (b): GN sigue significativo y mas fuerte (0.795 [0.254,1.249], +1sd 3.99 [1.27,6.27]); el efecto
de O2 deja de ser significativo (-0.035 [-0.383,0.360]) -- las features termicas nuevas absorben parte de la
señal que antes cargaba O2 (son construidas con el mismo O2/aire/GN de la lanza), y `b6_dT_teorico_sin_reaccion`
sale con IC que casi no cruza 0 (-0.074 [-0.171, 0.002]) pero de signo NEGATIVO (contra-intuitivo, +1 esperado),
coherente con la correlacion negativa de la seccion 1. Ninguna de las 4 features termicas es individualmente
significativa al 95 % en Reduccion; la mejora de R2 en (b) es conjunta, no atribuible a una sola.

Fusion, ambas variantes: ninguna palanca de lanza/carbon es significativa (todos los IC cruzan 0), consistente
con v4 (referencia lockbox 0.06-0.07). En variante (b), `b6_dT_teorico_sin_reaccion` es la UNICA palanca
significativa (theta=-0.024 [-0.039,-0.007]) y su signo es OPUESTO al teorico (se esperaba +1): a mayor "margen
termico teorico" segun el balance sin calibrar, MENOR el dT real predicho -- exactamente el reflejo del sesgo de
Fusion (seccion 1): la variable termica en Fusion no aporta fisica util, aporta el sesgo del balance.

## 4) Calor de reaccion aparente como estimador de la extraccion

Correlacion (DEV) de `b6_q_reaccion_aparente_MJ` (bruto) y de la version calibrada (`q_neto` de la seccion 2 SIN
el termino Sn/FeO, es decir el residuo del balance calibrado) contra el trazador y las extracciones en kg:

| fase | version | variable | n | Spearman | Pearson |
|---|---|---|---|---|---|
| Fusion | bruto | q_reaccion_trazador_MJ | 1749 | -0.253 | -0.164 |
| Fusion | bruto | sn_extraido_est_kg | 1749 | -0.270 | -0.169 |
| Fusion | bruto | feo_extraido_est_kg | 1749 | 0.046 (p=0.06) | -0.022 |
| Reduccion | bruto | q_reaccion_trazador_MJ | 1076 | 0.206 | 0.098 |
| Reduccion | bruto | sn_extraido_est_kg | 1076 | **0.248** | 0.085 |
| Reduccion | bruto | feo_extraido_est_kg | 1071 | 0.128 | 0.114 |
| Reduccion | bruto | feo_extraido_multi_kg | 777 | 0.042 (p=0.24) | -0.035 |
| Fusion | calibrado (sin r7) | q_reaccion_trazador_MJ | 1727 | **-0.296** | -0.235 |
| Fusion | calibrado (sin r7) | sn_extraido_est_kg | 1727 | -0.281 | -0.233 |
| Reduccion | calibrado (sin r7) | q_reaccion_trazador_MJ | 1056 | -0.223 | -0.163 |
| Reduccion | calibrado (sin r7) | sn_extraido_est_kg | 1056 | -0.214 | -0.151 |
| Reduccion | calibrado (sin r7) | feo_extraido_multi_kg | 759 | -0.113 (p=0.002) | -0.017 |

**Umbral de decision (|rho|>=0.30 en Reduccion):** el maximo |Spearman| observado en Reduccion es 0.248 (bruto,
vs `sn_extraido_est_kg`); con el balance calibrado el signo incluso se invierte (-0.21 a -0.22). **No se cumple
el umbral -> no se construye el estimador combinado** (ver log, seccion 4). En Fusion el calor de reaccion
aparente esta correlacionado negativamente con la extraccion real (-0.25 a -0.30): el balance de Fusion, sesgado
como se documento en la seccion 1, se mueve en la direccion CONTRARIA a la extraccion real -- mas Sn extraido
(mas calor absorbido de verdad) coincide con un q_reaccion_aparente MAS chico, lo cual es fisicamente absurdo y
confirma que en Fusion esta dominado por el sesgo estructural, no por la reaccion.

Por `orden_escalon_fase` (detalle en `b03_correlacion_reaccion.csv`): la correlacion en Reduccion no mejora en
ningun escalon individual (N por escalon ~270-360, suficiente para detectar rho=0.30 con p<0.05); el patron es
parejo, no hay una etapa donde el balance "vea" la extraccion mejor que otra.

**Techo documentado:** el balance de energia por escalon **no resuelve la extraccion** de forma independiente del
ensayo de escoria, ni en su version bruta ni calibrada. Razones: (1) muy pocas variables termicas de estado
(temperatura de cara interior, termocupla, gradiente) frente a la cantidad de calor en juego (decenas de miles de
MJ) -- el ruido de medicion de esas pocas variables, propagado a traves de `m*Cp*dT`, es del mismo orden que la
señal de reaccion (2 000-5 000 MJ); (2) la temperatura de cierre disponible es la de la cara interior del
refractario, no la temperatura del bano -- el desfase termico entre ambas no esta modelado; (3) la post-combustion
de CO/carbon con aire secundario (600-6500 Nm3/h en el modelo Rust) no esta en este dataset a nivel de escalon,
lo que introduce un sesgo de igual o mayor magnitud que la señal buscada (ver seccion 1, Fusion).

## 5) Sensibilidad

Efecto sobre R2 OOF/lockbox de la variante (b) y sobre la correlacion `q_reaccion_aparente` vs `sn_extraido_est_kg`
(DEV). Caso base: PCI=35.8 MJ/Nm3, PERDIDAS=100 %, CP_ESCORIA=1.2, H_CARGA=1.7.

*Nota metodologica sobre PCI:* `PCI_GN_MJ_NM3` esta declarado en `bal.SUPUESTOS` pero **no esta cableado** en
`agregar_balance_energia` -- la formula de `q_combustion` usa entalpias fijas de combustion del CH4 (DH_CH4_CO2 =
-802.3 kJ/mol, DH_CH4_CO = -519.3 kJ/mol), equivalentes a un PCI implicito de 802.3/22.414 = 35.79 MJ/Nm3 (el
valor por defecto). Sobreescribir `bal.SUPUESTOS["PCI_GN_MJ_NM3"]` y recomputar con
`bal.construir_features_v6(...)` (como indica el resto del enunciado) **no tiene ningun efecto** sobre
`b6_q_combustion_MJ`. Para no modificar `bal_features.py`, la sensibilidad a PCI se aproximo en el script
reescalando linealmente `b6_q_combustion_MJ` (y las columnas derivadas) por `PCI_nuevo/35.8` -- una aproximacion
razonable si se asume que la correccion sub-estequiometrica (fraccion a CO) no cambia con el PCI, pero es una
aproximacion local hecha fuera de la libreria compartida, no una sensibilidad nativa del bloque v6.

| parametro | valor | R2 OOF dT(b) Fusion | R2 lockbox dT(b) Fusion | rho Fusion | R2 OOF dT(b) Reduccion | R2 lockbox dT(b) Reduccion | rho Reduccion |
|---|---|---|---|---|---|---|---|
| base | -- | 0.466 | 0.052 | -0.270 | 0.474 | 0.501 | 0.248 |
| PCI_GN_MJ_NM3 (aprox.) | 33.0 | 0.473 | 0.048 | -0.507 | 0.474 | 0.503 | 0.130 |
| PCI_GN_MJ_NM3 (aprox.) | 38.0 | 0.463 | 0.062 | -0.047 | 0.474 | 0.507 | 0.321 |
| PERDIDAS_MJ_MIN x | 0.5 | 0.463 | 0.056 | -0.083 | 0.474 | 0.500 | 0.318 |
| PERDIDAS_MJ_MIN x | 1.5 | 0.471 | 0.053 | -0.431 | 0.474 | 0.505 | 0.160 |
| CP_ESCORIA_KJ_KG_K | 1.0 | 0.466 | 0.052 | -0.272 | 0.474 | 0.501 | 0.248 |
| CP_ESCORIA_KJ_KG_K | 1.4 | 0.466 | 0.052 | -0.267 | 0.474 | 0.501 | 0.247 |
| H_CARGA_MJ_KG | 1.3 | 0.466 | 0.068 | 0.231 | 0.475 | 0.504 | 0.311 |
| H_CARGA_MJ_KG | 2.2 | 0.473 | 0.047 | -0.656 | 0.475 | 0.506 | 0.170 |

**Lectura:** el R2 de la variante (b) es **muy insensible** a los 4 supuestos en ambas fases (Reduccion se
mantiene en 0.474-0.475 / 0.500-0.507; Fusion en 0.463-0.473 / 0.047-0.068) -- la mejora/deterioro de R2 al
agregar las features termicas es estructural (viene de que Reduccion tiene menos ruido de carga y Fusion esta
sesgado), no un artefacto de un supuesto concreto. La correlacion `q_reaccion_aparente` vs `sn_extraido_est_kg`
en cambio es **bastante sensible**, sobre todo a PCI y H_CARGA: en Reduccion oscila entre 0.13 y 0.32 (cruza el
umbral 0.30 con PCI=38 y con H_CARGA=1.3, pero no con los demas), y en Fusion oscila entre -0.05 y -0.66 (siempre
negativa salvo el escenario extremo PCI=38, casi nula). Esto confirma que el calor de reaccion aparente no es un
estimador robusto: su correlacion con la extraccion depende mas de que supuesto de calibracion se use que de una
señal fisica estable -- exactamente lo esperable si el balance no cierra (seccion 2) y el "ruido" de calibracion
es del mismo orden que la señal.

## Limitaciones

1. El balance de escalon usa la temperatura de CIERRE de la cara interior del refractario (`temperatura_horno_celsius`),
   no la temperatura del bano ni un promedio del escalon -- el desfase termico entre ambas no esta modelado y es,
   segun esta seccion, la limitacion dominante.
2. La post-combustion de CO/carbon con aire secundario (presente en el modelo Rust, 600-6500 Nm3/h) no esta en
   este dataset a nivel de escalon: es la causa mas probable del sesgo sistematico de Fusion (seccion 1).
3. `PCI_GN_MJ_NM3` no esta parametrizado en `bal_features.py` (dead code); la sensibilidad reportada en la
   seccion 5 es una aproximacion local hecha en `b03_balance_energia.py`, no una recalculo nativo del bloque v6.
4. La calibracion por NNLS/OLS (seccion 2) tiene R2 interno muy bajo (0.03-0.09) y colinealidad severa entre
   regresores (varios colapsan a 0 con NNLS, cambian de signo con OLS libre): los coeficientes calibrados NO deben
   interpretarse como estimaciones fisicas confiables de Cp_gas, H_CARGA, etc., solo como diagnostico de que el
   balance no es identificable con las variables de escalon disponibles.
5. La variante (c) (balance calibrado como predictor fisico puro de dT) es numericamente inestable en
   validacion cruzada (coeficientes NNLS sin regularizacion ni intercepto pueden explotar en subconjuntos de
   entrenamiento); se reporta solo el numero de lockbox (mas estable, fit con todo DEV) como referencia, y aun
   asi es claramente peor que predecir la media (R2 negativo en ambas fases).
6. La correlacion de la seccion 4 se calculo en DEV (362x0.825 ~ 299 batches); dado el techo bajo (rho<0.3) y la
   sensibilidad de la seccion 5, no se intento validar en lockbox (no aporta si el DEV ya no muestra señal robusta).

## Recomendacion explicita

- **(i) Features termicas en el modelo de dT: ADOPTAR en Reduccion** (b6_dT_teorico_sin_reaccion,
  b6_q_neto_por_min_MJ, b6_q_gases_MJ, b6_margen_termico_MJ como palancas adicionales del PLM de dT en Reduccion:
  R2 OOF 0.466->0.474, lockbox 0.465->0.501, mejora limpia y consistente). **RECHAZAR en Fusion** (R2 OOF
  0.481->0.466, lockbox 0.066->0.052, empeora; ademas la unica palanca termica significativa tiene signo
  contra-teorico, reflejo del sesgo de Fusion documentado en la seccion 1).
- **(ii) Balance calibrado como modelo fisico de dT: RECHAZAR.** R2 OOF de la calibracion 0.03-0.09 en ambas
  fases, coeficiente de `q_combustion` calibrado en 0.00-0.03 (deberia ser ~1 por construccion), varios
  regresores colapsan a 0 (NNLS) o cambian de signo (OLS libre), y el uso directo como predictor fisico de dT
  (variante c) es numericamente inestable y peor que la media en lockbox. El balance de escalon, con las
  variables termicas disponibles en este dataset, no cierra.
- **(iii) Estimador energetico de la extraccion: RECHAZAR.** max|Spearman| en Reduccion = 0.248 (<0.30, umbral
  del /goal), y con el balance calibrado el signo se invierte; en Fusion la correlacion es negativa (el balance
  sesgado apunta contra la extraccion real). No se construye el estimador combinado trazador+energia. El techo
  documentado (pocas variables termicas, T de cara interior != T de bano, post-combustion no medida) es
  estructural, no un problema de calibracion de parametros -- una ronda 2 tendria que resolver primero (1) y (2)
  de las limitaciones antes de que el calor de reaccion aparente aporte señal util.
