# B-02 -- Balance reductor/oxidante en kg de carbono equivalente como feature estructurada

Iteracion 6, bloque de balances. Script: `b02_balance_reductor.py` (semilla 42, log completo en
`b02_log.txt`, CSVs `b02_*.csv`). Usa `bal_features.py` (NO modificado) + `search_targets/st_targets.py`
(targets SDI/IRF de la iteracion 5) sobre el dataset v3 cacheado (3982 escalones, 362 batches; DEV = 299
batches / lockbox = 63 batches, ultimos 17.5% cronologicos). Criterio: OOF GroupKFold(5) por Batch en DEV;
lockbox solo se reporta, nunca se usa para elegir.

## Resumen ejecutivo

| Pregunta | Recomendacion | Por que |
|---|---|---|
| (i) `b6_reductor_neto_kg_ceq` (y `b6_ratio_reductor_demanda`/`b6_capacidad_sobre_reduccion_feo_kg`) como palanca del PLM | **RECHAZAR** (como sustituto de las palancas v4) | R2 OOF DEV no mejora en NINGUNO de los 6 targets probados (diferencias de 0.001-0.002, ruido); el IC de theta sigue cruzando 0 en todos los casos; el signo es correcto para FeO/SDI pero **incorrecto para Sn extraido** (el target donde v4 es mas fuerte, R2 0.976) y para la variante de Fusion. La correlacion marginal (sin controlar por estado) es NEGATIVA en los 3 targets probados, contraria a la hipotesis, por el confusor de inventario de Sn (ver Seccion 1). Unico punto a favor: el IC de theta sobre FeO se angosta de forma real (357 kg/sd vs 1594 kg/sd de los terminos de carbon de v4), evidencia de mejor identificacion aunque no de significancia. |
| (ii) Los acumulados `b6_cum_reductor_neto_kg_prev` / `b6_cum_oferta_reductora_kg_prev` como ESTADO | **RECHAZAR** | Anadirlos al estado (variantes d/e) no mejora el R2 OOF (plano o levemente peor) y **degrada el R2 lockbox de forma sistematica**: FeO 0.378->0.358, SDI 0.143->0.077/0.097. Son sumas monotonas dentro del batch, altamente correlacionadas con el orden del escalon y con la fecha del batch: el HGB de g(S) las usa para sobreajustar patrones de DEV que no generalizan.
| (iii) `b6_utilizacion_reductor` como diagnostico de arrastre | **RONDA 2** | La utilizacion agregada por batch (mediana 1.03 Fusion / 1.05 Reduccion) es muy superior al supuesto del modelo Rust (~0.80 F / ~0.52 R): una discrepancia real y grande que merece explicarse, pero la regresion contra proxies de arrastre (tiro, gas total, posicion de lanza, pellets, polvo/dross del batch) solo explica R2=0.124 y con signos parcialmente contra-intuitivos. No hay evidencia suficiente para usarla como diagnostico causal de arrastre; si para senalar que los supuestos de "carbon perdido al tiro" del modelo Rust no se reproducen con este balance simplificado (requiere cruce con B-03/B-04). |

---

## 1. Descriptiva y coherencia fisica

### 1.1 Medianas por fase y orden de escalon (IQR en el CSV `b02_descriptiva_por_escalon.csv`)

| fase | escalon | `b6_reductor_neto_kg_ceq` | `b6_ratio_reductor_demanda` | `b6_ceq_lanza_kg` | `b6_capacidad_sobre_reduccion_feo_kg` | `b6_cum_reductor_neto_kg_prev` |
|---|---|---:|---:|---:|---:|---:|
| Fusion | 0 | -895.6 | 0.1 | -304.3 | 0.0 | 0.0 |
| Fusion | 1 | -753.7 | 0.4 | -355.3 | 0.0 | -895.6 |
| Fusion | 2 | -873.3 | 0.6 | -418.7 | 0.0 | -1656.1 |
| Fusion | 3 | -1216.7 | 0.5 | -480.6 | 0.0 | -2551.7 |
| Fusion | 4 | -1549.7 | 0.5 | -543.6 | 0.0 | -3722.1 |
| Fusion | 5 | -2040.5 | 0.4 | -600.5 | 0.0 | -5315.6 |
| Fusion | 6 | -2732.7 | 0.4 | -811.4 | 0.0 | -7297.5 |
| Reduccion | 0 | -715.5 | 0.7 | 107.6 | 0.0 | -10000.0 |
| Reduccion | 1 | +250.9 | 1.7 | 62.4 | **1352.2** | -10662.2 |
| Reduccion | 2 | -36.3 | 0.8 | 39.0 | 0.0 | -10530.9 |
| Reduccion | 3 | -91.4 | 0.4 | 24.0 | 0.0 | -10588.3 |

Coherente con el modelo teorico: en Fusion el reductor neto es siempre negativo y se hace mas negativo
escalon a escalon (el carbon nunca alcanza para el Sn+Fe2O3 que va entrando; el deficit se acumula:
`b6_cum_reductor_neto_kg_prev` cae monotonicamente de 0 a -7298 kg_ceq al cierre de Fusion). En Reduccion
el patron es fisicamente sensato: R0 hereda el deficit fuerte de Fusion (arranca en -10000 acumulado) y
sigue en deficit (-715 kg neto) porque el Sn disponible sigue siendo alto; **R1 es el unico escalon con
mediana de reductor neto POSITIVA (+251 kg_ceq, ratio 1.7)** y es tambien el unico con capacidad de
sobre-reduccion de FeO no nula (mediana 1352 kg): coincide con la ventana donde el carbon alimentado ya
supera la demanda decreciente de Sn. R2-R3 vuelven a un deficit leve (-36 a -91 kg) al agotarse el carbon
remanente. Esto reproduce el "banco de reductor" que describe el modelo Rust.

### 1.2 Spearman: `b6_reductor_neto_kg_ceq` / `b6_capacidad_sobre_reduccion_feo_kg` vs targets, global y por tramo de avance

(n y p en `b02_spearman_por_tramo.csv`; filtro `~es_primer_escalon_batch`, R0 incluido con estado
heredado de F6)

| predictor | target | global rho | avance <0.84 | 0.84-0.96 | >0.96 |
|---|---|---:|---:|---:|---:|
| `b6_reductor_neto_kg_ceq` | `feo_extraido_est_kg` | **-0.328** (p<1e-33) | -0.016 (n.s.) | -0.089 (n.s.) | **-0.163** (p<1e-5) |
| `b6_reductor_neto_kg_ceq` | `b6_feo_extraido_multi_kg` | **-0.097** (p=0.003) | -- | -0.031 (n.s.) | **-0.100** (p=0.004) |
| `b6_reductor_neto_kg_ceq` | `sn_extraido_est_kg` | **-0.504** (p<1e-84) | **-0.747** (p<1e-49) | **-0.259** (p<1e-4) | +0.037 (n.s.) |
| `b6_capacidad_sobre_reduccion_feo_kg` | `feo_extraido_est_kg` | **-0.220** (p<1e-15) | NaN (casi todo 0) | -0.114 (n.s., p=0.10) | **-0.147** (p<1e-4) |
| `b6_capacidad_sobre_reduccion_feo_kg` | `sn_extraido_est_kg` | **-0.198** (p<1e-12) | NaN | **-0.224** (p<0.01) | +0.115 (p<0.001) |

**Resultado NO esperado**: la hipotesis del enunciado (correlacion positiva, mas fuerte a avance alto) NO
se confirma en la correlacion marginal (sin controlar por estado) -- el signo es NEGATIVO en todos los
casos con n suficiente, y para FeO se hace MAS negativo (no menos) a avance alto. La razon es un confusor
mecanico directo en la propia formula: `b6_reductor_neto_kg_ceq = oferta - 2*C_por_Sn*Sn_disponible -
C_por_Fe2O3*Fe2O3`, es decir el termino RESTA la demanda de Sn disponible; cuando el Sn disponible es alto
(escalones tempranos, avance bajo) la demanda es alta y el neto tiende a ser muy negativo, y es
precisamente ahi donde la extraccion de Sn (que tambien depende del inventario disponible, cinetica de
primer orden) es mayor. Ambas variables comparten el mismo termino `Sn_disponible` con signos opuestos por
construccion, lo que produce una correlacion negativa espuria antes de controlar por estado. Esto es
justamente el problema que el PLM (Seccion 2) intenta resolver via `g(S)` y residualizacion OOF -- pero
los resultados de la Seccion 2 muestran que, una vez controlado el estado, el efecto sigue sin ser
significativo y el signo se revierte tambien para Sn (ver 2.1).

### 1.3 Utilizacion del reductor por escalon (adelanto de Seccion 3)

`b6_utilizacion_reductor` (LEAKAGE/diagnostico, no feature): Fusion mediana 0.994 [P25 0.832, P75 1.230,
n=2119]; Reduccion mediana 0.906 [P25 0.379, P75 1.492, n=1266]. Coherente con el smoke test de
`bal_features.py`.

---

## 2. PLM v4 con y sin balance reductor (Reduccion y Fusion)

Variantes: **(a)** v4 original; **(b)** sustituye el/los termino(s) de carbon por `b6_reductor_neto_kg_ceq`
(mismo estado, mismo GN/O2/aire); **(c)** v4 + `b6_capacidad_sobre_reduccion_feo_kg` anadida; **(d)** v4 con
`b6_cum_reductor_neto_kg_prev` + `b6_cum_oferta_reductora_kg_prev` anadidas al ESTADO; **(e)** = (b)+(d).
Para los targets tipo "feo"/"sdi" el "v4 original" es `[tasa_feed_Carbon_kg_min, Cx_av_v4, tasa_gn_nm3_min,
exceso_o2_combustion_pct, tasa_aire_nm3_min]` (= `PALANCAS_V4[("Reducción","feo_kg")]`); para "sn" es
`[Cx_sn_v4, tasa_gn_nm3_min, exceso_o2_combustion_pct]` (= `PALANCAS_V4[("Reducción","sn_kg")]`); SDI usa la
misma familia de palancas que FeO (selectividad Sn/FeO). n_boot=150, n_splits=5.

### 2.1 R2 OOF (DEV) / R2 lockbox por target y variante (Reduccion)

| target | n_dev | (a) oof/lb | (b) oof/lb | (c) oof/lb | (d) oof/lb | (e) oof/lb |
|---|---:|---|---|---|---|---|
| `feo_extraido_est_kg` | 1045 | 0.367/0.378 | 0.368/**0.377** | 0.367/0.379 | 0.367/0.358 | 0.368/0.358 |
| `b6_feo_extraido_multi_kg` | 758 | -0.034/-1.04 | -0.035/-1.03 | -0.034/-1.05 | -0.031/-1.12 | -0.032/-1.10 |
| `sn_extraido_est_kg` | 1049 | **0.976/0.984** | 0.975/0.984 | 0.976/0.984 | 0.977/0.983 | 0.976/0.983 |
| `st_R22_sdi` | 1015 | 0.131/0.143 | 0.131/**0.165** | 0.132/0.132 | 0.132/0.077 | 0.131/0.097 |
| `b6_sdi_multi` | 727 | 0.022/-0.40 | 0.021/-0.40 | 0.022/-0.40 | 0.024/-0.37 | 0.024/-0.38 |

Lectura: `b6_feo_extraido_multi_kg` y `b6_sdi_multi` (targets del cierre multi-trazador, B-01) son
**no predecibles con ninguna variante** en este dataset (R2 OOF ~0, R2 lockbox muy negativo: peor que
predecir la media) -- no es un problema del balance reductor, es un problema del target multi-trazador en
si (ruido del cierre por diferencia entre escalones consecutivos, ver B-01). Para los 3 targets restantes,
las diferencias entre variantes son **de 0.001-0.002 en R2 OOF**, dentro del ruido de muestreo: ninguna
variante del balance reductor mejora de forma clara sobre (a). El unico movimiento apreciable es en el
lockbox de SDI, donde (b) sube de 0.143 a 0.165 pero (d)/(e) bajan a 0.077-0.097 -- inconsistente entre
variantes, tampoco es una senal limpia.

### 2.2 Tabla theta -- `feo_extraido_est_kg` (Reduccion), variantes (a) y (b)

| variante | palanca | theta (1 sd) | IC95% (1 sd) | signif. | signo teorico | concuerda |
|---|---|---:|---|:---:|:---:|:---:|
| (a) | `tasa_feed_Carbon_kg_min` | +413.7 kg | [-277.6, +1317.2] | No | + | Si |
| (a) | `Cx_av_v4` | -302.0 kg | [-1039.5, +278.6] | No | + | **No** |
| (a) | `tasa_gn_nm3_min` | +56.7 kg | [-39.6, +162.6] | No | + | Si |
| (a) | `exceso_o2_combustion_pct` | -50.4 kg | [-99.9, -13.5] | **Si** | - | Si |
| (b) | `b6_reductor_neto_kg_ceq` | +115.1 kg | [-11.5, +345.5] | No | + | Si |
| (b) | `tasa_gn_nm3_min` | +43.6 kg | [-67.6, +144.2] | No | + | Si |
| (b) | `exceso_o2_combustion_pct` | -40.9 kg | [-93.9, +1.1] | No | - | Si |

El IC de `tasa_feed_Carbon_kg_min` reportado en v4 (+414 [-340,+1298] kg/sd) se reproduce casi exacto
(+413.7 [-277.6,+1317.2]) -- confirma la consistencia del pipeline. `b6_reductor_neto_kg_ceq` SI angosta el
IC de forma sustancial (ancho 357 kg vs 1595 kg de `tasa_feed_Carbon_kg_min` solo, o vs la suma de
`tasa_feed_Carbon_kg_min` + `Cx_av_v4` que estan fuertemente correlacionados entre si y se "pisan" el
efecto uno al otro con signos opuestos -- +414 vs -302, ambos no significativos y de signo contrario en la
teoria). Es una mejora real de precision/identificacion, coherente con la hipotesis de partida, pero **no
alcanza significancia**: el limite inferior del IC queda en -11.5, a un pelo de cruzar cero.

### 2.3 Tabla theta -- `sn_extraido_est_kg` (Reduccion), variantes (a) y (b)

| variante | palanca | theta (1 sd) | IC95% (1 sd) | signif. | signo teorico | concuerda |
|---|---|---:|---|:---:|:---:|:---:|
| (a) | `Cx_sn_v4` | +460.9 kg | [-57.3, +898.8] | No | + | Si |
| (a) | `tasa_gn_nm3_min` | +81.4 kg | [+26.6, +146.6] | **Si** | + | Si |
| (a) | `exceso_o2_combustion_pct` | -0.4 kg | [-16.2, +20.3] | No | - | Si |
| (b) | `b6_reductor_neto_kg_ceq` | **-16.3 kg** | [-231.0, +186.7] | No | + | **No** |
| (b) | `tasa_gn_nm3_min` | +113.8 kg | [+52.9, +179.8] | **Si** | + | Si |
| (b) | `exceso_o2_combustion_pct` | +1.5 kg | [-19.4, +25.6] | No | - | **No** |

Aqui el resultado es claramente desfavorable a la hipotesis: sustituir `Cx_sn_v4` por
`b6_reductor_neto_kg_ceq` **invierte el signo** (de +461 a -16, ambos no significativos, pero el signo
teorico esperado era positivo). Es fisicamente explicable: `b6_reductor_neto_kg_ceq` ya tiene RESTADA la
demanda de Sn disponible (es "lo que sobra despues de servir a Sn"), asi que por construccion contiene
MENOS informacion sobre cuanto Sn hay para reducir que `Cx_sn_v4` (que es directamente carbon x Sn
disponible, la cinetica bimolecular correcta). El balance reductor esta diseñado para capturar el canal de
sobre-reduccion de FeO, no la cinetica primaria de Sn -- usarlo como sustituto en el target de Sn no tiene
soporte teorico y los datos lo confirman.

### 2.4 Tabla theta -- `st_R22_sdi` (Reduccion), variante (b)

| palanca | theta (1 sd) | IC95% (1 sd) | signif. | signo teorico* | concuerda |
|---|---:|---|:---:|:---:|:---:|
| `b6_reductor_neto_kg_ceq` | -0.065 | [-0.247, +0.109] | No | - | Si |
| `tasa_gn_nm3_min` | -0.129 | [-0.259, +0.023] | No | - | Si |
| `exceso_o2_combustion_pct` | +0.002 | [-0.065, +0.066] | No | + | Si |
| `tasa_aire_nm3_min` | +0.046 | [-0.007, +0.104] | No | + | Si |

(*) Signo teorico propio, no literal del enunciado: el SDI premia agotar Sn SIN metalizar Fe, asi que el
canal de "sobra de reductor" (lo que queda tras servir la demanda de Sn) es el mismo canal de
sobre-reduccion que PENALIZA el SDI -> signo negativo esperado para `b6_reductor_neto_kg_ceq` (distinto del
signo positivo esperado sobre FeO en kg bruto). Todos los signos concuerdan, ninguno es significativo. R2
OOF practicamente identico a (a) (0.131 vs 0.131); el R2 lockbox mejora de 0.143 a 0.165 pero (c)/(d)/(e) no
sostienen la mejora (0.132/0.077/0.097) -- no hay una senal consistente entre variantes.

### 2.5 Fusion -- `sn_extraido_est_kg`, (a) v4 vs (b) balance reductor

| variante | n_dev/lb | R2 OOF / lockbox | palanca nueva | theta (1 sd) | IC95% | concuerda |
|---|---|---|---|---:|---|:---:|
| (a) | 1749/374 | 0.7319 / 0.7235 | `relacion_C_Sn_carga` | +51.2 kg | [-84.2, +201.7] | Si (no sig.) |
| (b) | 1749/374 | 0.7319 / 0.7231 | `b6_ratio_reductor_demanda` | -12.3 kg | [-282.4, +222.6] | **No** |
| (b) | | | `b6_reductor_neto_kg_ceq` | +133.8 kg | [-568.4, +921.4] | Si (no sig., IC muy ancho) |

R2 practicamente identico (0.7319 en ambas, cuarta cifra decimal). `b6_ratio_reductor_demanda` sale con
signo contrario al esperado y un IC enorme; `b6_reductor_neto_kg_ceq` tiene el signo correcto pero un IC
~4x mas ancho que el de `relacion_C_Sn_carga` en (a). En Fusion el balance reductor no aporta nada sobre la
palanca v4 existente.

---

## 3. Utilizacion del reductor

### 3.1 Por batch (`sum(b6_c_consumido_teorico_kg) / sum(b6_oferta_reductora_kg_ceq)`), n=361 batches

| fase | media | sd | mediana | IQR | supuesto Rust ("perdido al tiro") | utilizacion Rust implicita |
|---|---:|---:|---:|---|---|---:|
| Fusion | 1.007 | 0.170 | 1.028 | [0.920, 1.121] | 19.5% perdido | ~0.80 |
| Reduccion | 1.025 | 0.354 | 1.045 | [0.866, 1.246] | 48% perdido | ~0.52 |

La utilizacion agregada por batch (y tambien la mediana por escalon de la Seccion 1.3: 0.99 F / 0.91 R)
esta sistematicamente MUY por encima del supuesto del modelo Rust, sobre todo en Reduccion (0.91-1.05
observado vs 0.52 asumido: casi el doble). Interpretacion honesta: `b6_utilizacion_reductor` es un
diagnostico POST-HOC (LEAKAGE) que usa el `sn_extraido_est_kg`/`feo_extraido_est_kg` MEDIDOS para calcular
cuanto carbon "deberia" haberse consumido, y lo compara contra la oferta calculada con los supuestos de
`SUPUESTOS` (CARBONO_FIJO, FE_MINERAL, FE_MET_DROSS). Que la mediana este cerca o por encima de 1 significa
que, con los supuestos actuales, el balance casi no deja margen para carbon "perdido" -- lo opuesto de la
hipotesis Rust de 48% de perdida en Reduccion. Esto puede deberse a (a) que `CARBONO_FIJO`/`FE_MINERAL` de
`SUPUESTOS` subestiman la demanda real (candidato: `FE_MINERAL` en 0.55 podria ser bajo), (b) que el
`c_consumido_teorico` post-hoc (basado solo en Sn/FeO reducidos) no captura toda la demanda real (p.ej. Fe
que se metaliza y luego se re-oxida, o perdidas de carbon que el modelo Rust sí modela por otras vias), o
(c) que el "perdido al tiro" del modelo Rust incluye combustion de carbon que este balance simplificado ya
cuenta como "oferta util" en otro renglon. No se puede dirimir con este experimento solo -- requiere el
balance de masa global (B-04) y el balance de energia (B-03) para cerrar la cuenta de forma independiente.

### 3.2 Regresion OLS HC3 (DEV, n=298) de la utilizacion de Reduccion sobre proxies de arrastre

| variable (z-score) | coef por sd | IC95% | p |
|---|---:|---|---:|
| const | 0.992 | [0.955, 1.030] | <0.001 |
| `tiro_horno_pct` medio | **+0.076** | [0.024, 0.127] | 0.004 |
| `total_process_gas_nm3_min` medio | +0.018 | [-0.029, 0.066] | 0.451 |
| `posicion_vertical_lanza_mm` media | **+0.089** | [0.046, 0.131] | <0.001 |
| `frac_pellets_carga` (Fusion) | **-0.080** | [-0.126, -0.035] | 0.001 |
| `f_polvo` (batch) | **+0.036** | [0.003, 0.068] | 0.034 |
| `f_dross` (batch) | -0.009 | [-0.048, 0.030] | 0.654 |

R2 = 0.124 (R2 ajustado 0.105). Tres coeficientes significativos, pero de lectura ambigua: mas tiro y
lanza mas alta (posicion vertical mayor) se asocian a MAYOR utilizacion (menos "carbon perdido" implicito),
contrario a la intuicion naive de que mas tiro = mas arrastre = mas carbon perdido = MENOR utilizacion.
Mas pellets en Fusion se asocia a MENOR utilizacion en Reduccion (unico signo intuitivo: pellets cambian la
matriz Fe/Sn que llega a Reduccion). Mas polvo del batch (`f_polvo`) se asocia a MAYOR utilizacion, tambien
contra-intuitivo si se esperaba que el polvo fuese "carbon perdido" junto con el Sn. Con R2=0.124 y signos
mixtos, esta regresion NO da una historia causal limpia de arrastre -- es mas probable que estas proxies
esten correlacionadas con el REGIMEN operativo general (velocidad de proceso, intensidad) que con perdidas
fisicas especificas de carbon. Recomendacion: RONDA 2, cruzar con B-03/B-04 antes de interpretar
causalmente.

---

## 4. Sensibilidad a supuestos

Variante (b) recalculada con `bal.SUPUESTOS[...]` mutado uno a la vez (recompute completo de
`bal.construir_features_v6`), targets `feo_extraido_est_kg` y `st_R22_sdi`:

| parametro | valor | target | R2 OOF | R2 lockbox | theta reductor_neto (1sd) | IC95% |
|---|---:|---|---:|---:|---:|---|
| CARBONO_FIJO | 0.60 | feo | 0.3687 | 0.3754 | +160.2 | [-13.7, +453.5] |
| CARBONO_FIJO | 0.75 (base) | feo | 0.3680 | 0.3768 | +115.1 | [-11.5, +345.5] |
| CARBONO_FIJO | 0.85 | feo | 0.3681 | 0.3776 | +105.7 | [-8.0, +320.5] |
| FE_MET_DROSS | 0.20 | feo | 0.3684 | 0.3761 | +127.1 | [-4.0, +370.3] |
| FE_MET_DROSS | 0.30 (base) | feo | 0.3680 | 0.3768 | +115.1 | [-11.5, +345.5] |
| FE_MET_DROSS | 0.40 | feo | 0.3674 | 0.3776 | +95.1 | [-24.4, +321.3] |
| FE_MINERAL | 0.45 | feo | 0.3677 | 0.3771 | +104.3 | [-19.4, +329.9] |
| FE_MINERAL | 0.55 (base) | feo | 0.3680 | 0.3768 | +115.1 | [-11.5, +345.5] |
| FE_MINERAL | 0.65 | feo | 0.3680 | 0.3763 | +115.1 | [-11.6, +346.0] |
| CARBONO_FIJO | 0.60 | sdi | 0.1317 | 0.1626 | -0.121 | [-0.342, +0.068] |
| CARBONO_FIJO | 0.75 (base) | sdi | 0.1310 | 0.1654 | -0.065 | [-0.247, +0.109] |
| CARBONO_FIJO | 0.85 | sdi | 0.1307 | 0.1658 | -0.040 | [-0.194, +0.103] |
| FE_MET_DROSS | 0.20 | sdi | 0.1312 | 0.1645 | -0.073 | [-0.259, +0.096] |
| FE_MET_DROSS | 0.40 | sdi | 0.1309 | 0.1651 | -0.057 | [-0.232, +0.132] |
| FE_MINERAL | 0.45 | sdi | 0.1311 | 0.1656 | -0.068 | [-0.247, +0.103] |
| FE_MINERAL | 0.65 | sdi | 0.1310 | 0.1657 | -0.062 | [-0.247, +0.103] |

**Estable en todos los sentidos relevantes**: el R2 OOF y lockbox no se mueven (variacion de 4a cifra
decimal) y el IC de theta CRUZA CERO en las 16 configuraciones probadas, sin excepcion -- la no
significancia no es un artefacto de los supuestos por defecto, es robusta al rango completo
(CARBONO_FIJO 0.60-0.85, FE_MET_DROSS 0.20-0.40, FE_MINERAL 0.45-0.65). El punto (favorable) tambien es
estable: el signo de theta es siempre el esperado (+ FeO, - SDI) y la magnitud se mueve de forma suave y
monotona (p.ej. CARBONO_FIJO mas alto -> menos "sobra" de reductor marginal -> theta mas chico), sin
cambios de signo ni saltos.

### 4.1 Variante "sin lanza" (`b6_c_fijo_kg + b6_ceq_fe_dross_kg - demanda`, sin el CO de la lanza)

| target | R2 OOF | R2 lockbox | theta reductor (1sd) | IC95% |
|---|---:|---:|---:|---|
| `feo_extraido_est_kg` (con lanza, variante b) | 0.3680 | 0.3768 | +115.1 | [-11.5, +345.5] |
| `feo_extraido_est_kg` (sin lanza) | 0.3673 | 0.3757 | +105.1 | [-39.7, +332.9] |
| `st_R22_sdi` (con lanza, variante b) | 0.1310 | 0.1654 | -0.065 | [-0.247, +0.109] |
| `st_R22_sdi` (sin lanza) | -- | 0.1620 | -0.080 | [-0.252, +0.093] |

Quitar el termino de CO de la lanza (`b6_ceq_lanza_kg`) apenas cambia el R2 (4a cifra decimal) y ensancha
un poco el IC de theta (de 357 a 373 kg para FeO). Esto dice que **el termino de la lanza aporta poco** al
poder del balance reductor en Reduccion: la mediana de `b6_ceq_lanza_kg` en Reduccion es +62 kg (escalon 1)
frente a `b6_c_fijo_kg` mediana ~364-1464 kg segun el escalon -- el carbon solido domina la oferta, el CO
de la lanza es un termino de segundo orden en esta fase (a diferencia de Fusion, donde `b6_ceq_lanza_kg` es
mucho mas negativo, -300 a -811 kg, porque el exceso de O2 alli SI quema carbon activamente).

---

## 5. Curvas de respuesta

Se usa la variante (b) en los 3 modelos (FeO, Sn, SDI) para tener un barrido conjunto y comparable de
`b6_reductor_neto_kg_ceq` (P5..P95 de DEV Reduccion: aprox. -1165 a +481 kg_ceq), con el resto del estado
fijo en la mediana DEV por tramo de avance y las demas palancas (GN, O2, aire) tambien fijas en su mediana
del tramo.

| tramo avance | `b6_reductor_neto_kg_ceq`=P5 (-1165) | =P95 (+481) | Δ FeO pred | Δ Sn pred | Δ J=Sn-0.6FeO | Δ SDI |
|---|---|---|---:|---:|---:|---:|
| <0.84 | FeO=1174, Sn=9865, J=9160, SDI=0.670 | FeO=1525, Sn=9813, J=8898, SDI=0.464 | +351 | -52 | **-263** | **-0.206** |
| 0.84-0.96 | FeO=580, Sn=2471, J=2123, SDI=0.267 | FeO=930, Sn=2419, J=1860, SDI=0.062 | +351 | -52 | **-263** | **-0.206** |
| >0.96 | FeO=281, Sn=317, J=148, SDI=-0.086 | FeO=631, Sn=264, J=-115, SDI=-0.292 | +351 | -52 | **-263** | **-0.206** |

(Nota: el delta P5->P95 es identico en los 3 tramos -- 351/-52/-263/-0.206 -- porque el PLM es
`g(S) + theta*(a - m(S))`: para S fijo, variar solo `a` produce un cambio `theta*(P95-P5)` que NO depende
de S; lo que cambia por tramo es unicamente el nivel base `g(S)`, no la pendiente. Es la misma limitacion
de linealidad senalada abajo.)

**No aparece un optimo interior en ningun tramo**: J y SDI son monotonos DECRECIENTES en todo el rango
barrido de `b6_reductor_neto_kg_ceq`, en los 3 tramos de avance -- el optimo cae siempre en el borde
inferior del rango (P5, el valor MAS negativo/oxidante). Esto es consistente con los theta de la Seccion
2.2-2.4: el efecto sobre FeO es positivo (+115 kg/sd) y sobre Sn es practicamente nulo o levemente negativo
(-16 kg/sd), asi que aumentar el reductor neto solo suma FeO sin compensar con mas Sn, y J = Sn - 0.6*FeO
cae con el. Dos limitaciones honestas: (1) el PLM es LINEAL en cada palanca dado el estado (theta
constante) -- un barrido de una sola palanca lineal estructuralmente NUNCA puede mostrar un maximo interior
salvo que la pendiente cambie de signo dentro del rango, cosa que theta no permite por construccion; el
optimo interior que predice la teoria (cruce de selectividad marginal FeO/Sn, exp 15) tendria que salir de
una INTERACCION explicita reductor_neto x avance (no probada aqui) o de la optimizacion CONJUNTA de varias
palancas con costos, no de un barrido 1-D; (2) dado que theta de Sn sobre `b6_reductor_neto_kg_ceq` no es
significativo y tiene el signo "equivocado" (Seccion 2.3), esta curva de Sn no debe leerse como una
relacion causal validada, sino como la extrapolacion lineal de un efecto estimado con mucha incertidumbre.

---

## 6. Limitaciones

- El diagnostico `b6_utilizacion_reductor` usa los MISMOS targets (`sn_extraido_est_kg`,
  `feo_extraido_est_kg`) que las variables que se intentan predecir en la Seccion 2: no es una validacion
  independiente, es una identidad contable sujeta a los mismos supuestos (`SUPUESTOS`) que la oferta.
- Los targets multi-trazador (`b6_feo_extraido_multi_kg`, `b6_sdi_multi`) no son predecibles con ningun
  conjunto de palancas probado (R2 OOF ~0, lockbox muy negativo): cualquier conclusion sobre el balance
  reductor usando estos targets especificos no es confiable (ver B-01 para el diagnostico de ruido del
  cierre multi-trazador).
- El signo teorico para SDI (Seccion 2.4) es una interpretacion propia, no esta en el enunciado original;
  se documenta explicitamente el razonamiento para que pueda auditarse o revisarse.
- La regresion de utilizacion (Seccion 3.2) tiene R2 bajo (0.124) y no se corrigio por multiplicidad de
  hipotesis (6 proxies); los p-valores individuales deben leerse como exploratorios.
- El PLM es lineal en las palancas: los IC angostos-pero-no-significativos (feo, IC [-11.5,+345.5]) sugieren
  que con mas datos (mas batches) el balance reductor podria cruzar el umbral de significancia para FeO/SDI
  especificamente -- no se descarta un valor futuro como ESTADO o como palanca ADICIONAL (no sustituta) en
  una ronda con mas datos o con una forma funcional que permita interaccion con el avance.

---

## Archivos generados

`b02_descriptiva_por_escalon.csv`, `b02_spearman_por_tramo.csv`, `b02_plm_reduccion_resumen.csv`,
`b02_plm_reduccion_theta.csv`, `b02_plm_fusion_resumen.csv`, `b02_plm_fusion_theta.csv`,
`b02_utilizacion_por_batch.csv`, `b02_ols_utilizacion_reduccion.csv`, `b02_sensibilidad_supuestos.csv`,
`b02_sin_lanza_resumen.csv`, `b02_theta_sin_lanza_feo_extraido_est_kg.csv`,
`b02_theta_sin_lanza_st_R22_sdi.csv`, `b02_curvas_respuesta.csv`.
