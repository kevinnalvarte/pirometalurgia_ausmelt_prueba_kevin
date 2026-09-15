# B-01 -- Cierre multi-trazador de la escoria en Reduccion

Script: `b01_multitrazador.py` (semilla 42). Log completo: `b01_log.txt`. CSVs: `b01_*.csv`.
Runtime total: 198 s. Dataset: 3982 escalones / 362 batches (`bal.cargar_df()`), DEV = 299 batches,
lockbox = 63 batches (`modelo_prescriptivo.split_dev_lockbox`). El lockbox solo se reporta; ninguna
decision de este documento se tomo mirandolo.

Pregunta: en Reduccion (sin carga), el cierre por 4 inertes (CaO, SiO2, Al2O3, MgO,
`bal_features.agregar_multitrazador`) da un cociente de masas m[t]/m[t-1] con menos ruido que el
trazador CaO unico (v3), y eso se traduce en mejores targets (`b6_feo_extraido_multi_kg`,
`b6_sdi_multi`) y en un residuo de masa (`b6_residuo_masa_kg`) con senal fisica.

**Resultado en una linea: el multi-trazador reduce el ruido de ensayo aislado (Monte Carlo), pero en
cada prueba empirica independiente (correlacion con KPI, HGB, PLM, autocorrelacion) es PEOR que el
trazador CaO unico ya adoptado en v3/v4/v5. Se recomienda RECHAZAR su adopcion como target y mantener
CaO. El residuo de masa es RONDA 2 (senal debil y no concluyente).**

---

## 1. Consistencia entre los 4 inertes

957/1448 escalones de Reduccion son validos para el cierre (sin carga, con al menos un inerte medido,
excluye R0). Sesgo y sd de cada inerte respecto al consenso ponderado (`b01_inertes_sesgo.csv`):

| inerte | n | sesgo medio (ln) | sd vs consenso | sd_rel asumida | peso asumido (norm) | peso empirico (norm) |
|---|---|---|---|---|---|---|
| CaO   | 957 | +0.0100 | 0.0850 | 0.020 | 0.296 | 0.245 |
| SiO2  | 957 | -0.0058 | 0.0709 | 0.015 | 0.526 | 0.353 |
| Al2O3 | 957 | +0.0071 | 0.0745 | 0.030 | 0.131 | 0.320 |
| MgO   | 952 | -0.0178 | 0.1472 | 0.050 | 0.047 | 0.082 |

MgO tiene el sesgo mas grande (casi el doble que los demas) y la sd mas alta -- es el inerte menos
fiable, como se sospechaba (aporte de refractario / matriz mas dificil de ensayar en trazas). Pero
**los pesos asumidos (`SD_REL_ENSAYO`) no reflejan bien la precision empirica**: SiO2 esta
sobreponderado (0.526 asumido vs 0.353 empirico) y Al2O3 esta muy subponderado (0.131 asumido vs
0.320 empirico, su sd vs consenso es casi igual a la de SiO2). CaO tambien esta algo sobreponderado.
Una recalibracion empirica de `SD_REL_ENSAYO` (o un esquema IRLS/EM, ya que el consenso actual se
construye con esos mismos pesos) es una propuesta razonable para ronda 2, aunque no resuelve por si
sola el problema de fondo (ver mas abajo).

Correlacion entre pares de inertes (ln-ratio crudo, `b01_inertes_correlacion.csv`):

| | CaO | SiO2 | Al2O3 | MgO |
|---|---|---|---|---|
| CaO | 1.00 | 0.25 | 0.13 | 0.47 |
| SiO2 | | 1.00 | 0.59 | 0.74 |
| Al2O3 | | | 1.00 | 0.46 |
| MgO | | | | 1.00 |

Si la unica fuente de discrepancia fuera ruido de ensayo independiente, las correlaciones entre
inertes deberian ser ~0. Que sean positivas y moderadas (0.13-0.74) confirma que SI existe una senal
real compartida (el evento de reduccion mueve a los 4 juntos), pero la heterogeneidad es marcada:
SiO2-MgO (0.74) y SiO2-Al2O3 (0.59) concuerdan bien entre si, mientras que **CaO es el que menos
concuerda con Al2O3 (0.13)** -- es el "raro" del grupo, compatible con que la cal (unica fuente de
CaO, tolva 6) tenga una dinamica de mezcla propia distinta de los oxidos que vienen de la ganga y el
refractario.

Dispersion ponderada entre trazadores (`b6_dispersion_trazadores`) y sesgo de cada inerte, por orden
de escalon dentro de Reduccion (`b01_dispersion_por_orden.csv`, `b01_inertes_sesgo_por_orden.csv`):

| orden | n | dispersion media | dispersion p50 | sesgo CaO | sesgo SiO2 | sesgo Al2O3 | sesgo MgO |
|---|---|---|---|---|---|---|---|
| 1 (primer escalon valido) | 348 | 0.0373 | 0.0241 | +0.0128 | -0.0056 | +0.0054 | -0.0326 |
| 2 | 353 | 0.0258 | 0.0166 | +0.0104 | -0.0079 | +0.0116 | -0.0030 |
| 3 (ultimo) | 256 | 0.0227 | 0.0133 | +0.0055 | -0.0031 | +0.0030 | -0.0084 |

La dispersion entre trazadores y el sesgo de CaO y de MgO son claramente mas altos en el **primer
escalon valido tras el cierre de Fusion** y decaen hacia el final de Reduccion. Es compatible con la
hipotesis de cal residual / mezcla no homogenea justo despues de que termina la carga (la cal de
tolva 6 y la escoria recien fundida no estan bien mezcladas en R0-R1), mas que con un aporte
acumulativo de refractario (que deberia crecer, no decaer, con el avance). Diagnostico: **si**, hay
un sesgo sistematico por inerte, concentrado en el arranque de Reduccion, y MgO es el inerte menos
fiable en todo el rango.

---

## 2. Ruido: Monte Carlo vs huella empirica

### 2a. Monte Carlo del ensayo (200 replicas, DEV, `b01_ruido_montecarlo.csv`)

Perturbando %CaO/%SiO2/%Al2O3/%MgO (relativo, `SD_REL_ENSAYO`) y %Sn (+-0.05 pt), %FeO (+-0.3 pt):

| version | sd ruido (kg) | sd target (kg) | R2 techo |
|---|---|---|---|
| FeO_ext single (v3) | 374 | 719 | 0.729 |
| FeO_ext multi | 281 | 2170 | **0.983** |
| Sn_ext single (v3) | 56 | 757 | 0.995 |
| Sn_ext multi | 47 | 785 | 0.996 |

Para FeO, el multi-trazador SI baja el ruido absoluto (281 vs 374 kg) y el techo de R2 sube mucho
(0.98 vs 0.73). Pero el techo sube sobre todo porque **la varianza del propio target casi se triplica
(2170 vs 719 kg)**, no solo porque baje el ruido -- es decir, el target multi tiene mucha mas
dispersion "real" (o al menos no-atribuible-a-ruido-de-ensayo-aislado) entre escalones. Esto es
exactamente lo que predice la seccion 1: si los 4 inertes no comparten una senal perfectamente comun
(correlaciones 0.13-0.74, sesgos distintos por orden), el promedio ponderado no es un simple
"promedio de ruido independiente" -- arrastra el desacuerdo entre inertes como varianza adicional que
el Monte Carlo (que solo perturba el ensayo, no el desacuerdo estructural) no captura.

### 2b. Huella empirica: autocorrelacion lag-1 de FeO_ext dentro del batch (`b01_autocorr_lag1.csv`)

| version | n pares | autocorr lag-1 |
|---|---|---|
| single (v3) | 953 | -0.313 |
| multi | 597 | **-0.654** |

Una serie puramente ruidosa, diferenciada (como es FeO_ext = -(inv[t]-inv[t-1])), tiene autocorrelacion
lag-1 teorica ~-0.5. El trazador **single (-0.31) esta MENOS cerca de ese limite de ruido puro** --
conserva mas momentum/senal real entre escalones consecutivos -- mientras que **multi (-0.65) esta
MAS cerca (incluso mas negativo) que el limite de ruido puro**, es decir, se comporta como una serie
mas dominada por ruido/oscilacion espuria que el propio trazador que se queria mejorar.

**Conclusion de la seccion 2**: el Monte Carlo aislado (que solo mide precision de ensayo) sugiere que
multi es mejor; la huella empirica (autocorrelacion) dice lo contrario. Se prioriza la huella empirica
porque es una medicion directa sobre los datos reales, no una simulacion de un solo mecanismo de
ruido.

---

## 3. Targets de batch: multi vs single

`b01_targets_correlaciones.csv`, `b01_targets_ols.csv`, `b01_targets_tercios.csv`,
`b01_targets_dosis_respuesta.csv`, `b01_w_sweep_sdi.csv`, `b01_single_vs_multi_ranking.csv`.

### Spearman vs `rendimiento_proxy_batch` (IC bootstrap 500, BH dentro de familia)

| target | version | DEV rho | DEV IC95% | p_bh DEV | lockbox rho | total rho |
|---|---|---|---|---|---|---|
| feo_ext | single | **0.274** | [0.149, 0.383] | <0.001 | -0.136 | **0.205** |
| feo_ext | multi | 0.150 | [0.043, 0.251] | 0.013 | -0.206 | 0.089 (ns) |
| ln_feo_ret | single | **0.315** | [0.206, 0.409] | <0.001 | -0.061 | **0.244** |
| ln_feo_ret | multi | 0.194 | [0.089, 0.301] | 0.002 | -0.167 | 0.133 |
| sdi | single | **0.319** | [0.201, 0.409] | <0.001 | 0.094 | **0.280** |
| sdi | multi | 0.230 | [0.117, 0.329] | <0.001 | -0.053 | 0.185 |
| sn_ext | single | 0.021 (ns) | | 0.71 | 0.044 | 0.047 (ns) |
| sn_ext | multi | 0.052 (ns) | | 0.42 | 0.037 | 0.065 (ns) |

En los 3 targets con senal real (feo_ext, ln_feo_ret, sdi) **single gana a multi en DEV con margen
claro** (rho ~0.27-0.32 vs ~0.15-0.23) y tambien en la muestra total. En lockbox (n=63) ambas
versiones son inestables y de signo variable -- consistente con la fragilidad de lockbox ya conocida
para estos targets (iteracion 5), no evidencia a favor de ninguna version. `sn_ext` es debil en
ambas versiones (coherente con hallazgos previos: el Sn extraido en Reduccion por si solo no
correlaciona con el KPI).

### Tercios cronologicos de DEV

Single es mayor o igual que multi en 7 de 9 comparaciones (3 targets x 3 tercios); solo `sdi` en el
primer tercio favorece a multi (0.224 vs 0.191, ambos debiles). El patron no es un artefacto de un
periodo particular: es consistente en el tiempo.

### OLS HC3 (controles + tendencia, DEV)

| target | version | coef (1sd) | IC95% | p | R2 ajustado |
|---|---|---|---|---|---|
| feo_ext | single | **1.515** | [0.898, 2.132] | <0.001 | 0.091 |
| feo_ext | multi | 0.703 | [0.222, 1.184] | 0.004 | 0.025 |
| sdi | single | **1.515** | [0.906, 2.124] | <0.001 | 0.123 |
| sdi | multi | 1.103 | [0.637, 1.569] | <0.001 | 0.073 |
| ln_feo_ret | single | **1.370** | [0.792, 1.948] | <0.001 | 0.105 |
| ln_feo_ret | multi | 0.880 | [0.416, 1.345] | <0.001 | 0.050 |
| sn_ext | single | 0.010 | [-0.597, 0.617] | 0.97 | 0.001 |
| sn_ext | multi | 0.463 | [-0.040, 0.966] | 0.071 | 0.010 |

Controlando por `feed_sn_total_t`, `ley_sn_conc_batch_pct`, `espesor_ladrillo_norm_mm`,
`frac_carga_secundaria_F`, `feed_dross_Fe_total_t` y tendencia cronologica: single duplica
aproximadamente el coeficiente y el R2 ajustado de multi en los 3 targets con senal.

### Dosis-respuesta por quintiles (DEV, media de `rendimiento_proxy_batch`)

| quintil | sdi single | sdi multi | feo_ext single | feo_ext multi |
|---|---|---|---|---|
| 1 | 66.5 | 67.5 | 67.5 | 68.5 |
| 5 | 70.9 | 69.9 | 71.1 | 70.5 |
| rango (Q5-Q1) | **4.4** | 2.5 | **3.6** | 2.0 |

El gradiente Q1->Q5 es ~1.7-1.8x mas empinado en single que en multi para ambos targets: single
discrimina mejor entre batches buenos y malos.

### Barrido de w (SDI multi, solo DEV)

rho maximo = 0.2315 en w=11; el minimo w que alcanza 95% del maximo es **w=6** (rho=0.224); w=10
(el valor v3 adoptado en iteracion 5) da rho=0.2296, esencialmente igual al maximo. El barrido con
multi-trazador no sugiere cambiar w -- **si se usara multi, w=10 seguiria siendo razonable**, pero el
techo de rho (0.23) sigue siendo inferior al de single (0.32 con el mismo w).

### Correlacion single vs multi (cambio de ranking de batches)

| target | n | rho |
|---|---|---|
| sdi | 348 | 0.580 |
| ln_feo_ret | 348 | 0.599 |
| feo_ext | 360 | 0.550 |
| sn_ext | 360 | 0.406 |

Las dos versiones comparten solo ~55-60% de concordancia de rango (rho de Spearman, no R2): el
multi-trazador reordena una fraccion sustancial de los batches respecto al trazador CaO -- no es un
simple "mismo ranking con menos ruido", es una senal genuinamente distinta (y, por las secciones 2 y 4,
peor).

---

## 4. Predictibilidad por escalon (HGB) y PLM v4

`b01_predictibilidad_escalon.csv`, `b01_plm_theta.csv`, `b01_plm_r2.csv`. Set A = STATE+CONTEXT V3
Reduccion seguras (37 features); set B = A + CONTROL (44: primitivas de accion + `exceso_o2_combustion_pct`
+ `Cx_sn_v4` + `Cx_av_v4`). Filas restringidas a donde AMBAS versiones son validas simultaneamente.

### HGB OOF GroupKFold5 (DEV) + lockbox

| target | version | n_dev | R2 OOF set A | R2 OOF set B | R2 lockbox set B |
|---|---|---|---|---|---|
| feo_ext | single | 769 | **0.221** | **0.239** | 0.019 |
| feo_ext | multi | 769 | 0.074 | 0.110 | -0.038 |
| sdi | single | 745 | **0.242** | **0.250** | -0.240 |
| sdi | multi | 745 | 0.056 | 0.073 | -0.135 |

Single es 2-4x mas predecible en DEV (criterio de seleccion) para ambos targets. En lockbox, sdi es
negativo en ambas versiones (fragilidad ya documentada de SDI fuera de DEV); feo_ext single se
mantiene positivo (aunque debil) y multi se vuelve negativo.

### PLM v4 (estado + palancas de `("Reducción","feo_kg")`: Carbon, Cx_av_v4, GN, exceso O2, aire)

| target | version | n_dev | R2 OOF interno | R2 lockbox |
|---|---|---|---|---|
| feo_ext | single | 1045 | **0.367** | **0.378** |
| feo_ext | multi | 758 | -0.034 | -1.044 |
| sdi | single | 1015 | **0.131** | **0.143** |
| sdi | multi | 727 | 0.022 | -0.398 |

El PLM de multi para feo_ext es peor que predecir la media (R2 negativo en DEV y catastrofico en
lockbox); single mantiene R2 ~0.37-0.38 consistente DEV/lockbox.

### Pregunta clave: ¿el multi-trazador estrecha el IC del carbon?

Ancho de IC95% (`ci_hi - ci_lo`) de las palancas de carbon:

| target | palanca | ancho IC single | ancho IC multi | razon multi/single |
|---|---|---|---|---|
| feo_ext | `tasa_feed_Carbon_kg_min` | 72.5 | 445.6 | **6.1x mas ancho** |
| feo_ext | `Cx_av_v4` | 77.8 | 441.5 | **5.7x mas ancho** |
| sdi | `tasa_feed_Carbon_kg_min` | 0.086 | 0.272 | **3.2x mas ancho** |
| sdi | `Cx_av_v4` | 0.107 | 0.273 | **2.5x mas ancho** |

**No, al contrario: el multi-trazador ENSANCHA el IC del carbon entre 2.5x y 6x.** Solo
`exceso_o2_combustion_pct` sale significativo, y unicamente en el modelo single de feo_ext
(theta=-4.66, IC[-9.18,-0.46], signo coherente con la teoria). Ningun theta de multi es significativo
en ninguno de los 4 ajustes, y varios cambian de signo respecto a la teoria (`Cx_av_v4` sale negativo
en feo_ext single tambien, contrario a la teoria -- unico punto donde single tampoco concuerda).

---

## 5. Residuo de masa (`b6_residuo_masa_kg`): senal o ruido

`b01_residuo_por_orden.csv`, `b01_residuo_correlaciones_escalon.csv`,
`b01_residuo_correlaciones_batch.csv`.

| orden | n | media (kg) | sd (kg) | p50 (kg) |
|---|---|---|---|---|
| 1 | 348 | +1080 | 11241 | +367 |
| 2 | 353 | -384 | 9215 | 0 |
| 3 | 256 | -11 | 2417 | -7 |

El residuo es grande, muy disperso y sesgado positivo justo en el primer escalon valido (mismo
patron que el sesgo de CaO/MgO de la seccion 1) y colapsa a ~0 con sd mucho menor hacia el final de
Reduccion -- de nuevo compatible con un artefacto de transicion Fusion->Reduccion, no con una fuga
continua de masa.

Spearman con controles del proceso (escalon, DEV, n~776-778):

| variable | rho | p |
|---|---|---|
| `tasa_gn_nm3_min` | +0.157 | <0.001 |
| `total_process_gas_nm3_min` | +0.133 | <0.001 |
| `posicion_vertical_lanza_mm` | +0.042 | 0.24 (ns) |
| `temperatura_horno_celsius_prev` | +0.047 | 0.19 (ns) |
| `tiro_horno_pct` | -0.014 | 0.70 (ns) |
| `b6_dispersion_trazadores` | +0.0002 | 0.996 (ns) |

Hay una asociacion debil pero significativa con el gas total y con GN (mas gas -> mas residuo
positivo), compatible con arrastre/turbulencia inducida por el gas de lanza (splash). Pero la
correlacion con `b6_dispersion_trazadores` es practicamente nula: si el residuo fuera solo ruido de
ensayo heredado del desacuerdo entre trazadores, deberia correlacionar con la dispersion, y no lo
hace -- el residuo no es simplemente "ruido de trazador" disfrazado.

A nivel batch (suma del residuo, DEV, n=297), sin embargo, no hay senal:

| kpi | rho | p |
|---|---|---|
| `f_polvo` | -0.062 | 0.29 (ns) |
| `sn_polvo_t` | -0.066 | 0.26 (ns) |
| `rendimiento_proxy_batch` | +0.054 | 0.36 (ns) |

**Conclusion mixta**: a nivel escalon hay un candidato debil de senal fisica (gas/turbulencia) que no
es pura dispersion de trazador, pero es demasiado debil (rho~0.13-0.16) para ser accionable, y no
sobrevive a la agregacion por batch (no correlaciona con polvo ni con el KPI). Combinado con el
patron de "artefacto de arranque" (orden 1), la explicacion mas parsimoniosa es una mezcla de ruido
de transicion Fusion->Reduccion con una senal fisica de gas/splash demasiado debil para usarse sola.

---

## Limitaciones

- El Monte Carlo de la seccion 2a solo perturba precision de ensayo (independiente por inerte); no
  modela el desacuerdo estructural/sesgo compartido documentado en la seccion 1 -- por eso su techo de
  R2 es optimista para multi. Se prioriza la evidencia empirica (autocorrelacion, HGB, PLM, correlacion
  con KPI) sobre el techo de Monte Carlo.
- Los pesos de `SD_REL_ENSAYO` (2%/1.5%/3%/5%) son supuestos de planta, no medidos en este dataset; la
  seccion 1 muestra que difieren bastante de la precision empirica (especialmente Al2O3, subponderado).
- El PLM y el HGB de multi tienen menos filas validas que single en algunas comparaciones (requieren
  los 4 inertes simultaneamente disponibles), lo que reduce potencia estadistica ademas de reducir
  senal -- aunque la seccion 4 ya restringe a filas donde ambas versiones son validas para el HGB, el
  PLM (por diseno de `ModeloPLM.fit`, que no admite ese cruce facilmente) usa el dropna propio de cada
  version y por tanto compara n distintos (1045 vs 758 para feo_ext); no se espera que esto explique
  una diferencia de la magnitud observada (R2 pasando de +0.37 a -0.03).
- El residuo de masa depende de los mismos supuestos de `bal.SUPUESTOS` (constante 1.135 = M_SnO/M_Sn
  para el canal de Sn); no se hizo sensibilidad a esa constante en este experimento.
- n=63 en lockbox es pequeno; las correlaciones alli cambian de signo entre versiones y no deben
  usarse para elegir nada (regla del proyecto), solo se reportan.

## Recomendaciones

**(i) Multi-trazador como estimador de masa en Reduccion: RECHAZAR.**
El techo de ruido por Monte Carlo mejora (R2_max 0.73->0.98 en FeO), pero toda la evidencia empirica
independiente (autocorrelacion lag-1 mas negativa que el limite de ruido puro, correlacion con KPI
sistematicamente menor en DEV y total, R2 OOF de HGB 2-4x menor, PLM con R2 negativo/catastrofico y
IC del carbon 2.5-6x mas anchos) apunta en la direccion contraria: el multi-trazador no reduce el
ruido relevante, agrega varianza no explicada (seccion 1: los 4 inertes no comparten una senal
limpia, especialmente CaO vs Al2O3 con rho=0.13). Mantener `masa_escoria_est_kg`/CaO unico como en
v3/v4/v5.

**(ii) `b6_sdi_multi` y `b6_feo_extraido_multi_kg` como targets: RECHAZAR.**
En cada bateria (Spearman+IC+BH, tercios, OLS HC3+tendencia, dosis-respuesta, HGB OOF, PLM) las
versiones multi quedan por debajo de `st_R22_sdi`/`st_R02_feo_ret` (single, ya adoptados en la
iteracion 5). El barrido de w confirma que w=10 seguiria siendo razonable si se usara multi, pero no
compensa el deficit de senal. No sustituir los targets ganadores de la iteracion 5.

**(iii) Residuo de masa como feature/diagnostico: RONDA 2 (inconcluso, mas cerca de RECHAZAR como
feature).**
Hay un candidato debil de senal fisica a nivel escalon (correlacion positiva significativa pero baja,
rho~0.13-0.16, con `tasa_gn_nm3_min`/`total_process_gas_nm3_min`, y NO con la dispersion entre
trazadores, lo que descarta que sea solo ruido de ensayo), pero no sobrevive a la agregacion por batch
(sin correlacion con `f_polvo`, `sn_polvo_t` ni `rendimiento_proxy_batch`). El patron dominante es un
artefacto de transicion en el primer escalon valido tras el cierre de Fusion (coincide con el mayor
sesgo de CaO/MgO de la seccion 1), mas compatible con mezcla incompleta que con arrastre continuo. No
adoptar como feature de prediccion; podria explorarse en ronda 2 como bandera de QC (marcar
batches/escalones con residuo grande en el arranque de Reduccion para revision), pero eso es un uso
distinto (auditoria, no prediccion/prescripcion).

## Propuesta ronda 2 (si se retoma esta linea)

1. Recalibrar `SD_REL_ENSAYO` con un esquema IRLS/EM (o pesos leave-one-out) en vez de los supuestos
   de planta, y repetir la bateria de las secciones 3-4 -- dado que Al2O3 esta claramente
   subponderado y CaO algo sobreponderado.
2. Probar el consenso excluyendo CaO (el inerte con menor correlacion promedio con los otros 3,
   rho=0.13 con Al2O3) para ver si SiO2+Al2O3+MgO solos (que si co-varian bien entre si, 0.46-0.74)
   dan un mejor techo empirico que el promedio de los 4.
3. Verificar si el desacuerdo entre inertes (o el residuo de masa) correlaciona con
   `espesor_ladrillo_norm_mm` (proxy de desgaste de refractario) a traves de batches -- no se probo
   en este experimento por alcance/tiempo.
4. Si se explora el residuo como bandera de QC, excluir explicitamente el primer escalon valido de
   Reduccion (orden=1) del calculo, dado el patron de artefacto de arranque documentado aqui.
