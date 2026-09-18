# E12-03 — beta variable en el tiempo por filtro de Kalman (H3)

Filtro propio en numpy (`e12_03_lib.py`): estado Z_b=[1,controles,xG,xC], beta_b=beta_{b-1}+eta_b (paseo aleatorio,
Q diagonal). Inicializacion "diffuse-to-proper": OLS en los primeros 40 batches da beta0,P0 (evita saltos del
prior difuso). Score prospectivo = beta_{G,b|b-1}*xG_b + beta_{C,b|b-1}*xC_b, un-paso-adelante puro. Misma tabla,
controles y evaluacion que E11-07 (`stats_final`, ALL/DEV/lockbox, batches desde el 100).

## 1. Hiperparametros

Variante (i) controles+intercepto FIJOS (Q=0 -> equivalentes a OLS recursivo/expansivo dentro del mismo filtro).
Variante (ii) controles con paseo lento (q_ctrl>0): **empeora monotonamente** t_ALL al crecer q_ctrl (1.61 en
q_ctrl=0 -> 0.80 en 0.0002 -> 0.60 en 0.0005 -> 0.44 en 0.001) — se descarta, (i) es preferible.

(q_G,q_C,R) para xG,xC se fijaron por verosimilitud predictiva un-paso-adelante en batches [40,100) (fallback
explicito del diseno por costo de permutar la seleccion). Resultado: **q_G=0, q_C=0** — la rejilla no detecta
deriva en el arranque de campana (los sub-OLS en [0,100) muestran movimiento de beta pero con SE tan grandes,
0.6-1.7, que no son distinguibles de beta constante). Esto colapsa el filtro "fijo" a un **OLS recursivo/expansivo
puro** para xG,xC tambien. Una variante **adaptativa** (re-selecciona q_G,q_C cada 20 batches por verosimilitud en
todo el pasado ya observado, rejilla 6x6, R fijo) sí encuentra q_G hasta 0.05 y q_C hasta 0.15 en tramos
posteriores — mas fiel al espiritu de H3 pero ~2.3s/corrida (no permutable a >=1000 reps en presupuesto razonable;
se reporta solo puntual, sin permutacion, tal como permite el diseno cuando la reseleccion no es barata).

## 2-3. Prueba prospectiva y comparacion (`e12_03_comparacion.csv`, `e12_03_scores.csv`)

| Variante | ALL t (p) | DEV t | Lockbox t (pendiente) | Spearman ALL |
|---|---|---|---|---|
| Kalman continuo (batch a batch) | 1.61 (0.054) | 2.24 | **-1.35 (-1.05)** | 0.126 |
| Kalman cada 10 (latencia = v9) | 1.64 (0.051) | 2.33 | **-1.48 (-1.13)** | 0.122 |
| Kalman adaptativo cada 20 | 1.79 (0.037) | 2.05 | -0.41 (-0.31) | 0.143 |
| **W=100 duro (referencia v9)** | **2.32 (0.010)** | 2.20 | **+0.78 (+0.37)** | 0.154 |
| Pond. exponencial semivida=60 | 1.96 (0.025) | 2.29 | -0.80 (-0.62) | 0.145 |
| Expansiva pura | 1.42 (0.078) | 2.22 | -1.04 (-0.81) | 0.110 |

Permutacion en bloques de 20 (1200 rep., hiperparametros FIJOS re-ejecutando todo el filtro sobre xG,xC
permutadas — `e12_03_02_permutacion.py` → `e12_03_permutacion.csv`): continuo t p=0.041, Spearman p=0.024;
cada-10 t p=0.035, Spearman p=0.027. **Ambos por encima del umbral p<0.01** de suficiencia y muy por encima del
p=0.0065 de W=100.

**Hallazgo central**: TODAS las variantes que actualizan de forma continua o suave (Kalman en cualquier version,
semivida, expansiva) dan **pendiente negativa en el lockbox** (los ultimos 63 batches); solo la ventana dura
W=100, que se re-ancla en bloques discretos de 10 sin arrastrar momentum del tramo medio de campana, mantiene
lockbox no-negativo. Esto coincide con el quiebre marginal de la seccion 5 (tau*≈idx 295-300, justo el limite
DEV/lockbox): parece haber un cambio de regimen leve al final de la campana que penaliza a los estimadores que
"siguen de cerca" el pasado reciente y favorece al que se reinicia cada 10 batches con una ventana fija de 100.

## 4. Calibracion y abstencion (`e12_03_calibracion.csv`, `e12_03_recomendacion.csv`)

Innovaciones estandarizadas z_b=v_b/sqrt(F_b) (batches >=100, n=261, filtro continuo fijo): media 0.168 (debiera
ser 0), varianza 1.275 (debiera ser 1 → el filtro es ~13-27% **demasiado confiado**), cobertura empirica de
±1.96·sd = **90.0%** (nominal 95%), Ljung-Box (h=5) p=0.0001 — **autocorrelacion residual significativa**
(lag1=0.125, lag3=0.253). **El IC del filtro NO esta bien calibrado**: subestima la incertidumbre real y deja
dependencia serial sin capturar (coherente con la autocorrelacion de corto alcance ya documentada en E11-07,
placebos lead-1).

Regla de abstencion |beta_j|/sd_j >= umbral (1, 1.5, 2), magnitud=min(1,|t|/2): la palanca G se "recomienda" en
85-96% de los batches evaluados (cae poco con el umbral); la palanca C se recomienda en solo 32% (umbral 1),
15% (1.5) y **0%** (umbral 2) — el carbon casi nunca alcanza significancia individual, coherente con "xC sola"
no significativa en E11-07. **Gatear EMPEORA el ajuste prospectivo** en vez de mejorarlo: t_ALL cae de 1.61
(sin gate) a 1.29 (umbral 1), 1.06 (1.5) y 0.91 (2.0) — con esta especificacion la abstencion no filtra ruido,
filtra tambien senal util (la sd del filtro, ya sobre-confiada, penaliza de mas a C).

## 5. Trayectoria suavizada y quiebre (`e12_03_trayectoria.csv`, `e12_03_chow.csv`, `e12_03_chow_curva.csv`)

Con los hiperparametros fijos del burn-in (q=0) el suavizador RTS colapsa a una recta (sin variacion) — no sirve
para describir. Para la seccion descriptiva se usaron hiperparametros elegidos por verosimilitud en TODA la
muestra (q_G=0.02, q_C=0.002; NO se usan en la prueba prospectiva causal): la trayectoria de beta_G_suavizado
baja de -1.3 (inicio) a un minimo ≈-1.8 hacia idx 120-160 y sube a +1.0 al final; beta_C_suavizado sube de ≈0.94 a
≈1.28, mas plano que en la ventana movil de E11-07 (por el suavizado con q_C pequeno). **La deriva es
predominantemente gradual** (compatible con paseo aleatorio), sin escalon abrupto claro en la trayectoria misma.

Prueba sup-F/Chow (rejilla tau cada 5 batches, 60..n-60; interaccion xG*post, xC*post; permutacion en bloques de
20, 1500 rep.): sup-F=5.69 en tau*=idx 295 (batch AP0297), **p=0.060** — marginal, no significativo a 0.05, pero
la localizacion del maximo coincide con el limite DEV/lockbox y con el deterioro de todos los estimadores suaves
en esa zona (seccion 3). No hay evidencia fuerte de quiebre discreto, pero tampoco se descarta un cambio leve de
regimen justo donde el lockbox comienza.

## 6. Veredicto

**El filtro de Kalman NO sustituye a la ventana dura W=100 en el asesor**, con esta implementacion:

- Precision nominal del beta: el Kalman fijo (recursivo/expansivo) da sd promedio menor que W=100 (sd_G 0.51 vs
  0.77; sd_C 0.74 vs 1.42, ≈35-48% mas ajustado) — pero la calibracion (seccion 4) muestra que esa precision es
  ~13-27% optimista, y sobre todo el punto estimado (con q≈0) queda "pegado" al pasado acumulado, sin rastrear
  bien la deriva de fin de campana que la ventana de 100 sí captura localmente.
- Evidencia prospectiva: ninguna variante Kalman (fija ni adaptativa) alcanza p<0.01 corregido ni lockbox
  no-negativo simultaneamente (criterios 1 y 3 de suficiencia); W=100 sigue siendo la unica especificacion que
  cumple ambos en este conjunto de comparaciones.
- La version **adaptativa** (re-estimar q_G,q_C cada 20 batches con el pasado) es la mas prometedora del grupo
  Kalman (t_ALL 1.79, lockbox -0.41, el menos malo) pero no se pudo validar con permutacion completa por costo
  (~2.3 s/corrida) y aun asi no alcanza a W=100.
- Aporte real de H3: el suavizador da una trayectoria descriptiva limpia (deriva gradual, no ruido puro) y el
  quiebre marginal en idx≈295-300 es una pista util — no para reemplazar W=100, sino para investigar que cambia
  en la planta justo antes del lockbox (candidato a explicar por que TODOS los estimadores que "siguen de cerca"
  fallan ahi mientras el reinicio discreto no).

**Recomendacion**: mantener W=100 (paso 10) como especificacion de produccion; no adoptar el Kalman de esta
iteracion. Si se retoma H3, priorizar (a) una seleccion de hiperparametros verdaderamente adaptativa y barata
(p.ej. rejilla mas chica, o formula cerrada tipo EWMA-RLS) que sea permutable a >=1000 reps, y (b) investigar
directamente el quiebre marginal cerca de idx 295-300 antes de confiar en cualquier estimador continuo ahi.
