# Hallazgos del EDA — Horno Ausmelt Sn (Lingo Smelter)

Este documento resume el conocimiento nuevo generado en el EDA diferencial/gradiente realizado en
`analisis_lingo_smelter.ipynb`, orientado a identificar relaciones pirometalúrgicas (estequiometría,
combustión, redox, ácido-base) entre lo que se consume en cada escalón de un batch y la respuesta del
horno/escoria en ese mismo escalón.

## 1. Estructura del dataset

- `df`: 3982 filas (escalones), 362 batches.
- **Todo batch tiene siempre 7 escalones de Fusión + 4 de Reducción** (estructura fija).
- `fecha_final` de un escalón coincide exactamente con `fecha_inicio` del escalón siguiente (mismo batch).
- `delta_tiempo` (duración del escalón, minutos): Fusión media ≈44 min (25-188), Reducción media ≈22 min
  (5-125).
- Se construyó `orden_escalon_fase` (posición 0-6 en Fusión, 0-3 en Reducción) que permite **alinear
  escalones equivalentes entre distintos batches** (útil para comparación funcional/curvas).
- Único KPI de negocio disponible: `rendimiento_proxy_batch` (uno por batch, a maximizar).

## 2. Hallazgo clave: `feed_*` y `volumen_*` NO son acumulados desde el inicio del batch

El diccionario de datos (`diccionario_datos.json`) describe columnas como `feed_Sn_kgf`, `feed_Fe_kgh`,
etc. como **"masa acumulada"**, lo que sugeriría un contador creciente desde el inicio del batch (y que
habría que aplicar `diff()` para obtener el consumo de cada escalón). La evidencia empírica **contradice
esa lectura literal**:

- **Tasa implícita estable:** `feed_total_kgh / delta_tiempo` se mueve en un rango físicamente razonable
  (~0-770 kg/min, mayor masa concentrada en 390-425 kg/min en Fusión). Si el valor fuera acumulado desde
  el inicio del batch, este cociente crecería sin límite a medida que avanzan los escalones.
- **Correlación orden-de-escalón vs. tasa (controlando por fase):** dentro de Fusión la correlación es
  apenas -0.21 (leve caída natural de ritmo al final de la fusión), no el ≈+1 esperado de un contador
  acumulado creciente.
- **Magnitud tras una caída:** cuando el valor baja de un escalón a otro *dentro de la misma fase*, el
  valor posterior no cae a ~0 (mediana ≈0.38× la mediana de esa fase, con casos incluso por encima de la
  mediana) — consistente con **fluctuación natural de una tasa de alimentación por escalón**, no con un
  acumulador que se reinicia.
- **Nombres originales de columnas** (antes del renombrado): *"Gas Natural total (Nm3)"*, *"O2 total
  (Nm3)"*, *"Aire total (Nm3)"*, *"TMH tolva: 3+4+5+7+1 (Kg)"*, *"CaO total (kg)"*, *"Carbon total F+R
  (Kg)"* — todas usan la palabra **"total"**, consistente con "total alimentado/inyectado **durante ese
  escalón**" (tal como el propio diccionario aclara explícitamente para `volumen_*`: *"durante el
  escalón"*), y no con un acumulado histórico del batch.

**Conclusión operativa:** `feed_*` y `volumen_*` ya representan, para cada fila, la masa/volumen de
reactivo consumido **en ese escalón**. No se les aplica `diff()`. En cambio, sí tiene sentido diferenciar
las variables de **estado** del horno/escoria (leyes, temperaturas, presiones), porque esas sí son
mediciones puntuales (o resultado de ensayo de laboratorio) tomadas al cierre de cada escalón, y su cambio
de un escalón a otro refleja el efecto neto de las reacciones ocurridas con los reactivos consumidos en
ese mismo escalón. El `diff()` de estado **no se resetea** en la transición Fusión→Reducción: esa
transición es, de hecho, la señal más informativa (ver sección 4).

Afecta a: `feed_Sn_kgf`, `feed_Fe_kgh`, `feed_dross_Fe_kgh`, `feed_CaO_kgh`, `feed_Carbon_kgh`,
`feed_total_kgh`, `volumen_gas_natural_inyectado_lanza_escalon_nm3`,
`volumen_o2_inyectado_lanza_escalon_nm3`, `volumen_aire_inyectado_lanza_escalon_nm3`.

## 3. Features construidas

### a) Tasas por minuto (normalización por duración del escalón)
$$ \text{tasa}_x = \frac{x_{\text{escalón}}}{\Delta t_{\text{escalón}}} \quad \left[\frac{kg}{min}\ \text{o}\ \frac{Nm^3}{min}\right] $$
Aplicado a los 9 `feed_*`/`volumen_*` listados arriba (ej. `tasa_feed_Carbon_kg_min`, `tasa_o2_nm3_min`).

### b) Gradientes de variables de estado
Para cada variable de estado $y$ (leyes de escoria, temperatura, presiones, tiro, posición de lanza):
$$ \Delta y_i = y_i - y_{i-1} \qquad \text{grad}(y)_i = \frac{\Delta y_i}{\Delta t_i} $$
donde $i-1$ es el escalón inmediatamente anterior **del mismo batch** (columnas `d_<var>` y
`grad_<var>`). El primer escalón de cada batch queda en `NaN` (no hay escalón previo *de ese batch*).

### c) Basicidad de la escoria
$$ B_2 = \frac{\%CaO}{\%SiO_2} \qquad\qquad B_4 = \frac{\%CaO + \%MgO}{\%SiO_2 + \%Al_2O_3} $$
Controla viscosidad y capacidad de fundente; en un sistema fayalítico FeO–SiO₂–CaO regula cuánto SnO₂
queda atrapado en la escoria.

### d) Balance estequiométrico de combustión en la lanza (proxy redox)
Asumiendo combustión de gas natural como metano puro:
$$ CH_4 + 2\,O_2 \rightarrow CO_2 + 2\,H_2O $$
$$ O_{2,\text{teórico}} = 2 \cdot V_{GN} \qquad O_{2,\text{disponible}} = V_{O_2} + 0.21 \cdot V_{aire} $$
$$ \text{exceso\_O}_2\,(\%) = 100 \times \frac{O_{2,\text{disponible}} - O_{2,\text{teórico}}}{O_{2,\text{teórico}}} $$
Positivo → atmósfera oxidante (excedente de O₂ más allá de lo necesario para quemar el gas natural,
disponible para oxidar el baño); negativo → atmósfera reductora (déficit de O₂, se generan CO/H₂ que
reducen SnO₂/FeO del baño). Columna: `exceso_o2_combustion_pct`.

### e) Relaciones reductor/carga y fundente/carga
$$ r_{C/carga} = \frac{feed_{Carbon}}{feed_{Sn} + feed_{Fe} + feed_{drossFe}} \qquad
   r_{CaO/carga} = \frac{feed_{CaO}}{feed_{Fe} + feed_{drossFe}} $$
`relacion_C_carga` **sólo es interpretable en Fusión** (mediana ≈0.17, rango físico razonable). En
Reducción casi no se alimenta mineral/dross fresco (denominador ≈0), por lo que el ratio se dispara a
valores sin sentido (hasta 7.8e8); ahí el indicador relevante de intensidad reductora es directamente
`tasa_feed_Carbon_kg_min`.

## 4. Relaciones pirometalúrgicas encontradas (evidencia empírica)

- **`exceso_o2_combustion_pct` separa nítidamente el régimen operativo:** Fusión mediana ≈+18.9%
  (oxidante, distribución estrecha), Reducción mediana ≈-4.8% (reductor, distribución más ancha) —
  validación numérica y visual (boxplot) fuerte del proxy redox.
- **Estequiometría de reducción de casiterita** ($SnO_2 + 2C \rightarrow Sn + 2CO_2$) visible en los
  datos: `tasa_feed_Carbon_kg_min` correlaciona (Spearman) **-0.76 con `Δ%Sn escoria` en Reducción** y
  **-0.33 en Fusión**. `tasa_gn_nm3_min`/`tasa_o2_nm3_min` también correlacionan fuerte (-0.68) con esa
  caída en Reducción, consistente con que la lanza aporta calor/CO reductor que sostiene la cinética.
- **El signo de `Δ%Sn escoria` está dominado por la fase**: positivo en Fusión (la escoria se está
  formando y concentrando SnO₂ desde la carga), negativo en Reducción (el carbón reduce el SnO₂ de vuelta
  a Sn metálico). Dentro de cada fase, a mayor tasa de carbón, mayor la magnitud del cambio.
- **Batch de ejemplo (AP0001):** el mayor salto de consumo de carbón (primer escalón de Reducción,
  ~63 kg/min) coincide exactamente con el cambio abrupto de exceso de O₂ (de +19% a -6%) y con la mayor
  caída de %Sn en escoria de todo el batch (Δ≈-13.7 puntos porcentuales) — ilustración directa y
  coherente del mecanismo de reducción.
- **Las leyes de escoria se mueven en bloque por composición cerrada (suma=100%):**
  `d_ley_sn_escoria_pct` correlaciona fuertemente negativo con `d_ley_fe_total_escoria_pct` /
  `d_ley_feo_escoria_pct` (-0.50 a -0.76 en Reducción) — al removerse Sn de la escoria, el resto de óxidos
  sube su participación porcentual. Es **un efecto de composición (cierre de 100%), no necesariamente
  causal**; considerar razones/masas en vez de % puros al modelar.
- **Agregados simples por batch (sumas/promedios por fase) correlacionan débil con
  `rendimiento_proxy_batch`** (|ρ|≤0.13 en Spearman, para `carbon_total_reduccion_kg`,
  `exceso_o2_medio_fusion/reduccion_pct`, `caida_ley_sn_reduccion_pct`, `basicidad_media_fusion`, etc.) →
  la relación entre química de proceso y rendimiento **no es capturable con agregados lineales simples**;
  probablemente depende de la combinación y el momento en que ocurren los eventos, no sólo el total del
  batch.

## 5. Calidad de datos

- **`ley_feo_escoria_pct` no es un ensayo de laboratorio independiente ni una columna del Excel.** La
  columna `FEO` del Excel fuente reportaba un cálculo estequiométrico a partir de `%Fe` (columna origen de
  `ley_fe_total_escoria_pct`):
  $$ \%FeO = \%Fe \times 1.2865 $$
  donde 1.2865 es el factor de conversión de óxido FeO/Fe (ratio de peso molecular FeO/Fe = 71.844/55.845),
  el factor estándar en metalurgia para convertir Fe elemental ensayado a su equivalente como óxido FeO.
  Verificado empíricamente contra el Excel fuente: pendiente de regresión `FEO ~ %Fe` = 1.2865, correlación
  0.96 (3849 filas con ambos valores no nulos). **`dataset_lingo_smelter.py` ahora ignora deliberadamente
  la columna `FEO` del Excel** (no está en `RENOMBRES` ni en `CAMPOS_BASE`) y **`ley_feo_escoria_pct` se
  recalcula explícitamente con esa misma fórmula en la primera celda de `analisis_lingo_smelter.ipynb`**,
  para dejar la dependencia visible en el código en vez de heredarla implícita del Excel. **Implicación:**
  `ley_feo_escoria_pct` es prácticamente colineal con `ley_fe_total_escoria_pct` y no aporta información
  química independiente — cualquier correlación, ratio (`ratio_sn_feo*`, `ratio_feo_sio2*`), interacción
  (`interaccion_C_x_FeO*`, `interaccion_FeO_x_B2*`, `interaccion_Sn_x_FeO*`) o hallazgo de importancia de
  features que use ambas leyes juntas debe reinterpretarse sabiendo que son, en esencia, la misma señal
  reescalada, no dos leyes independientes del ensayo.
- **Barrido de correlaciones para descartar otras variables "calculadas" disfrazadas de independientes**
  (mismo chequeo que delató a `ley_feo_escoria_pct`): se probaron todos los pares de columnas numéricas
  del dataset construido por correlación y, para los pares con |r|>0.9, la dispersión (IQR/mediana) del
  ratio entre ambas — la firma de un cálculo exacto es un ratio casi sin dispersión (caso FeO/Fe: ~2%).
  No apareció ningún otro par con esa firma. Los más altos, para contexto:
  - `feed_Sn_kgf` vs `feed_total_kgh`: r=0.9988 (la correlación más alta de todo el dataset), pero el
    ratio tiene ~5% de dispersión — consistente con que la ley de Sn del concentrado es bastante estable
    batch a batch, **no un cálculo exacto**. Aun así, casi no aportan información incremental una sobre
    la otra; tratar con la misma cautela que las leyes de escoria al usarlas juntas como features.
  - `temperatura_gas_pre_bhf_celsius` vs `temperatura_gas_pre_cuchilla_celsius`: r=0.969, pero la
    diferencia entre ambas varía de -38°C a +10°C (no es constante) — son dos sensores físicos distintos
    en la línea de gases, no una derivada la una de la otra.
  - Los volúmenes de gas natural/O2/aire de la lanza (r=0.91-0.99) tienen 20-40% de dispersión en su
    ratio, consistente con dosificación coordinada real (control de combustión), no con una fórmula.
- **2 registros con leyes fuera de rango físico [0, 100]%** (de 3982): `ley_sn_escoria_pct`=241.5% y
  `ley_al2o3_escoria_pct`=549.1%, ambos del batch `AP0263` — casi con certeza error de captura/laboratorio.
  Tratados como `NaN` (no se elimina la fila completa).
- **Nulos** (% del total): `presion_punta_lanza_kpa` ~13.6%, `presion_suministro_gn_lanza_kpa` ~11.6%,
  leyes de metales menores (`%Sb` ~8.3%, `%Cr` ~4.2%, `%As` ~3.5%), leyes principales (Fe, FeO, SiO2, CaO,
  Al2O3, MgO, Sn) ~3.0-3.3%, `feed_Fe_kgh`/`feed_dross_Fe_kgh` ~0.3%, resto de presiones/volúmenes/
  temperaturas de gas ~0.1-0.2%.
- `relacion_C_carga` inestable/no interpretable en Reducción (ver 3.e).
- `basicidad_B2`/`B4` pueden dispararse cuando `%SiO2` del ensayo queda muy cerca de 0 — recortar
  percentiles 1%-99% sólo para visualización, no para el dataset base.

## 6. Enfoque analítico recomendado para las siguientes etapas (implementado en la sección 6b)

1. **Unidad de análisis: el escalón, no el batch**, usando `Batch` + `orden_escalon_fase` (0-6 Fusión,
   0-3 Reducción) como índice alineado entre batches.
2. **Modelar por fase por separado** (Fusión = régimen oxidante/formación de escoria; Reducción = régimen
   reductor/recuperación de Sn — química y reactivos relevantes distintos).
3. **Variable objetivo a nivel de escalón** candidata: `Δ%Sn escoria` (o `grad_ley_sn_escoria_pct`), en
   función de `tasa_feed_Carbon_kg_min`, `exceso_o2_combustion_pct`, `basicidad_B2`,
   `grad_temperatura_horno_celsius` y variables de posición de lanza — modelo de "extensión de reacción
   por escalón", interpretable y traducible a recomendación operativa.
4. **De escalón a batch:** una vez validado el modelo a nivel de escalón, agregar/integrar sus
   predicciones a lo largo del batch para conectar con `sn_en_metal_crudo_batch_t` /
   `rendimiento_proxy_batch`, en vez de usar agregados lineales crudos (que ya mostraron ser insuficientes,
   ver sección 4).
5. **Explorar no-linealidad e interacciones** con modelos de gradient boosting (XGBoost/LightGBM, ya en
   `requirements.txt`), interpretados con SHAP para verificar que las relaciones aprendidas sean
   pirometalúrgicamente coherentes (ej. signo esperado de `tasa_feed_Carbon_kg_min` en Reducción) antes de
   usarlas de forma prescriptiva.
6. **Confirmar con el equipo de proceso:** la interpretación de `feed_*`/`volumen_*` como "consumo por
   escalón" (sección 2) y la corrección de las 2 leyes fuera de rango (sección 5), antes de construir el
   modelo final.

## 6b. Resultados del modelamiento a nivel de escalón (Pasos 6-8 del notebook)

Se implementó el enfoque de la sección 6: modelo por fase, target `d_ley_sn_escoria_pct` (Δ del escalón),
predictores `tasa_feed_Carbon_kg_min`, `tasa_gn/o2/aire_nm3_min`, `exceso_o2_combustion_pct`,
`basicidad_B2`, `grad_temperatura_horno_celsius`, `posicion_vertical_lanza_mm`, `delta_tiempo` y
`orden_escalon_fase`. Nota de entorno: **LightGBM no funciona en el `.venv` de este equipo** (falta
`vcomp140.dll`, requiere permisos de admin) — se usó **XGBoost** en su lugar (sí funciona), con
**scikit-learn** para la validación cruzada.

- **Modelo:** `XGBRegressor` (300 árboles, `max_depth=4`, `learning_rate=0.03`), validado con
  **`GroupKFold` agrupando por `Batch`** (5 folds) para que ningún batch comparta filas entre train y
  test — evita la fuga de información que produciría una validación cruzada aleatoria simple dado que
  cada batch aporta varias filas correlacionadas.
- **Desempeño out-of-fold (`GroupKFold` por `Batch`):**
  - **Reducción:** $R^2$≈**0.85**, MAE≈1.24 puntos porcentuales de `%Sn` (vs. 4.50 de un baseline que
    predice siempre el promedio) — mejora muy grande, confirma que la química de escalón explica la gran
    mayoría del cambio de ley en Reducción.
  - **Fusión:** $R^2$≈**0.36**, MAE≈2.20 (vs. 2.68 baseline) — mejora más modesta; en Fusión el cambio de
    ley depende de más factores no incluidos (ej. composición/granulometría de la carga sólida entrante).
- **Validación out-of-time (OOT), walk-forward por orden cronológico de batch (`TimeSeriesSplit`, 5
  folds, entrenando sólo con batches anteriores y evaluando en batches posteriores):** el `GroupKFold`
  anterior agrupa por batch pero **no respeta el orden temporal**, por lo que no garantiza que el modelo
  generalice hacia adelante en el tiempo. El chequeo OOT da $R^2$≈**0.80** en Reducción (vs. 0.85 OOF) y
  $R^2$≈**0.30** en Fusión (vs. 0.36 OOF) — una caída leve (~0.05-0.06) y estable entre los 5 folds
  cronológicos (Reducción: 0.79-0.81 en todos los folds, incluso con sólo 61 batches de entrenamiento en
  el primero). **Conclusión: no hay evidencia de deriva fuerte del proceso ni de que el resultado OOF
  estuviera inflado por fuga temporal**; se recomienda igualmente reentrenar periódicamente en producción
  dado el gap observado.
- **Interpretabilidad (SHAP, `shap.TreeExplainer`):** se calculó, para cada variable, la correlación de
  Pearson entre su valor y su propio valor SHAP (signo del efecto marginal aprendido por el modelo):
  - **`tasa_feed_Carbon_kg_min`: signo negativo en ambas fases** (-0.85 Fusión, -0.80 Reducción) —
    coherente con $SnO_2+2C\rightarrow Sn+2CO_2$ y con el signo ya visto en las correlaciones simples de
    la sección 4. **Chequeo de sensatez pirometalúrgica superado.**
  - **`orden_escalon_fase` es la variable más influyente en Reducción** (importancia SHAP media ≈2.9, muy
    por encima de las demás) — reflejo de que el mayor salto de reducción ocurre casi siempre en el
    primer escalón de Reducción (ver batch AP0001). Informativo pero poco accionable por sí solo.
  - **`posicion_vertical_lanza_mm` tiene efecto marginal fuerte** (-0.91 Fusión, -0.74 Reducción),
    variable no explorada a fondo en el EDA original; candidata a revisión adicional con el equipo de
    proceso.
  - `exceso_o2_combustion_pct` y `basicidad_B2` tienen efecto marginal menor una vez controlado por las
    demás variables — su asociación univariada (sección 4) parece mediada en parte por
    `tasa_feed_Carbon_kg_min`/`orden_escalon_fase` (colinealidad esperable en la transición de fase).
- **Gráficos de dependencia SHAP (Paso 7b) — ¿tiene sentido la forma de la tendencia, no sólo el signo?**
  Se grafica valor de la variable vs. su propio valor SHAP, con curva LOWESS y medias por tercil:
  - `tasa_feed_Carbon_kg_min`: tendencia **monótona decreciente y limpia** en ambas fases (tercil
    bajo→alto: Fusión +0.38→-0.30, Reducción +0.41→-0.63), sin umbral ni saturación — dosis-respuesta
    ideal, coherente con la estequiometría. **Chequeo superado sin reservas.**
  - `basicidad_B2`: monótona decreciente en Fusión (+0.37→-0.43, coherente con química ácido-base de la
    escoria). En Reducción el efecto es prácticamente nulo (tabla de terciles ≈0.00 a -0.04) — la curva
    LOWESS en Reducción está distorsionada por los mismos outliers de `%SiO2`≈0 de la sección 5 (el eje
    llega a ~8 vs. ~0.8 en Fusión); la tabla de terciles es la lectura confiable. Ambos resultados tienen
    sentido: la basicidad gobierna la formación de escoria (Fusión), no la reducción del Sn ya formado.
  - `posicion_vertical_lanza_mm`: tendencia **monótona decreciente fuerte** en ambas fases (Fusión
    +0.88→-0.62, Reducción +0.30→-0.44) — mecánicamente plausible (mayor inmersión de la lanza podría
    mejorar la transferencia de masa/calor), pero el diccionario de datos no aclara si un valor mayor es
    lanza más alta o más baja. **Palanca operativa de alto interés, pendiente de confirmar su sentido
    físico exacto con el equipo de proceso antes de cualquier uso prescriptivo.**
  - `exceso_o2_combustion_pct` y `grad_temperatura_horno_celsius`: efecto marginal pequeño y no monótono
    en las medias por tercil — consistente con que su rol es más de "indicador de régimen"/interacción
    que de palanca marginal de primer orden una vez controladas las demás variables.
- **De escalón a batch (Paso 8):** se sumó, por batch, el `Δ%Sn escoria` observado y el predicho
  out-of-fold en Reducción, y se correlacionó (Spearman) con `rendimiento_proxy_batch`:
  `predicho_reducción` ρ≈-0.18, `observado_reducción` ρ≈-0.12 — **algo mejor que el agregado lineal
  ingenuo de la sección 4** (`caida_ley_sn_reduccion_pct` ρ≈-0.10) **pero aún débil.**
  **Interpretación:** un buen modelo de escalón (que predice bien la *ley* de la escoria) no se traduce
  automáticamente en una fuerte explicación de `rendimiento_proxy_batch`, porque este último se define a
  partir de **masas** de Sn en metal crudo/dross/polvo, no directamente de la ley de la escoria. Cerrar
  ese vínculo requeriría la **masa de escoria por escalón** (no disponible en este dataset) para convertir
  `Δ%Sn escoria` en `Δ(masa de Sn en escoria)` y así poder hacer un balance de masa explícito con
  `sn_en_metal_crudo_batch_t`. **Se documenta como limitación de datos para la siguiente etapa.**

## 7. Notas sobre origen de los datos en gráficos

Todos los gráficos y estadísticos de este EDA provienen de **datos observados/históricos** del dataset
(`df`), sin ningún modelo ni predicción de por medio — se indica explícitamente en los títulos de cada
gráfico del notebook (`analisis_lingo_smelter.ipynb`). Las excepciones son los gráficos de la sección
"Paso 6-8" del notebook (modelamiento), donde se indica explícitamente cuándo un valor es una
**predicción out-of-fold** (`GroupKFold`, el modelo nunca vio ese batch en entrenamiento) en vez de una
observación directa.

## 8. Próximos pasos (planteados en la primera sesión; ver sección 9 para lo ya resuelto)

**Para madurar el modelo predictivo:**
1. **Conseguir masa de escoria por escalón** (si existe en planta) para cerrar el balance de masa entre
   `Δ%Sn escoria` y `rendimiento_proxy_batch` (ver limitación en sección 6b). **Sigue pendiente — ver
   sección 9.5, es el único cuello de botella real que queda.**
2. Replicar el modelo de escalón sobre otras leyes (`ley_fe_total_escoria_pct`, `ley_feo_escoria_pct`)
   para enriquecer el balance de masa/redox. Pendiente.
3. Confirmar con el equipo de proceso los hallazgos de la sección 2 (interpretación de `feed_*`/
   `volumen_*`), sección 5 (2 leyes corregidas) y el rol de `posicion_vertical_lanza_mm` (sección 6b).
   **Más urgente que antes: ver sección 9.3 (el signo de su efecto cambió en el modelo expandido).**
4. Cuantificar incertidumbre (regresión cuantílica / conformal prediction), no sólo el punto estimado.
   Pendiente.
5. Definir cadencia de reentrenamiento y monitoreo de error en producción (dado el gap OOF→OOT). Pendiente.

**Para pasar de predictivo a prescriptivo:**
6. Distinguir correlación de causalidad antes de prescribir dosis (SHAP explica qué aprendió el modelo,
   no prueba causalidad aislada de `orden_escalon_fase`); considerar pruebas operativas controladas.
7. Definir explícitamente la función objetivo y las restricciones operativas/seguras de cada palanca.
8. Construir una capa de optimización/recomendación (búsqueda numérica sobre el modelo + restricciones),
   traducida a reglas simples y explicables para el operador, no una caja negra.
9. Validar las prescripciones en batches históricos retenidos y luego en un piloto acotado con supervisión
   de proceso, antes de adopción generalizada.

## 9. Feature engineering avanzado y modelo expandido (Pasos 10-15, sesión posterior)

Sesión de continuación centrada en mejorar features y target del modelo de escalón (Pasos 6-9), motivada
por la sospecha de que el target debía normalizarse (por tiempo o por el Sn ya presente) y de que las
features tenían margen de mejora. Ver `consideracioness_pirometalurgia_ausmelt.md` secciones 26-35 para el
fundamento teórico completo de cada feature nueva.

### 9.1 Features nuevas con mayor impacto: estado previo (`_prev`)

El modelo original excluía, correctamente, las leyes de escoria **contemporáneas** como feature (fuga: el
target es el delta de esa misma columna). Pero también excluía el valor del **escalón anterior**
(`ley_sn_escoria_pct_prev`, etc.), que sí está disponible antes de la reacción del escalón actual y no es
fuga. Añadir el estado previo (leyes, basicidad, temperatura, posición de lanza, presión de punta, tiro,
temperatura de gas pre-BHF) fue la mejora individual más importante: con el mismo target original, el
$R^2$ (OOF, `GroupKFold` por Batch) subió de **0.363→0.586 en Fusión** y de **0.854→0.961 en Reducción**
sólo por agregar features (sin cambiar el target).

Otras familias de features agregadas (Paso 10 del notebook): acumulados dentro del batch hasta el escalón
anterior (`cum_feed_Carbon_kg_prev`, `relacion_C_cum_Sn_cum_prev`, etc.), razones composicionales con
estado previo (`ratio_sn_feo_prev`, `ratio_feo_sio2_prev`, `log_ratio_sn_feo_prev`), features adicionales
de lanza/redox (`oxygen_enrichment_pct`, `indice_severidad_reductora`, `total_process_gas_nm3_min`,
`lanza_pos_x_gas_total`, `margen_presion_gn`) y del tren de gases (`delta_T_gas_pre`).

### 9.2 Target: no existe un único ganador — depende de la fase

Se compararon 3 targets (misma validación `GroupKFold` por Batch, mismas features expandidas):
$$ T_1=\Delta\%Sn_{\text{escoria}} \qquad T_2=\frac{\Delta\%Sn_{\text{escoria}}}{\Delta t} \qquad
   T_3=\frac{\Delta\%Sn_{\text{escoria}}}{\%Sn_{\text{escoria},\,prev}} $$
($T_3$ excluye escalones con `ley_sn_escoria_pct_prev < 0.5%`, división inestable).

| Fase | $T_1$ (features v2) | $T_2$ (tasa) | $T_3$ (fracción) | Ganador |
|---|---|---|---|---|
| Fusión | 0.586 | 0.665 | **0.734** | $T_3$ — fracción relativa al Sn previo |
| Reducción | **0.961** | 0.931 | 0.569 (peor que el baseline 0.854) | $T_1$ — delta absoluto |

En Reducción, normalizar por `ley_sn_escoria_pct_prev` empeora el resultado porque en tramos avanzados ese
valor puede ser muy pequeño (percentil 1%≈0.73%) y dividir por casi-cero amplifica el ruido de laboratorio
justo donde más importa (cerca del `%Sn` objetivo ≈1%). El target absoluto ya captura la misma información
cinética a través de la *feature* `ley_sn_escoria_pct_prev`, sin ese problema numérico. En Fusión, en
cambio, el crecimiento de `%Sn` se comporta de forma más proporcional al nivel ya acumulado, y la fracción
relativa gana. **Configuración final adoptada:** features expandidas en ambas fases; target fracción
relativa en Fusión, target absoluto en Reducción.

### 9.3 Validación del modelo final (features expandidas + target por fase)

| Fase | $R^2$ OOF | $R^2$ OOT (walk-forward) | Diferencia |
|---|---|---|---|
| Fusión | 0.734 | 0.674 | -0.060 (mismo gap normal que el Paso 6b) |
| Reducción | 0.961 | 0.958 | -0.003 (prácticamente nulo) |

SHAP (`shap.TreeExplainer`) confirma que `ley_sn_escoria_pct_prev` es, por lejos, la variable más
influyente en ambas fases (importancia media 3.19 en Reducción). `ratio_sn_feo_prev` (Sn/FeO previo) es la
segunda más importante en Reducción, con signo negativo (-0.988): más Sn relativo a FeO al empezar el
escalón → mayor caída de `%Sn` — **validación cuantitativa directa de la competencia redox Sn-Fe**
(`FeO_{slag}+Sn_{metal}\rightleftharpoons SnO_{slag}+Fe_{metal}`, sección 5.2 de consideraciones).
`tasa_feed_Carbon_kg_min` conserva su signo negativo en ambas fases, igual que en el modelo original.

**Hallazgo de robustez importante:** el signo del efecto marginal de `posicion_vertical_lanza_mm` en
Reducción **cambia** de negativo (modelo original, Paso 7) a positivo (modelo expandido) al controlar por
el estado previo de la escoria. Indica que su efecto aparente en el Paso 7 estaba en gran parte mediado por
su correlación con `ley_sn_escoria_pct_prev`/`ratio_sn_feo_prev`, no controladas en ese momento. Se eleva la
prioridad de confirmar con el equipo de proceso tanto el sentido físico de esta variable (¿mayor valor =
lanza más alta o más baja?) como la dirección real de su efecto, antes de cualquier uso prescriptivo.

### 9.4 KPI de batch adicional: `recuperación_real_batch_pct` (ancorado en balance de masa)

$$ \text{recuperación\_real\_batch}\,(\%) = 100\times\frac{Sn_{\text{metal crudo}}}
   {\sum_{\text{escalones}} feed_{Sn}} $$
A diferencia de `rendimiento_proxy_batch` (reparte el Sn contabilizado entre 3 salidas, sin referencia al
feed), esta métrica compara directamente el Sn que salió como metal crudo contra el Sn que entró al batch —
más cercana al concepto de "recuperación real" de la sección 35 de consideraciones. Media 69.5% (rango
51-90%, ningún batch supera ~90%, razonable para el supuesto de batch aislado). Correlaciona
moderadamente con `rendimiento_proxy_batch` (Spearman $\rho$=0.552): miden cosas relacionadas pero no
idénticas.

### 9.5 El cuello de botella escalón→batch sigue siendo el mismo (confirmado, no resuelto)

Con el modelo expandido y el nuevo KPI, los agregados lineales simples de escalón mejoran su correlación
máxima con el KPI de batch (de |ρ|≤0.133 con `rendimiento_proxy_batch` a |ρ|≤0.216 con
`recuperación_real_batch_pct`), pero el puente predicción-de-escalón→batch sigue siendo débil (|ρ|≤0.22)
**pese a que el modelo de escalón ahora es mucho mejor** ($R^2$ Reducción 0.96 vs. 0.85). Esto confirma que
el cuello de botella no era la calidad del modelo de escalón, sino la **falta de masa de escoria por
escalón** — sin ella no se puede convertir `Δ%Sn escoria` en `Δ(masa de Sn en escoria)` para cerrar un
balance de masa explícito. Es, con diferencia, el dato más valioso a conseguir antes de construir la capa
prescriptiva.

## 10. Feature engineering profundo y selección parsimoniosa (Pasos 17-19, sesión posterior)

Sesión de continuación centrada explícitamente en (a) profundizar el feature engineering apoyándose en
partes de `consideracioness_pirometalurgia_ausmelt.md` aún no explotadas (secciones 13, 14, 44.2-44.5), y
(b) podar el modelo ganador del Paso 13 (`FEATURES_MODELO_V2`, 20 features) a la menor cantidad de
variables que no perjudicara el $R^2$ OOF ya alcanzado (0.734 Fusión, 0.961 Reducción).

### 10.1 Ocho features nuevas, cada una atada a una sección específica del documento

- **Interacciones explícitas carbón×estado** (sección 13: *"el efecto relevante es carbón × O₂ × T ×
  estado de la escoria, no carbón aislado"*): `interaccion_C_x_Sn_prev`, `interaccion_C_x_FeO_prev`,
  `interaccion_C_x_temp_prev`, `interaccion_C_x_exceso_o2` — todas productos de
  `tasa_feed_Carbon_kg_min` por una variable de estado previo/redox, válidas en ambas fases.
- **Suficiencia estequiométrica exacta** (sección 14, $m_C/m_{Sn}=2\times12.011/118.71\approx0.2024$):
  `exceso_C_estequiometrico_pct` — sólo como candidata en **Fusión** (en Reducción `feed_Sn_kgf` es ~0 por
  diseño de proceso: 31.8% de escalones con feed ≤0.01 kg, confirmado por diagnóstico antes de construirla).
- **Reductor por inventario de escoria** (sección 44.5): `carbon_sobre_inventario_sn_prev` = tasa de
  carbón / `%Sn_prev` (protegida con el mismo umbral 0.5% del Paso 11) — válida en ambas fases.
- **Proxies de carga/térmico relativo** (secciones 44.2-44.3): `oxidante_sobre_carga_total`,
  `carga_termica_relativa` — sólo como candidatas en **Fusión** (mismo problema de `feed_total_kgh`≈0 en
  Reducción, 27.3% de escalones con feed ≤0.01 kg).
- Se documentaron explícitamente ideas descartadas y por qué: `progreso_normalizado_batch` (sección 44.1)
  es un reescalamiento lineal de `orden_escalon_fase` dentro de un modelo por fase con estructura fija —
  no aporta nada a un modelo de árboles (invariante a transformaciones monótonas); productos crudos
  O₂×GN/O₂×aire (sección 13) ya están resumidos por `exceso_o2_combustion_pct`/`oxygen_enrichment_pct`.

### 10.2 Selección parsimoniosa: eliminación hacia atrás guiada por SHAP

Metodología: entrenar con todas las candidatas (`FEATURES_MODELO_V2` + las 8 nuevas: 28 en Fusión, 25 en
Reducción), ordenar por importancia SHAP ascendente, intentar remover la menos importante y validar con
`GroupKFold` (Paso 12-13); aceptar la remoción si el $R^2$ OOF se mantiene en o por encima del piso (el
$R^2$ OOF ganador del Paso 13), recalcular SHAP sobre el conjunto reducido y repetir; detener al primer
rechazo. Resultado:

| Fase | Features (V2→final) | $R^2$ OOF (V2→final) | $R^2$ OOT (V2→final) | Gap OOF-OOT (V2→final) |
|---|---|---|---|---|
| Fusión | 20→**7** (-65%) | 0.734→**0.738** | 0.674→**0.686** | -0.060→**-0.052** |
| Reducción | 20→**14** (-30%) | 0.961→0.961 (-0.0006) | 0.958→**0.960** | -0.003→**-0.001** |

**El resultado más importante no es sólo "no empeoró": el modelo parsimonioso generaliza mejor** (mayor
$R^2$ OOT y menor gap OOF→OOT en ambas fases) — la firma clásica de reducir varianza por sobreajuste a
redundancia entre features correlacionadas (se documentaron 23 pares con $|\rho|>0.85$ en Fusión y 34 en
Reducción entre las candidatas, mucha de ella esperada por construcción).

**Feature set final:**
- Fusión (7): `basicidad_B2`, `delta_tiempo`, `ley_sn_escoria_pct_prev`, `cum_feed_Carbon_kg_prev`,
  `lanza_pos_x_gas_total`, `interaccion_C_x_Sn_prev`\*, `oxidante_sobre_carga_total`\*
- Reducción (14): `tasa_gn_nm3_min`, `exceso_o2_combustion_pct`, `basicidad_B2`,
  `grad_temperatura_horno_celsius`, `posicion_vertical_lanza_mm`, `ley_sn_escoria_pct_prev`,
  `ley_feo_escoria_pct_prev`, `cum_feed_Carbon_kg_prev`, `indice_severidad_reductora`,
  `lanza_pos_x_gas_total`, `ratio_sn_feo_prev`, `interaccion_C_x_Sn_prev`\*, `interaccion_C_x_exceso_o2`\*,
  `carbon_sobre_inventario_sn_prev`\* (\* = nueva del Paso 17)

### 10.3 Hallazgo pirometalúrgico central: la interacción carbón×Sn_prev valida la cinética bimolecular

`interaccion_C_x_Sn_prev` (producto `tasa_feed_Carbon_kg_min` × `ley_sn_escoria_pct_prev`) es la **tercera
variable más importante en ambas fases** tras la poda, con signo negativo limpio (-0.734 Fusión, -0.974
Reducción): a mayor producto dosis×sustrato, mayor caída de `%Sn` — validación cuantitativa directa de
$rate\propto -k[C][SnO_x]$, la advertencia central de la sección 13 de consideraciones sobre modelar
interacciones y no sólo efectos principales.

Consecuencia notable: **`tasa_feed_Carbon_kg_min` y `orden_escalon_fase`** —las dos variables más citadas
del análisis original (Pasos 6-9)— **resultan prescindibles** una vez presentes sus versiones
normalizadas/derivadas (`interaccion_C_x_Sn_prev`, `carbon_sobre_inventario_sn_prev`,
`cum_feed_Carbon_kg_prev`, `ley_sn_escoria_pct_prev`): no dejan de importar, su información sólo queda
mejor representada por transformaciones más cercanas a la cinética/inventario real.

## 11. Segunda ronda: interpretabilidad y baja colinealidad del modelo final (Pasos 20-27, sesión posterior)

Sesión motivada por un pedido explícito del usuario tras revisar el modelo parsimonioso del Paso 19: (a)
que las features del modelo final **no correlacionen fuerte entre sí** (nunca se había verificado
explícitamente sobre el *set final*, sólo sobre el *pool de candidatas*, Paso 18), (b) evitar **cocientes
difíciles de interpretar** cuando exista una alternativa físicamente igual de fundamentada (productos,
diferencias, niveles), y (c) explorar variables **aún no usadas** de
`consideracioness_pirometalurgia_ausmelt.md`, manteniendo o mejorando la precisión ya alcanzada — con el
objetivo final declarado de usar el modelo como gemelo digital para prescripción operacional.

### 11.1 Ocho features nuevas, todas construidas con estado previo/acumulado (sin fuga)

| Feature | Sección del documento | Idea pirometalúrgica | Tipo |
|---|---|---|---|
| `temperatura_horno_celsius_prev` | 8 (superheat/margen térmico) | Nivel térmico de partida (complementa al gradiente ya usado) | nivel (°C) |
| `presion_punta_lanza_kpa_prev` | 17.3 (sensor hidrodinámico indirecto) | Régimen de burbujeo/turbulencia de partida | nivel (kPa) |
| `margen_presion_gn` | 17.4 (suministro vs. contrapresión) | Presión "de sobra" en la lanza de GN | diferencia (kPa) |
| `cum_feed_CaO_kg_prev` | 28 (inventario acumulado) | Fundente ya dosificado hasta el escalón anterior | masa (kg) |
| `cum_feed_Fe_kg_prev` | 28 (inventario acumulado) | Mineral de Fe ya cargado hasta el escalón anterior | masa (kg) |
| `interaccion_FeO_x_B2_prev` | 7.4, 7.7 (modificadores de red conjuntos) | Riqueza conjunta FeO×B2 al iniciar el escalón | producto |
| `interaccion_B2_x_temp_prev` | 8 (fluidez química Y térmica) | Basicidad sólo se traduce en fluidez con suficiente T | producto |
| `interaccion_Sn_x_FeO_prev` | 5.2, 32 (competencia redox Sn–Fe) | Alternativa robusta (sin dividir) a `ratio_sn_feo_prev` | producto |

`margen_presion_gn` y `presion_punta_lanza_kpa_prev` resultaron con nulos altos en Reducción (~32% y ~14%
respectivamente, heredados de las presiones base ya documentadas en la sección 5) — suficiente para
"atascar" la poda automática por pérdida de muestra (ver 11.3); se excluyeron del pool de Reducción una vez
confirmado (por su propia eliminación temprana en Fusión) que no aportaban lo suficiente para justificar
ese costo de muestra.

### 11.2 Diagnóstico de correlación cruzada ANTES de modelar (nuevo, no se había hecho en rondas previas)

Se calculó la correlación de Spearman entre las 8 candidatas nuevas y el set parsimonioso del Paso 19,
*antes* de entrenar nada: anticipó correctamente que `cum_feed_CaO_kg_prev`/`cum_feed_Fe_kg_prev`
correlacionan fuerte con acumulados/tiempo ya presentes en Fusión, y que `interaccion_Sn_x_FeO_prev`
correlaciona casi perfecto (0.99) con `ley_sn_escoria_pct_prev` en Reducción (donde `%FeO` varía poco). El
diagnóstico previo no reemplaza la validación empírica, pero explica *por qué* el algoritmo de poda termina
descartando lo que descarta.

### 11.3 Bug metodológico detectado y corregido: poda "atascada" por pérdida de muestra en Reducción

Al correr la misma poda voraz por SHAP del Paso 18 sobre parsimonioso+8 nuevas, Reducción no podó **nada**
en el primer intento (0 features removidas de 22): la razón no era que las 22 fueran indispensables, sino
que el `dropna()` conjunto (candidatas con muchos nulos incluidas) hizo caer el `n` utilizable de 1327 a
809 (-39%), y la configuración *inicial* ya arrancaba por debajo del piso de $R^2$ sólo por eso — ningún
intento de remoción individual lograba "recuperar" un piso que nunca debió perderse por una causa ajena a
la calidad de las features. **Lección metodológica para futuras rondas de poda:** si se agregan candidatas
con nulos considerablemente mayores que el resto del pool, filtrarlas (o imputarlas) *antes* de correr
`seleccion_parsimoniosa`, o la comparación queda contaminada por el tamaño de muestra, no por el mérito
real de las variables. Corregido excluyendo las 2 candidatas más nulas de Reducción; con eso la poda
recuperó su comportamiento normal (20→12 candidatas, $R^2$ OOF prácticamente igual al piso).

### 11.4 Redundancia real que la poda voraz por SHAP no elimina (limitación conocida del método, ahora cuantificada)

La eliminación hacia atrás es voraz: sólo evalúa remover la variable de *menor* SHAP en cada paso, así que
una variable puede sobrevivir toda la poda sin ser nunca la más débil, **aunque esté fuertemente
correlacionada con otra que también sobrevivió**. Verificado explícitamente sobre los sets que arrojó el
Paso 18-modificado: Reducción quedó con **7 pares** con $|\rho_{Spearman}|>0.85$ (un clúster de 4 variables
mutuamente correlacionadas, todas dominadas por `%Sn` previo) y Fusión con 3. Se resolvió con una segunda
ronda de **remociones dirigidas por interpretabilidad** (no por SHAP): probar remover, en orden de
preferencia manual, el miembro menos fundamental de cada par/clúster, aceptando la remoción sólo si el
$R^2$ OOF se mantenía dentro de la misma tolerancia (1e-3) del piso original (Paso 19). Resultado:
**Reducción bajó de 7 pares a 0** (y el $R^2$ OOF de hecho *subió* de 0.9607 a 0.9623 al quitar 3 variables
redundantes: `interaccion_Sn_x_FeO_prev`, `ratio_sn_feo_prev`, `lanza_pos_x_gas_total`) y **Fusión bajó de
3 pares a 1, verificado como necesario** (remover cualquiera de los dos miembros —`delta_tiempo` o
`cum_feed_CaO_kg_prev`— perjudica el $R^2$ muy por encima de la tolerancia; se conservan ambos a propósito).
**Conclusión metodológica:** la poda guiada sólo por SHAP (Paso 18) es necesaria pero no suficiente para
garantizar baja colinealidad en el set final; hace falta un chequeo de correlación explícito *después* de
podar, no sólo antes.

### 11.5 Configuración final adoptada — dos alternativas documentadas para Reducción

| Fase | Features (P19→final) | $R^2$ OOF (P19→final) | $R^2$ OOT (P19→final) | Pares $|\rho|>0.85$ final |
|---|---|---|---|---|
| Fusión | 7→**6** | 0.7376→**0.7394** (+0.0018) | 0.6861→**0.6889** (+0.0028) | 1 (necesario) |
| Reducción (Config. B, adoptada) | 14→**7** (-50%) | 0.9607→**0.9598** (-0.0009) | 0.9598→0.9514 (-0.0085) | 0 |
| Reducción (Config. A, alternativa desempeño) | 14→9 | **0.9623** (mejor de toda la ronda) | no recalculado | 1 (necesario) |

**Fusión final (6):** `basicidad_B2`, `delta_tiempo`, `ley_sn_escoria_pct_prev`, `interaccion_C_x_Sn_prev`,
`cum_feed_CaO_kg_prev`, `interaccion_Sn_x_FeO_prev` — **ningún cociente**, sólo niveles/masas físicas y dos
productos de interpretación directa.

**Reducción final, Config. B (7):** `tasa_gn_nm3_min`, `exceso_o2_combustion_pct`, `basicidad_B2`,
`grad_temperatura_horno_celsius`, `posicion_vertical_lanza_mm`, `ley_sn_escoria_pct_prev`,
`cum_feed_Carbon_kg_prev` — 50% menos features que el Paso 19, **0 pares correlacionados**, sin
`carbon_sobre_inventario_sn_prev` (el cociente de signo contra-intuitivo por efecto de piso ya documentado
en el Paso 18e).

**Config. A (9, alternativa)** agrega `carbon_sobre_inventario_sn_prev` e `interaccion_C_x_Sn_prev` a la
Config. B: mejor $R^2$ OOF absoluto (0.9623) pero conserva un cociente "difícil" y un par correlacionado
(aceptable, ya verificado como necesario). Se documenta como respaldo si en el futuro se prioriza
desempeño puro sobre interpretabilidad — la diferencia entre A y B (0.0025 de $R^2$ OOF) es menor que el
propio gap OOF→OOT ya aceptado como ruido normal en pasos anteriores.

### 11.6 Advertencia reforzada: el signo de `posicion_vertical_lanza_mm` sigue sin ser estable

Con el set final de Reducción (Config. B), el signo SHAP de `posicion_vertical_lanza_mm` vuelve a ser
**negativo** (-0.18), revirtiendo el signo **positivo** que había emergido en los Pasos 13b/18 al controlar
por estado previo (que a su vez había revertido el signo negativo del modelo original, Paso 7). Que el
signo cambie según qué otras variables lo acompañan en el modelo (negativo→positivo→negativo en 3
especificaciones distintas) es la evidencia más fuerte hasta ahora de que su efecto marginal **no está
identificado de forma estable** con los datos y features disponibles. Se reitera con la máxima prioridad
la recomendación de confirmar con planta el sentido físico de esta variable y su efecto real antes de
cualquier uso prescriptivo — no tratar el signo de ninguna versión del modelo como conclusión asentada.

### 11.7 Qué sigue para el gemelo digital prescriptivo

Con features ahora validadas en desempeño, de baja colinealidad y casi enteramente libres de cocientes de
difícil interpretación, el bloqueo principal para avanzar hacia el modelo dinámico estado→acción→estado
siguiente + optimización restringida (secciones 37-39 de consideraciones) ya no es la calidad de las
features de entrada sino **datos faltantes**: (1) masa de escoria por escalón (§9.5, para cerrar el balance
de masa escalón→batch) y (2) confirmación con planta del sentido físico de `posicion_vertical_lanza_mm`
(§11.6). Ambas son limitaciones de datos/dominio, no de metodología de modelado.

**Advertencia de interpretación:** el signo positivo de `carbon_sobre_inventario_sn_prev` en Reducción
(+0.657) no es "más dosis relativa → menos reducción"; es un efecto de piso mecánico (la razón se dispara
cuando `%Sn_prev` ya está cerca del umbral de 0.5%, donde el `Δ%Sn` absoluto está acotado por lo poco que
queda por reducir). `posicion_vertical_lanza_mm` conserva su signo positivo en Reducción (Paso 13b) también
tras la poda — se refuerza, no se resuelve, la prioridad de confirmar con planta su sentido físico antes de
cualquier uso prescriptivo.

**Qué NO cambió:** el target ganador por fase (T3 Fusión, T1 Reducción) y el esquema de un modelo por fase
se mantuvieron sin cambios — esta sesión sólo optimizó qué features usa cada modelo. El cuello de botella
escalón→batch (§9.5) sigue sin resolverse: sigue siendo la prioridad de datos más alta antes de la capa
prescriptiva.

## 12. `posicion_vertical_lanza_mm` no es monótona: tiene un óptimo interior, no un signo inestable (Pasos 46-47, sesión posterior)

Sesión motivada por una explicación pirometalúrgica explícita del usuario (TSL/Ausmelt, ver
`consideracioness_pirometalurgia_ausmelt.md` sección 16 y su desarrollo teórico completo): la profundidad
de inmersión de la lanza **no debería tener un efecto monótono**. Demasiado profunda perturba la interfaz
escoria-metal, re-entraína gotas de Sn y favorece sobre-reducción de FeO/hardhead; demasiado superficial
reduce tiempo de residencia/mezcla y puede volverse intermitente. El óptimo esperado es una **región
intermedia**, no un extremo — lo que implica que el resumen de 3 terciles usado en los Pasos 29/39 (que
sólo puede describir monotonía o no-monotonía muy gruesa) podía estar ocultando la forma real de la curva.

**Metodología:** sin reentrenar ningún modelo, se recalculó la dependencia SHAP de `posicion_vertical_lanza_mm`
con **10 deciles en vez de 3 terciles**, un **LOWESS angosto** (`frac=0.15`) superpuesto al LOWESS ancho ya
usado (`frac=0.4`), la **fracción de escalones con SHAP positivo por decil** (para distinguir un efecto de
signo limpio por zona de uno mezclado/ruidoso), y **SHAP interaction values** (`shap.TreeExplainer.shap_interaction_values`)
para cuantificar cuánto del efecto de posición es interacción con `tasa_gn_nm3_min` (flujo de gas de la
lanza) en vez de efecto propio aislado — sobre 3 combinaciones ya calculadas en el notebook: el modelo final
Config B (target `d_ley_sn_escoria_pct`, Paso 27) y el modelo de frontera (targets Δ%Sn y Δ%FeO, Paso 39),
ambos en Reducción.

**Resultado — la hipótesis del usuario se confirma cuantitativa y visualmente:**

- **Deciles rompen la monotonía que terciles no podían ver:** 3, 1 y 5 cambios de dirección en la secuencia
  de medias SHAP por decil (ΔSn Config B, ΔFeO frontera, ΔSn frontera respectivamente) — con sólo 3 terciles
  esto es indetectable por construcción.
- **El LOWESS angosto muestra una forma de "valle" clara y consistente en los 3 casos:** SHAP positivo en
  el extremo bajo de la variable cruda (~1000-2000mm), cae hasta un **mínimo alrededor de 3300-4200mm**, y
  se **recupera** hacia valores cercanos a cero/positivos en el extremo alto (>4500mm). El LOWESS ancho
  (frac=0.4) promedia esa recuperación final y la hacía lucir más monótona de lo que realmente es — la
  causa técnica de por qué los Pasos 29/39 reportaron "monótona decreciente".
- **Coincidencia pirometalúrgicamente coherente:** el mismo tramo (~3300-4200mm) concentra simultáneamente
  la mayor caída de `%Sn` (deseable) Y de `%FeO` (riesgo de hardhead) — consistente con una región de mayor
  intensidad de agitación/poder reductor que empuja ambas reacciones de reducción a la vez, exactamente el
  mecanismo de doble filo que describe la explicación del usuario, y no dos fenómenos independientes.
- **La interacción con el flujo de gas es sustancial, no un matiz:** el 64.5% de la magnitud SHAP total de
  `posicion_vertical_lanza_mm` (Config B) proviene de su interacción con `tasa_gn_nm3_min`, no de un efecto
  propio aislado. El scatter coloreado por tercil de gas muestra que el valle de SHAP más negativo está
  poblado desproporcionadamente por escalones de gas medio/alto — valida directamente la advertencia de la
  sección 16 de consideraciones sobre modelar `posición × flujo de gas` en vez de posición aislada.

**Reinterpretación de la "advertencia de signo inestable" de los Pasos 7/13b/18/27/29/39/42/45:** esos pasos
no eran ruido ni un efecto mal identificado — eran **proyecciones distintas de la misma curva no monótona**.
Cada especificación de modelo pondera de forma distinta las tres zonas (bajo/valle/alto) del rango de
posición según qué filas sobreviven el `dropna()` de esas features y qué otras variables acompañan al
modelo, y un resumen agregado de un solo signo (correlación, SHAP-tercil, o SHAP medio) de una curva en
valle puede dar positivo, negativo o casi cero dependiendo de qué zona domine esa muestra particular. La
serie negativo→positivo→negativo→(residual)positivo documentada en la sección 11.6 dejó de leerse como
"efecto no identificable" y pasa a leerse como la firma esperada de resumir con un solo número una relación
con óptimo interior.

**Limitación que sigue sin resolverse:** este análisis no determina si un valor mayor de
`posicion_vertical_lanza_mm` es lanza más alta o más baja (`diccionario_datos.json` sólo dice "Altura o
posición vertical de la lanza", ambiguo) — por eso el hallazgo se reporta en términos de la **forma de la
curva sobre la variable cruda** (óptimo interior ~3300-4200mm, peor desempeño en ambos extremos), sin
traducirlo todavía a "muy profundo"/"muy superficial" en el sentido físico de consideraciones. Confirmar el
sentido físico con planta sigue siendo la prioridad más alta antes de cualquier uso prescriptivo de esta
variable (§11.7), pero ahora con una hipótesis funcional mucho más precisa para validar: no "es mejor más
arriba o más abajo", sino "hay un rango intermedio de la variable cruda que concentra la mayor actividad
reductora (Sn y Fe simultáneamente), con menor efecto hacia ambos extremos". Implementado como Pasos 46-47
en `analisis_lingo_smelter.ipynb`.

## 13. Modelo predictivo v2: causalidad cross-fitted, selección avanzada, bake-off y modelo de batch (sesión 2026-09-13 tarde)

Sesión motivada por un pedido explícito de mejorar el modelo predictivo para prescripción, con más
experimentos/variantes, feature selection con técnicas avanzadas y análisis de causalidad. Todo lo
implementado vive en `modelo_predictivo_v2.py` (nuevo módulo, complementa — no reemplaza — a
`modelo_prescriptivo.py`) y en `feature_engineering.construir_features_trayectoria_batch` (nueva). Cinco
piezas, cada una ejecutada contra el dataset real con la misma disciplina OOF/OOT/lockbox ya establecida.

### 13.1 Causalidad: Double ML CON cross-fitting real, corrige un sesgo metodológico del Paso 42

El Paso 42 implementaba Frisch-Waugh-Lovell (residualizar acción y resultado contra el estado) pero
ajustaba las nuisance models **una sola vez sobre todo el dev set y tomaba residuos IN-SAMPLE** — un sesgo
de sobreajuste conocido en la literatura de Double ML (Chernozhukov et al. 2018), la razón por la que ese
método exige *cross-fitting*. Se reimplementó con cross-fitting real (GroupKFold por Batch, 5 folds,
residuos siempre out-of-fold) más 3 *refutation tests* al estilo DoWhy (placebo/acción barajada, confusor
aleatorio agregado a las nuisance models, estabilidad en submuestras por batch):

| Fase | Acción | θ (puntos de %Sn por unidad) | IC95% (bootstrap cluster) | Placebo (3σ) | Confusor | Subset mismo signo | Veredicto |
|---|---|---|---|---|---|---|---|
| Fusión | `delta_tiempo` | -0.039 | [-0.059, -0.018] | pasa | Δ0.2% | 100% | **ROBUSTO** |
| Fusión | `tasa_feed_Carbon_kg_min` | +0.003 | [-0.020, +0.027] (cruza 0) | no aplica | — | 90% | NO SIGNIFICATIVO |
| Reducción | `tasa_feed_Carbon_kg_min` | -0.0094 | [-0.016, -0.0033] | pasa | Δ23% | 100% | **ROBUSTO** |
| Reducción | `tasa_gn_nm3_min` | -0.0159 | [-0.0305, -0.0036] | no (umbral 3σ) | Δ0.5% | 100% | SIGNIFICATIVO MODERADO |
| Reducción | `exceso_o2_combustion_pct` | +0.0049 | [0.0022, 0.0081] | no (umbral 3σ) | Δ1.0% | 100% | SIGNIFICATIVO MODERADO |
| Reducción | `posicion_vertical_lanza_mm` | +0.0003 | [0.0002, 0.0004] | pasa | Δ1.1% | 100% | significativo pero **magnitud despreciable** |

Tres lecturas importantes:

- **La significancia de `tasa_feed_Carbon_kg_min` en Fusión que encontró el Paso 42 (ρ_residual=-0.062,
  p<0.05) se REVIERTE con cross-fitting real** (IC95% cruza 0): era, con alta probabilidad, un artefacto de
  sesgo de sobreajuste in-sample, no un efecto causal real. Esto no significa que el carbón no importe en
  Fusión (su señal sigue presente vía `interaccion_C_x_Sn_prev` y el nivel acumulado), sino que su efecto
  marginal PURO, lineal y aislado del resto del estado, no está identificado de forma confiable.
- **`tasa_feed_Carbon_kg_min` en Reducción es, con diferencia, el efecto causal más robusto de todo el
  análisis** (pasa las 4 verificaciones: IC95% no cruza 0, supera el umbral placebo más estricto, cambia
  <25% con un confusor aleatorio, 100% de submuestras con el mismo signo) — y sin embargo (§13.4) nunca fue
  ofrecida como acción optimizable en `modelo_prescriptivo.py`.
- **El efecto lineal de `posicion_vertical_lanza_mm` es estadísticamente no-nulo pero prácticamente
  despreciable** (0.0003 puntos de %Sn por mm — hacen falta ~3000mm, casi todo el rango observado, para
  mover 1 punto de %Sn). Cuantifica exactamente por qué un ajuste lineal la ve "casi plana": promedia una
  curva en valle que sube y baja (Pasos 46-47), y las contribuciones positivas/negativas se cancelan.

### 13.2 Selección de features con técnicas avanzadas (mutual information + permutation importance estable + Boruta-shadow)

Sobre el pool completo de 79 candidatas seguras (`feature_engineering.CLASIFICACION_FEATURES`, roles
STATE/ACTION_primitiva/DERIVED-*), se aplicaron 3 técnicas no usadas en rondas previas (que usaron
eliminación hacia atrás guiada por SHAP): mutual information (no paramétrica), permutation importance
repetida (5 folds × 5 semillas, reporta fracción de corridas con importancia > 0, no solo la media) y un
test Boruta-shadow manual (25 iteraciones, shadow features barajadas, confirma una variable si supera a la
*mejor* shadow en ≥60% de las iteraciones).

**Fusión**: confirma el set actual. `basicidad_B2_prev` (0/25) y `tasa_feed_Carbon_kg_min` (2/25) no superan
el test de shadow-features *sobre el pool completo* — pero esto no contradice su rol ya validado en el set
parsimonioso (Pasos 18-19, ablation directo sobre esas 7 variables): frente a 79 alternativas correlacionadas,
su señal queda absorbida por `interaccion_C_x_Sn_prev`/`cum_feed_CaO_kg_prev` (ya documentado en §10.3). Es
evidencia de **redundancia**, no de irrelevancia.

**Reducción**: hallazgo más fuerte. **5 de las 7 features en producción** obtienen 0-12% de confirmación
Boruta (indistinguibles de ruido en ese test): `exceso_o2_combustion_pct` (12%), `basicidad_B2_prev` (0%),
`grad_temperatura_horno_celsius_prev` (0%), `posicion_vertical_lanza_mm` (4%), `cum_feed_Carbon_kg_prev`
(0%). Solo `tasa_gn_nm3_min` (96%) y `ley_sn_escoria_pct_prev` (100%) sobreviven con margen. Esto **converge
con §13.1**: dos métodos independientes (residualización causal y shadow-feature test) coinciden en que
`posicion_vertical_lanza_mm`/`exceso_o2_combustion_pct` no tienen una señal marginal robusta una vez
disponibles las variables de interacción/ratio (ya validadas pirometalúrgicamente en §10.3/§11).

### 13.3 Config C (Reducción): set alternativo validado, desempeño equivalente pero más defendible

A partir de las features CONFIRMADAS/TENTATIVAS del Boruta, decorrelacionadas (mismo criterio de §11.4:
eliminar el miembro menos fundamental de cada par con |ρ_Spearman|>0.85) y filtrando 2 exclusiones
documentadas (`oxidante_sobre_carga_total`: denominador ≈0 en Reducción, §10.1; `interaccion_lanza_x_gn`:
usa `posicion_vertical_lanza_mm` contemporánea, incompatible con §13.4), se llegó a un set de 9 features:

```
ley_sn_escoria_pct_prev, interaccion_Sn_x_FeO_prev, interaccion_C_x_Sn_prev, tasa_gn_nm3_min,
interaccion_C_x_exceso_o2, posicion_vertical_lanza_mm_prev, tasa_feed_Carbon_kg_min,
interaccion_C_x_FeO_prev, exceso_o2_combustion_pct
```

Validado con la MISMA disciplina de 3 niveles, sobre las MISMAS filas que la Config B actual
(n_dev=1085-1086, n_lockbox=242): R² OOF/lockbox prácticamente empatados con la config actual (diferencias
de 0.001-0.004, dentro del ruido ya documentado entre folds). **El valor de Config C no es "más preciso",
es "más defendible"**: reemplaza 3 features en zona de ruido (Boruta) por features causalmente más
transparentes, y sustituye `posicion_vertical_lanza_mm` contemporánea por su versión `_prev` (rompe la
posible confusión S_t→A_t que motivó la disciplina de residualización de §13.1). Se documenta como
alternativa, no como reemplazo obligatorio — igual que "Config A/B" en §11.5.

### 13.4 Hallazgo operacional más importante de la sesión: la palanca de carbón nunca estuvo disponible en Reducción

`modelo_prescriptivo.ACCIONES_PRIMITIVAS["Reducción"]` (GN, O2, aire, posición de lanza) **nunca incluyó
`tasa_feed_Carbon_kg_min`**, y `FEATURES_STATE_ACTION["Reducción"]` solo usaba `cum_feed_Carbon_kg_prev`
(el acumulado PASADO, no la dosis del escalón actual). Es decir: la variable con el efecto causal más
robusto de todo el análisis (§13.1, pasa las 4 verificaciones) **nunca estuvo al alcance del optimizador**
— ni como feature de predicción del escalón actual, ni como acción recomendable. `modelo_predictivo_v2.py`
lo corrige: Config C incluye `tasa_feed_Carbon_kg_min` como feature, y
`ACCIONES_PRIMITIVAS_V2["Reducción"]` la agrega a las acciones optimizables (a cambio, `posicion_vertical_lanza_mm`
contemporánea SALE de las acciones — no se ofrece como palanca hasta validación de planta, misma razón que
motivó los Pasos 46-47; su `_prev` se mantiene como contexto de estado, no como acción).

### 13.5 Bake-off de algoritmos con restricciones monotónicas LOCALES (no globales)

Siguiendo `consideracioness_pirometalurgia_ausmelt.md` sección 84 ("no imponer monotonía global, solo sobre
submodelos físicos inequívocos"), se restringió `monotone_constraints`/`monotonic_cst` **solo** en las
variables de dosis de carbón (única relación confirmada "sin reservas" en todas las especificaciones
previas) — nunca en `posicion_vertical_lanza_mm` (óptimo interior confirmado, Pasos 46-47). Comparación
XGBoost / XGBoost+monotónico / HistGradientBoostingRegressor / HistGB+monotónico / ElasticNet, misma
validación de 3 niveles:

| Fase | Modelo ganador (lockbox) | R² lockbox actual → nuevo | MAE lockbox actual → nuevo |
|---|---|---|---|
| Fusión | XGBoost **sin** restricción (sin cambio) | 0.465 → 0.465 | 2.132 → 2.132 |
| Reducción | HistGB + monotónico(carbón), Config B | 0.965 → **0.973** | 0.687 → **0.593** |
| Reducción | HistGB + monotónico(carbón), Config C | 0.965 → 0.971 | 0.687 → 0.642 |

Coherente con §13.1: en Reducción, donde el efecto causal del carbón está robustamente validado, forzar
monotonía ahí mejora la generalización (regulariza con conocimiento de dominio verificado). En Fusión,
donde ese mismo efecto NO se valida de forma robusta tras cross-fitting, monotonizarlo empeora el ajuste
(0.447 vs. 0.465 lockbox) — exactamente el resultado esperable si la restricción fuera injustificada ahí.
ElasticNet (lineal) queda claramente por detrás en ambas fases, más en Fusión (lockbox R²=0.387, con una
brecha OOF→lockbox grande que sugiere mala generalización temporal de un ajuste puramente lineal).

### 13.6 Incertidumbre calibrada: CV+ conformal prediction reemplaza el std del ensamble

Pendiente explícito desde hallazgos.md §8 punto 4 ("cuantificar incertidumbre... conformal prediction").
Se implementó CV+ (Barber, Candés, Ramdas & Tibshirani, 2021) sobre el mismo ensamble de GroupKFold ya
entrenado: para un punto nuevo, cada modelo *fold* que no vio su propio conjunto de calibración aporta
`predicción ± residuo_OOF` a una nube combinada, de la que se toman los percentiles del intervalo. Cobertura
empírica medida en el LOCKBOX (nunca usado para calibrar nada):

| Fase | Nominal | Cobertura empírica | Ancho medio |
|---|---|---|---|
| Fusión | 80% | 89.6% | 8.23 pts %Sn |
| Fusión | 90% | 95.7% | 10.78 pts %Sn |
| Reducción | 80% | 91.3% | 0.59 (escala T3) |
| Reducción | 90% | 96.3% | 0.77 (escala T3) |

Las 4 combinaciones son **conservadoras** (cobertura real ≥ nominal) — la dirección segura para un sistema
prescriptivo (consideraciones §85: "si incertidumbre alta → no recomendar cambio agresivo"). Reemplaza la
`uncertainty` (std del ensamble) de `modelo_prescriptivo.py`, declarada ahí explícitamente como no calibrada.

### 13.7 Modelo de BATCH con features de trayectoria: mejora real pero sigue sin resolver el cuello de botella

Se construyó `feature_engineering.construir_features_trayectoria_batch` (perfiles de dosificación,
dispersión, fracción de escalones de Reducción en la zona de "valle" de lanza de los Pasos 46-47, en vez de
solo sumas/promedios simples ya documentados como débiles, §6) más una feature adicional informada por el
propio modelo de escalón ya validado (`suma_pred_delta_sn_reduccion_modelo`, rollup out-of-fold/out-of-sample,
nunca in-sample). Con un XGBoost pequeño y regularizado (lección de sobreajuste con 38 features sobre solo
299 batches DEV, documentada explícitamente) sobre las 10-11 features más importantes:

| Target | R² OOF | R² OOT walk-forward | R² lockbox |
|---|---|---|---|
| `rendimiento_proxy_batch` | 0.102 | 0.016 | 0.049 |

Mejora real sobre los baselines (agregados lineales simples ya daban R² OOT/lockbox **negativos**, peor que
predecir la media), pero sigue siendo una señal **modesta**. Confirma — no resuelve — el diagnóstico ya
documentado en §9.5/§11.7: el cuello de botella es la falta de masa de escoria y composición de carga por
escalón, no la técnica de modelado (ya se probó un modelo no lineal con features de trayectoria ricas, el
techo sigue bajo). Se declara explícitamente como diagnóstico complementario, no como base de prescripción
de batch completo.

### 13.8 Qué cambia y qué no para el uso prescriptivo

`modelo_prescriptivo.py` sigue siendo válido y usable tal cual — `modelo_predictivo_v2.py` lo complementa,
no lo reemplaza. Cambios recomendados para quien migre: (a) en Reducción, usar HistGB+monotónico(carbón) en
vez de XGBoost sin restricción (mejora validada, §13.5); (b) considerar Config C si se quiere poder
recomendar dosis de carbón (§13.4) a cambio de una pérdida marginal de MAE (0.642 vs. 0.593 lockbox); (c)
usar los intervalos CV+ (§13.6) en vez de la `uncertainty` no calibrada para cualquier decisión de "¿es
seguro recomendar este cambio?"; (d) tratar la tabla de causalidad (§13.1) como filtro adicional de qué
palancas recomendar de forma agresiva (`tasa_feed_Carbon_kg_min` en Reducción: alta confianza;
`tasa_gn_nm3_min`/`exceso_o2_combustion_pct`: confianza moderada; `posicion_vertical_lanza_mm`: no
recomendar cambios agresivos, efecto lineal despreciable y no-linealidad no explotable seguramente sin
más datos). Lo que NO cambió: el cuello de botella escalón→batch (§13.7) sigue siendo la limitación de
datos más importante del proyecto completo.

## 14. Modelo v3: carga por tolva, masa de escoria por trazador CaO y prescriptor en kg de Sn (sesión 2026-09-13 noche)

Sesión motivada por el pedido explícito de **volver a explorar y mejorar el modelo predictivo para el mejor prescriptor
operacional** (distinguir estado vs. control, capturar el efecto de las acciones dado el estado, optimizar cada escalón y
el rendimiento del batch), incorporando las columnas del Excel que las rondas anteriores no usaban y que el usuario
aclaró: `tmf Sn` y `tmh` por tolva (1 concentrado, 3 reciclo Sn/Fe, 4 mineral de Fe WF, 5 dross de Fe, 7 pellets/humos),
tolva 2 = carbón, tolva 6 = CaO. Todo el código vive en `dataset_lingo_smelter.py` (loader extendido, 61 columnas),
`feature_engineering.py` (bloque v3), `modelo_predictivo_v3.py` (nuevo módulo, complementa v1/v2) y
`experimentos/06-12_*.py` (+ `build_notebook_v3.py` → `modelo_v3_masa_escoria_y_prescriptor.ipynb`).

### 14.1 Verificaciones sobre el Excel (3982 filas, 362 batches) que cambian el significado de los datos

| Verificación | Resultado | Consecuencia |
|---|---|---|
| `TMH tolva 3+4+5+7+1` vs. suma de `tmh` tolvas 1,3,4,5,7 | igual en 3982/3982 filas | `feed_total_kgh` es exactamente la carga sólida sin cal ni carbón |
| `TMF Sn total F+R` vs. suma de `tmf Sn` por tolva | igual en 3982/3982 | `feed_Sn_kgf` = Σ Sn fino por corriente |
| `tmf_i / tmh_i` dentro de cada batch | std intra-batch = **0.000** en las 5 tolvas | el "Sn fino por tolva" es **tmh × ley del batch** (ensayo por corriente asignado al batch, no por escalón). Por eso `feed_Sn_kgf` ≈ `feed_total_kgh` (r=0.9988): la ley de la mezcla casi no varía (39.7% ± 1.7). La información nueva es (a) la **mezcla** por escalón y (b) la **ley por corriente** del batch (CONTEXT) |
| `CaO total (kg)` vs. `Kgmh tolva 6` | `CaO total` = **tolva 6 + tolva 3** exactamente (3982/3982) | la columna que las sesiones previas usaban como `feed_CaO_kgh` sumaba el reciclo Sn/Fe a la cal. **Corregido**: `feed_CaO_kgh` = tolva 6; la antigua queda como `feed_CaO_total_excel_kg` (DEPRECATED). Afecta `cum_feed_CaO_kg_prev`, `relacion_CaO_carga_Fe` y todo lo que dependía de la cal |
| `duracion plan` vs. duración real | 77.5% de escalones iguales; 13.1% más largos, 9.4% más cortos; r=0.935 | `duracion_plan_min` es una **ACTION primitiva real** (se decide antes); resuelve la ambigüedad ACTION/LEAKAGE de `delta_tiempo` (§11). En Fusión hay 163 escalones >10 min más largos que el plan |
| `HA-01 Estequiometría (%)`, `HA-01 Enriquecimiento de O2` | r = **1.000** con `exceso_o2_combustion_pct`+100 y con `oxygen_enrichment_pct` | valida los proxies de combustión de §3d; no aportan información nueva |
| `Flujo ... (Kg/h)` y `(Nm3/h)` | = masa o volumen del escalón / duración (r=1.000) | no se importan |
| `Rendimiento HA` | = Generación Sn crudo / (metal + dross + polvo); r=0.9988 con `rendimiento_proxy_batch`; la Generación excede al metal crudo por 0.72 ± 0.06 t | mismo KPI; y como el balance del batch cierra (feed Sn − metal − dross − polvo = −0.2 ± 3.4 t) equivale a la recuperación metal/feed (mediana 0.696) |
| Heel de escoria entre batches | corr %Sn del primer escalón vs. %Sn final del batch anterior = **−0.005** | los batches se tratan como independientes |
| Ley de Sn por corriente (mediana del batch) | concentrado 39.2%, reciclo t3 49.8% (0 en abril), mineral Fe WF t4 40.3%, dross t5 38.4%, pellets t7 37.7% | **pendiente con planta**: que a un mineral de Fe se le asigne ~40% Sn sugiere que t4 no es hematita pura |
| Refractario (`Espesor de ladrillo normalizado`) | decrece monótonamente 300 → 218 mm de abril a agosto 2026 | proxy de edad de campaña (CONTEXT); no correlaciona con el rendimiento (ρ=−0.07) |

### 14.2 Features v3 (feature_engineering.py, bloque v3) — entrada, contexto, estado y salida

- **ENTRADA (ACTION/DERIVED-ACTION, insumos del propio escalón)**: `tasa_feed_conc/reciclo/pellets_kg_min` (+ las de Fe/dross
  ya existentes), `frac_<corriente>_carga`, `frac_carga_secundaria` (reciclo + dross + pellets, ≈45% de la masa en Fusión),
  `ley_sn_carga_pct`, `relacion_CaO_carga` (cal/carga, mediana 0.097), `relacion_C_Sn_carga` (kg C por kg Sn cargado, mediana
  0.29 vs. estequiométrico 0.20), `tasa_feed_Carbon_fusion/reduccion_kg_min`, `duracion_plan_min`.
- **CONTEXTO (batch)**: `ley_sn_<corriente>_batch_pct`, `ley_sn_carga_batch_pct`, `espesor_ladrillo_norm_mm`, `escalon_idx`.
- **ESTADO (`_prev`, conocido antes del escalón)**: `cum_feed_total/conc/reciclo/pellets/carga_secundaria_kg_prev`,
  `termocupla_media_celsius_prev` (8 termocuplas de carcasa), `indice_irf_prev` (FeO/(SiO₂+Al₂O₃+CaO), índice de planta
  recalculado) y el **inventario por trazador CaO**: la cal (tolva 6) no se reduce ni volatiliza, luego
  `masa_escoria_est_kg = Σ CaO alimentado / (%CaO/100)`, `sn_inventario_escoria_est_kg = masa × %Sn/100`,
  `feo_inventario_escoria_est_kg`, `avance_reduccion_sn_prev = 1 − Sn en escoria / Sn cargado`. Supuestos declarados: el CaO
  de la ganga no está cuantificado (masa absoluta subestimada en un factor ~constante por batch), ruido del ensayo de %CaO
  ≈2% relativo (estimado de escalones consecutivos de Reducción sin carga), sin heel. Magnitudes (medianas): masa de
  escoria ≈65 t al fin de Fusión y ≈54 t al fin de Reducción (baja porque salen SnO y FeO), Sn en escoria ≈10.4 t al fin
  de Fusión y ≈0.7 t al final; el avance de reducción ya es ≈0.7-0.8 durante Fusión (la mayor parte del Sn va a metal
  durante la fusión misma).
- **SALIDA (TARGET, en masa)**: `sn_extraido_est_kg = Sn cargado − Δ(Sn en escoria)` (Sn que dejó la fase escoria: a
  metal crudo o a polvo), `frac_sn_extraido_escalon = sn_extraido / (inventario previo + cargado)` (fracción del Sn
  disponible; válida en ambas fases, a diferencia de T3), `tasa_extraccion_sn_kg_min`, `feo_extraido_est_kg` (kg de FeO
  que salieron de la escoria: la señal de sobre-reducción en masa), `selectividad_masa_sn_feo`, `d_masa_escoria_est_kg`.
  Medianas: Sn extraído por escalón de Fusión 3.3-11 t (fracción 0.42-0.52), R0 8.8 t (0.83), R1-R3 0.65/0.31/0.24 t.
- Todo clasificado en `CLASIFICACION_FEATURES` (222 columnas, 137 seguras; sin columnas sin clasificar).

### 14.3 Targets × feature sets × algoritmos (experimentos/06 y 11): dónde mejora v3 y dónde no

R² en el lockbox (últimos 63 batches), escala propia de cada target; OOF GroupKFold entre paréntesis:

| Fase | Target | v2 actual | v3 carga (+inventario) | pool seguro | Baseline trivial |
|---|---|---|---|---|---|
| Fusión | Δ%Sn (T1) | **0.47** (0.54) | 0.31-0.35 (0.55-0.57) | 0.28-0.33 (0.55-0.56) | — |
| Fusión | Sn extraído kg (M1) | 0.70 (0.71) | 0.62-0.71 (0.74-0.75) | **0.72** (0.75-0.76) | solo `feed_Sn_kgf`: **0.71** (0.69) |
| Fusión | fracción extraída (M2) | 0.02-0.04 (0.55-0.58) | ≈0 (0.59-0.61) | ≈0 (0.59-0.61) | — (en kg equivale a M1: 0.67-0.73) |
| Fusión | Δ%FeO | 0.10 (0.14-0.18) | 0.04-0.14 (0.18-0.22) | 0.08-0.12 (0.20-0.23) | — |
| Reducción | Δ%Sn (T1) | 0.97-0.98 (0.95) | 0.97 (0.96) | **0.98** (0.96) | — |
| Reducción | Sn extraído kg (M1) | 0.97 (0.92-0.93) | 0.98-0.99 (0.96-0.97) | **0.99** (0.97) | solo inventario previo: **0.98** (0.97) |
| Reducción | FeO extraído kg | 0.09-0.15 (0.09-0.15) | 0.21-0.32 (0.33-0.35) | 0.25-0.30 (0.34-0.36) | — |
| Reducción | Δ%FeO | 0.53-0.54 (0.45-0.47) | 0.52-0.57 (0.47-0.50) | 0.44 (0.51-0.52) | — |

Lecturas honestas:

1. **El target en masa es más predecible, pero casi todo por la carga.** En Fusión, `sn_extraido_est_kg` alcanza R²
   lockbox ≈0.70 vs. ≈0.45 del Δ%Sn, pero el baseline trivial "solo Sn cargado en el escalón" ya da 0.71 (OOF 0.69); los
   sets curados suben el OOF a 0.72-0.74 y no mejoran el lockbox. En Reducción, "solo inventario previo" da 0.98 (R0 extrae
   ≈83% del inventario). **La química explica poco R² adicional; su valor está en la respuesta a las palancas (§14.4), en la
   escala física común y en la restricción térmica, no en el R².**
2. **La fracción extraída (M2) no está rota: el lockbox es menos variable.** std de la fracción en Fusión: 0.156 en DEV vs.
   0.098 en lockbox (misma MAE ≈0.07) → R² ≈0 por construcción. Convertida a kg rinde igual que M1.
3. **Reducción ya era excelente y v3 lo mantiene** (Δ%Sn 0.97-0.98). La ganancia es de escala (kg) y de ΔT (ver 4).
4. **El ΔT pasa a ser predecible en Reducción** con termocuplas/refractario/gases/carga: R² OOF 0.44, lockbox 0.32-0.49
   (MAE 6-7 °C vs. std 11 °C; antes −0.04, §"Paso 38"). En Fusión sigue débil (0.39 / 0.10). El signo SHAP de
   `espesor_ladrillo_norm_mm` es positivo (refractario más grueso → menos pérdida → ΔT mayor), coherente.
5. **El Δ%FeO en puntos NO mide sobre-reducción en Reducción.** Al salir el Sn de la escoria (hasta −12 puntos en R0) el
   %FeO **sube por cierre de composición** aunque el FeO en masa **baje**: convertir el modelo en puntos a kg da R² = −1.8
   en el lockbox, mientras el modelo directo en kg da 0.27 (MAE 515 kg, std 814). Es ruidoso (ruido de trazador + ensayo
   ≈300-350 kg) pero es la **única señal de sobre-reducción con signo físico correcto**; reinterpreta la penalización
   `lambda_Fe·ΔFeO` (puntos) de v1/v2, que estaba confundida por el cierre de composición.
6. **Deriva de campaña en el lockbox (experimentos/12).** Dentro de DEV el walk-forward de Fusión Δ%Sn es ≈0.51-0.54 para
   todos los sets (sin deriva), pero cualquier familia v3 baja el lockbox (v2 0.44 → +carga 0.28, +contexto 0.32, todo 0.36).
   Causa: los últimos 63 batches (agosto 2026) operan **al final de la campaña de refractario**, fuera del rango de DEV:
   espesor 218-228 mm (mínimo DEV 235), termocuplas 378-410 °C (P95 DEV 384), ley de concentrado 41.5% (P95 DEV 41.4), baño
   más frío (995 vs. 1077 °C). Los árboles no extrapolan → **reentrenar/recalibrar conforme avanza la campaña** y preferir,
   para el modelo de ley de Fusión, sets sin contexto de campaña (v2 + `duracion_plan_min`: 0.517 / 0.450).

### 14.4 Efecto causal de las acciones dado el estado (DML cross-fitted con el estado v3 como control; experimentos/07 y 10)

Nuisance HistGB, GroupKFold por batch, IC95% bootstrap cluster, placebo 3σ y estabilidad en submuestras. Efecto por +1 sd
de la palanca sobre `sn_extraido_est_kg` (kg de Sn extraído de la escoria en el escalón):

| Fase | Palanca | θ por unidad | Efecto por +1 sd | Veredicto |
|---|---|---|---|---|
| Fusión | `tasa_feed_Carbon_kg_min` | +22.9 kg/(kg/min) [1.9, 38.2] | **+196 kg** | ROBUSTO (con Δ%Sn era NO significativo, §13.1) |
| Fusión | `relacion_CaO_carga` (cal/carga) | −1.8e4 [−2.8e4, −5.0e3] | **−423 kg** | ROBUSTO |
| Fusión | `tasa_feed_total_kg_min` | +8.6 [6.6, 11.1] | +338 kg | ROBUSTO |
| Fusión | `tasa_feed_pellets_kg_min` | +19.5 [11.6, 26.3] | +468 kg | ROBUSTO |
| Fusión | `tasa_feed_dross_Fe_kg_min` | +14.6 [5.8, 22.3] | +242 kg | ROBUSTO |
| Fusión | `tasa_gn_nm3_min` | +214 [153, 304] | **+315 kg** | ROBUSTO |
| Fusión | `posicion_vertical_lanza_mm` | −0.34 [−0.65, −0.07] | −415 kg | ROBUSTO (signo sobre la variable cruda; sentido físico sin confirmar) |
| Fusión | `exceso_o2_combustion_pct` | +2.3 [−24, +28] | ≈0 | NO significativo |
| Fusión | `duracion_plan_min` | +46 [−42, +144] | +453 kg | NO significativo (IC ancho) |
| Fusión | `frac_carga_secundaria` | +2188 [−14, +4670] | +201 kg | NO significativo (borde) |
| Reducción | `tasa_feed_Carbon_kg_min` | +8.7 [0.8, 14.7] | **+191 kg** | ROBUSTO |
| Reducción | `tasa_gn_nm3_min` | +19.8 [11.1, 34.5] | +99 kg | ROBUSTO |
| Reducción | `exceso_o2_combustion_pct` | −3.1 [−5.5, −0.9] | −35 kg | ROBUSTO (más oxidante → menos Sn extraído; sobre Δ%FeO +0.08 pts/sd ROBUSTO: retiene FeO) |
| Reducción | `posicion_vertical_lanza_mm` | −0.087 [−0.15, −0.03] | −63 kg | ROBUSTO (magnitud pequeña; sentido físico sin confirmar) |
| Reducción | `oxygen_enrichment_pct` | −9.0 [−25, +18] | −42 kg | NO significativo |
| Reducción | `duracion_plan_min` | +129 [−395, +758] | +757 kg | NO significativo (IC muy ancho: la duración de Reducción se decide según el avance) |

Sobre Δ%Sn en Reducción los mismos signos son robustos (carbón −0.38 pts/sd, GN −0.15, exceso O₂ +0.04, lanza +0.15),
y sobre Δ%FeO: GN −0.15 pts/sd (ROBUSTO), exceso O₂ +0.08 (ROBUSTO), carbón −0.07 (no significativo, IC cruza 0),
lanza −0.07 (moderado). Lectura conjunta: en Reducción el carbón es la palanca de Sn con menor efecto colateral
identificable sobre FeO en puntos, y el gas natural mueve ambos (Sn y FeO).

Sobre `d_ley_sn_escoria_pct` (puntos) en Fusión solo resultan robustas las palancas de **carga** (tasa total, dross, pellets:
+0.39 a +0.44 pts por sd — más carga → más SnO₂ acumulándose), y carbón, cal, GN, O₂ y lanza no son significativas: el
target de ley no permite identificar las palancas químicas en Fusión, el de masa sí. Sobre Δ%FeO en Fusión el carbón sale
**positivo** (+0.07 pts/sd) — el artefacto de cierre de composición del punto 5 de §14.3, no una oxidación de Fe.

**DML conjunto** (`modelo_predictivo_v3.efectos_marginales_palancas_v3`: todas las palancas residualizadas a la vez sobre
el mismo estado; experimentos/10_efectos_palancas_conjunto_v3.csv): ver tabla en el notebook (Paso 62). Complementa la
lectura univariada de arriba y es la que el prescriptor expone como "efecto promedio por unidad de palanca dado el estado",
junto con `sensibilidad_local_v3` (diferencias finitas del modelo no lineal en el estado concreto de cada escalón).

### 14.5 Configuración final v3 (`modelo_predictivo_v3.FEATURES_V3_POR_DEFECTO`, experimentos/10 y 11)

Sets curados (interpretables, sin cocientes difíciles, sin duplicados, signos SHAP verificados) elegidos por R² OOF; el
lockbox solo se reporta:

| Fase | Target (clave) | k | R² OOF | R² lockbox | MAE lockbox | Uso en el prescriptor |
|---|---|---|---|---|---|---|
| Fusión | `sn_extraido_est_kg` (sn_kg) | 13 | 0.722 | 0.680 | 1113 kg | objetivo J |
| Fusión | `d_ley_sn_escoria_pct` (sn_pts) | 7 | 0.517 | 0.450 | 2.14 pts | continuidad v1/v2, informativo |
| Fusión | `d_ley_feo_escoria_pct` (feo_pts) | 9 | 0.202 | 0.042 | 0.78 pts | informativo (sin penalización en J) |
| Fusión | `d_temperatura_horno_celsius` (dT) | 15 | 0.385 | 0.097 | 7.5 °C | ventana térmica (rara vez activa) |
| Reducción | `sn_extraido_est_kg` (sn_kg) | 10 | 0.971 | 0.983 | 347 kg | objetivo J |
| Reducción | `d_ley_sn_escoria_pct` (sn_pts) | 11 | 0.948 | 0.968 | 0.66 pts | continuidad v1/v2 (equivalente en pts del modelo kg: 0.975) |
| Reducción | `feo_extraido_est_kg` (feo_kg) | 11 | 0.286 | 0.271 | 515 kg | penalización λ_Fe en J |
| Reducción | `d_ley_feo_escoria_pct` (feo_pts) | 11 | 0.465 | 0.570 | 0.58 pts | informativo |
| Reducción | `d_temperatura_horno_celsius` (dT) | 14 | 0.444 | 0.320 | 7.2 °C | ventana térmica |

Signos SHAP (corr valor–SHAP) coherentes: Fusión sn_kg → `feed_Sn_kgf` +0.97, `relacion_CaO_carga` **−0.88**,
`interaccion_C_x_Sn_prev` +0.83, `tasa_feed_Carbon_kg_min` +0.79, `frac_carga_secundaria` +0.40, `basicidad_B2_prev` −0.20;
Reducción sn_kg → inventario +0.99, `avance_reduccion_sn_prev` −0.96, `interaccion_C_x_Sn_prev` +0.95, carbón +0.79,
GN +0.41, `exceso_o2_combustion_pct` **−0.42**; Reducción Δ%FeO → `cum_feed_Carbon_kg_prev` −0.92, `interaccion_C_x_FeO_prev`
−0.65, exceso O₂ +0.54. Selección automática (permutación estable + Boruta-shadow + decorrelación, experimentos/09)
usada como guía: su set k=8 para Fusión sn_kg (OOF 0.740 / lockbox 0.701) incluye `margen_presion_gn` (nulos) y
`carbon_sobre_inventario_sn_prev` (cociente de piso, §11.7); se prefirió el set curado por interpretabilidad.

**Cobertura de los intervalos CV+ en el lockbox** (nunca usado para calibrar; nominal 80% / 90%): Fusión sn_kg 89.3% / 97.1%
(ancho medio 5.1 / 7.1 t), sn_pts 87.5% / 94.1%, feo_pts 85.1% / 93.1%; Reducción sn_kg 93.3% / 96.2% (ancho 1.7 / 2.6 t),
sn_pts 92.6% / 96.7%, feo_kg 93.4% / 96.3% (ancho 2.6 / 3.1 t — el intervalo de FeO es tan ancho como su rango, coherente
con R² 0.27), feo_pts 91.7% / 95.9%; ΔT 98-100% (intervalos conservadores de ±20-30 °C). Todas ≥ nominal: dirección segura
para prescribir (consideraciones §85). Ver `experimentos/10_cobertura_cvplus_v3.csv`.

**DML conjunto (todas las palancas residualizadas a la vez sobre el estado v3; `experimentos/10_efectos_palancas_conjunto_v3.csv`)**:
en Fusión sobre kg de Sn extraído sólo quedan significativas la **cal** (−403 kg por +1 sd de `tasa_feed_CaO_kg_min`, IC95%
[−693, −109]), la **tasa de carga** (+365 [227, 529]) y el **gas natural** (+132 [1, 293]); el carbón, que era robusto en
el DML univariado (+196 kg/sd), pierde significación cuando se controla la tasa de carga con la que covaría (−22 [−197, +135]):
en Fusión la dosis de carbón sigue a la carga y su efecto propio, separado de ésta, no se identifica con estos datos. En
Reducción: GN +75 kg/sd [10, 140] sobre Sn pero también **+187 kg/sd de FeO reducido** [48, 332] (el GN empuja ambas
reducciones: es la palanca de menor selectividad); carbón +189 kg/sd de Sn [−0.2, 360] (borde) con efecto nulo sobre FeO
(−15 [−174, +207]) — la palanca más selectiva; aire −56 kg/sd de FeO [−112, −1]; O₂ baja la temperatura (−3.0 °C/sd
[−5.9, −0.5]) y también el aire (−1.1 °C/sd). La duración planificada no es identificable en ninguna fase (IC ±1 t):
se decide en función del avance, no al revés.

### 14.6 El rendimiento del batch: dos canales de pérdida, el precio de sobre-reducir (λ_Fe) y el precio de extraer Sn prematuramente en Fusión (λ_F)

`rendimiento_ha_batch` = Sn a metal / (metal + dross + polvo) se descompone en **Sn a dross de Fe** (12.7% ± 3.9 del Sn
cargado; ρ = −0.76 con el rendimiento) y **Sn a polvo** (18.3% ± 4.0; ρ = −0.54), **independientes entre sí** (ρ = 0.02).
Con features de trayectoria v3 (incl. rollups en masa) el KPI sigue siendo poco predecible (R² OOF 0.11-0.18, walk-forward
≈0, lockbox ≈0; Spearman en lockbox 0.24 rendimiento, 0.40 dross), pero las asociaciones son consistentes con el mecanismo
de sobre-reducción: caída total de FeO en Reducción (ρ = +0.27 con rendimiento, −0.25 con dross), `delta_FR_ley_feo` (+0.24),
selectividad en masa Sn/FeO (+0.25), fracción de FeO extraído (−0.26), IRF al fin de Fusión (+0.22). El canal de polvo no se
explica con los agregados disponibles (|ρ| ≤ 0.16; los pellets recirculados no correlacionan con el polvo del batch
anterior, r=0.03: vienen de stock).

**λ_Fe**: OLS entre batches `Sn_dross (t) = 5.7 + 0.31·FeO_extraído_Reducción (t)` (p=0.007; con feed y Sn del dross
alimentado como controles, 0.28 [0.08, 0.49]). Cada kg de FeO reducido en Reducción cuesta ≈0.3 kg de Sn a dross —
`LAMBDA_FE_KG_SN_POR_KG_FEO = 0.30` en el objetivo v3, en las mismas unidades que el Sn extraído.

**λ_F — ¿conviene extraer Sn durante la Fusión? No.** El Sn extraído de la escoria a lo largo del batch es ≈ fijo (Sn
cargado − ≈1 t en la escoria final), así que extraer más en Fusión sólo deja menos para Reducción; lo que varía entre
batches es **cuándo** se extrae. La fracción del Sn cargado que ya salió de la escoria al fin de Fusión (`F_avance_final`,
media 0.80, quintiles 0.70-0.90) **reduce** la fracción a metal: OLS `f_metal ~ F_avance_final` = **−0.143** (t=−3.7,
p<0.001) controlando ley del concentrado, refractario, temperatura media de Fusión y FeO reducido en Reducción (este
último −0.0070/t, p=0.005); **aumenta** el dross (+0.085, p=0.004; FeO reducido +0.0074/t, p<0.001) y reduce levemente el
polvo (−0.059, p=0.046). Por quintiles de avance: f_metal 0.705/0.717/0.690/0.682/0.681. Cada t de Sn conservada en la
escoria al fin de Fusión suma ≈0.13 t de metal crudo (`metal_t ~ 0.128·Sn_escoria_fin_Fusión_t + 0.51·feed_Sn_t`,
p=0.001). Es el principio clásico de dos etapas del horno de Sn (fundir oxidando manteniendo el Sn como SnO₂ en la
escoria, y reducir después con selectividad controlada), ahora cuantificado: `LAMBDA_F_KG_METAL_POR_KG_EXTRAIDO_FUSION =
0.13`. Resuelve además la ambigüedad `direccion_sn_fusion` que v1 dejó como parámetro de negocio (§"Paso 44"): la
evidencia apoya +1 (crecer %Sn en escoria durante Fusión es deseable). Nótese que las palancas que en §14.4 **aumentan**
la extracción en Fusión (carbón, GN, pellets, lanza baja) son, por lo tanto, palancas a **moderar** en Fusión, no a
maximizar.

### 14.7 Prescriptor v3 (`modelo_predictivo_v3.py`)

- **Roles**: `CONTEXT_V3` (ensayos del batch, refractario, escalón), `STATE_V3` (leyes/temperaturas previas, acumulados,
  inventario por trazador), `ACCIONES_PRIMITIVAS_V3` (duración planificada, tasas por tolva, cal, carbón, GN, O₂, aire).
  `ACCIONES_OPTIMIZABLES_V3` mueve por defecto carbón, cal, GN, O₂, aire y duración; la mezcla de carga se respeta como plan
  del operador (configurable), porque "cargar más" sería trivialmente óptimo para un objetivo en kg.
- **Simulador**: `recalcular_derivadas_v3` reconstruye todas las derivadas (fracciones, ley de la mezcla, cal/carga, C/Sn,
  estequiometría, enriquecimiento, interacciones) desde primitivas + estado + contexto; cambiar el feature set no exige
  reescribirlo. `simulate_action_v3` devuelve Sn extraído [kg], FeO reducido [kg], ΔT y T predicha, todos con intervalo CV+.
- **Objetivo en kg de metal equivalente**: Reducción `J_R = Sn_extraído − λ_Fe·FeO_reducido − costos(C, GN, O₂, tiempo) −
  w_T·(T fuera de P5-P95) − w_U·ancho CV+ − w_S·(1 − soporte)`; Fusión `J_F = −λ_F·Sn_extraído − costos − …` (λ_Fe = 0 en
  Fusión: régimen oxidante y señal de FeO dominada por la carga de Fe y el cierre de composición). Pesos de costo =
  placeholders sin precios reales. `ACCIONES_OPTIMIZABLES_V3`: Reducción carbón, GN, O₂, aire y duración planificada; Fusión
  **sólo carbón y cal** (gases, carga y duración quedan en el plan del operador: el modelo térmico de Fusión es débil y
  alargar el escalón sólo "carga más Sn"). Soporte histórico mínimo **duro** (P10 de DEV) además del término blando
  (w_S = 2000 kg): en la primera versión, sin ese umbral y con w_S = 300, el optimizador recomendaba +33 t de Sn por batch
  con soporte 0.07 — un artefacto de extrapolación que la restricción elimina (soporte recomendado 0.37 vs. histórico 0.29).
- **Vectorización**: `_predecir_lote`, `simular_acciones_lote_v3`, `objetivo_lote_v3` (un predict por modelo por lote de
  candidatos): optimizar un escalón (300 + 60 candidatos) tarda ≈0.4-1 s y un batch completo ≈5-20 s; la validación de
  cobertura CV+ pasa de minutos a <1 s.
- **Efecto de las acciones dado el estado**: `efectos_marginales_palancas_v3` (DML **conjunto**, todas las palancas a la vez,
  IC95%) y `sensibilidad_local_v3` (diferencias finitas del modelo no lineal en el estado concreto del escalón).
- **Reporte por batch**: `reporte_recomendaciones_batch_v3` / `resumen_reporte_batch_v3` en kg de Sn (Sn adicional extraído,
  FeO adicional reducido y saldo neto = Sn − λ_Fe·FeO), `evaluar_politica_en_lockbox_v3`.

**Evaluación de la política recomendada en 8 batches del lockbox** (`evaluar_politica_en_lockbox_v3`, cada escalón sobre su
estado real, sin encadenar; `experimentos/10_politica_lockbox_v3.csv`):

| Batch | ΔSn extraído Fusión (kg, − favorable) | ΔSn extraído Reducción (kg, + favorable) | ΔFeO reducido Reducción (kg, − favorable) | Metal equivalente (kg) | Soporte hist. → rec. | % escalones con mejora |
|---|---|---|---|---|---|---|
| AP0300 | −2238 | +733 | −113 | **+1058** | 0.29 → 0.38 | 90 |
| AP0301 | −1738 | +1043 | −354 | **+1376** | 0.38 → 0.44 | 100 |
| AP0302 | −1064 | +822 | +455 | +824 | 0.33 → 0.38 | 90 |
| AP0303 | −1451 | +911 | +1324 | +702 | 0.31 → 0.40 | 100 |
| AP0304 | −452 | +359 | −92 | +445 | 0.33 → 0.45 | 100 |
| AP0305 | −518 | +360 | +380 | +314 | 0.36 → 0.41 | 80 |
| AP0306 | −1138 | +542 | +115 | +655 | 0.39 → 0.44 | 100 |
| AP0307 | −1230 | +441 | +124 | +563 | 0.42 → 0.46 | 100 |

Lectura: el uplift estimado es de **+0.3 a +1.4 t de metal equivalente por batch** (mediana ≈ +0.7 t ≈ 2% de los ≈35 t de
metal), con soporte histórico de las recomendaciones **mayor** que el de la operación real (la restricción dura P10
descarta extrapolaciones; en escalones de soporte bajo, p.ej. AP0300 R1 con 0.12, la recomendación coincide con la acción
histórica). Es un uplift **estimado por el modelo**, no observado; ~1/3 proviene de conservar Sn en Fusión (vía λ_F, el
coeficiente menos preciso) y ~2/3 de Reducción. Ejemplo AP0300 (`10_reporte_batch_ejemplo_v3.csv`): en F1-F6 se recomienda
menos carbón y más cal dentro del soporte (−0.2 a −1.0 t de extracción prematura por escalón, ΔT +1 a +5 °C); en R0
+3.5 Nm³/min de GN y −1.5 kg/min de carbón (+478 kg Sn, −163 kg FeO); en R2-R3 alargar 5 min y subir GN. Sensibilidad local
en AP0300 R0: +1 kg/min de carbón → +12 kg Sn pero +48 kg FeO reducido (selectividad marginal 0.25, ya bajo el umbral
λ_Fe⁻¹); +1 Nm³/min de GN → +94 kg Sn y −33 kg FeO; en F2: +1 kg/min de cal → −46 kg de Sn extraído prematuramente.

### 14.8 Qué cambia respecto a v1/v2 y qué sigue pendiente

Cambia: (1) `feed_CaO_kgh` = cal pura; (2) la duración planificada reemplaza a la real como acción; (3) existe masa de
escoria/inventario por escalón (trazador CaO) y targets en kg; (4) el FeO en puntos deja de usarse como penalización
(cierre de composición) y se usa FeO en kg con λ_Fe físico; (5) se identifican causalmente las palancas químicas de Fusión
(carbón +, cal −, GN +, lanza −) **y se establece la dirección del objetivo de Fusión** (conservar el Sn en la escoria,
λ_F ≈ 0.13); (6) ΔT de Reducción modelable → restricción térmica; (7) el prescriptor reporta en kg de metal equivalente,
con soporte mínimo duro, y expone el efecto de cada palanca dado el estado. No cambia: Reducción como fase mejor modelada;
el KPI de batch sigue poco predecible; `posicion_vertical_lanza_mm` sigue solo como estado `_prev`.

Pendientes con planta: sentido físico de la posición de lanza; naturaleza de la tolva 4 (~40% Sn asignado); CaO de la ganga
del concentrado (calibraría la masa absoluta de escoria); precios de reactivos para los costos de J; y confirmar el protocolo
de cierre de escalón (por qué el 22.5% de los escalones no cumple el plan). Metodológicos: reentrenamiento por campaña
(§14.3 punto 6) y validación piloto de las recomendaciones antes de uso operativo.

## 15. Iteración 4: modelo prescriptor v4 (2026-09-14, mañana) — puntero

La iteración 4 (`experimentos/13-21`, diseño en `experimentos/ITERACION_4_diseno.md`) quedó documentada en
`candidate_win_model.md` y `win_analytics.ipynb` (PLM por fase, `modelo_predictivo_v4.py`, λ_Fe = 0.60, objetivo de Fusión
anclado en KPI, evidencia off-policy sólo en lockbox). No se repite aquí; las referencias "§15.x" de esa ficha apuntan a sus
secciones internas.

## 16. search_targets: qué target por escalón debe maximizar el recomendador (iteración 5, 2026-09-14 tarde)

Familia `experimentos/search_targets/` (README de la carpeta con el detalle): 36 targets candidatos por escalón (20 de
Reducción, 16 de Fusión), calculados desde teoría, con dirección y regla de agregación escalón→batch coherentes, evaluados
contra `rendimiento_proxy_batch` (ST-01 correlación con BH/controles, ST-02 predictibilidad, ST-03 palancas y cadena de
signos, ST-04 robustez, ST-05/06/07 síntesis).

### 16.1 Ganadores

- **Reducción — SDI, índice de agotamiento selectivo**: `SDI[t] = ln(Sn_inv[t−1]/Sn_inv[t]) + 10·ln(FeO_inv[t]/FeO_inv[t−1])`;
  suma telescópica por batch `ln(Sn_ini/Sn_fin) + 10·ln(FeO_fin/FeO_ini)`. Adimensional; la masa de escoria del trazador se
  cancela en cada log-cociente. ρ Spearman con el rendimiento en DEV **+0.319** [0.21, 0.43], p = 3·10⁻⁸ (BH sobre 37 tests
  < 10⁻⁵), positivo en los tres tercios cronológicos (+0.19/+0.35/+0.48), OLS HC3 con controles y tendencia **+1.55 pp de
  rendimiento por sd** (p < 0.001), monótono en los 5 quintiles (Q5 − Q1 = +4.4 pp), ρ −0.20 con dross y −0.18 con polvo.
  En el lockbox (fin de campaña) la asociación con el propio batch es +0.09 (n.s.) pero con el **batch siguiente** es +0.35
  (p = 0.005; OLS +1.22 pp/sd, p = 0.035): heel químico de escoria (ver 16.3). w = 10 se eligió sólo con DEV (menor w con
  ρ ≥ 0.95·máx; el máximo 0.33 está en w = 12-16). La referencia v4 `Sn − 0.6·FeO` da ρ +0.035 (n.s.).
- **Fusión — IRF al cierre del escalón**: `%FeO/(%SiO₂+%Al₂O₃+%CaO)` (índice de planta), agregado = último valor de la
  fase. ρ DEV **+0.195** [0.09, 0.30], p = 0.0007 (BH 0.006); lockbox **+0.33** (p = 0.008); OLS +1.21 pp/sd (p < 0.001);
  positivo en los 4 bloques temporales; ρ −0.21 con polvo. Único candidato de Fusión significativo tras BH que replica en
  lockbox. La referencia v4 (−Sn extraído) da +0.06 (n.s.).

### 16.2 Qué se descartó y por qué

- Extraer Sn en Reducción (kg, fracción, k aparente, Δavance, Sn residual): ρ ≈ 0 o negativo en DEV. El Sn extraído total
  es ≈ fijo; lo que decide el KPI es cuánto Fe se metaliza (dross) y el estado de la escoria. Por eso `Sn − λ·FeO` sólo
  correlaciona con λ ≫ 0.6 (barrido sin máximo interior hasta λ = 2) y el objetivo v4 no correlaciona.
- Selectividad como cociente de masas Sn/FeO: ρ 0.02 (hereda el ruido de dos diferencias del trazador; R² máximo ≈ 0).
- Conservación de Sn en Fusión (λ_F): ρ 0.06 en DEV. Térmicos y polvo: no correlacionan.
- Los tres de la familia FeO (fracción retenida, kg, Fe metalizado/Sn) son redundantes (ρ 0.88); el SDI los subsume y evita
  el objetivo degenerado "no reducir".

### 16.3 Hallazgo nuevo: heel químico de escoria y estado inicial de la Fusión

No hay heel de Sn (§14.1) pero sí de matriz: el IRF al cierre de F0 correlaciona ρ 0.46 (p 10⁻²⁰) con el IRF final de la
Reducción del batch anterior (SiO₂ 0.47, Al₂O₃ 0.32). El IRF final de Reducción del batch b predice el KPI del batch b+1
(ρ +0.23, p 8·10⁻⁶). En Fusión, la correlación del IRF con el KPI es máxima al cierre de F0-F1 (0.32-0.38 DEV, 0.54-0.57
lockbox) y decrece hasta F6 (0.19); el cambio de IRF dentro de la Fusión no aporta información una vez controlado el IRF
inicial (p 0.60), y ninguna palanca de lanza/carbón mueve el IRF en el escalón (PLM θ ≈ 0). Lectura: el target de Fusión es
de **conservación de estado** (no reducir FeO con GN/temperatura, no diluir con formadores de red), y la palanca real sobre
el estado inicial de la Fusión es la escoria final de la Reducción anterior, que el SDI ya premia.

### 16.4 Predictibilidad y palancas de los ganadores (ST-02/ST-07)

SDI por escalón: R² OOF 0.26 (estado), 0.25 (estado + control), lockbox 0.24; techo de ruido del trazador para la componente
FeO ≈ 0.82. PLM v4: GN **−0.153 sd/sd** [−0.25, −0.05] (reductor no selectivo), carbón +0.08 (IC cruza 0), O₂ +0.06,
aire +0.03; sobre la componente FeO sólo el GN es significativo (−0.023) y el carbón ≈ 0 (palanca selectiva); sobre la
componente Sn, carbón +0.09* y GN +0.06*. Cadena de signos coherente con exp 19 (GN total de Reducción sube dross y baja
metal; carbón total baja dross). IRF de Fusión: R² OOF 0.71 (estado), θ ≈ 0 para las 4 palancas.

### 16.5 Límites

Evidencia observacional con controles; ρ ≈ 0.3 explica ~10 % de la varianza de un KPI que entre batches es casi ruido
blanco (autocorrelación 0.12, controles + tendencia R² 0.025). El SDI sobre el propio batch no replica en el lockbox
(régimen de fin de campaña) y sí sobre el batch siguiente. Siguiente paso: reemplazar en `modelo_predictivo_v4` el
objetivo de Reducción por el SDI (o `J_R` con λ ≥ 2 en la escala de masa) y el de Fusión por conservación del IRF, y validar
con piloto.

### 16.6 Ronda 2 (misma tarde): ¿existe un target con ρ o R² > 0.70? Techo del KPI y mejor target alcanzable

Se pidió un target cuyo agregado correlacione > 0.70 con el rendimiento proxy. **No existe con este KPI**, y se demostró con tres
cotas independientes (`experimentos/search_targets/README.md` §R2, ST-08/10/11): (i) fiabilidad: el proxy y la recuperación
real (metal / Sn cargado) concuerdan sólo r = 0.57 → ρ_max ≈ 0.75 incluso para un predictor perfecto; el proxy correlaciona
−0.37 con el cierre del balance de Sn y sólo +0.50 con el metal (−0.74 dross, −0.60 polvo); (ii) estructura: el KPI es
exactamente 100·(1 − f_dross − f_polvo), el polvo aporta el 42 % de la varianza y es impredecible desde los escalones (R² OOF
≤ 0.09), y dross y polvo tienen autocorrelación anómala a lag 3 (0.38 y 0.48: sospecha de atribución periódica); (iii) techo
empírico: combinando los 38 agregados teóricos + heel + controles (Ridge/Lasso/HGB, CV 5×5) R² OOF 0.20 y ρ 0.46, walk-forward
cronológico R² ≈ 0; con toda la información de escalón no sube.

Mejor target nuevo (ST-09/ST-12, `R23_lnirf_R`): **ln IRF al cierre de cada escalón de Reducción**, agregado por media
ponderada por tiempo: ρ +0.304 [0.19, 0.41] (DEV, p 8·10⁻⁸), **+0.380 (lockbox, p 0.002)**, +0.331 (total, p 10⁻¹⁰); OLS +1.83
pp/sd (DEV) y +1.49 (lockbox, p 0.024); 4/4 quintiles crecientes (+3.9 pp); ρ +0.42 con el KPI del batch siguiente; Monte
Carlo de ruido ρ 0.303 ± 0.006; PLM: GN −0.24 sd/sd [−0.37, −0.13], O₂ +0.19 [+0.04, +0.38], carbón ≈ 0. Se descompone en
0.91·ln IRF fin de Fusión (estado heredado, heel) + 0.28·ln(FeO_fin/FeO_ini) (acción: la componente FeO del SDI), lo que
explica que replique en ambos regímenes. La suma estandarizada z(SDI) + z(lnIRF_R) es el mejor correlato: ρ 0.41 [0.30, 0.51]
DEV, 0.40 total (p 10⁻¹⁴), 0.26 lockbox (p 0.04), +5.7 pp entre quintiles extremos. Familia 2 (30 candidatos: balances de Fe
y Sn, NBO/T, FeO/SiO₂, Al₂O₃, sobrecalentamiento, T gases, fume, tiro): sólo la matriz durante la Reducción replica en lockbox;
el resto ≤ 0.13. Compuesto supervisado de 6 términos (ST-10): R² OOF 0.21, dominado por el estado inicial.

Conclusión: para superar ρ 0.5 hace falta cambiar el KPI (atribución consistente de dross y polvo por batch, o la recuperación
real, más predecible fuera de muestra), no el target. Con el proxy actual, ρ ≈ 0.3-0.4 es el máximo defendible.

### 16.7 Ronda 3: aplicar la recomendación (cambiar el KPI)

(a) Atribución de dross y polvo: la correlación cruzada con las señales de proceso a lags −6..+6 (`st_13_crosscorr.csv`)
muestra que el dross está atribuido al batch correcto (máximo en lag 0 con Sn y Fe cargados) y que el polvo no responde a
ninguna señal a ningún lag: su autocorrelación a lag 3 es una periodicidad del registro, no un retardo corregible. Desplazar
D y P o agrupar en ventanas no mejora la correlación de los targets y las ventanas inducen autocorrelación artificial. Sin
datos de planta sobre el pesaje y asignación del polvo no hay atribución alternativa defendible.
(b) Recuperación real (100·M/Sn cargado) como KPI: el mejor par es ln IRF en Reducción, ρ 0.26 (DEV, BH < 0.001), **0.50
(lockbox, p < 10⁻⁴)**, 0.32 (total, p < 10⁻⁹); IRF fin de Fusión 0.28/0.35/0.30; la conservación de Sn en Fusión (−Sn
extraído) reaparece con signo teórico (0.25/0.29/0.23, coherente con λ_F). 14 de 39 candidatos BH-significativos.
Conclusión: aplicada la recomendación, el máximo alcanzable sigue en ρ 0.3-0.5. Superar 0.70 requiere datos que el Excel no
contiene (Sn en dross y polvo pesado y asignado por batch, balance de Sn cerrado con heel y escoria residual).

## 17. Iteración 6: balances de masa y energía inspirados en el modelo teórico de los metalurgistas (2026-09-14, noche)

Familia `experimentos/balances/` (README propio con el detalle; diseño en `ITERACION_6_diseno.md`; librería `bal_features.py`).
Punto de partida: el análisis del modelo Rust `func-teorethical-model` (balance inverso por batch: los targets de %Sn en escoria y
temperatura son entradas y el solver despeja carbón, gas y oxígeno) y las ideas del equipo de planta (masa en vez de ley, conservación
de masa y energía, temperatura, pérdidas por volatilización/splash/finos, rendimiento refinado con Sn en escoria, Fe como monitor de
sobre-reducción). Cuatro experimentos paralelos (B-01 multi-trazador, B-02 balance reductor, B-03 balance de energía, B-04 balance
global y KPI refinado) más hipótesis propias (B-05).

### 17.1 Lo que se adopta

- **KPI refinado con Sn en escoria** (`feature_engineering.construir_resumen_batch_balance_global`, bloque v6):
  `recuperacion_refinada_pct = 100·M/(M+D+P+Sn_escoria_final)`, con la masa final de escoria por balance global de masa (estructura del
  modelo Rust: carga por tolva − humedad − Sn de salida como SnO₂ − FeO del Fe metalizado + O del Fe metálico escorificado − polvo
  no-Sn + ceniza). Masa final mediana 57.9 t (trazador CaO 53.5 t; el cociente entre ambos depende de la fracción de cal, ρ −0.35: es el
  artefacto contable de exp 18, por lo que no se usa un factor multiplicativo). Sn en escoria final 690 kg (1.3 % del Sn de salida, IQR
  1.0-1.7 %). Cierre del Sn: sesgo −1.9 % (p 4·10⁻⁸), sd 6.6 %. El KPI correlaciona 0.99 con el proxy pero reclasifica el 9.4 % de los
  batches de cuartil, es robusto a los supuestos (< 0.2 pp) y su techo de predictibilidad iguala o supera al del proxy (Ridge lockbox
  R² 0.089 vs 0.002). Con este KPI, en el lockbox replican los targets de matriz de escoria (ln IRF en Reducción ρ 0.38, p 0.002; IRF fin
  de Fusión 0.34) y no el SDI (0.09): refuerza la recomendación de §16.6 (lnIRF_R o z(SDI)+z(lnIRF_R) como objetivo de Reducción).
- **Features térmicas de balance de energía en el modelo de ΔT de Reducción** (`b6_q_neto_por_min_MJ`, `b6_dT_teorico_sin_reaccion`,
  `b6_q_gases_MJ`, `b6_margen_termico_MJ`; fórmulas en `bal_features.py`): R² OOF 0.466→0.474 y lockbox 0.465→0.501 (HGB 0.485/0.504).
  Ganancia pequeña; en Fusión no aportan (0.466/0.052 vs 0.481/0.066). Quedan como estado opcional del modelo térmico.

### 17.2 Lo que se rechaza (con la evidencia)

- **Cierre multi-inerte (CaO, SiO₂, Al₂O₃, MgO) para la masa de escoria en Reducción.** En las mismas filas, el trazador CaO re-anclado
  conserva señal por escalón (FeO reducido: R² OOF HGB 0.25, autocorrelación lag-1 −0.15) y cualquier consenso multi la pierde
  (R² ≈ 0, lag-1 −0.50..−0.68, sd del FeO reducido 1.1-2.0 t frente a 0.7 t); a nivel batch ρ con el KPI 0.23 frente a 0.32 del SDI v3;
  los IC del carbón se ensanchan 2.5-6×. Los ln-cocientes de SiO₂/Al₂O₃/MgO entre escalones tienen sd 7-15 % y el sesgo se concentra en
  el primer escalón de Reducción (mezcla incompleta de la cal). Lección: la lógica de "óxidos inertes" del modelo teórico sirve para el
  balance de diseño, no para trazar masa por escalón con los ensayos de planta.
- **Balance reductor/oxidante en kg C-eq como palanca del PLM** (`b6_reductor_neto_kg_ceq` = C fijo + CO de lanza + Fe metálico −
  2C·Sn disponible − C·Fe₂O₃). Descriptiva coherente con el modelo Rust (Fusión siempre en déficit; R1 único escalón con reductor neto
  positivo, +251 kg, capacidad de sobre-reducir 1.35 t de FeO). Pero R² OOF idéntico en 5 targets (Δ ≤ 0.002), θ(FeO) +115 [−12, +346]
  kg/sd (IC 4× más estrecho que el del carbón v4, sin llegar a significancia, estable en toda la grilla de supuestos), signo incorrecto
  sobre Sn (resta la demanda de Sn, que es el motor cinético). Los acumulados `_prev` degradan el lockbox (SDI 0.143→0.077).
- **Balance de energía físico calibrado como modelo de ΔT y calor de reacción aparente como estimador de la extracción.** Coeficiente de
  combustión calibrado 0.03 (teoría ~1), pérdidas 0, R² OOF 0.03-0.09; q_neto correlaciona −0.33 con ΔT en Reducción: la temperatura de
  cara interior no responde al balance térmico del baño. ρ(calor aparente, calor del trazador) −0.25 (F) / +0.21 (R): sin estimador
  independiente del ensayo de escoria. Faltan aire de post-combustión, temperatura de baño y análisis del carbón.
- **Cementación total Fe + SnO → FeO + Sn (ec. 6 del modelo Rust).** El C-eq del Fe metálico del dross en el PLM de Sn extraído en Fusión
  da θ −40..−85 kg/sd (signo contrario); a nivel batch el dross de Fe en Fusión no mueve f_dross ni f_metal. El dross de tolva 5 se
  escorifica, no cementa.
- **"Carbón perdido al tiro" 19.5 % (F) / 48 % (R) del modelo Rust.** La utilización del reductor por batch (consumo estequiométrico del
  Sn y FeO reducidos / oferta) es 1.03 (F) / 1.05 (R): el carbón alimentado coincide con la demanda estequiométrica; no hay margen para un
  48 % perdido. La regresión de la utilización contra proxies de arrastre (tiro, gas, lanza, pellets) explica R² 0.12 con signos mixtos.

### 17.3 Ronda 2 (señal débil o forma inadmisible)

Residuo de masa no explicado por reducción (sesgado +1.1 t en el primer escalón de Reducción; ρ 0.13-0.16 con gas de lanza; sin relación
con polvo ni KPI: bandera de QC, no feature); modelo multi-óxido de talón + ganga por tolva (coeficientes de SiO₂ saturan al 100 %, talón
en cota, R² hold-out ≤ 0, masa 5.4× la del balance; rehacer con pureza de cal fija, anclaje de escala y menos parámetros); avance de la
pérdida de FeO como estado (FeO OOF 0.367→0.383 pero lockbox 0.378→0.361; SDI OOF 0.131→0.154 pero lockbox 0.143→0.044: reevaluar con
reentrenamiento por campaña).

### 17.4 Monitor de Fe (idea del equipo de planta)

La pérdida de FeO (> 3 % del inventario) empieza en el primer escalón de Reducción en 283/362 batches; el %Sn de la escoria al inicio de
esa pérdida no discrimina batches (ρ −0.04 con f_dross), la pérdida total sí (ρ +0.17 con f_dross, −0.23 con el KPI, p 6·10⁻⁵) y retrasar
el inicio ayuda (ρ +0.15 con el KPI, p 0.009). Es exactamente la componente `ln(FeO_fin/FeO_ini)` del SDI: el monitor ya está en el
target. Feedback al equipo metalúrgico: la sobre-reducción del Fe no espera al agotamiento del Sn; el criterio de corte debe ser la
pérdida acumulada de FeO, no el %Sn.

## 18. Iteración 7: prescriptor v5 — objetivo por escalón en escala log anclado en el KPI refinado (2026-09-15, madrugada)

Familia `experimentos/v5/` (diseño `ITERACION_7_diseno.md`, librería `v5_lib.py`, módulo `modelo_predictivo_v5.py`, ficha
`candidate_win_model.md`, notebook `win_analytics_v1.ipynb`). Punto de partida: v4 (§15), targets ganadores de la iteración 5 (§16: SDI
y ln IRF) y KPI refinado de la iteración 6 (§17). Cinco experimentos: E7-01 componentes y anclaje de w (Sonnet), E7-02 Fusión (Sonnet),
E7-03 formas funcionales y E7-04 estabilidad (scripts de Sonnet, corridos por Fable tras el límite de sesión), E7-05/06 validación,
política cross-fitted y evidencia (Fable).

### 18.1 Lo que cambia respecto de v4

- **Targets en escala log** (`v5_ln_sn_dep` = ln(Sn disponible / Sn en escoria), `v5_ln_feo_ret` = ln(FeO[t]/FeO[t−1])): adimensionales,
  cancelan la masa del trazador, suman telescópicamente por batch y son las componentes del SDI. El objetivo v4 en kg (Sn − 0.6·FeO)
  no correlaciona con el KPI (ρ 0.035); las componentes sí (Σln_feo_ret ρ 0.29 DEV; Σln_sn_dep 0.35 en lockbox).
- **w y K anclados en el KPI refinado** (E7-01): OLS HC3 de la recuperación refinada sobre las dos componentes agregadas (DEV n 285,
  controles + tendencia): +2.12 pp por unidad de agotamiento (p 0.001) y +12.83 por unidad de retención de FeO (p < 0.001) → w = 6.04
  [3.8, 13.6]; K = 2.12 pp × 512 kg/pp ≈ 1 090 kg de Sn por unidad. El barrido de w sube en DEV hasta 12-14 y baja en lockbox; w = 6 es
  el anclaje causal. La variante ponderada por tiempo restante (retener FeO temprano vale más) gana en lockbox (0.32) y pierde en DEV
  (0.24): no se adopta (criterio DEV) pero queda registrada.
- **El carbón en Reducción se identifica sólo con el estado completo** (37: incluye acumulados de carbón y C/Sn acumulado): `Cx_sn`
  +0.18 sd/sd [0.01, 0.29]; con el estado curado (20) el IC cruza 0. Reparametrizar el carbón (dosis específica C/Sn disponible,
  carbón sobrante, log-dosis) no mejora R² ni identificación. Sobre la retención de FeO el carbón es ≈ 0 (selectivo) y el GN es la
  única palanca robusta (−0.021 sd/sd [−0.031, −0.005]: reductor no selectivo).
- **θ con signos de teoría** (`ModeloPLMSignos`, mínimos cuadrados acotados): un reductor no puede aumentar la retención de FeO ni una
  lanza más oxidante aumentar el agotamiento de Sn; los θ que el dato empuja contra su signo colapsan a 0. Es la única monotonía impuesta
  y sólo sobre el efecto lineal de las palancas; el estado sigue libre.
- **Cuadráticos** (E7-03, Robinson residualizado): ninguno mejora el OOF (agotamiento 0.760 → 0.759; retención 0.203 → 0.191); sólo el
  aire sobre el agotamiento sale cóncavo (vértice 117 Nm³/min, IC [88, 260]). Forma lineal por palanca; los óptimos interiores de J
  vienen del trade-off Sn/FeO (w), la ventana térmica y el soporte.
- **Soporte relativo**: el término absoluto `−w_S(1−s)` de v4 dominaba J (900 kg de diferencia entre candidatos frente a ~200 kg de
  física) y empujaba hacia la moda histórica; v5 penaliza sólo perder densidad respecto de la acción histórica.
- **Fusión** (E7-02): con el KPI refinado ningún control tiene efecto neto de batch: C/Sn baja dross (−0.008/sd, p 0.014) pero sube polvo
  (+0.006, p 0.04); T sube polvo (+0.008, p 0.05) y baja dross (−0.006); el heel de escoria (`ln IRF` al cierre de F0) domina (+1.7
  pp/sd, p < 10⁻¹⁰); Δln IRF durante la Fusión es nulo controlando el heel (p 0.8). Único término con signo teórico: conservar el Sn en
  la escoria (−0.45 pp por unidad de Σln_sn_dep, p 0.083 DEV; −0.59, p 0.014 total). Los β_C y β_T de v4 se retiran. Hallazgo de plan:
  más cal por t de carga → −0.87 pp/sd (p 0.038; −1.04 con heel, p 0.006).

### 18.2 Métricas y estabilidad

R² OOF / lockbox: agotamiento de Sn 0.766 / 0.802 (HGB 0.761 / 0.801), retención de FeO 0.164 / 0.187 (HGB 0.133 / 0.186; techo del
trazador ≈ 0.8), ΔT de Reducción 0.466 / 0.465, agotamiento en Fusión 0.187 / −0.39 (sólo se usan sus θ), ΔT de Fusión 0.481 / 0.066.
Walk-forward (bloques de 50 batches): 0.744 y 0.144. Reentrenamiento progresivo cada 10 batches sobre el lockbox: retención de FeO
0.122 → 0.286, SDI 0.141 → 0.216. θ por tercios: `Cx_sn` mismo signo en los 4 períodos (significativo en 2); GN sobre agotamiento
positivo en los 4; GN sobre retención de FeO negativo en tercios 2-3 y DEV completo, positivo en tercio 1 y lockbox (frágil).

### 18.3 Política y evidencia (E7-05/06; 362 batches recomendados por sistemas que no los vieron)

Recomendación media: Reducción carbón +3.4 kg/min con avance < 0.84 y −1.4 con 0.96-0.975, GN −0.7 a −1.7 Nm³/min, O₂ −0.2 a −1.5,
aire +1 a +2; Fusión carbón −3 a −6 kg/min y gas de lanza −0.2 a −0.5 dentro de la ventana térmica. Uplift físico (J) mediana 157 kg
Sn/batch (Reducción 85, Fusión 53).
- Cadena θ → agregado → KPI: la política cambia Σln_feo_ret +0.023 y Σln_sn_dep −0.03 por batch → **+0.20 pp de recuperación refinada
  [0.13, 0.30] en DEV y +0.28 [0.18, 0.43] en lockbox** (≈ +100-140 kg Sn/batch; > 97 % de batches positivos).
- Adherencia off-policy (OLS HC3 con controles, permutación, placebo): Fusión `dist_F` −0.70 pp/sd (p 0.025; permutación 0.004) en DEV,
  −0.61 sobre la media del batch actual y el siguiente (p 0.006), polvo +0.008/sd (p < 0.001 en DEV y total); por palanca, el carbón de
  Fusión −0.79 (p 0.024 DEV; media2 p 0.002 / 0.044 lockbox / 0.031 total). Placebo ≈ 0. Reducción: nula (cambios recomendados
  pequeños; sin potencia). Emparejamiento por estado: nulo dentro de cada régimen; +1.8 pp [1.1, 2.5] en TOTAL sobre la media del batch
  actual y el siguiente.

### 18.4 Límites y siguiente paso

Evidencia observacional; la de Reducción descansa en la cadena (target ↔ KPI, θ con IC) y no en la adherencia; la recomendación de
carbón en Fusión es de costo/conservación (θ = 0) con respaldo por el canal polvo pero efecto neto de batch n.s. Lockbox = fin de
campaña: producción con reentrenamiento progresivo. Siguiente paso: piloto A/B por batches con la política v5 (GN de Reducción y gas de
Fusión como primeras palancas), y cerrar la parsimonia del estado de Reducción sin perder la identificación del carbón.

### 18.5 Variante con la posición de lanza como palanca (E7-08, pedida por el usuario)

Se promovió `posicion_vertical_lanza_mm` a palanca con término lineal y cuadrático de signo libre (la convención de la medida no está
confirmada). Resultado: los términos lineales no son significativos y los cuadráticos sí en 3 de 5 modelos (agotamiento de Sn en
Reducción cóncavo con máximo ≈ 2 715 mm; retención de FeO convexa con mínimo ≈ 3 250 mm; ΔT de Fusión cóncava): la lanza tiene óptimo
interior, no dirección. La política resultante baja la lanza 120-180 mm en Reducción y la sube en F1-F2; el uplift estimado sube de
+0.20 a +0.36 pp por batch (cadena θ→agregado→KPI) y el físico de 182 a 252 kg, pero la evidencia off-policy no mejora (`dist_F`
−0.70 → −0.56 pp/sd; adherencia específica de la lanza n.s.). Se mantiene la base v5 como ganadora; la variante queda como candidata
para el piloto tras confirmar con planta el sentido de la medida (memo `experimentos/v5/e7_08_resultados.md`).

### 18.6 Diagnóstico del R² lockbox negativo en Fusión (E7-09)

El agotamiento de Sn en Fusión da R² −0.38 en lockbox pero conserva correlación real-predicho 0.37 (Spearman 0.37; DEV 0.43): el
signo negativo viene de un sesgo de nivel (+0.08, concentrado en F6 con +0.23) por el régimen de fin de campaña (horno ≈ 160 °C más
frío, extracción en Fusión 0.59 vs 0.62-0.65) y de una varianza del target 25 % menor. No es un modelo espurio, pero la palanca que
sostiene el término de conservación (GN +0.030 en DEV) cae a 0 ajustando sólo el lockbox y a +0.017 con reentrenamiento progresivo.
Consecuencia: Fusión es módulo de menor confianza; operar con reentrenamiento por campaña o dejar sólo ventana térmica y costos.
