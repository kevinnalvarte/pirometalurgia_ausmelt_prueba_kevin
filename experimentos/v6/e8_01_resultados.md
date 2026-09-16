# E8-01 — Variantes, calibración y robustez del estimador de masa v6

Script: `experimentos/v6/e8_01_masa_variantes.py` (etapas `grid|calib|sensib|ruido|bordes|anclaje|qc|all`).
Log completo: `experimentos/v6/e8_01_log.txt` (925 s totales; `e8_01_log_err.txt` sólo trae un warning de
`log(0)` esperado durante el bootstrap de ruido). Datos: `masa_v6.cargar_df_base()` (cache `df_v3.pkl`,
3982 filas x 362 batches), DEV=299 batches, lockbox=63 (`mp.split_dev_lockbox`).

No se modificó `masa_v6.py`. Se documentan dos observaciones de diseño al final (sin parche urgente).

## 1. Grilla fusión × reducción × (δ_R, w_obs_R, aC)

`e8_01_grilla.csv` — 66 variantes solicitadas, 65 firmas únicas evaluadas (una firma se repite porque
`aC` no afecta a fusión="CaO"+reducción="CaO"), más la fila de referencia v3 (`masa_escoria_est_kg`).
Cada fila trae la batería completa de `masa_v6.evaluar_masa` (física, escala vs balance, cierre de Sn,
suavidad en Fusión, R² OOF HGB GroupKFold(5) de `ln_feo_ret`/`ln_sn_dep`).

### 1.1 Resultado central: fusión física ("cierre") gana en las dos métricas que definieron la iteración

| fusión (aC=0) | rho_ratio_calcarga | ratio_bal_dev | ratio_bal_lb | F_sd_ln_masa_carga | R2_feo |
|---|---|---|---|---|---|
| **cierre** | **0.039** | **1.004** | **0.976** | 0.064 | 0.300 |
| G (trazador matriz) | 0.114 | 1.114 | 1.063 | **0.055** | 0.290 |
| CaO (trazador calibrado) | 0.136 | 1.090 | 1.015 | 0.077 | 0.274 |
| v3 (`masa_escoria_est_kg`) | **-0.350** | 1.078 | 1.119 | 0.082 | 0.148 |

`rho_ratio_calcarga` es exactamente el defecto que motivó la iteración 8 (v3 = -0.35). Con `fusion="cierre"`
(cierre físico de "resto" aplicado también en Fusión, sin dividir nunca por el trazador contemporáneo)
ese sesgo cae a 0.04 y `ratio_bal` (masa final del balance / masa final del estimador) queda prácticamente
en 1.00 en DEV y lockbox simultáneamente — ningún otro método logra eso en ambos splits a la vez. El
único costo es una suavidad en Fusión ligeramente peor (`F_sd_ln_masa_carga` 0.064 vs 0.055 de fusión="G"),
aceptable frente a la ganancia en insesgamiento.

### 1.2 Resultado central: mezclar el trazador en Reducción reintroduce el artefacto que se quería eliminar

Ordenando por R2_feo la tabla completa, el podio lo ocupan `reduccion="G"` puro (R2_feo=0.656, con aC
calibrado) y `reduccion="cierre+obs"` con w_obs_R alto (0.42-0.45 con w=0.30). Pero esas mismas filas son
las que peor puntúan en los diagnósticos físicos:

| reducción | frac_dM_pos_R | sd_dlnM_R | R2_feo | interpretación |
|---|---|---|---|---|
| cierre (δ=0.01, aC=0) | 0.035 | 0.0254 | 0.304 | referencia recomendada |
| cierre+obs, w=0.15 (δ=0.02, aC=0) | 0.036 | 0.0268 | 0.318 | leve mezcla, aceptable |
| cierre+obs, w=0.30 (δ=0.02, aC=cal) | 0.060 | 0.0302 | 0.407 | empieza a degradar la física |
| **G puro (aC=cal)** | **0.200** | **0.0348** | **0.656** | 20% de escalones sin carga "ganan masa": el mismo síntoma que delató a v3 |
| CaO puro (aC=0) | 0.292 | 0.0394 | 0.068-0.135 | peor en ambos frentes |

`reduccion="G"` calcula M_t de forma independiente en cada escalón dividiendo por el mismo ensayo (los
4 óxidos) que en parte comparte matriz analítica con %FeO del mismo pocillo de escoria; eso genera
correlación espuria entre M_t (denominador G) y el target `ln_feo_ret` (que usa %FeO), exactamente el
mecanismo de "reversión al ruido del ensayo" que la auditoría de CaO diagnosticó para v3. La evidencia:
`R2_feo_sinCaOprev` cae poco (0.656→0.635) —no es reversión a `%CaO_prev`— pero `frac_dM_pos_R`(20%) y
`sd_dlnM_R` (0.035, el peor de toda la grilla) muestran que la "M" en sí es fisicamente inconsistente.
**Conclusión: el R² más alto de la grilla no es el más confiable.** Se prioriza el método con mejor perfil
físico (`cierre` puro, `w_obs_R=0`) aunque su R2_feo (0.30) sea menor que el de las variantes con trazador.

### 1.3 δ_R (pérdida residual en Reducción): pequeño y positivo mejora la física sin mover mucho la escala

Con fusión="cierre", reducción="cierre", aC=0:

| δ_R | frac_dM_pos_R | sd_dlnM_R | ratio_bal_dev | ratio_bal_lb | R2_feo |
|---|---|---|---|---|---|
| 0.00 | 0.097 | 0.0254 | 1.004 | 0.976 | 0.300 |
| **0.01** | **0.035** | 0.0254 | 1.042 | 1.015 | 0.304 |
| 0.02 | 0.013 | 0.0254 | 1.079 | 1.056 | 0.300 |

δ_R=0.01 recorta a un tercio la fracción de escalones sin carga que "ganan masa" (9.7%→3.5%) desplazando
`ratio_bal` sólo 4 pp, muy por debajo de la dispersión del ratio (`sd_ln_ratio_bal`≈0.11, es decir ±11%).
δ_R=0.02 sobrecorrige la escala sin mejorar más el R². Se recomienda **δ_R=0.01**.

### 1.4 aC (coeficiente de ceniza de carbón): la calibración conjunta no es fiable — se recomienda aC=0

Ver el detalle en la sección de calibración: el coeficiente sale **negativo** (-0.56, IC95%
[-1.33, 0.20], p=0.15) — físicamente imposible (el carbón no puede restar matriz a la escoria) y
estadísticamente no distinguible de cero. Aparece con signo positivo o negativo dependiendo del split
(ver §2), consistente con colinealidad entre dosis de carbón y dosis de cal/carga, no con una señal real.
Aunque activar `con_carbon=True` sube R2_feo en casi todas las filas de la grilla (p.ej. cierre/cierre:
0.30→0.37), ese incremento es sospechoso de ser un artefacto de la misma familia que el de §1.2 (la
calibración con carbón se ajustó sobre los mismos 287 batches DEV que luego se usan para evaluar R²) y no
se sostiene con un coeficiente físicamente defendible. **Se recomienda aC=0** pese al R² más alto.

## 2. Calibración: estabilidad temporal y por fold

`e8_01_calibracion.csv` (full DEV, 3 tercios cronológicos de DEV, 5 folds aleatorios por batch).

| split | trazador | con_carbon | cal (pG/pCaO) | carga (gG/cCaO) |
|---|---|---|---|---|
| full DEV (n=287) | G | No | 0.703 (p=0.054) | 0.301 (p≈0) |
| tercio 1 (n=87) | G | No | 0.906 (p≈0) | 0.282 (p≈0) |
| tercio 2 (n=100) | G | No | **0.002 (p=0.999, ns)** | 0.369 (p=0.05) |
| tercio 3 (n=100) | G | No | 0.474 (p=0.60, ns) | 0.323 (p<0.001) |
| kfold(5) aleatorio | G | No | media 0.700, **sd 0.179** | media 0.302, sd 0.018 |
| kfold(5) aleatorio | G | Sí | media 1.036, sd 0.143 | media 0.340, sd 0.023 |
| kfold(5) aleatorio | CaO | No | media 0.254, sd 0.066 | media 0.080, sd 0.007 |
| kfold(5) aleatorio | CaO | Sí | media 0.370, sd 0.061 | media 0.093, sd 0.008 |

**Hallazgo importante:** el coeficiente de la **cal** (pG/pCaO) es muy inestable — cronológicamente
pasa de 0.91 a ~0 (no significativo) a 0.47 entre los tres tercios de la campaña, y su sd por fold
aleatorio (0.179, ~25% relativo) sextuplica la del coeficiente de **carga** (gG, sd 0.018, ~6% relativo).
El mismo patrón se repite en el trazador CaO. La interpretación más plausible es que la pureza/composición
efectiva de la cal (tolva 6) no es realmente constante en el tiempo (cambios de proveedor o de lote),
mientras que la contribución de la ganga de la carga es comparativamente estable. Esto no invalida usar
`pG=0.703` como constante poblacional (es el promedio DEV, y el §1.1 muestra que fusión="cierre" —que
sólo usa `pG` para anclar el primer escalón y no en cada paso— es poco sensible a este valor), pero es una
advertencia: **una recalibración periódica de pG/pCaO (p. ej. por trimestre o al detectar cambio de
proveedor de cal) es más defendible que confiar en una constante fija de campaña completa**, sobre todo si
el estimador se usara con fusión="G" (que sí depende de pG en cada escalón).

## 3. Sensibilidad ±30% en pG y gG

`e8_01_sensibilidad.csv` (fusión="G", reducción="cierre", 7 variantes). Confirma la predicción teórica:

| variante | R2_feo | R2_sn | sd_dlnM_R | frac_dM_pos_R | ratio_bal_dev |
|---|---|---|---|---|---|
| base | 0.290 | 0.695 | 0.02539 | 0.0975 | 1.114 |
| pG×0.7 | 0.278 | 0.702 | 0.02539 | 0.0975 | 1.180 |
| pG×1.3 | 0.289 | 0.691 | 0.02539 | 0.0975 | 1.055 |
| gG×0.7 | 0.284 | 0.708 | 0.02539 | 0.0975 | 1.473 |
| gG×1.3 | 0.277 | 0.692 | 0.02539 | 0.0975 | 0.896 |
| **ambos×0.7** | **0.290** | 0.708 | **0.02539** | **0.0975** | 1.591 |
| **ambos×1.3** | **0.290** | 0.697 | **0.02539** | **0.0975** | 0.857 |

Cuando se escalan **ambos** coeficientes por el mismo factor, `R2_feo`, `R2_sn`, `sd_dlnM_R`,
`frac_dM_pos_R` y `rho_ratio_calcarga` quedan **exactamente idénticos** al caso base (invarianza de escala
de los targets log, como predice la teoría); sólo `ratio_bal_*` (la comparación absoluta contra el balance)
se mueve, de 0.86 a 1.59. Esto confirma que la calibración contra el balance es indispensable para fijar
la escala absoluta, pero que el resto de la batería (y por tanto la elección de fusión/reducción/δ_R) es
robusta a errores moderados en pG/gG — no hace falta una calibración perfecta de esos coeficientes para
que el resto del diagnóstico sea válido.

## 4. Propagación de error de ensayo en Reducción

`e8_01_ruido.csv` (200 réplicas, %Sn ruido relativo sd 2%, %FeO sd 2.8%, %CaO sd 2.8%; fusión="G" fijo
para aislar el efecto del método de Reducción).

| orden R | señal sd(Δln M) | ruido sd(Δln M) cierre | **SNR cierre** | ruido sd(Δln M) CaO | **SNR CaO** | señal sd(ln FeO_ret) | ruido sd(ln FeO_ret) | **SNR FeO** |
|---|---|---|---|---|---|---|---|---|
| R0 | n/d (sin escalones sin carga) | n/d | n/d | 0.0279 | n/d | 0.0658 | 0.0510 | 1.29 |
| R1 | 0.0384 | 0.0130 | **2.96** | 0.0280 | 1.37 | 0.0728 | 0.0495 | 1.47 |
| R2 | 0.0154 | 0.0127 | 1.21 | 0.0280 | 0.55 | 0.0622 | 0.0492 | 1.26 |
| R3 | 0.0155 | 0.0128 | 1.21 | 0.0280 | 0.55 | 0.0529 | 0.0491 | 1.08 |

Dos hallazgos cuantitativos:
- **El cierre físico es ~2× más preciso que el trazador CaO** bajo el mismo nivel de ruido de ensayo:
  sd(Δln M) propagada ≈0.013 con cierre vs ≈0.028 con CaO, en cada orden de escalón. El SNR del cierre
  (1.2-3.0) supera siempre al del CaO (0.55-1.37) — confirma cuantitativamente el argumento cualitativo de
  la auditoría original.
- El SNR de `ln_feo_ret` (1.08-1.47) es moderado, no alto: coherente con el R2_feo OOF observado (~0.30).
  R0 no tiene escalones "sin carga" para medir la señal de Δln M (es la transición F→R, siempre trae
  carbón/carga) — limitación de la metodología, no del estimador.

## 5. Casos borde (sólo en el log, sin CSV propio)

- **F0** (primer escalón de Fusión, n=358): `m6_masa_kg/(cal+carga)` media 0.493, sd 0.049, rango
  [0.386, 0.716] — banda estrecha y sin degenerados. La misma razón con `masa_v3` tiene media 0.444 pero
  sd 0.107 y un mínimo de 2.1e-26 (división casi por cero): v6 es más robusto en el primer escalón porque
  `ley_min_G` evita denominadores ínfimos.
- **Ensayos faltantes**: Sn 3.04% (F=10,R=111), FeO 3.31% (F=17,R=115), CaO 3.29% (F=16,R=115) — la falta
  se concentra en Reducción. De las 133 filas con Sn o FeO faltante, 131 (98%) quedan con `m6_masa_kg`
  NaN (el cierre no puede calcular `resto_t` sin ambas leyes) y el escalón siguiente del mismo batch (28
  casos observados) se re-ancla con `obs_G` porque `M[i-1]` deja de ser finito — comportamiento esperado
  del código (branch `if not np.isfinite(Mp): M[i]=obs_G[i]`), no interpola el hueco.
- **Batches con cal≈0** (12 de 362, feed_CaO_kgh sumado <100 kg): masa mediana por escalón 37.7 t vs 47.5 t
  global — son batches distintos (más chicos), no un fallo del estimador; para ellos la entrada depende
  casi enteramente de `gG` (el coeficiente estable, §2), así que la inestabilidad de `pG` no los afecta.
- **resto_frac<0.4**: 0 de 3982 escalones. El piso `resto_min=0.30` nunca se activa en este dataset
  (mínimo empírico de `m6_resto_frac` = 0.468): es un seguro razonable para outliers futuros, no un ajuste
  fino necesario hoy.
- **m6_masa_kg NaN/0**: 134 filas (F=18, R=116, 0 en otras fases) — coincide con el conteo de ensayos
  faltantes; los modelos OOF ya excluyen estas filas vía `dropna`.

## 6. Diagnóstico de `m6_k_anclaje` (M_balance/M_fin por batch, LEAKAGE offline)

`e8_01_anclaje.csv` (362 batches, PARAMS_DEFAULT). k_anclaje: media 1.100, sd 0.133, mediana 1.103,
IQR [1.012, 1.190] — sesgo de ~10% consistente con `ratio_bal_dev` de la fila "G" de la grilla (PARAMS_DEFAULT
usa fusión="G").

Spearman univariado: `frac_carga_secundaria` 0.24, `feed_dross_Fe_kg` 0.26, `T_media_C` 0.24,
`espesor_ladrillo_mm` 0.27, `orden_cronologico` -0.27, `cal_sobre_carga` 0.11, `ley_sn_carga_batch_pct` -0.22
(todos p<0.05, n=362). Pero `espesor_ladrillo_mm` y `orden_cronologico` correlacionan entre sí **-1.00**
(por diseño: el espesor decrece monótonamente con la campaña) y `T_media_C`, `frac_carga_secundaria` y
`feed_dross_Fe_kg` correlacionan 0.5-0.85 con ese mismo eje temporal. **No son 6 causas independientes: es
mayormente un único eje "tiempo/campaña"** más dos señales propias (`cal_sobre_carga`, `ley_sn_carga_batch_pct`).
Una regresión OLS-HC3 multivariante con las 7 variables da R²=0.168 (17% de la varianza de k_anclaje
explicada); al controlar por las colineales, sólo sobreviven significativos `ley_sn_carga_batch_pct`
(coef -0.026, p<0.001) y, marginalmente, `cal_sobre_carga` (coef -0.98, p=0.049) — este último es
exactamente el mismo tipo de sesgo que motivó la iteración 8, ahora mucho más débil (ρ=0.11 vs -0.35 en v3)
pero no completamente nulo.

**Veredicto**: sí hay estructura (no es ruido puro), pero está dominada por una deriva temporal de campaña
que es la misma deriva que ya se ve en la inestabilidad de `pG` por tercios (§2) — probablemente originada
en cambios de calidad de cal/ladrillo/mineral secundario a lo largo de la campaña, no en un error funcional
del estimador. **No se recomienda incorporar `k_anclaje` como corrección poblacional estática**: cualquier
corrección fija estimada sobre el DEV histórico quedaría desactualizada tan pronto cambie la campaña (igual
que le pasaría a un pG fijo). La recomendación es la misma que en §2: **recalibrar pG/gG periódicamente**
(o cuando se detecte cambio de proveedor de cal / recambio de ladrillo) en vez de introducir una corrección
por batch en línea que reintroduciría dependencia de variables no disponibles causalmente en el momento de
prescribir.

## 7. Bandera QC de residuo de masa en Reducción

`e8_01_qc.csv` (por escalón, n=1332) y `e8_01_qc_batch.csv` (agregado por batch). Residuo =
ln(M_cierre) − ln(M_G) (ambos con fusión="G" fijo, para aislar el efecto del método de Reducción):
media -0.103, sd 0.055, prácticamente constante por orden de escalón (R0 -0.092, R1 -0.106, R2 -0.105,
R3 -0.108) — es un **sesgo estructural entre métodos** (el cierre acumula ~10% menos masa que la
sustitución directa por trazador G), no una anomalía de calidad de dato. La bandera QC se definió como
`|z-score dentro del orden de escalón| > 2` (demeanado por orden, para no confundir el sesgo estructural
con un outlier) y marca 3.5% de los escalones — razonable para un umbral de 2σ.

A nivel de batch, el residuo medio-absoluto **no correlaciona** con el polvo de fundición (Spearman
ρ=0.080, p=0.13) ni con el KPI `recuperacion_refinada_pct` (ρ=0.036, p=0.49); tampoco la fracción de
escalones marcados por batch (ρ=0.006, p=0.91). **Conclusión: esta bandera es útil como control interno de
consistencia entre dos formas de estimar M_t (detecta escalones donde el cierre físico y el trazador
divergen más de lo típico dentro de su propio orden), pero no es un proxy del desempeño metalúrgico del
batch** — no debe usarse como corrección del KPI ni como filtro de calidad de batches.

## 8. Observaciones de diseño en `masa_v6.py` (sin parche urgente; no se modificó el archivo)

1. **`calibrar_entradas(trazador="CaO", con_carbon=True)` calcula un coeficiente de carbón que
   `estimar_masa` nunca usa.** `PARAMS_DEFAULT` no tiene un análogo de `aC` para la rama CaO (`obs_CaO`
   sólo usa `pCaO`, `cCaO`). No es un bug (no se documenta lo contrario), pero es una asimetría con la
   rama G que puede confundir a quien calibre con `con_carbon=True` esperando que se aplique. Parche
   propuesto si se quisiera cerrar la simetría: agregar `caC` a `PARAMS_DEFAULT` y cambiar
   `obs_CaO = (p["pCaO"]*cum_cal + p["cCaO"]*cum_carga + p["caC"]*cum_carbon) / cao` en `estimar_masa`
   (requiere exponer `cum_carbon` igual que `cum_cal`/`cum_carga`). Dado que en §1.4 el coeficiente de
   carbón no resultó confiable ni siquiera en la rama G, no se recomienda implementar este parche por ahora.
2. **`frac_dM_pos_R`/`sd_dlnM_R` (definidos sobre escalones "sin carga") no excluyen el escalón
   inmediatamente posterior a un re-anclaje por ensayo faltante** (§5): ese escalón puede mostrar un salto
   de `M` grande por el cambio de método (cadena→`obs_G`) aunque no traiga carga, inflando ligeramente
   estos diagnósticos. Con sólo 28 casos de 3982 filas el efecto es pequeño, pero si se quisiera una
   medición más limpia, `evaluar_masa` podría recibir una máscara adicional de "recién re-anclado" para
   excluirlos (no implementado aquí porque requeriría exponer el estado interno del loop de
   `estimar_masa`, hoy no expuesto).

## 9. Veredicto y parámetros recomendados

```python
PARAMS_RECOMENDADOS = dict(
    fusion="cierre",       # cierre fisico tambien en Fusion: rho_ratio_calcarga 0.04 (vs 0.11-0.35),
                            # ratio_bal ~1.00 en DEV y lockbox simultaneamente
    reduccion="cierre",    # cierre fisico puro; NO mezclar el trazador (w_obs_R=0): mezclarlo reintroduce
                            # el artefacto de reversion al ruido de ensayo que motivo la iteracion 8
    pG=0.703, gG=0.301, aC=0.0,     # aC calibrado sale negativo y no significativo (p=0.15): no usar
    pCaO=0.256, cCaO=0.079,          # sin usar en este metodo; se dejan para diagnostico/QC (obs_CaO)
    delta_R=0.01,           # reduce frac_dM_pos_R de 9.7% a 3.5% con costo minimo en ratio_bal (+4pp)
    w_obs_R=0.0,
    resto_min=0.30,         # nunca se activa en DEV/lockbox (min empirico de resto_frac = 0.468); es
                            # un seguro para outliers futuros, no requiere ajuste
    ley_min_G=5.0, ley_min_cao=2.0,
)
```

Métricas de esta configuración (fila `signature=('cierre','cierre',0.01,0.0,False)` en `e8_01_grilla.csv`):
`frac_dM_pos_R`=0.035, `sd_dlnM_R`=0.0254, `rho_ratio_calcarga`=0.031, `sd_ln_ratio_bal`=0.113,
`ratio_bal_dev`=1.042, `ratio_bal_lb`=1.015, `R2_feo`=0.304, `R2_feo_sinCaOprev`=0.280 (brecha 0.024, chica),
`R2_sn`=0.693 (no degradado vs el resto de la grilla, que va de 0.68 a 0.72), `cierre_sn_dev`=0.989,
`cierre_sn_lb`=0.985 (ambos ≈1), `F_sd_ln_masa_carga`=0.064.

Difiere de `masa_v6.PARAMS_DEFAULT` sólo en `fusion` ("G"→"cierre") y `delta_R` (0.0→0.01); todo lo demás
coincide con la calibración ya presente en el módulo.

### Límites y advertencias para E8-02/E8-03/E8-04

- El R2_feo de ~0.30 es el techo defendible sin reintroducir el artefacto de trazador; **cualquier variante
  de esta grilla que reporte R2_feo sustancialmente mayor (>0.40) casi con seguridad está mezclando
  información contemporánea del mismo ensayo de escoria y debe tratarse con sospecha**, no como una mejora.
- `pG`/`pCaO` (coeficiente de la cal) son inestables entre tercios cronológicos de la campaña (§2); el valor
  poblacional de DEV completo es razonable como constante de arranque pero **debería revisarse
  periódicamente**, no asumirse fijo para siempre. Fusión="cierre" es la elección menos sensible a este
  riesgo porque `pG` sólo ancla el primer escalón del batch.
- La bandera QC de residuo (§7) sirve para auditoría de consistencia interna del estimador, no como
  predictor del KPI ni de calidad de batch.
- `m6_k_anclaje` (§6) no debe usarse en línea; es diagnóstico offline por construcción (usa salidas del
  batch) y su estructura observada es principalmente deriva temporal de campaña, mejor atendida con
  recalibración periódica de `pG`/`gG` que con una corrección por batch.
- Ensayos faltantes (~3%) dejan huecos de `m6_masa_kg`=NaN que no se interpolan; el uso de estos targets en
  E8-03/E8-04 debe seguir excluyéndolos vía `dropna` como ya hace `evaluar_masa`.
