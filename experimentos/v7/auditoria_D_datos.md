# Auditoría D — integridad de datos, anti-fuga y cadena de validación del enfoque v7 (2026-09-16)

Auditor D (integridad de datos de planta, features anti-fuga, validación de procesos por lotes). Alcance: `candidate_win_model_v7.md`,
hallazgos §1-2, §14.1, §19-20, `diccionario_datos.json`, `config_limpieza.yaml`, `dataset_lingo_smelter.py`, `feature_engineering.py`,
`experimentos/v6/masa_v6.py`, `modelo_predictivo_v6.py`, `modelo_predictivo_v7.py`, `experimentos/v7/v7_lib.py` y el Excel fuente
(hojas Datos, Diccionario, Tag Pi system). No se modificó nada del repo; las comprobaciones son scripts del scratchpad ejecutados con
`.venv/Scripts/python.exe` sobre `experimentos/cache/df_v6.pkl` (3982 × 281) y sobre el Excel. Convención de veredictos:
**RESCATAR** = el hecho sostiene la afirmación del modelo; **EN DUDA** = no invalida pero exige corrección o aviso; **DESCARTAR** = falso.

## Resumen de veredictos

| # | Punto | Veredicto | Corrección mínima |
|---|---|---|---|
| 1a | 16 features de `ESTADO_R_V7` + palancas + derivadas usan sólo cierre de t−1 y acción/plan de t | RESCATAR | — |
| 1b | Filtro anti-fuga `es_segura_v6_o_v5` | EN DUDA | `m6_G_pct` (contemporáneo) pasa el filtro; ninguna feature v7 lo usa |
| 1c | `espesor_ladrillo_norm_mm` como "campaña" | EN DUDA | es un reloj (ρ −0.9996 con la fecha); lockbox fuera del rango DEV; quitarlo no cuesta R² |
| 1d | T horno / tiro / lanza / termocuplas "al cierre" | EN DUDA (interpretación) | son tags PI agregados por escalón; el Excel no dice media o cierre; no es fuga porque se usa `_prev` |
| 1e | `ley_sn_carga_batch_pct` ex post | RESCATAR (pendiente planta) | no correlaciona con KPI ni con el cierre; confirmar que el ensayo precede al batch |
| 2a | Calibración pG/gG en DEV → targets | RESCATAR | targets de Reducción invariantes a pG/gG (corr 0.99999 por fold) |
| 2b | `delta_R` 0.01 | RESCATAR (inocuo) | desplazamiento constante exacto de −0.01/escalón; no afecta R², θ ni argmax |
| 2c | "3 % de escalones sin masa" | DESCARTAR el número | en Reducción falta el 8 % (R3: 29 % sin ensayo); en Fusión 0.6 % |
| 2d | F1 con gradiente imputado 0 | RESCATAR con aviso | F1 no es predecible en ΔT (R² 0.04) y diluye el R² de Fusión ΔT 0.48 → 0.38 |
| 3 | Estructura, NaN, rangos, duplicados, cal, GN | RESCATAR | 12 batches con cal < 1 t (abril, DEV); 5 escalones sin GN; sin sesgo de selección por KPI |
| 4a | Splits DEV/lockbox y GroupKFold | RESCATAR | sin solape temporal ni de batches; pG/gG sólo DEV; caché ≡ recomputo |
| 4b | OOF aleatorio vs cronológico | EN DUDA (FeO) | FeO 0.333 → 0.296 con folds cronológicos; Sn 0.767 → 0.773 |
| 5 | Reproducibilidad | RESCATAR con aviso | tabla reproducida a 1e-16; `v7_lib.CONFIG` es copia v6 tras `activar_v7`; sin versiones fijadas |
| 6 | Cobertura de la política 3577/3982 | RESCATAR | 362 primeros escalones + 43 con NaN; sin exclusión sistemática por KPI |

---

## 1. Fuga de información

**(a) Hecho.** Las 16 features de `ESTADO_R_V7` son: 4 leyes `_prev` (`agregar_estado_previo`: `groupby(Batch).shift(1)`),
`ratio_sn_feo_prev` e `interaccion_Sn_x_FeO_prev` (funciones de leyes `_prev`), `grad_temperatura_horno_celsius_prev` (= `shift(1)` del
gradiente, es decir (T[t−1]−T[t−2])/Δt[t−1]), `termocupla_media_celsius_prev` y `posicion_vertical_lanza_mm_prev` (`shift(1)`),
`cum_feed_Carbon_kg_prev`, `cum_aire_nm3_prev` (= cumsum − valor de t, excluye el escalón t), `relacion_C_cum_Sn_cum_prev` (cociente de dos
cum_prev), `m6_sn_inv_kg_prev`, `m6_feo_inv_kg_prev`, `m6_resto_frac_prev` (`shift(1)` enmascarado en el primer escalón) y
`espesor_ladrillo_norm_mm` (CONTEXT por batch). Palancas: `Cx_sn_v6 = tasa_C[t] · Sn_inv_prev/1000`, `Cx_av_v6 = tasa_C[t] · avance_prev`
(acción t × estado t−1), `exceso_o2_combustion_pct = 100·(O2 + 0.21·aire − 2·GN)/(2·GN)` con volúmenes del propio escalón,
`relacion_C_Sn_carga = C[t]/feed_Sn[t]` (acción × ensayo del batch), tasas = volumen/`delta_tiempo`. Las columnas de salida de batch
(`sn_en_metal/dross/polvo`, `rendimiento_*`, `recuperacion_*`) no aparecen en ningún ESTADO, PALANCAS ni FEATURES_SOPORTE (verificado
sobre los diccionarios en memoria tras `activar_v7`: lista vacía); sólo entran en el KPI (anclaje/evidencia).

**(b) Riesgos examinados y (c) comprobaciones.**

| Riesgo | Comprobación | Resultado |
|---|---|---|
| ¿`_prev` de T horno, tiro, lanza, termocuplas son medias o cierre de t−1? | Excel hoja Diccionario y Tag Pi system | Son tags del historiador PI agregados por escalón; el Diccionario no indica media ni cierre. `diccionario_datos.json` afirma "al cierre" sin respaldo. En cualquier caso el valor de t−1 (media o cierre) está disponible al inicio de t: **no hay fuga**, sólo ambigüedad de interpretación física |
| `espesor_ladrillo_norm_mm` como estado de campaña | Spearman con fecha del primer escalón; rangos DEV/lockbox; R² sin la feature | ρ = −0.9996 (95 valores únicos; 63 batches cambian de valor a mitad de batch). Rango lockbox [218.1, 230.5] disjunto de DEV [230.5, 300]: el HGB extrapola en lockbox. Sin espesor: Sn 0.767/0.800 (OOF/lockbox, vs 0.767/0.795), FeO 0.325/0.469 (vs 0.333/0.483). Con sólo {espesor, %Sn_prev} (k=2): Sn 0.751/0.801, FeO 0.276/0.447 |
| `ley_sn_carga_batch_pct` (ensayo por batch, ¿retro-calculado?) | ley por corriente = tmf/tmh (std intra-batch 0 en las 5 tolvas); correlaciones | Leyes distintas por corriente (t1≠t4 en 100 % de batches; t5≠t7 en 98 %, mediana 0.75 pp); ρ(ley conc., KPI) = 0.008 (p 0.89), ρ(ley conc., cierre de Sn) = 0.048; cierre de Sn del balance −1.9 % ± 6.6 %: si la ley se hubiera reconciliado con las salidas el cierre sería ≈ 0. Usa la mezcla de TODO el batch (incluye escalones futuros), pero la ley por corriente es constante y en Reducción `feed_Sn` ≈ 0 (mediana 476 kg/batch vs inventario de miles de kg); no está en `ESTADO_R_V7` (sí en el estado de Fusión, 25) |
| `grad_temperatura_horno_celsius_prev` en F1 | `_preparar_base_v7` imputa 0 sólo en Fusión orden 1 | 362 filas F1 imputadas; F2-F6 tiene mediana 0.02, IQR [−0.19, 0.26] → imputación al centro. Ver §2d |
| Filtro `es_segura_v6_o_v5` deja pasar LEAKAGE/TARGET | aplicado a las 281 columnas de la caché | Pasan 25 columnas sin rol seguro en fe/m6: `Cx_*_v4` (ok), 16 `b6_*` (declaradas seguras en `bal_features.SEGURAS_V6`), 7 `v5_*` derivadas (ok) y **`m6_G_pct`** (contemporáneo). Causa: `masa_v6.CLASIFICACION_M6["m6_G_pct"] = "STATE (contemporaneo: LEAKAGE ...)"` se copia a `fe.CLASIFICACION_FEATURES` y `fe.es_feature_segura_para_prescripcion` sólo mira `startswith("LEAKAGE")` → True; `masa_v6.es_segura_v6` sí lo rechaza, pero la unión con la rama v5/fe lo rescata. Ninguna feature de ESTADO/PALANCAS/SOPORTE v7 es insegura (0/0 con `es_segura_v6` y con `fe` puro) |
| Salidas de batch (metal/dross/polvo) en el modelo | búsqueda en todos los diccionarios de features | Ausentes. Constantes dentro del batch (0 batches con >1 valor) |
| Tasas con `delta_tiempo` real (conocido al cierre) | plan vs real por fase/orden | R0-R2: real = plan en el 100 % (std 0). R3: 85 % difiere (5-125 min, plan 13); F6: 93 % difiere (34-188 min). La palanca es un caudal (set point), correcto usar el real; pero corr(tasa_GN real, tasa_GN con plan) = 0.40 en F y 0.29 en R3, y J cuesta `tasa × duracion_plan`: en F6/R3 el costo está mal escalado hasta 2×. ρ(Δt real, ln_sn_dep) = 0.25 (p 2e-6) en F6 y 0.05 (n.s.) en R3: la duración endógena de F6 es una variable omitida del target, no una fuga |

**(d) Veredictos.** 1a RESCATAR. 1b EN DUDA → corregir el rol de `m6_G_pct` a `"LEAKAGE (…)"` o hacer que `fe.es_feature_segura_para_prescripcion`
use `"LEAKAGE" in rol`; añadir un test que recorra las 281 columnas. 1c EN DUDA → `espesor` es un índice de tiempo: reportar el OOF cronológico
(§4b) y, dado que no cuesta R², retirarlo del estado o sustituirlo por una medida real del refractario. 1d EN DUDA → cambiar "al cierre" por
"agregado PI por escalón (media o último valor: confirmar con planta)" en `diccionario_datos.json`. 1e RESCATAR condicionado a confirmar con
planta que la ley por corriente se conoce antes de cargar.

## 2. Targets y masa v6

**(a) Hecho.** `m6_ln_sn_dep[t] = ln((Sn_inv[t−1] + feed_Sn[t]) / Sn_inv[t])`, `m6_ln_feo_ret[t] = ln(FeO_inv[t]/FeO_inv[t−1])`, con
`M_t·r_t = M_{t−1}·r_{t−1}·e^{−δ_R} + pG·cal_t + gG·carga_t`, `r = 1 − 1.270·s − f`. pG/gG por OLS sin intercepto de `M_balance·G_fin`
sobre Σcal, Σcarga, sólo batches DEV con Σcal > 1000 kg (n = 287 de 299). Validez: Sn_inv > 20 kg, FeO_inv > 100 kg, no primer escalón.

**(b/c) Riesgos y comprobaciones.**

| Riesgo | Comprobación | Resultado |
|---|---|---|
| pG/gG calibrados con metal/dross/polvo de DEV → fuga hacia el OOF (el fold de test participa en la calibración) | recalibrar con DEV−fold_k (mismos 5 GroupKFold), recomputar targets y comparar con los del repo en el fold k | pG por fold 0.51 / 0.60 / 0.69 / 1.16 / 0.58 (IC DEV [−0.01, 1.42]: no identificado), gG 0.26-0.32; corr targets Reducción recalibrados vs repo **0.99999** (Sn y FeO), MAD 1e-6 en todos los folds. Motivo: en Reducción `ent_t ≈ 0` y `M_t/M_{t−1} = r_{t−1}/r_t·e^{−δ}` no depende de pG/gG |
| Fuga hacia el lockbox | recomputar `calibrar_entradas` con DEV, TODO y sólo lockbox | DEV 0.7032/0.3013 = `PARAMS_DEFAULT` (sólo DEV, verificado). TODO 1.41/0.23, lockbox 1.70/0.19 (pG deriva con la campaña: el "resto efectivo" por kg de carga ≈ 0.37 en ambos). Aplicar pG/gG del lockbox a todo: corr con el repo 0.995 (Sn, incluye Fusión) / 0.9996 (FeO) |
| `delta_R` = 0.01 | recomputar con δ = 0 y 0.02 | corr 1.000 con el repo; media de `ln_feo_ret` −0.0793 / −0.0893 / −0.0992: **un desplazamiento aditivo exacto de −δ por escalón**. No cambia R², θ, el argmax de la política ni la pendiente del anclaje (se absorbe en el intercepto); sólo el nivel físico de M |
| Umbrales 20/100 kg y `resto_min` 0.30 | conteo | 0 filas de Reducción bajo los umbrales; `resto_t` en R: mín 0.61, media 0.785 ± 0.029; nunca toca 0.30 |
| "3 % de escalones sin masa" (hallazgos §19.6) | NaN de `m6_masa_kg`, `m6_valido` por fase/orden | Fusión 0.6 %; **Reducción 8.0 %**; `m6_valido` R0-R2 0.98-0.99, **R3 0.71**: 105 batches (29 %) sin ningún ensayo en R3 (Sn, Fe, SiO2, CaO NaN a la vez; DEV 31.8 % vs lockbox 15.9 %). No es el umbral: es muestra no tomada. KPI mediana con/sin ensayo R3 68.7 vs 67.8 (MWU p 0.31): sin sesgo de selección. Consecuencia colateral: en esos 105 batches el KPI refinado toma `ley_fin` de R2 (antes del último escalón) y Σ_R usa 3 escalones |
| `feed_Sn_kgf` del escalón en `sn_disp` (tmh × ley del batch) | R con feed > 0 | 1446 escalones de R con feed > 0, mediana 476 kg/batch (dross/pellets residuales); es ACTION × CONTEXT |
| Leyes no normalizadas | Σ(%Sn+%Fe+%SiO2+%CaO+%Al2O3+%MgO) | media 97.1 ± 4.0, > 100 % en 888 filas (23 %; máx 112.7): `r_t` hereda el sesgo de nivel, que se cancela en los cocientes; no afecta a los targets log, sí al nivel de M |
| F1 imputado | R² OOF HGB (mismo estado/palancas) con y sin F1, DEV | sn_dep: sin F1 0.146 → con F1 0.189 (F1 solo 0.143; F2-F6 mejora a 0.169). ΔT: sin F1 0.478 → con F1 0.382 (**F1 solo 0.041**; F2-F6 0.469). F1 = 16.7 % de las filas de Fusión. Nota: la fila "PLM v5 ref" de la tabla (n 1689) también corre sobre la base con F1 imputado, que v5 nunca modeló |

**(d) Veredictos.** 2a RESCATAR (la única sensibilidad a pG/gG está en Fusión, corr 0.995, y la calibración es DEV-only). 2b RESCATAR: documentar
que δ_R es inocuo para el modelo y no contrastable con estos datos. 2c DESCARTAR la cifra "3 %": escribir "8 % de los escalones de Reducción
(29 % de R3) sin ensayo"; como sensibilidad, re-anclar K/w sólo con batches con ensayo R3 o imputar R3 = R2. 2d RESCATAR con aviso: para ΔT de
Fusión, añadir indicador `es_F1` o ajustar sin F1 (0.48 vs 0.38); para agotamiento la imputación ayuda.

## 3. Datos

| Comprobación | Resultado |
|---|---|
| Estructura | 3982 filas = 362 batches × (7 F + 4 R) exactamente; 0 duplicados (Batch, fecha_inicio) y 0 filas duplicadas; `fecha_final` = `fecha_inicio` siguiente en 3620/3620 pares |
| Excel → caché | Excel hoja Datos 3982 × 108, 362 batches; %Sn, T horno, feed_Sn, cal (tolva 6), posición de lanza: máx |diff| = 0.0 frente a la caché; %Sn NaN 120 (Excel) → 121 (caché) = AP0263 escalón 4 (241.5 %) puesto a NaN por `limpiar_dataset`; `df_v3` → `df_v6` recomputado con `agregar_masa_v6(PARAMS_DEFAULT)`: máx |diff| 0.0 en masa, targets y `Cx_sn_v6`, mismos NaN |
| Duraciones de Reducción | plan 30/25/20/13 exactos (R3 máx 15); real = plan exacto en R0-R2 (std 0); R3 real 5-125 (mediana 11) |
| NaN en `ESTADO_R_V7` (base sin primer escalón) | ≤ 0.9 % por columna; fila completa NaN por orden R0-R3: 0.6 / 0.8 / 1.1 / 1.4 %. Estado de Fusión (24 sin grad): 0.6-2.2 % (F6 2.2 %). Palancas: GN 0.1 %, exceso O2 0.2 %; `relacion_C_Sn_carga` NaN en 72.6 % de R (feed_Sn ≤ 20 kg) — sólo se usa en Fusión (0.1 %) |
| Leyes por fase/orden | NaN 0-2.2 % salvo **R3 29 %** (todas las leyes a la vez); ley_Sn < 0.5 % en 4 filas (3 R); %Fe = 0 en 0 filas; %Sn máx 29.4 tras limpieza |
| Rangos físicos | sin negativos en 20 columnas; T horno 881-1333 °C (0 fuera de [800, 1400]); tiro 0-99 % (9 filas < 30 %); lanza 13.6-8330 mm (2 filas < 100, 5 > 6000: convención sin confirmar, hallazgos §11.6/§12); exceso O2 −91..+135 %, 0 inf; termocuplas 255-417 °C |
| Cal | 0 batches con cal = 0; **12 batches con Σcal < 1 t** (AP0009-AP0020, 24-28 abril, todos DEV): excluidos de la calibración pG/gG por `cal_min`; masa v6 se calcula (3 % NaN) y KPI mediana 69.95 vs 68.37 |
| GN = 0 / NaN | 5 escalones (AP0007 R2-R3, AP0008 F0, AP0060 R2 = 0, AP0168 F3) → `exceso_o2` NaN → fila excluida del modelo (`division_segura`); 1 escalón con GN < 5 Nm³ |
| Carbón = 0 | 3 escalones en R, 3 en F |
| Selección por `dropna` | R sn_dep/feo_ret: 1325/1448 filas (91.5 %; R3 71 %; faltantes: target 120, inventario FeO 13, ratios 12); R ΔT 99.0 %; F sn_dep 98.5 %, F ΔT 99.2 %. ρ(fracción de filas retenidas por batch, KPI) = 0.04 (p 0.43) en R y −0.04/−0.05 en F: **sin sesgo de selección por KPI** |

Veredicto 3: RESCATAR. Correcciones: registrar en `config_limpieza.yaml`/diccionario la ausencia de ensayo en R3 (29 %) y los 12 batches de
abril con cal < 1 t como cohorte de arranque; considerar rango físico para `posicion_vertical_lanza_mm` (13.6 mm y 8330 mm) y `tiro` = 0.

## 4. Splits y caché

| Comprobación | Resultado |
|---|---|
| DEV/lockbox | lockbox = 63 = round(362 × 0.175) últimos batches por fecha del primer escalón; máx fecha DEV 2026-07-29 11:24 < mín lockbox 2026-07-29 18:58: sin solape |
| GroupKFold(5) de validación | 0 batches en más de un fold (298 batches DEV en el modelo de R); política `_folds_dev(seed 0)`: 60/60/60/60/59, 0 repetidos, cubren DEV |
| Caché `df_v6.pkl` | contiene `m6_k_anclaje` y `m6_masa_kg_anclada` (LEAKAGE, calculadas también para el lockbox: 693 filas) pero ambas rechazadas por todos los filtros; pG/gG = calibración DEV (recomputada: 0.7032/0.3013); la caché es idéntica al recomputo desde `df_v3` |
| KPI refinado DEV vs lockbox | misma fórmula; única asimetría: `ley_fin` procede de R2 en el 32 % de DEV vs 16 % del lockbox (§2c) |
| Anclajes K, w | DEV n = 297 (E9-00b); controles incluyen `espesor` e `idx_cronologico` |
| OOF aleatorio vs cronológico (5 bloques contiguos de batches, mismo PLM, mismas filas 1083) | Sn: 0.767 → **0.773**; FeO: 0.333 → **0.296** (−0.037); sin espesor: 0.775 / 0.301; k=12 sin espesor ni cum: 0.767 / 0.286. θ idénticos (se ajustan sobre todo DEV) |

Veredicto 4a RESCATAR. 4b EN DUDA para FeO: el GroupKFold aleatorio interpola en el tiempo (autocorrelación de campaña) y sobreestima ~0.04 el
R² de retención de FeO; coherente con el walk-forward 0.267 de E9-06. Corrección: reportar en la ficha el OOF cronológico junto al aleatorio
y usar el cronológico como criterio para FeO.

## 5. Reproducibilidad

| Comprobación | Resultado |
|---|---|
| `tabla_validacion_v7` recomputada (Reducción sn_dep y feo_ret, PLM) | 0.7666 / 0.2469 / 0.7952 / 0.2449 y 0.3329 / 0.0466 / 0.4829 / 0.0422, n 1083 / 242: máx |diff| vs `e9_07_tabla_validacion_v7.csv` = 8e-17; dos corridas idénticas (semillas: SEED 42 en HGB y bootstrap, `_folds_dev` 0, optimizador 0; GroupKFold determinista) |
| Orden de importación | `import v7_lib` solo → estado R 38, w 6.09 (v6 puro); tras `import modelo_predictivo_v7` → 16, w 7.86; `desactivar_v7()` restaura 38 / 6.09. **`v7_lib.CONFIG` es una copia hecha al importar y sigue en w 6.09 después de `activar_v7`, mientras `v7_lib.ESTADO` es el mismo objeto y pasa a 16**: un script que importe ambos y use `L.CONFIG` mezcla anclajes v6 con estado v7. Hoy no ocurre (E9-00..06 no importan mp7 y son v6 por diseño; E9-07/08 usan sólo mp7), pero el propio auditor cayó en la trampa inversa (script sin mp7 → sin F1) |
| Cadena Excel → caché → tablas | Excel ≡ caché en columnas base (diff 0); `df_v3` → `df_v6` exacto; `df_v6` → tabla exacto. No se re-ejecutó la construcción de `df_v3.pkl` desde el Excel (pipeline v3, fuera de alcance) |
| Versiones | Python 3.11.9, scikit-learn 1.9.1, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1; `requirements.txt` sólo con cotas inferiores (`>=`), sin lock |

Veredicto 5: RESCATAR con dos correcciones: (i) en `v7_lib` exponer `CONFIG` como referencia viva (`mp6.CONFIG_V6`) o documentar que tras
`activar_v7` hay que leer `mp6.CONFIG_V6`; (ii) fijar versiones (lock) y guardar un hash de `df_v6.pkl` y del Excel junto a las tablas.

## 6. Cobertura de la política cross-fitted

| Comprobación | Resultado |
|---|---|
| Filas | 3577 = 3620 escalones modelables (sin primer escalón) − 43; los 405 de 3982 excluidos son 362 primeros escalones (sin estado `_prev`, por diseño) + 43 |
| Motivos de los 43 | 16 `ley_sn_escoria_pct_prev` NaN, 12 `m6_sn_inv_kg_prev` NaN, 14 acción primitiva NaN (tasas de dross/pellets/reciclo/GN), 1 estado de FeO NaN |
| Cobertura por fase/orden | 97.5-99.4 % (mínimo F6 97.5 %, R2-R3 98.3 %); 362/362 batches con ≥ 1 recomendación; 23 batches con < 10 |
| ¿Excluye batches buenos/malos? | KPI mediana batches con 10 recomendaciones 68.3 vs < 10: 69.4 (MWU p 0.13); ρ(n recomendaciones, KPI) = −0.08 (p 0.15) |
| Avisos | R3 recibe recomendación aunque su target falte en el 29 % (predice desde el estado: correcto). En F6 y R3 la duración real es endógena (93 % / 85 % ≠ plan) y `J_costo` usa `duracion_plan`; en F6 ρ(Δt real, ln_sn_dep) = 0.25 |

Veredicto 6: RESCATAR. Corrección: en la ficha, sustituir "3577 de 3982" por "3577 de 3620 modelables (98.8 %); el resto son primeros
escalones"; añadir el indicador de duración endógena a la discusión de F6/R3.

## Correcciones priorizadas para la ficha v7

1. Reescribir "3 % de escalones sin masa" → 8 % de Reducción / 29 % de R3 sin ensayo; añadir sensibilidad del anclaje (K, w) a esos batches.
2. Reportar el OOF cronológico de FeO (0.296) junto al aleatorio (0.333) y adoptar el primero como criterio para ese target.
3. Retirar `espesor_ladrillo_norm_mm` del estado (o declararlo reloj de campaña) — no cuesta R² y evita extrapolar en lockbox.
4. Cerrar el agujero del filtro (`m6_G_pct`; `fe.es_feature_segura_para_prescripcion` con `"LEAKAGE" in rol`) y añadir un test de las 281 columnas.
5. Corregir `diccionario_datos.json`: T horno, tiro, lanza y termocuplas son agregados PI por escalón (media/último: confirmar con planta).
6. `v7_lib.CONFIG` vivo o documentado; fijar versiones; hash de caché y Excel.
7. ΔT de Fusión: indicador `es_F1` o ajuste sin F1 (0.48 vs 0.38); dejar constancia de que la fila "PLM v5 ref" corre sobre la base con F1.
8. Confirmar con planta que la ley de Sn por corriente se conoce antes del batch (hoy sin evidencia de retro-cálculo: cierre −1.9 % ± 6.6 %).

Scripts de comprobación (scratchpad de la sesión): `a1_datos.py`, `a2_fuga_tiempo_calib.py` (+ `a2_folds.csv`, `a2_calib_folds.csv`),
`a3_cobertura.py`, `a4_smoke.py`, `a5_excel.py`, `a6_final.py`, `a7_f1.py`.
