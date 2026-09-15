# balances — iteración 6: balances de masa y energía inspirados en `func-teorethical-model` (2026-09-14, noche)

Pregunta del `/goal`: usar la lógica del modelo teórico de los metalurgistas (balance de masa y energía por batch, Rust) para crear o
mejorar features y targets de escalón con sustento pirometalúrgico, incorporando las ideas del equipo de planta (masa en vez de ley,
conservación de masa y energía, variación de temperatura, pérdidas por volatilización/splash/finos, rendimiento refinado con Sn en
escoria, Fe en escoria como monitor de sobre-reducción). Diseño en `ITERACION_6_diseno.md`; librería `bal_features.py` (bloque `b6_*`);
ejecución paralela B-01..B-04 (Sonnet 5.1), hipótesis propias y síntesis B-05 (Fable 5.1).

## 0. Respuesta corta

| Idea (origen) | Implementación | Veredicto | Evidencia |
|---|---|---|---|
| Rendimiento refinado con el Sn de la escoria final (equipo de planta) | balance global de masa por batch (`construir_resumen_batch_balance_global`, bloque v6 de `feature_engineering.py`) → `recuperacion_refinada_pct` | **ADOPTAR** | masa final de escoria 57.9 t (trazador CaO 53.5 t); Sn en escoria final 690 kg = 1.3 % del Sn de salida; ρ 0.99 con el proxy pero reclasifica 9.4 % de batches de cuartil; robusto a supuestos (< 0.2 pp); techo de predictibilidad ≥ proxy en lockbox (R² 0.089 vs 0.002) |
| "Fe en escoria como monitor de sobre-reducción" (planta) | inicio y magnitud de la pérdida de FeO por batch (b05 H2) | **YA CUBIERTO por el SDI**; el escalón de inicio es diagnóstico complementario | la pérdida de FeO arranca en R0 en 78 % de los batches; pérdida total ρ −0.23 con el KPI (p 6·10⁻⁵), %Sn al inicio ρ −0.04 (n.s.), inicio más tardío ρ +0.15 (p 0.009) |
| Cierre multi-inerte (CaO, SiO₂, Al₂O₃, MgO) para masa de escoria en Reducción (Rust: "Dump 1/2") | `agregar_multitrazador` → `b6_sdi_multi`, `b6_feo_extraido_multi_kg` | **RECHAZAR** (todas las variantes) | autocorr lag-1 de FeO_ext −0.50..−0.68 (CaO único −0.15); R² OOF HGB ≈ 0 (CaO 0.25); ρ con KPI 0.23 vs 0.32; IC del carbón 2.5-6× más anchos |
| Carbón como demanda estequiométrica + O₂ libre + Fe metálico (Rust: `obj_carbon`, `free_oxygen`, ec. 6) | `b6_reductor_neto_kg_ceq`, ratio, capacidad, acumulados | **RECHAZAR** como sustituto de las palancas v4; acumulados **RECHAZAR** | R² OOF idéntico (Δ ≤ 0.002) en 5 targets; θ sobre FeO +115 [−12, +346] kg/sd (IC 4× más estrecho que el del carbón v4, pero cruza 0); signo incorrecto sobre Sn; acumulados degradan lockbox (SDI 0.143→0.077) |
| Cementación Fe + SnO → FeO + Sn con todo el Fe metálico (Rust ec. 6) | θ del C-eq del Fe del dross en el PLM de Sn extraído en Fusión (b05 H1) | **NO SE OBSERVA** (feedback al modelo Rust) | θ −40..−85 kg/sd (signo contrario); a nivel batch dross de Fe en Fusión no mueve f_dross ni f_metal |
| Balance de energía por escalón (Rust: balance térmico del horno; planta: "variación de temperatura") | `b6_q_*`, `b6_dT_teorico_sin_reaccion` en el modelo de ΔT; balance calibrado por NNLS; calor de reacción aparente | features térmicas **ADOPTAR como estado opcional en Reducción** (ganancia pequeña); balance físico **RECHAZAR**; estimador energético de la extracción **RECHAZAR** | ΔT Reducción R² OOF 0.466→0.474, lockbox 0.465→0.501 (HGB 0.485/0.504); Fusión sin mejora (0.48/0.05); balance calibrado R² OOF 0.03-0.09 (la T de cara interior no es la T de baño; coeficiente de combustión 0.03 en vez de 1); ρ(calor aparente, extracción) ≤ 0.25 |
| "Carbón perdido al tiro" 19.5 % F / 48 % R (Rust) | utilización del reductor por batch (`b6_utilizacion_reductor`) | **NO SE REPRODUCE** (feedback al modelo Rust); diagnóstico de arrastre RONDA 2 | utilización mediana 1.03 (F) / 1.05 (R): el carbón alimentado ≈ demanda estequiométrica; regresión contra proxies de arrastre R² 0.12 con signos mixtos |
| Residuo de masa no explicado por reducción (planta: splash, finos) | `b6_residuo_masa_kg` | RONDA 2 (bandera de QC, no feature) | sesgado +1.1 t en el primer escalón válido de Reducción (mezcla incompleta); ρ 0.13-0.16 con gas de lanza; sin correlación con polvo ni KPI por batch |
| Talón + ganga por tolva por regresión multi-óxido (Rust: talón 600 mm; composición del talón) | `b04` parte 2 (NNLS sobre log-cocientes) | RONDA 2 (inadmisible en la forma actual) | coeficientes de SiO₂ saturan en 100 %, talón de SiO₂ en cota, pureza de cal en cota inferior, R² hold-out ≤ 0, masa 5.4× la del balance |
| Avance de la pérdida de FeO como estado (b05b) | `b6_avance_feo_prev` en el estado del PLM | RONDA 2 | FeO OOF 0.367→0.383 pero lockbox 0.378→0.361; SDI OOF 0.131→0.154, lockbox 0.143→0.044 |

Lo que cambia para el recomendador: (1) el KPI de batch para evaluar políticas pasa a `recuperacion_refinada_pct` (con el Sn de
escoria); con ese KPI, en el lockbox replican los targets de matriz (ln IRF en Reducción ρ 0.38, IRF fin de Fusión 0.34) y no el SDI
(0.09), lo que refuerza la recomendación de la iteración 5 de usar lnIRF_R (o z(SDI)+z(lnIRF_R)) como objetivo de Reducción; (2) el
modelo térmico de Reducción puede incorporar `b6_q_neto_por_min_MJ` y `b6_dT_teorico_sin_reaccion` como estado; (3) nada de lo demás
sustituye a v4/v5.

Lo que se devuelve al equipo metalúrgico sobre su modelo teórico: la cementación total por Fe metálico (ec. 6) y el 48 % de carbón
perdido al tiro no se observan en los 362 batches; los óxidos "inertes" SiO₂/Al₂O₃/MgO no sirven como trazadores por escalón con la
precisión de ensayo actual; el balance global por batch sí cierra (sesgo −1.9 % del Sn, sd 6.6 %) y permite estimar el Sn en escoria.

## 1. Librería `bal_features.py`

Cuatro familias de columnas `b6_*` con clasificación anti-fuga (`CLASIFICACION_V6`, `SEGURAS_V6`) y supuestos parametrizados
(`SUPUESTOS`): A) cierre multi-trazador (Reducción), B) balance reductor/oxidante en kg C-eq, C) balance de energía por escalón (MJ),
D) balance global por batch (`construir_batch_balance`). Fórmulas en el docstring del módulo. Solo D pasó a `feature_engineering.py`.

## 2. Experimentos

| Script | Qué hace | Resultado | Memo |
|---|---|---|---|
| `b01_multitrazador.py` | consistencia entre inertes, Monte Carlo y huella empírica de ruido, targets multi vs CaO (ST-01/02/03), residuo de masa | multi rechazado; residuo RONDA 2 | `b01_resultados.md` |
| `b01b_multitrazador_ronda2.py` (Fable) | consensos alternativos (pesos empíricos, sin CaO, SiO₂+Al₂O₃) | todos rechazados | `b05_resultados_fable.md` |
| `b02_balance_reductor.py` | descriptiva, PLM v4 con variantes de palancas/estado, utilización, sensibilidad, curvas de respuesta | rechazado como sustituto; IC de FeO más estrecho; utilización ≈ 1 | `b02_resultados.md` |
| `b03_balance_energia.py` | descomposición, balance calibrado NNLS/OLS, modelo de ΔT con features térmicas, calor de reacción aparente, sensibilidad | térmicas: ganancia pequeña en Reducción; físico y estimador rechazados | `b03_resultados.md` |
| `b04_balance_global.py` | balance global, factor de escala, KPI refinado, targets vs KPI refinado, techo, modelo multi-óxido | KPI refinado adoptado; factor rechazado; multi-óxido RONDA 2 | `b04_resultados.md` |
| `b05_hipotesis_fable.py`, `b05b_avance_feo_estado.py` | cementación por dross de Fe (H1), monitor de Fe (H2), avance de FeO como estado | H1 no se observa; H2 cubierto por SDI; estado RONDA 2 | `b05_resultados_fable.md` |

## 3. Detalle de los hallazgos que cambian decisiones

**KPI refinado (B-04).** `recuperacion_refinada_pct = 100·M/(M+D+P+Sn_esc)`, con Sn_esc = masa de escoria por balance global ×
%Sn del último ensayo de Reducción. Tornado: humedad de carga y ley de Sn del polvo mueven la masa absoluta 7-8 t, pero el KPI < 0.15 pp.
Targets ganadores vs KPI refinado (DEV): SDI 0.317, lnIRF_R 0.297, ln FeO retenido 0.286, IRF fin F 0.207; lockbox: lnIRF_R 0.378
(p 0.002), IRF fin F 0.341 (p 0.006), SDI 0.087 (n.s.). Techo (Ridge, 44 agregados): R² OOF 0.158, lockbox 0.089 (proxy: 0.152 / 0.002).
`sn_perdido_escoria_frac` se explica con R² OOF 0.54 (HGB) una vez retiradas las 4 columnas terminales que comparten dato con él.

**Multi-trazador (B-01, B-01b).** Los ln-cocientes de SiO₂/Al₂O₃/MgO entre escalones consecutivos tienen sd 0.07-0.15 (7-15 %) vs
consenso; correlaciones entre inertes 0.13-0.74; el sesgo se concentra en el primer escalón válido de Reducción (mezcla incompleta de la
cal). A nivel escalón el trazador CaO re-anclado (cum cal / %CaO) conserva señal (R² OOF FeO 0.25, lag-1 −0.15) y el multi la pierde (R² ≈ 0,
lag-1 −0.5..−0.68). Los pesos de ensayo asumidos (`SD_REL_ENSAYO`) no coinciden con la precisión empírica (Al₂O₃ subponderado), pero
recalibrarlos no cambia el veredicto.

**Balance reductor (B-02).** Coherente con el modelo Rust en la descriptiva (Fusión siempre en déficit de reductor; R1 es el único
escalón con reductor neto positivo, +251 kg C-eq, y capacidad de sobre-reducir 1.35 t de FeO). Como palanca: θ(FeO) +115 [−12, +346]
kg/sd, IC 357 kg de ancho frente a 1 595 del carbón v4 (los dos términos de carbón de v4, `tasa_feed_Carbon` y `Cx_av`, se pisan con
signos opuestos); estable en toda la grilla de supuestos; sin ganancia de R²; signo incorrecto sobre Sn porque resta la demanda de Sn
(que es el propio motor cinético). El término de CO de la lanza es de segundo orden en Reducción. Sin óptimo interior en barridos 1-D
(el PLM es lineal por palanca).

**Balance de energía (B-03).** Descomposición mediana por escalón: Fusión combustión 47.8 GJ, carga 34.2 GJ, gases 18.1 GJ, pérdidas
5.3 GJ, neto −4.0 GJ (el balance sin calibrar omite el carbón quemado con aire de post-combustión, que no está en el dataset, y la
oxidación de Fe metálico/sulfuros); Reducción neto +4.5 GJ, ΔT teórico +66 K frente a −4 K medidos. Calibración NNLS: coeficiente de
combustión 0.03 (teoría ~1), pérdidas 0, R² OOF 0.03-0.09: la temperatura de cara interior no responde al balance térmico del baño
(q_neto correlaciona −0.33 con ΔT en Reducción). Modelo de ΔT: PLM v4 + térmicas en Reducción 0.474/0.501 vs 0.466/0.465 (HGB 0.485/0.504);
Fusión 0.466/0.052 vs 0.481/0.066. Calor de reacción aparente vs trazador: ρ −0.25 (F), +0.21 (R): sin estimador independiente.

## 4. Límites

Todos los balances heredan los supuestos de `SUPUESTOS` (PCI, carbono fijo, Fe del mineral y del dross, leyes de metal/dross/polvo,
humedad); las sensibilidades muestran que las conclusiones (adoptar/rechazar) no dependen de ellos, pero las magnitudes absolutas sí
(masa de escoria ±7 t). No hay aire de post-combustión, temperatura de baño ni análisis de carbón en el dataset: el balance de energía
por escalón no puede cerrarse con estos datos. El lockbox (fin de campaña) sigue siendo un régimen distinto: las features de estado
acumulado (reductor, avance de FeO) ganan en OOF y pierden allí.
