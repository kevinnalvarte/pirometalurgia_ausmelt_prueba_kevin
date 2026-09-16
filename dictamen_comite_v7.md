# Dictamen del comité de pirometalurgistas analíticos sobre el enfoque ganador v7 (2026-09-16)

Artefacto: https://claude.ai/artifact/UyEetgRSM2JjfRnNauLmV8 (fuente `experimentos/v7/figs/dictamen_v7.html`).

Comité: cuatro auditores independientes (A fundamento físico-químico Ausmelt/TSL, B inferencia estadística y causal, C operación y seguridad
del horno, D integridad de datos y validación) con acceso completo al repositorio y comprobaciones numéricas propias, más la presidencia
(síntesis, potencia del piloto, tasa de cambio implícita del objetivo). Informes: `experimentos/v7/auditoria_A_pirometalurgia.md`,
`auditoria_B_estadistica.md`, `auditoria_C_operacion.md`, `auditoria_D_datos.md`. Objeto auditado: `candidate_win_model_v7.md`,
`modelo_predictivo_v7.py`, hallazgos §19-20, memos E9-00..E9-08 y sus CSV.

Escala: **RESCATAR** = válido hoy con la evidencia disponible; **EN DUDA** = no invalidado pero no demostrado, con la prueba que lo resolvería;
**DESCARTAR** = falso, no identificado o inaplicable; se indica el sustituto.

---

## 0. Dictamen global

El enfoque v7 es la formulación más honesta que permiten los datos y su **núcleo predictivo de Reducción es válido**: masa de escoria por
cierre físico, targets logarítmicos agregables, modelo parcialmente lineal con cross-fitting por batch, estado de 16 features, validación sin
fuga y reproducible. Lo que **no está demostrado** es lo que convierte ese núcleo en recomendaciones: (i) los efectos de las palancas sobre la
retención de FeO, que sostienen el 82 % del uplift, no están identificados (sus intervalos libres cruzan cero; la cota de signo fabrica la
significancia); (ii) los pesos del objetivo (K, w) dependen del régimen (w 7.86 en desarrollo, ≈ 1 en el lockbox) y valoran un kg de Sn o de
FeO de forma muy distinta según el escalón; (iii) el optimizador produce soluciones de esquina (carbón en la cota P99 en el 52 % de los R0),
mueve el aire sin efecto identificado ni costo, y su término de incertidumbre es una constante; (iv) las recomendaciones de Reducción no son
ejecutables en tiempo real porque el estado exige el ensayo de laboratorio del escalón anterior; (v) la evidencia off-policy no tiene potencia
ni para demostrar mejora ni para excluir perjuicio. Se **descartan** la conservación de Sn en Fusión como principio, el aire como palanca, el
escalón F1 como escalón recomendable, la lectura "carbón selectivo", el techo de ruido de FeO 0.57 y la afirmación "las recomendaciones
mejoran el batch". La única recomendación con respaldo físico y estadístico convergente es **recortar el carbón en R2-R3**; la de más carbón en
R0 es de riesgo alto y no debe desplegarse.

Conclusión operativa: **no desplegar en línea**. Pasar a un piloto restringido y aleatorizado por escalón, con salvaguardas duras, tras las
correcciones de la fase 0. Hoy nadie puede afirmar que las recomendaciones "aportarán valor sí o sí"; sí puede afirmarse que, con las salvaguardas
de la hoja de ruta, el piloto no expone al horno a estados fuera del envolvente histórico y produce en 2-4 meses la evidencia que hoy falta.

---

## 1. Lo que se rescata como válido

| Componente | Por qué es válido | Evidencia | Auditor |
|---|---|---|---|
| Masa de escoria por cierre físico (v6) | Único uso coherente de un ensayo de Fe total sin especiación; supera al trazador de CaO; los targets de Reducción son invariantes a pG/gG (corr 0.99999 al recalibrar por fold) y a δ_R (desplazamiento aditivo exacto) | `masa_v6.py`, E8-01, chk D §2 | A, D |
| Targets `ln_sn_dep` y `ln_feo_ret` como contabilidad agregable | Identidad telescópica exacta (0.005 ± 0.018 / 0.000); un solo (K, w) por fase es coherente; el objetivo del batch depende sólo del estado terminal | E9-00 §1-2, E9-03 §4 | A, B |
| `ln_feo_ret` como proxy de hardhead (retención de **Fe total** en escoria) | Complemento por balance del Fe metalizado; f_dross ~ Σln_feo_ret −0.045/unidad (p 0.04) con el signo correcto | A §2 | A |
| Forma PLM con cross-fitting por batch | Ningún algoritmo de 14 ni la forma cinética estructural la supera (Δ ≤ +0.005, IC cruza 0); curva de aprendizaje saturada desde n ≈ 150 | E9-01, E9-05 | B |
| Estado de 16 features (k16) como predictor | 0.767 ± 0.003 / 0.326 ± 0.007 en 4 particiones aleatorias frente a FULL 0.765 / 0.305; sin fuga (todas usan cierre de t−1 o acción de t); sin salidas de batch en estado, palancas ni soporte | E9-06, chk B §2, D §1 | B, D |
| Efecto del carbón sobre el agotamiento de Sn (`Cx_sn` +0.178 [0.041, 0.285]) y del GN sobre el Sn (+0.047 [0.011, 0.086]) | Óptimos interiores: el IC acotado coincide con el libre (1 % de réplicas en la cota); robustos a 4 semillas y 3 sets | chk B §1 | B |
| Término de legado (FeO retenido → KPI del batch siguiente) como **señal** | +10.4 pp/unidad (p 2e-4), sobrevive a KPI actual como control (+9.4, p 0.0015), HAC, placebo inverso (p 0.70) y **replica en lockbox (+19.8, p 0.011)**; es el hallazgo mejor protegido frente a pruebas múltiples | E9-00b, chk B §3 | B (A duda del mecanismo, ver §2) |
| Recorte de carbón en R2-R3 (−1.3 / −2.6 / −1.4 kg/min) | Coincide con la práctica TSL y con la selectividad marginal medida (5.3 → 0.44 kg Sn/kg FeO de R0 a R3): en R3 cada kg de Fe reducido saca 0.44 kg de Sn y arrastra ~0.65 al dross | A §6 | A, C |
| Cadena de validación | OOF GroupKFold por batch como criterio; DEV/lockbox sin solape; calibración pG/gG sólo DEV; tabla reproducida a 1e-16; cobertura 3577/3620 escalones modelables sin sesgo por KPI | D §4-6 | D |
| Abstención sobre la lanza y la cal | La convención de la lanza no está confirmada; la cal no tiene contraste en el rango histórico (P25-P75 9.1-10.2 % de la carga) | E9-02, A §3 | A |
| Procedimiento de reentrenamiento por campaña y de cambio de gestión | ΔT de Fusión 0.04 → 0.16-0.32 con reentrenamiento; rango en vez de punto, códigos de razón, registro de aceptación | E9-04, C §5-6 | C |

## 2. Lo que queda en duda (y la prueba que lo resuelve)

| Componente | Duda | Cómo se resuelve |
|---|---|---|
| Efectos de GN, O₂, aire y carbón sobre la retención de FeO | IC libres: GN [−0.0174, +0.0002], O₂ [−0.0009, +0.0068], aire [−0.0033, +0.0047], carbón ±0.045; GN sólo se identifica en el último tercio; "carbón selectivo" = no identificado (residuos carbón/Cx_av r 0.97) | Piloto aleatorizado por escalón: GN ±1 sd en R1-R3 (~900 escalones por brazo) y carbón ±1 sd en R0-R1 (~55 escalones por brazo, ≈ 15 batches) |
| Pesos del objetivo K y w | K_Sn IC [1.8, 5.5]; K_FeO atenuado por errores-en-variables ×1.2-1.4; w 7.86 en DEV frente a ≈ 1 en lockbox; 1 kg de Sn agotado vale 2.0-2.5 kg de Sn a metal (imposible físicamente, ≤ 1 + pérdidas evitadas); 1 kg FeO retenido vale 6.9 kg Sn en R0 y 0.68 en R3 | Objetivo lineal en kg (ΔSn_kg − λ·ΔFeO_kg, λ ≈ 0.5-1.5) como contraste; sensibilidad de la política a w ∈ [1, 11]; K corregido por EIV; re-anclaje con el nuevo holdout |
| Mecanismo del legado | Señal estadística sólida, pero persiste dos batches (un talón físico decae en uno) y su signo contradice el mecanismo de talón metálico; asimetría hacia adelante 0.25 vs 0.07 hacia atrás | Confirmar con planta el protocolo de sangrado y el talón residual; hasta entonces `CONFIG_V7_SIN_LEGADO` (w 6.09, K 2.89) como base y el legado como sensibilidad |
| Cierre de masa en la transición F6 → R0 | Fe³⁺ de la Fusión oxidante (30 % → −0.04 en `ln_feo_ret` de R0, 27 % de la señal); Fe metálico del dross alimentado en R0 (+0.03 a +0.14); Sn²⁺ vs Sn⁴⁺ (K 1.27 vs 1.135) | Titulación Fe²⁺/Fe³⁺ y Sn²⁺/Sn⁴⁺ en escoria F6 y R0 de 20-30 batches; %Fe del dross; descontar el FeO equivalente del Fe alimentado |
| `ln_sn_dep` como cinética y su ceguera al polvo | R0 es carbón-limitado (0.18 kg C/kg Sn, orden ≈ 0 en Sn); 9.3 t de Sn por batch van a polvo (18 %) sin medición por fase; KPI ~ carbón de R0 −0.43 pp/sd (p 0.10), polvo +0.003/sd (p 0.04-0.09) | Pesaje/ensayo de polvo de BHF por batch y fase; piso C* (Sn de equilibrio) como sensibilidad; forma `ln(1 + C/Sn_disp)` para el carbón |
| Estado k16 como estado "físico" | `termocupla_media` correlaciona −0.97 con el espesor (edad de campaña); `cum_aire` es un reloj (0.92 con el tiempo); `espesor` es la fecha (ρ −0.9996) y su rango en lockbox es disjunto del de DEV; se pierden T de horno y Al₂O₃ | Retirar `espesor` (no cuesta R²: 0.767/0.800, 0.325/0.469); recuperar T horno y Al₂O₃ (k ≈ 18) si el carbón sigue identificado; reentrenar por campaña obligatorio |
| Métrica de generalización de FeO | OOF aleatorio 0.333 frente a cronológico 0.296 y walk-forward 0.261; lockbox gastado (≥ 3 decisiones lo tocaron) | Reportar el OOF cronológico y el walk-forward como criterio para FeO; pre-registrar un nuevo holdout (primeros N batches del piloto) |
| Techo de ruido de FeO | El cálculo 0.57 ignora que el ruido de t−1 está en el estado y es revertible; techo corregido 0.79 (PLM al 42 %); la inferencia "ruido ≤ 2.8 %" y la autocorrelación −0.11 no son diagnósticas | Techo = 1 − var(ruido no revertible)/var_total; medir la fracción de R² que es reversión de ruido (OOF con y sin `%FeO_prev`) |
| Fusión: recorte de carbón −4.4 kg/min | Defendible por costo y polvo (f_polvo ~ C_F +0.005/sd, p 0.01), no por conservar Sn; el carbón de Fusión es en parte combustible (el exceso de O₂ quema 6-12 kg C/min); el 39 % del uplift son costos con precios placeholder (carbón implícito 1 500 USD/t) | Re-optimizar con precios reales; recorte ≤ 1 sd del orden y ≥ P10; bajar O₂ en proporción (λ constante); prohibido en fin de campaña |
| Latencia del ensayo | R0-R2 duran exactamente 30/25/20 min y no esperan al laboratorio; con latencia ≥ 15 min, R1-R3 no son recomendables en tiempo real (7 de 16 features son de laboratorio) | Registrar hora de toma y de reporte del ensayo; arquitectura de dos niveles (receta de Fusión + plan R0-R3 al inicio de Reducción con F5 conocida y F6 simulada, re-plan tras R0) |
| Uplift modelado +0.27 / +0.35 pp | IC honesto con K, w y θ propagados ≈ [0.12, 0.63]; "> 99 % positivo" es tautológico (el optimizador maximiza la misma J); maldición del optimizador no corregida | Recomputar sólo con θ interiores; IC propagado; evaluación del uplift con un modelo distinto del que optimiza |
| Cal y basicidad | Sin contraste en el rango histórico, no refutada físicamente (canales polvo, Sn en escoria y FeO responden a B2 con p < 0.01) | Piloto de cal (riesgo bajo) antes que el de carbón: B2 0.45-0.65 |

## 3. Lo que se descarta (y su sustituto)

| Componente | Motivo | Sustituto |
|---|---|---|
| Conservación de Sn en Fusión (β_F ≠ 0) como principio | Contrario a la selectividad termodinámica de la primera etapa TSL; β_F −0.58 p 0.17; R² lockbox −0.70 | β_F = 0; Fusión gobernada por costos reales, ventana térmica dura y polvo |
| Aire como palanca | Setpoint de protección de lanza (120 Nm³/min, P25-P75 119-122); θ nulos en Sn, FeO y ΔT; costo cero en J; recomendaciones de hasta +26 Nm³/min | Aire y O₂ congelados al plan; sólo reingresan con costo y θ identificado |
| Escalón F1 como escalón recomendable | ΔT R² 0.04-0.07; agotamiento sin generalización; uplift 6.6 kg = recorte genérico de carbón | F1 hereda la receta de Fusión (nivel 0) |
| "Carbón selectivo" (θ ≈ 0 sobre FeO) como hallazgo | Es ausencia de identificación (IC libre ±0.045/sd, 5× el efecto de GN) | Declarar "no identificado"; piloto por escalón |
| Signo negativo de O₂ sobre ΔT como física | Control reactivo (corr O₂–T_prev −0.53): más O₂ cuando el horno está frío | ΔT sólo como ventana, no para inferir el signo del O₂ |
| Techo de ruido de FeO 0.571 y la inferencia "ruido real ≤ 2.8 %" | Cálculo que ignora la reversión del ruido de t−1 vía el estado | Techo corregido 0.79 |
| Cifra "3 % de escalones sin masa" | Reducción: 8 %; R3: 29 % sin ningún ensayo (105 batches; en ellos el KPI refinado usa la ley de R2) | Documentar; sensibilidad del anclaje con y sin esos batches |
| Evidencia off-policy como prueba de valor | Polvo q-BH 0.24 en 18 pruebas y sin réplica en lockbox; dist_R con signo contrario; placebo por permutación no informativo; MDE 0.59 pp/sd | Piloto aleatorizado por escalón; la regresión de adherencia sólo como monitoreo |
| Afirmación "las recomendaciones mejoran el batch" | Cálculo del modelo, no evidencia; potencia 32 %; un A/B sobre el KPI exigiría ~3 500 batches por brazo (≈ 8 años) | Endpoints de escalón y de consumo (§5); el KPI como seguimiento de seguridad |
| Término de incertidumbre de J tal como está | Constante entre candidatos (`n_boot=0` en la política cross-fitted; sd del ancho 0 en 20/20 curvas) | Bootstrap activo o IC empírico de θ que dependa de la acción |
| Límites globales de fase y carbón "en la cota" en R0 | 52 % de los R0 en la cota P99 (66 kg/min; histórico 8 %); saltos desde estados sin soporte (AP0058: 3.4 → 58 kg/min) | Límites P5-P95 por orden, |Δ| ≤ 1 sd del orden, techo P90 en R0, sin recomendación con soporte < P10 |

## 4. Correcciones obligatorias antes de cualquier uso (fase 0, sin datos nuevos)

1. Reportar θ libre y acotado; reclasificar como "no identificados" los efectos sobre FeO de carbón, O₂ y aire; recomputar el uplift sólo
   con θ interiores (Cx_sn, GN sobre Sn) y con IC que propague K, w y θ.
2. Optimizador: límites P5-P95 por orden de escalón; |Δ| ≤ 1 sd del orden; carbón de R0 ≤ P90 del orden y nunca en la cota; aire y O₂
   congelados; sin recomendación si el soporte del estado es < P10, faltan leyes de t−1 o hay código de razón activo; ventana térmica como
   restricción dura (T_pred − MAE ≥ P10 del orden); IC real dependiente de la acción; costos con precios de planta (los actuales son placeholders).
3. Objetivo: base `CONFIG_V7_SIN_LEGADO` (w 6.09, K 2.89) con el legado como sensibilidad; β_F = 0; contraste con objetivo lineal en kg
   (λ 0.5-1.5) y barrido de w ∈ [1, 11]: la política debe conservar el signo en todo el rango o se recorta a lo que lo conserva.
4. Estado: retirar `espesor_ladrillo_norm_mm` (reloj); evaluar k18 con T de horno y Al₂O₃; reportar OOF cronológico y walk-forward para FeO.
5. Fusión: β_F = 0; F1 sin recomendación; ΔT con indicador de F1 o sin F1; recorte de carbón acotado y acompañado de −O₂ proporcional.
6. Datos y código: cerrar el agujero del filtro anti-fuga (`m6_G_pct`) y añadir un test sobre las 281 columnas (**hecho en esta sesión**);
   documentar el 8 %/29 % de escalones sin ensayo; `diccionario_datos.json` con "agregado PI por escalón"; `v7_lib.CONFIG` vivo; versiones
   fijadas y hash de caché; corregir el techo de ruido y la cifra de escalones sin masa en la ficha y en hallazgos §20.
7. Declarar el lockbox gastado; pre-registrar como nuevo holdout los primeros batches del piloto.

## 5. Hoja de ruta para llegar a un prescriptor robusto, seguro y rentable

| Fase | Qué | Criterio de salida | Plazo orientativo |
|---|---|---|---|
| **0. Corrección y prescriptor conservador (v7.1)** | Las 7 correcciones de §4; política reducida a dos palancas (carbón de Fusión F2-F6 acotado; carbón de Reducción con techo P90 en R0 y recorte en R2-R3; GN R1-R3 con piso P10); arquitectura de dos niveles | Política que conserva el signo bajo w ∈ [1, 11] y bajo objetivo lineal en kg; ninguna recomendación fuera de P5-P95 del orden; IC dependiente de la acción; caso económico con precios reales | 2-3 semanas de analítica |
| **1. Datos de planta que hoy faltan** | Hora de toma y reporte del ensayo; Fe²⁺/Fe³⁺ y Sn²⁺/Sn⁴⁺ en F6 y R0 (20-30 batches); %Fe del dross y del metal crudo; polvo de BHF por batch y fase; composición del GN; separación de aire de camisa y de proceso; protocolo de sangrado y talón; ley de Sn por corriente antes del batch; convención de la lanza; precios (Sn neto, carbón, GN, O₂, aire, lanza/refractario, hora de horno, mermas); T de baño si existe | Cada dato con su dueño y fecha; con ellos se corrige el cierre F6→R0, el legado, el canal polvo y el caso económico | 4-8 semanas en paralelo con la fase 0 |
| **2. Modo sombra** | El sistema recomienda sin mostrarse durante 30-60 batches: registra recomendación, acción real, código de razón, MAE por orden, sesgo, soporte vivo, % en cota, escalones sin estado | MAE ≤ 1.2× la de referencia; sesgo por orden |b| ≤ 0.10; % en cota ≤ 30 %; latencia del ensayo medida; sin evento de seguridad atribuible | 30-60 batches (2-3 semanas) |
| **3. Piloto aleatorizado por escalón (identificación de θ)** | Carbón en R0-R1: ±1 sd alrededor de la recomendación, con techo P90 del orden, bloqueado por batch (≈ 55 escalones por brazo, ≈ 15-30 batches por brazo); después GN en R1-R3 con piso P10 (≈ 230 batches por brazo). Endpoints: `ln_sn_dep` y `ln_feo_ret` del escalón; polvo por fase; T pre-BHF; eventos | θ del carbón y del GN con IC que excluya 0 **sin cotas de signo**; ausencia de eventos (espumado, T gas > 200 °C, Fe en metal fuera de especificación) | 1-3 meses |
| **4. Piloto de política restringida (no inferioridad y valor)** | A/B por batches alternos, 120 batches, dos palancas, aire y O₂ congelados; endpoints primarios factibles: consumo de carbón de Fusión (10 por brazo), Sn en escoria final ajustado (50-110 por brazo), `ln_sn_dep` de R0 ajustado (80 por brazo); KPI refinado, f_dross, f_polvo como seguridad; paradas explícitas (dross tratado > control + 2 pp en 20 batches; KPI < control − 1 sd en 10 batches; rechazo del operador > 50 %; cualquier evento) | No inferioridad demostrada en KPI, dross y polvo; endpoints primarios en la dirección predicha; caso económico positivo con precios reales | 40-60 días |
| **5. Decisión de despliegue** | Con los resultados de 3 y 4: re-anclar K, w y β con el nuevo holdout; θ re-estimados con datos aleatorizados; política final | Uplift con IC propagado > 0 y ≥ costo de operación del sistema; drift bajo control; procedimientos de congelar/desactivar/revertir probados | — |
| **6. Mejoras de modelo (en paralelo, sin bloquear)** | Objetivo lineal en kg como alternativa; piso C* de equilibrio; descuento del Fe alimentado en R0; θ heterogéneo (carbón × T, carbón × B2); ganga por corriente; estado en línea (CO/CO₂, T de baño, XRF/LIBS rápido) para recomendar R1-R3 en tiempo real; piloto de cal (B2 0.45-0.65) | Cada mejora con OOF cronológico y walk-forward, y sin perder la identificación del carbón | continuo |

Garantías de "no perjuicio" durante las fases 2-4: el sistema corre por debajo de los enclavamientos de planta y nunca los sustituye; la acción
histórica y la receta estándar son siempre admisibles; ninguna recomendación sale del envolvente P5-P95 del orden ni supera 1 sd de cambio;
carbón de R0 con techo P90; aire y O₂ intocados; prohibido recortar carbón de Fusión en fin de campaña o con T previa en el decil inferior;
desactivación automática con 3 indicadores de drift en alerta o un evento de seguridad; el operador conserva la decisión y registra el motivo.

## 6. Qué puede afirmarse hoy y qué no

- Puede afirmarse: el modelo predictivo de Reducción es el mejor que permiten estos datos (forma, estado, algoritmo), su agregación por escalón
  es exacta, el carbón y el GN mueven el agotamiento de Sn con efectos identificados, retener FeO se asocia con mejor recuperación en este
  batch y en el siguiente, y el recorte de carbón tardío es coherente con la física y con la selectividad medida.
- No puede afirmarse: que la política actual mejore el KPI (ni que no lo empeore), que los efectos sobre la retención de FeO existan con la
  magnitud usada, que los pesos del objetivo sean estables entre campañas, que la conservación de Sn en Fusión aporte, ni que las
  recomendaciones sean ejecutables en R1-R3 con la latencia actual del laboratorio.
- La rentabilidad defendible hoy es del orden de 20 t de Sn por campaña (≈ 0.6 MUSD a 30 kUSD/t) frente a las 50-65 t del cálculo del modelo;
  el 39 % del uplift modelado son costos con precios placeholder. El caso económico se cierra en la fase 1 con precios reales y en la fase 4
  con datos aleatorizados.
