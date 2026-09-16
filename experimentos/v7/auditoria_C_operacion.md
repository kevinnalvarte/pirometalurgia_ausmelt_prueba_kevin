# Auditoría C — aplicabilidad operativa y seguridad del prescriptor v7 (2026-09-16)

Auditor C (operación de hornos TSL/Ausmelt, implantación de recomendadores en línea). Fuentes: `candidate_win_model_v7.md`, `hallazgos.md` §20,
`modelo_predictivo_v5.py` (`objetivo_lote_v5`, `optimizar_accion_v5`, `PESOS_V5_DEFAULT`, `PERCENTIL_SOPORTE_MINIMO`, `FEATURES_SOPORTE_V5`),
`modelo_prescriptivo.py` / `modelo_predictivo_v3.py` (soporte kNN), `modelo_predictivo_v7.py`, `descripcion_fases.md`, `diccionario_datos.json`,
y comprobaciones numéricas propias sobre `e9_07_recomendaciones_v7_escalones.csv` (3 577 escalones, 362 batches cross-fitted),
`e9_07_curvas_respuesta_v7.csv`, `e9_07_efectos_control_v7.csv`, `e9_07_por_batch_v7.csv` y `e8_05_recomendaciones_v6_escalones.csv`.
Nada del repo fue modificado; los scripts de comprobación viven en el scratchpad de la sesión.

Qué hace el sistema (síntesis operativa): en cada escalón t toma el estado al cierre de t−1 (leyes de escoria de laboratorio, inventarios del
cierre físico, T, lanza, carbón acumulado), y busca al azar 300 + 60 candidatos de (carbón, GN, O₂, aire) dentro de los P1-P99 **globales de la
fase** (DEV), maximizando `J = K·(ln_sn_dep + w·ln_feo_ret) − costos − 2·(T fuera de P5-P95) − IC − soporte`, con K = 1 805 kg Sn/unidad,
K·w = 14 187 kg Sn por unidad de ln_feo_ret, y la acción histórica siempre admisible. Duración y cargas no se optimizan (plan).

## 0. Veredictos

| # | Punto | Veredicto | Salvaguarda mínima |
|---|---|---|---|
| 1 | Latencia del ensayo | **EN DUDA** (viable sólo con arquitectura de 2 niveles) | Fusión = receta estática; R0 y perfil R1-R3 planificados al inicio de Reducción; nada por escalón en tiempo real en R1-R3 |
| 2 | Magnitud de los cambios | **EN DUDA** (R0 saturado en la cota; aire sin fundamento) | Límites por orden; |Δ| ≤ 1 sd del orden; aire y O₂ congelados; no recomendar con soporte histórico < P10 |
| 3 | Riesgos físicos | **EN DUDA** | Restricciones duras: T baño previa ≥ P10 del orden, T gas pre-BHF ≤ 195 °C, presión de punta, códigos de razón del operador |
| 4 | F1 | **DESCARTAR** como escalón recomendable | F1 hereda la receta de Fusión, sin recomendación específica |
| 5 | Reentrenamiento / deriva | **RESCATAR** con procedimiento | Progresivo cada 10 batches, θ con signos fijos, 6 indicadores de drift, congelar automático |
| 6 | Cambio de gestión | **RESCATAR** | Rango + justificación por componente, registro aceptar/rechazar, piloto A/B con parada |
| 7 | Rentabilidad | **EN DUDA** (39 % del uplift son costos con precios placeholder) | Pedir 8 precios/costos a planta antes del caso económico |

Veredicto global: **EN DUDA → piloto restringido** (2 palancas: carbón de Fusión y carbón/GN de Reducción; aire y O₂ fuera), no despliegue en línea.

## 1. Latencia de la información

(a) El estado del escalón t exige `ley_sn/fe/sio2/cao_prev`, es decir el ensayo de laboratorio de la muestra tomada al cierre de t−1. Además, los
inventarios `m6_*_prev` y `ratio_sn_feo_prev` derivan de esas leyes. Las variables en línea del estado (termocuplas, gradiente de T, lanza,
carbón/aire acumulados, espesor) son 7 de 16.

(b) Duraciones reales (P5/P50/P95, 362 batches): R0 30/30/30, R1 25/25/25, R2 20/20/20, R3 10/11/23 min; F1 30/32/34 … F6 58/70/101 min. Las
duraciones de Reducción son **exactamente** las del plan en > 90 % de los casos: el escalón no espera al ensayo. Un XRF de pastilla prensada con
enfriamiento, molienda y prensado tarda típicamente 30-60 min desde la toma (**dato a confirmar con el laboratorio**: hora de toma, hora de
reporte). Con cualquier latencia ≥ 15 min, la ley del cierre de R0 llega cuando R1 ya terminó; la de R1 cuando R2/R3 terminaron. El operador
histórico tampoco disponía de esas leyes en tiempo real, lo que es coherente con el modelo (no hay confusión vía laboratorio), pero significa que
**el sistema, tal como está evaluado, no es ejecutable en R1-R3**. El 7.7 % de los escalones de Reducción no tiene leyes (nulos): tampoco
tendría estado.

(c) Qué se puede recomendar antes del ensayo: la política de Fusión es casi constante (carbón −3 kg/min en todos los órdenes, GN/O₂/aire
≈ 0; 71-74 % de los escalones cambian, r = 0.99 con v6) → no necesita la ley contemporánea. En R0 la recomendación es "carbón en la cota"
en el 51 % de los batches (ver §2): la ley de F6 aporta poco. El simulador de E9-03 predice el estado a un paso con R² 0.69 (Sn_inv) y 0.89 (FeO),
pero cae a 0.12 en órdenes tardíos: sirve para R0 (con la ley de F5 + predicción de F6), no para R2-R3.

(d) **EN DUDA → arquitectura propuesta**:
- Nivel 0 (antes del batch, sin laboratorio): receta de Fusión revisada (F2-F6) con el recorte de carbón acotado (§2), como cambio de estándar
  operativo, no como recomendación por escalón.
- Nivel 1 (al inicio de Reducción): plan R0-R3 calculado con la ley de F5 (conocida ≥ 60 min antes) + F6 simulada; si el laboratorio reporta
  F6 en ≤ 15 min, se recalcula R0.
- Nivel 2 (tras el reporte de R0, ≈ fin de R1): re-plan de R2-R3. Ninguna recomendación por escalón en tiempo real en R1-R3 con el modelo
  actual; para ello haría falta un estado en línea (CO/CO₂ en gases, T de baño continua, análisis rápido LIBS/XRF en línea).
- Registrar en el histórico de planta la **hora de reporte del ensayo** para cerrar esta incógnita en el piloto.

## 2. Magnitud y plausibilidad de los cambios (rec − hist, cross-fitted; sd = sd histórica DEV del mismo orden)

| Fase / orden | Palanca | hist P50 | Δ P5 | Δ P50 | Δ P95 | % \|Δ\| > 1 sd | % rec > P95 del orden | % rec < P5 del orden |
|---|---|---|---|---|---|---|---|---|
| F1-F6 | carbón kg/min | 35-52 | −13 … −23 | −2.3 … −3.2 | 0 | 16-25 % | 0-1 % | 9-21 % |
| F1-F6 | GN Nm³/min | 30.5-30.9 | −1.2 … −1.5 | 0 | +0.6 … +1.1 | 1-6 % | 1-3 % | 4 % |
| F1-F6 | O₂ / aire | 47-48 / 120 | −3 / −3 | 0 / 0 | +1.3 … +3.2 / +1.7 … +3.2 | 2-9 % / 0-1 % | 1-11 % / 1-7 % | 3 % / 4-13 % |
| R0 | carbón | 62.4 | −1.5 | **+1.5** | **+8.7** (máx +54.8) | 11 % | **49 %** | 2 % |
| R1 | carbón | 30.1 | −13.7 | −1.5 | +6.2 | 15 % | 2 % | 6 % |
| R2 | carbón | 11.1 | −6.8 | −1.5 | +6.3 | 26 % | 3 % | 12 % |
| R3 | carbón | 5.6 | −6.4 | −1.5 | +4.3 | **34 %** | 3 % | **33 %** |
| R0-R3 | GN | 28.3 / 24.9 / 18.8 / 16.7 | −2.8 … −4.4 | 0 / −0.6 / −0.7 / −0.7 | +0.8 … +3.6 | 7-16 % | 1-10 % | 2-5 % |
| R0-R3 | aire | 120-121 | −1.5 … −4.0 | 0 … +0.7 | **+5.7 / +12.4 / +25.7 / +7.3** | 2-6 % | **20 / 14 / 25 / 9 %** | 1-2 % |
| R0-R3 | O₂ | 28.6 / 23.5 / 11.1 / 6.8 | −2.6 … −5.9 | 0 … +0.6 | +2.8 … +6.4 | 3-17 % | 29 / 8 / 1 / 5 % | 1-2 % |

Comprobaciones adicionales:
- **Saturación en la cota**: en R0 el carbón recomendado es ≥ 66.0 kg/min (P99 global de la fase = 66.2) en el **50.6 %** de los batches (histórico:
  7.8 %); sube en el 69 % y baja en el 9 %. El PLM es lineal en `Cx_sn` (θ +0.178/sd, sin rendimientos decrecientes) y el costo del carbón es
  0.05 kg Sn/kg: el óptimo es una **solución de esquina**, no un óptimo interior. Una recomendación "al máximo permitido" no es una recomendación,
  es la caja de búsqueda.
- **Casos extremos**: AP0060 (hist 18.6 → rec 64.3 kg/min, uplift 922 kg), AP0058 (hist 3.4 → rec 58.2, **soporte histórico 0.00**), AP0181 (43 → 64).
  El modelo no puede saber por qué el operador corrió R0 con poco carbón (espumado, lanza, baño frío, parada): recomienda "normalizar" a ciegas.
  50 escalones tienen soporte < 0.05 y 59 estados históricos están fuera de la nube DEV; la regla "acción base siempre admisible" deja que el
  optimizador salte desde un estado sin soporte a la moda histórica.
- **Límites globales vs por orden**: en Reducción los límites son P1-P99 de toda la fase (carbón 3.2-66.2, GN 10.4-29.6, O₂ 2.9-31.1, aire 73-130),
  mientras que por orden son R0 35-66, R1 18-48, R2 4-27, R3 3-19 kg/min de carbón. En la práctica sólo el 0.3-7 % de las recomendaciones cae fuera del
  P1-P99 de su orden (R2 carbón 7 %, R1 aire 7 %), pero nada impide estructuralmente 60 kg/min en R3: los límites **deben ser por orden**.
- **Soporte mínimo (P10)**: 8.6 % (F) y 5.9 % (R) de las recomendaciones quedan por debajo del P10 del soporte histórico; el umbral duro se aplica
  contra el P10 del fold y la acción base es admisible aunque esté fuera. El soporte recomendado es < histórico en el 26 % de R (penalización blanda).
- **El término de IC no discrimina**: en las curvas de respuesta la desviación estándar del ancho del IC dentro de cada curva es 0.000 (20/20 curvas), y
  `J_incertidumbre_rec == J_incertidumbre_hist` en el 100 % de los escalones (política cross-fitted con `n_boot=0`). La "penalización por
  incertidumbre" es una constante: **no frena ninguna recomendación agresiva**.
- **Aire**: θ sobre Sn +0.0067/sd [−0.010, +0.022] (n.s.), sobre FeO +0.0008 [0, +0.005] (n.s., en cota), sobre ΔT −0.57 °C (n.s.), y **costo 0** en
  `PESOS_V5_DEFAULT`. La recomendación "aire +2..+4 Nm³/min" (P95 +25.7 en R2, 25 % por encima del P95 histórico del orden) es el producto de
  coeficientes nulos por K·w = 14 187 kg/unidad y un precio cero. **DESCARTAR** el aire como palanca hasta que tenga costo y θ significativo.
- **O₂**: θ sobre Sn en cota (−0.008 n.s.), sobre ΔT −4.2 °C/sd (sig.). Δ ≈ +0.3 con P95 +6.4 en R1. EN DUDA, congelar en el piloto.
- **v7 vs v6**: Fusión r 0.99 (mismo signo 100 %); Reducción carbón r 0.73 (77 % mismo signo), GN 0.75, O₂ 0.62, aire 0.97. Dos versiones del mismo
  sistema discrepan en el signo del carbón de Reducción en 1 de cada 4 escalones: la recomendación de carbón tardío no es estable a la especificación.

Veredicto **EN DUDA**. Salvaguardas: (i) límites P5-P95 **por orden** para la búsqueda y **techo de |Δ| ≤ 1 sd del orden** (≈ 6-8 kg/min de carbón,
1.5-3 Nm³/min de GN); (ii) sin recomendación si el soporte del estado histórico < P10 o si faltan leyes; (iii) aire y O₂ fijos en el plan;
(iv) activar el bootstrap (`n_boot ≥ 200`) para que el IC dependa de la acción, o sustituir por un IC empírico de θ; (v) presentar R0 como
"mantener carbón ≥ mediana del orden (62 kg/min)" y no como "66".

## 3. Riesgos físicos y de seguridad

| Riesgo | Qué recomienda el sistema | Comprobación | Riesgo real | Salvaguarda |
|---|---|---|---|---|
| (i) Más carbón temprano en R → espumado, sobre-reducción local, hardhead | R0 +1.5 (P95 +8.7, máx +55) kg/min, +44 kg/escalón P50, +262 kg P95 | Exceso O₂ de lanza rec P50 −0.8 % (hist −4.8 %); ningún R0 < −15 %; `pred_feo_ret` rec−hist +0.0018 (más FeO retenido); dosis C/Sn P95 baja 0.52 → 0.47 | Modelo no ve espumado (sin sensor de nivel/CO); R0 tiene ya el máximo de carbón y de caída de %Sn (−12 pts) del batch; +260 kg en 30 min sobre una escoria con 16 % Sn es el punto de mayor generación de CO | Techo P90 del orden; prohibido subir carbón si T previa < P10 del orden (baño frío) o si el operador marcó "espumado"; observación de la lanza/tiro; Fe en metal crudo por batch como KPI de parada |
| (ii) Menos GN → T baja, viscosidad, Sn atrapado | GN −0.5 Nm³/min P50 (P5 −3.3; −2 % … −13 %) en R1-R3 | θ_dT GN +2.9 °C/sd; ΔT rec−hist P5 −2.7 °C, P50 −0.4 °C; `pred_T_rec` < P5 en 5.1 % (hist 4.8 %) | Bajo en magnitud; pero la palanca GN→FeO "sólo se identifica en el último tercio" (§20.4): es la menos robusta | Piso de GN = P10 del orden; piso de T predicha = P10 del orden (no P5 global); no recortar GN si T previa está en el decil inferior |
| (iii) Más aire → oxidación de Sn a polvo, desgaste de lanza/refractario, T de gases | Aire +0.6 P50, P95 +5.7 … +25.7 Nm³/min; O₂ +0.3 | Sin θ significativo (§2); T gas pre-BHF Reducción P50 192 °C, **P99 198 °C** (Fusión 201 °C): el BHF opera ya en su techo; presión de punta ρ +0.22 con aire | Post-combustión y polvo (18 % del Sn ya va a polvo); mangas | **DESCARTAR** aire como palanca; T gas pre-BHF ≤ 195 °C como restricción dura si alguna vez se optimiza |
| (iv) Menos carbón/gas en Fusión → escoria fría al entrar en R | Carbón −3 kg/min P50 (P5 −13 … −23; 9-21 % por debajo del P5 del orden), −995 kg/batch P50 (−2 750 P95) de 13.6 t; GN −48 Nm³ | θ_dT carbón +0.9 °C/sd (n.s.): el modelo dice que el carbón de Fusión no calienta; `pred_T` en F6 rec−hist P50 0.0, P5 −2.5 °C | Físicamente el carbón que combustiona en el baño es fuente de calor: un recorte del 20 % sin efecto térmico es lo que el histórico no muestra porque el operador compensa con GN. Además Fe más oxidado (magnetita) → viscosidad | Recorte de carbón de Fusión ≤ 1 sd del orden (5-8 kg/min) y ≥ P10 del orden; observar T de F6 y viscosidad al sangrar; piloto |
| (v) Ventana térmica P5-P95 como salvaguarda | Penaliza 2 kg Sn/°C fuera de 942-1177 °C (R) / 959-1212 °C (F) | 8.7-8.9 % de las recomendaciones tienen `J_temperatura < 0` (heredado del estado); la penalización de una excursión de 10 °C (20 kg) es menor que el uplift típico (37 kg) | La ventana no es de seguridad: es el rango observado, aplicado a una predicción con R² 0.35-0.46 (MAE 7-9 °C) | Convertir en restricción dura (`T_pred − MAE ≥ P10 del orden`) y no en penalización blanda |

Variables de seguridad **no modeladas**: T de gases pre-BHF/pre-cuchilla (techo de mangas), presión de punta de lanza (P99 110 kPa en R: lanza
inmersa/obstruida), contrapresión de aire/GN, tiro (P1 55 %), nivel de baño y de escoria (espumado, no hay sensor), CO/CO₂ en gases, Fe en
metal crudo (hardhead), estado de la lanza (consumo), refractario (sólo el espesor normalizado como contexto). Ninguna entra en `J`; el
sistema debe correr **debajo** del sistema de enclavamientos, nunca sustituirlo. Veredicto **EN DUDA**.

## 4. Escalón F1

F1 nunca se modelaba (gradiente NaN); v7 lo incluye imputando 0. R² OOF de ΔT **por orden**: F1 0.07, F2 0.36, F3 0.48, F4 0.52, F5 0.51, F6 0.32.
El agotamiento de Fusión tiene R² OOF 0.19 y **lockbox −0.70** (no generaliza). Uplift en F1: 6.6 kg/escalón (mediana; el más bajo de Fusión),
71 % de los escalones cambian, recomendación idéntica a la del resto (carbón −3.0 kg/min, GN −0.2). En Fusión el uplift medio de 13.9 kg/escalón
se compone de **10.4 kg de costo de carbón (placeholder)** + 3.8 kg de conservación de Sn + 0 térmico: F1 no aporta información propia, sólo el
recorte genérico de carbón. **DESCARTAR F1 como escalón recomendable**: hereda la receta de Fusión (Nivel 0) y no se le atribuye uplift.

## 5. Reentrenamiento por campaña y deriva

- Evidencia de deriva ya en los datos: lockbox = fin de campaña de refractario; Fusión Sn lockbox −0.70; FeO por tercios 0.40/0.30/0.28; GN→FeO sólo
  en el último tercio; `espesor_ladrillo_norm_mm` (300 → 218 mm) es feature de estado en Reducción: en una campaña nueva vuelve a 300 y el HGB
  tratará los batches como los de abril (proxy de tiempo, no de física).
- Procedimiento: (1) reentrenamiento progresivo cada 10 batches (E9-04: ΔT de Fusión 0.04 → 0.16-0.32), θ con signos fijos y estado `ESTADO_R_V7`
  congelado; (2) w, K y β_F **no** se re-estiman en el piloto (son el objetivo, no el modelo); (3) cada reentrenamiento se valida OOF por batch y se
  compara con el anterior (Δ R² dentro de ±0.05, θ_Cx_sn dentro de su IC) antes de activarse; (4) al cambiar de campaña (refractario nuevo, cambio de
  concentrado o de reciclo) el sistema arranca en modo **sombra** (recomienda, no se muestra) hasta 30 batches con MAE ≤ 1.2× la de referencia.
- Indicadores de drift (ventana móvil de 30 batches; referencia DEV): MAE de ln_sn_dep R ≤ 0.30 (ref 0.247); sesgo medio del residuo por orden
  |b| ≤ 0.10; soporte P50 de los estados vivos ≥ 0.35 (ref 0.42); fracción de recomendaciones en la cota ≤ 30 % (ref R0 51 %: ya en alerta);
  fracción de escalones sin estado (leyes faltantes) ≤ 10 %; T de baño y %FeO final dentro de ±1 sd del histórico.
- Congelar/desactivar: 2 indicadores en alerta → congelar (sin reentrenar, sólo mostrar); 3 → desactivar (modo sombra); cualquier evento de seguridad
  (espumado, T gas > 200 °C, lanza) → desactivar y revisar. Reversión: la receta estándar de planta vigente se conserva como acción base en todo
  momento (ya es admisible por diseño), y cada recomendación guarda versión del modelo, fold/fecha de entrenamiento y hash de datos.

## 6. Cambio de gestión y piloto

- Presentación al operador: **rango, no punto** (θ ± IC95 % → Δ recomendado como [Δ_lo, Δ_hi], recortado a ±1 sd del orden), con la descomposición
  J_sn / J_feo / J_costo / J_T en kg Sn y una frase de justificación ("retener FeO: +0.002 ln → +25 kg Sn en este y el siguiente batch"). Mostrar
  también el soporte (densidad histórica) en tres colores y la T predicha con su MAE.
- Registro: por escalón, recomendación, valor aplicado, aceptado/rechazado/modificado, **código de razón** (espumado, lanza, baño frío, calidad de
  carbón, orden del supervisor), hora de toma y de reporte del ensayo. Sin códigos de razón el siguiente reentrenamiento aprende de los rechazos
  como si fueran decisiones de proceso.
- KPI de seguimiento por batch: recuperación refinada, Sn en escoria final (kg, mediana 736, sd 535), %FeO final (17.7), Fe en metal crudo, f_dross
  (12.6 %), f_polvo (18.1 %), T de baño en F6 y R3, T gas pre-BHF máx, consumo de carbón F/R, GN, O₂, aire, duración, eventos.
- Piloto A/B: batches alternos (par/impar) con la misma dotación, 2 palancas (carbón de Fusión F2-F6 acotado; carbón R0 ≥ mediana con techo P90 y
  GN R1-R3 con piso P10), aire y O₂ congelados, 120 batches (≈ 40 días a 3.0 batches/día). Potencia: sobre el KPI (sd 4.5 pp) detectar +0.3 pp exige
  **3 500 batches por brazo** (+1 pp: 317): el KPI **no puede** validar el caso en una campaña. Endpoints primarios factibles: consumo de carbón de
  Fusión (sd 794 kg → −995 kg detectable con 10/brazo), Sn en escoria final ajustado por covariables (≈ 50-110/brazo para 200-300 kg), ln_sn_dep de
  R0 ajustado (≈ 80/brazo). El KPI y f_polvo/f_dross quedan como secundarios de seguridad.
- Criterios de parada: un evento de espumado/slopping o T gas > 200 °C atribuible; Fe en metal crudo fuera de especificación en 3 batches tratados;
  f_dross del brazo tratado > control + 2 pp sobre 20 batches; KPI tratado < control − 1 sd sostenido 10 batches; rechazo del operador > 50 %.

## 7. Rentabilidad

| Concepto | Valor | Base |
|---|---|---|
| Uplift modelado por batch | +0.27 pp DEV / +0.35 lockbox = 137 / 179 kg Sn eq | mediana `uplift_batch_kg`; "> 99 % positivos" es **por construcción** (la histórica es admisible) |
| Descomposición (mediana por batch) | J_sn + J_feo +146 kg; **J_costo +53 kg (39 %)** | costos placeholder: C 0.05, GN 0.03, O₂ 0.01 kg Sn/unidad, aire 0 |
| Consumos rec − hist por batch | carbón **−1 037 kg** (F −995, R −32), GN −82 Nm³, O₂ −19, aire +126 Nm³ | sobre 16.5 t C, 11.4 kNm³ GN, 16.4 kNm³ O₂ por batch |
| Placeholder implícito (Sn a 30 kUSD/t) | carbón 1 500 USD/t, GN 0.9 USD/Nm³, O₂ 0.3 USD/Nm³ | ≈ 4-6× el precio probable del carbón, 2-3× el del GN |
| Parte de Reducción tardía (75 % de los escalones R) | uplift ≈ 40 kg/escalón, del cual J_feo +45..+51 y J_sn −4..−12 | apoyada en θ de FeO **no significativos** × K·w = 14 187 kg/unidad |
| Parte defendible (θ significativos: Cx_sn en R0, conservación en F) | ≈ 30 kg (R0) + 6 × 3.8 (F) ≈ **50-60 kg Sn/batch** | orden de magnitud, no estimación |
| Anual (3.04 batches/día → ~1 100 batches/año; 362 por campaña de 119 días) | bruto 50-65 t Sn/campaña ≈ 1.5-1.9 MUSD (150-200 t/año); defendible ≈ 20 t/campaña ≈ 0.6 MUSD; carbón −375 t/campaña ≈ 0.1 MUSD a 300 USD/t | Sn a 30 kUSD/t (placeholder) |

Precios/costos a pedir a planta para cerrar el caso: (1) precio neto realizado del Sn (LME − TC/RC − pagables) y valor del Sn en dross y en polvo
(reproceso); (2) carbón: precio puesto en planta y carbono fijo; (3) GN: tarifa USD/Nm³ (o por MMBtu con PCS); (4) O₂: costo marginal USD/Nm³ (VPSA/
criogénico, contrato take-or-pay); (5) aire: kWh/Nm³ del compresor × tarifa; (6) consumo y costo de lanza y refractario por campaña (sensibles a aire/
O₂); (7) valor del tiempo de horno (USD/h) si alguna palanca alarga el batch; (8) costo de las mermas: hardhead, escoria final con Sn (736 kg/batch).
Con el placeholder de carbón corregido a su precio real, el recorte de carbón de Fusión (3/4 del uplift de Fusión) deja de pagar y la política de
Fusión debe re-optimizarse antes del piloto. Veredicto **EN DUDA**.

## 8. Resumen de salvaguardas para el piloto (checklist)

1. Límites de búsqueda P5-P95 **por orden**; |Δ| ≤ 1 sd del orden; R0 carbón ≤ P90 del orden y nunca "en la cota".
2. Aire y O₂ congelados al plan; costo del aire > 0 y θ significativo antes de reincorporarlos.
3. Sin recomendación si: soporte del estado < P10, leyes de t−1 faltantes, T previa < P10 del orden, o código de razón activo (espumado, lanza).
4. Restricciones duras: T_pred − MAE ≥ P10 del orden; T gas pre-BHF ≤ 195 °C; sistema subordinado a enclavamientos.
5. IC real (bootstrap activo) mostrado como rango; recalcular `PESOS_V5_DEFAULT` con precios de planta.
6. Arquitectura de 2 niveles (receta de Fusión + plan de Reducción al inicio de R, re-plan tras R0); registrar hora de reporte del ensayo.
7. Reentrenamiento progresivo cada 10 batches con validación previa; 6 indicadores de drift; modo sombra 30 batches al cambiar de campaña.
8. Piloto A/B de 120 batches con endpoints de escalón y consumo (no el KPI), códigos de razón y criterios de parada explícitos.
