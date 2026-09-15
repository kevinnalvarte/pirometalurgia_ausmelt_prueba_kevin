# search_targets — ¿qué target por escalón debe maximizar el recomendador? (iteración 5, 2026-09-14)

Familia de experimentos que evalúa **36 variables objetivo candidatas por escalón** (20 de Reducción, 16 de Fusión), todas
calculables desde teoría pirometalúrgica, con una regla de agregación escalón→batch coherente, contra el KPI
`rendimiento_proxy_batch` = Sn a metal / (metal + dross + polvo). Diseño en `SEARCH_TARGETS_diseno.md`; librería `st_targets.py`;
ejecución paralela (ST-01..04, Sonnet 5) y síntesis/hipótesis/decisión (ST-05..07, Fable 5.1).

## 0. Respuesta corta

| Fase | Target ganador por escalón | Agregación batch | ρ Spearman vs rendimiento (DEV, n=286-298) | lockbox (n=63) | OLS con controles (pp de rendimiento por +1 sd) |
|---|---|---|---|---|---|
| **Reducción** | **SDI** = `ln(Sn_inv[t−1]/Sn_inv[t]) + 10·ln(FeO_inv[t]/FeO_inv[t−1])` (agotamiento log del Sn + 10 × retención log del FeO) | suma telescópica = `ln(Sn_ini/Sn_fin) + 10·ln(FeO_fin/FeO_ini)` | **+0.319** [IC95 % 0.21, 0.43], p = 3·10⁻⁸ (BH sobre 37 tests: p < 10⁻⁵); positivo en los 3 tercios cronológicos (+0.19 / +0.35 / +0.48) | +0.09 (n.s.) sobre el propio batch; **+0.35 (p = 0.005) sobre el batch siguiente** (heel de escoria); +0.33 (p = 0.010) sobre la media de ambos | **+1.55** [0.9, 2.2] p < 0.001 (DEV); +1.42 (total) |
| **Fusión** | **IRF al cierre del escalón** = `%FeO / (%SiO₂ + %Al₂O₃ + %CaO)` (índice de planta; conservar la matriz FeO-rica) | último valor de la fase (estado terminal) | **+0.195** [0.09, 0.30], p = 0.0007 (BH: 0.006); tercios +0.16 / +0.17 / +0.31 | **+0.33 (p = 0.008)** | **+1.21** p < 0.001 (DEV); +1.35 p = 0.055 (lockbox) |
| (referencia v4) | Reducción `Sn_ext − 0.6·FeO_ext` [kg] | suma | +0.035 (p = 0.55) | +0.04 | +0.07 (n.s.) |
| (referencia v4) | Fusión `−Sn_ext` [kg] | suma | +0.063 (p = 0.28) | +0.19 | +0.11 (n.s.) |

Los dos ganadores son los **únicos** candidatos que cumplen a la vez: significancia tras BH en DEV, coeficiente positivo y
significativo con controles de contexto y tendencia de campaña, dosis-respuesta monótona (SDI: 5/5 quintiles crecientes,
+4.4 pp entre Q5 y Q1; IRF: +2.0 pp), y correlación negativa con ambos canales de pérdida (SDI: dross −0.20, polvo −0.18;
IRF: polvo −0.21). Frente al techo del KPI (ruido blanco entre batches: autocorrelación 0.12, controles + tendencia R² 0.025),
ρ ≈ 0.3 es una señal fuerte: el SDI solo explica más varianza del KPI (R² adj 0.10-0.17) que cualquier set de agregados de
las iteraciones anteriores (0.11-0.18 con decenas de features, hallazgos §14.6).

## 1. Cómo se buscó (resumen del método)

1. **Registro de candidatos** (`st_registro.csv`): fórmula, unidad, dirección teórica (+1 maximizar / −1 minimizar / 0
   exploratorio), regla de agregación y referencia teórica. Familias: extracción de Sn, sobre-reducción (FeO), selectividad,
   metal-equivalente (Sn − λ·FeO), conservación (Fusión), matriz de escoria, térmico, polvo.
2. **Agregación coherente** (Guía 10.8): masas → suma (telescópica para Δinventario); fracciones de 1er orden → supervivencia
   `1 − Π(1 − f)` o retención `Π r = fin/ini`; constantes cinéticas → `−ln(fin/ini)/Σdt`; cocientes → cociente de sumas;
   intensivas → media ponderada por tiempo; estados terminales → último valor. ST-04 verificó que la regla coherente no es
   peor que reglas incoherentes (media de cocientes, máximo, etc.).
3. **ST-01** correlación batch (Spearman/Pearson, IC bootstrap, BH por fase, OLS HC3 con controles, dosis-respuesta,
   barrido de λ, combinación F+R, redundancia). **ST-02** predictibilidad por escalón (HGB OOF GroupKFold por batch, estado
   solo vs estado + CONTROL, techo de ruido del trazador). **ST-03** PLM v4 (θ de carbón/GN/O₂/aire con IC) y cadena de
   signos vs exp 19. **ST-04** robustez (agregaciones alternativas, tercios cronológicos, Monte Carlo del ruido de ensayo,
   bootstrap del ranking). **ST-05** hipótesis de 2ª ronda (índice compuesto log, origen del IRF, heel). **ST-06** métricas
   finales de los ganadores. **ST-07** PLM y predictibilidad de los ganadores.
4. **Criterio**: primario ρ(DEV) > 0 significativo tras BH + signo en lockbox + robusto a controles; secundario predictible
   por escalón y con palancas identificadas; terciario parsimonia/estabilidad. El lockbox nunca se usó para elegir (el peso
   w del SDI se fijó con la regla "menor w con ρ_DEV ≥ 0.95·máx" sobre DEV).

## 2. Qué se encontró (por familia)

**Reducción.** La familia *sobre-reducción* domina en DEV: fracción de FeO retenida R08 (ρ +0.31, p_BH 1.5·10⁻⁶), FeO reducido
en kg R02 (+0.27), Fe metalizado por Sn cargado R20 (+0.22); son redundantes entre sí (ρ 0.88). La familia *extracción de Sn*
(Sn extraído kg, fracción, k aparente, Δavance, Sn residual) tiene ρ ≈ 0 o negativo en DEV: **extraer más Sn en Reducción no
sube el KPI** (el Sn extraído total es ≈ fijo; lo que varía es cuánto va a dross/polvo). Por eso la familia metal-equivalente
`Sn − λ·FeO` sólo correlaciona cuando λ ≫ 0.6 (λ = 2: ρ 0.17; el barrido no encuentra máximo interior) y el objetivo v4
(λ = 0.6) da ρ 0.035. La *selectividad* como cociente de masas (Sn/FeO) es ruidosa (ρ 0.02): el cociente hereda el ruido de
dos diferencias de trazador. En el lockbox (agosto, fin de campaña, horno ≈80 °C más frío, Reducción fija en 88 min) el patrón
cambia: el FeO retenido pierde la asociación con el propio batch (−0.06) y la gana el agotamiento de Sn (k aparente +0.35,
fracción +0.31; su sd cae de 0.067 a 0.018: son unos pocos batches fríos que ni agotan ni rinden). Ambas componentes están
en trade-off (ρ −0.38 entre batches; −0.56 entre escalones).

**El índice compuesto SDI resuelve el trade-off en escala logarítmica.** `ln(Sn_ini/Sn_fin)` y `ln(FeO_fin/FeO_ini)` son
adimensionales, la masa de escoria del trazador se cancela en cada cociente (sólo quedan las leyes y el cociente de %CaO), y la
suma por escalón es exactamente telescópica. El barrido de w (`st_06_w_seleccion.csv`) muestra que ρ_DEV crece con w y satura
en 0.33 (w ≥ 12); la regla de parsimonia sobre DEV da **w* = 10** (ρ 0.319). Con w = 10 el índice es positivo en los tres
tercios de DEV y en el lockbox sobre el batch siguiente; con w ≤ 4 sería positivo también sobre el propio batch del lockbox
(+0.24, p = 0.06) a costa de DEV (0.23). La componente de agotamiento (peso 1) evita el objetivo degenerado "no reducir".

**Heel químico de escoria (hallazgo nuevo, ST-05).** El IRF al cierre del primer escalón (F0) correlaciona ρ 0.46 (p 10⁻²⁰) con
el IRF final de la Reducción del batch anterior (SiO₂: 0.47; Al₂O₃: 0.32); no hay heel de Sn (hallazgos §14.1) pero **sí de
matriz de escoria**. Consecuencia: el target de Reducción tiene dos canales hacia el KPI: (i) propio batch (menos Fe
metalizado → menos dross: λ_Fe, exp 15/19) y (ii) batch siguiente (escoria final FeO-rica y pobre en formadores de red →
mejor matriz inicial): ρ(SDI_b, KPI_{b+1}) = +0.11 (DEV, p 0.06), **+0.35 (lockbox, p 0.005)**, +0.15 (total, p 0.005); OLS
con controles del batch siguiente y el KPI propio: +1.22 pp/sd (lockbox, p 0.035), +0.56 (total, p 0.047). Sobre la media de
los dos KPIs: ρ 0.28 (DEV), 0.33 (lockbox), 0.28 (total, p 10⁻⁷).

**Fusión.** Sólo la familia *matriz de escoria* correlaciona: IRF al fin de Fusión (F10, +0.19 / +0.33), %FeO final (+0.12 /
+0.18), FeO retenido por Fe cargado (+0.11 / +0.25). Las familias *conservación de Sn* (−Sn extraído, fracción retenida:
ρ 0.06), *térmico* y *polvo* no correlacionan en DEV. El IRF correlaciona con el KPI ya al cierre de F0 (ρ 0.32 DEV / 0.57
lockbox) y la correlación **decrece** a lo largo de la Fusión (F1 0.38 → F6 0.19): la señal es en gran parte el **estado
inicial** de la escoria (heel + primera carga); el cambio de IRF durante la Fusión no aporta información adicional una vez
controlado el IRF de F0 (OLS: ΔIRF −0.34 pp/sd, p 0.60) y ningún otro target de Fusión sobrevive a ese control
(`st_05_fusion_control_irfF0.csv`). Determinantes operativos del IRF final (OLS, R² 0.28): GN/carga (−), temperatura (−),
dross de Fe cargado (−), carga secundaria (+), mineral de Fe (+); cal y C/Sn no significativos. Lectura: en Fusión el
recomendador debe **conservar** la matriz FeO-rica (no reducir FeO con GN/temperatura alta), y la palanca más fuerte sobre el
estado inicial de la Fusión es la escoria final de la Reducción anterior (es decir, el target de Reducción).

## 3. Defensa cualitativa (pirometalurgia)

- **SDI (Reducción).** `SnO₂ + 2C → Sn + 2CO` compite con `FeO + C → Fe + CO` (Guía §3.3, 5.2-5.3). El Fe metalizado forma
  hardhead/dross que ocluye Sn (canal dross; λ_Fe 0.30-0.60 kg Sn/kg FeO) y el fume sube con la severidad (canal polvo).
  Cerca del final (avance > 0.96) la selectividad marginal FeO/Sn se degrada 10× (exp 15). El SDI premia agotar Sn
  (fuerza impulsora, cinética de 1er orden en masa: `ln(Sn_ini/Sn_fin) = k·t`) y penaliza 10× la pérdida logarítmica de
  FeO. Es la versión adimensional y aditiva del criterio de corte de Reducción (Guía 11.3): seguir reduciendo sólo mientras
  el Sn cae más rápido de lo que cae el FeO. Además la escoria final rica en FeO y pobre en SiO₂/Al₂O₃ es un heel de
  matriz fluida (fayalítica) para el batch siguiente (§7.4, 8): menor viscosidad/liquidus → mejor coalescencia y
  sedimentación de gotas de Sn (§4.4-4.5).
- **IRF (Fusión).** FeO/(SiO₂+Al₂O₃+CaO) es la razón modificador/formadores: un IRF alto es una escoria fayalítica fluida
  (§7.2-7.4, 5.5) con reserva de FeO que tampona el potencial de oxígeno y protege de la sobre-reducción posterior; un IRF
  bajo (SiO₂/Al₂O₃ altos) es viscoso y de alto liquidus, atrapa Sn y favorece arrastre (ρ con polvo −0.21). Conservarlo en
  Fusión equivale al principio de dos etapas (fundir oxidando, reducir después): no consumir FeO ni Sn con GN/carbón en
  Fusión, mantener la temperatura acotada.
- **Coherencia con lo ya validado**: λ_Fe (exp 15/19), λ_F (14.6), selectividad por avance (exp 15), "cal no es palanca"
  (exp 18) y "GN no selectivo" (v4) son casos particulares de estos dos targets.

## 4. Predictibilidad por escalón y palancas (ST-02, ST-03, ST-07)

HGB OOF GroupKFold(5) por batch en DEV (lockbox entrenando con DEV); A = STATE+CONTEXT v3, B = A + CONTROL (carbón, GN, O₂,
aire, exceso O₂, Cx), C = sólo CONTROL. PLM v4: θ por +1 sd de la palanca, IC95 % bootstrap por batch (300).

| Target (escalón) | n DEV | sd | R² OOF A | R² OOF B | R² lockbox B | θ carbón | θ GN | θ O₂ | θ aire | Lectura |
|---|---|---|---|---|---|---|---|---|---|---|
| **SDI** (Reducción) | 1015 | 0.76 | 0.262 | 0.251 | 0.242 | +0.084 [−0.11, +0.25] | **−0.153 [−0.25, −0.05]** | +0.059 [−0.05, +0.18] | +0.031 [−0.01, +0.09] | GN (reductor no selectivo) baja el SDI; carbón/O₂/aire con signo teórico, IC cruza 0 |
| `ln_feo_ret` (componente FeO) | 1015 | 0.091 | 0.267 | 0.267 | 0.255 | +0.001 | **−0.023 [−0.035, −0.012]** | +0.007 | +0.001 | el FeO retenido sólo responde al GN; el carbón es selectivo (≈0 sobre FeO) |
| `ln_sn_dep` (componente Sn) | 1015 | 0.66 | 0.743 | 0.746 | 0.810 | **+0.089 [0.00, +0.18]** | **+0.059 [+0.01, +0.11]** | −0.010 | +0.002 | agotamiento de Sn: carbón (+, `Cx_sn` +0.17*) y GN (+) |
| **IRF** al cierre (Fusión) | 1769 | 0.040 | 0.709 | 0.714 | 0.596 | +0.001 | −0.001 | +0.002 | +0.000 | el IRF lo fija el estado (IRF previo, carga); las palancas de lanza no lo mueven en el rango histórico |
| ΔIRF (Fusión) | 1768 | 0.023 | 0.368 | 0.374 | 0.218 | +0.001 | −0.001 [−0.003, 0.000] | +0.001 | +0.001 | GN borderline negativo (consume FeO), resto ≈0 |

Techo de ruido del trazador (ST-02, Monte Carlo de ±2 % en %CaO, ±0.05 pt en %Sn, ±0.3 pt en %FeO): las componentes de FeO
tienen R² máximo ≈ 0.82-0.84 (sd de ruido 380 kg de FeO por escalón, 0.034 en la fracción retenida) y exp 15 estimó 0.68 con
otro método; R² OOF ≈ 0.27 es por tanto un tercio-mitad de lo alcanzable: la parte no explicada es en buena medida química
no capturada por el estado (mezcla, dosis local de carbón), no sólo ruido. El SDI hereda ese ruido. Las 36 candidatas (`st_02_predictibilidad.csv`): la extracción de Sn en kg es "trivial" (R² 0.97, regla inventario ×
fracción); las de FeO 0.26-0.36; las térmicas de gases (T pre-BHF) son las únicas que ganan mucho R² con las palancas (+0.33
a +0.39) porque miden la lanza, no el baño.

ST-03 (PLM sobre los 36 candidatos, `st_03_theta.csv`, 72 ajustes sin fallos): la componente FeO en kg (R02) responde a GN
−242 kg FeO/sd [−370, −145]* y a O₂ +164 [+54, +309]* (lanza oxidante retiene FeO), carbón −92 [−319, +81] (n.s.); la fracción
retenida (R08) sólo al GN (−0.019/sd*, rel. −0.22); en Fusión ni el IRF ni el %FeO final responden a las palancas (θ_rel ≤ 0.06),
salvo carbón +0.09 pt de %FeO (signo contrario al teórico, borderline). Cadena de signos (`st_03_resumen_cadena.csv`): el GN
es la única palanca con θ significativo y evidencia de batch (exp 19 GN_R −1.04 pp/sd, p 0.03) y su cadena es **coherente**
para R08, R02, R20 y R16 (más GN → menos FeO retenido → menos rendimiento); el carbón queda indeterminado en todos (θ con IC
que cruza 0, exp 19 sobre rendimiento n.s.); en Fusión todas las cadenas son indeterminadas.

**Cadena de signos (ST-03/ST-07 vs exp 19).** Para que "maximizar el target" empuje las palancas hacia el KPI hace falta
sign(θ_palanca→target) = sign(palanca→KPI de batch). En Reducción, exp 19 encontró GN total +f_dross (p 0.04) / −f_metal
(p 0.03) y carbón total −f_dross (p 0.03): el SDI reproduce ambas direcciones (GN −0.15*, carbón +0.08). El objetivo v4
`Sn − 0.6·FeO` tenía al GN con efecto neto ≈ +30 kg/sd (subía el GN) — el SDI corrige esa incoherencia sin cambiar λ a mano.
En Fusión, exp 19 halló C/Sn −f_dross (p 0.011) y T media +f_polvo (p 0.027); el IRF no responde a esas palancas en el
escalón (θ ≈ 0), así que la cadena es indeterminada: el target de Fusión se defiende por su asociación con el KPI y por la
teoría, no por palancas identificadas (ver §6).

## 5. Robustez (ST-04)

| Prueba | Familia FeO-retenido (R08 / R02 / R20) | IRF de Fusión (F10) |
|---|---|---|
| Agregación coherente vs mejor regla (DEV) | brecha 0.027 / 0.005 / 0.013: la regla coherente (producto = fin/ini; suma) es ≈ la mejor | la regla "último valor" (0.195) es peor que "primer valor" (IRF al cierre de F1: 0.38): brecha 0.19 |
| Tercios cronológicos de DEV + lockbox | 3/4 bloques positivos (DEV +0.29/+0.32/+0.37; lockbox −0.06) | **4/4 positivos** (mín. +0.16; lockbox +0.33); ρ con el KPI residualizado por controles + tendencia +0.20 |
| Monte Carlo del ruido de ensayo (200 réplicas; %CaO ±2 %, %Sn ±0.05, %FeO ±0.3) | R08: ρ 0.291 ± 0.020 (P5-P95 0.26-0.33), significativo en el 100 % de las réplicas; R02 0.240 ± 0.019 (100 %); R20 0.142 ± 0.021 (92 %) | ρ 0.189 ± 0.010 (0.17-0.20), significativo en el 100 % |
| Bootstrap del ranking dentro de la fase (500) | R08 rango mediano 1, P(top-3) = 1.00; R02 rango 2 (1.00); R20 rango 3 (0.85) | rango mediano 1, P(top-3) = 0.93 |

Veredicto de ST-04: en Reducción sólo los tres de la familia FeO-retenido son robustos (los 17 restantes se desvanecen con ruido
o cambian de signo entre bloques); en Fusión el IRF es el más estable (signo en 4/4 bloques, ranking 1, insensible al ruido)
pero su agregación "estado terminal" no es la más informativa: el IRF **temprano** (F0-F1) correlaciona más (0.32-0.38) que el
final (0.19) — es la misma observación de ST-05 (la señal es el estado inicial de la escoria). El SDI (w = 10) hereda la
robustez de R08 (ρ_MC y ranking idénticos en la práctica, ya que la componente FeO domina) y añade la componente de Sn, que
es la que aporta la señal en el lockbox.

## 6. Límites honestos

- ρ ≈ 0.3 es "fuerte" sólo relativo al techo del KPI (que es casi ruido blanco entre batches); el SDI explica ~10 % de la
  varianza del KPI. No es un experimento: la evidencia es observacional con controles.
- El SDI sobre el propio batch no replica en el lockbox (fin de campaña); replica sobre el batch siguiente. Reentrenar
  por campaña y validar con piloto A/B siguen siendo necesarios.
- w = 10 se eligió con DEV y una regla de parsimonia; el óptimo empírico está en 12-16 y el rango 4-16 da ρ_DEV 0.23-0.33.
- En Fusión, la asociación del IRF con el KPI es mayormente de estado inicial (heel + primera carga); el valor operativo del
  target de Fusión es conservar ese estado, y el cambio de IRF dentro de la Fusión no tiene evidencia propia.
- Los targets de trazador heredan su ruido (R² OOF ≈ 0.3 para las componentes de FeO; ST-02/ST-04).

## 7. Archivos

| Script | Salidas |
|---|---|
| `st_targets.py` | `st_registro.csv`, `st_batch.csv`, `cache_df_st.pkl` |
| `st_01_correlacion_batch.py` | `st_01_ranking.csv`, `st_01_correlaciones.csv`, `st_01_ols_controles.csv`, `st_01_dosis_respuesta.csv`, `st_01_lambda_sweep.csv`, `st_01_combinacion_FR.csv`, `st_01_redundancia.csv` |
| `st_02_predictibilidad_escalon.py` | `st_02_predictibilidad.csv`, `st_02_techo_ruido.csv`, `st_02_importancia.csv` |
| `st_03_identificacion_control.py` | `st_03_theta.csv`, `st_03_cadena_signos.csv`, `st_03_resumen_cadena.csv` |
| `st_04_robustez.py` | `st_04_agregaciones.csv`, `st_04_estabilidad_temporal.csv`, `st_04_ruido.csv`, `st_04_ranking_bootstrap.csv` |
| `st_05_hipotesis_ronda2.py` (+ inline) | `st_05_sdi_sweep.csv`, `st_05_irf_controles.csv`, `st_05_irf_F0_origen.csv`, `st_05_efecto_batch_siguiente.csv`, `st_05_fusion_control_irfF0.csv` |
| `st_06_ganador.py` | `st_06_metricas_ganadores.csv`, `st_06_w_seleccion.csv`, `st_06_efecto_siguiente_batch.csv`, `st_06_targets_escalon.csv` |
| `st_07_plm_ganadores.py` | `st_07_theta_ganadores.csv`, `st_07_predictibilidad_ganadores.csv` |

Figuras en `figs/`. Logs `st_0N_log.txt`.

---

# Ronda 2 (2026-09-14): ¿existe un target con ρ o R² > 0.70? Techo del KPI y mejor target alcanzable

Diseño en `SEARCH_TARGETS_ronda2.md`. Experimentos: ST-08 techo/fiabilidad del KPI (Fable), ST-09 familia 2 de 30 candidatos
teóricos (Sonnet), ST-10 compuesto supervisado con validación cruzada (Sonnet), ST-11 estructura del KPI y canales (Sonnet),
ST-12 formalización del mejor hallazgo (Fable).

## R2.1 Respuesta corta

**El umbral 0.70 no es alcanzable con este KPI y estos datos, y no por falta de candidatos: es un límite del propio KPI.**
Tres cotas independientes lo muestran:

| Cota | Evidencia | Máximo alcanzable |
|---|---|---|
| Fiabilidad del KPI (ST-08) | dos medidas del mismo rendimiento (proxy = M/(M+D+P) y recuperación real = M/Sn cargado) concuerdan sólo r = 0.57; el proxy correlaciona −0.37 con el cierre del balance de Sn y sólo +0.50 con el metal (−0.74 con dross, −0.60 con polvo) | ρ_max ≈ √0.57 ≈ 0.75 incluso para un predictor perfecto del rendimiento "verdadero" |
| Estructura del KPI (ST-11) | `rendimiento ≡ 100·(1 − f_dross − f_polvo)`; el polvo explica el 42 % de la varianza y es impredecible desde los escalones (R² OOF ≤ 0.09, lockbox ≈ 0); dross y polvo muestran autocorrelación anómala a lag 3 (0.38 y 0.48) sugestiva de atribución periódica | sólo el canal dross (58 %) es potencialmente explicable → R² ≤ ~0.5 |
| Techo empírico (ST-10, ST-08b) | combinando los 38 agregados teóricos + heel + controles con Ridge/Lasso/HGB en CV 5×5: R² OOF 0.20, ρ 0.46; walk-forward cronológico R² ≈ 0. Con **toda** la información de escalón (2003 agregados: media/primero/último/suma de cada columna por fase + heel; HGB, CV 5×2): R² OOF 0.16, ρ 0.44 (rendimiento); 0.25 / 0.50 (f_dross); ≈0 (f_polvo); lockbox R² −0.10, ρ 0.40; walk-forward 0.07 (`st_08b_techo_modelos.csv`; Ridge con 2003 features diverge y sólo conserva ρ 0.47) | ρ ≈ 0.45-0.50 con toda la información; R² ≈ 0.16-0.25 |

## R2.2 Mejor target encontrado en la ronda 2: ln IRF durante la Reducción (`R23_lnirf_R`)

```
lnIRF_R[t] = ln( %FeO[t] / (%SiO2[t] + %Al2O3[t] + %CaO[t]) )   al cierre de cada escalón de Reducción
agregación = media ponderada por el tiempo del escalón (estado intensivo, Guía 10.8)
```

| Métrica | DEV (n=299) | tercios DEV | lockbox (n=63) | total (n=362) |
|---|---|---|---|---|
| ρ Spearman vs rendimiento | **+0.304** [0.19, 0.41], p = 8·10⁻⁸ | +0.33 / +0.26 / +0.43 | **+0.380**, p = 0.002 | **+0.331**, p = 1·10⁻¹⁰ |
| OLS HC3 por sd con controles + tendencia | +1.83 pp (p < 0.001) | | +1.49 pp (p = 0.024) | |
| batch siguiente (heel) | | | ρ +0.42 | |
| canales | dross −0.14, polvo −0.24 | | | |
| dosis-respuesta (quintiles DEV) | 4/4 crecientes, Q5 − Q1 = +3.9 pp | | | |
| Monte Carlo ruido de ensayo (50 rep) | ρ 0.303 ± 0.006 | | | |
| Palancas (PLM v4, escalón, θ por sd / sd target) | GN **−0.24** [−0.37, −0.13]; O₂ **+0.19** [+0.04, +0.38]; carbón −0.02 (n.s.); aire +0.03 | | | |

Descomposición: `lnIRF_R_twmean ≈ 0.91·lnIRF_finFusión + 0.28·ln(FeO_fin/FeO_ini)` (R² 0.81): el target une el **estado heredado**
(matriz de escoria al fin de Fusión, que a su vez viene del heel del batch anterior) con la **acción del batch** (retención de
FeO, la componente del SDI). Por eso replica en los dos regímenes: en DEV pesa la acción, en el lockbox el estado.

Comparación con los ganadores de la ronda 1 (misma batería, `st_12_metricas.csv`):

| Target | ρ DEV | ρ lockbox | ρ total | OLS lockbox | ρ batch siguiente (lockbox) |
|---|---|---|---|---|---|
| SDI (ronda 1) | +0.319 | +0.094 (n.s.) | +0.280 | +0.26 (n.s.) | +0.35 |
| ln FeO retenido | +0.315 | −0.061 | +0.244 | −0.29 | +0.37 |
| IRF fin Fusión (ronda 1) | +0.195 | +0.331 | +0.225 | +1.32 (p 0.06) | +0.33 |
| **ln IRF en Reducción (twmean)** | **+0.304** | **+0.380** | **+0.331** | **+1.49 (p 0.02)** | **+0.42** |
| z(SDI) + z(lnIRF_R) | **+0.410** [0.30, 0.51] | +0.256 (p 0.04) | **+0.399**, p 10⁻¹⁴ | +1.04 (p 0.07) | +0.45 |

Fundamento: durante la Reducción no entra carga, así que el IRF sólo puede caer por reducción de FeO a Fe⁰ (dross) o subir
por retención; un IRF alto sostenido a lo largo de la fase significa una escoria fayalítica fluida (coalescencia y sedimentación
de gotas de Sn, Guía §4.4-4.5, 5.5) y una reserva de FeO que mantiene la ventana selectiva (§3.3) y deja un heel FeO-rico al
batch siguiente. Como target de recomendador: "mantener alto el IRF en cada escalón de Reducción" = no reducir FeO (menos GN,
lanza algo más oxidante), exactamente la dirección de la evidencia de batch (exp 19: GN total de Reducción sube dross y baja
metal).

## R2.3 Familia 2 (ST-09) y compuesto supervisado (ST-10): qué más se probó

- Familia 2 (30 candidatos: balances de Fe y Sn, NBO/T, FeO/SiO₂, Al₂O₃/modificadores, sobrecalentamiento, T gases, intensidad
  de fume Arrhenius, gas por carga, tiro, compuestos SDI + IRF): sólo la familia *matriz durante la Reducción* replica en
  lockbox (IRF twmean 0.30/0.38; índice de fluidez IRF − k·Al₂O₃ 0.30/0.39; FeO/SiO₂ 0.29/0.36; IRF al inicio de R 0.24/0.38).
  Balances de Fe en Fusión ρ ≤ 0.13; metal-equivalente normalizado ≤ 0.12; térmicos y polvo ≈ 0. Ver `st_09_ranking.csv`.
- Compuesto supervisado (ST-10, ≤ 6 términos elegidos por CV): `2.0·z(IRF F0) + 0.93·z(−FeO reducido) + 0.82·z(SDI) +
  0.72·z(log Sn/FeO fin F) − 0.57·z(Δlog Sn/FeO F) − 0.41·z(ΔT F)` → R² OOF 0.21 ± 0.01, ρ 0.46, lockbox R² 0.11 / ρ 0.35. El
  término dominante es el estado inicial (heel); los dos últimos no son significativos.
- Canal polvo (ST-11, 12 candidatos): ninguno sobrevive BH contra f_polvo; los más honestos (T del baño, intensidad de fume)
  dan ρ ≈ −0.15/−0.20 con f_polvo en DEV y lockbox. Canal dross: SDI, IRF final y %FeO final de Reducción (ρ 0.31-0.32).

## R2.4 Qué se recomienda con esta evidencia

1. Target de Reducción para el recomendador: **ln IRF al cierre del escalón** (o, equivalentemente en dirección, el SDI): ambos
   ordenan las palancas igual (menos GN, carbón selectivo, lanza no reductora). Para correlación de batch, `lnIRF_R` es el más
   robusto entre regímenes; el SDI es el más "de acción" (no depende del estado heredado). La suma estandarizada de ambos es el
   mejor correlato encontrado (ρ 0.41 DEV / 0.40 total, p 10⁻¹⁴).
2. Fusión: conservar el IRF (ronda 1); su valor está sobre todo en el estado con que arranca la Reducción.
3. Para superar ρ 0.5-0.7 hace falta cambiar el **KPI**, no el target: medir el Sn en dross y polvo por batch con una
   atribución consistente (o usar la recuperación real, que es más predecible fuera de muestra: ρ lockbox 0.50 en ST-10), y
   validar con piloto. Con el proxy actual, ρ ≈ 0.3-0.4 es el máximo defendible para un target teórico de un solo término.

---

# Ronda 3 (2026-09-14): aplicar la recomendación — cambiar el KPI

Recomendación de la ronda 2: cambiar el KPI, no el target. Dos vías con los datos existentes: (a) recuperación real
(metal / Sn cargado) como objetivo; (b) corregir la atribución de dross y polvo por batch. ST-13 (Fable) y ST-15 (Sonnet).

## R3.1 ¿Se puede corregir la atribución de dross y polvo con los datos? No.

Correlación cruzada ρ(salida[b], señal de proceso[b−k]) para k = −6..+6 (`st_13_crosscorr.csv`): el dross D correlaciona con el
Sn cargado y el Fe cargado al máximo en k = 0 (0.26 y 0.22) y con el FeO reducido en k = 0/+1 (0.17/0.25), es decir, **D sí está
atribuido al batch correcto**, sólo que débilmente ligado al proceso; el polvo P no correlaciona con ninguna señal a ningún
lag (|ρ| ≤ 0.19) y su autocorrelación a lag 3 (0.48) no coincide con ninguna señal de proceso: es una periodicidad propia del
registro de polvo (recolección/pesaje por ciclos), no un retardo corregible. Desplazar D y P (k = 1, 2, 3) o agrupar en
ventanas de 3-5 batches no produce un KPI con el que los targets correlacionen más en DEV (ventana 3: SDI 0.22, lnIRF_R 0.23;
desplazamiento 1: 0.15-0.21), y las ventanas inducen autocorrelación artificial (0.80-0.85 a lag 1). Sin datos de planta sobre
cómo se pesa y asigna el polvo, no hay atribución alternativa defendible.

## R3.2 Recuperación real como KPI (K1 = 100·M/Sn cargado)

| Target | ρ DEV (n 299) | ρ lockbox (n 63) | ρ total (n 362) |
|---|---|---|---|
| IRF fin de Fusión (`F10`) | +0.278 (p_BH < 0.001) | +0.353 (p 0.005) | +0.300 (p < 10⁻⁸) |
| **ln IRF en Reducción (`R23`)** | +0.262 (p_BH < 0.001) | **+0.497 (p < 10⁻⁴)** | **+0.317 (p < 10⁻⁹)** |
| −Sn extraído en Fusión (`F01`, conservación λ_F) | +0.251 | +0.290 | +0.233 |
| SDI (`R22`) | +0.12 | +0.27 | +0.14 |
| z(SDI) + z(lnIRF_R) | +0.26 | +0.45 | +0.31 |

14 de 39 candidatos son BH-significativos en DEV contra la recuperación real (`st_13_candidatos_vs_recuperacion_real.csv`); con
este KPI la familia *conservación de Sn en Fusión* reaparece con signo teórico (menos Sn extraído en Fusión → más metal por Sn
cargado, λ_F de §14.6), y la *matriz de escoria* (IRF en ambas fases) sigue siendo la más fuerte y la más estable fuera de
muestra (lnIRF_R: 0.50 en lockbox). El SDI pierde peso porque su componente FeO actúa sobre el dross, que no entra en K1.

## R3.3 Techo con los KPIs alternativos (ST-15)

HGB sobre 39 agregados + controles + heel, CV 5×3 en DEV y DEV→lockbox (`st_15_techo.csv`):

| KPI | R² OOF DEV | ρ OOF DEV | R² lockbox | ρ lockbox |
|---|---|---|---|---|
| K0 rendimiento proxy (referencia) | 0.148 | 0.433 | **0.122** | 0.482 |
| K1 recuperación real | 0.098 | 0.387 | −0.016 | 0.396 |
| K6 sólo canal dross, 100·(1 − f_dross) | 0.132 | 0.399 | 0.049 | **0.575** |
| K7 M/(M+D) | 0.163 | 0.436 | 0.041 | **0.593** |
| K3/K4 ventanas móviles 3/5 | 0.26/0.32 | 0.51/0.57 | **−0.54 / −1.50** | 0.36/0.25 |
| K5 D y P desplazados 1-3 batches | ≤ 0 | 0.13-0.21 | ≤ 0.06 | 0.15-0.35 |

Ningún KPI alternativo honesto (sin ventana) es más predecible que el proxy en lockbox. Las ventanas móviles parecen más
predecibles en CV aleatoria (R² 0.26-0.32) pero es fuga por construcción (el valor de b usa b±1..2 que caen en el fold de
entrenamiento): en lockbox su R² es catastrófico. Ningún par target-KPI supera ρ 0.7 en ningún subconjunto; los que superan
0.5 (lnIRF_R vs K5 desplazamiento 1: 0.52; vs ventanas suavizadas: 0.54-0.65) ocurren sólo en el lockbox (n ≈ 60, régimen de
fin de campaña, efecto "batch siguiente" ya documentado) y no replican en DEV (0.12-0.34), por lo que no pueden usarse para
elegir. lnIRF_R es el target que correlaciona de forma más consistente con todos los KPIs en lockbox (0.33-0.52).

## R3.4 Conclusión de la ronda 3

Aplicada la recomendación, el mejor par target-KPI es **ln IRF en Reducción vs recuperación real**: ρ 0.26 (DEV), 0.50 (lockbox),
0.32 (total, p < 10⁻⁹). Sigue por debajo de 0.70: la recuperación real hereda su propio ruido (metal en t correlaciona −0.43 con
la temperatura media del baño y −0.40 con el dross de Fe cargado a *cualquier* lag: efectos de campaña, no de batch) y el KPI
por ventana móvil no es una alternativa legítima (autocorrelación inducida). El umbral 0.70 requiere datos que el Excel no
contiene: pesaje y asignación por batch del Sn en dross y polvo (o un balance de Sn cerrado por batch), y un KPI de
recuperación con el Sn residual de escoria y el heel contabilizados.
