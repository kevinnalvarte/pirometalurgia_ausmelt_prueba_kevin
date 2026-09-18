# E18-03 — Tres replanteamientos evaluables hoy sobre el asesor v10 (receta)

Motivación: el asesor v10 (`asesor_receta_v10.py`) vale −1.58 pp/sd de GN sobre receta en régimen normal (t −3.9) pero el
signo se invierte en los últimos 63 batches (fin de campaña, horno ~975 vs ~1070 °C, receta de GN R0 subida a 30.8).
El usuario exige replantear QUÉ se demuestra, no re-ajustar el modelo. Scripts: `experimentos/v16/e18_03_lib.py`
(motor compartido, reutiliza `asesor_receta_v10`, `e11_07_lib`, `e13_03_lib`), `e18_03_A_alcance.py`, `e18_03_B_unilateral.py`,
`e18_03_C_seguridad.py`. n = 362 batches, DEV = 299 (idx 1-299), lockbox = 63 (idx 300-362).

## A. Alcance declarado ex-ante (4 reglas, ninguna mira el KPI)

% de escalones en alcance por tramo de campaña (`e18_03_A_pct_por_tramo.csv`):

| Regla | idx 1-100 | 101-200 | 201-300 | 301-362 (lockbox) | global |
|---|---|---|---|---|---|
| A1 T_prev≥1000 °C | 100% | 93.5% | 42.8% | 18.5% | 68.4% |
| A2 T_prev≥1020 °C | 100% | 86.5% | 27.0% | 5.6% | 59.9% |
| A3 espesor≥P20 expansivo | 10% | 0% | 0% | 0% | 2.8% (solo batches 21-30) |
| A4 receta GN R0≤28.3 | 92% | 94.0% | 34.0% | **0%** | 60.8% |

**A3 es inutilizable**: el espesor de ladrillo cae monótonamente con la campaña (corr = −0.997 con idx_cronologico), así
que "≥ P20 de lo visto hasta la fecha" solo es cierto en los 10 primeros batches con historia — sin variación fuera del
calentamiento, la prueba prospectiva (empieza en el batch 100) da varianza nula (t = NaN). **A4 colapsa casi exactamente
en "excluir el lockbox"**: la planta subió la receta de GN de R0 justo cuando cambió de régimen, así que A4 no es una
regla operativa distinta de "usar solo DEV".

Evidencia DENTRO del alcance (exposición = media del desvío solo en escalones con mask, 0 si ninguno; `e18_03_A_ols_dentro.csv`,
`e18_03_A_resumen.csv`, `e18_03_A_jackknife.csv`, `e18_03_A_calibracion.csv`):

| Regla | OLS GN→KPI (t) | OLS GN→dross (t) | Prospectivo t | p_perm simple | **p corregida (máx. 4 reglas)** | Calibración IC95 | Jackknife −10 (%t>1.645) | Fuera de alcance GN→KPI (t) |
|---|---|---|---|---|---|---|---|---|
| A1 | −1.89 | +2.06 | 1.19 | 0.081 | 0.237 | [−1.72, 1.34] | 1% | **−2.25 (contradice)** |
| A2 | −2.78 | +2.43 | 2.02 | 0.021 | 0.051 | [−1.64, 1.51] | 65% | −1.24 (ns, ok) |
| A3 | −0.40 | +1.24 | NaN | — | — | — | — | −3.08 (=global, no informativo) |
| A4 | **−3.23** | **+3.10** | **2.33** | **0.008** | 0.022 | [−0.65, 1.40] | 92% | −0.46 (ns, ok) |

**Ninguna de las 4 reglas cruza p corregida < 0.01** (A4, la mejor, queda en 0.022) y en ninguna la calibración excluye 0.
A1 además se contradice: el efecto sigue siendo significativo (t −2.25) **fuera** de su propio alcance declarado, lo que
invalida a A1 como regla de alcance limpia. A4 es la única sin contradicción fuera de alcance, pero es funcionalmente
el split DEV/lockbox ya conocido, no una nueva frontera operativa.

## B. Regla unilateral "GN nunca por encima de la receta"

63.0% de los escalones ejecutan GN > receta (exceso medio +1.69 Nm³/min; por orden: R0 66%/+1.04, R1 52%/+1.92,
R2 73%/+2.16, R3 60%/+1.63). Exposición física (sin modelo): x⁺ = media(max(dev_gn,0)), x⁻ = media(min(dev_gn,0))
(`e18_03_B_descriptivo_por_orden.csv`).

Regresión conjunta KPI/dross/Sn-escoria ~ x⁺ + x⁻ + carbón + controles (`e18_03_B_asimetria_*.csv`):

| y | β⁺ (t) | β⁻ (t) | H0 β⁺=β⁻ (p) |
|---|---|---|---|
| KPI | **−0.63 (−2.63)** | −0.28 (−1.23, ns) | 0.356 (no distinguible) |
| dross (f_dross·100) | +0.46 (+2.16) | +0.46 (+2.62) | 0.996 (simétrico: cualquier desvío de receta ensucia) |
| Sn en escoria (·100) | +0.10 (+1.95) | **−0.12 (−2.39)** | **0.008 (asimetría confirmada)** |

Para KPI el exceso (x⁺) es significativo y el defecto (x⁻) no, pero la prueba formal de asimetría no alcanza
significancia (p 0.36): **es sugerente, no está confirmada estadísticamente para el KPI**; sí está confirmada para
Sn-en-escoria (operar por debajo de receta reduce la pérdida de Sn, operar por encima la aumenta).

Ganancia esperada −β⁺·E[x⁺], bootstrap por bloques de 10 (`e18_03_B_ganancia_boot.csv`):

| Segmento | Ganancia pp/batch | IC95 |
|---|---|---|
| Total (n=361) | +0.65 | **[0.09, 1.23]** |
| DEV (n=298) | +0.93 | **[0.31, 1.53]** |
| Dentro de A1 (T≥1000, n=361) | +0.32 | [−0.01, 0.67] (toca 0) |

Prospectivo de x⁺ solo (`e18_03_B_prospectivo_xplus.csv`): n=261, pendiente 0.78, **t=2.70, p_perm=0.001** — más
fuerte que el asesor completo (t 2.6-2.7, p 0.002-0.012) y sin necesitar estimar dirección ni β para ejecutarse.

Seguridad térmica (`e18_03_B_seguridad_dT.csv/_Tcierre.csv`, cluster por Batch): ΔT del escalón ~ x⁺(escalón) −0.54
(t −1.85, p=0.064, marginal) y x⁻(escalón) **+0.68 (t +2.82, p=0.005)** — operar por debajo de receta se asocia a
MÁS calentamiento, no a enfriamiento; no hay señal de riesgo térmico. T horno al cierre de Reducción ~ x⁺ −3.86
(t −1.41, ns), x⁻ −0.12 (t −0.05, ns): tampoco hay efecto significativo de operar por debajo sobre la temperatura final.

## C. Expediente de seguridad (`asesor_receta_v10.replay_prospectivo()`, 888 escalones recomendados)

**C1** (`e18_03_C1_resumen.csv`): GN: 0% fuera de [P1,P99] histórico, 7.5% fuera de [P5,P95]; paso máximo 3.02 Nm³/min
(16.9% de la receta, P95 12.6%). Carbón: 2.5% fuera de [P1,P99] (no es 0), 11.4% fuera de [P5,P95]; paso máximo
6.94 kg/min (**41.9% de la receta** — outlier a vigilar), P95 24.8%.

**C2** (`e18_03_C2_gemelos.csv`): 17.9% de escalones "gemelos" (ejecutado a ≤0.25 sd del setpoint del asesor); 37/222
batches gemelos (≥2 de 4 escalones). Tras BH sobre 8 indicadores, solo 2 son significativos y **ambos favorables**:
presión de punta de lanza más baja en gemelos (41.0 vs 49.7 kPa, q=0.0003) y desvío de duración menor (−0.55 vs
+0.04 min, q=0.0008, más apegado al plan). dT del escalón, tiro del horno, dross, polvo, Sn en escoria y KPI: sin
diferencia significativa (q≥0.08). **Ningún indicador empeora.**

**C3**: 0 batches con GN ejecutado ≤−1 sd en los 4 escalones; relajando a ≥3 de 4, solo 4 batches — T_media_R 1004.7
vs 1045.2 °C (≈40 °C más frío), dross igual, KPI 65.5 vs 68.3 (peor). **n=4 es insuficiente para concluir**; no se
puede descartar causalidad inversa (el operador corta GN porque el horno ya viene mal).

**C4** (`e18_03_C4_o2.csv`): con ΔO2=2·ΔGN y piso 0, 4.8% de recomendaciones de O2 serían negativas sin piso; **23.6%
quedarían bajo el P1 histórico del orden y 35.9% bajo el P5** — el piso 0 es insuficiente. Piso propuesto = P5
histórico por orden (Nm³/min): R0 22.0, R1 16.2, R2 5.5, R3 3.1.

## Veredicto final

Ninguno de los tres replanteamientos entrega hoy un "desplegar en planta sin condiciones". **A (alcance)** falla: A3 es
vacío, A1 se contradice fuera de su propio alcance, y la única regla limpia (A4) coincide casi 1:1 con excluir el
lockbox — no aísla un régimen operativo nuevo, solo repite el split ya conocido, y ni así cruza p<0.01 ni calibra con
IC fuera de 0. **C (seguridad)** es la más sólida como *expediente*: los pasos recomendados están dentro del rango
histórico (excepto el paso de carbón, que necesita un tope explícito) y no hay ningún indicador de seguridad que
empeore en los "gemelos históricos" — pero un expediente de seguridad no es una demostración de eficacia, y el
piso de O2 debe corregirse antes de cualquier prueba. **B (regla unilateral "GN nunca sobre receta")** es el único
que hoy sostiene una afirmación acotada y con evidencia: gana +0.65 a +0.93 pp/batch (IC95 excluye 0 en total y en
DEV), el efecto prospectivo de solo el exceso es más fuerte que el del asesor completo (t 2.70, p 0.001) y no muestra
riesgo térmico. Su límite honesto: la asimetría formal (exceso daña / defecto es neutro) no está confirmada
estadísticamente para el KPI (p=0.36, solo para Sn-en-escoria) y dentro del alcance térmico A1 el IC toca 0.

**Frase sostenible ante un comité**: *"No podemos afirmar que el asesor adaptativo v10 sea robusto para desplegarse en
planta bajo ninguna regla de alcance evaluada. Sí podemos afirmar, con IC95 que excluye 0 y sin evidencia de riesgo
térmico, que una disciplina simple y sin modelo — no ejecutar gas natural por encima de la receta de planta — habría
valido +0.65 pp de KPI por batch en toda la campaña (+0.93 en régimen normal), y que un piloto de ESA disciplina
puntual (no del asesor adaptativo completo) es defendible como próximo paso, con el piso de O2 corregido a P5
histórico y un tope explícito al paso de carbón."**
