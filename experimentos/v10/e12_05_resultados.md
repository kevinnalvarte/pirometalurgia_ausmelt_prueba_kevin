# E12-05 — palancas no evaluadas por el canal directo acción → KPI (H5)

Candidatas: posición vertical de lanza, O₂ de lanza, enriquecimiento O₂, aire, estequiometría de planta, caudal total
de gas de lanza (GN+O₂+aire) y presión de punta. Método común (`ITERACION_12_diseno.md`): residuo a−E[a|S] (HGB
cross-fitted por batch, `experimentos/v10/e12_05_lib.py`, grupo simple R=R0-R3 / F=F1-F6, sin split Rt/Rl), z por
(fase,orden) en DEV, agregado por batch. `posicion_vertical_lanza_mm_prev` está correctamente en el estado (S), no
se filtra. DEV n=298-299, lockbox n=63. Scripts `e12_05_0{1..5}_*.py`; CSVs y esta memo en `experimentos/v10/`.

## 1. Descriptivo (`e12_05_r2_oof.csv`, `_icc.csv`, `_deriva.csv`, `_corr_{R,F}.csv`)

R² OOF de E[a|S]: **lanza en Fusión = 0.91** (casi determinada por el estado/orden — la lanza sigue el nivel del
baño, poco margen de decisión libre, coherente con E7-08); lanza en Reducción 0.39. O₂ y enriquecimiento muy
predecibles (R²=0.76-0.85 en Reducción): son casi receta. **Aire y estequiometría casi no predecibles por el estado
(R²≈-0.04/-0.05 en Reducción)** — mucha varianza "libre", pero eso no implica señal sobre KPI. ICC bajo (0.05-0.26)
salvo **presión de punta (ICC 0.44 Reducción / 0.50 Fusión)**: varía sobre todo ENTRE batches, no dentro — más
compatible con un rasgo de la punta/batch que con una decisión de escalón (pista de "consecuencia", no acción).
Colinealidad confirmada: **O₂-GN r=0.75-0.81** en ambas fases (como predecía el diseño); caudal total correlaciona
fuerte con O₂ y aire (r=0.73-0.88, es su suma). Deriva: varias candidatas (o2, caudal, gn, carbon, aire en Fusión)
muestran tendencia significativa con `idx_cronologico` en TOTAL (p<0.001) pero NO en DEV (p>0.1) — la tendencia se
concentra en/alrededor del lockbox, mismo patrón de "deriva de campaña" que GN_R en E11-01.

## 2. OLS conjunta KPI y canales (`e12_05_ols_*.csv`)

**Ninguna de las 14 combinaciones candidata×grupo cruza p<0.05 en DEV** (una a una, KPI ~ xG+xC+controles+candidata).
Más cercanas: F_esteq (coef −0.99, p=.058), F_o2 (coef −0.73, p=.099); resto p entre .16 y .93. q-BH dentro de la
familia (14): todas ≥ 0.63. La regresión **conjunta** (todas las candidatas del grupo a la vez) da coeficientes
inestables y con signos que cambian entre DEV/lockbox (p.ej. R_enriq lockbox +8.5, p=.003) — artefacto de
colinealidad (O₂-GN-caudal-aire comparten varianza), no evidencia. Canales (DEV, sólo hallazgos p<0.10, no sobreviven
BH): O2_R y enriq_R → `sn_perdido_escoria_frac` +0.0025/+0.0014 (p=.031/.040); lanza_F → `f_dross` −0.0119 (p=.028);
esteq_F → `f_metal` −0.0106 (p=.046) y `f_dross` +0.0113 (p=.057). Tercios: sólo R_o2 y R_esteq muestran signo
estable 3/3 (negativo), pero con t/p débiles en DEV. **Bootstrap (1000, por batch): no se ejecuta para ninguna
candidata porque ninguna pasó p<0.05 en DEV** (criterio del diseño).

## 3. Lanza — forma no monótona (`e12_05_lanza_*.csv`)

**(a)** Lineal+cuadrático del residuo sobre KPI (batch): **ningún término es significativo** en R ni F, DEV/lockbox/
total (p entre .35 y .99); el signo del cuadrático ni siquiera es estable entre muestras. La curvatura de E7-08 (que
vivía en el target de agotamiento de Sn/retención de FeO a nivel de escalón) **no replica** en el KPI agregado por
batch. **(b)** Nivel absoluto: KPI residualizado (por xG+xC+controles) por quintil de posición media en Reducción es
**casi monótono creciente** (−0.36, −0.09, −0.06, +0.09, +0.44 pp; ΔQ5−Q1≈0.8pp), NO una curva con óptimo interior
claro; el vértice ajustado (2856mm) tiene IC95% bootstrap **[1998, 5022] mm — cubre casi todo el rango observado**,
no identificado. **(c)** Mecanismo de escalón (residuo-sobre-residuo, grupo R, n=1448/1326): lineal sobre
`m6_ln_sn_dep` p=.068 (marginal) y sobre ΔT p=.043 (linda con el umbral, sin corrección); `m8_ln_feo_ret_dross50`
n.s. (p=.168); **todos los términos cuadráticos n.s.** (p>.10). **(d)** Dinámica intra-batch: el residuo z no arrastra
tendencia con el orden (ya absorbida por el estado, p=.27-.35); correlación con proxies de desgaste (gas acumulado)
es significativa pero minúscula en Reducción (ρ=−0.07/−0.08, p<.01) y nula en Fusión; **63-75% de los cambios
escalón-a-escalón superan 300mm** (saltos discretos grandes), lo que sugiere reposicionamiento programado/decisión
más que deriva mecánica continua por desgaste — el pequeño componente de desgaste no domina el patrón.

## 4. Placebos (`e12_05_placebos.csv`, sólo F_esteq y F_o2 — las más cercanas a pasar)

F_esteq: limpio en los 5 placebos (lead1 p=.22, lead3 p=.63, pre-ley p=.91, pre-feed p=.19, pre-KPI p=.23).
**F_o2: pre-KPI_prev significativo (coef −0.88, p=.043)** — el O₂ de Fusión se correlaciona con el KPI del batch
ANTERIOR (ajuste reactivo del operador), mismo patrón de contaminación que carbón_F↔ley_sn_conc en E11-01: resta
crédito causal a F_o2 aunque su OLS ya era n.s.

## 5. Prueba prospectiva (`e12_05_prospectiva.csv`, W=100, paso=10, permutación bloques=20, n=1200)

Base {xG,xC} DEV: t=2.20 (p=.014, n=198) — reproduce el resultado ya establecido. Añadir cualquier candidata da como
máximo mejoras marginales de t (R_o2 2.20→2.35; F_o2 2.20→2.43; F_lanza 2.20→2.28) y en la mayoría empeora
(R_lanza→1.54, R_enriq→1.93, F_enriq→1.92). **Ninguna mejora supera la nula por permutación**: p_perm más bajos
F_o2=.108, R_ppunta=.116, R_o2=.117; resto .21-.96. F_esteq mejora mucho **lockbox** (t 0.78→2.10, p=.018) y
**total** (t 2.32→2.81, p=.0025) pero empeora levemente DEV (2.20→2.06) y su p_perm=.438 — no significativo. F_o2
lockbox se vuelve inestable (t→0.20, sd_score dispara a 2.4): otra señal de fragilidad, coherente con el placebo
fallido.

## Veredictos

| candidata | veredicto | motivo |
|---|---|---|
| posición vertical de lanza | **RECHAZAR** | sin lineal/cuadrático en KPI (a); nivel absoluto sin óptimo interior identificado, vértice no acotado (b); mecanismo débil/no robusto (c); R²=0.91 en Fusión = casi sin margen de decisión; permutación R p=.96, F p=.21 |
| O₂ de lanza | **RECHAZAR** (colinealidad con GN) | DEV p=.10-.16, q_BH>.6; permutación p=.11-.12; lockbox inestable en Fusión; placebo pre-KPI contamina F_o2 |
| enriquecimiento O₂ | **RECHAZAR** | DEV p=.49-.58; permutación p=.57-.66 |
| aire | **RECHAZAR** | DEV p=.87-.89; permutación p=.35-.68; colineal con caudal (r hasta .88) |
| estequiometría de planta | **EN DUDA** | único con signo replicado DEV(p=.058)-lockbox(p=.094)-total(p=.0025) y placebos limpios, pero NO cruza p<0.05 en DEV ni la permutación prospectiva (p=.438) — no cumple el criterio de adopción; candidato a re-testear con más potencia |
| caudal total de gas de lanza | **RECHAZAR** | combinación lineal de gn+o2+aire, hereda su falta de señal (DEV p=.31-.79; permutación p=.42-.73) |
| presión de punta de lanza | **RECHAZAR — más CONSECUENCIA que acción** | ICC más alto de todas (0.44-0.50, varía entre batches, no dentro); 13% de dato faltante; DEV p=.33/.95; permutación p=.12/.37 |

**Ninguna candidata cumple el criterio de suficiencia para planta** (pendiente prospectiva p<0.01 corregida). No se
propone ninguna incorporación al asesor v9 desde esta ola; la posición de lanza (hipótesis central de H5, heredada de
E7-08) queda específicamente descartada como palanca directa sobre el KPI de batch, aunque el hallazgo original de
E7-08 (curvatura a nivel de escalón sobre targets de Sn/FeO) no se contradice — son cantidades distintas.
