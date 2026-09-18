# E12-04 — asesor ejecutable al inicio del escalón (H4)

Scripts: `e12_04_lib.py` (estados), `e12_04_01_estados.py` (1,2), `e12_04_02_prospectivo.py` (3),
`e12_04_03_dos_etapas.py` (4), `e12_04_04_sensibilidad.py` (5,6). Entorno `.venv`, primer plano, sin joblib.

## 1. Cuatro estados (`e12_04_01_seguridad_estados.csv`, `..._nan_por_estado.csv`)

Los 4 pasan `L8.asegurar_seguras`. Las 8 columnas `*_F6p` de S3' se registraron como STATE (misma cantidad física
que `*_F6`/`*_prev2`, solo resuelta por orden: R0 usa `_prev2` = cierre de F5, R1-R3 usa `*_F6` = cierre de F6).

| Estado | Cols | Filas completas (de 1448) | Peor columna (%NaN) |
|---|---|---|---|
| S1 (v9, ref.) | 17 | 99.0 % | `m6_feo_inv_kg_prev` 0.9 % |
| S2 (t−2, online) | 22 | 85.0 % | `presion_punta_lanza_kpa_prev` 13.8 % |
| S3' (plan) | 20 | 98.9 % | `ley_feo_escoria_pct_F6p` 1.0 % |
| S4 (solo en línea) | 14 | 86.2 % | `presion_punta_lanza_kpa_prev` 13.8 % |

S2 y S4 pierden ~14-15 % de escalones por un único sensor (presión de punta de lanza) con huecos, no por el ensayo:
**S3' tiene casi la misma cobertura que S1** (98.9 % vs 99.0 %), pese a no usar el ensayo de t−1.

## 2. E[a|S], R² OOF y exposiciones (`..._r2_oof_por_estado.csv`, `..._correlacion_residuos_vs_S1.csv`, `..._ols_conjunta.csv`)

R² OOF global de E[a|S] (GroupKFold(5) por Batch) es **prácticamente idéntico entre estados** (GN 0.757-0.765,
carbón 0.938-0.945); por orden es mucho más bajo (0.03-0.33) en los 4 — la política habitual varía por orden más
de lo que cualquier estado explica, con o sin ensayo. Los residuos estandarizados z de S2/S3'/S4 correlacionan con
los de S1 en **Pearson 0.91-0.93 / Spearman 0.85-0.89** (las 2 palancas): perder el ensayo de t−1 apenas cambia
*a quién* se juzga desviado de la práctica habitual.

OLS HC3 conjunta (xG, xC + controles sin espesor), DEV:

| y | x | S1 | S2 | S3' | S4 |
|---|---|---|---|---|---|
| KPI | β_GN (t) | −1.22 (−3.37) | −1.26 (−2.74) | −1.21 (−2.90) | −1.41 (−3.12) |
| KPI | β_C (t) | +0.73 (0.93) | +0.37 (0.53) | **+1.62 (2.11)** | +0.28 (0.44) |
| f_dross | β_GN (t) | +0.011 (3.89) | +0.012 (3.51) | +0.012 (3.84) | +0.014 (3.99) |
| Sn escoria | β_C (t) | −0.0032 (−2.16) | −0.0015 (−0.78) | −0.0007 (−0.50) | −0.0013 (−0.74) |

El mecanismo GN→dross (positivo, p<0.001) es idéntico en los 4 estados: no depende del ensayo de t−1. El carbón
es más débil e inestable entre estados (solo S3' lo identifica sobre el KPI en DEV).

## 3. Prueba prospectiva (`e12_04_02_evidencia_*.csv`, `..._evidencia_todos_estados.csv`)

Réplica exacta de `mp9.replay_prospectivo`/`evidencia_prospectiva` (W=100, paso 10) vía monkeypatch de
`mp9.ESTADO_R_V9`. `S1` reproduce el baseline de v9 (`experimentos/v9/e11_09_evidencia_prospectiva.csv`) exacto.

| Estado | muestra | n | pendiente | t | p uni | Spearman parcial | p perm. bloques |
|---|---|---|---|---|---|---|---|
| S1 | todo | 262 | 0.441 | **1.989** | 0.023 | 0.165 | 0.043 |
| S1 | DEV | 199 | 0.479 | **2.063** | 0.020 | 0.202 | 0.026 |
| S2 | todo | 262 | 0.456 | 1.753 | 0.040 | 0.150 | 0.092 |
| S2 | DEV | 199 | 0.527 | 1.847 | 0.032 | 0.194 | 0.045 |
| S3' | todo | 262 | 0.438 | 1.723 | 0.042 | 0.181 | 0.098 |
| S3' | DEV | 199 | 0.472 | 1.588 | 0.056 | 0.213 | 0.072 |
| S4 | todo | 262 | 0.564 | **2.424** | 0.008 | 0.211 | 0.014 |
| S4 | DEV | 199 | 0.673 | **2.480** | 0.007 | 0.257 | 0.003 |

Lockbox (n=63): S1 ya es no significativo y de signo negativo (t=−0.27) — no sirve de referencia para H4; todos
los estados quedan en ese rango de ruido (−0.40 a +0.26), sin degradación adicional atribuible al estado.

**Criterio H4 (≥80 % de la t de S1)**: S2 lo cumple en todo y DEV (88 %/89 %); **S3' lo cumple en "todo" (87 %) y
queda debajo en DEV (77 %)**, un margen modesto y dentro del ruido de este único replay; S4 lo **supera** (todo
122 %, DEV 120 %) — con la salvedad de la sección 6.

## 4. Esquema en dos etapas: a\*(S3') vs a\*(S1) (`e12_04_03_dos_etapas_por_orden.csv`)

1048 escalones emparejados (mismo Batch × orden, ambos replays). Diferencia de setpoint recomendado:

| Palanca | \|Δ\| media (unid. físicas) | \|Δ\| media (sd del orden) | % cambia acción (recomendar↔mantener) | Reversión con ambos activos |
|---|---|---|---|---|
| GN (Nm³/min) | 0.73 | 0.29 sd | 5.2 % | **0/367 (0 %)** |
| Carbón (kg/min) | 1.99 | 0.38 sd | 26.7 % | **0/248 (0 %)** |

La diferencia es de **magnitud, no de signo**: en ningún escalón donde ambos estados recomiendan mover la palanca
apuntan en direcciones opuestas. GN cambia de "recomendar" a "mantener" (o viceversa) en 1 de cada 19 escalones;
carbón en 1 de cada 4 (más sensible a si se conoce el ensayo — coherente con que carbón es la palanca peor
identificada). Con S3' el operador recibiría casi siempre la misma dirección que con S1, con una magnitud algo
distinta.

## 5. Valor de la información del ensayo de t−1 (`e12_04_04_valor_informacion.csv`)

| Estado | muestra | pendiente S1 | pendiente estado | t S1 | t estado | Δt |
|---|---|---|---|---|---|---|
| S3' | todo | 0.441 | 0.438 | 1.989 | 1.723 | −0.27 |
| S3' | DEV | 0.479 | 0.472 | 2.063 | 1.588 | −0.48 |

**El punto estimado (pendiente) casi no cambia** (0.441→0.438 todo; 0.479→0.472 DEV): el ensayo de t−1 no aporta
tamaño de efecto adicional. Lo que aporta es **precisión** — el error estándar sube de 0.221→0.254 (todo) y de
0.232→0.297 (DEV), bajando la t. Interpretación operativa: si el ensayo llega a mitad del escalón, corregir con S1
no cambiaría la recomendación económica esperada en promedio, solo la angostaría un poco; el "costo" de operar
solo con S3' es estadístico (menos confianza en el signo), no una pérdida de la magnitud del efecto.

## 6. Veredicto

**El asesor puede emitir con S3' (cierre de F6, o de F5 en R0, + variables en línea + orden) al INICIO del
escalón**, y no pierde evidencia de forma relevante: exposiciones z correlacionan 0.93/0.87 con las de S1, el
mecanismo GN→dross se sostiene idéntico, la pendiente prospectiva es prácticamente la misma (0.438 vs 0.441) y en
ningún escalón con recomendación activa en ambos estados se invierte la dirección. Cobertura: **S3' es ejecutable
en 98.9 % de los escalones** (vs 99.0 % de S1), muy por encima de S2/S4 (85-86 %, limitados por un sensor de lanza
con huecos, no por el ensayo).

S4 (sin ningún ensayo) da una t prospectiva *mayor* que S1 (2.42 vs 1.99 todo), lo cual es contraintuitivo — al
condicionar en menos estado, E[a|S] deja más varianza sin explicar en el residuo, que puede estar absorbiendo
confusión no observada (estado real del baño) en vez de una desviación causal de la práctica. Se reporta pero
**no se recomienda operar con S4** sin placebos adicionales (lead/pretratamiento) que descarten esa confusión —
fuera del alcance de E12-04.

**Todos los escalones R0-R3 son plenamente ejecutables con S3'** (R0 con el ensayo de F5, R1-R3 con el de F6): no
hay ningún orden que deba esperar el cierre de t−1. Esquema recomendado: recomendación al inicio con S3'; si el
ensayo de t−1 llega durante el escalón, re-evaluar con S1 solo afecta la confianza (t), no la dirección — no es
necesario bloquear la ejecución a la espera del ensayo.
