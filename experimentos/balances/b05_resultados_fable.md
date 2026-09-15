# B-05a (Fable) — hipótesis propias derivadas del modelo Rust

Script `b05_hipotesis_fable.py`; salidas `b05_h1_plm_fusion.csv`, `b05_h1_batch_ols.csv`, `b05_h2_monitor_fe.csv`, `b05_log.txt`.

## H1. ¿Cementa el Fe metálico del dross al SnO en Fusión? (ec. 6 del modelo Rust: Fe + SnO → FeO + Sn con todo el Fe)

**No se observa.** PLM v4 del Sn extraído en Fusión (DEV, n = 1749, R² OOF 0.73 en las tres variantes):

| Palanca (θ por +1 sd, kg Sn) | v4 | física (C fijo, Fe metálico del dross, C-eq lanza, GN) | física + cal |
|---|---|---|---|
| C fijo del carbón | (C/Sn +51 [−66, +189]) | +14 [−131, +147] | +51 [−86, +197] |
| **Fe metálico del dross (C-eq)** | — | **−40 [−150, +47]** | **−85 [−195, −1]** |
| C-eq de la lanza | — | −65 [−147, +47] | −91 [−173, +8] |
| GN | +221 [+114, +339] | +176 [+67, +289] | +201 [+94, +306] |
| cal/carga | −440 [−715, −181] | — | −468 [−746, −173] |

El Fe metálico cargado no extrae Sn de la escoria en Fusión; si acaso lo retiene (θ negativo). A nivel batch (OLS HC3, DEV, por sd,
controles: Sn cargado, ley, espesor, tendencia): dross de Fe en Fusión → f_dross +0.0025 (p 0.39), f_metal −0.0055 (p 0.25),
Sn extraído en Fusión +0.42 t (p 0.13); el mineral de Fe (t4) **reduce** el Sn extraído en Fusión (−1.10 t/sd, p 0.003) y el carbón
también (−1.21 t/sd, p < 0.001; es el mecanismo de β_C: el carbón de Fusión baja el dross, exp 19: aquí −0.0062 f_dross/sd, p 0.016).

Lectura: la hipótesis del modelo teórico de que el 100 % del Fe metálico disponible cementa SnO no tiene respaldo en los escalones. El
dross de Fe de tolva 5 se comporta como una corriente ferrífera que se escorifica (aporta FeO a la matriz), no como reductor. Para el
modelo Rust esto significa que la ec. 6 debería tener una extensión parcial (o nula) y que el término `b6_ceq_fe_dross_kg` del balance
reductor debe ponderarse hacia cero (B-02 lo evalúa por sensibilidad de `FE_MET_DROSS`).

## H2. Monitor de Fe: ¿cuándo empieza la pérdida de FeO y qué dice del batch?

Con el inventario multi-trazador de FeO (`b6_feo_inv_multi_kg`), una pérdida > 3 % del inventario previo (por encima del ruido de
ensayo, ~2 %) ocurre en 357/362 batches y **empieza en el primer escalón de Reducción en 283 (78 %)**, en R1 en 58, R2 en 12, R3 en 4.
Es decir, la sobre-reducción del Fe no espera al agotamiento del Sn: arranca con la Reducción.

| Señal del monitor (DEV, n ≈ 284-296) | ρ vs f_dross | ρ vs rendimiento proxy | ρ vs SDI |
|---|---|---|---|
| %Sn de la escoria al inicio de la pérdida de FeO | −0.04 (n.s.) | −0.04 (n.s.) | −0.11 (p 0.06) |
| avance del Sn al inicio | +0.04 (n.s.) | +0.03 (n.s.) | +0.18 (p 0.002) |
| escalón de inicio (más tarde = mejor) | −0.12 (p 0.04) | **+0.15 (p 0.009)** | +0.33 (p 10⁻⁸) |
| pérdida total de FeO en la fase (fracción) | **+0.17 (p 0.005)** | **−0.23 (p 6·10⁻⁵)** | −0.68 (construcción) |

Lectura: el %Sn al que comienza a caer el FeO no discrimina batches; lo que importa es **cuánto FeO se pierde en total** (la
componente FeO del SDI) y, secundariamente, **retrasar el inicio** de esa pérdida. En el lockbox (n = 61) el avance al inicio da
ρ +0.21 con f_dross (p 0.10), mismo signo. El "monitor de Fe" que pide el equipo metalúrgico ya está capturado por `ln(FeO_fin/FeO_ini)`
del SDI; el escalón de inicio es un diagnóstico complementario (posible feature STATE para R1-R3: "ya empezó la pérdida de FeO").

## B-01b. Ronda 2 del multi-trazador (consensos alternativos), `b01b_multitrazador_ronda2.py`

Mismas filas de Reducción (587-596 escalones válidos, 285 batches DEV). Todas las variantes multi (4 inertes con pesos asumidos o
empíricos, 3 sin CaO, 2 sólo SiO₂+Al₂O₃) muestran la huella de una serie dominada por ruido y ninguna predictibilidad por escalón;
el trazador CaO re-anclado conserva señal:

| variante | autocorr lag-1 FeO_ext | sd FeO_ext (kg) | ρ Σln FeO_ret vs KPI (DEV) | ρ SDI vs KPI (DEV) | R² OOF HGB FeO_ext |
|---|---|---|---|---|---|
| CaO único (v3, mismas filas) | **−0.15** | **691** | 0.137 (p 0.02) | 0.176 (p 0.003) | **0.248** |
| 4 inertes, pesos asumidos | −0.65 | 1975 | 0.194 | 0.230 | −0.017 |
| 4 inertes, pesos empíricos | −0.50 | 1139 | 0.200 | 0.233 | −0.005 |
| 3 inertes sin CaO | −0.63 | 1529 | 0.195 | 0.231 | 0.004 |
| SiO₂ + Al₂O₃ | −0.68 | 2012 | 0.197 | 0.225 | −0.015 |

Lectura: los ln-cocientes de SiO₂, Al₂O₃ y MgO entre escalones consecutivos tienen 3-4 veces más varianza que la que explica un
cambio real de masa; a nivel batch la suma telescópica cancela parte de ese ruido (ρ 0.23 vs 0.18 en estas filas), pero a nivel
escalón no hay nada que modelar. Decisión: **RECHAZAR** el multi-trazador en todas sus formas; mantener el trazador CaO. Valor
metalúrgico del hallazgo: la lógica de "óxidos inertes" del modelo Rust vale para el balance de diseño, pero los ensayos de SiO₂,
Al₂O₃ y MgO de planta no tienen la precisión (o los óxidos no la inercia: desgaste de refractario, mezcla incompleta en R0-R1) para
usarse como trazadores por escalón.

## B-05b. Avance de la pérdida de FeO como feature de ESTADO (`b05b_avance_feo_estado.py`)

`b6_avance_feo_prev = 1 − FeO_inv_prev[t] / FeO_inv_prev[R0]` (trazador CaO, STATE) y bandera `> 3 %`, añadidas al estado del PLM v4
de Reducción:

| target | estado | R² OOF DEV | R² lockbox |
|---|---|---|---|
| FeO reducido (kg) | v4 | 0.367 | 0.378 |
| FeO reducido (kg) | v4 + avance FeO | 0.383 | 0.361 |
| Sn extraído (kg) | v4 / + avance FeO | 0.976 / 0.976 | 0.984 / 0.984 |
| SDI | v4 | 0.131 | 0.143 |
| SDI | v4 + avance FeO | 0.154 | 0.044 |

Mejora en OOF (+0.016 FeO, +0.023 SDI) pero empeora en lockbox (−0.017 y −0.10): no cumple el criterio de adopción (mejorar OOF sin
empeorar lockbox). El θ del carbón no cambia de calidad (IC sigue cruzando 0). Decisión: **RONDA 2** (revisar con reentrenamiento por
campaña; el lockbox es fin de campaña y la trayectoria de FeO cambia de régimen allí).
