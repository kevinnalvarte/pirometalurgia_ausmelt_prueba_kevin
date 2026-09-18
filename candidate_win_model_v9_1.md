# candidate_win_model_v9_1 — asesor de planta v9.1 y protocolo de piloto (iteración 12, 2026-09-17)

Código: `asesor_planta_v9_1.py` (sobre `modelo_predictivo_v9.py`). Diseño y memos: `experimentos/v10/ITERACION_12_diseno.md`,
`experimentos/v10/e12_0N_resultados.md`. Ficha base: `candidate_win_model_v9.md` (todo lo que no se menciona aquí sigue igual). Hallazgos §24.

## 0. Veredicto de la iteración 12

Se exploraron las cinco mejoras propuestas y cuatro adicionales. **Ninguna mejora de modelado aumenta la precisión del valor de los controles;
la única que cambia el sistema es la ejecutabilidad.** El límite es de información, no de método: 362 batches de una sola campaña, KPI con
sd 4.4 pp (residual 3.9) y una exposición observacional con sd 0.64 acotan el error de β_GN en ≥ 0.32 pp aun usando todos los datos con
efecto constante (t máximo ≈ 3.4, que es lo que E11-01 ya obtiene en DEV); con deriva de campaña, menos. Lo que sí entrega la precisión que
falta es un piloto aleatorizado simétrico embebido en el asesor: **80 batches (≈ 4 semanas a 3 batches/día) → 80 % de potencia**.

| Mejora explorada | Exp. | Resultado |
|---|---|---|
| Duración de R3 / F6 como palanca (criterio de corte) | E12-01 | **No**: R3 es casi exógena (R² 0.15 desde el estado) y agota +5 kg Sn/min (p 0.02) sin mover FeO; KPI +0.37 pp/sd n.s.; empeora la prueba prospectiva (t 2.33 → 1.98). F6 confundida con el tamaño de carga (placebo t 5.9) |
| Valoración por canal de pérdida | E12-02 | **No**: equivalente al KPI directo por linealidad (t 2.10 vs 2.05); ventanas por canal / EB / sin polvo se invierten en lockbox; se(β) ×1.05-1.12. Identidad exacta: KPI = 100·f_metal·(1 − Sn_escoria) |
| β por filtro de Kalman | E12-03 | **No**: t 1.61-1.79 vs 2.32 de W=100; pendiente negativa en lockbox; filtro 13-27 % sobre-confiado; abstenerse por \|t\| empeora |
| Estado ejecutable al inicio del escalón | E12-04 | **Sí (adoptado)**: S3' (cierre de F6; en R0 cierre de F5; + en línea) conserva 87 % de la t prospectiva, exposiciones r 0.91-0.93 con las de v9, GN → dross t 3.5-4.0 en los cuatro estados, 0 % de inversiones de dirección |
| Lanza, O₂, enriquecimiento, aire, caudal total, presión de punta | E12-05 | **No**: 0/14 con p < 0.05 en DEV (q-BH ≥ 0.63); lanza sin óptimo identificable sobre el KPI; estequiometría de planta "en duda" |
| Ajuste por covariables pre-tratamiento (talón IRF, KPI lag 1-3, estado al cierre de F6) | E12-06 | **No**: R² de controles 0.08 → 0.36 y se(β_G) −8 %, pero la evidencia prospectiva no mejora (t 1.58 → 1.31-1.78). Nota: quitar 5 batches mueve la t base de 2.05 a 1.58 → la evidencia observacional es frágil |
| Objetivo físico sin pesos ajustados (ΔSn_kg − λ·ΔFeO_kg, λ = 0.6 por composición del dross) | E12-08 | **No**: DEV premia retener FeO (sólo-FeO t +2.0..+2.2), lockbox premia agotar Sn (t −2.4): dos regímenes con valoración opuesta; ningún λ fijo sirve a ambos |
| Potencia del piloto con ruido real | E12-07 | tratado vs control (−1 sd): 150-200 batches; **simétrico ±1 sd: 80 batches → 0.80 (β −1.1), 0.96 (β −1.5), 0.52 (β −0.75)**; dross como endpoint: 0.71 / 0.92 / 0.48 |
| Lazo cerrado simulado con ruido real (400 batches) | E12-09 | ver §3 |

## 1. Qué es v9.1

| Componente | Definición |
|---|---|
| STATE ejecutable (20) | leyes e inventarios al cierre de F6 (`*_F6p`; en R0 = cierre de F5), orden, termocupla, T horno y gradiente previos, posición de lanza previa, acumulados de carbón / GN / O₂ / aire / Sn, C:Sn acumulado, tiempo de fase — todo disponible al INICIO del escalón |
| CONTROL | GN (palanca principal; intensidad de fuego de lanza) y carbón (secundaria); **O₂ acompaña al GN (ΔO₂ = 2·ΔGN, piso 0); aire = setpoint de protección de lanza, no se toca**; Fusión = receta |
| Valor | β (pp de KPI por unidad de desviación media estandarizada respecto de la práctica habitual), OLS HC3 sobre los últimos 100 batches, re-anclado cada 10 |
| Modo explotación | a* = Ê[a\|S] + min(1, \|t\|/2)·signo(β̂)·sd_res si \|t\| ≥ 1; caja [P5, P95] del orden; carbón R0 ≤ P90; sin recorte de GN con T_prev < P10 |
| Modo exploración (piloto) | brazo ∈ {−1, +1} por batch, bloques permutados de 4 (`asignar_brazos`, semilla registrada): GN de R0-R3 = Ê[GN\|S] ± 1 sd_res (≈ ±2.3-3.3 Nm³/min), dentro de la caja; mismo resguardo de horno frío |
| Análisis | `analizar_piloto`: ITT outcome ~ brazo + controles pre-tratamiento (HC3) para KPI, `f_dross` y Sn en escoria; miradas en 40 / 60 / 80 batches con límites tipo O'Brien-Fleming (\|z\| ≥ 2.37 / 1.94 / 1.68) |

## 2. Estado frente al criterio de suficiencia para planta (fijado a priori en el diseño)

| Criterio | Estado |
|---|---|
| 1. Relación prospectiva > 0 con p corregida < 0.01 | **No** (0.010-0.027 según esquema; frágil a la muestra) |
| 2. Calibración en [0.6, 1.4] | **No** (0.44-0.53: sobre-atribución ×2) |
| 3. No negativa en lockbox | Sí, débil (nula; el asesor se abstiene en GN) |
| 4. Ganancia calibrada ≥ +0.5 pp con IC que excluya 0 | Sí en DEV-prospectivo: +0.70 pp por sd de puntaje [0.28, 1.17]; IC cruza 0 en lockbox |
| 5. Mecanismo por palanca | GN sí (escalón → dross, 99 % de ventanas); carbón parcial (sólo canal escoria) |
| 6. Ejecutable al inicio del escalón | **Sí** (E12-04) |

Lectura: el asesor **no** tiene todavía la precisión para operar en lazo de explotación con ganancia garantizada, y está demostrado (9 intentos)
que no se obtiene con más modelado sobre estos datos. **Sí** está listo para usarse en planta en modo piloto, que es la forma de obtenerla.

## 3. Por qué usarlo en planta agrega valor esperado sin riesgo relevante (E12-09)

Simulación con el ruido real del KPI (bootstrap por bloques), 400 batches, 400 réplicas; ganancia real media en pp de KPI por batch respecto de
la práctica habitual:

| Efecto real del GN | Sin asesor | Asesor observacional | **Piloto 80 batches ±1 sd + explotación con 30 % de exploración** |
|---|---|---|---|
| constante −1.1 (estimación DEV) | 0.00 | +0.56 [0.10, 0.93]; P(< 0) 1.2 % | **+0.49 [0.29, 0.64]; P(< 0) 0.2 %**; tras el piloto +0.61; se(β) 0.54 |
| constante −0.5 | 0.00 | +0.10 [−0.04, 0.28] | +0.10 [−0.01, 0.21] |
| **nulo** | 0 | 0 | **0 (sin daño: la perturbación simétrica tiene costo esperado 0)** |
| deriva como la estimada (−0.3 → −2.6 → +0.4) | 0.01 | +0.70 [0.28, 0.88] | +0.29 [0.18, 0.41]; P(< 0) 0 % |
| cambio brusco de signo (−1.5 → +1.5) | 0.00 | +0.68 [0.31, 1.01] | +0.49 [0.31, 0.65]; P(< 0) 0 % |

0.5 pp ≈ 250 kg de Sn a metal por batch ≈ 750 kg/día. Supuestos: efecto lineal dentro de ±1 sd (dosis-respuesta observada casi lineal,
E11-01) y cumplimiento con ruido 0.3 sd. La columna "observacional" es optimista (en la simulación la variación natural es exógena; en la
planta no lo es): por eso el camino recomendado es el de la derecha, que además reduce a la mitad el error de β y la dispersión del resultado.

## 4. Protocolo de piloto

1. **Sombra (10-15 batches)**: `recomendar_en_vivo` con el estado ejecutable, sin ejecutar; verificar con metalurgia setpoints, cajas, O₂ en piso
   (en R3 el O₂ del plan es ~4-7 Nm³/min: un recorte de GN de 2.3 lo lleva a ~2; acordar el piso real) y el resguardo térmico.
2. **Piloto aleatorizado (80 batches)**: GN de R0-R3 a Ê[GN|S] ± 1 sd_res según `asignar_brazos` (semilla y lista en acta); carbón, aire, Fusión
   y duración según receta. Registro por escalón: setpoint recomendado, ejecutado, motivo de desvío.
3. **Endpoints**: primario KPI refinado (ITT); secundarios `f_dross`, Sn en escoria final, retención de FeO por escalón (n = 4 por batch), ΔT
   (seguridad). Miradas en 40 / 60 / 80 con los límites de §1. Parada de seguridad: ΔT del escalón fuera de P1-P99 histórico en 3 batches
   seguidos de un brazo, o |z| de daño ≥ 2.37 en la primera mirada.
4. **Después**: β causal → modo explotación con 30 % de batches de exploración permanente (mantiene el re-anclaje vivo ante la deriva de
   campaña) y, con el mismo esquema, segundo factor: carbón R0-R1 +1 sd (factorial 2×2, mismo número de batches).
5. Re-entrenar Ê[a|S] y cajas al inicio de cada campaña (20-30 batches de sombra tras el cambio de ladrillo).

## 5. Reproducir

    .venv/Scripts/python.exe asesor_planta_v9_1.py                          # asesor con todo el histórico + ejemplo en vivo y brazos
    .venv/Scripts/python.exe experimentos/v10/e12_06_covariables.py | e12_07_potencia_piloto.py | e12_08_objetivo_fisico.py | e12_09_lazo_cerrado.py

## 6. Tercera ronda (iteración 13, `experimentos/v11/`): calibración, auditoría adversarial y lectura de decisión

| Exploración | Resultado |
|---|---|
| E13-01 contracción James-Stein de parte positiva, fijada a priori y sin hiperparámetros: β̃ = β̂·max(0, 1 − 1/t²) (el error de β̂ atenuaba la pendiente: dilución) | Prueba rápida (exposiciones cross-fitted, W 100): calibración 0.53 → **0.70** (0.65-0.87 con W 60-120); p por permutación 0.0055; corregida sobre 6 ventanas 0.023 → 0.006; lockbox +0.88 (n.s.). **Adoptada** (`CONFIG_V9["contraccion"] = "js"`) |
| E13-03 auditoría adversarial (Sonnet) de E13-01 | Reproduce. Placebos con JS **limpios** (lead 1 t 0.05; sobrevive a controlar KPI y exposiciones del batch anterior, t 2.30) y sin bloque cronológico dominante (leave-one-block-out t ≥ 1.96). Pero: con bloques de 10 o desplazamiento circular p corregida 0.010-0.012; sobre toda la familia de contracciones × ventanas **p 0.014**; 43 % de 81 combinaciones razonables dan p < 0.01 (xG + xC juntas 25/27); jackknife t > 1.645 en 95 % (−5 batches), 68 % (−10), 59 % (−20). **Criterio 2 (calibración) sostenido; criterio 1 (p < 0.01) no** |
| E13-02 replay ESTRICTO del asesor de planta (estado ejecutable; práctica habitual entrenada sólo con el pasado) | sin contracción: pendiente 0.44, t 1.72, perm. por bloques 0.098; con JS: 0.58, t 1.41, perm. 0.16; Spearman parcial 0.19 (p 0.002) y terciles −1.15 / +0.04 / +1.12 pp en ambos |
| E13-04 corrección de deriva de receta: Ê[a\|S] + residuo medio por orden de los últimos k batches | k 20: pendiente **0.74**, t 1.89 (p 0.029), perm. 0.087, lockbox +0.64, terciles −0.99 / −0.08 / +1.08; k 40: 0.57, t 1.40 (sin mejora) → sensible a k; se deja k = 20 en `CONFIG_PLANTA` como valor operativo razonable, no como hallazgo |
| E13-05 lectura de decisión (bootstrap por bloques del replay estricto final) | ganancia calibrada de seguir la recomendación **+0.80 pp/batch, IC95 [0.04, 1.92], IC90 [0.16, 1.70]; P(> 0) 0.98; P(> 0.25 pp) 0.91** (k 40: +0.57 [−0.06, 1.47], P(> 0) 0.96) |

**Estado final frente al criterio de suficiencia**: 2 (calibración) ✔ con JS; 3 (lockbox no negativo) ✔; 4 (ganancia calibrada con IC que excluye 0) ✔ por poco;
5 (mecanismo) ✔ GN / parcial carbón; 6 (ejecutable) ✔; **1 (p corregida < 0.01 estable) ✘**: la p honesta es 0.01-0.03 en la prueba rápida y 0.03-0.09 en
el replay estricto. Tres rondas y 15 exploraciones muestran que ese umbral no es alcanzable con datos observacionales de una campaña (t máx teórica ≈ 3.4
con efecto constante; menor con deriva). Lectura de decisión: usar el asesor tiene valor esperado positivo (P ≈ 0.96-0.98) con pasos ≤ 1 sd dentro de la
práctica histórica y costo esperado 0 bajo el nulo; la confirmación con p < 0.01 la da el piloto aleatorizado embebido (80 batches).
