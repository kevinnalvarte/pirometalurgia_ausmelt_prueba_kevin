# Modelo ganador final — asesor por escalón v11 (cierre de la iteración, 2026-09-17)

Código: `asesor_fisico_v11.py` (modelo físico intra-batch + recomendación) sobre `asesor_receta_v10.py` (receta de planta, regla y conmutador de régimen).
Ficha técnica: `candidate_win_model_v11.md`. Despliegue: `propuesta_despliegue_v11.md`. Hallazgos §26-§29. Verificación final: `experimentos/v19/`.

## 1. Qué recomienda

En cada escalón de Reducción (R0-R3), al inicio del escalón y sin esperar ningún ensayo: **ejecutar el GN de lanza como máximo en el valor de la receta de planta**
(O₂ acompaña el recorte con ΔO₂ = 2·ΔGN y piso P5 histórico del orden; aire, carbón y Fusión según receta). Para cada escalón informa los kg de FeO que se dejan de
reducir (con IC), los kg de Sn que dejan de ir a dross y el efecto esperado sobre el KPI. La regla se suspende cuando planta declara la receta de horno frío
(GN de R0 > 28.3 Nm³/min, fin de campaña) o con T previa < 960 °C.

## 2. Condiciones de aceptación del proyecto → evidencia

| Condición del usuario | Cumplimiento |
|---|---|
| Variables con sentido pirometalúrgico, parsimonia y estabilidad | **Target**: retención de FeO del escalón (lo contrario = Fe metalizado → hardhead/dross). **STATE** (5 + receta + orden): leyes previas de Sn y FeO, inventarios previos de Sn y FeO, T horno previa. **CONTROL**: GN ejecutado − receta. 4 parámetros de efecto (θ por orden). θ_R3 entre −0.0058 y −0.0082 en 14 re-estimaciones expansivas |
| STATE y CONTROL diferenciados; recomendación robusta, estable, coherente y significativa | La recomendación depende sólo de la receta (conocida antes del batch) y del régimen; θ conjunto p < 1e-5; mismo signo en los 3 tercios cronológicos y en fin de campaña |
| El modelo distingue robustamente el efecto del CONTROL | Diseño intra-batch (efectos fijos de batch y de orden): compara escalones del mismo batch, eliminando mineral, talón de escoria, campaña y turno; robusto a controlar la respuesta del operador al escalón anterior (θ 0.0160 → 0.0163) |
| Sin monotonías supuestas; forma por teoría | Dosis-respuesta within aproximadamente lineal en ±1 sd y sin asimetría significativa (E18-02); el óptimo no es "menos GN siempre": el Sn final de la escoria se verificó insensible sólo dentro del rango observado → recomendación acotada a la receta (k = 0) y a ≤ 1 sd en piloto; régimen de horno frío excluido por física y por dato |
| Sin fuga; generaliza en batches no vistos; simulación escalón a escalón | θ estimado sólo con batches anteriores → kg de FeO extra reducido por batch predicen el FeO total reducido medido por ensayo: **pendiente 0.76 [0.45, 1.07], t 4.8, p permutación 0.0002** (256 batches no vistos; DEV 0.79; fin de campaña 0.49, mismo signo). `replay_regla` recorre cada batch escalón a escalón |
| Paralelización con subagentes Sonnet; Fable diseña y decide | 8 rondas, ≈ 30 experimentos, 3 auditorías adversariales (E13-03, E15-02, E18-03) |
| **Las decisiones derivadas producen un incremento robusto, significativo y coherente del rendimiento del batch (obligatorio)** | Adherencia a la recomendación (−exceso de GN sobre la receta) → KPI refinado: **DEV +1.00 pp por Nm³/min [0.49, 1.52], t 3.8, p 0.0001; toda la campaña +0.72 [0.28, 1.17], t 3.2, p 0.0015**; cuartiles monótonos (−0.76 / −0.87 / +0.66 / +0.98 pp). **Prospectiva en batches no vistos: p ≤ 0.0085 en las 6 ventanas; p corregida por multiplicidad 0.0035; circular 0.003**; jackknife −5/−10/−20/−40 batches: 97-100 % con t > 1.645. No es persistencia: la parte propia del batch (innovación) +0.91 (t 3.1) y con efectos fijos por bloques de 20 batches +1.06 (t 3.6). Valor: **+0.66 pp/batch [0.10, 1.19] en toda la campaña; +0.91 [0.28, 1.53] en régimen normal (≈ 470 kg de Sn a metal por batch)**; por la vía física +0.41 [0.19, 0.63] |
| Target agregable escalón → fase → batch | Σ_t ln_feo_ret = ln(FeO_fin / FeO_ini neto) (telescópica, exacta); FeO extra reducido por batch = Σ_t −θ_o·(GN − receta)_t·FeO_inv_{t−1} |
| Relación positiva significativa entre aplicar la recomendación y el rendimiento, con sustento pirometalúrgico | Cadena cerrada con mediciones independientes: exceso de GN → más FeO reducido (ensayo de escoria, t 3.8-4.8) → más dross (pesaje, t 2.9-4.1) → menos KPI (t −3.1 a −3.8); el Sn "extra" que agota el GN es adelanto (el Sn de la escoria final no cambia, \|t\| ≤ 1.6); selectividad marginal del GN en R3 = 0.49 kg Sn / kg FeO [0.31, 0.72] |
| Documentación por iteración | `hallazgos.md` §23-§29, fichas v9 → v11, memos por experimento, `experimentos/README.md` |

## 3. Límites que se declaran (no invalidan la aceptación; acotan el uso)

1. **Fin de campaña** (últimos ~60 batches, horno < 1000 °C, receta de horno frío): la relación con el KPI no se sostiene (−0.51, n.s.) → la regla está suspendida allí por
   diseño; el mecanismo sobre el FeO conserva el signo. Una sola campaña no permite modelar ese régimen.
2. **Evidencia observacional**: la identificación cuasi-experimental por revisiones de receta no funcionó (instrumento débil). La confirmación causal es el piloto
   (receta ± 1 sd, 80 batches), que el despliegue por niveles habilita sin riesgo: el nivel 0 sólo pide cumplir el estándar vigente.
3. **Carbón, O₂, aire, lanza, duración de R3/F6 y Fusión**: sin palanca demostrada → receta.
4. El valor del asesor adaptativo v10 (GN por debajo de la receta según β re-anclado) por encima de la regla simple no está demostrado; queda para el piloto.

## 4. Siguiente paso operativo

Nivel 0 (`propuesta_despliegue_v11.md`): tablero de desvío ejecutado − receta por escalón + regla GN ≤ receta en régimen normal; nivel 1: sombra pre-registrada del
asesor; nivel 2: piloto. Re-estimar θ y revisar el conmutador al inicio de cada campaña.
