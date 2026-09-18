# Propuesta de despliegue por niveles (iteración 18, 2026-09-17)

Restricciones del proyecto: sólo existe la campaña actual (362 batches) y el piloto en planta no se autoriza hasta demostrar un modelo robusto y preciso.
Experimentos de esta ronda: `experimentos/v16/` (E18-01..07). Fichas: `candidate_win_model_v10.md` (asesor por receta), `candidate_win_model_v9_1.md` (piloto).

## 1. Alternativas evaluadas

| Alternativa | Exp. | Resultado |
|---|---|---|
| Revisiones de la receta (49 versiones) como experimento natural: IV y estudio de eventos | E18-01 | **No sirve**: instrumento débil (F 2.8-10.4), 2SLS con IC95 [−1.9, +4.3] pp por Nm³/min, y la receta FUTURA predice el KPI de hoy (t −2.05): receta y campaña derivan juntas |
| Diseño intra-batch (efectos fijos de batch y orden) con el desvío exacto respecto de la receta, sobre los resultados físicos del escalón | E18-02, E18-05 | **Sí**: GN → agotamiento de Sn +0.022 por Nm³/min (p 1e-5) y → retención de FeO −0.0033 (p < 1e-5); **estable en los 3 tercios cronológicos y en DEV vs lockbox**; R² fuera de muestra 0.54 / 0.61. Selectividad marginal del GN (kg Sn agotado por kg FeO reducido): **R0 11.2 [6.3, 25.8]; R1 0.75; R2 0.78; R3 0.49 [0.31, 0.72], P(< 0.9) = 0.996** |
| Estado FINAL de la Reducción como endpoint (ensayo, no pesaje) | E18-06 | **Cierra el mecanismo**: el desvío de GN en cualquier orden NO cambia el Sn que queda en la escoria final (\|t\| ≤ 1.6: el Sn "extra" agotado es adelanto); el GN tardío (R2-R3) sobre la receta reduce **+102 kg de FeO por Nm³/min (t 3.8)** → dross +0.29 pp (t 2.9) → KPI −0.43 pp (t −3.1) en régimen normal. En fin de campaña el signo sobre el FeO se conserva (+52, n.s.); la inversión del KPI viene del canal polvo |
| Coeficientes físicos de conversión (batch) | E18-04 | 1 kg de FeO reducido → +0.43 kg de Sn a dross [0.18, 0.67] y +0.29 a polvo (DEV); la conversión Sn agotado → metal no es identificable (atenuación) |
| Alcance declarado ex-ante (T ≥ 1000 / 1020 °C, vida de ladrillo, receta de horno frío) | E18-03 A | Ninguna regla da p corregida < 0.01; la mejor (receta de GN de R0 ≤ 28.3) equivale a excluir el fin de campaña (p corregida 0.022) |
| **Regla de disciplina sin modelo: "GN nunca por encima de la receta"** | E18-03 B, E18-07 | 63 % de los escalones exceden la receta (+1.06 Nm³/min de exceso medio; 67 % en R2-R3). Exceso → KPI −0.86 pp por Nm³/min (t −3.1 DEV). **Prueba prospectiva: t 2.71, p permutación 0.003, p circular 0.003** (sin ningún parámetro que elegir; DEV t 3.6). **Ganancia esperada +0.66 pp/batch, IC95 [0.10, 1.19], P(> 0) 0.99 en toda la campaña; +0.91 [0.28, 1.53] en régimen normal**. En fin de campaña la regla habría costado −0.85 [−2.26, +0.40] → debe suspenderse allí |
| Expediente de seguridad | E18-03 C | Recomendaciones de GN 0 % fuera de [P1, P99] histórico; "gemelos históricos" (escalones ya ejecutados a ≤ 0.25 sd del setpoint recomendado): ningún indicador de seguridad empeora tras BH (2 mejoran: presión de punta de lanza, apego a la duración); operar bajo la receta NO enfría el horno (+0.7 °C/escalón). Correcciones: piso de O₂ = P5 histórico por orden (22.0 / 16.2 / 5.5 / 3.1 Nm³/min); tope al paso de carbón |

## 2. Qué se puede sostener hoy ante un comité (con los datos existentes)

1. **Hecho físico, preciso y estable entre regímenes** (n ≈ 1 400 escalones, comparación dentro del mismo batch): cada Nm³/min de GN por encima de la receta reduce
   más FeO; en R3 rinde 0.49 kg de Sn agotado por kg de FeO reducido, y ese Sn se habría agotado igual (el Sn final de la escoria no cambia). El Fe metalizado
   arrastra 0.43 kg de Sn a dross por kg de FeO.
2. **Hecho de operación**: el 63 % de los escalones se ejecuta con GN por encima del estándar que la propia planta fijó.
3. **Consecuencia medida en batches no vistos**: el exceso de GN predice peor KPI de forma prospectiva (p 0.003) y cumplir la receta habría valido +0.9 pp por batch en
   régimen normal (≈ 470 kg de Sn a metal por batch), IC95 que excluye 0.
4. **Límite conocido**: en fin de campaña (horno < 1000 °C, planta ya opera la receta de horno frío con GN de R0 > 28.3) la regla no aplica: se suspende.

## 3. Despliegue por niveles

| Nivel | Qué es | Requiere aprobación de piloto | Evidencia que lo respalda |
|---|---|---|---|
| **0. Disciplina de receta** | Monitoreo por escalón del desvío ejecutado − receta y regla "GN ≤ receta" en régimen normal (`asesor_receta_v10.regla_disciplina_gn`); O₂ acompaña con piso P5; aire intacto; suspendida con receta de horno frío o T previa < 1000 °C | **No**: es cumplir el estándar vigente de planta, no un experimento | §2 (1-3); seguridad E18-03 C |
| **1. Sombra pre-registrada** | El asesor v10 puntúa cada batch nuevo sin intervenir; análisis fijado de antemano (pendiente del KPI sobre el puntaje, permutación por bloques, miradas cada 40 batches) | No (no interviene) | Genera la evidencia prospectiva genuina y cubre el arranque de la próxima campaña |
| **2. Asesor adaptativo v10** (GN por debajo de la receta cuando β lo respalda; carbón) | Pasos ≤ 1 sd alrededor de la receta | Sí | Se habilita cuando el nivel 0-1 acumule ≥ 80-100 batches con la relación confirmada; protocolo en `candidate_win_model_v9_1.md` §4 |

El nivel 0 rompe el círculo "sin modelo robusto no hay piloto / sin piloto no hay prueba": no pide permiso para experimentar, sólo para cumplir la receta, y su propio
despliegue (cumplimiento que sube del 37 % a ~100 %) produce la variación que confirma o refuta el efecto en las semanas siguientes.

## 4. Lo que sigue sin estar demostrado (no se afirma)

Causalidad estricta (la identificación cuasi-experimental por revisiones de receta falló); el valor del asesor adaptativo completo por encima de la regla simple; el
comportamiento en fin de campaña (una sola campaña); el carbón como palanca (nunca significativo solo; intra-batch con signo contraintuitivo).
