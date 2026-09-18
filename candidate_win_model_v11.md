# candidate_win_model_v11 — asesor por escalón basado en el modelo físico intra-batch (iteración 19, 2026-09-17)

Código: `asesor_fisico_v11.py` (sobre `asesor_receta_v10.py` y `experimentos/v16/e18_02_lib.py`). Experimentos: `experimentos/v16/` (E18), `experimentos/v17/`
(E19-01). Propuesta de despliegue: `propuesta_despliegue_v11.md`. Fichas previas: `candidate_win_model_v10.md`, `candidate_win_model_v9_1.md`. Hallazgos §27-§28.

## 0. Resumen

| | Reducción (`WIN-R v11`) | Fusión |
|---|---|---|
| Target del escalón | `m8_ln_feo_ret_dross50` = ln(FeO_inv[t] / (FeO_inv[t−1] + FeO eq. del Fe del dross alimentado)): retención de FeO (lo contrario = Fe metalizado → hardhead/dross) | receta |
| STATE | leyes previas de Sn y FeO, inventarios previos de Sn y FeO, T horno previa; receta del escalón (GN, O₂); orden | — |
| CONTROL | **GN ejecutado respecto de la receta** (Nm³/min); O₂ acompaña (ΔO₂ = 2·ΔGN, piso P5 del orden: 22.0 / 16.2 / 5.5 / 3.1); aire intacto; carbón = receta (no utilizable como palanca) | — |
| Forma | efectos fijos de batch y de orden + θ_o·(GN − receta) por orden + estado lineal; estimador within, errores agrupados por batch (compara escalones DENTRO del mismo batch: elimina mineral, talón de escoria, campaña, turno) | — |
| θ_o (por +1 Nm³/min; IC95 bootstrap por batch) | R0 −0.0035 [−0.0075, −0.0010]; R1 −0.0022 [−0.0043, −0.0005]; R2 −0.0027 [−0.0048, −0.0011]; **R3 −0.0058 [−0.0081, −0.0034]** (t −4.4); conjunto p < 1e-5 | — |
| Agregación escalón → batch (exacta) | FeO extra reducido por el desvío de GN: F_b = Σ_t −θ_o(t)·(GN − receta)_t·FeO_inv_{t−1} [kg]; Σ_t ln_feo_ret = ln(FeO_fin / FeO_ini neto) | — |
| Valoración | física, no ajustada al KPI: el Sn "extra" agotado por el GN es adelanto (el Sn de la escoria final no cambia, \|t\| ≤ 1.6) → sólo queda el costo del Fe metalizado: λ = 0.43 kg de Sn a dross por kg de FeO reducido [0.18, 0.67]; en régimen normal −0.28 pp de KPI por 100 kg de FeO [−0.43, −0.13] | — |
| Recomendación | GN* = min(GN previsto, receta) − k·sd_desvío (k = 0: disciplina; k ≤ 1: piloto); suspendida con receta de horno frío (GN de R0 > 28.3 Nm³/min) o T previa < 960 °C (≈ P10 histórico) | receta |

## 1. Validación del modelo (E18-02, E18-05, E19-01)

| Prueba | Resultado |
|---|---|
| Precisión del efecto de escalón (n ≈ 1 400) | GN → retención de FeO −0.0033 por Nm³/min (p < 1e-5); GN → agotamiento de Sn +0.022 (p 1e-5) |
| Estabilidad entre regímenes | mismo signo y p < 0.01 en los 3 tercios cronológicos y en DEV vs lockbox; θ_R3 entre −0.0058 y −0.0082 en las 14 re-estimaciones expansivas (batches 100 → 360) |
| Fuera de muestra (GroupKFold por batch) | correlación predicho-observado del target within 0.61 (FeO) / 0.54 (Sn) |
| Causalidad inversa dentro del batch | el desvío responde al resultado de t−1 (p 0.01) pero controlarlo no cambia θ (0.0160 → 0.0163) |
| **Prospectiva estricta, agregada a batch** (θ sólo con batches anteriores; 256 batches) | F_b predice el **FeO total reducido medido por ensayo: pendiente 0.76 [0.45, 1.07], t 4.8, p permutación por bloques 0.0002** (calibración compatible con 1); DEV 0.79 (t 4.6); lockbox 0.49 (n.s., n 63, mismo signo) |
| Cadena a pérdidas | F_b → dross +0.19 pp por 100 kg (t 3.0, p perm. 0.014; DEV t 3.4; lockbox ≈ 0); F_b → KPI −0.28 pp por 100 kg en régimen normal (t −3.7); en fin de campaña +0.50 (t 2.6) por el canal polvo (−0.45 pp por 100 kg, t −3.2) |
| Selectividad marginal del GN (kg Sn agotado / kg FeO reducido) | R0 11.2 [6.3, 25.8]; R1 0.75; R2 0.78; **R3 0.49 [0.31, 0.72], P(< 0.9) 0.996** |
| Evidencia de batch del exceso de GN (sin modelo) | 63 % de los escalones con GN > receta (+1.06 Nm³/min); prospectiva t 2.71, p perm. 0.003, p circular 0.003; ganancia +0.66 pp/batch [0.10, 1.19] en toda la campaña, +0.91 [0.28, 1.53] en régimen normal |
| Seguridad (E18-03 C) | 0 % de setpoints de GN fuera de [P1, P99] histórico; "gemelos históricos" sin indicadores adversos (BH); operar bajo la receta no enfría el horno (+0.7 °C/escalón) |

## 2. Valor esperado de la regla en la historia (`asesor_fisico_v11.replay_regla`; 281 de 362 batches en régimen)

| Regla | FeO que se deja de reducir | Sn que no va a dross | KPI esperado (régimen normal) |
|---|---|---|---|
| k = 0 (GN ≤ receta) | 146 kg/batch (P10 3, P90 319) | 63 kg [26, 98] | **+0.41 pp [0.19, 0.63]** |
| k = 0.5 sd bajo la receta | 339 kg | 146 kg [61, 227] | +0.95 pp [0.44, 1.46] |
| k = 1 sd | 531 kg | 228 kg [96, 356] | +1.49 pp [0.69, 2.28] |

k > 0 extrapola por debajo de la receta: se mantiene dentro de [P5, P95] histórico pero el Sn final de la escoria sólo se verificó insensible dentro del rango
observado (±1 sd). Por escalón (ejemplo): limitar el GN a la receta evita 64 / 153 / 247 / 422 kg de FeO en R0 / R1 / R2 / R3 cuando el exceso es de ~2 Nm³/min.

## 3. Estado frente al criterio de suficiencia para planta

| Criterio | v11 |
|---|---|
| 1. Relación prospectiva con p corregida < 0.01 estable | **Cumple para el modelo físico**: FeO reducido p 0.0002 (una sola especificación, sin ventanas que elegir: θ expansivo), estable por tercios y por régimen; exceso de GN → KPI p 0.003. Sobre el KPI en todos los regímenes: no (p 0.06; se invierte en fin de campaña) |
| 2. Calibración en [0.6, 1.4] | **Cumple**: 0.76 [0.45, 1.07] sobre el FeO reducido |
| 3. No negativa en fin de campaña | Cumple para el FeO (0.49, mismo signo) y dross (≈ 0); para el KPI no → la regla se suspende allí con un conmutador que declara la propia planta (receta de horno frío) |
| 4. Ganancia con IC que excluya 0 | **Cumple** en régimen normal: +0.41 pp [0.19, 0.63] (k = 0) por la vía física; +0.91 [0.28, 1.53] por la vía directa |
| 5. Mecanismo | **Cumple**: GN → Fe metalizado (ensayo) → dross → KPI; el Sn extra es adelanto |
| 6. Ejecutable al inicio del escalón | **Cumple**: receta + inventario de FeO previo (del cierre anterior disponible; la recomendación no depende de su valor para el signo) |

## 4. Lo que no se afirma

Causalidad estricta (sin aleatorización; la identificación por revisiones de receta falló, E18-01). Valor en fin de campaña (una sola campaña; allí el KPI responde por el
canal polvo con signo contrario). Carbón como palanca. Extrapolación por debajo de la receta más allá de 1 sd.

## 5. Reproducir

    .venv/Scripts/python.exe asesor_fisico_v11.py                                   # θ por orden, valor de la regla, ejemplo por escalón
    .venv/Scripts/python.exe experimentos/v17/e19_01_modelo_fisico_prospectivo.py   # validación prospectiva estricta
    .venv/Scripts/python.exe experimentos/v16/e18_05_balance_por_orden.py | e18_06_estado_final.py | e18_07_regla_final.py
