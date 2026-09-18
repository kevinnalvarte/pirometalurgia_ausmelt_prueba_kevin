# Iteración 12 (2026-09-17) — mejoras del asesor v9 hacia precisión y robustez de planta

Diseño (Fable 5.1). Punto de partida: v9 (`modelo_predictivo_v9.py`, `candidate_win_model_v9.md`, hallazgos §23). Diagnóstico: el cuello de
botella no es la predicción del escalón sino la **precisión del valor de cada control sobre el KPI** (β_GN t 1.5-3 en ventana de 100;
β_C t < 2; calibración 0.44-0.53; ganancia esperada +0.5 pp/batch frente a sd del KPI 4.5 pp) y el conjunto de palancas (sólo GN y carbón).

Dato de planta (usuario, 2026-09-17): la muestra de escoria se toma al FINAL del escalón; los escalones son contiguos (gap 0) y de duración
fija (F0-F5 27..52 ± 1.5 min; R0 30, R1 25, R2 20 exactos) salvo **F6 (58-100 min) y R3 (10-23 min; plan 13, desvío −8..+112)**, cuya
duración depende de cuándo el laboratorio puede medir → variación de la duración en parte EXÓGENA al proceso (cuasi-experimento natural).

## Hipótesis

- **H1 (duración de R3 y F6 como palanca; criterio de corte)**: minutos de R3 por encima de lo esperable dado el estado al inicio de R3 → más Sn
  agotado pero más Fe reducido (dross) → efecto neto sobre el KPI con óptimo interior dependiente del Sn que queda; ídem F6 (más tiempo de
  fusión/oxidación antes de reducir). Identificación favorecida porque la duración la fija en parte la disponibilidad del laboratorio. → E12-01.
- **H2 (valoración por canal)**: estimar β por canal de pérdida (dross, Sn en escoria, polvo) y combinarlos (β_KPI = −Σ β_canal), con ventana larga
  para los canales estables (dross, escoria) y corta para el inestable (polvo), mejora la pendiente/t prospectiva y la calibración frente a β
  sobre el KPI compuesto. → E12-02.
- **H3 (β variable en el tiempo por estado-espacio)**: un filtro de Kalman (β_t = β_{t−1} + η) con varianzas por máxima verosimilitud en el pasado
  mejora la prueba prospectiva frente a la ventana dura W=100 y da un IC honesto para abstenerse. → E12-03.
- **H4 (ejecutabilidad sin el ensayo de t−1)**: con el estado "plan" (cierre de F6 + variables en línea de t−1) en E[a|S] la evidencia prospectiva
  se conserva (≥ 80 % de la pendiente-t) → el asesor puede emitir la recomendación al INICIO del escalón y corregirla al llegar el ensayo. → E12-04.
- **H5 (posición de lanza y otras palancas no evaluadas por el canal directo)**: posición vertical de lanza (óptimo interior en E7-08), O₂/enriquecimiento,
  aire: efecto directo prospectivo sobre KPI/canales con forma cuadrática donde la teoría lo pide. → E12-05.

## Método común (obligatorio)

Exposición = residuo de la palanca respecto de E[palanca | estado al inicio del escalón] (HGB cross-fitted por batch; `experimentos/v9/v9_lib.py`),
estandarizado por orden, agregado por batch. Evaluación: (a) OLS HC3 conjunta con xG, xC (tabla `experimentos/v9/e11_06b_tabla_batch.csv`) +
controles sin `espesor`; DEV / lockbox / total; tercios; canales; placebos (lead 1, lead 3, pre-tratamiento); (b) **prueba prospectiva** con
`experimentos/v9/e11_07_lib.py` (`walk_forward_score`, W=100, paso 10, permutación por bloques de 20, ≥ 1000) comparando {xG, xC} vs {xG, xC, nueva}.
Una mejora se adopta si: signo con sentido físico; p < 0.05 conjunta en DEV y mismo signo en total; placebos limpios; y **mejora la prueba
prospectiva** (t o Spearman) sin empeorar el lockbox. Todo lo probado se reporta. Entorno: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`,
`OMP_NUM_THREADS=3`, primer plano, sin joblib. Salidas en `experimentos/v10/e12_0N_*`.

## Criterio de suficiencia para planta (fijado aquí)

El asesor se considera listo para piloto de campo con expectativa de valor si, en el replay prospectivo: (1) pendiente > 0 con p corregida < 0.01;
(2) calibración (pendiente) en [0.6, 1.4]; (3) relación no negativa en el lockbox; (4) ganancia esperada calibrada ≥ +0.5 pp/batch con IC que
excluya 0; (5) cada palanca recomendada con mecanismo de escalón/canal significativo; (6) recomendación ejecutable al inicio del escalón.
