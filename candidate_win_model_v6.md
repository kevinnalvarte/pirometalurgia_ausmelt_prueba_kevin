# Candidato v6 (iteración 8, 2026-09-16): masa de escoria por cierre físico + prescriptor v6

Ficha resumida; detalle en `hallazgos.md` §19, memos `experimentos/v6/e8_0N_resultados.md` y el artefacto
https://claude.ai/artifact/MWuQCtJCnqNtFVXJ9no3SP. La ficha v5 sigue en `candidate_win_model.md`.

## 1. Inferencia de masa de escoria (adoptada)

- `experimentos/v6/masa_v6.py::agregar_masa_v6(df)`; parámetros `PARAMS_DEFAULT` (fusion="cierre", reduccion="cierre", pG 0.703,
  gG 0.301, aC 0, δ_R 0.01). Todo inferible en línea con el ensayo del escalón (Sn, Fe totales) y las alimentaciones acumuladas.
- Fórmula: `r_t = 1 − 1.270·s_t − f_t`; `M_t·r_t = M_{t−1}·r_{t−1}·e^{−δ_R[R]} + pG·cal_t + gG·carga_t`. Solo SnO₂ y FeO salen de la
  escoria; el resto se conserva. Coeficientes efectivos calibrados en DEV contra la masa final del balance global (metal+dross+polvo).
- Diagnósticos: 3.5 % de escalones sin carga que ganan masa (v3 31 %), ρ(escala, cal/carga) 0.03 (v3 −0.35), balance/M 1.04/1.02,
  cierre de Sn 0.99, ruido propagado ~½ del trazador CaO.
- Columnas: `m6_masa_kg[_prev]`, `m6_sn_inv_kg[_prev]`, `m6_feo_inv_kg[_prev]`, `m6_avance_prev`, `m6_resto_frac_prev`, targets
  `m6_ln_sn_dep`, `m6_ln_feo_ret`, palancas `Cx_sn_v6`, `Cx_av_v6`, `m6_dosis_C_sn`, `m6_exceso_C_pos`; `m6_k_anclaje` LEAKAGE.

## 2. Prescriptor v6 (`modelo_predictivo_v6.py`, candidato)

- Misma arquitectura que v5 (PLM con signos, J con costos/ventana térmica/IC/soporte relativo, política cross-fitted) sobre estado y
  targets v6. CONFIG_V6: w 6.09 [4.05, 9.99], K 2.893 pp/unidad, β_F −0.58.
- Validación (mismas filas que v5): FeO_ret 0.329/0.486 (v5 0.169/0.194); Sn_dep 0.765/0.800; ΔT R 0.464/0.351; Fusión Sn_dep
  0.161/−0.47. θ: GN +0.052 (Sn) / −0.012 (FeO); Cx_sn_v6 +0.16 [0.03, 0.29].
- Política: Fusión carbón −0.5 sd, GN −0.13 sd; Reducción carbón +1.5 kg/min (avance <0.84) y −2.5..−4.2 después; GN −0.12 sd.
- Evidencia: cadena +0.26 pp [0.18, 0.36] DEV / +0.37 lockbox (~134/189 kg Sn/batch); adherencia Fusión −0.42 pp/sd (p 0.17),
  polvo +0.006/sd (p 0.006); emparejamiento con señales mixtas. No supera empíricamente a v5; coincide con v5 salvo en el carbón
  de Reducción. Requiere piloto A/B.

## 3. Cómo reproducir

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/v6/masa_v6.py            # calibración + batería
    .venv/Scripts/python.exe experimentos/v6/e8_05_validacion_v6.py validacion   # tablas, θ, predicciones, curvas
    .venv/Scripts/python.exe experimentos/v6/e8_05_validacion_v6.py 0..4|lockbox # política cross-fitted (Start-Process, ~10 min c/u)
    .venv/Scripts/python.exe experimentos/v6/e8_05_validacion_v6.py evidencia
    .venv/Scripts/python.exe experimentos/v6/e8_06_datos_figuras.py all         # datos del artefacto

## 4. Pendientes

Análisis de CaO/humedad de la cal y ganga de la carga (cierra pG, gG); pesaje de polvo (δ_R); recalibración por campaña; piloto A/B
(GN de Reducción, carbón de Fusión y de Reducción temprana); reentrenamiento progresivo; Fusión sin target predictivo en lockbox.
