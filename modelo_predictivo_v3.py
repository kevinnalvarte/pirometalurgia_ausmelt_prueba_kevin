"""Modelo predictivo/prescriptivo v3 de Lingo Smelter (sesion 2026-09-13 noche).

Extiende `modelo_prescriptivo.py` (v1) y `modelo_predictivo_v2.py` (v2) -- ambos siguen
siendo validos -- con las piezas nuevas de esta ronda. Que cambia y por que (detalle y
evidencia en hallazgos.md seccion 14):

1. **Datos nuevos del Excel** (dataset_lingo_smelter.py): alimentacion y Sn fino POR TOLVA
   (concentrado t1, reciclo Sn/Fe t3, mineral Fe WF t4, dross Fe t5, pellets/humos t7),
   desglose del carbon por etapa, duracion PLANIFICADA del escalon, 8 termocuplas de
   carcasa + espesor de ladrillo, y correccion de `feed_CaO_kgh` (antes 'CaO total' =
   tolva 6 + tolva 3; ahora tolva 6 = cal pura).

2. **Features v3 pirometalurgicamente interpretables** (feature_engineering.py, bloque v3):
   - ENTRADA (ACTION): tasas y fracciones de mezcla por corriente, ley de Sn de la mezcla,
     cal/carga, C/Sn cargado, duracion planificada, desglose de carbon.
   - CONTEXTO: ensayo de Sn por corriente del batch, espesor de refractario, escalon.
   - ESTADO: inventario acumulado por corriente, masa de escoria estimada por TRAZADOR CaO
     (la cal no se reduce ni volatiliza: masa_escoria = CaO acumulado / %CaO), inventario
     de Sn y FeO en escoria (kg), avance de reduccion, termocuplas previas, IRF previo.
   - SALIDA (TARGET) EN MASA: sn_extraido_est_kg (kg de Sn que salieron de la escoria en
     el escalon: a metal o a polvo), frac_sn_extraido_escalon, feo_extraido_est_kg,
     selectividad en masa. Resuelven el "cuello de botella" declarado desde hallazgos.md
     6b/9.5: antes no habia masa de escoria por escalon.

3. **Resultado predictivo (honesto)**: en FUSION el target en masa `sn_extraido_est_kg`
   alcanza R2 lockbox ~0.68-0.72 vs ~0.45 del delta de ley, PERO el baseline trivial "solo
   Sn cargado en el escalon" ya da 0.71: la quimica agrega ~0.03 de R2 OOF. En REDUCCION
   "solo inventario previo" da 0.98. El valor de las features v3 no esta en el R2 sino en
   (a) la RESPUESTA a las palancas (punto 4), (b) la escala fisica comun en kg y (c) el
   DeltaT de Reduccion, que pasa a ser predecible (R2 lockbox ~0.3-0.5, antes -0.04) y
   habilita una restriccion termica. Deriva de campana: el lockbox (agosto 2026) opera al
   final de la campana de refractario, fuera del rango de entrenamiento -> reentrenar por
   campana (hallazgos.md 14.3).

4. **Causalidad (DML cross-fitted con el estado v3 como control)**: con el target en masa,
   la dosis de carbon en FUSION pasa a tener efecto causal ROBUSTO y positivo sobre el Sn
   extraido (con el target de ley no era significativa, hallazgos.md 13.1). Ver
   experimentos/07_efectos_causales_v3.csv y hallazgos.md 14 para el resto de palancas.

5. **Prescriptor v3**: (a) las ACCIONES incluyen ahora la mezcla de carga por tolva, la cal,
   el carbon, los gases de lanza y la duracion PLANIFICADA (no la real); (b) el simulador
   reconstruye TODAS las features derivadas desde primitivas (`recalcular_derivadas_v3`),
   por lo que cambiar el feature set no exige reescribirlo; (c) la funcion objetivo esta
   en UNIDADES FISICAS COMUNES (kg de Sn a METAL, equivalente) con dos coeficientes
   estimados de los propios batches:
     - REDUCCION: J = Sn extraido [kg] - lambda_Fe * FeO reducido [kg] - costos - ...,
       lambda_Fe ~ 0.30 kg Sn a dross por kg FeO reducido (OLS entre batches, IC95% 0.08-0.49).
     - FUSION: J = -lambda_F * Sn extraido [kg] - costos - ..., con lambda_F ~ 0.13 kg de
       metal PERDIDO por kg de Sn extraido prematuramente en Fusion: la fraccion del Sn
       cargado que ya salio de la escoria al fin de Fusion REDUCE la fraccion a metal
       (OLS -0.143, t=-3.7, controlando ley del concentrado, refractario, temperatura y
       FeO reducido) y AUMENTA el dross (+0.085, p=0.004); cada t de Sn conservada en la
       escoria al fin de Fusion suma ~0.13 t de metal (p=0.001). Es el principio clasico
       de dos etapas (fundir oxidando manteniendo el Sn en la escoria; reducirlo despues
       con selectividad controlada). Como el Sn extraido a lo largo del batch es ~fijo
       (Sn cargado - Sn en escoria final ~1 t), "extraer mas" en Fusion no crea Sn, solo
       lo reparte peor. Por eso en Fusion el optimizador mueve SOLO carbon y cal (quimica)
       y deja gases, carga y duracion en el plan del operador (el modelo de DeltaT de
       Fusion es debil para sostener una recomendacion energetica).
   (d) restriccion termica blanda usando el modelo de DeltaT (ventana P5-P95 historica
   por fase); (e) intervalos CV+ calibrados (v2) para Sn, FeO y T; (f) soporte historico
   minimo DURO (P10) ademas de la penalizacion blanda; (g) todo vectorizado (un predict
   por modelo por lote de candidatos: ~0.4 s por escalon).

Uso tipico:

    import modelo_predictivo_v3 as mp3
    df = mp3.construir_dataset_modelo_v3("Datos Lingo smelter fase II.xlsx")
    sistema = mp3.entrenar_sistema_v3(df)
    print(mp3.tabla_validacion_v3(df))                       # OOF + lockbox por fase/target
    reporte = mp3.reporte_recomendaciones_batch_v3(sistema, df, "AP0350")
    print(mp3.resumen_reporte_batch_v3(reporte))

Limitaciones que este modulo NO resuelve (se declaran): la masa de escoria por trazador
CaO esta subestimada en un factor ~constante por batch (el CaO de la ganga del concentrado
no esta cuantificado) y hereda ~2% de ruido relativo del ensayo de %CaO; el sentido fisico
de `posicion_vertical_lanza_mm` sigue sin confirmarse con planta (se usa solo como estado
`_prev`, no como palanca); los costos de reactivos son pesos placeholder sin precios reales;
y el KPI de batch sigue siendo poco predecible (R2 lockbox ~0, Spearman ~0.25-0.40) por
ruido contable y solo 362 batches -- el valor prescriptivo esta en el escalon, con el
eslabon escalon->batch dado por lambda_Fe (dross) y por el balance de Sn en masa.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import dataset_lingo_smelter as dls
import feature_engineering as fe
import modelo_prescriptivo as mp
import modelo_predictivo_v2 as mp2

FASES = mp.FASES

# =============================================================================
# 0) Dataset
# =============================================================================

def construir_dataset_modelo_v3(ruta_excel: str, hoja: str = "Datos") -> pd.DataFrame:
    """Excel -> limpieza -> features base + modelado + v3 (carga por tolva, inventario, targets en masa)."""
    df = dls.construir_dataset(ruta_excel, hoja=hoja, clean_mode=True)
    df = fe.construir_features_v3(df)
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
    # Colas extremas de los targets en masa (por %CaO o %Sn muy bajos): se anulan (0.5%/99.5%
    # por fase), nunca se recortan a un valor -- un target anulado simplemente no entrena.
    for col in ["sn_extraido_est_kg", "frac_sn_extraido_escalon", "feo_extraido_est_kg", "d_sn_inventario_escoria_est_kg"]:
        df[col] = df.groupby("fase_proceso")[col].transform(lambda s: s.where(s.between(*s.quantile([0.005, 0.995]))))
    return df


# =============================================================================
# 1) Roles v3: CONTEXT / STATE / ACTION por fase
# =============================================================================

CONTEXT_V3: dict[str, list[str]] = {
    "Fusión": ["orden_escalon_fase", "tiempo_fase", "espesor_ladrillo_norm_mm", "ley_sn_conc_batch_pct",
               "ley_sn_reciclo_batch_pct", "ley_sn_Fe_batch_pct", "ley_sn_dross_batch_pct", "ley_sn_pellets_batch_pct",
               "ley_sn_carga_batch_pct"],
    "Reducción": ["orden_escalon_fase", "tiempo_fase", "espesor_ladrillo_norm_mm", "ley_sn_carga_batch_pct",
                  "ley_sn_dross_batch_pct", "ley_sn_pellets_batch_pct"],
}

STATE_V3: dict[str, list[str]] = {
    "Fusión": ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev",
               "basicidad_B2_prev", "basicidad_B4_prev", "indice_estructural_molar_prev", "indice_irf_prev",
               "ratio_sn_feo_prev", "ratio_sn_sio2_prev", "interaccion_Sn_x_FeO_prev", "interaccion_FeO_x_B2_prev",
               "temperatura_horno_celsius_prev", "grad_temperatura_horno_celsius_prev", "termocupla_media_celsius_prev",
               "tiro_horno_pct_prev", "posicion_vertical_lanza_mm_prev", "temperatura_gas_pre_bhf_celsius_prev",
               "cum_feed_Sn_kg_prev", "cum_feed_total_kg_prev", "cum_feed_CaO_kg_prev", "cum_feed_Carbon_kg_prev",
               "cum_feed_Fe_kg_prev", "cum_feed_dross_Fe_kg_prev", "cum_feed_conc_kg_prev", "cum_feed_reciclo_kg_prev",
               "cum_feed_pellets_kg_prev", "cum_feed_carga_secundaria_kg_prev", "cum_gn_nm3_prev", "cum_o2_nm3_prev", "cum_aire_nm3_prev",
               "relacion_C_cum_Sn_cum_prev", "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev",
               "feo_inventario_escoria_est_kg_prev", "avance_reduccion_sn_prev"],
    "Reducción": ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev",
                  "basicidad_B2_prev", "basicidad_B4_prev", "indice_estructural_molar_prev", "indice_irf_prev",
                  "ratio_sn_feo_prev", "ratio_sn_sio2_prev", "log_ratio_sn_feo_prev", "interaccion_Sn_x_FeO_prev",
                  "interaccion_FeO_x_B2_prev", "temperatura_horno_celsius_prev", "grad_temperatura_horno_celsius_prev",
                  "termocupla_media_celsius_prev", "tiro_horno_pct_prev", "posicion_vertical_lanza_mm_prev",
                  "temperatura_gas_pre_bhf_celsius_prev", "cum_feed_Sn_kg_prev", "cum_feed_total_kg_prev",
                  "cum_feed_CaO_kg_prev", "cum_feed_Carbon_kg_prev", "cum_gn_nm3_prev", "cum_o2_nm3_prev", "cum_aire_nm3_prev",
                  "relacion_C_cum_Sn_cum_prev", "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev",
                  "feo_inventario_escoria_est_kg_prev", "avance_reduccion_sn_prev"],
}

# Primitivas de ACCION del escalon t (lo que el operador fija). `duracion_plan_min` reemplaza
# a `delta_tiempo` (duracion real, ambigua ACTION/LEAKAGE): el simulador asume que el plan se
# cumple (77.5% de los escalones historicos lo cumplen exactamente).
ACCIONES_PRIMITIVAS_V3: dict[str, list[str]] = {
    "Fusión": ["duracion_plan_min", "tasa_feed_conc_kg_min", "tasa_feed_reciclo_kg_min", "tasa_feed_Fe_kg_min",
               "tasa_feed_dross_Fe_kg_min", "tasa_feed_pellets_kg_min", "tasa_feed_CaO_kg_min", "tasa_feed_Carbon_kg_min",
               "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
    "Reducción": ["duracion_plan_min", "tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min",
                  "tasa_feed_dross_Fe_kg_min", "tasa_feed_pellets_kg_min"],
}
# Subconjunto que el optimizador MUEVE por defecto: quimica/energia/tiempo. La mezcla de carga
# (tolvas 1/3/4/5/7) se trata como plan dado del operador (se puede habilitar via
# `acciones_a_optimizar` en optimizar_accion_v3) -- recomendar "cargar mas" seria trivial
# para un objetivo en kg y no es una decision del escalon sino del plan del batch.
# En FUSION la duracion tampoco se optimiza por defecto: con las tasas de carga fijas, alargar el escalon
# solo "carga mas Sn" (mas Sn extraido en kg por construccion) sin crear Sn -- la carga total del batch
# esta dada por el plan. En REDUCCION (sin carga) la duracion si es una palanca fisica real (mas tiempo de
# reduccion) con su costo de tiempo en J.
ACCIONES_OPTIMIZABLES_V3: dict[str, list[str]] = {
    "Fusión": ["tasa_feed_Carbon_kg_min", "tasa_feed_CaO_kg_min"],  # gases/carga/duracion: plan del operador (ver docstring 5c)
    "Reducción": ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "duracion_plan_min"],
}
# Soporte historico minimo exigido a una recomendacion (restriccion DURA, ademas de la penalizacion blanda
# de J): percentil de la distribucion de support_score de los escalones historicos DEV de esa fase. Una
# recomendacion por debajo se descarta (consideraciones sec. 85: no recomendar cambios agresivos fuera de
# lo observado). Si ningun candidato lo cumple, se devuelve la accion base (historica) sin cambio.
PERCENTIL_SOPORTE_MINIMO = 0.10

# Targets por fase y clave. "sn_kg" es el objetivo principal (kg de Sn extraido de la escoria);
# "sn_pts" se mantiene por continuidad con v1/v2 (misma escala que hallazgos.md); "feo_*" es
# la senal de sobre-reduccion; "dT" la restriccion termica.
TARGETS_V3: dict[str, dict[str, str]] = {
    "Fusión": {"sn_kg": "sn_extraido_est_kg", "sn_pts": "d_ley_sn_escoria_pct", "feo_pts": "d_ley_feo_escoria_pct",
               "dT": "d_temperatura_horno_celsius"},
    "Reducción": {"sn_kg": "sn_extraido_est_kg", "sn_pts": "d_ley_sn_escoria_pct", "feo_kg": "feo_extraido_est_kg",
                  "feo_pts": "d_ley_feo_escoria_pct", "dT": "d_temperatura_horno_celsius"},
}

# Feature sets por fase y target (seleccion experimentos/09: permutacion estable + Boruta-shadow
# + decorrelacion |rho|>0.85 + curva de tamano; ver hallazgos.md 14). Se cargan desde
# FEATURES_V3_POR_DEFECTO y se pueden sobreescribir en entrenar_sistema_v3(features=...).
FEATURES_V3_POR_DEFECTO: dict[str, dict[str, list[str]]] = {
    "Fusión": {
        # curado_B_13 (experimentos/11): R2 OOF 0.722 / lockbox 0.680; el baseline trivial "solo Sn
        # cargado" ya da 0.685 / 0.714 -> la quimica agrega poco R2 pero aporta la RESPUESTA a las
        # palancas (carbon +, cal/carga -, GN +, O2 -), que es lo que necesita el prescriptor.
        "sn_kg": ["feed_Sn_kgf", "sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev",
                  "feo_inventario_escoria_est_kg_prev", "ley_sn_escoria_pct_prev", "interaccion_C_x_Sn_prev",
                  "tasa_feed_Carbon_kg_min", "relacion_CaO_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                  "basicidad_B2_prev", "temperatura_horno_celsius_prev", "frac_carga_secundaria"],
        # v2 (7 features) con duracion_plan_min en lugar de delta_tiempo: R2 OOF 0.517 / lockbox 0.450
        "sn_pts": ["basicidad_B2_prev", "duracion_plan_min", "ley_sn_escoria_pct_prev", "interaccion_C_x_Sn_prev",
                   "cum_feed_CaO_kg_prev", "interaccion_Sn_x_FeO_prev", "tasa_feed_Carbon_kg_min"],
        # informativo (R2 OOF 0.20 / lockbox 0.04): en Fusion el %FeO sube con la carga de Fe y por
        # cierre de composicion; NO se usa como penalizacion en J (ver PESOS_V3_POR_FASE).
        "feo_pts": ["ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_sn_escoria_pct_prev", "tasa_feed_CaO_kg_min",
                    "tasa_feed_Carbon_kg_min", "exceso_o2_combustion_pct", "tasa_feed_Fe_kg_min", "tasa_feed_dross_Fe_kg_min",
                    "duracion_plan_min"],
        # R2 OOF 0.39 / lockbox 0.10 (debil en Fusion): la ventana termica de J rara vez se activa aqui.
        "dT": ["temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "espesor_ladrillo_norm_mm", "tasa_gn_nm3_min",
               "tasa_o2_nm3_min", "tasa_aire_nm3_min", "exceso_o2_combustion_pct", "tasa_feed_total_kg_min",
               "tasa_feed_Carbon_kg_min", "tasa_feed_CaO_kg_min", "duracion_plan_min", "orden_escalon_fase",
               "masa_escoria_est_kg_prev", "grad_temperatura_horno_celsius_prev", "tiro_horno_pct_prev"],
    },
    "Reducción": {
        # curado_A_10: R2 OOF 0.971 / lockbox 0.983 (baseline "solo inventario previo": 0.973 / 0.984 --
        # el valor del set esta en la respuesta a carbon/GN/O2, no en el R2).
        "sn_kg": ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev",
                  "ley_feo_escoria_pct_prev", "tasa_feed_Carbon_kg_min", "interaccion_C_x_Sn_prev", "duracion_plan_min",
                  "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "temperatura_horno_celsius_prev"],
        # Config C (v2) + inventario + duracion planificada: R2 OOF 0.948 / lockbox 0.968
        "sn_pts": mp2.FEATURES_STATE_ACTION_V2["Reducción"] + ["sn_inventario_escoria_est_kg_prev", "duracion_plan_min"],
        # kg de FeO que salieron de la escoria (trazador CaO): R2 OOF 0.29 / lockbox 0.27, MAE ~515 kg
        # (std 814). Ruidoso pero es la UNICA senal de sobre-reduccion en masa: el d%FeO en puntos sube
        # por cierre de composicion cuando sale el Sn (convertirlo a kg da R2 negativo).
        "feo_kg": ["feo_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_feo_escoria_pct_prev",
                   "ley_cao_escoria_pct_prev", "ley_sn_escoria_pct_prev", "sn_inventario_escoria_est_kg_prev",
                   "tasa_feed_Carbon_kg_min", "interaccion_C_x_FeO_prev", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                   "duracion_plan_min"],
        # informativo (R2 OOF 0.47 / lockbox 0.57), continuidad con v1/v2
        "feo_pts": ["ley_feo_escoria_pct_prev", "ley_sn_escoria_pct_prev", "interaccion_Sn_x_FeO_prev", "ratio_sn_sio2_prev",
                    "tasa_feed_Carbon_kg_min", "interaccion_C_x_FeO_prev", "cum_feed_Carbon_kg_prev", "tasa_gn_nm3_min",
                    "exceso_o2_combustion_pct", "duracion_plan_min", "grad_temperatura_horno_celsius_prev"],
        # R2 OOF 0.44 / lockbox 0.32-0.49 (antes -0.04): habilita la ventana termica de J en Reduccion.
        "dT": ["temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "espesor_ladrillo_norm_mm", "tasa_gn_nm3_min",
               "tasa_o2_nm3_min", "tasa_aire_nm3_min", "exceso_o2_combustion_pct", "tasa_feed_Carbon_kg_min", "duracion_plan_min",
               "orden_escalon_fase", "masa_escoria_est_kg_prev", "grad_temperatura_horno_celsius_prev", "tiro_horno_pct_prev",
               "ley_sn_escoria_pct_prev"],
    },
}

for _fase, _por_target in FEATURES_V3_POR_DEFECTO.items():
    for _k, _feats in _por_target.items():
        _inseguras = [f for f in _feats if f not in fe.filtrar_features_seguras(_feats)]
        if _inseguras:
            raise AssertionError(f"FEATURES_V3_POR_DEFECTO[{_fase}][{_k}] contiene features inseguras: {_inseguras}")
del _fase, _por_target, _k, _feats, _inseguras

# Relaciones monotonas locales (consideraciones sec. 84): SOLO carbon, cuyo efecto causal esta
# validado (v2: Reduccion sobre ley; v3: ambas fases sobre kg). Signo segun el target: mas carbon
# -> MAS Sn extraido (kg) / MENOS d_ley_sn (pts) / MAS FeO extraido (kg) / MENOS d_ley_feo.
VARIABLES_CARBON = {"tasa_feed_Carbon_kg_min", "interaccion_C_x_Sn_prev", "interaccion_C_x_FeO_prev",
                    "tasa_feed_Carbon_reduccion_kg_min", "cum_feed_Carbon_kg_prev"}
SIGNO_MONOTONO_CARBON = {"sn_extraido_est_kg": +1, "frac_sn_extraido_escalon": +1, "feo_extraido_est_kg": +1,
                         "d_ley_sn_escoria_pct": -1, "tasa_relativa_sn_escoria": -1, "d_ley_feo_escoria_pct": -1}


def construir_modelo_v3(features: list[str], target_col: str, monotono: bool = True):
    signo = SIGNO_MONOTONO_CARBON.get(target_col, 0) if monotono else 0
    cst = [signo if f in VARIABLES_CARBON else 0 for f in features]
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15,
                                         l2_regularization=1.0, random_state=42, monotonic_cst=cst)


# =============================================================================
# 2) Simulador: primitivas de accion + estado + contexto -> fila completa de features
# =============================================================================

def recalcular_derivadas_v3(fase: str, state: dict, action: dict, context: dict | None = None) -> dict:
    """Reconstruye TODAS las features derivadas de accion (v1/v2/v3) a partir de las
    primitivas de `action`, el `state` (columnas `_prev`/cum/inventario) y el `context`
    (ensayos del batch, escalon, refractario), exactamente como lo haria feature_engineering
    sobre datos reales. Devuelve un dict con state + context + action + derivadas.

    Supuesto explicito: la duracion real del escalon = `duracion_plan_min` (el plan se
    cumple); `delta_tiempo` se rellena con ese valor para las features que lo usan.
    """
    fila = dict(state)
    fila.update(context or {})
    fila.update(action)
    dt = float(action["duracion_plan_min"])
    fila["delta_tiempo"] = dt
    fila["duracion_plan_min"] = dt

    # --- carga solida por corriente (kg del escalon) y Sn fino via ensayo del batch (contexto)
    feeds = {}
    for corriente, (col_h, col_f, col_tasa) in fe.TOLVAS_CARGA.items():
        tasa = float(action.get(col_tasa, state.get(col_tasa, 0.0)) or 0.0)
        fila[col_tasa] = tasa
        feeds[corriente] = tasa * dt
        fila[col_h] = feeds[corriente]
        ley = fila.get(f"ley_sn_{corriente}_batch_pct", np.nan)
        fila[col_f] = feeds[corriente] * (0.0 if pd.isna(ley) else ley) / 100
    feed_total = sum(feeds.values())
    feed_sn = sum(fila[fe.TOLVAS_CARGA[c][1]] for c in fe.TOLVAS_CARGA)
    fila["feed_total_kgh"] = feed_total
    fila["feed_Sn_kgf"] = feed_sn
    fila["tasa_feed_total_kg_min"] = feed_total / dt
    fila["tasa_feed_Sn_kg_min"] = feed_sn / dt
    carga_valida = feed_total if feed_total > fe.UMBRAL_CARGA_KG else np.nan
    for corriente in fe.TOLVAS_CARGA:
        fila[f"frac_{corriente}_carga"] = feeds[corriente] / carga_valida
    secundaria = sum(feeds[c] for c in fe.CORRIENTES_SECUNDARIAS)
    fila["feed_carga_secundaria_kgh"] = secundaria
    fila["frac_carga_secundaria"] = secundaria / carga_valida
    fila["ley_sn_carga_pct"] = 100 * feed_sn / carga_valida

    # --- cal y carbon
    tasa_cao = float(action.get("tasa_feed_CaO_kg_min", state.get("tasa_feed_CaO_kg_min", 0.0)) or 0.0)
    tasa_c = float(action.get("tasa_feed_Carbon_kg_min", 0.0) or 0.0)
    fila["tasa_feed_CaO_kg_min"] = tasa_cao
    fila["tasa_feed_Carbon_kg_min"] = tasa_c
    fila["feed_CaO_kgh"] = tasa_cao * dt
    fila["feed_Carbon_kgh"] = tasa_c * dt
    fila["tasa_feed_Carbon_fusion_kg_min"] = tasa_c if fase == "Fusión" else 0.0
    fila["tasa_feed_Carbon_reduccion_kg_min"] = tasa_c if fase == "Reducción" else 0.0
    fila["relacion_CaO_carga"] = fila["feed_CaO_kgh"] / carga_valida
    carga_fe = feeds["Fe"] + feeds["dross"]
    fila["relacion_CaO_carga_Fe"] = fila["feed_CaO_kgh"] / carga_fe if carga_fe > 1e-6 else np.nan
    fila["relacion_C_Sn_carga"] = fila["feed_Carbon_kgh"] / feed_sn if feed_sn > fe.UMBRAL_SN_FED_KG else np.nan
    carga_ox = feed_sn + carga_fe
    fila["relacion_C_carga"] = fila["feed_Carbon_kgh"] / carga_ox if carga_ox > 1e-6 else np.nan
    fila["exceso_C_estequiometrico_pct"] = (100 * (fila["feed_Carbon_kgh"] - fe.FACTOR_ESTEQ_C_SN * feed_sn) / (fe.FACTOR_ESTEQ_C_SN * feed_sn)
                                            if feed_sn > 1e-6 else np.nan)

    # --- lanza (gases)
    gn, o2, aire = (float(action[k]) for k in ("tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"))
    fila["volumen_gas_natural_inyectado_lanza_escalon_nm3"] = gn * dt
    fila["volumen_o2_inyectado_lanza_escalon_nm3"] = o2 * dt
    fila["volumen_aire_inyectado_lanza_escalon_nm3"] = aire * dt
    o2_teo = fe.RELACION_O2_CH4 * gn
    o2_disp = o2 + fe.FRACCION_O2_EN_AIRE * aire
    fila["o2_teorico_combustion_nm3"] = o2_teo * dt
    fila["o2_disponible_nm3"] = o2_disp * dt
    fila["exceso_o2_combustion_pct"] = 100 * (o2_disp - o2_teo) / o2_teo if o2_teo > 1e-9 else np.nan
    fila["estequiometria_lanza_planta_pct"] = fila["exceso_o2_combustion_pct"] + 100 if pd.notna(fila["exceso_o2_combustion_pct"]) else np.nan
    fila["oxygen_enrichment_pct"] = 100 * o2_disp / (o2 + aire) if (o2 + aire) > 1e-9 else np.nan
    fila["enriquecimiento_o2_lanza_planta_pct"] = fila["oxygen_enrichment_pct"]
    fila["total_process_gas_nm3_min"] = gn + o2 + aire
    fila["indice_severidad_reductora"] = tasa_c / o2_disp if o2_disp > 1e-3 else np.nan
    fila["carga_termica_relativa"] = (gn * dt) / carga_valida
    fila["oxidante_sobre_carga_total"] = (o2_disp * dt) / carga_valida
    lanza = float(action.get("posicion_vertical_lanza_mm", state.get("posicion_vertical_lanza_mm_prev", np.nan)))
    fila["posicion_vertical_lanza_mm"] = lanza
    fila["lanza_pos_x_gas_total"] = lanza * fila["total_process_gas_nm3_min"]
    fila["interaccion_lanza_x_gn"] = lanza * gn

    # --- interacciones accion x estado previo
    sn_prev = state.get("ley_sn_escoria_pct_prev", np.nan)
    feo_prev = state.get("ley_feo_escoria_pct_prev", np.nan)
    t_prev = state.get("temperatura_horno_celsius_prev", np.nan)
    fila["interaccion_C_x_Sn_prev"] = tasa_c * sn_prev
    fila["interaccion_C_x_FeO_prev"] = tasa_c * feo_prev
    fila["interaccion_C_x_temp_prev"] = tasa_c * t_prev
    fila["interaccion_C_x_exceso_o2"] = tasa_c * fila["exceso_o2_combustion_pct"]
    fila["carbon_sobre_inventario_sn_prev"] = tasa_c / sn_prev if (pd.notna(sn_prev) and sn_prev >= fe.UMBRAL_LEY_SN_PREV) else np.nan
    if "interaccion_Sn_x_FeO_prev" not in fila:
        fila["interaccion_Sn_x_FeO_prev"] = sn_prev * feo_prev
    return fila


def fila_a_state_action_context(fase: str, row: pd.Series) -> tuple[dict, dict, dict]:
    """Descompone una fila historica en (state, action, context) segun los roles v3."""
    state = {f: row.get(f, np.nan) for f in STATE_V3[fase]}
    action = {a: row.get(a, np.nan) for a in ACCIONES_PRIMITIVAS_V3[fase]}
    context = {c: row.get(c, np.nan) for c in CONTEXT_V3[fase]}
    return state, action, context


# =============================================================================
# 3) Entrenamiento: ensambles GroupKFold + residuos OOF (CV+) por target
# =============================================================================

@dataclass
class SistemaPrescriptivoV3:
    features: dict = field(default_factory=dict)        # fase -> {clave_target: [features]}
    ensambles: dict = field(default_factory=dict)       # (fase, clave_target) -> list[modelo]
    residuos_oof: dict = field(default_factory=dict)    # (fase, clave_target) -> list[np.ndarray]
    soporte: dict = field(default_factory=dict)         # fase -> soporte historico (kNN) sobre features de sn_kg
    limites_accion: dict = field(default_factory=dict)  # fase -> {primitiva: (p1, p99)}
    ventana_temperatura: dict = field(default_factory=dict)  # fase -> (p5, p95) de temperatura_horno_celsius
    soporte_minimo: dict = field(default_factory=dict)       # fase -> umbral duro de support_score (PERCENTIL_SOPORTE_MINIMO historico)
    batches_dev: set = field(default_factory=set)
    batches_lockbox: set = field(default_factory=set)
    n_entrenamiento: dict = field(default_factory=dict)


def entrenar_sistema_v3(df: pd.DataFrame, features: dict | None = None, pct_lockbox: float = mp.N_LOCKBOX_PCT,
                        monotono: bool = True) -> SistemaPrescriptivoV3:
    """Entrena, SOLO con batches DEV, un ensamble GroupKFold por (fase, target) + residuos OOF
    (CV+), soporte historico (kNN sobre el feature set de sn_kg) y limites P1-P99 de cada
    primitiva de accion. El lockbox queda intacto para `tabla_validacion_v3`."""
    features = features or FEATURES_V3_POR_DEFECTO
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df, pct_lockbox)
    sistema = SistemaPrescriptivoV3(features=features, batches_dev=batches_dev, batches_lockbox=batches_lockbox)
    for fase in FASES:
        sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        for clave, target_col in TARGETS_V3[fase].items():
            feats = features[fase][clave]
            d = sub.dropna(subset=[target_col])
            modelos, residuos = [], []
            for tr, te in GroupKFold(n_splits=mp.N_SPLITS_OOF).split(d[feats], d[target_col], d["Batch"]):
                m = construir_modelo_v3(feats, target_col, monotono)
                m.fit(d[feats].iloc[tr], d[target_col].iloc[tr])
                modelos.append(m)
                residuos.append(np.abs(d[target_col].iloc[te].to_numpy() - m.predict(d[feats].iloc[te])))
            sistema.ensambles[(fase, clave)] = modelos
            sistema.residuos_oof[(fase, clave)] = residuos
            sistema.n_entrenamiento[(fase, clave)] = len(d)
        feats_sop = [f for f in features[fase]["sn_kg"]]
        sub_sop = sub.dropna(subset=feats_sop)
        sistema.soporte[fase] = mp._construir_soporte_historico(sub_sop, feats_sop)
        scores_hist = _support_scores_lote(sistema.soporte[fase], sub_sop)
        sistema.soporte_minimo[fase] = float(np.quantile(scores_hist, PERCENTIL_SOPORTE_MINIMO))
        acciones = sub[ACCIONES_PRIMITIVAS_V3[fase]].dropna()
        sistema.limites_accion[fase] = {a: (float(acciones[a].quantile(0.01)), float(acciones[a].quantile(0.99)))
                                        for a in ACCIONES_PRIMITIVAS_V3[fase]}
        t = sub["temperatura_horno_celsius"].dropna()
        sistema.ventana_temperatura[fase] = (float(t.quantile(0.05)), float(t.quantile(0.95)))
    return sistema


def _predecir_lote(sistema: SistemaPrescriptivoV3, fase: str, clave: str, X: pd.DataFrame, alpha: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prediccion (media del ensamble) + intervalo CV+ para MUCHAS filas a la vez. Misma definicion
    que `modelo_predictivo_v2.intervalo_cv_plus` (para cada modelo k y cada residuo OOF r del fold que
    ese modelo no vio: pred_k -/+ r; percentiles alpha/2 y 1-alpha/2 de la nube combinada), pero
    vectorizada: un predict por modelo para todas las filas, y los cuantiles por fila sobre una matriz
    (n_filas x total_residuos). Es lo que hace viable el optimizador y la validacion de cobertura."""
    feats = sistema.features[fase][clave]
    modelos = sistema.ensambles[(fase, clave)]
    residuos = sistema.residuos_oof[(fase, clave)]
    P = np.column_stack([m.predict(X[feats]) for m in modelos])          # (n, k)
    pred = P.mean(axis=1)
    lo_nube = np.concatenate([P[:, [k]] - residuos[k][None, :] for k in range(len(modelos))], axis=1)
    hi_nube = np.concatenate([P[:, [k]] + residuos[k][None, :] for k in range(len(modelos))], axis=1)
    lo = np.quantile(lo_nube, alpha / 2, axis=1)
    hi = np.quantile(hi_nube, 1 - alpha / 2, axis=1)
    return pred, lo, hi


def _predecir(sistema: SistemaPrescriptivoV3, fase: str, clave: str, fila: dict, alpha: float = 0.20) -> tuple[float, float, float]:
    feats = sistema.features[fase][clave]
    x = pd.DataFrame([[fila.get(f, np.nan) for f in feats]], columns=feats)
    pred, lo, hi = _predecir_lote(sistema, fase, clave, x, alpha)
    return float(pred[0]), float(lo[0]), float(hi[0])


def _support_scores_lote(soporte_fase: dict, X: pd.DataFrame) -> np.ndarray:
    feats = soporte_fase["features"]
    xs = soporte_fase["scaler"].transform(X[feats])
    dist, _ = soporte_fase["nn"].kneighbors(xs, n_neighbors=soporte_fase["k"])
    return np.exp(-dist.mean(axis=1) / soporte_fase["dist_ref_media"])


# =============================================================================
# 4) Simulador contrafactual y funcion objetivo en kg de Sn
# =============================================================================

# kg de Sn que terminan en dross de Fe por cada kg de FeO reducido en Reduccion (OLS entre
# batches: pendiente 0.28-0.31, IC95% [0.08, 0.49]; ver hallazgos.md 14). Es el "precio" fisico
# de sobre-reducir, en las MISMAS unidades que el objetivo.
LAMBDA_FE_KG_SN_POR_KG_FEO = 0.30

# kg de metal final perdidos por cada kg de Sn extraido PREMATURAMENTE de la escoria durante Fusion
# (OLS entre batches: metal_t ~ 0.128 * Sn_en_escoria_al_fin_de_Fusion_t + 0.51 * feed_Sn_t, p=0.001;
# f_metal ~ -0.143 * avance_reduccion_al_fin_de_Fusion con controles, t=-3.7). Ver hallazgos.md 14.6.
LAMBDA_F_KG_METAL_POR_KG_EXTRAIDO_FUSION = 0.13

PESOS_V3_DEFAULT = dict(
    w_sn=1.0,                     # kg Sn extraido de la escoria (a metal o polvo)
    lambda_fe=LAMBDA_FE_KG_SN_POR_KG_FEO,  # kg Sn perdido a dross por kg FeO reducido (solo Reduccion, ver PESOS_V3_POR_FASE)
    costo_carbon_kgSn_por_kg=0.02,  # placeholders sin precios reales: 1 kg C ~ 0.02 kg Sn
    costo_gn_kgSn_por_nm3=0.01,
    costo_o2_kgSn_por_nm3=0.005,
    costo_tiempo_kgSn_por_min=0.5,  # costo de oportunidad del tiempo de horno
    w_temp=2.0,                     # kg Sn por grado fuera de la ventana P5-P95
    w_incertidumbre=0.25,           # kg Sn por kg de ancho del intervalo CV+ de Sn
    w_soporte=2000.0,               # kg Sn por (1 - support_score): perder 0.2 de soporte cuesta 400 kg (~1 sd de efecto de palanca)
)


# Ajustes por fase sobre PESOS_V3_DEFAULT (todo en kg de metal equivalente):
# - FUSION: w_sn = -lambda_F (extraer Sn prematuramente CUESTA metal, ver docstring 5c) y lambda_fe = 0
#   (regimen oxidante, +19% O2 en el 100% de los batches; la unica senal de FeO -d%FeO en puntos- esta
#   dominada por la entrada de Fe con la carga y el cierre de composicion). La prediccion de FeO se
#   sigue reportando como informacion.
# - REDUCCION: w_sn = 1 (el Sn extraido va a metal) y lambda_fe = 0.30.
PESOS_V3_POR_FASE = {"Fusión": dict(w_sn=-LAMBDA_F_KG_METAL_POR_KG_EXTRAIDO_FUSION, lambda_fe=0.0), "Reducción": dict(w_sn=1.0)}


def simulate_action_v3(sistema: SistemaPrescriptivoV3, fase: str, state: dict, action: dict, context: dict | None = None,
                       alpha: float = 0.20) -> dict:
    """state + action (+context) -> Sn extraido [kg], FeO reducido [kg], DeltaT [C], todos con
    intervalo CV+ (nominal 1-alpha), mas support_score y las derivadas usadas."""
    fila = recalcular_derivadas_v3(fase, state, action, context)
    sn_kg, sn_lo, sn_hi = _predecir(sistema, fase, "sn_kg", fila, alpha)
    sn_pts, sn_pts_lo, sn_pts_hi = _predecir(sistema, fase, "sn_pts", fila, alpha)
    masa_prev = float(state.get("masa_escoria_est_kg_prev", np.nan))
    if fase == "Reducción":
        feo_kg, feo_lo, feo_hi = _predecir(sistema, fase, "feo_kg", fila, alpha)
        feo_pts, _, _ = _predecir(sistema, fase, "feo_pts", fila, alpha)
    else:
        feo_pts, feo_pts_lo, feo_pts_hi = _predecir(sistema, fase, "feo_pts", fila, alpha)
        # kg de FeO que dejan la escoria ~ -d%FeO * masa_prev / 100 (aprox. de primer orden; en
        # Fusion entra FeO con la carga y el signo suele ser negativo = FeO en aumento)
        f = -masa_prev / 100 if pd.notna(masa_prev) else np.nan
        feo_kg, feo_lo, feo_hi = feo_pts * f, min(feo_pts_lo * f, feo_pts_hi * f), max(feo_pts_lo * f, feo_pts_hi * f)
    dT, dT_lo, dT_hi = _predecir(sistema, fase, "dT", fila, alpha)
    t_prev = float(state.get("temperatura_horno_celsius_prev", np.nan))
    return {
        "fase": fase,
        "pred_sn_extraido_kg": sn_kg, "intervalo_sn_kg": (sn_lo, sn_hi),
        "pred_d_ley_sn_pts": sn_pts, "intervalo_d_ley_sn_pts": (sn_pts_lo, sn_pts_hi),
        "pred_feo_reducido_kg": max(0.0, feo_kg) if pd.notna(feo_kg) else np.nan, "pred_feo_extraido_kg": feo_kg,
        "intervalo_feo_kg": (feo_lo, feo_hi), "pred_d_ley_feo_pts": feo_pts,
        "pred_dT": dT, "intervalo_dT": (dT_lo, dT_hi), "pred_temperatura": t_prev + dT if pd.notna(t_prev) else np.nan,
        "support_score": mp.calcular_support_score(sistema.soporte[fase], fila),
        "derivadas": {k: fila[k] for k in ("exceso_o2_combustion_pct", "oxygen_enrichment_pct", "relacion_C_Sn_carga",
                                           "relacion_CaO_carga", "ley_sn_carga_pct", "frac_carga_secundaria") if k in fila},
    }


def objetivo_v3(sistema: SistemaPrescriptivoV3, fase: str, state: dict, action: dict, context: dict | None = None,
                pesos: dict | None = None) -> tuple[float, dict]:
    """J [kg Sn] = w_sn * Sn_extraido - lambda_fe * FeO_reducido - costos(C, GN, O2, tiempo)
                   - w_temp * exceso fuera de la ventana termica - w_incert * ancho CV+ (Sn)
                   - w_soporte * (1 - support_score).
    Mismas unidades en ambas fases; los pesos son parametros de negocio explicitos."""
    p = dict(PESOS_V3_DEFAULT, **PESOS_V3_POR_FASE.get(fase, {}), **(pesos or {}))
    sim = simulate_action_v3(sistema, fase, state, action, context)
    dt = float(action["duracion_plan_min"])
    costo = (p["costo_carbon_kgSn_por_kg"] * action["tasa_feed_Carbon_kg_min"] * dt
             + p["costo_gn_kgSn_por_nm3"] * action["tasa_gn_nm3_min"] * dt
             + p["costo_o2_kgSn_por_nm3"] * action["tasa_o2_nm3_min"] * dt
             + p["costo_tiempo_kgSn_por_min"] * dt)
    t_min, t_max = sistema.ventana_temperatura[fase]
    t_pred = sim["pred_temperatura"]
    fuera = 0.0 if pd.isna(t_pred) else max(0.0, t_min - t_pred) + max(0.0, t_pred - t_max)
    feo_pen = 0.0 if pd.isna(sim["pred_feo_reducido_kg"]) else sim["pred_feo_reducido_kg"]
    ancho = sim["intervalo_sn_kg"][1] - sim["intervalo_sn_kg"][0]
    J = (p["w_sn"] * sim["pred_sn_extraido_kg"] - p["lambda_fe"] * feo_pen - costo
         - p["w_temp"] * fuera - p["w_incertidumbre"] * ancho - p["w_soporte"] * (1 - sim["support_score"]))
    sim["J"] = J
    sim["componentes_J"] = dict(sn=p["w_sn"] * sim["pred_sn_extraido_kg"], feo=-p["lambda_fe"] * feo_pen, costo=-costo,
                                temperatura=-p["w_temp"] * fuera, incertidumbre=-p["w_incertidumbre"] * ancho,
                                soporte=-p["w_soporte"] * (1 - sim["support_score"]))
    return J, sim


def simular_acciones_lote_v3(sistema: SistemaPrescriptivoV3, fase: str, state: dict, acciones: list[dict],
                             context: dict | None = None, alpha: float = 0.20) -> pd.DataFrame:
    """Version por LOTE de `simulate_action_v3`: evalua muchas acciones candidatas sobre el mismo estado con
    un solo predict por modelo (columnas: pred_sn_extraido_kg, sn_lo, sn_hi, pred_feo_reducido_kg,
    pred_dT, pred_temperatura, support_score, ...)."""
    filas = [recalcular_derivadas_v3(fase, state, a, context) for a in acciones]
    todas = sorted({f for k in sistema.features[fase] for f in sistema.features[fase][k]} | set(sistema.soporte[fase]["features"]))
    X = pd.DataFrame([[f.get(c, np.nan) for c in todas] for f in filas], columns=todas)
    out = pd.DataFrame(index=range(len(acciones)))
    sn, sn_lo, sn_hi = _predecir_lote(sistema, fase, "sn_kg", X, alpha)
    out["pred_sn_extraido_kg"], out["sn_lo"], out["sn_hi"] = sn, sn_lo, sn_hi
    out["pred_d_ley_sn_pts"] = _predecir_lote(sistema, fase, "sn_pts", X, alpha)[0]
    if fase == "Reducción":
        feo, feo_lo, feo_hi = _predecir_lote(sistema, fase, "feo_kg", X, alpha)
        out["pred_d_ley_feo_pts"] = _predecir_lote(sistema, fase, "feo_pts", X, alpha)[0]
    else:
        feo_pts, feo_pts_lo, feo_pts_hi = _predecir_lote(sistema, fase, "feo_pts", X, alpha)
        masa_prev = float(state.get("masa_escoria_est_kg_prev", np.nan))
        f = -masa_prev / 100 if pd.notna(masa_prev) else np.nan
        feo, feo_lo, feo_hi = feo_pts * f, np.minimum(feo_pts_lo * f, feo_pts_hi * f), np.maximum(feo_pts_lo * f, feo_pts_hi * f)
        out["pred_d_ley_feo_pts"] = feo_pts
    out["pred_feo_extraido_kg"], out["feo_lo"], out["feo_hi"] = feo, feo_lo, feo_hi
    out["pred_feo_reducido_kg"] = np.where(np.isnan(feo), np.nan, np.maximum(0.0, feo))
    dT, dT_lo, dT_hi = _predecir_lote(sistema, fase, "dT", X, alpha)
    t_prev = float(state.get("temperatura_horno_celsius_prev", np.nan))
    out["pred_dT"], out["dT_lo"], out["dT_hi"] = dT, dT_lo, dT_hi
    out["pred_temperatura"] = t_prev + dT if pd.notna(t_prev) else np.nan
    out["support_score"] = _support_scores_lote(sistema.soporte[fase], X)
    for k in ("exceso_o2_combustion_pct", "oxygen_enrichment_pct", "relacion_C_Sn_carga", "relacion_CaO_carga", "ley_sn_carga_pct", "frac_carga_secundaria"):
        out[f"derivada__{k}"] = [f.get(k, np.nan) for f in filas]
    return out


def objetivo_lote_v3(sistema: SistemaPrescriptivoV3, fase: str, state: dict, acciones: list[dict], context: dict | None = None,
                     pesos: dict | None = None) -> pd.DataFrame:
    """`objetivo_v3` para un lote de acciones (misma formula de J, calculada por filas)."""
    p = dict(PESOS_V3_DEFAULT, **PESOS_V3_POR_FASE.get(fase, {}), **(pesos or {}))
    sim = simular_acciones_lote_v3(sistema, fase, state, acciones, context)
    A = pd.DataFrame(acciones)
    dt = A["duracion_plan_min"].to_numpy(dtype=float)
    costo = (p["costo_carbon_kgSn_por_kg"] * A["tasa_feed_Carbon_kg_min"].to_numpy(dtype=float) * dt
             + p["costo_gn_kgSn_por_nm3"] * A["tasa_gn_nm3_min"].to_numpy(dtype=float) * dt
             + p["costo_o2_kgSn_por_nm3"] * A["tasa_o2_nm3_min"].to_numpy(dtype=float) * dt
             + p["costo_tiempo_kgSn_por_min"] * dt)
    t_min, t_max = sistema.ventana_temperatura[fase]
    t_pred = sim["pred_temperatura"].to_numpy(dtype=float)
    fuera = np.where(np.isnan(t_pred), 0.0, np.maximum(0.0, t_min - t_pred) + np.maximum(0.0, t_pred - t_max))
    feo_pen = np.nan_to_num(sim["pred_feo_reducido_kg"].to_numpy(dtype=float), nan=0.0)
    ancho = (sim["sn_hi"] - sim["sn_lo"]).to_numpy()
    sim["J_sn"] = p["w_sn"] * sim["pred_sn_extraido_kg"]
    sim["J_feo"] = -p["lambda_fe"] * feo_pen
    sim["J_costo"] = -costo
    sim["J_temperatura"] = -p["w_temp"] * fuera
    sim["J_incertidumbre"] = -p["w_incertidumbre"] * ancho
    sim["J_soporte"] = -p["w_soporte"] * (1 - sim["support_score"])
    sim["J"] = sim[["J_sn", "J_feo", "J_costo", "J_temperatura", "J_incertidumbre", "J_soporte"]].sum(axis=1)
    return sim


def _sim_fila_a_dict(fase: str, fila: pd.Series) -> dict:
    return {"fase": fase, "pred_sn_extraido_kg": float(fila["pred_sn_extraido_kg"]), "intervalo_sn_kg": (float(fila["sn_lo"]), float(fila["sn_hi"])),
            "pred_d_ley_sn_pts": float(fila["pred_d_ley_sn_pts"]), "pred_feo_reducido_kg": float(fila["pred_feo_reducido_kg"]),
            "pred_feo_extraido_kg": float(fila["pred_feo_extraido_kg"]), "intervalo_feo_kg": (float(fila["feo_lo"]), float(fila["feo_hi"])),
            "pred_d_ley_feo_pts": float(fila["pred_d_ley_feo_pts"]), "pred_dT": float(fila["pred_dT"]), "intervalo_dT": (float(fila["dT_lo"]), float(fila["dT_hi"])),
            "pred_temperatura": float(fila["pred_temperatura"]), "support_score": float(fila["support_score"]),
            "J": float(fila["J"]) if "J" in fila else np.nan,
            "componentes_J": {k[2:]: float(fila[k]) for k in ("J_sn", "J_feo", "J_costo", "J_temperatura", "J_incertidumbre", "J_soporte") if k in fila},
            "derivadas": {k.replace("derivada__", ""): float(fila[k]) for k in fila.index if k.startswith("derivada__")}}


def optimizar_accion_v3(sistema: SistemaPrescriptivoV3, fase: str, state: dict, accion_base: dict, context: dict | None = None,
                        acciones_a_optimizar: list[str] | None = None, n_candidatos: int = 300, n_refinamiento: int = 60,
                        semilla: int = 0, pesos: dict | None = None) -> tuple[dict, float, dict]:
    """Busqueda aleatoria en P1-P99 historico de cada primitiva optimizable (el resto se mantiene en
    `accion_base`, p.ej. la mezcla de carga planificada) + refinamiento local gaussiano alrededor del mejor
    candidato; ambos por LOTE (un predict por modelo). El termino de soporte de J ya penaliza salir de la
    region observada; los percentiles solo evitan candidatos fisicamente absurdos."""
    optimizables = acciones_a_optimizar or ACCIONES_OPTIMIZABLES_V3[fase]
    limites = {a: sistema.limites_accion[fase][a] for a in optimizables}
    rng = np.random.default_rng(semilla)
    candidatos = [dict(accion_base)]  # la accion historica/base siempre es candidata
    for _ in range(n_candidatos):
        cand = dict(accion_base)
        for a, (lo, hi) in limites.items():
            cand[a] = float(rng.uniform(lo, hi))
        candidatos.append(cand)
    sim = objetivo_lote_v3(sistema, fase, state, candidatos, context, pesos)
    umbral = sistema.soporte_minimo.get(fase, 0.0)
    J_adm = sim["J"].where(sim["support_score"] >= umbral)
    J_adm.iloc[0] = sim["J"].iloc[0]  # la accion base siempre es admisible (es lo que se hizo / se planea)
    i_best = int(J_adm.idxmax())
    mejor_accion, mejor_J, mejor_fila = candidatos[i_best], float(sim.loc[i_best, "J"]), sim.loc[i_best]
    if n_refinamiento > 0:
        refin = []
        for _ in range(n_refinamiento):
            cand = dict(mejor_accion)
            for a, (lo, hi) in limites.items():
                cand[a] = float(np.clip(cand[a] + rng.normal(0, 0.08 * (hi - lo)), lo, hi))
            refin.append(cand)
        sim_r = objetivo_lote_v3(sistema, fase, state, refin, context, pesos)
        J_r = sim_r["J"].where(sim_r["support_score"] >= umbral)
        if J_r.notna().any():
            j = int(J_r.idxmax())
            if float(sim_r.loc[j, "J"]) > mejor_J:
                mejor_accion, mejor_J, mejor_fila = refin[j], float(sim_r.loc[j, "J"]), sim_r.loc[j]
    return mejor_accion, mejor_J, _sim_fila_a_dict(fase, mejor_fila)


# =============================================================================
# 5) Validacion (OOF dev + lockbox) por fase y target, con escalas fisicas comunes
# =============================================================================

def tabla_validacion_v3(df: pd.DataFrame, features: dict | None = None, monotono: bool = True) -> pd.DataFrame:
    """R2/MAE OOF (GroupKFold por Batch, DEV) y lockbox por (fase, target). Para los targets de
    ley se agrega la escala fisica comun a v1/v2 (puntos de %Sn); para sn_kg se agrega el
    equivalente en puntos de %Sn (-kg/masa_prev*100) para comparar contra el modelo de ley."""
    features = features or FEATURES_V3_POR_DEFECTO
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    filas = []
    for fase in FASES:
        sub = df_base.loc[df_base["fase_proceso"] == fase]
        for clave, target_col in TARGETS_V3[fase].items():
            feats = features[fase][clave]
            cols = list(dict.fromkeys(feats + [target_col, "Batch", "masa_escoria_est_kg_prev", "feed_Sn_kgf", "d_ley_sn_escoria_pct"]))
            d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=[target_col]).reset_index(drop=True)
            d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=[target_col]).reset_index(drop=True)
            oof = np.full(len(d_dev), np.nan)
            for tr, te in GroupKFold(n_splits=mp.N_SPLITS_OOF).split(d_dev[feats], d_dev[target_col], d_dev["Batch"]):
                m = construir_modelo_v3(feats, target_col, monotono)
                m.fit(d_dev[feats].iloc[tr], d_dev[target_col].iloc[tr])
                oof[te] = m.predict(d_dev[feats].iloc[te])
            m = construir_modelo_v3(feats, target_col, monotono)
            m.fit(d_dev[feats], d_dev[target_col])
            p_lb = m.predict(d_lb[feats])
            fila = {"fase": fase, "target": clave, "columna": target_col, "n_features": len(feats), "n_dev": len(d_dev), "n_lockbox": len(d_lb),
                    "r2_oof": r2_score(d_dev[target_col], oof), "mae_oof": mean_absolute_error(d_dev[target_col], oof),
                    "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb),
                    "std_target_lockbox": float(d_lb[target_col].std())}
            if clave == "sn_kg" and fase == "Reducción":
                # equivalente en puntos de %Sn (solo Reduccion, donde la masa de escoria casi no cambia
                # dentro del escalon): d%Sn ~ (feed_Sn - sn_extraido)/masa_prev*100. En Fusion la carga
                # cambia la masa en cada escalon y esta aproximacion no es valida (se deja NaN).
                ok = d_lb["masa_escoria_est_kg_prev"].gt(1000) & d_lb["d_ley_sn_escoria_pct"].notna()
                eq = (d_lb["feed_Sn_kgf"] - p_lb) / d_lb["masa_escoria_est_kg_prev"] * 100
                fila["r2_lockbox_equiv_dSn_pts"] = r2_score(d_lb.loc[ok, "d_ley_sn_escoria_pct"], eq[ok]) if ok.sum() > 10 else np.nan
            filas.append(fila)
    return pd.DataFrame(filas).set_index(["fase", "target"])


def validar_cobertura_lockbox_v3(df: pd.DataFrame, sistema: SistemaPrescriptivoV3, fase: str, clave: str = "sn_kg",
                                 alpha: float = 0.20) -> dict:
    """Cobertura empirica del intervalo CV+ en el lockbox (nunca usado para calibrar), vectorizada."""
    df_base = mp.dataset_base_modelo(df)
    target_col = TARGETS_V3[fase][clave]
    feats = sistema.features[fase][clave]
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(sistema.batches_lockbox)].dropna(subset=[target_col])
    _, lo, hi = _predecir_lote(sistema, fase, clave, sub[feats], alpha)
    y = sub[target_col].to_numpy()
    return {"fase": fase, "target": clave, "nominal": 1 - alpha, "cobertura_empirica": float(np.mean((lo <= y) & (y <= hi))),
            "ancho_medio": float(np.mean(hi - lo)), "n": len(sub)}


# =============================================================================
# 6) De escalon a batch: reporte de recomendaciones y eslabon con el rendimiento
# =============================================================================

def reporte_recomendaciones_batch_v3(sistema: SistemaPrescriptivoV3, df: pd.DataFrame, batch_id: str,
                                     acciones_a_optimizar: dict | None = None, n_candidatos: int = 300,
                                     semilla: int = 0, pesos: dict | None = None) -> pd.DataFrame:
    """Recomendacion escalon a escalon para UN batch: en cada escalon se parte del ESTADO REAL
    observado (no encadenado, misma razon que v1: no hay modelos validados para encadenar
    basicidad/SiO2/CaO), se compara la accion historica con la recomendada sobre el mismo
    estado, y se reportan Sn extraido [kg], FeO reducido [kg], T predicha, soporte y J."""
    df_base = mp.dataset_base_modelo(df)
    sub = df_base.loc[df_base["Batch"] == batch_id].sort_values("fecha_inicio")
    if sub.empty:
        raise ValueError(f"Batch '{batch_id}' sin escalones modelables.")
    filas = []
    for _, row in sub.iterrows():
        fase = row["fase_proceso"]
        state, accion_hist, context = fila_a_state_action_context(fase, row)
        if any(pd.isna(v) for v in accion_hist.values()) or pd.isna(state.get("ley_sn_escoria_pct_prev")):
            continue
        opt = (acciones_a_optimizar or {}).get(fase)
        J_h, sim_h = objetivo_v3(sistema, fase, state, accion_hist, context, pesos)
        accion_opt, J_o, sim_o = optimizar_accion_v3(sistema, fase, state, accion_hist, context, opt, n_candidatos, semilla=semilla, pesos=pesos)
        fila = {"Batch": batch_id, "fase": fase, "orden_escalon_fase": row["orden_escalon_fase"],
                "ley_sn_escoria_pct_prev": state["ley_sn_escoria_pct_prev"], "sn_inventario_escoria_est_kg_prev": state.get("sn_inventario_escoria_est_kg_prev"),
                "sn_extraido_real_kg": row.get("sn_extraido_est_kg"), "d_ley_sn_real_pts": row.get("d_ley_sn_escoria_pct"),
                "feo_extraido_real_kg": row.get("feo_extraido_est_kg"), "temperatura_real": row.get("temperatura_horno_celsius"),
                "pred_sn_kg_historico": sim_h["pred_sn_extraido_kg"], "pred_sn_kg_recomendado": sim_o["pred_sn_extraido_kg"],
                "pred_feo_reducido_kg_historico": sim_h["pred_feo_reducido_kg"], "pred_feo_reducido_kg_recomendado": sim_o["pred_feo_reducido_kg"],
                "pred_T_historico": sim_h["pred_temperatura"], "pred_T_recomendado": sim_o["pred_temperatura"],
                "support_historico": sim_h["support_score"], "support_recomendado": sim_o["support_score"],
                "J_historico": J_h, "J_recomendado": J_o, "uplift_J_kgSn": J_o - J_h}
        for a in ACCIONES_PRIMITIVAS_V3[fase]:
            fila[f"hist__{a}"] = accion_hist[a]
            fila[f"rec__{a}"] = accion_opt[a]
        for k, v in sim_o["derivadas"].items():
            fila[f"rec_derivada__{k}"] = v
        filas.append(fila)
    rep = pd.DataFrame(filas)
    if rep.empty:
        warnings.warn(f"Batch '{batch_id}': ningun escalon con insumos completos.")
    return rep


def resumen_reporte_batch_v3(reporte: pd.DataFrame, lambda_fe: float = LAMBDA_FE_KG_SN_POR_KG_FEO,
                             lambda_f: float = LAMBDA_F_KG_METAL_POR_KG_EXTRAIDO_FUSION) -> dict:
    """Agrega el reporte a nivel batch en kg de METAL EQUIVALENTE (recomendado - historico, cada escalon
    evaluado sobre su propio estado real, sin encadenar):
        metal_equivalente_kg = -lambda_f * dSn_extraido_Fusion + 1.0 * dSn_extraido_Reduccion
                               - lambda_fe * dFeO_reducido_Reduccion
    Signos: en Fusion un dSn_extraido NEGATIVO es favorable (menos extraccion prematura); en Reduccion
    un dSn_extraido POSITIVO es favorable y un dFeO_reducido NEGATIVO es favorable (menos dross).
    NO es la suma de "Sn extraido" de ambas fases (ese total es ~fijo en el batch) y es un uplift
    ESTIMADO por el modelo, nunca observado en produccion."""
    if reporte.empty:
        return {}
    fus = reporte["fase"] == "Fusión"; red = reporte["fase"] == "Reducción"
    d_sn_f = float((reporte.loc[fus, "pred_sn_kg_recomendado"] - reporte.loc[fus, "pred_sn_kg_historico"]).sum())
    d_sn_r = float((reporte.loc[red, "pred_sn_kg_recomendado"] - reporte.loc[red, "pred_sn_kg_historico"]).sum())
    d_feo_r = float((reporte.loc[red, "pred_feo_reducido_kg_recomendado"] - reporte.loc[red, "pred_feo_reducido_kg_historico"]).sum())
    return {"n_escalones_evaluados": int(len(reporte)),
            "fusion_sn_extraido_adicional_kg": d_sn_f, "reduccion_sn_extraido_adicional_kg": d_sn_r,
            "reduccion_feo_reducido_adicional_kg": d_feo_r,
            "metal_equivalente_kg": -lambda_f * d_sn_f + d_sn_r - lambda_fe * d_feo_r,
            "uplift_J_total_kgSn": float(reporte["uplift_J_kgSn"].sum()),
            "pct_escalones_con_mejora": float((reporte["uplift_J_kgSn"] > 0).mean() * 100),
            "support_medio_historico": float(reporte["support_historico"].mean()),
            "support_medio_recomendado": float(reporte["support_recomendado"].mean())}


def evaluar_politica_en_lockbox_v3(sistema: SistemaPrescriptivoV3, df: pd.DataFrame, n_batches: int = 10, semilla: int = 0,
                                   n_candidatos: int = 150) -> pd.DataFrame:
    """Aplica `reporte_recomendaciones_batch_v3` a los primeros `n_batches` del lockbox y devuelve
    el resumen por batch (uplift estimado por el modelo, soporte). Sirve para auditar que las
    recomendaciones no se apoyan en regiones sin soporte historico."""
    orden = [b for b in mp.orden_cronologico_batches(df) if b in sistema.batches_lockbox][:n_batches]
    filas = []
    for b in orden:
        rep = reporte_recomendaciones_batch_v3(sistema, df, b, n_candidatos=n_candidatos, semilla=semilla)
        r = resumen_reporte_batch_v3(rep)
        r["Batch"] = b
        r["rendimiento_ha_batch"] = float(df.loc[df["Batch"] == b, "rendimiento_ha_batch"].iloc[0])
        filas.append(r)
    return pd.DataFrame(filas).set_index("Batch")


# =============================================================================
# 7) Efecto de las acciones dado el estado: DML conjunto (causal, promedio) + sensibilidad
#    local del modelo (finita, en el estado concreto del escalon)
# =============================================================================

PALANCAS_EXPLICATIVAS_V3: dict[str, list[str]] = {
    "Fusión": ["tasa_feed_Carbon_kg_min", "tasa_feed_CaO_kg_min", "tasa_feed_total_kg_min", "tasa_gn_nm3_min",
               "tasa_o2_nm3_min", "tasa_aire_nm3_min", "duracion_plan_min"],
    "Reducción": ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "duracion_plan_min"],
}


def _oof_hgb(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    oof = np.full(len(X), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=seed)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def efectos_marginales_palancas_v3(df: pd.DataFrame, fase: str, target_col: str | None = None,
                                   palancas: list[str] | None = None, state: list[str] | None = None,
                                   n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Efecto causal PROMEDIO de cada palanca sobre el target, controlando el estado v3 y las
    DEMAS palancas a la vez (modelo parcialmente lineal multivariado, Robinson 1988, con
    cross-fitting GroupKFold por Batch): y - E[y|S] = sum_j theta_j (a_j - E[a_j|S]) + e.
    Devuelve theta por unidad y por 1 desviacion estandar de cada palanca, con IC95% por
    bootstrap cluster por batch. Complementa (no reemplaza) el modelo no lineal: es la lectura
    "cuanto Sn extra por unidad de palanca, en promedio, dado el estado" que pide un
    prescriptor operacional, con signo e incertidumbre explicitos."""
    target_col = target_col or TARGETS_V3[fase]["sn_kg"]
    palancas = palancas or PALANCAS_EXPLICATIVAS_V3[fase]
    state = state or [s for s in STATE_V3[fase] + CONTEXT_V3[fase] if s not in palancas]
    df_base = mp.dataset_base_modelo(df)
    dev, _ = mp.split_dev_lockbox(df)
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev)]
    d = sub.dropna(subset=palancas + [target_col]).reset_index(drop=True)
    state = [s for s in state if d[s].notna().mean() > 0.7]
    g = d["Batch"].to_numpy()
    y_res = d[target_col].to_numpy() - _oof_hgb(d[state], d[target_col], g, seed)
    A_res = np.column_stack([d[a].to_numpy() - _oof_hgb(d[state], d[a], g, seed) for a in palancas])

    def _theta(idx):
        A, y = A_res[idx], y_res[idx]
        return np.linalg.lstsq(A, y, rcond=None)[0]

    theta = _theta(np.arange(len(d)))
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    idx_by = {b: np.where(g == b)[0] for b in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
        try:
            boots.append(_theta(idx))
        except np.linalg.LinAlgError:
            continue
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    sd_a = d[palancas].std().to_numpy()
    out = pd.DataFrame({"palanca": palancas, "theta_por_unidad": theta, "ci95_lo": lo, "ci95_hi": hi,
                        "sd_palanca": sd_a, "efecto_por_1sd": theta * sd_a,
                        "efecto_1sd_ci95_lo": lo * sd_a, "efecto_1sd_ci95_hi": hi * sd_a})
    out["significativo_ci95"] = (out["ci95_lo"] > 0) | (out["ci95_hi"] < 0)
    out["n"] = len(d)
    out["target"] = target_col
    out["fase"] = fase
    return out.set_index("palanca")


def sensibilidad_local_v3(sistema: SistemaPrescriptivoV3, fase: str, state: dict, action: dict, context: dict | None = None,
                          palancas: list[str] | None = None, paso_relativo: float = 0.10) -> pd.DataFrame:
    """Sensibilidad LOCAL del modelo no lineal en el estado concreto del escalon: para cada
    palanca, diferencia finita central (+/- paso_relativo del rango P1-P99 historico) de las
    predicciones de Sn extraido [kg], FeO reducido [kg] y DeltaT [C]. Es el "efecto de la accion
    dado ESTE estado" (puede diferir del promedio causal de `efectos_marginales_palancas_v3`
    porque el modelo captura interacciones, p.ej. carbon x Sn previo)."""
    palancas = palancas or ACCIONES_OPTIMIZABLES_V3[fase]
    filas = []
    for a in palancas:
        lo, hi = sistema.limites_accion[fase][a]
        h = paso_relativo * (hi - lo)
        a_mas, a_menos = dict(action), dict(action)
        a_mas[a] = min(hi, action[a] + h)
        a_menos[a] = max(lo, action[a] - h)
        s_mas = simulate_action_v3(sistema, fase, state, a_mas, context)
        s_menos = simulate_action_v3(sistema, fase, state, a_menos, context)
        delta = a_mas[a] - a_menos[a]
        if delta <= 0:
            continue
        filas.append({"palanca": a, "valor_actual": action[a], "paso": delta,
                      "d_sn_kg_por_unidad": (s_mas["pred_sn_extraido_kg"] - s_menos["pred_sn_extraido_kg"]) / delta,
                      "d_feo_reducido_kg_por_unidad": (s_mas["pred_feo_reducido_kg"] - s_menos["pred_feo_reducido_kg"]) / delta,
                      "d_T_por_unidad": (s_mas["pred_dT"] - s_menos["pred_dT"]) / delta,
                      "d_sn_kg_en_el_paso": s_mas["pred_sn_extraido_kg"] - s_menos["pred_sn_extraido_kg"],
                      "d_feo_reducido_kg_en_el_paso": s_mas["pred_feo_reducido_kg"] - s_menos["pred_feo_reducido_kg"],
                      "d_T_en_el_paso": s_mas["pred_dT"] - s_menos["pred_dT"]})
    return pd.DataFrame(filas).set_index("palanca")
