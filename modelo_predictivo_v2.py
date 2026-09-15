"""Modelo predictivo/prescriptivo v2 de Lingo Smelter -- incremento validado sobre
`modelo_prescriptivo.py` (Ronda 2, sesion 2026-09-13). Este modulo NO reemplaza al
anterior (que sigue siendo valido y se sigue pudiendo usar tal cual): lo EXTIENDE con
5 piezas nuevas, cada una ejecutada contra el dataset real y validada con la misma
disciplina de 3 niveles (OOF GroupKFold / OOT walk-forward / FINAL_LOCKBOX) ya
establecida, mas tecnicas que el proyecto todavia no habia aplicado.

Resumen de que cambia y por que (ver hallazgos.md seccion 13 para el detalle completo):

1. **Bake-off de algoritmos con restricciones MONOTONICAS locales** (consideraciones
   sec. 84: "no imponer monotonia global, solo sobre submodelos fisicos inequivocos").
   Se restringe SOLO la relacion carbon->Sn (la unica confirmada "sin reservas" en TODAS
   las especificaciones previas, hallazgos.md 6b/9/10), nunca `posicion_vertical_lanza_mm`
   (optimo interior confirmado, Pasos 46-47 -- monotonizarla seria incorrecto).
   Resultado (lockbox, escala fisica): en Reduccion, `HistGradientBoostingRegressor` +
   monotonic_cst(carbon) le gana a XGBoost sin restriccion (R2 0.9648->0.9728,
   MAE 0.687->0.593); en Fusion, XGBoost SIN restriccion se mantiene mejor (R2
   0.465 vs 0.447 monotonico) -- coherente con el punto 3: ahi el efecto causal del
   carbon no se valida de forma robusta tras cross-fitting, forzar monotonia ahi es
   sobre-restringir una relacion incierta.

2. **Intervalos de prediccion CV+ (conformal, Barber, Candes, Ramdas & Tibshirani 2021)**
   sobre el mismo ensamble de GroupKFold que ya se entrenaba: reemplaza el `uncertainty`
   (std del ensamble, declarado explicitamente NO calibrado en modelo_prescriptivo.py)
   por un intervalo con **cobertura empirica validada en el lockbox** (nunca antes
   calibrado en este proyecto -- hallazgos.md seccion 8 punto 4, "pendiente"). Cobertura
   medida (lockbox, nunca usado para calibrar nada): Fusion 89.6%/95.7% para nominal
   80%/90%; Reduccion 90.5%/95.0% -- las 4 son conservadoras (cobertura real >= nominal),
   la direccion segura para un sistema prescriptivo (consideraciones sec. 85).

3. **Causalidad con Double ML CROSS-FITTED** (Chernozhukov et al. 2018) + refutation
   tests (placebo/random-confounder/subsample-stability, al estilo DoWhy). Corrige una
   limitacion metodologica real del Paso 42 (residualizaba con un modelo ajustado UNA vez
   sobre TODO el dev set y tomaba residuos IN-SAMPLE -- sesgo de sobreajuste conocido).
   Con cross-fitting real: `tasa_feed_Carbon_kg_min` en Reduccion y `delta_tiempo` en
   Fusion quedan **robustamente validados** (IC95% no cruza 0, pasan placebo estricto
   3-sigma, robustos a un confusor aleatorio, 100% de subsamples con el mismo signo);
   la significancia que el Paso 42 encontro para `tasa_feed_Carbon_kg_min` EN FUSION
   **se revierte** (IC95%=[-0.020,0.027], cruza 0) -- evidencia de que era un artefacto
   de no usar cross-fitting, no un efecto real; y el efecto lineal de
   `posicion_vertical_lanza_mm` resulta tecnicamente no-nulo pero de magnitud
   PRACTICAMENTE DESPRECIABLE (theta=0.0003 puntos de %Sn por mm -- para 1 punto de
   %Sn hacen falta ~3000mm, casi todo el rango observado), cuantificando por que un
   ajuste lineal la ve "casi plana": promedia una curva en valle que sube y baja
   (Pasos 46-47), tal como se anticipaba.

4. **Seleccion de features con tecnicas avanzadas** (mutual information + permutation
   importance ESTABLE [5 folds x 5 semillas] + Boruta-shadow [25 iteraciones]) sobre el
   pool completo de 79 candidatas seguras (`feature_engineering.CLASIFICACION_FEATURES`).
   En Fusion confirma el set actual (la aparente "redundancia" de `basicidad_B2_prev` y
   `tasa_feed_Carbon_kg_min` frente al pool completo ya estaba explicada en hallazgos.md
   10.3: su señal queda absorbida por `interaccion_C_x_Sn_prev`/`cum_feed_CaO_kg_prev`,
   no es evidencia de irrelevancia). En Reduccion, converge con el punto 3: 3 de las 7
   features actuales (`exceso_o2_combustion_pct` vía Boruta puro, `basicidad_B2_prev`,
   `grad_temperatura_horno_celsius_prev`) no superan un test independiente de ruido (0-12%
   de confirmacion sobre 25 iteraciones) -- de ahi **Config C** (ver mas abajo).

5. **Hallazgo operacional mas importante de la ronda**: `modelo_prescriptivo.py` NUNCA
   ofrecia `tasa_feed_Carbon_kg_min` como accion optimizable en Reduccion
   (`ACCIONES_PRIMITIVAS["Reducción"]` solo tenia GN/O2/aire/lanza) ni como feature de
   ese modelo (solo `cum_feed_Carbon_kg_prev`, el acumulado PASADO) -- la palanca con el
   efecto causal mas robusto de todo el analisis (punto 3) estaba **fuera del alcance del
   optimizador**. Config C la incorpora en ambos lugares.

**Config C (Reduccion, reemplaza a `FEATURES_STATE_ACTION["Reducción"]` para quien migre)**:
features = ['ley_sn_escoria_pct_prev', 'interaccion_Sn_x_FeO_prev', 'interaccion_C_x_Sn_prev',
'tasa_gn_nm3_min', 'interaccion_C_x_exceso_o2', 'posicion_vertical_lanza_mm_prev',
'tasa_feed_Carbon_kg_min', 'interaccion_C_x_FeO_prev', 'exceso_o2_combustion_pct'].
Desempeno (lockbox, escala fisica, HistGB+monotonic(carbon), n_dev=1085/n_lockbox=242,
MISMAS filas que la config actual): R2=0.9708 (vs. 0.9648 actual), MAE=0.642 (vs. 0.687).
Mejora real pero MODESTA en R2/MAE -- el valor principal de Config C no es "mas preciso",
es "mas defendible" (menos features en zona de ruido, elimina la confusion S_t->A_t de
`posicion_vertical_lanza_mm` contemporanea) Y **desbloquea recomendar dosis de carbon**,
que Config B nunca pudo hacer (punto 5). `posicion_vertical_lanza_mm` (contemporanea) SALE
de `ACCIONES_PRIMITIVAS_V2["Reducción"]` -- no se ofrece como palanca optimizable hasta
validacion de planta (misma razon que motivo Pasos 46-47), pero su version `_prev` se
mantiene como contexto de STATE (informativa para prediccion, no para recomendacion).

6. **Modelo de BATCH con features de TRAYECTORIA** (`feature_engineering.construir_features_trayectoria_batch`,
   no solo agregados lineales simples -- ya documentados como debiles, hallazgos.md
   seccion 6). Mejora sobre los baselines (R2 pasa de NEGATIVO/nulo a ~0.05-0.12 en
   walk-forward/lockbox), pero sigue siendo un resultado MODESTO: refuerza, no resuelve,
   el diagnostico ya documentado (hallazgos.md 9.5/11.7) de que falta masa de escoria y
   composicion de carga por escalon para cerrar el balance escalon->batch. Se declara
   explicitamente como señal real pero insuficiente para prescripcion de batch completo.

Uso tipico:

    import modelo_predictivo_v2 as mp2

    df = mp2.mp.construir_dataset_modelo("Datos Lingo smelter fase II.xlsx")
    sistema = mp2.entrenar_sistema_v2(df)
    print(mp2.tabla_validacion_v2(df))                    # bake-off + Config C vs. actual
    print(mp2.tabla_efectos_causales())                    # DML cross-fitted + refutation
    modelo_batch, metricas_batch = mp2.entrenar_modelo_batch(df, sistema)

Limitaciones heredadas que este modulo tampoco resuelve (se declaran, no se ocultan):
no hay masa de escoria por escalon (punto 6); el sentido fisico exacto de
`posicion_vertical_lanza_mm` (mayor valor = lanza mas alta o mas baja) sigue sin
confirmar con planta; los intervalos CV+ son *conformal* bajo supuestos de
intercambiabilidad que la estructura de batches/tiempo no garantiza estrictamente --
se validaron empiricamente en el lockbox (punto 2), pero deben re-chequearse si el
proceso cambia de regimen (nueva campaña, cambio de concentrado).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

import feature_engineering as fe
import modelo_prescriptivo as mp  # reutiliza dataset/split/soporte/anti-fuga ya validados

FASES = mp.FASES

# =============================================================================
# 1) Bake-off: algoritmo + restricciones monotonicas GANADORAS por fase
# =============================================================================

# Unica familia con relacion monotonica confirmada "sin reservas" en TODAS las
# especificaciones previas (hallazgos.md 6b/9/10): a mayor dosis/interaccion de
# carbon, mayor caida (mas negativo) del target fisico. NUNCA se restringe
# `posicion_vertical_lanza_mm` (optimo interior, Pasos 46-47) ni ninguna otra
# variable sin ese mismo nivel de evidencia (consideraciones sec. 84).
VARIABLES_MONOTONAS_NEGATIVAS = {"tasa_feed_Carbon_kg_min", "cum_feed_Carbon_kg_prev", "interaccion_C_x_Sn_prev"}


def _monotone_list(features: list[str]) -> list[int]:
    return [-1 if f in VARIABLES_MONOTONAS_NEGATIVAS else 0 for f in features]


def _builder_fusion(features: list[str]):
    return xgb.XGBRegressor(**mp.PARAMS_XGB)  # bake-off: sin restriccion gana en Fusion


def _builder_reduccion(features: list[str]):
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15,
        l2_regularization=1.0, random_state=42, monotonic_cst=_monotone_list(features),
    )


MODEL_BUILDERS = {"Fusión": _builder_fusion, "Reducción": _builder_reduccion}

# =============================================================================
# Config C (Reduccion): feature set alternativo tras el audit de seleccion (punto 4)
# y el hallazgo operacional del punto 5 (incorpora tasa_feed_Carbon_kg_min).
# =============================================================================

FEATURES_STATE_ACTION_V2: dict[str, list[str]] = {
    "Fusión": mp.FEATURES_STATE_ACTION["Fusión"],  # sin cambios: el audit confirma el set actual
    "Reducción": [
        "ley_sn_escoria_pct_prev", "interaccion_Sn_x_FeO_prev", "interaccion_C_x_Sn_prev",
        "tasa_gn_nm3_min", "interaccion_C_x_exceso_o2", "posicion_vertical_lanza_mm_prev",
        "tasa_feed_Carbon_kg_min", "interaccion_C_x_FeO_prev", "exceso_o2_combustion_pct",
    ],
}

FEATURES_STATE_V2: dict[str, list[str]] = {
    "Fusión": mp.FEATURES_STATE["Fusión"],
    "Reducción": ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "posicion_vertical_lanza_mm_prev"],
}

# `posicion_vertical_lanza_mm` SALE de las acciones optimizables de Reduccion (triple
# evidencia de inestabilidad: SHAP Pasos 7/13b/27/42/45, no-monotonia Pasos 46-47, y ahora
# Boruta ~0-4% aqui) hasta validacion de planta; `tasa_feed_Carbon_kg_min` ENTRA (punto 5).
ACCIONES_PRIMITIVAS_V2: dict[str, list[str]] = {
    "Fusión": mp.ACCIONES_PRIMITIVAS["Fusión"],
    "Reducción": ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"],
}

for _fase, _feats in FEATURES_STATE_ACTION_V2.items():
    _inseguras = [f for f in _feats if f not in fe.filtrar_features_seguras(_feats)]
    if _inseguras:
        raise AssertionError(f"FEATURES_STATE_ACTION_V2[{_fase}] contiene features inseguras: {_inseguras}")
del _fase, _feats, _inseguras


def recalcular_derivadas_v2(fase: str, state: dict, action: dict) -> dict:
    """Version 2 de `modelo_prescriptivo.recalcular_derivadas`: en Reduccion ya no
    recibe `posicion_vertical_lanza_mm` como accion (ver ACCIONES_PRIMITIVAS_V2) --
    su `_prev` viene fijo desde `state` (contexto observado, no una accion a mover)."""
    fila = dict(state)
    if fase == "Fusión":
        fila["delta_tiempo"] = action["delta_tiempo"]
        fila["tasa_feed_Carbon_kg_min"] = action["tasa_feed_Carbon_kg_min"]
        fila["interaccion_C_x_Sn_prev"] = action["tasa_feed_Carbon_kg_min"] * state["ley_sn_escoria_pct_prev"]
    else:
        fila["tasa_gn_nm3_min"] = action["tasa_gn_nm3_min"]
        fila["tasa_feed_Carbon_kg_min"] = action["tasa_feed_Carbon_kg_min"]
        o2_teorico = fe.RELACION_O2_CH4 * action["tasa_gn_nm3_min"]
        o2_disponible = action["tasa_o2_nm3_min"] + fe.FRACCION_O2_EN_AIRE * action["tasa_aire_nm3_min"]
        fila["exceso_o2_combustion_pct"] = 100 * (o2_disponible - o2_teorico) / o2_teorico if o2_teorico > 0 else np.nan
        fila["interaccion_C_x_exceso_o2"] = action["tasa_feed_Carbon_kg_min"] * fila["exceso_o2_combustion_pct"]
        fila["interaccion_C_x_Sn_prev"] = action["tasa_feed_Carbon_kg_min"] * state["ley_sn_escoria_pct_prev"]
        fila["interaccion_C_x_FeO_prev"] = action["tasa_feed_Carbon_kg_min"] * state["ley_feo_escoria_pct_prev"]
        fila["interaccion_Sn_x_FeO_prev"] = state["ley_sn_escoria_pct_prev"] * state["ley_feo_escoria_pct_prev"]

    faltantes = [f for f in FEATURES_STATE_ACTION_V2[fase] if f not in fila]
    if faltantes:
        raise KeyError(f"recalcular_derivadas_v2 no pudo construir: {faltantes}")
    return fila


# =============================================================================
# Validacion de 3 niveles del bake-off + Config C (reproduce los numeros del docstring)
# =============================================================================

def tabla_validacion_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Bake-off (algoritmo actual vs. ganador v2) x (feature set actual vs. Config C en
    Reduccion), en la MISMA escala fisica y con la MISMA disciplina OOF/lockbox ya usada
    en `modelo_prescriptivo.tabla_validacion_3_niveles`."""
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    filas = []
    variantes = {
        "Fusión": [("actual (XGBoost, set actual)", mp.FEATURES_STATE_ACTION["Fusión"], _builder_fusion)],
        "Reducción": [
            ("actual (XGBoost, Config B)", mp.FEATURES_STATE_ACTION["Reducción"], lambda f: xgb.XGBRegressor(**mp.PARAMS_XGB)),
            ("v2 (HistGB+monotonic, Config B)", mp.FEATURES_STATE_ACTION["Reducción"], _builder_reduccion),
            ("v2 (HistGB+monotonic, Config C)", FEATURES_STATE_ACTION_V2["Reducción"], _builder_reduccion),
        ],
    }
    for fase, variantes_fase in variantes.items():
        target = mp.TARGET_PRESCRIPTIVO_GANADOR[fase]
        for nombre, feats, builder in variantes_fase:
            cols = list(dict.fromkeys(feats + [target, "d_ley_sn_escoria_pct", "ley_sn_escoria_pct_prev", "Batch"]))
            d_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev), cols].dropna(subset=feats + [target])
            d_lb = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + [target])

            oof = np.full(len(d_dev), np.nan)
            for tr, te in GroupKFold(n_splits=mp.N_SPLITS_OOF).split(d_dev[feats], d_dev[target], d_dev["Batch"]):
                m = builder(feats)
                m.fit(d_dev[feats].iloc[tr], d_dev[target].iloc[tr])
                oof[te] = m.predict(d_dev[feats].iloc[te])
            pred_oof = mp.convertir_a_delta_sn_fisico(fase, oof, d_dev["ley_sn_escoria_pct_prev"].to_numpy())

            m_final = builder(feats)
            m_final.fit(d_dev[feats], d_dev[target])
            pred_lb = mp.convertir_a_delta_sn_fisico(fase, m_final.predict(d_lb[feats]), d_lb["ley_sn_escoria_pct_prev"].to_numpy())

            filas.append({
                "fase": fase, "variante": nombre, "n_features": len(feats),
                "n_dev": len(d_dev), "n_lockbox": len(d_lb),
                "r2_oof": r2_score(d_dev["d_ley_sn_escoria_pct"], pred_oof),
                "mae_oof": mean_absolute_error(d_dev["d_ley_sn_escoria_pct"], pred_oof),
                "r2_lockbox": r2_score(d_lb["d_ley_sn_escoria_pct"], pred_lb),
                "mae_lockbox": mean_absolute_error(d_lb["d_ley_sn_escoria_pct"], pred_lb),
            })
    return pd.DataFrame(filas).set_index(["fase", "variante"])


# =============================================================================
# 2) CV+ conformal prediction (reemplaza el std del ensamble no calibrado)
# =============================================================================

@dataclass
class SistemaPrescriptivoV2:
    ensambles: dict = field(default_factory=dict)       # (fase, target_key) -> list[modelo]
    residuos_oof: dict = field(default_factory=dict)    # (fase, target_key) -> list[np.ndarray] (uno por fold)
    soporte: dict = field(default_factory=dict)
    limites_accion: dict = field(default_factory=dict)
    batches_dev: set = field(default_factory=set)
    batches_lockbox: set = field(default_factory=set)


def _entrenar_ensamble_v2(sub_dev_fase: pd.DataFrame, features: list[str], target_col: str, builder, n_splits: int = mp.N_SPLITS_OOF):
    """Como `modelo_prescriptivo._entrenar_ensamble`, pero ademas devuelve los residuos
    OOF |y-pred| de CADA fold (para CV+): el modelo que no vio el fold `k`, evaluado en
    el fold `k`. Son la materia prima del intervalo conformal (Barber et al. 2021)."""
    sub = sub_dev_fase.dropna(subset=features + [target_col])
    modelos, residuos = [], []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(sub[features], sub[target_col], sub["Batch"]):
        m = builder(features)
        m.fit(sub[features].iloc[train_idx], sub[target_col].iloc[train_idx])
        modelos.append(m)
        residuos.append(np.abs(sub[target_col].iloc[test_idx].to_numpy() - m.predict(sub[features].iloc[test_idx])))
    return modelos, residuos


def intervalo_cv_plus(modelos_fold: list, residuos_fold: list[np.ndarray], x_row_df: pd.DataFrame, alpha: float = 0.20) -> tuple[float, float]:
    """CV+ (Barber, Candes, Ramdas & Tibshirani 2021): para cada residuo OOF historico,
    usa la prediccion del modelo QUE NO LO VIO evaluada en el punto nuevo +/- ese residuo;
    el intervalo es el percentil alpha/2 - (1-alpha/2) de la nube combinada. Cobertura
    empirica validada en el lockbox: ver docstring del modulo, punto 2."""
    lo, hi = [], []
    for m, residuos in zip(modelos_fold, residuos_fold):
        pred = m.predict(x_row_df)[0]
        lo.extend(pred - residuos)
        hi.extend(pred + residuos)
    return float(np.quantile(lo, alpha / 2)), float(np.quantile(hi, 1 - alpha / 2))


def validar_cobertura_lockbox(df: pd.DataFrame, fase: str, alpha: float = 0.20) -> dict:
    """Reproduce la validacion de cobertura empirica del docstring del modulo (punto 2),
    sobre el LOCKBOX (nunca usado para entrenar ni calibrar nada)."""
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    feats = FEATURES_STATE_ACTION_V2[fase]
    target = mp.TARGET_PRESCRIPTIVO_GANADOR[fase]
    builder = MODEL_BUILDERS[fase]
    cols = list(dict.fromkeys(feats + [target, "Batch"]))
    sub_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev), cols].dropna()
    sub_lb = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_lockbox), cols].dropna()

    modelos_fold, residuos_fold = _entrenar_ensamble_v2(sub_dev, feats, target, builder)
    cubiertos, anchos = [], []
    for i in range(len(sub_lb)):
        lo, hi = intervalo_cv_plus(modelos_fold, residuos_fold, sub_lb[feats].iloc[[i]], alpha)
        cubiertos.append(lo <= sub_lb[target].iloc[i] <= hi)
        anchos.append(hi - lo)
    return {"fase": fase, "alpha": alpha, "nominal": 1 - alpha,
            "cobertura_empirica": float(np.mean(cubiertos)), "ancho_medio": float(np.mean(anchos)), "n": len(sub_lb)}


# =============================================================================
# Entrenamiento del sistema completo v2 (bake-off + Config C + CV+ + soporte)
# =============================================================================

def entrenar_sistema_v2(df: pd.DataFrame, pct_lockbox: float = mp.N_LOCKBOX_PCT) -> SistemaPrescriptivoV2:
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df, pct_lockbox)
    sistema = SistemaPrescriptivoV2(batches_dev=batches_dev, batches_lockbox=batches_lockbox)

    for fase in FASES:
        sub_fase_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        feats = FEATURES_STATE_ACTION_V2[fase]
        builder = MODEL_BUILDERS[fase]

        modelos_sn, res_sn = _entrenar_ensamble_v2(sub_fase_dev, feats, mp.TARGET_PRESCRIPTIVO_GANADOR[fase], builder)
        modelos_feo, res_feo = _entrenar_ensamble_v2(sub_fase_dev, feats, mp.TARGET_FEO, builder)
        sistema.ensambles[(fase, "delta_Sn")] = modelos_sn
        sistema.residuos_oof[(fase, "delta_Sn")] = res_sn
        sistema.ensambles[(fase, "delta_FeO")] = modelos_feo
        sistema.residuos_oof[(fase, "delta_FeO")] = res_feo

        sistema.soporte[fase] = mp._construir_soporte_historico(sub_fase_dev, feats)

        sub_acciones = sub_fase_dev[ACCIONES_PRIMITIVAS_V2[fase]].dropna()
        sistema.limites_accion[fase] = {
            a: (float(sub_acciones[a].quantile(0.01)), float(sub_acciones[a].quantile(0.99)))
            for a in ACCIONES_PRIMITIVAS_V2[fase]
        }
    return sistema


def simulate_action_v2(sistema: SistemaPrescriptivoV2, fase: str, state: dict, action: dict, alpha: float = 0.20) -> dict:
    """Como `modelo_prescriptivo.simulate_action`, pero con intervalo CV+ CALIBRADO
    (punto 2) en vez de `uncertainty` = std del ensamble sin calibrar."""
    fila = recalcular_derivadas_v2(fase, state, action)
    feats = FEATURES_STATE_ACTION_V2[fase]
    x = pd.DataFrame([[fila[f] for f in feats]], columns=feats)

    def _predecir(nombre_target):
        modelos = sistema.ensambles[(fase, nombre_target)]
        residuos = sistema.residuos_oof[(fase, nombre_target)]
        preds = np.array([m.predict(x)[0] for m in modelos])
        lo, hi = intervalo_cv_plus(modelos, residuos, x, alpha)
        return float(preds.mean()), lo, hi

    pred_sn_raw, lo_sn_raw, hi_sn_raw = _predecir("delta_Sn")
    ley_prev = np.array([state["ley_sn_escoria_pct_prev"]])
    pred_sn = float(mp.convertir_a_delta_sn_fisico(fase, np.array([pred_sn_raw]), ley_prev)[0])
    lo_sn = float(mp.convertir_a_delta_sn_fisico(fase, np.array([lo_sn_raw]), ley_prev)[0])
    hi_sn = float(mp.convertir_a_delta_sn_fisico(fase, np.array([hi_sn_raw]), ley_prev)[0])
    if lo_sn > hi_sn:
        lo_sn, hi_sn = hi_sn, lo_sn  # T3 escalada por un ley_prev negativo no deberia ocurrir, pero se ordena por seguridad

    pred_feo, lo_feo, hi_feo = _predecir("delta_FeO")

    return {
        "fase": fase, "pred_delta_sn": pred_sn, "pred_delta_feo": pred_feo,
        "intervalo_delta_sn_cvplus": (lo_sn, hi_sn), "intervalo_delta_feo_cvplus": (lo_feo, hi_feo),
        "ancho_intervalo_sn": hi_sn - lo_sn,
        "support_score": mp.calcular_support_score(sistema.soporte[fase], fila),
    }


# =============================================================================
# 3) Causalidad: Double ML cross-fitted + refutation tests (reemplaza el Paso 42)
# =============================================================================

PARAMS_NUISANCE = mp.PARAMS_XGB

# Acciones candidatas evaluadas (union de las del Paso 42 + tasa_feed_Carbon_kg_min en
# Reduccion, que el Paso 42 no habia evaluado porque en ese momento no era ni feature ni
# accion de ese modelo -- ver punto 5).
ACCIONES_CANDIDATAS_CAUSAL: dict[str, list[str]] = {
    "Fusión": ["delta_tiempo", "tasa_feed_Carbon_kg_min"],
    "Reducción": ["tasa_gn_nm3_min", "exceso_o2_combustion_pct", "posicion_vertical_lanza_mm", "tasa_feed_Carbon_kg_min"],
}
TARGET_CAUSAL_FISICO = "d_ley_sn_escoria_pct"  # misma escala fisica en ambas fases


def _oof_predict_generic(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    oof = np.full(len(X), np.nan)
    params = dict(PARAMS_NUISANCE, random_state=seed)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = m.predict(X.iloc[test_idx])
    return oof


def _cluster_bootstrap_ci(a_res: np.ndarray, y_res: np.ndarray, batches: np.ndarray, n_boot: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(batches)
    idx_by_batch = {b: np.where(batches == b)[0] for b in uniq}
    thetas = []
    for _ in range(n_boot):
        sample_batches = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_batch[b] for b in sample_batches])
        a_b, y_b = a_res[idx], y_res[idx]
        if np.var(a_b) < 1e-12:
            continue
        thetas.append(np.cov(a_b, y_b, bias=True)[0, 1] / np.var(a_b))
    thetas = np.array(thetas)
    return float(np.percentile(thetas, 2.5)), float(np.percentile(thetas, 97.5))


def dml_partialling_out(sub: pd.DataFrame, state_feats: list[str], accion: str, target_col: str, n_splits: int = 5, seed: int = 0) -> dict:
    """Double ML de Robinson/partialling-out CON CROSS-FITTING real (a diferencia del
    Paso 42, que residualizaba in-sample). theta = efecto marginal medio de `accion`
    sobre `target_col`, controlando `state_feats`, en las MISMAS unidades del target."""
    cols = list(dict.fromkeys(state_feats + [accion, target_col, "Batch"]))
    d = sub[cols].dropna().reset_index(drop=True)
    X = d[state_feats]
    y_hat = _oof_predict_generic(X, d[target_col], d["Batch"].to_numpy(), n_splits, seed)
    a_hat = _oof_predict_generic(X, d[accion], d["Batch"].to_numpy(), n_splits, seed)
    y_res = d[target_col].to_numpy() - y_hat
    a_res = d[accion].to_numpy() - a_hat
    theta = float(np.cov(a_res, y_res, bias=True)[0, 1] / np.var(a_res))
    ci_lo, ci_hi = _cluster_bootstrap_ci(a_res, y_res, d["Batch"].to_numpy(), seed=seed)
    return {"theta": theta, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(d), "a_res": a_res, "y_res": y_res, "batches": d["Batch"].to_numpy()}


def refutation_placebo(sub, state_feats, accion, target_col, n_reps=12, seed=0) -> np.ndarray:
    """Placebo: baraja la accion (misma distribucion marginal, sin relacion real con el
    resto); theta deberia concentrarse cerca de 0 si el metodo no encuentra efectos
    espurios. Umbral de referencia (no un p-valor formal): |theta_real| > 3*std(placebo)."""
    d = sub[list(dict.fromkeys(state_feats + [accion, target_col, "Batch"]))].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    thetas = []
    for r in range(n_reps):
        d2 = d.copy()
        d2[accion] = rng.permutation(d2[accion].to_numpy())
        thetas.append(dml_partialling_out(d2, state_feats, accion, target_col, seed=seed + r)["theta"])
    return np.array(thetas)


def refutation_random_confounder(sub, state_feats, accion, target_col, seed=0) -> dict:
    """Agrega una variable de ruido aleatorio a las nuisance models; un theta robusto no
    deberia moverse mucho (umbral informal: <50% de cambio relativo)."""
    d = sub.copy()
    d["_ruido_confusor"] = np.random.default_rng(seed).normal(size=len(d))
    return dml_partialling_out(d, state_feats + ["_ruido_confusor"], accion, target_col, seed=seed)


def refutation_subset_stability(sub, state_feats, accion, target_col, frac=0.7, n_reps=10, seed=0) -> np.ndarray:
    """Reestima theta sobre submuestras aleatorias POR BATCH; un efecto real deberia
    mantener signo estable entre submuestras (no ser un artefacto de pocos batches)."""
    batches = sub["Batch"].unique()
    rng = np.random.default_rng(seed)
    thetas = []
    for r in range(n_reps):
        sample_b = rng.choice(batches, size=int(len(batches) * frac), replace=False)
        try:
            thetas.append(dml_partialling_out(sub.loc[sub["Batch"].isin(sample_b)], state_feats, accion, target_col, seed=seed + r)["theta"])
        except Exception:
            continue
    return np.array(thetas)


def tabla_efectos_causales(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla de efectos causales cross-fitted + refutation tests (punto 3). Si `df` es
    None, reconstruye el dataset desde cero (mismo pipeline que `entrenar_sistema_v2`)."""
    if df is None:
        df = mp.construir_dataset_modelo("Datos Lingo smelter fase II.xlsx")
    df_base = mp.dataset_base_modelo(df)
    batches_dev, _ = mp.split_dev_lockbox(df)

    filas = []
    for fase, acciones in ACCIONES_CANDIDATAS_CAUSAL.items():
        state_feats = mp.FEATURES_STATE[fase]
        sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        for accion in acciones:
            base = dml_partialling_out(sub, state_feats, accion, TARGET_CAUSAL_FISICO)
            placebo = refutation_placebo(sub, state_feats, accion, TARGET_CAUSAL_FISICO)
            conf = refutation_random_confounder(sub, state_feats, accion, TARGET_CAUSAL_FISICO)
            subset = refutation_subset_stability(sub, state_feats, accion, TARGET_CAUSAL_FISICO)

            significativo = (base["ci_lo"] > 0) or (base["ci_hi"] < 0)
            pasa_placebo = bool(abs(base["theta"]) > 3 * placebo.std()) if placebo.std() > 0 else bool(abs(base["theta"]) > 0)
            cambio_confusor_pct = 100 * abs(conf["theta"] - base["theta"]) / (abs(base["theta"]) + 1e-9)
            frac_mismo_signo = float(np.mean(np.sign(subset) == np.sign(base["theta"]))) if len(subset) else np.nan

            robusto = bool(significativo and pasa_placebo and cambio_confusor_pct < 50 and frac_mismo_signo >= 0.8)
            filas.append({
                "fase": fase, "accion": accion, "n": base["n"],
                "theta_por_unidad": base["theta"], "ci95_lo": base["ci_lo"], "ci95_hi": base["ci_hi"],
                "significativo_ci95": significativo, "pasa_placebo_3sigma": pasa_placebo,
                "cambio_pct_con_confusor": cambio_confusor_pct, "frac_subset_mismo_signo": frac_mismo_signo,
                "veredicto": "ROBUSTO" if robusto else ("SIGNIFICATIVO_MODERADO" if significativo else "NO_SIGNIFICATIVO"),
            })
    return pd.DataFrame(filas).set_index(["fase", "accion"])


# =============================================================================
# 6) Modelo de BATCH con features de trayectoria (complementa, no reemplaza, al de escalon)
# =============================================================================

PARAMS_XGB_BATCH = dict(n_estimators=80, learning_rate=0.03, max_depth=2, min_child_weight=10,
                         subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, random_state=42, n_jobs=-1)


def _rollup_modelo_escalon_reduccion(df_base: pd.DataFrame, sistema: SistemaPrescriptivoV2,
                                      batches_dev: set, batches_lockbox: set) -> pd.Series:
    """Suma por batch del Delta_Sn fisico predicho por el ENSAMBLE v2 de Reduccion, SIEMPRE
    fuera de muestra (promedio de las k predicciones out-of-fold, cada una de un modelo que
    no vio ese batch en entrenamiento): una feature de batch NO LINEAL informada por el
    modelo de escalon ya validado, en vez de solo sumas crudas de reactivos."""
    fase = "Reducción"
    feats = FEATURES_STATE_ACTION_V2[fase]
    target = mp.TARGET_PRESCRIPTIVO_GANADOR[fase]
    cols = list(dict.fromkeys(feats + [target, "ley_sn_escoria_pct_prev", "Batch"]))
    sub_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev), cols].dropna()
    sub_lb = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_lockbox), cols].dropna()

    modelos = sistema.ensambles[(fase, "delta_Sn")]
    pred_dev = np.mean([m.predict(sub_dev[feats]) for m in modelos], axis=0)
    pred_lb = np.mean([m.predict(sub_lb[feats]) for m in modelos], axis=0)
    fisico_dev = mp.convertir_a_delta_sn_fisico(fase, pred_dev, sub_dev["ley_sn_escoria_pct_prev"].to_numpy())
    fisico_lb = mp.convertir_a_delta_sn_fisico(fase, pred_lb, sub_lb["ley_sn_escoria_pct_prev"].to_numpy())

    resultado = pd.concat([
        sub_dev[["Batch"]].assign(p=fisico_dev), sub_lb[["Batch"]].assign(p=fisico_lb),
    ])
    return resultado.groupby("Batch")["p"].sum().rename("suma_pred_delta_sn_reduccion_modelo")


def entrenar_modelo_batch(df: pd.DataFrame, sistema: SistemaPrescriptivoV2 | None = None,
                           target_col: str = "rendimiento_proxy_batch") -> tuple:
    """Modelo de batch con features de trayectoria (punto 6): XGBoost chico y regularizado
    (pocos batches -- 362 -- relativo al numero de features, ver hallazgos.md seccion 13
    para la leccion de sobreajuste con el pool completo de 38 features). Devuelve el modelo
    entrenado en TODO dev y un dict con las metricas de validacion de 3 niveles (honestas:
    no se oculta que el resultado es modesto).
    """
    if sistema is None:
        sistema = entrenar_sistema_v2(df)
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    orden_batches = mp.orden_cronologico_batches(df)

    tabla = fe.construir_features_trayectoria_batch(df)
    rollup = _rollup_modelo_escalon_reduccion(df_base, sistema, batches_dev, batches_lockbox)
    tabla = tabla.join(rollup, how="left")

    features_top10 = [
        "reduccion_feo_caida_total", "reduccion_ley_feo_final", "reduccion_carbon_total_kg",
        "fusion_carbon_total_kg", "fusion_temp_mean", "reduccion_o2_total_nm3",
        "reduccion_lanza_pos_mean", "fusion_o2_exceso_mean", "reduccion_sn_caida_total",
        "delta_FR_ley_feo_escoria_pct", "suma_pred_delta_sn_reduccion_modelo",
    ]
    features_top10 = [f for f in features_top10 if f in tabla.columns]

    d = tabla.dropna(subset=[target_col]).copy()
    orden_validos = [b for b in orden_batches if b in d.index]
    d = d.loc[orden_validos]
    dev_mask = d.index.isin(batches_dev)
    d_dev, d_lb = d.loc[dev_mask], d.loc[~dev_mask]
    X_dev, y_dev = d_dev[features_top10], d_dev[target_col]
    X_lb, y_lb = d_lb[features_top10], d_lb[target_col]
    med = X_dev.median()

    def _oof():
        out = np.full(len(X_dev), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X_dev):
            m = xgb.XGBRegressor(**PARAMS_XGB_BATCH)
            m.fit(X_dev.iloc[tr].fillna(med), y_dev.iloc[tr])
            out[te] = m.predict(X_dev.iloc[te].fillna(med))
        return out

    def _walkforward():
        out = np.full(len(X_dev), np.nan)
        for tr, te in TimeSeriesSplit(5).split(X_dev):
            m = xgb.XGBRegressor(**PARAMS_XGB_BATCH)
            m.fit(X_dev.iloc[tr].fillna(X_dev.iloc[tr].median()), y_dev.iloc[tr])
            out[te] = m.predict(X_dev.iloc[te].fillna(X_dev.iloc[tr].median()))
        return out

    oof = _oof()
    wf = _walkforward()
    valid_wf = ~np.isnan(wf)
    modelo = xgb.XGBRegressor(**PARAMS_XGB_BATCH)
    modelo.fit(X_dev.fillna(med), y_dev)
    pred_lb = modelo.predict(X_lb.fillna(med))

    metricas = {
        "target": target_col, "features": features_top10, "n_dev": len(d_dev), "n_lockbox": len(d_lb),
        "r2_oof": r2_score(y_dev, oof), "r2_oot_walkforward": r2_score(y_dev.to_numpy()[valid_wf], wf[valid_wf]),
        "r2_lockbox": r2_score(y_lb, pred_lb), "mae_lockbox": mean_absolute_error(y_lb, pred_lb),
        "advertencia": ("Senal real pero MODESTA (R2 tipico 0.05-0.12): no usar para prescripcion de "
                        "batch completo, solo como diagnostico complementario. hallazgos.md 9.5/11.7: "
                        "el cuello de botella es la falta de masa de escoria/composicion de carga por "
                        "escalon, no la tecnica de modelado (ya se probo XGBoost + features de "
                        "trayectoria no lineales, ver hallazgos.md seccion 13)."),
    }
    return modelo, metricas
