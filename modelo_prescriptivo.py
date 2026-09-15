"""Modelo predictivo/prescriptivo final de Lingo Smelter (horno Ausmelt de Sn).

Este modulo CONSOLIDA en codigo reutilizable el sistema ya validado en
`analisis_lingo_smelter.ipynb` (Pasos 30-47: "de modelo descriptivo a gemelo
digital prescriptivo"). No re-deriva metodologia nueva: reproduce exactamente
las decisiones ya tomadas y verificadas ahi (features sin fuga, target ganador
por fase, algoritmo, pesos por defecto de la funcion objetivo), y las empaqueta
como funciones/objetos importables desde un script o notebook, en vez de
depender de 47 celdas ejecutadas en orden con variables de alcance global.

Lo que agrega este modulo respecto al notebook (parte pedida explicitamente el
2026-09-13, "por ende tambien el rendimiento del batch"):
  - `reporte_recomendaciones_batch`: recomendacion escalon-por-escalon para UN
    batch completo (no solo una muestra de 12 escalones de Reduccion como en
    el Paso 44), usando en cada escalon su propio ESTADO HISTORICO REAL (no
    una trayectoria encadenada / simulada hacia adelante -- ver razon en el
    docstring de esa funcion) y agregando el resultado a un indicador de
    "puntos de %Sn adicionales recuperables de la escoria" por batch.
  - `evaluar_relacion_selectividad_vs_rendimiento`: cuantifica, con la
    evidencia empirica disponible (Spearman), el eslabon causal cualitativo
    que conecta el nivel de escalon con `rendimiento_proxy_batch`: sobre-
    reducir FeO en Reduccion arrastra Fe hacia el metal/hardhead, lo que
    empiricamente termina como `sn_en_dross_fe_batch_t` (Sn atrapado en dross
    de Fe) en vez de `sn_en_metal_crudo_batch_t` -- el mecanismo por el que
    `indice_selectividad_sn_feo` (Paso 39 del notebook) no es solo un
    diagnostico de escalon sino la palanca mas defendible sobre el rendimiento
    del batch. Se reporta el numero real (que puede ser debil, ver
    hallazgos.md seccion 6b), no se asume que sea fuerte.

Arquitectura final por fase (veredicto del notebook, Paso 45; ver tambien
`consideracioness_pirometalurgia_ausmelt.md` y `guia_fenomenologica_ausmelt_sn.md`):

    Algoritmo:      XGBoost (empate tecnico con HistGradientBoostingRegressor,
                    Paso 40; se elige por continuidad con SHAP ya validado).
    Features:       FEATURES_STATE_ACTION[fase] -- 7 variables por fase, TODAS
                    disponibles ANTES de decidir la accion del escalon t (sin
                    fuga; ver feature_engineering.CLASIFICACION_FEATURES).
    Target ganador: TARGET_PRESCRIPTIVO_GANADOR[fase] (T1 delta absoluto en
                    Fusion, T3 fraccion relativa en Reduccion -- Paso 36).
    Validacion:     3 niveles (OOF GroupKFold-por-Batch / OOT TimeSeriesSplit
                    walk-forward / FINAL_LOCKBOX ultimos batches, nunca usados
                    para elegir nada) -- Paso 41.
    Desempeno
    (lockbox, escala fisica Delta_Sn/Delta_FeO):
        Fusion:    R2 Delta_Sn=0.462  R2 Delta_FeO=0.156
        Reduccion: R2 Delta_Sn=0.965  R2 Delta_FeO=0.517

Uso tipico:

    import modelo_prescriptivo as mp

    df = mp.construir_dataset_modelo("Datos Lingo smelter fase II.xlsx")
    sistema = mp.entrenar_sistema(df)
    print(mp.tabla_validacion_3_niveles(sistema, df))

    reporte = mp.reporte_recomendaciones_batch(sistema, df, batch_id="AP0350")
    print(reporte)

Limitaciones heredadas del notebook que este modulo NO resuelve (se declaran,
no se ocultan): `posicion_vertical_lanza_mm` tiene un optimo interior
confirmado (Paso 47) pero de sentido fisico (lanza mas alta o mas baja)
todavia sin confirmar con planta; `delta_tiempo` es ambigua entre ACTION y
LEAKAGE segun si el escalon termina por decision previa o por parada endogena
(Paso 31); no existe modelo confiable de Delta_T (Paso 38, R2 OOT=-0.04); y no
hay masa de escoria por escalon, por lo que ninguna cantidad en puntos de %Sn/
%FeO de este modulo se debe leer como toneladas ni como delta exacto de
`rendimiento_proxy_batch` (hallazgos.md seccion 6b/9.5).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import dataset_lingo_smelter as dls
import feature_engineering as fe

# =============================================================================
# Configuracion del modelo final (fija; ver docstring del modulo y Paso 45)
# =============================================================================

FASES = ["Fusión", "Reducción"]

PARAMS_XGB = dict(
    n_estimators=300, learning_rate=0.03, max_depth=4,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42, n_jobs=-1,
)

N_SPLITS_OOF = 5
N_SPLITS_OOT = 5
N_LOCKBOX_PCT = 0.175
K_VECINOS_SOPORTE = 10

# STATE puro: conocido antes de decidir la accion del escalon t (Paso 33).
FEATURES_STATE: dict[str, list[str]] = {
    "Fusión": ["basicidad_B2_prev", "ley_sn_escoria_pct_prev", "cum_feed_CaO_kg_prev", "interaccion_Sn_x_FeO_prev"],
    "Reducción": ["basicidad_B2_prev", "grad_temperatura_horno_celsius_prev", "ley_sn_escoria_pct_prev", "cum_feed_Carbon_kg_prev"],
}

# ACTION puro: primitivas de decision del escalon t usadas como feature de control (Paso 33).
FEATURES_ACTION: dict[str, list[str]] = {
    "Fusión": ["delta_tiempo", "tasa_feed_Carbon_kg_min"],
    "Reducción": ["tasa_gn_nm3_min", "exceso_o2_combustion_pct", "posicion_vertical_lanza_mm"],
}

# Primitivas de ACCION reales que el optimizador puede mover (ver recalcular_derivadas):
# nunca se optimiza una razon/derivada directamente (Paso 43).
ACCIONES_PRIMITIVAS: dict[str, list[str]] = {
    "Fusión": ["delta_tiempo", "tasa_feed_Carbon_kg_min"],
    "Reducción": ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "posicion_vertical_lanza_mm"],
}

# M3 = State+Action, features del modelo prescriptivo final por fase (Paso 32/33/45), SIN fuga.
FEATURES_STATE_ACTION: dict[str, list[str]] = {
    "Fusión": [
        "basicidad_B2_prev", "delta_tiempo", "ley_sn_escoria_pct_prev",
        "interaccion_C_x_Sn_prev", "cum_feed_CaO_kg_prev",
        "interaccion_Sn_x_FeO_prev", "tasa_feed_Carbon_kg_min",
    ],
    "Reducción": [
        "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "basicidad_B2_prev",
        "grad_temperatura_horno_celsius_prev", "posicion_vertical_lanza_mm",
        "ley_sn_escoria_pct_prev", "cum_feed_Carbon_kg_prev",
    ],
}

# Target ganador tras revalidar en features SIN fuga, en la MISMA escala fisica (Paso 35-36):
# invierte respecto a la eleccion original con features contemporaneas (Paso 13).
TARGET_PRESCRIPTIVO_GANADOR: dict[str, str] = {"Fusión": "d_ley_sn_escoria_pct", "Reducción": "tasa_relativa_sn_escoria"}
TARGET_FEO = "d_ley_feo_escoria_pct"

PESOS_J_R_DEFAULT = dict(V_Sn=1.0, lambda_Fe=1.0, lambda_C=0.05, lambda_E=0.02, lambda_t=0.01, lambda_U=0.3, lambda_S=3.0)
PESOS_J_F_DEFAULT = dict(w_Sn=1.0, direccion_sn_fusion=1.0, w_FeO=0.3, w_B2=0.5, w_E=0.02, w_t=0.01, w_U=0.3, w_S=3.0, basicidad_optima=1.4)

# Verificacion en tiempo de importacion: ninguna feature del modelo prescriptivo
# puede estar clasificada como LEAKAGE/TARGET en el registro independiente de
# feature_engineering.py (disciplina anti-fuga, ver su docstring de modulo).
for _fase, _feats in FEATURES_STATE_ACTION.items():
    _inseguras = [f for f in _feats if f not in fe.filtrar_features_seguras(_feats)]
    if _inseguras:
        raise AssertionError(f"FEATURES_STATE_ACTION[{_fase}] contiene features inseguras: {_inseguras}")
del _fase, _feats, _inseguras


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# =============================================================================
# Construccion del dataset de modelado
# =============================================================================

def construir_dataset_modelo(ruta_excel: str, hoja: str = "Datos") -> pd.DataFrame:
    """Pipeline completo: Excel -> limpieza -> features base -> features de modelado.

    Aplica, en orden, `dataset_lingo_smelter.construir_dataset` (clean_mode=True),
    `feature_engineering.construir_features`, `feature_engineering.construir_features_modelado`
    y `feature_engineering.agregar_target_tasa_relativa_sn` (T3, target ganador de
    Reduccion). Devuelve el DataFrame completo (incluye el primer escalon de cada
    batch, sin `_prev` valido); usar `dataset_base_modelo` para el subconjunto
    modelable.
    """
    df = dls.construir_dataset(ruta_excel, hoja=hoja, clean_mode=True)
    df = fe.construir_features(df)
    df = fe.construir_features_modelado(df)
    df = fe.agregar_target_tasa_relativa_sn(df)
    return df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)


def dataset_base_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Excluye el primer escalon de cada batch (sin estado `_prev` valido). Paso 13."""
    return df.loc[~df["es_primer_escalon_batch"]].reset_index(drop=True)


def orden_cronologico_batches(df: pd.DataFrame) -> list:
    """Batches ordenados por la fecha de inicio de su primer escalon."""
    return df.groupby("Batch")["fecha_inicio"].min().sort_values().index.tolist()


def split_dev_lockbox(df: pd.DataFrame, pct_lockbox: float = N_LOCKBOX_PCT) -> tuple[set, set]:
    """Ultimos `pct_lockbox` batches cronologicos = FINAL_LOCKBOX; el resto = DEV (Paso 41).

    El lockbox nunca debe usarse para elegir features/target/hiperparametros/pesos
    de la funcion objetivo -- solo como auditoria final de un modelo ya congelado.
    """
    orden_batches = orden_cronologico_batches(df)
    n_lockbox = int(round(len(orden_batches) * pct_lockbox))
    if n_lockbox <= 0:  # pct_lockbox=0: todo es DEV (util para entrenar sub-sistemas en cross-fitting)
        return set(orden_batches), set()
    batches_lockbox = set(orden_batches[-n_lockbox:])
    batches_dev = set(orden_batches[:-n_lockbox])
    return batches_dev, batches_lockbox


def convertir_a_delta_sn_fisico(fase: str, pred: np.ndarray, ley_sn_prev: np.ndarray) -> np.ndarray:
    """Convierte la prediccion del target ganador de `fase` a Delta_Sn fisico (puntos de %Sn).

    T1 (Fusion) ya es Delta_Sn; T3 (Reduccion, fraccion relativa) se multiplica por
    el nivel de partida `ley_sn_escoria_pct_prev` (Paso 35).
    """
    pred = np.asarray(pred, dtype=float)
    if TARGET_PRESCRIPTIVO_GANADOR[fase] == "d_ley_sn_escoria_pct":
        return pred
    return pred * np.asarray(ley_sn_prev, dtype=float)


# =============================================================================
# Validacion: OOF (GroupKFold), OOT (walk-forward), y utilidades compartidas
# =============================================================================

def oof_groupkfold(sub: pd.DataFrame, features: list[str], target_col: str, n_splits: int = N_SPLITS_OOF) -> np.ndarray:
    """Prediccion out-of-fold agrupando por Batch (ningun batch comparte filas train/test)."""
    X, y, groups = sub[features].to_numpy(), sub[target_col].to_numpy(), sub["Batch"].to_numpy()
    oof = np.full(len(sub), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        modelo = xgb.XGBRegressor(**PARAMS_XGB)
        modelo.fit(X[train_idx], y[train_idx])
        oof[test_idx] = modelo.predict(X[test_idx])
    return oof


def oot_walkforward(sub: pd.DataFrame, features: list[str], target_col: str,
                     orden_batches: list, n_splits: int = N_SPLITS_OOT) -> pd.DataFrame:
    """Prediccion walk-forward (TimeSeriesSplit por orden cronologico de Batch).

    Devuelve `sub` reordenado cronologicamente con una columna `pred_oot` (NaN en
    los batches usados solo como train del primer fold, que nunca quedan en un
    conjunto de test).
    """
    sub_ordenado = sub.set_index("Batch").loc[[b for b in orden_batches if b in set(sub["Batch"])]].reset_index()
    batches_orden = sub_ordenado["Batch"].drop_duplicates().tolist()
    X_full = sub_ordenado[features].to_numpy()
    y_full = sub_ordenado[target_col].to_numpy()
    batch_arr = sub_ordenado["Batch"].to_numpy()

    oot_pred = np.full(len(sub_ordenado), np.nan)
    for train_pos, test_pos in TimeSeriesSplit(n_splits=n_splits).split(batches_orden):
        batches_train = set(batches_orden[i] for i in train_pos)
        batches_test = set(batches_orden[i] for i in test_pos)
        mask_train = np.isin(batch_arr, list(batches_train))
        mask_test = np.isin(batch_arr, list(batches_test))
        modelo = xgb.XGBRegressor(**PARAMS_XGB)
        modelo.fit(X_full[mask_train], y_full[mask_train])
        oot_pred[mask_test] = modelo.predict(X_full[mask_test])
    return sub_ordenado.assign(pred_oot=oot_pred)


def tabla_validacion_3_niveles(df_base: pd.DataFrame, batches_dev: set, batches_lockbox: set) -> pd.DataFrame:
    """Reproduce la tabla de validacion en 3 niveles del Paso 41/45: OOF dev / OOT dev / lockbox.

    Todo convertido a la escala fisica correspondiente (Delta_Sn en puntos de %Sn
    para el target ganador de cada fase; Delta_FeO en su escala nativa). Sirve como
    chequeo de fidelidad de esta consolidacion contra los numeros ya reportados en
    el notebook (Paso 45): Fusion Delta_Sn R2 lockbox~0.462, Reduccion Delta_Sn R2
    lockbox~0.965.
    """
    orden_batches = orden_cronologico_batches(df_base)
    filas = []
    for fase in FASES:
        for nombre_target, target_col, es_delta_sn in [
            ("Delta_Sn (escala fisica)", TARGET_PRESCRIPTIVO_GANADOR[fase], True),
            ("Delta_FeO", TARGET_FEO, False),
        ]:
            feats = FEATURES_STATE_ACTION[fase]
            cols = list(dict.fromkeys(feats + [target_col, "d_ley_sn_escoria_pct", "ley_sn_escoria_pct_prev", "Batch"]))
            sub_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev), cols].dropna(subset=feats + [target_col])
            sub_lockbox = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + [target_col])

            oof_pred = oof_groupkfold(sub_dev, feats, target_col)
            sub_dev_oof = sub_dev.assign(pred=oof_pred)

            res_oot = oot_walkforward(sub_dev, feats, target_col, orden_batches)
            sub_oot_valido = res_oot.loc[res_oot["pred_oot"].notna()].rename(columns={"pred_oot": "pred"})

            modelo_final = xgb.XGBRegressor(**PARAMS_XGB)
            modelo_final.fit(sub_dev[feats], sub_dev[target_col])
            sub_lockbox = sub_lockbox.assign(pred=modelo_final.predict(sub_lockbox[feats]))

            def _metrica(d):
                if es_delta_sn:
                    y = d["d_ley_sn_escoria_pct"]
                    pred = convertir_a_delta_sn_fisico(fase, d["pred"], d["ley_sn_escoria_pct_prev"])
                else:
                    y, pred = d[target_col], d["pred"]
                return r2_score(y, pred), mean_absolute_error(y, pred), _rmse(y, pred)

            r2_oof, mae_oof, _ = _metrica(sub_dev_oof)
            r2_oot, mae_oot, _ = _metrica(sub_oot_valido)
            r2_lb, mae_lb, rmse_lb = _metrica(sub_lockbox)
            filas.append({
                "fase": fase, "target": nombre_target,
                "n_dev": len(sub_dev), "n_lockbox": len(sub_lockbox),
                "r2_oof_dev": r2_oof, "mae_oof_dev": mae_oof,
                "r2_oot_dev": r2_oot, "mae_oot_dev": mae_oot,
                "r2_lockbox": r2_lb, "mae_lockbox": mae_lb, "rmse_lockbox": rmse_lb,
            })
    return pd.DataFrame(filas).set_index(["fase", "target"])


# =============================================================================
# Sistema prescriptivo: ensambles + soporte historico + simulador + optimizador
# =============================================================================

@dataclass
class SistemaPrescriptivo:
    """Contenedor del modelo final ya entrenado: ensambles, soporte historico y limites de accion.

    No se re-entrena en cada llamada a `simulate_action`/`optimizar_accion`; se
    construye UNA vez con `entrenar_sistema(df)` y se reutiliza.
    """
    ensambles: dict = field(default_factory=dict)          # (fase, "delta_Sn"|"delta_FeO") -> list[XGBRegressor]
    soporte: dict = field(default_factory=dict)             # fase -> {"scaler","nn","dist_ref_media","features"}
    limites_accion: dict = field(default_factory=dict)      # fase -> {primitiva: (p1, p99)}
    batches_dev: set = field(default_factory=set)
    batches_lockbox: set = field(default_factory=set)


def _entrenar_ensamble(sub_dev_fase: pd.DataFrame, features: list[str], target_col: str,
                        n_splits: int = N_SPLITS_OOF) -> list:
    """Ensamble de N_SPLITS modelos (uno por fold de entrenamiento de un GroupKFold por Batch).

    Se usa el PROMEDIO de sus predicciones como punto estimado y la DESVIACION
    ESTANDAR entre modelos como `uncertainty` relativa (Paso 43) -- no es un
    intervalo de confianza calibrado (subestima la incertidumbre real de
    generalizacion, ver Paso 43/45 punto 11), solo sirve para comparar candidatos
    de accion entre si.
    """
    sub = sub_dev_fase.dropna(subset=features + [target_col])
    modelos = []
    for train_idx, _ in GroupKFold(n_splits=n_splits).split(sub[features], sub[target_col], sub["Batch"]):
        modelo = xgb.XGBRegressor(**PARAMS_XGB)
        modelo.fit(sub[features].iloc[train_idx], sub[target_col].iloc[train_idx])
        modelos.append(modelo)
    return modelos


def _construir_soporte_historico(sub_dev_fase: pd.DataFrame, features: list[str],
                                  k_vecinos: int = K_VECINOS_SOPORTE) -> dict:
    """Infraestructura de `support_score` (Paso 42): densidad historica de (State, Action)."""
    sub = sub_dev_fase[features].dropna()
    scaler = StandardScaler().fit(sub)
    Xs = scaler.transform(sub)
    nn = NearestNeighbors(n_neighbors=k_vecinos + 1).fit(Xs)
    dist_ref, _ = nn.kneighbors(Xs)
    dist_ref_media = float(dist_ref[:, 1:].mean())
    return {"scaler": scaler, "nn": nn, "dist_ref_media": dist_ref_media, "features": features, "k": k_vecinos}


def calcular_support_score(soporte_fase: dict, fila_features: dict) -> float:
    """exp(-dist_media / dist_ref_media): ~0.34-0.37 es la densidad TIPICA de un punto

    historico (no 1.0, que solo ocurre para un punto identico a otro ya
    observado); decae hacia 0 a medida que (State, Action) se aleja de la region
    observada. Paso 42/47 del notebook.
    """
    feats = soporte_fase["features"]
    x = pd.DataFrame([[fila_features[f] for f in feats]], columns=feats)
    xs = soporte_fase["scaler"].transform(x)
    dist, _ = soporte_fase["nn"].kneighbors(xs, n_neighbors=soporte_fase["k"])
    return float(np.exp(-dist.mean() / soporte_fase["dist_ref_media"]))


def entrenar_sistema(df: pd.DataFrame, pct_lockbox: float = N_LOCKBOX_PCT) -> SistemaPrescriptivo:
    """Entrena el sistema prescriptivo completo (ensambles + soporte + limites de accion).

    Usa SOLO batches DEV (nunca lockbox) para entrenar y calibrar, siguiendo la
    disciplina del Paso 41: el lockbox se reserva integramente para
    `tabla_validacion_3_niveles`, nunca para ajustar nada del sistema.
    """
    df_base = dataset_base_modelo(df)
    batches_dev, batches_lockbox = split_dev_lockbox(df, pct_lockbox)

    sistema = SistemaPrescriptivo(batches_dev=batches_dev, batches_lockbox=batches_lockbox)
    for fase in FASES:
        sub_fase_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        feats_sa = FEATURES_STATE_ACTION[fase]

        sistema.ensambles[(fase, "delta_Sn")] = _entrenar_ensamble(sub_fase_dev, feats_sa, TARGET_PRESCRIPTIVO_GANADOR[fase])
        sistema.ensambles[(fase, "delta_FeO")] = _entrenar_ensamble(sub_fase_dev, feats_sa, TARGET_FEO)
        sistema.soporte[fase] = _construir_soporte_historico(sub_fase_dev, feats_sa)

        sub_acciones = sub_fase_dev[ACCIONES_PRIMITIVAS[fase]].dropna()
        sistema.limites_accion[fase] = {
            a: (float(sub_acciones[a].quantile(0.01)), float(sub_acciones[a].quantile(0.99)))
            for a in ACCIONES_PRIMITIVAS[fase]
        }
    return sistema


# =============================================================================
# Simulador contrafactual: state + action -> resultado metalurgico siguiente
# =============================================================================

def recalcular_derivadas(fase: str, state: dict, action: dict) -> dict:
    """Recalcula, a partir de las primitivas de ACCION + el STATE dado, todas las
    features derivadas que necesita `FEATURES_STATE_ACTION[fase]`.

    Nunca se optimiza una derivada/razon directamente (Paso 43): el optimizador
    siempre mueve primitivas de `ACCIONES_PRIMITIVAS`, y esta funcion reconstruye
    el resto exactamente como lo haria `feature_engineering` sobre datos reales.
    """
    fila = dict(state)
    if fase == "Fusión":
        fila["delta_tiempo"] = action["delta_tiempo"]
        fila["tasa_feed_Carbon_kg_min"] = action["tasa_feed_Carbon_kg_min"]
        fila["interaccion_C_x_Sn_prev"] = action["tasa_feed_Carbon_kg_min"] * state["ley_sn_escoria_pct_prev"]
    else:
        fila["tasa_gn_nm3_min"] = action["tasa_gn_nm3_min"]
        fila["posicion_vertical_lanza_mm"] = action["posicion_vertical_lanza_mm"]
        o2_teorico = fe.RELACION_O2_CH4 * action["tasa_gn_nm3_min"]
        o2_disponible = action["tasa_o2_nm3_min"] + fe.FRACCION_O2_EN_AIRE * action["tasa_aire_nm3_min"]
        fila["exceso_o2_combustion_pct"] = 100 * (o2_disponible - o2_teorico) / o2_teorico if o2_teorico > 0 else np.nan

    faltantes = [f for f in FEATURES_STATE_ACTION[fase] if f not in fila]
    if faltantes:
        raise KeyError(f"recalcular_derivadas no pudo construir: {faltantes} (revisar state/action de entrada)")
    return fila


def simulate_action(sistema: SistemaPrescriptivo, fase: str, state: dict, action: dict) -> dict:
    """Simulador contrafactual state+action -> resultado metalurgico siguiente (Paso 43).

    state: dict con FEATURES_STATE[fase] (conocidas al cierre del escalon anterior).
    action: dict con ACCIONES_PRIMITIVAS[fase] (candidata a evaluar).
    Devuelve pred_delta_sn / pred_delta_feo (puntos de %Sn/%FeO, escala fisica),
    `uncertainty` (std del ensamble, orientativa/relativa, no calibrada) y
    `support_score` (densidad historica de esa combinacion State+Action).
    No se ofrece `pred_temperature`: el modelo de Delta_T no generaliza OOT
    (R2=-0.04, Paso 38) y forzar una prediccion no confiable seria peor que no
    ofrecerla.
    """
    fila = recalcular_derivadas(fase, state, action)
    feats = FEATURES_STATE_ACTION[fase]
    x = pd.DataFrame([[fila[f] for f in feats]], columns=feats)

    def _predecir(nombre_target):
        preds = np.array([m.predict(x)[0] for m in sistema.ensambles[(fase, nombre_target)]])
        return float(preds.mean()), float(preds.std())

    pred_sn_raw, unc_sn_raw = _predecir("delta_Sn")
    pred_sn = float(convertir_a_delta_sn_fisico(fase, np.array([pred_sn_raw]), np.array([state["ley_sn_escoria_pct_prev"]]))[0])
    # La incertidumbre tambien se reescala a la unidad fisica cuando el target es T3 (fraccion relativa).
    unc_sn = unc_sn_raw if TARGET_PRESCRIPTIVO_GANADOR[fase] == "d_ley_sn_escoria_pct" else unc_sn_raw * abs(state["ley_sn_escoria_pct_prev"])
    pred_feo, unc_feo = _predecir("delta_FeO")

    return {
        "fase": fase,
        "pred_delta_sn": pred_sn, "pred_delta_feo": pred_feo,
        "pred_temperature": None,
        "uncertainty": {"delta_sn_std": unc_sn, "delta_feo_std": unc_feo, "temperature": None},
        "support_score": calcular_support_score(sistema.soporte[fase], fila),
    }


# =============================================================================
# Funcion objetivo (J_R, J_F) y optimizador restringido (Paso 44)
# =============================================================================

def costo_reactivos_reduccion(action: dict) -> float:
    return action["tasa_gn_nm3_min"] + 0.5 * action["tasa_o2_nm3_min"] + 0.05 * action["tasa_aire_nm3_min"]


def costo_reactivos_fusion(action: dict) -> float:
    return action["tasa_feed_Carbon_kg_min"]


def objetivo_J_R(sistema: SistemaPrescriptivo, state: dict, action: dict, pesos: dict = PESOS_J_R_DEFAULT) -> tuple[float, dict]:
    """Funcion objetivo de Reduccion (maximizar extraccion de Sn sin sobre-reducir FeO).

    J_R = V_Sn*(-Delta_Sn) + lambda_Fe*Delta_FeO - lambda_C*C - lambda_E*E - lambda_t*t
          - lambda_U*uncertainty - lambda_S*(1-support_score)

    -Delta_Sn es positivo porque en Reduccion Delta_Sn es negativo (mas caida de
    %Sn = mejor). Delta_FeO entra SIN signo adicional: una caida fuerte de %FeO
    (sobre-reduccion hacia Fe metalico/hardhead) vuelve el termino negativo y
    penaliza -- el mecanismo que evita perseguir el minimo de %Sn a cualquier
    costo de Fe reducido. Pesos configurables, NUNCA "definitivos" (Paso 44):
    ajustar segun el apetito de riesgo de sobre-reduccion que defina el equipo
    de proceso.
    """
    sim = simulate_action(sistema, "Reducción", state, action)
    C = costo_reactivos_reduccion(action)
    E = C  # proxy: mismo consumo de gases aproxima tambien el termino energetico (sin precios reales en el dataset)
    t = action.get("delta_tiempo", state.get("delta_tiempo_tipico", 20))
    unc = sim["uncertainty"]["delta_sn_std"] + sim["uncertainty"]["delta_feo_std"]
    J = (pesos["V_Sn"] * (-sim["pred_delta_sn"]) + pesos["lambda_Fe"] * sim["pred_delta_feo"]
         - pesos["lambda_C"] * C - pesos["lambda_E"] * E - pesos["lambda_t"] * t
         - pesos["lambda_U"] * unc - pesos["lambda_S"] * (1 - sim["support_score"]))
    return J, sim


def objetivo_J_F(sistema: SistemaPrescriptivo, state: dict, action: dict, pesos: dict = PESOS_J_F_DEFAULT) -> tuple[float, dict]:
    """Funcion objetivo de Fusion (multiobjetivo, sin pesos "correctos" definitivos -- Paso 44).

    `direccion_sn_fusion` es un parametro de NEGOCIO explicito, no una asuncion
    oculta: +1 trata crecer %Sn en escoria durante Fusion como deseable (carga
    que se recuperara despues en Reduccion); -1 lo trata como indeseable en si
    mismo. Este modulo no decide cual es correcto -- lo deja como argumento.
    """
    sim = simulate_action(sistema, "Fusión", state, action)
    E = costo_reactivos_fusion(action)
    t = action.get("delta_tiempo", 44)
    unc = sim["uncertainty"]["delta_sn_std"] + sim["uncertainty"]["delta_feo_std"]
    penal_b2 = abs(state.get("basicidad_B2_prev", pesos["basicidad_optima"]) - pesos["basicidad_optima"])
    J = (pesos["w_Sn"] * pesos["direccion_sn_fusion"] * sim["pred_delta_sn"] + pesos["w_FeO"] * sim["pred_delta_feo"]
         - pesos["w_B2"] * penal_b2 - pesos["w_E"] * E - pesos["w_t"] * t
         - pesos["w_U"] * unc - pesos["w_S"] * (1 - sim["support_score"]))
    return J, sim


def optimizar_accion(sistema: SistemaPrescriptivo, fase: str, state: dict,
                      n_candidatos: int = 300, semilla: int = 0, pesos: dict | None = None) -> tuple[dict, float, dict]:
    """Busqueda aleatoria de la mejor accion candidata para `state`, acotada a los

    percentiles P1-P99 historicos (DEV) de cada primitiva (`sistema.limites_accion`).
    Deliberadamente simple y barata (sin busqueda masiva, Paso 44 seccion 5 del
    pedido original): el `support_score` ya penaliza salir de la region observada
    dentro de J, por lo que no hace falta una restriccion dura adicional aparte de
    los percentiles (que solo evitan evaluar candidatos fisicamente absurdos).
    """
    objetivo, pesos_default = (objetivo_J_R, PESOS_J_R_DEFAULT) if fase == "Reducción" else (objetivo_J_F, PESOS_J_F_DEFAULT)
    pesos = pesos or pesos_default
    limites = sistema.limites_accion[fase]
    rng = np.random.default_rng(semilla)

    mejor_J, mejor_accion, mejor_sim = -np.inf, None, None
    for _ in range(n_candidatos):
        candidato = {a: rng.uniform(lo, hi) for a, (lo, hi) in limites.items()}
        J, sim = objetivo(sistema, state, candidato, pesos)
        if J > mejor_J:
            mejor_J, mejor_accion, mejor_sim = J, candidato, sim
    return mejor_accion, mejor_J, mejor_sim


# =============================================================================
# De escalon a BATCH: reporte de recomendaciones y eslabon con el rendimiento
# =============================================================================

def reporte_recomendaciones_batch(sistema: SistemaPrescriptivo, df: pd.DataFrame, batch_id: str,
                                   n_candidatos: int = 300, semilla: int = 0,
                                   pesos_R: dict | None = None, pesos_F: dict | None = None) -> pd.DataFrame:
    """Recomendacion escalon-por-escalon para UN batch completo, lista para revision operativa.

    Metodologia (deliberada, se declara para que no se lea como mas de lo que
    es): en CADA escalon del batch se usa su propio ESTADO HISTORICO REAL (el
    `_prev` observado, no un estado resultante de encadenar recomendaciones
    anteriores). Esto es intencional, no una limitacion de implementacion: el
    modulo NO tiene modelos validados para las variables de estado que no son
    %Sn/%FeO (temperatura -- Paso 38, R2 OOT=-0.04 -- ni CaO/SiO2, que
    determinan `basicidad_B2_prev`), asi que encadenar una simulacion completo-
    batch tendria que mantenerlas artificialmente congeladas en su valor
    historico mientras %Sn/%FeO si cambian con la accion recomendada -- un
    supuesto peor (y que compone error escalon a escalon) que evaluar cada
    escalon de forma independiente contra su propio estado real observado, que
    es exactamente la misma metodologia ya validada en el Paso 44 (12 escalones
    de muestra), aqui aplicada a los ~10 escalones completos de un batch.

    Cada fila compara la accion HISTORICA (la que el operador realmente aplico)
    contra la accion RECOMENDADA (optimizador restringido, Paso 44) sobre el
    MISMO estado de partida real, con sus predicciones de Delta_Sn/Delta_FeO,
    `support_score` y el valor de la funcion objetivo J.
    """
    df_base = dataset_base_modelo(df)
    sub = df_base.loc[df_base["Batch"] == batch_id].sort_values("fecha_inicio")
    if sub.empty:
        raise ValueError(f"Batch '{batch_id}' no tiene escalones modelables (revisar que exista y no sea solo su primer escalon).")

    filas = []
    for _, row in sub.iterrows():
        fase = row["fase_proceso"]
        state = {f: row[f] for f in FEATURES_STATE[fase]}
        accion_hist = {f: row[f] for f in ACCIONES_PRIMITIVAS[fase]}
        if any(pd.isna(v) for v in list(state.values()) + list(accion_hist.values())):
            continue  # escalon con algun insumo faltante: se omite del reporte, no se imputa silenciosamente

        objetivo = objetivo_J_R if fase == "Reducción" else objetivo_J_F
        pesos = (pesos_R if fase == "Reducción" else pesos_F)
        J_hist, sim_hist = objetivo(sistema, state, accion_hist, pesos) if pesos else objetivo(sistema, state, accion_hist)
        accion_opt, J_opt, sim_opt = optimizar_accion(sistema, fase, state, n_candidatos, semilla, pesos)

        fila = {
            "Batch": batch_id, "fase": fase, "orden_escalon_fase": row["orden_escalon_fase"],
            "ley_sn_escoria_pct_prev": state["ley_sn_escoria_pct_prev"],
            "d_ley_sn_escoria_pct_real": row["d_ley_sn_escoria_pct"],
            "d_ley_feo_escoria_pct_real": row.get("d_ley_feo_escoria_pct", np.nan),
            "pred_delta_sn_historico": sim_hist["pred_delta_sn"], "pred_delta_sn_recomendado": sim_opt["pred_delta_sn"],
            "pred_delta_feo_historico": sim_hist["pred_delta_feo"], "pred_delta_feo_recomendado": sim_opt["pred_delta_feo"],
            "support_historico": sim_hist["support_score"], "support_recomendado": sim_opt["support_score"],
            "J_historico": J_hist, "J_recomendado": J_opt, "uplift_J": J_opt - J_hist,
        }
        for a in ACCIONES_PRIMITIVAS[fase]:
            fila[f"accion_historica__{a}"] = accion_hist[a]
            fila[f"accion_recomendada__{a}"] = accion_opt[a]
        filas.append(fila)

    reporte = pd.DataFrame(filas)
    if reporte.empty:
        warnings.warn(f"Batch '{batch_id}': ningun escalon tenia todos los insumos necesarios; reporte vacio.")
    return reporte


def resumen_reporte_batch(reporte: pd.DataFrame) -> dict:
    """Agrega `reporte_recomendaciones_batch` a un resumen a nivel de batch.

    Ambos indicadores siguientes usan la MISMA formula (recomendado -
    historico), sumada solo sobre escalones de Reduccion -- se documenta el
    signo de cada uno por separado en vez de invertir la formula entre
    variables (mas transparente que "normalizar" el signo a mano y arriesgar
    una inversion equivocada):

    `delta_sn_recomendado_menos_historico_reduccion`: dado que Delta_Sn es
    NEGATIVO cuando se extrae Sn de la escoria, un valor NEGATIVO aqui es
    FAVORABLE (el paquete de recomendaciones, evaluado escalon por escalon
    sobre su propio estado real, extraeria en conjunto mas %Sn de la escoria
    que la politica historica); un valor positivo es desfavorable.
    `delta_feo_recomendado_menos_historico_reduccion`: dado que Delta_FeO
    NEGATIVO significa mas FeO reducido (riesgo de hardhead), un valor
    POSITIVO aqui es FAVORABLE (el paquete recomendado reduciria en conjunto
    MENOS FeO que la politica historica -- menor riesgo de sobre-reduccion);
    un valor negativo es una alerta a revisar antes de aceptar el paquete
    completo, no solo escalon a escalon.

    Ninguno de los dos es una cantidad de toneladas ni un delta exacto de
    `rendimiento_proxy_batch`: no hay masa de escoria por escalon en el
    dataset para cerrar ese balance (hallazgos.md seccion 6b/9.5), y es un
    uplift ESTIMADO por el propio modelo, nunca observado en produccion
    (Paso 44).
    """
    if reporte.empty:
        return {}
    es_reduccion = reporte["fase"] == "Reducción"
    return {
        "n_escalones_evaluados": len(reporte),
        "n_escalones_reduccion": int(es_reduccion.sum()),
        "delta_sn_recomendado_menos_historico_reduccion": float((reporte.loc[es_reduccion, "pred_delta_sn_recomendado"]
                                                                  - reporte.loc[es_reduccion, "pred_delta_sn_historico"]).sum()),
        "delta_feo_recomendado_menos_historico_reduccion": float((reporte.loc[es_reduccion, "pred_delta_feo_recomendado"]
                                                                   - reporte.loc[es_reduccion, "pred_delta_feo_historico"]).sum()),
        "uplift_J_medio": float(reporte["uplift_J"].mean()),
        "pct_escalones_con_mejora": float((reporte["uplift_J"] > 0).mean() * 100),
        "support_medio_historico": float(reporte["support_historico"].mean()),
        "support_medio_recomendado": float(reporte["support_recomendado"].mean()),
    }


def evaluar_relacion_selectividad_vs_rendimiento(df: pd.DataFrame) -> pd.DataFrame:
    """Evidencia empirica del eslabon escalon -> rendimiento del batch (mecanismo redox Sn-Fe).

    Mecanismo cualitativo (guia_fenomenologica_ausmelt_sn.md; consideraciones
    secciones 5.3/11.3-11.4): sobre-reducir FeO en Reduccion arrastra Fe hacia el
    metal/hardhead; ese Fe reducido en exceso es exactamente lo que la
    contabilidad del batch registra como `sn_en_dross_fe_batch_t` (Sn atrapado en
    dross de Fe) en vez de `sn_en_metal_crudo_batch_t` -- el denominador y
    numerador, respectivamente, de `rendimiento_proxy_batch`
    (feature_engineering.agregar_rendimiento_proxy_batch). `indice_selectividad_sn_feo`
    (caida de %Sn / caida de %FeO por escalon, feature_engineering.agregar_indice_selectividad_sn_feo)
    operacionaliza esa selectividad: un batch con selectividad media ALTA en
    Reduccion extrajo Sn "limpio" (poco FeO arrastrado); uno con selectividad
    BAJA sobre-redujo FeO por cada punto de Sn ganado.

    Se reporta la correlacion (Spearman) REAL entre la selectividad media de
    Reduccion de cada batch y (a) `rendimiento_proxy_batch`, (b) la fraccion de
    Sn de salida que terminó en dross de Fe -- sin asumir de antemano que sea
    fuerte: hallazgos.md seccion 6b ya documento que agregados lineales simples
    de escalon correlacionan debil (|rho|<=0.18) con `rendimiento_proxy_batch`,
    precisamente por no existir masa de escoria por escalon para cerrar un
    balance de masa explicito.
    """
    df_ind = fe.agregar_indice_selectividad_sn_feo(df)
    df_red = df_ind.loc[df_ind["fase_proceso"] == "Reducción"]

    resumen_batch = df_red.groupby("Batch").agg(
        selectividad_media_reduccion=("indice_selectividad_sn_feo", "mean"),
        n_escalones_selectividad_valida=("indice_selectividad_sn_feo", "count"),
        rendimiento_proxy_batch=("rendimiento_proxy_batch", "first"),
        sn_en_metal_crudo_batch_t=("sn_en_metal_crudo_batch_t", "first"),
        sn_en_dross_fe_batch_t=("sn_en_dross_fe_batch_t", "first"),
        sn_en_polvo_fundicion_batch_t=("sn_en_polvo_fundicion_batch_t", "first"),
    ).reset_index()
    sn_total = (resumen_batch["sn_en_metal_crudo_batch_t"] + resumen_batch["sn_en_dross_fe_batch_t"]
                + resumen_batch["sn_en_polvo_fundicion_batch_t"])
    resumen_batch["frac_sn_a_dross_fe_pct"] = 100 * fe.division_segura(resumen_batch["sn_en_dross_fe_batch_t"], sn_total)

    filas = []
    valido = resumen_batch.dropna(subset=["selectividad_media_reduccion", "rendimiento_proxy_batch"])
    for col_objetivo in ["rendimiento_proxy_batch", "frac_sn_a_dross_fe_pct"]:
        sub = valido.dropna(subset=[col_objetivo])
        rho, p = spearmanr(sub["selectividad_media_reduccion"], sub[col_objetivo])
        filas.append({
            "objetivo": col_objetivo, "n_batches": len(sub),
            "rho_spearman_vs_selectividad_media_reduccion": rho, "p_valor": p,
        })
    return pd.DataFrame(filas).set_index("objetivo")
