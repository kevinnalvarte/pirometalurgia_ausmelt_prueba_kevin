"""Prototipo: (a) bake-off de algoritmos (XGBoost baseline, XGBoost con restricciones
MONOTONICAS en las variables de dosis de carbon -- unica relacion confirmada "sin reservas"
en TODAS las especificaciones previas, hallazgos.md 6b/9/10 -- HistGradientBoostingRegressor,
ElasticNet lineal) bajo la MISMA validacion de 3 niveles ya establecida (OOF/OOT/lockbox); y
(b) intervalos de prediccion CV+ (Barber, Candes, Ramdas & Tibshirani 2021) construidos sobre
el ENSAMBLE de K modelos de GroupKFold que modelo_prescriptivo.py ya entrena -- reemplaza la
'incertidumbre no calibrada' (std del ensamble, declarada explicitamente como no confiable en
modelo_prescriptivo.py) por un intervalo con cobertura empirica VALIDADA en el lockbox (nunca
usado para calibrar nada).
"""
import os
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, r"C:\Users\user\Desktop\Linea de Sn\Lingo Smelter")
import modelo_prescriptivo as mp

OUT_DIR = os.path.dirname(__file__)

df = mp.construir_dataset_modelo(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datos Lingo smelter fase II.xlsx"))
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
orden_batches = mp.orden_cronologico_batches(df)

# Variables con relacion NEGATIVA confirmada "sin reservas" en todas las especificaciones
# previas (dosis-respuesta monotona limpia, hallazgos.md 6b: "chequeo superado sin reservas"):
# a mayor dosis/interaccion de carbon, mayor caida (mas negativo) de d_ley_sn_escoria_pct / mas
# negativo el target T3. Unica familia de variables tratada como candidata a restriccion
# monotonica -- posicion_vertical_lanza_mm queda EXCLUIDA a proposito (Pasos 46-47: optimo
# interior confirmado, monotonizarla seria pirometalurgicamente incorrecto).
VARIABLES_MONOTONAS_NEGATIVAS = {
    "tasa_feed_Carbon_kg_min", "cum_feed_Carbon_kg_prev", "interaccion_C_x_Sn_prev",
}


def construir_monotone_constraints(features):
    return tuple(-1 if f in VARIABLES_MONOTONAS_NEGATIVAS else 0 for f in features)


def modelos_bakeoff(features):
    mono = construir_monotone_constraints(features)
    return {
        "XGBoost (actual, sin restriccion)": lambda: xgb.XGBRegressor(**mp.PARAMS_XGB),
        "XGBoost + monotonico(carbon)": lambda: xgb.XGBRegressor(**dict(mp.PARAMS_XGB, monotone_constraints=mono)),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15,
            l2_regularization=1.0, random_state=42),
        "HistGB + monotonico(carbon)": lambda: HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15,
            l2_regularization=1.0, random_state=42,
            monotonic_cst=[(-1 if f in VARIABLES_MONOTONAS_NEGATIVAS else 0) for f in features]),
        "ElasticNet (lineal)": lambda: ElasticNetCV(cv=5, random_state=0, max_iter=5000),
    }


def evaluar_bakeoff(fase):
    feats = mp.FEATURES_STATE_ACTION[fase]
    target = mp.TARGET_PRESCRIPTIVO_GANADOR[fase]
    cols = list(dict.fromkeys(feats + [target, "d_ley_sn_escoria_pct", "ley_sn_escoria_pct_prev", "Batch"]))
    sub_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev), cols].dropna()
    sub_lb = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_lockbox), cols].dropna()

    modelos = modelos_bakeoff(feats)
    filas = []
    for nombre, modelo_fn in modelos.items():
        necesita_scaler = nombre.startswith("ElasticNet")
        scaler = StandardScaler().fit(sub_dev[feats]) if necesita_scaler else None

        def fit_predict(X_tr, y_tr, X_te):
            if scaler is not None:
                X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
            m = modelo_fn()
            m.fit(X_tr, y_tr)
            return m.predict(X_te)

        # OOF (GroupKFold por Batch)
        oof = np.full(len(sub_dev), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(sub_dev[feats], sub_dev[target], sub_dev["Batch"]):
            oof[te] = fit_predict(sub_dev[feats].iloc[tr], sub_dev[target].iloc[tr], sub_dev[feats].iloc[te])
        pred_oof_fisico = mp.convertir_a_delta_sn_fisico(fase, oof, sub_dev["ley_sn_escoria_pct_prev"].to_numpy())

        # Walk-forward OOT
        res_oot = mp.oot_walkforward(sub_dev, feats, target, orden_batches) if nombre.startswith("XGBoost (actual") else None

        # Lockbox: entrena en TODO dev, predice lockbox
        X_dev_fit, X_lb_fit = sub_dev[feats], sub_lb[feats]
        if scaler is not None:
            X_dev_fit, X_lb_fit = scaler.transform(X_dev_fit), scaler.transform(X_lb_fit)
        m_final = modelo_fn()
        m_final.fit(X_dev_fit, sub_dev[target])
        pred_lb_raw = m_final.predict(X_lb_fit)
        pred_lb_fisico = mp.convertir_a_delta_sn_fisico(fase, pred_lb_raw, sub_lb["ley_sn_escoria_pct_prev"].to_numpy())

        y_dev_fisico = sub_dev["d_ley_sn_escoria_pct"].to_numpy()
        y_lb_fisico = sub_lb["d_ley_sn_escoria_pct"].to_numpy()

        filas.append({
            "fase": fase, "modelo": nombre,
            "r2_oof": r2_score(y_dev_fisico, pred_oof_fisico),
            "mae_oof": mean_absolute_error(y_dev_fisico, pred_oof_fisico),
            "r2_lockbox": r2_score(y_lb_fisico, pred_lb_fisico),
            "mae_lockbox": mean_absolute_error(y_lb_fisico, pred_lb_fisico),
            "n_dev": len(sub_dev), "n_lockbox": len(sub_lb),
        })
    return pd.DataFrame(filas)


tabla_bakeoff = pd.concat([evaluar_bakeoff(fase) for fase in mp.FASES], ignore_index=True)
print(tabla_bakeoff.to_string(index=False))
tabla_bakeoff.to_csv(f"{OUT_DIR}/tabla_bakeoff.csv", index=False)


# =============================================================================
# CV+ conformal prediction sobre el ensamble YA existente (GroupKFold, 5 modelos)
# =============================================================================
def construir_cv_plus(fase, target_col):
    feats = mp.FEATURES_STATE_ACTION[fase]
    cols = list(dict.fromkeys(feats + [target_col, "Batch"]))
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev), cols].dropna().reset_index(drop=True)
    X, y, groups = sub[feats], sub[target_col].to_numpy(), sub["Batch"].to_numpy()

    modelos_fold, residuos_fold = [], []
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(**mp.PARAMS_XGB)
        m.fit(X.iloc[tr], y[tr])
        modelos_fold.append(m)
        residuos_fold.append(np.abs(y[te] - m.predict(X.iloc[te])))  # |residuo| absoluto OOF de ese fold
    return modelos_fold, residuos_fold, feats


def intervalo_cv_plus(modelos_fold, residuos_fold, x_row_df, alpha=0.20):
    """CV+ (Barber et al. 2021): junta, para cada punto historico de calibracion,
    la prediccion del modelo QUE NO LO VIO evaluada en x_nuevo +/- su propio residuo OOF,
    y toma los percentiles alpha/2 y 1-alpha/2 de esa nube combinada."""
    lo_candidatos, hi_candidatos = [], []
    for m, residuos in zip(modelos_fold, residuos_fold):
        pred_en_x = m.predict(x_row_df)[0]
        lo_candidatos.extend(pred_en_x - residuos)
        hi_candidatos.extend(pred_en_x + residuos)
    return float(np.quantile(lo_candidatos, alpha / 2)), float(np.quantile(hi_candidatos, 1 - alpha / 2))


def validar_cobertura_lockbox(fase, target_col, alpha=0.20):
    feats = mp.FEATURES_STATE_ACTION[fase]
    modelos_fold, residuos_fold, feats = construir_cv_plus(fase, target_col)
    cols = list(dict.fromkeys(feats + [target_col, "Batch"]))
    sub_lb = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_lockbox), cols].dropna().reset_index(drop=True)

    cubiertos, anchos = [], []
    for i in range(len(sub_lb)):
        x_row = sub_lb[feats].iloc[[i]]
        lo, hi = intervalo_cv_plus(modelos_fold, residuos_fold, x_row, alpha)
        y_real = sub_lb[target_col].iloc[i]
        cubiertos.append(lo <= y_real <= hi)
        anchos.append(hi - lo)
    cobertura = float(np.mean(cubiertos))
    return cobertura, float(np.mean(anchos)), len(sub_lb)


print("\n=== Validacion de cobertura empirica CV+ en LOCKBOX (nunca usado para calibrar nada) ===")
for fase in mp.FASES:
    target = mp.TARGET_PRESCRIPTIVO_GANADOR[fase]
    for alpha, nombre_nivel in [(0.20, "80%"), (0.10, "90%")]:
        cobertura, ancho_medio, n = validar_cobertura_lockbox(fase, target, alpha)
        print(f"{fase:12s} target={target:26s} nominal={nombre_nivel}  cobertura_empirica={cobertura:.3f}  ancho_medio={ancho_medio:.4f}  n_lockbox={n}")

print("\nListo.")
