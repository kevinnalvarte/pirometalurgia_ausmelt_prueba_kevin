"""Seguimiento rapido del modelo de batch: dado que 38 features sobre 299 batches DEV
dio R2 OOT walk-forward NEGATIVO (peor que predecir la media) pese a R2 OOF/lockbox
positivos -- firma clasica de sobreajuste/inestabilidad temporal con demasiadas features
correlacionadas para el tamano de muestra -- se prueba una version mas chica (top-10 por
importancia XGBoost) y mas regularizada, para ver si la parsimonia (misma leccion ya
aprendida en Pasos 18-19 a nivel de escalon) recupera estabilidad OOT a nivel de BATCH.
"""
import os
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, r"C:\Users\user\Desktop\Linea de Sn\Lingo Smelter")
import modelo_prescriptivo as mp

OUT_DIR = os.path.dirname(__file__)
df_batch = pd.read_csv(f"{OUT_DIR}/df_batch_features.csv", index_col="Batch")

df = mp.construir_dataset_modelo(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datos Lingo smelter fase II.xlsx"))
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
orden_batches = mp.orden_cronologico_batches(df)

TOP10 = ["reduccion_feo_caida_total", "reduccion_ley_feo_final", "reduccion_carbon_total_kg",
         "fusion_carbon_total_kg", "fusion_temp_mean", "reduccion_o2_total_nm3",
         "reduccion_lanza_pos_mean", "fusion_o2_exceso_mean", "reduccion_sn_caida_total",
         "delta_FR_ley_feo_escoria_pct"]
TOP5 = TOP10[:5]

PARAMS_SMALL = dict(n_estimators=80, learning_rate=0.03, max_depth=2,
                     min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                     reg_lambda=5.0, random_state=42, n_jobs=-1)


def evaluar(features, target_col, nombre_set, n_estimators_grid=(80,)):
    d = df_batch.dropna(subset=[target_col]).copy()
    orden_validos = [b for b in orden_batches if b in d.index]
    d = d.loc[orden_validos]
    dev_mask = d.index.isin(batches_dev)
    d_dev, d_lb = d.loc[dev_mask], d.loc[~dev_mask]
    X_dev, y_dev = d_dev[features], d_dev[target_col]
    X_lb, y_lb = d_lb[features], d_lb[target_col]

    def oof(modelo_fn, X, y, seed=0):
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        out = np.full(len(X), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            med = X.iloc[tr].median()
            m = modelo_fn(); m.fit(X.iloc[tr].fillna(med), y.iloc[tr])
            out[te] = m.predict(X.iloc[te].fillna(med))
        return out

    def wf(modelo_fn, X, y):
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        out = np.full(len(X), np.nan)
        for tr, te in TimeSeriesSplit(5).split(X):
            med = X.iloc[tr].median()
            m = modelo_fn(); m.fit(X.iloc[tr].fillna(med), y.iloc[tr])
            out[te] = m.predict(X.iloc[te].fillna(med))
        return out

    for nombre, modelo_fn in [
        ("XGBoost chico regularizado", lambda: xgb.XGBRegressor(**PARAMS_SMALL)),
        ("ElasticNet", lambda: ElasticNetCV(cv=5, random_state=0, max_iter=5000)),
    ]:
        o = oof(modelo_fn, X_dev, y_dev)
        w = wf(modelo_fn, X_dev, y_dev)
        valid_w = ~np.isnan(w)
        med = X_dev.median()
        mf = modelo_fn(); mf.fit(X_dev.fillna(med), y_dev)
        pred_lb = mf.predict(X_lb.fillna(med))
        print(f"[{nombre_set}] {nombre} (target={target_col}, k={len(features)}): "
              f"R2_oof={r2_score(y_dev, o):.4f}  "
              f"R2_oot_wf={r2_score(y_dev.to_numpy()[valid_w], w[valid_w]) if valid_w.sum()>3 else float('nan'):.4f}  "
              f"R2_lockbox={r2_score(y_lb, pred_lb):.4f}  MAE_lockbox={mean_absolute_error(y_lb, pred_lb):.4f}")


for target_col in ["rendimiento_proxy_batch", "recuperacion_real_batch_pct"]:
    print(f"\n===== {target_col} =====")
    evaluar(TOP10, target_col, "top10")
    evaluar(TOP5, target_col, "top5")
    evaluar(["reduccion_sn_caida_total"], target_col, "baseline_univariado")
print("\nListo.")
