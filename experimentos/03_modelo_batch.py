"""Prototipo: modelo de RENDIMIENTO DE BATCH con features no-lineales ricas.

Motivacion (hallazgos.md 6b/9.5/11.7): el cuello de botella documentado en TODAS las rondas
anteriores es que los agregados lineales simples por batch (sumas/promedios, construir_resumen_batch_eda)
correlacionan debil (|Spearman|<=0.13-0.22) con rendimiento_proxy_batch/recuperacion_real_batch_pct,
y nunca se probo un modelo NO LINEAL (XGBoost) sobre un pool de features de batch mas rico
(perfiles de dosificacion, fraccion de escalones en la zona de riesgo de posicion de lanza
hallada en Pasos 46-47, indices cineticos ya construidos a nivel de escalon, rollup del propio
modelo de escalon ya validado) -- eso es exactamente lo que se hace aqui.
"""
import os
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

sys.path.insert(0, r"C:\Users\user\Desktop\Linea de Sn\Lingo Smelter")
import feature_engineering as fe
import modelo_prescriptivo as mp

OUT_DIR = os.path.dirname(__file__)

df = mp.construir_dataset_modelo(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datos Lingo smelter fase II.xlsx"))
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
orden_batches = mp.orden_cronologico_batches(df)

VALLE_LANZA = (3300, 4200)  # Pasos 46-47: zona de mayor caida simultanea de %Sn y %FeO en Reduccion
BASICIDAD_OPTIMA = 1.4  # ya usada como parametro de negocio en PESOS_J_F_DEFAULT


# -----------------------------------------------------------------------------
# 1) Rollup del modelo de escalon YA VALIDADO (nonlinear), sin fuga batch->batch
# -----------------------------------------------------------------------------
def construir_rollup_modelo_escalon(df_base, batches_dev, batches_lockbox):
    """pred_delta_sn fisico por escalon de Reduccion: OOF (GroupKFold) para DEV,
    prediccion directa de un modelo entrenado en TODO DEV para LOCKBOX (ambos casos:
    el modelo nunca vio el batch que esta prediciendo)."""
    fase = "Reducción"
    feats = mp.FEATURES_STATE_ACTION[fase]
    target = mp.TARGET_PRESCRIPTIVO_GANADOR[fase]
    cols = list(dict.fromkeys(feats + [target, "ley_sn_escoria_pct_prev", "Batch"]))

    sub_dev = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev), cols].dropna()
    sub_lb = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_lockbox), cols].dropna()

    oof = mp.oof_groupkfold(sub_dev, feats, target)
    pred_dev = mp.convertir_a_delta_sn_fisico(fase, oof, sub_dev["ley_sn_escoria_pct_prev"].to_numpy())

    modelo_final = xgb.XGBRegressor(**mp.PARAMS_XGB)
    modelo_final.fit(sub_dev[feats], sub_dev[target])
    pred_lb = mp.convertir_a_delta_sn_fisico(fase, modelo_final.predict(sub_lb[feats]), sub_lb["ley_sn_escoria_pct_prev"].to_numpy())

    resultado = pd.concat([
        sub_dev[["Batch"]].assign(pred_delta_sn_modelo=pred_dev),
        sub_lb[["Batch"]].assign(pred_delta_sn_modelo=pred_lb),
    ])
    return resultado


rollup = construir_rollup_modelo_escalon(df_base, batches_dev, batches_lockbox)
rollup_agg = rollup.groupby("Batch")["pred_delta_sn_modelo"].sum().rename("suma_pred_delta_sn_reduccion_modelo")


# -----------------------------------------------------------------------------
# 2) Features de batch ricas (no lineales: perfiles, fracciones, dispersion, no solo sumas)
# -----------------------------------------------------------------------------
def slope(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or np.all(np.isnan(y)):
        return np.nan
    x = np.arange(len(y))
    m = ~np.isnan(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


filas = []
for batch_id, g in df.groupby("Batch"):
    g = g.sort_values("fecha_inicio")
    gf = g.loc[g["fase_proceso"] == "Fusión"]
    gr = g.loc[g["fase_proceso"] == "Reducción"]
    if gf.empty or gr.empty:
        continue

    fila = {"Batch": batch_id, "rendimiento_proxy_batch": g["rendimiento_proxy_batch"].iloc[0]}

    # ---- Fusion: perfil de dosificacion, no solo total ----
    fila["fusion_carbon_total_kg"] = gf["feed_Carbon_kgh"].sum()
    fila["fusion_carbon_rate_mean"] = gf["tasa_feed_Carbon_kg_min"].mean()
    fila["fusion_carbon_rate_cv"] = gf["tasa_feed_Carbon_kg_min"].std() / (gf["tasa_feed_Carbon_kg_min"].mean() + 1e-9)
    fila["fusion_b2_mean"] = gf["basicidad_B2"].mean()
    fila["fusion_frac_b2_optimo"] = (gf["basicidad_B2"].sub(BASICIDAD_OPTIMA).abs() < 0.3).mean()
    fila["fusion_o2_exceso_mean"] = gf["exceso_o2_combustion_pct"].mean()
    fila["fusion_o2_exceso_std"] = gf["exceso_o2_combustion_pct"].std()
    fila["fusion_duracion_total_min"] = gf["delta_tiempo"].sum()
    fila["fusion_ley_sn_pico"] = gf["ley_sn_escoria_pct"].max()
    fila["fusion_ley_sn_final"] = gf["ley_sn_escoria_pct"].iloc[-1]
    fila["fusion_temp_mean"] = gf["temperatura_horno_celsius"].mean()
    fila["fusion_temp_trend"] = slope(gf["temperatura_horno_celsius"].to_numpy())
    fila["fusion_cao_total_kg"] = gf["feed_CaO_kgh"].sum()
    fila["fusion_interaccion_C_x_Sn_prev_mean"] = gf["interaccion_C_x_Sn_prev"].mean()

    # ---- Transicion F->R (ya construida, Paso 3 de hallazgos) ----
    ultimo_f = gf.iloc[-1]
    primero_r = gr.iloc[0]
    for var in ["ley_sn_escoria_pct", "ley_feo_escoria_pct", "temperatura_horno_celsius",
                "presion_punta_lanza_kpa", "exceso_o2_combustion_pct"]:
        if var in g.columns:
            fila[f"delta_FR_{var}"] = primero_r.get(var, np.nan) - ultimo_f.get(var, np.nan)

    # ---- Reduccion: perfil de dosificacion + zona de riesgo de lanza (Pasos 46-47) ----
    tasa_c = gr["tasa_feed_Carbon_kg_min"].to_numpy()
    fila["reduccion_carbon_total_kg"] = gr["feed_Carbon_kgh"].sum()
    fila["reduccion_carbon_rate_first"] = tasa_c[0] if len(tasa_c) else np.nan
    fila["reduccion_carbon_decay_ratio"] = (tasa_c[-1] / tasa_c[0]) if len(tasa_c) and tasa_c[0] > 1e-6 else np.nan
    fila["reduccion_gn_total_nm3"] = gr["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum()
    fila["reduccion_o2_total_nm3"] = gr["volumen_o2_inyectado_lanza_escalon_nm3"].sum()
    fila["reduccion_aire_total_nm3"] = gr["volumen_aire_inyectado_lanza_escalon_nm3"].sum()
    fila["reduccion_o2_exceso_mean"] = gr["exceso_o2_combustion_pct"].mean()
    fila["reduccion_lanza_pos_mean"] = gr["posicion_vertical_lanza_mm"].mean()
    fila["reduccion_lanza_pos_std"] = gr["posicion_vertical_lanza_mm"].std()
    fila["reduccion_frac_en_valle_lanza"] = gr["posicion_vertical_lanza_mm"].between(*VALLE_LANZA).mean()
    fila["reduccion_selectividad_media"] = fe.division_segura(
        -gr["d_ley_sn_escoria_pct"], -gr["d_ley_feo_escoria_pct"]
    ).replace([np.inf, -np.inf], np.nan).mean()
    fila["reduccion_duracion_total_min"] = gr["delta_tiempo"].sum()
    fila["reduccion_ley_sn_inicio"] = gr["ley_sn_escoria_pct_prev"].iloc[0] if "ley_sn_escoria_pct_prev" in gr else np.nan
    fila["reduccion_ley_sn_final"] = gr["ley_sn_escoria_pct"].iloc[-1]
    fila["reduccion_ley_feo_final"] = gr["ley_feo_escoria_pct"].iloc[-1] if "ley_feo_escoria_pct" in gr else np.nan
    fila["reduccion_feo_caida_total"] = gr["d_ley_feo_escoria_pct"].sum() if "d_ley_feo_escoria_pct" in gr else np.nan
    fila["reduccion_sn_caida_total"] = gr["d_ley_sn_escoria_pct"].sum()
    fila["n_escalones_reduccion"] = len(gr)

    filas.append(fila)

df_batch = pd.DataFrame(filas).set_index("Batch")
df_batch = df_batch.join(rollup_agg, how="left")
df_batch["recuperacion_real_batch_pct"] = fe.construir_resumen_batch_masa(df).set_index("Batch")["recuperacion_real_batch_pct"]
print(f"Dataset de batch: {df_batch.shape}")
df_batch.to_csv(f"{OUT_DIR}/df_batch_features.csv")

FEATURES_BATCH = [c for c in df_batch.columns if c not in ("rendimiento_proxy_batch", "recuperacion_real_batch_pct")]
print(f"n features de batch: {len(FEATURES_BATCH)}")

# -----------------------------------------------------------------------------
# 3) Validacion: baseline lineal simple (lo YA probado, para comparar de forma justa)
#    vs. ElasticNet (lineal regularizado, todas las features) vs. XGBoost (no lineal)
# -----------------------------------------------------------------------------
PARAMS_XGB_BATCH = dict(n_estimators=200, learning_rate=0.03, max_depth=3,
                         min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                         reg_lambda=2.0, random_state=42, n_jobs=-1)


def evaluar_modelo_batch(df_batch, target_col, orden_batches, batches_dev, batches_lockbox):
    d = df_batch.dropna(subset=[target_col]).copy()
    orden_validos = [b for b in orden_batches if b in d.index]
    d = d.loc[orden_validos]
    dev_mask = d.index.isin(batches_dev)
    d_dev, d_lb = d.loc[dev_mask], d.loc[~dev_mask]

    resultados = {}

    # --- Baseline: mejor correlato lineal simple ya documentado ---
    baseline_col = "reduccion_sn_caida_total"
    rho, p = spearmanr(d_dev[baseline_col], d_dev[target_col])
    resultados["baseline_simple_rho_spearman"] = (rho, p)

    X_dev_full = d_dev[FEATURES_BATCH]
    y_dev = d_dev[target_col]
    X_lb_full = d_lb[FEATURES_BATCH]
    y_lb = d_lb[target_col]

    # --- KFold OOF (no hay agrupacion por batch aqui: 1 fila = 1 batch) ---
    def oof_generic(modelo_fn, X, y, n_splits=5, seed=0):
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        oof = np.full(len(X), np.nan)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
            med = X_tr.median()
            X_tr, X_te = X_tr.fillna(med), X_te.fillna(med)
            m = modelo_fn()
            m.fit(X_tr, y.iloc[tr])
            oof[te] = m.predict(X_te)
        return oof

    def walkforward_generic(modelo_fn, X, y, n_splits=5):
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        pred = np.full(len(X), np.nan)
        tss = TimeSeriesSplit(n_splits=n_splits)
        for tr, te in tss.split(X):
            X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
            med = X_tr.median()
            X_tr, X_te = X_tr.fillna(med), X_te.fillna(med)
            m = modelo_fn()
            m.fit(X_tr, y.iloc[tr])
            pred[te] = m.predict(X_te)
        return pred

    modelos = {
        "ElasticNet (lineal, todas)": lambda: ElasticNetCV(cv=5, random_state=0, max_iter=5000),
        "XGBoost (no lineal, todas)": lambda: xgb.XGBRegressor(**PARAMS_XGB_BATCH),
    }

    for nombre, modelo_fn in modelos.items():
        oof = oof_generic(modelo_fn, X_dev_full, y_dev)
        wf = walkforward_generic(modelo_fn, X_dev_full, y_dev)
        valid_wf = ~np.isnan(wf)

        med_full = X_dev_full.median()
        m_final = modelo_fn()
        m_final.fit(X_dev_full.fillna(med_full), y_dev)
        pred_lb = m_final.predict(X_lb_full.fillna(med_full))

        resultados[nombre] = {
            "r2_oof": r2_score(y_dev, oof), "mae_oof": mean_absolute_error(y_dev, oof),
            "r2_oot_walkforward": r2_score(y_dev.to_numpy()[valid_wf], wf[valid_wf]) if valid_wf.sum() > 3 else np.nan,
            "r2_lockbox": r2_score(y_lb, pred_lb), "mae_lockbox": mean_absolute_error(y_lb, pred_lb),
            "n_dev": len(d_dev), "n_lockbox": len(d_lb),
        }
        if nombre.startswith("XGBoost"):
            imp = pd.Series(m_final.feature_importances_, index=FEATURES_BATCH).sort_values(ascending=False)
            resultados["_importancia_xgb"] = imp
    return resultados


for target_col in ["rendimiento_proxy_batch", "recuperacion_real_batch_pct"]:
    print(f"\n========== TARGET: {target_col} ==========")
    res = evaluar_modelo_batch(df_batch, target_col, orden_batches, batches_dev, batches_lockbox)
    print("Baseline simple (reduccion_sn_caida_total) Spearman:", res["baseline_simple_rho_spearman"])
    for nombre in ["ElasticNet (lineal, todas)", "XGBoost (no lineal, todas)"]:
        print(f"\n{nombre}:")
        for k, v in res[nombre].items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print("\nTop-15 importancia XGBoost:")
    print(res["_importancia_xgb"].head(15).round(4))
    res["_importancia_xgb"].to_csv(f"{OUT_DIR}/importancia_batch_{target_col}.csv")

print("\nListo.")
