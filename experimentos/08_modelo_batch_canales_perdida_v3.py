"""Modelo de BATCH v3: rendimiento y sus dos canales de perdida (Sn a dross Fe, Sn a polvo).

Hallazgo previo (esta sesion): rendimiento_ha_batch = metal/(metal+dross+polvo) se descompone
en dos canales INDEPENDIENTES (Spearman entre si = 0.02): frac_sn_dross (rho=-0.76 con el
rendimiento) y frac_sn_polvo (rho=-0.54). Modelarlos por separado con features de
trayectoria v3 (incluyendo los rollups EN MASA por trazador CaO: Sn en escoria al fin de
Fusion, Sn/FeO extraidos en Reduccion, selectividad en masa, mezcla de carga) y comparar con
el techo previo (R2 lockbox ~0.05 para rendimiento_proxy_batch, hallazgos.md 13.7).

Validacion: KFold OOF (batches) + walk-forward cronologico + lockbox (ultimos 17.5%).
Salidas: experimentos/08_resultados_batch.csv, 08_features_batch_v3.csv, 08_importancia_batch_<target>.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import dataset_lingo_smelter as dls  # noqa: E402
import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

df = dls.construir_dataset(RAIZ / "Datos Lingo smelter fase II.xlsx", clean_mode=True)
df = fe.construir_features_v3(df)
df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
orden = mp.orden_cronologico_batches(df)

# ------------------------------------------------------------------ tabla de batch v3
tray = fe.construir_features_trayectoria_batch(df)  # features ya existentes (v2)
filas = []
for b, g in df.groupby("Batch"):
    g = g.sort_values("fecha_inicio"); gf = g[g.fase_proceso == "Fusión"]; gr = g[g.fase_proceso == "Reducción"]
    if gf.empty or gr.empty:
        continue
    feed_sn = g["feed_Sn_kgf"].sum() / 1000
    f = {"Batch": b,
         "rendimiento_ha_batch": g["rendimiento_ha_batch"].iloc[0],
         "frac_sn_metal": g["sn_en_metal_crudo_batch_t"].iloc[0] / feed_sn,
         "frac_sn_dross": g["sn_en_dross_fe_batch_t"].iloc[0] / feed_sn,
         "frac_sn_polvo": g["sn_en_polvo_fundicion_batch_t"].iloc[0] / feed_sn,
         "feed_sn_total_t": feed_sn, "feed_total_t": g["feed_total_kgh"].sum() / 1000,
         # mezcla de carga del batch (Fusion)
         "F_frac_conc": gf["feed_conc_t1_kgh"].sum() / gf["feed_total_kgh"].sum(),
         "F_frac_reciclo": gf["feed_reciclo_t3_kgh"].sum() / gf["feed_total_kgh"].sum(),
         "F_frac_Fe": gf["feed_Fe_kgh"].sum() / gf["feed_total_kgh"].sum(),
         "F_frac_dross": gf["feed_dross_Fe_kgh"].sum() / gf["feed_total_kgh"].sum(),
         "F_frac_pellets": gf["feed_pellets_t7_kgh"].sum() / gf["feed_total_kgh"].sum(),
         "F_frac_secundaria": gf["feed_carga_secundaria_kgh"].sum() / gf["feed_total_kgh"].sum(),
         "ley_sn_conc_batch_pct": g["ley_sn_conc_batch_pct"].iloc[0], "ley_sn_carga_batch_pct": g["ley_sn_carga_batch_pct"].iloc[0],
         "ley_sn_reciclo_batch_pct": g["ley_sn_reciclo_batch_pct"].iloc[0], "ley_sn_dross_batch_pct": g["ley_sn_dross_batch_pct"].iloc[0],
         "F_cal_por_t_carga": gf["feed_CaO_kgh"].sum() / gf["feed_total_kgh"].sum(),
         "F_C_por_t_Sn": gf["feed_Carbon_kgh"].sum() / max(gf["feed_Sn_kgf"].sum(), 1),
         "R_C_por_t_Sn_escoria_inicio": gr["feed_Carbon_kgh"].sum() / max(gf["sn_inventario_escoria_est_kg"].iloc[-1], 1) if pd.notna(gf["sn_inventario_escoria_est_kg"].iloc[-1]) else np.nan,
         "F_feed_rate_mean": gf["tasa_feed_total_kg_min"].mean(), "F_feed_rate_cv": gf["tasa_feed_total_kg_min"].std() / max(gf["tasa_feed_total_kg_min"].mean(), 1e-6),
         "F_gn_por_t_carga": gf["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum() / (gf["feed_total_kgh"].sum() / 1000),
         "F_o2_enriq_mean": gf["oxygen_enrichment_pct"].mean(), "R_o2_enriq_mean": gr["oxygen_enrichment_pct"].mean(),
         "F_gn_rate_mean": gf["tasa_gn_nm3_min"].mean(), "R_gn_rate_mean": gr["tasa_gn_nm3_min"].mean(),
         "F_tiro_mean": gf["tiro_horno_pct"].mean(), "R_tiro_mean": gr["tiro_horno_pct"].mean(),
         "F_tgas_mean": gf["temperatura_gas_pre_bhf_celsius"].mean(), "R_tgas_mean": gr["temperatura_gas_pre_bhf_celsius"].mean(),
         "F_temp_max": gf["temperatura_horno_celsius"].max(), "R_temp_mean": gr["temperatura_horno_celsius"].mean(), "R_temp_max": gr["temperatura_horno_celsius"].max(),
         "F_termocupla_mean": gf["termocupla_media_celsius"].mean(), "espesor_ladrillo_norm_mm": g["espesor_ladrillo_norm_mm"].iloc[0],
         "F_desvio_duracion_total": gf["desvio_duracion_min"].sum(), "R_desvio_duracion_total": gr["desvio_duracion_min"].sum(),
         "F_lanza_mean": gf["posicion_vertical_lanza_mm"].mean(), "F_lanza_std": gf["posicion_vertical_lanza_mm"].std(),
         "F_tipp_mean": gf["presion_punta_lanza_kpa"].mean(), "R_tipp_mean": gr["presion_punta_lanza_kpa"].mean(),
         # estado de escoria
         "F_b2_final": gf["basicidad_B2"].iloc[-1], "F_irf_final": gf["indice_irf"].iloc[-1], "R_b2_final": gr["basicidad_B2"].iloc[-1],
         "F_feo_final": gf["ley_feo_escoria_pct"].iloc[-1], "R_feo_final": gr["ley_feo_escoria_pct"].iloc[-1],
         "R_sio2_final": gr["ley_sio2_escoria_pct"].iloc[-1], "R_cao_final": gr["ley_cao_escoria_pct"].iloc[-1], "R_al2o3_final": gr["ley_al2o3_escoria_pct"].iloc[-1], "R_mgo_final": gr["ley_mgo_escoria_pct"].iloc[-1],
         # rollups EN MASA (trazador CaO)
         "F_masa_escoria_final_t": gf["masa_escoria_est_kg"].iloc[-1] / 1000, "R_masa_escoria_final_t": gr["masa_escoria_est_kg"].iloc[-1] / 1000,
         "F_sn_escoria_final_t": gf["sn_inventario_escoria_est_kg"].iloc[-1] / 1000, "R_sn_escoria_final_t": gr["sn_inventario_escoria_est_kg"].iloc[-1] / 1000,
         "F_sn_extraido_t": gf["sn_extraido_est_kg"].sum() / 1000, "R_sn_extraido_t": gr["sn_extraido_est_kg"].sum() / 1000,
         "R_feo_extraido_t": gr["feo_extraido_est_kg"].sum() / 1000, "F_feo_escoria_final_t": gf["feo_inventario_escoria_est_kg"].iloc[-1] / 1000,
         "R_feo_extraido_frac": gr["feo_extraido_est_kg"].sum() / max(gf["feo_inventario_escoria_est_kg"].iloc[-1], 1) if pd.notna(gf["feo_inventario_escoria_est_kg"].iloc[-1]) else np.nan,
         "R_selectividad_masa": (gr["sn_extraido_est_kg"].sum() / gr["feo_extraido_est_kg"].sum()) if gr["feo_extraido_est_kg"].sum() > 0 else np.nan,
         "F_avance_reduccion_final": 1 - gf["sn_inventario_escoria_est_kg"].iloc[-1] / max(gf["feed_Sn_kgf"].sum(), 1),
         "R_sn_extraido_frac": gr["sn_extraido_est_kg"].sum() / max(gf["sn_inventario_escoria_est_kg"].iloc[-1], 1) if pd.notna(gf["sn_inventario_escoria_est_kg"].iloc[-1]) else np.nan,
         "R_frac_sn_extraido_R0": gr["frac_sn_extraido_escalon"].iloc[0],
         }
    filas.append(f)
tabla = pd.DataFrame(filas).set_index("Batch")
tabla = tabla.join(tray.drop(columns=[c for c in tray.columns if c in tabla.columns]), how="left")
tabla.to_csv(RAIZ / "experimentos" / "08_features_batch_v3.csv")
TARGETS = ["rendimiento_ha_batch", "frac_sn_dross", "frac_sn_polvo", "frac_sn_metal"]
EXCLUIR = set(TARGETS) | {"rendimiento_proxy_batch", "recuperacion_real_batch_pct", "feed_sn_total_t", "n_escalones_reduccion"}
FEATS = [c for c in tabla.columns if c not in EXCLUIR and tabla[c].notna().mean() > 0.8]
print(f"tabla batch {tabla.shape}; features candidatas {len(FEATS)}")

# ------------------------------------------------------------------ correlaciones univariadas
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
cors = pd.DataFrame({t: {f: spearmanr(tabla[f], tabla[t], nan_policy="omit")[0] for f in FEATS} for t in TARGETS})
cors.to_csv(RAIZ / "experimentos" / "08_spearman_batch.csv")
print("\n=== Spearman |rho| > 0.2 con algun target ===")
print(cors.loc[cors.abs().max(axis=1) > 0.2].round(3).sort_values("rendimiento_ha_batch").to_string())

# ------------------------------------------------------------------ modelos
PARAMS = dict(n_estimators=150, learning_rate=0.03, max_depth=2, min_child_weight=8, subsample=0.7, colsample_bytree=0.6, reg_lambda=5.0, random_state=42, n_jobs=-1)
tabla = tabla.loc[[b for b in orden if b in tabla.index]]
dev_mask = tabla.index.isin(batches_dev)
res = []
for target in TARGETS:
    d = tabla.dropna(subset=[target])
    dm = d.index.isin(batches_dev)
    X_dev, y_dev, X_lb, y_lb = d.loc[dm, FEATS], d.loc[dm, target], d.loc[~dm, FEATS], d.loc[~dm, target]
    med = X_dev.median()
    for nombre, mk in [("XGB_chico", lambda: xgb.XGBRegressor(**PARAMS)),
                       ("ElasticNet", lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 1.0], cv=5, random_state=0)))]:
        oof = np.full(len(X_dev), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X_dev):
            m = mk(); m.fit(X_dev.iloc[tr].fillna(med), y_dev.iloc[tr]); oof[te] = m.predict(X_dev.iloc[te].fillna(med))
        wf = np.full(len(X_dev), np.nan)
        for tr, te in TimeSeriesSplit(5).split(X_dev):
            m = mk(); m.fit(X_dev.iloc[tr].fillna(med), y_dev.iloc[tr]); wf[te] = m.predict(X_dev.iloc[te].fillna(med))
        v = ~np.isnan(wf)
        m = mk(); m.fit(X_dev.fillna(med), y_dev); p_lb = m.predict(X_lb.fillna(med))
        fila = dict(target=target, modelo=nombre, n_dev=len(X_dev), n_lockbox=len(X_lb),
                    r2_oof=r2_score(y_dev, oof), r2_walkforward=r2_score(y_dev[v], wf[v]), r2_lockbox=r2_score(y_lb, p_lb),
                    mae_lockbox=mean_absolute_error(y_lb, p_lb), rho_lockbox=spearmanr(y_lb, p_lb)[0])
        res.append(fila); print(fila, flush=True)
        if nombre == "XGB_chico":
            pi = permutation_importance(m, X_lb.fillna(med), y_lb, n_repeats=10, random_state=0, scoring="r2")
            imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
            imp.to_csv(RAIZ / "experimentos" / f"08_importancia_batch_{target}.csv")
            print(f"  top-12 permutation (lockbox) {target}:\n{imp.head(12).round(4).to_string()}")
            # version parsimoniosa: top-8 por importancia OOF-dev (evita elegir con lockbox)
            m_dev = xgb.XGBRegressor(**PARAMS).fit(X_dev.fillna(med), y_dev)
            pi_dev = permutation_importance(m_dev, X_dev.fillna(med), y_dev, n_repeats=5, random_state=0, scoring="r2")
            top8 = list(pd.Series(pi_dev.importances_mean, index=FEATS).sort_values(ascending=False).head(8).index)
            oof8 = np.full(len(X_dev), np.nan)
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X_dev):
                mm = xgb.XGBRegressor(**PARAMS).fit(X_dev.iloc[tr][top8].fillna(med[top8]), y_dev.iloc[tr]); oof8[te] = mm.predict(X_dev.iloc[te][top8].fillna(med[top8]))
            mm = xgb.XGBRegressor(**PARAMS).fit(X_dev[top8].fillna(med[top8]), y_dev); p8 = mm.predict(X_lb[top8].fillna(med[top8]))
            fila8 = dict(target=target, modelo="XGB_top8", n_dev=len(X_dev), n_lockbox=len(X_lb), r2_oof=r2_score(y_dev, oof8), r2_walkforward=np.nan,
                         r2_lockbox=r2_score(y_lb, p8), mae_lockbox=mean_absolute_error(y_lb, p8), rho_lockbox=spearmanr(y_lb, p8)[0], features=";".join(top8))
            res.append(fila8); print(fila8, flush=True)
pd.DataFrame(res).to_csv(RAIZ / "experimentos" / "08_resultados_batch.csv", index=False)
print(pd.DataFrame(res).round(3).to_string())
