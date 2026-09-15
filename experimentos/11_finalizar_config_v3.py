"""Finalizacion de la configuracion v3: sets CURADOS (interpretables, sin cocientes dificiles,
sin duplicados) por (fase, target), evaluados con OOF GroupKFold (criterio de seleccion) y
lockbox (solo reporte). Incluye:
  (a) R2 "trivial" de referencia: sn_extraido_est_kg con SOLO el Sn cargado (Fusion) o SOLO el
      inventario previo (Reduccion) -> cuanto agregan realmente las palancas quimicas.
  (b) Comparacion en escala kg del FeO en Reduccion: modelo en puntos convertido (x masa_prev)
      vs modelo directo en kg.
  (c) Diagnostico de la fraccion extraida (frac_sn_extraido_escalon) dev vs lockbox.
  (d) Chequeo de signos SHAP (corr valor vs SHAP) de los sets elegidos.
Salidas: experimentos/11_sets_curados_resultados.csv, 11_shap_signos.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v2 as mp2  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402

SALIDA = RAIZ / "experimentos"
df = mp3.construir_dataset_modelo_v3(RAIZ / "Datos Lingo smelter fase II.xlsx")
df_base = mp.dataset_base_modelo(df)
dev, lb = mp.split_dev_lockbox(df)

def evaluar(fase, target, feats, monot=True):
    sub = df_base.loc[df_base["fase_proceso"] == fase].dropna(subset=[target])
    d_dev = sub[sub["Batch"].isin(dev)].reset_index(drop=True); d_lb = sub[sub["Batch"].isin(lb)].reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(5).split(d_dev[feats], d_dev[target], d_dev["Batch"]):
        m = mp3.construir_modelo_v3(feats, target, monot).fit(d_dev[feats].iloc[tr], d_dev[target].iloc[tr]); oof[te] = m.predict(d_dev[feats].iloc[te])
    m = mp3.construir_modelo_v3(feats, target, monot).fit(d_dev[feats], d_dev[target]); p = m.predict(d_lb[feats])
    return dict(r2_oof=r2_score(d_dev[target], oof), mae_oof=mean_absolute_error(d_dev[target], oof), r2_lockbox=r2_score(d_lb[target], p),
                mae_lockbox=mean_absolute_error(d_lb[target], p), n_dev=len(d_dev), n_lb=len(d_lb)), m, d_dev, d_lb, p

CANDIDATOS = {
    ("Fusión", "sn_extraido_est_kg"): {
        "trivial_solo_Sn_cargado": ["feed_Sn_kgf"],
        "trivial_Sn_cargado+inventario": ["feed_Sn_kgf", "sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev"],
        "curado_A_10": ["feed_Sn_kgf", "sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev",
                        "tasa_feed_Carbon_kg_min", "relacion_CaO_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                        "basicidad_B2_prev", "temperatura_horno_celsius_prev"],
        "curado_B_13": ["feed_Sn_kgf", "sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "feo_inventario_escoria_est_kg_prev",
                        "ley_sn_escoria_pct_prev", "interaccion_C_x_Sn_prev", "tasa_feed_Carbon_kg_min", "relacion_CaO_carga",
                        "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "basicidad_B2_prev", "temperatura_horno_celsius_prev",
                        "frac_carga_secundaria"],
        "curado_C_15": ["feed_Sn_kgf", "duracion_plan_min", "sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev",
                        "feo_inventario_escoria_est_kg_prev", "ley_sn_escoria_pct_prev", "interaccion_C_x_Sn_prev", "relacion_CaO_carga",
                        "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "oxygen_enrichment_pct", "basicidad_B2_prev",
                        "temperatura_horno_celsius_prev", "tasa_feed_pellets_kg_min", "posicion_vertical_lanza_mm_prev"],
        "auto09_k8": ["feed_Sn_kgf", "margen_presion_gn", "basicidad_B4_prev", "interaccion_C_x_Sn_prev", "cum_feed_reciclo_kg_prev",
                      "posicion_vertical_lanza_mm_prev", "carbon_sobre_inventario_sn_prev", "sn_inventario_escoria_est_kg_prev"],
    },
    ("Fusión", "d_ley_sn_escoria_pct"): {
        "v2_actual_7": mp2.FEATURES_STATE_ACTION_V2["Fusión"],
        "v2+masa_9": mp2.FEATURES_STATE_ACTION_V2["Fusión"] + ["masa_escoria_est_kg_prev", "feed_Sn_kgf"],
        "v2_plan_7": [f if f != "delta_tiempo" else "duracion_plan_min" for f in mp2.FEATURES_STATE_ACTION_V2["Fusión"]],
    },
    ("Fusión", "d_ley_feo_escoria_pct"): {
        "v2_actual_7": mp2.FEATURES_STATE_ACTION_V2["Fusión"],
        "curado_feo_9": ["ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_sn_escoria_pct_prev", "tasa_feed_CaO_kg_min",
                         "tasa_feed_Carbon_kg_min", "exceso_o2_combustion_pct", "tasa_feed_Fe_kg_min", "tasa_feed_dross_Fe_kg_min", "duracion_plan_min"],
    },
    ("Reducción", "sn_extraido_est_kg"): {
        "trivial_solo_inventario": ["sn_inventario_escoria_est_kg_prev"],
        "trivial_inventario+avance": ["sn_inventario_escoria_est_kg_prev", "avance_reduccion_sn_prev", "masa_escoria_est_kg_prev"],
        "curado_A_10": ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev",
                        "tasa_feed_Carbon_kg_min", "interaccion_C_x_Sn_prev", "duracion_plan_min", "tasa_gn_nm3_min",
                        "exceso_o2_combustion_pct", "temperatura_horno_celsius_prev"],
        "curado_B_13": ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "avance_reduccion_sn_prev", "ley_sn_escoria_pct_prev",
                        "ley_feo_escoria_pct_prev", "interaccion_Sn_x_FeO_prev", "tasa_feed_Carbon_kg_min", "interaccion_C_x_Sn_prev",
                        "duracion_plan_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "interaccion_C_x_exceso_o2",
                        "temperatura_horno_celsius_prev"],
    },
    ("Reducción", "d_ley_sn_escoria_pct"): {
        "v2_configC_9": mp2.FEATURES_STATE_ACTION_V2["Reducción"],
        "v2C+masa_11": mp2.FEATURES_STATE_ACTION_V2["Reducción"] + ["sn_inventario_escoria_est_kg_prev", "duracion_plan_min"],
        "v2C+ratio_10": mp2.FEATURES_STATE_ACTION_V2["Reducción"] + ["ratio_sn_feo_prev"],
    },
    ("Reducción", "d_ley_feo_escoria_pct"): {
        "v2_configC_9": mp2.FEATURES_STATE_ACTION_V2["Reducción"],
        "curado_feo_11": ["ley_feo_escoria_pct_prev", "ley_sn_escoria_pct_prev", "interaccion_Sn_x_FeO_prev", "ratio_sn_sio2_prev",
                          "tasa_feed_Carbon_kg_min", "interaccion_C_x_FeO_prev", "cum_feed_Carbon_kg_prev", "tasa_gn_nm3_min",
                          "exceso_o2_combustion_pct", "duracion_plan_min", "grad_temperatura_horno_celsius_prev"],
    },
    ("Reducción", "feo_extraido_est_kg"): {
        "curado_feo_kg_11": ["feo_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_feo_escoria_pct_prev", "ley_cao_escoria_pct_prev",
                             "ley_sn_escoria_pct_prev", "sn_inventario_escoria_est_kg_prev", "tasa_feed_Carbon_kg_min", "interaccion_C_x_FeO_prev",
                             "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "duracion_plan_min"],
    },
    ("Fusión", "d_temperatura_horno_celsius"): {"v3_dT": mp3.FEATURES_V3_POR_DEFECTO["Fusión"]["dT"]},
    ("Reducción", "d_temperatura_horno_celsius"): {"v3_dT": mp3.FEATURES_V3_POR_DEFECTO["Reducción"]["dT"]},
}

filas, modelos = [], {}
for (fase, target), sets in CANDIDATOS.items():
    for nombre, feats in sets.items():
        ev, m, d_dev, d_lb, p = evaluar(fase, target, feats)
        ev.update(fase=fase, target=target, set=nombre, n_features=len(feats), features=";".join(feats))
        filas.append(ev); modelos[(fase, target, nombre)] = (m, d_dev, d_lb, p, feats)
        print(f"{fase:9s} | {target:26s} | {nombre:30s} | k={len(feats):2d} | R2 oof={ev['r2_oof']:.3f} lb={ev['r2_lockbox']:.3f} | MAE lb={ev['mae_lockbox']:.3f}", flush=True)
res = pd.DataFrame(filas); res.to_csv(SALIDA / "11_sets_curados_resultados.csv", index=False)

# (b) FeO en kg en Reduccion: pts convertido vs kg directo
print("\n=== (b) FeO Reduccion en escala kg (lockbox) ===")
m_pts, _, d_lb_pts, p_pts, _ = modelos[("Reducción", "d_ley_feo_escoria_pct", "curado_feo_11")]
m_kg, _, d_lb_kg, p_kg, _ = modelos[("Reducción", "feo_extraido_est_kg", "curado_feo_kg_11")]
comun = d_lb_pts.merge(d_lb_kg[["Batch", "orden_escalon_fase", "feo_extraido_est_kg"]], on=["Batch", "orden_escalon_fase"], suffixes=("", "_kg"))
conv = -p_pts * d_lb_pts["masa_escoria_est_kg_prev"].to_numpy() / 100
ok = d_lb_pts["feo_extraido_est_kg"].notna() & d_lb_pts["masa_escoria_est_kg_prev"].notna()
print(f"pts->kg (x masa_prev): R2={r2_score(d_lb_pts.loc[ok,'feo_extraido_est_kg'], conv[ok]):.3f} MAE={mean_absolute_error(d_lb_pts.loc[ok,'feo_extraido_est_kg'], conv[ok]):.0f} kg")
print(f"kg directo:            R2={r2_score(d_lb_kg['feo_extraido_est_kg'], p_kg):.3f} MAE={mean_absolute_error(d_lb_kg['feo_extraido_est_kg'], p_kg):.0f} kg  (std target lb={d_lb_kg['feo_extraido_est_kg'].std():.0f})")

# (c) fraccion extraida: dev vs lockbox
print("\n=== (c) frac_sn_extraido_escalon: distribucion dev vs lockbox ===")
for fase in mp.FASES:
    s = df_base.loc[df_base["fase_proceso"] == fase, ["Batch", "frac_sn_extraido_escalon", "sn_extraido_est_kg", "feed_Sn_kgf", "ley_sn_conc_batch_pct"]]
    for nombre, bts in [("dev", dev), ("lockbox", lb)]:
        x = s[s["Batch"].isin(bts)]
        print(f"{fase:9s} {nombre:8s} frac: mean={x['frac_sn_extraido_escalon'].mean():.3f} std={x['frac_sn_extraido_escalon'].std():.3f} | kg: mean={x['sn_extraido_est_kg'].mean():.0f} std={x['sn_extraido_est_kg'].std():.0f} | feedSn mean={x['feed_Sn_kgf'].mean():.0f} | ley conc={x['ley_sn_conc_batch_pct'].mean():.2f}")

# (d) signos SHAP de los sets curados principales
print("\n=== (d) signos SHAP (corr Pearson valor vs SHAP; media |SHAP|) ===")
filas_shap = []
for clave in [("Fusión", "sn_extraido_est_kg", "curado_B_13"), ("Reducción", "sn_extraido_est_kg", "curado_B_13"),
              ("Reducción", "d_ley_feo_escoria_pct", "curado_feo_11"), ("Fusión", "d_ley_feo_escoria_pct", "curado_feo_9"),
              ("Reducción", "d_temperatura_horno_celsius", "v3_dT"), ("Fusión", "d_temperatura_horno_celsius", "v3_dT")]:
    m, d_dev, d_lb, p, feats = modelos[clave]
    X = d_dev[feats]
    expl = shap.TreeExplainer(m); sv = expl.shap_values(X)
    for j, f in enumerate(feats):
        x = X[f].to_numpy(); v = sv[:, j]; okk = ~np.isnan(x)
        c = np.corrcoef(x[okk], v[okk])[0, 1] if okk.sum() > 10 and np.std(x[okk]) > 0 else np.nan
        filas_shap.append(dict(fase=clave[0], target=clave[1], set=clave[2], feature=f, corr_valor_shap=c, media_abs_shap=float(np.abs(v).mean())))
    print(f"-- {clave}")
    print(pd.DataFrame([r for r in filas_shap if (r['fase'], r['target'], r['set']) == clave]).sort_values("media_abs_shap", ascending=False)[["feature", "corr_valor_shap", "media_abs_shap"]].round(3).to_string(index=False))
pd.DataFrame(filas_shap).to_csv(SALIDA / "11_shap_signos.csv", index=False)
print("\nLISTO")
