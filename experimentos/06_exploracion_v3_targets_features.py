"""Exploracion v3 (2026-09-13 noche): targets x feature sets x algoritmos, por fase.

Pregunta: con las features nuevas de carga por tolva, plan de duracion, refractario e
inventario/masa por trazador CaO (feature_engineering.construir_features_v3), que
combinacion (target, feature set, algoritmo) predice mejor el efecto de las acciones
dado el estado, en cada fase? Misma disciplina que rondas anteriores: OOF GroupKFold por
Batch sobre DEV + lockbox (ultimos 17.5% batches, nunca usado para elegir nada).

Salidas: experimentos/06_resultados_targets_features.csv, 06_importancia_<fase>_<target>.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import dataset_lingo_smelter as dls  # noqa: E402
import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v2 as mp2  # noqa: E402

SALIDA = RAIZ / "experimentos"

# ----------------------------------------------------------------------------- datos
df = dls.construir_dataset(RAIZ / "Datos Lingo smelter fase II.xlsx", clean_mode=True)
df = fe.construir_features_v3(df)
df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
print(f"df {df.shape}; base {df_base.shape}; dev={len(batches_dev)} lockbox={len(batches_lockbox)}")

# Winsorizacion suave de targets en masa (colas por %CaO/%Sn muy bajos)
def _clip_target(s: pd.Series, lo=0.005, hi=0.995) -> pd.Series:
    q1, q2 = s.quantile(lo), s.quantile(hi)
    return s.where(s.between(q1, q2))

for col in ["sn_extraido_est_kg", "frac_sn_extraido_escalon", "feo_extraido_est_kg", "d_sn_inventario_escoria_est_kg"]:
    df_base[col] = df_base.groupby("fase_proceso")[col].transform(_clip_target)

# ----------------------------------------------------------------------------- feature sets
COMUNES_V3_CARGA = [
    "tasa_feed_conc_kg_min", "tasa_feed_reciclo_kg_min", "tasa_feed_Fe_kg_min",
    "tasa_feed_dross_Fe_kg_min", "tasa_feed_pellets_kg_min", "tasa_feed_CaO_kg_min",
    "tasa_feed_total_kg_min", "ley_sn_carga_pct", "frac_carga_secundaria",
    "relacion_CaO_carga", "relacion_C_Sn_carga", "duracion_plan_min",
    "ley_sn_conc_batch_pct", "ley_sn_reciclo_batch_pct", "ley_sn_dross_batch_pct",
    "ley_sn_pellets_batch_pct", "ley_sn_Fe_batch_pct", "espesor_ladrillo_norm_mm",
    "termocupla_media_celsius_prev", "temperatura_horno_celsius_prev",
    "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min",
    "exceso_o2_combustion_pct", "oxygen_enrichment_pct", "posicion_vertical_lanza_mm_prev",
    "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev",
    "indice_irf_prev", "basicidad_B2_prev", "orden_escalon_fase", "tiempo_fase",
    "tasa_feed_Carbon_kg_min", "ley_sn_escoria_pct_prev", "interaccion_C_x_Sn_prev",
]
INVENTARIO = [
    "cum_feed_total_kg_prev", "cum_feed_Sn_kg_prev", "cum_feed_carga_secundaria_kg_prev",
    "cum_feed_Carbon_kg_prev", "cum_feed_CaO_kg_prev", "masa_escoria_est_kg_prev",
    "sn_inventario_escoria_est_kg_prev", "feo_inventario_escoria_est_kg_prev",
    "avance_reduccion_sn_prev",
]

def _dedup(lista):
    return list(dict.fromkeys(lista))

FEATURE_SETS = {
    "Fusión": {
        "v2_actual": mp2.FEATURES_STATE_ACTION_V2["Fusión"],
        "v3_carga": _dedup(mp2.FEATURES_STATE_ACTION_V2["Fusión"] + COMUNES_V3_CARGA),
        "v3_carga+inventario": _dedup(mp2.FEATURES_STATE_ACTION_V2["Fusión"] + COMUNES_V3_CARGA + INVENTARIO),
    },
    "Reducción": {
        "v2_actual": mp2.FEATURES_STATE_ACTION_V2["Reducción"],
        "v3_carga": _dedup(mp2.FEATURES_STATE_ACTION_V2["Reducción"] + [
            c for c in COMUNES_V3_CARGA if c not in {"frac_carga_secundaria", "ley_sn_carga_pct", "relacion_CaO_carga", "relacion_C_Sn_carga"}
        ] + ["tasa_feed_Carbon_reduccion_kg_min", "grad_temperatura_horno_celsius_prev"]),
        "v3_carga+inventario": None,  # se completa abajo
    },
}
FEATURE_SETS["Reducción"]["v3_carga+inventario"] = _dedup(FEATURE_SETS["Reducción"]["v3_carga"] + INVENTARIO)
# pool completo de features seguras (para el Boruta posterior, script 07b)
POOL_SEGURO = [c for c in fe.filtrar_features_seguras(list(df_base.columns))
               if df_base[c].dtype != object and c not in {"Batch", "fase_proceso", "fecha_inicio", "fecha_final", "es_fusion"}
               and not c.startswith("es_") and c != "feed_CaO_total_excel_kg"]
POOL_SEGURO = [c for c in POOL_SEGURO if df_base[c].notna().mean() > 0.6]
for fase in FEATURE_SETS:
    FEATURE_SETS[fase]["v3_pool_seguro"] = _dedup([c for c in POOL_SEGURO if not (fase == "Reducción" and c in {"frac_carga_secundaria", "ley_sn_carga_pct", "relacion_CaO_carga", "relacion_C_Sn_carga", "exceso_C_estequiometrico_pct", "oxidante_sobre_carga_total", "carga_termica_relativa", "relacion_C_carga"})])

for fase, sets in FEATURE_SETS.items():
    for nombre, feats in sets.items():
        inseguras = [f for f in feats if f not in fe.filtrar_features_seguras(feats)]
        assert not inseguras, (fase, nombre, inseguras)
        print(f"{fase:10s} {nombre:22s} n_features={len(feats)}")

# ----------------------------------------------------------------------------- targets
# (nombre, columna, como convertir la prediccion a escala fisica Delta%Sn (o None), como
#  convertir a kg de Sn extraido (o None))
TARGETS = {
    "Fusión": [
        ("T1 d_ley_sn (pts %Sn)", "d_ley_sn_escoria_pct"),
        ("T3 tasa_relativa_sn", "tasa_relativa_sn_escoria"),
        ("M1 sn_extraido_est_kg", "sn_extraido_est_kg"),
        ("M2 frac_sn_extraido_escalon", "frac_sn_extraido_escalon"),
        ("F1 d_ley_feo (pts %FeO)", "d_ley_feo_escoria_pct"),
    ],
    "Reducción": [
        ("T1 d_ley_sn (pts %Sn)", "d_ley_sn_escoria_pct"),
        ("T3 tasa_relativa_sn", "tasa_relativa_sn_escoria"),
        ("M1 sn_extraido_est_kg", "sn_extraido_est_kg"),
        ("M2 frac_sn_extraido_escalon", "frac_sn_extraido_escalon"),
        ("F1 d_ley_feo (pts %FeO)", "d_ley_feo_escoria_pct"),
        ("F2 feo_extraido_est_kg", "feo_extraido_est_kg"),
    ],
}

MONOTONAS_NEG = {"tasa_feed_Carbon_kg_min", "cum_feed_Carbon_kg_prev", "interaccion_C_x_Sn_prev", "tasa_feed_Carbon_reduccion_kg_min"}

def builder_hgb(features, target_col, monotona=True):
    signo = -1
    if target_col in {"sn_extraido_est_kg", "frac_sn_extraido_escalon", "feo_extraido_est_kg"}:
        signo = +1  # mas carbon -> mas Sn extraido (kg) => monotona POSITIVA
    cst = [signo if (monotona and f in MONOTONAS_NEG) else 0 for f in features]
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15,
                                         l2_regularization=1.0, random_state=42, monotonic_cst=cst)

def builder_xgb(features, target_col, monotona=False):
    return xgb.XGBRegressor(**mp.PARAMS_XGB)

ALGOS = {"HistGB+mono(C)": lambda f, t: builder_hgb(f, t, True), "XGBoost": builder_xgb}

def a_fisico_delta_sn(target_col, pred, d):
    """Convierte la prediccion a puntos de %Sn cuando es derivable, si no devuelve None."""
    if target_col == "d_ley_sn_escoria_pct":
        return pred
    if target_col == "tasa_relativa_sn_escoria":
        return pred * d["ley_sn_escoria_pct_prev"].to_numpy()
    return None

def a_kg_sn_extraido(target_col, pred, d):
    if target_col == "sn_extraido_est_kg":
        return pred
    if target_col == "frac_sn_extraido_escalon":
        disponible = d["sn_inventario_escoria_est_kg_prev"].to_numpy() + d["feed_Sn_kgf"].to_numpy()
        return pred * disponible
    return None

filas = []
importancias = {}
for fase in mp.FASES:
    sub_fase = df_base.loc[df_base["fase_proceso"] == fase].copy()
    todas_targets = [t[1] for t in TARGETS[fase]]
    # filas comunes: todos los targets disponibles (comparabilidad entre targets)
    comun = sub_fase.dropna(subset=todas_targets + ["ley_sn_escoria_pct_prev", "sn_inventario_escoria_est_kg_prev"])
    d_dev = comun.loc[comun["Batch"].isin(batches_dev)].reset_index(drop=True)
    d_lb = comun.loc[comun["Batch"].isin(batches_lockbox)].reset_index(drop=True)
    print(f"\n=== {fase}: filas comunes dev={len(d_dev)} lockbox={len(d_lb)} ===")
    for nombre_t, tcol in TARGETS[fase]:
        for nombre_fs, feats in FEATURE_SETS[fase].items():
            for nombre_algo, builder in ALGOS.items():
                oof = np.full(len(d_dev), np.nan)
                for tr, te in GroupKFold(n_splits=5).split(d_dev[feats], d_dev[tcol], d_dev["Batch"]):
                    m = builder(feats, tcol)
                    m.fit(d_dev[feats].iloc[tr], d_dev[tcol].iloc[tr])
                    oof[te] = m.predict(d_dev[feats].iloc[te])
                m_final = builder(feats, tcol)
                m_final.fit(d_dev[feats], d_dev[tcol])
                pred_lb = m_final.predict(d_lb[feats])

                fila = {"fase": fase, "target": nombre_t, "feature_set": nombre_fs, "algoritmo": nombre_algo,
                        "n_features": len(feats), "n_dev": len(d_dev), "n_lockbox": len(d_lb),
                        "r2_oof": r2_score(d_dev[tcol], oof), "mae_oof": mean_absolute_error(d_dev[tcol], oof),
                        "r2_lockbox": r2_score(d_lb[tcol], pred_lb), "mae_lockbox": mean_absolute_error(d_lb[tcol], pred_lb)}
                # escalas fisicas comunes
                f_oof, f_lb = a_fisico_delta_sn(tcol, oof, d_dev), a_fisico_delta_sn(tcol, pred_lb, d_lb)
                if f_oof is not None:
                    fila["r2_oof_dSn_pts"] = r2_score(d_dev["d_ley_sn_escoria_pct"], f_oof)
                    fila["r2_lockbox_dSn_pts"] = r2_score(d_lb["d_ley_sn_escoria_pct"], f_lb)
                    fila["mae_lockbox_dSn_pts"] = mean_absolute_error(d_lb["d_ley_sn_escoria_pct"], f_lb)
                k_oof, k_lb = a_kg_sn_extraido(tcol, oof, d_dev), a_kg_sn_extraido(tcol, pred_lb, d_lb)
                if k_oof is not None:
                    fila["r2_oof_kgSn"] = r2_score(d_dev["sn_extraido_est_kg"], k_oof)
                    fila["r2_lockbox_kgSn"] = r2_score(d_lb["sn_extraido_est_kg"], k_lb)
                    fila["mae_lockbox_kgSn"] = mean_absolute_error(d_lb["sn_extraido_est_kg"], k_lb)
                filas.append(fila)
                print(f"{fase:9s} | {nombre_t:28s} | {nombre_fs:20s} | {nombre_algo:14s} | R2 oof={fila['r2_oof']:.3f} lb={fila['r2_lockbox']:.3f}"
                      + (f" | dSn lb={fila['r2_lockbox_dSn_pts']:.3f}" if "r2_lockbox_dSn_pts" in fila else "")
                      + (f" | kgSn lb={fila['r2_lockbox_kgSn']:.3f}" if "r2_lockbox_kgSn" in fila else ""), flush=True)
                # importancia (permutation, sobre lockbox) solo para el pool completo con XGBoost
                if nombre_fs == "v3_pool_seguro" and nombre_algo == "XGBoost":
                    from sklearn.inspection import permutation_importance
                    pi = permutation_importance(m_final, d_lb[feats], d_lb[tcol], n_repeats=5, random_state=0, scoring="r2")
                    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
                    imp.to_csv(SALIDA / f"06_importancia_perm_{fase}_{tcol}.csv")
                    importancias[(fase, tcol)] = imp

res = pd.DataFrame(filas)
res.to_csv(SALIDA / "06_resultados_targets_features.csv", index=False)
pd.set_option("display.width", 250)
print("\n\n===== RESUMEN (lockbox) =====")
print(res.pivot_table(index=["fase", "target"], columns=["feature_set", "algoritmo"], values="r2_lockbox").round(3).to_string())
print("\n===== RESUMEN (OOF) =====")
print(res.pivot_table(index=["fase", "target"], columns=["feature_set", "algoritmo"], values="r2_oof").round(3).to_string())
for k, imp in importancias.items():
    print(f"\n--- top 20 permutation importance (lockbox) {k} ---")
    print(imp.head(20).round(4).to_string())
