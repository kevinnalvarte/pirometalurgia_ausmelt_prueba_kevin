"""Seleccion de features v3 por (fase, target): permutation importance estable + Boruta-shadow
+ decorrelacion + validacion del tamano del set (OOF GroupKFold + lockbox).

Uso: python experimentos/09_seleccion_features_v3.py
Salidas: experimentos/09_ranking_<fase>_<target>.csv, 09_curva_tamano_<fase>_<target>.csv,
         09_sets_finales.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import dataset_lingo_smelter as dls  # noqa: E402
import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

SALIDA = RAIZ / "experimentos"
df = dls.construir_dataset(RAIZ / "Datos Lingo smelter fase II.xlsx", clean_mode=True)
df = fe.construir_features_v3(df)
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
for col in ["sn_extraido_est_kg", "frac_sn_extraido_escalon", "feo_extraido_est_kg"]:
    df_base[col] = df_base.groupby("fase_proceso")[col].transform(lambda s: s.where(s.between(*s.quantile([0.005, 0.995]))))

EXCLUIR_SIEMPRE = {"Batch", "fase_proceso", "fecha_inicio", "fecha_final", "es_fusion", "feed_CaO_total_excel_kg",
                   "estequiometria_lanza_planta_pct", "enriquecimiento_o2_lanza_planta_pct",  # duplicados exactos
                   "o2_teorico_combustion_nm3", "o2_disponible_nm3",  # volumenes brutos (redundantes con tasas)
                   "sn_conc_t1_kgf", "sn_reciclo_t3_kgf", "sn_Fe_t4_kgf", "sn_dross_Fe_t5_kgf", "sn_pellets_t7_kgf",  # = tmh x ley batch
                   "feed_carga_secundaria_kgh"}
SOLO_FUSION = {"frac_carga_secundaria", "ley_sn_carga_pct", "relacion_CaO_carga", "relacion_C_Sn_carga",
               "exceso_C_estequiometrico_pct", "oxidante_sobre_carga_total", "carga_termica_relativa", "relacion_C_carga",
               "frac_conc_carga", "frac_reciclo_carga", "frac_Fe_carga", "frac_dross_carga", "frac_pellets_carga", "relacion_CaO_carga_Fe"}
# Nota: `delta_tiempo` (duracion REAL) se mantiene en el pool como referencia; `duracion_plan_min`
# es la version ACTION limpia. Ambas se reportan para poder compararlas.

def pool_para(fase):
    pool = [c for c in fe.filtrar_features_seguras(list(df_base.columns)) if c not in EXCLUIR_SIEMPRE and df_base[c].dtype != object and not c.startswith("es_")]
    pool = [c for c in pool if df_base.loc[df_base["fase_proceso"] == fase, c].notna().mean() > 0.7]
    if fase == "Reducción":
        pool = [c for c in pool if c not in SOLO_FUSION]
    return pool

PARES = [("Fusión", "sn_extraido_est_kg"), ("Fusión", "d_ley_sn_escoria_pct"), ("Fusión", "d_ley_feo_escoria_pct"),
         ("Reducción", "tasa_relativa_sn_escoria"), ("Reducción", "sn_extraido_est_kg"), ("Reducción", "d_ley_sn_escoria_pct"),
         ("Reducción", "d_ley_feo_escoria_pct"), ("Reducción", "feo_extraido_est_kg")]

def hgb(seed=0):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=15, l2_regularization=1.0, random_state=seed)

def perm_estable(d, feats, target, n_seeds=3):
    acc = pd.DataFrame(0.0, index=feats, columns=["media", "frac_pos"])
    k = 0
    for seed in range(n_seeds):
        for tr, te in GroupKFold(5).split(d[feats], d[target], d["Batch"]):
            m = hgb(seed).fit(d[feats].iloc[tr], d[target].iloc[tr])
            pi = permutation_importance(m, d[feats].iloc[te], d[target].iloc[te], n_repeats=3, random_state=seed, scoring="r2")
            acc["media"] += pi.importances_mean; acc["frac_pos"] += (pi.importances_mean > 0); k += 1
    return acc / k

def boruta_shadow(d, feats, target, n_iter=20, seed=0):
    rng = np.random.default_rng(seed); hits = pd.Series(0, index=feats, dtype=float)
    X = d[feats].to_numpy(); y = d[target].to_numpy()
    for it in range(n_iter):
        Xs = X.copy()
        for j in range(Xs.shape[1]):
            Xs[:, j] = rng.permutation(Xs[:, j])
        Xall = np.hstack([X, Xs])
        m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.6, random_state=seed + it, n_jobs=-1)
        m.fit(Xall, y)
        imp = m.feature_importances_
        umbral = imp[len(feats):].max()
        hits += (imp[:len(feats)] > umbral)
    return hits / n_iter

def decorrelar(d, ranking, umbral=0.85):
    """Recorre el ranking de mejor a peor; descarta una feature si |Spearman| > umbral con alguna ya aceptada."""
    aceptadas = []
    corr = d[ranking].corr(method="spearman").abs()
    for f in ranking:
        if all(corr.loc[f, a] <= umbral for a in aceptadas if pd.notna(corr.loc[f, a])):
            aceptadas.append(f)
    return aceptadas

def evaluar(d_dev, d_lb, feats, target, monot=True):
    signo_c = +1 if target in {"sn_extraido_est_kg", "feo_extraido_est_kg", "frac_sn_extraido_escalon"} else -1
    mono = {"tasa_feed_Carbon_kg_min", "interaccion_C_x_Sn_prev", "tasa_feed_Carbon_reduccion_kg_min", "cum_feed_Carbon_kg_prev"}
    cst = [signo_c if (monot and f in mono) else 0 for f in feats]
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(5).split(d_dev[feats], d_dev[target], d_dev["Batch"]):
        m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15, l2_regularization=1.0, random_state=42, monotonic_cst=cst)
        m.fit(d_dev[feats].iloc[tr], d_dev[target].iloc[tr]); oof[te] = m.predict(d_dev[feats].iloc[te])
    m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15, l2_regularization=1.0, random_state=42, monotonic_cst=cst)
    m.fit(d_dev[feats], d_dev[target]); p = m.predict(d_lb[feats])
    return dict(r2_oof=r2_score(d_dev[target], oof), mae_oof=mean_absolute_error(d_dev[target], oof), r2_lockbox=r2_score(d_lb[target], p), mae_lockbox=mean_absolute_error(d_lb[target], p))

sets_finales = {}
for fase, target in PARES:
    pool = pool_para(fase)
    sub = df_base.loc[df_base["fase_proceso"] == fase].dropna(subset=[target]).reset_index(drop=True)
    d_dev = sub.loc[sub["Batch"].isin(batches_dev)].reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox)].reset_index(drop=True)
    print(f"\n===== {fase} | {target} | pool={len(pool)} | dev={len(d_dev)} lb={len(d_lb)} =====", flush=True)
    # XGB/Boruta no acepta NaN? XGBoost si acepta NaN. HistGB tambien.
    pi = perm_estable(d_dev, pool, target)
    bo = boruta_shadow(d_dev, pool, target)
    rank = pd.DataFrame({"perm_media": pi["media"], "perm_frac_pos": pi["frac_pos"], "boruta_frac": bo})
    rank["score"] = rank["perm_media"].rank(pct=True) * 0.5 + rank["boruta_frac"].rank(pct=True) * 0.5
    rank = rank.sort_values("score", ascending=False)
    rank.to_csv(SALIDA / f"09_ranking_{fase}_{target}.csv")
    print(rank.head(25).round(4).to_string(), flush=True)
    candidatas = [f for f in rank.index if rank.loc[f, "boruta_frac"] >= 0.3 or rank.loc[f, "perm_frac_pos"] >= 0.8]
    candidatas = candidatas[:40]
    decor = decorrelar(d_dev, candidatas)
    print(f"candidatas={len(candidatas)} -> decorrelacionadas={len(decor)}: {decor}", flush=True)
    curva = []
    for k in [4, 6, 8, 10, 12, 15, 20, len(decor)]:
        k = min(k, len(decor)); feats = decor[:k]
        ev = evaluar(d_dev, d_lb, feats, target); ev.update(k=k, features=";".join(feats)); curva.append(ev)
        print(f"  k={k:2d}: R2 oof={ev['r2_oof']:.3f} lb={ev['r2_lockbox']:.3f} mae_lb={ev['mae_lockbox']:.3f}", flush=True)
    curva = pd.DataFrame(curva).drop_duplicates(subset="k")
    curva.to_csv(SALIDA / f"09_curva_tamano_{fase}_{target}.csv", index=False)
    # set final: el k mas chico cuyo R2 OOF >= max(R2 OOF) - 0.01 (parsimonia con tolerancia)
    mejor = curva["r2_oof"].max()
    k_sel = int(curva.loc[curva["r2_oof"] >= mejor - 0.01, "k"].min())
    sets_finales[f"{fase}|{target}"] = {"k": k_sel, "features": decor[:k_sel], "r2_oof": float(curva.loc[curva.k == k_sel, "r2_oof"].iloc[0]),
                                        "r2_lockbox": float(curva.loc[curva.k == k_sel, "r2_lockbox"].iloc[0])}
    print(f"  -> set final k={k_sel}: {decor[:k_sel]}", flush=True)
    (SALIDA / "09_sets_finales.json").write_text(json.dumps(sets_finales, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nLISTO")
