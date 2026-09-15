"""Prototipo: Double ML con cross-fitting (Chernozhukov et al. 2018, partialling-out)
para las acciones principales, comparado contra el FWL de una sola pasada del Paso 42.

Mejora metodologica sobre Paso 42:
  - Paso 42: ajusta E[A|S] y E[Y|S] UNA vez sobre TODO el dev set y toma residuos IN-SAMPLE
    (mismo dato para entrenar y residualizar) -> XGBoost puede sobreajustar y contaminar el residuo.
  - Aqui: cross-fitting real (K folds agrupados por Batch), residuos siempre OUT-OF-FOLD,
    estimador final theta = regresion OLS robusta de Y_res ~ A_res (Robinson/partialling-out),
    error estandar por cluster de Batch (bootstrap por cluster), y refutation tests
    (placebo/random treatment, random common cause, subsample stability) al estilo DoWhy.
"""
import os
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, r"C:\Users\user\Desktop\Linea de Sn\Lingo Smelter")
import dataset_lingo_smelter as dls
import feature_engineering as fe
import modelo_prescriptivo as mp

PARAMS_NUISANCE = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
                        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=1.0, random_state=42, n_jobs=-1)

df = mp.construir_dataset_modelo(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datos Lingo smelter fase II.xlsx"))
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)

TARGET_FISICO = "d_ley_sn_escoria_pct"  # misma escala fisica en las 2 fases para el analisis causal

# Candidatas de accion a evaluar por fase (uniendo lo que ya se evaluo en Paso 42 + alguna extra)
ACCIONES_CANDIDATAS = {
    "Fusión": ["delta_tiempo", "tasa_feed_Carbon_kg_min"],
    "Reducción": ["tasa_gn_nm3_min", "exceso_o2_combustion_pct", "posicion_vertical_lanza_mm", "tasa_feed_Carbon_kg_min"],
}


def _oof_predict(X, y, groups, n_splits=5, seed=0):
    """Prediccion OOF (GroupKFold por Batch) de un XGBoost -- nuisance model cross-fitted."""
    oof = np.full(len(X), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    params = dict(PARAMS_NUISANCE, random_state=seed)
    for train_idx, test_idx in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = m.predict(X.iloc[test_idx])
    return oof


def _cluster_bootstrap_ci(a_res, y_res, batches, n_boot=300, seed=0):
    """IC 95% de theta = cov(A_res,Y_res)/var(A_res) por bootstrap clusterizado en Batch."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(batches)
    thetas = []
    idx_by_batch = {b: np.where(batches == b)[0] for b in uniq}
    for _ in range(n_boot):
        sample_batches = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_batch[b] for b in sample_batches])
        a_b, y_b = a_res[idx], y_res[idx]
        var_a = np.var(a_b)
        if var_a < 1e-12:
            continue
        thetas.append(np.cov(a_b, y_b, bias=True)[0, 1] / var_a)
    thetas = np.array(thetas)
    return float(np.mean(thetas)), float(np.percentile(thetas, 2.5)), float(np.percentile(thetas, 97.5)), float(np.std(thetas))


def dml_partialling_out(sub, state_feats, accion, target_col, n_splits=5, seed=0):
    """Devuelve theta (efecto marginal medio de `accion` sobre `target_col`, controlando `state_feats`
    via cross-fitting), IC 95% (bootstrap cluster por Batch) y los residuos (para refutation tests)."""
    cols = list(dict.fromkeys(state_feats + [accion, target_col, "Batch"]))
    d = sub[cols].dropna().reset_index(drop=True)
    X = d[state_feats]
    y_hat = _oof_predict(X, d[target_col], d["Batch"].to_numpy(), n_splits, seed)
    a_hat = _oof_predict(X, d[accion], d["Batch"].to_numpy(), n_splits, seed)
    y_res = d[target_col].to_numpy() - y_hat
    a_res = d[accion].to_numpy() - a_hat
    theta_mean, lo, hi, se = _cluster_bootstrap_ci(a_res, y_res, d["Batch"].to_numpy(), seed=seed)
    theta_point = float(np.cov(a_res, y_res, bias=True)[0, 1] / np.var(a_res))
    return {
        "theta": theta_point, "theta_boot_mean": theta_mean, "ci_lo": lo, "ci_hi": hi, "se_boot": se,
        "n": len(d), "a_res": a_res, "y_res": y_res, "batches": d["Batch"].to_numpy(),
    }


def refutation_placebo(sub, state_feats, accion, target_col, n_reps=12, seed=0):
    """Placebo: reemplaza la accion real por ruido con la MISMA distribucion marginal (shuffle),
    reestima theta -- si el metodo es valido, theta_placebo debe concentrarse cerca de 0."""
    cols = list(dict.fromkeys(state_feats + [accion, target_col, "Batch"]))
    d = sub[cols].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    thetas = []
    for r in range(n_reps):
        d2 = d.copy()
        d2[accion] = rng.permutation(d2[accion].to_numpy())
        res = dml_partialling_out(d2, state_feats, accion, target_col, seed=seed + r)
        thetas.append(res["theta"])
    return np.array(thetas)


def refutation_random_confounder(sub, state_feats, accion, target_col, seed=0):
    """Agrega una variable de ruido aleatorio como 'confusor' adicional en las nuisance models;
    el theta no deberia cambiar sustancialmente si el resultado es robusto."""
    d = sub.copy()
    rng = np.random.default_rng(seed)
    d["_ruido_confusor"] = rng.normal(size=len(d))
    return dml_partialling_out(d, state_feats + ["_ruido_confusor"], accion, target_col, seed=seed)


def refutation_subset_stability(sub, state_feats, accion, target_col, frac=0.7, n_reps=10, seed=0):
    """Reestima theta sobre submuestras aleatorias (por Batch) del `frac` de los datos; theta deberia
    ser estable en signo/magnitud si el efecto es real y no un artefacto de unos pocos batches."""
    batches = sub["Batch"].unique()
    rng = np.random.default_rng(seed)
    thetas = []
    for r in range(n_reps):
        sample_b = rng.choice(batches, size=int(len(batches) * frac), replace=False)
        sub_r = sub.loc[sub["Batch"].isin(sample_b)]
        try:
            res = dml_partialling_out(sub_r, state_feats, accion, target_col, seed=seed + r)
            thetas.append(res["theta"])
        except Exception:
            continue
    return np.array(thetas)


resultados = []
detalle_residuos = {}
for fase, acciones in ACCIONES_CANDIDATAS.items():
    state_feats = mp.FEATURES_STATE[fase]
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
    for accion in acciones:
        base = dml_partialling_out(sub, state_feats, accion, TARGET_FISICO)
        placebo = refutation_placebo(sub, state_feats, accion, TARGET_FISICO)
        conf = refutation_random_confounder(sub, state_feats, accion, TARGET_FISICO)
        subset = refutation_subset_stability(sub, state_feats, accion, TARGET_FISICO)

        # correlacion FWL de una sola pasada (Paso 42, para comparar directamente)
        from scipy.stats import spearmanr, pearsonr
        rho_bruto, p_bruto = spearmanr(sub[accion], sub[TARGET_FISICO])

        fila = {
            "fase": fase, "accion": accion, "n": base["n"],
            "theta_dml_crossfit": base["theta"], "ci95_lo": base["ci_lo"], "ci95_hi": base["ci_hi"],
            "se_boot": base["se_boot"],
            "significativo_95": (base["ci_lo"] > 0) or (base["ci_hi"] < 0),
            "rho_bruto_spearman": rho_bruto,
            "placebo_theta_media": float(placebo.mean()), "placebo_theta_std": float(placebo.std()),
            "placebo_pasa": bool(abs(base["theta"]) > 3 * placebo.std()) if placebo.std() > 0 else None,
            "confounder_theta": conf["theta"],
            "confounder_cambio_pct": 100 * abs(conf["theta"] - base["theta"]) / (abs(base["theta"]) + 1e-9),
            "subset_theta_media": float(subset.mean()) if len(subset) else np.nan,
            "subset_theta_std": float(subset.std()) if len(subset) else np.nan,
            "subset_frac_mismo_signo": float(np.mean(np.sign(subset) == np.sign(base["theta"]))) if len(subset) else np.nan,
        }
        resultados.append(fila)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()}, ensure_ascii=False))

tabla = pd.DataFrame(resultados)
tabla.to_csv(os.path.join(os.path.dirname(__file__), "tabla_dml_crossfit.csv"), index=False)
print("\nGuardado tabla_dml_crossfit.csv")
