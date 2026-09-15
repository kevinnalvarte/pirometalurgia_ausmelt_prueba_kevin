"""Experimento 14: formas de modelo para identificar el efecto de las palancas de CONTROL.

Compara, con la MISMA validacion (OOF GroupKFold(5) por Batch sobre DEV; lockbox solo reporte),
cinco formas de modelo para (fase, target) en {Fusion:sn_kg, Reduccion:sn_kg, Reduccion:feo_kg}:

  A. HGB v3 (baseline, monotonia solo en carbon) -- modelo_predictivo_v3.construir_modelo_v3
  B. HGB con monotonia teorica extendida (tabla de ITERACION_4_diseno.md)
  C. Modelo parcialmente lineal (Robinson) con cross-fitting interno; variante completa y
     parsimoniosa (solo ACCIONES_OPTIMIZABLES_V3 + control de carga en Fusion)
  D. Cinetico log-lineal: D1 HGB sobre log k, D2 Ridge sobre log k, D3 PLM sobre log k;
     reconvertido a kg via disponible*(1-exp(-k*dt))
  E. HGB sobre la fraccion extraida (frac_sn_extraido_escalon), convertida a kg

Ademas: tabla de consistencia de signos entre formas, diagnostico carbon-vs-carga en Fusion,
curva de parsimonia (eliminacion hacia atras) para la mejor forma.

Salidas: 14_comparacion_formas_modelo.csv, 14_efectos_palancas_por_forma.csv, 14_parsimonia.csv,
14_log.txt. Ejecutar con .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=4.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import feature_engineering as fe  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

LOG_PATH = RAIZ / "experimentos" / "14_log.txt"
_LOG_LINES: list[str] = []


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _LOG_LINES.append(line)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")


SEED = 42
N_SPLITS = 5
N_BOOT = 200

# Combinaciones principales del experimento 14
COMBOS = [
    ("Fusión", "sn_kg"),
    ("Reducción", "sn_kg"),
    ("Reducción", "feo_kg"),
]


# =============================================================================
# 0) Dataset
# =============================================================================

def cargar_dev_lockbox() -> tuple[pd.DataFrame, pd.DataFrame, set, set]:
    df = pd.read_pickle(RAIZ / "experimentos" / "cache" / "df_v3.pkl")
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    return df, df_base, batches_dev, batches_lockbox


# =============================================================================
# 1) Forma A y B: HGB con distintas restricciones de monotonia
# =============================================================================

def hgb_v3(feats: list[str], target_col: str, monotonic_cst: list[int] | None = None,
           max_iter: int = 300, seed: int = SEED) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=max_iter,
                                          min_samples_leaf=15, l2_regularization=1.0, random_state=seed,
                                          monotonic_cst=monotonic_cst)


def monotonia_teorica(fase: str, target_col: str, feats: list[str]) -> list[int]:
    """Signos de la tabla teorica de ITERACION_4_diseno.md, restringidos a las columnas presentes."""
    es_sn = target_col == "sn_extraido_est_kg"
    es_feo = target_col == "feo_extraido_est_kg"
    signos: dict[str, int] = {}
    if es_sn:
        signos["tasa_feed_Carbon_kg_min"] = +1
        signos["interaccion_C_x_Sn_prev"] = +1
        signos["exceso_o2_combustion_pct"] = -1 if fase == "Reducción" else 0
        signos["tasa_gn_nm3_min"] = +1
        signos["duracion_plan_min"] = +1
        signos["sn_inventario_escoria_est_kg_prev"] = +1
        signos["ley_sn_escoria_pct_prev"] = +1
        signos["feed_Sn_kgf"] = +1
        if fase == "Fusión":
            signos["relacion_CaO_carga"] = -1
    elif es_feo:
        signos["tasa_feed_Carbon_kg_min"] = +1
        signos["interaccion_C_x_FeO_prev"] = +1
        signos["tasa_gn_nm3_min"] = +1
        signos["exceso_o2_combustion_pct"] = -1
        signos["feo_inventario_escoria_est_kg_prev"] = +1
    return [signos.get(f, 0) for f in feats]


def oof_lockbox_generico(df_base: pd.DataFrame, batches_dev: set, batches_lockbox: set, fase: str,
                          target_col: str, feats: list[str], fit_predict_fn) -> dict:
    """Esqueleto comun de validacion: fit_predict_fn(train_df, test_df) -> np.ndarray de predicciones."""
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    cols = list(dict.fromkeys(feats + [target_col, "Batch"]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats + [target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + [target_col]).reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats], d_dev[target_col], d_dev["Batch"]):
        oof[te] = fit_predict_fn(d_dev.iloc[tr], d_dev.iloc[te])
    p_lb = fit_predict_fn(d_dev, d_lb)
    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof), "mae_oof": mean_absolute_error(d_dev[target_col], oof),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb),
            "y_dev": d_dev[target_col].to_numpy(), "oof": oof, "y_lb": d_lb[target_col].to_numpy(), "p_lb": p_lb,
            "d_dev": d_dev, "d_lb": d_lb}


def forma_hgb(df_base, batches_dev, batches_lockbox, fase, target_col, feats, monotonic_cst=None, seed=SEED):
    def fit_predict(train_df, test_df):
        m = hgb_v3(feats, target_col, monotonic_cst=monotonic_cst, seed=seed)
        m.fit(train_df[feats], train_df[target_col])
        return m.predict(test_df[feats])
    res = oof_lockbox_generico(df_base, batches_dev, batches_lockbox, fase, target_col, feats, fit_predict)
    # modelo final sobre todo DEV, para SHAP/permutacion de signos
    m_full = hgb_v3(feats, target_col, monotonic_cst=monotonic_cst, seed=seed)
    m_full.fit(res["d_dev"][feats], res["d_dev"][target_col])
    res["modelo_full"] = m_full
    return res


# =============================================================================
# 2) Forma C: modelo parcialmente lineal (Robinson) con cross-fitting interno
# =============================================================================

def hgb_nuisance(max_iter=200, seed=SEED):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=max_iter,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=seed)


def _oof_hgb_cols(X: pd.DataFrame, cols: pd.DataFrame, groups: np.ndarray, seed=SEED, max_iter=200) -> pd.DataFrame:
    """OOF (GroupKFold) de cada columna de `cols` regresionada sobre X (cross-fitting interno)."""
    out = pd.DataFrame(index=cols.index, columns=cols.columns, dtype=float)
    for tr, te in GroupKFold(N_SPLITS).split(X, cols.iloc[:, 0], groups):
        for c in cols.columns:
            m = hgb_nuisance(max_iter=max_iter, seed=seed)
            m.fit(X.iloc[tr], cols[c].iloc[tr])
            out.loc[out.index[te], c] = m.predict(X.iloc[te])
    return out


def plm_fit_theta(train: pd.DataFrame, S_cols: list[str], palancas: list[str], target_col: str,
                   seed=SEED, max_iter=200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fitting interno sobre `train`: devuelve theta (lstsq), y_res y A_res (para bootstrap)."""
    g = train["Batch"].to_numpy()
    todas = pd.concat([train[[target_col]], train[palancas]], axis=1)
    todas.columns = [target_col] + palancas
    oof_nuis = _oof_hgb_cols(train[S_cols], todas, g, seed=seed, max_iter=max_iter)
    y_res = train[target_col].to_numpy() - oof_nuis[target_col].to_numpy()
    A_res = np.column_stack([train[a].to_numpy() - oof_nuis[a].to_numpy() for a in palancas])
    theta = np.linalg.lstsq(A_res, y_res, rcond=None)[0]
    return theta, y_res, A_res


def plm_full_models(train: pd.DataFrame, S_cols: list[str], palancas: list[str], target_col: str,
                     seed=SEED, max_iter=200) -> tuple[HistGradientBoostingRegressor, dict]:
    g_full = hgb_nuisance(max_iter=max_iter, seed=seed)
    g_full.fit(train[S_cols], train[target_col])
    m_full = {}
    for a in palancas:
        m = hgb_nuisance(max_iter=max_iter, seed=seed)
        m.fit(train[S_cols], train[a])
        m_full[a] = m
    return g_full, m_full


def plm_predict(test: pd.DataFrame, S_cols: list[str], palancas: list[str], theta: np.ndarray,
                 g_full, m_full: dict) -> np.ndarray:
    pred = g_full.predict(test[S_cols]).astype(float)
    for j, a in enumerate(palancas):
        pred = pred + theta[j] * (test[a].to_numpy() - m_full[a].predict(test[S_cols]))
    return pred


def forma_plm(df_base, batches_dev, batches_lockbox, fase, target_col, S_cols, palancas, seed=SEED, max_iter=200):
    feats_needed = list(dict.fromkeys(S_cols + palancas))
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    cols = list(dict.fromkeys(feats_needed + [target_col, "Batch"]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats_needed + [target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats_needed + [target_col]).reset_index(drop=True)

    # --- OOF (nested: cross-fit interno dentro de cada train externo)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats_needed], d_dev[target_col], d_dev["Batch"]):
        train, test = d_dev.iloc[tr], d_dev.iloc[te]
        theta, _, _ = plm_fit_theta(train, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)
        g_full, m_full = plm_full_models(train, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)
        oof[te] = plm_predict(test, S_cols, palancas, theta, g_full, m_full)

    # --- lockbox: entrenar con todo DEV
    theta_dev, y_res_dev, A_res_dev = plm_fit_theta(d_dev, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)
    g_full_dev, m_full_dev = plm_full_models(d_dev, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)
    p_lb = plm_predict(d_lb, S_cols, palancas, theta_dev, g_full_dev, m_full_dev)

    # --- theta + IC95% (bootstrap cluster por Batch) sobre todo DEV
    g = d_dev["Batch"].to_numpy()
    uniq = np.unique(g)
    idx_by = {b: np.where(g == b)[0] for b in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
        try:
            boots.append(np.linalg.lstsq(A_res_dev[idx], y_res_dev[idx], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    boots = np.array(boots)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    sd_a = d_dev[palancas].std().to_numpy()
    theta_tab = pd.DataFrame({"palanca": palancas, "theta_unidad": theta_dev, "ci_lo": lo, "ci_hi": hi,
                              "theta_1sd": theta_dev * sd_a, "ci_lo_1sd": lo * sd_a, "ci_hi_1sd": hi * sd_a})
    theta_tab["significativo"] = (theta_tab["ci_lo"] > 0) | (theta_tab["ci_hi"] < 0)

    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof), "mae_oof": mean_absolute_error(d_dev[target_col], oof),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb),
            "theta_tab": theta_tab, "d_dev": d_dev, "d_lb": d_lb}


# =============================================================================
# 3) Forma D: cinetico log-lineal
# =============================================================================

def construir_columnas_cineticas(df_base: pd.DataFrame, fase: str, target_key: str) -> pd.DataFrame:
    """Anade columnas log_k, disponible y n_recortados para (fase, target_key)."""
    df = df_base.copy()
    sub_mask = df["fase_proceso"] == fase
    if target_key == "sn_kg":
        frac_raw = df["frac_sn_extraido_escalon"]
        disponible = df["sn_inventario_escoria_est_kg_prev"] + df["feed_Sn_kgf"]
    else:  # feo_kg (solo Reduccion)
        disponible = df["feo_inventario_escoria_est_kg_prev"]
        frac_raw = df["feo_extraido_est_kg"] / disponible.where(disponible > 1.0)
    frac_clip = frac_raw.clip(0.01, 0.99)
    n_recortados = int(((frac_raw < 0.01) | (frac_raw > 0.99)).loc[sub_mask].sum())
    dt = df["duracion_plan_min"].where(df["duracion_plan_min"] > 0)
    k = -np.log(1 - frac_clip) / dt
    df["_frac_raw"], df["_frac_clip"], df["_disponible"], df["_k"], df["_log_k"] = (
        frac_raw, frac_clip, disponible, k, np.log(k.where(k > 0)))
    return df, n_recortados


def kg_desde_k(k_pred: np.ndarray, disponible: np.ndarray, dt: np.ndarray) -> np.ndarray:
    return disponible * (1 - np.exp(-np.exp(k_pred) * dt))
    # nota: k_pred aqui es log_k predicho; se exponencia antes de usar


def forma_d1_hgb_logk(df_cin, batches_dev, batches_lockbox, fase, feats):
    sub = df_cin.loc[df_cin["fase_proceso"] == fase]
    cols = list(dict.fromkeys(feats + ["_log_k", "_disponible", "duracion_plan_min", "Batch",
                                        df_cin.attrs.get("target_col", "")]))
    cols = [c for c in cols if c]
    target_col = df_cin.attrs["target_col"]
    cols = list(dict.fromkeys(cols + [target_col]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats + ["_log_k", target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + ["_log_k", target_col]).reset_index(drop=True)

    def _kg(d, logk_pred):
        return kg_desde_k(logk_pred, d["_disponible"].to_numpy(), d["duracion_plan_min"].to_numpy())

    oof_kg = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats], d_dev["_log_k"], d_dev["Batch"]):
        m = hgb_v3(feats, "_log_k", seed=SEED)
        m.fit(d_dev[feats].iloc[tr], d_dev["_log_k"].iloc[tr])
        pred = m.predict(d_dev[feats].iloc[te])
        oof_kg[te] = _kg(d_dev.iloc[te], pred)
    m_full = hgb_v3(feats, "_log_k", seed=SEED)
    m_full.fit(d_dev[feats], d_dev["_log_k"])
    p_lb = _kg(d_lb, m_full.predict(d_lb[feats]))
    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof_kg), "mae_oof": mean_absolute_error(d_dev[target_col], oof_kg),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb)}


def forma_d2_ridge_logk(df_cin, batches_dev, batches_lockbox, fase, feats):
    sub = df_cin.loc[df_cin["fase_proceso"] == fase]
    target_col = df_cin.attrs["target_col"]
    cols = list(dict.fromkeys(feats + ["_log_k", "_disponible", "duracion_plan_min", "Batch", target_col]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats + ["_log_k", target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + ["_log_k", target_col]).reset_index(drop=True)

    def _kg(d, logk_pred):
        return kg_desde_k(logk_pred, d["_disponible"].to_numpy(), d["duracion_plan_min"].to_numpy())

    oof_kg = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats], d_dev["_log_k"], d_dev["Batch"]):
        sc = StandardScaler().fit(d_dev[feats].iloc[tr])
        Xtr = sc.transform(d_dev[feats].iloc[tr]); Xte = sc.transform(d_dev[feats].iloc[te])
        m = Ridge(alpha=1.0, random_state=SEED)
        m.fit(Xtr, d_dev["_log_k"].iloc[tr])
        pred = m.predict(Xte)
        oof_kg[te] = _kg(d_dev.iloc[te], pred)
    sc_full = StandardScaler().fit(d_dev[feats])
    m_full = Ridge(alpha=1.0, random_state=SEED).fit(sc_full.transform(d_dev[feats]), d_dev["_log_k"])
    p_lb = _kg(d_lb, m_full.predict(sc_full.transform(d_lb[feats])))
    coefs = pd.Series(m_full.coef_, index=feats)
    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof_kg), "mae_oof": mean_absolute_error(d_dev[target_col], oof_kg),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb),
            "coefs": coefs}


def forma_d3_plm_logk(df_cin, batches_dev, batches_lockbox, fase, S_cols, palancas):
    target_col = df_cin.attrs["target_col"]
    sub = df_cin.loc[df_cin["fase_proceso"] == fase]
    feats_needed = list(dict.fromkeys(S_cols + palancas))
    cols = list(dict.fromkeys(feats_needed + ["_log_k", "_disponible", "duracion_plan_min", "Batch", target_col]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats_needed + ["_log_k", target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats_needed + ["_log_k", target_col]).reset_index(drop=True)

    def _kg(d, logk_pred):
        return kg_desde_k(logk_pred, d["_disponible"].to_numpy(), d["duracion_plan_min"].to_numpy())

    oof_kg = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats_needed], d_dev["_log_k"], d_dev["Batch"]):
        train, test = d_dev.iloc[tr], d_dev.iloc[te]
        theta, _, _ = plm_fit_theta(train, S_cols, palancas, "_log_k", seed=SEED, max_iter=150)
        g_full, m_full = plm_full_models(train, S_cols, palancas, "_log_k", seed=SEED, max_iter=150)
        pred = plm_predict(test, S_cols, palancas, theta, g_full, m_full)
        oof_kg[te] = _kg(test, pred)
    theta_dev, _, _ = plm_fit_theta(d_dev, S_cols, palancas, "_log_k", seed=SEED, max_iter=150)
    g_full_dev, m_full_dev = plm_full_models(d_dev, S_cols, palancas, "_log_k", seed=SEED, max_iter=150)
    p_lb_logk = plm_predict(d_lb, S_cols, palancas, theta_dev, g_full_dev, m_full_dev)
    p_lb = _kg(d_lb, p_lb_logk)
    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof_kg), "mae_oof": mean_absolute_error(d_dev[target_col], oof_kg),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb),
            "theta_dev": theta_dev, "palancas": palancas}


# =============================================================================
# 4) Forma E: HGB sobre la fraccion, convertida a kg
# =============================================================================

def forma_e_frac(df_base, batches_dev, batches_lockbox, fase, target_key, feats):
    df = df_base.copy()
    if target_key == "sn_kg":
        frac_col = "frac_sn_extraido_escalon"
        disponible = df["sn_inventario_escoria_est_kg_prev"] + df["feed_Sn_kgf"]
        target_col = "sn_extraido_est_kg"
    else:
        disponible = df["feo_inventario_escoria_est_kg_prev"]
        target_col = "feo_extraido_est_kg"
        frac_col = "_frac_feo_tmp"
        df[frac_col] = df[target_col] / disponible.where(disponible > 1.0)
    df["_disponible_e"] = disponible
    sub = df.loc[df["fase_proceso"] == fase]
    cols = list(dict.fromkeys(feats + [frac_col, "_disponible_e", "Batch", target_col]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats + [frac_col, target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + [frac_col, target_col]).reset_index(drop=True)

    oof_kg = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats], d_dev[frac_col], d_dev["Batch"]):
        m = hgb_v3(feats, frac_col, seed=SEED)
        m.fit(d_dev[feats].iloc[tr], d_dev[frac_col].iloc[tr])
        pred_frac = m.predict(d_dev[feats].iloc[te])
        oof_kg[te] = pred_frac * d_dev["_disponible_e"].iloc[te].to_numpy()
    m_full = hgb_v3(feats, frac_col, seed=SEED)
    m_full.fit(d_dev[feats], d_dev[frac_col])
    p_lb = m_full.predict(d_lb[feats]) * d_lb["_disponible_e"].to_numpy()
    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof_kg), "mae_oof": mean_absolute_error(d_dev[target_col], oof_kg),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb)}


# =============================================================================
# 5) Signos de las palancas por forma (A: SHAP-lite via permutacion+correlacion; C/D3: theta; D2: coef)
# =============================================================================

def signo_por_correlacion_shap(modelo, X: pd.DataFrame, feats: list[str]) -> pd.Series:
    """Signo de cada feature: correlacion (Pearson) entre su valor y su contribucion marginal,
    aproximada por diferencias finitas del modelo alrededor del valor observado (barato, sin
    dependencias de librerias externas de SHAP)."""
    base_pred = modelo.predict(X[feats])
    signos = {}
    for f in feats:
        x = X[f].to_numpy()
        rng = np.ptp(x)
        if rng <= 0 or np.isnan(rng):
            signos[f] = 0.0
            continue
        h = 0.01 * rng
        Xp, Xm = X[feats].copy(), X[feats].copy()
        Xp[f] = X[f] + h
        Xm[f] = X[f] - h
        dpred = (modelo.predict(Xp) - modelo.predict(Xm)) / (2 * h)
        # signo dominante ponderado por |dpred| (evita que zonas planas dominen el promedio)
        w = np.abs(dpred)
        signos[f] = float(np.sign(np.sum(np.sign(dpred) * w))) if w.sum() > 0 else 0.0
    return pd.Series(signos)


# =============================================================================
# main
# =============================================================================

def main():
    t0 = time.time()
    log("Cargando dataset v3 y split DEV/lockbox...")
    df, df_base, batches_dev, batches_lockbox = cargar_dev_lockbox()
    log(f"df_base={len(df_base)} filas; DEV batches={len(batches_dev)}, lockbox batches={len(batches_lockbox)}")

    filas_comp = []
    filas_theta = []
    dic_mejor_forma = {}  # (fase, target_key) -> (feats, S_cols, palancas) de la mejor forma para parsimonia

    for fase, target_key in COMBOS:
        target_col = mp3.TARGETS_V3[fase][target_key]
        feats_v3 = mp3.FEATURES_V3_POR_DEFECTO[fase][target_key]
        log(f"\n===== {fase} / {target_key} ({target_col}), {len(feats_v3)} features v3 =====")

        # --- A: HGB v3 baseline
        log("Forma A: HGB v3 baseline...")
        res_a = forma_hgb(df_base, batches_dev, batches_lockbox, fase, target_col, feats_v3, monotonic_cst=None)
        log(f"  A: r2_oof={res_a['r2_oof']:.4f} mae_oof={res_a['mae_oof']:.1f} "
            f"r2_lb={res_a['r2_lockbox']:.4f} mae_lb={res_a['mae_lockbox']:.1f}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "A_hgb_v3", "n_features": len(feats_v3),
                           "r2_oof": res_a["r2_oof"], "mae_oof": res_a["mae_oof"],
                           "r2_lockbox": res_a["r2_lockbox"], "mae_lockbox": res_a["mae_lockbox"]})

        # --- B: HGB con monotonia teorica extendida
        log("Forma B: HGB monotonia teorica extendida...")
        cst_b = monotonia_teorica(fase, target_col, feats_v3)
        res_b = forma_hgb(df_base, batches_dev, batches_lockbox, fase, target_col, feats_v3, monotonic_cst=cst_b)
        d_oof = res_b["r2_oof"] - res_a["r2_oof"]
        d_lb = res_b["r2_lockbox"] - res_a["r2_lockbox"]
        veredicto_b = ("mejora" if d_oof > 0.005 else ("empeora" if d_oof < -0.005 else "sin cambio"))
        log(f"  B: r2_oof={res_b['r2_oof']:.4f} (d={d_oof:+.4f}, {veredicto_b}) "
            f"r2_lb={res_b['r2_lockbox']:.4f} (d={d_lb:+.4f}) cst={dict(zip(feats_v3, cst_b))}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "B_hgb_monotono_teorico", "n_features": len(feats_v3),
                           "r2_oof": res_b["r2_oof"], "mae_oof": res_b["mae_oof"],
                           "r2_lockbox": res_b["r2_lockbox"], "mae_lockbox": res_b["mae_lockbox"],
                           "vs_A_oof": d_oof, "vs_A_lockbox": d_lb, "veredicto_vs_A": veredicto_b})

        # --- señales para signos de A/B (correlacion pseudo-SHAP)
        for nombre_forma, res in [("A_hgb_v3", res_a), ("B_hgb_monotono_teorico", res_b)]:
            signos = signo_por_correlacion_shap(res["modelo_full"], res["d_dev"], feats_v3)
            for f, s in signos.items():
                if f in mp3.PALANCAS_EXPLICATIVAS_V3.get(fase, []) or f in ["tasa_feed_Carbon_kg_min", "tasa_feed_CaO_kg_min",
                                                                             "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                                                                             "duracion_plan_min", "relacion_CaO_carga"]:
                    filas_theta.append({"fase": fase, "target": target_col, "forma": nombre_forma, "palanca": f,
                                        "theta_unidad": np.nan, "theta_1sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                                        "significativo": np.nan, "signo": s})

        # --- C: PLM completo
        S_base = [s for s in (mp3.STATE_V3[fase] + mp3.CONTEXT_V3[fase])
                  if s not in mp3.PALANCAS_EXPLICATIVAS_V3.get(fase, [])]
        S_base = [s for s in S_base if df_base.loc[df_base["fase_proceso"] == fase, s].notna().mean() > 0.7]
        palancas_c = list(mp3.PALANCAS_EXPLICATIVAS_V3.get(fase, []))

        # términos teóricos extra (i)/(ii)/(iii)
        df_extra = df_base.copy()
        if target_key in ("sn_kg",):
            term_cxsn = "interaccion_C_x_Sn_prev"  # ya existe: tasa_feed_Carbon_kg_min * ley_sn_escoria_pct_prev
            if term_cxsn not in palancas_c:
                palancas_c = palancas_c + [term_cxsn]
        if target_key == "feo_kg":
            term_cxfeo = "interaccion_C_x_FeO_prev"  # tasa_feed_Carbon_kg_min * ley_feo_escoria_pct_prev
            if term_cxfeo not in palancas_c:
                palancas_c = palancas_c + [term_cxfeo]
        if fase == "Fusión":
            df_extra["_tasa_CaO_sq"] = df_extra["tasa_feed_CaO_kg_min"] ** 2
            palancas_c = palancas_c + ["_tasa_CaO_sq"]
        df_extra["_gn_x_temp"] = df_extra["tasa_gn_nm3_min"] * df_extra["temperatura_horno_celsius_prev"]
        palancas_c = palancas_c + ["_gn_x_temp"]
        palancas_c = [p for p in dict.fromkeys(palancas_c) if p not in S_base]

        log(f"Forma C (completa): S={len(S_base)} vars, palancas={palancas_c}")
        res_c = forma_plm(df_extra, batches_dev, batches_lockbox, fase, target_col, S_base, palancas_c, max_iter=150)
        log(f"  C_completo: r2_oof={res_c['r2_oof']:.4f} mae_oof={res_c['mae_oof']:.1f} "
            f"r2_lb={res_c['r2_lockbox']:.4f} mae_lb={res_c['mae_lockbox']:.1f}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "C_plm_completo",
                           "n_features": len(S_base) + len(palancas_c),
                           "r2_oof": res_c["r2_oof"], "mae_oof": res_c["mae_oof"],
                           "r2_lockbox": res_c["r2_lockbox"], "mae_lockbox": res_c["mae_lockbox"],
                           "vs_A_oof": res_c["r2_oof"] - res_a["r2_oof"], "vs_A_lockbox": res_c["r2_lockbox"] - res_a["r2_lockbox"]})
        for _, row in res_c["theta_tab"].iterrows():
            filas_theta.append({"fase": fase, "target": target_col, "forma": "C_plm_completo", "palanca": row["palanca"],
                                "theta_unidad": row["theta_unidad"], "theta_1sd": row["theta_1sd"],
                                "ci_lo": row["ci_lo_1sd"], "ci_hi": row["ci_hi_1sd"],
                                "significativo": row["significativo"], "signo": np.sign(row["theta_unidad"])})

        # --- C parsimoniosa: solo ACCIONES_OPTIMIZABLES_V3 (+ tasa de carga como control en Fusion)
        palancas_pars = list(mp3.ACCIONES_OPTIMIZABLES_V3.get(fase, []))
        S_pars = list(S_base)
        if fase == "Fusión":
            palancas_pars = palancas_pars + ["tasa_feed_total_kg_min"]
        palancas_pars = [p for p in dict.fromkeys(palancas_pars) if p not in S_pars]
        log(f"Forma C (parsimoniosa): palancas={palancas_pars}")
        res_c_p = forma_plm(df_base, batches_dev, batches_lockbox, fase, target_col, S_pars, palancas_pars, max_iter=150)
        log(f"  C_parsimoniosa: r2_oof={res_c_p['r2_oof']:.4f} r2_lb={res_c_p['r2_lockbox']:.4f}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "C_plm_parsimonioso",
                           "n_features": len(S_pars) + len(palancas_pars),
                           "r2_oof": res_c_p["r2_oof"], "mae_oof": res_c_p["mae_oof"],
                           "r2_lockbox": res_c_p["r2_lockbox"], "mae_lockbox": res_c_p["mae_lockbox"],
                           "vs_A_oof": res_c_p["r2_oof"] - res_a["r2_oof"], "vs_A_lockbox": res_c_p["r2_lockbox"] - res_a["r2_lockbox"]})
        for _, row in res_c_p["theta_tab"].iterrows():
            filas_theta.append({"fase": fase, "target": target_col, "forma": "C_plm_parsimonioso", "palanca": row["palanca"],
                                "theta_unidad": row["theta_unidad"], "theta_1sd": row["theta_1sd"],
                                "ci_lo": row["ci_lo_1sd"], "ci_hi": row["ci_hi_1sd"],
                                "significativo": row["significativo"], "signo": np.sign(row["theta_unidad"])})

        # --- diagnostico carbon-vs-carga (solo Fusion, reportado una vez)
        if fase == "Fusión" and target_key == "sn_kg":
            sub_f = df_base.loc[df_base["fase_proceso"] == "Fusión"]
            corr_tasa = sub_f["tasa_feed_Carbon_kg_min"].corr(sub_f["tasa_feed_total_kg_min"])
            corr_feed = sub_f["tasa_feed_Carbon_kg_min"].corr(sub_f["feed_Sn_kgf"])
            log(f"Diagnostico carbon-vs-carga (Fusion): corr(C, tasa_carga_total)={corr_tasa:.3f}, "
                f"corr(C, feed_Sn_kgf)={corr_feed:.3f}")
            # PLM con relacion_C_Sn_carga en lugar de tasa_feed_Carbon_kg_min
            palancas_rel = [p if p != "tasa_feed_Carbon_kg_min" else "relacion_C_Sn_carga" for p in palancas_c]
            palancas_rel = [p for p in dict.fromkeys(palancas_rel) if p not in S_base]
            res_rel = forma_plm(df_extra, batches_dev, batches_lockbox, fase, target_col, S_base, palancas_rel, max_iter=150)
            fila_rel = res_rel["theta_tab"].set_index("palanca").loc["relacion_C_Sn_carga"]
            log(f"  PLM con relacion_C_Sn_carga: theta={fila_rel['theta_unidad']:.3f} "
                f"IC95%=[{fila_rel['ci_lo']:.3f},{fila_rel['ci_hi']:.3f}] "
                f"significativo={bool(fila_rel['significativo'])} r2_oof={res_rel['r2_oof']:.4f}")
            filas_theta.append({"fase": fase, "target": target_col, "forma": "C_plm_relacion_C_Sn_carga",
                                "palanca": "relacion_C_Sn_carga", "theta_unidad": fila_rel["theta_unidad"],
                                "theta_1sd": fila_rel["theta_1sd"], "ci_lo": fila_rel["ci_lo_1sd"], "ci_hi": fila_rel["ci_hi_1sd"],
                                "significativo": fila_rel["significativo"], "signo": np.sign(fila_rel["theta_unidad"])})
            filas_comp.append({"fase": fase, "target": target_col, "forma": "C_plm_relacion_C_Sn_carga",
                               "n_features": len(S_base) + len(palancas_rel),
                               "r2_oof": res_rel["r2_oof"], "mae_oof": res_rel["mae_oof"],
                               "r2_lockbox": res_rel["r2_lockbox"], "mae_lockbox": res_rel["mae_lockbox"]})
            with open(RAIZ / "experimentos" / "14_diagnostico_carbon_carga.txt", "w", encoding="utf-8") as fh:
                fh.write(f"corr(tasa_feed_Carbon_kg_min, tasa_feed_total_kg_min) en Fusion = {corr_tasa:.4f}\n")
                fh.write(f"corr(tasa_feed_Carbon_kg_min, feed_Sn_kgf) en Fusion = {corr_feed:.4f}\n")
                fh.write(f"PLM con relacion_C_Sn_carga (en vez de tasa absoluta): theta={fila_rel['theta_unidad']:.4f} "
                         f"unidades kg Sn extra por kg-C/kg-Sn, IC95%=[{fila_rel['ci_lo']:.4f},{fila_rel['ci_hi']:.4f}], "
                         f"significativo={bool(fila_rel['significativo'])}\n")
                fh.write(f"r2_oof PLM con relacion_C_Sn_carga = {res_rel['r2_oof']:.4f} vs PLM con tasa absoluta = {res_c['r2_oof']:.4f}\n")

        # --- D: cinetico log-lineal
        log("Forma D: cinetico log-lineal...")
        df_cin, n_recortados = construir_columnas_cineticas(df_base, fase, target_key)
        df_cin.attrs["target_col"] = target_col
        log(f"  filas recortadas a [0.01,0.99] en {fase}/{target_key}: {n_recortados}")
        res_d1 = forma_d1_hgb_logk(df_cin, batches_dev, batches_lockbox, fase, feats_v3)
        log(f"  D1 (HGB log k): r2_oof={res_d1['r2_oof']:.4f} r2_lb={res_d1['r2_lockbox']:.4f}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "D1_hgb_logk", "n_features": len(feats_v3),
                           "r2_oof": res_d1["r2_oof"], "mae_oof": res_d1["mae_oof"],
                           "r2_lockbox": res_d1["r2_lockbox"], "mae_lockbox": res_d1["mae_lockbox"],
                           "n_recortados_frac": n_recortados})
        res_d2 = forma_d2_ridge_logk(df_cin, batches_dev, batches_lockbox, fase, feats_v3)
        log(f"  D2 (Ridge log k): r2_oof={res_d2['r2_oof']:.4f} r2_lb={res_d2['r2_lockbox']:.4f}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "D2_ridge_logk", "n_features": len(feats_v3),
                           "r2_oof": res_d2["r2_oof"], "mae_oof": res_d2["mae_oof"],
                           "r2_lockbox": res_d2["r2_lockbox"], "mae_lockbox": res_d2["mae_lockbox"],
                           "n_recortados_frac": n_recortados})
        for f, c in res_d2["coefs"].items():
            if f in ["tasa_feed_Carbon_kg_min", "tasa_feed_CaO_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                     "duracion_plan_min", "relacion_CaO_carga"]:
                filas_theta.append({"fase": fase, "target": target_col, "forma": "D2_ridge_logk", "palanca": f,
                                    "theta_unidad": c, "theta_1sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                                    "significativo": np.nan, "signo": np.sign(c)})

        S_cin = [s for s in S_base if s in df_cin.columns]
        palancas_cin = [p for p in palancas_c if p in df_cin.columns and p not in S_cin]
        res_d3 = forma_d3_plm_logk(df_cin, batches_dev, batches_lockbox, fase, S_cin, palancas_cin)
        log(f"  D3 (PLM log k): r2_oof={res_d3['r2_oof']:.4f} r2_lb={res_d3['r2_lockbox']:.4f}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "D3_plm_logk",
                           "n_features": len(S_cin) + len(palancas_cin),
                           "r2_oof": res_d3["r2_oof"], "mae_oof": res_d3["mae_oof"],
                           "r2_lockbox": res_d3["r2_lockbox"], "mae_lockbox": res_d3["mae_lockbox"],
                           "n_recortados_frac": n_recortados})
        for p, th in zip(res_d3["palancas"], res_d3["theta_dev"]):
            if p in ["tasa_feed_Carbon_kg_min", "tasa_feed_CaO_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                     "duracion_plan_min"]:
                filas_theta.append({"fase": fase, "target": target_col, "forma": "D3_plm_logk", "palanca": p,
                                    "theta_unidad": th, "theta_1sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                                    "significativo": np.nan, "signo": np.sign(th)})

        # --- E: HGB sobre fraccion
        log("Forma E: HGB sobre fraccion extraida...")
        res_e = forma_e_frac(df_base, batches_dev, batches_lockbox, fase, target_key, feats_v3)
        log(f"  E: r2_oof={res_e['r2_oof']:.4f} r2_lb={res_e['r2_lockbox']:.4f}")
        filas_comp.append({"fase": fase, "target": target_col, "forma": "E_hgb_fraccion", "n_features": len(feats_v3),
                           "r2_oof": res_e["r2_oof"], "mae_oof": res_e["mae_oof"],
                           "r2_lockbox": res_e["r2_lockbox"], "mae_lockbox": res_e["mae_lockbox"]})

        dic_mejor_forma[(fase, target_key)] = {"res_a": res_a, "feats_v3": feats_v3}

    tabla_comp = pd.DataFrame(filas_comp)
    tabla_theta = pd.DataFrame(filas_theta)
    tabla_comp.to_csv(RAIZ / "experimentos" / "14_comparacion_formas_modelo.csv", index=False)
    tabla_theta.to_csv(RAIZ / "experimentos" / "14_efectos_palancas_por_forma.csv", index=False)
    log(f"\nGuardado 14_comparacion_formas_modelo.csv ({len(tabla_comp)} filas)")
    log(f"Guardado 14_efectos_palancas_por_forma.csv ({len(tabla_theta)} filas)")

    # =========================================================================
    # 6) Curva de parsimonia para la mejor forma (A: HGB v3) por eliminacion hacia atras
    # =========================================================================
    log("\n===== Curva de parsimonia (eliminacion hacia atras, forma A) =====")
    filas_pars = []
    for fase, target_key in COMBOS:
        target_col = mp3.TARGETS_V3[fase][target_key]
        feats = list(mp3.FEATURES_V3_POR_DEFECTO[fase][target_key])
        log(f"-- {fase}/{target_key}: partiendo de {len(feats)} features")
        restantes = list(feats)
        historial = []
        while len(restantes) >= 1:
            res = forma_hgb(df_base, batches_dev, batches_lockbox, fase, target_col, restantes, monotonic_cst=None)
            historial.append((len(restantes), res["r2_oof"], res["r2_lockbox"], list(restantes)))
            filas_pars.append({"fase": fase, "target": target_col, "n_features": len(restantes),
                               "r2_oof": res["r2_oof"], "r2_lockbox": res["r2_lockbox"], "features": ";".join(restantes)})
            if len(restantes) == 1:
                break
            importancias = pd.Series(res["modelo_full"].feature_importances_ if hasattr(res["modelo_full"], "feature_importances_")
                                     else np.zeros(len(restantes)), index=restantes)
            if importancias.sum() == 0:
                # HGB no expone feature_importances_; usar permutacion barata (una repeticion)
                base_r2 = res["r2_oof"]
                perdidas = {}
                rng = np.random.default_rng(SEED)
                d_dev = res["d_dev"]
                for f in restantes:
                    xp = d_dev.copy()
                    xp[f] = rng.permutation(xp[f].to_numpy())
                    pred_perm = res["modelo_full"].predict(xp[restantes])
                    perdidas[f] = base_r2 - r2_score(d_dev[target_col], pred_perm)
                peor = min(perdidas, key=perdidas.get)
            else:
                peor = importancias.idxmin()
            restantes = [f for f in restantes if f != peor]
        r2_max = max(h[1] for h in historial)
        candidatos = [h for h in historial if h[1] >= r2_max - 0.01 and h[0] <= 9]
        if candidatos:
            mejor = max(candidatos, key=lambda h: h[1])
        else:
            mejor = min(historial, key=lambda h: h[0])
        log(f"   set <=9 features con perdida <=0.01 de R2 OOF (max={r2_max:.4f}): "
            f"k={mejor[0]}, r2_oof={mejor[1]:.4f}, r2_lockbox={mejor[2]:.4f}")
        log(f"   features: {mejor[3]}")

    tabla_pars = pd.DataFrame(filas_pars)
    tabla_pars.to_csv(RAIZ / "experimentos" / "14_parsimonia.csv", index=False)
    log(f"Guardado 14_parsimonia.csv ({len(tabla_pars)} filas)")

    log(f"\nTiempo total: {time.time() - t0:.1f} s")
    flush_log()


if __name__ == "__main__":
    main()
