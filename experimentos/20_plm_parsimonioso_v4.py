"""Experimento 20: modelo parcialmente lineal PARSIMONIOSO por fase (candidato v4).

Forma: y_hat = g_hat(S) + sum_j theta_j * (a_j - m_hat_j(S)), con S un estado curado pequeno
(HGB no lineal, max_depth 3-4) y palancas a_j lineales con estructura cinetica (interacciones
carbon x inventario / carbon x avance, terminos cuadraticos de saturacion). Cross-fitting interno
GroupKFold(5) por Batch para los residuos de Robinson (1988); theta por OLS; IC95% por bootstrap
cluster por Batch (150 replicas). Validacion externa: OOF GroupKFold(5) por Batch en DEV (anidado)
y lockbox (entrenando con todo DEV).

Reutiliza (copiando, sin importar) la logica de cross-fitting interno de
`experimentos/14_formas_modelo_palancas.py` (plm_fit_theta, plm_full_models, plm_predict).

Salidas: 20_validacion_v4.csv, 20_theta_v4.csv, 20_curva_parsimonia.csv, 20_curvas_respuesta.csv,
figs_20/*.png, 20_log.txt. Ejecutar con .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8,
OMP_NUM_THREADS=4. Sin joblib/multiprocessing; disenado para < 9 min.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import modelo_prescriptivo as mp  # noqa: E402

EXP_DIR = RAIZ / "experimentos"
FIGS_DIR = EXP_DIR / "figs_20"
FIGS_DIR.mkdir(exist_ok=True)
LOG_PATH = EXP_DIR / "20_log.txt"
_LOG_LINES: list[str] = []


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")


SEED = 42
N_SPLITS = 5
N_BOOT = 150
MAX_ITER_NUIS = 150
MAX_ITER_HGB = 200


# =============================================================================
# 0) Dataset y columnas derivadas de estructura cinetica
# =============================================================================

def cargar_dev_lockbox():
    df = pd.read_pickle(EXP_DIR / "cache" / "df_v3.pkl")
    df_base = mp.dataset_base_modelo(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    return df, df_base, batches_dev, batches_lockbox


def preparar_columnas(df_base: pd.DataFrame) -> pd.DataFrame:
    """Anade las columnas de estructura cinetica/saturacion (funciones deterministas de
    primitivas ACTION/STATE ya vetadas como seguras -> heredan esa seguridad)."""
    d = df_base.copy()
    d["_Cx_sn"] = d["tasa_feed_Carbon_kg_min"] * d["sn_inventario_escoria_est_kg_prev"] / 1000.0
    d["_Cx_feo"] = d["tasa_feed_Carbon_kg_min"] * d["avance_reduccion_sn_prev"]
    d["_relC_sq"] = d["relacion_C_Sn_carga"] ** 2
    d["_relCaO_sq"] = d["relacion_CaO_carga"] ** 2
    d["_disponible_fusion"] = d["sn_inventario_escoria_est_kg_prev"] + d["feed_Sn_kgf"]
    return d


# =============================================================================
# 1) PLM (Robinson) con cross-fitting interno -- copiado/adaptado de experimentos/14
# =============================================================================

def hgb_nuisance(max_iter: int = MAX_ITER_NUIS, seed: int = SEED) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=max_iter,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=seed)


def _oof_hgb_cols(X: pd.DataFrame, cols: pd.DataFrame, groups: np.ndarray, seed=SEED,
                   max_iter=MAX_ITER_NUIS) -> pd.DataFrame:
    out = pd.DataFrame(index=cols.index, columns=cols.columns, dtype=float)
    for tr, te in GroupKFold(N_SPLITS).split(X, cols.iloc[:, 0], groups):
        for c in cols.columns:
            m = hgb_nuisance(max_iter=max_iter, seed=seed)
            m.fit(X.iloc[tr], cols[c].iloc[tr])
            out.loc[out.index[te], c] = m.predict(X.iloc[te])
    return out


def plm_fit_theta(train: pd.DataFrame, S_cols, palancas, target_col, seed=SEED, max_iter=MAX_ITER_NUIS):
    g = train["Batch"].to_numpy()
    todas = train[[target_col] + palancas].copy()
    todas.columns = [target_col] + palancas
    oof_nuis = _oof_hgb_cols(train[S_cols], todas, g, seed=seed, max_iter=max_iter)
    y_res = train[target_col].to_numpy() - oof_nuis[target_col].to_numpy()
    A_res = np.column_stack([train[a].to_numpy() - oof_nuis[a].to_numpy() for a in palancas])
    theta = np.linalg.lstsq(A_res, y_res, rcond=None)[0]
    return theta, y_res, A_res


def plm_full_models(train: pd.DataFrame, S_cols, palancas, target_col, seed=SEED, max_iter=MAX_ITER_NUIS):
    g_full = hgb_nuisance(max_iter=max_iter, seed=seed)
    g_full.fit(train[S_cols], train[target_col])
    m_full = {}
    for a in palancas:
        m = hgb_nuisance(max_iter=max_iter, seed=seed)
        m.fit(train[S_cols], train[a])
        m_full[a] = m
    return g_full, m_full


def plm_predict(test: pd.DataFrame, S_cols, palancas, theta, g_full, m_full) -> np.ndarray:
    pred = g_full.predict(test[S_cols]).astype(float)
    for j, a in enumerate(palancas):
        pred = pred + theta[j] * (test[a].to_numpy() - m_full[a].predict(test[S_cols]))
    return pred


def ajustar_plm(df_ext: pd.DataFrame, batches_dev: set, batches_lockbox: set, fase: str, target_col: str,
                 S_cols: list, palancas: list, pass_cols: list | None = None, seed=SEED,
                 max_iter=MAX_ITER_NUIS, n_boot=N_BOOT, target_es_kg=True, disponible_fn=None,
                 target_kg_col=None, skip_lockbox=False) -> dict:
    pass_cols = pass_cols or []
    feats_needed = list(dict.fromkeys(S_cols + palancas))
    sub = df_ext.loc[df_ext["fase_proceso"] == fase]
    cols = list(dict.fromkeys(feats_needed + pass_cols + [target_col, "Batch"]))
    need_dropna = list(dict.fromkeys(feats_needed + [target_col] + pass_cols))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=need_dropna).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=need_dropna).reset_index(drop=True)

    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats_needed], d_dev[target_col], d_dev["Batch"]):
        train, test = d_dev.iloc[tr], d_dev.iloc[te]
        theta, _, _ = plm_fit_theta(train, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)
        g_full, m_full = plm_full_models(train, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)
        oof[te] = plm_predict(test, S_cols, palancas, theta, g_full, m_full)

    theta_dev, y_res_dev, A_res_dev = plm_fit_theta(d_dev, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)
    g_full_dev, m_full_dev = plm_full_models(d_dev, S_cols, palancas, target_col, seed=seed, max_iter=max_iter)

    p_lb = None
    if not skip_lockbox:
        p_lb = plm_predict(d_lb, S_cols, palancas, theta_dev, g_full_dev, m_full_dev)

    g = d_dev["Batch"].to_numpy()
    uniq = np.unique(g)
    idx_by = {b: np.where(g == b)[0] for b in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
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

    res = {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
           "r2_oof": r2_score(d_dev[target_col], oof), "mae_oof": mean_absolute_error(d_dev[target_col], oof),
           "theta_tab": theta_tab, "d_dev": d_dev, "d_lb": d_lb, "oof": oof, "p_lb": p_lb,
           "g_full_dev": g_full_dev, "m_full_dev": m_full_dev, "theta_dev": theta_dev,
           "S_cols": S_cols, "palancas": palancas, "target_col": target_col}
    if not skip_lockbox:
        res["r2_lockbox"] = r2_score(d_lb[target_col], p_lb)
        res["mae_lockbox"] = mean_absolute_error(d_lb[target_col], p_lb)
    else:
        res["r2_lockbox"] = np.nan
        res["mae_lockbox"] = np.nan

    if target_es_kg:
        res["r2_oof_kg"] = res["r2_oof"]
        res["mae_oof_kg"] = res["mae_oof"]
        res["r2_lockbox_kg"] = res["r2_lockbox"]
        res["mae_lockbox_kg"] = res["mae_lockbox"]
    else:
        disp_dev = disponible_fn(d_dev)
        kg_oof_pred = oof * disp_dev
        kg_true_dev = d_dev[target_kg_col].to_numpy()
        res["r2_oof_kg"] = r2_score(kg_true_dev, kg_oof_pred)
        res["mae_oof_kg"] = mean_absolute_error(kg_true_dev, kg_oof_pred)
        if not skip_lockbox:
            disp_lb = disponible_fn(d_lb)
            kg_lb_pred = p_lb * disp_lb
            kg_true_lb = d_lb[target_kg_col].to_numpy()
            res["r2_lockbox_kg"] = r2_score(kg_true_lb, kg_lb_pred)
            res["mae_lockbox_kg"] = mean_absolute_error(kg_true_lb, kg_lb_pred)
        else:
            res["r2_lockbox_kg"] = np.nan
            res["mae_lockbox_kg"] = np.nan
    return res


# =============================================================================
# 2) HGB baseline (comparacion) -- v3 y forma E (fraccion)
# =============================================================================

def hgb_v3_model(max_iter=MAX_ITER_HGB, seed=SEED):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=max_iter,
                                          min_samples_leaf=15, l2_regularization=1.0, random_state=seed)


def ajustar_hgb_baseline(df_base, batches_dev, batches_lockbox, fase, target_col, feats):
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    cols = list(dict.fromkeys(feats + [target_col, "Batch"]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats + [target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + [target_col]).reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats], d_dev[target_col], d_dev["Batch"]):
        m = hgb_v3_model()
        m.fit(d_dev[feats].iloc[tr], d_dev[target_col].iloc[tr])
        oof[te] = m.predict(d_dev[feats].iloc[te])
    m_full = hgb_v3_model()
    m_full.fit(d_dev[feats], d_dev[target_col])
    p_lb = m_full.predict(d_lb[feats])
    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof), "mae_oof": mean_absolute_error(d_dev[target_col], oof),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb)}


def ajustar_hgb_fraccion(df_base, batches_dev, batches_lockbox, fase, feats):
    df = df_base.copy()
    frac_col, target_col = "frac_sn_extraido_escalon", "sn_extraido_est_kg"
    df["_disp"] = df["sn_inventario_escoria_est_kg_prev"] + df["feed_Sn_kgf"]
    sub = df.loc[df["fase_proceso"] == fase]
    cols = list(dict.fromkeys(feats + [frac_col, "_disp", "Batch", target_col]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), cols].dropna(subset=feats + [frac_col, target_col]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), cols].dropna(subset=feats + [frac_col, target_col]).reset_index(drop=True)
    oof_kg = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(d_dev[feats], d_dev[frac_col], d_dev["Batch"]):
        m = hgb_v3_model()
        m.fit(d_dev[feats].iloc[tr], d_dev[frac_col].iloc[tr])
        oof_kg[te] = m.predict(d_dev[feats].iloc[te]) * d_dev["_disp"].iloc[te].to_numpy()
    m_full = hgb_v3_model()
    m_full.fit(d_dev[feats], d_dev[frac_col])
    p_lb = m_full.predict(d_lb[feats]) * d_lb["_disp"].to_numpy()
    return {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
            "r2_oof": r2_score(d_dev[target_col], oof_kg), "mae_oof": mean_absolute_error(d_dev[target_col], oof_kg),
            "r2_lockbox": r2_score(d_lb[target_col], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target_col], p_lb)}


# =============================================================================
# 3) Signos teoricos (tabla ITERACION_4_diseno.md) para juzgar concordancia de theta
# =============================================================================

SIGNO_TEORICO = {
    ("R_sn", "_Cx_sn"): +1, ("R_sn", "tasa_feed_Carbon_kg_min"): +1,
    ("R_sn", "tasa_gn_nm3_min"): +1, ("R_sn", "exceso_o2_combustion_pct"): -1,
    ("R_sn", "duracion_plan_min"): +1,
    ("R_feo", "tasa_feed_Carbon_kg_min"): +1, ("R_feo", "_Cx_feo"): +1,
    ("R_feo", "tasa_gn_nm3_min"): +1, ("R_feo", "exceso_o2_combustion_pct"): -1,
    ("R_feo", "tasa_aire_nm3_min"): -1, ("R_feo", "duracion_plan_min"): +1,
    ("R_dT", "tasa_gn_nm3_min"): +1, ("R_dT", "tasa_o2_nm3_min"): 0,
    ("R_dT", "tasa_aire_nm3_min"): -1, ("R_dT", "tasa_feed_Carbon_kg_min"): -1,
    ("R_dT", "duracion_plan_min"): 0,
    ("F_sn", "relacion_C_Sn_carga"): +1, ("F_sn", "relacion_CaO_carga"): -1,
    ("F_sn", "carga_termica_relativa"): +1, ("F_sn", "exceso_o2_combustion_pct"): 0,
    ("F_sn", "tasa_feed_total_kg_min"): +1, ("F_sn", "duracion_plan_min"): +1,
    ("F_sn", "_relC_sq"): -1, ("F_sn", "_relCaO_sq"): -1, ("F_sn", "feed_Sn_kgf"): +1,
}


def filas_theta_desde_res(res, fase, target_col, variante, clave_teorica):
    filas = []
    for _, row in res["theta_tab"].iterrows():
        st = SIGNO_TEORICO.get((clave_teorica, row["palanca"]), None)
        signo_obs = float(np.sign(row["theta_unidad"]))
        if st is None:
            concuerda = np.nan
        elif st == 0:
            concuerda = np.nan
        else:
            concuerda = bool(signo_obs == st)
        filas.append({"fase": fase, "target": target_col, "variante": variante, "palanca": row["palanca"],
                      "theta_unidad": row["theta_unidad"], "theta_1sd": row["theta_1sd"],
                      "ci_lo": row["ci_lo_1sd"], "ci_hi": row["ci_hi_1sd"], "significativo": bool(row["significativo"]),
                      "signo_teorico": st, "concuerda": concuerda})
    return filas


def fila_validacion(fase, target_col, variante, res, n_estado, n_palancas):
    return {"fase": fase, "target": target_col, "variante": variante, "n_estado": n_estado, "n_palancas": n_palancas,
            "n_dev": res["n_dev"], "n_lockbox": res["n_lockbox"],
            "r2_oof": res["r2_oof"], "mae_oof": res["mae_oof"],
            "r2_lockbox": res["r2_lockbox"], "mae_lockbox": res["mae_lockbox"],
            "r2_oof_kg": res["r2_oof_kg"], "r2_lockbox_kg": res["r2_lockbox_kg"]}


# =============================================================================
# 4) Curva de parsimonia (Reduccion sn_kg y feo_kg): quitar 1 a 1 hasta 3 vars de estado
# =============================================================================

def curva_parsimonia(df_ext, batches_dev, batches_lockbox, fase, target_col, S_cols, palancas, max_removals=3):
    filas = []
    S_actual = list(S_cols)
    res = ajustar_plm(df_ext, batches_dev, batches_lockbox, fase, target_col, S_actual, palancas,
                       skip_lockbox=True, n_boot=5)
    filas.append({"fase": fase, "target": target_col, "n_estado": len(S_actual), "r2_oof": res["r2_oof"],
                  "variable_eliminada": None, "estado_restante": ";".join(S_actual)})
    for _ in range(max_removals):
        if len(S_actual) <= 1:
            break
        g_full_dev, m_full_dev, theta_dev, d_dev = res["g_full_dev"], res["m_full_dev"], res["theta_dev"], res["d_dev"]
        base_pred = plm_predict(d_dev, S_actual, palancas, theta_dev, g_full_dev, m_full_dev)
        base_r2 = r2_score(d_dev[target_col], base_pred)
        rng = np.random.default_rng(SEED)
        perdidas = {}
        for s in S_actual:
            dp = d_dev.copy()
            dp[s] = rng.permutation(dp[s].to_numpy())
            pred_perm = plm_predict(dp, S_actual, palancas, theta_dev, g_full_dev, m_full_dev)
            perdidas[s] = base_r2 - r2_score(d_dev[target_col], pred_perm)
        peor = min(perdidas, key=perdidas.get)
        S_actual = [s for s in S_actual if s != peor]
        res = ajustar_plm(df_ext, batches_dev, batches_lockbox, fase, target_col, S_actual, palancas,
                           skip_lockbox=True, n_boot=5)
        filas.append({"fase": fase, "target": target_col, "n_estado": len(S_actual), "r2_oof": res["r2_oof"],
                      "variable_eliminada": peor, "estado_restante": ";".join(S_actual)})
    return pd.DataFrame(filas)


# =============================================================================
# 5) Curvas de respuesta: 3 estados representativos de Reduccion, barrido de palancas, J = Sn - 0.30*FeO
# =============================================================================

def estado_representativo(df_dev_fase, avance_objetivo, cols_necesarias, ventana=0.03):
    d = df_dev_fase
    ventana_actual = ventana
    sel = (d["avance_reduccion_sn_prev"] - avance_objetivo).abs() <= ventana_actual
    intentos = 0
    while sel.sum() < 8 and intentos < 6:
        ventana_actual *= 1.7
        sel = (d["avance_reduccion_sn_prev"] - avance_objetivo).abs() <= ventana_actual
        intentos += 1
    rep = d.loc[sel, cols_necesarias].median()
    return rep, int(sel.sum()), ventana_actual


def curvas_respuesta_reduccion(res_sn, res_feo, df_ext, batches_dev):
    S_sn, pal_sn = res_sn["S_cols"], res_sn["palancas"]
    S_feo, pal_feo = res_feo["S_cols"], res_feo["palancas"]
    cols_all = list(dict.fromkeys(S_sn + pal_sn + S_feo + pal_feo +
                                   ["avance_reduccion_sn_prev", "sn_inventario_escoria_est_kg_prev"]))
    sub_dev = df_ext.loc[(df_ext["fase_proceso"] == "Reducción") & df_ext["Batch"].isin(batches_dev)]
    grids = {
        "tasa_feed_Carbon_kg_min": np.linspace(0.0, 70.0, 15),
        "tasa_gn_nm3_min": np.linspace(sub_dev["tasa_gn_nm3_min"].quantile(0.01), sub_dev["tasa_gn_nm3_min"].quantile(0.99), 15),
        "exceso_o2_combustion_pct": np.linspace(sub_dev["exceso_o2_combustion_pct"].quantile(0.01),
                                                 sub_dev["exceso_o2_combustion_pct"].quantile(0.99), 15),
    }
    filas = []
    resumen_optimo = []
    for nombre, avance in [("R0", 0.75), ("R1", 0.93), ("R3", 0.98)]:
        rep, n_rep, vent = estado_representativo(sub_dev, avance, cols_all)
        rep = rep.copy()
        rep["avance_reduccion_sn_prev"] = avance
        for var_barrida, grid in grids.items():
            test = pd.DataFrame([rep.to_dict()] * len(grid))
            test[var_barrida] = grid
            test["_Cx_sn"] = test["tasa_feed_Carbon_kg_min"] * test["sn_inventario_escoria_est_kg_prev"] / 1000.0
            test["_Cx_feo"] = test["tasa_feed_Carbon_kg_min"] * test["avance_reduccion_sn_prev"]
            sn_pred = plm_predict(test, S_sn, pal_sn, res_sn["theta_dev"], res_sn["g_full_dev"], res_sn["m_full_dev"])
            feo_pred = plm_predict(test, S_feo, pal_feo, res_feo["theta_dev"], res_feo["g_full_dev"], res_feo["m_full_dev"])
            J = sn_pred - 0.30 * feo_pred
            for i in range(len(grid)):
                filas.append({"estado": nombre, "avance_objetivo": avance, "n_rep": n_rep, "ventana": vent,
                              "variable_barrida": var_barrida, "valor": grid[i],
                              "sn_kg_pred": sn_pred[i], "feo_kg_pred": feo_pred[i], "J": J[i]})
            if var_barrida in ("tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min"):
                i_max = int(np.argmax(J))
                frontera = "borde_inf" if i_max == 0 else ("borde_sup" if i_max == len(grid) - 1 else "interior")
                resumen_optimo.append({"estado": nombre, "variable_barrida": var_barrida,
                                       "valor_optimo": grid[i_max], "J_optimo": J[i_max], "tipo": frontera})
    return pd.DataFrame(filas), pd.DataFrame(resumen_optimo)


def graficar_curvas_respuesta(df_curvas: pd.DataFrame):
    for estado in df_curvas["estado"].unique():
        sub = df_curvas[df_curvas["estado"] == estado]
        variables = sub["variable_barrida"].unique()
        fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4), squeeze=False)
        for j, var in enumerate(variables):
            ax = axes[0, j]
            s2 = sub[sub["variable_barrida"] == var].sort_values("valor")
            ax.plot(s2["valor"], s2["sn_kg_pred"], label="Sn extraido (kg)", color="tab:blue")
            ax.plot(s2["valor"], s2["feo_kg_pred"], label="FeO extraido (kg)", color="tab:orange")
            ax.plot(s2["valor"], s2["J"], label="J = Sn - 0.30*FeO", color="tab:green", linestyle="--")
            ax.set_xlabel(var)
            ax.set_ylabel("kg")
            ax.set_title(f"{estado}: {var}")
            ax.legend(fontsize=7)
            ax.axhline(0, color="grey", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"curva_respuesta_{estado}.png", dpi=110)
        plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    t0 = time.time()
    log("Cargando dataset v3 y split DEV/lockbox...")
    df, df_base, batches_dev, batches_lockbox = cargar_dev_lockbox()
    df_ext = preparar_columnas(df_base)
    log(f"df_base={len(df_base)} filas; DEV batches={len(batches_dev)}, lockbox batches={len(batches_lockbox)}")

    filas_val = []
    filas_theta = []

    # ---------------------------------------------------------------
    # Reduccion / sn_extraido_est_kg: 3 variantes de palanca de carbon
    # ---------------------------------------------------------------
    S_R = ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev",
           "ley_feo_escoria_pct_prev", "avance_reduccion_sn_prev", "temperatura_horno_celsius_prev",
           "orden_escalon_fase"]
    target_R_sn = "sn_extraido_est_kg"
    pal_v1 = ["_Cx_sn", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "duracion_plan_min"]
    pal_v2 = pal_v1 + ["tasa_feed_Carbon_kg_min"]
    pal_v3 = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "duracion_plan_min"]

    log("\n===== Reduccion / sn_extraido_est_kg =====")
    feats_v3_R_sn = ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev",
                      "ley_feo_escoria_pct_prev", "tasa_feed_Carbon_kg_min", "interaccion_C_x_Sn_prev",
                      "duracion_plan_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                      "temperatura_horno_celsius_prev"]
    res_hgb_R_sn = ajustar_hgb_baseline(df_ext, batches_dev, batches_lockbox, "Reducción", target_R_sn, feats_v3_R_sn)
    log(f"  HGB_v3 baseline: r2_oof={res_hgb_R_sn['r2_oof']:.4f} r2_lb={res_hgb_R_sn['r2_lockbox']:.4f}")
    filas_val.append({"fase": "Reducción", "target": target_R_sn, "variante": "HGB_v3_baseline",
                      "n_estado": len(feats_v3_R_sn), "n_palancas": 0, "n_dev": res_hgb_R_sn["n_dev"],
                      "n_lockbox": res_hgb_R_sn["n_lockbox"], "r2_oof": res_hgb_R_sn["r2_oof"],
                      "mae_oof": res_hgb_R_sn["mae_oof"], "r2_lockbox": res_hgb_R_sn["r2_lockbox"],
                      "mae_lockbox": res_hgb_R_sn["mae_lockbox"], "r2_oof_kg": res_hgb_R_sn["r2_oof"],
                      "r2_lockbox_kg": res_hgb_R_sn["r2_lockbox"]})

    res_R_sn = {}
    for nombre, pal in [("v1_Cx", pal_v1), ("v2_Cx_y_Carbon", pal_v2), ("v3_solo_Carbon", pal_v3)]:
        log(f"Ajustando PLM Reduccion/sn_kg variante {nombre} (S={len(S_R)}, palancas={pal})...")
        t1 = time.time()
        res = ajustar_plm(df_ext, batches_dev, batches_lockbox, "Reducción", target_R_sn, S_R, pal)
        log(f"  {nombre}: r2_oof={res['r2_oof']:.4f} r2_lb={res['r2_lockbox']:.4f} ({time.time()-t1:.1f}s)")
        filas_val.append(fila_validacion("Reducción", target_R_sn, nombre, res, len(S_R), len(pal)))
        filas_theta.extend(filas_theta_desde_res(res, "Reducción", target_R_sn, nombre, "R_sn"))
        res_R_sn[nombre] = res
    res_sn_v1 = res_R_sn["v1_Cx"]  # usado luego para curvas de respuesta

    # ---------------------------------------------------------------
    # Reduccion / feo_extraido_est_kg
    # ---------------------------------------------------------------
    log("\n===== Reduccion / feo_extraido_est_kg =====")
    S_R_feo = ["feo_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_feo_escoria_pct_prev",
               "ley_cao_escoria_pct_prev", "sn_inventario_escoria_est_kg_prev", "avance_reduccion_sn_prev",
               "temperatura_horno_celsius_prev"]
    pal_feo = ["tasa_feed_Carbon_kg_min", "_Cx_feo", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
               "tasa_aire_nm3_min", "duracion_plan_min"]
    target_R_feo = "feo_extraido_est_kg"
    feats_v3_R_feo = ["feo_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_feo_escoria_pct_prev",
                       "ley_cao_escoria_pct_prev", "ley_sn_escoria_pct_prev", "sn_inventario_escoria_est_kg_prev",
                       "tasa_feed_Carbon_kg_min", "interaccion_C_x_FeO_prev", "tasa_gn_nm3_min",
                       "exceso_o2_combustion_pct", "duracion_plan_min"]
    res_hgb_R_feo = ajustar_hgb_baseline(df_ext, batches_dev, batches_lockbox, "Reducción", target_R_feo, feats_v3_R_feo)
    log(f"  HGB_v3 baseline: r2_oof={res_hgb_R_feo['r2_oof']:.4f} r2_lb={res_hgb_R_feo['r2_lockbox']:.4f}")
    filas_val.append({"fase": "Reducción", "target": target_R_feo, "variante": "HGB_v3_baseline",
                      "n_estado": len(feats_v3_R_feo), "n_palancas": 0, "n_dev": res_hgb_R_feo["n_dev"],
                      "n_lockbox": res_hgb_R_feo["n_lockbox"], "r2_oof": res_hgb_R_feo["r2_oof"],
                      "mae_oof": res_hgb_R_feo["mae_oof"], "r2_lockbox": res_hgb_R_feo["r2_lockbox"],
                      "mae_lockbox": res_hgb_R_feo["mae_lockbox"], "r2_oof_kg": res_hgb_R_feo["r2_oof"],
                      "r2_lockbox_kg": res_hgb_R_feo["r2_lockbox"]})

    t1 = time.time()
    res_feo_v1 = ajustar_plm(df_ext, batches_dev, batches_lockbox, "Reducción", target_R_feo, S_R_feo, pal_feo)
    log(f"  PLM v1: r2_oof={res_feo_v1['r2_oof']:.4f} r2_lb={res_feo_v1['r2_lockbox']:.4f} ({time.time()-t1:.1f}s)")
    filas_val.append(fila_validacion("Reducción", target_R_feo, "v1", res_feo_v1, len(S_R_feo), len(pal_feo)))
    filas_theta.extend(filas_theta_desde_res(res_feo_v1, "Reducción", target_R_feo, "v1", "R_feo"))

    # ---------------------------------------------------------------
    # Reduccion / d_temperatura_horno_celsius (restriccion termica)
    # ---------------------------------------------------------------
    log("\n===== Reduccion / d_temperatura_horno_celsius =====")
    S_dT = ["temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "masa_escoria_est_kg_prev",
            "orden_escalon_fase", "grad_temperatura_horno_celsius_prev"]
    pal_dT = ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min", "duracion_plan_min"]
    target_dT = "d_temperatura_horno_celsius"
    t1 = time.time()
    res_dT = ajustar_plm(df_ext, batches_dev, batches_lockbox, "Reducción", target_dT, S_dT, pal_dT)
    log(f"  PLM v1: r2_oof={res_dT['r2_oof']:.4f} r2_lb={res_dT['r2_lockbox']:.4f} ({time.time()-t1:.1f}s)")
    filas_val.append(fila_validacion("Reducción", target_dT, "v1", res_dT, len(S_dT), len(pal_dT)))
    filas_theta.extend(filas_theta_desde_res(res_dT, "Reducción", target_dT, "v1", "R_dT"))

    # ---------------------------------------------------------------
    # Fusion / target A: frac_sn_extraido_escalon -> kg ; target B: sn_extraido_est_kg directo
    # ---------------------------------------------------------------
    log("\n===== Fusion / sn (frac y kg) =====")
    S_F = ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev",
           "ley_feo_escoria_pct_prev", "basicidad_B2_prev", "temperatura_horno_celsius_prev", "orden_escalon_fase"]
    pal_F_base = ["relacion_C_Sn_carga", "relacion_CaO_carga", "carga_termica_relativa",
                  "exceso_o2_combustion_pct", "tasa_feed_total_kg_min", "duracion_plan_min",
                  "_relC_sq", "_relCaO_sq"]
    pal_F_B = pal_F_base + ["feed_Sn_kgf"]

    feats_v3_F_sn = ["feed_Sn_kgf", "sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev",
                      "feo_inventario_escoria_est_kg_prev", "ley_sn_escoria_pct_prev", "interaccion_C_x_Sn_prev",
                      "tasa_feed_Carbon_kg_min", "relacion_CaO_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
                      "basicidad_B2_prev", "temperatura_horno_celsius_prev", "frac_carga_secundaria"]
    res_hgb_F = ajustar_hgb_baseline(df_ext, batches_dev, batches_lockbox, "Fusión", "sn_extraido_est_kg", feats_v3_F_sn)
    log(f"  HGB_v3 baseline: r2_oof={res_hgb_F['r2_oof']:.4f} r2_lb={res_hgb_F['r2_lockbox']:.4f}")
    filas_val.append({"fase": "Fusión", "target": "sn_extraido_est_kg", "variante": "HGB_v3_baseline",
                      "n_estado": len(feats_v3_F_sn), "n_palancas": 0, "n_dev": res_hgb_F["n_dev"],
                      "n_lockbox": res_hgb_F["n_lockbox"], "r2_oof": res_hgb_F["r2_oof"],
                      "mae_oof": res_hgb_F["mae_oof"], "r2_lockbox": res_hgb_F["r2_lockbox"],
                      "mae_lockbox": res_hgb_F["mae_lockbox"], "r2_oof_kg": res_hgb_F["r2_oof"],
                      "r2_lockbox_kg": res_hgb_F["r2_lockbox"]})

    res_hgb_Ffrac = ajustar_hgb_fraccion(df_ext, batches_dev, batches_lockbox, "Fusión", feats_v3_F_sn)
    log(f"  HGB_forma_E (fraccion->kg): r2_oof={res_hgb_Ffrac['r2_oof']:.4f} r2_lb={res_hgb_Ffrac['r2_lockbox']:.4f}")
    filas_val.append({"fase": "Fusión", "target": "sn_extraido_est_kg", "variante": "HGB_E_fraccion",
                      "n_estado": len(feats_v3_F_sn), "n_palancas": 0, "n_dev": res_hgb_Ffrac["n_dev"],
                      "n_lockbox": res_hgb_Ffrac["n_lockbox"], "r2_oof": res_hgb_Ffrac["r2_oof"],
                      "mae_oof": res_hgb_Ffrac["mae_oof"], "r2_lockbox": res_hgb_Ffrac["r2_lockbox"],
                      "mae_lockbox": res_hgb_Ffrac["mae_lockbox"], "r2_oof_kg": res_hgb_Ffrac["r2_oof"],
                      "r2_lockbox_kg": res_hgb_Ffrac["r2_lockbox"]})

    t1 = time.time()
    res_F_A = ajustar_plm(df_ext, batches_dev, batches_lockbox, "Fusión", "frac_sn_extraido_escalon", S_F, pal_F_base,
                           pass_cols=["feed_Sn_kgf", "sn_extraido_est_kg"], target_es_kg=False,
                           disponible_fn=lambda d: (d["sn_inventario_escoria_est_kg_prev"] + d["feed_Sn_kgf"]).to_numpy(),
                           target_kg_col="sn_extraido_est_kg")
    log(f"  PLM A (frac->kg): r2_oof_frac={res_F_A['r2_oof']:.4f} r2_oof_kg={res_F_A['r2_oof_kg']:.4f} "
        f"r2_lb_kg={res_F_A['r2_lockbox_kg']:.4f} ({time.time()-t1:.1f}s)")
    fila_a = fila_validacion("Fusión", "frac_sn_extraido_escalon", "A_frac_a_kg", res_F_A, len(S_F), len(pal_F_base))
    filas_val.append(fila_a)
    filas_theta.extend(filas_theta_desde_res(res_F_A, "Fusión", "frac_sn_extraido_escalon", "A_frac_a_kg", "F_sn"))

    t1 = time.time()
    res_F_B = ajustar_plm(df_ext, batches_dev, batches_lockbox, "Fusión", "sn_extraido_est_kg", S_F, pal_F_B)
    log(f"  PLM B (kg directo): r2_oof={res_F_B['r2_oof']:.4f} r2_lb={res_F_B['r2_lockbox']:.4f} ({time.time()-t1:.1f}s)")
    filas_val.append(fila_validacion("Fusión", "sn_extraido_est_kg", "B_kg_directo", res_F_B, len(S_F), len(pal_F_B)))
    filas_theta.extend(filas_theta_desde_res(res_F_B, "Fusión", "sn_extraido_est_kg", "B_kg_directo", "F_sn"))

    tabla_val = pd.DataFrame(filas_val)
    tabla_theta = pd.DataFrame(filas_theta)
    tabla_val.to_csv(EXP_DIR / "20_validacion_v4.csv", index=False)
    tabla_theta.to_csv(EXP_DIR / "20_theta_v4.csv", index=False)
    log(f"\nGuardado 20_validacion_v4.csv ({len(tabla_val)} filas) y 20_theta_v4.csv ({len(tabla_theta)} filas)")

    # ---------------------------------------------------------------
    # Curva de parsimonia: Reduccion sn_kg (variante v1) y feo_kg
    # ---------------------------------------------------------------
    log("\n===== Curva de parsimonia (Reduccion sn_kg y feo_kg) =====")
    t1 = time.time()
    curva_sn = curva_parsimonia(df_ext, batches_dev, batches_lockbox, "Reducción", target_R_sn, S_R, pal_v1)
    curva_sn["target_key"] = "sn_kg"
    log(f"  sn_kg: {curva_sn[['n_estado','r2_oof','variable_eliminada']].to_dict('records')} ({time.time()-t1:.1f}s)")
    t1 = time.time()
    curva_feo = curva_parsimonia(df_ext, batches_dev, batches_lockbox, "Reducción", target_R_feo, S_R_feo, pal_feo)
    curva_feo["target_key"] = "feo_kg"
    log(f"  feo_kg: {curva_feo[['n_estado','r2_oof','variable_eliminada']].to_dict('records')} ({time.time()-t1:.1f}s)")
    tabla_pars = pd.concat([curva_sn, curva_feo], ignore_index=True)
    tabla_pars.to_csv(EXP_DIR / "20_curva_parsimonia.csv", index=False)
    log(f"Guardado 20_curva_parsimonia.csv ({len(tabla_pars)} filas)")

    # ---------------------------------------------------------------
    # Curvas de respuesta: 3 estados representativos de Reduccion, J = Sn - 0.30*FeO
    # ---------------------------------------------------------------
    log("\n===== Curvas de respuesta (Reduccion, 3 estados) =====")
    t1 = time.time()
    df_curvas, df_optimo = curvas_respuesta_reduccion(res_sn_v1, res_feo_v1, df_ext, batches_dev)
    df_curvas.to_csv(EXP_DIR / "20_curvas_respuesta.csv", index=False)
    log(f"Guardado 20_curvas_respuesta.csv ({len(df_curvas)} filas) ({time.time()-t1:.1f}s)")
    log("Optimos de J por estado/variable:")
    for _, row in df_optimo.iterrows():
        log(f"  {row['estado']} / {row['variable_barrida']}: optimo en {row['valor_optimo']:.2f} "
            f"(J={row['J_optimo']:.1f} kg, {row['tipo']})")
    df_optimo.to_csv(EXP_DIR / "20_curvas_respuesta_optimo.csv", index=False)
    graficar_curvas_respuesta(df_curvas)
    log(f"Guardadas figuras en {FIGS_DIR}")

    log(f"\nTiempo total: {time.time() - t0:.1f} s")
    flush_log()


if __name__ == "__main__":
    main()
