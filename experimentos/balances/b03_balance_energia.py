"""B-03 -- balance de energia por escalon (features termicas y calor de reaccion aparente).

Ver experimentos/balances/ITERACION_6_diseno.md y experimentos/balances/bal_features.py (bloque C).
NO se modifica bal_features.py: las variantes de supuestos sobreescriben bal.SUPUESTOS[k]["valor"] y
recomputan con bal.construir_features_v6(pd.read_pickle(bal.CACHE)).

Secciones:
  1) Descomposicion del balance por fase / orden_escalon_fase; q_neto vs dT medido.
  2) Calibracion del balance por regresion fisica (NNLS con signo restringido + OLS libre),
     OOF GroupKFold(5) por batch, lockbox; teoria vs calibrado; variante LEAKAGE (extraccion real).
  3) Modelo de dT: PLM v4 (a) vs PLM v4 + features termicas b6 (b) vs balance calibrado / (m*Cp) (c)
     vs HGB con el set (b) (d). R2 OOF/lockbox por fase; theta con IC.
  4) Calor de reaccion aparente como estimador independiente de la extraccion (correlaciones).
  5) Sensibilidad: PCI, PERDIDAS, CP_ESCORIA, H_CARGA.

Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/balances/b03_balance_energia.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos" / "balances"))
sys.path.insert(0, str(RAIZ / "experimentos" / "search_targets"))

import bal_features as bal          # noqa: E402
import feature_engineering as fe    # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_prescriptivo as mp    # noqa: E402
import st_targets as st             # noqa: E402

FASES = ["Fusión", "Reducción"]


def log(*a):
    print(*a, flush=True)


def csv(df: pd.DataFrame, nombre: str):
    p = HERE / nombre
    df.to_csv(p, index=False)
    log(f"  -> {p.name} ({df.shape[0]}x{df.shape[1]})")


t_inicio = time.time()

# =============================================================================
# 0) Datos
# =============================================================================
log("=" * 90)
log("0) CARGA DE DATOS")
df = bal.cargar_df()
dev_b, lock_b = mp.split_dev_lockbox(df)
df["set"] = np.where(df["Batch"].isin(lock_b), "lockbox", "dev")
df_base = mp.dataset_base_modelo(df)  # excluye primer escalon del batch (sin _prev)
log(f"df: {df.shape}, DEV batches={len(dev_b)}, lockbox batches={len(lock_b)}")
log(f"df_base (sin primer escalon de batch): {df_base.shape}")

COLS_Q = ["b6_q_combustion_MJ", "b6_q_carbon_lanza_MJ", "b6_q_carga_MJ", "b6_q_gases_MJ",
          "b6_q_perdidas_MJ", "b6_q_neto_disponible_MJ", "b6_dT_teorico_sin_reaccion",
          "d_temperatura_horno_celsius"]

# =============================================================================
# 1) DESCOMPOSICION
# =============================================================================
log("=" * 90)
log("1) DESCOMPOSICION DEL BALANCE POR FASE Y ORDEN DE ESCALON")

filas_desc = []
for fase in FASES:
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    for orden, g in sub.groupby("orden_escalon_fase"):
        fila = {"fase": fase, "orden_escalon_fase": int(orden), "n": len(g)}
        for c in COLS_Q:
            fila[f"{c}_mediana"] = float(g[c].median())
            fila[f"{c}_p25"] = float(g[c].quantile(0.25))
            fila[f"{c}_p75"] = float(g[c].quantile(0.75))
        entradas = g["b6_q_combustion_MJ"] + g["b6_q_carbon_lanza_MJ"]
        salidas = g["b6_q_carga_MJ"] + g["b6_q_gases_MJ"] + g["b6_q_perdidas_MJ"]
        fila["participacion_combustion_pct"] = float((g["b6_q_combustion_MJ"] / entradas).median() * 100)
        fila["participacion_carbon_lanza_pct"] = float((g["b6_q_carbon_lanza_MJ"] / entradas).median() * 100)
        fila["participacion_carga_pct"] = float((g["b6_q_carga_MJ"] / salidas).median() * 100)
        fila["participacion_gases_pct"] = float((g["b6_q_gases_MJ"] / salidas).median() * 100)
        fila["participacion_perdidas_pct"] = float((g["b6_q_perdidas_MJ"] / salidas).median() * 100)
        filas_desc.append(fila)
tabla_desc = pd.DataFrame(filas_desc)
csv(tabla_desc, "b03_descomposicion.csv")

# resumen global por fase (todas las ordenes)
filas_glob = []
for fase in FASES:
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    fila = {"fase": fase, "n": len(sub)}
    for c in COLS_Q:
        fila[f"{c}_mediana"] = float(sub[c].median())
    filas_glob.append(fila)
tabla_glob = pd.DataFrame(filas_glob)
csv(tabla_glob, "b03_descomposicion_global.csv")
log(tabla_glob.round(1).to_string())

# correlacion q_neto (y dT_teorico_sin_reaccion) vs dT medido, en DEV, por fase
filas_corr_qneto = []
for fase in FASES:
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & (df_base["set"] == "dev")]
    for col in ["b6_q_neto_disponible_MJ", "b6_q_neto_por_min_MJ", "b6_dT_teorico_sin_reaccion"]:
        x, y = sub[col], sub["d_temperatura_horno_celsius"]
        m = x.notna() & y.notna()
        rho, p_rho = stats.spearmanr(x[m], y[m])
        r, p_r = stats.pearsonr(x[m], y[m])
        filas_corr_qneto.append({"fase": fase, "columna": col, "n": int(m.sum()),
                                 "spearman": float(rho), "p_spearman": float(p_rho),
                                 "pearson": float(r), "p_pearson": float(p_r)})
tabla_corr_qneto = pd.DataFrame(filas_corr_qneto)
csv(tabla_corr_qneto, "b03_correlacion_qneto_dT.csv")
log(tabla_corr_qneto.round(4).to_string())

log(f"[{time.time()-t_inicio:.0f}s] seccion 1 lista")

# =============================================================================
# 2) CALIBRACION DEL BALANCE POR REGRESION FISICA
# =============================================================================
log("=" * 90)
log("2) CALIBRACION DEL BALANCE (NNLS signo-restringido + OLS libre)")

NEED_BAL = ["masa_escoria_est_kg_prev", "d_temperatura_horno_celsius", "b6_q_combustion_MJ", "b6_c_fijo_kg",
            "feed_dross_Fe_kgh", "feed_total_kgh", "b6_v_gas_salida_nm3", "temperatura_horno_celsius_prev",
            "delta_tiempo", "sn_inventario_escoria_est_kg_prev", "feed_Sn_kgf", "feed_Fe_kgh",
            "sn_extraido_est_kg", "feo_extraido_est_kg", "Batch"]
# b6_feo_extraido_multi_kg (Reduccion, multi-trazador) es NaN en Fusion: se anexa aparte solo donde se usa
# (variante LEAKAGE), para no perder todas las filas de Fusion al hacer dropna sobre NEED_BAL.

REGRESORES = ["q_combustion", "c_fijo", "dross_fe", "feed_total", "q_gas_sensible", "delta_tiempo", "sn_disponible", "feed_fe_mineral"]
SIGNOS = np.array([+1, +1, +1, -1, -1, -1, -1, -1])
# teoria: coeficiente esperado (unidad natural) o rango, segun ITERACION_6_diseno / SUPUESTOS
TEORIA_TXT = {
    "q_combustion": "~1 (paso directo, MJ->MJ)",
    "c_fijo": "9.2 (a CO) - 32.8 (a CO2) MJ/kg C",
    "dross_fe": "0 - 1.5 MJ/kg dross (4.9 MJ/kgFe x 0.30 FE_MET_DROSS)",
    "feed_total": "-1.3 a -2.2 MJ/kg (H_CARGA)",
    "q_gas_sensible": "-1.35 a -2.10 (Cp gas kJ/Nm3K, aqui ya en MJ con /1000)",
    "delta_tiempo": "-117 (Fusion) / -158 (Reduccion) MJ/min (PERDIDAS)",
    "sn_disponible": "-(3.0 MJ/kg Sn) x fraccion media extraida/escalon",
    "feed_fe_mineral": "-1.519 MJ/kgFe x 0.55 FE_MINERAL = -0.84 MJ/kg mineral",
}


def _design(sub: pd.DataFrame, cp_escoria: float = None):
    cp = cp_escoria if cp_escoria is not None else bal.S("CP_ESCORIA_KJ_KG_K")
    y = sub["masa_escoria_est_kg_prev"] * cp * sub["d_temperatura_horno_celsius"] / 1000.0
    r1 = sub["b6_q_combustion_MJ"]
    r2 = sub["b6_c_fijo_kg"]
    r3 = sub["feed_dross_Fe_kgh"].fillna(0.0)
    r4 = sub["feed_total_kgh"].fillna(0.0)
    r5 = sub["b6_v_gas_salida_nm3"] * (sub["temperatura_horno_celsius_prev"] - 25.0) / 1000.0
    r6 = sub["delta_tiempo"]
    r7 = sub["sn_inventario_escoria_est_kg_prev"].fillna(0.0) + sub["feed_Sn_kgf"].fillna(0.0)
    r8 = sub["feed_Fe_kgh"].fillna(0.0)
    X = np.column_stack([r1, r2, r3, r4, r5, r6, r7, r8]).astype(float)
    return X, y.to_numpy(dtype=float)


def _design_leakage(sub: pd.DataFrame, usar_feo_multi: bool):
    """r7 (Sn disponible) reemplazado por la extraccion REAL (LEAKAGE): sn_extraido + feo_extraido."""
    X, y = _design(sub)
    sn_ext = sub["sn_extraido_est_kg"].clip(lower=0).fillna(0.0).to_numpy()
    if usar_feo_multi:
        feo_ext = sub["b6_feo_extraido_multi_kg"].clip(lower=0).fillna(0.0).to_numpy()
    else:
        feo_ext = sub["feo_extraido_est_kg"].clip(lower=0).fillna(0.0).to_numpy()
    X[:, 6] = bal.Q_SN_MJ_KG / 1.0 * sn_ext + bal.Q_FEO_MJ_KG / 1.0 * feo_ext  # ya en "kg-equivalente termico"
    # nota: se deja como una sola columna combinada (misma posicion que sn_disponible) para no
    # cambiar el numero de regresores; su signo teorico sigue siendo negativo (endotermico)
    return X, y


def fit_nnls(X, y, signos=SIGNOS):
    Xs = X * signos[None, :]
    coef_s, _ = nnls(Xs, y)
    return coef_s * signos


def fit_ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def oof_r2_calibrado(sub: pd.DataFrame, fit_fn, n_splits=5, design_fn=_design):
    X, y = design_fn(sub)
    grupos = sub["Batch"].to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, grupos):
        coef = fit_fn(X[tr], y[tr])
        oof[te] = X[te] @ coef
    return oof, float(r2_score(y, oof))


def bootstrap_ci_coef(sub: pd.DataFrame, fit_fn, n_boot=150, seed=SEED, design_fn=_design):
    X, y = design_fn(sub)
    grupos = sub["Batch"].to_numpy()
    uniq = np.unique(grupos)
    idx_by = {b: np.where(grupos == b)[0] for b in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
        try:
            boots.append(fit_fn(X[idx], y[idx]))
        except Exception:  # noqa: BLE001
            continue
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    return lo, hi


resultados_calib = {}
filas_coef, filas_r2 = [], []
for fase in FASES:
    sub_all = df_base.loc[df_base["fase_proceso"] == fase, NEED_BAL].dropna()
    # se conserva el indice original de df_base (NO reset_index) para poder escribir de vuelta en
    # df_base_cal (seccion 4) con .loc[sub.index, ...] sin desalinear filas.
    sub_dev = sub_all.loc[sub_all["Batch"].isin(dev_b)]
    sub_lb = sub_all.loc[sub_all["Batch"].isin(lock_b)]
    resultados_calib[fase] = dict(dev=sub_dev, lb=sub_lb)

    for nombre, fit_fn in [("nnls", fit_nnls), ("ols", fit_ols)]:
        coef = fit_fn(*_design(sub_dev))
        oof, r2_oof = oof_r2_calibrado(sub_dev, fit_fn)
        Xlb, ylb = _design(sub_lb)
        r2_lb = float(r2_score(ylb, Xlb @ coef))
        lo, hi = bootstrap_ci_coef(sub_dev, fit_fn, n_boot=150)
        for j, reg in enumerate(REGRESORES):
            filas_coef.append({"fase": fase, "metodo": nombre, "regresor": reg, "coef": float(coef[j]),
                               "ci_lo": float(lo[j]), "ci_hi": float(hi[j]), "signo_teorico": int(SIGNOS[j]),
                               "concuerda_signo": bool(np.sign(coef[j]) == SIGNOS[j]) if coef[j] != 0 else np.nan,
                               "teoria": TEORIA_TXT[reg]})
        filas_r2.append({"fase": fase, "metodo": nombre, "variante": "estandar", "n_dev": len(sub_dev),
                         "n_lockbox": len(sub_lb), "r2_oof": r2_oof, "r2_lockbox": r2_lb})

    # variante LEAKAGE (Reduccion: usa b6_feo_extraido_multi_kg si esta disponible; si no, feo_extraido_est_kg)
    usar_multi = fase == "Reducción"
    if usar_multi:
        multi_col = df_base.loc[df_base["fase_proceso"] == fase, ["b6_feo_extraido_multi_kg"]]
        sub_dev_leak = sub_dev.join(multi_col, how="left").dropna(subset=["b6_feo_extraido_multi_kg"])
        sub_lb_leak = sub_lb.join(multi_col, how="left").dropna(subset=["b6_feo_extraido_multi_kg"])
    else:
        sub_dev_leak, sub_lb_leak = sub_dev, sub_lb
    for nombre, fit_fn in [("nnls", fit_nnls), ("ols", fit_ols)]:
        design_fn = (lambda s, um=usar_multi: _design_leakage(s, um))
        coef = fit_fn(*design_fn(sub_dev_leak))
        oof, r2_oof = oof_r2_calibrado(sub_dev_leak, fit_fn, design_fn=design_fn)
        Xlb, ylb = design_fn(sub_lb_leak)
        r2_lb = float(r2_score(ylb, Xlb @ coef)) if len(ylb) else np.nan
        filas_r2.append({"fase": fase, "metodo": nombre, "variante": "leakage_extraccion_real",
                         "n_dev": len(sub_dev_leak), "n_lockbox": len(sub_lb_leak), "r2_oof": r2_oof, "r2_lockbox": r2_lb})

tabla_coef = pd.DataFrame(filas_coef)
tabla_r2_calib = pd.DataFrame(filas_r2)
csv(tabla_coef, "b03_calibracion_coeficientes.csv")
csv(tabla_r2_calib, "b03_calibracion_r2.csv")
log(tabla_r2_calib.round(4).to_string())
log(tabla_coef.round(4).to_string())
log(f"[{time.time()-t_inicio:.0f}s] seccion 2 lista")

# =============================================================================
# 3) MODELO DE dT: variantes (a) PLM v4, (b) PLM v4 + termicas, (c) fisico puro, (d) HGB set(b)
# =============================================================================
log("=" * 90)
log("3) MODELO DE dT -- variantes")

FEATS_TERMICAS_B = ["b6_dT_teorico_sin_reaccion", "b6_q_neto_por_min_MJ", "b6_q_gases_MJ", "b6_margen_termico_MJ"]
assert all(c in bal.SEGURAS_V6 for c in FEATS_TERMICAS_B), "features termicas no clasificadas como seguras"

SIGNO_TERMICAS_B = {"b6_dT_teorico_sin_reaccion": +1, "b6_q_neto_por_min_MJ": +1, "b6_q_gases_MJ": -1, "b6_margen_termico_MJ": +1}


def _hgb(seed=SEED, max_iter=150):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=max_iter, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=seed)


def evaluar_plm_dT(df_in: pd.DataFrame, fase: str, estado: list, palancas: list, target: str,
                    n_splits=5, n_boot=150, seed=SEED):
    sub = df_in.loc[df_in["fase_proceso"] == fase]
    need = list(dict.fromkeys(estado + palancas + [target, "Batch"]))
    d_dev = sub.loc[sub["Batch"].isin(dev_b), need].dropna().reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(lock_b), need].dropna().reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev[estado], d_dev[target], d_dev["Batch"]):
        m = mp4.ModeloPLM(estado, palancas, target).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
        oof[te] = m.predict(d_dev.iloc[te])
    m_full = mp4.ModeloPLM(estado, palancas, target).fit(d_dev, n_boot=n_boot, seed=seed)
    p_lb = m_full.predict(d_lb) if len(d_lb) else np.array([])
    r2_oof = float(r2_score(d_dev[target], oof))
    r2_lb = float(r2_score(d_lb[target], p_lb)) if len(d_lb) else np.nan
    return dict(r2_oof=r2_oof, r2_lockbox=r2_lb, n_dev=len(d_dev), n_lockbox=len(d_lb)), m_full


def evaluar_hgb_dT(df_in: pd.DataFrame, fase: str, features: list, target: str, n_splits=5, seed=SEED):
    sub = df_in.loc[df_in["fase_proceso"] == fase]
    need = list(dict.fromkeys(features + [target, "Batch"]))
    d_dev = sub.loc[sub["Batch"].isin(dev_b), need].dropna().reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(lock_b), need].dropna().reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev[features], d_dev[target], d_dev["Batch"]):
        mdl = _hgb(seed).fit(d_dev[features].iloc[tr], d_dev[target].iloc[tr])
        oof[te] = mdl.predict(d_dev[features].iloc[te])
    mdl_full = _hgb(seed).fit(d_dev[features], d_dev[target])
    p_lb = mdl_full.predict(d_lb[features]) if len(d_lb) else np.array([])
    r2_oof = float(r2_score(d_dev[target], oof))
    r2_lb = float(r2_score(d_lb[target], p_lb)) if len(d_lb) else np.nan
    return dict(r2_oof=r2_oof, r2_lockbox=r2_lb, n_dev=len(d_dev), n_lockbox=len(d_lb))


def variante_c_fisica(df_in: pd.DataFrame, fase: str, n_splits=5):
    """dT_pred = balance_calibrado(NNLS, DEV) / (masa_prev*Cp/1000). OOF por fold + lockbox."""
    sub = df_in.loc[df_in["fase_proceso"] == fase, NEED_BAL].dropna()
    d_dev = sub.loc[sub["Batch"].isin(dev_b)].reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(lock_b)].reset_index(drop=True)
    X, y = _design(d_dev)
    cap = d_dev["masa_escoria_est_kg_prev"] * bal.S("CP_ESCORIA_KJ_KG_K") / 1000.0
    grupos = d_dev["Batch"].to_numpy()
    oof_dT = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, grupos):
        coef = fit_nnls(X[tr], y[tr])
        oof_dT[te] = (X[te] @ coef) / cap.to_numpy()[te]
    coef_full = fit_nnls(X, y)
    Xlb, ylb = _design(d_lb)
    cap_lb = d_lb["masa_escoria_est_kg_prev"] * bal.S("CP_ESCORIA_KJ_KG_K") / 1000.0
    dT_pred_lb = (Xlb @ coef_full) / cap_lb.to_numpy()
    r2_oof = float(r2_score(d_dev["d_temperatura_horno_celsius"], oof_dT))
    r2_lb = float(r2_score(d_lb["d_temperatura_horno_celsius"], dT_pred_lb)) if len(d_lb) else np.nan
    return dict(r2_oof=r2_oof, r2_lockbox=r2_lb, n_dev=len(d_dev), n_lockbox=len(d_lb))


filas_dT, filas_theta = [], []
modelos_ab = {}
for fase in FASES:
    estado = mp4.ESTADO_V4[(fase, "dT")]
    palancas_a = mp4.PALANCAS_V4[(fase, "dT")]
    palancas_b = palancas_a + FEATS_TERMICAS_B
    target = "d_temperatura_horno_celsius"

    res_a, m_a = evaluar_plm_dT(df_base, fase, estado, palancas_a, target)
    filas_dT.append({"fase": fase, "variante": "a_PLM_v4", **res_a})
    signo_a = mp4.SIGNO_TEORICO_V4[(fase, "dT")]
    t_a = m_a.tabla_theta(signo_a).reset_index()
    t_a.insert(0, "variante", "a_PLM_v4"); t_a.insert(0, "fase", fase)
    filas_theta.append(t_a)

    res_b, m_b = evaluar_plm_dT(df_base, fase, estado, palancas_b, target)
    filas_dT.append({"fase": fase, "variante": "b_PLM_v4_termicas", **res_b})
    signo_b = dict(signo_a, **SIGNO_TERMICAS_B)
    t_b = m_b.tabla_theta(signo_b).reset_index()
    t_b.insert(0, "variante", "b_PLM_v4_termicas"); t_b.insert(0, "fase", fase)
    filas_theta.append(t_b)
    modelos_ab[fase] = (m_a, m_b)

    res_c = variante_c_fisica(df_base, fase)
    filas_dT.append({"fase": fase, "variante": "c_fisico_calibrado", **res_c})

    feats_d = list(dict.fromkeys(estado + palancas_b))
    res_d = evaluar_hgb_dT(df_base, fase, feats_d, target)
    filas_dT.append({"fase": fase, "variante": "d_HGB_set_b", **res_d})

    log(f"  {fase}: a={res_a['r2_oof']:.3f}/{res_a['r2_lockbox']:.3f}  b={res_b['r2_oof']:.3f}/{res_b['r2_lockbox']:.3f}  "
        f"c={res_c['r2_oof']:.3f}/{res_c['r2_lockbox']:.3f}  d={res_d['r2_oof']:.3f}/{res_d['r2_lockbox']:.3f}")

tabla_dT = pd.DataFrame(filas_dT)
tabla_theta_dT = pd.concat(filas_theta, ignore_index=True)
csv(tabla_dT, "b03_dT_variantes.csv")
csv(tabla_theta_dT, "b03_theta_dT.csv")
log(f"[{time.time()-t_inicio:.0f}s] seccion 3 lista")

# =============================================================================
# 4) CALOR DE REACCION APARENTE COMO ESTIMADOR DE LA EXTRACCION
# =============================================================================
log("=" * 90)
log("4) CALOR DE REACCION APARENTE vs TRAZADOR")

OTRAS_COLS = ["b6_q_reaccion_trazador_MJ", "sn_extraido_est_kg", "feo_extraido_est_kg", "b6_feo_extraido_multi_kg"]


def tabla_correlacion_reaccion(df_in: pd.DataFrame, col_aparente: str, etiqueta: str) -> pd.DataFrame:
    filas = []
    for fase in FASES:
        sub = df_in.loc[(df_in["fase_proceso"] == fase) & (df_in["set"] == "dev")]
        for otra in OTRAS_COLS:
            x, y = sub[col_aparente], sub[otra]
            m = x.notna() & y.notna()
            if m.sum() < 10:
                filas.append({"fase": fase, "orden_escalon_fase": "todos", "aparente": etiqueta, "variable": otra, "n": int(m.sum())})
                continue
            rho, p_rho = stats.spearmanr(x[m], y[m])
            r, p_r = stats.pearsonr(x[m], y[m])
            filas.append({"fase": fase, "orden_escalon_fase": "todos", "aparente": etiqueta, "variable": otra,
                         "n": int(m.sum()), "spearman": float(rho), "p_spearman": float(p_rho),
                         "pearson": float(r), "p_pearson": float(p_r)})
            for orden, g in sub.groupby("orden_escalon_fase"):
                xg, yg = g[col_aparente], g[otra]
                mg = xg.notna() & yg.notna()
                if mg.sum() < 15:
                    continue
                rho2, p2 = stats.spearmanr(xg[mg], yg[mg])
                filas.append({"fase": fase, "orden_escalon_fase": int(orden), "aparente": etiqueta, "variable": otra,
                             "n": int(mg.sum()), "spearman": float(rho2), "p_spearman": float(p2),
                             "pearson": np.nan, "p_pearson": np.nan})
    return pd.DataFrame(filas)


tabla_corr_raw = tabla_correlacion_reaccion(df_base, "b6_q_reaccion_aparente_MJ", "raw")

# version calibrada: q_neto_calibrado - cap_bano*dT_medido, sin el termino de Sn/FeO (r7)
df_base_cal = df_base.copy()
df_base_cal["b6_q_reaccion_aparente_cal_MJ"] = np.nan
for fase in FASES:
    sub_dev = resultados_calib[fase]["dev"]
    coef = fit_nnls(*_design(sub_dev))
    for nombre, sub in [("dev", resultados_calib[fase]["dev"]), ("lb", resultados_calib[fase]["lb"])]:
        X, y = _design(sub)
        pred_no_r7 = X[:, [0, 1, 2, 3, 4, 5, 7]] @ coef[[0, 1, 2, 3, 4, 5, 7]]
        q_ap_cal = y - pred_no_r7
        df_base_cal.loc[sub.index, "b6_q_reaccion_aparente_cal_MJ"] = q_ap_cal
tabla_corr_cal = tabla_correlacion_reaccion(df_base_cal, "b6_q_reaccion_aparente_cal_MJ", "calibrado_sin_r7")

tabla_corr_reaccion = pd.concat([tabla_corr_raw, tabla_corr_cal], ignore_index=True)
csv(tabla_corr_reaccion, "b03_correlacion_reaccion.csv")
resumen4 = tabla_corr_reaccion[tabla_corr_reaccion["orden_escalon_fase"] == "todos"]
log(resumen4.round(4).to_string())

# umbral de decision (tarea 4): |rho| >= 0.3 en Reduccion (overall, raw o calibrado) con cualquier
# columna trazadora -> construir estimador combinado; si no, documentar el techo.
red_rows = resumen4[(resumen4["fase"] == "Reducción")]
max_abs_rho_red = red_rows["spearman"].abs().max()
log(f"max |spearman| en Reduccion (overall, raw+calibrado): {max_abs_rho_red:.3f}")
UMBRAL_COMBINADO = 0.30
construir_combinado = bool(max_abs_rho_red >= UMBRAL_COMBINADO)
log(f"construir estimador combinado (umbral {UMBRAL_COMBINADO}): {construir_combinado}")

resultado_combinado = None
if construir_combinado:
    # Monte Carlo del ruido del trazador (aplicado sobre feo_extraido_est_kg, Reduccion) + varianza del
    # residuo de la calibracion como ruido del estimador energetico; combinacion por inversa de varianza.
    rng = np.random.default_rng(SEED)
    sub = df_base.loc[(df_base["fase_proceso"] == "Reducción") & (df_base["set"] == "dev")].dropna(
        subset=["ley_cao_escoria_pct", "ley_sn_escoria_pct", "ley_feo_escoria_pct", "masa_escoria_est_kg_prev",
                "sn_extraido_est_kg", "b6_q_reaccion_aparente_MJ"]).reset_index(drop=True)
    n_mc = 300
    sn_mc = np.zeros((n_mc, len(sub)))
    for k in range(n_mc):
        d_cao = rng.normal(0, 2.0, len(sub))
        d_sn = rng.normal(0, 0.05, len(sub))
        masa_pert = sub["masa_escoria_est_kg_prev"] * (1 + d_cao / 100.0)
        sn_pert = masa_pert * (sub["ley_sn_escoria_pct"] + d_sn) / 100.0
        sn_mc[k] = sn_pert
    var_trazador = np.nanvar(sn_mc, axis=0)
    coef = fit_nnls(*_design(resultados_calib["Reducción"]["dev"]))
    Xd, yd = _design(resultados_calib["Reducción"]["dev"])
    oof_bal, _ = oof_r2_calibrado(resultados_calib["Reducción"]["dev"], fit_nnls)
    var_energia_mj = float(np.nanvar(yd - oof_bal))
    var_energia_kg = var_energia_mj / (bal.Q_SN_MJ_KG ** 2)  # a unidades de kg Sn
    w_traz = 1.0 / np.maximum(var_trazador, 1e-6)
    w_energ = 1.0 / max(var_energia_kg, 1e-6)
    sn_energia_kg = (sub["b6_q_reaccion_aparente_MJ"] - bal.Q_FEO_MJ_KG * sub["feo_extraido_est_kg"].clip(lower=0).fillna(0)) / bal.Q_SN_MJ_KG
    sn_comb = (w_traz * sub["sn_extraido_est_kg"] + w_energ * sn_energia_kg) / (w_traz + w_energ)
    resultado_combinado = dict(var_trazador_medio=float(np.mean(var_trazador)), var_energia_kg=var_energia_kg,
                               peso_medio_trazador=float(np.mean(w_traz / (w_traz + w_energ))))
    log(f"combinado: {resultado_combinado}")
else:
    log("Reduccion: |rho| < 0.30 para todas las variables trazadoras -> NO se construye el estimador combinado.")
    log("Techo: el balance de energia por escalon no resuelve la extraccion de forma independiente")
    log("(pocas variables termicas medidas, T de cara interior != T de bano, post-combustion no medida).")

log(f"[{time.time()-t_inicio:.0f}s] seccion 4 lista")

# =============================================================================
# 5) SENSIBILIDAD
# =============================================================================
log("=" * 90)
log("5) SENSIBILIDAD (PCI, PERDIDAS, CP_ESCORIA, H_CARGA)")

DH_CH4_CO2_BASE, DH_CH4_CO_BASE = bal.DH_CH4_CO2, bal.DH_CH4_CO  # -802.3, -519.3 kJ/mol => PCI base 35.8


def _recomputar(supuestos_override: dict | None) -> pd.DataFrame:
    """Recomputa el bloque v6 con SUPUESTOS sobreescritos (PCI se maneja aparte, ver mas abajo)."""
    orig = {k: bal.SUPUESTOS[k]["valor"] for k in (supuestos_override or {})}
    for k, v in (supuestos_override or {}).items():
        bal.SUPUESTOS[k]["valor"] = v
    try:
        d = bal.construir_features_v6(pd.read_pickle(bal.CACHE))
    finally:
        for k, v in orig.items():
            bal.SUPUESTOS[k]["valor"] = v
    d["set"] = np.where(d["Batch"].isin(lock_b), "lockbox", "dev")
    return mp.dataset_base_modelo(d)


def _escalar_pci(d: pd.DataFrame, pci_nuevo: float) -> pd.DataFrame:
    """PCI_GN_MJ_NM3 no esta cableado en agregar_balance_energia (usa entalpias CH4 fijas,
    equivalentes a PCI=35.8 MJ/Nm3 -802.3/22.414). Se aproxima la sensibilidad reescalando
    linealmente q_combustion (y las columnas derivadas) por pci_nuevo/35.8, sin tocar bal_features.py."""
    d = d.copy()
    factor = pci_nuevo / 35.8
    d["b6_q_combustion_MJ"] = d["b6_q_combustion_MJ"] * factor
    q_neto = d["b6_q_combustion_MJ"] + d["b6_q_carbon_lanza_MJ"] - d["b6_q_carga_MJ"] - d["b6_q_gases_MJ"] - d["b6_q_perdidas_MJ"]
    d["b6_q_neto_disponible_MJ"] = q_neto
    d["b6_q_neto_por_min_MJ"] = q_neto / d["delta_tiempo"]
    cap = d["masa_escoria_est_kg_prev"] * bal.S("CP_ESCORIA_KJ_KG_K") / 1000.0
    d["b6_dT_teorico_sin_reaccion"] = q_neto / cap.where(cap > 1)
    d["b6_margen_termico_MJ"] = q_neto - d["b6_q_reduccion_max_MJ"]
    d["b6_q_reaccion_aparente_MJ"] = q_neto - cap * d["d_temperatura_horno_celsius"]
    return d


def _r2_variante_b_y_corr(d: pd.DataFrame) -> dict:
    out = {}
    for fase in FASES:
        estado = mp4.ESTADO_V4[(fase, "dT")]
        palancas_b = mp4.PALANCAS_V4[(fase, "dT")] + FEATS_TERMICAS_B
        res, _ = evaluar_plm_dT(d, fase, estado, palancas_b, "d_temperatura_horno_celsius", n_boot=0)
        out[f"r2_oof_dT_b_{fase}"] = res["r2_oof"]
        out[f"r2_lockbox_dT_b_{fase}"] = res["r2_lockbox"]
        sub = d.loc[(d["fase_proceso"] == fase) & (d["set"] == "dev")]
        x, y = sub["b6_q_reaccion_aparente_MJ"], sub["sn_extraido_est_kg"]
        m = x.notna() & y.notna()
        rho, _ = stats.spearmanr(x[m], y[m]) if m.sum() > 10 else (np.nan, np.nan)
        out[f"rho_reaccion_sn_{fase}"] = float(rho)
    return out


filas_sens = []
base_case = df_base
r0 = _r2_variante_b_y_corr(base_case)
filas_sens.append({"parametro": "base", "valor": "PCI=35.8/PERD=100%/CP=1.2/HCARGA=1.7", **r0})

for pci in [33.0, 38.0]:
    d = _escalar_pci(df_base.copy(), pci)
    r = _r2_variante_b_y_corr(d)
    filas_sens.append({"parametro": "PCI_GN_MJ_NM3", "valor": pci, **r})

for factor in [0.5, 1.5]:
    perd_orig = dict(bal.SUPUESTOS["PERDIDAS_MJ_MIN"]["valor"])
    nuevo = {k: v * factor for k, v in perd_orig.items()}
    d = _recomputar({"PERDIDAS_MJ_MIN": nuevo})
    r = _r2_variante_b_y_corr(d)
    filas_sens.append({"parametro": "PERDIDAS_MJ_MIN_factor", "valor": factor, **r})

for cp in [1.0, 1.4]:
    d = _recomputar({"CP_ESCORIA_KJ_KG_K": cp})
    r = _r2_variante_b_y_corr(d)
    filas_sens.append({"parametro": "CP_ESCORIA_KJ_KG_K", "valor": cp, **r})

for hc in [1.3, 2.2]:
    d = _recomputar({"H_CARGA_MJ_KG": hc})
    r = _r2_variante_b_y_corr(d)
    filas_sens.append({"parametro": "H_CARGA_MJ_KG", "valor": hc, **r})

tabla_sens = pd.DataFrame(filas_sens)
csv(tabla_sens, "b03_sensibilidad.csv")
log(tabla_sens.round(4).to_string())

log(f"[{time.time()-t_inicio:.0f}s] TODO listo, total {time.time()-t_inicio:.0f}s")
