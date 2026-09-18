"""E11-03 (H3) -- pasos 3-7: anclaje por el canal de la accion, canales, sobre-identificacion, comparacion de
objetivos por CV anidada y moderacion GN/carbon x Sn disponible. Lee experimentos/v9/e11_03_tabla_batch.pkl
(generada por e11_03_construir.py). .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3,
primer plano, sin joblib.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v9", RAIZ / "experimentos" / "v8"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import v9_lib as L  # noqa: E402

SALIDA = RAIZ / "experimentos" / "v9"
SEED = 42
KG_SN_POR_PP = 512.0   # v8: ~kg de Sn a metal por pp de KPI (sanity)

t = pd.read_pickle(SALIDA / "e11_03_tabla_batch.pkl")
CONTROLES_FULL = list(L.CONTROLES)
CONTROLES_SIN_ESP = [c for c in CONTROLES_FULL if c != "espesor_ladrillo_norm_mm"]
DEV, LB = t[t.es_dev].copy(), t[~t.es_dev].copy()
KPI = L.KPI


def controles_de(sample_name: str) -> list[str]:
    return CONTROLES_SIN_ESP if sample_name == "LOCKBOX" else CONTROLES_FULL


# --------------------------------------------------------------------------- utilidades
def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def regresion_ancla(d: pd.DataFrame, x_cols: list[str], controles: list[str], y_col: str = KPI,
                    n_boot: int = 1000, seed: int = SEED) -> dict:
    dd = d.dropna(subset=[y_col] + x_cols + controles)
    X = sm.add_constant(dd[x_cols + controles], has_constant="add")
    fit = sm.OLS(dd[y_col], X).fit(cov_type="HC3")
    Xn = X.to_numpy(); y = dd[y_col].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(dd)
    boots = np.zeros((n_boot, len(x_cols)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        beta = _ols(y[idx], Xn[idx])
        boots[b] = beta[1:1 + len(x_cols)]
    out = {"n": n, "params": {c: float(fit.params[c]) for c in x_cols}, "se": {c: float(fit.bse[c]) for c in x_cols},
           "t": {c: float(fit.tvalues[c]) for c in x_cols}, "p": {c: float(fit.pvalues[c]) for c in x_cols},
           "boots": boots, "x_cols": x_cols, "r2": float(fit.rsquared)}
    return out


def ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


# --------------------------------------------------------------------------- paso 3: anclaje por accion
def paso3():
    filas = []
    for version, x_cols in (("LOG_ident", ["D_sn_log_ident", "D_feo_log_ident"]),
                            ("LOG_all", ["D_sn_log_all", "D_feo_log_all"]),
                            ("KG_ident", ["D_sn_kg_ident", "D_feo_kg_ident"]),
                            ("KG_all", ["D_sn_kg_all", "D_feo_kg_all"])):
        for nombre, d in (("DEV", DEV), ("LOCKBOX", LB), ("TOTAL", t)):
            r = regresion_ancla(d, x_cols, controles_de(nombre), n_boot=1000)
            k_sn, k_feo = r["params"][x_cols[0]], r["params"][x_cols[1]]
            b_sn, b_feo = r["boots"][:, 0], r["boots"][:, 1]
            if version.startswith("LOG"):
                w = k_feo / k_sn if k_sn else np.nan
                w_boot = np.divide(b_feo, b_sn, out=np.full_like(b_feo, np.nan), where=b_sn != 0)
                lo, hi = ci(w_boot)
                pp_t_sn = pp_t_feo = np.nan
            else:
                w = -k_feo / k_sn if k_sn else np.nan
                w_boot = np.divide(-b_feo, b_sn, out=np.full_like(b_feo, np.nan), where=b_sn != 0)
                lo, hi = ci(w_boot)
                pp_t_sn, pp_t_feo = 1000 * k_sn, 1000 * k_feo   # pp de KPI por tonelada
            filas.append(dict(version=version, muestra=nombre, n=r["n"], K_Sn=k_sn, K_Sn_t=r["t"][x_cols[0]],
                              K_Sn_p=r["p"][x_cols[0]], K_FeO=k_feo, K_FeO_t=r["t"][x_cols[1]], K_FeO_p=r["p"][x_cols[1]],
                              w_o_lambda=w, ci_lo=lo, ci_hi=hi, pp_por_t_Sn=pp_t_sn, pp_por_t_FeO=pp_t_feo, r2=r["r2"]))
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA / "e11_03_anclaje.csv", index=False)
    print(df.round(4).to_string())
    return df


# --------------------------------------------------------------------------- paso 4: canales
def paso4():
    filas = []
    canales = ["f_metal", "f_dross", "f_polvo", "sn_perdido_escoria_frac"]
    for version, x_cols in (("LOG_ident", ["D_sn_log_ident", "D_feo_log_ident"]),
                            ("KG_ident", ["D_sn_kg_ident", "D_feo_kg_ident"])):
        for nombre, d in (("DEV", DEV), ("TOTAL", t)):
            for canal in canales:
                dd = d.copy(); dd[canal] = 100 * dd[canal]
                r = regresion_ancla(dd, x_cols, controles_de(nombre), y_col=canal, n_boot=200)
                filas.append(dict(version=version, muestra=nombre, canal=canal, n=r["n"],
                                  coef_Sn=r["params"][x_cols[0]], p_Sn=r["p"][x_cols[0]],
                                  coef_FeO=r["params"][x_cols[1]], p_FeO=r["p"][x_cols[1]]))
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA / "e11_03_canales.csv", index=False)
    print(df.round(4).to_string())
    return df


# --------------------------------------------------------------------------- paso 5: sobre-identificacion
def paso5():
    filas = []
    for version, x_cols in (("LOG_ident", ["D_sn_log_ident", "D_feo_log_ident"]),
                            ("KG_ident", ["D_sn_kg_ident", "D_feo_kg_ident"])):
        for nombre, d in (("DEV", DEV), ("TOTAL", t)):
            extra = ["x_R_gn", "x_R_carbon"]
            controles = controles_de(nombre)
            dd = d.dropna(subset=[KPI] + x_cols + extra + controles)
            X = sm.add_constant(dd[x_cols + extra + controles], has_constant="add")
            fit = sm.OLS(dd[KPI], X).fit(cov_type="HC3")
            wald = fit.wald_test(np.eye(len(X.columns))[[X.columns.get_loc(c) for c in extra]], scalar=True)
            filas.append(dict(version=version, muestra=nombre, n=len(dd), F=float(wald.statistic), p=float(wald.pvalue),
                              coef_gn=float(fit.params["x_R_gn"]), p_gn=float(fit.pvalues["x_R_gn"]),
                              coef_carbon=float(fit.params["x_R_carbon"]), p_carbon=float(fit.pvalues["x_R_carbon"]),
                              coef_Sn_con_extra=float(fit.params[x_cols[0]]), coef_FeO_con_extra=float(fit.params[x_cols[1]])))
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA / "e11_03_sobreidentificacion.csv", index=False)
    print(df.round(4).to_string())
    return df


# --------------------------------------------------------------------------- paso 6: comparacion de objetivos (CV anidada)
def evaluar_score_fijo(d: pd.DataFrame, score_col: str, controles: list[str], kpi: str = KPI, n_perm: int = 2000, seed: int = SEED) -> dict:
    """Mismo contraste final de `L.prueba_anidada` (pendiente HC3 + controles, Spearman parcial, permutacion) pero
    para un puntaje SIN parametros libres (objetivo (a): K, w de v8 fijos) -> no hace falta anidar."""
    dd = d.dropna(subset=[kpi, score_col] + controles).copy()
    fit = L.ols_hc3(dd[kpi], dd[[score_col] + controles])
    tval = float(fit.tvalues[score_col])
    ry = L.ols_hc3(dd[kpi], dd[controles]).resid
    rs = L.ols_hc3(dd[score_col], dd[controles]).resid
    rho, p_rho = stats.spearmanr(ry, rs)
    rng = np.random.default_rng(seed)
    r_obs = float(np.corrcoef(rs, ry)[0, 1])
    tp = [np.corrcoef(rng.permutation(rs.to_numpy()), ry)[0, 1] for _ in range(n_perm)]
    p_perm = float((np.sum(np.array(tp) >= r_obs) + 1) / (n_perm + 1))
    q = pd.qcut(dd[score_col], 3, labels=False, duplicates="drop")
    ter = ry.groupby(q).mean()
    return {"n": len(dd), "pendiente": float(fit.params[score_col]), "t": tval, "p_unilateral": float(stats.norm.sf(tval)),
           "spearman_parcial": float(rho), "p_perm_unilateral": p_perm,
           "kpi_tercio_bajo": float(ter.iloc[0]), "kpi_tercio_alto": float(ter.iloc[-1])}


def paso6():
    t2 = t.copy()
    t2["score_a"] = t2["agg_sn"] + 5.93 * t2["agg_feo"]   # (a) J log v8, K/w fijos (sin ajuste)
    dev2 = t2[t2.es_dev]
    filas = []
    ra_dev = evaluar_score_fijo(dev2, "score_a", CONTROLES_FULL)
    ra_tot = evaluar_score_fijo(t2, "score_a", CONTROLES_FULL)
    filas.append(dict(objetivo="a_log_v8_fijo", muestra="DEV_gkf(0 param)", **ra_dev))
    filas.append(dict(objetivo="a_log_v8_fijo", muestra="TOTAL_crono(0 param)", **ra_tot))

    specs = {
        "b_log_Kact": (["D_sn_log_ident", "D_feo_log_ident"], None),
        "c_kg_lambda_act": (["D_sn_kg_ident", "D_feo_kg_ident"], None),
        "d_beta_libres_GN_C": (["x_R_gn", "x_R_carbon"], None),
        "e_kg_lambda_act_signo": (["D_sn_kg_ident", "D_feo_kg_ident"], {"D_sn_kg_ident": 1, "D_feo_kg_ident": -1}),
    }
    for nombre, (expo, signos) in specs.items():
        r_dev = L.prueba_anidada(DEV, expo, controles=CONTROLES_FULL, esquema="gkf", n_folds=5, n_rep=10, n_perm=2000, signos=signos)
        r_dev.pop("score")
        filas.append(dict(objetivo=nombre, muestra="DEV_gkf(10 rep)", **r_dev))
        r_tot = L.prueba_anidada(t2, expo, controles=CONTROLES_FULL, esquema="crono", n_folds=6, n_perm=2000, signos=signos)
        r_tot.pop("score")
        filas.append(dict(objetivo=nombre, muestra="TOTAL_crono(6)", **r_tot))
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA / "e11_03_cv_objetivos.csv", index=False)
    print(df.round(4).to_string())
    return df


# --------------------------------------------------------------------------- paso 7: moderacion GN/carbon x Sn disponible
def paso7():
    esc = pd.read_pickle(SALIDA / "e11_03_escalon.pkl")
    esc_dev = esc[esc.es_dev].copy()
    anc = pd.read_csv(SALIDA / "e11_03_anclaje.csv")
    fila = anc[(anc.version == "KG_ident") & (anc.muestra == "DEV")].iloc[0]
    k_sn_dev, k_feo_dev = float(fila["K_Sn"]), float(fila["K_FeO"])
    k_sn_v8, k_feo_v8 = 1.0 / KG_SN_POR_PP, -3.0 / KG_SN_POR_PP   # contraste v8 (lineal_kg, lambda=3.0)

    sd_gn_o = esc_dev.groupby("orden_escalon_fase")["a_res_gn_sn"].std()

    def valor_gn(d_, k_sn, k_feo):
        sd = d_["orden_escalon_fase"].map(sd_gn_o)
        shift_sn = d_["theta_gn_sn"] * sd
        shift_feo = d_["theta_gn_feo"] * sd
        dsn = d_["m6_sn_inv_kg"] * (1 - np.exp(-shift_sn))
        dfeo = d_["m6_feo_inv_kg"] * (1 - np.exp(shift_feo))
        return k_sn * dsn + k_feo * dfeo

    esc_dev["val_gn_dev"] = valor_gn(esc_dev, k_sn_dev, k_feo_dev)
    esc_dev["val_gn_v8"] = valor_gn(esc_dev, k_sn_v8, k_feo_v8)
    dsn_c = esc_dev["m6_sn_inv_kg"] * (1 - np.exp(-esc_dev["theta_cxsn_sn"] * esc_dev["m6_sn_inv_kg_prev"] / 1000.0))
    esc_dev["val_carbon_dev"] = k_sn_dev * dsn_c
    esc_dev["val_carbon_v8"] = k_sn_v8 * dsn_c

    filas = []
    rng = np.random.default_rng(SEED)
    for o in range(4):
        sub = esc_dev[esc_dev.orden_escalon_fase == o].dropna(subset=["val_gn_dev", "val_gn_v8", "m6_sn_inv_kg_prev"])
        batches = sub["Batch"].to_numpy()
        for etiqueta, col in (("dev_fit", "val_gn_dev"), ("v8_contraste", "val_gn_v8")):
            X = sm.add_constant(sub[["m6_sn_inv_kg_prev"]])
            fit = sm.OLS(sub[col], X).fit()
            b0, b1 = fit.params["const"], fit.params["m6_sn_inv_kg_prev"]
            umbral = -b0 / b1 if b1 else np.nan
            ths = []
            for _ in range(1000):
                idx = rng.integers(0, len(sub), len(sub))
                s = sub.iloc[idx]
                Xb = sm.add_constant(s[["m6_sn_inv_kg_prev"]])
                try:
                    fb = sm.OLS(s[col], Xb).fit()
                    ths.append(-fb.params["const"] / fb.params["m6_sn_inv_kg_prev"])
                except Exception:
                    pass
            lo, hi = np.percentile(ths, [2.5, 97.5]) if ths else (np.nan, np.nan)
            pct_bajo = float((sub["m6_sn_inv_kg_prev"] < umbral).mean() * 100) if np.isfinite(umbral) else np.nan
            filas.append(dict(orden=o, coef_set=etiqueta, n=len(sub), b0=b0, b1=b1, umbral_sn_inv_kg=umbral,
                              ci_lo=lo, ci_hi=hi, pct_escalones_bajo_umbral=pct_bajo,
                              val_medio_pp=float(sub[col].mean())))
        sub_c = esc_dev[esc_dev.orden_escalon_fase == o]
        filas.append(dict(orden=o, coef_set="carbon_dev_fit", n=len(sub_c), b0=np.nan, b1=np.nan, umbral_sn_inv_kg=np.nan,
                          ci_lo=np.nan, ci_hi=np.nan, pct_escalones_bajo_umbral=np.nan,
                          val_medio_pp=float(sub_c["val_carbon_dev"].mean())))
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA / "e11_03_moderacion.csv", index=False)
    print(df.round(5).to_string())
    return df


if __name__ == "__main__":
    import sys as _s
    etapa = _s.argv[1] if len(_s.argv) > 1 else "todo"
    if etapa in ("3", "todo"):
        print("== paso 3: anclaje ==")
        paso3()
    if etapa in ("4", "todo"):
        print("== paso 4: canales ==")
        paso4()
    if etapa in ("5", "todo"):
        print("== paso 5: sobre-identificacion ==")
        paso5()
    if etapa in ("6", "todo"):
        print("== paso 6: comparacion de objetivos (CV anidada) ==")
        paso6()
    if etapa in ("7", "todo"):
        print("== paso 7: moderacion GN/carbon x Sn disponible ==")
        paso7()
