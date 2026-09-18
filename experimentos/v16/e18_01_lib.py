"""E18-01 -- libreria: tabla a nivel batch para el experimento natural de revisiones de receta de GN/carbon
en Reduccion, mas un motor manual de OLS-HC3 y 2SLS (robusto HC1 o Newey-West) sin depender de linearmodels
(no instalado en el venv). Todo en numpy/statsmodels puro.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v8", RAIZ / "experimentos" / "v9"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import asesor_receta_v10 as A  # noqa: E402
import v9_lib as L  # noqa: E402

KPI = A.KPI
CONTROLES = list(A.CONTROLES)          # feed_sn_total_t, ley_sn_conc_batch_pct, frac_carga_secundaria_F, feed_dross_Fe_total_t, idx_cronologico
CONTROLES_SIN_IDX = [c for c in CONTROLES if c != "idx_cronologico"]
CANALES = ["f_dross", "f_polvo", "f_metal", "sn_perdido_escoria_frac"]
PALANCAS = ["gn", "carbon"]


def tabla_batch() -> pd.DataFrame:
    """Batch x (receta media R0-R3, ejecutado medio R0-R3, desvio, KPI, canales, controles, es_dev, T_media_R,
    variables de tendencia flexible: idx centrado^1..3, bloques de 40)."""
    r, bt = A.escalones_con_receta()
    g = r.groupby("Batch")
    out = pd.DataFrame(index=g.size().index)
    for p, (_, col) in A.PALANCAS.items():
        out[f"plan_{p}"] = g[f"plan__{p}"].mean()
        out[f"ejec_{p}"] = g[col].mean()
        out[f"dev_{p}"] = out[f"ejec_{p}"] - out[f"plan_{p}"]
    out["plan_aire"] = g["plan__aire"].mean()
    out["ejec_aire"] = g[A.PALANCAS["aire"][1]].mean()
    out["n_pasos"] = g.size()
    out["temperatura_horno_prev_media"] = g["temperatura_horno_celsius_prev"].mean() if "temperatura_horno_celsius_prev" in r.columns else np.nan
    t = bt.join(out, how="inner")
    t = t[t["n_pasos"] >= 3].sort_values("idx_cronologico").copy()
    t["idxc"] = (t["idx_cronologico"] - t["idx_cronologico"].mean()) / 100.0
    t["idxc2"] = t["idxc"] ** 2
    t["idxc3"] = t["idxc"] ** 3
    t["bloque40"] = (t["idx_cronologico"] // 40).astype(int)
    t["fin_campana"] = t["idx_cronologico"] >= (t["idx_cronologico"].max() - 62)   # ultimos 63 batches
    t["kpi_prev"] = t[KPI].shift(1)
    t["dross_prev10"] = t["f_dross"].rolling(10).mean().shift(1)
    t["kpi_prev10"] = t[KPI].rolling(10).mean().shift(1)
    return t


def bloque_dummies(t: pd.DataFrame, col: str = "bloque40", drop_first: bool = True) -> pd.DataFrame:
    d = pd.get_dummies(t[col], prefix="b40", drop_first=drop_first)
    return d.astype(float)


def controles_tendencia(t: pd.DataFrame, tendencia: str, controles: list[str] | None = None) -> pd.DataFrame:
    """Matriz de controles + tendencia flexible. tendencia in {'lineal','cubica','bloque40','ninguna'}."""
    controles = CONTROLES_SIN_IDX if controles is None else controles
    X = t[controles].copy()
    if tendencia == "lineal":
        X["idxc"] = t["idxc"]
    elif tendencia == "cubica":
        X[["idxc", "idxc2", "idxc3"]] = t[["idxc", "idxc2", "idxc3"]]
    elif tendencia == "bloque40":
        X = X.join(bloque_dummies(t))
    elif tendencia == "ninguna":
        pass
    else:
        raise ValueError(tendencia)
    return X


# --------------------------------------------------------------------------- OLS robusto (HC3 / Newey-West)

def ols_robusto(y: pd.Series, X: pd.DataFrame, cov: str = "HC3", nw_lags: int = 5):
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    if cov == "HAC":
        return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return sm.OLS(y, Xc).fit(cov_type=cov)


def _bartlett_hac(G: np.ndarray, lags: int) -> np.ndarray:
    """Meat de Newey-West (kernel Bartlett) para g_t = fila de G (n x m), ya centrada implicitamente por ser score."""
    n = G.shape[0]
    S = G.T @ G / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1)
        Gl = G[l:].T @ G[:-l] / n
        S += w * (Gl + Gl.T)
    return S


def iv2sls(y: np.ndarray, Xend: np.ndarray, Xexog: np.ndarray, Zexcl: np.ndarray, cov: str = "HC1", nw_lags: int = 5,
           nombres_end: list[str] | None = None, nombres_exog: list[str] | None = None):
    """2SLS manual con constante incluida en Xexog. Devuelve dict con beta, se, t, p, R2, F de primera etapa
    (por variable endogena, excluyendo su(s) propio(s) instrumento(s) del resto), y objeto residual para Hausman."""
    from scipy import stats
    n = y.shape[0]
    X = np.column_stack([Xend, Xexog])          # k = k_end + k_exog
    Z = np.column_stack([Zexcl, Xexog])         # l = k_z + k_exog  (l >= k para identificacion)
    k, l = X.shape[1], Z.shape[1]
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    Pz = Z @ ZtZ_inv @ Z.T
    Xhat = Pz @ X
    XhatXhat_inv = np.linalg.pinv(Xhat.T @ Xhat)
    beta = XhatXhat_inv @ (Xhat.T @ y)
    e = y - X @ beta                             # residuo con X REAL (no Xhat) -- clave para SE correcto
    if cov == "HAC":
        G = Xhat * e[:, None]
        S = _bartlett_hac(G, nw_lags)
        V = XhatXhat_inv @ (n * S) @ XhatXhat_inv
    else:  # HC1
        dfc = n / max(n - k, 1)
        meat = Xhat.T @ (np.diag(e ** 2)) @ Xhat if n < 2000 else Xhat.T @ (Xhat * (e ** 2)[:, None])
        V = dfc * XhatXhat_inv @ meat @ XhatXhat_inv
    se = np.sqrt(np.diag(V))
    tvals = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(tvals)))
    yhat = X @ beta
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    k_end = Xend.shape[1]
    nombres = (nombres_end or [f"end{i}" for i in range(k_end)]) + (nombres_exog or [f"exog{i}" for i in range(Xexog.shape[1])])
    out = {"beta": dict(zip(nombres, beta)), "se": dict(zip(nombres, se)), "t": dict(zip(nombres, tvals)),
           "p": dict(zip(nombres, pvals)), "r2": r2, "n": n, "e": e, "beta_arr": beta, "V": V, "nombres": nombres}
    return out


def primera_etapa_F(x_end: np.ndarray, Zexcl_own: np.ndarray, Xexog: np.ndarray, Zexcl_otros: np.ndarray | None = None,
                     cov: str = "HC3") -> dict:
    """Regresion de UNA variable endogena sobre su(s) instrumento(s) propio(s) + instrumentos de otras endogenas (si joint) +
    exogenas; F robusto de exclusion del/los instrumento(s) propio(s) (test conjunto si Zexcl_own tiene >1 columna)."""
    import statsmodels.api as sm
    Zexcl_own = np.atleast_2d(Zexcl_own.T).T if Zexcl_own.ndim == 1 else Zexcl_own
    cols = [Zexcl_own]
    n_own = Zexcl_own.shape[1]
    if Zexcl_otros is not None and Zexcl_otros.size:
        cols.append(Zexcl_otros)
    cols.append(Xexog)
    Z = np.column_stack(cols)
    mod = sm.OLS(x_end, sm.add_constant(Z, has_constant="add")).fit(cov_type=cov)
    # los primeros n_own coeficientes tras la constante son los instrumentos propios
    idx_own = list(range(1, 1 + n_own))
    Rmat = np.zeros((n_own, len(mod.params)))
    for i, j in enumerate(idx_own):
        Rmat[i, j] = 1.0
    ft = mod.f_test(Rmat)
    return {"F": float(np.asarray(ft.fvalue).squeeze()), "p": float(np.asarray(ft.pvalue).squeeze()),
            "coef_own": [float(mod.params[j]) for j in idx_own], "r2": mod.rsquared, "n": int(mod.nobs)}


def dwh_control_function(y: np.ndarray, x_end: np.ndarray, Xexog: np.ndarray, Zexcl: np.ndarray, cov: str = "HC3") -> dict:
    """Durbin-Wu-Hausman via funcion de control: v_hat = residuo de x_end ~ Zexcl+Xexog; se agrega v_hat a
    y ~ x_end + Xexog; el t del coeficiente de v_hat es el test DWH (H0: exogeneidad)."""
    import statsmodels.api as sm
    Z = np.column_stack([Zexcl, Xexog]) if Zexcl.ndim > 1 else np.column_stack([Zexcl[:, None], Xexog])
    fs = sm.OLS(x_end, sm.add_constant(Z, has_constant="add")).fit()
    v_hat = fs.resid
    X2 = np.column_stack([x_end[:, None] if x_end.ndim == 1 else x_end, Xexog, v_hat])
    ss = sm.OLS(y, sm.add_constant(X2, has_constant="add")).fit(cov_type=cov)
    j = X2.shape[1]   # ultima columna = v_hat
    return {"t_vhat": float(ss.tvalues[j]), "p_vhat": float(ss.pvalues[j]), "coef_end_ols_cf": float(ss.params[1])}
