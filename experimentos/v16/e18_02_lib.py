"""e18_02_lib -- utilidades del experimento E18-02 (diseño intra-batch, efectos fijos de batch y orden).

Panel balanceado: 362 batches x 4 escalones de Reducción (R0-R3) = 1448 filas, cada (Batch, orden) aparece UNA vez
(no es un panel repetido en el tiempo: es un diseño factorial completo batch x orden). Los efectos fijos de batch y
orden se remueven con demeaning alternante (proyecciones alternadas), exacto para diseños balanceados y válido
también con el 1 % de filas con NaN en el estado (no exactamente balanceado ahí).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))
import asesor_receta_v10 as ar  # noqa: E402

STATE = ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev",
         "temperatura_horno_celsius_prev"]
TARGETS = {"sn_dep": "m6_ln_sn_dep", "feo_ret": "m8_ln_feo_ret_dross50", "dT": "d_temperatura_horno_celsius"}
SIGNO_BUENO = {"sn_dep": +1, "feo_ret": -1, "dT": 0}   # sn_dep: mas alto = mejor (mas Sn agotado); feo_ret: mas bajo = mejor
BATCH, ORDEN = "Batch", "orden_escalon_fase"


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    r, bt = ar.escalones_con_receta()
    r = r.copy()
    r["dev_carbon_x_sn"] = r["dev__carbon"] * r["m6_sn_inv_kg_prev"] / 1000.0
    r["es_dev"] = r[BATCH].map(bt["es_dev"])
    r["idx_cronologico"] = r[BATCH].map(bt["idx_cronologico"])
    return r, bt


def demean_two_way(df: pd.DataFrame, cols: list[str], batch_col: str = BATCH, orden_col: str = ORDEN,
                    iters: int = 40) -> pd.DataFrame:
    """Dos vias de efectos fijos (batch, orden) por proyecciones alternadas. Exacto en el limite; con 40
    iteraciones el residuo de no-convergencia es < 1e-10 incluso con las pocas filas con NaN en el estado."""
    x = df[cols].astype(float).to_numpy().copy()
    b = df[batch_col].to_numpy(); o = df[orden_col].to_numpy()
    for _ in range(iters):
        m = pd.DataFrame(x, columns=cols).groupby(b, sort=False).transform("mean").to_numpy()
        x = x - m
        m = pd.DataFrame(x, columns=cols).groupby(o, sort=False).transform("mean").to_numpy()
        x = x - m
    return pd.DataFrame(x, columns=[f"{c}__tilde" for c in cols], index=df.index)


def within_ols(d: pd.DataFrame, y: str, X: list[str], extra_fe: bool = True, iters: int = 40) -> pd.DataFrame:
    """Estimador within (batch + orden si extra_fe) con errores agrupados por batch (HC cluster, sm). Devuelve
    tabla con theta por unidad fisica, theta por sd (de la variacion usada: within si extra_fe, cruda si no),
    IC95, t, p, n, n_clusters, R2 within."""
    import statsmodels.api as sm
    cols = [y] + X
    d = d.dropna(subset=cols + [BATCH, ORDEN]).reset_index(drop=True)
    if extra_fe:
        til = demean_two_way(d, cols, iters=iters)
    else:
        til = d[cols].astype(float) - d[cols].astype(float).mean()
        til.columns = [f"{c}__tilde" for c in cols]
    yt = til[f"{y}__tilde"].to_numpy()
    Xt = til[[f"{x}__tilde" for x in X]].to_numpy()
    fit = sm.OLS(yt, Xt).fit(cov_type="cluster", cov_kwds={"groups": d[BATCH].to_numpy()})
    sd_x = Xt.std(axis=0, ddof=1)
    filas = []
    for i, x in enumerate(X):
        b, se, t, p = fit.params[i], fit.bse[i], fit.tvalues[i], fit.pvalues[i]
        ci = fit.conf_int()[i]
        filas.append(dict(variable=x, theta=b, se=se, ci_lo=ci[0], ci_hi=ci[1], t=t, p=p,
                           theta_por_sd=b * sd_x[i], sd_usada=sd_x[i]))
    tab = pd.DataFrame(filas)
    tab.attrs["n"] = len(d); tab.attrs["n_clusters"] = d[BATCH].nunique(); tab.attrs["r2_within"] = float(fit.rsquared)
    return tab


def resumen_attrs(tab: pd.DataFrame) -> dict:
    return {"n": tab.attrs.get("n"), "n_clusters": tab.attrs.get("n_clusters"), "r2_within": tab.attrs.get("r2_within")}
