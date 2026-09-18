"""E11-01 -- helpers compartidos (regresion conjunta, BH, bootstrap por batch, exposicion fisica, sensemakr manual).
Uso: sys.path.insert(0, "experimentos/v9"); import e11_01_common as C
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v9"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v9_lib as L  # noqa: E402

PALANCAS3 = ["carbon", "gn", "exo2"]
CANALES_PP = ["f_metal", "f_dross", "f_polvo", "sn_perdido_escoria_frac"]  # x100 -> pp
PRINCIPALES = ["x_Rt_gn", "x_Rl_gn", "x_R_gn", "x_F_carbon", "x_R_carbon", "x_F_exo2"]  # nombres usados en el memo


def exposiciones_R_combinada(res: pd.DataFrame, palancas=None, columna="z", ponderar_dur=False) -> pd.DataFrame:
    """Como L.exposiciones_batch pero con Reduccion agrupada (Rt+Rl -> "R") ademas de F."""
    palancas = palancas or PALANCAS3
    r2 = res.copy()
    r2["grupo"] = r2["grupo"].replace({"Rt": "R", "Rl": "R"})
    return L.exposiciones_batch(r2, palancas=palancas, grupos=("R",), columna=columna, ponderar_dur=ponderar_dur)


def exposiciones_fisicas(res: pd.DataFrame, palancas=None, grupos=("F", "Rt", "Rl")) -> pd.DataFrame:
    """Sigma residuo (unidades fisicas, NO z) x duracion_min por batch y grupo: kg de carbon / Nm3 de GN
    por encima de lo esperado dado el estado, acumulado en el grupo."""
    palancas = palancas or PALANCAS3
    filas = {}
    for g in grupos:
        sub = res[res["grupo"] == g]
        for p in palancas:
            v = sub[f"res__{p}"] * sub["dur_min"]
            filas[f"xf_{g}_{p}"] = v.groupby(sub["Batch"]).sum(min_count=1)
    return pd.DataFrame(filas)


def bh(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values (misma indexacion que pvals)."""
    p = pvals.dropna().sort_values()
    n = len(p)
    q = (p * n / (np.arange(n) + 1)).to_numpy()
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = pd.Series(np.minimum(q, 1.0), index=p.index)
    return out.reindex(pvals.index)


def ols_familia(y: pd.Series, X: pd.DataFrame, exposiciones: list[str]) -> pd.DataFrame:
    """OLS HC3 y ~ X (X incluye exposiciones + controles). Tabla de coef/IC/p SOLO para `exposiciones`."""
    fit = L.ols_hc3(y, X)
    ci = fit.conf_int()
    filas = []
    for e in exposiciones:
        filas.append({"exposicion": e, "coef": float(fit.params[e]), "ci_lo": float(ci.loc[e, 0]), "ci_hi": float(ci.loc[e, 1]),
                      "p": float(fit.pvalues[e]), "n": int(fit.nobs)})
    return pd.DataFrame(filas)


def regresion_conjunta(t: pd.DataFrame, exposiciones: list[str], controles: list[str], ys: list[str] = None) -> pd.DataFrame:
    """Para cada y en ys (default KPI + canales x100), OLS HC3 y ~ exposiciones + controles. Devuelve tabla larga."""
    ys = ys or [L.KPI] + CANALES_PP
    filas = []
    cols = list(dict.fromkeys(exposiciones + controles))
    d = t.dropna(subset=cols + [c for c in ys if c in ("recuperacion_refinada_pct",)] + CANALES_PP + [L.KPI]).copy()
    # y por y (evita perder filas por NaN en un canal que no se usa)
    for y in ys:
        col = y
        yy = t[col] * (100 if col in CANALES_PP else 1)
        dd = pd.concat([yy.rename("_y"), t[cols]], axis=1).dropna()
        if len(dd) < 20:
            continue
        tab = ols_familia(dd["_y"], dd[cols], exposiciones)
        tab.insert(0, "y", y)
        filas.append(tab)
    out = pd.concat(filas, ignore_index=True)
    return out


def bootstrap_batch_coefs(t: pd.DataFrame, exposiciones: list[str], controles: list[str], kpi: str, n_boot: int = 1000,
                          seed: int = 42) -> pd.DataFrame:
    """Bootstrap por batch (resample con reemplazo de FILAS de `t`, que ya es 1 fila = 1 batch) de los coef OLS (sin HC3, solo lstsq)."""
    cols = list(dict.fromkeys(exposiciones + controles))
    d = t.dropna(subset=cols + [kpi]).reset_index(drop=True)
    y = d[kpi].to_numpy(float)
    Xc = d[controles].to_numpy(float)
    Xe = d[exposiciones].to_numpy(float)
    n = len(d)
    rng = np.random.default_rng(seed)

    def fit(idx):
        Xcs = Xc[idx]; mu, sd = Xcs.mean(0), Xcs.std(0) + 1e-12
        Z = np.column_stack([np.ones(len(idx)), (Xcs - mu) / sd, Xe[idx]])
        return np.linalg.lstsq(Z, y[idx], rcond=None)[0][1 + Xc.shape[1]:]

    boots = np.array([fit(rng.integers(0, n, n)) for _ in range(n_boot)])
    obs = fit(np.arange(n))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"exposicion": exposiciones, "coef_obs": obs, "ci_lo_boot": lo, "ci_hi_boot": hi,
                         "se_boot": boots.std(0), "n": n})


def rv_sensemakr(t_stat: float, dof: int, q: float = 1.0, alpha: float = 0.05) -> dict:
    """Partial R2, Robustness Value RV_q y RV_{q,alpha} (Cinelli & Hazlett 2020), formulas cerradas."""
    from scipy import stats
    f = t_stat / np.sqrt(dof)                      # partial Cohen's f
    r2 = t_stat ** 2 / (t_stat ** 2 + dof)          # partial R2
    fq = q * abs(f)
    rv_q = 0.5 * (np.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2)
    t_alpha = stats.t.ppf(1 - alpha / 2, dof)
    fqa_num = (q * abs(t_stat) - t_alpha)
    if fqa_num <= 0:
        rv_qa = 0.0
    else:
        fqa = fqa_num / np.sqrt(dof)
        rv_qa = 0.5 * (np.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2)
    return {"t": t_stat, "dof": dof, "f_partial": f, "r2_partial": r2, "RV_q1": rv_q, "RV_q1_alpha05": rv_qa}


if __name__ == "__main__":
    res = L.residuos_escalon()
    t = L.tabla_batch(res=res).join(exposiciones_R_combinada(res)).join(exposiciones_fisicas(res))
    print(t.shape, [c for c in t.columns if c.startswith("x_R_") or c.startswith("xf_")])
