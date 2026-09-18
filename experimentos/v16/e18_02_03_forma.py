"""E18-02 tarea 3: forma de la respuesta within -- dosis-respuesta por quintiles, asimetria (paso de mas vs paso de
menos), heterogeneidad por orden temprano/tardio (R0-R1 vs R2-R3) y por Sn disponible relativo (tercil de
m6_sn_inv_kg_prev tras remover el efecto de orden)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import e18_02_lib as L

SAL = Path(__file__).parent
LEVERS = {"gn": "dev__gn", "carbon": "dev__carbon"}
OTHER = {"gn": "dev__carbon", "carbon": "dev__gn"}


def _fit(r, y, X):
    return L.within_ols(r, y, X)


def dosis_respuesta(r: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for clave, target in L.TARGETS.items():
        cols = [target, "dev__gn", "dev__carbon"] + L.STATE
        d = r.dropna(subset=cols).reset_index(drop=True)
        til = L.demean_two_way(d, cols)
        for palanca, col in LEVERS.items():
            x = til[f"{col}__tilde"]; y = til[f"{target}__tilde"]
            q = pd.qcut(x, 5, labels=False, duplicates="drop")
            g = pd.DataFrame({"q": q, "x": x, "y": y}).groupby("q").agg(x_media=("x", "mean"), y_media=("y", "mean"),
                                                                          y_se=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s))), n=("y", "size"))
            g.insert(0, "palanca", palanca); g.insert(0, "target", clave); filas.append(g.reset_index())
    return pd.concat(filas, ignore_index=True)


def asimetria(r: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for clave, target in L.TARGETS.items():
        for palanca, col in LEVERS.items():
            cols = [target, col, OTHER[palanca]] + L.STATE
            d = r.dropna(subset=cols).reset_index(drop=True)
            til = L.demean_two_way(d, cols)
            xt = til[f"{col}__tilde"].to_numpy()
            pos = np.clip(xt, 0, None); neg = np.clip(xt, None, 0)
            X = np.column_stack([pos, neg, til[[f"{c}__tilde" for c in [OTHER[palanca]] + L.STATE]].to_numpy()])
            y = til[f"{target}__tilde"].to_numpy()
            fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[L.BATCH].to_numpy()})
            for i, nm in enumerate(["pos_paso_de_mas", "neg_paso_de_menos"]):
                filas.append(dict(target=clave, palanca=palanca, parte=nm, theta=fit.params[i], se=fit.bse[i],
                                   ci_lo=fit.conf_int()[i][0], ci_hi=fit.conf_int()[i][1], t=fit.tvalues[i], p=fit.pvalues[i], n=len(d)))
    return pd.DataFrame(filas)


def heterogeneidad_orden(r: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for clave, target in L.TARGETS.items():
        for palanca, col in LEVERS.items():
            base_cols = [target, col, OTHER[palanca]] + L.STATE
            d = r.dropna(subset=base_cols).reset_index(drop=True)
            early = d[L.ORDEN].isin([0, 1]).astype(float)
            d = d.assign(**{f"{col}_temprano": d[col] * early, f"{col}_tardio": d[col] * (1 - early)})
            reg_cols = [f"{col}_temprano", f"{col}_tardio", OTHER[palanca]] + L.STATE
            til = L.demean_two_way(d, [target] + reg_cols)
            X = til[[f"{c}__tilde" for c in reg_cols]].to_numpy(); y = til[f"{target}__tilde"].to_numpy()
            fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[L.BATCH].to_numpy()})
            for i, nm in enumerate(["R0_R1_temprano", "R2_R3_tardio"]):
                filas.append(dict(target=clave, palanca=palanca, tramo=nm, theta=fit.params[i], se=fit.bse[i],
                                   ci_lo=fit.conf_int()[i][0], ci_hi=fit.conf_int()[i][1], t=fit.tvalues[i], p=fit.pvalues[i], n=len(d)))
    return pd.DataFrame(filas)


def heterogeneidad_sn(r: pd.DataFrame) -> pd.DataFrame:
    d0 = r.dropna(subset=["m6_sn_inv_kg_prev"]).copy()
    ln_sn = np.log(d0["m6_sn_inv_kg_prev"].clip(lower=1))
    resid = ln_sn - ln_sn.groupby(d0[L.ORDEN]).transform("mean")   # Sn disponible relativo al tipico de ese orden
    d0["tercil_sn"] = pd.qcut(resid, 3, labels=["T1_bajo", "T2_medio", "T3_alto"])
    filas = []
    for clave, target in L.TARGETS.items():
        for palanca, col in LEVERS.items():
            base_cols = [target, col, OTHER[palanca]] + L.STATE
            d = d0.dropna(subset=base_cols + ["tercil_sn"]).reset_index(drop=True)
            dummies = pd.get_dummies(d["tercil_sn"], prefix=col)
            for c in dummies.columns:
                d[c] = d[col] * dummies[c]
            reg_cols = list(dummies.columns) + [OTHER[palanca]] + L.STATE
            til = L.demean_two_way(d, [target] + reg_cols)
            X = til[[f"{c}__tilde" for c in reg_cols]].to_numpy(); y = til[f"{target}__tilde"].to_numpy()
            fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[L.BATCH].to_numpy()})
            for i, nm in enumerate(dummies.columns):
                filas.append(dict(target=clave, palanca=palanca, tercil=nm, theta=fit.params[i], se=fit.bse[i],
                                   ci_lo=fit.conf_int()[i][0], ci_hi=fit.conf_int()[i][1], t=fit.tvalues[i], p=fit.pvalues[i], n=len(d)))
    return pd.DataFrame(filas)


def main():
    r, bt = L.cargar()
    dr = dosis_respuesta(r); dr.to_csv(SAL / "e18_02_dosis_respuesta.csv", index=False)
    asi = asimetria(r); asi.to_csv(SAL / "e18_02_asimetria.csv", index=False)
    het_o = heterogeneidad_orden(r); het_o.to_csv(SAL / "e18_02_heterogeneidad_orden.csv", index=False)
    het_sn = heterogeneidad_sn(r); het_sn.to_csv(SAL / "e18_02_heterogeneidad_sn.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== dosis-respuesta (quintiles) ===")
    print(dr.round(4).to_string(index=False))
    print("\n=== asimetria ===")
    print(asi.round(5).to_string(index=False))
    print("\n=== heterogeneidad por orden ===")
    print(het_o.round(5).to_string(index=False))
    print("\n=== heterogeneidad por Sn disponible (tercil relativo al orden) ===")
    print(het_sn.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
