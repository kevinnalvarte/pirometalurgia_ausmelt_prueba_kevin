"""E18-02 tarea 4: causalidad inversa DENTRO del batch. (a) dev_t responde al target de t-1? (within, controlando
STATE_t). (b) si responde, se agrega target_{t-1} al modelo principal y se reporta el cambio en theta. (c) placebo:
dev_{t+1} -> target_t (debe ser ~0 tras controlar STATE_t)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

import e18_02_lib as L

SAL = Path(__file__).parent


def _con_lags(r: pd.DataFrame) -> pd.DataFrame:
    r = r.sort_values([L.BATCH, L.ORDEN]).reset_index(drop=True).copy()
    g = r.groupby(L.BATCH)
    for clave, target in L.TARGETS.items():
        r[f"lag__{clave}"] = g[target].shift(1)
    for pal, col in (("gn", "dev__gn"), ("carbon", "dev__carbon")):
        r[f"lead__{col}"] = g[col].shift(-1)
    return r


def reversion(r: pd.DataFrame) -> pd.DataFrame:
    """dev_t (gn, carbon) ~ target_{t-1} + STATE_t, within batch+orden, para t=1,2,3."""
    filas = []
    for clave, target in L.TARGETS.items():
        for pal, col in (("gn", "dev__gn"), ("carbon", "dev__carbon")):
            cols = [col, f"lag__{clave}"] + L.STATE
            d = r.dropna(subset=cols).reset_index(drop=True)
            til = L.demean_two_way(d, cols)
            X = til[[f"{c}__tilde" for c in [f"lag__{clave}"] + L.STATE]].to_numpy(); y = til[f"{col}__tilde"].to_numpy()
            fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[L.BATCH].to_numpy()})
            filas.append(dict(target_lag=clave, palanca_dev=pal, theta_lag=fit.params[0], se=fit.bse[0],
                               ci_lo=fit.conf_int()[0][0], ci_hi=fit.conf_int()[0][1], t=fit.tvalues[0], p=fit.pvalues[0], n=len(d)))
    return pd.DataFrame(filas)


def modelo_aumentado(r: pd.DataFrame) -> pd.DataFrame:
    """theta_G, theta_C del modelo principal (t=1,2,3) con y sin target_{t-1} como control."""
    filas = []
    for clave, target in L.TARGETS.items():
        cols_lag = [target, f"lag__{clave}"]
        base = ["dev__gn", "dev__carbon"] + L.STATE
        d = r.dropna(subset=[target, f"lag__{clave}"] + base).reset_index(drop=True)
        for tag, X in (("sin_lag", base), ("con_lag", base + [f"lag__{clave}"])):
            tab = L.within_ols(d, target, X)
            tab.insert(0, "target", clave); tab.insert(1, "control_lag", tag)
            tab["n"] = L.resumen_attrs(tab)["n"]
            filas.append(tab[tab["variable"].isin(["dev__gn", "dev__carbon"])])
    return pd.concat(filas, ignore_index=True)


def placebo(r: pd.DataFrame) -> pd.DataFrame:
    """target_t ~ dev_{t+1}(gn,carbon) + STATE_t, within batch+orden, para t=0,1,2."""
    filas = []
    for clave, target in L.TARGETS.items():
        cols = [target, "lead__dev__gn", "lead__dev__carbon"] + L.STATE
        d = r.dropna(subset=cols).reset_index(drop=True)
        til = L.demean_two_way(d, cols)
        X = til[[f"{c}__tilde" for c in ["lead__dev__gn", "lead__dev__carbon"] + L.STATE]].to_numpy()
        y = til[f"{target}__tilde"].to_numpy()
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[L.BATCH].to_numpy()})
        for i, pal in enumerate(["lead_dev_gn", "lead_dev_carbon"]):
            filas.append(dict(target=clave, palanca_lead=pal, theta=fit.params[i], se=fit.bse[i],
                               ci_lo=fit.conf_int()[i][0], ci_hi=fit.conf_int()[i][1], t=fit.tvalues[i], p=fit.pvalues[i], n=len(d)))
    return pd.DataFrame(filas)


def main():
    r, bt = L.cargar(); r = _con_lags(r)
    rev = reversion(r); rev.to_csv(SAL / "e18_02_reversion.csv", index=False)
    aug = modelo_aumentado(r); aug.to_csv(SAL / "e18_02_modelo_aumentado.csv", index=False)
    plc = placebo(r); plc.to_csv(SAL / "e18_02_placebo.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== reversion: dev_t ~ target_(t-1) + STATE_t ===")
    print(rev.round(5).to_string(index=False))
    print("\n=== modelo aumentado (theta_gn, theta_carbon con/sin lag) ===")
    print(aug[["target", "control_lag", "variable", "theta", "ci_lo", "ci_hi", "t", "p", "n"]].round(5).to_string(index=False))
    print("\n=== placebo: target_t ~ dev_(t+1) + STATE_t ===")
    print(plc.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
