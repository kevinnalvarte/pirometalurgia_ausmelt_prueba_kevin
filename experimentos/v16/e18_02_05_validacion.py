"""E18-02 tarea 5: validacion fuera de muestra del modelo within. (a) GroupKFold por batch: theta estimado sin los
batches de prueba -> correlacion entre el target demeaned PREDICHO por theta*dev y el observado en los batches de
prueba (demeaned con su propia media, sin fuga). (b) estabilidad de theta por tercios cronologicos. (c) estabilidad
DEV vs lockbox (bt.es_dev)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import e18_02_lib as L

SAL = Path(__file__).parent
X_BASE = ["dev__gn", "dev__carbon"] + L.STATE


def _fit_coefs(d: pd.DataFrame, target: str, X: list[str]) -> np.ndarray:
    """Coeficientes theta (incl. STATE) del estimador within, ajustados SOLO con `d`."""
    import statsmodels.api as sm
    cols = [target] + X
    til = L.demean_two_way(d, cols)
    Xt = til[[f"{c}__tilde" for c in X]].to_numpy(); yt = til[f"{target}__tilde"].to_numpy()
    return np.linalg.lstsq(Xt, yt, rcond=None)[0]


def oos_groupkfold(r: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    filas = []
    for clave, target in L.TARGETS.items():
        cols = [target] + X_BASE
        d = r.dropna(subset=cols).reset_index(drop=True)
        gkf = GroupKFold(n_splits=n_splits)
        pred, obs = [], []
        for tr, te in gkf.split(d, groups=d[L.BATCH]):
            beta = _fit_coefs(d.iloc[tr].reset_index(drop=True), target, X_BASE)
            d_te = d.iloc[te].reset_index(drop=True)
            til_te = L.demean_two_way(d_te, cols)   # demeaning propio del fold de prueba (sin usar el train)
            yhat = til_te[[f"{c}__tilde" for c in X_BASE]].to_numpy() @ beta
            pred.append(yhat); obs.append(til_te[f"{target}__tilde"].to_numpy())
        pred, obs = np.concatenate(pred), np.concatenate(obs)
        pear = np.corrcoef(pred, obs)[0, 1]
        spear = pd.Series(pred).corr(pd.Series(obs), method="spearman")
        filas.append(dict(target=clave, n=len(pred), pearson=pear, spearman=spear, r2_oos=1 - np.var(obs - pred) / np.var(obs)))
    return pd.DataFrame(filas)


def por_tercios(r: pd.DataFrame) -> pd.DataFrame:
    filas = []
    orden_b = r.groupby(L.BATCH)["idx_cronologico"].first().sort_values()
    terc = pd.qcut(orden_b, 3, labels=["T1_temprano", "T2_medio", "T3_tardio"])
    for clave, target in L.TARGETS.items():
        for lab in terc.cat.categories:
            batches = terc[terc == lab].index
            d = r[r[L.BATCH].isin(batches)]
            tab = L.within_ols(d, target, X_BASE)
            tab.insert(0, "target", clave); tab.insert(1, "tercio", lab)
            tab["n"] = L.resumen_attrs(tab)["n"]; tab["n_batches"] = len(batches)
            filas.append(tab[tab["variable"].isin(["dev__gn", "dev__carbon"])])
    return pd.concat(filas, ignore_index=True)


def dev_vs_lockbox(r: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for clave, target in L.TARGETS.items():
        for tag, sub in (("DEV", r[r["es_dev"]]), ("lockbox", r[~r["es_dev"]])):
            tab = L.within_ols(sub, target, X_BASE)
            tab.insert(0, "target", clave); tab.insert(1, "conjunto", tag)
            tab["n"] = L.resumen_attrs(tab)["n"]; tab["n_batches"] = sub[L.BATCH].nunique()
            filas.append(tab[tab["variable"].isin(["dev__gn", "dev__carbon"])])
    return pd.concat(filas, ignore_index=True)


def main():
    r, bt = L.cargar()
    oos = oos_groupkfold(r); oos.to_csv(SAL / "e18_02_oos_groupkfold.csv", index=False)
    terc = por_tercios(r); terc.to_csv(SAL / "e18_02_tercios.csv", index=False)
    dvl = dev_vs_lockbox(r); dvl.to_csv(SAL / "e18_02_dev_vs_lockbox.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== GroupKFold(5) por batch: correlacion pred vs obs (demeaned) ===")
    print(oos.round(4).to_string(index=False))
    print("\n=== theta por tercios cronologicos ===")
    print(terc[["target", "tercio", "variable", "theta", "ci_lo", "ci_hi", "t", "p", "n_batches"]].round(5).to_string(index=False))
    print("\n=== theta DEV vs lockbox ===")
    print(dvl[["target", "conjunto", "variable", "theta", "ci_lo", "ci_hi", "t", "p", "n_batches"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
