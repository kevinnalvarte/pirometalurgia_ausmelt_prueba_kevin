"""E18-05 (Fable): selectividad marginal del GN por ORDEN con el diseno intra-batch (efectos fijos de batch y de orden; theta_GN especifico por orden)
y bootstrap por batch del cociente  R_o = kg Sn agotado / kg FeO reducido  por +1 Nm3/min de GN. El GN por encima de la receta conviene en el orden o
si  lambda/mu < R_o  (lambda = kg de Sn perdidos a dross+polvo por kg de FeO reducido; mu = fraccion del Sn agotado que llega a metal)."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v16")
import numpy as np, pandas as pd
import e18_02_lib as E
r, bt = E.cargar(); ys = ["m6_ln_sn_dep", "m8_ln_feo_ret_dross50"]
for o in range(4):
    r[f"g{o}"] = r["dev__gn"].where(r[E.ORDEN] == o, 0.0)
X = [f"g{o}" for o in range(4)] + ["dev__carbon"] + E.STATE
d = r.dropna(subset=ys + X).reset_index(drop=True); til = E.demean_two_way(d, ys + X); til["Batch"] = d.Batch.to_numpy(); til["es_dev"] = d.es_dev.to_numpy()
inv = d.groupby(E.ORDEN)[["m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev"]].median()
def fit(t):
    Xt = t[[f"{x}__tilde" for x in X]].to_numpy(); out = {}
    for y in ys:
        out[y] = np.linalg.lstsq(Xt, t[f"{y}__tilde"].to_numpy(), rcond=None)[0][:4]
    return out
def tabla(t, nb=1000, seed=0):
    rng = np.random.default_rng(seed); ub = t.Batch.unique(); idx = {b: np.where(t.Batch.to_numpy() == b)[0] for b in ub}; base = fit(t); B = []
    for _ in range(nb):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]); f = fit(t.iloc[ii]); B.append(np.r_[f[ys[0]], f[ys[1]]])
    B = np.array(B); filas = []
    for o in range(4):
        sn = inv.m6_sn_inv_kg_prev[o]; fe = inv.m6_feo_inv_kg_prev[o]
        dsn = base[ys[0]][o] * sn; dfe = -base[ys[1]][o] * fe; bs_sn = B[:, o] * sn; bs_fe = -B[:, 4 + o] * fe; R = bs_sn / np.where(bs_fe > 1e-6, bs_fe, np.nan)
        filas.append({"orden": f"R{o}", "theta_sn": base[ys[0]][o], "t_sn": base[ys[0]][o] / B[:, o].std(), "theta_feo": base[ys[1]][o], "t_feo": base[ys[1]][o] / B[:, 4 + o].std(),
                      "kg_Sn_agotado": dsn, "kg_FeO_reducido": dfe, "R_Sn/FeO": dsn / dfe if dfe > 0 else np.nan, "R_P5": np.nanpercentile(R, 5), "R_P95": np.nanpercentile(R, 95),
                      "P(R<0.5)": np.nanmean(R < 0.5), "P(R<0.7)": np.nanmean(R < 0.7), "P(R<0.9)": np.nanmean(R < 0.9), "P(R>0.9)": np.nanmean(R > 0.9)})
    return pd.DataFrame(filas)
pd.set_option("display.width", 230)
for nm, t in (("TOTAL", til), ("DEV", til[til.es_dev]), ("LOCKBOX", til[~til.es_dev])):
    T = tabla(t.reset_index(drop=True)); T.insert(0, "muestra", nm); print(T.round(3).to_string()); T.to_csv(f"experimentos/v16/e18_05_balance_{nm}.csv", index=False)
print("inventarios tipicos por orden:\n", inv.round(0).to_string())
