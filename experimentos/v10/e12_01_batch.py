"""E12-01 tarea 3: nivel batch. zD_R3, zD_F6 como exposiciones batch (un valor por batch, R3/F6 ocurren una vez por
batch). OLS HC3 conjunta KPI y canales ~ xG + xC + zD_R3 + zD_F6 + controles (DEV/lockbox/total, sin espesor); tercios
cronologicos; bootstrap por batch (1000). Forma cuadratica + interaccion zD_R3 x z(%Sn al inicio de R3): busca el
%Sn de corte donde el minuto extra deja de convenir.
"""
import sys
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import e11_07_lib as E
import v9_lib as L

SALIDA = L.RAIZ / "experimentos" / "v10"
SEED = 42
CTRL = E.C_SIN_ESPESOR
CANALES = ["f_dross", "f_polvo", "f_metal", "sn_perdido_escoria_frac"]

exp = pd.read_csv(SALIDA / "e12_01_exposiciones.csv", index_col=0)
t = E.cargar_tabla().join(exp[["zD_R3", "zD_F6", "R3_ley_sn_escoria_pct_prev"]], how="left")
dev_mask_all = t["es_dev"]
mu_ley, sd_ley = t.loc[dev_mask_all, "R3_ley_sn_escoria_pct_prev"].mean(), t.loc[dev_mask_all, "R3_ley_sn_escoria_pct_prev"].std()
t["z_leySn"] = (t["R3_ley_sn_escoria_pct_prev"] - mu_ley) / sd_ley
t["zD_R3_sq"] = t["zD_R3"] ** 2
t["zD_R3_x_leySn"] = t["zD_R3"] * t["z_leySn"]
X = ["xG", "xC", "zD_R3", "zD_F6"]

print("n total =", len(t), "n dev =", dev_mask_all.sum(), "n lockbox =", (~dev_mask_all).sum())

# --- 1) OLS HC3 conjunta KPI + canales, DEV / lockbox / total
filas = []
for y in [E.KPI] + CANALES:
    for nombre, sub in (("DEV", t[dev_mask_all]), ("LOCKBOX", t[~dev_mask_all]), ("TOTAL", t)):
        d = sub.dropna(subset=[y] + X + CTRL)
        if len(d) < 20:
            continue
        fit = L.ols_hc3(d[y], d[X + CTRL])
        for v in X:
            filas.append({"y": y, "muestra": nombre, "n": len(d), "var": v, "coef": fit.params[v], "se": fit.bse[v],
                          "t": fit.tvalues[v], "p": fit.pvalues[v]})
tab1 = pd.DataFrame(filas)
print("\n--- OLS HC3: y ~ xG + xC + zD_R3 + zD_F6 + controles (sin espesor) ---")
print(tab1[tab1["var"].isin(["zD_R3", "zD_F6"])].round(4).to_string())
tab1.to_csv(SALIDA / "e12_01_batch_ols.csv", index=False)

# --- 2) tercios cronologicos (DEV)
dev = t[dev_mask_all].sort_values("idx_cronologico").copy()
dev["tercio"] = pd.qcut(np.arange(len(dev)), 3, labels=["T1", "T2", "T3"])
filas_t = []
for terc, sub in dev.groupby("tercio", observed=True):
    d = sub.dropna(subset=[E.KPI] + X + CTRL)
    fit = L.ols_hc3(d[E.KPI], d[X + CTRL])
    for v in ("zD_R3", "zD_F6"):
        filas_t.append({"tercio": terc, "n": len(d), "var": v, "coef": fit.params[v], "t": fit.tvalues[v], "p": fit.pvalues[v]})
tercios = pd.DataFrame(filas_t)
print("\n--- estabilidad por tercios cronologicos (DEV, KPI ~ ...) ---")
print(tercios.round(4).to_string())
tercios.to_csv(SALIDA / "e12_01_batch_tercios.csv", index=False)

# --- 3) bootstrap por batch (1000), DEV y TOTAL
def boot_coefs(d: pd.DataFrame, y: str, Xc: list[str], n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    d = d.dropna(subset=[y] + Xc + CTRL).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(d)
    out = np.full((n_boot, len(Xc)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        dd = d.iloc[idx]
        try:
            fit = L.ols_hc3(dd[y], dd[Xc + CTRL])
            out[b] = fit.params[Xc].to_numpy()
        except Exception:
            pass
    return pd.DataFrame(out, columns=Xc)


for nombre, sub in (("DEV", t[dev_mask_all]), ("TOTAL", t)):
    bt = boot_coefs(sub, E.KPI, X)
    print(f"\n--- bootstrap por batch (1000), {nombre}: IC95% de zD_R3 / zD_F6 sobre KPI ---")
    for v in ("zD_R3", "zD_F6"):
        lo, hi = bt[v].quantile(.025), bt[v].quantile(.975)
        print(f"  {v}: media={bt[v].mean():.4f}  IC95=[{lo:.4f}, {hi:.4f}]  P(coef>0)={float((bt[v] > 0).mean()):.3f}")
    bt.to_csv(SALIDA / f"e12_01_batch_bootstrap_{nombre.lower()}.csv", index=False)

# --- 4) forma cuadratica + interaccion (DEV, TOTAL)
Xq = ["xG", "xC", "zD_R3", "zD_R3_sq", "zD_R3_x_leySn", "z_leySn", "zD_F6"]
filas_q = []
for nombre, sub in (("DEV", t[dev_mask_all]), ("TOTAL", t)):
    d = sub.dropna(subset=[E.KPI] + Xq + CTRL)
    fit = L.ols_hc3(d[E.KPI], d[Xq + CTRL])
    for v in Xq:
        filas_q.append({"muestra": nombre, "n": len(d), "var": v, "coef": fit.params[v], "se": fit.bse[v], "t": fit.tvalues[v], "p": fit.pvalues[v]})
quad = pd.DataFrame(filas_q)
print("\n--- KPI ~ xG+xC+zD_R3+zD_R3^2+zD_R3:z_leySn+z_leySn+zD_F6+controles ---")
print(quad.round(4).to_string())
quad.to_csv(SALIDA / "e12_01_batch_cuadratica.csv", index=False)

# --- 5) %Sn de corte: derivada en zD_R3=0 -> b1 + b3*z_leySn = 0 -> z_leySn* = -b1/b3; bootstrap CI
def corte_ley(d: pd.DataFrame, seed: int = SEED, n_boot: int = 1000) -> dict:
    d = d.dropna(subset=[E.KPI] + Xq + CTRL).reset_index(drop=True)
    fit = L.ols_hc3(d[E.KPI], d[Xq + CTRL])
    b1, b3 = fit.params["zD_R3"], fit.params["zD_R3_x_leySn"]
    corte_z = -b1 / b3 if b3 != 0 else np.nan
    corte_pct = mu_ley + corte_z * sd_ley
    rng = np.random.default_rng(seed)
    n = len(d)
    cortes = []
    signos_b3 = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        dd = d.iloc[idx]
        try:
            f = L.ols_hc3(dd[E.KPI], dd[Xq + CTRL])
            b1b, b3b = f.params["zD_R3"], f.params["zD_R3_x_leySn"]
            signos_b3.append(np.sign(b3b))
            if abs(b3b) > 1e-8:
                cz = -b1b / b3b
                cortes.append(mu_ley + cz * sd_ley)
        except Exception:
            pass
    cortes = np.array(cortes)
    return {"b1_zD_R3": b1, "b3_interaccion": b3, "corte_z": corte_z, "corte_pct_Sn": corte_pct,
            "corte_pct_Sn_IC_lo": np.nanpercentile(cortes, 2.5) if len(cortes) else np.nan,
            "corte_pct_Sn_IC_hi": np.nanpercentile(cortes, 97.5) if len(cortes) else np.nan,
            "frac_cortes_en_rango_dev": float(np.mean((cortes >= d["R3_ley_sn_escoria_pct_prev"].min()) & (cortes <= d["R3_ley_sn_escoria_pct_prev"].max()))) if len(cortes) else np.nan,
            "frac_b3_mismo_signo": float(np.mean(np.array(signos_b3) == np.sign(b3))) if signos_b3 else np.nan,
            "n_boot_validos": len(cortes)}


filas_corte = []
for nombre, sub in (("DEV", t[dev_mask_all]), ("TOTAL", t)):
    r = corte_ley(sub)
    r["muestra"] = nombre
    filas_corte.append(r)
    print(f"\n--- corte de %Sn en escoria (inicio de R3) donde el minuto extra deja de convenir, {nombre} ---")
    print({k: (round(v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in r.items()})
pd.DataFrame(filas_corte).to_csv(SALIDA / "e12_01_batch_corte.csv", index=False)

t.to_csv(SALIDA / "e12_01_tabla_batch_completa.csv")
