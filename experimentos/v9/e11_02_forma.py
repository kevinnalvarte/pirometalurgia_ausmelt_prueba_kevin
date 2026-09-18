"""E11-02 tarea 2: forma funcional (lineal vs ln vs terciles/escalon) de beta_GN(m) y beta_C(m); tarea 3: orden vs Sn disponible."""
import sys
from pathlib import Path
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd
import statsmodels.api as sm

import v9_lib as L
import v8_lib as L8
import modelo_predictivo_v8 as mp8

SEED = 42
OUT = Path("experimentos/v9")
PRIMARIO = "ley_sn_escoria_pct_prev"
LN_PRIMARIO = True

df = L8.cargar_df()
res = L.residuos_escalon(df)
base = mp8._preparar_base(df)
t_batch = L.tabla_batch(df, res)
dev_b, lockbox_b = L8.mp.split_dev_lockbox(df)

MODS = ["ley_sn_escoria_pct_prev", "m6_sn_inv_kg_prev", "ratio_sn_feo_prev", "m6_avance_prev",
        "ley_feo_escoria_pct_prev", "m6_feo_inv_kg_prev", "temperatura_horno_celsius_prev"]
R = res[res.fase_proceso == "Reducción"].merge(
    base[["Batch", "fase_proceso", "orden_escalon_fase"] + MODS].drop_duplicates(["Batch", "fase_proceso", "orden_escalon_fase"]),
    on=["Batch", "fase_proceso", "orden_escalon_fase"], how="left")
R["es_dev"] = R.Batch.isin(dev_b)


def controles_ok(d, nm="DEV", controles=None):
    controles = controles or L.CONTROLES
    keep = list(controles)
    if nm != "DEV" and "espesor_ladrillo_norm_mm" in keep and "idx_cronologico" in d.columns:
        dd = d[["espesor_ladrillo_norm_mm", "idx_cronologico"]].dropna()
        if len(dd) > 5 and abs(dd.corr().iloc[0, 1]) > 0.9:
            keep = [c for c in keep if c != "espesor_ladrillo_norm_mm"]
    return keep


def ols_fit(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")


def feat_lineal(R, mod, ln):
    x = np.log(R[mod]) if ln else R[mod].astype(float)
    mu, sd = float(x[R.es_dev].mean()), float(x[R.es_dev].std())
    mtil = (x - mu) / sd
    out = {}
    for p in ("gn", "carbon"):
        z = R[f"z__{p}"]
        out[f"x_R_{p}"] = z.groupby(R.Batch).mean()
        out[f"x_R_{p}_x_m"] = (z * mtil).groupby(R.Batch).mean()
    return pd.DataFrame(out), mu, sd, mtil


def feat_terciles(R, mod, por_orden=True):
    """3 exposiciones por palanca: x_t1 (tercil bajo), x_t2 (medio), x_t3 (alto) del moderador; cortes con DEV."""
    if por_orden:
        q1 = R.loc[R.es_dev].groupby("orden_escalon_fase")[mod].quantile(1 / 3)
        q2 = R.loc[R.es_dev].groupby("orden_escalon_fase")[mod].quantile(2 / 3)
        c1 = R["orden_escalon_fase"].map(q1); c2 = R["orden_escalon_fase"].map(q2)
    else:
        c1 = R.loc[R.es_dev, mod].quantile(1 / 3); c2 = R.loc[R.es_dev, mod].quantile(2 / 3)
    bajo = R[mod] <= c1; alto = R[mod] > c2; medio = ~bajo & ~alto
    out = {}
    for p in ("gn", "carbon"):
        z = R[f"z__{p}"]
        out[f"x_{p}_t1"] = z.where(bajo, 0).groupby(R.Batch).mean()
        out[f"x_{p}_t2"] = z.where(medio, 0).groupby(R.Batch).mean()
        out[f"x_{p}_t3"] = z.where(alto, 0).groupby(R.Batch).mean()
    return pd.DataFrame(out)


print("=== TAREA 2: forma funcional (moderador primario, DEV) ===")
filas_forma = []
for mod in MODS:
    ln_ok = mod not in ("m6_avance_prev", "temperatura_horno_celsius_prev")
    feat_lin, mu, sd, mtil = feat_lineal(R, mod, ln_ok)
    t2 = t_batch.join(feat_lin)
    feat_ter_o = feat_terciles(R, mod, por_orden=True)
    feat_ter_g = feat_terciles(R, mod, por_orden=False)
    t2 = t2.join(feat_ter_o, rsuffix="_o").join(feat_ter_g, rsuffix="_g")
    for nm, mask in (("DEV", t2.es_dev), ("TOTAL", pd.Series(True, index=t2.index))):
        d = t2[mask]
        ctr = controles_ok(d, nm)
        # lineal
        Xg = ["x_R_gn", "x_R_gn_x_m", "x_R_carbon", "x_R_carbon_x_m"]
        dd = d.dropna(subset=Xg + ctr + [L.KPI])
        f_lin = ols_fit(dd[L.KPI], dd[Xg + ctr])
        r2_lin = f_lin.rsquared
        # terciles por orden
        Xt_o = ["x_gn_t1", "x_gn_t2", "x_gn_t3", "x_carbon_t1", "x_carbon_t2", "x_carbon_t3"]
        ddo = d.dropna(subset=Xt_o + ctr + [L.KPI])
        f_ter_o = ols_fit(ddo[L.KPI], ddo[Xt_o + ctr])
        r2_ter_o = f_ter_o.rsquared
        # terciles globales
        Xt_g = [c + "_g" for c in Xt_o]
        ddg = d.dropna(subset=Xt_g + ctr + [L.KPI])
        f_ter_g = ols_fit(ddg[L.KPI], ddg[Xt_g + ctr])
        r2_ter_g = f_ter_g.rsquared
        filas_forma.append({
            "moderador": mod, "subset": nm, "n": len(dd),
            "r2_lineal": r2_lin, "p_int_gn_lineal": f_lin.pvalues["x_R_gn_x_m"],
            "r2_tercil_orden": r2_ter_o, "n_tercil_orden": len(ddo),
            "gn_t1": f_ter_o.params["x_gn_t1"], "p_gn_t1": f_ter_o.pvalues["x_gn_t1"],
            "gn_t2": f_ter_o.params["x_gn_t2"], "p_gn_t2": f_ter_o.pvalues["x_gn_t2"],
            "gn_t3": f_ter_o.params["x_gn_t3"], "p_gn_t3": f_ter_o.pvalues["x_gn_t3"],
            "c_t1": f_ter_o.params["x_carbon_t1"], "p_c_t1": f_ter_o.pvalues["x_carbon_t1"],
            "c_t2": f_ter_o.params["x_carbon_t2"], "p_c_t2": f_ter_o.pvalues["x_carbon_t2"],
            "c_t3": f_ter_o.params["x_carbon_t3"], "p_c_t3": f_ter_o.pvalues["x_carbon_t3"],
            "r2_tercil_global": r2_ter_g, "n_tercil_global": len(ddg),
            "gn_t1_g": f_ter_g.params["x_gn_t1_g"], "p_gn_t1_g": f_ter_g.pvalues["x_gn_t1_g"],
            "gn_t2_g": f_ter_g.params["x_gn_t2_g"], "p_gn_t2_g": f_ter_g.pvalues["x_gn_t2_g"],
            "gn_t3_g": f_ter_g.params["x_gn_t3_g"], "p_gn_t3_g": f_ter_g.pvalues["x_gn_t3_g"],
        })
tabla_forma = pd.DataFrame(filas_forma)
tabla_forma.to_csv(OUT / "e11_02_forma_funcional.csv", index=False)
cols_show = ["moderador", "subset", "n", "r2_lineal", "p_int_gn_lineal", "gn_t1", "p_gn_t1", "gn_t2", "p_gn_t2", "gn_t3", "p_gn_t3"]
print(tabla_forma[tabla_forma.subset == "DEV"][cols_show].round(3).to_string())

# ---------------------------------------------------------------- curva beta_GN(m), beta_C(m) con bandas bootstrap (primario)
print("\n=== curva beta(m) primario:", PRIMARIO, "===")
feat_lin, mu, sd, mtil = feat_lineal(R, PRIMARIO, LN_PRIMARIO)
t2 = t_batch.join(feat_lin)


def boot_betas(d, Xcols, kpi, ctr, n_boot=1000, seed=SEED):
    dd = d.dropna(subset=Xcols + ctr + [kpi]).reset_index(drop=False)
    Xc = dd[ctr].to_numpy(float); Xe = dd[Xcols].to_numpy(float); y = dd[kpi].to_numpy(float)
    n = len(dd)
    Z = np.column_stack([np.ones(n), Xc, Xe])
    beta0 = np.linalg.lstsq(Z, y, rcond=None)[0][1 + Xc.shape[1]:]
    rng = np.random.default_rng(seed)
    BB = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            bb = np.linalg.lstsq(Z[idx], y[idx], rcond=None)[0][1 + Xc.shape[1]:]
        except np.linalg.LinAlgError:
            continue
        BB.append(bb)
    return beta0, np.array(BB)


d_dev = t2[t2.es_dev]
ctr = controles_ok(d_dev)
b_gn, BB_gn = boot_betas(d_dev, ["x_R_gn", "x_R_gn_x_m"], L.KPI, ctr)
b_c, BB_c = boot_betas(d_dev, ["x_R_carbon", "x_R_carbon_x_m"], L.KPI, ctr)
m_grid_std = np.linspace(np.nanpercentile(mtil[R.es_dev], 5), np.nanpercentile(mtil[R.es_dev], 95), 25)
filas_curva = []
for mstd in m_grid_std:
    m_orig = np.exp(mu + sd * mstd) if LN_PRIMARIO else (mu + sd * mstd)
    beta_gn = b_gn[0] + b_gn[1] * mstd
    beta_gn_ci = np.percentile(BB_gn[:, 0] + BB_gn[:, 1] * mstd, [2.5, 97.5])
    beta_c = b_c[0] + b_c[1] * mstd
    beta_c_ci = np.percentile(BB_c[:, 0] + BB_c[:, 1] * mstd, [2.5, 97.5])
    filas_curva.append({"m_std": mstd, "m_orig": m_orig, "beta_gn": beta_gn, "beta_gn_lo": beta_gn_ci[0], "beta_gn_hi": beta_gn_ci[1],
                         "beta_c": beta_c, "beta_c_lo": beta_c_ci[0], "beta_c_hi": beta_c_ci[1]})
pd.DataFrame(filas_curva).to_csv(OUT / "e11_02_curva_beta.csv", index=False)
print(pd.DataFrame(filas_curva)[["m_orig", "beta_gn", "beta_gn_lo", "beta_gn_hi", "beta_c", "beta_c_lo", "beta_c_hi"]].round(3).to_string())

print("\n=== TAREA 3: orden vs Sn disponible (moderador primario) ===")
orden_c = (R["orden_escalon_fase"] - R.loc[R.es_dev, "orden_escalon_fase"].mean()) / R.loc[R.es_dev, "orden_escalon_fase"].std()
out3 = {}
for p in ("gn", "carbon"):
    z = R[f"z__{p}"]
    out3[f"x_R_{p}"] = z.groupby(R.Batch).mean()
    out3[f"x_R_{p}_x_sn"] = (z * mtil).groupby(R.Batch).mean()
    out3[f"x_R_{p}_x_orden"] = (z * orden_c).groupby(R.Batch).mean()
t3 = t_batch.join(pd.DataFrame(out3))
filas3 = []
for nm, mask in (("DEV", t3.es_dev), ("TOTAL", pd.Series(True, index=t3.index))):
    d = t3[mask]
    ctr = controles_ok(d, nm)
    especs = {
        "solo_sn": ["x_R_gn", "x_R_gn_x_sn", "x_R_carbon", "x_R_carbon_x_sn"],
        "solo_orden": ["x_R_gn", "x_R_gn_x_orden", "x_R_carbon", "x_R_carbon_x_orden"],
        "ambos": ["x_R_gn", "x_R_gn_x_sn", "x_R_gn_x_orden", "x_R_carbon", "x_R_carbon_x_sn", "x_R_carbon_x_orden"],
    }
    for nombre, X in especs.items():
        dd = d.dropna(subset=X + ctr + [L.KPI])
        f = ols_fit(dd[L.KPI], dd[X + ctr])
        row = {"subset": nm, "espec": nombre, "n": len(dd)}
        for c in X:
            row[f"b_{c}"] = f.params[c]; row[f"p_{c}"] = f.pvalues[c]
        filas3.append(row)
tabla3 = pd.DataFrame(filas3)
tabla3.to_csv(OUT / "e11_02_orden_vs_sn.csv", index=False)
print(tabla3.round(3).to_string())

# dentro de cada orden (Rt vs Rl), persiste la moderacion por Sn?
print("\n--- dentro de Rt (R0-R1) y Rl (R2-R3) por separado ---")
out3b = {}
for g in ("Rt", "Rl"):
    sub = R[R.grupo == g]
    for p in ("gn", "carbon"):
        z = sub[f"z__{p}"]
        out3b[f"xi_{g}_{p}"] = z.groupby(sub.Batch).mean()
        out3b[f"xi_{g}_{p}_x_sn"] = (z * mtil.loc[sub.index]).groupby(sub.Batch).mean()
t3b = t_batch.join(pd.DataFrame(out3b))
filas3b = []
for nm, mask in (("DEV", t3b.es_dev), ("TOTAL", pd.Series(True, index=t3b.index))):
    d = t3b[mask]
    ctr = controles_ok(d, nm)
    X = [c for c in out3b if True]
    dd = d.dropna(subset=X + ctr + [L.KPI])
    f = ols_fit(dd[L.KPI], dd[X + ctr])
    row = {"subset": nm, "n": len(dd)}
    for c in X:
        row[f"b_{c}"] = f.params[c]; row[f"p_{c}"] = f.pvalues[c]
    filas3b.append(row)
tabla3b = pd.DataFrame(filas3b)
tabla3b.to_csv(OUT / "e11_02_orden_dentro.csv", index=False)
print(tabla3b.round(3).to_string())

print("\nscript forma/orden completo")
