"""E11-02 -- el valor de cada reductor (GN, carbon) sobre el KPI del batch depende del Sn disponible (H2).
Ver experimentos/v9/ITERACION_11_diseno.md y experimentos/v9/e11_02_resultados.md.
.venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3, sin joblib, primer plano.
"""
import sys
from pathlib import Path
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import v9_lib as L
import v8_lib as L8
import modelo_predictivo_v8 as mp8

SEED = 42
OUT = Path("experimentos/v9")

MODERADORES = {
    "ley_sn_escoria_pct_prev": True,
    "m6_sn_inv_kg_prev": True,
    "ratio_sn_feo_prev": True,
    "m6_avance_prev": False,
    "ley_feo_escoria_pct_prev": True,
    "m6_feo_inv_kg_prev": True,
    "temperatura_horno_celsius_prev": False,
}
PRIMARIO = "ley_sn_escoria_pct_prev"   # a priori por teoria: actividad ~ concentracion (ley), no inventario absoluto
SENSIB = ["m6_sn_inv_kg_prev"]

print("cargando datos...")
df = L8.cargar_df()
res = L.residuos_escalon(df)
base = mp8._preparar_base(df)
t_batch = L.tabla_batch(df, res)
dev_b, lockbox_b = L8.mp.split_dev_lockbox(df)

R = res[res.fase_proceso == "Reducción"].merge(
    base[["Batch", "fase_proceso", "orden_escalon_fase"] + list(MODERADORES)].drop_duplicates(
        ["Batch", "fase_proceso", "orden_escalon_fase"]),
    on=["Batch", "fase_proceso", "orden_escalon_fase"], how="left")
print("R shape", R.shape, "n batches", R.Batch.nunique())


def controles_ok(d, nm="DEV", controles=None):
    """DEV conserva los controles completos (convencion v8/v9); en LB/TOTAL se quita espesor si es colineal con idx_cronologico
    (diseno E11-02: lockbox = fin de campana, casi disjunto de DEV en el reloj de desgaste del refractario)."""
    controles = controles or L.CONTROLES
    keep = list(controles)
    if nm != "DEV" and "espesor_ladrillo_norm_mm" in keep and "idx_cronologico" in d.columns:
        dd = d[["espesor_ladrillo_norm_mm", "idx_cronologico"]].dropna()
        if len(dd) > 5 and abs(dd.corr().iloc[0, 1]) > 0.9:
            keep = [c for c in keep if c != "espesor_ladrillo_norm_mm"]
    return keep


def zscore(x, mu=None, sd=None):
    mu = x.mean() if mu is None else mu
    sd = x.std() if sd is None else sd
    return (x - mu) / sd, mu, sd


def features_mod(R, mod, ln, subset_batches=None):
    """Devuelve DataFrame indexado por Batch con x_R_gn, x_R_gn_x_m, x_R_carbon, x_R_carbon_x_m para el moderador `mod`
    (estandarizado con media/sd de DEV, ln opcional), y (mu, sd) usados."""
    x = np.log(R[mod]) if ln else R[mod].astype(float)
    ref = R.Batch.isin(dev_b)
    mu, sd = float(x[ref].mean()), float(x[ref].std())
    mtil = (x - mu) / sd
    out = {}
    for p in ("gn", "carbon"):
        z = R[f"z__{p}"]
        out[f"x_R_{p}"] = z.groupby(R.Batch).mean()
        out[f"x_R_{p}_x_m"] = (z * mtil).groupby(R.Batch).mean()
    return pd.DataFrame(out), mu, sd


def ols_fit(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")


# ============================================================== TAREA 1: b_j0, b_j1 por moderador, DEV/LB/TOTAL
print("\n=== TAREA 1: b_j0, b_j1 por moderador ===")
filas1 = []
cache_feat = {}   # (mod, ln) -> (feat, mu, sd)
for mod, permite_ln in MODERADORES.items():
    for ln in ([False, True] if permite_ln else [False]):
        feat, mu, sd = features_mod(R, mod, ln)
        cache_feat[(mod, ln)] = (feat, mu, sd)
        t2 = t_batch.join(feat)
        Xg = ["x_R_gn", "x_R_gn_x_m"]
        for nm, mask in (("DEV", t2.es_dev), ("LB", ~t2.es_dev), ("TOTAL", pd.Series(True, index=t2.index))):
            d = t2[mask]
            ctr = controles_ok(d, nm)
            dd = d.dropna(subset=Xg + ["x_R_carbon", "x_R_carbon_x_m"] + ctr + [L.KPI])
            f = ols_fit(dd[L.KPI], dd[Xg + ["x_R_carbon", "x_R_carbon_x_m"] + ctr])
            row = {"moderador": mod, "ln": ln, "subset": nm, "n": len(dd)}
            for c in Xg + ["x_R_carbon", "x_R_carbon_x_m"]:
                row[f"b_{c}"] = f.params[c]
                row[f"se_{c}"] = f.bse[c]
                row[f"p_{c}"] = f.pvalues[c]
            filas1.append(row)
tabla1 = pd.DataFrame(filas1)
tabla1.to_csv(OUT / "e11_02_tabla1_moderadores.csv", index=False)
print(tabla1[tabla1.subset == "DEV"][["moderador", "ln", "n", "b_x_R_gn", "p_x_R_gn", "b_x_R_gn_x_m", "p_x_R_gn_x_m",
                                       "b_x_R_carbon", "p_x_R_carbon", "b_x_R_carbon_x_m", "p_x_R_carbon_x_m"]].round(3).to_string())

# canales (mecanismo), DEV, todos los moderadores
print("\n--- canales (DEV) ---")
filas1c = []
for (mod, ln), (feat, mu, sd) in cache_feat.items():
    t2 = t_batch.join(feat)
    d = t2[t2.es_dev]
    ctr = controles_ok(d)
    Xg = ["x_R_gn", "x_R_gn_x_m", "x_R_carbon", "x_R_carbon_x_m"]
    for canal in L.CANALES:
        dd = d.dropna(subset=Xg + ctr + [canal])
        f = ols_fit(dd[canal], dd[Xg + ctr])
        row = {"moderador": mod, "ln": ln, "canal": canal, "n": len(dd)}
        for c in Xg:
            row[f"b_{c}"] = f.params[c]; row[f"p_{c}"] = f.pvalues[c]
        filas1c.append(row)
tabla1c = pd.DataFrame(filas1c)
tabla1c.to_csv(OUT / "e11_02_tabla1_canales.csv", index=False)

# ============================================================== umbral m* (GN) con bootstrap por batch
print("\n=== umbral m* GN, bootstrap 1000 ===")


def boot_umbral(d, Xcols, kpi, ctr, n_boot=1000, seed=SEED):
    dd = d.dropna(subset=Xcols + ctr + [kpi]).reset_index(drop=False)
    Xc = dd[ctr].to_numpy(float); Xe = dd[Xcols].to_numpy(float); y = dd[kpi].to_numpy(float)
    n = len(dd)
    Z = np.column_stack([np.ones(n), Xc, Xe])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    b0, b1 = beta[1 + Xc.shape[1]], beta[1 + Xc.shape[1] + 1]
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Zb, yb = Z[idx], y[idx]
        try:
            bb = np.linalg.lstsq(Zb, yb, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        bb0, bb1 = bb[1 + Xc.shape[1]], bb[1 + Xc.shape[1] + 1]
        if abs(bb1) > 1e-8:
            ms.append(-bb0 / bb1)
    ms = np.array(ms)
    return b0, b1, ms


filas_um = []
for (mod, ln), (feat, mu, sd) in cache_feat.items():
    t2 = t_batch.join(feat)
    for nm, mask in (("DEV", t2.es_dev), ("LB", ~t2.es_dev), ("TOTAL", pd.Series(True, index=t2.index))):
        d = t2[mask]
        ctr = controles_ok(d, nm)
        b0, b1, ms = boot_umbral(d, ["x_R_gn", "x_R_gn_x_m"], L.KPI, ctr)
        if len(ms) < 50:
            continue
        m_star_std = -b0 / b1 if abs(b1) > 1e-9 else np.nan
        m_orig = np.exp(mu + sd * m_star_std) if ln else (mu + sd * m_star_std)
        lo_std, hi_std = np.percentile(ms, [2.5, 97.5])
        lo_orig = np.exp(mu + sd * lo_std) if ln else (mu + sd * lo_std)
        hi_orig = np.exp(mu + sd * hi_std) if ln else (mu + sd * hi_std)
        # percentil del umbral en la distribucion observada del moderador (escalones del subset)
        batches_sub = set(t2.index[mask])
        obs = R.loc[R.Batch.isin(batches_sub), mod].dropna()
        pct = float((obs < m_orig).mean() * 100) if np.isfinite(m_orig) else np.nan
        filas_um.append({"moderador": mod, "ln": ln, "subset": nm, "n": len(d), "b_gn0": b0, "b_gn1": b1,
                          "m_star": m_orig, "m_star_ci_lo": min(lo_orig, hi_orig), "m_star_ci_hi": max(lo_orig, hi_orig),
                          "pct_replicas_finito": len(ms) / 1000, "percentil_obs": pct})
tabla_um = pd.DataFrame(filas_um)
tabla_um.to_csv(OUT / "e11_02_umbral_gn.csv", index=False)
print(tabla_um[(tabla_um.moderador == PRIMARIO)].round(3).to_string())

print("\nscript parte 1 completo")
