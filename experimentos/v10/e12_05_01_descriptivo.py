"""E12-05 tarea 1: descriptivo por candidata (R2 OOF de E[a|S], ICC entre/intra batch, deriva con la campana,
correlacion entre residuos de las candidatas y con los de GN/carbon)."""
import sys; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
from scipy import stats
import e12_05_lib as E

res = E.residuos_escalon_e12()
t = E.tabla_batch_e12(res)
t.to_csv(E.SALIDA / "e12_05_tabla_batch.csv")

CANDS = list(E.PALANCAS) + list(E.PALANCAS_REF)
GRUPOS = ("F", "R")

# --- 1. R2 OOF de E[a|S] (DEV), ya calculado en el modulo; recomputamos aqui para dejar en CSV
from sklearn.metrics import r2_score
filas_r2 = []
for (f, g), s in res[res.es_dev].groupby(["fase_proceso", "grupo"]):
    for p in CANDS:
        col_a, col_ah = f"a__{p}", f"ahat__{p}"
        if col_a not in s.columns:
            continue
        m = s[col_ah].notna() & s[col_a].notna()
        if m.sum() > 5:
            filas_r2.append({"fase": f, "grupo": g, "candidata": p, "n": int(m.sum()),
                              "r2_oof": round(r2_score(s[col_a][m], s[col_ah][m]), 4)})
r2_tab = pd.DataFrame(filas_r2)
r2_tab.to_csv(E.SALIDA / "e12_05_r2_oof.csv", index=False)
print("== R2 OOF de E[a|S] por candidata (DEV) =="); print(r2_tab.to_string(index=False))

# --- 2. ICC entre/intra batch (escalon, DEV, z__palanca), por grupo
filas_icc = []
for g in GRUPOS:
    sub = res[(res.grupo == g) & (res.es_dev)]
    for p in CANDS:
        col = f"z__{p}"
        if col not in sub.columns:
            continue
        d = sub[["Batch", col]].dropna()
        if d[col].std() < 1e-9 or d["Batch"].nunique() < 10:
            continue
        gm = d.groupby("Batch")[col]
        n_i = gm.count(); k = len(n_i); N = n_i.sum()
        grand = d[col].mean()
        msb = ((gm.mean() - grand) ** 2 * n_i).sum() / (k - 1)
        msw = gm.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (N - k)
        n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) != 0 else np.nan
        icc = max(0.0, min(1.0, icc)) if np.isfinite(icc) else np.nan
        filas_icc.append({"grupo": g, "candidata": p, "n_escalones": int(N), "n_batches": int(k),
                           "icc": round(float(icc), 4) if np.isfinite(icc) else np.nan})
icc_tab = pd.DataFrame(filas_icc)
icc_tab.to_csv(E.SALIDA / "e12_05_icc.csv", index=False)
print("\n== ICC (residuo z, entre-batch / total) =="); print(icc_tab.to_string(index=False))

# --- 3. Deriva con la campana (Spearman batch-level exposure vs idx_cronologico), DEV y TOTAL
filas_deriva = []
for g in GRUPOS:
    for p in CANDS:
        col = f"x_{g}_{p}"
        if col not in t.columns:
            continue
        for muestra, tt in (("DEV", t[t.es_dev]), ("TOTAL", t)):
            d = tt[[col, "idx_cronologico"]].dropna()
            if len(d) < 20:
                continue
            rho, pv = stats.spearmanr(d[col], d["idx_cronologico"])
            filas_deriva.append({"grupo": g, "candidata": p, "muestra": muestra, "n": len(d),
                                  "spearman_vs_idx": round(rho, 3), "p": round(pv, 4)})
deriva_tab = pd.DataFrame(filas_deriva)
deriva_tab.to_csv(E.SALIDA / "e12_05_deriva.csv", index=False)
print("\n== Deriva con la campana (Spearman x_g_p vs idx_cronologico) =="); print(deriva_tab.to_string(index=False))

# --- 4. Correlacion entre residuos de candidatas (y con GN/carbon), batch-level, por grupo, DEV
for g in GRUPOS:
    cols = [f"x_{g}_{p}" for p in CANDS if f"x_{g}_{p}" in t.columns]
    dd = t.loc[t.es_dev, cols].dropna()
    corr = dd.corr(method="pearson")
    corr.to_csv(E.SALIDA / f"e12_05_corr_{g}.csv")
    print(f"\n== Correlacion (Pearson) exposiciones batch-level, grupo {g}, DEV, n={len(dd)} ==")
    print(corr.round(2).to_string())
