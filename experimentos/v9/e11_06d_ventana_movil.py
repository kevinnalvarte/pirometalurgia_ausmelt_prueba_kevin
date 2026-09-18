"""E11-06d (Fable): re-anclaje adaptativo. beta(GN_R, carbon_R -> KPI) estimado con los ULTIMOS W batches (ventana movil), aplicado a los
25 batches siguientes (prospectivo puro). Variante con compuerta de significancia (solo se usa un beta si |t| > tmin en la ventana)."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import v9_lib as L
t = pd.read_csv(L.SALIDA / "e11_06b_tabla_batch.csv", index_col=0).sort_values("idx_cronologico")
X = ["xG", "xC"]; C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]
d0 = t.dropna(subset=X + C + [L.KPI]).copy(); n = len(d0); filas = []
for W in (60, 80, 100, 120, 150, None):
    for tmin in (0.0, 1.0, 1.64):
        for paso in (10, 25):
            d = d0.copy(); d["score"] = np.nan; start = (W or 100)
            for s in range(start, n, paso):
                tr = d.iloc[(s - W) if W else 0:s]; te = d.iloc[s:s + paso]
                f = L.ols_hc3(tr[L.KPI], tr[X + C]); b = f.params[X].where(f.tvalues[X].abs() > tmin, 0.0)
                d.loc[te.index, "score"] = te[X].to_numpy() @ b.to_numpy()
            for nm, m in (("todo", d.score.notna()), ("DEV", d.score.notna() & d.es_dev), ("LB", d.score.notna() & ~d.es_dev)):
                dd = d[m]
                if dd.score.std() < 1e-9: continue
                f = L.ols_hc3(dd[L.KPI], dd[["score"] + C]); rho, p = stats.spearmanr(dd.score, dd[L.KPI])
                q = pd.qcut(dd.score.rank(method="first"), 3, labels=False); ry = L.ols_hc3(dd[L.KPI], dd[C]).resid
                filas.append({"W": W or "expansiva", "tmin": tmin, "paso": paso, "muestra": nm, "n": len(dd), "frac_score_no_nulo": (dd.score.abs() > 1e-9).mean().round(2),
                              "pendiente": f.params["score"], "t": f.tvalues["score"], "p_uni": stats.norm.sf(f.tvalues["score"]), "spearman": rho,
                              "kpi_tercio_bajo": ry[q == 0].mean(), "kpi_tercio_alto": ry[q == 2].mean()})
out = pd.DataFrame(filas); out.to_csv(L.SALIDA / "e11_06d_ventana_movil.csv", index=False)
pd.set_option("display.width", 250); print(out.round(3).to_string())
