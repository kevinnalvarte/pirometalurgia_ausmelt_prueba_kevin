"""E11-06c (Fable): walk-forward prospectivo del puntaje de concordancia (beta estimado SOLO con el pasado) y localizacion de la ruptura de regimen."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import v9_lib as L
t = pd.read_csv(L.SALIDA / "e11_06b_tabla_batch.csv", index_col=0).sort_values("idx_cronologico")
X = ["xG", "xC"]; C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]   # espesor = reloj colineal con idx
d = t.dropna(subset=X + C + [L.KPI]).copy(); n = len(d); d["pos"] = np.arange(n)
for bloque, mintrain in ((25, 100), (50, 100)):
    d["score"] = np.nan; filas = []
    for s in range(mintrain, n, bloque):
        tr, te = d.iloc[:s], d.iloc[s:s + bloque]
        f = L.ols_hc3(tr[L.KPI], tr[X + C]); d.loc[te.index, "score"] = te[X].to_numpy() @ f.params[X].to_numpy()
        ry = te[L.KPI] - te[L.KPI].mean(); r = np.corrcoef(d.loc[te.index, "score"], ry)[0, 1]
        filas.append({"inicio": s, "n": len(te), "T_R": te.T_media_R.mean().round(0), "bG": f.params["xG"].round(2), "bC": f.params["xC"].round(2), "r_score_kpi": round(r, 3), "frac_lockbox": (~te.es_dev).mean().round(2)})
    print(f"--- bloque {bloque}"); print(pd.DataFrame(filas).to_string())
    for nm, m in (("WF solo DEV", d.es_dev & d.score.notna()), ("WF lockbox", ~d.es_dev & d.score.notna()), ("WF todo", d.score.notna())):
        dd = d[m]; f = L.ols_hc3(dd[L.KPI], dd[["score"] + C]); rho, p = stats.spearmanr(dd.score, dd[L.KPI])
        print(f"{nm}: n {len(dd)} pendiente {f.params['score']:+.2f} t {f.tvalues['score']:+.2f} p_unilateral {stats.norm.sf(f.tvalues['score']):.4f} spearman {rho:+.3f}")
# coeficiente movil (ventana de 80 batches) de xG y xC sobre KPI y canales + temperatura media
filas = []
for s in range(0, n - 80 + 1, 20):
    w = d.iloc[s:s + 80]; row = {"ini": s, "T_R": round(w.T_media_R.mean()), "espesor": round(w.espesor_ladrillo_norm_mm.mean(), 1)}
    for y in (L.KPI, "f_dross", "f_polvo", "f_metal"):
        f = L.ols_hc3(w[y] * (1 if y == L.KPI else 100), w[X + C]); row[f"G>{y[:7]}"] = round(f.params["xG"], 2); row[f"tG>{y[:7]}"] = round(f.tvalues["xG"], 1)
        if y == L.KPI: row["C>KPI"] = round(f.params["xC"], 2); row["tC"] = round(f.tvalues["xC"], 1)
    filas.append(row)
print(pd.DataFrame(filas).to_string())
