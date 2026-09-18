"""E17-01 (Fable): con la exposicion EXACTA (ejecutado - receta), la temperatura del horno modera el valor del GN?  Modelo por batch:
  KPI = c + bG*x_G + bGT*x_GT + bC*x_C (+ bCT*x_CT) + controles (+ persistencia: KPI_prev, x_G_prev)
  x_G = media_t d_gn_t ; x_GT = media_t d_gn_t * zT_t  (zT = (T_horno_prev - 1050)/70, constantes fijas a priori, no estimadas)
Pruebas: DEV / LB / total; HOLDOUT real: modelo ajustado SOLO en DEV -> puntaje en lockbox; walk-forward expansivo y W=100; permutacion por bloques."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import asesor_receta_v10 as A, v9_lib as L, e11_07_lib as W7
r, bt = A.escalones_con_receta(); dev_b, _ = A.L8.mp.split_dev_lockbox(A.L8.cargar_df())
sd = r[r.Batch.isin(dev_b)].groupby("orden_escalon_fase")[["dev__gn", "dev__carbon"]].std()
r["dG"] = r.dev__gn / r.orden_escalon_fase.map(sd.dev__gn); r["dC"] = r.dev__carbon / r.orden_escalon_fase.map(sd.dev__carbon)
for Tcol, T0, Ts in (("temperatura_horno_celsius_prev", 1050.0, 70.0),):
    r["zT"] = (r[Tcol] - T0) / Ts
    g = r.groupby("Batch"); E = pd.DataFrame({"xG": g.dG.mean(), "xC": g.dC.mean(), "xGT": (r.dG * r.zT).groupby(r.Batch).mean(), "xCT": (r.dC * r.zT).groupby(r.Batch).mean(), "zTm": g.zT.mean()})
    t = bt.join(E).sort_values("idx_cronologico"); t["kpi_prev"] = t[L.KPI].shift(1); t["xG_prev"] = t.xG.shift(1); t.to_csv("experimentos/v15/e17_01_tabla.csv")
    C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]
    print("zT medio: DEV %.2f  LB %.2f ; solape: %% escalones LB con zT > P10 de DEV: %.2f" % (r[r.Batch.isin(dev_b)].zT.mean(), r[~r.Batch.isin(dev_b)].zT.mean(), (r[~r.Batch.isin(dev_b)].zT > r[r.Batch.isin(dev_b)].zT.quantile(.1)).mean()))
    for Xs in (["xG", "xC"], ["xG", "xGT", "xC"], ["xG", "xGT", "xC", "xCT"], ["xG", "xGT", "xC", "zTm"], ["xG", "xGT", "xC", "kpi_prev", "xG_prev"]):
        for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
            for y in (L.KPI, "f_dross", "f_polvo"):
                dd = tt.dropna(subset=Xs + C + [y]); f = L.ols_hc3(dd[y] * (1 if y == L.KPI else 100), dd[Xs + C]); print(nm, y[:8], len(dd), " ".join(f"{c}:{f.params[c]:+.2f}(t{f.tvalues[c]:+.1f})" for c in Xs))
        # holdout real DEV -> LB
        d1 = t[t.es_dev].dropna(subset=Xs + C + [L.KPI]); f = L.ols_hc3(d1[L.KPI], d1[Xs + C]); xs_a = [x for x in Xs if x.startswith("x") and not x.endswith("prev")]
        d2 = t[~t.es_dev].dropna(subset=Xs + C + [L.KPI]).copy(); d2["score"] = d2[xs_a].to_numpy() @ f.params[xs_a].to_numpy(); q = L.ols_hc3(d2[L.KPI], d2[["score"] + C]); rho, p = stats.spearmanr(d2.score, d2[L.KPI])
        print(">> HOLDOUT DEV->LB", Xs, "pendiente %.2f t %.2f | spearman %.3f p %.3f | efecto GN implicito en LB (zT medio %.2f): %.2f" % (q.params["score"], q.tvalues["score"], rho, p, d2.zTm.mean() if "zTm" in d2 else np.nan, f.params["xG"] + f.params.get("xGT", 0) * t[~t.es_dev].zTm.mean()))
