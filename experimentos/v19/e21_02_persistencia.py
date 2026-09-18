"""E21-02 (Fable): la adherencia es persistente entre batches (lag 1 predice el KPI actual). Se separa la parte PROPIA del batch (innovacion) de la persistente:
(a) KPI ~ adh + adh_lag1 + adh_lag2 + KPI_lag1 (+controles); (b) adh menos su media movil de los 10 batches anteriores (innovacion) ; (c) efectos fijos por bloques de 20 batches."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import v9_lib as L, e11_07_lib as W7
t = pd.read_csv("experimentos/v16/e18_07_tabla.csv", index_col=0).sort_values("idx_cronologico"); K = L.KPI; C = W7.C_SIN_ESPESOR; t["adh"] = -t.exc_todo
for k in (1, 2): t[f"adh_l{k}"] = t.adh.shift(k)
t["kpi_l1"] = t[K].shift(1); t["adh_mm10"] = t.adh.shift(1).rolling(10, min_periods=5).mean(); t["adh_innov"] = t.adh - t.adh_mm10; t["bloque"] = (np.arange(len(t)) // 20)
print("autocorrelacion de la adherencia: lag1 %.2f lag2 %.2f lag5 %.2f | del KPI: lag1 %.2f" % (t.adh.autocorr(1), t.adh.autocorr(2), t.adh.autocorr(5), t[K].autocorr(1)))
for nm, tt in (("DEV", t[t.es_dev]), ("TOTAL", t)):
    for Xs in (["adh"], ["adh", "adh_l1", "adh_l2", "kpi_l1"], ["adh_innov", "adh_mm10"], ["adh_innov", "adh_mm10", "kpi_l1"]):
        d = tt.dropna(subset=[K] + Xs + C); f = L.ols_hc3(d[K], d[Xs + C]); print(f"{nm:5s} n {len(d)} " + " ".join(f"{x}:{f.params[x]:+.2f}(t{f.tvalues[x]:+.1f})" for x in Xs))
    d = tt.dropna(subset=[K, "adh"] + C).copy(); cols = [K, "adh"] + C
    dm = d[cols] - d.groupby("bloque")[cols].transform("mean"); f = L.ols_hc3(dm[K], dm[["adh"] + [c for c in C if c != "idx_cronologico"]]); print(f"{nm:5s} efectos fijos por bloque de 20 batches: adh {f.params.adh:+.2f} (t {f.tvalues.adh:+.1f}) n {len(d)}")
# prospectivo con la innovacion
P = W7.preparar(t.dropna(subset=["adh_innov"]), ["adh_innov"]); X, Cc, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]; rng = np.random.default_rng(0); blo = W7.bloques_cronologicos(n, 20)
def wf(Xe, W=100, paso=10):
    sc = np.full(n, np.nan); s = W
    while s < n:
        q = W7.ols_classic(np.column_stack([Cc[s-W:s], Xe[s-W:s]]), y[s-W:s]); sc[s:s+paso] = Xe[s:s+paso] @ q["beta"][-1:]; s += paso
    return sc
o = W7.stats_final(wf(X), y, Cc); nul = np.array([W7.stats_final(wf(W7.permutar_bloques(X, blo, rng)), y, Cc)["t"] for _ in range(2000)])
print("PROSPECTIVO con la innovacion de adherencia: pendiente %.2f t %.2f spearman %.3f p_perm %.4f | placebo lead1 t %.2f, lag1 t %.2f" % (o["pendiente"], o["t"], o["spearman"], (np.sum(nul >= o["t"]) + 1) / 2001,
      W7.stats_final(wf(np.roll(X, -1, axis=0)), y, Cc)["t"], W7.stats_final(wf(np.roll(X, 1, axis=0)), y, Cc)["t"]))
