"""E21-01 (Fable): verificacion final de la condicion obligatoria para la recomendacion de v11 ("GN <= receta").
Medida de NO-adherencia del operador: exceso medio de GN sobre la receta en R0-R3 (Nm3/min). Adherencia = -exceso.
(1) KPI ~ adherencia (OLS HC3; DEV, lockbox, total; dosis-respuesta por cuartiles); (2) prueba prospectiva para 6 ventanas con p corregida por multiplicidad
(estadistico maximo bajo permutacion por bloques) y desplazamiento circular; (3) jackknife; (4) placebos (lead 1, lead 3, KPI anterior)."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import v9_lib as L, e11_07_lib as W7
t = pd.read_csv("experimentos/v16/e18_07_tabla.csv", index_col=0).sort_values("idx_cronologico"); K = L.KPI; C = W7.C_SIN_ESPESOR; t["adh"] = -t.exc_todo
for nm, tt in (("DEV", t[t.es_dev]), ("LOCKBOX", t[~t.es_dev]), ("TOTAL", t)):
    d = tt.dropna(subset=[K, "adh"] + C); f = L.ols_hc3(d[K], d[["adh"] + C]); print(f"{nm:8s} n {len(d)} KPI por +1 Nm3/min de adherencia: {f.params.adh:+.3f} [{f.conf_int().loc['adh',0]:+.3f}, {f.conf_int().loc['adh',1]:+.3f}] t {f.tvalues.adh:+.2f} p {f.pvalues.adh:.4f}")
d = t.dropna(subset=[K, "adh"] + C).copy(); d["res"] = L.ols_hc3(d[K], d[C]).resid; q = pd.qcut(d.adh, 4, labels=["Q1 (mas exceso)", "Q2", "Q3", "Q4 (mas adherente)"])
print(d.groupby(q, observed=True).agg(n=("res", "size"), exceso_medio=("exc_todo", "mean"), kpi_residual=("res", "mean")).round(2).to_string())
def wf(Xe, Cc, yy, W, paso=10):
    n = len(yy); sc = np.full(n, np.nan); s = 100 if (W is None or W < 100) else W
    while s < n:
        lo = 0 if W is None else s - W; q_ = W7.ols_classic(np.column_stack([Cc[lo:s], Xe[lo:s]]), yy[lo:s]); sc[s:s+paso] = Xe[s:s+paso] @ q_["beta"][-1:]; s += paso
    return sc
P = W7.preparar(t, ["adh"]); X, Cc, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]; rng = np.random.default_rng(0); blo = W7.bloques_cronologicos(n, 20); Ws = [60, 80, 100, 120, 150, None]
perms = [W7.permutar_bloques(X, blo, rng) for _ in range(2000)]; obs = {}; nul = {}
for W in Ws:
    sc = wf(X, Cc, y, W); obs[W] = W7.stats_final(sc, y, Cc); nul[W] = np.array([W7.stats_final(wf(Xp, Cc, y, W), y, Cc)["t"] for Xp in perms])
    dv = W7.stats_final(sc, y, Cc, P["es_dev"]); lb = W7.stats_final(sc, y, Cc, ~P["es_dev"])
    print(f"W={str(W):5s} n {obs[W]['n']} pendiente {obs[W]['pendiente']:.2f} t {obs[W]['t']:.2f} spearman {obs[W]['spearman']:.3f} p_perm {(np.sum(nul[W] >= obs[W]['t'])+1)/2001:.4f} | DEV t {dv['t']:.2f} | LB t {lb['t']:.2f}")
tmax = max(o["t"] for o in obs.values()); nmax = np.max(np.column_stack([nul[W] for W in Ws]), axis=1); print("p corregida por multiplicidad sobre las 6 ventanas: %.4f" % ((np.sum(nmax >= tmax) + 1) / 2001))
sc = wf(X, Cc, y, 100); o = W7.stats_final(sc, y, Cc); circ = np.array([W7.stats_final(wf(np.roll(X, k, axis=0), Cc, y, 100), y, Cc)["t"] for k in range(15, n - 15)]); print("p circular (W=100): %.4f" % ((np.sum(circ >= o["t"]) + 1) / (len(circ) + 1)))
for m in (5, 10, 20, 40):
    ts = []
    for _ in range(300):
        keep = np.sort(rng.choice(n, n - m, replace=False)); ts.append(W7.stats_final(wf(X[keep], Cc[keep], y[keep], 100), y[keep], Cc[keep])["t"])
    print(f"jackknife -{m}: t P5 {np.percentile(ts,5):.2f} mediana {np.median(ts):.2f} | % t>1.645: {100*np.mean(np.array(ts)>1.645):.0f} | % t>2.33: {100*np.mean(np.array(ts)>2.33):.0f}")
for nm, Xp in (("lead 1", np.roll(X, -1, axis=0)), ("lead 3", np.roll(X, -3, axis=0)), ("lag 1", np.roll(X, 1, axis=0))):
    print("placebo", nm, "t %.2f" % W7.stats_final(wf(Xp, Cc, y, 100), y, Cc)["t"])
print("placebo outcome = KPI anterior: t %.2f" % W7.stats_final(wf(X, Cc, np.roll(y, 1), 100), np.roll(y, 1), Cc)["t"])
