"""E13-01 (Fable): contraccion James-Stein (parte positiva) del valor de cada control, fijada a priori y sin hiperparametros:
   beta~_j = beta^_j * max(0, 1 - 1/t_j^2)   (t_j = beta^_j / se_j en la ventana).
Motivo: la pendiente del KPI sobre el puntaje (calibracion 0.44-0.53) esta atenuada por el error de estimacion de beta^ (dilucion): el
estimador de Bayes empirico del efecto real dado (beta^, se) es beta^*(1 - se^2/beta^2)+ -> puntaje calibrado por construccion.
Evalua: calibracion, t, Spearman, p por permutacion en bloques (re-ejecutando todo), correccion por multiplicidad sobre W, lockbox, y
FRAGILIDAD (jackknife: quitar 5 batches al azar 300 veces)."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
t = W7.cargar_tabla(); P = W7.preparar(t, ["xG", "xC"]); X, C, y, n, dev = P["Xexp"], P["Xctrl"], P["y"], P["n"], P["es_dev"]
def wf(Xe, Cc, yy, W=100, paso=10, modo="js", start=None):
    nn = len(yy); sc = np.full(nn, np.nan); s = start or (W or 100)
    while s < nn:
        lo = 0 if W is None else s - W; Z = np.column_stack([Cc[lo:s], Xe[lo:s]]); r = W7.ols_classic(Z, yy[lo:s]); b, se = r["beta"][-2:], r["se"][-2:]
        if modo == "js": b = b * np.clip(1 - (se / np.where(b == 0, 1e-9, b)) ** 2, 0, None)
        sc[s:s + paso] = Xe[s:s + paso] @ b; s += paso
    return sc
rng = np.random.default_rng(0); blo = W7.bloques_cronologicos(n, 20); Ws = [60, 80, 100, 120, 150, None]; filas = []
perm = [W7.permutar_bloques(X, blo, rng) for _ in range(2000)]
for modo in ("ols", "js"):
    obs = {}; nul = {}
    for Wd in Ws:
        sc = wf(X, C, y, Wd, modo=modo, start=max(Wd or 100, 100) if Wd != 60 and Wd != 80 else 100)
        m = ~np.isnan(sc); obs[Wd] = W7.stats_final(sc, y, C); nul[Wd] = np.array([W7.stats_final(wf(Xp, C, y, Wd, modo=modo, start=max(Wd or 100, 100) if Wd not in (60, 80) else 100), y, C)["t"] for Xp in perm[:1000 if Wd != 100 else 2000]])
        for nm, mk in (("todo", None), ("DEV", dev), ("LB", ~dev)):
            st = W7.stats_final(sc, y, C, mk); st.update({"modo": modo, "W": Wd or "exp", "muestra": nm, "p_perm": (np.sum(nul[Wd] >= obs[Wd]["t"]) + 1) / (len(nul[Wd]) + 1) if nm == "todo" else np.nan}); filas.append(st)
    tmax_obs = max(o["t"] for o in obs.values()); tmax_nul = np.max(np.column_stack([nul[w][:1000] for w in Ws]), axis=1)
    print(modo, "p corregida por multiplicidad (max sobre W):", round((np.sum(tmax_nul >= tmax_obs) + 1) / 1001, 4), "| p W=100:", round((np.sum(nul[100] >= obs[100]["t"]) + 1) / (len(nul[100]) + 1), 4))
out = pd.DataFrame(filas); out.to_csv("experimentos/v11/e13_01_contraccion_js.csv", index=False); pd.set_option("display.width", 200)
print(out[["modo", "W", "muestra", "n", "pendiente", "se", "t", "p_perm", "spearman", "sd_score"]].round(3).to_string())
# fragilidad: quitar 5 batches al azar
for modo in ("ols", "js"):
    ts, sl = [], []
    for _ in range(300):
        keep = np.sort(rng.choice(n, n - 5, replace=False)); sc = wf(X[keep], C[keep], y[keep], 100, modo=modo); st = W7.stats_final(sc, y[keep], C[keep]); ts.append(st["t"]); sl.append(st["pendiente"])
    print(modo, "jackknife-5: t P5 %.2f P50 %.2f P95 %.2f | pendiente P5 %.2f P50 %.2f P95 %.2f | %% t>1.645: %.0f" % (*np.percentile(ts, [5, 50, 95]), *np.percentile(sl, [5, 50, 95]), 100 * np.mean(np.array(ts) > 1.645)))
