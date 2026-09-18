"""E12-10 (Fable): estimacion ROBUSTA de beta (el ruido del KPI es de pesaje/atribucion: colas pesadas?). Walk-forward W=100, paso 10:
OLS vs Huber (IRLS) vs KPI winsorizado con cuantiles del PASADO vs regresion sobre rangos. Misma evaluacion prospectiva + permutacion."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import e11_07_lib as W7
t = W7.cargar_tabla(); P = W7.preparar(t, ["xG", "xC"]); X, Cc, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
r0 = W7.ols_classic(np.column_stack([Cc, X]), y)["resid"]; print("curtosis exceso residuo KPI: %.2f  | %% |res|>2.5 sd: %.1f" % (stats.kurtosis(r0), 100 * (np.abs(r0) > 2.5 * r0.std()).mean()))
def huber(Z, yy, k=1.345, it=30):
    b = np.linalg.lstsq(Z, yy, rcond=None)[0]
    for _ in range(it):
        r = yy - Z @ b; s = np.median(np.abs(r - np.median(r))) / 0.6745 + 1e-9; w = np.minimum(1, k * s / np.maximum(np.abs(r), 1e-9)); sw = np.sqrt(w)
        b = np.linalg.lstsq(Z * sw[:, None], yy * sw, rcond=None)[0]
    return b
def wf(metodo, Xe):
    sc = np.full(n, np.nan)
    for s in range(100, n, 10):
        Z = np.column_stack([np.ones(100), Cc[s-100:s], Xe[s-100:s]]); yy = y[s-100:s].copy()
        if metodo == "winsor": lo, hi = np.percentile(yy, [5, 95]); yy = np.clip(yy, lo, hi)
        if metodo == "rangos": yy = stats.rankdata(yy) / len(yy) * yy.std() * 3.46
        b = huber(Z, yy) if metodo.startswith("huber") else np.linalg.lstsq(Z, yy, rcond=None)[0]
        sc[s:s+10] = Xe[s:s+10] @ b[-2:]
    return sc
rng = np.random.default_rng(0); blo = W7.bloques_cronologicos(n, 20); filas = []
for m in ("ols", "huber", "winsor", "rangos"):
    sc = wf(m, X); obs = W7.stats_final(sc, y, Cc)
    nul = [W7.stats_final(wf(m, W7.permutar_bloques(X, blo, rng)), y, Cc)["t"] for _ in range(300 if m == "huber" else 1000)]
    for nm, mk in (("todo", None), ("DEV", P["es_dev"]), ("LB", ~P["es_dev"])):
        st = W7.stats_final(sc, y, Cc, mk); st.update({"metodo": m, "muestra": nm, "p_perm": (np.sum(np.array(nul) >= obs["t"]) + 1) / (len(nul) + 1) if nm == "todo" else np.nan}); filas.append(st)
    # evaluacion robusta del contraste final: Spearman ya lo es; ademas pendiente Huber del KPI sobre el puntaje
out = pd.DataFrame(filas); out.to_csv("experimentos/v10/e12_10_robusto.csv", index=False); pd.set_option("display.width", 200)
print(out[["metodo", "muestra", "n", "pendiente", "t", "p_uni", "p_perm", "spearman", "sd_score"]].round(3).to_string())
