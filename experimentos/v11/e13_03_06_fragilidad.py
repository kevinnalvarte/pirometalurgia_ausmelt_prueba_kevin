"""E13-03 punto 6: fragilidad con JS (W=100). Jackknife quitando 5/10/20 batches al azar (300 rep c/u):
distribucion de t y calibracion. Leave-one-block-out (bloques cronologicos de 20): ¿algun bloque sostiene
todo el resultado?"""
import sys, time
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
import e13_03_lib as L

t0 = time.time()
tab = W7.cargar_tabla(); P = W7.preparar(tab, ["xG", "xC"])
X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
rng = np.random.default_rng(0)

filas = []
for k in (5, 10, 20):
    ts, sl = [], []
    for _ in range(300):
        keep = np.sort(rng.choice(n, n - k, replace=False))
        sc = L.wf(X[keep], C[keep], y[keep], 100, modo="js")
        st = W7.stats_final(sc, y[keep], C[keep])
        ts.append(st["t"]); sl.append(st["pendiente"])
    ts = np.array(ts); sl = np.array(sl)
    p5, p50, p95 = np.percentile(ts, [5, 50, 95])
    s5, s50, s95 = np.percentile(sl, [5, 50, 95])
    pct = 100 * np.mean(ts > 1.645)
    filas.append({"quitando": k, "t_P5": p5, "t_P50": p50, "t_P95": p95, "pend_P5": s5, "pend_P50": s50, "pend_P95": s95, "pct_t_gt_1.645": pct})
    print(f"jackknife-{k}: t P5 {p5:.2f} P50 {p50:.2f} P95 {p95:.2f} | pend P50 {s50:.2f} | %t>1.645 {pct:.0f}  [{time.time()-t0:.0f}s]")

pd.DataFrame(filas).to_csv("experimentos/v11/e13_03_06_jackknife.csv", index=False)

# leave-one-block-out (bloques cronologicos de 20)
bl20 = L.bloques(n, 20, seed=0)
obs_full = W7.stats_final(L.wf(X, C, y, 100, modo="js"), y, C)
filas_b = []
for i, b in enumerate(bl20):
    keep = np.setdiff1d(np.arange(n), b)
    sc = L.wf(X[keep], C[keep], y[keep], 100, modo="js")
    st = W7.stats_final(sc, y[keep], C[keep])
    filas_b.append({"bloque_excluido": i, "batches": f"{b[0]}-{b[-1]}", "n": st["n"], "pendiente": st["pendiente"], "t": st["t"]})
outb = pd.DataFrame(filas_b)
outb.to_csv("experimentos/v11/e13_03_06_loo_bloques.csv", index=False)
print("\nObservado completo: pendiente %.3f t %.3f" % (obs_full["pendiente"], obs_full["t"]))
print(outb.round(3).to_string())
print("t min/max al excluir un bloque:", round(outb["t"].min(), 3), round(outb["t"].max(), 3))
print("cuantos bloques excluidos hacen t<1.645:", int((outb["t"] < 1.645).sum()), "de", len(outb))
