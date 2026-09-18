"""E15-02 punto 5: placebos con JS (W=100). lead1, lead3, lag1 (exposicion de otro batch usada "como si
fuera" la actual); kpi_prev (outcome = KPI del batch anterior). Contraste final "limpio de persistencia":
agrega KPI_prev, d_gn_prev, d_carbon_prev como CONTROLES en la regresion final (no en el ajuste de beta)."""
import sys, time
sys.path.insert(0, "experimentos/v13")
import numpy as np, pandas as pd
import e15_02_lib as E
import e11_07_lib as W7
import e13_03_lib as L13

t0 = time.time()
_, t = E.cargar()
P = W7.preparar(t, ["d_gn", "d_carbon"])
X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
bl20 = L13.bloques(n, 20, seed=0)
rng = np.random.default_rng(0)
Xperms = [L13.permutar_bloques(X, bl20, rng) for _ in range(500)]


def shift_rows(A, k):
    Ash = np.roll(A, -k, axis=0)
    if k > 0:
        Ash[-k:] = np.nan
    elif k < 0:
        Ash[:-k] = np.nan
    return Ash


filas = []
for nombre, k in (("lead1", 1), ("lead3", 3), ("lag1", -1)):
    Xk = shift_rows(X, k)
    m = ~np.isnan(Xk).any(axis=1)
    Xk2, C2, y2 = Xk[m], C[m], y[m]
    sc = L13.wf(Xk2, C2, y2, 100, modo="js")
    obs = W7.stats_final(sc, y2, C2)
    nts = np.array([W7.stats_final(L13.wf(shift_rows(Xp, k)[m], C2, y2, 100, modo="js"), y2, C2)["t"] for Xp in Xperms])
    p = L13.p_perm(obs["t"], nts)
    filas.append({"placebo": nombre, "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_uni_perm": p})
    print(nombre, obs["pendiente"], obs["t"], f"[{time.time()-t0:.0f}s]")

# kpi_prev: outcome = KPI del batch ANTERIOR, exposiciones = las actuales (score normal)
y_prev = shift_rows(y.reshape(-1, 1), -1).ravel()
m = ~np.isnan(y_prev)
sc = L13.wf(X[m], C[m], y[m], 100, modo="js")
obs = W7.stats_final(sc, y_prev[m], C[m])
nts = np.array([W7.stats_final(L13.wf(Xp[m], C[m], y[m], 100, modo="js"), y_prev[m], C[m])["t"] for Xp in Xperms])
p = L13.p_perm(obs["t"], nts)
filas.append({"placebo": "kpi_prev", "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_uni_perm": p})
print("kpi_prev", obs["pendiente"], obs["t"], f"[{time.time()-t0:.0f}s]")

out = pd.DataFrame(filas)
out.to_csv("experimentos/v13/e15_02_05_placebos.csv", index=False)
print(out.round(4).to_string())

# contraste final "limpio de persistencia": controlar por KPI_prev, d_gn_prev, d_carbon_prev
xg_prev = shift_rows(X[:, [0]], -1).ravel(); xc_prev = shift_rows(X[:, [1]], -1).ravel()
y_prev_all = shift_rows(y.reshape(-1, 1), -1).ravel()
m2 = ~np.isnan(xg_prev) & ~np.isnan(y_prev_all)
C_ext = np.column_stack([C[m2], y_prev_all[m2], xg_prev[m2], xc_prev[m2]])
sc_full = L13.wf(X, C, y, 100, modo="js")
obs2 = W7.stats_final(sc_full[m2], y[m2], C_ext)
nts2 = np.array([W7.stats_final(L13.wf(Xp, C, y, 100, modo="js")[m2], y[m2], C_ext)["t"] for Xp in Xperms])
p2 = L13.p_perm(obs2["t"], nts2)
print("limpio_de_persistencia: n=%d pendiente=%.3f t=%.3f p_perm=%.4f  [%ds]" % (obs2["n"], obs2["pendiente"], obs2["t"], p2, time.time() - t0))
pd.DataFrame([{"variante": "limpio_de_persistencia", "n": obs2["n"], "pendiente": obs2["pendiente"], "t": obs2["t"], "p_perm": p2}]).to_csv(
    "experimentos/v13/e15_02_05_limpio_persistencia.csv", index=False)
