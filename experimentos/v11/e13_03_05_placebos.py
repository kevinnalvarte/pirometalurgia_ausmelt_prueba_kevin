"""E13-03 punto 5: placebos con JS (W=100). lead1, lead3, lag1 (exposicion de otro batch usada "como si
fuera" la actual); kpi_prev (outcome = KPI del batch anterior). Y el puntaje "limpio de persistencia":
agregar KPI_prev, xG_prev, xC_prev como controles en la regresion FINAL (no en el ajuste de beta)."""
import sys, time
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
import e13_03_lib as L

t0 = time.time()
tab = W7.cargar_tabla(); P = W7.preparar(tab, ["xG", "xC"])
X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
bl20 = L.bloques(n, 20, seed=0)
rng = np.random.default_rng(0)
Xperms = [L.permutar_bloques(X, bl20, rng) for _ in range(500)]


def shift_rows(A, k):
    """Desplaza FILAS k posiciones (k>0 adelanta = usa el valor de k batches despues); recorta bordes con NaN."""
    Ash = np.roll(A, -k, axis=0)
    if k > 0:
        Ash[-k:] = np.nan
    elif k < 0:
        Ash[:-k] = np.nan
    return Ash


filas = []
# --- lead1, lead3, lag1: exposicion de otro batch, outcome = KPI del batch actual
for nombre, k in (("lead1", 1), ("lead3", 3), ("lag1", -1)):
    Xk = shift_rows(X, k)
    m = ~np.isnan(Xk).any(axis=1)
    Xk2, C2, y2 = Xk[m], C[m], y[m]
    sc = L.wf(Xk2, C2, y2, 100, modo="js")
    obs = W7.stats_final(sc, y2, C2)
    nts = np.array([W7.stats_final(L.wf(shift_rows(Xp, k)[m], C2, y2, 100, modo="js"), y2, C2)["t"] for Xp in Xperms])
    p = L.p_perm(obs["t"], nts)  # mismo signo/direccion que la prueba principal (una cola, positiva)
    filas.append({"placebo": nombre, "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_uni_perm": p})
    print(nombre, obs["pendiente"], obs["t"], f"[{time.time()-t0:.0f}s]")

# --- kpi_prev: outcome = KPI del batch ANTERIOR, exposiciones = las actuales (score normal)
y_prev = shift_rows(y.reshape(-1, 1), -1).ravel()  # y_prev[i] = y[i-1]
m = ~np.isnan(y_prev)
sc = L.wf(X[m], C[m], y[m], 100, modo="js")  # beta se sigue ajustando/aplicando sobre la muestra recortada
obs = W7.stats_final(sc, y_prev[m], C[m])
nts = np.array([W7.stats_final(L.wf(Xp[m], C[m], y[m], 100, modo="js"), y_prev[m], C[m])["t"] for Xp in Xperms])
p = L.p_perm(obs["t"], nts)
filas.append({"placebo": "kpi_prev", "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_uni_perm": p})
print("kpi_prev", obs["pendiente"], obs["t"], f"[{time.time()-t0:.0f}s]")

out = pd.DataFrame(filas)
out.to_csv("experimentos/v11/e13_03_05_placebos.csv", index=False)
print(out.round(4).to_string())

# --- puntaje limpio de persistencia: agregar y_prev, xG_prev, xC_prev como CONTROLES en la regresion final
xg_prev = shift_rows(X[:, [0]], -1).ravel(); xc_prev = shift_rows(X[:, [1]], -1).ravel()
y_prev_all = shift_rows(y.reshape(-1, 1), -1).ravel()
m2 = ~np.isnan(xg_prev) & ~np.isnan(y_prev_all)
C_ext = np.column_stack([C[m2], y_prev_all[m2], xg_prev[m2], xc_prev[m2]])
sc_full = L.wf(X, C, y, 100, modo="js")
obs2 = W7.stats_final(sc_full[m2], y[m2], C_ext)
nts2 = np.array([W7.stats_final(L.wf(Xp, C, y, 100, modo="js")[m2], y[m2], C_ext)["t"] for Xp in Xperms])
p2 = L.p_perm(obs2["t"], nts2)
print("limpio_de_persistencia: n=%d pendiente=%.3f t=%.3f p_perm=%.4f  [%ds]" % (obs2["n"], obs2["pendiente"], obs2["t"], p2, time.time() - t0))
pd.DataFrame([{"variante": "limpio_de_persistencia", "n": obs2["n"], "pendiente": obs2["pendiente"], "t": obs2["t"], "p_perm": p2}]).to_csv(
    "experimentos/v11/e13_03_05_limpio_persistencia.csv", index=False)
