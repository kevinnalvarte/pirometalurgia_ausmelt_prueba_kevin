"""E13-03 punto 1: reproducir E13-01 (JS, W=100, semilla 0) y repetir con otras 3 semillas de permutacion."""
import sys, time
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
import e13_03_lib as L

t0 = time.time()
tab = W7.cargar_tabla(); P = W7.preparar(tab, ["xG", "xC"])
X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
bl20 = L.bloques(n, 20, seed=0)

sc = L.wf(X, C, y, 100, modo="js")
obs = W7.stats_final(sc, y, C)
filas = []
for seed in (0, 1, 2, 3):
    rng = np.random.default_rng(seed)
    nul = np.array([W7.stats_final(L.wf(L.permutar_bloques(X, bl20, rng), C, y, 100, modo="js"), y, C)["t"] for _ in range(1000)])
    p = L.p_perm(obs["t"], nul)
    filas.append({"seed": seed, "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_perm": p})
    print(f"seed={seed} p_perm={p:.4f}  [{time.time()-t0:.0f}s]")

out = pd.DataFrame(filas)
out.to_csv("experimentos/v11/e13_03_01_repro.csv", index=False)
print(out.round(4).to_string())
