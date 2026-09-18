"""E13-03 punto 3: familia de contracciones (sin, JS, umbral duro 1/1.5/2, lineal, Bayes empirico
online, JS conjunta) x 6 ventanas. Para cada celda: calibracion, t, p_perm (nula bloques de 20, 500 rep).
Y p corregida por multiplicidad sobre TODA la familia (contracciones x ventanas): max t bajo la MISMA
permutacion para todas las celdas (comparacion pareada, no 42 nulas independientes)."""
import sys, time
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
import e13_03_lib as L

t0 = time.time()
tab = W7.cargar_tabla(); P = W7.preparar(tab, ["xG", "xC"])
X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
Ws = [60, 80, 100, 120, 150, None]
REPS = 500
bl20 = L.bloques(n, 20, seed=0)
rng = np.random.default_rng(0)
Xperms = [L.permutar_bloques(X, bl20, rng) for _ in range(REPS)]


def start_de(Wd):
    return 100 if Wd in (60, 80) else max(Wd or 100, 100)


filas = []
obs_grid = {}   # (modo,W) -> t
nul_grid = {}   # (modo,W) -> array de REPS t's
for modo in L.MODOS:
    for Wd in Ws:
        st0 = start_de(Wd)
        sc = L.wf(X, C, y, Wd, modo=modo, start=st0)
        obs = W7.stats_final(sc, y, C)
        obs_grid[(modo, Wd)] = obs["t"]
        nts = np.array([W7.stats_final(L.wf(Xp, C, y, Wd, modo=modo, start=st0), y, C)["t"] for Xp in Xperms])
        nul_grid[(modo, Wd)] = nts
        p = L.p_perm(obs["t"], nts)
        filas.append({"modo": modo, "W": Wd or "exp", "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_perm": p})
    print(modo, "listo", f"[{time.time()-t0:.0f}s]")

out = pd.DataFrame(filas)
out.to_csv("experimentos/v11/e13_03_03_familia.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string())

# p corregida por multiplicidad sobre TODA la familia (comparacion pareada: misma permutacion j
# para todas las 48 celdas -> max_t bajo H0 por replica)
claves = list(obs_grid.keys())
tmax_obs = max(obs_grid.values())
tmax_nul = np.max(np.column_stack([nul_grid[k] for k in claves]), axis=1)
p_fam = L.p_perm(tmax_obs, tmax_nul)
mejor = max(claves, key=lambda k: obs_grid[k])
print("t max observado en la familia:", round(tmax_obs, 3), "en", mejor)
print("p corregida por multiplicidad sobre TODA la familia (contracciones x ventanas):", round(p_fam, 4))
print("< 0.05:", p_fam < 0.05, " | < 0.01:", p_fam < 0.01)
