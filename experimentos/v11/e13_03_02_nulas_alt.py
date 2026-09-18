"""E13-03 punto 2: nulas alternativas para JS (Fable) -- bloques de 10/20/40, desplazamiento circular
COMPLETO (todos los k en [15, n-15]), permutacion de SIGNO por bloque de 20. Para cada una: p_perm en
W=100 (primaria) y p corregida por multiplicidad sobre las 6 ventanas (max t)."""
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


def start_de(Wd):
    return 100 if Wd in (60, 80) else max(Wd or 100, 100)


def nulas_de(null_kind, rng):
    """Devuelve lista de arrays Xexp permutados segun el tipo de nula."""
    if null_kind == "circular":
        return L.circular_shifts_todos(X, margen=15)  # exhaustivo, ~n-30
    if null_kind.startswith("bloque"):
        tam = int(null_kind.replace("bloque", ""))
        bl = L.bloques(n, tam, seed=0)
        return [L.permutar_bloques(X, bl, rng) for _ in range(REPS)]
    if null_kind == "signo20":
        bl = L.bloques(n, 20, seed=0)
        return [L.signo_bloques(X, bl, rng) for _ in range(REPS)]
    raise ValueError(null_kind)


filas = []
resumen = []
for null_kind in ("bloque10", "bloque20", "bloque40", "circular", "signo20"):
    rng = np.random.default_rng(0)
    Xperms = nulas_de(null_kind, rng)
    obs_t = {}; nul_t = {}
    for Wd in Ws:
        st0 = start_de(Wd)
        sc = L.wf(X, C, y, Wd, modo="js", start=st0)
        obs = W7.stats_final(sc, y, C); obs_t[Wd] = obs["t"]
        nts = np.array([W7.stats_final(L.wf(Xp, C, y, Wd, modo="js", start=st0), y, C)["t"] for Xp in Xperms])
        nul_t[Wd] = nts
        p = L.p_perm(obs["t"], nts)
        filas.append({"nula": null_kind, "W": Wd or "exp", "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "n_reps": len(nts), "p_perm": p})
    tmax_obs = max(obs_t.values())
    tmax_nul = np.max(np.column_stack([nul_t[w][:REPS] for w in Ws]), axis=1)
    p_corr = L.p_perm(tmax_obs, tmax_nul)
    resumen.append({"nula": null_kind, "p_W100": [f["p_perm"] for f in filas if f["nula"] == null_kind and f["W"] == 100][0], "p_corregida_6W": p_corr})
    print(null_kind, "p_W100=%.4f p_corregida(6W)=%.4f" % (resumen[-1]["p_W100"], p_corr), f"[{time.time()-t0:.0f}s]")

pd.DataFrame(filas).to_csv("experimentos/v11/e13_03_02_nulas_alt.csv", index=False)
res = pd.DataFrame(resumen); res.to_csv("experimentos/v11/e13_03_02_nulas_alt_resumen.csv", index=False)
print(res.round(4).to_string())
