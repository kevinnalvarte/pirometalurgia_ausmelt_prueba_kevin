"""E15-02 punto 3: reproduce la prueba prospectiva y corrige por multiplicidad. Grid W in {60,80,100,120,
150,expansiva} x {JS, sin contraccion(OLS)} = 12 combos. Para cada nula (bloques de 10/20(primaria)/40,
circular exhaustivo, signo por bloque de 20) se calcula p por combo y p CORREGIDA (maximo t sobre las 12
combinaciones bajo la MISMA nula). Exposicion = d_gn, d_carbon tal como E15-01 (estandarizado por DEV,
para reproducir exacto)."""
import sys, time
sys.path.insert(0, "experimentos/v13")
import numpy as np, pandas as pd
import e15_02_lib as E
import e11_07_lib as W7
import e13_03_lib as L13

t0 = time.time()
_, t = E.cargar()
P0 = W7.preparar(t, ["d_gn", "d_carbon"])
X, C, y, n = P0["Xexp"], P0["Xctrl"], P0["y"], P0["n"]
Ws = [60, 80, 100, 120, 150, None]
MODOS = ["js", "ols"]
REPS = 500


def start_de(Wd):
    return 100 if Wd in (60, 80) else max(Wd or 100, 100)


def nulas_de(null_kind, rng):
    if null_kind == "circular":
        return L13.circular_shifts_todos(X, margen=15)
    if null_kind.startswith("bloque"):
        tam = int(null_kind.replace("bloque", ""))
        bl = L13.bloques(n, tam, seed=0)
        return [L13.permutar_bloques(X, bl, rng) for _ in range(REPS)]
    if null_kind == "signo20":
        bl = L13.bloques(n, 20, seed=0)
        return [L13.signo_bloques(X, bl, rng) for _ in range(REPS)]
    raise ValueError(null_kind)


filas, resumen = [], []
for null_kind in ("bloque10", "bloque20", "bloque40", "circular", "signo20"):
    rng = np.random.default_rng(0)
    Xperms = nulas_de(null_kind, rng)
    obs_t, nul_t = {}, {}
    for Wd in Ws:
        for modo in MODOS:
            st0 = start_de(Wd)
            sc = L13.wf(X, C, y, Wd, modo=modo, start=st0)
            obs = W7.stats_final(sc, y, C); obs_t[(Wd, modo)] = obs["t"]
            nts = np.array([W7.stats_final(L13.wf(Xp, C, y, Wd, modo=modo, start=st0), y, C)["t"] for Xp in Xperms])
            nul_t[(Wd, modo)] = nts
            p = L13.p_perm(obs["t"], nts)
            filas.append({"nula": null_kind, "W": Wd or "exp", "modo": modo, "n": obs["n"],
                          "pendiente": obs["pendiente"], "t": obs["t"], "n_reps": len(nts), "p_perm": p})
    tmax_obs = max(obs_t.values())
    tmax_nul = np.max(np.column_stack([nul_t[k][:REPS] for k in obs_t]), axis=1)
    p_corr = L13.p_perm(tmax_obs, tmax_nul)
    p_w100_js = [f["p_perm"] for f in filas if f["nula"] == null_kind and f["W"] == 100 and f["modo"] == "js"][0]
    resumen.append({"nula": null_kind, "p_W100_JS": p_w100_js, "p_corregida_12combos": p_corr})
    print(null_kind, "p_W100_JS=%.4f p_corregida(12 combos W x modo)=%.4f" % (p_w100_js, p_corr), f"[{time.time()-t0:.0f}s]")

out = pd.DataFrame(filas); out.to_csv("experimentos/v13/e15_02_03_multiplicidad.csv", index=False)
res = pd.DataFrame(resumen); res.to_csv("experimentos/v13/e15_02_03_resumen.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string())
