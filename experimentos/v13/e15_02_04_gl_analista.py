"""E15-02 punto 4: grados de libertad del analista con JS, W=100 fijo. paso in {5,10,20}; start in
{100,120,150}; controles con/sin idx_cronologico, o con espesor en vez de idx; exposiciones: dG+dC (full,
receta original E15-01), solo dG, solo dC, dG solo en R2-R3, dG solo en R0-R1 (desde el pkl de escalones),
y dG+dC+dO2+dAire. Reusa la MISMA permutacion (bloques de 20) para combos con igual n."""
import sys, time
sys.path.insert(0, "experimentos/v13")
import numpy as np, pandas as pd
import e15_02_lib as E
import e11_07_lib as W7
import e13_03_lib as L13

t0 = time.time()
r, t = E.cargar()
REPS = 300

t = t.copy()
t["d_gn_R01"] = E.exposicion_ordenes(r, t, "gn", [0, 1])
t["d_gn_R23"] = E.exposicion_ordenes(r, t, "gn", [2, 3])

CTRL_FULL = list(W7.C_SIN_ESPESOR)
CTRL_SIN_IDX = [c for c in CTRL_FULL if c != "idx_cronologico"]
CTRL_ESPESOR = CTRL_SIN_IDX + ["espesor_ladrillo_norm_mm"]
CTRL_SETS = {"full(+idx)": CTRL_FULL, "sin_idx": CTRL_SIN_IDX, "espesor_en_vez_idx": CTRL_ESPESOR}
XCOL_SETS = {"dG+dC": ["d_gn", "d_carbon"], "solo_dG": ["d_gn"], "solo_dC": ["d_carbon"],
             "dG_R01": ["d_gn_R01"], "dG_R23": ["d_gn_R23"], "dG+dC+dO2+dAire": ["d_gn", "d_carbon", "d_o2", "d_aire"]}

perm_cache = {}  # n -> (Xexp_base, lista de permutaciones, cols_base)

filas = []
for xcol_name, xcols in XCOL_SETS.items():
    for ctrl_name, ctrl in CTRL_SETS.items():
        Pc = W7.preparar(t, xcols, controles=ctrl)
        Xe, Cc, yy, nn = Pc["Xexp"], Pc["Xctrl"], Pc["y"], Pc["n"]
        if nn not in perm_cache:
            bl_i = L13.bloques(nn, 20, seed=0)
            perm_cache[nn] = (None, bl_i)
        rng_i, bl_i = perm_cache[nn]
        rng_local = np.random.default_rng(0)
        Xperms = [L13.permutar_bloques(Xe, bl_i, rng_local) for _ in range(REPS)]
        for paso in (5, 10, 20):
            for start in (100, 120, 150):
                sc = L13.wf(Xe, Cc, yy, 100, paso=paso, modo="js", start=start)
                obs = W7.stats_final(sc, yy, Cc)
                if obs["n"] < 10 or np.isnan(obs["t"]):
                    continue
                nts = np.array([W7.stats_final(L13.wf(Xp, Cc, yy, 100, paso=paso, modo="js", start=start), yy, Cc)["t"] for Xp in Xperms])
                p = L13.p_perm(obs["t"], nts)
                filas.append({"xcols": xcol_name, "controles": ctrl_name, "paso": paso, "start": start,
                              "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_perm": p})
        print(xcol_name, ctrl_name, f"[{time.time()-t0:.0f}s]")

out = pd.DataFrame(filas)
out.to_csv("experimentos/v13/e15_02_04_gl_analista.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string())
frac01 = (out["p_perm"] < 0.01).mean(); frac05 = (out["p_perm"] < 0.05).mean()
print(f"\nn combinaciones: {len(out)} | fraccion p<0.01: {frac01:.2f} | fraccion p<0.05: {frac05:.2f}")
for xn in XCOL_SETS:
    sub = out[out.xcols == xn]
    print(xn, "p<0.01:", round((sub.p_perm < 0.01).mean(), 2), "p<0.05:", round((sub.p_perm < 0.05).mean(), 2), "n_combos:", len(sub))
