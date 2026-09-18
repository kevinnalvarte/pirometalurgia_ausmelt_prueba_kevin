"""E13-03 punto 4: grados de libertad del analista con contraccion JS, W=100 fijo.
paso in {5,10,20}; start in {100,120,150}; controles con/sin idx_cronologico, o con espesor en vez de
idx; exposiciones {xG,xC} / solo xG / solo xC. Se reutiliza LA MISMA permutacion (bloques de 20, 500
rep) sobre Xexp para todas las combinaciones (solo cambian paso/start/controles/columnas de Xexp, no la
nula en si)."""
import sys, time
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
import e13_03_lib as L

t0 = time.time()
tab = W7.cargar_tabla()
REPS = 500

CTRL_FULL = list(W7.C_SIN_ESPESOR)                       # incluye idx_cronologico
CTRL_SIN_IDX = [c for c in CTRL_FULL if c != "idx_cronologico"]
CTRL_ESPESOR = CTRL_SIN_IDX + ["espesor_ladrillo_norm_mm"]  # reemplaza idx por espesor
CTRL_SETS = {"full(+idx)": CTRL_FULL, "sin_idx": CTRL_SIN_IDX, "espesor_en_vez_idx": CTRL_ESPESOR}
XCOL_SETS = {"xG+xC": ["xG", "xC"], "solo_xG": ["xG"], "solo_xC": ["xC"]}

# Xexp base para el generador de permutaciones (bloques definidos sobre la muestra mas grande, xG+xC)
P0 = W7.preparar(tab, ["xG", "xC"], controles=CTRL_FULL)
n0 = P0["n"]; bl20 = L.bloques(n0, 20, seed=0)
rng = np.random.default_rng(0)
Xperms_full = [L.permutar_bloques(P0["Xexp"], bl20, rng) for _ in range(REPS)]

filas = []
for xcol_name, xcols in XCOL_SETS.items():
    for ctrl_name, ctrl in CTRL_SETS.items():
        Pc = W7.preparar(tab, xcols, controles=ctrl)
        Xe, Cc, yy, nn = Pc["Xexp"], Pc["Xctrl"], Pc["y"], Pc["n"]
        # nula propia (mismos bloques, misma semilla, pero recortada a las columnas usadas); si nn != n0
        # (dropna distinto por set de controles) se regenera la lista de permutaciones para esa muestra.
        if nn == n0:
            if xcols == ["xG"]:
                Xperms = [Xp[:, [0]] for Xp in Xperms_full]
            elif xcols == ["xC"]:
                Xperms = [Xp[:, [1]] for Xp in Xperms_full]
            else:
                Xperms = Xperms_full
        else:
            rng_i = np.random.default_rng(0); bl_i = L.bloques(nn, 20, seed=0)
            Xperms = [L.permutar_bloques(Xe, bl_i, rng_i) for _ in range(REPS)]
        for paso in (5, 10, 20):
            for start in (100, 120, 150):
                sc = L.wf(Xe, Cc, yy, 100, paso=paso, modo="js", start=start)
                obs = W7.stats_final(sc, yy, Cc)
                if obs["n"] < 10 or np.isnan(obs["t"]):
                    continue
                nts = np.array([W7.stats_final(L.wf(Xp, Cc, yy, 100, paso=paso, modo="js", start=start), yy, Cc)["t"] for Xp in Xperms])
                p = L.p_perm(obs["t"], nts)
                filas.append({"xcols": xcol_name, "controles": ctrl_name, "paso": paso, "start": start,
                              "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_perm": p})
        print(xcol_name, ctrl_name, f"[{time.time()-t0:.0f}s]")

out = pd.DataFrame(filas)
out.to_csv("experimentos/v11/e13_03_04_gl_analista.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string())
frac01 = (out["p_perm"] < 0.01).mean(); frac05 = (out["p_perm"] < 0.05).mean()
print(f"n combinaciones: {len(out)} | fraccion p<0.01: {frac01:.2f} | fraccion p<0.05: {frac05:.2f}")
