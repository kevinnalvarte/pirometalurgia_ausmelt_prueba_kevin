"""E12-01 tarea 5: prueba prospectiva (e11_07_lib), W=100, paso 10, permutacion en bloques de 20 (1000 reps).
Specs: {xG,xC} vs {xG,xC,zD_R3} vs {xG,xC,zD_R3,zD_F6}. Reporta pendiente/t/spearman pooled, DEV, lockbox aparte."""
import sys, time
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import e11_07_lib as E
import v9_lib as L

SALIDA = L.RAIZ / "experimentos" / "v10"
N_PERM = 1000
SEED = 42

exp = pd.read_csv(SALIDA / "e12_01_exposiciones.csv", index_col=0)
t = E.cargar_tabla().join(exp[["zD_R3", "zD_F6"]], how="left")

SPECS = {"P0_xG_xC": ["xG", "xC"], "P1_+zD_R3": ["xG", "xC", "zD_R3"], "P2_+zD_R3+zD_F6": ["xG", "xC", "zD_R3", "zD_F6"]}

filas = []
for nombre, X in SPECS.items():
    D = E.preparar(t, X)
    Xexp, Xctrl, y, es_dev, n = D["Xexp"], D["Xctrl"], D["y"], D["es_dev"], D["n"]
    score = E.walk_forward_score(Xexp, Xctrl, y, W=100, paso=E.PASO, tmin=0.0)
    for muestra, m in (("TODO", np.ones(n, bool)), ("DEV", es_dev), ("LOCKBOX", ~es_dev)):
        r = E.stats_final(score, y, Xctrl, m)
        r.update({"spec": nombre, "muestra": muestra, "vars": ",".join(X)})
        filas.append(r)
    print(nombre, "n =", n, "n_dev =", es_dev.sum())

tab = pd.DataFrame(filas)[["spec", "vars", "muestra", "n", "pendiente", "se", "t", "p_uni", "spearman", "sd_score"]]
print("\n--- prueba prospectiva (walk-forward W=100, paso 10) ---")
print(tab.round(4).to_string())
tab.to_csv(SALIDA / "e12_01_prospectivo_walkforward.csv", index=False)

# --- permutacion en bloques de 20 (TODO, W=100), por spec
filas_perm = []
for nombre, X in SPECS.items():
    D = E.preparar(t, X)
    Xexp, Xctrl, y, n = D["Xexp"], D["Xctrl"], D["y"], D["n"]
    mask = np.ones(n, bool)
    score = E.walk_forward_score(Xexp, Xctrl, y, W=100, paso=E.PASO, tmin=0.0)
    r = E.stats_final(score, y, Xctrl, mask)
    obs_t, obs_rho = r["t"], r["spearman"]
    bloques = E.bloques_cronologicos(n, 20)
    rng = np.random.default_rng(SEED)
    null_t, null_rho = np.empty(N_PERM), np.empty(N_PERM)
    t0 = time.time()
    for i in range(N_PERM):
        Xp = E.permutar_bloques(Xexp, bloques, rng)
        sp = E.walk_forward_score(Xp, Xctrl, y, W=100, paso=E.PASO, tmin=0.0)
        rp = E.stats_final(sp, y, Xctrl, mask)
        null_t[i], null_rho[i] = rp["t"], rp["spearman"]
    p_t = float((np.sum(null_t >= obs_t) + 1) / (N_PERM + 1))
    p_rho = float((np.sum(null_rho >= obs_rho) + 1) / (N_PERM + 1))
    print(f"{nombre}: obs t={obs_t:.3f} (p_perm={p_t:.4f})  obs spearman={obs_rho:.3f} (p_perm={p_rho:.4f})  [{time.time()-t0:.0f}s]")
    filas_perm.append({"spec": nombre, "n": n, "obs_t": obs_t, "p_perm_t": p_t, "obs_spearman": obs_rho, "p_perm_spearman": p_rho})

perm_tab = pd.DataFrame(filas_perm)
perm_tab.to_csv(SALIDA / "e12_01_prospectivo_permutacion.csv", index=False)
print("\n--- resumen permutacion en bloques (n_perm=1000) ---")
print(perm_tab.round(4).to_string())
