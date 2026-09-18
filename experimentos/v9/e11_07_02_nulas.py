"""E11-07 tarea 2: nulas que respetan la adaptatividad y la autocorrelacion.
(a) permutacion de (xG,xC) por fila DENTRO de bloques cronologicos de 20 batches, walk-forward completo, para
    W=100 (primaria) y para el minimo p sobre todas las W (correccion por multiplicidad).
(b) desplazamiento circular de las exposiciones respecto del KPI, k in [15, n-15], W=100.
"""
import sys, time; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import e11_07_lib as E

N_PERM = 2000
t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Xexp, Xctrl, y, n = D["Xexp"], D["Xctrl"], D["y"], D["n"]
mask_todo = np.ones(n, bool)


def stat_W(Xe, W):
    score = E.walk_forward_score(Xe, Xctrl, y, W, paso=E.PASO, tmin=0.0)
    r = E.stats_final(score, y, Xctrl, mask_todo)
    return r["t"], r["spearman"]

# --- observado
obs = {}
for W in E.WS_MULTIPLICIDAD:
    tval, rho = stat_W(Xexp, W)
    obs[W] = (tval, rho)
    print("obs W", W, "t", round(tval, 3), "spearman", round(rho, 3))
obs_t100, obs_rho100 = obs[100]
obs_best_t = max(v[0] for v in obs.values())
obs_best_rho = max(v[1] for v in obs.values())

# --- (a) permutacion en bloques de 20
bloques = E.bloques_cronologicos(n, 20)
rng = np.random.default_rng(E.SEED)
t0 = time.time()
null_t100, null_rho100 = np.empty(N_PERM), np.empty(N_PERM)
null_best_t, null_best_rho = np.empty(N_PERM), np.empty(N_PERM)
for i in range(N_PERM):
    Xp = E.permutar_bloques(Xexp, bloques, rng)
    ts, rhos = [], []
    for W in E.WS_MULTIPLICIDAD:
        tv, rv = stat_W(Xp, W)
        ts.append(tv); rhos.append(rv)
        if W == 100:
            null_t100[i], null_rho100[i] = tv, rv
    null_best_t[i] = max(ts); null_best_rho[i] = max(rhos)
    if (i + 1) % 200 == 0:
        print(f"  perm {i+1}/{N_PERM}  {time.time()-t0:.0f}s")
print(f"permutacion en bloques: {time.time()-t0:.0f}s total")

def p_uni(obs_v, null_arr):
    return float((np.sum(null_arr >= obs_v) + 1) / (len(null_arr) + 1))

filas_perm = [
    {"nula": "bloques20", "spec": "W100_primaria", "estadistico": "t_pendiente", "obs": obs_t100, "p_uni": p_uni(obs_t100, null_t100)},
    {"nula": "bloques20", "spec": "W100_primaria", "estadistico": "spearman", "obs": obs_rho100, "p_uni": p_uni(obs_rho100, null_rho100)},
    {"nula": "bloques20", "spec": "min_p_sobre_W (mejor de 6)", "estadistico": "t_pendiente", "obs": obs_best_t, "p_uni": p_uni(obs_best_t, null_best_t)},
    {"nula": "bloques20", "spec": "min_p_sobre_W (mejor de 6)", "estadistico": "spearman", "obs": obs_best_rho, "p_uni": p_uni(obs_best_rho, null_best_rho)},
]

# --- (b) desplazamiento circular, W=100
ks = np.arange(15, n - 15)
circ_t, circ_rho = np.empty(len(ks)), np.empty(len(ks))
t0 = time.time()
for j, k in enumerate(ks):
    Xs = E.circular_shift(Xexp, int(k))
    circ_t[j], circ_rho[j] = stat_W(Xs, 100)
print(f"desplazamiento circular: {time.time()-t0:.0f}s ({len(ks)} shifts)")
filas_perm += [
    {"nula": "circular_k15..n-15", "spec": "W100_primaria", "estadistico": "t_pendiente", "obs": obs_t100, "p_uni": p_uni(obs_t100, circ_t)},
    {"nula": "circular_k15..n-15", "spec": "W100_primaria", "estadistico": "spearman", "obs": obs_rho100, "p_uni": p_uni(obs_rho100, circ_rho)},
]

out = pd.DataFrame(filas_perm)
out.to_csv(E.SALIDA / "e11_07_nulas.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string())

# guardar tambien las distribuciones nulas crudas (para auditoria / graficos futuros)
np.savez(E.SALIDA / "e11_07_nulas_dist.npz", null_t100=null_t100, null_rho100=null_rho100,
         null_best_t=null_best_t, null_best_rho=null_best_rho, circ_t=circ_t, circ_rho=circ_rho, ks=ks)

pd.DataFrame({"W": list(obs.keys()), "t_obs": [v[0] for v in obs.values()], "spearman_obs": [v[1] for v in obs.values()]}).to_csv(
    E.SALIDA / "e11_07_obs_por_W.csv", index=False)
