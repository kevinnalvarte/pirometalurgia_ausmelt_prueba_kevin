"""E12-03 tarea 2: permutacion en bloques cronologicos de 20 (>=1000 rep.), re-ejecutando TODO el filtro de Kalman
(las dos variantes de hiperparametros fijos: continua y periodica-10; el filtro corre desde cero sobre (xG,xC)
permutadas). Hiperparametros (q_G,q_C,R) se mantienen FIJOS en los valores elegidos con los datos reales
(e12_03_01_filtro.py, burn-in [40,100)) -- no se re-seleccionan dentro de cada permutacion: la variante adaptativa
tarda ~4.5s/corrida (demasiado para >=1000 reps en primer plano) y ITERACION_12_diseno.md permite fijar los
hiperparametros por costo, dejandolo dicho aqui."""
import sys
import time

sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v10")
import numpy as np
import pandas as pd
import e11_07_lib as E
import e12_03_lib as K

N_PERM = 1200
t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Z = K.construir_Z(D)
y, Xctrl = D["y"], D["Xctrl"]
nc = Xctrl.shape[1]
n = D["n"]
es_dev = D["es_dev"]

GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
hp = K.elegir_hiperparametros(Z, y, nc, GRID, burnin=K.BURNIN, q_ctrl=0.0, ninit=K.NINIT)
Qd = np.array([0.0] * (1 + nc) + [hp["q_G"], hp["q_C"]])
print("hiperparametros fijos:", hp["q_G"], hp["q_C"], hp["R"])


def stat_de(Zp):
    sc_c, _ = K.score_continuous(Zp, y, Qd, hp["R"], start=K.START)
    sc_p = K.score_periodic(Zp, y, Qd, hp["R"], start=K.START, paso=K.PASO)
    rc = E.stats_final(sc_c, y, Xctrl); rp = E.stats_final(sc_p, y, Xctrl)
    return rc["t"], rc["spearman"], rp["t"], rp["spearman"]


# --- observado -----------------------------------------------------------------------------------------------
obs_tc, obs_rc, obs_tp, obs_rp = stat_de(Z)
print(f"obs continuo: t={obs_tc:.3f} rho={obs_rc:.3f} | periodico10: t={obs_tp:.3f} rho={obs_rp:.3f}")

# --- permutacion en bloques de 20 (permuta filas de xG,xC dentro de cada bloque; controles e idx intactos) -----
bloques = E.bloques_cronologicos(n, 20)
rng = np.random.default_rng(K.SEED)
null_tc = np.empty(N_PERM); null_rc = np.empty(N_PERM)
null_tp = np.empty(N_PERM); null_rp = np.empty(N_PERM)
t0 = time.time()
for i in range(N_PERM):
    Xp = E.permutar_bloques(D["Xexp"], bloques, rng)
    Zp = Z.copy(); Zp[:, -2:] = Xp
    null_tc[i], null_rc[i], null_tp[i], null_rp[i] = stat_de(Zp)
    if (i + 1) % 200 == 0:
        print(f"  perm {i+1}/{N_PERM}  {time.time()-t0:.0f}s")
print(f"permutacion: {time.time()-t0:.0f}s total")


def p_uni(obs_v, null_arr):
    return float((np.sum(null_arr >= obs_v) + 1) / (len(null_arr) + 1))


filas = [
    {"variante": "kalman_continuo_qctrl0", "estadistico": "t_pendiente", "obs": obs_tc, "p_uni": p_uni(obs_tc, null_tc)},
    {"variante": "kalman_continuo_qctrl0", "estadistico": "spearman", "obs": obs_rc, "p_uni": p_uni(obs_rc, null_rc)},
    {"variante": "kalman_periodico10_qctrl0", "estadistico": "t_pendiente", "obs": obs_tp, "p_uni": p_uni(obs_tp, null_tp)},
    {"variante": "kalman_periodico10_qctrl0", "estadistico": "spearman", "obs": obs_rp, "p_uni": p_uni(obs_rp, null_rp)},
]
out = pd.DataFrame(filas)
out.to_csv(K.SALIDA / "e12_03_permutacion.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string())
np.savez(K.SALIDA / "e12_03_permutacion_dist.npz", null_tc=null_tc, null_rc=null_rc, null_tp=null_tp, null_rp=null_rp)
