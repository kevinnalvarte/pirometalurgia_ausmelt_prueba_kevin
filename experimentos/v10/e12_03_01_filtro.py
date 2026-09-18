"""E12-03 tarea 1-3: filtro de Kalman (beta_t = beta_{t-1} + eta_t) para xG,xC, controles/intercepto fijos
(Q=0, equivalente a OLS recursivo/expansivo dentro del mismo filtro -- variante (i); variante (ii) con controles
paseo lento se probo aparte y EMPEORA t/DEV/lockbox en todos los q_ctrl>0 ensayados, ver e12_03_resultados.md).

Hiperparametros (q_G, q_C, R) fijados UNA VEZ por verosimilitud predictiva un-paso-adelante en los batches
[NINIT=40, BURNIN=100) (solo pasado respecto del inicio de la evaluacion prospectiva, batch 100) -- fallback
explicito de ITERACION_12_diseno.md (H3) por costo de permutacion. Tambien se corre una variante ADAPTATIVA que
re-selecciona (q_G,q_C) cada 20 batches por verosimilitud en el pasado ya observado (ventana expansiva desde 40),
mas cara pero mas fiel al espiritu de H3; se reporta como punto de comparacion (no se permuta por costo, ver
tarea 2 en el script siguiente).

Compara contra: W=100 (referencia v9), pond. exponencial semivida=60, expansiva pura (todos via e11_07_lib, misma
funcion stats_final para que las comparaciones sean apples-to-apples)."""
import sys
import time

sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v10")
import numpy as np
import pandas as pd
import e11_07_lib as E
import e12_03_lib as K

t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Z = K.construir_Z(D)
y, Xexp, Xctrl = D["y"], D["Xexp"], D["Xctrl"]
nc = Xctrl.shape[1]
n = D["n"]
es_dev = D["es_dev"]
idx = D["idx"]

GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# --- hiperparametros fijos (q_ctrl = 0, variante (i)) -----------------------------------------------------------
hp = K.elegir_hiperparametros(Z, y, nc, GRID, burnin=K.BURNIN, q_ctrl=0.0, ninit=K.NINIT)
print(f"hiperparametros fijos (burn-in [{K.NINIT},{K.BURNIN})): q_G={hp['q_G']}, q_C={hp['q_C']}, R={hp['R']:.3f}")
hp["tabla"].to_csv(K.SALIDA / "e12_03_hiperparametros.csv", index=False)
Qd_fijo = np.array([0.0] * (1 + nc) + [hp["q_G"], hp["q_C"]])

score_cont, res_cont = K.score_continuous(Z, y, Qd_fijo, hp["R"], start=K.START)
score_per = K.score_periodic(Z, y, Qd_fijo, hp["R"], start=K.START, paso=K.PASO)

# --- variante adaptativa (q_ctrl=0), re-estimacion cada 20 batches -----------------------------------------------
GRID_ADAPT = [0.0, 0.001, 0.005, 0.02, 0.05, 0.15]
t0 = time.time()
score_adapt, info_adapt = K.kalman_adaptive_score(Z, y, nc, GRID_ADAPT, first_select=K.BURNIN, reestim_step=20,
                                                   ninit=K.NINIT, q_ctrl=0.0, start=K.START)
print(f"adaptativo: {time.time()-t0:.1f}s")

# --- comparaciones (mismas W/alternativas que E11-07, funcion stats_final identica) ------------------------------
score_w100 = E.walk_forward_score(Xexp, Xctrl, y, 100, paso=E.PASO)
score_hl60 = E.walk_forward_score(Xexp, Xctrl, y, None, paso=E.PASO, start=K.START, halflife=60)
score_expansiva = E.walk_forward_score(Xexp, Xctrl, y, None, paso=E.PASO, start=K.START)

specs = {
    "kalman_continuo_qctrl0": score_cont,
    "kalman_periodico10_qctrl0": score_per,
    "kalman_adaptativo20_qctrl0": score_adapt,
    "W100_duro (referencia v9)": score_w100,
    "exp_semivida60": score_hl60,
    "expansiva": score_expansiva,
}

filas = []
scores_df = {"idx_cronologico": idx, "T_media_R": D["T_media_R"], "kpi": y}
for nombre, sc in specs.items():
    scores_df[nombre] = sc
    m = ~np.isnan(sc)
    for etiqueta, msk in [("todo", m), ("DEV", m & es_dev), ("lockbox", m & ~es_dev)]:
        r = E.stats_final(sc, y, Xctrl, msk)
        r["variante"] = nombre; r["muestra"] = etiqueta
        filas.append(r)

out = pd.DataFrame(filas)[["variante", "muestra", "n", "pendiente", "se", "t", "p_uni", "spearman", "sd_score"]]
out.to_csv(K.SALIDA / "e12_03_comparacion.csv", index=False)
pd.DataFrame(scores_df).to_csv(K.SALIDA / "e12_03_scores.csv", index=False)

pd.set_option("display.width", 220)
print(out.round(4).to_string())
