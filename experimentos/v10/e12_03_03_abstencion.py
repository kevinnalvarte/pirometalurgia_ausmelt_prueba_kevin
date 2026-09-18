"""E12-03 tarea 4: incertidumbre para abstenerse usando P_{b|b-1} del filtro (variante continua, hiperparametros
fijos de e12_03_01_filtro.py).
(a) Calibracion de las innovaciones un-paso-adelante estandarizadas z_i = v_i/sqrt(F_i): media, varianza,
    autocorrelacion (lag 1-5) y Ljung-Box; cobertura empirica de bandas +-1 y +-1.96 sd.
(b) Regla "recomendar la palanca j solo si |beta_j|/sd_j >= umbral" (umbrales 1, 1.5, 2) con magnitud
    ~ min(1, |t_j|/2): % de batches con recomendacion por palanca a lo largo de la campana (DEV/lockbox/tercios
    cronologicos) y desempeno prospectivo del puntaje gateado vs el puntaje crudo."""
import sys

sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v10")
import numpy as np
import pandas as pd
from scipy import stats as sst
import e11_07_lib as E
import e12_03_lib as K

t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Z = K.construir_Z(D)
y, Xctrl, Xexp = D["y"], D["Xctrl"], D["Xexp"]
nc = Xctrl.shape[1]
n = D["n"]
es_dev = D["es_dev"]
idx = D["idx"]

GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
hp = K.elegir_hiperparametros(Z, y, nc, GRID, burnin=K.BURNIN, q_ctrl=0.0, ninit=K.NINIT)
Qd = np.array([0.0] * (1 + nc) + [hp["q_G"], hp["q_C"]])
score_cont, res = K.score_continuous(Z, y, Qd, hp["R"], start=K.START)

# ------------------------------------------------------------------------------------------------- (a) calibracion
lo = K.START
z = res["innov"][lo:] / np.sqrt(res["Fvar"][lo:])
mean_z, var_z = float(np.mean(z)), float(np.var(z))
autocorr = {k: float(np.corrcoef(z[:-k], z[k:])[0, 1]) for k in range(1, 6)}
h = 5
nz = len(z)
Q_lb = nz * (nz + 2) * sum(autocorr[k] ** 2 / (nz - k) for k in range(1, h + 1))
p_lb = float(1 - sst.chi2.cdf(Q_lb, h))
cov_1sd = float(np.mean(np.abs(z) <= 1.0))
cov_2sd = float(np.mean(np.abs(z) <= 1.96))
calib = pd.DataFrame([{
    "n": nz, "media_z": mean_z, "var_z": var_z, **{f"autocorr_lag{k}": v for k, v in autocorr.items()},
    "ljungbox_Q_h5": Q_lb, "ljungbox_p": p_lb, "cobertura_1sd_esperada_0.683": cov_1sd,
    "cobertura_1.96sd_esperada_0.95": cov_2sd,
}])
calib.to_csv(K.SALIDA / "e12_03_calibracion.csv", index=False)
print(calib.round(4).to_string(index=False))

# ------------------------------------------------------------------------------------------------- (b) recomendacion
beta_pred = res["beta_pred"]  # (n, p): ultimas 2 columnas = G, C
P_pred_diag = res["P_pred_diag"]
bG, bC = beta_pred[:, -2], beta_pred[:, -1]
sdG, sdC = np.sqrt(P_pred_diag[:, -2]), np.sqrt(P_pred_diag[:, -1])
tG, tC = np.divide(bG, sdG, out=np.zeros_like(bG), where=sdG > 0), np.divide(bC, sdC, out=np.zeros_like(bC), where=sdC > 0)
magG, magC = np.minimum(1.0, np.abs(tG) / 2), np.minimum(1.0, np.abs(tC) / 2)
mask_eval = np.zeros(n, bool); mask_eval[K.START:] = True

tercios_lbl = np.full(n, "", dtype=object)
ord_idx = np.argsort(idx)
terc = np.array_split(ord_idx, 3)
for j, grp in enumerate(terc):
    tercios_lbl[grp] = f"tercio{j+1}"

filas_reco = []
for umbral in (1.0, 1.5, 2.0):
    gateG = (np.abs(tG) >= umbral).astype(float)
    gateC = (np.abs(tC) >= umbral).astype(float)
    score_gate = gateG * magG * bG * Xexp[:, 0] + gateC * magC * bC * Xexp[:, 1]
    score_gate_eval = np.where(mask_eval, score_gate, np.nan)
    for etiqueta, msk in [("todo", mask_eval), ("DEV", mask_eval & es_dev), ("lockbox", mask_eval & ~es_dev)]:
        r = E.stats_final(score_gate_eval, y, Xctrl, msk)
        filas_reco.append({"umbral": umbral, "muestra": etiqueta, "n": r["n"], "pendiente": r["pendiente"],
                            "t": r["t"], "p_uni": r["p_uni"], "spearman": r["spearman"],
                            "pct_recom_G": float(gateG[msk].mean()) if msk.sum() else np.nan,
                            "pct_recom_C": float(gateC[msk].mean()) if msk.sum() else np.nan})
    for terc_nombre in ("tercio1", "tercio2", "tercio3"):
        msk = mask_eval & (tercios_lbl == terc_nombre)
        if msk.sum() == 0:
            continue
        filas_reco.append({"umbral": umbral, "muestra": terc_nombre, "n": int(msk.sum()), "pendiente": np.nan,
                            "t": np.nan, "p_uni": np.nan, "spearman": np.nan,
                            "pct_recom_G": float(gateG[msk].mean()), "pct_recom_C": float(gateC[msk].mean())})

# referencia: puntaje crudo (sin gate), para comparar
for etiqueta, msk in [("todo", mask_eval), ("DEV", mask_eval & es_dev), ("lockbox", mask_eval & ~es_dev)]:
    r = E.stats_final(score_cont, y, Xctrl, msk)
    filas_reco.append({"umbral": "sin_gate", "muestra": etiqueta, "n": r["n"], "pendiente": r["pendiente"],
                        "t": r["t"], "p_uni": r["p_uni"], "spearman": r["spearman"], "pct_recom_G": 1.0, "pct_recom_C": 1.0})

out_reco = pd.DataFrame(filas_reco)
out_reco.to_csv(K.SALIDA / "e12_03_recomendacion.csv", index=False)
pd.set_option("display.width", 220)
print(out_reco.round(4).to_string(index=False))
