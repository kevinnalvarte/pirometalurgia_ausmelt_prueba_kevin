"""E11-07 tarea 7: trayectoria de beta_G(t) y beta_C(t) (ventana dura W=100, paso=10) con IC (clasico, HC-like vía OLS
homocedastico dentro de la ventana), junto con T_media_R y espesor_ladrillo_norm_mm promedio de la ventana de entrenamiento."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as E

t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Xexp, Xctrl, y, n = D["Xexp"], D["Xctrl"], D["y"], D["n"]
T_R, espesor, idx_batch = D["T_media_R"], D["espesor"], D["idx"]
W, paso = 100, E.PASO

filas = []
for s in range(W, n, paso):
    lo = s - W
    Ztr = np.column_stack([np.ones(W), Xctrl[lo:s], Xexp[lo:s]])
    ytr = y[lo:s]
    r = E.ols_classic(np.column_stack([Xctrl[lo:s], Xexp[lo:s]]), ytr)
    beta, se = r["beta"], r["se"]
    ncc = Xctrl.shape[1]
    bG, bC = beta[1 + ncc], beta[1 + ncc + 1]
    seG, seC = se[1 + ncc], se[1 + ncc + 1]
    filas.append({
        "s_inicio_ventana": lo, "s_fin_ventana": s, "batch_inicio": idx_batch[lo], "batch_fin": idx_batch[s - 1],
        "batch_test_ini": idx_batch[s], "batch_test_fin": idx_batch[min(s + paso, n) - 1],
        "beta_G": bG, "ic_lo_G": bG - 1.96 * seG, "ic_hi_G": bG + 1.96 * seG,
        "beta_C": bC, "ic_lo_C": bC - 1.96 * seC, "ic_hi_C": bC + 1.96 * seC,
        "T_media_R_ventana": float(np.nanmean(T_R[lo:s])), "espesor_ladrillo_norm_mm_ventana": float(np.nanmean(espesor[lo:s])),
        "T_media_R_test": float(np.nanmean(T_R[s:min(s + paso, n)])), "es_dev_frac_test": float(D["es_dev"][s:min(s + paso, n)].mean()),
    })
out = pd.DataFrame(filas)
out.to_csv(E.SALIDA / "e11_07_trayectoria_beta.csv", index=False)
pd.set_option("display.width", 240)
print(out.round(3).to_string())
