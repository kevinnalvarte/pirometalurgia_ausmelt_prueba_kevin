"""E11-07 tarea 6: magnitud, W=100 paso=10. IC bootstrap por bloques de 10 batches (1000 reps) de:
(a) la pendiente de KPI ~ score + controles;
(b) la diferencia de KPI residualizado por controles entre el tercio alto y el tercio bajo del puntaje prospectivo.
Expresado en pp de KPI y en kg de Sn/batch (1 pp ~= 512 kg Sn). Compatibilidad de la pendiente con 1 (calibracion)."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as E

KG_SN_POR_PP = 512.0
N_BOOT = 1000
BLOQUE = 10
rng = np.random.default_rng(E.SEED)

t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Xexp, Xctrl, y, n = D["Xexp"], D["Xctrl"], D["y"], D["n"]
score_full = E.walk_forward_score(Xexp, Xctrl, y, 100, paso=E.PASO, tmin=0.0)
m = ~np.isnan(score_full)
score, yy, Xc = score_full[m], y[m], Xctrl[m]
n_s = len(score)
print("n puntuado (W=100, paso 10):", n_s)


def tercil_diff(score_arr, y_arr, Xc_arr):
    resid_y = E.ols_classic(Xc_arr, y_arr)["resid"]
    rank = pd.Series(score_arr).rank(method="first").to_numpy()
    q = pd.qcut(rank, 3, labels=False)
    return float(resid_y[q == 2].mean() - resid_y[q == 0].mean())


pend_obs = E.stats_final(score_full, y, Xctrl, np.ones(n, bool))["pendiente"]
diff_obs = tercil_diff(score, yy, Xc)
print("observado: pendiente", round(pend_obs, 4), "| diff tercio alto-bajo (pp, resid. controles)", round(diff_obs, 4))

bloques = E.bloques_cronologicos(n_s, BLOQUE)
nb = len(bloques)
pend_boot = np.empty(N_BOOT); diff_boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    elegidos = rng.integers(0, nb, size=nb)
    idxcat = np.concatenate([bloques[i] for i in elegidos])
    sB, yB, XcB = score[idxcat], yy[idxcat], Xc[idxcat]
    beta = E._lstsq_beta(np.column_stack([np.ones(len(yB)), sB, XcB]), yB)
    pend_boot[b] = beta[1]
    diff_boot[b] = tercil_diff(sB, yB, XcB)

ci_pend = np.percentile(pend_boot, [2.5, 97.5])
ci_diff = np.percentile(diff_boot, [2.5, 97.5])
print(f"pendiente: obs {pend_obs:.3f}  IC95 boot [{ci_pend[0]:.3f}, {ci_pend[1]:.3f}]  (1 = calibracion perfecta; incluido? {ci_pend[0] <= 1 <= ci_pend[1]})")
print(f"diff tercio alto-bajo (pp KPI, resid.): obs {diff_obs:.3f}  IC95 boot [{ci_diff[0]:.3f}, {ci_diff[1]:.3f}]")
print(f"en kg Sn/batch (x{KG_SN_POR_PP}): diff obs {diff_obs*KG_SN_POR_PP:.0f}  IC95 [{ci_diff[0]*KG_SN_POR_PP:.0f}, {ci_diff[1]*KG_SN_POR_PP:.0f}]")

sd_score = float(np.std(score))
desplaz_1sd_pp = pend_obs * sd_score
out = pd.DataFrame([{
    "n_puntuado": n_s, "pendiente_obs": pend_obs, "pendiente_ic_lo": ci_pend[0], "pendiente_ic_hi": ci_pend[1],
    "calibracion_incluye_1": bool(ci_pend[0] <= 1 <= ci_pend[1]),
    "diff_tercio_alto_bajo_pp_obs": diff_obs, "diff_ic_lo_pp": ci_diff[0], "diff_ic_hi_pp": ci_diff[1],
    "diff_obs_kgSn": diff_obs * KG_SN_POR_PP, "diff_ic_lo_kgSn": ci_diff[0] * KG_SN_POR_PP, "diff_ic_hi_kgSn": ci_diff[1] * KG_SN_POR_PP,
    "sd_score": sd_score, "desplaz_1sd_score_en_pp_kpi": desplaz_1sd_pp, "desplaz_1sd_score_en_kgSn": desplaz_1sd_pp * KG_SN_POR_PP,
    "n_boot": N_BOOT, "bloque_batches": BLOQUE}])
out.to_csv(E.SALIDA / "e11_07_magnitud.csv", index=False)
print(out.round(3).to_string())
