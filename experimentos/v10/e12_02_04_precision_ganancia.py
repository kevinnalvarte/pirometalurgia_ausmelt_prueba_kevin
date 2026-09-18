"""E12-02 tarea 5: (a) compara la precision (SE a lo largo de la campana, y dispersion bootstrap por bloques del
promedio) de beta_KPI,GN y beta_KPI,C bajo el esquema A (referencia) vs el mejor esquema (elegido por t en 'total'
en e12_02_esquemas_observado.csv); (b) ganancia esperada calibrada de la politica (pp/batch) con IC bootstrap por
bloques de 10 (1000 rep.), para A y para el mejor esquema, en total y en lockbox."""
import sys, time; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
import e12_02_lib as C
import e11_07_lib as E

KG_SN_POR_PP = C.KG_SN_POR_PP
N_BOOT = 1000
BLOQUE = 10
rng = np.random.default_rng(C.SEED)

t = C.cargar()
D = C.preparar_todo(t)
Xexp, Xctrl, y_kpi, ycan, n = D["Xexp"], D["Xctrl"], D["y_kpi"], D["ycan"], D["n"]
es_dev = D["es_dev"]

obs = pd.read_csv(C.SALIDA / "e12_02_esquemas_observado.csv")
mejor = obs[obs.muestra == "total"].sort_values("t_clasico", ascending=False).iloc[0]["esquema"]
print("Mejor esquema (t en 'total'):", mejor)

# --- (a) precision de beta_KPI,GN y beta_KPI,C a lo largo de la campana ---------------------------------------
sgA, bA, seA = C.beta_kpi_series_por_esquema("A_directo", Xexp, Xctrl, y_kpi, ycan)
sgM, bM, seM = C.beta_kpi_series_por_esquema(mejor, Xexp, Xctrl, y_kpi, ycan)
assert np.array_equal(sgA, sgM)
filas_prec = []
for j, palanca in enumerate(C.X_PRINCIPAL):
    filas_prec.append({"palanca": palanca, "esquema": "A_directo", "se_mediana": float(np.nanmedian(seA[:, j])),
                        "se_p25": float(np.nanpercentile(seA[:, j], 25)), "se_p75": float(np.nanpercentile(seA[:, j], 75)),
                        "beta_mediana": float(np.nanmedian(bA[:, j]))})
    filas_prec.append({"palanca": palanca, "esquema": mejor, "se_mediana": float(np.nanmedian(seM[:, j])),
                        "se_p25": float(np.nanpercentile(seM[:, j], 25)), "se_p75": float(np.nanpercentile(seM[:, j], 75)),
                        "beta_mediana": float(np.nanmedian(bM[:, j]))})
out_prec = pd.DataFrame(filas_prec)
out_prec["razon_se_vs_A"] = out_prec.groupby("palanca")["se_mediana"].transform(lambda s: s / s.iloc[0])
out_prec.to_csv(C.SALIDA / "e12_02_precision_beta.csv", index=False)
print("\n=== Precision de beta_KPI,GN / beta_KPI,C a lo largo de la campana (SE mediana de la ventana en curso) ===")
print(out_prec.round(4).to_string())

pd.DataFrame({"s": sgA, "beta_GN_A": bA[:, 0], "se_GN_A": seA[:, 0], "beta_C_A": bA[:, 1], "se_C_A": seA[:, 1],
              f"beta_GN_{mejor}": bM[:, 0], f"se_GN_{mejor}": seM[:, 0], f"beta_C_{mejor}": bM[:, 1], f"se_C_{mejor}": seM[:, 1]}
             ).to_csv(C.SALIDA / "e12_02_trayectoria_beta_kpi.csv", index=False)

# --- (b) ganancia esperada calibrada (pp/batch), bootstrap por bloques de 10 -----------------------------------
npz = np.load(C.SALIDA / "e12_02_scores_obs.npz")
MASKS = {"total": np.ones(n, bool), "lockbox": ~es_dev}


def tercil_diff(score_arr, y_arr, Xc_arr):
    resid_y = E.ols_classic(Xc_arr, y_arr)["resid"]
    rank = pd.Series(score_arr).rank(method="first").to_numpy()
    q = pd.qcut(rank, 3, labels=False)
    return float(resid_y[q == 2].mean() - resid_y[q == 0].mean())


filas_gan = []
for nombre in ["A_directo", mejor]:
    score_full = npz[nombre]
    for nombre_m, m in MASKS.items():
        mv = m & ~np.isnan(score_full)
        score, yy, Xc = score_full[mv], y_kpi[mv], Xctrl[mv]
        n_s = len(score)
        if n_s < 30:
            continue
        pend_obs = E.stats_final(score_full, y_kpi, Xctrl, mv)["pendiente"]
        sd_score = float(np.std(score))
        diff_obs = tercil_diff(score, yy, Xc)
        bloques = E.bloques_cronologicos(n_s, BLOQUE, seed=C.SEED)
        nb = len(bloques)
        pend_boot = np.empty(N_BOOT); diff_boot = np.empty(N_BOOT); desplaz_boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            elegidos = rng.integers(0, nb, size=nb)
            idxcat = np.concatenate([bloques[i] for i in elegidos])
            sB, yB, XcB = score[idxcat], yy[idxcat], Xc[idxcat]
            beta = E._lstsq_beta(np.column_stack([np.ones(len(yB)), sB, XcB]), yB)
            pend_boot[b] = beta[1]
            diff_boot[b] = tercil_diff(sB, yB, XcB)
            desplaz_boot[b] = beta[1] * np.std(sB)
        ci_pend = np.percentile(pend_boot, [2.5, 97.5])
        ci_diff = np.percentile(diff_boot, [2.5, 97.5])
        ci_desplaz = np.percentile(desplaz_boot, [2.5, 97.5])
        desplaz_1sd = pend_obs * sd_score
        filas_gan.append({"esquema": nombre, "muestra": nombre_m, "n": n_s, "pendiente_obs": pend_obs,
                           "pendiente_ic_lo": ci_pend[0], "pendiente_ic_hi": ci_pend[1],
                           "calibracion_incluye_1": bool(ci_pend[0] <= 1 <= ci_pend[1]),
                           "ganancia_1sd_score_pp_obs": desplaz_1sd, "ganancia_1sd_ic_lo_pp": ci_desplaz[0],
                           "ganancia_1sd_ic_hi_pp": ci_desplaz[1], "ganancia_1sd_excluye_0": bool(ci_desplaz[0] > 0 or ci_desplaz[1] < 0),
                           "tercil_diff_pp_obs": diff_obs, "tercil_diff_ic_lo_pp": ci_diff[0], "tercil_diff_ic_hi_pp": ci_diff[1],
                           "tercil_diff_kgSn_obs": diff_obs * KG_SN_POR_PP})
out_gan = pd.DataFrame(filas_gan)
out_gan.to_csv(C.SALIDA / "e12_02_ganancia.csv", index=False)
print("\n=== Ganancia esperada calibrada (bootstrap bloques de 10, 1000 rep.) ===")
print(out_gan.round(4).to_string())
