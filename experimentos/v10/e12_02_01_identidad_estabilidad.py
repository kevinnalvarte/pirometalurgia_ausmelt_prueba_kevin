"""E12-02 tarea 1: identidad KPI<->canales, varianza/autocorrelacion por canal, estabilidad temporal de
beta_canal,j (ventanas moviles 80-100) y clasificacion estable/inestable (teoria a priori vs regla prospectiva
sin fuga de e12_02_lib.clasif_prospectiva)."""
import sys; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
from scipy import stats as sst
import e12_02_lib as C
import e11_07_lib as E

t = C.cargar()

# --- 1a. identidad exacta ---------------------------------------------------------------------------------
kpi = t["recuperacion_refinada_pct"]
cand = 100 * t["f_metal"] * (1 - t["sn_perdido_escoria_frac"])
cand_aditivo = 100 * (1 - t["f_dross"] - t["f_polvo"] - t["sn_perdido_escoria_frac"])
suma_mdp = t["f_metal"] + t["f_dross"] + t["f_polvo"]
filas_id = [
    {"formula": "100*f_metal*(1-sn_perdido_escoria_frac)  [EXACTA]", "max_abs_diff": float((kpi - cand).abs().max()), "corr": float(kpi.corr(cand))},
    {"formula": "100*(1-f_dross-f_polvo-sn_perdido_escoria_frac)  [aditiva, aprox 1er orden]", "max_abs_diff": float((kpi - cand_aditivo).abs().max()), "corr": float(kpi.corr(cand_aditivo))},
    {"formula": "f_metal+f_dross+f_polvo (debe ser 1)", "max_abs_diff": float((suma_mdp - 1).abs().max()), "corr": np.nan},
]
pd.DataFrame(filas_id).to_csv(C.SALIDA / "e12_02_identidad.csv", index=False)
print("=== Identidad KPI <-> canales ===")
print(pd.DataFrame(filas_id).round(6).to_string())

# --- 1b. varianza y autocorrelacion por canal (pp) --------------------------------------------------------
filas_ac = []
for c in C.CANALES + ["recuperacion_refinada_pct"]:
    col = c + "_pp" if c in C.CANALES else c
    s = t.sort_values("idx_cronologico")[col].dropna().to_numpy()
    sd = float(np.std(s))
    acf = []
    for lag in (1, 2, 3, 6):
        x0, x1 = s[:-lag], s[lag:]
        r = float(np.corrcoef(x0, x1)[0, 1])
        acf.append(r)
    filas_ac.append({"canal": c, "n": len(s), "media_pp": float(np.mean(s)), "sd_pp": sd,
                      "acf1": acf[0], "acf2": acf[1], "acf3": acf[2], "acf6": acf[3]})
out_ac = pd.DataFrame(filas_ac)
out_ac.to_csv(C.SALIDA / "e12_02_varianza_autocorr.csv", index=False)
print("\n=== Varianza y autocorrelacion por canal (pp) ===")
print(out_ac.round(4).to_string())

# --- 1c. estabilidad temporal de beta_canal,j: ventanas moviles trailing 80/90/100, comparadas contra el signo
#         de muestra completa (diagnostico descriptivo, usa TODA la muestra a proposito, ver tarea 3) --------
D = {c: C.preparar_canal(t, c + "_pp") for c in C.CANALES}
filas_est = []
detalle_rolling = []
for c in C.CANALES:
    Xexp, Xctrl, y, n = D[c]["Xexp"], D[c]["Xctrl"], D[c]["y"], D[c]["n"]
    nc = Xctrl.shape[1]
    beta_full = E.ols_classic(np.column_stack([Xctrl, Xexp]), y)["beta"][1 + nc:]
    for j, palanca in enumerate(C.X_PRINCIPAL):
        signo_full = np.sign(beta_full[j])
        for W in (80, 90, 100):
            s_grid = np.arange(W, n, 10)
            signos = []
            for s in s_grid:
                lo = s - W
                b = E.ols_classic(np.column_stack([Xctrl[lo:s], Xexp[lo:s]]), y[lo:s])["beta"][1 + nc + j]
                signos.append(np.sign(b))
                detalle_rolling.append({"canal": c, "palanca": palanca, "W": W, "s": int(s), "beta": b})
            signos = np.array(signos)
            frac_mismo_signo = float(np.mean(signos == signo_full)) if signo_full != 0 else np.nan
            filas_est.append({"canal": c, "palanca": palanca, "W": W, "beta_muestra_completa": beta_full[j],
                               "signo_muestra_completa": int(signo_full), "n_ventanas": len(signos),
                               "frac_mismo_signo": frac_mismo_signo, "estable_ge80pct": bool(frac_mismo_signo >= 0.8)})
out_est = pd.DataFrame(filas_est)
out_est.to_csv(C.SALIDA / "e12_02_estabilidad_rolling.csv", index=False)
pd.DataFrame(detalle_rolling).to_csv(C.SALIDA / "e12_02_estabilidad_rolling_detalle.csv", index=False)
print("\n=== Estabilidad temporal beta_canal,j (ventanas moviles 80-100, frac. mismo signo que muestra completa) ===")
print(out_est.round(3).to_string())

# resumen por (canal,palanca) promediando W=80/90/100
resumen = out_est.groupby(["canal", "palanca"]).agg(frac_media=("frac_mismo_signo", "mean"),
                                                      estable_en_las_3_W=("estable_ge80pct", "all")).reset_index()
resumen.to_csv(C.SALIDA / "e12_02_estabilidad_resumen.csv", index=False)
print("\n=== Resumen estable/inestable por (canal,palanca), datos-driven (usa toda la muestra) ===")
print(resumen.round(3).to_string())

# --- 1d/3. clasificacion SIN FUGA: teoria a priori vs regla prospectiva de 2 ventanas disjuntas ------------
print("\n=== Clasificacion SIN FUGA por canal (regla prospectiva vs teoria a priori) ===")
filas_clasif = []
for c in C.CANALES:
    ycan = {c + "_pp": D[c]["y"]}
    Xexp, Xctrl = D[c]["Xexp"], D[c]["Xctrl"]
    # palanca "relevante" de teoria (si existe) o la primera con mayor |t| en muestra completa si no hay teoria
    j_teoria = next((j for (cc, j) in C.SIGNO_TEORIA if cc == c), None)
    if j_teoria is None:
        nc = Xctrl.shape[1]
        r = E.ols_classic(np.column_stack([Xctrl, Xexp]), ycan[c + "_pp"])
        j_teoria = int(np.argmax(np.abs(r["t"][1 + nc:])))
    df_cl = C.clasif_prospectiva(Xexp, Xctrl, ycan, c + "_pp", j_teoria)
    df_cl["canal"] = c; df_cl["palanca"] = C.X_PRINCIPAL[j_teoria]
    df_cl.to_csv(C.SALIDA / f"e12_02_clasif_prospectiva_{c}.csv", index=False)
    frac_estable = float(df_cl["estable"].mean())
    filas_clasif.append({"canal": c, "palanca_evaluada": C.X_PRINCIPAL[j_teoria], "estable_teoria_a_priori": C.CANAL_ESTABLE_TEORIA[c],
                          "frac_pasos_estable_prospectivo": frac_estable, "n_pasos": len(df_cl),
                          "coincide_mayoria_con_teoria": bool((frac_estable >= 0.5) == C.CANAL_ESTABLE_TEORIA[c])})
out_clasif = pd.DataFrame(filas_clasif)
out_clasif.to_csv(C.SALIDA / "e12_02_clasif_comparacion.csv", index=False)
print(out_clasif.round(3).to_string())
