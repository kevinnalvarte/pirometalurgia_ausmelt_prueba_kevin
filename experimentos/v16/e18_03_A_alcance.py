"""E18-03 A -- alcance declarado ex-ante (4 reglas) evaluado con el mismo motor del asesor v10."""
import sys, time
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
import e18_03_lib as E

t0 = time.time()
SAL = "experimentos/v16"
r, bt = E.cargar()
reglas = E.reglas_alcance(r, bt)
print("== % escalones en alcance (global) ==")
for k, m in reglas.items():
    print(f"  {k}: {100*m.mean():.1f}%  (n_escalones={int(m.sum())}, n_batches>=1esc={r.loc[m,'Batch'].nunique()})")

# 1) % por tramo de campana
tramos = pd.DataFrame({k: E.pct_alcance_por_tramo(r, m) for k, m in reglas.items()})
tramos.to_csv(f"{SAL}/e18_03_A_pct_por_tramo.csv")
print("\n== % en alcance por tramo (idx_cronologico) ==")
print(tramos.round(1).to_string())

filas_resumen = []
Ps_prosp = {}
ols_filas = []
fuera_filas = []
jack_filas = []
calib_filas = []

for k, m in reglas.items():
    expo_dentro = E.exposicion_alcance(r, m, dentro=True)
    expo_fuera = E.exposicion_alcance(r, m, dentro=False)
    n_batches_con_signal = int((expo_dentro.abs().sum(axis=1) > 0).sum())

    # 2) OLS conjunta HC3 (KPI + dross) dentro del alcance
    ols = E.ols_conjunta(bt, expo_dentro, y_cols=(E.KPI, "f_dross", "sn_perdido_escoria_frac"))
    ols.insert(0, "regla", k)
    ols_filas.append(ols)

    # OLS conjunta fuera del alcance (control: el asesor no deberia "funcionar" aqui)
    ols_fuera = E.ols_conjunta(bt, expo_fuera, y_cols=(E.KPI,))
    ols_fuera.insert(0, "regla", k)
    fuera_filas.append(ols_fuera)

    # 3) prueba prospectiva W=100/paso10, JS -- solo si hay variacion fuera del calentamiento
    P = E.preparar_batch(bt, expo_dentro)
    Ps_prosp[k] = P
    prosp = E.prospectivo(P)

    # 5) jackknife -10 y 6) calibracion bootstrap (solo si prospectivo es evaluable)
    if np.isfinite(prosp["t"]):
        jk = E.jackknife_quitar_k(P, k=10, reps=300)
        cal = E.calibracion_boot(P, reps=1000)
    else:
        jk = {"t_P5": np.nan, "t_P50": np.nan, "t_P95": np.nan, "pct_t_gt_1.645": np.nan, "pend_P50": np.nan}
        cal = {"pendiente_P50": np.nan, "IC95_lo": np.nan, "IC95_hi": np.nan, "P_mayor_0": np.nan}
    jk["regla"] = k; jk["n_batches_con_senal"] = n_batches_con_signal
    cal["regla"] = k
    jack_filas.append(jk); calib_filas.append(cal)

    filas_resumen.append({"regla": k, "n_batches_con_senal": n_batches_con_signal,
                           "prosp_n": prosp["n"], "prosp_pendiente": prosp["pendiente"], "prosp_t": prosp["t"]})
    print(f"\n[{k}] n_batches_con_senal={n_batches_con_signal}  prospectivo: n={prosp['n']} pendiente={prosp['pendiente']:.3f} t={prosp['t']:.2f}  [{time.time()-t0:.0f}s]")

pd.concat(ols_filas, ignore_index=True).to_csv(f"{SAL}/e18_03_A_ols_dentro.csv", index=False)
pd.concat(fuera_filas, ignore_index=True).to_csv(f"{SAL}/e18_03_A_ols_fuera.csv", index=False)
pd.DataFrame(jack_filas).to_csv(f"{SAL}/e18_03_A_jackknife.csv", index=False)
pd.DataFrame(calib_filas).to_csv(f"{SAL}/e18_03_A_calibracion.csv", index=False)

# 4) multiplicidad: estadistico maximo sobre las 4 reglas (misma reordenacion por replica)
mult = E.maxstat_multiplicidad(Ps_prosp)
mult.to_csv(f"{SAL}/e18_03_A_multiplicidad.csv", index=False)
print("\n== multiplicidad (maximo t sobre 4 reglas) ==")
print(mult.round(4).to_string())

resumen = pd.DataFrame(filas_resumen).merge(mult, left_on="regla", right_on="regla")
resumen.to_csv(f"{SAL}/e18_03_A_resumen.csv", index=False)
print("\n== resumen final A ==")
print(resumen.round(4).to_string())
print(f"\n[{time.time()-t0:.0f}s] listo")
