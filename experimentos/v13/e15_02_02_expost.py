"""E15-02 punto 2: es la hoja "Alimentacion" receta ex-ante o pudo llenarse/ajustarse ex-post?
(a) niveles discretos: cuenta de valores distintos por columna/orden.
(b) estabilidad por tramos: numero de "corridas" (segmentos de valor constante) vs n, y si los cambios
    caen en fechas puntuales (la receta es una tabla de planta revisada de tanto en tanto, no un valor
    por-batch).
(c) la receta responde al estado pre-Reduccion (leyes/feed conocidos ANTES de Reduccion)? OLS + R2 con
    permutacion (legítimo: contextualizar el plan no es fuga).
(d) el DESVIO (ejecutado - receta) correlaciona con variables pre-tratamiento? Eso SI seria confusion
    inversa: el operador se desvia porque el batch "se ve dificil", no porque el desvio cause el KPI.
"""
import sys, time
sys.path.insert(0, "experimentos/v13")
import numpy as np, pandas as pd
import e15_02_lib as E
import v9_lib as L

t0 = time.time()
r, t = E.cargar()

# --- (a) niveles discretos
filas_a = []
for p in ("gn", "carbon", "o2", "aire"):
    for o in range(4):
        sub = r.loc[r.orden_escalon_fase == o, f"plan__{p}"].dropna()
        filas_a.append({"palanca": p, "orden": o, "n": len(sub), "valores_distintos": sub.nunique()})
out_a = pd.DataFrame(filas_a)
out_a.to_csv("experimentos/v13/e15_02_02_niveles.csv", index=False)
print("(a) niveles discretos por columna:\n", out_a.to_string())

# --- (b) corridas (segmentos de valor constante) en orden cronologico
filas_b = []
for p in ("gn", "carbon", "o2", "aire"):
    for o in range(4):
        sub = r.loc[r.orden_escalon_fase == o].sort_values("idx_cronologico")[f"plan__{p}"].dropna()
        cambios = int((sub.diff().fillna(0) != 0).sum())
        filas_b.append({"palanca": p, "orden": o, "n": len(sub), "n_corridas": cambios + 1,
                         "largo_medio_corrida": len(sub) / (cambios + 1)})
out_b = pd.DataFrame(filas_b)
out_b.to_csv("experimentos/v13/e15_02_02_corridas.csv", index=False)
print("\n(b) corridas (tabla revisada por tramos, no por batch):\n", out_b.round(1).to_string())

# --- (c) la receta responde al estado pre-Reduccion?
ESTADO_PRE = ["feed_sn_total_t", "ley_sn_conc_batch_pct", "frac_carga_secundaria_F", "feed_dross_Fe_total_t", "F_lnirf_F0"]
tt = t.join(pd.read_csv("experimentos/v13/e15_01_tabla_batch.csv", index_col=0)[["p_gn", "p_carbon"]], rsuffix="_dup")
rng = np.random.default_rng(0)


def r2_perm(y, X, reps=1000):
    dd = pd.concat([y, X], axis=1).dropna()
    yv, Xv = dd.iloc[:, 0], dd.iloc[:, 1:]
    f = L.ols_hc3(yv, Xv)
    r2 = f.rsquared
    n = len(dd)
    r2s = np.empty(reps)
    for i in range(reps):
        yp = yv.to_numpy()[rng.permutation(n)]
        r2s[i] = L.ols_hc3(pd.Series(yp, index=yv.index), Xv).rsquared
    p = (np.sum(r2s >= r2) + 1) / (reps + 1)
    return r2, p, f


filas_c = []
for p in ("p_gn", "p_carbon"):
    r2, pv, f = r2_perm(tt[p], tt[ESTADO_PRE])
    filas_c.append({"receta": p, "R2": r2, "p_perm_R2": pv, "n": f.nobs})
    print(f"\n(c) {p} ~ estado_pre_Reduccion: R2={r2:.3f} p_perm={pv:.4f} n={int(f.nobs)}")
    print(f.summary2().tables[1].round(3).to_string())
out_c = pd.DataFrame(filas_c)
out_c.to_csv("experimentos/v13/e15_02_02_receta_vs_estado.csv", index=False)

# --- (d) el DESVIO correlaciona con variables pre-tratamiento? (confusion inversa)
tt["kpi_prev"] = tt.sort_values("idx_cronologico")[E.KPI].shift(1)
VARS_PRE = ESTADO_PRE + ["kpi_prev"]
filas_d = []
from scipy import stats as sst
for dvar in ("d_gn", "d_carbon"):
    for v in VARS_PRE:
        dd = tt[[dvar, v]].dropna()
        rho, pv = sst.pearsonr(dd[dvar], dd[v])
        filas_d.append({"desvio": dvar, "var_pre": v, "n": len(dd), "corr": rho, "p": pv})
out_d = pd.DataFrame(filas_d)
out_d.to_csv("experimentos/v13/e15_02_02_desvio_vs_pretrat.csv", index=False)
print("\n(d) correlacion desvio vs variables pre-tratamiento:\n", out_d.round(4).to_string())
print(f"\n[{time.time()-t0:.0f}s]")
