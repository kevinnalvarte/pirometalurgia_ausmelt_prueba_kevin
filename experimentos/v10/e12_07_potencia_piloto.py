"""E12-07 (Fable): potencia de un piloto ALEATORIZADO por batch (GN de Reduccion a -d sd_res vs practica habitual) con el ruido REAL del KPI
(bootstrap por bloques de los residuos, conserva la autocorrelacion a lag 3). Endpoints: KPI, f_dross, y el target de escalon ln_feo_ret."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
t = W7.cargar_tabla(); t["kpi_l3"] = t[W7.KPI].shift(3); t["dross_l3"] = t["f_dross"].shift(3)
C = W7.C_SIN_ESPESOR + ["F_lnirf_F0"]
d = t.dropna(subset=C + ["xG", "xC", W7.KPI, "f_dross", "kpi_l3"]).copy()
res = {}
for y, extra in ((W7.KPI, ["kpi_l3"]), ("f_dross", ["dross_l3"])):
    Z = d[C + extra + ["xG", "xC"]].to_numpy(float); yy = d[y].to_numpy(float) * (100 if y == "f_dross" else 1)
    r = W7.ols_classic(Z, yy); res[y] = r["resid"]; print(y, "sd bruta %.2f  sd residual %.2f  beta_xG %.2f (t %.1f)" % (yy.std(), r["resid"].std(), r["beta"][-2], r["t"][-2]))
print("sd de xG observacional: %.2f ; n batches/dia: %.2f" % (d.xG.std(), len(t) / ((pd.to_datetime(t.fecha_batch).max() - pd.to_datetime(t.fecha_batch).min()).days)))
rng = np.random.default_rng(1); filas = []
def sim(resid, beta, N, delta, cumpl_sd=0.6, nsim=2000, bloque=10):
    n = len(resid); rej = 0
    for _ in range(nsim):
        st = rng.integers(0, n - bloque, size=N // bloque + 1); e = np.concatenate([resid[s:s + bloque] for s in st])[:N]
        a = rng.permutation(np.r_[np.ones(N // 2), np.zeros(N - N // 2)])          # 1 = brazo tratado (GN -delta sd en R0-R3)
        x = -delta * a + rng.normal(0, cumpl_sd / 2, N)                           # exposicion realizada (ruido de cumplimiento)
        y = beta * x + e; X = np.column_stack([np.ones(N), a]); b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
        se = np.sqrt(r @ r / (N - 2) * np.linalg.inv(X.T @ X)[1, 1]); rej += (b[1] / se > 1.645)   # unilateral 5 %: tratado mejora KPI
    return rej / nsim
for N in (30, 40, 60, 80, 100, 150, 200):
    for delta in (1.0, 1.5):
        for beta in (-0.75, -1.1, -1.5):
            filas.append({"N_batches": N, "delta_sd": delta, "beta_pp_por_z": beta, "efecto_pp": -beta * delta, "potencia_KPI": sim(res[W7.KPI], beta, N, delta),
                          "potencia_dross": sim(-res["f_dross"], beta * 0.75, N, delta)})   # beta_dross ~ +1.13 pp por z (E11-01) = 0.75 x |beta_KPI|
out = pd.DataFrame(filas); out.to_csv("experimentos/v10/e12_07_potencia_piloto.csv", index=False)
pd.set_option("display.width", 200); print(out.round(2).to_string())
# --- diseno simetrico: brazos -delta vs +delta (ambos dentro de la practica historica): duplica el contraste
def sim_pm(resid, beta, N, delta, nsim=2000, bloque=10):
    n = len(resid); rej = 0
    for _ in range(nsim):
        st = rng.integers(0, n - bloque, size=N // bloque + 1); e = np.concatenate([resid[s:s + bloque] for s in st])[:N]
        a = rng.permutation(np.r_[np.ones(N // 2), -np.ones(N - N // 2)]); x = delta * a + rng.normal(0, 0.3, N)
        y = beta * x + e; X = np.column_stack([np.ones(N), a]); b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
        se = np.sqrt(r @ r / (N - 2) * np.linalg.inv(X.T @ X)[1, 1]); rej += (b[1] / se < -1.645)
    return rej / nsim
f2 = [{"N_batches": N, "delta_sd": dl, "beta": be, "potencia_KPI_pm": sim_pm(res[W7.KPI], be, N, dl), "potencia_dross_pm": sim_pm(-res["f_dross"], be * 0.75, N, dl)}
      for N in (30, 40, 60, 80, 100, 120) for dl in (1.0,) for be in (-0.75, -1.1, -1.5)]
o2 = pd.DataFrame(f2); o2.to_csv("experimentos/v10/e12_07_potencia_piloto_simetrico.csv", index=False); print(o2.round(2).to_string())
