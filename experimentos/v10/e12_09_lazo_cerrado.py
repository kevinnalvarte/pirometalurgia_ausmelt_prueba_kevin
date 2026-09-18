"""E12-09 (Fable): simulacion de lazo cerrado del asesor en planta con el RUIDO REAL del KPI (bootstrap por bloques de los residuos).
Modelo de planta por batch: KPI_b = beta_b * x_b + e_b ; x_b = exposicion media de GN en Reduccion (unidades de z). Escenarios de beta_b real:
 constante -1.1 | constante -0.5 | NULO 0 | deriva como la estimada (-0.3 -> -2.6 -> +0.4) | cambio de signo brusco (-1.5 -> +1.5 a mitad).
Estrategias: (S0) sin asesor x~N(0,0.64); (S1) asesor observacional v9 (beta de los ultimos 100 batches, recomienda sign(beta)*min(1,|t|/2) si |t|>=1,
cumplimiento con ruido 0.3); (S2) piloto aleatorizado +-1 sd de 80 batches y luego explotacion con 30 % de batches de exploracion permanente.
Historia inicial: 100 batches observacionales. Horizonte: 400 batches (~4.4 meses a 3/dia)."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7
t = W7.cargar_tabla(); C = W7.C_SIN_ESPESOR + ["F_lnirf_F0"]; d = t.dropna(subset=C + ["xG", "xC", W7.KPI])
resid = W7.ols_classic(d[C + ["xG", "xC"]].to_numpy(float), d[W7.KPI].to_numpy(float))["resid"]; nres = len(resid)
H0, H, W = 100, 400, 100
esc = {"constante -1.1": np.full(H0 + H, -1.1), "constante -0.5": np.full(H0 + H, -0.5), "nulo": np.zeros(H0 + H),
       "deriva estimada": np.interp(np.arange(H0 + H), [0, 120, 250, 380, 499], [-0.3, -2.6, -1.2, 0.4, 0.4]),
       "cambio de signo": np.r_[np.full(H0 + H // 2, -1.5), np.full(H - H // 2, 1.5)]}
rng = np.random.default_rng(3)
def ruido(n):
    st = rng.integers(0, nres - 10, size=n // 10 + 1); return np.concatenate([resid[s:s + 10] for s in st])[:n]
def beta_hat(x, y):
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    se = np.sqrt(r @ r / (len(x) - 2) * np.linalg.inv(X.T @ X)[1, 1]); return b[1], se
def correr(beta, estrategia):
    n = H0 + H; e = ruido(n); x = np.zeros(n); y = np.zeros(n); x[:H0] = rng.normal(0, 0.64, H0); y[:H0] = beta[:H0] * x[:H0] + e[:H0]
    b, se = 0.0, 1.0
    for i in range(H0, n):
        if (i - H0) % 10 == 0:
            b, se = beta_hat(x[max(0, i - W):i], y[max(0, i - W):i])
        tt = b / se; rec = np.sign(b) * min(1.0, abs(tt) / 2) if abs(tt) >= 1 else 0.0
        if estrategia == "S0":   x[i] = rng.normal(0, 0.64)
        elif estrategia == "S1": x[i] = rec + rng.normal(0, 0.3)
        else:
            explora = (i - H0 < 80) or (rng.random() < 0.30)
            x[i] = (rng.choice([-1.0, 1.0]) if explora else rec) + rng.normal(0, 0.3)
        y[i] = beta[i] * x[i] + e[i]
    g = beta[H0:] * x[H0:]                       # ganancia REAL de KPI respecto de la practica habitual (x=0), sin ruido
    return g.mean(), g[80:].mean(), se
filas = []
for nm, beta in esc.items():
    for s in ("S0", "S1", "S2"):
        R = np.array([correr(beta, s) for _ in range(400)])
        filas.append({"escenario": nm, "estrategia": s, "ganancia_pp_batch": R[:, 0].mean(), "P5": np.percentile(R[:, 0], 5), "P95": np.percentile(R[:, 0], 95),
                      "P(ganancia<0)": (R[:, 0] < 0).mean(), "ganancia_tras_piloto_pp": R[:, 1].mean(), "se_beta_final": R[:, 2].mean()})
out = pd.DataFrame(filas); out.to_csv("experimentos/v10/e12_09_lazo_cerrado.csv", index=False)
pd.set_option("display.width", 220); print(out.round(3).to_string())
