"""E15-02 punto 8: magnitud honesta. (a) pendiente de calibracion (score prospectivo JS W=100) con IC
bootstrap por bloques de 10. (b) ganancia calibrada, en pp de KPI por batch, de "ejecutar el GN 1 sd por
debajo de donde se ejecuto" (fija d_gn=-1, receta original) y de "ejecutar exactamente la receta" (dev=0
en GN y carbon) relativo al comportamiento DEV promedio real, con bootstrap por bloques de 10 (IC95,
P(>0))."""
import sys, time
sys.path.insert(0, "experimentos/v13")
import numpy as np, pandas as pd
import e15_02_lib as E
import e11_07_lib as W7
import e13_03_lib as L13

t0 = time.time()
_, t = E.cargar()
REPS = 2000
rng = np.random.default_rng(0)


def bootstrap_bloques(n, tam, rng):
    """Bloques cronologicos contiguos de tamano `tam`, remuestreados CON reemplazo hasta cubrir n filas."""
    bloques = [np.arange(i, min(i + tam, n)) for i in range(0, n, tam)]
    elegidos = rng.choice(len(bloques), size=len(bloques), replace=True)
    idxs = np.concatenate([bloques[i] for i in elegidos])
    return idxs[:n] if len(idxs) >= n else idxs


# --- (a) calibracion: pendiente del score prospectivo (JS, W=100) con IC bootstrap bloques de 10
P = W7.preparar(t, ["d_gn", "d_carbon"])
X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
sc = L13.wf(X, C, y, 100, modo="js")
mask = ~np.isnan(sc)
s_obs, y_obs, C_obs = sc[mask], y[mask], C[mask]
nobs = mask.sum()
pends = np.empty(REPS)
for i in range(REPS):
    ix = bootstrap_bloques(nobs, 10, rng)
    r_ = W7.ols_classic(np.column_stack([s_obs[ix], C_obs[ix]]), y_obs[ix])
    pends[i] = r_["beta"][1]
ci_lo, ci_hi = np.percentile(pends, [2.5, 97.5])
p_gt0 = float(np.mean(pends > 0))
print(f"(a) calibracion: pendiente obs = {W7.stats_final(sc, y, C)['pendiente']:.3f} | "
      f"bootstrap IC95 [{ci_lo:.3f}, {ci_hi:.3f}] | P(pendiente>0) = {p_gt0:.4f}  [{time.time()-t0:.0f}s]")
pd.DataFrame({"pendiente_boot": pends}).to_csv("experimentos/v13/e15_02_08_calibracion_boot.csv", index=False)

# --- (b) ganancia calibrada usando la regresion DEV estatica (d_gn, d_carbon, controles) por batch
dev = t[t.es_dev].dropna(subset=["d_gn", "d_carbon", E.KPI] + E.CTRL).sort_values("idx_cronologico").reset_index(drop=True)
Xd = dev[["d_gn", "d_carbon"] + E.CTRL].to_numpy(float)
yd = dev[E.KPI].to_numpy(float)  # KPI ya esta en puntos porcentuales
nd = len(dev)
b_gn_rep = np.empty(REPS); b_c_rep = np.empty(REPS); mu_gn_rep = np.empty(REPS); mu_c_rep = np.empty(REPS)
for i in range(REPS):
    ix = bootstrap_bloques(nd, 10, rng)
    r_ = W7.ols_classic(Xd[ix], yd[ix])
    b_gn_rep[i], b_c_rep[i] = r_["beta"][1], r_["beta"][2]
    mu_gn_rep[i], mu_c_rep[i] = Xd[ix, 0].mean(), Xd[ix, 1].mean()

gain_1sd_gn = -b_gn_rep  # ejecutar GN 1 sd por debajo de lo ejecutado (d_gn -> d_gn - 1)
gain_receta_exacta = -(b_gn_rep * mu_gn_rep + b_c_rep * mu_c_rep)  # llevar dev a 0 desde el promedio real DEV

for nombre, arr in (("gain_1sd_menos_GN", gain_1sd_gn), ("gain_receta_exacta(dev=0)", gain_receta_exacta)):
    lo, me, hi = np.percentile(arr, [2.5, 50, 97.5])
    pg0 = float(np.mean(arr > 0))
    print(f"(b) {nombre}: mediana {me:.3f} pp | IC95 [{lo:.3f}, {hi:.3f}] | P(>0) = {pg0:.4f}")

out_b = pd.DataFrame({"gain_1sd_menos_GN": gain_1sd_gn, "gain_receta_exacta": gain_receta_exacta})
out_b.to_csv("experimentos/v13/e15_02_08_ganancias_boot.csv", index=False)
print(f"\n[media real DEV] d_gn={Xd[:,0].mean():.3f} d_carbon={Xd[:,1].mean():.3f} (n={nd})")
print(f"b_gn observado (DEV, punto): {W7.ols_classic(Xd, yd)['beta'][1]:.3f}")
print(f"[{time.time()-t0:.0f}s]")
