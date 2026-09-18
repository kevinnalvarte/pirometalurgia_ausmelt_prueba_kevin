"""E12-02 tarea 2: evaluacion prospectiva de los 6+1 esquemas de valoracion por canal (A-F, con C en dos variantes
de ventana para el canal inestable). Para cada esquema: pendiente HC3, t, Spearman parcial, p por permutacion en
bloques cronologicos de 20 (N_PERM>=1000), calibracion, diferencia de tercios; en total, DEV-prospectivo y lockbox.
Guarda tambien las distribuciones nulas (npz) para la correccion de multiplicidad (e12_02_03)."""
import sys, time; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
from scipy import stats as sst
import e12_02_lib as C
import e11_07_lib as E

N_PERM = 1500
SEED = C.SEED
rng = np.random.default_rng(SEED)

t = C.cargar()
D = C.preparar_todo(t)
Xexp, Xctrl, y_kpi, ycan, n = D["Xexp"], D["Xctrl"], D["y_kpi"], D["ycan"], D["n"]
es_dev = D["es_dev"]
MASKS = {"total": np.ones(n, bool), "dev": es_dev, "lockbox": ~es_dev}
bloques = E.bloques_cronologicos(n, 20, seed=SEED)


def tercil_diff(score_arr, y_arr, Xc_arr):
    if len(score_arr) < 12 or np.nanstd(score_arr) < 1e-10:
        return np.nan
    resid_y = E.ols_classic(Xc_arr, y_arr)["resid"]
    rank = pd.Series(score_arr).rank(method="first").to_numpy()
    q = pd.qcut(rank, 3, labels=False)
    return float(resid_y[q == 2].mean() - resid_y[q == 0].mean())


def stats_por_mascara(score):
    out = {}
    for nombre_m, m in MASKS.items():
        mv = m & ~np.isnan(score)
        r = E.stats_final(score, y_kpi, Xctrl, mv)
        diff = tercil_diff(score[mv], y_kpi[mv], Xctrl[mv])
        out[nombre_m] = {**r, "tercil_diff_pp": diff}
    return out


def stats_hc3(score, nombre_m, m):
    mv = m & ~np.isnan(score)
    return E.stats_final_hc3(score, y_kpi, D["t_df"], D["controles"], mv, C.KPI)


# --- observado -----------------------------------------------------------------------------------------------
print("=== Observado por esquema (pendiente=coef. de score sobre KPI + controles) ===")
scores_obs = {}
filas_obs = []
for nombre in C.ESQUEMAS:
    sc = C.score_esquema(nombre, Xexp, Xctrl, y_kpi, ycan, n)
    scores_obs[nombre] = sc
    stp = stats_por_mascara(sc)
    for nombre_m in MASKS:
        h = stats_hc3(sc, nombre_m, MASKS[nombre_m])
        r = stp[nombre_m]
        fila = {"esquema": nombre, "muestra": nombre_m, "n": r["n"], "pendiente_hc3": h["pendiente"],
                "t_hc3": h["t"], "spearman_hc3": h["spearman"], "pendiente_clasica": r["pendiente"],
                "t_clasico": r["t"], "spearman_clasico": r["spearman"], "tercil_diff_pp": r["tercil_diff_pp"],
                "sd_score": r.get("sd_score", np.nan)}
        filas_obs.append(fila)
        print(f"{nombre:16s} {nombre_m:8s} n={r['n']:3d} pend_hc3={h['pendiente']:+.3f} t_hc3={h['t']:+.2f} "
              f"spearman={h['spearman']:+.3f} tercil={r['tercil_diff_pp']:+.3f}")
out_obs = pd.DataFrame(filas_obs)
out_obs.to_csv(C.SALIDA / "e12_02_esquemas_observado.csv", index=False)

# --- permutacion en bloques de 20, N_PERM reps, MISMO sorteo para todos los esquemas (para task 4) ------------
print(f"\npermutando ({N_PERM} reps x {len(C.ESQUEMAS)} esquemas)...")
null_t = {nombre: {m: np.empty(N_PERM) for m in MASKS} for nombre in C.ESQUEMAS}
null_rho = {nombre: {m: np.empty(N_PERM) for m in MASKS} for nombre in C.ESQUEMAS}
t0 = time.time()
for i in range(N_PERM):
    Xp = E.permutar_bloques(Xexp, bloques, rng)
    for nombre in C.ESQUEMAS:
        scp = C.score_esquema(nombre, Xp, Xctrl, y_kpi, ycan, n)
        for nombre_m, m in MASKS.items():
            mv = m & ~np.isnan(scp)
            r = E.stats_final(scp, y_kpi, Xctrl, mv)
            null_t[nombre][nombre_m][i] = r["t"]
            null_rho[nombre][nombre_m][i] = r["spearman"]
    if (i + 1) % 300 == 0:
        print(f"  perm {i+1}/{N_PERM}  {time.time()-t0:.0f}s")
print(f"permutacion total: {time.time()-t0:.0f}s")


def p_uni(obs_v, null_arr):
    if np.isnan(obs_v):
        return np.nan
    return float((np.sum(null_arr >= obs_v) + 1) / (len(null_arr) + 1))


filas_p = []
for nombre in C.ESQUEMAS:
    for nombre_m in MASKS:
        obs_row = out_obs[(out_obs.esquema == nombre) & (out_obs.muestra == nombre_m)].iloc[0]
        p_t = p_uni(obs_row["t_clasico"], null_t[nombre][nombre_m])
        p_rho = p_uni(obs_row["spearman_clasico"], null_rho[nombre][nombre_m])
        filas_p.append({"esquema": nombre, "muestra": nombre_m, "t_clasico_obs": obs_row["t_clasico"],
                         "p_perm_bloques20_t": p_t, "spearman_obs": obs_row["spearman_clasico"], "p_perm_bloques20_rho": p_rho})
out_p = pd.DataFrame(filas_p)
out_p.to_csv(C.SALIDA / "e12_02_esquemas_pvalores.csv", index=False)
print("\n=== p por permutacion en bloques de 20 (>=1000 rep., no corregido por multiplicidad de esquemas) ===")
print(out_p.round(4).to_string())

# guardar todo para e12_02_03 (multiplicidad) y auditoria
np.savez(C.SALIDA / "e12_02_nulas_dist.npz",
         **{f"t__{nombre}__{m}": null_t[nombre][m] for nombre in C.ESQUEMAS for m in MASKS},
         **{f"rho__{nombre}__{m}": null_rho[nombre][m] for nombre in C.ESQUEMAS for m in MASKS})
np.savez(C.SALIDA / "e12_02_scores_obs.npz", **{nombre: scores_obs[nombre] for nombre in C.ESQUEMAS},
         y_kpi=y_kpi, es_dev=es_dev, idx=D["idx"].astype(str))
print("\nGuardado: e12_02_esquemas_observado.csv, e12_02_esquemas_pvalores.csv, e12_02_nulas_dist.npz, e12_02_scores_obs.npz")
