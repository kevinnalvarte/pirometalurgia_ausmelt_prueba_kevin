"""ST-04 -- robustez de los 36 targets candidatos por escalon (Fusion/Reduccion).

Ver `experimentos/search_targets/SEARCH_TARGETS_diseno.md`. Cuatro bloques:
  1) agregaciones alternativas (sum/mean/twmean/last/first/min/max/median) vs la
     agregacion "coherente" registrada en `st_targets.REGISTRO` (copia de st_batch.csv).
  2) estabilidad temporal: 3 tercios cronologicos de DEV + lockbox como 4o bloque,
     y correlacion con el KPI residualizado (controles + idx_cronologico).
  3) sensibilidad al ruido del trazador: Monte Carlo perturbando %CaO/%Sn/%FeO de
     escoria y recalculando TODA la cadena de inventarios/targets desde cero.
  4) estabilidad del ranking: bootstrap de batches en DEV.

Datos: `cache_df_st.pkl` (escalones, con st_<id> ya incluidos -- se recalculan desde
las columnas base para evitar duplicados) y `st_batch.csv` (batches, referencia).
Salidas: st_04_agregaciones.csv, st_04_estabilidad_temporal.csv, st_04_ruido.csv,
st_04_ranking_bootstrap.csv, st_04_log.txt, figs/st_04_*.png.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import st_targets as st  # noqa: E402

FIGS = BASE / "figs"
FIGS.mkdir(exist_ok=True)

LOGF = open(BASE / "st_04_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOGF.write(msg + "\n")
    LOGF.flush()


RNG_SEED = 42
KPI = "rendimiento_proxy_batch"
REGISTRO = st.REGISTRO
CAND_IDS = list(REGISTRO.keys())
FASE_DE = {k: v["fase"] for k, v in REGISTRO.items()}
CANDS_POR_FASE = {
    fase: [k for k in CAND_IDS if FASE_DE[k] == fase] for fase in st.FASES
}

t_inicio = time.time()
log("=" * 92)
log("ST-04 ROBUSTEZ -- targets candidatos por escalon (Fusion/Reduccion)")
log(f"inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"candidatos: {len(CAND_IDS)} ({len(CANDS_POR_FASE['Reducción'])} Reduccion, "
    f"{len(CANDS_POR_FASE['Fusión'])} Fusion)")


def spearman_np(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan, np.nan, int(m.sum())
    rho, p = stats.spearmanr(x[m], y[m])
    return float(rho), float(p), int(m.sum())


# =============================================================================
# 0) DATOS -- reconstruir desde columnas base (cache_df_st.pkl ya trae st_<id>
#    horneados; los recalculamos desde cero para no duplicar columnas y para
#    poder reproducir exactamente el mismo pipeline al perturbar en el bloque 3)
# =============================================================================
df_raw = pd.read_pickle(BASE / "cache_df_st.pkl")
BASE_COLS = [c for c in df_raw.columns if not c.startswith("st_")]
df0 = df_raw[BASE_COLS].sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
batch_csv = pd.read_csv(BASE / "st_batch.csv").set_index("Batch")

df_st0 = st.construir_df_targets(df0)
batch0 = st.construir_batch(df_st0)

diff_kpi = (batch0[KPI] - batch_csv.loc[batch0.index, KPI]).abs().max()
diff_r01 = (batch0["st_R01_sn_ext"] - batch_csv.loc[batch0.index, "st_R01_sn_ext"]).abs().max()
log(f"reconstruccion vs st_batch.csv: max|d KPI|={diff_kpi:.2e}, max|d st_R01_sn_ext|={diff_r01:.2e}")
assert diff_kpi < 1e-6 and diff_r01 < 1e-6, "la reconstruccion no coincide con st_batch.csv"

es_lockbox = batch0["es_lockbox"]
dev_idx = batch0.index[~es_lockbox]
lockbox_idx = batch0.index[es_lockbox]
log(f"DEV n={len(dev_idx)}, lockbox n={len(lockbox_idx)}, total={len(batch0)}")

# =============================================================================
# 1) AGREGACIONES ALTERNATIVAS
# =============================================================================
log("\n" + "=" * 92)
log("1) AGREGACIONES ALTERNATIVAS (sum/mean/twmean/last/first/min/max/median vs coherente)")
t0 = time.time()

REGLAS_SIMPLES = ["sum", "mean", "twmean", "last", "first", "min", "max", "median"]


def _agg_simple(x: pd.Series, dt: pd.Series, rule: str) -> float:
    m = x.notna()
    if rule == "twmean":
        m = m & dt.gt(0)
    if not m.any():
        return np.nan
    xv = x[m]
    if rule == "sum":
        return float(xv.sum())
    if rule == "mean":
        return float(xv.mean())
    if rule == "median":
        return float(xv.median())
    if rule == "min":
        return float(xv.min())
    if rule == "max":
        return float(xv.max())
    if rule == "twmean":
        return float(np.average(xv, weights=dt[m]))
    if rule == "last":
        return float(xv.iloc[-1])
    if rule == "first":
        return float(xv.iloc[0])
    raise ValueError(rule)


filas_alt = []
for b, g in df_st0.groupby("Batch", sort=False):
    g = g.sort_values("fecha_inicio")
    gf, gr = g[g["fase_proceso"] == "Fusión"], g[g["fase_proceso"] == "Reducción"]
    fila = {"Batch": b}
    for id_ in CAND_IDS:
        sub = gf if FASE_DE[id_] == "Fusión" else gr
        col = "st_" + id_
        for rule in REGLAS_SIMPLES:
            fila[f"{id_}__{rule}"] = _agg_simple(sub[col], sub["dt"], rule) if len(sub) else np.nan
    filas_alt.append(fila)
alt = pd.DataFrame(filas_alt).set_index("Batch")
alt[KPI] = batch0[KPI]

filas_res = []
for id_ in CAND_IDS:
    fase = FASE_DE[id_]
    reglas_vals = {}
    for rule in REGLAS_SIMPLES:
        col = f"{id_}__{rule}"
        rho_d, p_d, n_d = spearman_np(alt.loc[dev_idx, col], alt.loc[dev_idx, KPI])
        rho_l, p_l, n_l = spearman_np(alt.loc[lockbox_idx, col], alt.loc[lockbox_idx, KPI])
        reglas_vals[rule] = dict(rho_dev=rho_d, p_dev=p_d, n_dev=n_d,
                                  rho_lockbox=rho_l, p_lockbox=p_l, n_lockbox=n_l)
    rho_c_d, p_c_d, n_c_d = spearman_np(batch0.loc[dev_idx, "st_" + id_], batch0.loc[dev_idx, KPI])
    rho_c_l, p_c_l, n_c_l = spearman_np(batch0.loc[lockbox_idx, "st_" + id_], batch0.loc[lockbox_idx, KPI])
    reglas_vals["coherente"] = dict(rho_dev=rho_c_d, p_dev=p_c_d, n_dev=n_c_d,
                                     rho_lockbox=rho_c_l, p_lockbox=p_c_l, n_lockbox=n_c_l)
    for rule, v in reglas_vals.items():
        filas_res.append(dict(id=id_, fase=fase, regla=rule, agg_registrada=REGISTRO[id_]["agg"], **v))

agregaciones_long = pd.DataFrame(filas_res)

resumen_agg = []
for id_ in CAND_IDS:
    sub = agregaciones_long[agregaciones_long["id"] == id_]
    coh = sub[sub["regla"] == "coherente"].iloc[0]
    simples = sub[sub["regla"] != "coherente"]
    if simples["rho_dev"].notna().any():
        best = simples.loc[simples["rho_dev"].idxmax()]
    else:
        best = None
    rho_coh = coh["rho_dev"]
    rho_best = best["rho_dev"] if best is not None else np.nan
    resumen_agg.append(dict(
        id=id_, fase=FASE_DE[id_], agg_registrada=REGISTRO[id_]["agg"],
        rho_coherente_dev=rho_coh, p_coherente_dev=coh["p_dev"],
        mejor_regla_dev=best["regla"] if best is not None else np.nan,
        rho_mejor_dev=rho_best,
        brecha=(rho_best - rho_coh) if pd.notna(rho_best) and pd.notna(rho_coh) else np.nan,
    ))
resumen_agg = pd.DataFrame(resumen_agg).set_index("id")

piv = agregaciones_long.pivot(index="id", columns="regla",
                               values=["rho_dev", "p_dev", "n_dev", "rho_lockbox", "p_lockbox", "n_lockbox"])
piv.columns = [f"{a}__{b}" for a, b in piv.columns]
piv = piv.reset_index().set_index("id")
out1 = piv.join(resumen_agg[["fase", "agg_registrada", "rho_coherente_dev", "p_coherente_dev",
                              "mejor_regla_dev", "rho_mejor_dev", "brecha"]])
out1 = out1.reset_index()
out1.to_csv(BASE / "st_04_agregaciones.csv", index=False)
agregaciones_long.to_csv(BASE / "st_04_agregaciones_long.csv", index=False)

log(f"st_04_agregaciones.csv escrito ({time.time()-t0:.1f}s). "
    f"n_candidatos={len(resumen_agg)}, brecha media={resumen_agg['brecha'].mean():.3f}, "
    f"brecha max={resumen_agg['brecha'].max():.3f}")
grandes = resumen_agg[resumen_agg["brecha"] > 0.05].sort_values("brecha", ascending=False)
log(f"candidatos con brecha > 0.05 (regla incoherente gana claramente): {len(grandes)}")
for id_, r in grandes.iterrows():
    log(f"  {id_} [{r['fase']}] agg_registrada={r['agg_registrada']} coherente_rho={r['rho_coherente_dev']:.3f} "
        f"-> mejor={r['mejor_regla_dev']} rho={r['rho_mejor_dev']:.3f} (brecha={r['brecha']:.3f})")

# =============================================================================
# 2) ESTABILIDAD TEMPORAL
# =============================================================================
log("\n" + "=" * 92)
log("2) ESTABILIDAD TEMPORAL (3 tercios cronologicos de DEV + lockbox)")
t0 = time.time()

dev_sorted = batch0.loc[dev_idx].sort_values("idx_cronologico")
terciles = np.array_split(np.array(dev_sorted.index), 3)
bloques = {"tercio1": terciles[0], "tercio2": terciles[1], "tercio3": terciles[2], "lockbox": np.array(lockbox_idx)}
log("tamano de bloques: " + ", ".join(f"{k}={len(v)}" for k, v in bloques.items()))

controles = st.CONTROLES_BATCH
X = batch0[controles + ["idx_cronologico"]].copy()
X = sm.add_constant(X)
y = batch0[KPI]
m_ols = X.notna().all(axis=1) & y.notna()
modelo_ols = sm.OLS(y[m_ols], X[m_ols]).fit()
resid = pd.Series(np.nan, index=batch0.index)
resid.loc[m_ols] = modelo_ols.resid
log(f"OLS KPI ~ controles + idx_cronologico (muestra total, n={int(m_ols.sum())}): "
    f"R2={modelo_ols.rsquared:.3f}")

filas_temp = []
for id_ in CAND_IDS:
    col = batch0["st_" + id_]
    rhos = {}
    for nombre, idxb in bloques.items():
        rho, p, n = spearman_np(col.loc[idxb], batch0.loc[idxb, KPI])
        rhos[f"rho_{nombre}"] = rho
        rhos[f"p_{nombre}"] = p
        rhos[f"n_{nombre}"] = n
    vals = [rhos[f"rho_{b}"] for b in bloques if pd.notna(rhos[f"rho_{b}"])]
    n_pos = sum(1 for v in vals if v > 0)
    rho_res, p_res, n_res = spearman_np(col, resid)
    filas_temp.append(dict(id=id_, fase=FASE_DE[id_], **rhos,
                            n_bloques_validos=len(vals),
                            n_bloques_signo_positivo=n_pos,
                            rho_min=min(vals) if vals else np.nan,
                            rho_max=max(vals) if vals else np.nan,
                            rho_media=float(np.mean(vals)) if vals else np.nan,
                            rho_residual_total=rho_res, p_residual_total=p_res, n_residual_total=n_res))
estabilidad_temporal = pd.DataFrame(filas_temp).set_index("id")
estabilidad_temporal.to_csv(BASE / "st_04_estabilidad_temporal.csv")
log(f"st_04_estabilidad_temporal.csv escrito ({time.time()-t0:.1f}s)")
robustos_temp = estabilidad_temporal[estabilidad_temporal["n_bloques_signo_positivo"] >= 3]
log(f"candidatos con signo positivo en >=3/4 bloques: {len(robustos_temp)}/{len(estabilidad_temporal)}")

# =============================================================================
# 3) SENSIBILIDAD AL RUIDO DEL TRAZADOR (Monte Carlo)
# =============================================================================
log("\n" + "=" * 92)
log("3) SENSIBILIDAD AL RUIDO DEL TRAZADOR (Monte Carlo, perturbando %CaO/%Sn/%FeO)")


def perturbar(d0: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = d0.copy()
    n = len(d)
    cao0 = d["ley_cao_escoria_pct"]
    sn0 = d["ley_sn_escoria_pct"]
    feo0 = d["ley_feo_escoria_pct"]
    cao = cao0 * (1 + rng.normal(0, 0.02, n))
    sn = (sn0 + rng.normal(0, 0.05, n)).clip(lower=0)
    feo = (feo0 + rng.normal(0, 0.3, n)).clip(lower=0)
    d["ley_cao_escoria_pct"] = cao
    d["ley_sn_escoria_pct"] = sn
    d["ley_feo_escoria_pct"] = feo

    cum_cao_incl = d.groupby("Batch")["feed_CaO_kgh"].cumsum()
    masa = cum_cao_incl / (cao.where(cao > 0.5) / 100)
    sn_inv = masa * sn / 100
    feo_inv = masa * feo / 100
    d["masa_escoria_est_kg"] = masa
    d["sn_inventario_escoria_est_kg"] = sn_inv
    d["feo_inventario_escoria_est_kg"] = feo_inv
    masa_prev = masa.groupby(d["Batch"]).shift(1)
    sn_inv_prev = sn_inv.groupby(d["Batch"]).shift(1)
    feo_inv_prev = feo_inv.groupby(d["Batch"]).shift(1)
    d["masa_escoria_est_kg_prev"] = masa_prev
    d["sn_inventario_escoria_est_kg_prev"] = sn_inv_prev
    d["feo_inventario_escoria_est_kg_prev"] = feo_inv_prev

    d_masa = masa - masa_prev
    d_sn_inv = sn_inv - sn_inv_prev
    sn_extraido = d["feed_Sn_kgf"] - d_sn_inv
    disponible = sn_inv_prev + d["feed_Sn_kgf"]
    d["d_masa_escoria_est_kg"] = d_masa
    d["sn_extraido_est_kg"] = sn_extraido
    d["frac_sn_extraido_escalon"] = sn_extraido / disponible.where(disponible.abs() > 20.0)
    feo_extraido = -(feo_inv - feo_inv_prev)
    d["feo_extraido_est_kg"] = feo_extraido

    sio2 = d["ley_sio2_escoria_pct"]
    d["basicidad_B2"] = cao / sio2.where(sio2.abs() > 1e-6)
    denom_irf = sio2 + d["ley_al2o3_escoria_pct"] + cao
    d["indice_irf"] = feo / denom_irf.where(denom_irf.abs() > 1e-6)
    return d


# calibrar N de replicas por tiempo real de una iteracion
rng_probe = np.random.default_rng(RNG_SEED)
t_probe = time.time()
d_p = perturbar(df0, rng_probe)
d_p_st = st.construir_df_targets(d_p)
b_p = st.construir_batch(d_p_st)
t_por_rep = time.time() - t_probe
log(f"tiempo de 1 replica (perturbar + construir_df_targets + construir_batch): {t_por_rep:.1f}s")

N_MC_OBJETIVO = 200
presupuesto_s = 40 * 60  # 40 min tope razonable para este bloque
n_mc = min(N_MC_OBJETIVO, max(20, int(presupuesto_s / max(t_por_rep, 0.1))))
if n_mc < N_MC_OBJETIVO:
    log(f"AVISO: 200 replicas tomarian ~{200*t_por_rep/60:.0f} min; se reduce a {n_mc} replicas "
        f"(~{n_mc*t_por_rep/60:.0f} min) para mantener el costo razonable.")
else:
    n_mc = N_MC_OBJETIVO
    log(f"usando las {N_MC_OBJETIVO} replicas solicitadas (~{N_MC_OBJETIVO*t_por_rep/60:.0f} min estimados)")

t0 = time.time()
rng = np.random.default_rng(RNG_SEED)
rho_mat = np.full((n_mc, len(CAND_IDS)), np.nan)
p_mat = np.full((n_mc, len(CAND_IDS)), np.nan)
rank_mat = np.full((n_mc, len(CAND_IDS)), np.nan)
id_pos = {id_: j for j, id_ in enumerate(CAND_IDS)}

for rep in range(n_mc):
    d_p = perturbar(df0, rng)
    d_p_st = st.construir_df_targets(d_p)
    b_p = st.construir_batch(d_p_st)
    b_p_dev = b_p.loc[b_p.index.intersection(dev_idx)]
    kpi_rep = b_p_dev[KPI]
    rho_rep = {}
    for id_ in CAND_IDS:
        rho, p, n = spearman_np(b_p_dev["st_" + id_], kpi_rep)
        rho_mat[rep, id_pos[id_]] = rho
        p_mat[rep, id_pos[id_]] = p
        rho_rep[id_] = rho
    for fase, ids in CANDS_POR_FASE.items():
        s = pd.Series({i: rho_rep[i] for i in ids})
        ranks = s.rank(ascending=False, method="min", na_option="bottom")
        for i in ids:
            rank_mat[rep, id_pos[i]] = ranks[i]
    if (rep + 1) % max(1, n_mc // 10) == 0 or rep == n_mc - 1:
        log(f"  MC replica {rep+1}/{n_mc} ({time.time()-t0:.0f}s transcurridos)")

filas_ruido = []
for id_ in CAND_IDS:
    j = id_pos[id_]
    rho_col = rho_mat[:, j]
    p_col = p_mat[:, j]
    rank_col = rank_mat[:, j]
    ok = np.isfinite(rho_col)
    rho_base, p_base, n_base = spearman_np(batch0.loc[dev_idx, "st_" + id_], batch0.loc[dev_idx, KPI])
    filas_ruido.append(dict(
        id=id_, fase=FASE_DE[id_], n_replicas=int(ok.sum()),
        rho_dev_base=rho_base,
        rho_mc_media=float(np.nanmean(rho_col)) if ok.any() else np.nan,
        rho_mc_sd=float(np.nanstd(rho_col)) if ok.any() else np.nan,
        rho_mc_p5=float(np.nanpercentile(rho_col, 5)) if ok.any() else np.nan,
        rho_mc_p95=float(np.nanpercentile(rho_col, 95)) if ok.any() else np.nan,
        frac_rho_positivo=float(np.nanmean(rho_col > 0)) if ok.any() else np.nan,
        frac_rho_pos_y_sig005=float(np.nanmean((rho_col > 0) & (p_col < 0.05))) if ok.any() else np.nan,
        rango_medio_ruido=float(np.nanmean(rank_col)) if np.isfinite(rank_col).any() else np.nan,
    ))
ruido = pd.DataFrame(filas_ruido).set_index("id")
ruido.to_csv(BASE / "st_04_ruido.csv")
log(f"st_04_ruido.csv escrito ({time.time()-t0:.1f}s, {n_mc} replicas)")
fragiles_ruido = ruido[ruido["frac_rho_pos_y_sig005"] < 0.5].sort_values("frac_rho_pos_y_sig005")
log(f"candidatos que pierden significancia positiva en >=50% de las replicas (fragiles al ruido): "
    f"{len(fragiles_ruido)}/{len(ruido)}")

# =============================================================================
# 4) ESTABILIDAD DEL RANKING (bootstrap de batches en DEV)
# =============================================================================
log("\n" + "=" * 92)
log("4) ESTABILIDAD DEL RANKING (bootstrap de batches, DEV, 500 replicas)")
t0 = time.time()

N_BOOT = 500
rng4 = np.random.default_rng(RNG_SEED)
dev_arr = np.array(dev_idx)
n_dev_ = len(dev_arr)
vals_by_cand = {id_: batch0.loc[dev_idx, "st_" + id_].to_numpy(dtype=float) for id_ in CAND_IDS}
kpi_dev = batch0.loc[dev_idx, KPI].to_numpy(dtype=float)

boot_ranks = {id_: np.full(N_BOOT, np.nan) for id_ in CAND_IDS}
boot_rho = {id_: np.full(N_BOOT, np.nan) for id_ in CAND_IDS}
for rep in range(N_BOOT):
    sel = rng4.integers(0, n_dev_, n_dev_)
    rho_rep = {}
    for id_ in CAND_IDS:
        x = vals_by_cand[id_][sel]
        y = kpi_dev[sel]
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < 8:
            rho_rep[id_] = np.nan
        else:
            rho_rep[id_] = stats.spearmanr(x[mm], y[mm])[0]
        boot_rho[id_][rep] = rho_rep[id_]
    for fase, ids in CANDS_POR_FASE.items():
        s = pd.Series({i: rho_rep[i] for i in ids})
        ranks = s.rank(ascending=False, method="min", na_option="bottom")
        for i in ids:
            boot_ranks[i][rep] = ranks[i]

filas_boot = []
for id_ in CAND_IDS:
    r = boot_ranks[id_]
    filas_boot.append(dict(
        id=id_, fase=FASE_DE[id_],
        rho_boot_media=float(np.nanmean(boot_rho[id_])),
        rango_mediano=float(np.nanmedian(r)),
        rango_p25=float(np.nanpercentile(r, 25)),
        rango_p75=float(np.nanpercentile(r, 75)),
        prob_top3=float(np.nanmean(r <= 3)),
    ))
ranking_bootstrap = pd.DataFrame(filas_boot).set_index("id")
ranking_bootstrap.to_csv(BASE / "st_04_ranking_bootstrap.csv")
log(f"st_04_ranking_bootstrap.csv escrito ({time.time()-t0:.1f}s)")

# =============================================================================
# 5) FIGURAS
# =============================================================================
log("\n" + "=" * 92)
log("5) FIGURAS")


def _heatmap_fase(fase: str, ax):
    ids = CANDS_POR_FASE[fase]
    reglas_cols = REGLAS_SIMPLES + ["coherente"]
    mat = np.full((len(ids), len(reglas_cols)), np.nan)
    for i, id_ in enumerate(ids):
        sub = agregaciones_long[agregaciones_long["id"] == id_].set_index("regla")
        for jc, rule in enumerate(reglas_cols):
            if rule in sub.index:
                mat[i, jc] = sub.loc[rule, "rho_dev"]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(reglas_cols)))
    ax.set_xticklabels(reglas_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_title(f"{fase}: rho Spearman (DEV) por regla de agregacion")
    for i in range(len(ids)):
        for jc in range(len(reglas_cols)):
            v = mat[i, jc]
            if np.isfinite(v):
                ax.text(jc, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                         color="white" if abs(v) > 0.25 else "black")
    return im


fig, axes = plt.subplots(1, 2, figsize=(15, 9))
im0 = _heatmap_fase("Reducción", axes[0])
im1 = _heatmap_fase("Fusión", axes[1])
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
fig.suptitle("ST-04.1 -- Agregaciones alternativas: rho(candidato, KPI) por regla, DEV")
fig.tight_layout()
fig.savefig(FIGS / "st_04_heatmap_agregaciones.png", dpi=130)
plt.close(fig)
log("figs/st_04_heatmap_agregaciones.png")

# rho por bloque temporal (lineas), 6 mejores por fase (por rho_coherente_dev)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
bloque_orden = ["tercio1", "tercio2", "tercio3", "lockbox"]
for ax, fase in zip(axes, st.FASES):
    top6 = resumen_agg[resumen_agg["fase"] == fase].sort_values("rho_coherente_dev", ascending=False).head(6)
    for id_ in top6.index:
        row = estabilidad_temporal.loc[id_]
        ys = [row[f"rho_{b}"] for b in bloque_orden]
        ax.plot(bloque_orden, ys, marker="o", label=id_)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title(fase)
    ax.legend(fontsize=6.5, loc="best")
    ax.set_ylabel("rho Spearman vs KPI")
fig.suptitle("ST-04.2 -- Estabilidad temporal: rho por bloque (6 mejores por fase)")
fig.tight_layout()
fig.savefig(FIGS / "st_04_estabilidad_temporal.png", dpi=130)
plt.close(fig)
log("figs/st_04_estabilidad_temporal.png")

# boxplot ruido, 8 mejores por fase
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, fase in zip(axes, st.FASES):
    top8 = resumen_agg[resumen_agg["fase"] == fase].sort_values("rho_coherente_dev", ascending=False).head(8)
    data = [rho_mat[:, id_pos[id_]][np.isfinite(rho_mat[:, id_pos[id_]])] for id_ in top8.index]
    ax.boxplot(data, tick_labels=list(top8.index), showmeans=True)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.set_title(fase)
    ax.set_ylabel("rho Spearman (MC, ruido trazador)")
fig.suptitle(f"ST-04.3 -- Sensibilidad al ruido: rho bajo {n_mc} perturbaciones (8 mejores por fase)")
fig.tight_layout()
fig.savefig(FIGS / "st_04_boxplot_ruido.png", dpi=130)
plt.close(fig)
log("figs/st_04_boxplot_ruido.png")

# =============================================================================
# 6) VEREDICTO -- robustos vs fragiles por fase
# =============================================================================
log("\n" + "=" * 92)
log("6) VEREDICTO POR FASE: candidatos ROBUSTOS vs FRAGILES/ARTEFACTO")
log("Criterios: ROBUSTO = signo positivo en >=3/4 bloques temporales, Y frac_rho_pos_y_sig005 (MC) >= 0.5, "
    "Y brecha (mejor_regla_incoherente - coherente) < 0.05.")

resumen_final = resumen_agg.join(estabilidad_temporal[["n_bloques_signo_positivo", "rho_min", "rho_max",
                                                         "rho_media", "rho_residual_total"]])
resumen_final = resumen_final.join(ruido[["rho_mc_media", "rho_mc_sd", "frac_rho_positivo",
                                           "frac_rho_pos_y_sig005", "rango_medio_ruido"]])
resumen_final = resumen_final.join(ranking_bootstrap[["rango_mediano", "prob_top3"]])
resumen_final["robusto"] = (
    (resumen_final["n_bloques_signo_positivo"] >= 3)
    & (resumen_final["frac_rho_pos_y_sig005"] >= 0.5)
    & (resumen_final["brecha"].fillna(1.0) < 0.05)
)
resumen_final.to_csv(BASE / "st_04_resumen_final.csv")

for fase in st.FASES:
    sub = resumen_final[resumen_final["fase"] == fase].sort_values("rho_coherente_dev", ascending=False)
    log(f"\n--- {fase} ---")
    robustos = sub[sub["robusto"]]
    fragiles = sub[~sub["robusto"]]
    log(f"ROBUSTOS ({len(robustos)}/{len(sub)}):")
    for id_, r in robustos.iterrows():
        log(f"  {id_}: rho_coherente_dev={r['rho_coherente_dev']:.3f} (p={r['p_coherente_dev']:.4f}), "
            f"bloques+={int(r['n_bloques_signo_positivo'])}/4, rho_mc={r['rho_mc_media']:.3f}+-{r['rho_mc_sd']:.3f} "
            f"(frac sig={r['frac_rho_pos_y_sig005']:.2f}), brecha={r['brecha']:.3f}, "
            f"rango_mediano_boot={r['rango_mediano']:.0f}, P(top3)={r['prob_top3']:.2f}")
    log(f"FRAGILES / POSIBLE ARTEFACTO ({len(fragiles)}/{len(sub)}):")
    for id_, r in fragiles.iterrows():
        motivos = []
        if r["n_bloques_signo_positivo"] < 3:
            motivos.append(f"signo inestable ({int(r['n_bloques_signo_positivo'])}/4 bloques +)")
        if pd.isna(r["frac_rho_pos_y_sig005"]) or r["frac_rho_pos_y_sig005"] < 0.5:
            fv = r["frac_rho_pos_y_sig005"]
            motivos.append(f"se desvanece con ruido (frac sig={fv:.2f})" if pd.notna(fv) else "ruido: sin datos")
        if pd.notna(r["brecha"]) and r["brecha"] >= 0.05:
            motivos.append(f"depende de regla incoherente {r['mejor_regla_dev']} (brecha={r['brecha']:.3f})")
        log(f"  {id_}: rho_coherente_dev={r['rho_coherente_dev']:.3f} -- " + "; ".join(motivos))

log(f"\ntiempo total: {(time.time()-t_inicio)/60:.1f} min")
log(f"fin: {time.strftime('%Y-%m-%d %H:%M:%S')}")
LOGF.close()
