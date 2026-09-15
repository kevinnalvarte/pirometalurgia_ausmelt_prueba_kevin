"""B-04 -- balance global de masa por batch, KPI refinado con Sn en escoria, y modelo multi-oxido
de la escoria (talon + ganga por tolva). Ver experimentos/balances/ITERACION_6_diseno.md.

Secciones:
  1) Balance global: distribuciones, factor de escala del trazador (dependencias), cierre de Sn,
     tornado de sensibilidad a los supuestos de leyes/humedad/ceniza.
  2) KPI refinado: recuperacion_refinada_pct vs proxy/real; Sn en escoria final; cambio de ranking.
  3) Targets ganadores (bateria ST-01) vs KPI refinado: Spearman+IC, BH, OLS HC3, quintiles.
  4) Techo de predictibilidad del KPI refinado (patron ST-15): HGB / Ridge, OOF + lockbox.
  5) Modelo multi-oxido de la escoria (talon + ganga por tolva): log-cocientes observables,
     least_squares con cotas, validacion GroupKFold(5) por batch.

Ejecutar: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/balances/b04_balance_global.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "search_targets"))
import bal_features as bal  # noqa: E402
import st_targets as st  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)
N_BOOT = 500

LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


t0 = time.time()

# =============================================================================
# 0) Carga de datos
# =============================================================================
df = bal.cargar_df()                      # escalones + b6_* (bal_features)
batch_bal = bal.construir_batch_balance(df)  # 362 filas: balance global de masa
df_st = st.construir_df_targets(df)       # escalones + st_* (comparten el mismo df_v3 base)
batch_st = st.construir_batch(df_st)      # 362 filas: agregados st_* + KPIs proxy/real
log(f"[0] df {df.shape}, batch_bal {batch_bal.shape}, batch_st {batch_st.shape}  ({time.time()-t0:.1f}s)")
assert set(batch_bal.index) == set(batch_st.index), "los indices de batch (Batch) deben coincidir"

# duracion total de Reduccion por batch (minutos)
dur_r = df[df["fase_proceso"] == "Reducción"].groupby("Batch")["delta_tiempo"].sum()
batch_bal["duracion_reduccion_min"] = dur_r.reindex(batch_bal.index)

# agregados multi-trazador (bal, familia A) por batch: suma sobre escalones de Reduccion
mt = df[df["fase_proceso"] == "Reducción"].groupby("Batch").agg(
    b6_sdi_multi_batch=("b6_sdi_multi", "sum"),
    b6_feo_extraido_multi_kg_batch=("b6_feo_extraido_multi_kg", "sum"),
    n_b6_sdi_multi=("b6_sdi_multi", "count"),
)
batch_bal = batch_bal.join(mt.reindex(batch_bal.index))
batch_bal["b6_feo_extraido_multi_neg"] = -batch_bal["b6_feo_extraido_multi_kg_batch"]  # direccion -1: menos FeO = mejor

# tabla maestra: balance global + targets st_* + KPIs de la familia ST + controles
BATCH = batch_bal.join(batch_st[[c for c in batch_st.columns if c not in batch_bal.columns]])
BATCH.to_csv(HERE / "b04_batch_balance.csv", encoding="utf-8")
log(f"[0] b04_batch_balance.csv escrito ({BATCH.shape})")

DEV_IDX = BATCH.index[~BATCH["es_lockbox"]]
LOCK_IDX = BATCH.index[BATCH["es_lockbox"]]
MUESTRAS = {"dev": DEV_IDX, "lockbox": LOCK_IDX, "total": BATCH.index}
log(f"[0] DEV n={len(DEV_IDX)}, lockbox n={len(LOCK_IDX)}")


# =============================================================================
# utilidades estadisticas (patron st_01 / st_15)
# =============================================================================

def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    n = len(x)
    if n < 8:
        return np.nan, np.nan
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    rxb, ryb = rx[idx], ry[idx]
    mx, my = rxb.mean(axis=1, keepdims=True), ryb.mean(axis=1, keepdims=True)
    dx, dy = rxb - mx, ryb - my
    num = (dx * dy).sum(axis=1)
    den = np.sqrt((dx ** 2).sum(axis=1) * (dy ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r = r[np.isfinite(r)]
    if len(r) < n_boot * 0.5:
        return np.nan, np.nan
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def corr_pair(x: pd.Series, y: pd.Series, seed: int = 0) -> dict:
    sub = pd.concat([x, y], axis=1).dropna()
    n = len(sub)
    out = dict(n=n, rho=np.nan, p_rho=np.nan, r=np.nan, p_r=np.nan, ci_low=np.nan, ci_high=np.nan)
    if n < 8:
        return out
    xv, yv = sub.iloc[:, 0].to_numpy(float), sub.iloc[:, 1].to_numpy(float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return out
    rho, p_rho = stats.spearmanr(xv, yv)
    r, p_r = stats.pearsonr(xv, yv)
    ci_low, ci_high = bootstrap_ci_spearman(xv, yv, seed=seed)
    out.update(rho=rho, p_rho=p_rho, r=r, p_r=p_r, ci_low=ci_low, ci_high=ci_high)
    return out


def fit_ols(sub: pd.DataFrame, regressors: list[str], y_col: str):
    used = [c for c in regressors if sub[c].std(ddof=1) >= 1e-8]
    if not used:
        return None, used
    X = sm.add_constant(sub[used], has_constant="add")
    y = sub[y_col]
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception:
        return None, used
    return model, used


def percentiles(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return dict(n=0, mediana=np.nan, p5=np.nan, p25=np.nan, p75=np.nan, p95=np.nan, media=np.nan, sd=np.nan)
    return dict(n=len(s), mediana=s.median(), p5=s.quantile(.05), p25=s.quantile(.25),
                p75=s.quantile(.75), p95=s.quantile(.95), media=s.mean(), sd=s.std())


# =============================================================================
# 1) BALANCE GLOBAL: distribuciones, factor de escala, cierre de Sn, tornado
# =============================================================================
log("\n" + "=" * 78 + "\n1) BALANCE GLOBAL DE MASA\n" + "=" * 78)

COLS_BALANCE = ["carga_solida_kg", "feed_sn_kg", "carbon_kg", "cal_kg", "dross_fe_in_kg", "mineral_fe_kg",
                "pellets_kg", "sn_metal_kg", "sn_dross_kg", "sn_polvo_kg",
                "masa_escoria_balance_kg", "masa_escoria_trazador_kg", "factor_escala_trazador",
                "ley_sn_escoria_final_pct", "ley_feo_escoria_final_pct",
                "sn_escoria_final_balance_kg", "sn_escoria_final_trazador_kg",
                "cierre_sn_kg", "cierre_sn_frac", "rendimiento_proxy_batch", "f_metal", "f_dross", "f_polvo",
                "recuperacion_real_pct", "recuperacion_refinada_pct", "sn_perdido_escoria_frac",
                "duracion_reduccion_min"]
dist_rows = [dict(columna=c, **percentiles(BATCH[c])) for c in COLS_BALANCE]
dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(HERE / "b04_distribuciones.csv", index=False, encoding="utf-8")
log("\n[1.1] Distribuciones (todas las columnas del balance, n=362):")
log(dist_df.round(3).to_string(index=False))

# --- factor_escala_trazador vs dependencias
BATCH["ratio_cal_carga"] = BATCH["cal_kg"] / BATCH["carga_solida_kg"]
dep_vars = ["ratio_cal_carga", "frac_carga_secundaria_F", "pellets_kg", "idx_cronologico"]
fe_rows = []
for v in dep_vars:
    r = corr_pair(BATCH[v], BATCH["factor_escala_trazador"], seed=hash(("fe", v)) % (2**31))
    sub = BATCH[[v, "factor_escala_trazador"]].dropna()
    model, used = (None, [])
    coef = ci_lo = ci_hi = p_ols = r2 = np.nan
    if len(sub) >= 15 and sub[v].std(ddof=1) > 1e-8:
        Xc = sm.add_constant(sub[[v]], has_constant="add")
        try:
            m = sm.OLS(sub["factor_escala_trazador"], Xc).fit(cov_type="HC3")
            coef, p_ols, r2 = m.params[v], m.pvalues[v], m.rsquared
            ci = m.conf_int().loc[v]
            ci_lo, ci_hi = ci[0], ci[1]
        except Exception:
            pass
    fe_rows.append(dict(variable=v, n=r["n"], rho=r["rho"], p_rho=r["p_rho"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                         ols_coef=coef, ols_ci_low=ci_lo, ols_ci_high=ci_hi, ols_p=p_ols, ols_r2=r2))
fe_df = pd.DataFrame(fe_rows)
fe_df.to_csv(HERE / "b04_factor_escala_corr.csv", index=False, encoding="utf-8")
log("\n[1.2] factor_escala_trazador vs dependencias (Spearman + OLS univariado, total n=362):")
log(fe_df.round(4).to_string(index=False))

# --- cierre_sn_frac: media, sd, t-test vs 0, dependencia de ley_sn_conc_batch_pct y feed_dross_Fe_total_t
cierre = BATCH["cierre_sn_frac"].dropna()
t_stat, p_t = stats.ttest_1samp(cierre, 0.0)
log(f"\n[1.3] cierre_sn_frac: media={cierre.mean():.5f}, sd={cierre.std():.5f}, n={len(cierre)}, "
    f"t={t_stat:.3f}, p={p_t:.4g} (H0: media=0)")
cierre_dep_rows = []
for v in ["ley_sn_conc_batch_pct", "feed_dross_Fe_total_t"]:
    r = corr_pair(BATCH[v], BATCH["cierre_sn_frac"], seed=hash(("cierre", v)) % (2**31))
    cierre_dep_rows.append(dict(variable=v, **{k: r[k] for k in ["n", "rho", "p_rho", "ci_low", "ci_high"]}))
cierre_df = pd.DataFrame(cierre_dep_rows)
cierre_summary = pd.DataFrame([dict(media=cierre.mean(), sd=cierre.std(), n=len(cierre), t_stat=t_stat, p_ttest=p_t)])
cierre_df.to_csv(HERE / "b04_cierre_sn.csv", index=False, encoding="utf-8")
cierre_summary.to_csv(HERE / "b04_cierre_sn_ttest.csv", index=False, encoding="utf-8")
log(cierre_df.round(4).to_string(index=False))

# --- tornado de sensibilidad
log("\n[1.4] Tornado de sensibilidad de masa_escoria_balance_kg y sn_escoria_final_balance_kg:")
TORNADO_GRID = {
    "LEY_SN_METAL": [0.94, 0.96, 0.98],
    "LEY_SN_DROSS": [0.55, 0.65, 0.75],
    "LEY_SN_POLVO": [0.45, 0.55, 0.65],
    "HUMEDAD_CARGA": [0.03, 0.06, 0.09],
    "FE_MET_DROSS": [0.2, 0.3, 0.4],
    "CENIZA_CARBON": [0.05, 0.10, 0.15],
}
base_default = {k: bal.SUPUESTOS[k]["valor"] for k in TORNADO_GRID}
tornado_rows = []
for param, grid in TORNADO_GRID.items():
    for val in grid:
        sup = {param: val}
        bb = bal.construir_batch_balance(df, supuestos=sup)
        tornado_rows.append(dict(parametro=param, valor=val, es_base=(val == base_default[param]),
                                  mediana_masa_escoria_kg=bb["masa_escoria_balance_kg"].median(),
                                  mediana_sn_escoria_final_kg=bb["sn_escoria_final_balance_kg"].median(),
                                  mediana_recuperacion_refinada_pct=bb["recuperacion_refinada_pct"].median()))
tornado_df = pd.DataFrame(tornado_rows)
tornado_df.to_csv(HERE / "b04_tornado.csv", index=False, encoding="utf-8")
log(tornado_df.round(3).to_string(index=False))
tornado_rango = tornado_df.groupby("parametro").agg(
    rango_masa_escoria_kg=("mediana_masa_escoria_kg", lambda s: s.max() - s.min()),
    rango_sn_escoria_kg=("mediana_sn_escoria_final_kg", lambda s: s.max() - s.min()),
    rango_recuperacion_refinada_pp=("mediana_recuperacion_refinada_pct", lambda s: s.max() - s.min()),
).sort_values("rango_masa_escoria_kg", ascending=False)
tornado_rango.to_csv(HERE / "b04_tornado_rango.csv", encoding="utf-8")
log("\n  Rango (max-min de la mediana) por parametro, ordenado por impacto en masa_escoria_balance_kg:")
log(tornado_rango.round(3).to_string())

# =============================================================================
# 2) KPI REFINADO
# =============================================================================
log("\n" + "=" * 78 + "\n2) KPI REFINADO\n" + "=" * 78)

KPI_SET = ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_real_pct", "f_metal"]
kpi_corr_rows = []
for i, a in enumerate(KPI_SET):
    for b in KPI_SET[i + 1:]:
        sub = BATCH[[a, b]].dropna()
        rho, p_rho = stats.spearmanr(sub[a], sub[b])
        r, p_r = stats.pearsonr(sub[a], sub[b])
        kpi_corr_rows.append(dict(kpi_a=a, kpi_b=b, n=len(sub), rho=rho, p_rho=p_rho, r=r, p_r=p_r))
kpi_corr_df = pd.DataFrame(kpi_corr_rows)
kpi_sd = pd.DataFrame([dict(kpi=k, sd=BATCH[k].std(), media=BATCH[k].mean()) for k in KPI_SET])
kpi_corr_df.to_csv(HERE / "b04_kpi_correlaciones.csv", index=False, encoding="utf-8")
kpi_sd.to_csv(HERE / "b04_kpi_sd.csv", index=False, encoding="utf-8")
log("\n[2.1] Correlaciones entre KPIs (Pearson/Spearman) y sd:")
log(kpi_corr_df.round(4).to_string(index=False))
log(kpi_sd.round(4).to_string(index=False))

# distribucion de sn_perdido_escoria_frac (%) y Sn escoria: balance vs trazador
sn_perdido_pct = 100 * BATCH["sn_perdido_escoria_frac"]
dist_sn_perdido = percentiles(sn_perdido_pct)
log(f"\n[2.2] sn_perdido_escoria_frac (%% del Sn de salida): {dist_sn_perdido}")

sn_bt = BATCH[["sn_escoria_final_balance_kg", "sn_escoria_final_trazador_kg"]].dropna()
r_sn, p_sn = stats.pearsonr(sn_bt["sn_escoria_final_balance_kg"], sn_bt["sn_escoria_final_trazador_kg"])
rho_sn, prho_sn = stats.spearmanr(sn_bt["sn_escoria_final_balance_kg"], sn_bt["sn_escoria_final_trazador_kg"])
ratio_sn = (sn_bt["sn_escoria_final_balance_kg"] / sn_bt["sn_escoria_final_trazador_kg"]).replace([np.inf, -np.inf], np.nan)
log(f"[2.2b] Sn escoria final balance vs trazador: n={len(sn_bt)}, r={r_sn:.4f}, rho={rho_sn:.4f}, "
    f"ratio balance/trazador mediana={ratio_sn.median():.3f} (IQR {ratio_sn.quantile(.25):.3f}-{ratio_sn.quantile(.75):.3f})")

# Spearman Sn escoria final con ley_sn_escoria_final_pct, st_R22_sdi, st_R23_lnirf_R, duracion Reduccion
sn_dep_vars = ["ley_sn_escoria_final_pct", "st_R22_sdi", "st_R23_lnirf_R", "duracion_reduccion_min"]
sn_dep_rows = []
for v in sn_dep_vars:
    r = corr_pair(BATCH[v], BATCH["sn_escoria_final_balance_kg"], seed=hash(("sndep", v)) % (2**31))
    sn_dep_rows.append(dict(variable=v, **{k: r[k] for k in ["n", "rho", "p_rho", "ci_low", "ci_high"]}))
sn_dep_df = pd.DataFrame(sn_dep_rows)
sn_dep_df.to_csv(HERE / "b04_sn_escoria_dependencias.csv", index=False, encoding="utf-8")
log("\n[2.3] Sn en escoria final (balance, kg) vs:")
log(sn_dep_df.round(4).to_string(index=False))

# ranking: Spearman proxy vs refinado, batches que cambian de cuartil
rank_sub = BATCH[["rendimiento_proxy_batch", "recuperacion_refinada_pct"]].dropna()
rho_rank, p_rank = stats.spearmanr(rank_sub["rendimiento_proxy_batch"], rank_sub["recuperacion_refinada_pct"])
q_proxy = pd.qcut(rank_sub["rendimiento_proxy_batch"], 4, labels=[1, 2, 3, 4])
q_ref = pd.qcut(rank_sub["recuperacion_refinada_pct"], 4, labels=[1, 2, 3, 4])
cambia_cuartil = (q_proxy.astype(int) != q_ref.astype(int))
log(f"\n[2.4] Spearman(proxy, refinado) = {rho_rank:.4f} (p={p_rank:.4g}, n={len(rank_sub)}); "
    f"batches que cambian de cuartil: {cambia_cuartil.sum()}/{len(rank_sub)} ({100*cambia_cuartil.mean():.1f}%)")
ranking_df = pd.DataFrame({"rendimiento_proxy_batch": rank_sub["rendimiento_proxy_batch"],
                           "recuperacion_refinada_pct": rank_sub["recuperacion_refinada_pct"],
                           "cuartil_proxy": q_proxy.astype(int), "cuartil_refinado": q_ref.astype(int),
                           "cambia_cuartil": cambia_cuartil})
ranking_df.to_csv(HERE / "b04_ranking_cambio.csv", encoding="utf-8")

# =============================================================================
# 3) TARGETS GANADORES vs KPI REFINADO (patron ST-01)
# =============================================================================
log("\n" + "=" * 78 + "\n3) TARGETS GANADORES vs KPI REFINADO (ST-01)\n" + "=" * 78)

TARGETS_B04 = ["st_R22_sdi", "st_R23_lnirf_R", "st_F10_irf_next", "st_F01_sn_ext", "st_R21_ln_feo_ret",
               "st_R02_feo_ret", "st_F16_sn_ret_frac", "st_F05_ley_feo_next",
               "b6_sdi_multi_batch", "b6_feo_extraido_multi_neg"]
KPIS_B04 = ["recuperacion_refinada_pct", "sn_perdido_escoria_frac", "rendimiento_proxy_batch", "recuperacion_real_pct"]
NEG_KPIS_B04 = {"sn_perdido_escoria_frac"}  # menor = mejor
CONTROLES = st.CONTROLES_BATCH

corr3_rows = []
for tgt in TARGETS_B04:
    for kpi in KPIS_B04:
        for mname, idx in MUESTRAS.items():
            r = corr_pair(BATCH.loc[idx, tgt], BATCH.loc[idx, kpi], seed=hash((tgt, kpi, mname)) % (2**31))
            row = dict(target=tgt, kpi=kpi, muestra=mname, **r)
            if kpi in NEG_KPIS_B04:
                row["signo_esperado"] = -1
                row["signo_ok"] = (r["rho"] < 0) if pd.notna(r["rho"]) else np.nan
            else:
                row["signo_esperado"] = 1
                row["signo_ok"] = (r["rho"] > 0) if pd.notna(r["rho"]) else np.nan
            corr3_rows.append(row)
corr3 = pd.DataFrame(corr3_rows)

# BH por KPI (dentro de KPI, en DEV, sobre los 10 targets)
corr3["p_bh"] = np.nan
corr3["significativo_bh"] = pd.Series([np.nan] * len(corr3), dtype=object)
for kpi in KPIS_B04:
    m = (corr3["kpi"] == kpi) & (corr3["muestra"] == "dev")
    pvals = corr3.loc[m, "p_rho"].to_numpy()
    valid = ~np.isnan(pvals)
    p_bh = np.full(len(pvals), np.nan)
    sig_bh = np.full(len(pvals), np.nan, dtype=object)
    if valid.sum() > 0:
        rej, pb, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        p_bh[valid] = pb
        sig_bh[valid] = rej
    corr3.loc[m, "p_bh"] = p_bh
    corr3.loc[m, "significativo_bh"] = sig_bh
corr3.to_csv(HERE / "b04_targets_vs_kpi.csv", index=False, encoding="utf-8")
log(f"\n[3.1] b04_targets_vs_kpi.csv escrito ({len(corr3)} filas). Resumen DEV vs recuperacion_refinada_pct:")
resumen3 = corr3[(corr3["kpi"] == "recuperacion_refinada_pct") & (corr3["muestra"] == "dev")].sort_values("rho", ascending=False)
log(resumen3[["target", "n", "rho", "p_rho", "p_bh", "significativo_bh", "ci_low", "ci_high"]].round(4).to_string(index=False))

# comparacion contra el proxy (mismo target, mismo KPI set del proxy) para responder "mas o menos explicable"
log("\n[3.2] Comparacion rho_dev: recuperacion_refinada_pct vs rendimiento_proxy_batch, por target:")
comp_rows = []
for tgt in TARGETS_B04:
    rr = corr3[(corr3["target"] == tgt) & (corr3["muestra"] == "dev")]
    rho_ref = rr[rr["kpi"] == "recuperacion_refinada_pct"]["rho"].iloc[0]
    rho_proxy = rr[rr["kpi"] == "rendimiento_proxy_batch"]["rho"].iloc[0]
    comp_rows.append(dict(target=tgt, rho_dev_refinado=rho_ref, rho_dev_proxy=rho_proxy, delta=rho_ref - rho_proxy))
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(HERE / "b04_targets_comparacion_proxy_refinado.csv", index=False, encoding="utf-8")
log(comp_df.round(4).to_string(index=False))
log(f"  media(delta) = {comp_df['delta'].mean():.4f}  (positivo = el refinado esta MAS correlacionado con los targets)")

# OLS HC3 con controles + tendencia (estandarizado, media/sd de total)
z_vars = TARGETS_B04 + CONTROLES + ["idx_cronologico"]
means_t = {v: BATCH[v].mean() for v in z_vars}
stds_t = {v: BATCH[v].std(ddof=1) for v in z_vars}
BATCH_Z = BATCH.copy()
for v in z_vars:
    sdv = stds_t[v]
    BATCH_Z["z_" + v] = (BATCH[v] - means_t[v]) / sdv if sdv and sdv > 1e-12 else np.nan
ctrl_z = ["z_" + c for c in CONTROLES]
idx_z = "z_idx_cronologico"

ols3_rows = []
for tgt in TARGETS_B04:
    tz = "z_" + tgt
    for kpi in KPIS_B04:
        for variante, extra in [("A_controles", ctrl_z), ("B_controles_idx", ctrl_z + [idx_z]), ("C_solo", [])]:
            for mname, idx in MUESTRAS.items():
                sub = BATCH_Z.loc[idx, [tz] + extra + [kpi]].dropna()
                n = len(sub)
                if n < 15 or sub[tz].std(ddof=1) < 1e-8:
                    ols3_rows.append(dict(target=tgt, kpi=kpi, variante=variante, muestra=mname, n=n,
                                          coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, r2_adj=np.nan))
                    continue
                model, used = fit_ols(sub, [tz] + extra, kpi)
                if model is None or tz not in model.params.index:
                    ols3_rows.append(dict(target=tgt, kpi=kpi, variante=variante, muestra=mname, n=n,
                                          coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, r2_adj=np.nan))
                    continue
                ci = model.conf_int().loc[tz]
                ols3_rows.append(dict(target=tgt, kpi=kpi, variante=variante, muestra=mname, n=n,
                                      coef=model.params[tz], ci_low=ci[0], ci_high=ci[1], p=model.pvalues[tz],
                                      r2_adj=model.rsquared_adj))
ols3_df = pd.DataFrame(ols3_rows)
ols3_df.to_csv(HERE / "b04_targets_ols.csv", index=False, encoding="utf-8")
log(f"\n[3.3] b04_targets_ols.csv escrito ({len(ols3_df)} filas).")

# quintiles en DEV vs recuperacion_refinada_pct
quint_rows = []
for tgt in TARGETS_B04:
    sub = pd.concat([BATCH.loc[DEV_IDX, tgt], BATCH.loc[DEV_IDX, "recuperacion_refinada_pct"]], axis=1).dropna()
    sub.columns = ["x", "y"]
    n = len(sub)
    if n < 25 or sub["x"].std(ddof=1) < 1e-8:
        continue
    try:
        sub["q"] = pd.qcut(sub["x"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except Exception:
        continue
    grp = sub.groupby("q")["y"].agg(["mean", "sem", "count"]).reindex([1, 2, 3, 4, 5])
    monotonia = int(sum(1 for i in range(1, 5) if pd.notna(grp["mean"].iloc[i]) and pd.notna(grp["mean"].iloc[i - 1])
                         and grp["mean"].iloc[i] > grp["mean"].iloc[i - 1]))
    for q in [1, 2, 3, 4, 5]:
        quint_rows.append(dict(target=tgt, quintil=q, n=int(grp.loc[q, "count"]) if pd.notna(grp.loc[q, "count"]) else 0,
                               mean_recuperacion_refinada=grp.loc[q, "mean"], sem=grp.loc[q, "sem"], monotonia=monotonia))
quint_df = pd.DataFrame(quint_rows)
quint_df.to_csv(HERE / "b04_targets_quintiles.csv", index=False, encoding="utf-8")
log(f"[3.4] b04_targets_quintiles.csv escrito ({len(quint_df)} filas).")

# =============================================================================
# 4) TECHO DE PREDICTIBILIDAD DEL KPI REFINADO (patron ST-15)
# =============================================================================
log("\n" + "=" * 78 + "\n4) TECHO DE PREDICTIBILIDAD (HGB / Ridge)\n" + "=" * 78)

STCOLS = st.columnas_targets()  # 39
FEATURES4 = STCOLS + CONTROLES   # factor_escala_trazador EXCLUIDO (usa las salidas del propio batch)
log(f"FEATURES = {len(FEATURES4)} ({len(STCOLS)} st_ + {len(CONTROLES)} controles); "
    f"factor_escala_trazador excluido (usa masa_escoria_trazador y masa_escoria_balance, ambas derivadas del batch).")

X_all = BATCH[FEATURES4]
dev_mask = ~BATCH["es_lockbox"]


def make_pipe(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def model_factory(name):
    if name == "HGB":
        return lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                                       min_samples_leaf=10, random_state=SEED)
    if name == "Ridge":
        return lambda: RidgeCV(alphas=np.logspace(-3, 4, 30))
    raise ValueError(name)


def oof_repeated(X, y, mfac, n_repeats=3, n_splits=5, seed0=0):
    n = len(y)
    filas = []
    for rep in range(n_repeats):
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=seed0 + rep)
        oof = np.full(n, np.nan)
        for tr, te in kf.split(X):
            pipe = make_pipe(mfac())
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = pipe.predict(X.iloc[te])
        r2 = r2_score(y, oof)
        rho, p = stats.spearmanr(y, oof)
        filas.append(dict(rep=rep, r2=r2, rho=rho, p=p))
    return pd.DataFrame(filas)


def fit_predict_lockbox(Xd, yd, Xl, yl, mfac):
    pipe = make_pipe(mfac())
    m = yd.notna()
    pipe.fit(Xd[m], yd[m])
    pred = pipe.predict(Xl)
    ml = yl.notna()
    if ml.sum() < 5:
        return dict(r2=np.nan, rho=np.nan, p=np.nan, n=int(ml.sum()))
    r2 = r2_score(yl[ml], pred[ml])
    rho, p = stats.spearmanr(yl[ml], pred[ml])
    return dict(r2=r2, rho=rho, p=p, n=int(ml.sum()))


techo_rows = []
for kpi in KPIS_B04:
    y_full = BATCH[kpi]
    yd_full, yl_full = y_full[dev_mask], y_full[~dev_mask]
    m_dev = yd_full.notna()
    Xd, yd = X_all[dev_mask].loc[m_dev], yd_full[m_dev]
    Xl, yl = X_all[~dev_mask], yl_full
    for modelo in ["HGB", "Ridge"]:
        mfac = model_factory(modelo)
        rep_df = oof_repeated(Xd, yd, mfac, n_repeats=3, n_splits=5, seed0=0)
        lb = fit_predict_lockbox(Xd, yd, Xl, yl, mfac)
        fila = dict(kpi=kpi, modelo=modelo, n_dev=len(yd),
                    r2_oof_mean=rep_df.r2.mean(), r2_oof_sd=rep_df.r2.std(),
                    rho_oof_mean=rep_df.rho.mean(), rho_oof_sd=rep_df.rho.std(),
                    r2_lockbox=lb["r2"], rho_lockbox=lb["rho"], p_lockbox=lb["p"], n_lockbox=lb["n"])
        techo_rows.append(fila)
        log(f"{kpi:28s} {modelo:6s} n_dev={len(yd):3d}  R2_OOF {fila['r2_oof_mean']:+.3f}(+-{fila['r2_oof_sd']:.3f})"
            f"  rho_OOF {fila['rho_oof_mean']:+.3f}(+-{fila['rho_oof_sd']:.3f})  ||  lockbox R2 {fila['r2_lockbox']:+.3f}"
            f"  rho {fila['rho_lockbox']:+.3f} (n={fila['n_lockbox']})")
techo_df = pd.DataFrame(techo_rows)
techo_df.to_csv(HERE / "b04_techo.csv", index=False, encoding="utf-8")
log(f"\n[4] b04_techo.csv escrito. Referencia ST-15 (proxy, HGB): R2 OOF 0.148 / rho 0.43 DEV; lockbox R2 0.122 / rho 0.48.")

# --- 4b) robustez: sn_perdido_escoria_frac (y recuperacion_refinada_pct) usan ley_sn_escoria_pct/
# ley_feo_escoria_pct del ULTIMO escalon de Reduccion; 4 de los 39 st_ (agg='last') usan las MISMAS
# columnas crudas en ese mismo escalon terminal (R13_sn_residual, R14_log_sn_feo_next, F05_ley_feo_next,
# F11_log_sn_feo_next comparten fila/ensayo con el KPI, aunque tecnicamente son TARGET no LEAKAGE segun
# CLASIFICACION_V6/REGISTRO). Repetimos el techo SIN esas 4 columnas para separar "predictibilidad real"
# de "comparten el mismo ensayo terminal".
log("\n[4b] Robustez: techo SIN columnas terminales que comparten ensayo con el KPI "
    "(R13_sn_residual, R14_log_sn_feo_next, F05_ley_feo_next, F11_log_sn_feo_next):")
LEAKY_LAST = ["st_R13_sn_residual", "st_R14_log_sn_feo_next", "st_F05_ley_feo_next", "st_F11_log_sn_feo_next"]
FEATURES4B = [c for c in FEATURES4 if c not in LEAKY_LAST]
X_all_b = BATCH[FEATURES4B]
techo_rows_b = []
for kpi in ["recuperacion_refinada_pct", "sn_perdido_escoria_frac"]:
    y_full = BATCH[kpi]
    yd_full, yl_full = y_full[dev_mask], y_full[~dev_mask]
    m_dev = yd_full.notna()
    Xd, yd = X_all_b[dev_mask].loc[m_dev], yd_full[m_dev]
    Xl, yl = X_all_b[~dev_mask], yl_full
    for modelo in ["HGB", "Ridge"]:
        mfac = model_factory(modelo)
        rep_df = oof_repeated(Xd, yd, mfac, n_repeats=3, n_splits=5, seed0=0)
        lb = fit_predict_lockbox(Xd, yd, Xl, yl, mfac)
        fila = dict(kpi=kpi, modelo=modelo, n_dev=len(yd), n_features=len(FEATURES4B),
                    r2_oof_mean=rep_df.r2.mean(), r2_oof_sd=rep_df.r2.std(),
                    rho_oof_mean=rep_df.rho.mean(), rho_oof_sd=rep_df.rho.std(),
                    r2_lockbox=lb["r2"], rho_lockbox=lb["rho"], p_lockbox=lb["p"], n_lockbox=lb["n"])
        techo_rows_b.append(fila)
        log(f"{kpi:28s} {modelo:6s} n_dev={len(yd):3d}  R2_OOF {fila['r2_oof_mean']:+.3f}(+-{fila['r2_oof_sd']:.3f})"
            f"  rho_OOF {fila['rho_oof_mean']:+.3f}(+-{fila['rho_oof_sd']:.3f})  ||  lockbox R2 {fila['r2_lockbox']:+.3f}"
            f"  rho {fila['rho_lockbox']:+.3f} (n={fila['n_lockbox']})")
techo_df_b = pd.DataFrame(techo_rows_b)
techo_df_b.to_csv(HERE / "b04_techo_sin_terminales.csv", index=False, encoding="utf-8")
log("[4b] b04_techo_sin_terminales.csv escrito.")

# =============================================================================
# 5) MODELO MULTI-OXIDO DE LA ESCORIA (talon + ganga por tolva)
# =============================================================================
log("\n" + "=" * 78 + "\n5) MODELO MULTI-OXIDO (talon + ganga por tolva)\n" + "=" * 78)

TOLVAS = {
    "t1_conc": "feed_conc_t1_kgh", "t3_reciclo": "feed_reciclo_t3_kgh", "t4_fe": "feed_Fe_kgh",
    "t5_dross": "feed_dross_Fe_kgh", "t7_pellets": "feed_pellets_t7_kgh",
    "t6_cal": "feed_CaO_kgh", "carbon": "feed_Carbon_kgh",
}
OXIDOS = ["cao", "sio2", "al2o3", "mgo"]
LEY_COL = {"cao": "ley_cao_escoria_pct", "sio2": "ley_sio2_escoria_pct",
           "al2o3": "ley_al2o3_escoria_pct", "mgo": "ley_mgo_escoria_pct"}
SD_REL = {"cao": 0.02, "sio2": 0.015, "al2o3": 0.03, "mgo": 0.05}
PARES = [("sio2", "cao"), ("al2o3", "cao"), ("mgo", "cao")]

d = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
for tk, tc in TOLVAS.items():
    d["cum_" + tk] = d[tc].fillna(0).groupby(d["Batch"]).cumsum()

# supuesto fisico: t6_cal solo aporta CaO; carbon (ceniza) solo aporta SiO2/Al2O3 (CENIZA_CARBON ~ 10%)
FREE_PAIRS: list[tuple[str, str]] = []
for tk in TOLVAS:
    for ox in OXIDOS:
        if tk == "t6_cal" and ox != "cao":
            continue
        if tk == "carbon" and ox not in ("sio2", "al2o3"):
            continue
        FREE_PAIRS.append((ox, tk))
N_A = len(FREE_PAIRS)
N_H = len(OXIDOS)
log(f"[5.0] parametros libres: {N_A} coeficientes a_(oxido,tolva) + {N_H} talones H_X = {N_A + N_H} total.")
log(f"      supuesto fisico: t6_cal (cal) solo aporta CaO; carbon (ceniza) solo aporta SiO2/Al2O3.")

LOWER = np.array([0.7 if p == ("cao", "t6_cal") else 0.0 for p in FREE_PAIRS] + [0.0] * N_H)
UPPER = np.array([1.0 if p == ("cao", "t6_cal") else 1.0 for p in FREE_PAIRS] + [10000.0] * N_H)


def unpack(params: np.ndarray):
    a = {p: params[i] for i, p in enumerate(FREE_PAIRS)}
    H = {ox: params[N_A + j] for j, ox in enumerate(OXIDOS)}
    return a, H


def compute_MX(cum: pd.DataFrame, a: dict, H: dict, ox: str) -> pd.Series:
    s = pd.Series(H[ox], index=cum.index, dtype=float)
    for (ox2, tk) in FREE_PAIRS:
        if ox2 == ox:
            s = s + a[(ox2, tk)] * cum["cum_" + tk]
    return s.clip(lower=1e-3)  # piso numerico: evita log(0) en escalones sin cal/ceniza aun


def residuals(params: np.ndarray, cum: pd.DataFrame, ley: pd.DataFrame) -> np.ndarray:
    a, H = unpack(params)
    M = {ox: compute_MX(cum, a, H, ox) for ox in OXIDOS}
    out = []
    for num, den in PARES:
        obs = np.log(ley[LEY_COL[num]].to_numpy(float) / ley[LEY_COL[den]].to_numpy(float))
        pred = np.log((M[num] / M[den]).to_numpy(float))
        out.append(np.clip(obs - pred, -15, 15))
    return np.concatenate(out)


# subconjunto con ensayo simultaneo de los 4 oxidos
oxi_cols = list(LEY_COL.values())
mask_assay = d[oxi_cols].notna().all(axis=1)
cum_cols = ["cum_" + tk for tk in TOLVAS]
D_FIT = d.loc[mask_assay, ["Batch"] + cum_cols + oxi_cols].reset_index(drop=True)
log(f"[5.1] escalones con ensayo simultaneo de CaO/SiO2/Al2O3/MgO: {len(D_FIT)} de {len(d)}.")

x0 = np.concatenate([
    np.array([0.85 if p == ("cao", "t6_cal") else 0.05 for p in FREE_PAIRS]),
    np.array([200.0] * N_H),
])


def fit_full(cum: pd.DataFrame, ley: pd.DataFrame, x_init: np.ndarray):
    res = least_squares(residuals, x_init, args=(cum, ley), bounds=(LOWER, UPPER),
                        loss="soft_l1", f_scale=0.3, max_nfev=4000)
    return res


t5 = time.time()
res_full = fit_full(D_FIT, D_FIT, x0)
log(f"[5.2] ajuste completo (n={len(D_FIT)}): cost={res_full.cost:.3f}, exito={res_full.success}, "
    f"nfev={res_full.nfev}  ({time.time()-t5:.1f}s)")

a_full, H_full = unpack(res_full.x)

# --- multi-start: estabilidad de los parametros
log("\n[5.3] Multi-start (5 semillas) -- estabilidad de parametros:")
starts = []
for i in range(5):
    rng_i = np.random.default_rng(SEED + i)
    xi = LOWER + rng_i.uniform(0.05, 0.5, size=len(LOWER)) * (UPPER - LOWER).clip(max=500)
    xi = np.clip(xi, LOWER, UPPER)
    r = least_squares(residuals, xi, args=(D_FIT, D_FIT), bounds=(LOWER, UPPER), loss="soft_l1", f_scale=0.3, max_nfev=4000)
    starts.append(r.x)
starts_arr = np.array(starts + [res_full.x])
param_names = [f"a_{p[0]}_{p[1]}" for p in FREE_PAIRS] + [f"H_{ox}" for ox in OXIDOS]
stab_df = pd.DataFrame(starts_arr, columns=param_names)
stab_summary = pd.DataFrame({"media": stab_df.mean(), "sd": stab_df.std(), "min": stab_df.min(), "max": stab_df.max(),
                             "lower_bound": LOWER, "upper_bound": UPPER})
stab_summary["en_cota_inf"] = np.isclose(stab_summary["media"], stab_summary["lower_bound"], atol=1e-3)
stab_summary["en_cota_sup"] = np.isclose(stab_summary["media"], stab_summary["upper_bound"], atol=1e-2)
stab_summary.to_csv(HERE / "b04_multioxido_estabilidad.csv", encoding="utf-8")
log(stab_summary.round(4).to_string())

# --- GroupKFold(5) por batch: R2 de los log-cocientes predichos en hold-out vs modelo trivial (media)
log("\n[5.4] GroupKFold(5) por batch -- R2 de log-cocientes en hold-out:")
groups = D_FIT["Batch"].to_numpy()
gkf = GroupKFold(n_splits=5)
cv_rows = []
oof_pred = np.full(len(D_FIT) * len(PARES), np.nan)
oof_obs = np.full(len(D_FIT) * len(PARES), np.nan)
n1 = len(D_FIT)
for fold, (tr, te) in enumerate(gkf.split(D_FIT, groups=groups)):
    cum_tr, ley_tr = D_FIT.iloc[tr], D_FIT.iloc[tr]
    cum_te, ley_te = D_FIT.iloc[te], D_FIT.iloc[te]
    rng_f = np.random.default_rng(SEED + 100 + fold)
    x_init_f = np.clip(x0 * rng_f.uniform(0.8, 1.2, size=len(x0)), LOWER, UPPER)
    r = least_squares(residuals, x_init_f, args=(cum_tr, ley_tr), bounds=(LOWER, UPPER), loss="soft_l1", f_scale=0.3, max_nfev=4000)
    a_f, H_f = unpack(r.x)
    M_te = {ox: compute_MX(cum_te, a_f, H_f, ox) for ox in OXIDOS}
    for k, (num, den) in enumerate(PARES):
        obs = np.log(ley_te[LEY_COL[num]].to_numpy(float) / ley_te[LEY_COL[den]].to_numpy(float))
        pred = np.log((M_te[num] / M_te[den]).to_numpy(float))
        obs_tr = np.log(ley_tr[LEY_COL[num]].to_numpy(float) / ley_tr[LEY_COL[den]].to_numpy(float))
        baseline = np.nanmean(obs_tr)
        ss_res = np.nansum((obs - pred) ** 2)
        ss_tot = np.nansum((obs - baseline) ** 2)
        r2_modelo = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ss_res_triv = np.nansum((obs - baseline) ** 2)
        r2_trivial = 1 - ss_res_triv / ss_tot if ss_tot > 0 else np.nan
        rho_f, p_f = stats.spearmanr(obs, pred) if len(obs) > 5 else (np.nan, np.nan)
        cv_rows.append(dict(fold=fold, par=f"{num}/{den}", n=len(te), r2_modelo=r2_modelo, r2_trivial=r2_trivial,
                            rho=rho_f, p=p_f))
        sl = slice(k * n1, (k + 1) * n1)
        idx_global = te
        oof_pred[k * n1 + idx_global] = pred
        oof_obs[k * n1 + idx_global] = obs
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(HERE / "b04_multioxido_cv.csv", index=False, encoding="utf-8")
log(cv_df.round(4).to_string(index=False))
cv_agg = cv_df.groupby("par")[["r2_modelo", "r2_trivial", "rho"]].mean()
log("\n  Promedio por par (5 folds):")
log(cv_agg.round(4).to_string())

# --- coeficientes finales (ajuste completo) con su interpretacion
coef_rows = []
for (ox, tk) in FREE_PAIRS:
    coef_rows.append(dict(oxido=ox, tolva=tk, columna_tolva=TOLVAS[tk], a_kg_por_kg=a_full[(ox, tk)],
                          en_cota_inf=np.isclose(a_full[(ox, tk)], LOWER[param_names.index(f"a_{ox}_{tk}")], atol=1e-3),
                          en_cota_sup=np.isclose(a_full[(ox, tk)], UPPER[param_names.index(f"a_{ox}_{tk}")], atol=1e-2)))
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(HERE / "b04_multioxido_coeficientes.csv", index=False, encoding="utf-8")
log("\n[5.5] Coeficientes de ganga por tolva (kg oxido / kg tolva), ajuste completo:")
log(coef_df.round(5).to_string(index=False))
H_df = pd.DataFrame([dict(oxido=ox, H_kg=H_full[ox]) for ox in OXIDOS])
H_df.to_csv(HERE / "b04_multioxido_talon.csv", index=False, encoding="utf-8")
log("\n  Talon H_X (kg de oxido, constante por batch):")
log(H_df.round(1).to_string(index=False))

# --- masa de escoria por escalon a partir del modelo multi-oxido (todos los escalones, no solo ensayo)
d["cum_all"] = 1  # placeholder no usado
M_all = {ox: compute_MX(d, a_full, H_full, ox) for ox in OXIDOS}
m_est = {}
for ox in OXIDOS:
    ley = d[LEY_COL[ox]]
    m_est[ox] = np.where(ley.notna() & (ley > 0.05), M_all[ox] / (ley / 100), np.nan)
m_est_df = pd.DataFrame(m_est, index=d.index)
w = pd.Series({ox: 1 / SD_REL[ox] ** 2 for ox in OXIDOS})
W = m_est_df.notna() * w
masa_multi = (m_est_df.fillna(0) * W).sum(axis=1) / W.sum(axis=1).replace(0, np.nan)
d["b6_masa_escoria_multioxido_kg"] = masa_multi

salida_escalon = d[["Batch", "fecha_inicio", "escalon_idx", "fase_proceso", "b6_masa_escoria_multioxido_kg"]].copy()
for ox in OXIDOS:
    salida_escalon[f"m_est_{ox}_kg"] = m_est_df[ox]
salida_escalon.to_csv(HERE / "b04_masa_multioxido_escalon.csv", index=False, encoding="utf-8")
log(f"\n[5.6] b04_masa_multioxido_escalon.csv escrito ({salida_escalon.shape}); "
    f"{salida_escalon['b6_masa_escoria_multioxido_kg'].notna().sum()} escalones con masa multi-oxido estimada.")

# --- comparacion por batch: masa final multi-oxido vs balance global vs trazador v3
gR = d[d["fase_proceso"] == "Reducción"]
ult_mo = gR.dropna(subset=["b6_masa_escoria_multioxido_kg"]).groupby("Batch")["b6_masa_escoria_multioxido_kg"].last()
BATCH["masa_escoria_multioxido_kg"] = ult_mo.reindex(BATCH.index)

cmp_bal = BATCH[["masa_escoria_multioxido_kg", "masa_escoria_balance_kg"]].dropna()
cmp_tr = BATCH[["masa_escoria_multioxido_kg", "masa_escoria_trazador_kg"]].dropna()
factor_mo_bal = (cmp_bal["masa_escoria_multioxido_kg"] / cmp_bal["masa_escoria_balance_kg"])
factor_mo_tr = (cmp_tr["masa_escoria_multioxido_kg"] / cmp_tr["masa_escoria_trazador_kg"]).replace([np.inf, -np.inf], np.nan)
rho_mo_bal, _ = stats.spearmanr(cmp_bal["masa_escoria_multioxido_kg"], cmp_bal["masa_escoria_balance_kg"])
rho_mo_tr, _ = stats.spearmanr(cmp_tr["masa_escoria_multioxido_kg"], cmp_tr["masa_escoria_trazador_kg"])
log(f"\n[5.7] masa final multi-oxido vs balance global: n={len(cmp_bal)}, rho={rho_mo_bal:.4f}, "
    f"factor mediana={factor_mo_bal.median():.3f} (IQR {factor_mo_bal.quantile(.25):.3f}-{factor_mo_bal.quantile(.75):.3f})")
log(f"      masa final multi-oxido vs trazador v3 (CaO solo): n={len(cmp_tr)}, rho={rho_mo_tr.round(4) if pd.notna(rho_mo_tr) else np.nan}, "
    f"factor mediana={factor_mo_tr.median():.3f} (IQR {factor_mo_tr.quantile(.25):.3f}-{factor_mo_tr.quantile(.75):.3f})")

mo_batch_out = BATCH[["masa_escoria_multioxido_kg", "masa_escoria_balance_kg", "masa_escoria_trazador_kg"]].copy()
mo_batch_out["factor_multioxido_vs_balance"] = factor_mo_bal
mo_batch_out["factor_multioxido_vs_trazador"] = factor_mo_tr
mo_batch_out.to_csv(HERE / "b04_multioxido_batch.csv", encoding="utf-8")

# --- talon como fraccion de la escoria final
talon_total_kg = sum(H_full.values())
masa_final_mediana = BATCH["masa_escoria_multioxido_kg"].median()
frac_talon = talon_total_kg / masa_final_mediana if masa_final_mediana else np.nan
log(f"\n[5.8] Talon (suma H_X de los 4 oxidos inertes) = {talon_total_kg:.0f} kg; "
    f"mediana masa final multi-oxido = {masa_final_mediana:.0f} kg -> fraccion = {frac_talon:.3f}. "
    f"Referencia modelo Rust: ~15 t a 600 mm de espesor y 2.32 t/m3 (nota: H_X aqui es solo la masa de los 4 "
    f"oxidos inertes del talon, no el talon fisico completo que tambien lleva FeO/SnO/otros).")

log(f"\nTiempo total: {time.time()-t0:.1f}s")
(HERE / "b04_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("listo.")
