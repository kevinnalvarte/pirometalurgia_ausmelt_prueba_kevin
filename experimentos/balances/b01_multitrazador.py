"""B-01 -- cierre multi-trazador de la escoria en Reduccion (`bal_features.agregar_multitrazador`)
vs el trazador CaO unico (v3). Ver `experimentos/balances/ITERACION_6_diseno.md`.

Secciones:
  1) Consistencia entre los 4 inertes (sesgo/sd vs consenso, correlacion entre pares, dispersion
     por orden de escalon).
  2) Ruido: (a) Monte Carlo (200 replicas) perturbando leyes, sd de ruido y techo R2 single vs multi;
     (b) huella empirica -- autocorrelacion lag-1 de FeO_ext dentro del batch, single vs multi.
  3) Targets de batch: b6_sdi_multi, b6_ln_feo_ret_multi, b6_feo_extraido_multi_kg, b6_sn_extraido_multi_kg
     vs st_R22_sdi/st_R21_ln_feo_ret/st_R02_feo_ret/st_R01_sn_ext -- Spearman IC boot500, OLS HC3+tendencia,
     tercios, dosis-respuesta, barrido de w (SDI), correlacion single vs multi (ranking de batches).
  4) Predictibilidad por escalon (HGB OOF DEV+lockbox, sets A/B) y PLM v4 (palancas feo_kg) para
     FeO_ext y SDI, version v3 vs multi -- tabla theta con IC y ancho de IC del carbon.
  5) Residuo de masa: distribucion por orden, Spearman con controles del proceso y con KPIs de batch.

Ejecutar: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/balances/b01_multitrazador.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RAIZ / "experimentos" / "search_targets"))

import bal_features as bal  # noqa: E402
import st_targets as st  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import feature_engineering as fe  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import r2_score, mean_absolute_error  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)
N_BOOT_SPEARMAN = 500
N_REPLICAS_MC = 200
KPIS = st.KPIS
CONTROLES = st.CONTROLES_BATCH
NEG_KPIS = {"f_dross", "f_polvo"}
HGB_KW = dict(max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
              l2_regularization=1.0, random_state=SEED)

_LOG: list[str] = []


def log(msg: object = "") -> None:
    s = str(msg)
    print(s)
    _LOG.append(s)


def flush_log() -> None:
    (HERE / "b01_log.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


# =============================================================================
# utilidades estadisticas (mismo patron que search_targets/st_01)
# =============================================================================

def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT_SPEARMAN) -> tuple[float, float]:
    n = len(x)
    if n < 8:
        return np.nan, np.nan
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    idx = RNG.integers(0, n, size=(n_boot, n))
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


def corr_pair(x: pd.Series, y: pd.Series) -> dict:
    sub = pd.concat([x, y], axis=1).dropna()
    n = len(sub)
    out = dict(n=n, rho=np.nan, p_rho=np.nan, ci_low=np.nan, ci_high=np.nan)
    if n < 8:
        return out
    xv, yv = sub.iloc[:, 0].to_numpy(float), sub.iloc[:, 1].to_numpy(float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return out
    rho, p_rho = stats.spearmanr(xv, yv)
    ci_low, ci_high = bootstrap_ci_spearman(xv, yv)
    out.update(rho=rho, p_rho=p_rho, ci_low=ci_low, ci_high=ci_high)
    return out


def fit_ols(sub: pd.DataFrame, regressors: list[str], y_col: str):
    used = [c for c in regressors if sub[c].std(ddof=1) >= 1e-8]
    if not used:
        return None, used
    X = sm.add_constant(sub[used], has_constant="add")
    try:
        model = sm.OLS(sub[y_col], X).fit(cov_type="HC3")
    except Exception:
        return None, used
    return model, used


def es_segura(nombre: str) -> bool:
    """Anti-fuga (mismo criterio que ST-02): STATE/CONTEXT/ACTION seguras de feature_engineering,
    o columnas `_v4` (Cx_sn_v4/Cx_av_v4: derivadas STATE_prev x ACTION, seguras por construccion)."""
    return nombre.endswith("_v4") or fe.es_feature_segura_para_prescripcion(nombre)


# =============================================================================
# 0) Carga
# =============================================================================

def cargar() -> tuple[pd.DataFrame, pd.DataFrame, set, set]:
    df = bal.cargar_df()
    df = st.construir_df_targets(df)
    df = mp4.agregar_columnas_v4(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    batch = st.construir_batch(df)
    return df, batch, batches_dev, batches_lockbox


# =============================================================================
# 1) Consistencia entre los 4 inertes
# =============================================================================

INERTES = ("cao", "sio2", "al2o3", "mgo")


def ln_ratios_por_inerte(df: pd.DataFrame) -> pd.DataFrame:
    """Replica el calculo interno de `bal.agregar_multitrazador` pero conserva el ln-ratio de
    CADA inerte por separado (bal_features solo expone el de CaO y el consenso)."""
    d = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    g = d.groupby("Batch")
    sin_carga = d["feed_total_kgh"].fillna(0) < bal.S("UMBRAL_SIN_CARGA_KG")
    es_red = d["fase_proceso"] == "Reducción"
    valido = sin_carga & es_red & ~d["es_primer_escalon_fase"].astype(bool)
    out = pd.DataFrame(index=d.index)
    for k in INERTES:
        col = bal.COL_INERTE[k]
        x = d[col].where(d[col] > 0.3)
        xp = g[col].shift(1).where(lambda s: s > 0.3)
        out[k] = (np.log(xp / x)).where(valido)
    out["Batch"] = d["Batch"].to_numpy()
    out["orden_escalon_fase"] = d["orden_escalon_fase"].to_numpy()
    out["consenso"] = d["b6_ln_ratio_masa_multi"].to_numpy()
    out["dispersion"] = d["b6_dispersion_trazadores"].to_numpy()
    out["valido"] = valido.to_numpy()
    return out


def tarea1_consistencia_inertes(df: pd.DataFrame) -> None:
    log("\n" + "=" * 78)
    log("TAREA 1 -- consistencia entre los 4 inertes")
    log("=" * 78)
    L = ln_ratios_por_inerte(df)
    Lv = L[L["valido"]].copy()
    log(f"escalones validos de Reduccion (sin carga, no R0): {len(Lv)} / {L['valido'].notna().sum()}")

    # sesgo/sd de cada inerte respecto al consenso
    rows = []
    for k in INERTES:
        d = (Lv[k] - Lv["consenso"]).dropna()
        w_asumido = 1.0 / bal.SD_REL_ENSAYO[k] ** 2
        sd_emp = float(d.std()) if len(d) > 5 else np.nan
        rows.append(dict(inerte=k, n=len(d), sesgo_medio=float(d.mean()) if len(d) else np.nan,
                          sd_vs_consenso=sd_emp, sd_rel_asumida=bal.SD_REL_ENSAYO[k],
                          peso_asumido=w_asumido, peso_empirico=(1.0 / sd_emp ** 2) if sd_emp and sd_emp > 0 else np.nan))
    sesgo_df = pd.DataFrame(rows)
    tot_asum = sesgo_df["peso_asumido"].sum()
    tot_emp = sesgo_df["peso_empirico"].sum()
    sesgo_df["peso_asumido_norm"] = sesgo_df["peso_asumido"] / tot_asum
    sesgo_df["peso_empirico_norm"] = sesgo_df["peso_empirico"] / tot_emp
    sesgo_df.to_csv(HERE / "b01_inertes_sesgo.csv", index=False, encoding="utf-8")
    log("\nSesgo/sd de cada inerte respecto al consenso multi-trazador (DEV+lockbox, Reduccion valida):")
    log(sesgo_df.round(4).to_string(index=False))

    # correlacion entre pares de inertes (ln-ratio crudo, no vs consenso)
    corr_inertes = Lv[list(INERTES)].corr(method="spearman")
    corr_inertes.to_csv(HERE / "b01_inertes_correlacion.csv", encoding="utf-8")
    log("\nCorrelacion (Spearman) entre pares de inertes (ln-ratio crudo):")
    log(corr_inertes.round(3).to_string())

    # dispersion de trazadores por orden de escalon de Reduccion
    disp_rows = []
    for orden, g in Lv.groupby("orden_escalon_fase"):
        s = g["dispersion"].dropna()
        if len(s) < 3:
            continue
        disp_rows.append(dict(orden_escalon_fase=int(orden), n=len(s), media=float(s.mean()), sd=float(s.std()),
                               p5=float(s.quantile(0.05)), p50=float(s.quantile(0.5)), p95=float(s.quantile(0.95))))
        # sesgo de CADA inerte tambien por orden (para detectar CaO residual en escalones tempranos)
    disp_df = pd.DataFrame(disp_rows)
    disp_df.to_csv(HERE / "b01_dispersion_por_orden.csv", index=False, encoding="utf-8")
    log("\nDispersion ponderada entre trazadores (b6_dispersion_trazadores) por orden de escalon:")
    log(disp_df.round(4).to_string(index=False))

    sesgo_por_orden = []
    for k in INERTES:
        for orden, g in Lv.groupby("orden_escalon_fase"):
            d = (g[k] - g["consenso"]).dropna()
            if len(d) < 3:
                continue
            sesgo_por_orden.append(dict(inerte=k, orden_escalon_fase=int(orden), n=len(d), sesgo_medio=float(d.mean())))
    sesgo_orden_df = pd.DataFrame(sesgo_por_orden)
    sesgo_orden_df.to_csv(HERE / "b01_inertes_sesgo_por_orden.csv", index=False, encoding="utf-8")
    log("\nSesgo medio de cada inerte vs consenso, por orden de escalon (posible cal residual en escalones tempranos):")
    log(sesgo_orden_df.round(4).to_string(index=False))
    flush_log()


# =============================================================================
# 2) Ruido: (a) Monte Carlo, (b) huella empirica lag-1, (c) techo R2
# =============================================================================

def _replica_perturbada(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Perturba %CaO/%SiO2/%Al2O3/%MgO (relativo, sd de bal.SD_REL_ENSAYO) y %Sn (+-0.05 pt),
    %FeO (+-0.3 pt); recalcula la cadena v3 (trazador CaO unico, formula exacta de
    `feature_engineering.agregar_inventario_masa_escoria`: masa[t] = cum_feed_CaO_kg[t]/(%CaO[t]/100))
    y luego llama `bal.agregar_multitrazador` (que usa esa misma masa v3 como ancla) para obtener
    la version multi-trazador bajo la MISMA perturbacion."""
    d = df.drop(columns=[c for c in df.columns if c.startswith("b6_") or c.startswith("st_")]).copy()
    for k, col in bal.COL_INERTE.items():
        sd_rel = bal.SD_REL_ENSAYO[k]
        d[col] = df[col] * (1.0 + rng.normal(0.0, sd_rel, size=len(df)))
    d["ley_sn_escoria_pct"] = df["ley_sn_escoria_pct"] + rng.normal(0.0, 0.05, size=len(df))
    d["ley_feo_escoria_pct"] = df["ley_feo_escoria_pct"] + rng.normal(0.0, 0.3, size=len(df))

    cum_cao_incl = d.groupby("Batch")["feed_CaO_kgh"].cumsum()
    masa = cum_cao_incl / (d["ley_cao_escoria_pct"].where(d["ley_cao_escoria_pct"] > 0.5) / 100.0)
    sn_inv = masa * d["ley_sn_escoria_pct"] / 100.0
    feo_inv = masa * d["ley_feo_escoria_pct"] / 100.0
    masa_prev = masa.groupby(d["Batch"]).shift(1)
    sn_inv_prev = sn_inv.groupby(d["Batch"]).shift(1)
    feo_inv_prev = feo_inv.groupby(d["Batch"]).shift(1)
    d["masa_escoria_est_kg"] = masa
    d["masa_escoria_est_kg_prev"] = masa_prev
    d["sn_inventario_escoria_est_kg"] = sn_inv
    d["sn_inventario_escoria_est_kg_prev"] = sn_inv_prev
    d["feo_inventario_escoria_est_kg"] = feo_inv
    d["feo_inventario_escoria_est_kg_prev"] = feo_inv_prev
    d["sn_extraido_est_kg"] = d["feed_Sn_kgf"].fillna(0.0) - (sn_inv - sn_inv_prev)
    d["feo_extraido_est_kg"] = -(feo_inv - feo_inv_prev)  # v3 (CaO unico), formula del enunciado

    d = bal.agregar_multitrazador(d)
    return d


def tarea2a_monte_carlo(df: pd.DataFrame, batches_dev: set) -> pd.DataFrame:
    log("\n" + "=" * 78)
    log("TAREA 2a -- Monte Carlo del ensayo (200 replicas): sd de ruido single vs multi")
    log("=" * 78)
    mask = (df["fase_proceso"] == "Reducción") & df["Batch"].isin(batches_dev) & df["b6_feo_extraido_multi_kg"].notna()
    idxs = np.where(mask.to_numpy())[0]
    n = len(idxs)
    log(f"filas DEV validas (multi-trazador): {n}")

    cols_track = ["feo_extraido_est_kg", "b6_feo_extraido_multi_kg", "sn_extraido_est_kg", "b6_sn_extraido_multi_kg"]
    acc = {c: np.full((N_REPLICAS_MC, n), np.nan, dtype=np.float32) for c in cols_track}

    t0 = time.time()
    n_hechas = N_REPLICAS_MC
    for k in range(N_REPLICAS_MC):
        try:
            dft = _replica_perturbada(df, RNG)
        except Exception as exc:  # noqa: BLE001
            log(f"MC replica {k}: ERROR -- {exc}")
            n_hechas = k
            break
        for c in cols_track:
            acc[c][k, :] = dft[c].to_numpy(dtype=np.float32)[idxs]
        if (k + 1) % 50 == 0:
            log(f"  MC: {k + 1}/{N_REPLICAS_MC} replicas ({time.time() - t0:.1f}s)")
    log(f"MC terminado: {n_hechas} replicas en {time.time() - t0:.1f}s")

    rows = []
    for c in cols_track:
        vals = acc[c][:n_hechas]
        per_row_sd = np.nanstd(vals, axis=0, ddof=1)
        sd_ruido = float(np.sqrt(np.nanmean(per_row_sd ** 2)))
        sd_target = float(np.nanstd(df.loc[mask, c].to_numpy(dtype=float), ddof=1))
        r2_max = 1.0 - (sd_ruido ** 2) / (sd_target ** 2) if sd_target > 0 else np.nan
        rows.append(dict(columna=c, n=n, n_replicas=n_hechas, sd_ruido=sd_ruido, sd_target=sd_target, r2_max=r2_max))
        log(f"  {c:28s}: sd_ruido={sd_ruido:10.4g}  sd_target={sd_target:10.4g}  R2_max={r2_max:+.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "b01_ruido_montecarlo.csv", index=False, encoding="utf-8")
    flush_log()
    return out


def tarea2b_autocorr_lag1(df: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 78)
    log("TAREA 2b -- huella empirica: autocorrelacion lag-1 de FeO_ext dentro del batch")
    log("=" * 78)
    rows = []
    for label, col in [("single_v3", "feo_extraido_est_kg"), ("multi", "b6_feo_extraido_multi_kg")]:
        d = df[(df["fase_proceso"] == "Reducción") & df[col].notna()][["Batch", "orden_escalon_fase", col]].copy()
        d = d.sort_values(["Batch", "orden_escalon_fase"])
        d["prev"] = d.groupby("Batch")[col].shift(1)
        d["mismo_batch"] = d["Batch"] == d["Batch"].shift(1)
        pares = d[d["mismo_batch"] & d["prev"].notna()]
        if len(pares) > 8:
            r, p = stats.pearsonr(pares["prev"], pares[col])
        else:
            r, p = np.nan, np.nan
        rows.append(dict(version=label, n_pares=len(pares), autocorr_lag1=r, p=p))
        log(f"  {label:10s}: n_pares={len(pares):4d}  autocorr_lag1={r:+.4f}  p={p:.4g}")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "b01_autocorr_lag1.csv", index=False, encoding="utf-8")
    flush_log()
    return out


# =============================================================================
# 3) Targets de batch
# =============================================================================

def _agregado_multi_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: sumas de las columnas b6_* de Reduccion (mismo criterio 'sum telescopica'
    que search_targets: min_count=1 -> NaN si ningun escalon valido)."""
    red = df[df["fase_proceso"] == "Reducción"]
    g = red.groupby("Batch")
    out = pd.DataFrame(index=g.size().index)
    out["b6_sdi_multi"] = g["b6_sdi_multi"].sum(min_count=1)
    out["b6_ln_feo_ret_multi"] = g["b6_ln_feo_ret_multi"].sum(min_count=1)
    out["b6_feo_extraido_multi_kg_raw"] = g["b6_feo_extraido_multi_kg"].sum(min_count=1)
    out["b6_feo_extraido_multi_kg"] = -out["b6_feo_extraido_multi_kg_raw"]  # direccion -1 (como st_R02_feo_ret)
    out["b6_sn_extraido_multi_kg"] = g["b6_sn_extraido_multi_kg"].sum(min_count=1)
    out["n_valido_multi"] = g["b6_ln_ratio_masa_multi"].apply(lambda s: s.notna().sum())
    return out


PARES_TARGETS = [
    ("sdi", "b6_sdi_multi", "st_R22_sdi"),
    ("ln_feo_ret", "b6_ln_feo_ret_multi", "st_R21_ln_feo_ret"),
    ("feo_ext", "b6_feo_extraido_multi_kg", "st_R02_feo_ret"),
    ("sn_ext", "b6_sn_extraido_multi_kg", "st_R01_sn_ext"),
]


def tarea3_targets_batch(df: pd.DataFrame, batch: pd.DataFrame, batches_dev: set, batches_lockbox: set) -> None:
    log("\n" + "=" * 78)
    log("TAREA 3 -- targets de batch: multi-trazador vs single (CaO)")
    log("=" * 78)
    agg = _agregado_multi_batch(df)
    b = batch.join(agg, how="left")
    dev = b[~b["es_lockbox"]]
    lockbox = b[b["es_lockbox"]]
    muestras = {"dev": dev, "lockbox": lockbox, "total": b}
    log(f"batch total={len(b)}, dev={len(dev)}, lockbox={len(lockbox)}")

    # --- 3.1 Spearman IC boot vs los 5 KPIs, dev/lockbox/total ---
    rows = []
    for nombre, col_multi, col_single in PARES_TARGETS:
        for version, col in [("multi", col_multi), ("single", col_single)]:
            for kpi in KPIS:
                for mname, mdf in muestras.items():
                    r = corr_pair(mdf[col], mdf[kpi])
                    rows.append(dict(target=nombre, version=version, columna=col, kpi=kpi, muestra=mname, **r))
    corr_df = pd.DataFrame(rows)
    # BH dentro de (target, version) en DEV vs rendimiento_proxy_batch
    corr_df["p_bh"] = np.nan
    mask_base = (corr_df["kpi"] == "rendimiento_proxy_batch") & (corr_df["muestra"] == "dev")
    pvals = corr_df.loc[mask_base, "p_rho"].to_numpy()
    valid = ~np.isnan(pvals)
    if valid.sum():
        p_bh = np.full(len(pvals), np.nan)
        _, p_bh[valid], _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        corr_df.loc[mask_base, "p_bh"] = p_bh
    corr_df.to_csv(HERE / "b01_targets_correlaciones.csv", index=False, encoding="utf-8")
    log("\nSpearman vs rendimiento_proxy_batch (DEV), multi vs single:")
    show = corr_df[mask_base].sort_values(["target", "version"])
    log(show[["target", "version", "n", "rho", "p_rho", "p_bh", "ci_low", "ci_high"]].round(4).to_string(index=False))

    # --- 3.2 tercios cronologicos de DEV ---
    dev_ = dev.copy()
    dev_["tercio"] = pd.qcut(dev_["idx_cronologico"], 3, labels=[1, 2, 3])
    rows_t = []
    for nombre, col_multi, col_single in PARES_TARGETS:
        for version, col in [("multi", col_multi), ("single", col_single)]:
            for t in [1, 2, 3]:
                sub = dev_[dev_["tercio"] == t]
                r = corr_pair(sub[col], sub["rendimiento_proxy_batch"])
                rows_t.append(dict(target=nombre, version=version, tercio=t, **r))
    tercios_df = pd.DataFrame(rows_t)
    tercios_df.to_csv(HERE / "b01_targets_tercios.csv", index=False, encoding="utf-8")
    log("\nSpearman vs rendimiento_proxy_batch por tercio cronologico de DEV:")
    log(tercios_df.round(4).to_string(index=False))

    # --- 3.3 OLS HC3 con controles + tendencia ---
    z_vars = [c for _, c, _ in PARES_TARGETS] + [c for _, _, c in PARES_TARGETS] + CONTROLES + ["idx_cronologico"]
    z_vars = list(dict.fromkeys(z_vars))
    means_t = {v: b[v].mean() for v in z_vars}
    stds_t = {v: b[v].std(ddof=1) for v in z_vars}
    bz = b.copy()
    for v in z_vars:
        s = stds_t[v]
        bz["z_" + v] = (b[v] - means_t[v]) / s if s and s > 1e-12 else np.nan
    ctrl_z = ["z_" + c for c in CONTROLES]
    idx_z = "z_idx_cronologico"

    rows_ols = []
    for nombre, col_multi, col_single in PARES_TARGETS:
        for version, col in [("multi", col_multi), ("single", col_single)]:
            cz = "z_" + col
            for variante, extra in [("controles", ctrl_z), ("controles_tendencia", ctrl_z + [idx_z]), ("solo", [])]:
                for mname, mdf in muestras.items():
                    sub = bz.loc[mdf.index, [cz] + extra + ["rendimiento_proxy_batch"]].dropna()
                    n = len(sub)
                    if n < 15 or sub[cz].std(ddof=1) < 1e-8:
                        rows_ols.append(dict(target=nombre, version=version, variante=variante, muestra=mname,
                                              n=n, coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, r2_adj=np.nan))
                        continue
                    model, used = fit_ols(sub, [cz] + extra, "rendimiento_proxy_batch")
                    if model is None or cz not in model.params.index:
                        rows_ols.append(dict(target=nombre, version=version, variante=variante, muestra=mname,
                                              n=n, coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, r2_adj=np.nan))
                        continue
                    ci = model.conf_int().loc[cz]
                    rows_ols.append(dict(target=nombre, version=version, variante=variante, muestra=mname, n=n,
                                         coef=model.params[cz], ci_low=ci[0], ci_high=ci[1], p=model.pvalues[cz],
                                         r2_adj=model.rsquared_adj))
    ols_df = pd.DataFrame(rows_ols)
    ols_df.to_csv(HERE / "b01_targets_ols.csv", index=False, encoding="utf-8")
    log("\nOLS HC3 (variante controles_tendencia, DEV):")
    show2 = ols_df[(ols_df["variante"] == "controles_tendencia") & (ols_df["muestra"] == "dev")]
    log(show2[["target", "version", "n", "coef", "ci_low", "ci_high", "p", "r2_adj"]].round(4).to_string(index=False))

    # --- 3.4 dosis-respuesta por quintiles (DEV) ---
    rows_q = []
    for nombre, col_multi, col_single in PARES_TARGETS:
        for version, col in [("multi", col_multi), ("single", col_single)]:
            sub = pd.concat([dev[col], dev["rendimiento_proxy_batch"]], axis=1).dropna()
            sub.columns = ["x", "y"]
            if len(sub) < 25 or sub["x"].std(ddof=1) < 1e-8:
                continue
            sub["q"] = pd.qcut(sub["x"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
            grp = sub.groupby("q")["y"].agg(["mean", "sem", "count"]).reindex([1, 2, 3, 4, 5])
            for q in [1, 2, 3, 4, 5]:
                rows_q.append(dict(target=nombre, version=version, quintil=q,
                                    n=int(grp.loc[q, "count"]) if pd.notna(grp.loc[q, "count"]) else 0,
                                    media_rendimiento=grp.loc[q, "mean"], sem=grp.loc[q, "sem"]))
    dr_df = pd.DataFrame(rows_q)
    dr_df.to_csv(HERE / "b01_targets_dosis_respuesta.csv", index=False, encoding="utf-8")

    # --- 3.5 barrido de w (SDI multi), solo DEV ---
    ws = list(range(2, 17))
    rows_w = []
    ln_sn_dep = df["b6_ln_sn_dep_multi"]
    ln_feo_ret = df["b6_ln_feo_ret_multi"]
    red_mask = df["fase_proceso"] == "Reducción"
    for w in ws:
        sdi_w = (ln_sn_dep + w * ln_feo_ret).where(red_mask)
        batch_w = sdi_w.groupby(df["Batch"]).sum(min_count=1)
        idx_dev = [ix for ix in dev.index if ix in batch_w.index]
        r = corr_pair(batch_w.loc[idx_dev], dev.loc[idx_dev, "rendimiento_proxy_batch"])
        rows_w.append(dict(w=w, n=r["n"], rho_dev=r["rho"], p_dev=r["p_rho"]))
    w_df = pd.DataFrame(rows_w)
    rho_max = w_df["rho_dev"].max()
    w_df["cumple_95pct_max"] = w_df["rho_dev"] >= 0.95 * rho_max
    w_star = int(w_df.loc[w_df["cumple_95pct_max"], "w"].min())
    w_df.to_csv(HERE / "b01_w_sweep_sdi.csv", index=False, encoding="utf-8")
    rho_w10 = float(w_df.loc[w_df["w"] == 10, "rho_dev"].iloc[0])
    log(f"\nBarrido de w (SDI multi, DEV): rho_max={rho_max:.4f} en w={int(w_df.loc[w_df['rho_dev'].idxmax(),'w'])}; "
        f"w* (menor w con rho>=0.95*max) = {w_star} (rho={float(w_df.loc[w_df.w==w_star,'rho_dev'].iloc[0]):.4f}); "
        f"w=10 (v3): rho={rho_w10:.4f}")
    log(w_df.round(4).to_string(index=False))

    # --- 3.6 correlacion single vs multi (cambio de ranking de batches) ---
    rows_sm = []
    for nombre, col_multi, col_single in PARES_TARGETS:
        r = corr_pair(b[col_multi], b[col_single])
        rows_sm.append(dict(target=nombre, **r))
        log(f"  single vs multi -- {nombre:10s}: n={r['n']:4d} rho={r['rho']:+.4f} IC[{r['ci_low']:.3f},{r['ci_high']:.3f}]")
    sm_df = pd.DataFrame(rows_sm)
    sm_df.to_csv(HERE / "b01_single_vs_multi_ranking.csv", index=False, encoding="utf-8")
    flush_log()


# =============================================================================
# 4) Predictibilidad por escalon (HGB) y PLM v4
# =============================================================================

def _feature_sets_reduccion() -> dict[str, list[str]]:
    state_context = list(dict.fromkeys(mp3.STATE_V3["Reducción"] + mp3.CONTEXT_V3["Reducción"]))
    setA = fe.filtrar_features_seguras(state_context)
    control = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min",
               "exceso_o2_combustion_pct", "Cx_sn_v4", "Cx_av_v4"]
    inseguras = [c for c in control if not es_segura(c)]
    if inseguras:
        raise AssertionError(f"control inseguro: {inseguras}")
    setB = list(dict.fromkeys(setA + control))
    return dict(A=setA, B=setB)


def _hgb_oof_lockbox(dev: pd.DataFrame, lockbox: pd.DataFrame, feats: list[str], y_col: str) -> dict:
    n_splits = min(5, dev["Batch"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(dev), np.nan)
    for tr, te in gkf.split(dev[feats], dev[y_col], dev["Batch"]):
        m = HistGradientBoostingRegressor(**HGB_KW)
        m.fit(dev[feats].iloc[tr], dev[y_col].iloc[tr])
        oof[te] = m.predict(dev[feats].iloc[te])
    r2_oof = r2_score(dev[y_col], oof)
    mae_oof = mean_absolute_error(dev[y_col], oof)
    m_full = HistGradientBoostingRegressor(**HGB_KW).fit(dev[feats], dev[y_col])
    if len(lockbox):
        pred_lb = m_full.predict(lockbox[feats])
        r2_lb = r2_score(lockbox[y_col], pred_lb)
        mae_lb = mean_absolute_error(lockbox[y_col], pred_lb)
    else:
        r2_lb, mae_lb = np.nan, np.nan
    return dict(n_dev=len(dev), n_lockbox=len(lockbox), r2_oof=r2_oof, mae_oof=mae_oof, r2_lockbox=r2_lb, mae_lockbox=mae_lb)


PARES_ESCALON = [
    ("feo_ext", "feo_extraido_est_kg", "b6_feo_extraido_multi_kg"),
    ("sdi", "st_R22_sdi", "b6_sdi_multi"),
]


def tarea4_predictibilidad(df: pd.DataFrame, batches_dev: set, batches_lockbox: set) -> None:
    log("\n" + "=" * 78)
    log("TAREA 4 -- predictibilidad por escalon (HGB OOF) y PLM v4 (palancas feo_kg)")
    log("=" * 78)
    feats = _feature_sets_reduccion()
    log(f"Set A ({len(feats['A'])} features seguras STATE+CONTEXT V3 Reduccion)")
    log(f"Set B ({len(feats['B'])} = A + CONTROL)")

    rows = []
    for nombre, col_v3, col_multi in PARES_ESCALON:
        base_mask = (df["fase_proceso"] == "Reducción") & df[col_v3].notna() & df[col_multi].notna()
        base = df[base_mask]
        dev = base[base["Batch"].isin(batches_dev)].reset_index(drop=True)
        lockbox = base[base["Batch"].isin(batches_lockbox)].reset_index(drop=True)
        log(f"\n-- {nombre}: n_dev={len(dev)}, n_lockbox={len(lockbox)} (filas donde AMBAS versiones son validas) --")
        for version, col in [("v3_single", col_v3), ("multi", col_multi)]:
            lo, hi = dev[col].quantile([0.005, 0.995])
            dev = dev.copy()
            lockbox = lockbox.copy()
            dev["y"] = dev[col].clip(lo, hi)
            lockbox["y"] = lockbox[col].clip(lo, hi)
            for setname in ["A", "B"]:
                r = _hgb_oof_lockbox(dev, lockbox, feats[setname], "y")
                rows.append(dict(target=nombre, version=version, columna=col, set=setname, **r))
                log(f"  {version:10s} set{setname}: n_dev={r['n_dev']:4d} r2_oof={r['r2_oof']:+.4f} "
                    f"r2_lockbox={r['r2_lockbox']:+.4f} mae_oof={r['mae_oof']:.3g}")
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(HERE / "b01_predictibilidad_escalon.csv", index=False, encoding="utf-8")
    flush_log()

    # --- PLM v4 (palancas feo_kg) para las 2 versiones de FeO y de SDI ---
    log("\n-- PLM v4 (estado/palancas de feo_kg) --")
    estado = mp4.ESTADO_V4[("Reducción", "feo_kg")]
    palancas = mp4.PALANCAS_V4[("Reducción", "feo_kg")]
    signo = mp4.SIGNO_TEORICO_V4[("Reducción", "feo_kg")]
    log(f"estado ({len(estado)}): {estado}")
    log(f"palancas: {palancas}")

    theta_rows = []
    r2_rows = []
    for nombre, col_v3, col_multi in PARES_ESCALON:
        for version, col in [("v3_single", col_v3), ("multi", col_multi)]:
            need = list(dict.fromkeys(estado + palancas + [col, "Batch"]))
            sub = df[(df["fase_proceso"] == "Reducción")][need].dropna().reset_index(drop=True)
            sub_dev = sub[sub["Batch"].isin(batches_dev)].reset_index(drop=True)
            sub_lb = sub[sub["Batch"].isin(batches_lockbox)].reset_index(drop=True)
            if len(sub_dev) < 60:
                log(f"  {nombre}/{version}: DEV insuficiente (n={len(sub_dev)}), omitido")
                continue
            m = mp4.ModeloPLM(estado, palancas, col).fit(sub_dev, n_splits=5, n_boot=200, seed=SEED)
            t = m.tabla_theta(signo).reset_index()
            t.insert(0, "version", version)
            t.insert(0, "target", nombre)
            theta_rows.append(t)
            r2_lb = r2_score(sub_lb[col], m.predict(sub_lb)) if len(sub_lb) else np.nan
            r2_rows.append(dict(target=nombre, version=version, n_dev=len(sub_dev), n_lockbox=len(sub_lb),
                                r2_oof_interno=m.r2_oof_interno, r2_lockbox=r2_lb))
            log(f"\n  {nombre}/{version} (n_dev={len(sub_dev)}, n_lockbox={len(sub_lb)}): "
                f"R2_oof_interno={m.r2_oof_interno:+.4f}  R2_lockbox={r2_lb:+.4f}")
            log(t.set_index("palanca")[["theta_unidad", "ci_lo", "ci_hi", "significativo", "signo_teorico", "concuerda"]].round(5).to_string())
    theta_df = pd.concat(theta_rows, ignore_index=True) if theta_rows else pd.DataFrame()
    theta_df.to_csv(HERE / "b01_plm_theta.csv", index=False, encoding="utf-8")
    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(HERE / "b01_plm_r2.csv", index=False, encoding="utf-8")

    # ancho de IC del carbon (tasa_feed_Carbon_kg_min, Cx_av_v4) v3 vs multi
    log("\nAncho de IC (ci_hi - ci_lo) de las palancas de carbon, v3 vs multi:")
    for pal in ["tasa_feed_Carbon_kg_min", "Cx_av_v4"]:
        sub = theta_df[theta_df["palanca"] == pal]
        for nombre in ["feo_ext", "sdi"]:
            s = sub[sub["target"] == nombre]
            if not len(s):
                continue
            anchos = {}
            for _, rr in s.iterrows():
                anchos[rr["version"]] = rr["ci_hi"] - rr["ci_lo"]
            log(f"  {nombre:10s} / {pal:24s}: " + ", ".join(f"{v}={a:.5g}" for v, a in anchos.items()))
    flush_log()


# =============================================================================
# 5) Residuo de masa
# =============================================================================

COLS_RESIDUO_ESCALON = ["tiro_horno_pct", "total_process_gas_nm3_min", "tasa_gn_nm3_min",
                        "posicion_vertical_lanza_mm", "temperatura_horno_celsius_prev", "b6_dispersion_trazadores"]


def tarea5_residuo_masa(df: pd.DataFrame, batch: pd.DataFrame, batches_dev: set, batches_lockbox: set) -> None:
    log("\n" + "=" * 78)
    log("TAREA 5 -- residuo de masa (b6_residuo_masa_kg): senal vs ruido")
    log("=" * 78)
    red = df[(df["fase_proceso"] == "Reducción") & df["b6_residuo_masa_kg"].notna()]
    log(f"n escalones con residuo valido: {len(red)}")

    # distribucion por orden de escalon
    rows_o = []
    for orden, g in red.groupby("orden_escalon_fase"):
        s = g["b6_residuo_masa_kg"]
        rows_o.append(dict(orden_escalon_fase=int(orden), n=len(s), media=float(s.mean()), sd=float(s.std()),
                            p5=float(s.quantile(0.05)), p50=float(s.quantile(0.5)), p95=float(s.quantile(0.95))))
    orden_df = pd.DataFrame(rows_o)
    orden_df.to_csv(HERE / "b01_residuo_por_orden.csv", index=False, encoding="utf-8")
    log("\nb6_residuo_masa_kg por orden de escalon de Reduccion:")
    log(orden_df.round(2).to_string(index=False))

    # Spearman con controles del proceso y con la dispersion entre trazadores (dev/lockbox/total)
    dev_r = red[red["Batch"].isin(batches_dev)]
    lockbox_r = red[red["Batch"].isin(batches_lockbox)]
    muestras = {"dev": dev_r, "lockbox": lockbox_r, "total": red}
    rows_c = []
    for col in COLS_RESIDUO_ESCALON:
        for mname, mdf in muestras.items():
            r = corr_pair(mdf["b6_residuo_masa_kg"], mdf[col])
            rows_c.append(dict(variable=col, muestra=mname, **r))
    corr_df = pd.DataFrame(rows_c)
    corr_df.to_csv(HERE / "b01_residuo_correlaciones_escalon.csv", index=False, encoding="utf-8")
    log("\nSpearman de b6_residuo_masa_kg con controles del proceso (DEV):")
    log(corr_df[corr_df["muestra"] == "dev"].round(4).to_string(index=False))

    # suma por batch vs f_polvo, sn_polvo_t, rendimiento_proxy_batch
    res_batch = red.groupby("Batch")["b6_residuo_masa_kg"].sum(min_count=1)
    b = batch.join(res_batch.rename("b6_residuo_masa_kg_batch"), how="left")
    dev = b[~b["es_lockbox"]]
    lockbox = b[b["es_lockbox"]]
    muestras_b = {"dev": dev, "lockbox": lockbox, "total": b}
    rows_b = []
    for kpi in ["f_polvo", "sn_polvo_t", "rendimiento_proxy_batch"]:
        for mname, mdf in muestras_b.items():
            r = corr_pair(mdf["b6_residuo_masa_kg_batch"], mdf[kpi])
            rows_b.append(dict(kpi=kpi, muestra=mname, **r))
    corr_batch_df = pd.DataFrame(rows_b)
    corr_batch_df.to_csv(HERE / "b01_residuo_correlaciones_batch.csv", index=False, encoding="utf-8")
    log("\nSpearman de la suma por batch de b6_residuo_masa_kg con KPIs (DEV):")
    log(corr_batch_df[corr_batch_df["muestra"] == "dev"].round(4).to_string(index=False))
    flush_log()


# =============================================================================
# main
# =============================================================================

def main() -> None:
    t0 = time.time()
    log(f"B-01 multitrazador -- inicio {time.strftime('%Y-%m-%d %H:%M:%S')}")
    df, batch, batches_dev, batches_lockbox = cargar()
    log(f"df {df.shape}, batch {batch.shape}, DEV batches={len(batches_dev)}, lockbox batches={len(batches_lockbox)} "
        f"({time.time()-t0:.1f}s)")
    flush_log()

    try:
        tarea1_consistencia_inertes(df)
    except Exception:
        log("TAREA 1 ERROR:\n" + traceback.format_exc())
        flush_log()

    try:
        df_mc = tarea2a_monte_carlo(df, batches_dev)
    except Exception:
        log("TAREA 2a ERROR:\n" + traceback.format_exc())
        flush_log()

    try:
        tarea2b_autocorr_lag1(df)
    except Exception:
        log("TAREA 2b ERROR:\n" + traceback.format_exc())
        flush_log()

    try:
        tarea3_targets_batch(df, batch, batches_dev, batches_lockbox)
    except Exception:
        log("TAREA 3 ERROR:\n" + traceback.format_exc())
        flush_log()

    try:
        tarea4_predictibilidad(df, batches_dev, batches_lockbox)
    except Exception:
        log("TAREA 4 ERROR:\n" + traceback.format_exc())
        flush_log()

    try:
        tarea5_residuo_masa(df, batch, batches_dev, batches_lockbox)
    except Exception:
        log("TAREA 5 ERROR:\n" + traceback.format_exc())
        flush_log()

    log(f"\nTiempo total: {time.time()-t0:.1f}s")
    flush_log()


if __name__ == "__main__":
    main()
