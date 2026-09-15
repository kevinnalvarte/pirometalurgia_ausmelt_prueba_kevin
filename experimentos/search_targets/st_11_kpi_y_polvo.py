"""ST-11 -- KPI alternativos y canal polvo/dross (search_targets, ronda 2, 2026-09-14).

Pregunta (SEARCH_TARGETS_ronda2.md): (1) estructura del KPI rendimiento_proxy_batch = 100*M/(M+D+P) --
cuanto explica cada canal, correlaciones entre M/D/P y con Sn cargado, autocorrelacion de D y P entre
batches y con pellets recirculados (tolva 7), evidencia de atribucion por stock/periodo; (2) que KPI
alternativo (propio, f_dross, f_polvo, recuperacion real, media de 2 batches, media movil de 3) es mas
predecible con los agregados st_* + controles + heel (IRF fin de Reduccion del batch anterior); (3)/(4)
candidatos de polvo (>=10) y de dross (>=6) por escalon, con fundamento teorico (Guia_Ausmelt_Sn.md /
consideracioness_pirometalurgia_ausmelt.md), evaluados contra f_polvo / f_dross y rendimiento.

Datos: cache_df_st.pkl (escalones, ya trae st_R01..R20/F01..F16 del registro v1; aqui se RECALCULAN
via st_targets.construir_df_targets para incluir tambien los ganadores R21 (FeO retenido log) y R22
(SDI) que no estaban cacheados) y st_batch.csv (una fila por batch; se reconstruye una version propia
equivalente con st.construir_batch para tener SDI/IRF-heel disponibles, y se valida contra el CSV
provisto).

Entorno: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/search_targets/st_11_kpi_y_polvo.py
Sin joblib.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import st_targets as st  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)
N_BOOT = 1000
KPIS = ["rendimiento_proxy_batch", "f_metal", "f_dross", "f_polvo", "recuperacion_real_pct"]
CONTROLES = list(st.CONTROLES_BATCH)  # feed_sn_total_t, ley_sn_conc_batch_pct, espesor_ladrillo_norm_mm,
                                       # frac_carga_secundaria_F, feed_dross_Fe_total_t

LOG: list[str] = []


def log(msg: object = "") -> None:
    s = str(msg)
    print(s)
    LOG.append(s)


# =============================================================================
# utilidades estadisticas (mismo patron que st_01)
# =============================================================================

def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
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


def corr_pair(x: pd.Series, y: pd.Series, boot: bool = True) -> dict:
    sub = pd.concat([x, y], axis=1).dropna()
    out = dict(n=len(sub), rho=np.nan, p_rho=np.nan, r=np.nan, p_r=np.nan, ci_low=np.nan, ci_high=np.nan)
    if len(sub) < 8:
        return out
    xv, yv = sub.iloc[:, 0].to_numpy(float), sub.iloc[:, 1].to_numpy(float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return out
    rho, p_rho = stats.spearmanr(xv, yv)
    r, p_r = stats.pearsonr(xv, yv)
    out.update(n=len(sub), rho=rho, p_rho=p_rho, r=r, p_r=p_r)
    if boot:
        cl, ch = bootstrap_ci_spearman(xv, yv)
        out.update(ci_low=cl, ci_high=ch)
    return out


def ols_std_hc3(sub: pd.DataFrame, x_col: str, y_col: str, controles: list[str]) -> dict:
    """OLS HC3 de y ~ const + z(x) + controles (sin estandarizar controles). Devuelve theta por +1sd de x."""
    d = sub[[x_col, y_col] + controles].dropna().copy()
    out = dict(n=len(d), theta=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    if len(d) < 15 or d[x_col].std(ddof=1) < 1e-10:
        return out
    z = (d[x_col] - d[x_col].mean()) / d[x_col].std(ddof=1)
    used_c = [c for c in controles if d[c].std(ddof=1) >= 1e-8]
    X = pd.concat([z.rename("z_x")] + [d[c] for c in used_c], axis=1)
    X = sm.add_constant(X, has_constant="add")
    try:
        model = sm.OLS(d[y_col], X).fit(cov_type="HC3")
    except Exception:
        return out
    out.update(n=len(d), theta=model.params["z_x"], p=model.pvalues["z_x"],
                ci_low=model.conf_int().loc["z_x", 0], ci_high=model.conf_int().loc["z_x", 1])
    return out


def bh(pvals: pd.Series) -> pd.Series:
    p = pvals.to_numpy(float)
    m = np.isfinite(p)
    out = np.full(len(p), np.nan)
    if m.sum():
        out[m] = multipletests(p[m], method="fdr_bh")[1]
    return pd.Series(out, index=pvals.index)


# =============================================================================
# 0) Carga y reconstruccion (df escalon + batch), con SDI/IRF-heel
# =============================================================================

def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    df_raw = pd.read_pickle(HERE / "cache_df_st.pkl")
    df_raw = df_raw.drop(columns=[c for c in df_raw.columns if c.startswith("st_")])  # se recalculan (incluye ganadores R21/R22)
    df = st.construir_df_targets(df_raw)          # agrega st_R01..R22 / st_F01..F16 (incluye ganadores)
    batch = st.construir_batch(df)                 # una fila por batch: st_*, KPIs, controles, es_lockbox, idx_cronologico
    log(f"df escalon {df.shape}, batch reconstruido {batch.shape}  ({time.time()-t0:.1f}s)")

    # validacion contra el st_batch.csv provisto (mismas columnas salvo R21/R22, que no estaban cacheadas)
    ref = pd.read_csv(HERE / "st_batch.csv").set_index("Batch")
    comunes = [c for c in ref.columns if c in batch.columns and pd.api.types.is_float_dtype(ref[c])
               and pd.api.types.is_numeric_dtype(batch[c]) and not pd.api.types.is_bool_dtype(batch[c])]
    max_diffs = {}
    for c in comunes:
        d = (ref[c] - batch.loc[ref.index, c]).abs()
        max_diffs[c] = float(d.max()) if d.notna().any() else np.nan
    peor = pd.Series(max_diffs).sort_values(ascending=False)
    log(f"Validacion vs st_batch.csv provisto: {len(comunes)} columnas comunes; "
        f"max |diff| peor caso = {peor.iloc[0]:.6g} ({peor.index[0]})")
    log(peor.head(5).to_string())

    return df, batch


# =============================================================================
# 1) Estructura del KPI
# =============================================================================

def parte1_estructura(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 90)
    log("PARTE 1 -- Estructura del KPI")
    log("=" * 90)
    filas = []

    # -- 1.1 identidad y regresion rendimiento ~ f_dross + f_polvo
    sub = batch[["rendimiento_proxy_batch", "f_dross", "f_polvo", "f_metal"]].dropna()
    identidad = 100 * (1 - sub["f_dross"] - sub["f_polvo"])
    r_ident, _ = stats.pearsonr(sub["rendimiento_proxy_batch"], identidad)
    log(f"\n1.1 Identidad rendimiento_proxy_batch == 100*(1-f_dross-f_polvo): "
        f"r={r_ident:.6f}, max|diff|={np.abs(sub['rendimiento_proxy_batch']-identidad).max():.4f} pp "
        f"(n={len(sub)}) -> {'identidad exacta (redondeo)' if r_ident > 0.9999 else 'NO es identidad exacta, revisar'}")

    X = sm.add_constant(sub[["f_dross", "f_polvo"]])
    m = sm.OLS(sub["rendimiento_proxy_batch"], X).fit(cov_type="HC3")
    log(f"OLS rendimiento ~ const + f_dross + f_polvo: R2={m.rsquared:.4f}  "
        f"coef f_dross={m.params['f_dross']:.2f} (p={m.pvalues['f_dross']:.2e}), "
        f"coef f_polvo={m.params['f_polvo']:.2f} (p={m.pvalues['f_polvo']:.2e})")

    var_d, var_p = sub["f_dross"].var(ddof=1), sub["f_polvo"].var(ddof=1)
    cov_dp = sub[["f_dross", "f_polvo"]].cov().iloc[0, 1]
    var_total_teorico = 10000 * (var_d + var_p + 2 * cov_dp)
    var_total_obs = sub["rendimiento_proxy_batch"].var(ddof=1)
    share_d = 10000 * (var_d + cov_dp) / var_total_teorico
    share_p = 10000 * (var_p + cov_dp) / var_total_teorico
    log(f"Descomposicion de varianza (Var(rendimiento) = 10000*[Var(fd)+Var(fp)+2Cov(fd,fp)]): "
        f"Var(rend) observada={var_total_obs:.2f}, teorica={var_total_teorico:.2f}")
    log(f"  sd(f_dross)={np.sqrt(var_d)*100:.3f} pp, sd(f_polvo)={np.sqrt(var_p)*100:.3f} pp, "
        f"corr(f_dross,f_polvo)={cov_dp/np.sqrt(var_d*var_p):.3f}")
    log(f"  Share de la varianza del KPI atribuible a f_dross = {share_d*100:.1f}%, a f_polvo = {share_p*100:.1f}% "
        f"(suman 100% por construccion; polvo domina si sd_polvo > sd_dross)")
    filas.append(dict(seccion="1.1_identidad", item="r_identidad", valor=r_ident))
    filas.append(dict(seccion="1.1_regresion", item="R2_fd_fp", valor=m.rsquared))
    filas.append(dict(seccion="1.1_regresion", item="coef_f_dross", valor=m.params["f_dross"]))
    filas.append(dict(seccion="1.1_regresion", item="coef_f_polvo", valor=m.params["f_polvo"]))
    filas.append(dict(seccion="1.1_varianza", item="sd_f_dross_pp", valor=np.sqrt(var_d) * 100))
    filas.append(dict(seccion="1.1_varianza", item="sd_f_polvo_pp", valor=np.sqrt(var_p) * 100))
    filas.append(dict(seccion="1.1_varianza", item="corr_fd_fp", valor=cov_dp / np.sqrt(var_d * var_p)))
    filas.append(dict(seccion="1.1_varianza", item="share_var_f_dross_pct", valor=share_d * 100))
    filas.append(dict(seccion="1.1_varianza", item="share_var_f_polvo_pct", valor=share_p * 100))

    # -- 1.2 correlaciones M, D, P (t) entre si y con Sn cargado
    log("\n1.2 Correlaciones M/D/P (t) y con Sn cargado (feed_sn_total_t):")
    pares = [("sn_metal_t", "sn_dross_t"), ("sn_metal_t", "sn_polvo_t"), ("sn_dross_t", "sn_polvo_t"),
             ("sn_metal_t", "feed_sn_total_t"), ("sn_dross_t", "feed_sn_total_t"), ("sn_polvo_t", "feed_sn_total_t")]
    for a, b in pares:
        r = corr_pair(batch[a], batch[b])
        log(f"  {a:<18} vs {b:<18}  rho={r['rho']:.3f} [IC {r['ci_low']:.2f},{r['ci_high']:.2f}] "
            f"p={r['p_rho']:.2e}  r_pearson={r['r']:.3f}  n={r['n']}")
        filas.append(dict(seccion="1.2_correlaciones_MDP", item=f"{a}_vs_{b}", valor=r["rho"],
                            p=r["p_rho"], n=r["n"], ci_low=r["ci_low"], ci_high=r["ci_high"]))

    # -- 1.3 autocorrelacion D, P (lags 1-5) + cruce con pellets del batch siguiente/anterior
    log("\n1.3 Autocorrelacion de D y P entre batches consecutivos (lags 1-5, orden idx_cronologico):")
    b = batch.sort_values("idx_cronologico").reset_index()
    for col in ["sn_dross_t", "sn_polvo_t", "rendimiento_proxy_batch"]:
        for lag in range(1, 6):
            x, y = b[col], b[col].shift(lag)
            r = corr_pair(x, y, boot=False)
            log(f"  {col:<24} lag={lag}  rho={r['rho']:.3f}  p={r['p_rho']:.3f}  n={r['n']}")
            filas.append(dict(seccion="1.3_autocorr", item=f"{col}_lag{lag}", valor=r["rho"], p=r["p_rho"], n=r["n"]))

    # pellets (tolva 7) cargados por batch: masa total y Sn contenido, cruzados con P (sn_polvo_t) en lags -5..5
    pellets = df.groupby("Batch").agg(pellets_t7_kg=("feed_pellets_t7_kgh", "sum"),
                                        sn_pellets_t7_kg=("sn_pellets_t7_kgf", "sum"))
    b = b.merge(pellets, left_on="Batch", right_index=True, how="left")
    log("\n  Cruce P (Sn en polvo, t) del batch b vs pellets cargados (tolva 7, Sn kg) en el batch b+k:")
    for k in range(-5, 6):
        x = b["sn_polvo_t"]
        y = b["sn_pellets_t7_kg"].shift(-k)   # shift(-k): valor de pellets EN el batch b+k alineado con la fila b
        r = corr_pair(x, y, boot=False)
        log(f"    k={k:+d}  rho(P_b, pellets_Sn_{{b+k}})={r['rho']:.3f}  p={r['p_rho']:.3f}  n={r['n']}")
        filas.append(dict(seccion="1.3_polvo_vs_pellets", item=f"k{k:+d}", valor=r["rho"], p=r["p_rho"], n=r["n"]))

    # -- 1.4 P vs masa de carga, duracion total, n escalones
    log("\n1.4 P (sn_polvo_t) vs carga total, duracion total y n de escalones del batch:")
    carga = df.groupby("Batch").agg(carga_total_kg=("feed_total_kgh", "sum"),
                                      duracion_total_min=("delta_tiempo", "sum"),
                                      n_escalones=("escalon_idx", "count"))
    bb = batch.join(carga)
    for col in ["carga_total_kg", "duracion_total_min", "n_escalones"]:
        r = corr_pair(bb["sn_polvo_t"], bb[col])
        log(f"  sn_polvo_t vs {col:<20} rho={r['rho']:.3f} [IC {r['ci_low']:.2f},{r['ci_high']:.2f}] p={r['p_rho']:.2e} n={r['n']}")
        filas.append(dict(seccion="1.4_polvo_vs_escala", item=col, valor=r["rho"], p=r["p_rho"], n=r["n"],
                            ci_low=r["ci_low"], ci_high=r["ci_high"]))
        r2 = corr_pair(bb["sn_dross_t"], bb[col])
        log(f"  sn_dross_t vs {col:<20} rho={r2['rho']:.3f} p={r2['p_rho']:.2e} n={r2['n']}")
        filas.append(dict(seccion="1.4_dross_vs_escala", item=col, valor=r2["rho"], p=r2["p_rho"], n=r2["n"]))

    # -- 1.5 evidencia de atribucion por stock/periodo: valores unicos, rachas, correlacion con media movil
    log("\n1.5 Evidencia de atribucion por stock/periodo (valores unicos, rachas de valores iguales, corr con media movil):")
    bo = batch.sort_values("idx_cronologico")
    for col in ["sn_dross_t", "sn_polvo_t", "sn_metal_t"]:
        s = bo[col].reset_index(drop=True)
        n_unique = s.nunique()
        # rachas de valores consecutivos identicos
        eq = (s.diff() == 0)
        rachas = []
        run = 1
        for v in eq:
            if v:
                run += 1
            else:
                if run > 1:
                    rachas.append(run)
                run = 1
        if run > 1:
            rachas.append(run)
        max_racha = max(rachas) if rachas else 1
        n_rachas = len(rachas)
        log(f"  {col:<14} n={len(s)}  valores_unicos={n_unique} ({100*n_unique/len(s):.1f}%)  "
            f"rachas(valor repetido consecutivo)={n_rachas}  racha_max={max_racha}")
        filas.append(dict(seccion="1.5_stock_unicos", item=col, valor=n_unique, n=len(s)))
        filas.append(dict(seccion="1.5_stock_rachas", item=col, valor=max_racha, n=n_rachas))
        for win in (3, 5, 7):
            ma = s.rolling(win, center=True, min_periods=win).mean()
            r = corr_pair(s, ma, boot=False)
            log(f"      corr({col}, media_movil_centrada_{win}) rho={r['rho']:.3f} p={r['p_rho']:.2e} n={r['n']}")
            filas.append(dict(seccion="1.5_stock_corr_ma", item=f"{col}_ma{win}", valor=r["rho"], p=r["p_rho"], n=r["n"]))

    out = pd.DataFrame(filas)
    out.to_csv(HERE / "st_11_estructura_kpi.csv", index=False)

    # figuras
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col, lbl in zip(axes, ["sn_dross_t", "sn_polvo_t", "rendimiento_proxy_batch"],
                              ["D (dross, t Sn)", "P (polvo, t Sn)", "rendimiento_proxy_batch"]):
        bo2 = batch.sort_values("idx_cronologico")
        lags = list(range(1, 6))
        rhos = [corr_pair(bo2[col], bo2[col].shift(l), boot=False)["rho"] for l in lags]
        ax.bar(lags, rhos, color="#4C72B0")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("lag (batches)")
        ax.set_ylabel("rho autocorr")
    fig.tight_layout()
    fig.savefig(FIGS / "st_11_autocorrelacion_DP.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = list(range(-5, 6))
    rhos_k = []
    for k in ks:
        x = b["sn_polvo_t"]
        y = b["sn_pellets_t7_kg"].shift(-k)
        rhos_k.append(corr_pair(x, y, boot=False)["rho"])
    ax.bar(ks, rhos_k, color="#DD8452")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("k (batch b+k)")
    ax.set_ylabel("rho(P_b, pellets_Sn_{b+k})")
    ax.set_title("Polvo (P) del batch b vs pellets (T7) cargados en b+k")
    fig.tight_layout()
    fig.savefig(FIGS / "st_11_polvo_vs_pellets_lags.png", dpi=130)
    plt.close(fig)

    log(f"\nGuardado st_11_estructura_kpi.csv ({len(out)} filas)")
    return out


# =============================================================================
# 2) KPIs alternativos y su predictibilidad
# =============================================================================

def heel_irf_prev(df: pd.DataFrame, batch: pd.DataFrame) -> pd.Series:
    """IRF al cierre de Reduccion del batch anterior (orden cronologico)."""
    red = df[df["fase_proceso"] == "Reducción"].dropna(subset=["irf"])
    ult = red.sort_values(["Batch", "fecha_inicio"]).groupby("Batch")["irf"].last()
    bo = batch[["idx_cronologico"]].join(ult.rename("irf_fin_reduccion")).sort_values("idx_cronologico")
    heel = bo["irf_fin_reduccion"].shift(1)
    heel.name = "heel_irf_prev"
    return heel


def oof_metrics(X: np.ndarray, y: np.ndarray, model_factory, n_repeats: int = 5, n_folds: int = 5, seed0: int = SEED) -> dict:
    r2s, rhos = [], []
    for r in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed0 + r)
        preds = np.full(len(y), np.nan)
        for tr, te in kf.split(X):
            m = model_factory()
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        r2s.append(r2_score(y, preds))
        rho, _ = stats.spearmanr(y, preds)
        rhos.append(rho)
    return dict(r2_oof_mean=float(np.mean(r2s)), r2_oof_sd=float(np.std(r2s)),
                rho_oof_mean=float(np.mean(rhos)), rho_oof_sd=float(np.std(rhos)))


def parte2_predictibilidad(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 90)
    log("PARTE 2 -- KPIs alternativos y su predictibilidad")
    log("=" * 90)

    heel = heel_irf_prev(df, batch)
    b = batch.join(heel).sort_values("idx_cronologico").copy()

    b["rend_mean_b_bnext"] = (b["rendimiento_proxy_batch"] + b["rendimiento_proxy_batch"].shift(-1)) / 2
    b["rend_ma3_centrada"] = b["rendimiento_proxy_batch"].rolling(3, center=True, min_periods=3).mean()

    st_cols = [c for c in b.columns if c.startswith("st_")]
    feat_cols = st_cols + CONTROLES + ["heel_irf_prev"]
    log(f"n features = {len(feat_cols)} ({len(st_cols)} st_* + {len(CONTROLES)} controles + 1 heel)")

    targets = ["rendimiento_proxy_batch", "f_dross", "f_polvo", "recuperacion_real_pct",
               "rend_mean_b_bnext", "rend_ma3_centrada"]

    dev_mask = ~b["es_lockbox"]
    filas = []
    for tgt in targets:
        d = b[feat_cols + [tgt, "es_lockbox"]].dropna(subset=[tgt])
        dev = d[~d["es_lockbox"]]
        lockbox = d[d["es_lockbox"]]
        X_dev, y_dev = dev[feat_cols].to_numpy(float), dev[tgt].to_numpy(float)
        X_lb, y_lb = lockbox[feat_cols].to_numpy(float), lockbox[tgt].to_numpy(float)
        n_dev, n_lb = len(dev), len(lockbox)
        if n_dev < 30:
            log(f"  {tgt}: n_dev={n_dev} insuficiente, se omite")
            continue

        # HGB (nativo con NaN)
        hgb_factory = lambda: HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=12, l2_regularization=1.0, random_state=42)
        m_hgb = oof_metrics(X_dev, y_dev, hgb_factory)
        hgb_full = hgb_factory().fit(X_dev, y_dev)
        r2_lb_hgb = r2_score(y_lb, hgb_full.predict(X_lb)) if n_lb >= 15 else np.nan
        rho_lb_hgb = stats.spearmanr(y_lb, hgb_full.predict(X_lb))[0] if n_lb >= 15 else np.nan

        # Ridge con CV interna para alpha (imputacion mediana + escalado)
        ridge_factory = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                            ("sc", StandardScaler()),
                                            ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 25)))])
        m_ridge = oof_metrics(X_dev, y_dev, ridge_factory)
        ridge_full = ridge_factory().fit(X_dev, y_dev)
        r2_lb_ridge = r2_score(y_lb, ridge_full.predict(X_lb)) if n_lb >= 15 else np.nan
        rho_lb_ridge = stats.spearmanr(y_lb, ridge_full.predict(X_lb))[0] if n_lb >= 15 else np.nan

        for modelo, met, r2lb, rholb in [("HGB", m_hgb, r2_lb_hgb, rho_lb_hgb),
                                            ("Ridge", m_ridge, r2_lb_ridge, rho_lb_ridge)]:
            log(f"  {tgt:<24} {modelo:<6} n_dev={n_dev:<4} sd={y_dev.std():.3f}  "
                f"R2_oof_dev={met['r2_oof_mean']:.3f}+-{met['r2_oof_sd']:.3f}  "
                f"rho_oof_dev={met['rho_oof_mean']:.3f}+-{met['rho_oof_sd']:.3f}  "
                f"R2_lockbox={r2lb:.3f}  rho_lockbox={rholb:.3f}  n_lb={n_lb}")
            filas.append(dict(target=tgt, modelo=modelo, n_dev=n_dev, n_lockbox=n_lb, sd_dev=float(y_dev.std()),
                                R2_oof_dev=met["r2_oof_mean"], R2_oof_dev_sd=met["r2_oof_sd"],
                                rho_oof_dev=met["rho_oof_mean"], rho_oof_dev_sd=met["rho_oof_sd"],
                                R2_lockbox=r2lb, rho_lockbox=rholb))

    out = pd.DataFrame(filas)
    out.to_csv(HERE / "st_11_kpis_predictibilidad.csv", index=False)

    # cual KPI es mas predecible (por R2_oof_dev promedio entre HGB/Ridge)
    resumen = out.groupby("target")["R2_oof_dev"].mean().sort_values(ascending=False)
    log("\nRanking de predictibilidad (R2_oof_dev promedio HGB+Ridge):")
    log(resumen.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    piv = out.pivot(index="target", columns="modelo", values="R2_oof_dev").loc[resumen.index]
    piv.plot(kind="barh", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xlabel("R2 OOF (DEV, KFold 5x5)")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_title("Predictibilidad de KPIs alternativos")
    fig.tight_layout()
    fig.savefig(FIGS / "st_11_predictibilidad_kpis.png", dpi=130)
    plt.close(fig)

    log(f"\nGuardado st_11_kpis_predictibilidad.csv ({len(out)} filas)")
    return out


# =============================================================================
# 3) y 4) Candidatos de polvo y de dross por escalon
# =============================================================================

def agg_twmean(df: pd.DataFrame, col: str) -> pd.Series:
    def f(g):
        m = g[col].notna() & g["delta_tiempo"].gt(0)
        return np.average(g.loc[m, col], weights=g.loc[m, "delta_tiempo"]) if m.any() else np.nan
    return df.groupby("Batch").apply(f)


def agg_sum_exposicion(df: pd.DataFrame, col: str) -> pd.Series:
    d = df.copy()
    d["_exp"] = d[col] * d["delta_tiempo"]
    return d.groupby("Batch")["_exp"].sum(min_count=1)


def agg_sum(df: pd.DataFrame, col: str, phase: str | None = None) -> pd.Series:
    d = df if phase is None else df[df["fase_proceso"] == phase]
    return d.groupby("Batch")[col].sum(min_count=1)


def agg_last(df: pd.DataFrame, col: str, phase: str | None = None) -> pd.Series:
    d = df if phase is None else df[df["fase_proceso"] == phase]
    d = d.dropna(subset=[col]).sort_values(["Batch", "fecha_inicio"])
    return d.groupby("Batch")[col].last()


def agg_ratio_sum(df: pd.DataFrame, num: str, den: str) -> pd.Series:
    g = df.groupby("Batch").agg(n=(num, "sum"), d=(den, "sum"))
    return (g["n"] / g["d"]).where(g["d"] > 0)


def preparar_columnas_polvo(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["T_K"] = d["temperatura_horno_celsius"] + 273.15
    d["gas_total_escalon_nm3"] = (d["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0)
                                    + d["volumen_o2_inyectado_lanza_escalon_nm3"].fillna(0)
                                    + d["volumen_aire_inyectado_lanza_escalon_nm3"].fillna(0))
    d["gas_por_t_carga"] = d["gas_total_escalon_nm3"] / (d["feed_total_kgh"] / 1000).where(d["feed_total_kgh"] > 100)
    d["intensidad_fume_raw"] = d["total_process_gas_nm3_min"] * np.exp(-30000.0 / d["T_K"])
    d["carga_fina_frac"] = d["frac_conc_carga"].fillna(0) + d["frac_pellets_carga"].fillna(0)
    d["delta_T_gas_pre_raw"] = d["delta_T_gas_pre"]
    return d


def candidatos_polvo() -> list[dict]:
    return [
        dict(id="P01_T_gas_bhf", label="-T gas pre-BHF", col=lambda d: -d["temperatura_gas_pre_bhf_celsius"],
              teoria="Guia (consideracioness) 22.3: T pre-BHF proxy de carga termica / fume aguas abajo. Minimizar."),
        dict(id="P02_T_gas_cuchilla", label="-T gas pre-cuchilla", col=lambda d: -d["temperatura_gas_pre_cuchilla_celsius"],
              teoria="Guia 22.3: segundo punto del circuito de gases, mismo mecanismo."),
        dict(id="P03_delta_T_gas", label="-|dT gas pre|", col=lambda d: -d["delta_T_gas_pre_raw"].abs(),
              teoria="Guia 34: delta_T_gas = T_pre_cuchilla - T_pre_BHF; exploratorio (sin signo universal, 22.3)."),
        dict(id="P04_tiro", label="-tiro_horno_pct", col=lambda d: -d["tiro_horno_pct"],
              teoria="Guia 21.2: demasiado tiro sube infiltracion de aire falso, volumen de gas y carryover de particulas."),
        dict(id="P05_gas_total_min", label="-gas total lanza (Nm3/min)", col=lambda d: -d["total_process_gas_nm3_min"],
              teoria="Guia 15.1: momentum del jet ~ flujo*velocidad; mas flujo -> mas turbulencia/splashing (22.2 arrastre mecanico)."),
        dict(id="P06_gas_por_carga", label="-gas Nm3 / t carga", col=lambda d: -d["gas_por_t_carga"],
              teoria="Intensidad de gas relativa al tamano de carga del escalon (arrastre mecanico normalizado)."),
        dict(id="P07_presion_punta", label="presion punta lanza (kPa)", col=lambda d: d["presion_punta_lanza_kpa"],
              teoria="Guia 17.3: sin maximo/signo universal (inmersion vs restriccion); exploratorio, se deja signo crudo."),
        dict(id="P08_posicion_lanza", label="posicion vertical lanza (mm)", col=lambda d: d["posicion_vertical_lanza_mm"],
              teoria="Guia 16: posicion es proxy de inmersion, sin signo universal; exploratorio."),
        dict(id="P09_T_bano", label="-T horno (bano)", col=lambda d: -d["temperatura_horno_celsius"],
              teoria="Guia 4.3 / 9.3: mayor T de bano -> mayor volatilizacion de SnO -> fume. Minimizar."),
        dict(id="P10_intensidad_fume", label="-gas*exp(-30000/T_K)", col=lambda d: -d["intensidad_fume_raw"],
              teoria="Intensidad de fume tipo Arrhenius (gas portador * presion de vapor relativa de SnO); Guia 4.3, 22.2."),
        dict(id="P11_carga_fina", label="-frac(conc+pellets) en carga", col=lambda d: -d["carga_fina_frac"],
              teoria="Guia 22.2: arrastre mecanico aumenta con feed fino (concentrado t1 + pellets recirculados t7)."),
        dict(id="P12_tasa_carga", label="-tasa carga total (kg/min)", col=lambda d: -d["tasa_feed_total_kg_min"],
              teoria="Guia fenomenologica 9.2 / 15.1: tasa de carga alta -> mas arrastre/splash mecanico."),
    ]


def preparar_columnas_dross(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["feo_ext_pos"] = d["feo_ext"].clip(lower=0)
    d["fe_mineral_kg"] = d["feed_Fe_kgh"].fillna(0)
    d["fe_dross_kg"] = d["feed_dross_Fe_kgh"].fillna(0)
    d["sn_dross_alimentado_kg"] = d["sn_Fe_t4_kgf"].fillna(0) + d["sn_dross_Fe_t5_kgf"].fillna(0)
    d["_sdi_ln_sn_dep"] = st._ln_sn_dep(d)
    d["_sdi_ln_feo_ret"] = st._ln_feo_ret(d)
    d["sdi_raw"] = d["_sdi_ln_sn_dep"] + st.W_SDI * d["_sdi_ln_feo_ret"]
    return d


def candidatos_dross() -> list[dict]:
    """agg: 'sum' (masa, todo el batch), 'last' (estado terminal, fin de Reduccion), 'sdi' (suma telescopica, solo Reduccion)."""
    return [
        dict(id="D01_feo_reducido_total", agg="sum", col="feo_ext_pos", phase=None, signo=-1,
              teoria="Guia 5.3/11.3: FeO reducido a Fe0 forma hardhead/dross que ocluye Sn. Minimizar (total del batch)."),
        dict(id="D02_fe_dross_t5", agg="sum", col="fe_dross_kg", phase=None, signo=-1,
              teoria="Fe cargado como dross (tolva 5): mas Fe disponible para volver a formar dross. Minimizar."),
        dict(id="D03_fe_mineral_t4", agg="sum", col="fe_mineral_kg", phase=None, signo=-1,
              teoria="Fe cargado como mineral WF (tolva 4): mas Fe disponible para reducirse y formar dross. Minimizar."),
        dict(id="D04_sn_dross_alimentado", agg="sum", col="sn_dross_alimentado_kg", phase=None, signo=-1,
              teoria="Sn que entra ya en corrientes ferriferas (t4+t5): hipotesis de reciclo hacia dross de salida. Minimizar."),
        dict(id="D05_feo_final", agg="last", col="ley_feo_escoria_pct", phase="Reducción", signo=+1,
              teoria="Guia 7.4: %FeO final alto = matriz fayalitica conservada, menos Fe metalizado a dross. Maximizar."),
        dict(id="D06_irf_final", agg="last", col="irf", phase="Reducción", signo=+1,
              teoria="IRF = FeO/(SiO2+Al2O3+CaO) al cierre de Reduccion (fin de batch): matriz FeO-rica retiene Sn. Maximizar."),
        dict(id="D07_sdi", agg="sdi", col="sdi_raw", phase="Reducción", signo=+1,
              teoria="SDI ganador de Reduccion (README v5): agotamiento de Sn - 10*retencion log de FeO. Maximizar."),
    ]


def evaluar_candidato(batch: pd.DataFrame, serie: pd.Series, kpi_neg: str, kpi_pos: str, controles: list[str]) -> dict:
    """serie ya orientada mayor=mejor. Evalua contra kpi_neg (se espera rho<0) y kpi_pos (se espera rho>0),
    en DEV y lockbox, + OLS HC3 con controles en DEV."""
    b = batch.join(serie.rename("cand"))
    dev, lb = b[~b["es_lockbox"]], b[b["es_lockbox"]]
    r_neg_dev = corr_pair(dev["cand"], dev[kpi_neg])
    r_pos_dev = corr_pair(dev["cand"], dev[kpi_pos])
    r_neg_lb = corr_pair(lb["cand"], lb[kpi_neg], boot=False)
    r_pos_lb = corr_pair(lb["cand"], lb[kpi_pos], boot=False)
    ols_neg = ols_std_hc3(dev, "cand", kpi_neg, controles)
    ols_pos = ols_std_hc3(dev, "cand", kpi_pos, controles)
    return dict(n_dev=r_neg_dev["n"], n_lockbox=r_neg_lb["n"],
                rho_dev_negKPI=r_neg_dev["rho"], p_dev_negKPI=r_neg_dev["p_rho"],
                ci_low_negKPI=r_neg_dev["ci_low"], ci_high_negKPI=r_neg_dev["ci_high"],
                rho_lockbox_negKPI=r_neg_lb["rho"], p_lockbox_negKPI=r_neg_lb["p_rho"],
                rho_dev_posKPI=r_pos_dev["rho"], p_dev_posKPI=r_pos_dev["p_rho"],
                ci_low_posKPI=r_pos_dev["ci_low"], ci_high_posKPI=r_pos_dev["ci_high"],
                rho_lockbox_posKPI=r_pos_lb["rho"], p_lockbox_posKPI=r_pos_lb["p_rho"],
                theta_ols_negKPI=ols_neg["theta"], p_ols_negKPI=ols_neg["p"],
                theta_ols_posKPI=ols_pos["theta"], p_ols_posKPI=ols_pos["p"])


def parte3_polvo(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 90)
    log("PARTE 3 -- Canal polvo por escalon (>=10 candidatos)")
    log("=" * 90)
    d = preparar_columnas_polvo(df)
    cands = candidatos_polvo()
    filas = []
    for c in cands:
        raw = c["col"](d)
        for agg_name, agg_fn in [("twmean", agg_twmean), ("sum_exposicion", agg_sum_exposicion)]:
            tmp = d.copy()
            tmp["_c"] = raw
            serie = agg_fn(tmp, "_c")
            res = evaluar_candidato(batch, serie, "f_polvo", "rendimiento_proxy_batch", CONTROLES)
            fila = dict(id=c["id"], label=c["label"], agg=agg_name, teoria=c["teoria"], **res)
            filas.append(fila)
            log(f"  {c['id']:<24} [{agg_name:<14}] rho_dev(f_polvo)={res['rho_dev_negKPI']:+.3f} (p={res['p_dev_negKPI']:.3f})  "
                f"rho_dev(rend)={res['rho_dev_posKPI']:+.3f} (p={res['p_dev_posKPI']:.3f})  "
                f"lockbox(f_polvo)={res['rho_lockbox_negKPI']:+.3f}  lockbox(rend)={res['rho_lockbox_posKPI']:+.3f}  "
                f"theta_ols(rend)={res['theta_ols_posKPI']:+.3f}pp/sd (p={res['p_ols_posKPI']:.3f})")
        # variante coherente ratio-de-sumas para el candidato de razon (P06)
        if c["id"] == "P06_gas_por_carga":
            tmp = d.copy()
            serie = -agg_ratio_sum(tmp, "gas_total_escalon_nm3", "feed_total_kgh")
            res = evaluar_candidato(batch, serie, "f_polvo", "rendimiento_proxy_batch", CONTROLES)
            filas.append(dict(id=c["id"], label=c["label"] + " (ratio de sumas)", agg="ratio_sum",
                                teoria=c["teoria"] + " Variante coherente Guia 10.8 (cociente de sumas).", **res))
            log(f"  {c['id']:<24} [ratio_sum     ] rho_dev(f_polvo)={res['rho_dev_negKPI']:+.3f} (p={res['p_dev_negKPI']:.3f})  "
                f"rho_dev(rend)={res['rho_dev_posKPI']:+.3f} (p={res['p_dev_posKPI']:.3f})")

    out = pd.DataFrame(filas)
    out["p_dev_negKPI_bh"] = bh(out["p_dev_negKPI"])
    out["p_dev_posKPI_bh"] = bh(out["p_dev_posKPI"])
    out = out.sort_values("rho_dev_posKPI", ascending=False)
    out.to_csv(HERE / "st_11_polvo.csv", index=False)

    log("\nTop 5 candidatos de polvo por rho_dev(rendimiento):")
    log(out[["id", "agg", "rho_dev_posKPI", "p_dev_posKPI_bh", "rho_dev_negKPI", "p_dev_negKPI_bh",
             "rho_lockbox_posKPI", "n_dev"]].head(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 7))
    top = out.drop_duplicates("id").nlargest(12, "rho_dev_posKPI") if False else out.copy()
    best_by_id = out.loc[out.groupby("id")["rho_dev_posKPI"].idxmax()].sort_values("rho_dev_posKPI")
    colors = ["#55A868" if p < 0.05 else "#C44E52" for p in best_by_id["p_dev_posKPI_bh"]]
    ax.barh(best_by_id["id"] + " (" + best_by_id["agg"] + ")", best_by_id["rho_dev_posKPI"], color=colors)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("rho Spearman DEV vs rendimiento_proxy_batch (mejor agregacion por candidato)")
    ax.set_title("Candidatos de polvo (verde: p_BH<0.05)")
    fig.tight_layout()
    fig.savefig(FIGS / "st_11_polvo_ranking.png", dpi=130)
    plt.close(fig)

    log(f"\nGuardado st_11_polvo.csv ({len(out)} filas)")
    return out


def parte4_dross(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 90)
    log("PARTE 4 -- Canal dross por escalon (>=6 candidatos)")
    log("=" * 90)
    d = preparar_columnas_dross(df)
    cands = candidatos_dross()
    filas = []
    for c in cands:
        if c["agg"] == "sum":
            serie = c["signo"] * agg_sum(d, c["col"], phase=c["phase"])
        elif c["agg"] == "last":
            serie = c["signo"] * agg_last(d, c["col"], phase=c["phase"])
        elif c["agg"] == "sdi":
            # suma telescopica sobre Reduccion (igual que el registro st_targets: R22_sdi)
            dr = d[d["fase_proceso"] == "Reducción"]
            serie = c["signo"] * dr.groupby("Batch")[c["col"]].sum(min_count=1)
        else:
            raise ValueError(c["agg"])
        res = evaluar_candidato(batch, serie, "f_dross", "rendimiento_proxy_batch", CONTROLES)
        filas.append(dict(id=c["id"], agg=c["agg"], teoria=c["teoria"], **res))
        log(f"  {c['id']:<28} [{c['agg']:<5}] rho_dev(f_dross)={res['rho_dev_negKPI']:+.3f} (p={res['p_dev_negKPI']:.3f})  "
            f"rho_dev(rend)={res['rho_dev_posKPI']:+.3f} (p={res['p_dev_posKPI']:.3f})  "
            f"lockbox(f_dross)={res['rho_lockbox_negKPI']:+.3f}  lockbox(rend)={res['rho_lockbox_posKPI']:+.3f}  "
            f"theta_ols(rend)={res['theta_ols_posKPI']:+.3f}pp/sd (p={res['p_ols_posKPI']:.3f})  n_dev={res['n_dev']}")

    out = pd.DataFrame(filas)
    out["p_dev_negKPI_bh"] = bh(out["p_dev_negKPI"])
    out["p_dev_posKPI_bh"] = bh(out["p_dev_posKPI"])
    out = out.sort_values("rho_dev_posKPI", ascending=False)
    out.to_csv(HERE / "st_11_dross.csv", index=False)

    log("\nRanking de candidatos de dross por rho_dev(rendimiento):")
    log(out[["id", "agg", "rho_dev_posKPI", "p_dev_posKPI_bh", "rho_dev_negKPI", "p_dev_negKPI_bh",
             "rho_lockbox_posKPI", "n_dev"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#55A868" if p < 0.05 else "#C44E52" for p in out["p_dev_posKPI_bh"]]
    ax.barh(out["id"], out["rho_dev_posKPI"], color=colors)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("rho Spearman DEV vs rendimiento_proxy_batch")
    ax.set_title("Candidatos de dross (verde: p_BH<0.05)")
    fig.tight_layout()
    fig.savefig(FIGS / "st_11_dross_ranking.png", dpi=130)
    plt.close(fig)

    log(f"\nGuardado st_11_dross.csv ({len(out)} filas)")
    return out


# =============================================================================
# main
# =============================================================================

def main() -> None:
    t0 = time.time()
    df, batch = cargar()
    parte1_estructura(df, batch)
    parte2_predictibilidad(df, batch)
    parte3_polvo(df, batch)
    parte4_dross(df, batch)
    log(f"\nTiempo total: {time.time()-t0:.1f}s")
    with open(HERE / "st_11_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
