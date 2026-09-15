"""ST-15 (Sonnet) -- techo de predictibilidad de KPIs ALTERNATIVOS y su correlacion con los targets
teoricos ganadores (SDI, lnIRF_R, ln_feo_ret, IRF fin Fusion).

Motivacion (README.md secciones 0 y R2): rendimiento_proxy_batch = 100*M/(M+D+P) tiene fiabilidad baja
(r=0.57 con recuperacion_real; ST-08). Se propuso: (a) recuperacion real = 100*M/Sn_cargado, (b) una
atribucion consistente de dross (D) y polvo (P). Aqui se construyen 11 KPIs alternativos (K0 referencia +
K1..K7, con K5 en 3 variantes de retardo) y se mide:
  1) su techo de predictibilidad (HGB / Ridge, CV honesta, DEV->lockbox) con X = 39 agregados st_* +
     5 controles + 2 heel (igual matriz que ST-10),
  2) su correlacion de Spearman (con IC bootstrap) con los targets ganadores de escalon,
  3) para los KPIs de ventana movil (K3, K3b, K4): la comparacion JUSTA -- el mismo target suavizado con
     la MISMA ventana, para no confundir "el KPI es mas predecible" con "promediar K vecinos reduce ruido".

M = sn_en_metal_crudo_batch_t, D = sn_en_dross_fe_batch_t, P = sn_en_polvo_fundicion_batch_t (toneladas).
Sn_cargado = sum(feed_Sn_kgf)/1000 (t). Sn_escoria_final = ultimo sn_inventario_escoria_est_kg no nulo de
Reduccion / 1000 (t).

Salidas: st_15_kpis.csv, st_15_techo.csv, st_15_correlaciones_targets.csv, st_15_log.txt,
figs/st_15_techo.png, figs/st_15_heatmap_rho.png, figs/st_15_ventana_justa.png
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent.parent))
import st_targets as st  # noqa: E402

LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


t0 = time.time()
FIGS = AQUI / "figs"
FIGS.mkdir(exist_ok=True)

# =============================================================================
# 0) Cargar df de escalones + batch (incluye ganadores st_R21/R22/R23; ~5-15s)
# =============================================================================
df = st.cargar_df()
batch = st.construir_batch(df)
log(f"df {df.shape}, batch {batch.shape}  ({time.time()-t0:.1f}s)")

orden = list(batch.sort_values("idx_cronologico").index)  # orden cronologico de Batch labels
n_batch = len(orden)
log(f"n batches = {n_batch}, DEV = {(~batch.es_lockbox).sum()}, lockbox = {batch.es_lockbox.sum()}")

# =============================================================================
# 1) Construccion de los 11 KPIs alternativos (en orden cronologico; indices = Batch labels)
# =============================================================================
g = df.groupby("Batch")
M = g["sn_en_metal_crudo_batch_t"].first().reindex(orden)
D = g["sn_en_dross_fe_batch_t"].first().reindex(orden)
P = g["sn_en_polvo_fundicion_batch_t"].first().reindex(orden)
Sn_cargado = batch["feed_sn_total_t"].reindex(orden)  # = sum(feed_Sn_kgf)/1000, ya en st.construir_batch

R_fase = df[df.fase_proceso == "Reducción"].dropna(subset=["sn_inventario_escoria_est_kg"]).groupby("Batch").last()
Sn_escoria_final = R_fase["sn_inventario_escoria_est_kg"].reindex(orden) / 1000

log(f"M/D/P/Sn_cargado/Sn_escoria_final no-NaN: {M.notna().sum()}/{D.notna().sum()}/{P.notna().sum()}/"
    f"{Sn_cargado.notna().sum()}/{Sn_escoria_final.notna().sum()} de {n_batch}")

KPI = {}
KPI["K0"] = batch["rendimiento_proxy_batch"].reindex(orden)  # referencia: 100*M/(M+D+P)
KPI["K1"] = batch["recuperacion_real_pct"].reindex(orden)    # 100*M/Sn_cargado (ya calculado igual)
KPI["K2"] = 100 * M / (Sn_cargado - Sn_escoria_final)        # corregida por Sn residual en escoria final

# ventanas moviles (sumas de M, D, P sobre la ventana, luego el cociente)
M3c = M.rolling(3, center=True, min_periods=3).sum()
D3c = D.rolling(3, center=True, min_periods=3).sum()
P3c = P.rolling(3, center=True, min_periods=3).sum()
KPI["K3"] = 100 * M3c / (M3c + D3c + P3c)                    # ventana centrada b-1..b+1

M3f = M.rolling(3, min_periods=3).sum().shift(-2)
D3f = D.rolling(3, min_periods=3).sum().shift(-2)
P3f = P.rolling(3, min_periods=3).sum().shift(-2)
KPI["K3b"] = 100 * M3f / (M3f + D3f + P3f)                   # ventana adelantada b..b+2

M5c = M.rolling(5, center=True, min_periods=5).sum()
D5c = D.rolling(5, center=True, min_periods=5).sum()
P5c = P.rolling(5, center=True, min_periods=5).sum()
KPI["K4"] = 100 * M5c / (M5c + D5c + P5c)                    # ventana centrada b-2..b+2

for k in (1, 2, 3):
    Dk = D.shift(-k)  # D del batch b+k
    Pk = P.shift(-k)  # P del batch b+k
    KPI[f"K5_k{k}"] = 100 * M / (M + Dk + Pk)

KPI["K6"] = 100 * (M + P) / (M + D + P)                      # solo canal dross: 100*(1 - f_dross)
KPI["K7"] = M / (M + D)                                      # metal vs dross (fraccion, sin *100 -- asi se pidio)

KPI_NAMES = list(KPI.keys())
log(f"KPIs construidos: {KPI_NAMES}")
for k, s in KPI.items():
    log(f"  {k:8s} n_no_nan={s.notna().sum():3d}  media={s.mean():.3f}  sd={s.std():.3f}")

kpi_df = pd.DataFrame(KPI)
kpi_out = kpi_df.copy()
kpi_out.insert(0, "idx_cronologico", batch["idx_cronologico"].reindex(orden))
kpi_out.insert(1, "es_lockbox", batch["es_lockbox"].reindex(orden))
kpi_out.insert(2, "M_t", M); kpi_out.insert(3, "D_t", D); kpi_out.insert(4, "P_t", P)
kpi_out.insert(5, "Sn_cargado_t", Sn_cargado); kpi_out.insert(6, "Sn_escoria_final_t", Sn_escoria_final)
kpi_out.index.name = "Batch"
kpi_out.to_csv(AQUI / "st_15_kpis.csv")
log(f"guardado st_15_kpis.csv ({kpi_out.shape})")

# dataframe maestro (index = Batch, orden original de `batch`) con KPIs unidos
full = batch.join(kpi_df)

# =============================================================================
# 2) Techo de predictibilidad: HGB / Ridge, CV KFold5 x3 rep, DEV->lockbox
#    X = 39 agregados st_* + 5 controles + 2 heel (igual matriz que ST-10)
# =============================================================================
log("\n=== 2) Techo de predictibilidad de cada KPI (HGB / Ridge) ===")

F0 = df[(df.fase_proceso == "Fusión") & (df.orden_escalon_fase == 0)].set_index("Batch")["indice_irf"]
batch["heel_irf_F0"] = F0.reindex(batch.index)
R_fin = df[df.fase_proceso == "Reducción"].groupby("Batch")["indice_irf"].last()
R_fin_cron = R_fin.reindex(orden)
heel_prev = R_fin_cron.shift(1)
batch["heel_irf_R_prev"] = heel_prev.reindex(batch.index)
HEEL = ["heel_irf_F0", "heel_irf_R_prev"]

STCOLS = st.columnas_targets()          # 39 (36 ronda1 + R21 + R22 + R23)
CONTROLES = st.CONTROLES_BATCH          # 5
FEATURES = STCOLS + CONTROLES + HEEL    # 46
log(f"FEATURES = {len(FEATURES)} ({len(STCOLS)} st_ + {len(CONTROLES)} controles + {len(HEEL)} heel)")

full = full.drop(columns=[c for c in HEEL if c in full.columns], errors="ignore").join(batch[HEEL])

dev_mask = ~full.es_lockbox
X_all = full[FEATURES]


def make_pipe(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()),
                      ("model", model)])


def model_factory(name):
    if name == "HGB":
        return lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.04,
                                                       min_samples_leaf=10, random_state=42)
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
        rho, p = spearmanr(y, oof)
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
    rho, p = spearmanr(yl[ml], pred[ml])
    return dict(r2=r2, rho=rho, p=p, n=int(ml.sum()))


filas_techo = []
for kpi_name in KPI_NAMES:
    y_full = full[kpi_name]
    yd_full = y_full[dev_mask]
    yl_full = y_full[~dev_mask]
    mfull_dev = yd_full.notna()
    Xd = X_all[dev_mask].loc[mfull_dev]
    yd = yd_full[mfull_dev]
    Xl = X_all[~dev_mask]
    yl = yl_full
    for modelo in ["HGB", "Ridge"]:
        mfac = model_factory(modelo)
        rep_df = oof_repeated(Xd, yd, mfac, n_repeats=3, n_splits=5, seed0=0)
        lb = fit_predict_lockbox(Xd, yd, Xl, yl, mfac)
        fila = dict(kpi=kpi_name, modelo=modelo, n_dev=len(yd),
                    r2_oof_mean=rep_df.r2.mean(), r2_oof_sd=rep_df.r2.std(),
                    rho_oof_mean=rep_df.rho.mean(), rho_oof_sd=rep_df.rho.std(),
                    r2_lockbox=lb["r2"], rho_lockbox=lb["rho"], p_lockbox=lb["p"], n_lockbox=lb["n"])
        filas_techo.append(fila)
        log(f"{kpi_name:8s} {modelo:6s} n_dev={len(yd):3d}  R2_OOF {fila['r2_oof_mean']:+.3f}(+-{fila['r2_oof_sd']:.3f})"
            f"  rho_OOF {fila['rho_oof_mean']:+.3f}(+-{fila['rho_oof_sd']:.3f})  ||  lockbox R2 {fila['r2_lockbox']:+.3f}"
            f"  rho {fila['rho_lockbox']:+.3f} (n={fila['n_lockbox']})")

techo_df = pd.DataFrame(filas_techo)
techo_df.to_csv(AQUI / "st_15_techo.csv", index=False)
log(f"guardado st_15_techo.csv ({techo_df.shape})  [{time.time()-t0:.0f}s]")

# =============================================================================
# 3) Correlaciones targets teoricos x KPIs (Spearman, p, IC bootstrap 1000) en DEV/lockbox/total
# =============================================================================
log("\n=== 3) Correlaciones Spearman targets ganadores x KPIs alternativos ===")

TARGETS_BASE = ["st_R22_sdi", "st_R23_lnirf_R", "st_R21_ln_feo_ret", "st_F10_irf_next"]
# composite z(SDI) + z(lnIRF_R), estandarizado con media/sd de DEV (evita fuga de lockbox a la estandarizacion)
mu_sdi, sd_sdi = full.loc[dev_mask, "st_R22_sdi"].mean(), full.loc[dev_mask, "st_R22_sdi"].std()
mu_irf, sd_irf = full.loc[dev_mask, "st_R23_lnirf_R"].mean(), full.loc[dev_mask, "st_R23_lnirf_R"].std()
full["z_sdi_plus_lnirfR"] = (full["st_R22_sdi"] - mu_sdi) / sd_sdi + (full["st_R23_lnirf_R"] - mu_irf) / sd_irf
TARGETS = TARGETS_BASE + ["z_sdi_plus_lnirfR"]
log(f"targets: {TARGETS} (z_sdi_plus_lnirfR estandarizado con media/sd de DEV)")


def boot_spearman(xv: np.ndarray, yv: np.ndarray, n_boot=1000, seed=0):
    """rho, p puntuales (scipy) + IC 95% via bootstrap vectorizado (argsort doble = rango sin promediar
    empates; aproximacion rapida valida para variables continuas, ver nota en el log)."""
    n = len(xv)
    if n < 10:
        return dict(rho=np.nan, p=np.nan, n=n, ci_lo=np.nan, ci_hi=np.nan)
    rho, p = spearmanr(xv, yv)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    X = xv[idx]
    Y = yv[idx]
    rx = np.argsort(np.argsort(X, axis=1), axis=1).astype(float)
    ry = np.argsort(np.argsort(Y, axis=1), axis=1).astype(float)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    num = (rx * ry).sum(1)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_boot = num / den
    rho_boot = rho_boot[np.isfinite(rho_boot)]
    if len(rho_boot) < 50:
        ci_lo, ci_hi = np.nan, np.nan
    else:
        ci_lo, ci_hi = np.percentile(rho_boot, [2.5, 97.5])
    return dict(rho=rho, p=p, n=n, ci_lo=ci_lo, ci_hi=ci_hi)


def subset_mask(name):
    if name == "DEV":
        return dev_mask
    if name == "lockbox":
        return ~dev_mask
    return pd.Series(True, index=full.index)


filas_corr = []
for target in TARGETS:
    for kpi_name in KPI_NAMES:
        for subset in ["DEV", "lockbox", "total"]:
            m = subset_mask(subset)
            d = full.loc[m, [target, kpi_name]].dropna()
            r = boot_spearman(d[target].values, d[kpi_name].values, n_boot=1000, seed=hash((target, kpi_name, subset)) % (2**31))
            filas_corr.append(dict(target=target, kpi=kpi_name, subset=subset, comparacion="raw", **r))

corr_df = pd.DataFrame(filas_corr)
log(f"correlaciones raw calculadas: {len(corr_df)} filas  [{time.time()-t0:.0f}s]")

# resumen legible en el log: solo DEV/lockbox/total, targets base, para los KPIs mas relevantes
resumen = corr_df.pivot_table(index=["target", "kpi"], columns="subset", values="rho")
for target in TARGETS:
    log(f"\n-- {target} --")
    sub = resumen.loc[target].reindex(KPI_NAMES)
    log(sub.round(3).to_string())

# =============================================================================
# 4) Comparacion JUSTA para los KPIs de ventana movil (K3, K3b, K4): mismo target suavizado
#    con la MISMA ventana (media movil), correlacionado con el KPI (ya suavizado por construccion).
# =============================================================================
log("\n=== 4) Comparacion justa (autocorrelacion inducida por la ventana movil) ===")

VENTANAS = {"K3": "centered3", "K3b": "forward3", "K4": "centered5"}


def smooth_like(s: pd.Series, kind: str) -> pd.Series:
    s_chron = s.reindex(orden)
    if kind == "centered3":
        out = s_chron.rolling(3, center=True, min_periods=3).mean()
    elif kind == "forward3":
        out = s_chron.rolling(3, min_periods=3).mean().shift(-2)
    elif kind == "centered5":
        out = s_chron.rolling(5, center=True, min_periods=5).mean()
    else:
        raise ValueError(kind)
    return out.reindex(full.index)


filas_fair = []
for kpi_name, kind in VENTANAS.items():
    for target in TARGETS:
        target_smooth = smooth_like(full[target], kind)
        for subset in ["DEV", "lockbox", "total"]:
            m = subset_mask(subset)
            d = pd.concat([target_smooth[m], full.loc[m, kpi_name]], axis=1).dropna()
            d.columns = ["target_smooth", "kpi"]
            r = boot_spearman(d["target_smooth"].values, d["kpi"].values, n_boot=1000,
                               seed=hash((kpi_name, target, subset, "fair")) % (2**31))
            filas_fair.append(dict(target=target, kpi=kpi_name, subset=subset, comparacion="smoothed_fair", **r))
            log(f"{kpi_name} ({kind}) x {target:20s} [{subset:8s}] raw rho={resumen.loc[(target, kpi_name), subset] if (target,kpi_name) in resumen.index else np.nan:+.3f}"
                f"  ||  suavizado-justo rho={r['rho']:+.3f} (p={r['p']:.3g}, n={r['n']})")

fair_df = pd.DataFrame(filas_fair)
corr_all = pd.concat([corr_df, fair_df], ignore_index=True)
corr_all.to_csv(AQUI / "st_15_correlaciones_targets.csv", index=False)
log(f"guardado st_15_correlaciones_targets.csv ({corr_all.shape})  [{time.time()-t0:.0f}s]")

# pares que superan |rho| 0.5 o 0.7 (raw), para verificar honestidad
log("\n=== Pares target x KPI con |rho| > 0.5 (raw, cualquier subset) ===")
altos = corr_df[(corr_df.rho.abs() > 0.5)].sort_values("rho", key=lambda s: s.abs(), ascending=False)
if len(altos):
    log(altos[["target", "kpi", "subset", "rho", "p", "n", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
else:
    log("(ninguno)")
log("\n=== Pares target x KPI con |rho| > 0.7 (raw, cualquier subset) ===")
muyaltos = corr_df[(corr_df.rho.abs() > 0.7)].sort_values("rho", key=lambda s: s.abs(), ascending=False)
if len(muyaltos):
    log(muyaltos[["target", "kpi", "subset", "rho", "p", "n", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
else:
    log("(ninguno)")

# =============================================================================
# 5) Figuras
# =============================================================================
log("\n=== 5) Figuras ===")

# 5a: techo (R2 OOF HGB) por KPI, DEV vs lockbox
fig, ax = plt.subplots(figsize=(9, 4.5))
th_hgb = techo_df[techo_df.modelo == "HGB"].set_index("kpi").reindex(KPI_NAMES)
x = np.arange(len(KPI_NAMES))
ax.bar(x - 0.18, th_hgb.r2_oof_mean, width=0.36, yerr=th_hgb.r2_oof_sd, label="R2 OOF DEV (HGB)", color="#2b6cb0")
ax.bar(x + 0.18, th_hgb.r2_lockbox, width=0.36, label="R2 lockbox (HGB, DEV->lockbox)", color="#c53030")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(KPI_NAMES, rotation=20)
ax.set_ylabel("R2"); ax.set_title("Techo de predictibilidad por KPI (HGB, X=39 st_ + controles + heel)")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIGS / "st_15_techo.png", dpi=120)
plt.close(fig)

# 5b: heatmap rho (target x KPI) en DEV
fig, ax = plt.subplots(figsize=(9, 4.5))
mat = resumen["DEV"].unstack("kpi").reindex(index=TARGETS, columns=KPI_NAMES) if "DEV" in resumen.columns else None
if mat is None:
    mat = corr_df[corr_df.subset == "DEV"].pivot(index="target", columns="kpi", values="rho").reindex(index=TARGETS, columns=KPI_NAMES)
im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
ax.set_xticks(range(len(KPI_NAMES))); ax.set_xticklabels(KPI_NAMES, rotation=20)
ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels(TARGETS)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(v) > 0.3 else "black")
ax.set_title("rho Spearman (DEV): targets ganadores x KPIs alternativos")
fig.colorbar(im, ax=ax, shrink=0.8, label="rho")
plt.tight_layout()
fig.savefig(FIGS / "st_15_heatmap_rho.png", dpi=120)
plt.close(fig)

# 5c: comparacion justa raw vs suavizado, por KPI de ventana, target base
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
for ax, kpi_name in zip(axes, VENTANAS):
    raw_vals = [corr_df[(corr_df.target == t) & (corr_df.kpi == kpi_name) & (corr_df.subset == "total")].rho.iloc[0] for t in TARGETS_BASE]
    fair_vals = [fair_df[(fair_df.target == t) & (fair_df.kpi == kpi_name) & (fair_df.subset == "total")].rho.iloc[0] for t in TARGETS_BASE]
    xpos = np.arange(len(TARGETS_BASE))
    ax.bar(xpos - 0.18, raw_vals, width=0.36, label="target crudo vs KPI suavizado", color="#2b6cb0")
    ax.bar(xpos + 0.18, fair_vals, width=0.36, label="target suavizado (misma ventana) vs KPI suavizado", color="#2f855a")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels([t.replace("st_", "") for t in TARGETS_BASE], rotation=30, fontsize=8)
    ax.set_title(kpi_name)
axes[0].set_ylabel("rho Spearman (total)")
axes[0].legend(fontsize=7, loc="upper left")
plt.suptitle("Comparacion justa: autocorrelacion inducida por la ventana movil")
plt.tight_layout()
fig.savefig(FIGS / "st_15_ventana_justa.png", dpi=120)
plt.close(fig)

log(f"\ntiempo total {time.time()-t0:.1f}s")
(AQUI / "st_15_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("listo.")
