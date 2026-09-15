"""ST-10 (Sonnet) -- compuesto supervisado con validacion cruzada honesta.

Pregunta: cual es el R2/rho maximo alcanzable contra rendimiento_proxy_batch con una combinacion de
agregados TEORICOS por batch (sin fuga, CV honesta), y que "compuesto" parsimonioso (<=6 terminos) lo
consigue?

Matriz X: 38 columnas st_<id> (36 candidatos ronda1 + R21_ln_feo_ret + R22_sdi=SDI ganador Reduccion;
F10_irf_next=ganador Fusion ya esta en las 38) + 2 heel (IRF cierre F0 propio batch, IRF final Reduccion
del batch anterior en orden cronologico) + 5 controles de contexto + familia2 (st_09_batch.csv) si existe.

Y: rendimiento_proxy_batch, f_dross, f_polvo, recuperacion_real_pct, kpi_dos_batches (media del propio y
el siguiente batch en orden cronologico).

Modelos (CV anidada sobre DEV, imputacion mediana + escalado DENTRO de cada fold):
  Ridge (RidgeCV interno), Lasso (LassoCV interno), ElasticNet (ElasticNetCV interno),
  HGB (max_depth=3, max_iter=200, lr=0.05, min_samples_leaf=10, sin tuning),
  OLS_ganadores (solo st_R22_sdi + st_F10_irf_next + heel_irf_F0 + heel_irf_R_prev).
CV: KFold(5) repetido 5x (semillas 0..4, no agrupado: cada fila ya es un batch) + una variante de
walk-forward por bloques cronologicos (5 bloques, forward-chaining) para rendimiento_proxy_batch.
DEV->lockbox: entrena con todo DEV, predice lockbox.
"Techo" = R2 OOF de HGB con TODAS las columnas (== la fila HGB de rendimiento_proxy_batch).

Compuesto parsimonioso: candidatos restringidos a agg en {sum, product, last} (aditivo/telescopico por
escalon, o estado terminal) + heel (estados terminales) -- excluye ratio/twmean/kapp/survival/controles,
que no se pueden escribir como target de escalon simple. Lasso path + estabilidad (100 bootstraps de
LassoCV) + forward stepwise por CV (<=6 terminos, KFold5 simple) -> compuesto elegido = mejor R2 OOF con
<=6 terminos; coeficientes estandarizados con IC bootstrap (100) via OLS.

Salidas: st_10_modelos.csv, st_10_compuesto.csv, st_10_curva_terminos.csv, st_10_pred_oof.csv,
st_10_log.txt, figs/st_10_*.png
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression, lasso_path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent.parent))
import st_targets as st  # noqa: E402

RNG = np.random.RandomState(0)
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


t0 = time.time()

# =============================================================================
# 1) Construccion de la matriz de batch (recalculada, incluye R21/R22)
# =============================================================================
df = st.cargar_df()
batch = st.construir_batch(df)
log(f"df {df.shape}, batch {batch.shape}  ({time.time()-t0:.1f}s)")

# --- heel: IRF al cierre de F0 (propio batch) e IRF final de Reduccion del batch anterior (cronologico)
F0 = df[(df.fase_proceso == "Fusión") & (df.orden_escalon_fase == 0)].set_index("Batch")["indice_irf"]
batch["heel_irf_F0"] = F0.reindex(batch.index)
R_fin = df[df.fase_proceso == "Reducción"].groupby("Batch")["indice_irf"].last()
orden_cron = batch.sort_values("idx_cronologico").index
R_fin_cron = R_fin.reindex(orden_cron)
heel_prev = R_fin_cron.shift(1)
batch["heel_irf_R_prev"] = heel_prev.reindex(batch.index)
HEEL = ["heel_irf_F0", "heel_irf_R_prev"]
log(f"heel: irf_F0 no-NaN={batch.heel_irf_F0.notna().sum()}/362, irf_R_prev no-NaN={batch.heel_irf_R_prev.notna().sum()}/362 "
    f"(NaN esperado en el primer batch cronologico: no hay Reduccion anterior)")

# --- familia 2 (st_09_batch.csv), si existe
F2_PATH = AQUI / "st_09_batch.csv"
tiene_f2 = F2_PATH.exists()
f2_cols: list[str] = []
if tiene_f2:
    f2 = pd.read_csv(F2_PATH, index_col=0)
    f2_cols = [c for c in f2.columns if c.startswith("st_") and c not in batch.columns]
    batch = batch.join(f2[f2_cols], how="left")
    log(f"familia2 (st_09_batch.csv) encontrada: {len(f2_cols)} columnas anadidas: {f2_cols}")
else:
    log("familia2 (st_09_batch.csv) NO existe todavia -> se continua SIN ella (ST-09 no habia terminado al arrancar ST-10).")

# --- KPI de dos batches: media del propio y el siguiente (orden cronologico)
chron = batch.sort_values("idx_cronologico")
nxt = chron["rendimiento_proxy_batch"].shift(-1)
kpi2 = (chron["rendimiento_proxy_batch"] + nxt) / 2
batch["kpi_dos_batches"] = kpi2.reindex(batch.index)
log(f"kpi_dos_batches: {batch.kpi_dos_batches.notna().sum()}/362 no-NaN (NaN esperado en el ultimo batch cronologico)")

STCOLS = st.columnas_targets()  # 38: 36 ronda1 + R21 + R22 (ganadores ya incluidos)
CONTROLES = st.CONTROLES_BATCH  # 5
FEATURES = STCOLS + HEEL + CONTROLES + f2_cols
log(f"FEATURES totales: {len(FEATURES)} (= {len(STCOLS)} st_ + {len(HEEL)} heel + {len(CONTROLES)} controles + {len(f2_cols)} familia2)")

YS = ["rendimiento_proxy_batch", "f_dross", "f_polvo", "recuperacion_real_pct", "kpi_dos_batches"]

dev_mask = ~batch.es_lockbox
DEV = batch[dev_mask].copy()
LB = batch[batch.es_lockbox].copy()
log(f"DEV n={len(DEV)}, lockbox n={len(LB)}")

reg_tabla = st.tabla_registro()
AGG_OK = {"sum", "product", "last"}
allowed_ids = reg_tabla[reg_tabla["agg"].isin(AGG_OK)].index.tolist()
ALLOWED_STCOLS = ["st_" + i for i in allowed_ids]
ALLOWED = ALLOWED_STCOLS + HEEL  # + f2 si declarase agg compatible (no disponible aqui)
log(f"candidatos elegibles para el compuesto (agg in sum/product/last, o heel terminal): {len(ALLOWED)}")
log(str(sorted(ALLOWED)))

FIGS = AQUI / "figs"
FIGS.mkdir(exist_ok=True)


# =============================================================================
# 2) Infraestructura CV: imputacion + escalado DENTRO de cada fold
# =============================================================================
def make_pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def model_factory(name):
    if name == "Ridge":
        return lambda: RidgeCV(alphas=np.logspace(-3, 3, 25))
    if name == "Lasso":
        return lambda: LassoCV(cv=5, alphas=50, max_iter=20000, random_state=0)
    if name == "ElasticNet":
        return lambda: ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1], cv=5, alphas=30,
                                     max_iter=20000, random_state=0)
    if name == "HGB":
        return lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                                       min_samples_leaf=10, random_state=0)
    if name == "OLS_ganadores":
        return lambda: LinearRegression()
    raise ValueError(name)


def oof_repeated(X: pd.DataFrame, y: pd.Series, mfac, n_repeats=5, n_splits=5, seed0=0):
    """KFold(n_splits) repetido n_repeats veces (semillas distintas). Devuelve metricas por repeticion."""
    n = len(y)
    filas = []
    oof_rep0 = np.full(n, np.nan)
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed0 + rep)
        oof = np.full(n, np.nan)
        for tr, te in kf.split(X):
            pipe = make_pipe(mfac())
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = pipe.predict(X.iloc[te])
        r2 = r2_score(y, oof)
        rho, p = spearmanr(y, oof)
        mae = mean_absolute_error(y, oof)
        filas.append(dict(rep=rep, r2=r2, rho=rho, p=p, mae=mae))
        if rep == 0:
            oof_rep0 = oof.copy()
    d = pd.DataFrame(filas)
    return d, oof_rep0


def walk_forward(X: pd.DataFrame, y: pd.Series, idx_cron: pd.Series, mfac, n_blocks=5):
    """Forward-chaining cronologico: entrena en bloques 1..k, testea en el bloque k+1."""
    orden = idx_cron.sort_values().index
    orden = [i for i in orden if i in X.index]
    bloques = np.array_split(np.array(orden), n_blocks)
    preds = pd.Series(index=X.index, dtype=float)
    for k in range(1, n_blocks):
        tr_idx = np.concatenate(bloques[:k])
        te_idx = bloques[k]
        pipe = make_pipe(mfac())
        pipe.fit(X.loc[tr_idx], y.loc[tr_idx])
        preds.loc[te_idx] = pipe.predict(X.loc[te_idx])
    m = preds.notna()
    r2 = r2_score(y[m], preds[m]) if m.sum() > 5 else np.nan
    rho, p = spearmanr(y[m], preds[m]) if m.sum() > 5 else (np.nan, np.nan)
    mae = mean_absolute_error(y[m], preds[m]) if m.sum() > 5 else np.nan
    return dict(r2=r2, rho=rho, p=p, mae=mae, n=int(m.sum()))


def fit_predict_lockbox(Xd, yd, Xl, yl, mfac):
    pipe = make_pipe(mfac())
    m = yd.notna()
    pipe.fit(Xd[m], yd[m])
    pred = pipe.predict(Xl)
    ml = yl.notna()
    r2 = r2_score(yl[ml], pred[ml])
    rho, p = spearmanr(yl[ml], pred[ml])
    mae = mean_absolute_error(yl[ml], pred[ml])
    return dict(r2=r2, rho=rho, p=p, mae=mae, n=int(ml.sum())), pred


# =============================================================================
# 3) Correr todos los modelos x KPIs
# =============================================================================
MODELOS = ["Ridge", "Lasso", "ElasticNet", "HGB", "OLS_ganadores"]
GANADORES_COLS = ["st_R22_sdi", "st_F10_irf_next"] + HEEL

filas_modelos = []
pred_oof_rows = []  # para st_10_pred_oof.csv (guardamos rep 0 de cada modelo/kpi)

log("\n=== Modelos x KPI (KFold5 x5 repeticiones, DEV) + DEV->lockbox ===")
for y_name in YS:
    yd_full = DEV[y_name]
    yl_full = LB[y_name]
    mfull_dev = yd_full.notna()
    for modelo in MODELOS:
        cols = GANADORES_COLS if modelo == "OLS_ganadores" else FEATURES
        Xd = DEV.loc[mfull_dev, cols]
        yd = yd_full[mfull_dev]
        Xl = LB[cols]
        yl = yl_full
        mfac = model_factory(modelo)

        rep_df, oof0 = oof_repeated(Xd, yd, mfac, n_repeats=5, n_splits=5, seed0=0)
        lb_metrics, pred_lb = fit_predict_lockbox(Xd, yd, Xl, yl, mfac)

        fila = dict(kpi=y_name, modelo=modelo, n_dev=len(yd),
                    r2_oof_mean=rep_df.r2.mean(), r2_oof_sd=rep_df.r2.std(),
                    r2_oof_min=rep_df.r2.min(), r2_oof_max=rep_df.r2.max(),
                    rho_oof_mean=rep_df.rho.mean(), rho_oof_sd=rep_df.rho.std(),
                    rho_oof_min=rep_df.rho.min(), rho_oof_max=rep_df.rho.max(),
                    mae_oof_mean=rep_df.mae.mean(),
                    r2_lockbox=lb_metrics["r2"], rho_lockbox=lb_metrics["rho"], p_lockbox=lb_metrics["p"],
                    mae_lockbox=lb_metrics["mae"], n_lockbox=lb_metrics["n"])
        filas_modelos.append(fila)
        log(f"{y_name:26s} {modelo:14s} n_dev={len(yd):3d}  R2_OOF {fila['r2_oof_mean']:+.3f}"
            f"(±{fila['r2_oof_sd']:.3f})  rho_OOF {fila['rho_oof_mean']:+.3f}(±{fila['rho_oof_sd']:.3f})"
            f"  MAE {fila['mae_oof_mean']:.3f}  ||  lockbox R2 {fila['r2_lockbox']:+.3f} rho {fila['rho_lockbox']:+.3f}"
            f" (p={fila['p_lockbox']:.3g}, n={fila['n_lockbox']})")

        for i, idx in enumerate(Xd.index):
            pred_oof_rows.append(dict(Batch=idx, kpi=y_name, modelo=modelo, y_real=yd.loc[idx], y_pred_oof=oof0[i]))

modelos_df = pd.DataFrame(filas_modelos)
modelos_df.to_csv(AQUI / "st_10_modelos.csv", index=False)

# --- walk-forward por bloques cronologicos (solo rendimiento_proxy_batch, todos los modelos)
log("\n=== Walk-forward cronologico por bloques (5 bloques, forward-chaining), rendimiento_proxy_batch ===")
filas_wf = []
y_name = "rendimiento_proxy_batch"
yd_full = DEV[y_name]
mfull_dev = yd_full.notna()
for modelo in MODELOS:
    cols = GANADORES_COLS if modelo == "OLS_ganadores" else FEATURES
    Xd = DEV.loc[mfull_dev, cols]
    yd = yd_full[mfull_dev]
    mfac = model_factory(modelo)
    r = walk_forward(Xd, yd, DEV.loc[mfull_dev, "idx_cronologico"], mfac, n_blocks=5)
    r.update(kpi=y_name, modelo=modelo)
    filas_wf.append(r)
    log(f"walk-forward  {modelo:14s} n={r['n']:3d}  R2 {r['r2']:+.3f}  rho {r['rho']:+.3f} (p={r['p']:.3g})  MAE {r['mae']:.3f}")
wf_df = pd.DataFrame(filas_wf)
wf_df.to_csv(AQUI / "st_10_walkforward.csv", index=False)

techo_row = modelos_df[(modelos_df.kpi == "rendimiento_proxy_batch") & (modelos_df.modelo == "HGB")].iloc[0]
log(f"\n*** TECHO (HGB, todas las {len(FEATURES)} columnas, rendimiento_proxy_batch): "
    f"R2_OOF = {techo_row.r2_oof_mean:+.3f} (±{techo_row.r2_oof_sd:.3f}), rho_OOF = {techo_row.rho_oof_mean:+.3f}, "
    f"R2_lockbox = {techo_row.r2_lockbox:+.3f} ***")

pd.DataFrame(pred_oof_rows).to_csv(AQUI / "st_10_pred_oof.csv", index=False)


# =============================================================================
# 4) Seleccion parsimoniosa: Lasso path + estabilidad (bootstrap) + forward stepwise
# =============================================================================
log("\n=== Seleccion parsimoniosa (target = rendimiento_proxy_batch) ===")
y_name = "rendimiento_proxy_batch"
yd_full = DEV[y_name]
mfull = yd_full.notna()
X_allowed = DEV.loc[mfull, ALLOWED]
y_sel = yd_full[mfull]

imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(X_allowed), index=X_allowed.index, columns=X_allowed.columns)
scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(X_imp), index=X_imp.index, columns=X_imp.columns)
y_std = (y_sel - y_sel.mean()) / y_sel.std()

# --- lasso path (sobre todo DEV, para ver orden de entrada de variables)
alphas, coefs, _ = lasso_path(X_std.values, y_std.values, n_alphas=80, max_iter=20000)
orden_entrada = []
entrado = set()
for j in range(coefs.shape[1] - 1, -1, -1):  # alphas de mayor a menor -> entra primero el mas fuerte
    activos = np.where(np.abs(coefs[:, j]) > 1e-8)[0]
    for a in activos:
        if a not in entrado:
            entrado.add(a)
            orden_entrada.append((X_std.columns[a], alphas[j]))
log("Orden de entrada al Lasso path (variable, alpha de entrada), primeras 10:")
for v, a in orden_entrada[:10]:
    log(f"  {v:28s} alpha={a:.4f}")

# --- estabilidad: 100 bootstraps de DEV, LassoCV, frecuencia de seleccion
N_BOOT = 100
sel_count = pd.Series(0, index=ALLOWED)
rng = np.random.RandomState(1)
for b in range(N_BOOT):
    idx_b = rng.choice(X_allowed.index, size=len(X_allowed), replace=True)
    Xb = X_allowed.loc[idx_b]
    yb = y_sel.loc[idx_b]
    Xb_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xb), columns=Xb.columns)
    Xb_std = pd.DataFrame(StandardScaler().fit_transform(Xb_imp), columns=Xb.columns)
    yb_std = (yb.values - yb.values.mean()) / yb.values.std()
    lcv = LassoCV(cv=5, alphas=30, max_iter=20000, random_state=0).fit(Xb_std.values, yb_std)
    activos = np.abs(lcv.coef_) > 1e-6
    sel_count.loc[Xb_std.columns[activos]] += 1
freq = (sel_count / N_BOOT).sort_values(ascending=False)
log(f"\nEstabilidad (frecuencia de seleccion en {N_BOOT} bootstraps LassoCV), top 12:")
log(freq.head(12).round(2).to_string())

# --- forward stepwise por CV (KFold5 simple, sin repeticion, para velocidad), hasta 8 terminos
def cv_r2_ols(cols, X, y, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    Xc = X[cols]
    for tr, te in kf.split(Xc):
        pipe = make_pipe(LinearRegression())
        pipe.fit(Xc.iloc[tr], y.iloc[tr])
        oof[te] = pipe.predict(Xc.iloc[te])
    return r2_score(y, oof), oof


MAX_TERMS = 8
candidatos = list(freq.index)  # buscar en orden de estabilidad para acotar el espacio (pool completo ALLOWED es chico igual)
seleccionados: list[str] = []
curva = []
restantes = list(ALLOWED)
for k in range(1, MAX_TERMS + 1):
    mejor = None
    for c in restantes:
        cols = seleccionados + [c]
        r2, _ = cv_r2_ols(cols, X_allowed, y_sel)
        if mejor is None or r2 > mejor[1]:
            mejor = (c, r2)
    seleccionados.append(mejor[0])
    restantes.remove(mejor[0])
    curva.append(dict(n_terminos=k, variable_agregada=mejor[0], r2_oof_cv5=mejor[1],
                       terminos=";".join(seleccionados)))
    log(f"stepwise k={k}: +{mejor[0]:28s} R2_OOF(CV5) = {mejor[1]:+.3f}  |  set = {seleccionados}")

curva_df = pd.DataFrame(curva)
curva_df.to_csv(AQUI / "st_10_curva_terminos.csv", index=False)

# elegir compuesto <=6 terminos con mejor R2 OOF (entre k=1..6)
mejor_fila = curva_df[curva_df.n_terminos <= 6].sort_values("r2_oof_cv5", ascending=False).iloc[0]
COMPUESTO = mejor_fila.terminos.split(";")
log(f"\n*** COMPUESTO ELEGIDO (<=6 terminos, mejor R2_OOF CV5): {COMPUESTO} "
    f"R2_OOF(CV5)={mejor_fila.r2_oof_cv5:+.3f} ***")

# metricas del compuesto con el protocolo completo (KFold5 x5 repeticiones) + DEV->lockbox
mfac_ols = model_factory("OLS_ganadores")  # LinearRegression
Xd_c = DEV.loc[mfull, COMPUESTO]
rep_c, oof_c0 = oof_repeated(Xd_c, y_sel, mfac_ols, n_repeats=5, n_splits=5, seed0=0)
Xl_c = LB[COMPUESTO]
lb_c, pred_lb_c = fit_predict_lockbox(Xd_c, y_sel, Xl_c, LB[y_name], mfac_ols)
log(f"Compuesto (protocolo completo, 5x5): R2_OOF={rep_c.r2.mean():+.3f}(±{rep_c.r2.std():.3f}) "
    f"rho_OOF={rep_c.rho.mean():+.3f}(±{rep_c.rho.std():.3f}) MAE={rep_c.mae.mean():.3f}  ||  "
    f"lockbox R2={lb_c['r2']:+.3f} rho={lb_c['rho']:+.3f} (p={lb_c['p']:.3g}, n={lb_c['n']})")

# coeficientes estandarizados del compuesto (OLS) con IC bootstrap (100)
X_c_std = pd.DataFrame(StandardScaler().fit_transform(
    pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xd_c), columns=COMPUESTO, index=Xd_c.index)),
    columns=COMPUESTO, index=Xd_c.index)
m_full = sm.OLS(y_sel.values, sm.add_constant(X_c_std.values)).fit(cov_type="HC3")
coef_full = dict(zip(["const"] + COMPUESTO, m_full.params))
r2_ols_full = m_full.rsquared

boot_coefs = np.full((N_BOOT, len(COMPUESTO) + 1), np.nan)
rng2 = np.random.RandomState(2)
for b in range(N_BOOT):
    idx_b = rng2.choice(Xd_c.index, size=len(Xd_c), replace=True)
    Xb = Xd_c.loc[idx_b]
    yb = y_sel.loc[idx_b]
    Xb_imp = SimpleImputer(strategy="median").fit_transform(Xb)
    Xb_std = StandardScaler().fit_transform(Xb_imp)
    mb = sm.OLS(yb.values, sm.add_constant(Xb_std)).fit()
    boot_coefs[b, :] = mb.params

filas_comp = []
for j, nombre in enumerate(["const"] + COMPUESTO):
    vals = boot_coefs[:, j]
    fila_dir = None
    if nombre != "const":
        base_id = nombre.replace("st_", "") if nombre.startswith("st_") else None
        fila_dir = reg_tabla.loc[base_id, "familia"] if base_id in reg_tabla.index else "heel"
    filas_comp.append(dict(termino=nombre, coef_std=coef_full[nombre],
                            ic_lo=np.percentile(vals, 2.5), ic_hi=np.percentile(vals, 97.5),
                            p_full=m_full.pvalues[j], familia=fila_dir,
                            agg=reg_tabla.loc[nombre.replace("st_", ""), "agg"] if nombre.startswith("st_") and nombre.replace("st_", "") in reg_tabla.index else ("heel_terminal" if nombre in HEEL else "const"),
                            teoria=reg_tabla.loc[nombre.replace("st_", ""), "teoria"] if nombre.startswith("st_") and nombre.replace("st_", "") in reg_tabla.index else ""))
comp_df = pd.DataFrame(filas_comp)
comp_df["r2_ols_dev"] = r2_ols_full
comp_df["r2_oof_5x5"] = rep_c.r2.mean()
comp_df["rho_oof_5x5"] = rep_c.rho.mean()
comp_df["r2_lockbox"] = lb_c["r2"]
comp_df["rho_lockbox"] = lb_c["rho"]
comp_df.to_csv(AQUI / "st_10_compuesto.csv", index=False)
log("\nCoeficientes estandarizados del compuesto (OLS, DEV, IC bootstrap 100):")
log(comp_df[["termino", "coef_std", "ic_lo", "ic_hi", "p_full", "agg", "familia"]].round(4).to_string(index=False))

# formula por escalon (texto)
log("\nFormula por escalon del compuesto elegido (cada termino ya viene orientado mayor=mejor):")
formula_terms = []
for _, row in comp_df[comp_df.termino != "const"].iterrows():
    w = row["coef_std"]
    formula_terms.append(f"{w:+.3f}*z({row['termino']})")
log("  score_batch = " + "  ".join(formula_terms))
log("  (z = estandarizado con media/sd de DEV; cada st_<id> con agg='sum' o 'product' es una suma/producto")
log("   telescopico de un termino por escalon -> el recomendador puede maximizar el termino de escalon")
log("   correspondiente en cada paso; los terminos 'last'/heel son estados terminales de la fase/heel,")
log("   validos como target del ULTIMO escalon o como covariable de estado inicial, no como suma por paso.)")


# =============================================================================
# 5) Permutation importance de HGB (rendimiento_proxy_batch, todas las columnas)
# =============================================================================
log("\n=== Permutation importance HGB (rendimiento_proxy_batch, todas las columnas) ===")
y_name = "rendimiento_proxy_batch"
yd_full = DEV[y_name]
mfull = yd_full.notna()
Xd_all = DEV.loc[mfull, FEATURES]
yd_all = yd_full[mfull]
rng3 = np.random.RandomState(3)
idx_shuf = rng3.permutation(Xd_all.index)
n_test = int(0.2 * len(idx_shuf))
te_idx, tr_idx = idx_shuf[:n_test], idx_shuf[n_test:]
pipe_hgb = make_pipe(model_factory("HGB")())
pipe_hgb.fit(Xd_all.loc[tr_idx], yd_all.loc[tr_idx])
pi = permutation_importance(pipe_hgb, Xd_all.loc[te_idx], yd_all.loc[te_idx], n_repeats=30,
                             random_state=3, scoring="r2")
pi_df = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                       "importance_sd": pi.importances_std}).sort_values("importance_mean", ascending=False)
pi_df.to_csv(AQUI / "st_10_permutation_importance.csv", index=False)
log(pi_df.head(15).round(4).to_string(index=False))


# =============================================================================
# 6) Figuras
# =============================================================================
# 6a: R2 OOF vs numero de terminos (forward stepwise)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(curva_df.n_terminos, curva_df.r2_oof_cv5, "o-", color="#2b6cb0")
ax.axvline(len(COMPUESTO), color="#c53030", ls="--", lw=1, label=f"compuesto elegido (k={len(COMPUESTO)})")
ax.axhline(techo_row.r2_oof_mean, color="#2f855a", ls=":", lw=1.2, label=f"techo HGB todas las cols ({techo_row.r2_oof_mean:.3f})")
ax.set_xlabel("numero de terminos (forward stepwise, OLS, CV5)")
ax.set_ylabel("R2 OOF (CV5, DEV)")
ax.set_title("R2 OOF vs numero de terminos del compuesto")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIGS / "st_10_curva_terminos.png", dpi=120)
plt.close(fig)

# 6b: permutation importance HGB (top 15)
fig, ax = plt.subplots(figsize=(7, 5.5))
top = pi_df.head(15).iloc[::-1]
ax.barh(top.feature, top.importance_mean, xerr=top.importance_sd, color="#2b6cb0")
ax.set_xlabel("caida de R2 al permutar (30 repeticiones, held-out 20% DEV)")
ax.set_title("Permutation importance HGB (rendimiento_proxy_batch)")
plt.tight_layout()
fig.savefig(FIGS / "st_10_permutation_importance.png", dpi=120)
plt.close(fig)

# 6c: scatter OOF pred vs real, DEV (HGB) y lockbox (HGB DEV-> lockbox)
hgb_oof_rows = [r for r in pred_oof_rows if r["modelo"] == "HGB" and r["kpi"] == "rendimiento_proxy_batch"]
hgb_oof_df = pd.DataFrame(hgb_oof_rows)
mfac_hgb = model_factory("HGB")
lb_metrics_hgb, pred_lb_hgb = fit_predict_lockbox(Xd_all, yd_all, LB[FEATURES], LB[y_name], mfac_hgb)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
axes[0].scatter(hgb_oof_df.y_real, hgb_oof_df.y_pred_oof, s=18, alpha=0.6, color="#2b6cb0")
lims = [min(hgb_oof_df.y_real.min(), hgb_oof_df.y_pred_oof.min()), max(hgb_oof_df.y_real.max(), hgb_oof_df.y_pred_oof.max())]
axes[0].plot(lims, lims, "k--", lw=1)
axes[0].set_xlabel("rendimiento real (DEV)")
axes[0].set_ylabel("rendimiento OOF pred (HGB, rep 0)")
r2_show = modelos_df[(modelos_df.kpi == "rendimiento_proxy_batch") & (modelos_df.modelo == "HGB")].r2_oof_mean.iloc[0]
axes[0].set_title(f"DEV OOF (HGB): R2={r2_show:.3f}")

ml = LB[y_name].notna()
axes[1].scatter(LB[y_name][ml], pred_lb_hgb[ml], s=18, alpha=0.6, color="#c53030")
lims2 = [min(LB[y_name][ml].min(), pred_lb_hgb[ml].min()), max(LB[y_name][ml].max(), pred_lb_hgb[ml].max())]
axes[1].plot(lims2, lims2, "k--", lw=1)
axes[1].set_xlabel("rendimiento real (lockbox)")
axes[1].set_ylabel("rendimiento pred (HGB, DEV->lockbox)")
axes[1].set_title(f"Lockbox (HGB): R2={lb_metrics_hgb['r2']:.3f}")
plt.tight_layout()
fig.savefig(FIGS / "st_10_scatter_oof_lockbox.png", dpi=120)
plt.close(fig)

# 6d: scatter del compuesto elegido (OLS), DEV OOF y lockbox
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
axes[0].scatter(y_sel, oof_c0, s=18, alpha=0.6, color="#2b6cb0")
lims = [min(y_sel.min(), np.nanmin(oof_c0)), max(y_sel.max(), np.nanmax(oof_c0))]
axes[0].plot(lims, lims, "k--", lw=1)
axes[0].set_xlabel("rendimiento real (DEV)")
axes[0].set_ylabel("pred OOF (compuesto, rep 0)")
axes[0].set_title(f"DEV OOF (compuesto): R2={rep_c.r2.mean():.3f}")

axes[1].scatter(LB[y_name][ml], pred_lb_c[ml], s=18, alpha=0.6, color="#c53030")
lims2 = [min(LB[y_name][ml].min(), pred_lb_c[ml].min()), max(LB[y_name][ml].max(), pred_lb_c[ml].max())]
axes[1].plot(lims2, lims2, "k--", lw=1)
axes[1].set_xlabel("rendimiento real (lockbox)")
axes[1].set_ylabel("pred (compuesto, DEV->lockbox)")
axes[1].set_title(f"Lockbox (compuesto): R2={lb_c['r2']:.3f}")
plt.tight_layout()
fig.savefig(FIGS / "st_10_scatter_compuesto.png", dpi=120)
plt.close(fig)

log(f"\ntiempo total {time.time()-t0:.1f}s")
(AQUI / "st_10_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("listo.")
