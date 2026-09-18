"""E10-04a (H4) -- anclajes robustos del objetivo de Reduccion (K, w) y objetivo lineal en kg como contraste.

Preguntas (ITERACION_10_diseno.md, E10-04a):
 1. Anclaje base (sin legado) del KPI sobre Sum_R ln_sn_dep y Sum_R ln_feo_ret + controles: K_Sn, K_FeO, w = K_FeO/K_Sn,
    IC bootstrap por batch (1000 replicas); replica con la agregacion EXACTA por cociente de inventarios (primer/ultimo
    escalon VALIDO de Reduccion) en vez de la suma por escalon; y solo con batches con ensayo en R3.
 2. Correccion por errores-en-variables (EIV): fiabilidad lambda = 1 - var_ruido/var_total de Sum_R ln_feo_ret (y de
    Sum_R ln_sn_dep) por dos vias: (a) ruido de ensayo (2 % Sn, 2.8 % FeO, dos extremos: var = 2*sigma^2 -- e9_05_techo.py,
    e8_01_masa_variantes.py) y (b) el residuo OOF de un HGB del target por escalon (estado k16) agregado por batch, como
    cota superior de var_ruido. K corregido = K/lambda.
 3. Legado (E9-00b): KPI_next ~ Sum_R ln_feo_ret + Sum_R ln_sn_dep + KPI_actual + controles; y anclaje sobre
    y = KPI + KPI_next. Sensibilidad, no base.
 4. Objetivo lineal en kg (auditoria A Sec.4): Delta Sn_kg_R (Sn agotado) y Delta FeO_kg_R (FeO reducido) desde
    m6_sn_inv_kg / m6_feo_inv_kg del primer/ultimo escalon VALIDO; KPI ~ DeltaSn_kg_R + DeltaFeO_kg_R (por 1000 kg) +
    controles -> lambda_kg = -coef_FeO/coef_Sn (kg Sn equivalente por kg FeO reducido); version con
    sn_escoria_final_balance_kg (balance global) como control adicional / contraste.
 5. Barrido de w en {1,2,4,6.09,7.86,11}: Spearman de J_R(w) = Sum ln_sn_dep + w*Sum ln_feo_ret vs KPI en DEV/lockbox/total
    y tercios cronologicos.

Salidas: e10_04a_anclaje_base.csv, e10_04a_eiv.csv, e10_04a_legado.csv, e10_04a_lineal_kg.csv, e10_04a_barrido_w.csv,
memo e10_04a_resultados.md, log e10_04a_log.txt.

Entorno: `.venv/Scripts/python.exe`, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3, sin joblib.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "experimentos" / "v8"))
sys.path.insert(0, str(RAIZ / "experimentos" / "v7"))
sys.path.insert(0, str(RAIZ))
import v8_lib as L  # noqa: E402
import feature_engineering as fe  # noqa: E402

OUT = Path(__file__).resolve().parent
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


SEED = L.SEED
K = L.KPI_PRINCIPAL
C = L.CONTROLES_BATCH
SN, FE = "R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret"
SIGMA_SN, SIGMA_FEO = 0.02, 0.028   # ruido log-normal por ensayo (e9_05_techo.py / e8_01_masa_variantes.py)

# --------------------------------------------------------------------------- datos
log("cargando datos...")
df = L.cargar_df()
batch = L.construir_batch(df).reset_index()  # columna 'Batch'
dev_b, lockbox_b = L.split(df)
batch["muestra_dev"] = batch["Batch"].isin(dev_b)
batch["muestra_lockbox"] = batch["Batch"].isin(lockbox_b)
assert (batch["muestra_dev"] == batch["es_dev"]).all()
log(f"batch {batch.shape}")


# --------------------------------------------------------------------------- utilidades OLS + bootstrap
def ols_boot(dd: pd.DataFrame, y: str, xcols: list[str], controles: list[str], n_boot: int = 1000, seed: int = SEED):
    """OLS HC3 + bootstrap por fila (cada fila = 1 batch, no hay clustering adicional que hacer)."""
    cols = xcols + controles
    sub = dd[[y] + cols].dropna().reset_index(drop=True)
    m = sm.OLS(sub[y], sm.add_constant(sub[cols])).fit(cov_type="HC3")
    rng = np.random.default_rng(seed)
    boots = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = sub.iloc[idx]
        try:
            p = sm.OLS(b[y], sm.add_constant(b[cols])).fit().params
            boots.append(p.reindex(["const"] + cols).to_numpy())
        except Exception:
            continue
    boots = pd.DataFrame(boots, columns=["const"] + cols)
    return m, boots, n


def resumen_kw(nombre: str, m, boots: pd.DataFrame, n: int, sn_col: str, fe_col: str) -> dict:
    k_sn, k_feo = m.params[sn_col], m.params[fe_col]
    w = k_feo / k_sn if k_sn else np.nan
    w_boot = (boots[fe_col] / boots[sn_col]).replace([np.inf, -np.inf], np.nan)
    return dict(
        spec=nombre, n=n, K_Sn=k_sn, K_Sn_lo=m.conf_int().loc[sn_col, 0], K_Sn_hi=m.conf_int().loc[sn_col, 1], K_Sn_p=m.pvalues[sn_col],
        K_FeO=k_feo, K_FeO_lo=m.conf_int().loc[fe_col, 0], K_FeO_hi=m.conf_int().loc[fe_col, 1], K_FeO_p=m.pvalues[fe_col],
        w=w, w_boot_lo=np.nanpercentile(w_boot, 2.5), w_boot_hi=np.nanpercentile(w_boot, 97.5),
        frac_ambos_positivos=float(np.nanmean((boots[sn_col] > 0) & (boots[fe_col] > 0))),
        r2adj=m.rsquared_adj,
    )


# --------------------------------------------------------------------------- 1) anclaje base (sin legado)
log("1) anclaje base...")
filas1 = []
for muestra, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
    m, boots, n = ols_boot(sub, K, [SN, FE], C)
    filas1.append({"muestra": muestra, **resumen_kw("suma_escalon", m, boots, n, SN, FE)})

# 1b) agregacion exacta por cociente de inventarios (primer/ultimo escalon VALIDO de Reduccion)
log("1b) agregacion por cociente de inventarios (primer/ultimo escalon valido)...")
r = df[df["fase_proceso"].eq("Reducción")].sort_values(["Batch", "fecha_inicio"])
filas_inv = []
for b, g in r.groupby("Batch", sort=False):
    gv = g[g["m6_valido"].astype(bool)]
    tiene_r3 = bool((g["orden_escalon_fase"].eq(3) & g["m6_valido"].astype(bool)).any())
    if len(gv) == 0:
        filas_inv.append(dict(Batch=b, sn_ini=np.nan, sn_fin=np.nan, feo_ini=np.nan, feo_fin=np.nan,
                              n_validos=0, tiene_r3=tiene_r3))
        continue
    filas_inv.append(dict(
        Batch=b, sn_ini=float(gv["m6_sn_inv_kg_prev"].iloc[0]), sn_fin=float(gv["m6_sn_inv_kg"].iloc[-1]),
        feo_ini=float(gv["m6_feo_inv_kg_prev"].iloc[0]), feo_fin=float(gv["m6_feo_inv_kg"].iloc[-1]),
        n_validos=len(gv), tiene_r3=tiene_r3,
    ))
inv = pd.DataFrame(filas_inv)
inv["ln_sn_dep_inv"] = np.log(inv["sn_ini"] / inv["sn_fin"])
inv["ln_feo_ret_inv"] = np.log(inv["feo_fin"] / inv["feo_ini"])
inv["dSn_kg_R"] = inv["sn_ini"] - inv["sn_fin"]
inv["dFeO_kg_R"] = inv["feo_ini"] - inv["feo_fin"]
batch = batch.merge(inv[["Batch", "ln_sn_dep_inv", "ln_feo_ret_inv", "dSn_kg_R", "dFeO_kg_R", "n_validos", "tiene_r3"]],
                    on="Batch", how="left")
log(f"batches con ensayo en R3: {batch['tiene_r3'].mean():.3f} (esperado ~0.71)")

for muestra, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
    m, boots, n = ols_boot(sub, K, ["ln_sn_dep_inv", "ln_feo_ret_inv"], C)
    filas1.append({"muestra": muestra, **resumen_kw("cociente_inventarios", m, boots, n, "ln_sn_dep_inv", "ln_feo_ret_inv")})

# 1c) solo batches con ensayo en R3
for muestra, sub in [("dev", batch[batch["es_dev"] & batch["tiene_r3"]]),
                     ("lockbox", batch[batch["es_lockbox"] & batch["tiene_r3"]]),
                     ("total", batch[batch["tiene_r3"]])]:
    m, boots, n = ols_boot(sub, K, [SN, FE], C)
    filas1.append({"muestra": muestra, **resumen_kw("solo_con_ensayo_R3", m, boots, n, SN, FE)})

anclaje_base = pd.DataFrame(filas1)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("\n== 1) anclaje base ==\n", anclaje_base.round(4).to_string())
anclaje_base.to_csv(OUT / "e10_04a_anclaje_base.csv", index=False)

# --------------------------------------------------------------------------- 2) correccion EIV
log("2) EIV: HGB por escalon (estado k16) para el residuo OOF...")
from sklearn.model_selection import GroupKFold  # noqa: E402

dR = L.base_fase(df, "Reducción")
S16 = L.ESTADO_R_V7
L.asegurar_seguras(S16)
targets_eiv = {"m6_ln_feo_ret": FE, "m6_ln_sn_dep": SN}
filas_eiv = []
res_por_batch = {}
for tgt, sum_col in targets_eiv.items():
    need = S16 + [tgt, "Batch"]
    d = dR[need].dropna().reset_index(drop=True)
    d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(5).split(d_dev, d_dev[tgt], d_dev["Batch"]):
        mdl = L.hgb_nuisance(SEED).fit(d_dev[S16].iloc[tr], d_dev[tgt].iloc[tr])
        oof[te] = mdl.predict(d_dev[S16].iloc[te])
    resid = d_dev[tgt].to_numpy() - oof
    d_dev = d_dev.assign(resid=resid)
    resid_batch = d_dev.groupby("Batch")["resid"].sum()
    res_por_batch[tgt] = resid_batch
    var_resid_b = float(resid_batch.var(ddof=1))
    var_total = float(batch.loc[batch["es_dev"], sum_col].var(ddof=1))
    sigma = SIGMA_FEO if tgt == "m6_ln_feo_ret" else SIGMA_SN
    var_ruido_a = 2 * sigma ** 2
    lam_a = 1 - var_ruido_a / var_total
    lam_b = 1 - var_resid_b / var_total
    filas_eiv.append(dict(target=tgt, sum_col=sum_col, n_batches_dev_hgb=len(resid_batch), var_total_dev=var_total,
                          sigma_ensayo=sigma, var_ruido_a_ensayo=var_ruido_a, lambda_a_ensayo=lam_a,
                          var_ruido_b_hgb_oof=var_resid_b, lambda_b_hgb=lam_b,
                          lambda_lo=min(lam_a, lam_b), lambda_hi=max(lam_a, lam_b)))
eiv = pd.DataFrame(filas_eiv)

# K corregido: usa el anclaje base DEV (suma por escalon) de la seccion 1
fila_dev = anclaje_base.query("muestra == 'dev' and spec == 'suma_escalon'").iloc[0]
k_sn_dev, k_feo_dev = fila_dev["K_Sn"], fila_dev["K_FeO"]
lam_sn = eiv.query("target == 'm6_ln_sn_dep'").iloc[0]
lam_feo = eiv.query("target == 'm6_ln_feo_ret'").iloc[0]
filas_corr = []
for lam_feo_val, etiqueta_feo in [(lam_feo["lambda_a_ensayo"], "a_ensayo"), (lam_feo["lambda_b_hgb"], "b_hgb")]:
    for lam_sn_val, etiqueta_sn in [(lam_sn["lambda_a_ensayo"], "a_ensayo"), (lam_sn["lambda_b_hgb"], "b_hgb")]:
        lam_feo_c = float(np.clip(lam_feo_val, 0.05, 1.0))
        lam_sn_c = float(np.clip(lam_sn_val, 0.05, 1.0))
        k_feo_corr = k_feo_dev / lam_feo_c
        k_sn_corr = k_sn_dev / lam_sn_c
        filas_corr.append(dict(lambda_feo_via=etiqueta_feo, lambda_sn_via=etiqueta_sn, lambda_feo=lam_feo_c, lambda_sn=lam_sn_c,
                               K_Sn_corr=k_sn_corr, K_FeO_corr=k_feo_corr, w_corr=k_feo_corr / k_sn_corr))
corr = pd.DataFrame(filas_corr)
print("\n== 2) EIV: fiabilidad lambda y K/w corregidos ==\n", eiv.round(4).to_string())
print("\n-- K,w corregidos (combinaciones de via) --\n", corr.round(3).to_string())
eiv.to_csv(OUT / "e10_04a_eiv.csv", index=False)
corr.to_csv(OUT / "e10_04a_eiv_corregido.csv", index=False)

# --------------------------------------------------------------------------- 3) legado
log("3) legado...")
batch = batch.sort_values("fecha_batch").reset_index(drop=True)
batch["KPI_actual"] = batch[K]
batch["KPI_next"] = batch[f"{K}_next"]
batch["KPI_sum2"] = batch["KPI_actual"] + batch["KPI_next"]
batch["KPI_prev"] = batch["KPI_actual"].shift(1)
filas_leg = []
for muestra, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
    # KPI_next ~ FeO + Sn + KPI_actual + controles
    dd = sub[["KPI_next", SN, FE, "KPI_actual"] + C].dropna()
    if len(dd) > 15:
        m = sm.OLS(dd["KPI_next"], sm.add_constant(dd[[SN, FE, "KPI_actual"] + C])).fit(cov_type="HC3")
        for c in [SN, FE, "KPI_actual"]:
            filas_leg.append(dict(muestra=muestra, spec="next_controlando_actual", variable=c, coef=m.params[c],
                                  lo=m.conf_int().loc[c, 0], hi=m.conf_int().loc[c, 1], p=m.pvalues[c], n=len(dd), r2adj=m.rsquared_adj))
    # placebo temporal: el batch ANTERIOR no puede depender de lo que hago hoy
    ddp = sub[["KPI_prev", SN, FE] + C].dropna()
    if len(ddp) > 15:
        m = sm.OLS(ddp["KPI_prev"], sm.add_constant(ddp[[SN, FE] + C])).fit(cov_type="HC3")
        for c in [SN, FE]:
            filas_leg.append(dict(muestra=muestra, spec="placebo_prev", variable=c, coef=m.params[c],
                                  lo=m.conf_int().loc[c, 0], hi=m.conf_int().loc[c, 1], p=m.pvalues[c], n=len(ddp), r2adj=m.rsquared_adj))
legado = pd.DataFrame(filas_leg)

filas_leg_kw = []
for muestra, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
    for y in ["KPI_actual", "KPI_sum2"]:
        dd2 = sub[[y, SN, FE] + C].dropna()
        if len(dd2) > 15:
            m, boots, n = ols_boot(sub, y, [SN, FE], C)
            filas_leg_kw.append({"muestra": muestra, "y": y, **resumen_kw(y, m, boots, n, SN, FE)})
legado_kw = pd.DataFrame(filas_leg_kw)
print("\n== 3) legado: efectos sobre KPI_next / placebo ==\n", legado.round(4).to_string())
print("\n== 3) legado: anclaje K,w (KPI_actual vs KPI_actual+KPI_next) ==\n", legado_kw.round(3).to_string())
pd.concat([legado, legado_kw], axis=0, ignore_index=True, sort=False).to_csv(OUT / "e10_04a_legado.csv", index=False)

# --------------------------------------------------------------------------- 4) objetivo lineal en kg
log("4) objetivo lineal en kg...")
batch["dSn_kg_R_k"] = batch["dSn_kg_R"] / 1000.0
batch["dFeO_kg_R_k"] = batch["dFeO_kg_R"] / 1000.0
try:
    bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
    batch = batch.merge(bal[["sn_escoria_final_balance_kg"]].reset_index(), on="Batch", how="left")
    tiene_sn_escoria = True
except Exception as e:
    log("no se pudo construir sn_escoria_final_balance_kg:", repr(e))
    tiene_sn_escoria = False
if tiene_sn_escoria:
    batch["sn_escoria_final_balance_kg_k"] = batch["sn_escoria_final_balance_kg"] / 1000.0

filas_kg = []
for muestra, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
    m, boots, n = ols_boot(sub, K, ["dSn_kg_R_k", "dFeO_kg_R_k"], C)
    k_sn, k_feo = m.params["dSn_kg_R_k"], m.params["dFeO_kg_R_k"]
    lam_kg = -k_feo / k_sn if k_sn else np.nan
    lam_boot = (-boots["dFeO_kg_R_k"] / boots["dSn_kg_R_k"]).replace([np.inf, -np.inf], np.nan)
    filas_kg.append(dict(muestra=muestra, spec="lineal_kg", n=n, coef_dSn_por1000kg=k_sn, coef_dSn_p=m.pvalues["dSn_kg_R_k"],
                         coef_dFeO_por1000kg=k_feo, coef_dFeO_p=m.pvalues["dFeO_kg_R_k"], lambda_kg=lam_kg,
                         lambda_kg_lo=np.nanpercentile(lam_boot, 2.5), lambda_kg_hi=np.nanpercentile(lam_boot, 97.5),
                         r2adj=m.rsquared_adj))
    if tiene_sn_escoria:
        m2, boots2, n2 = ols_boot(sub, K, ["dSn_kg_R_k", "dFeO_kg_R_k", "sn_escoria_final_balance_kg_k"], C)
        k_sn2, k_feo2 = m2.params["dSn_kg_R_k"], m2.params["dFeO_kg_R_k"]
        lam_kg2 = -k_feo2 / k_sn2 if k_sn2 else np.nan
        lam_boot2 = (-boots2["dFeO_kg_R_k"] / boots2["dSn_kg_R_k"]).replace([np.inf, -np.inf], np.nan)
        filas_kg.append(dict(muestra=muestra, spec="lineal_kg_con_sn_escoria_final", n=n2, coef_dSn_por1000kg=k_sn2,
                             coef_dSn_p=m2.pvalues["dSn_kg_R_k"], coef_dFeO_por1000kg=k_feo2, coef_dFeO_p=m2.pvalues["dFeO_kg_R_k"],
                             lambda_kg=lam_kg2, lambda_kg_lo=np.nanpercentile(lam_boot2, 2.5), lambda_kg_hi=np.nanpercentile(lam_boot2, 97.5),
                             r2adj=m2.rsquared_adj, coef_sn_escoria_final_por1000kg=m2.params["sn_escoria_final_balance_kg_k"],
                             coef_sn_escoria_final_p=m2.pvalues["sn_escoria_final_balance_kg_k"]))
lineal_kg = pd.DataFrame(filas_kg)
print("\n== 4) objetivo lineal en kg ==\n", lineal_kg.round(4).to_string())
lineal_kg.to_csv(OUT / "e10_04a_lineal_kg.csv", index=False)

# --------------------------------------------------------------------------- 5) barrido de w
log("5) barrido de w...")
ws = [1, 2, 4, 6.09, 7.86, 11]
batch["tercio_cron"] = pd.qcut(batch["idx_cronologico"], 3, labels=False)
filas_w = []
for w in ws:
    batch[f"J_R_w{w}"] = batch[SN] + w * batch[FE]
    for muestra, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
        dd = sub[[f"J_R_w{w}", K]].dropna()
        rho, p = stats.spearmanr(dd[f"J_R_w{w}"], dd[K])
        filas_w.append(dict(w=w, muestra=muestra, tercio="todos", rho=rho, p=p, n=len(dd)))
    for terc in sorted(batch["tercio_cron"].dropna().unique()):
        dd = batch.loc[batch["tercio_cron"] == terc, [f"J_R_w{w}", K]].dropna()
        rho, p = stats.spearmanr(dd[f"J_R_w{w}"], dd[K])
        filas_w.append(dict(w=w, muestra="total", tercio=f"tercio_{int(terc)}", rho=rho, p=p, n=len(dd)))
barrido_w = pd.DataFrame(filas_w)
print("\n== 5) barrido de w: Spearman J_R(w) vs KPI ==\n", barrido_w.round(4).to_string())
barrido_w.to_csv(OUT / "e10_04a_barrido_w.csv", index=False)

log("FIN")
print("\nFIN")
