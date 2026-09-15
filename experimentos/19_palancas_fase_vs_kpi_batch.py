"""Experimento 19 -- Palancas operacionales por FASE vs canales de perdida del batch.

Pregunta: que decisiones de Fusion y de Reduccion se asocian, con controles, a
f_metal / f_dross / f_polvo / rendimiento_proxy_batch (hallazgos.md 14.6, ITERACION_4_diseno.md).

Todo el codigo vive aqui; no se modifica ningun modulo de la raiz ni hallazgos.md.
Ejecutar en primer plano, sin joblib/multiprocessing (362 filas de batch):

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/19_palancas_fase_vs_kpi_batch.py

Salidas: 19_batch_dataset.csv, 19_regresiones_kpi.csv, 19_pd_kpi.csv, 19_vif.csv,
19_hgb_resumen.csv, figuras en figs_19/.

Convenciones de agregacion (documentadas aqui porque el enunciado no las fija todas):
 - "dur_F"/"dur_R": suma de `duracion_plan_min` (es la ACCION planificada; memoria del
   proyecto: "duracion plan es ACTION"), no el tiempo real transcurrido.
 - Medias "ponderadas por tiempo" (exceso_o2_F, exceso_o2_R): promedio de
   `exceso_o2_combustion_pct` ponderado por `delta_tiempo` (tiempo real del escalon).
 - El resto de las medias (enriq_o2_F, T_media_F, T_media_R, lanza_media_F, lanza_media_R)
   son medias simples entre escalones de la fase (el enunciado no pide ponderacion).
 - Estandarizacion para las regresiones OLS: se calcula UNA vez con media/sd de la
   muestra total (362 batches, tras dropna) y se reutiliza igual en DEV, TOTAL y
   LOCKBOX, para que los coeficientes ("por +1 sd") sean directamente comparables
   entre las tres muestras (chequeo de consistencia de signo).
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp  # noqa: E402

FIGS = Path(__file__).resolve().parent / "figs_19"
FIGS.mkdir(exist_ok=True)
OUT = Path(__file__).resolve().parent
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# =============================================================================
# 0) Dataset de escalones y construccion del dataset batch-level
# =============================================================================
log("cargando dataset v3 (escalones) ...")
df = pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl")
df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
log(f"df escalones: {df.shape}, batches dev={len(batches_dev)} lockbox={len(batches_lockbox)}")


def wmean(g, col, w="delta_tiempo"):
    v = g[col].to_numpy(dtype=float)
    ww = g[w].to_numpy(dtype=float)
    m = np.isfinite(v) & np.isfinite(ww) & (ww > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.average(v[m], weights=ww[m]))


filas = []
for batch_id, g in df.groupby("Batch"):
    g = g.sort_values("fecha_inicio")
    gf = g.loc[g["fase_proceso"] == "Fusión"]
    gr = g.loc[g["fase_proceso"] == "Reducción"]
    if gf.empty or gr.empty:
        continue

    # ---- KPIs de batch ----
    sn_metal_t = g["sn_en_metal_crudo_batch_t"].iloc[0]
    sn_dross_t = g["sn_en_dross_fe_batch_t"].iloc[0]
    sn_polvo_t = g["sn_en_polvo_fundicion_batch_t"].iloc[0]
    tot_out = sn_metal_t + sn_dross_t + sn_polvo_t
    f_metal = sn_metal_t / tot_out if tot_out else np.nan
    f_dross = sn_dross_t / tot_out if tot_out else np.nan
    f_polvo = sn_polvo_t / tot_out if tot_out else np.nan

    # ---- Fusion: decisiones ----
    carbon_total_F = gf["feed_Carbon_kgh"].sum(min_count=1)
    feed_sn_F = gf["feed_Sn_kgf"].sum(min_count=1)
    C_por_Sn_F = carbon_total_F / feed_sn_F if feed_sn_F else np.nan
    cal_total_F = gf["feed_CaO_kgh"].sum(min_count=1)
    feed_total_F = gf["feed_total_kgh"].sum(min_count=1)
    cal_por_carga_F = cal_total_F / feed_total_F if feed_total_F else np.nan
    gn_total_F = gf["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum(min_count=1)
    gn_por_t_carga_F = gn_total_F / (feed_total_F / 1000) if feed_total_F else np.nan
    exceso_o2_F = wmean(gf, "exceso_o2_combustion_pct")
    enriq_o2_F = gf["oxygen_enrichment_pct"].mean()
    T_media_F = gf["temperatura_horno_celsius"].mean()
    T_final_F = gf["temperatura_horno_celsius"].iloc[-1]
    dur_F = gf["duracion_plan_min"].sum(min_count=1)
    feed_rate_F = feed_total_F / dur_F if dur_F else np.nan
    lanza_media_F = gf["posicion_vertical_lanza_mm"].mean()
    lanza_std_F = gf["posicion_vertical_lanza_mm"].std()

    # ---- estado al fin de Fusion ----
    ley_sn_finF = gf["ley_sn_escoria_pct"].iloc[-1]
    ley_feo_finF = gf["ley_feo_escoria_pct"].iloc[-1]
    b2_finF = gf["basicidad_B2"].iloc[-1]
    sn_inv_finF_kg = gf["sn_inventario_escoria_est_kg"].iloc[-1]
    masa_finF_kg = gf["masa_escoria_est_kg"].iloc[-1]
    F_avance_final = 1 - sn_inv_finF_kg / feed_sn_F if feed_sn_F else np.nan

    # ---- Reduccion: decisiones ----
    carbon_total_R = gr["feed_Carbon_kgh"].sum(min_count=1)
    carbon_por_Sn_inv_R = carbon_total_R / sn_inv_finF_kg if sn_inv_finF_kg and sn_inv_finF_kg > 0 else np.nan
    gn_total_R = gr["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum(min_count=1)
    exceso_o2_R = wmean(gr, "exceso_o2_combustion_pct")
    dur_R = gr["duracion_plan_min"].sum(min_count=1)
    carbon_rate_R0 = gr["tasa_feed_Carbon_kg_min"].iloc[0]
    carbon_R0_kg = gr["feed_Carbon_kgh"].iloc[0]
    perfil_carbon_R = carbon_R0_kg / carbon_total_R if carbon_total_R else np.nan
    T_media_R = gr["temperatura_horno_celsius"].mean()
    lanza_media_R = gr["posicion_vertical_lanza_mm"].mean()

    # ---- resultados/mecanismo de Reduccion (para M4, no como "decision") ----
    FeO_reducido_R_t = gr["feo_extraido_est_kg"].sum(min_count=1) / 1000
    Sn_extraido_R_t = gr["sn_extraido_est_kg"].sum(min_count=1) / 1000
    # umbral 0.5 t (no solo >0): con FeO_reducido_R_t cerca de cero el cociente explota
    # (se vieron valores de hasta 1e24 con el umbral ">0" ingenuo) -- se exige un
    # denominador con magnitud fisica minima; se winsoriza ademas mas abajo.
    selectividad_R = Sn_extraido_R_t / FeO_reducido_R_t if FeO_reducido_R_t and FeO_reducido_R_t > 0.5 else np.nan
    ley_sn_final = gr["ley_sn_escoria_pct"].iloc[-1]
    sn_inv_final_kg = gr["sn_inventario_escoria_est_kg"].iloc[-1]

    # ---- controles ----
    feed_sn_total_t = g["feed_Sn_kgf"].sum(min_count=1) / 1000
    avance_final_batch = 1 - sn_inv_final_kg / (feed_sn_total_t * 1000) if feed_sn_total_t else np.nan
    ley_sn_conc_batch_pct = g["ley_sn_conc_batch_pct"].iloc[0]
    espesor_ladrillo_norm_mm = g["espesor_ladrillo_norm_mm"].mean()
    frac_carga_secundaria_F = gf["frac_carga_secundaria"].mean()
    feed_dross_Fe_total_t = g["feed_dross_Fe_kgh"].sum(min_count=1) / 1000
    fecha_batch = g["fecha_inicio"].min()

    filas.append(dict(
        Batch=batch_id,
        f_metal=f_metal, f_dross=f_dross, f_polvo=f_polvo,
        rendimiento_proxy_batch=g["rendimiento_proxy_batch"].iloc[0],
        sn_metal_t=sn_metal_t, sn_dross_t=sn_dross_t, sn_polvo_t=sn_polvo_t,
        # decisiones Fusion
        C_por_Sn_F=C_por_Sn_F, cal_por_carga_F=cal_por_carga_F, gn_por_t_carga_F=gn_por_t_carga_F,
        exceso_o2_F=exceso_o2_F, enriq_o2_F=enriq_o2_F, T_media_F=T_media_F, T_final_F=T_final_F,
        dur_F=dur_F, feed_rate_F=feed_rate_F, lanza_media_F=lanza_media_F, lanza_std_F=lanza_std_F,
        carbon_total_F=carbon_total_F, cal_total_F=cal_total_F, feed_total_F=feed_total_F, feed_sn_F=feed_sn_F,
        # estado fin Fusion
        ley_sn_finF=ley_sn_finF, ley_feo_finF=ley_feo_finF, b2_finF=b2_finF,
        F_avance_final=F_avance_final, sn_inv_finF_t=sn_inv_finF_kg / 1000 if pd.notna(sn_inv_finF_kg) else np.nan,
        masa_finF_t=masa_finF_kg / 1000 if pd.notna(masa_finF_kg) else np.nan,
        # decisiones Reduccion
        carbon_total_R=carbon_total_R, carbon_por_Sn_inv_R=carbon_por_Sn_inv_R, gn_total_R=gn_total_R,
        exceso_o2_R=exceso_o2_R, dur_R=dur_R, carbon_rate_R0=carbon_rate_R0, perfil_carbon_R=perfil_carbon_R,
        T_media_R=T_media_R, lanza_media_R=lanza_media_R,
        # resultados/mecanismo Reduccion
        FeO_reducido_R_t=FeO_reducido_R_t, Sn_extraido_R_t=Sn_extraido_R_t, selectividad_R=selectividad_R,
        ley_sn_final=ley_sn_final, avance_final_batch=avance_final_batch,
        # controles
        feed_sn_total_t=feed_sn_total_t, ley_sn_conc_batch_pct=ley_sn_conc_batch_pct,
        espesor_ladrillo_norm_mm=espesor_ladrillo_norm_mm, frac_carga_secundaria_F=frac_carga_secundaria_F,
        feed_dross_Fe_total_t=feed_dross_Fe_total_t, fecha_batch=fecha_batch,
        es_lockbox=batch_id in batches_lockbox,
    ))

batch = pd.DataFrame(filas).set_index("Batch")
batch = batch.replace([np.inf, -np.inf], np.nan)
# winsorizacion final de seguridad sobre selectividad_R (cociente Sn/FeO): recorta la
# cola residual mas alla del P1/P99 aun con el umbral de denominador minimo de arriba
if batch["selectividad_R"].notna().sum() > 10:
    lo_w, hi_w = batch["selectividad_R"].quantile([0.01, 0.99])
    n_clip = ((batch["selectividad_R"] < lo_w) | (batch["selectividad_R"] > hi_w)).sum()
    batch["selectividad_R"] = batch["selectividad_R"].clip(lo_w, hi_w)
    log(f"selectividad_R: winsorizada a [{lo_w:.2f}, {hi_w:.2f}] ({n_clip} valores recortados)")
batch.to_csv(OUT / "19_batch_dataset.csv")
log(f"dataset batch: {batch.shape} ({batch['es_lockbox'].sum()} lockbox)")

KPIS = ["f_dross", "f_polvo", "f_metal", "rendimiento_proxy_batch"]
DEC_F = ["C_por_Sn_F", "cal_por_carga_F", "gn_por_t_carga_F", "exceso_o2_F", "enriq_o2_F",
         "T_media_F", "T_final_F", "dur_F", "feed_rate_F", "lanza_media_F", "lanza_std_F"]
DEC_R = ["carbon_total_R", "carbon_por_Sn_inv_R", "gn_total_R", "exceso_o2_R", "dur_R",
         "carbon_rate_R0", "perfil_carbon_R", "T_media_R", "lanza_media_R"]
MECANISMO = ["F_avance_final", "FeO_reducido_R_t", "selectividad_R"]
CONTROLES = ["feed_sn_total_t", "ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm",
             "frac_carga_secundaria_F", "feed_dross_Fe_total_t"]
TODAS_VARS = sorted(set(DEC_F + DEC_R + MECANISMO + CONTROLES))

# =============================================================================
# 1) Estandarizacion comun (media/sd de la muestra total, reutilizada en DEV/TOTAL/LOCKBOX)
# =============================================================================
mu = batch[TODAS_VARS].mean()
sd = batch[TODAS_VARS].std()
batch_z = batch.copy()
for c in TODAS_VARS:
    batch_z[c] = (batch[c] - mu[c]) / sd[c]

dev_mask = ~batch.index.to_series().map(lambda b: b in batches_lockbox)
log(f"n total={len(batch)}, n dev={dev_mask.sum()}, n lockbox={(~dev_mask).sum()}")


def vif_filtrar(d: pd.DataFrame, vars_: list, umbral: float = 10.0):
    """Elimina iterativamente la variable con mayor VIF hasta que todas <= umbral."""
    vars_ = list(vars_)
    eliminadas = []
    while len(vars_) > 1:
        X = sm.add_constant(d[vars_].dropna())
        vifs = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
            index=vars_,
        )
        peor = vifs.idxmax()
        if vifs[peor] <= umbral:
            return vars_, eliminadas, vifs
        eliminadas.append((peor, vifs[peor]))
        vars_.remove(peor)
    X = sm.add_constant(d[vars_].dropna())
    vifs = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=vars_) if len(vars_) > 1 else pd.Series(dtype=float)
    return vars_, eliminadas, vifs


def ols_hc3(d: pd.DataFrame, dep: str, vars_: list):
    """OLS HC3. Si alguna variable es (casi) constante dentro de esta muestra concreta
    (p.ej. dur_R = 88 min fijo en las 63 batches del lockbox: cambio de protocolo, no
    señal), se excluye de ESE ajuste para evitar coeficientes espurios por colinealidad
    con la constante -- devuelve tambien la lista de variables realmente usadas."""
    cols = [dep] + vars_
    dd = d.dropna(subset=cols)
    vars_ok = [v for v in vars_ if dd[v].std() > 1e-8]
    if len(dd) < len(vars_ok) + 5 or not vars_ok:
        return None
    X = sm.add_constant(dd[vars_ok])
    y = dd[dep]
    mod = sm.OLS(y, X).fit(cov_type="HC3")
    return mod, dd, vars_ok


# =============================================================================
# 2) VIF para M3 (conjunto) por KPI -- las variables son las mismas para los 4 KPIs
#    (dependen solo de X), se calcula una vez sobre DEV
# =============================================================================
log("=== VIF (modelo conjunto M3, DEV) ===")
vars_m3, elim_m3, vifs_m3 = vif_filtrar(batch_z.loc[dev_mask], DEC_F + DEC_R + CONTROLES)
log(f"variables eliminadas por VIF>10: {elim_m3}")
log(f"VIF finales:\n{vifs_m3.round(2).to_string()}")
vif_rows = [{"variable": v, "vif": vifs_m3[v], "eliminada": False} for v in vars_m3]
vif_rows += [{"variable": v, "vif": val, "eliminada": True} for v, val in elim_m3]
pd.DataFrame(vif_rows).to_csv(OUT / "19_vif.csv", index=False)

MODELOS = {
    "M1_Fusion": DEC_F + CONTROLES,
    "M2_Reduccion": DEC_R + CONTROLES,
    "M3_Conjunto": vars_m3,
    "M4_Mecanismo": MECANISMO + CONTROLES,
}

# =============================================================================
# 3) Regresiones OLS HC3 (DEV, TOTAL, LOCKBOX) para cada KPI x modelo
# =============================================================================
log("=== Regresiones OLS HC3 (variables estandarizadas) ===")
filas_reg = []
for kpi in KPIS:
    for nombre_modelo, vars_ in MODELOS.items():
        for muestra, dd in [("dev", batch_z.loc[dev_mask]), ("total", batch_z), ("lockbox", batch_z.loc[~dev_mask])]:
            out = ols_hc3(dd, kpi, vars_)
            if out is None:
                continue
            mod, ddd, vars_ok = out
            excluidas = set(vars_) - set(vars_ok)
            if excluidas:
                log(f"  [{kpi}/{nombre_modelo}/{muestra}] variable(s) sin varianza excluida(s): {excluidas}")
            ci = mod.conf_int()
            for v in vars_ok:
                filas_reg.append(dict(
                    kpi=kpi, modelo=nombre_modelo, muestra=muestra, variable=v,
                    coef_por_sd=mod.params[v], ci_lo=ci.loc[v, 0], ci_hi=ci.loc[v, 1],
                    p=mod.pvalues[v], r2_adj=mod.rsquared_adj, n=int(mod.nobs),
                ))
    log(f"  {kpi}: listo")

tabla_reg = pd.DataFrame(filas_reg)


def marca_sig(p):
    return p < 0.05


tabla_reg["significativa"] = tabla_reg["p"].apply(marca_sig)

# consistencia de signo dev/total/lockbox (solo si dev es significativa; el lockbox no
# necesita serlo con n=63, solo mismo signo)
piv_signo = tabla_reg.pivot_table(index=["kpi", "modelo", "variable"], columns="muestra",
                                   values="coef_por_sd", aggfunc="first")


def es_consistente(row):
    vals = [row.get(m) for m in ["dev", "total", "lockbox"]]
    if any(v is None or pd.isna(v) for v in vals):
        return False
    signos = [np.sign(v) for v in vals]
    return len(set(signos)) == 1 and signos[0] != 0


piv_signo["consistente"] = piv_signo.apply(es_consistente, axis=1)
tabla_reg = tabla_reg.merge(
    piv_signo["consistente"].reset_index(), on=["kpi", "modelo", "variable"], how="left"
)
tabla_reg.to_csv(OUT / "19_regresiones_kpi.csv", index=False)
log(f"guardado 19_regresiones_kpi.csv ({len(tabla_reg)} filas)")

# =============================================================================
# 4) HGB pequeno + OOF (KFold 5, DEV) + permutation importance (lockbox) + PD
# =============================================================================
log("=== HGB no parametrico (OOF + permutation importance + PD) ===")
DEC_TODAS = DEC_F + DEC_R
filas_hgb = []
filas_pd = []
top5_por_kpi = {}

for kpi in KPIS:
    feats = DEC_TODAS + CONTROLES
    d = batch.dropna(subset=[kpi] + feats)
    dm = d.index.to_series().map(lambda b: b in batches_dev)
    X_dev, y_dev = d.loc[dm, feats], d.loc[dm, kpi]
    X_lb, y_lb = d.loc[~dm, feats], d.loc[~dm, kpi]
    med = X_dev.median()

    def mk():
        return HistGradientBoostingRegressor(max_depth=2, max_iter=150, min_samples_leaf=20, random_state=0)

    oof = np.full(len(X_dev), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X_dev):
        m = mk()
        m.fit(X_dev.iloc[tr].fillna(med), y_dev.iloc[tr])
        oof[te] = m.predict(X_dev.iloc[te].fillna(med))
    r2_oof = 1 - np.nansum((y_dev.to_numpy() - oof) ** 2) / np.nansum((y_dev.to_numpy() - y_dev.mean()) ** 2)

    modelo_dev = mk().fit(X_dev.fillna(med), y_dev)
    r2_lb = modelo_dev.score(X_lb.fillna(med), y_lb) if len(X_lb) else np.nan

    pi = permutation_importance(modelo_dev, X_lb.fillna(med), y_lb, n_repeats=5, random_state=0, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    imp_std = pd.Series(pi.importances_std, index=feats)
    for f in feats:
        filas_hgb.append(dict(kpi=kpi, variable=f, importancia_perm_lockbox=imp[f], importancia_perm_std=imp_std[f],
                               r2_oof_dev=r2_oof, r2_lockbox=r2_lb, n_dev=len(X_dev), n_lockbox=len(X_lb)))
    # PD/optimo interior solo sobre DECISIONES (no controles), como pide el enunciado
    imp_decisiones = imp.loc[[f for f in DEC_TODAS if f in imp.index]].sort_values(ascending=False)
    top5 = list(imp_decisiones.head(5).index)
    top5_por_kpi[kpi] = top5
    log(f"  {kpi}: R2 OOF(dev)={r2_oof:.3f} R2(lockbox)={r2_lb:.3f} top5 perm-imp (todas)={list(imp.head(5).index)} top5 decisiones={top5}")

    # PD 1-D de las top5 decisiones (fit en DEV completo)
    for f in top5:
        idx = feats.index(f)
        pdres = partial_dependence(modelo_dev, X_dev.fillna(med), [idx], kind="average", grid_resolution=20)
        grid = pdres["grid_values"][0]
        vals = pdres["average"][0]
        for gv, av in zip(grid, vals):
            filas_pd.append(dict(kpi=kpi, variable=f, grid_value=gv, pd_value=av))

tabla_hgb = pd.DataFrame(filas_hgb)
tabla_hgb.to_csv(OUT / "19_hgb_resumen.csv", index=False)
tabla_pd = pd.DataFrame(filas_pd)
tabla_pd.to_csv(OUT / "19_pd_kpi.csv", index=False)
log("guardado 19_hgb_resumen.csv y 19_pd_kpi.csv")


# =============================================================================
# 5) Deteccion de optimo interior en las curvas PD -> termino cuadratico en OLS
# =============================================================================
def tiene_optimo_interior(grid: np.ndarray, vals: np.ndarray, margen: float = 0.15, salto_min: float = 0.20) -> bool:
    """Heuristica estricta: el extremo (max o min) de la curva PD debe (a) caer en el
    interior del rango (no en el borde) y (b) superar a AMBOS extremos por al menos
    `salto_min` * rango-total (evita marcar como "optimo interior" simple ruido/monotonia
    casi plana de un modelo con R2 OOF bajo/negativo)."""
    if len(grid) < 6:
        return False
    n = len(grid)
    lo, hi = int(n * margen), int(n * (1 - margen))
    rango = vals.max() - vals.min()
    if rango <= 0:
        return False
    extremo0, extremo1 = vals[0], vals[-1]
    i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
    interior_max = (lo < i_max < hi) and (vals[i_max] - max(extremo0, extremo1)) / rango > salto_min
    interior_min = (lo < i_min < hi) and (min(extremo0, extremo1) - vals[i_min]) / rango > salto_min
    return bool(interior_max or interior_min)


log("=== Deteccion de optimo interior (PD) y terminos cuadraticos ===")
optimos = {}
for kpi in KPIS:
    for f in top5_por_kpi.get(kpi, []):
        sub = tabla_pd.loc[(tabla_pd.kpi == kpi) & (tabla_pd.variable == f)].sort_values("grid_value")
        if sub.empty:
            continue
        flag = tiene_optimo_interior(sub["grid_value"].to_numpy(), sub["pd_value"].to_numpy())
        optimos[(kpi, f)] = flag
        if flag:
            log(f"  optimo interior sugerido: {kpi} ~ {f}")

filas_cuad = []
for (kpi, var), flag in optimos.items():
    if not flag:
        continue
    modelos_aplicables = []
    if var in DEC_F:
        modelos_aplicables += ["M1_Fusion", "M3_Conjunto"]
    if var in DEC_R:
        modelos_aplicables += ["M2_Reduccion", "M3_Conjunto"]
    for nombre_modelo in set(modelos_aplicables):
        vars_ = MODELOS[nombre_modelo]
        if var not in vars_:
            continue
        for muestra, dd in [("dev", batch_z.loc[dev_mask]), ("total", batch_z), ("lockbox", batch_z.loc[~dev_mask])]:
            dd2 = dd.copy()
            dd2[f"{var}_sq"] = dd2[var] ** 2
            out = ols_hc3(dd2, kpi, vars_ + [f"{var}_sq"])
            if out is None:
                continue
            mod, ddd, vars_ok = out
            if f"{var}_sq" not in vars_ok:
                continue
            ci = mod.conf_int()
            filas_cuad.append(dict(
                kpi=kpi, modelo=f"{nombre_modelo}+cuad({var})", muestra=muestra, variable=f"{var}_sq",
                coef_por_sd=mod.params[f"{var}_sq"], ci_lo=ci.loc[f"{var}_sq", 0], ci_hi=ci.loc[f"{var}_sq", 1],
                p=mod.pvalues[f"{var}_sq"], r2_adj=mod.rsquared_adj, n=int(mod.nobs),
            ))
tabla_cuad = pd.DataFrame(filas_cuad)
if not tabla_cuad.empty:
    tabla_cuad["significativa"] = tabla_cuad["p"] < 0.05
    tabla_cuad["consistente"] = np.nan
    tabla_cuad.to_csv(OUT / "19_regresiones_cuadraticas.csv", index=False)
    log(f"guardado 19_regresiones_cuadraticas.csv ({len(tabla_cuad)} filas)")
else:
    log("ninguna variable con optimo interior detectado en el top5 de permutation importance")

# =============================================================================
# 6) Figuras
# =============================================================================
log("=== Figuras ===")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kpi in KPIS:
        sub = tabla_reg.loc[(tabla_reg.kpi == kpi) & (tabla_reg.modelo == "M3_Conjunto") & (tabla_reg.muestra == "dev")]
        sub = sub.sort_values("coef_por_sd")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(sub))))
        y = np.arange(len(sub))
        colores = ["#DD8452" if s else "#8899AA" for s in sub["significativa"]]
        ax.errorbar(sub["coef_por_sd"], y, xerr=[sub["coef_por_sd"] - sub["ci_lo"], sub["ci_hi"] - sub["coef_por_sd"]],
                    fmt="o", color="black", ecolor="gray", capsize=3, zorder=2)
        ax.scatter(sub["coef_por_sd"], y, c=colores, zorder=3, s=40)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["variable"])
        ax.set_xlabel(f"coeficiente por +1 sd (M3, DEV)")
        ax.set_title(f"Coeficientes estandarizados M3 -- {kpi}")
        fig.tight_layout()
        fig.savefig(FIGS / f"coef_M3_{kpi}.png", dpi=130)
        plt.close(fig)
    log("figuras de coeficientes guardadas")

    for kpi in KPIS:
        vars_top = top5_por_kpi.get(kpi, [])[:3]
        if not vars_top:
            continue
        fig, axes = plt.subplots(1, len(vars_top), figsize=(4.5 * len(vars_top), 3.6))
        if len(vars_top) == 1:
            axes = [axes]
        for ax, v in zip(axes, vars_top):
            sub = tabla_pd.loc[(tabla_pd.kpi == kpi) & (tabla_pd.variable == v)].sort_values("grid_value")
            ax.plot(sub["grid_value"], sub["pd_value"], color="#4C72B0")
            ax.set_xlabel(v)
            ax.set_ylabel(kpi)
            ax.set_title(f"PD: {v}")
        fig.tight_layout()
        fig.savefig(FIGS / f"pd_{kpi}.png", dpi=130)
        plt.close(fig)
    log("figuras PD guardadas")
except Exception as e:
    log("figuras fallaron:", repr(e))

log("=== FIN experimento 19 ===")
log(f"tiempo total: {time.time() - T0:.1f}s")
