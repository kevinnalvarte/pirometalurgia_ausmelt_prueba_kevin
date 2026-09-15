"""E7-01 -- Reduccion: componentes logaritmicas del SDI, parametrizacion de las
palancas y anclaje del peso w en el KPI refinado (iteracion 7, ver
experimentos/v5/ITERACION_7_diseno.md).

Salidas (experimentos/v5/):
  e7_01_plm_comparacion.csv  -- una fila por (target, estado, parametrizacion)
  e7_01_theta.csv            -- una fila por (target, estado, parametrizacion, palanca)
  e7_01_w_anclaje.csv        -- OLS HC3 batch, w = coef(feo)/coef(sn), IC bootstrap
  e7_01_w_sweep.csv          -- barrido de w y variantes temporales, rho Spearman
  e7_01_cadena_signos.csv    -- palanca -> signo sobre ln_sn_dep / ln_feo_ret / J
  e7_01_resultados.md        -- memo
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import v5_lib as v5

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ============================================================================
# 0) Carga
# ============================================================================
log("cargando datos...")
df = v5.cargar_df()
dR = v5.base_fase(df, "Reducción")
batch = v5.construir_batch_v5(df)
log(f"df {df.shape}  dR {dR.shape}  batch {batch.shape}")

# ============================================================================
# 1) Parametrizaciones de palancas (tarea 2)
# ============================================================================
PALANCAS_POR_PARAM = {
    "P1_primitivas": {
        "Sn": v5.PALANCAS_PRIMITIVAS, "FeO": v5.PALANCAS_PRIMITIVAS, "SDI": v5.PALANCAS_PRIMITIVAS,
    },
    "P2_cinetica_v4": {
        "Sn": ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "FeO": ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "SDI": ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    },
    "P3_dosis_sobrante": {
        "Sn": ["v5_dosis_C_sn", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "FeO": ["v5_exceso_C_pos", "v5_C_x_avance", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "SDI": ["v5_exceso_C_pos", "v5_C_x_avance", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    },
    "P4_log_dosis": {
        "Sn": ["v5_ln_dosis_C", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "FeO": ["v5_ln_dosis_C", "v5_exceso_C_pos", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "SDI": ["v5_ln_dosis_C", "v5_exceso_C_pos", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    },
    "P5_minima": {
        "Sn": ["v5_dosis_C_sn", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
        "FeO": ["v5_exceso_C_pos", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
    },
}

TARGET_GRUPO = {
    "v5_ln_sn_dep": "Sn", "sn_extraido_est_kg": "Sn",
    "v5_ln_feo_ret": "FeO", "feo_extraido_est_kg": "FeO",
    "v5_sdi_w10": "SDI",
}

ESTADOS = {"FULL": v5.ESTADO_R_FULL, "CURADO": v5.ESTADO_R_CURADO}
N_BOOT_ESTADO = {"FULL": 100, "CURADO": 300}

# ============================================================================
# 2) Comparacion de PLMs
# ============================================================================
log("=== tarea 2: comparacion de PLMs ===")
filas_met, tablas_theta = [], []
for target, grupo in TARGET_GRUPO.items():
    signo = v5.SIGNO_TEORICO_V5.get(target, {})
    for estado_nombre, estado in ESTADOS.items():
        for param_nombre, mapping in PALANCAS_POR_PARAM.items():
            if grupo not in mapping:
                continue
            palancas = mapping[grupo]
            log(f"{target:22s} {estado_nombre:6s} {param_nombre:16s} palancas={palancas}")
            try:
                met, tabla, _pred = v5.evaluar_plm(
                    dR, estado, palancas, target,
                    n_boot=N_BOOT_ESTADO[estado_nombre], signo_teorico=signo,
                )
            except Exception as exc:  # noqa: BLE001
                log("  ERROR:", repr(exc))
                continue
            met.update(estado=estado_nombre, parametrizacion=param_nombre, grupo=grupo)
            filas_met.append(met)
            tabla = tabla.reset_index()
            tabla.insert(1, "estado", estado_nombre)
            tabla.insert(2, "parametrizacion", param_nombre)
            tabla.insert(3, "grupo", grupo)
            tablas_theta.append(tabla)
            log(f"  R2_oof={met['r2_oof']:.3f}  R2_lockbox={met['r2_lockbox']:.3f}  n_dev={met['n_dev']} n_lb={met['n_lockbox']}")

df_met = pd.DataFrame(filas_met)
df_theta = pd.concat(tablas_theta, ignore_index=True)
df_met.to_csv(OUT / "e7_01_plm_comparacion.csv", index=False)
df_theta.to_csv(OUT / "e7_01_theta.csv", index=False)
log(f"guardado e7_01_plm_comparacion.csv ({len(df_met)} filas) y e7_01_theta.csv ({len(df_theta)} filas)")


def elegir_parametrizacion(df_met: pd.DataFrame, df_theta: pd.DataFrame, target: str, estado_nombre: str = "CURADO") -> dict:
    """Criterio de eleccion (solo DEV): mayor R2 OOF; empate +/-0.01 resuelto por
    (a) mas palancas con IC que no cruza 0 y signo teorico correcto, (b) menos palancas."""
    sub = df_met[(df_met["target"] == target) & (df_met["estado"] == estado_nombre)].copy()
    if sub.empty:
        return {}
    r2max = sub["r2_oof"].max()
    cand = sub[sub["r2_oof"] >= r2max - 0.01].copy()
    n_ok = []
    for _, row in cand.iterrows():
        th = df_theta[(df_theta["target"] == target) & (df_theta["estado"] == estado_nombre) &
                      (df_theta["parametrizacion"] == row["parametrizacion"])]
        if "concuerda" in th.columns:
            ok = int((th["significativo"] & (th["concuerda"] == True)).sum())  # noqa: E712
        else:
            ok = int(th["significativo"].sum())
        n_ok.append(ok)
    cand["n_sig_concuerda"] = n_ok
    cand = cand.sort_values(["n_sig_concuerda", "n_palancas"], ascending=[False, True])
    ganador = cand.iloc[0].to_dict()
    return ganador


elegidos = {}
for target in ["v5_ln_sn_dep", "v5_ln_feo_ret", "v5_sdi_w10"]:
    g = elegir_parametrizacion(df_met, df_theta, target, "CURADO")
    elegidos[target] = g
    if g:
        log(f"elegida para {target} (CURADO): {g['parametrizacion']}  R2_oof={g['r2_oof']:.3f}  R2_lockbox={g.get('r2_lockbox', np.nan):.3f}")

# ============================================================================
# 3) Anclaje de w en el KPI (nivel batch)
# ============================================================================
log("=== tarea 3: anclaje de w ===")
CONTROLES = v5.CONTROLES_BATCH


def ols_w(dfin: pd.DataFrame, kpi_col: str, n_boot: int = 1000, seed: int = 0) -> dict:
    d = dfin[["R_sum_ln_sn_dep", "R_sum_ln_feo_ret", kpi_col] + CONTROLES].dropna()
    if len(d) < 30:
        return {}
    X = sm.add_constant(d[["R_sum_ln_sn_dep", "R_sum_ln_feo_ret"] + CONTROLES])
    res = sm.OLS(d[kpi_col], X).fit(cov_type="HC3")
    b_sn, b_feo = res.params["R_sum_ln_sn_dep"], res.params["R_sum_ln_feo_ret"]
    out = {
        "kpi": kpi_col, "n": len(d),
        "coef_sn": b_sn, "p_sn": res.pvalues["R_sum_ln_sn_dep"],
        "coef_feo": b_feo, "p_feo": res.pvalues["R_sum_ln_feo_ret"],
        "w_puntual": b_feo / b_sn if b_sn != 0 else np.nan,
        "r2adj": res.rsquared_adj,
    }
    rng = np.random.default_rng(seed)
    batches = d.index.to_numpy()
    ws, fracs = [], []
    for _ in range(n_boot):
        samp = rng.choice(batches, size=len(batches), replace=True)
        dd = d.loc[samp]
        Xb = sm.add_constant(dd[["R_sum_ln_sn_dep", "R_sum_ln_feo_ret"] + CONTROLES].reset_index(drop=True))
        try:
            rb = sm.OLS(dd[kpi_col].reset_index(drop=True), Xb).fit()
        except Exception:  # noqa: BLE001
            continue
        bs, bf = rb.params["R_sum_ln_sn_dep"], rb.params["R_sum_ln_feo_ret"]
        fracs.append(float(bs > 0 and bf > 0))
        if bs != 0:
            ws.append(bf / bs)
    ws = np.array(ws)
    ws = ws[np.isfinite(ws)]
    out["w_ci_lo"] = float(np.percentile(ws, 2.5)) if len(ws) else np.nan
    out["w_ci_med"] = float(np.percentile(ws, 50)) if len(ws) else np.nan
    out["w_ci_hi"] = float(np.percentile(ws, 97.5)) if len(ws) else np.nan
    out["frac_ambos_positivos"] = float(np.mean(fracs)) if fracs else np.nan
    return out


batch_dev = batch[batch["es_dev"]].copy()
batch_dev["recuperacion_refinada_pct_mean_cur_next"] = batch_dev[["recuperacion_refinada_pct", "recuperacion_refinada_pct_next"]].mean(axis=1)
batch_dev["rendimiento_proxy_batch_mean_cur_next"] = batch_dev[["rendimiento_proxy_batch", "rendimiento_proxy_batch_next"]].mean(axis=1)
batch_dev["f_dross_mean_cur_next"] = batch_dev[["f_dross", "f_dross_next"]].mean(axis=1)
batch_dev["recuperacion_real_pct_mean_cur_next"] = batch_dev[["recuperacion_real_pct", "recuperacion_real_pct_next"]].mean(axis=1)

filas_w = []
_seed_ctr = 0
for outcome in ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "f_dross", "recuperacion_real_pct"]:
    for variante, col in [("actual", outcome), ("next", f"{outcome}_next"), ("mean_cur_next", f"{outcome}_mean_cur_next")]:
        _seed_ctr += 1
        r = ols_w(batch_dev, col, n_boot=1000, seed=100 + _seed_ctr)
        if r:
            r.update(outcome=outcome, variante=variante)
            filas_w.append(r)
            log(f"w_anclaje {outcome:26s} {variante:14s} n={r['n']:3d} coef_sn={r['coef_sn']:.3f}(p{r['p_sn']:.3f}) "
                f"coef_feo={r['coef_feo']:.3f}(p{r['p_feo']:.3f}) w={r['w_puntual']:.2f} "
                f"[{r['w_ci_lo']:.2f},{r['w_ci_hi']:.2f}] frac++={r['frac_ambos_positivos']:.2f}")

df_w = pd.DataFrame(filas_w)
df_w.to_csv(OUT / "e7_01_w_anclaje.csv", index=False)
log(f"guardado e7_01_w_anclaje.csv ({len(df_w)} filas)")

w_kpi_principal = df_w[(df_w["outcome"] == "recuperacion_refinada_pct") & (df_w["variante"] == "actual")]
w_anchor = float(w_kpi_principal["w_puntual"].iloc[0]) if len(w_kpi_principal) else v5.W_SDI_ITER5
w_anchor_pos = max(w_anchor, 0.0)
log(f"w anclado (KPI actual) = {w_anchor:.3f}  (usado en tareas 4-5 como w_anchor, clip>=0 -> {w_anchor_pos:.3f})")

# ============================================================================
# 3b) Barrido de w
# ============================================================================
log("=== tarea 3: barrido de w ===")
W_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 30]


def spearman_ic(x: np.ndarray, y: np.ndarray, n_boot: int = 500, seed: int = 0):
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    idx_bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        idx_bs.append(stats.spearmanr(x[idx], y[idx])[0])
    return float(rho), float(p), float(np.nanpercentile(idx_bs, 2.5)), float(np.nanpercentile(idx_bs, 97.5))


filas_sweep = []


def sweep_variable(nombre_variante: str, col_feo: str, kpi: str = v5.KPI_PRINCIPAL):
    for w in W_GRID:
        j = batch["R_sum_ln_sn_dep"] + w * batch[col_feo]
        for split_nombre, mask in [("dev", batch["es_dev"]), ("lockbox", batch["es_lockbox"]), ("total", pd.Series(True, index=batch.index))]:
            d = pd.concat([j[mask], batch.loc[mask, kpi]], axis=1).dropna()
            if len(d) < 15:
                continue
            rho, p, lo, hi = spearman_ic(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy(), n_boot=400, seed=int(w * 7 + 1))
            filas_sweep.append(dict(variante=nombre_variante, w=w, split=split_nombre, kpi=kpi, n=len(d),
                                     rho=rho, p=p, ci_lo=lo, ci_hi=hi))


sweep_variable("simple", "R_sum_ln_feo_ret")
log("barrido w (variante simple) listo")

# ============================================================================
# 4) Variante ponderada por tiempo restante
# ============================================================================
log("=== tarea 4: variante ponderada por tiempo restante ===")
dfR_full = df[df["fase_proceso"] == "Reducción"].sort_values(["Batch", "orden_escalon_fase"]).copy()

pesos = []
for b, g in dfR_full.groupby("Batch", sort=False):
    g = g.sort_values("orden_escalon_fase")
    n_r = len(g)
    orden = g["orden_escalon_fase"].to_numpy()
    peso_orden = (n_r - orden) / n_r if n_r > 0 else np.full(len(g), np.nan)
    dt = g["delta_tiempo"].to_numpy(dtype=float)
    dt_fill = np.nan_to_num(dt, nan=0.0)
    rev_cum = np.cumsum(dt_fill[::-1])[::-1]
    total_t = dt_fill.sum()
    peso_tiempo = rev_cum / total_t if total_t > 0 else np.full(len(g), np.nan)
    pesos.append(pd.DataFrame({"Batch": b, "fecha_inicio": g["fecha_inicio"].to_numpy(),
                                "peso_orden_restante": peso_orden, "peso_tiempo_restante": peso_tiempo}))
df_pesos = pd.concat(pesos, ignore_index=True)
dfR_full = dfR_full.merge(df_pesos, on=["Batch", "fecha_inicio"], how="left")
dfR_full["v5_ln_feo_ret_wt_orden"] = dfR_full["v5_ln_feo_ret"] * dfR_full["peso_orden_restante"]
dfR_full["v5_ln_feo_ret_wt_tiempo"] = dfR_full["v5_ln_feo_ret"] * dfR_full["peso_tiempo_restante"]

agg = dfR_full.groupby("Batch").agg(
    R_sum_ln_feo_ret_wt_orden=("v5_ln_feo_ret_wt_orden", lambda s: s.sum(min_count=1)),
    R_sum_ln_feo_ret_wt_tiempo=("v5_ln_feo_ret_wt_tiempo", lambda s: s.sum(min_count=1)),
)
batch = batch.join(agg, how="left")
log("pesos temporales agregados a batch: R_sum_ln_feo_ret_wt_orden, R_sum_ln_feo_ret_wt_tiempo")

sweep_variable("tiempo_orden", "R_sum_ln_feo_ret_wt_orden")
sweep_variable("tiempo_real", "R_sum_ln_feo_ret_wt_tiempo")
log("barrido w (variantes temporales) listo")

df_sweep = pd.DataFrame(filas_sweep)
df_sweep.to_csv(OUT / "e7_01_w_sweep.csv", index=False)
log(f"guardado e7_01_w_sweep.csv ({len(df_sweep)} filas)")


def regla_parsimonia(sub: pd.DataFrame) -> float:
    """Menor w con rho_DEV >= 0.95 * max(rho_DEV)."""
    d = sub[sub["split"] == "dev"].copy()
    if d.empty:
        return np.nan
    rmax = d["rho"].max()
    ok = d[d["rho"] >= 0.95 * rmax]
    return float(ok["w"].min()) if len(ok) else np.nan


w_parsimonia = {}
for variante in ["simple", "tiempo_orden", "tiempo_real"]:
    sub = df_sweep[df_sweep["variante"] == variante]
    w_parsimonia[variante] = regla_parsimonia(sub)
    log(f"w parsimonia ({variante}) = {w_parsimonia[variante]}")

# comparacion directa de rho en w=10 y w=w_anchor por variante (dev/lockbox/total)
comparacion_variantes = []
for variante, col_feo in [("simple", "R_sum_ln_feo_ret"), ("tiempo_orden", "R_sum_ln_feo_ret_wt_orden"), ("tiempo_real", "R_sum_ln_feo_ret_wt_tiempo")]:
    for w_nombre, w_val in [("w10", 10.0), ("w_anchor", w_anchor_pos)]:
        j = batch["R_sum_ln_sn_dep"] + w_val * batch[col_feo]
        for split_nombre, mask in [("dev", batch["es_dev"]), ("lockbox", batch["es_lockbox"]), ("total", pd.Series(True, index=batch.index))]:
            d = pd.concat([j[mask], batch.loc[mask, v5.KPI_PRINCIPAL]], axis=1).dropna()
            if len(d) < 15:
                continue
            rho, p, lo, hi = spearman_ic(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy(), n_boot=400, seed=1)
            comparacion_variantes.append(dict(variante=variante, w_nombre=w_nombre, w=w_val, split=split_nombre,
                                               n=len(d), rho=rho, p=p, ci_lo=lo, ci_hi=hi))
df_comp_variantes = pd.DataFrame(comparacion_variantes)
df_comp_variantes.to_csv(OUT / "e7_01_w_sweep_variantes_resumen.csv", index=False)
log("guardado e7_01_w_sweep_variantes_resumen.csv")

# ============================================================================
# 5) Cadena de signos
# ============================================================================
log("=== tarea 5: cadena de signos ===")


def theta_de(target: str, estado_nombre: str, param_nombre: str) -> pd.DataFrame:
    return df_theta[(df_theta["target"] == target) & (df_theta["estado"] == estado_nombre) &
                     (df_theta["parametrizacion"] == param_nombre)].set_index("palanca")


param_sn = elegidos.get("v5_ln_sn_dep", {}).get("parametrizacion")
param_feo = elegidos.get("v5_ln_feo_ret", {}).get("parametrizacion")
param_sdi = elegidos.get("v5_sdi_w10", {}).get("parametrizacion")
log(f"parametrizacion elegida -> Sn: {param_sn}  FeO: {param_feo}  SDI: {param_sdi}")

th_sn = theta_de("v5_ln_sn_dep", "CURADO", param_sn) if param_sn else pd.DataFrame()
th_feo = theta_de("v5_ln_feo_ret", "CURADO", param_feo) if param_feo else pd.DataFrame()

palancas_union = sorted(set(th_sn.index) | set(th_feo.index))
filas_cadena = []
for p in palancas_union:
    theta_sn = float(th_sn.loc[p, "theta_unidad"]) if p in th_sn.index else np.nan
    theta_feo = float(th_feo.loc[p, "theta_unidad"]) if p in th_feo.index else np.nan
    sig_sn = float(th_sn.loc[p, "theta_unidad"]) if p in th_sn.index else 0.0
    sig_feo = float(th_feo.loc[p, "theta_unidad"]) if p in th_feo.index else 0.0
    sd_sn = float(th_sn.loc[p, "sd_palanca"]) if p in th_sn.index else np.nan
    sd_feo = float(th_feo.loc[p, "sd_palanca"]) if p in th_feo.index else np.nan
    sd_p = np.nanmean([sd_sn, sd_feo]) if not (np.isnan(sd_sn) and np.isnan(sd_feo)) else np.nan
    j_w10 = sig_sn + 10.0 * sig_feo
    j_wanchor = sig_sn + w_anchor_pos * sig_feo
    signo_evid_batch = {
        "tasa_feed_Carbon_kg_min": "batch: -f_dross p0.03 (carbon reduce dross)",
        "tasa_gn_nm3_min": "batch: +f_dross p0.04 / -f_metal p0.03 (GN sube dross)",
        "tasa_aire_nm3_min": "batch: lanza no es palanca (exp19)",
        "tasa_o2_nm3_min": "batch: lanza no es palanca (exp19)",
    }.get(p, "")
    filas_cadena.append(dict(
        palanca=p,
        theta_ln_sn_dep=theta_sn, sig_ln_sn_dep=("+" if theta_sn > 0 else "-" if theta_sn < 0 else "0") if p in th_sn.index else "n/a",
        theta_ln_feo_ret=theta_feo, sig_ln_feo_ret=("+" if theta_feo > 0 else "-" if theta_feo < 0 else "0") if p in th_feo.index else "n/a",
        sd_palanca_aprox=sd_p,
        theta_J_w10_por_unidad=j_w10, theta_J_wanchor_por_unidad=j_wanchor,
        theta_J_w10_por_sd=j_w10 * sd_p if not np.isnan(sd_p) else np.nan,
        theta_J_wanchor_por_sd=j_wanchor * sd_p if not np.isnan(sd_p) else np.nan,
        signo_neto_w10=("+" if j_w10 > 0 else "-" if j_w10 < 0 else "0"),
        signo_neto_wanchor=("+" if j_wanchor > 0 else "-" if j_wanchor < 0 else "0"),
        evidencia_batch_exp19=signo_evid_batch,
    ))
df_cadena = pd.DataFrame(filas_cadena)
df_cadena.to_csv(OUT / "e7_01_cadena_signos.csv", index=False)
log(f"guardado e7_01_cadena_signos.csv ({len(df_cadena)} filas)")

log(f"TOTAL {time.time() - T0:.1f}s")

# ============================================================================
# 6) Memo
# ============================================================================
log("=== escribiendo memo ===")


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa: BLE001
        return "NA"


memo = []
memo.append("# E7-01 -- Reduccion: componentes log del SDI, parametrizacion de palancas y anclaje de w\n")
memo.append(f"Generado {time.strftime('%Y-%m-%d %H:%M')}. dR (escalones modelables Reduccion) = {dR.shape[0]} filas; "
            f"batch = {batch.shape[0]} filas ({batch['es_dev'].sum()} DEV / {batch['es_lockbox'].sum()} lockbox).\n")

memo.append("## 1. Comparacion de PLMs (R2 OOF DEV / R2 lockbox), estado CURADO\n")
memo.append("| target | parametrizacion | n_palancas | R2_oof | MAE_oof | R2_lockbox | n_dev | n_lockbox |")
memo.append("|---|---|---:|---:|---:|---:|---:|---:|")
for _, r in df_met[df_met["estado"] == "CURADO"].sort_values(["target", "parametrizacion"]).iterrows():
    memo.append(f"| {r['target']} | {r['parametrizacion']} | {r['n_palancas']} | {fmt(r['r2_oof'])} | {fmt(r['mae_oof'])} | "
                f"{fmt(r['r2_lockbox'])} | {r['n_dev']} | {r['n_lockbox']} |")

memo.append("\n## 1b. Mismo, estado FULL\n")
memo.append("| target | parametrizacion | n_palancas | R2_oof | MAE_oof | R2_lockbox | n_dev | n_lockbox |")
memo.append("|---|---|---:|---:|---:|---:|---:|---:|")
for _, r in df_met[df_met["estado"] == "FULL"].sort_values(["target", "parametrizacion"]).iterrows():
    memo.append(f"| {r['target']} | {r['parametrizacion']} | {r['n_palancas']} | {fmt(r['r2_oof'])} | {fmt(r['mae_oof'])} | "
                f"{fmt(r['r2_lockbox'])} | {r['n_dev']} | {r['n_lockbox']} |")

memo.append("\n## 2. Parametrizacion elegida (criterio: solo DEV, estado CURADO)\n")
for target, g in elegidos.items():
    if not g:
        memo.append(f"- {target}: sin resultado")
        continue
    memo.append(f"- **{target}** -> {g['parametrizacion']} (R2_oof={fmt(g['r2_oof'])}, R2_lockbox={fmt(g.get('r2_lockbox'))}, "
                f"n_palancas={g['n_palancas']}, n_sig_concuerda={g.get('n_sig_concuerda', 'NA')})")
    th = df_theta[(df_theta["target"] == target) & (df_theta["estado"] == "CURADO") & (df_theta["parametrizacion"] == g["parametrizacion"])]
    memo.append("")
    memo.append("  | palanca | theta_unidad | IC | theta_1sd | IC_1sd | signif | signo_teorico | concuerda |")
    memo.append("  |---|---:|---|---:|---|---|---:|---|")
    for _, r in th.iterrows():
        memo.append(f"  | {r['palanca']} | {fmt(r['theta_unidad'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
                    f"{fmt(r['theta_1sd'])} | [{fmt(r['ci_lo_1sd'])}, {fmt(r['ci_hi_1sd'])}] | {r['significativo']} | "
                    f"{r.get('signo_teorico', 'NA')} | {r.get('concuerda', 'NA')} |")
    memo.append("")

memo.append("## 3. Anclaje de w en el KPI (OLS HC3, batch DEV, unidades naturales)\n")
memo.append("| outcome | variante | n | coef_sn(p) | coef_feo(p) | w puntual | w IC95% boot | frac(sn>0 & feo>0) | R2adj |")
memo.append("|---|---|---:|---|---|---:|---|---:|---:|")
for _, r in df_w.iterrows():
    memo.append(f"| {r['outcome']} | {r['variante']} | {r['n']} | {fmt(r['coef_sn'])}({fmt(r['p_sn'])}) | "
                f"{fmt(r['coef_feo'])}({fmt(r['p_feo'])}) | {fmt(r['w_puntual'], 2)} | "
                f"[{fmt(r['w_ci_lo'], 2)}, {fmt(r['w_ci_hi'], 2)}] (mediana {fmt(r['w_ci_med'], 2)}) | "
                f"{fmt(r['frac_ambos_positivos'], 2)} | {fmt(r['r2adj'])} |")

memo.append(f"\nw anclado usado en tareas 4-5 (KPI actual, recuperacion_refinada_pct): **{fmt(w_anchor, 2)}** "
            f"(clip a >=0 -> {fmt(w_anchor_pos, 2)}).\n")

memo.append("## 3b. Barrido de w (rho Spearman con KPI, variante simple)\n")
memo.append("| w | rho_dev | IC_dev | rho_lockbox | rho_total |")
memo.append("|---:|---:|---|---:|---:|")
piv = df_sweep[df_sweep["variante"] == "simple"]
for w in W_GRID:
    d_dev = piv[(piv["w"] == w) & (piv["split"] == "dev")]
    d_lb = piv[(piv["w"] == w) & (piv["split"] == "lockbox")]
    d_tot = piv[(piv["w"] == w) & (piv["split"] == "total")]
    if d_dev.empty:
        continue
    rd = d_dev.iloc[0]
    rl = d_lb.iloc[0] if len(d_lb) else None
    rt = d_tot.iloc[0] if len(d_tot) else None
    memo.append(f"| {w} | {fmt(rd['rho'])} | [{fmt(rd['ci_lo'])}, {fmt(rd['ci_hi'])}] | "
                f"{fmt(rl['rho']) if rl is not None else 'NA'} | {fmt(rt['rho']) if rt is not None else 'NA'} |")
memo.append(f"\nw por regla de parsimonia (min w con rho_DEV >= 0.95*max), variante simple: **{w_parsimonia['simple']}**.\n")

memo.append("## 4. Variante ponderada por tiempo restante\n")
memo.append("| variante | w usado | split | n | rho | IC |")
memo.append("|---|---|---|---:|---:|---|")
for _, r in df_comp_variantes.iterrows():
    memo.append(f"| {r['variante']} | {r['w_nombre']}={fmt(r['w'],2)} | {r['split']} | {r['n']} | {fmt(r['rho'])} | "
                f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] |")
memo.append(f"\nw parsimonia por variante: simple={w_parsimonia['simple']}, tiempo_orden={w_parsimonia['tiempo_orden']}, "
            f"tiempo_real={w_parsimonia['tiempo_real']}.\n")

memo.append("## 5. Cadena de signos (parametrizacion elegida: Sn={} / FeO={})\n".format(param_sn, param_feo))
memo.append("| palanca | signo ln_sn_dep | signo ln_feo_ret | J w10 (unidad/sd) | J w_anchor (unidad/sd) | evidencia batch exp19 |")
memo.append("|---|---|---|---|---|---|")
for _, r in df_cadena.iterrows():
    memo.append(f"| {r['palanca']} | {r['sig_ln_sn_dep']} ({fmt(r['theta_ln_sn_dep'])}) | {r['sig_ln_feo_ret']} ({fmt(r['theta_ln_feo_ret'])}) | "
                f"{r['signo_neto_w10']} ({fmt(r['theta_J_w10_por_sd'])}) | {r['signo_neto_wanchor']} ({fmt(r['theta_J_wanchor_por_sd'])}) | "
                f"{r['evidencia_batch_exp19']} |")

memo.append("\n## Veredicto\n")
memo.append("(completar tras revisar los numeros arriba: parametrizacion y w recomendados, justificacion piro y estadistica.)\n")
memo.append("\n## Limites\n")
memo.append("- Reducción tiene solo 4 escalones/batch: la variante temporal tiene poca resolucion (pesos {1, 0.75, 0.5, 0.25}/4 aprox).\n"
            "- w anclado por OLS de batch depende de los controles y es sensible al outcome (actual vs next vs mean); se reporta rango, no un unico numero.\n"
            "- El PLM cross-fitted con ~350-400 filas de DEV y 5+ palancas tiene poca potencia para IC angostos; los theta con IC amplios no deben leerse como nulos.\n"
            "- El KPI del batch siguiente introduce autocorrelacion cronologica (idx_cronologico ya es control, pero no es un IV).\n")

(OUT / "e7_01_resultados.md").write_text("\n".join(memo), encoding="utf-8")
log("memo escrito: e7_01_resultados.md")
log("FIN")
