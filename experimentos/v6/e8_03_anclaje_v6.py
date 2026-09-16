"""E8-03 -- Anclaje del objetivo del prescriptor en el KPI refinado con los targets v6 (masa de escoria por
cierre fisico / trazador de matriz G, ver `masa_v6.py`) y estimacion de los efectos de las palancas (theta)
con PLM, comparando con v5 (iteracion 8, ver `experimentos/v6/ITERACION_8_diseno.md`).

Replica las tareas de E7-01 (Reduccion: anclaje de w, bateria Spearman, PLM y cadena de signos,
`experimentos/v5/e7_01_reduccion_componentes.py`) y E7-02 (Fusion: conservacion de Sn controlando el heel,
`experimentos/v5/e7_02_fusion.py`) con los targets v6 (`m6_ln_sn_dep`, `m6_ln_feo_ret`) y compara contra los
targets v5 (`v5_ln_sn_dep`, `v5_ln_feo_ret`) sobre las MISMAS filas cuando corresponde.

Ejecutar (background, ~30-50 min por el PLM cross-fitted):
    cd "C:\\Users\\user\\Desktop\\Linea de Sn\\Lingo Smelter"
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/v6/e8_03_anclaje_v6.py \
        > experimentos/v6/e8_03_log.txt 2>&1

Salidas (experimentos/v6/): e8_03_w_anclaje.csv, e8_03_w_sweep.csv, e8_03_bateria.csv, e8_03_fusion.csv,
e8_03_plm_comparacion.csv, e8_03_theta.csv, e8_03_reversion.csv, e8_03_cadena_signos.csv,
e8_03_resultados.md.

No modifica modulos de la raiz, `masa_v6.py` ni `hallazgos.md`.
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
HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
for _p in (HERE, RAIZ, RAIZ / "experimentos" / "v5", RAIZ / "experimentos" / "balances"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import v6_lib as v6  # noqa: E402
import v5_lib as v5  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

OUT = HERE
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa: BLE001
        return "NA"


# =============================================================================
# 0) Carga
# =============================================================================
log("cargando datos (v6_lib.cargar_df = v5_lib + masa_v6)...")
df = v6.cargar_df()
dR = v6.base_fase(df, "Reducción")
dF = v6.base_fase(df, "Fusión")
batch = v6.construir_batch_v6(df)
log(f"df {df.shape}  dR {dR.shape}  dF {dF.shape}  batch {batch.shape} "
    f"({int(batch['es_dev'].sum())} DEV / {int(batch['es_lockbox'].sum())} lockbox)")

KPI = v5.KPI_PRINCIPAL
CONTROLES = v5.CONTROLES_BATCH
COL_SN_V6, COL_FEO_V6 = "R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret"
COL_SN_V5, COL_FEO_V5 = "R_sum_ln_sn_dep", "R_sum_ln_feo_ret"

# =============================================================================
# 1) Anclaje de w (OLS HC3, batch DEV, mismos controles + tendencia idx_cronologico que E7-01)
# =============================================================================
log("=== tarea 1: anclaje de w (v6 vs v5, misma muestra) ===")


def ols_w(dfin: pd.DataFrame, col_sn: str, col_feo: str, kpi_col: str, controles=CONTROLES,
          n_boot: int = 1000, seed: int = 0) -> dict:
    d = dfin[[col_sn, col_feo, kpi_col] + controles].dropna()
    if len(d) < 30:
        return {}
    X = sm.add_constant(d[[col_sn, col_feo] + controles])
    res = sm.OLS(d[kpi_col], X).fit(cov_type="HC3")
    b_sn, b_feo = res.params[col_sn], res.params[col_feo]
    out = {"col_sn": col_sn, "col_feo": col_feo, "kpi": kpi_col, "n": len(d),
           "coef_sn": float(b_sn), "p_sn": float(res.pvalues[col_sn]),
           "coef_feo": float(b_feo), "p_feo": float(res.pvalues[col_feo]),
           "w_puntual": float(b_feo / b_sn) if b_sn != 0 else np.nan, "r2adj": float(res.rsquared_adj)}
    rng = np.random.default_rng(seed)
    idxs = d.index.to_numpy()
    ws, fracs = [], []
    for _ in range(n_boot):
        samp = rng.choice(idxs, size=len(idxs), replace=True)
        dd = d.loc[samp]
        Xb = sm.add_constant(dd[[col_sn, col_feo] + controles].reset_index(drop=True))
        try:
            rb = sm.OLS(dd[kpi_col].reset_index(drop=True), Xb).fit()
        except Exception:  # noqa: BLE001
            continue
        bs, bf = rb.params[col_sn], rb.params[col_feo]
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
batch_dev[f"{KPI}_mean_cur_next"] = batch_dev[[KPI, f"{KPI}_next"]].mean(axis=1)
kpi_variantes = {"actual": KPI, "next": f"{KPI}_next", "mean_cur_next": f"{KPI}_mean_cur_next"}

# muestra comun para v5 y v6 (dropna sobre la union): comparacion honesta con el mismo n
cols_comunes = [COL_SN_V6, COL_FEO_V6, COL_SN_V5, COL_FEO_V5] + list(kpi_variantes.values()) + CONTROLES
batch_dev_comun = batch_dev.dropna(subset=cols_comunes)
log(f"muestra comun DEV (v5 y v6 + controles + 3 variantes KPI, sin NaN): n={len(batch_dev_comun)}")

filas_w = []
seed_ctr = 0
for targets_nombre, (col_sn, col_feo) in [("v6", (COL_SN_V6, COL_FEO_V6)), ("v5", (COL_SN_V5, COL_FEO_V5))]:
    for variante, kpi_col in kpi_variantes.items():
        seed_ctr += 1
        r = ols_w(batch_dev_comun, col_sn, col_feo, kpi_col, n_boot=1000, seed=100 + seed_ctr)
        if r:
            r.update(targets=targets_nombre, variante=variante, muestra="dev")
            filas_w.append(r)
            log(f"w_anclaje {targets_nombre} {variante:14s} n={r['n']:3d} coef_sn={fmt(r['coef_sn'])}(p{fmt(r['p_sn'])}) "
                f"coef_feo={fmt(r['coef_feo'])}(p{fmt(r['p_feo'])}) w={fmt(r['w_puntual'], 2)} "
                f"[{fmt(r['w_ci_lo'], 2)},{fmt(r['w_ci_hi'], 2)}]")

# lockbox: solo reporte (mismos controles, muestra comun de lockbox)
batch_lb = batch[batch["es_lockbox"]].copy()
batch_lb_comun = batch_lb.dropna(subset=[COL_SN_V6, COL_FEO_V6, COL_SN_V5, COL_FEO_V5, KPI] + CONTROLES)
log(f"muestra comun lockbox: n={len(batch_lb_comun)}")
for targets_nombre, (col_sn, col_feo) in [("v6", (COL_SN_V6, COL_FEO_V6)), ("v5", (COL_SN_V5, COL_FEO_V5))]:
    seed_ctr += 1
    r = ols_w(batch_lb_comun, col_sn, col_feo, KPI, n_boot=1000, seed=200 + seed_ctr)
    if r:
        r.update(targets=targets_nombre, variante="actual", muestra="lockbox")
        filas_w.append(r)
        log(f"w_anclaje LOCKBOX(reporte) {targets_nombre} n={r['n']:3d} w={fmt(r['w_puntual'], 2)} "
            f"[{fmt(r['w_ci_lo'], 2)},{fmt(r['w_ci_hi'], 2)}]")

df_w = pd.DataFrame(filas_w)
df_w.to_csv(OUT / "e8_03_w_anclaje.csv", index=False)
log(f"guardado e8_03_w_anclaje.csv ({len(df_w)} filas)")

w_v6 = df_w[(df_w["targets"] == "v6") & (df_w["variante"] == "actual") & (df_w["muestra"] == "dev")]
w_v5 = df_w[(df_w["targets"] == "v5") & (df_w["variante"] == "actual") & (df_w["muestra"] == "dev")]
w_anchor_v6 = max(float(w_v6["w_puntual"].iloc[0]), 0.0) if len(w_v6) else 6.04
w_anchor_v5 = max(float(w_v5["w_puntual"].iloc[0]), 0.0) if len(w_v5) else 6.04
log(f"w anclado v6={fmt(w_anchor_v6, 2)}  w anclado v5={fmt(w_anchor_v5, 2)} (referencia E7-01: 6.04)")

# =============================================================================
# 1b) Barrido de w
# =============================================================================
log("=== tarea 1b: barrido de w ===")
W_GRID = [2, 4, 6, 8, 10, 15]


def spearman_ic(x: np.ndarray, y: np.ndarray, n_boot: int = 500, seed: int = 0):
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bs.append(stats.spearmanr(x[idx], y[idx])[0])
    return float(rho), float(p), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


filas_sweep = []
for targets_nombre, (col_sn, col_feo) in [("v6", (COL_SN_V6, COL_FEO_V6)), ("v5", (COL_SN_V5, COL_FEO_V5))]:
    for w in W_GRID:
        j = batch[col_sn] + w * batch[col_feo]
        for split_nombre, mask in [("dev", batch["es_dev"]), ("lockbox", batch["es_lockbox"]),
                                    ("total", pd.Series(True, index=batch.index))]:
            d = pd.concat([j[mask], batch.loc[mask, KPI]], axis=1).dropna()
            if len(d) < 15:
                continue
            rho, p, lo, hi = spearman_ic(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy(), n_boot=400, seed=int(w * 7 + 1))
            filas_sweep.append(dict(targets=targets_nombre, w=w, split=split_nombre, n=len(d), rho=rho, p=p, ci_lo=lo, ci_hi=hi))
df_sweep = pd.DataFrame(filas_sweep)
df_sweep.to_csv(OUT / "e8_03_w_sweep.csv", index=False)
log(f"guardado e8_03_w_sweep.csv ({len(df_sweep)} filas)")
for targets_nombre in ("v6", "v5"):
    sub = df_sweep[(df_sweep["targets"] == targets_nombre) & (df_sweep["split"] == "dev")]
    log(f"  sweep {targets_nombre}: " + " ".join(f"w={int(r.w)}:rho={fmt(r.rho)}" for r in sub.itertuples()))

# =============================================================================
# 2) Bateria Spearman (v6 y v5) contra KPI actual, f_dross, f_polvo, KPI del batch siguiente
# =============================================================================
log("=== tarea 2: bateria Spearman ===")
VARS_BATERIA = [COL_SN_V6, COL_FEO_V6, "F_sum_m6_ln_sn_dep", COL_SN_V5, COL_FEO_V5, "F_sum_ln_sn_dep"]
KPIS_BATERIA = [KPI, "f_dross", "f_polvo", f"{KPI}_next"]
filas_bat = []
for col in VARS_BATERIA:
    for kpi_c in KPIS_BATERIA:
        r = v6.bateria_batch(batch, col, kpi=kpi_c)
        filas_bat.append(r)
        log(f"  {col:24s} vs {kpi_c:28s} rho_dev={fmt(r.get('rho_dev'))} p_dev={fmt(r.get('p_dev'))} "
            f"rho_lockbox={fmt(r.get('rho_lockbox'))}")
df_bateria = pd.DataFrame(filas_bat)
df_bateria.to_csv(OUT / "e8_03_bateria.csv", index=False)
log(f"guardado e8_03_bateria.csv ({len(df_bateria)} filas)")

# =============================================================================
# 3) Fusion: KPI ~ F_lnirf_F0 (heel) + F_sum_m6_ln_sn_dep, controles E7-02 (= CONTROLES_BATCH)
# =============================================================================
log("=== tarea 3: Fusion, conservacion de Sn controlando el heel ===")


def ols_hc3_batch_sd(data: pd.DataFrame, kpi: str, modelo_vars: list[str], controles: list[str],
                     muestras=("dev", "total", "lockbox")) -> list[dict]:
    """Estandarizado (mu/sd de la muestra TOTAL tras dropna), igual convencion que e7_02_fusion.ols_hc3_batch."""
    base_cols = list(dict.fromkeys(modelo_vars + controles))
    cols = [kpi] + base_cols
    d_total = data[cols + ["es_dev", "es_lockbox"]].dropna(subset=cols).copy()
    mu, sd = d_total[base_cols].mean(), d_total[base_cols].std()
    Z = (d_total[base_cols] - mu) / sd
    filas = []
    for nombre in muestras:
        sub = d_total[d_total["es_dev"]] if nombre == "dev" else (d_total[d_total["es_lockbox"]] if nombre == "lockbox" else d_total)
        if len(sub) < 20:
            continue
        X = sm.add_constant(Z.loc[sub.index, base_cols])
        res = sm.OLS(sub[kpi], X).fit(cov_type="HC3")
        for v in modelo_vars:
            filas.append(dict(kpi=kpi, variable=v, muestra=nombre, escala="por_sd", coef=float(res.params[v]),
                              ci_lo=float(res.conf_int().loc[v, 0]), ci_hi=float(res.conf_int().loc[v, 1]),
                              p=float(res.pvalues[v]), r2_adj=float(res.rsquared_adj), n=len(sub)))
    return filas


def ols_natural_batch(data: pd.DataFrame, kpi: str, modelo_vars: list[str], controles: list[str],
                      muestras=("dev", "total", "lockbox")) -> list[dict]:
    """Sin estandarizar: coeficiente en pp de KPI por unidad natural de `modelo_vars` (para comparar con
    CONFIG_V5['pp_kpi_por_unidad_sn_dep_F'] = -0.45)."""
    base_cols = list(dict.fromkeys(modelo_vars + controles))
    cols = [kpi] + base_cols
    d_total = data[cols + ["es_dev", "es_lockbox"]].dropna(subset=cols).copy()
    filas = []
    for nombre in muestras:
        sub = d_total[d_total["es_dev"]] if nombre == "dev" else (d_total[d_total["es_lockbox"]] if nombre == "lockbox" else d_total)
        if len(sub) < 20:
            continue
        X = sm.add_constant(sub[base_cols])
        res = sm.OLS(sub[kpi], X).fit(cov_type="HC3")
        for v in modelo_vars:
            filas.append(dict(kpi=kpi, variable=v, muestra=nombre, escala="natural", coef=float(res.params[v]),
                              ci_lo=float(res.conf_int().loc[v, 0]), ci_hi=float(res.conf_int().loc[v, 1]),
                              p=float(res.pvalues[v]), r2_adj=float(res.rsquared_adj), n=len(sub)))
    return filas


filas_fusion = []
for targets_nombre, col_sn in [("v6", "F_sum_m6_ln_sn_dep"), ("v5", "F_sum_ln_sn_dep")]:
    modelo_vars = ["F_lnirf_F0", col_sn]
    for fs in (ols_hc3_batch_sd(batch, KPI, modelo_vars, CONTROLES), ols_natural_batch(batch, KPI, modelo_vars, CONTROLES)):
        for f in fs:
            f["targets"] = targets_nombre
            filas_fusion.append(f)
df_fusion = pd.DataFrame(filas_fusion)
df_fusion.to_csv(OUT / "e8_03_fusion.csv", index=False)
log(f"guardado e8_03_fusion.csv ({len(df_fusion)} filas)")
for _, r in df_fusion[(df_fusion["muestra"] == "dev") & (df_fusion["variable"].str.contains("sn_dep")) & (df_fusion["escala"] == "natural")].iterrows():
    log(f"  Fusion natural DEV {r['targets']}: coef={fmt(r['coef'])} p={fmt(r['p'])} [{fmt(r['ci_lo'])},{fmt(r['ci_hi'])}] n={r['n']}")

# =============================================================================
# 4) PLM por escalon (Reduccion: Sn/FeO; Fusion: Sn) -- v6 vs v5, con y sin signos teoricos
# =============================================================================
log("=== tarea 4: PLM por escalon (v6 vs v5, con/sin signos, FULL/CURADO) ===")

PARAM_CINETICA_R = {
    "Sn": {"v6": (v6.PALANCAS_V6[("Reducción", "sn_dep")], v6.SIGNO_TEORICO_V6[("Reducción", "sn_dep")]),
           "v5": (mp5.PALANCAS_V5[("Reducción", "sn_dep")], mp5.SIGNO_TEORICO_V5[("Reducción", "sn_dep")])},
    "FeO": {"v6": (v6.PALANCAS_V6[("Reducción", "feo_ret")], v6.SIGNO_TEORICO_V6[("Reducción", "feo_ret")]),
            "v5": (mp5.PALANCAS_V5[("Reducción", "feo_ret")], mp5.SIGNO_TEORICO_V5[("Reducción", "feo_ret")])},
}
PARAM_DOSIS_R = {
    "Sn": {"v6": (["m6_dosis_C_sn", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                  dict(v6.SIGNO_TEORICO_V6[("Reducción", "sn_dep")], m6_dosis_C_sn=+1)),
           "v5": (["v5_dosis_C_sn", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                  mp5.SIGNO_TEORICO_V5[("Reducción", "sn_dep")])},
    "FeO": {"v6": (["m6_exceso_C_pos", "m6_C_x_avance", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                   dict(v6.SIGNO_TEORICO_V6[("Reducción", "feo_ret")], m6_exceso_C_pos=-1, m6_C_x_avance=-1)),
            "v5": (["v5_exceso_C_pos", "v5_C_x_avance", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                   mp5.SIGNO_TEORICO_V5[("Reducción", "feo_ret")])},
}
TARGET_R = {"Sn": ("m6_ln_sn_dep", "v5_ln_sn_dep"), "FeO": ("m6_ln_feo_ret", "v5_ln_feo_ret")}
ESTADOS_R = {"FULL": (v6.ESTADO_R_FULL_V6, mp5.ESTADO_R_FULL), "CURADO": (v6.ESTADO_R_CURADO_V6, v5.ESTADO_R_CURADO)}
# combinaciones (estado, parametrizacion) a correr. CURADO con ambas parametrizaciones (barato, como en
# E7-01). FULL (37 features) se prueba UNA sola vez por grupo (solo con signos, solo v6) como control de
# robustez -- un grid FULL completo (2 versiones x 2 signos) resulto en >10 min por corrida en un timing
# previo (ver e8_03_resultados.md, Limites); E7-01 ya mostro que FULL vs CURADO cambia el R2_oof en <=0.01
# para ln_sn_dep y de forma no sistematica para ln_feo_ret, asi que no se re-corre el grid completo aqui.
COMBOS_R = [("CURADO", "cinetica"), ("CURADO", "dosis_sobrante")]
PARAM_R = {"cinetica": PARAM_CINETICA_R, "dosis_sobrante": PARAM_DOSIS_R}

filas_met, tablas_theta = [], []
n_total = len(COMBOS_R) * 2 * 2 * 2 + 2 * 2  # grupos x targets(v5,v6) x signos x + Fusion
i_run = 0
for estado_nombre, param_nombre in COMBOS_R:
    for grupo in ("Sn", "FeO"):
        target_v6, target_v5 = TARGET_R[grupo]
        estado_v6, estado_v5 = ESTADOS_R[estado_nombre]
        palancas_v6, signos_v6 = PARAM_R[param_nombre][grupo]["v6"]
        palancas_v5, signos_v5 = PARAM_R[param_nombre][grupo]["v5"]
        # filas comunes v6/v5 (misma muestra): dropna sobre la union de todo lo que usan ambos modelos
        cols_union = list(dict.fromkeys(estado_v6 + estado_v5 + palancas_v6 + palancas_v5 +
                                        [target_v6, target_v5, "Batch", "fecha_inicio", "orden_escalon_fase"]))
        d_comun = dR.dropna(subset=[c for c in cols_union if c not in ("Batch", "fecha_inicio", "orden_escalon_fase")]).reset_index(drop=True)
        for con_signos in (True, False):
            for targets_nombre, target, estado, palancas, signos in [
                ("v6", target_v6, estado_v6, palancas_v6, signos_v6),
                ("v5", target_v5, estado_v5, palancas_v5, signos_v5),
            ]:
                i_run += 1
                log(f"[{i_run}/{n_total}] Reduccion {grupo:3s} {estado_nombre:6s} {param_nombre:14s} "
                    f"{targets_nombre} signos={con_signos} n_comun={len(d_comun)}")
                try:
                    if con_signos:
                        met, tabla, _pred = v6.evaluar_plm_signos(d_comun, estado, palancas, target, n_boot=300, signo_teorico=signos)
                    else:
                        met, tabla, _pred = v5.evaluar_plm(d_comun, estado, palancas, target, n_boot=300, signo_teorico=signos)
                except Exception as exc:  # noqa: BLE001
                    log("  ERROR:", repr(exc))
                    continue
                met.update(fase="Reducción", grupo=grupo, estado=estado_nombre, parametrizacion=param_nombre,
                           targets=targets_nombre, con_signos=con_signos, n_comun=len(d_comun))
                filas_met.append(met)
                tabla = tabla.reset_index()
                tabla.insert(0, "fase", "Reducción"); tabla.insert(1, "grupo", grupo)
                tabla.insert(2, "estado", estado_nombre); tabla.insert(3, "parametrizacion", param_nombre)
                tabla.insert(4, "targets", targets_nombre); tabla.insert(5, "con_signos", con_signos)
                tablas_theta.append(tabla)
                log(f"  R2_oof={fmt(met['r2_oof'])}  R2_lockbox={fmt(met['r2_lockbox'])}  n_dev={met['n_dev']} n_lb={met['n_lockbox']}")

# Fusion: m6_ln_sn_dep / v5_ln_sn_dep con ESTADO_F_CURADO_V6/V5 y palancas de Fusion v5
palancas_F, signos_F = mp5.PALANCAS_V5[("Fusión", "sn_dep")], mp5.SIGNO_TEORICO_V5[("Fusión", "sn_dep")]
estado_F_v6, estado_F_v5 = v6.ESTADO_F_CURADO_V6, v5.ESTADO_F_CURADO
cols_union_F = list(dict.fromkeys(estado_F_v6 + estado_F_v5 + palancas_F +
                                  ["m6_ln_sn_dep", "v5_ln_sn_dep", "Batch", "fecha_inicio", "orden_escalon_fase"]))
dF_comun = dF.dropna(subset=[c for c in cols_union_F if c not in ("Batch", "fecha_inicio", "orden_escalon_fase")]).reset_index(drop=True)
for con_signos in (True, False):
    for targets_nombre, target, estado in [("v6", "m6_ln_sn_dep", estado_F_v6), ("v5", "v5_ln_sn_dep", estado_F_v5)]:
        i_run += 1
        log(f"[{i_run}/{n_total}] Fusion Sn CURADO {targets_nombre} signos={con_signos} n_comun={len(dF_comun)}")
        try:
            if con_signos:
                met, tabla, _pred = v6.evaluar_plm_signos(dF_comun, estado, palancas_F, target, n_boot=300, signo_teorico=signos_F)
            else:
                met, tabla, _pred = v5.evaluar_plm(dF_comun, estado, palancas_F, target, n_boot=300, signo_teorico=signos_F)
        except Exception as exc:  # noqa: BLE001
            log("  ERROR:", repr(exc))
            continue
        met.update(fase="Fusión", grupo="Sn", estado="CURADO", parametrizacion="fusion_v5", targets=targets_nombre,
                   con_signos=con_signos, n_comun=len(dF_comun))
        filas_met.append(met)
        tabla = tabla.reset_index()
        tabla.insert(0, "fase", "Fusión"); tabla.insert(1, "grupo", "Sn")
        tabla.insert(2, "estado", "CURADO"); tabla.insert(3, "parametrizacion", "fusion_v5")
        tabla.insert(4, "targets", targets_nombre); tabla.insert(5, "con_signos", con_signos)
        tablas_theta.append(tabla)
        log(f"  R2_oof={fmt(met['r2_oof'])}  R2_lockbox={fmt(met['r2_lockbox'])}")

df_met = pd.DataFrame(filas_met)
df_theta = pd.concat(tablas_theta, ignore_index=True)
df_met.to_csv(OUT / "e8_03_plm_comparacion.csv", index=False)
df_theta.to_csv(OUT / "e8_03_theta.csv", index=False)
log(f"guardado e8_03_plm_comparacion.csv ({len(df_met)} filas) y e8_03_theta.csv ({len(df_theta)} filas)")

# =============================================================================
# 5) Control de reversion de ruido (ln_feo_ret v6 vs v5): R2 OOF con/sin CaO_prev/B2/IRF y sin FeO_inv
# =============================================================================
log("=== tarea 5: control de reversion de ruido de %CaO / %FeO ===")


def r2_oof_generico(df_fase: pd.DataFrame, estado: list[str], target: str, n_splits: int = 5, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    dev, _lb = mp.split_dev_lockbox(df_fase)
    need = list(dict.fromkeys(estado + [target, "Batch"]))
    d = df_fase.loc[df_fase["Batch"].isin(dev), need].dropna().reset_index(drop=True)
    if len(d) < 50:
        return np.nan, len(d)
    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits).split(d[estado], d[target], d["Batch"]):
        h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=seed)
        h.fit(d.iloc[tr][estado], d.iloc[tr][target])
        oof[te] = h.predict(d.iloc[te][estado])
    r2 = float(1 - np.nanmean((d[target] - oof) ** 2) / np.nanvar(d[target]))
    return r2, len(d)


DROP_CAO_MATRIZ = ["ley_cao_escoria_pct_prev", "basicidad_B2_prev", "indice_irf_prev"]
filas_rev = []
for targets_nombre, target, estado_full, drop_feo_inv in [
    ("v6", "m6_ln_feo_ret", v6.ESTADO_R_CURADO_V6, ["ley_feo_escoria_pct_prev", "m6_feo_inv_kg_prev"]),
    ("v5", "v5_ln_feo_ret", v5.ESTADO_R_CURADO, ["ley_feo_escoria_pct_prev", "feo_inventario_escoria_est_kg_prev"]),
]:
    r_full, n_full = r2_oof_generico(dR, estado_full, target)
    estado_sin_cao = [c for c in estado_full if c not in DROP_CAO_MATRIZ]
    r_sin_cao, n_sin_cao = r2_oof_generico(dR, estado_sin_cao, target)
    estado_sin_feo = [c for c in estado_full if c not in drop_feo_inv]
    r_sin_feo, n_sin_feo = r2_oof_generico(dR, estado_sin_feo, target)
    filas_rev.append(dict(targets=targets_nombre, target=target, n_estado_full=len(estado_full),
                          r2_full=r_full, n_full=n_full,
                          r2_sin_CaO_B2_IRF=r_sin_cao, n_sin_cao=n_sin_cao,
                          r2_sin_FeO_inv=r_sin_feo, n_sin_feo=n_sin_feo,
                          drop_cao=",".join(DROP_CAO_MATRIZ), drop_feo=",".join(drop_feo_inv)))
    log(f"  {targets_nombre} {target}: R2_full={fmt(r_full)} (n={n_full})  R2_sinCaO/B2/IRF={fmt(r_sin_cao)}  R2_sinFeOinv={fmt(r_sin_feo)}")
df_reversion = pd.DataFrame(filas_rev)
df_reversion.to_csv(OUT / "e8_03_reversion.csv", index=False)
log(f"guardado e8_03_reversion.csv ({len(df_reversion)} filas)")

# =============================================================================
# 6) Cadena de signos palanca -> ln_sn_dep / ln_feo_ret -> J (w anclado), v6 y v5
# =============================================================================
log("=== tarea 6: cadena de signos ===")


def theta_de(targets_nombre: str, grupo: str, estado_nombre: str, param_nombre: str, con_signos: bool) -> pd.DataFrame:
    m = ((df_theta["targets"] == targets_nombre) & (df_theta["grupo"] == grupo) & (df_theta["estado"] == estado_nombre) &
         (df_theta["parametrizacion"] == param_nombre) & (df_theta["con_signos"] == con_signos))
    return df_theta[m].set_index("palanca")


filas_cadena = []
for targets_nombre, w_anchor in [("v6", w_anchor_v6), ("v5", w_anchor_v5)]:
    th_sn = theta_de(targets_nombre, "Sn", "CURADO", "cinetica", True)
    th_feo = theta_de(targets_nombre, "FeO", "CURADO", "cinetica", True)
    palancas_union = sorted(set(th_sn.index) | set(th_feo.index))
    for p in palancas_union:
        theta_sn = float(th_sn.loc[p, "theta_unidad"]) if p in th_sn.index else np.nan
        theta_feo = float(th_feo.loc[p, "theta_unidad"]) if p in th_feo.index else np.nan
        sig_sn = theta_sn if p in th_sn.index and np.isfinite(theta_sn) else 0.0
        sig_feo = theta_feo if p in th_feo.index and np.isfinite(theta_feo) else 0.0
        sd_sn = float(th_sn.loc[p, "sd_palanca"]) if p in th_sn.index else np.nan
        sd_feo = float(th_feo.loc[p, "sd_palanca"]) if p in th_feo.index else np.nan
        sd_p = np.nanmean([sd_sn, sd_feo]) if not (np.isnan(sd_sn) and np.isnan(sd_feo)) else np.nan
        j_w = sig_sn + w_anchor * sig_feo
        filas_cadena.append(dict(
            targets=targets_nombre, palanca=p, w_anchor=w_anchor,
            theta_ln_sn_dep=theta_sn, sig_ln_sn_dep=("+" if theta_sn > 0 else "-" if theta_sn < 0 else "0") if p in th_sn.index else "n/a",
            theta_ln_feo_ret=theta_feo, sig_ln_feo_ret=("+" if theta_feo > 0 else "-" if theta_feo < 0 else "0") if p in th_feo.index else "n/a",
            sd_palanca_aprox=sd_p, theta_J_wanchor_por_unidad=j_w,
            theta_J_wanchor_por_sd=j_w * sd_p if not np.isnan(sd_p) else np.nan,
            signo_neto_wanchor=("+" if j_w > 0 else "-" if j_w < 0 else "0"),
        ))
df_cadena = pd.DataFrame(filas_cadena)
df_cadena.to_csv(OUT / "e8_03_cadena_signos.csv", index=False)
log(f"guardado e8_03_cadena_signos.csv ({len(df_cadena)} filas)")

log(f"TOTAL calculo {time.time() - T0:.1f}s")
