"""E7-02 -- Fusion: que objetivo por escalon es defendible con el KPI refinado (Iteracion 7).

Reancla beta_C / beta_T (exp 19) con el KPI refinado (`recuperacion_refinada_pct`), prueba terminos
cuadraticos (optimo interior) y si el cambio de la matriz (IRF) durante la Fusion aporta algo una vez
controlado el estado inicial (heel de escoria del batch anterior). PLM por escalon de los targets v5
en Fusion y cadena de signos palanca -> target de escalon -> KPI de batch.

Ejecutar (POSIX, ~5-10 min por el PLM -> background + monitorear el log):
    cd "C:\\Users\\user\\Desktop\\Linea de Sn\\Lingo Smelter"
    export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
    .venv/Scripts/python.exe experimentos/v5/e7_02_fusion.py > experimentos/v5/e7_02_log.txt 2>&1

Salidas (experimentos/v5/): e7_02_batch_regresiones.csv, e7_02_matriz_fusion.csv, e7_02_plm_fusion.csv,
e7_02_theta_fusion.csv, e7_02_cadena_signos.csv, e7_02_resultados.md.

No modifica modulos de la raiz ni hallazgos.md. Sin joblib/multiprocessing. Solo sklearn/statsmodels.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RAIZ))
import v5_lib as v5  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


SN_SALIDA_BATCH_KG = 51250.0
N_ESCALONES_FUSION = 7
KPI_PORCENTAJE = {"recuperacion_refinada_pct", "recuperacion_real_pct", "rendimiento_proxy_batch"}  # unidad: pp
DECISIONES_F = ["C_por_Sn_F", "gn_por_t_carga_F", "exceso_o2_F", "T_media_F", "cal_por_carga_F",
                "lanza_media_F", "feed_rate_F"]
KPIS_BATCH = ["recuperacion_refinada_pct", "f_dross", "f_polvo", "rendimiento_proxy_batch", "recuperacion_real_pct"]

PALANCAS_PRIMITIVAS = v5.PALANCAS_PRIMITIVAS  # a
PALANCAS_DERIVADAS = ["relacion_C_Sn_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]  # b
PALANCAS_TERMICAS = ["relacion_C_Sn_carga", "v5_gn_esp_nm3_t", "exceso_o2_combustion_pct"]  # c
EXTRA_B6 = ["b6_q_neto_por_min_MJ", "b6_dT_teorico_sin_reaccion"]

TARGETS_ESCALON = ["v5_d_lnirf", "v5_ln_sn_dep", "sn_extraido_est_kg", "d_temperatura_horno_celsius"]

# mapeo palanca de escalon -> decision de batch homologa (para la cadena de signos, paso 5)
PALANCA_A_BATCH = [
    dict(palanca="tasa_feed_Carbon_kg_min", set_="a", batch_var="C_por_Sn_F"),
    dict(palanca="relacion_C_Sn_carga", set_="b", batch_var="C_por_Sn_F"),
    dict(palanca="tasa_gn_nm3_min", set_="a", batch_var="gn_por_t_carga_F"),
    dict(palanca="v5_gn_esp_nm3_t", set_="c", batch_var="gn_por_t_carga_F"),
    dict(palanca="tasa_o2_nm3_min", set_="a", batch_var="exceso_o2_F"),
    dict(palanca="exceso_o2_combustion_pct", set_="b", batch_var="exceso_o2_F"),
    dict(palanca="tasa_aire_nm3_min", set_="a", batch_var=None),
]


def kg_por_escalon(kpi: str, coef: float) -> float:
    factor = 0.01 if kpi in KPI_PORCENTAJE else 1.0
    return coef * factor * SN_SALIDA_BATCH_KG / N_ESCALONES_FUSION


# =====================================================================================
# 1) Carga: escalones (dF) + batch v5 (KPI refinado) + decisiones de Fusion de exp 19
# =====================================================================================
log("cargando df de escalones (cache v3+v4+b6+v5)...")
df = v5.cargar_df()
dF = v5.base_fase(df, "Fusión")
batch = v5.construir_batch_v5(df)
log("batch v5", batch.shape, "dF (Fusion)", dF.shape)

e19 = pd.read_csv(RAIZ / "experimentos" / "19_batch_dataset.csv").set_index("Batch")
cols_e19 = ["C_por_Sn_F", "cal_por_carga_F", "gn_por_t_carga_F", "exceso_o2_F", "enriq_o2_F", "T_final_F",
            "dur_F", "feed_rate_F", "lanza_media_F", "ley_feo_finF", "b2_finF", "F_avance_final"]
batch = batch.join(e19[cols_e19], how="left")
log("batch + exp19", batch.shape, "NaN en decisiones F:", int(batch[DECISIONES_F].isna().sum().sum()))


# =====================================================================================
# 2) Regresiones de batch re-ancladas (OLS HC3, estandarizadas con mu/sd de la muestra TOTAL)
# =====================================================================================
def ols_hc3_batch(data: pd.DataFrame, kpi: str, modelo_vars: list[str], controles: list[str],
                   quad_vars: list[str] | None = None, extra_vars: list[str] | None = None,
                   muestras=("dev", "total", "lockbox")) -> tuple[list[dict], pd.Series, pd.Series]:
    """OLS HC3 de `kpi` sobre `modelo_vars` (+ `extra_vars` sin estandarizar propio, ya en unidades
    del estado) + `controles`, todas estandarizadas con mu/sd de la muestra TOTAL (dev+lockbox) tras
    dropna (convencion exp 19). `quad_vars` (subconjunto de modelo_vars) agrega el termino z^2 (centrado
    por construccion). Devuelve (filas, mu, sd) de las variables estandarizadas (para reconstruir el
    vertice en unidades originales)."""
    extra_vars = extra_vars or []
    quad_vars = quad_vars or []
    base_cols = list(dict.fromkeys(modelo_vars + controles + extra_vars))
    cols = [kpi] + base_cols
    d_total = data[cols + ["es_dev", "es_lockbox"]].dropna(subset=cols).copy()
    mu, sd = d_total[base_cols].mean(), d_total[base_cols].std()
    Z = (d_total[base_cols] - mu) / sd
    for v in quad_vars:
        Z[f"{v}_sq"] = Z[v] ** 2
    reportable = modelo_vars + [f"{v}_sq" for v in quad_vars] + extra_vars
    regresores = base_cols + [f"{v}_sq" for v in quad_vars]
    filas = []
    for nombre in muestras:
        if nombre == "dev":
            sub = d_total[d_total["es_dev"]]
        elif nombre == "lockbox":
            sub = d_total[d_total["es_lockbox"]]
        else:
            sub = d_total
        if len(sub) < 20:
            continue
        X = sm.add_constant(Z.loc[sub.index, regresores])
        res = sm.OLS(sub[kpi], X).fit(cov_type="HC3")
        for v in reportable:
            filas.append(dict(kpi=kpi, variable=v, muestra=nombre, coef_por_sd=float(res.params[v]),
                               ci_lo=float(res.conf_int().loc[v, 0]), ci_hi=float(res.conf_int().loc[v, 1]),
                               p=float(res.pvalues[v]), r2_adj=float(res.rsquared_adj), n=len(sub)))
    return filas, mu, sd


def vertice_boot(data: pd.DataFrame, kpi: str, modelo_vars: list[str], controles: list[str],
                  quad_var: str, muestra: str = "dev", n_boot: int = 300, seed: int = 0):
    """Bootstrap (resample de filas, batches independientes) del vertice x* = mu + sd*(-b1/(2 b2))
    de la parabola z*b1 + z^2*b2 en unidades originales de `quad_var`, sobre la muestra `muestra`."""
    base_cols = list(dict.fromkeys(modelo_vars + controles))
    cols = [kpi] + base_cols
    d_total = data[cols + ["es_dev", "es_lockbox"]].dropna(subset=cols).copy()
    mu, sd = d_total[base_cols].mean(), d_total[base_cols].std()
    sub = d_total[d_total["es_dev"]] if muestra == "dev" else (d_total[d_total["es_lockbox"]] if muestra == "lockbox" else d_total)
    Z = (sub[base_cols] - mu) / sd
    Z[f"{quad_var}_sq"] = Z[quad_var] ** 2
    regresores = base_cols + [f"{quad_var}_sq"]

    def _vertice(idx):
        X = sm.add_constant(Z.iloc[idx][regresores])
        res = sm.OLS(sub[kpi].iloc[idx], X).fit()
        b1, b2 = res.params[quad_var], res.params[f"{quad_var}_sq"]
        if abs(b2) < 1e-9:
            return np.nan
        zstar = -b1 / (2 * b2)
        return float(mu[quad_var] + zstar * sd[quad_var])

    x0 = _vertice(np.arange(len(sub)))
    rng = np.random.default_rng(seed)
    boots = [_vertice(rng.integers(0, len(sub), len(sub))) for _ in range(n_boot)]
    boots = np.array(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.nan, np.nan) if len(boots) < n_boot * 0.5 else (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    return x0, lo, hi, len(sub)


log("=== paso 2: regresiones de batch (M0/M1/M2/M3) ===")
filas_batch = []
vertices = []
for kpi in KPIS_BATCH:
    # M0: una a una
    for v in DECISIONES_F:
        fs, _, _ = ols_hc3_batch(batch, kpi, [v], v5.CONTROLES_BATCH)
        for f in fs:
            f["modelo"] = "M0"
        filas_batch += fs
    # M1: conjunta
    fs, _, _ = ols_hc3_batch(batch, kpi, DECISIONES_F, v5.CONTROLES_BATCH)
    for f in fs:
        f["modelo"] = "M1"
    filas_batch += fs
    # M2: M1 + cuadraticos centrados de C_por_Sn_F y T_media_F
    fs, mu2, sd2 = ols_hc3_batch(batch, kpi, DECISIONES_F, v5.CONTROLES_BATCH, quad_vars=["C_por_Sn_F", "T_media_F"])
    for f in fs:
        f["modelo"] = "M2"
    filas_batch += fs
    # vertice si el cuadratico es significativo en DEV o TOTAL
    for qv in ["C_por_Sn_F", "T_media_F"]:
        fila_dev = [f for f in fs if f["variable"] == f"{qv}_sq" and f["muestra"] == "dev"]
        fila_tot = [f for f in fs if f["variable"] == f"{qv}_sq" and f["muestra"] == "total"]
        sig_dev = bool(fila_dev) and (fila_dev[0]["ci_lo"] > 0 or fila_dev[0]["ci_hi"] < 0)
        sig_tot = bool(fila_tot) and (fila_tot[0]["ci_lo"] > 0 or fila_tot[0]["ci_hi"] < 0)
        if sig_dev or sig_tot:
            for muestra in ("dev", "total"):
                x0, lo, hi, n = vertice_boot(batch, kpi, DECISIONES_F, v5.CONTROLES_BATCH, qv, muestra=muestra)
                vertices.append(dict(kpi=kpi, variable=qv, muestra=muestra, vertice=x0, ci_lo=lo, ci_hi=hi, n=n,
                                      sig_dev=sig_dev, sig_total=sig_tot))
    # M3: M1 + F_lnirf_F0
    fs, _, _ = ols_hc3_batch(batch, kpi, DECISIONES_F + ["F_lnirf_F0"], v5.CONTROLES_BATCH)
    for f in fs:
        f["modelo"] = "M3"
    filas_batch += fs

reg_batch = pd.DataFrame(filas_batch)
reg_batch["significativo"] = (reg_batch["ci_lo"] > 0) | (reg_batch["ci_hi"] < 0)
reg_batch["kg_sn_por_escalon"] = [kg_por_escalon(r.kpi, r.coef_por_sd) for r in reg_batch.itertuples()]
reg_batch["kg_sn_ci_lo"] = [kg_por_escalon(r.kpi, r.ci_lo) for r in reg_batch.itertuples()]
reg_batch["kg_sn_ci_hi"] = [kg_por_escalon(r.kpi, r.ci_hi) for r in reg_batch.itertuples()]
reg_batch = reg_batch[["kpi", "modelo", "muestra", "variable", "coef_por_sd", "ci_lo", "ci_hi", "p", "r2_adj", "n",
                       "significativo", "kg_sn_por_escalon", "kg_sn_ci_lo", "kg_sn_ci_hi"]]
reg_batch.to_csv(HERE / "e7_02_batch_regresiones.csv", index=False)
vertices_df = pd.DataFrame(vertices)
if len(vertices_df):
    vertices_df.to_csv(HERE / "e7_02_vertices.csv", index=False)
log("e7_02_batch_regresiones.csv", reg_batch.shape, "vertices:", len(vertices_df))


# =====================================================================================
# 3) Aporta el cambio de la matriz durante la Fusion (controlando el heel F0)?
# =====================================================================================
log("=== paso 3: matriz de Fusion (IRF) vs KPI, controlando F_lnirf_F0 ===")
filas_matriz = []
kpi_m = "recuperacion_refinada_pct"
specs = [
    ("F0_sumdlnirf", ["F_lnirf_F0", "F_sum_d_lnirf"]),
    ("F0_lnirffin", ["F_lnirf_F0", "F_lnirf_fin"]),
    ("F0_sumlnsndep", ["F_lnirf_F0", "F_sum_ln_sn_dep"]),
]
for nombre_spec, vars_spec in specs:
    fs, _, _ = ols_hc3_batch(batch, kpi_m, vars_spec, v5.CONTROLES_BATCH, muestras=("dev", "total", "lockbox"))
    for f in fs:
        f["spec"] = nombre_spec
    filas_matriz += fs
matriz = pd.DataFrame(filas_matriz)
matriz["significativo"] = (matriz["ci_lo"] > 0) | (matriz["ci_hi"] < 0)
matriz = matriz[["spec", "kpi", "muestra", "variable", "coef_por_sd", "ci_lo", "ci_hi", "p", "r2_adj", "n", "significativo"]]
matriz.to_csv(HERE / "e7_02_matriz_fusion.csv", index=False)
log("e7_02_matriz_fusion.csv", matriz.shape)


# =====================================================================================
# 4) PLM por escalon en Fusion
# =====================================================================================
log("=== paso 4: PLM por escalon en Fusion ===")
combos = []
for t in TARGETS_ESCALON:
    combos.append((t, "a_primitivas", PALANCAS_PRIMITIVAS))
    combos.append((t, "b_derivadas", PALANCAS_DERIVADAS))
    combos.append((t, "c_termicas", PALANCAS_TERMICAS))
combos.append(("d_temperatura_horno_celsius", "c_termicas+b6", PALANCAS_TERMICAS + EXTRA_B6))

metricas, thetas = [], []
for target, label, palancas in combos:
    log(f"  PLM target={target} palancas={label} ({len(palancas)})")
    try:
        met, tabla, _ = v5.evaluar_plm(dF, v5.ESTADO_F_CURADO, palancas, target, n_boot=300,
                                        signo_teorico=v5.SIGNO_TEORICO_V5.get(target))
    except Exception as exc:  # noqa: BLE001
        log(f"    FALLO: {exc}")
        continue
    met["palancas_label"] = label
    metricas.append(met)
    tabla = tabla.reset_index()
    tabla["palancas_label"] = label
    thetas.append(tabla)
    log(f"    r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f} n_dev={met['n_dev']}")

plm_metricas = pd.DataFrame(metricas)
plm_metricas.to_csv(HERE / "e7_02_plm_fusion.csv", index=False)
theta_fusion = pd.concat(thetas, ignore_index=True) if thetas else pd.DataFrame()
theta_fusion.to_csv(HERE / "e7_02_theta_fusion.csv", index=False)
log("e7_02_plm_fusion.csv", plm_metricas.shape, "e7_02_theta_fusion.csv", theta_fusion.shape)


# =====================================================================================
# 5) Cadena de signos: palanca -> target de escalon -> KPI de batch
# =====================================================================================
log("=== paso 5: cadena de signos ===")


def signo_sig(ci_lo, ci_hi, valor):
    if pd.isna(ci_lo) or pd.isna(ci_hi):
        return np.nan
    return np.sign(valor) if (ci_lo > 0 or ci_hi < 0) else np.nan


# efecto agregado de cada target de escalon sobre el KPI de batch (DEV), con signo solo si es significativo
agg_ln_sn_dep, _, _ = ols_hc3_batch(batch, kpi_m, ["F_sum_ln_sn_dep"], v5.CONTROLES_BATCH, muestras=("dev",))
agg_d_lnirf = [f for f in filas_matriz if f["spec"] == "F0_sumdlnirf" and f["variable"] == "F_sum_d_lnirf" and f["muestra"] == "dev"]
m1_dev = reg_batch[(reg_batch["kpi"] == kpi_m) & (reg_batch["modelo"] == "M1") & (reg_batch["muestra"] == "dev")].set_index("variable")

sign_agg = {}
if agg_ln_sn_dep:
    r = agg_ln_sn_dep[0]
    sign_agg["v5_ln_sn_dep"] = signo_sig(r["ci_lo"], r["ci_hi"], r["coef_por_sd"])
    sign_agg["sn_extraido_est_kg"] = sign_agg["v5_ln_sn_dep"]  # mismo fenomeno (agotamiento de Sn en Fusion)
if agg_d_lnirf:
    r = agg_d_lnirf[0]
    sign_agg["v5_d_lnirf"] = signo_sig(r["ci_lo"], r["ci_hi"], r["coef_por_sd"])
if "T_media_F" in m1_dev.index:
    r = m1_dev.loc["T_media_F"]
    sign_agg["d_temperatura_horno_celsius"] = signo_sig(r["ci_lo"], r["ci_hi"], r["coef_por_sd"])

filas_cadena = []
for entry in PALANCA_A_BATCH:
    palanca, set_, batch_var = entry["palanca"], entry["set_"], entry["batch_var"]
    for target in TARGETS_ESCALON:
        label_map = {"a": "a_primitivas", "b": "b_derivadas", "c": "c_termicas"}
        tabla_t = theta_fusion[(theta_fusion["target"] == target) & (theta_fusion["palancas_label"] == label_map[set_])]
        fila_theta = tabla_t[tabla_t["palanca"] == palanca]
        theta_sign = np.nan
        if len(fila_theta):
            th = fila_theta.iloc[0]
            theta_sign = signo_sig(th["ci_lo"], th["ci_hi"], th["theta_unidad"])
        agg_s = sign_agg.get(target, np.nan)
        kpi_implicado = theta_sign * agg_s if pd.notna(theta_sign) and pd.notna(agg_s) else np.nan
        batch_sign = np.nan
        if batch_var is not None and batch_var in m1_dev.index:
            r = m1_dev.loc[batch_var]
            batch_sign = signo_sig(r["ci_lo"], r["ci_hi"], r["coef_por_sd"])
        if pd.isna(kpi_implicado) or pd.isna(batch_sign):
            clasif = "indeterminada"
        elif kpi_implicado == batch_sign:
            clasif = "coherente"
        else:
            clasif = "incoherente"
        filas_cadena.append(dict(palanca=palanca, target_escalon=target, palancas_label=label_map[set_],
                                  theta_signo=theta_sign, agg_target_sobre_kpi_signo=agg_s,
                                  kpi_implicado_signo=kpi_implicado, batch_var=batch_var,
                                  batch_directo_signo=batch_sign, clasificacion=clasif))

cadena = pd.DataFrame(filas_cadena)
cadena.to_csv(HERE / "e7_02_cadena_signos.csv", index=False)
log("e7_02_cadena_signos.csv", cadena.shape)
log(cadena["clasificacion"].value_counts().to_string())

log(f"TOTAL {time.time() - T0:.1f}s -- listo")
