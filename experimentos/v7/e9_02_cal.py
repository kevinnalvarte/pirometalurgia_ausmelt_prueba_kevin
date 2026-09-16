"""E9-02 -- La cal (CaO, tolva 6) y la basicidad como palanca bajo la masa v6 (Iteracion 9, 2026-09-16).

Reevalua el "efecto de la cal" de E7-02 (KPI refinado + trazador CaO v3, ver hallazgos SS19.1: ese efecto era
CONTABLE, no metalurgico) ahora que existe la masa v6 (cierre fisico sin dividir por %CaO). Etapas (una por
invocacion; ver ITERACION_9_diseno.md):

    kpi        -- verifica si recuperacion_refinada_pct depende del trazador CaO antiguo (via la fila que
                  elige como "ultimo ensayo de Reduccion"); construye kpi_v6 si corresponde y decide cual usar.
    batch      -- regresiones de batch (OLS HC3 + bootstrap por batch) del KPI y canales sobre cal/carga,
                  cal/Sn, B2 medio/final de Fusion; cuadratico; dosis-respuesta por quintiles; KPI del batch
                  siguiente (heel).
    escalon_F  -- PLM con signos en Fusion: cal (+ C/Sn, GN, O2, aire) sobre agotamiento de Sn, dT, dB2, dlnIRF.
    escalon_R  -- distribucion de la cal en Reduccion; PLM si hay soporte.
    basicidad  -- optimo interior de B2/IRF al cierre de Fusion sobre los resultados de Reduccion y el KPI.
    resumen    -- e9_02_resumen.csv + memo e9_02_resultados.md con el veredicto.
    all        -- las 6 en secuencia (usar solo en foreground si se sabe que cabe en el timeout).

Ejecutar (POSIX):
    cd "/c/Users/user/Desktop/Linea de Sn/Lingo Smelter"
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v7/e9_02_cal.py <etapa>

Sin joblib/loky, sin LightGBM. Salidas en experimentos/v7/ (e9_02_*.csv, e9_02_resultados.md, e9_02_log.txt).
"""
from __future__ import annotations

import json
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
import v7_lib as L  # noqa: E402
import feature_engineering as fe  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# cal_por_carga_esc = feed_CaO_kgh / feed_total_kgh: cociente de dos ACTION_primitiva ya registradas
# (feed_CaO_kgh, feed_total_kgh), misma naturaleza que relacion_C_Sn_carga (DERIVED-ACTION registrada).
# Se registra en caliente para que pase el chequeo anti-fuga del PLM (evaluar_plm_signos) sin tocar
# feature_engineering.py.
fe.CLASIFICACION_FEATURES.setdefault("cal_por_carga_esc", "DERIVED-ACTION")

DECISION_PATH = HERE / "e9_02_kpi_decision.json"
KG_SN_POR_PP = L.KG_SN_POR_PP_KPI  # ~512 kg Sn / pp KPI
CONTROLES = L.CONTROLES_BATCH


# =====================================================================================================
# datos compartidos
# =====================================================================================================
def cargar():
    df = L.cargar_df()
    batch = L.construir_batch(df)
    return df, batch


def ley_fin_por_disponibilidad(df: pd.DataFrame, col_masa: str | None) -> pd.DataFrame:
    """Por batch: ultimo ensayo `ley_sn_escoria_pct` de Reduccion entre las filas donde `col_masa` (si se da)
    tambien esta disponible. `col_masa=None` = sin filtro (ultimo ensayo disponible, punto de comparacion).
    Devuelve ley_fin y el indice de fila elegido (para comparar si dos criterios eligen el mismo escalon)."""
    filas = []
    for b, g in df.sort_values(["Batch", "fecha_inicio"]).groupby("Batch", sort=False):
        gr = g[g["fase_proceso"] == "Reducción"]
        cols_req = ["ley_sn_escoria_pct"] + ([col_masa] if col_masa else [])
        sub = gr.dropna(subset=cols_req)
        if len(sub) == 0:
            filas.append(dict(Batch=b, ley_fin=np.nan, idx_fin=-1, masa_fin=np.nan))
            continue
        filas.append(dict(Batch=b, ley_fin=float(sub["ley_sn_escoria_pct"].iloc[-1]), idx_fin=int(sub.index[-1]),
                          masa_fin=float(sub[col_masa].iloc[-1]) if col_masa else np.nan))
    return pd.DataFrame(filas).set_index("Batch")


def construir_extra_cal(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de cal/basicidad a nivel de BATCH (retrospectivas, DIAG/EDA -- nunca features de escalon)."""
    filas = []
    for b, g in df.sort_values(["Batch", "fecha_inicio"]).groupby("Batch", sort=False):
        gf = g[g["fase_proceso"] == "Fusión"]
        gr = g[g["fase_proceso"] == "Reducción"]
        cal_F = float(gf["feed_CaO_kgh"].fillna(0).sum())
        carga_F = float(gf["feed_total_kgh"].fillna(0).sum())
        sn_F_t = float(gf["feed_Sn_kgf"].fillna(0).sum()) / 1000.0
        cal_R = float(gr["feed_CaO_kgh"].fillna(0).sum())
        b2_f = gf["basicidad_B2"]
        irf_f = gf["ley_feo_escoria_pct"] / (gf["ley_sio2_escoria_pct"] + gf["ley_al2o3_escoria_pct"] + gf["ley_cao_escoria_pct"])
        cal_pos_R = gr["feed_CaO_kgh"].fillna(0).gt(0)
        filas.append(dict(
            Batch=b,
            cal_por_carga_F=cal_F / carga_F if carga_F > 0 else np.nan,
            cal_por_sn_F=cal_F / sn_F_t if sn_F_t > 0 else np.nan,
            cal_kg_F=cal_F, cal_kg_R=cal_R,
            b2_media_F=float(b2_f.mean()) if b2_f.notna().any() else np.nan,
            b2_fin_F=float(b2_f.dropna().iloc[-1]) if b2_f.notna().any() else np.nan,
            irf_fin_F=float(irf_f.dropna().iloc[-1]) if irf_f.notna().any() else np.nan,
            frac_escalones_cal_R_pos=float(cal_pos_R.mean()) if len(gr) else np.nan,
            n_escalones_R=len(gr),
        ))
    return pd.DataFrame(filas).set_index("Batch")


def kpi_col_activo() -> str:
    if DECISION_PATH.exists():
        d = json.loads(DECISION_PATH.read_text())
        return d.get("kpi_col", "recuperacion_refinada_pct")
    return "recuperacion_refinada_pct"


def preparar_batch(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """batch + extra de cal/basicidad + (si la decision de la etapa `kpi` fue v6) la columna kpi_v6."""
    extra = construir_extra_cal(df)
    b = batch.join(extra, how="left")
    if DECISION_PATH.exists():
        cmp_path = HERE / "e9_02_kpi_comparacion.csv"
        if cmp_path.exists():
            cmp = pd.read_csv(cmp_path).set_index("Batch")
            b = b.join(cmp[["kpi_v6", "sn_perdido_escoria_frac_v6"]], how="left")
            b["kpi_v6_next"] = b["kpi_v6"].shift(-1)
    return b


# =====================================================================================================
# etapa 1: kpi -- dependencia del trazador CaO antiguo
# =====================================================================================================
def etapa_kpi(df: pd.DataFrame, batch: pd.DataFrame):
    log("=== etapa kpi ===")
    ley_v3 = ley_fin_por_disponibilidad(df, "masa_escoria_est_kg")
    ley_v6 = ley_fin_por_disponibilidad(df, "m6_masa_kg")
    ley_libre = ley_fin_por_disponibilidad(df, None)
    cmp = ley_v3.add_suffix("_v3").join(ley_v6.add_suffix("_v6")).join(ley_libre.add_suffix("_libre"))
    cmp["fila_distinta_v3_v6"] = cmp["idx_fin_v3"] != cmp["idx_fin_v6"]
    cmp["fila_distinta_v3_libre"] = cmp["idx_fin_v3"] != cmp["idx_fin_libre"]
    cmp["dif_ley_v3_v6"] = cmp["ley_fin_v3"] - cmp["ley_fin_v6"]
    n = len(cmp)
    n_dist_v6 = int(cmp["fila_distinta_v3_v6"].sum())
    n_dist_libre = int(cmp["fila_distinta_v3_libre"].sum())
    log(f"filas donde v3 (trazador CaO) elige un escalon 'final' distinto de v6: {n_dist_v6}/{n} "
        f"({100*n_dist_v6/n:.1f}%); distinto del ultimo ensayo disponible sin filtro: {n_dist_libre}/{n} "
        f"({100*n_dist_libre/n:.1f}%)")
    log(f"dif ley_fin (v3-v6) media={cmp['dif_ley_v3_v6'].mean():.4f} pp, sd={cmp['dif_ley_v3_v6'].std():.4f}, "
        f"|dif|>0.5pp en {int((cmp['dif_ley_v3_v6'].abs()>0.5).sum())} batches")

    # kpi_v6: sustituir el termino de Sn en escoria final por R_m6_masa_fin_kg x %Sn final (v6, sin filtro v3)
    bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
    mdp = df.groupby("Batch").agg(M_t=("sn_en_metal_crudo_batch_t", "first"), D_t=("sn_en_dross_fe_batch_t", "first"),
                                  P_t=("sn_en_polvo_fundicion_batch_t", "first"))
    mdp[["M_kg", "D_kg", "P_kg"]] = mdp[["M_t", "D_t", "P_t"]] * 1000.0
    mdp["sn_out_kg"] = mdp[["M_kg", "D_kg", "P_kg"]].sum(axis=1)
    tab = mdp.join(ley_v6[["ley_fin"]].rename(columns={"ley_fin": "ley_fin_v6"})).join(batch[["R_m6_masa_fin_kg"]])
    tab["sn_esc_v6_kg"] = tab["R_m6_masa_fin_kg"] * tab["ley_fin_v6"] / 100.0
    tab["kpi_v6"] = 100 * tab["M_kg"] / (tab["sn_out_kg"] + tab["sn_esc_v6_kg"])
    tab["sn_perdido_escoria_frac_v6"] = tab["sn_esc_v6_kg"] / (tab["sn_out_kg"] + tab["sn_esc_v6_kg"])

    out = bal[["recuperacion_refinada_pct", "sn_escoria_final_balance_kg", "sn_perdido_escoria_frac"]].join(
        tab[["kpi_v6", "sn_esc_v6_kg", "sn_perdido_escoria_frac_v6", "R_m6_masa_fin_kg"]]).join(
        cmp[["ley_fin_v3", "ley_fin_v6", "dif_ley_v3_v6", "fila_distinta_v3_v6"]])
    out["dif_pp"] = out["kpi_v6"] - out["recuperacion_refinada_pct"]
    out = out.replace([np.inf, -np.inf], np.nan)

    from scipy.stats import spearmanr, pearsonr
    d = out[["recuperacion_refinada_pct", "kpi_v6"]].dropna()
    rho, p_s = spearmanr(d["recuperacion_refinada_pct"], d["kpi_v6"])
    r, p_r = pearsonr(d["recuperacion_refinada_pct"], d["kpi_v6"])
    mae_pp = float(out["dif_pp"].abs().mean())
    q_o = pd.qcut(d["recuperacion_refinada_pct"], 4, labels=False, duplicates="drop")
    q_v6 = pd.qcut(d["kpi_v6"], 4, labels=False, duplicates="drop")
    frac_reclas = float((q_o != q_v6).mean())
    log(f"kpi original vs kpi_v6 (n={len(d)}): rho={rho:.4f} (p={p_s:.1e}) pearson={r:.4f} "
        f"MAE={mae_pp:.4f} pp, max|dif|={out['dif_pp'].abs().max():.3f} pp, reclasificados de cuartil={frac_reclas:.1%}")

    # decision: si la dependencia (filtro v3) y la diferencia numerica son chicas, el KPI original ya es
    # aceptable (la dependencia del trazador es solo en la SELECCION de fila, marginal); si no, usar kpi_v6.
    dependencia_material = (n_dist_v6 / n > 0.05) or (mae_pp > 0.05) or (rho < 0.995)
    decision = "kpi_v6" if dependencia_material else "recuperacion_refinada_pct"
    kpi_col = "kpi_v6" if decision == "kpi_v6" else "recuperacion_refinada_pct"
    log(f"DECISION: usar '{kpi_col}' en las etapas siguientes "
        f"(dependencia material del trazador v3 = {dependencia_material})")

    out.reset_index().to_csv(HERE / "e9_02_kpi_comparacion.csv", index=False)
    DECISION_PATH.write_text(json.dumps({
        "kpi_col": kpi_col, "rho": rho, "pearson": r, "mae_pp": mae_pp, "frac_reclasificados": frac_reclas,
        "n_fila_distinta_v3_v6": n_dist_v6, "n_batches": n, "dependencia_material": dependencia_material,
    }, indent=2))
    log("guardado e9_02_kpi_comparacion.csv y e9_02_kpi_decision.json")


# =====================================================================================================
# etapa 2: batch -- cal/carga, cal/Sn, B2 vs KPI y canales
# =====================================================================================================
def boot_coef_batch(d0: pd.DataFrame, y: str, x: str, controles: list[str], n_boot: int = 500, seed: int = 0) -> dict:
    cols = [y, x] + controles
    d = d0[cols].dropna()
    if len(d) < 30:
        return dict(coef=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=len(d))
    Xc = [x] + controles
    Z = (d[Xc] - d[Xc].mean()) / d[Xc].std()
    X = sm.add_constant(Z)
    base = sm.OLS(d[y], X).fit(cov_type="HC3")
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        dd = d.iloc[idx]
        sd = dd[Xc].std()
        if (sd < 1e-9).any():
            continue
        Zb = sm.add_constant((dd[Xc] - dd[Xc].mean()) / sd)
        try:
            boots.append(sm.OLS(dd[y].to_numpy(), Zb.to_numpy()).fit().params[1])
        except Exception:
            continue
    boots = np.array(boots)
    lo, hi = (np.nan, np.nan) if len(boots) < n_boot * 0.5 else (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return dict(coef=float(base.params[x]), p=float(base.pvalues[x]), ci_lo_hc3=float(base.conf_int().loc[x, 0]),
               ci_hi_hc3=float(base.conf_int().loc[x, 1]), ci_lo=lo, ci_hi=hi, n=n)


def quintiles_ic(d0: pd.DataFrame, x: str, ys: list[str], n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    d = d0[[x] + ys].dropna(subset=[x])
    if len(d) < 50:
        return pd.DataFrame()
    q = pd.qcut(d[x], 5, labels=False, duplicates="drop")
    filas = []
    rng = np.random.default_rng(seed)
    for y in ys:
        for qi in sorted(q.dropna().unique()):
            vals = d.loc[q == qi, y].dropna().to_numpy()
            if len(vals) < 5:
                continue
            boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
            filas.append(dict(variable=y, quintil=int(qi), media=float(np.mean(vals)),
                              ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                              n=len(vals), x_media=float(d.loc[q == qi, x].mean())))
    return pd.DataFrame(filas)


def etapa_batch(df: pd.DataFrame, batch: pd.DataFrame):
    log("=== etapa batch ===")
    b = preparar_batch(df, batch)
    kpi_col = kpi_col_activo()
    kpi_next_col = f"{kpi_col}_next" if f"{kpi_col}_next" in b.columns else "recuperacion_refinada_pct_next"
    log(f"KPI activo: {kpi_col} (next: {kpi_next_col})")

    palancas_cal = ["cal_por_carga_F", "cal_por_sn_F", "b2_media_F", "b2_fin_F"]
    targets = [kpi_col, "f_dross", "f_polvo", "sn_perdido_escoria_frac", "R_sum_m6_ln_feo_ret",
              "R_sum_m6_ln_sn_dep", "T_media_R", kpi_next_col]
    log(f"n batches con cal_por_carga_F: {b['cal_por_carga_F'].notna().sum()}; "
        f"P25/mediana/P75 = {b['cal_por_carga_F'].quantile([.25,.5,.75]).round(4).to_dict()}")

    filas = []
    for tgt in targets:
        if tgt not in b.columns:
            continue
        for pal in palancas_cal:
            r = L.bateria_batch(b, pal, kpi=tgt, controles=CONTROLES)
            r["target"] = tgt
            filas.append(r)
    reg = pd.DataFrame(filas)
    reg.to_csv(HERE / "e9_02_batch_regresiones.csv", index=False)
    log(f"e9_02_batch_regresiones.csv {reg.shape}")

    # cuadratico de cal_por_carga_F sobre KPI y R_sum_m6_ln_feo_ret/sn_dep (optimo interior?)
    filas_q = []
    for tgt in [kpi_col, "R_sum_m6_ln_feo_ret", "R_sum_m6_ln_sn_dep"]:
        cols = [tgt, "cal_por_carga_F"] + CONTROLES
        d = b[cols].dropna()
        if len(d) < 30:
            continue
        Xc = ["cal_por_carga_F"] + CONTROLES
        Z = (d[Xc] - d[Xc].mean()) / d[Xc].std()
        Z["cal_por_carga_F_sq"] = Z["cal_por_carga_F"] ** 2
        for muestra, sub_idx in [("dev", b.loc[d.index, "es_dev"]), ("total", pd.Series(True, index=d.index))]:
            sel = sub_idx[sub_idx].index if muestra == "dev" else d.index
            if len(sel) < 30:
                continue
            X = sm.add_constant(Z.loc[sel, Xc + ["cal_por_carga_F_sq"]])
            res = sm.OLS(d.loc[sel, tgt], X).fit(cov_type="HC3")
            b1, b2c = res.params["cal_por_carga_F"], res.params["cal_por_carga_F_sq"]
            mu, sd = d["cal_por_carga_F"].mean(), d["cal_por_carga_F"].std()
            vertice = mu + sd * (-b1 / (2 * b2c)) if abs(b2c) > 1e-9 else np.nan
            filas_q.append(dict(target=tgt, muestra=muestra, coef_lin=b1, coef_sq=b2c,
                                p_sq=float(res.pvalues["cal_por_carga_F_sq"]),
                                ci_lo_sq=float(res.conf_int().loc["cal_por_carga_F_sq", 0]),
                                ci_hi_sq=float(res.conf_int().loc["cal_por_carga_F_sq", 1]),
                                vertice_cal_por_carga=vertice, n=len(sel)))
    quad = pd.DataFrame(filas_q)
    quad.to_csv(HERE / "e9_02_cuadratico.csv", index=False)
    log(f"e9_02_cuadratico.csv {quad.shape}")
    if len(quad):
        log(quad.to_string())

    # dosis-respuesta por quintiles de cal_por_carga_F (DEV)
    dq = quintiles_ic(b[b["es_dev"]], "cal_por_carga_F",
                      [kpi_col, "f_dross", "f_polvo", "sn_perdido_escoria_frac", "R_sum_m6_ln_feo_ret",
                       "R_sum_m6_ln_sn_dep", "T_media_R"])
    dq.to_csv(HERE / "e9_02_dosis_respuesta.csv", index=False)
    log(f"e9_02_dosis_respuesta.csv {dq.shape}")

    # bootstrap por batch del coeficiente principal (cal_por_carga_F -> KPI), DEV y total
    filas_boot = []
    for muestra, sub in [("dev", b[b["es_dev"]]), ("total", b)]:
        for pal in palancas_cal:
            r = boot_coef_batch(sub, kpi_col, pal, CONTROLES, n_boot=500, seed=0)
            r.update(dict(muestra=muestra, palanca=pal, target=kpi_col))
            filas_boot.append(r)
    bootdf = pd.DataFrame(filas_boot)
    bootdf.to_csv(HERE / "e9_02_boot_coef.csv", index=False)
    log(f"e9_02_boot_coef.csv {bootdf.shape}")
    log(bootdf[["muestra", "palanca", "coef", "ci_lo", "ci_hi", "p", "n"]].to_string())


# =====================================================================================================
# etapa 3: escalon_F -- PLM con signos, cal como palanca adicional
# =====================================================================================================
PALANCAS_CAL_F = ["tasa_feed_CaO_kg_min", "relacion_C_Sn_carga", "tasa_gn_nm3_min",
                  "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]

SIGNOS_CAL_F = {
    "m6_ln_sn_dep": dict(L.SIGNOS[("Fusión", "sn_dep")], tasa_feed_CaO_kg_min=0),
    "d_temperatura_horno_celsius": {"tasa_feed_CaO_kg_min": -1, "relacion_C_Sn_carga": 0, "tasa_gn_nm3_min": +1,
                                    "exceso_o2_combustion_pct": 0, "tasa_aire_nm3_min": -1},
    "d_basicidad_B2": {"tasa_feed_CaO_kg_min": +1, "relacion_C_Sn_carga": 0, "tasa_gn_nm3_min": 0,
                       "exceso_o2_combustion_pct": 0, "tasa_aire_nm3_min": 0},
    "v5_d_lnirf": {"tasa_feed_CaO_kg_min": -1, "relacion_C_Sn_carga": 0, "tasa_gn_nm3_min": -1,
                  "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": 0},
}


def preparar_dF(df: pd.DataFrame) -> pd.DataFrame:
    dF = L.base_fase(df, "Fusión").copy()
    dF["cal_por_carga_esc"] = dF["feed_CaO_kgh"] / dF["feed_total_kgh"]
    dF["d_basicidad_B2"] = dF["basicidad_B2"] - dF["basicidad_B2_prev"]
    return dF


def estabilidad_tercios(d_dev: pd.DataFrame, estado: list[str], palancas: list[str], target: str,
                        signos: dict, n_terciles: int = 3) -> pd.DataFrame:
    import modelo_predictivo_v5 as mp5
    terc = L.tercios_cronologicos(d_dev, n_terciles)
    filas = []
    for t in sorted(terc.dropna().unique()):
        sub = d_dev[terc == t].reset_index(drop=True)
        if len(sub) < 80:
            continue
        m = mp5.ModeloPLMSignos(estado, palancas, target, signos).fit(sub, n_splits=3, n_boot=150, seed=42)
        tabla = m.tabla_theta(signos)
        tabla["tercio"] = int(t)
        tabla["n"] = len(sub)
        filas.append(tabla.reset_index().rename(columns={"index": "palanca_idx"}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def etapa_escalon_F(df: pd.DataFrame):
    log("=== etapa escalon_F ===")
    dF = preparar_dF(df)
    L.asegurar_seguras(PALANCAS_CAL_F)

    combos = [
        ("m6_ln_sn_dep", [c for c in L.ESTADO[("Fusión", "sn_dep")] if c != "tasa_feed_CaO_kg_min"]),
        ("d_temperatura_horno_celsius", [c for c in L.ESTADO[("Fusión", "dT")] if c != "tasa_feed_CaO_kg_min"]),
        ("d_basicidad_B2", [c for c in L.ESTADO[("Fusión", "d_lnirf")] if c != "tasa_feed_CaO_kg_min"]),
        ("v5_d_lnirf", [c for c in L.ESTADO[("Fusión", "d_lnirf")] if c != "tasa_feed_CaO_kg_min"]),
    ]
    metricas, thetas, estab = [], [], []
    for target, estado in combos:
        L.asegurar_seguras(estado)
        log(f"  PLM target={target} (estado {len(estado)}, palancas {len(PALANCAS_CAL_F)})")
        signos = SIGNOS_CAL_F[target]
        met, tabla, _ = L.evaluar_plm_signos(dF, estado, PALANCAS_CAL_F, target, n_boot=300, signo_teorico=signos)
        met["signos"] = json.dumps(signos)
        metricas.append(met)
        thetas.append(tabla)
        log(f"    r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f} n_dev={met['n_dev']}")
        log(tabla[["palanca", "theta_1sd", "ci_lo_1sd", "ci_hi_1sd", "p" if "p" in tabla.columns else "significativo"]].to_string())

        # estabilidad por tercios (solo sobre DEV, misma preparacion que evaluar_plm_signos)
        import modelo_prescriptivo as mp
        dev, _ = mp.split_dev_lockbox(dF)
        need = list(dict.fromkeys(estado + PALANCAS_CAL_F + [target, "Batch", "fecha_inicio", "orden_escalon_fase"]))
        d_dev = dF.loc[dF["Batch"].isin(dev), need].dropna().reset_index(drop=True)
        tabla_est = estabilidad_tercios(d_dev, estado, PALANCAS_CAL_F, target, signos)
        if len(tabla_est):
            tabla_est["target"] = target
            estab.append(tabla_est)

    pd.DataFrame(metricas).to_csv(HERE / "e9_02_plm_fusion.csv", index=False)
    pd.concat(thetas, ignore_index=True).to_csv(HERE / "e9_02_theta_fusion.csv", index=False)
    if estab:
        pd.concat(estab, ignore_index=True).to_csv(HERE / "e9_02_estabilidad_fusion.csv", index=False)
    log("guardado e9_02_plm_fusion.csv, e9_02_theta_fusion.csv, e9_02_estabilidad_fusion.csv")


# =====================================================================================================
# etapa 4: escalon_R -- distribucion de la cal en Reduccion y PLM si hay soporte
# =====================================================================================================
def etapa_escalon_R(df: pd.DataFrame):
    log("=== etapa escalon_R ===")
    dR = L.base_fase(df, "Reducción").copy()
    cal = dR["feed_CaO_kgh"].fillna(0)
    frac_pos = float((cal > 0).mean())
    log(f"n escalones Reduccion: {len(dR)}; fraccion con cal>0: {frac_pos:.3f} (n={int((cal>0).sum())}); "
        f"cal>0 describe: {cal[cal>0].describe().round(3).to_dict()}")
    resumen = pd.DataFrame([dict(n_escalones=len(dR), frac_cal_pos=frac_pos, n_cal_pos=int((cal > 0).sum()),
                                 media_cal_kgh_pos=float(cal[cal > 0].mean()) if frac_pos > 0 else np.nan)])
    resumen.to_csv(HERE / "e9_02_distribucion_cal_R.csv", index=False)

    if frac_pos < 0.05 or (cal > 0).sum() < 100:
        log("soporte insuficiente (<5% de escalones o <100 filas con cal>0): no se ajusta PLM de Reduccion.")
        return

    combos = [
        ("m6_ln_sn_dep", L.ESTADO[("Reducción", "sn_dep")], L.PALANCAS[("Reducción", "sn_dep")] + ["tasa_feed_CaO_kg_min"],
        dict(L.SIGNOS[("Reducción", "sn_dep")], tasa_feed_CaO_kg_min=0)),
        ("m6_ln_feo_ret", L.ESTADO[("Reducción", "feo_ret")], L.PALANCAS[("Reducción", "feo_ret")] + ["tasa_feed_CaO_kg_min"],
        dict(L.SIGNOS[("Reducción", "feo_ret")], tasa_feed_CaO_kg_min=0)),
    ]
    metricas, thetas = [], []
    for target, estado, palancas, signos in combos:
        L.asegurar_seguras(estado); L.asegurar_seguras(palancas)
        log(f"  PLM target={target} (estado {len(estado)}, palancas {len(palancas)})")
        met, tabla, _ = L.evaluar_plm_signos(dR, estado, palancas, target, n_boot=300, signo_teorico=signos)
        metricas.append(met); thetas.append(tabla)
        log(f"    r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f} n_dev={met['n_dev']}")
    pd.DataFrame(metricas).to_csv(HERE / "e9_02_plm_reduccion.csv", index=False)
    pd.concat(thetas, ignore_index=True).to_csv(HERE / "e9_02_theta_reduccion.csv", index=False)
    log("guardado e9_02_plm_reduccion.csv, e9_02_theta_reduccion.csv")


# =====================================================================================================
# etapa 5: basicidad -- optimo interior al cierre de Fusion
# =====================================================================================================
def etapa_basicidad(df: pd.DataFrame, batch: pd.DataFrame):
    log("=== etapa basicidad ===")
    b = preparar_batch(df, batch)
    kpi_col = kpi_col_activo()
    controles_ext = CONTROLES + ["T_media_R", "F_lnirf_F0"]

    filas = []
    for var in ["b2_fin_F", "irf_fin_F"]:
        for tgt in [kpi_col, "R_sum_m6_ln_feo_ret", "R_sum_m6_ln_sn_dep"]:
            cols = [tgt, var] + controles_ext
            d = b[cols].dropna()
            if len(d) < 40:
                continue
            Xc = [var] + controles_ext
            Z = (d[Xc] - d[Xc].mean()) / d[Xc].std()
            Z[f"{var}_sq"] = Z[var] ** 2
            for muestra in ("dev", "total"):
                sel = d.index[b.loc[d.index, "es_dev"]] if muestra == "dev" else d.index
                if len(sel) < 40:
                    continue
                X = sm.add_constant(Z.loc[sel, Xc + [f"{var}_sq"]])
                res = sm.OLS(d.loc[sel, tgt], X).fit(cov_type="HC3")
                b1, b2c = res.params[var], res.params[f"{var}_sq"]
                mu, sd = d[var].mean(), d[var].std()
                vert = mu + sd * (-b1 / (2 * b2c)) if abs(b2c) > 1e-9 else np.nan
                filas.append(dict(variable=var, target=tgt, muestra=muestra, coef_lin=b1, coef_sq=b2c,
                                  p_sq=float(res.pvalues[f"{var}_sq"]),
                                  ci_lo_sq=float(res.conf_int().loc[f"{var}_sq", 0]),
                                  ci_hi_sq=float(res.conf_int().loc[f"{var}_sq", 1]),
                                  vertice=vert, n=len(sel)))
    quad = pd.DataFrame(filas)
    quad.to_csv(HERE / "e9_02_basicidad_cuadratico.csv", index=False)
    log(f"e9_02_basicidad_cuadratico.csv {quad.shape}")
    if len(quad):
        log(quad.to_string())

    # binned deciles (DEV) con IC bootstrap
    filas_dec = []
    rng = np.random.default_rng(0)
    for var in ["b2_fin_F", "irf_fin_F"]:
        d = b.loc[b["es_dev"], [var, kpi_col, "R_sum_m6_ln_feo_ret", "R_sum_m6_ln_sn_dep"]].dropna(subset=[var])
        if len(d) < 60:
            continue
        dec = pd.qcut(d[var], 10, labels=False, duplicates="drop")
        for tgt in [kpi_col, "R_sum_m6_ln_feo_ret", "R_sum_m6_ln_sn_dep"]:
            for di in sorted(dec.dropna().unique()):
                vals = d.loc[dec == di, tgt].dropna().to_numpy()
                if len(vals) < 5:
                    continue
                boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(400)]
                filas_dec.append(dict(variable=var, target=tgt, decil=int(di), x_media=float(d.loc[dec == di, var].mean()),
                                      media=float(np.mean(vals)), ci_lo=float(np.percentile(boots, 2.5)),
                                      ci_hi=float(np.percentile(boots, 97.5)), n=len(vals)))
    decdf = pd.DataFrame(filas_dec)
    decdf.to_csv(HERE / "e9_02_basicidad_deciles.csv", index=False)
    log(f"e9_02_basicidad_deciles.csv {decdf.shape}")


# =====================================================================================================
# etapa 6: resumen
# =====================================================================================================
def etapa_resumen():
    log("=== etapa resumen ===")

    def _try_csv(name):
        p = HERE / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    decision = json.loads(DECISION_PATH.read_text()) if DECISION_PATH.exists() else {}
    kpi_col = decision.get("kpi_col", "recuperacion_refinada_pct")
    boot = _try_csv("e9_02_boot_coef.csv")
    theta_f = _try_csv("e9_02_theta_fusion.csv")
    plm_f = _try_csv("e9_02_plm_fusion.csv")
    quad = _try_csv("e9_02_cuadratico.csv")
    reg = _try_csv("e9_02_batch_regresiones.csv")
    dist_r = _try_csv("e9_02_distribucion_cal_R.csv")
    bas_quad = _try_csv("e9_02_basicidad_cuadratico.csv")

    filas = []
    if len(boot):
        dev = boot[(boot["muestra"] == "dev") & (boot["palanca"] == "cal_por_carga_F")]
        if len(dev):
            r = dev.iloc[0]
            sig = bool(r["p"] < 0.05) if pd.notna(r["p"]) else False
            filas.append(dict(item="batch: cal_por_carga_F -> KPI (DEV)", coef_pp_sd=r["coef"], p=r["p"],
                              ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], significativo_p05=sig,
                              kg_sn_por_batch=r["coef"] * KG_SN_POR_PP if kpi_col != "R_sum_m6_ln_feo_ret" else np.nan))
    if len(theta_f):
        cal_rows = theta_f[theta_f["palanca"] == "tasa_feed_CaO_kg_min"]
        for _, r in cal_rows.iterrows():
            filas.append(dict(item=f"escalon_F: cal -> {r['target']}", coef_pp_sd=r.get("theta_1sd", np.nan),
                              p=np.nan, ci_lo=r.get("ci_lo_1sd", np.nan), ci_hi=r.get("ci_hi_1sd", np.nan),
                              significativo_p05=bool(r.get("significativo", False)), kg_sn_por_batch=np.nan))
    resumen = pd.DataFrame(filas)
    resumen.to_csv(HERE / "e9_02_resumen.csv", index=False)
    log(f"e9_02_resumen.csv {resumen.shape}")
    log(resumen.to_string() if len(resumen) else "resumen vacio (faltan etapas previas)")

    # veredicto segun criterio del diseno
    p_ok = False
    signo_ok = False
    rango_ok = False
    if len(boot):
        dev = boot[(boot["muestra"] == "dev") & (boot["palanca"] == "cal_por_carga_F")]
        if len(dev) and pd.notna(dev.iloc[0]["p"]):
            p_ok = bool(dev.iloc[0]["p"] < 0.05)
    cal_signo_b2 = np.nan
    if len(theta_f):
        r = theta_f[(theta_f["palanca"] == "tasa_feed_CaO_kg_min") & (theta_f["target"] == "d_basicidad_B2")]
        if len(r):
            cal_signo_b2 = float(r.iloc[0].get("theta_unidad", np.nan))
            signo_ok = bool(r.iloc[0].get("significativo", False)) and cal_signo_b2 > 0
    veredicto = "PALANCA DE PLAN" if (p_ok and signo_ok) else "NO SE ADOPTA COMO PALANCA DE PLAN (evidencia insuficiente)"
    log(f"VEREDICTO preliminar: {veredicto} (p_batch<0.05={p_ok}, señal mecánica ΔB2 confirmada={signo_ok})")

    resumen_txt = {
        "kpi_usado": kpi_col, "decision_kpi": decision,
        "veredicto": veredicto, "p_batch_ok": p_ok, "senal_b2_ok": signo_ok, "theta_cal_sobre_dB2": cal_signo_b2,
    }
    (HERE / "e9_02_veredicto.json").write_text(json.dumps(resumen_txt, indent=2, default=str))
    log("guardado e9_02_veredicto.json (el memo e9_02_resultados.md se redacta a mano con estos numeros)")


# =====================================================================================================
def main():
    etapa = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"cargando datos (etapa={etapa})...")
    df, batch = cargar()
    log(f"df {df.shape} batch {batch.shape}")

    if etapa in ("kpi", "all"):
        etapa_kpi(df, batch)
    if etapa in ("batch", "all"):
        etapa_batch(df, batch)
    if etapa in ("escalon_F", "all"):
        etapa_escalon_F(df)
    if etapa in ("escalon_R", "all"):
        etapa_escalon_R(df)
    if etapa in ("basicidad", "all"):
        etapa_basicidad(df, batch)
    if etapa in ("resumen", "all"):
        etapa_resumen()

    log("FIN")


if __name__ == "__main__":
    main()
