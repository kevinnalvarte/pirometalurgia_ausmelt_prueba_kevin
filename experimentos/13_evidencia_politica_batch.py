"""Experimento 13 -- evidencia off-policy de que la politica recomendada por v3 se asocia a
mayor rendimiento de batch (ver experimentos/ITERACION_4_diseno.md).

Idea: generar recomendaciones CROSS-FITTED para los 362 batches (el sistema que recomienda a
un batch nunca lo vio en su entrenamiento: 5 folds sobre DEV + sistema entrenado en todo DEV
para el lockbox), medir cuanto se ALEJO cada batch de esa recomendacion (adherencia historica,
"dist_total") y correlacionar/regresionar esa distancia contra el rendimiento observado del
batch. Hipotesis: mas lejos de la recomendacion => peor rendimiento (dist_total negativo).

Salidas:
    13_recomendaciones_oof_escalones.csv  -- detalle por escalon (reporte v3 + fold + es_lockbox)
    13_recomendaciones_oof_por_batch.csv  -- una fila por batch (adherencia + KPIs + contexto)
    13_evidencia_regresiones.csv          -- todas las regresiones OLS (dev/lockbox/total, reales/placebo)
    13_log.txt                            -- log de ejecucion (stdout+stderr redirigido por el caller)
    figs_13/*.png                         -- scatter, dosis-respuesta, histogramas de dir_*
"""
from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402

SALIDA = RAIZ / "experimentos"
FIGS = SALIDA / "figs_13"
FIGS.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 300)

N_CANDIDATOS = 150
SEMILLA = 0
N_JOBS = 5
N_FOLDS = 5
N_PERM = 1000

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# =============================================================================
# 1) Recomendaciones cross-fitted para TODOS los batches
# =============================================================================

def _generar_reporte(sistema, df: pd.DataFrame, batch_id: str, fold: int, es_lockbox: bool):
    try:
        rep = mp3.reporte_recomendaciones_batch_v3(sistema, df, batch_id, n_candidatos=N_CANDIDATOS, semilla=SEMILLA)
        if rep.empty:
            return None, (batch_id, "reporte vacio (sin escalones con insumos completos)")
        rep = rep.copy()
        rep["fold"] = fold
        rep["es_lockbox"] = es_lockbox
        return rep, None
    except Exception as e:  # noqa: BLE001
        return None, (batch_id, f"{type(e).__name__}: {e}")


def _folds_dev(df):
    dev, lockbox = mp.split_dev_lockbox(df)
    dev_list = sorted(dev)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(dev_list))
    fold_de_batch = {dev_list[i]: int(pos % N_FOLDS) for pos, i in enumerate(perm)}
    folds = {k: sorted(b for b, f in fold_de_batch.items() if f == k) for k in range(N_FOLDS)}
    return dev, lockbox, folds


def correr_fold(etiqueta: str):
    """Etapa 1 (por proceso independiente): genera los reportes de un fold ('0'..'4') o del
    lockbox ('lockbox') de forma SECUENCIAL y guarda 13_parcial_<etiqueta>.csv.
    (La version joblib/loky se colgaba en Windows: sin progreso en 3 h)."""
    log("cargando dataset")
    df = pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl")
    dev, lockbox, folds = _folds_dev(df)
    t_fold = time.time()
    if etiqueta == "lockbox":
        sistema = mp3.entrenar_sistema_v3(df)
        batches, k, es_lb = sorted(lockbox), -1, True
    else:
        k = int(etiqueta)
        train_batches = dev - set(folds[k])
        sistema = mp3.entrenar_sistema_v3(df[df["Batch"].isin(train_batches)], pct_lockbox=0.0)
        batches, es_lb = folds[k], False
    log(f"{etiqueta}: sistema entrenado ({time.time()-t_fold:.1f}s); {len(batches)} batches")
    reportes, errores = [], []
    for i, b in enumerate(batches):
        rep, err = _generar_reporte(sistema, df, b, k, es_lb)
        if err is not None:
            errores.append(err); log(f"  ERROR {b}: {err}")
        if rep is not None:
            reportes.append(rep)
        if (i + 1) % 5 == 0:
            log(f"  {etiqueta}: {i+1}/{len(batches)} batches ({time.time()-t_fold:.0f}s)")
    esc = pd.concat(reportes, ignore_index=True) if reportes else pd.DataFrame()
    esc.to_csv(SALIDA / f"13_parcial_{etiqueta}.csv", index=False)
    log(f"{etiqueta}: LISTO {len(reportes)} ok, {len(errores)} fallidos, {time.time()-t_fold:.0f}s")


def main():
    """Etapa 2: une los parciales de todos los folds + lockbox y corre metricas/analisis/figuras."""
    log("cargando dataset")
    df = pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl")
    dev, lockbox, folds = _folds_dev(df)
    errores = []
    parciales = [SALIDA / f"13_parcial_{e}.csv" for e in [str(k) for k in range(N_FOLDS)] + ["lockbox"]]
    faltan = [p.name for p in parciales if not p.exists()]
    if faltan:
        raise SystemExit(f"faltan parciales: {faltan}")
    escalones = pd.concat([pd.read_csv(p) for p in parciales], ignore_index=True)
    escalones.to_csv(SALIDA / "13_recomendaciones_oof_escalones.csv", index=False)
    log("escalones guardados", escalones.shape, "batches con reporte:", escalones["Batch"].nunique())

    # =========================================================================
    # 2) Metricas por batch
    # =========================================================================
    log("calculando sd_DEV por palanca/fase (para normalizar distancias)")
    df_base = mp.dataset_base_modelo(df)
    sd_por_fase_palanca = {}
    for fase in mp3.FASES:
        sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev)]
        for palanca in mp3.ACCIONES_OPTIMIZABLES_V3[fase]:
            sd = float(sub[palanca].std())
            sd_por_fase_palanca[(fase, palanca)] = sd if sd > 1e-9 else np.nan
            log(f"  sd_DEV[{fase}][{palanca}] = {sd:.4g}")

    filas_batch = []
    for batch_id, rep in escalones.groupby("Batch"):
        resumen = mp3.resumen_reporte_batch_v3(rep)
        fila = dict(resumen)
        fila["Batch"] = batch_id
        fila["fold"] = int(rep["fold"].iloc[0])
        fila["es_lockbox"] = bool(rep["es_lockbox"].iloc[0])

        dist_cols, dir_cols = {}, {}
        sin_cambio_flags = []
        for fase in mp3.FASES:
            sub_f = rep.loc[rep["fase"] == fase]
            if sub_f.empty:
                continue
            cambio_fase = pd.Series(False, index=sub_f.index)
            for palanca in mp3.ACCIONES_OPTIMIZABLES_V3[fase]:
                sd = sd_por_fase_palanca.get((fase, palanca), np.nan)
                hist = sub_f[f"hist__{palanca}"]
                rec = sub_f[f"rec__{palanca}"]
                if pd.isna(sd):
                    continue
                dist_esc = (hist - rec).abs() / sd
                dir_esc = (rec - hist) / sd
                dist_cols[f"dist_{fase}_{palanca}"] = float(dist_esc.mean())
                dir_cols[f"dir_{fase}_{palanca}"] = float(dir_esc.mean())
                cambio_fase = cambio_fase | (hist != rec)
            sin_cambio_flags.append(~cambio_fase)
        fila.update(dist_cols)
        fila.update(dir_cols)
        dist_vals = [v for k, v in dist_cols.items()]
        fila["dist_total"] = float(np.mean(dist_vals)) if dist_vals else np.nan
        for fase in mp3.FASES:
            vals_f = [v for k, v in dist_cols.items() if k.startswith(f"dist_{fase}_")]
            fila[f"dist_{'F' if fase == 'Fusión' else 'R'}"] = float(np.mean(vals_f)) if vals_f else np.nan
        if sin_cambio_flags:
            sin_cambio_all = pd.concat(sin_cambio_flags)
            fila["pct_escalones_sin_cambio"] = float(sin_cambio_all.mean())
        else:
            fila["pct_escalones_sin_cambio"] = np.nan

        # --- KPIs y contexto del batch (desde el dataset completo, no solo escalones OOF) ---
        g = df.loc[df["Batch"] == batch_id]
        g_fus = g.loc[g["fase_proceso"] == "Fusión"]
        m, d_, p_ = g["sn_en_metal_crudo_batch_t"].iloc[0], g["sn_en_dross_fe_batch_t"].iloc[0], g["sn_en_polvo_fundicion_batch_t"].iloc[0]
        tot = m + d_ + p_
        fila["rendimiento_proxy_batch"] = float(g["rendimiento_proxy_batch"].iloc[0])
        fila["rendimiento_ha_batch"] = float(g["rendimiento_ha_batch"].iloc[0]) if "rendimiento_ha_batch" in g.columns else np.nan
        fila["frac_sn_metal"] = float(m / tot) if tot > 0 else np.nan
        fila["frac_sn_dross"] = float(d_ / tot) if tot > 0 else np.nan
        fila["frac_sn_polvo"] = float(p_ / tot) if tot > 0 else np.nan
        fila["ley_sn_conc_batch_pct"] = float(g["ley_sn_conc_batch_pct"].iloc[0])
        fila["ley_sn_carga_batch_pct"] = float(g["ley_sn_carga_batch_pct"].iloc[0])
        fila["espesor_ladrillo_norm_mm"] = float(g["espesor_ladrillo_norm_mm"].mean())
        fila["sn_cargado_batch_kg"] = float(g["feed_Sn_kgf"].sum())
        fila["temperatura_media_fusion"] = float(g_fus["temperatura_horno_celsius"].mean()) if len(g_fus) else np.nan
        fila["frac_carga_secundaria_media_fusion"] = float(g_fus["frac_carga_secundaria"].mean()) if len(g_fus) else np.nan
        fila["fecha_inicio_batch"] = g["fecha_inicio"].min()
        fila["metal_equivalente_kg_por_t_cargada"] = (
            1000 * fila.get("metal_equivalente_kg", np.nan) / fila["sn_cargado_batch_kg"] if fila.get("sn_cargado_batch_kg", 0) else np.nan
        )
        filas_batch.append(fila)

    por_batch = pd.DataFrame(filas_batch).sort_values("Batch").reset_index(drop=True)
    por_batch.to_csv(SALIDA / "13_recomendaciones_oof_por_batch.csv", index=False)
    log("por_batch guardado", por_batch.shape)

    # =========================================================================
    # 3) Analisis estadistico
    # =========================================================================
    log("=== analisis estadistico ===")
    resultados_reg = analisis_estadistico(por_batch, sd_por_fase_palanca)
    resultados_reg.to_csv(SALIDA / "13_evidencia_regresiones.csv", index=False)
    log("regresiones guardadas", resultados_reg.shape)

    log("=== figuras ===")
    hacer_figuras(por_batch)

    log(f"LISTO. n_batches_con_reporte={por_batch.shape[0]} n_errores={len(errores)}")
    log(f"tiempo total: {time.time()-T0:.1f}s")


# =============================================================================
# helpers de estadistica
# =============================================================================

DIST_VARS = ["dist_total", "dist_F", "dist_R"]
KPI_VARS = ["rendimiento_proxy_batch", "frac_sn_dross", "frac_sn_polvo"]
CONTROLES = ["ley_sn_conc_batch_pct", "ley_sn_carga_batch_pct", "espesor_ladrillo_norm_mm",
             "sn_cargado_batch_kg", "temperatura_media_fusion", "frac_carga_secundaria_media_fusion"]


def _dist_cols_individuales(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("dist_") and c not in ("dist_total", "dist_F", "dist_R")
            and not c.startswith("dist_total_placebo")]


def _placebo_dist_total(por_batch: pd.DataFrame, escalones: pd.DataFrame | None, sd_por_fase_palanca: dict, seed: int = 0) -> pd.Series:
    """No usado directamente (ver generar_placebo_por_batch); mantiene firma para claridad."""
    raise NotImplementedError


def generar_placebo(escalones: pd.DataFrame, sd_por_fase_palanca: dict, seed: int = 0) -> pd.DataFrame:
    """Para cada escalon, genera una accion 'recomendada' aleatoria uniforme en P1-P99 HISTORICO
    (calculado sobre DEV, mismas primitivas) de cada palanca optimizable de esa fase, y calcula
    dist_total_placebo por batch con la MISMA normalizacion (sd_DEV) que la politica real."""
    rng = np.random.default_rng(seed)
    limites = {}
    for fase in mp3.FASES:
        for palanca in mp3.ACCIONES_OPTIMIZABLES_V3[fase]:
            col = f"hist__{palanca}"
            vals = escalones.loc[escalones["fase"] == fase, col].dropna()
            limites[(fase, palanca)] = (float(vals.quantile(0.01)), float(vals.quantile(0.99)))

    filas = []
    for batch_id, rep in escalones.groupby("Batch"):
        dist_cols = {}
        for fase in mp3.FASES:
            sub_f = rep.loc[rep["fase"] == fase]
            if sub_f.empty:
                continue
            for palanca in mp3.ACCIONES_OPTIMIZABLES_V3[fase]:
                sd = sd_por_fase_palanca.get((fase, palanca), np.nan)
                if pd.isna(sd):
                    continue
                lo, hi = limites[(fase, palanca)]
                hist = sub_f[f"hist__{palanca}"].to_numpy()
                rec_placebo = rng.uniform(lo, hi, size=len(hist))
                dist_esc = np.abs(hist - rec_placebo) / sd
                dist_cols[f"dist_{fase}_{palanca}"] = float(np.nanmean(dist_esc))
        vals = list(dist_cols.values())
        filas.append({"Batch": batch_id, "dist_total_placebo": float(np.mean(vals)) if vals else np.nan})
    return pd.DataFrame(filas)


def _fmt_ols(res, label, subset, n) -> list[dict]:
    filas = []
    ci = res.conf_int()
    for var in res.params.index:
        filas.append({
            "analisis": label, "subset": subset, "n": n, "variable": var,
            "coef": res.params[var], "se": res.bse[var], "p": res.pvalues[var],
            "ci95_lo": ci.loc[var, 0], "ci95_hi": ci.loc[var, 1], "r2": res.rsquared,
        })
    return filas


def _ols_hc3(data: pd.DataFrame, formula: str):
    d = data.dropna(subset=[c for c in _vars_de_formula(formula) if c in data.columns])
    if len(d) < 15:
        return None, d
    m = smf.ols(formula, data=d).fit(cov_type="HC3")
    return m, d


def _vars_de_formula(formula: str) -> list[str]:
    lhs, rhs = formula.split("~")
    vars_ = [lhs.strip()] + [t.strip() for t in rhs.replace("+", " ").split()]
    return [v for v in vars_ if v and v not in ("1",)]


def analisis_estadistico(por_batch: pd.DataFrame, sd_por_fase_palanca: dict) -> pd.DataFrame:
    filas = []
    subsets = {
        "DEV": por_batch.loc[~por_batch["es_lockbox"]],
        "LOCKBOX": por_batch.loc[por_batch["es_lockbox"]],
        "TOTAL": por_batch,
    }

    # --- (a) correlaciones spearman/pearson ---
    dist_individuales = _dist_cols_individuales(por_batch)
    vars_dist = DIST_VARS + dist_individuales
    vars_y = KPI_VARS + ["metal_equivalente_kg_por_t_cargada"]
    for nombre_sub, sub in subsets.items():
        for y in vars_y:
            for x in vars_dist:
                d = sub[[x, y]].dropna()
                if len(d) < 8:
                    continue
                rho, p_s = stats.spearmanr(d[x], d[y])
                r, p_p = stats.pearsonr(d[x], d[y])
                filas.append({"analisis": "correlacion", "subset": nombre_sub, "n": len(d), "variable": f"{y}~{x}",
                              "coef": rho, "se": np.nan, "p": p_s, "ci95_lo": np.nan, "ci95_hi": np.nan,
                              "r2": np.nan, "metodo": "spearman"})
                filas.append({"analisis": "correlacion", "subset": nombre_sub, "n": len(d), "variable": f"{y}~{x}",
                              "coef": r, "se": np.nan, "p": p_p, "ci95_lo": np.nan, "ci95_hi": np.nan,
                              "r2": np.nan, "metodo": "pearson"})
        log(f"  correlaciones {nombre_sub}: ok")

    # --- (b) OLS con controles ---
    controles_presentes = [c for c in CONTROLES if c in por_batch.columns]
    formula_controles = " + ".join(controles_presentes)
    for nombre_sub, sub in subsets.items():
        for y in KPI_VARS:
            f1 = f"{y} ~ dist_total + {formula_controles}"
            m1, d1 = _ols_hc3(sub, f1)
            if m1 is not None:
                for fila in _fmt_ols(m1, f"OLS_{y}_dist_total", nombre_sub, len(d1)):
                    fila["metodo"] = "ols_hc3"; filas.append(fila)
            f2 = f"{y} ~ dist_F + dist_R + {formula_controles}"
            m2, d2 = _ols_hc3(sub, f2)
            if m2 is not None:
                for fila in _fmt_ols(m2, f"OLS_{y}_dist_F_R", nombre_sub, len(d2)):
                    fila["metodo"] = "ols_hc3"; filas.append(fila)
        log(f"  OLS controles {nombre_sub}: ok")

    # --- traduccion del coeficiente dist_total -> rendimiento_proxy_batch (TOTAL) ---
    y = "rendimiento_proxy_batch"
    f1 = f"{y} ~ dist_total + {formula_controles}"
    m_total, d_total = _ols_hc3(por_batch, f1)
    if m_total is not None:
        sd_dist = d_total["dist_total"].std()
        coef = m_total.params["dist_total"]
        mediana = d_total["dist_total"].median()
        p10 = d_total["dist_total"].quantile(0.10)
        ganancia_p10 = coef * (p10 - mediana)
        filas.append({"analisis": "traduccion_dist_total", "subset": "TOTAL", "n": len(d_total),
                      "variable": "pp_rendimiento_por_1sd_menos_dist_total", "coef": -coef * sd_dist,
                      "se": np.nan, "p": m_total.pvalues["dist_total"], "ci95_lo": np.nan, "ci95_hi": np.nan,
                      "r2": m_total.rsquared, "metodo": "traduccion"})
        filas.append({"analisis": "traduccion_dist_total", "subset": "TOTAL", "n": len(d_total),
                      "variable": "pp_rendimiento_ganancia_mediana_a_p10", "coef": ganancia_p10,
                      "se": np.nan, "p": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan,
                      "r2": m_total.rsquared, "metodo": "traduccion"})
        log(f"  traduccion: coef_dist_total={coef:.4f} sd_dist={sd_dist:.4f} mediana={mediana:.3f} p10={p10:.3f} "
            f"ganancia_mediana_a_p10={ganancia_p10:.4f} pp")

    # --- (c) dosis-respuesta por quintiles ---
    for nombre_sub, sub in subsets.items():
        d = sub.dropna(subset=["dist_total", "rendimiento_proxy_batch"])
        if len(d) < 15:
            continue
        d = d.copy()
        try:
            d["quintil"] = pd.qcut(d["dist_total"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        agg = d.groupby("quintil")["rendimiento_proxy_batch"].agg(["mean", "sem", "count"])
        for q, row in agg.iterrows():
            filas.append({"analisis": "dosis_respuesta_quintil", "subset": nombre_sub, "n": int(row["count"]),
                          "variable": f"quintil_{q}", "coef": row["mean"], "se": row["sem"], "p": np.nan,
                          "ci95_lo": row["mean"] - 1.96 * row["sem"], "ci95_hi": row["mean"] + 1.96 * row["sem"],
                          "r2": np.nan, "metodo": "quintil"})
        # test de tendencia: OLS rendimiento ~ rango_quintil (0..4)
        m_trend = sm.OLS(d["rendimiento_proxy_batch"], sm.add_constant(d["quintil"].astype(float))).fit(cov_type="HC3")
        filas.append({"analisis": "dosis_respuesta_tendencia", "subset": nombre_sub, "n": len(d),
                      "variable": "pendiente_quintil", "coef": m_trend.params["quintil"], "se": m_trend.bse["quintil"],
                      "p": m_trend.pvalues["quintil"], "ci95_lo": m_trend.conf_int().loc["quintil", 0],
                      "ci95_hi": m_trend.conf_int().loc["quintil", 1], "r2": m_trend.rsquared, "metodo": "ols_hc3"})
        log(f"  dosis-respuesta {nombre_sub}: pendiente={m_trend.params['quintil']:.4f} p={m_trend.pvalues['quintil']:.4f}")

    # --- (d) placebo: politica aleatoria ---
    escalones = pd.read_csv(SALIDA / "13_recomendaciones_oof_escalones.csv")
    placebo = generar_placebo(escalones, sd_por_fase_palanca, seed=SEMILLA)
    por_batch_pl = por_batch.merge(placebo, on="Batch", how="left")
    for nombre_sub in ("DEV", "LOCKBOX", "TOTAL"):
        sub = por_batch_pl if nombre_sub == "TOTAL" else por_batch_pl.loc[por_batch_pl["es_lockbox"] == (nombre_sub == "LOCKBOX")]
        for y in KPI_VARS:
            f_pl = f"{y} ~ dist_total_placebo + {formula_controles}"
            m_pl, d_pl = _ols_hc3(sub, f_pl)
            if m_pl is not None:
                for fila in _fmt_ols(m_pl, f"PLACEBO_OLS_{y}", nombre_sub, len(d_pl)):
                    fila["metodo"] = "ols_hc3_placebo"; filas.append(fila)
            d = sub[["dist_total_placebo", y]].dropna()
            if len(d) >= 8:
                rho, p_s = stats.spearmanr(d["dist_total_placebo"], d[y])
                filas.append({"analisis": "correlacion_placebo", "subset": nombre_sub, "n": len(d),
                              "variable": f"{y}~dist_total_placebo", "coef": rho, "se": np.nan, "p": p_s,
                              "ci95_lo": np.nan, "ci95_hi": np.nan, "r2": np.nan, "metodo": "spearman_placebo"})
        log(f"  placebo {nombre_sub}: ok")

    # --- permutacion: 1000 permutaciones de dist_total entre batches, coef OLS con controles ---
    rng = np.random.default_rng(SEMILLA)
    for nombre_sub, sub in subsets.items():
        d = sub.dropna(subset=["dist_total", "rendimiento_proxy_batch"] + controles_presentes).reset_index(drop=True)
        if len(d) < 15:
            continue
        f = f"rendimiento_proxy_batch ~ dist_total + {formula_controles}"
        m_obs = smf.ols(f, data=d).fit()
        coef_obs = m_obs.params["dist_total"]
        coefs_perm = np.empty(N_PERM)
        d_perm = d.copy()
        for i in range(N_PERM):
            d_perm["dist_total"] = rng.permutation(d["dist_total"].to_numpy())
            m_p = smf.ols(f, data=d_perm).fit()
            coefs_perm[i] = m_p.params["dist_total"]
        p_perm = float(np.mean(np.abs(coefs_perm) >= np.abs(coef_obs)))
        filas.append({"analisis": "permutacion_dist_total", "subset": nombre_sub, "n": len(d),
                      "variable": "coef_dist_total_vs_null", "coef": coef_obs, "se": float(coefs_perm.std()),
                      "p": p_perm, "ci95_lo": float(np.percentile(coefs_perm, 2.5)),
                      "ci95_hi": float(np.percentile(coefs_perm, 97.5)), "r2": np.nan, "metodo": "permutacion_1000"})
        log(f"  permutacion {nombre_sub}: coef_obs={coef_obs:.5f} p_perm={p_perm:.4f}")

    # --- (f) analisis direccional por palanca ---
    dir_cols = [c for c in por_batch.columns if c.startswith("dir_")]
    for col in dir_cols:
        fase_palanca = col[len("dir_"):]
        media_dir = por_batch[col].mean()
        if pd.isna(media_dir) or abs(media_dir) < 1e-9:
            continue
        signo_deseado = np.sign(media_dir)  # el modelo pide (rec-hist)/sd de este signo en promedio
        fase = "Fusión" if fase_palanca.startswith("Fusión_") else "Reducción"
        palanca = fase_palanca[len(fase) + 1:]
        col_hist_medio = f"_hist_medio_{fase_palanca}"
        # valor historico medio de la palanca por batch (relativo al contexto): usamos el promedio
        # de hist__<palanca> en los escalones de esa fase del batch, ya presente via dist/dir; lo
        # recomputamos desde escalones para tener el nivel (no la distancia)
        hist_medio = escalones.loc[escalones["fase"] == fase].groupby("Batch")[f"hist__{palanca}"].mean()
        d = por_batch.set_index("Batch")[["rendimiento_proxy_batch"] + controles_presentes + ["es_lockbox"]].join(hist_medio.rename("hist_medio"))
        d = d.dropna(subset=["hist_medio", "rendimiento_proxy_batch"])
        if len(d) < 15:
            continue
        mediana_hist = d["hist_medio"].median()
        # "operar en la direccion deseada" = por encima de la mediana si el modelo pide MAS (+1),
        # por debajo de la mediana si el modelo pide MENOS (-1)
        d["en_direccion_deseada"] = (d["hist_medio"] >= mediana_hist) if signo_deseado > 0 else (d["hist_medio"] <= mediana_hist)
        for nombre_sub in ("DEV", "LOCKBOX", "TOTAL"):
            dd = d if nombre_sub == "TOTAL" else d.loc[d["es_lockbox"] == (nombre_sub == "LOCKBOX")]
            if len(dd) < 15 or dd["en_direccion_deseada"].nunique() < 2:
                continue
            g_si = dd.loc[dd["en_direccion_deseada"], "rendimiento_proxy_batch"]
            g_no = dd.loc[~dd["en_direccion_deseada"], "rendimiento_proxy_batch"]
            if len(g_si) < 3 or len(g_no) < 3:
                continue
            t_stat, p_t = stats.ttest_ind(g_si, g_no, equal_var=False)
            filas.append({"analisis": "direccional_ttest", "subset": nombre_sub, "n": len(dd),
                          "variable": f"{fase_palanca}_dir{'mas' if signo_deseado > 0 else 'menos'}",
                          "coef": float(g_si.mean() - g_no.mean()), "se": np.nan, "p": p_t,
                          "ci95_lo": np.nan, "ci95_hi": np.nan, "r2": np.nan, "metodo": "ttest_welch"})
            f_dir = f"rendimiento_proxy_batch ~ en_direccion_deseada + {formula_controles}"
            dd2 = dd.copy(); dd2["en_direccion_deseada"] = dd2["en_direccion_deseada"].astype(int)
            m_dir, d_dir = _ols_hc3(dd2, f_dir)
            if m_dir is not None:
                for fila in _fmt_ols(m_dir, f"OLS_direccional_{fase_palanca}", nombre_sub, len(d_dir)):
                    fila["metodo"] = "ols_hc3_direccional"; filas.append(fila)
        log(f"  direccional {fase_palanca}: signo_deseado={signo_deseado:+.0f} media_dir={media_dir:.3f}")

    return pd.DataFrame(filas)


def hacer_figuras(por_batch: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) scatter rendimiento vs dist_total, coloreado dev/lockbox
    fig, ax = plt.subplots(figsize=(7, 5))
    for es_lb, color, label in [(False, "tab:blue", "DEV (cross-fitted)"), (True, "tab:red", "LOCKBOX")]:
        d = por_batch.loc[por_batch["es_lockbox"] == es_lb]
        ax.scatter(d["dist_total"], d["rendimiento_proxy_batch"], c=color, label=label, alpha=0.6, s=25)
    ax.set_xlabel("dist_total (distancia media normalizada a la recomendacion)")
    ax.set_ylabel("rendimiento_proxy_batch (%)")
    ax.set_title("Rendimiento del batch vs. distancia a la politica recomendada v3")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "scatter_rendimiento_vs_dist_total.png", dpi=130); plt.close(fig)

    # 2) dosis-respuesta por quintiles (dev, lockbox, total)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (nombre, es_lb) in zip(axes, [("DEV", False), ("LOCKBOX", True), ("TOTAL", None)]):
        d = por_batch if es_lb is None else por_batch.loc[por_batch["es_lockbox"] == es_lb]
        d = d.dropna(subset=["dist_total", "rendimiento_proxy_batch"])
        if len(d) < 10:
            continue
        try:
            d = d.copy()
            d["quintil"] = pd.qcut(d["dist_total"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        agg = d.groupby("quintil")["rendimiento_proxy_batch"].agg(["mean", "sem"])
        ax.errorbar(agg.index, agg["mean"], yerr=1.96 * agg["sem"], marker="o", capsize=4)
        ax.set_title(nombre); ax.set_xlabel("quintil de dist_total (0=mas cerca, 4=mas lejos)")
    axes[0].set_ylabel("rendimiento_proxy_batch (%)")
    fig.suptitle("Dosis-respuesta: rendimiento por quintil de distancia a la politica recomendada")
    fig.tight_layout(); fig.savefig(FIGS / "dosis_respuesta_quintiles.png", dpi=130); plt.close(fig)

    # 3) histograma de dir_* por palanca y fase
    dir_cols = [c for c in por_batch.columns if c.startswith("dir_")]
    if dir_cols:
        n = len(dir_cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, dir_cols):
            vals = por_batch[col].dropna()
            ax.hist(vals, bins=25, color="tab:purple", alpha=0.7)
            ax.axvline(0, color="k", lw=1)
            ax.axvline(vals.mean(), color="tab:red", lw=1.5, ls="--")
            ax.set_title(col.replace("dir_", ""), fontsize=9)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Direccion media pedida por el modelo: (rec-hist)/sd_DEV por palanca y fase")
        fig.tight_layout(); fig.savefig(FIGS / "histograma_direccion_palancas.png", dpi=130); plt.close(fig)

    log("figuras guardadas en", FIGS)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "analisis":
        correr_fold(sys.argv[1])
    else:
        main()
