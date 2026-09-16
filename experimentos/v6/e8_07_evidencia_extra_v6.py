"""E8-07: evidencia extendida de la politica v6 (replica E7-06, `experimentos/v5/e7_06_evidencia_extra.py`)
sobre el KPI refinado del batch, comparada con v5 en la MISMA muestra de 362 batches.

Entradas: experimentos/v6/e8_05_por_batch_v6.csv (politica v6 cross-fitted, una fila por batch),
experimentos/v6/e8_05_recomendaciones_v6_escalones.csv (recomendaciones por escalon),
experimentos/v5/e7_05_por_batch_v5.csv, experimentos/v5/e7_05_recomendaciones_v5_escalones.csv (equivalentes v5),
experimentos/v6/e8_05_evidencia_politica_v6.csv y experimentos/v5/e7_05_evidencia_politica_v5.csv (para el veredicto).

Cuatro capas (A/B/C como E7-06, D y E nuevas de esta iteracion):
  A) Emparejamiento por estado (v6): batches "cerca" de la politica (tercil inferior de dist) vs "lejos"
     (tercil superior), emparejados 1:1 sobre covariables de estado/contexto; diferencia de KPI con IC bootstrap.
  B) Dosis-respuesta (v6): KPI refinado, f_dross, f_polvo por tercil de adherencia (dist_F/dist_R/dist_total),
     DEV/lockbox/total, tendencia OLS con controles.
  C) Cadena theta -> agregado -> KPI (v6 Y v5, misma muestra de batches): cambio predicho de Sum m6_ln_sn_dep /
     Sum m6_ln_feo_ret (v6) o Sum ln_sn_dep / Sum ln_feo_ret (v5) de Reduccion si se siguiera cada politica,
     traducido a pp de KPI con las pendientes de batch (OLS HC3, controles de E8-03 = v5_lib.CONTROLES_BATCH,
     DEV), IC bootstrap por batch (1000 remuestreos de la regresion).
  D) Comparacion directa v5 vs v6 por batch: correlacion de dist_R/dist_F/dist_total y de uplift fisico;
     coincidencia de signo de las recomendaciones (dir_*) por palanca.
  E) Resumen de la politica v6 por tramo de avance (Reduccion) y por orden de escalon (Fusion): mediana de
     (rec - hist) por palanca en unidades fisicas (kg/min, Nm3/min) y en sd (DEV).
Salidas: e8_07_matching.csv, e8_07_dosis_respuesta.csv, e8_07_cadena.csv, e8_07_comparacion_v5_v6.csv,
e8_07_politica_por_tramo.csv, e8_07_resultados.md
Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v6/e8_07_evidencia_extra_v6.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
V5_DIR = RAIZ / "experimentos" / "v5"
for _p in (HERE, RAIZ, V5_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
import modelo_prescriptivo as mp  # noqa: E402
import v6_lib as v6  # noqa: E402
import v5_lib as v5lib  # noqa: E402
import modelo_predictivo_v6 as mp6  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402

SALIDA = HERE
pd.set_option("display.width", 220)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# =============================================================================
# 0) Carga
# =============================================================================
log("cargando df v6 (mp6.agregar_columnas_v6 sobre cache df_v3.pkl)...")
df = mp6.agregar_columnas_v6(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
batch6 = v6.construir_batch_v6(df).reset_index()
log(f"df {df.shape}  batch6 {batch6.shape} ({int(batch6.es_dev.sum())} DEV / {int(batch6.es_lockbox.sum())} lockbox)")

EXTRA = ["Batch", "F_lnirf_F0", "R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "R_sum_ln_sn_dep", "R_sum_ln_feo_ret",
         "R_lnirf_twmean", "T_media_R", "feed_sn_total_t", "frac_carga_secundaria_F"]

pb6 = pd.read_csv(SALIDA / "e8_05_por_batch_v6.csv").merge(batch6[EXTRA], on="Batch", how="left")
pb5 = pd.read_csv(V5_DIR / "e7_05_por_batch_v5.csv").merge(batch6[EXTRA], on="Batch", how="left")
assert (pb5["Batch"].values == pb6["Batch"].values).all(), "v5 y v6 deben compartir la misma muestra de 362 batches"
log(f"pb6 {pb6.shape}  pb5 {pb5.shape}  (misma muestra de batches: {len(pb6)})")

KPIS = ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_refinada_pct_next", "recuperacion_refinada_pct_media2"]
COV = ["ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm", "sn_cargado_batch_kg", "temperatura_media_fusion",
       "frac_carga_secundaria_media_fusion", "feed_dross_Fe_total_t", "F_lnirf_F0", "idx_cronologico"]
CONTROLES_E803 = list(v5lib.CONTROLES_BATCH)   # controles de E8-03 (= v5.CONTROLES_BATCH): feed_sn_total_t, ley_sn_conc_batch_pct,
                                                # espesor_ladrillo_norm_mm, frac_carga_secundaria_F, feed_dross_Fe_total_t, idx_cronologico
rng = np.random.default_rng(0)

# =============================================================================
# A) Emparejamiento por estado (v6)
# =============================================================================
log("=== A) emparejamiento por estado (v6) ===")


def matching(sub: pd.DataFrame, dist_col: str, kpi: str, n_boot: int = 1000) -> dict:
    d = sub[[dist_col, kpi] + COV].dropna()
    if len(d) < 30:
        return {}
    q1, q3 = d[dist_col].quantile([1 / 3, 2 / 3])
    cerca, lejos = d[d[dist_col] <= q1], d[d[dist_col] >= q3]
    sc = StandardScaler().fit(d[COV])
    nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(cerca[COV]))
    dist_m, idx = nn.kneighbors(sc.transform(lejos[COV]))
    y_lejos = lejos[kpi].to_numpy(); y_cerca_m = cerca[kpi].to_numpy()[idx[:, 0]]
    dif = y_cerca_m - y_lejos                      # >0: seguir la politica se asocia a mayor KPI
    bs = [np.mean(dif[rng.integers(0, len(dif), len(dif))]) for _ in range(n_boot)]
    t, p = stats.ttest_rel(y_cerca_m, y_lejos)
    smd = np.abs((cerca[COV].to_numpy()[idx[:, 0]].mean(0) - lejos[COV].to_numpy().mean(0)) / (d[COV].std().to_numpy() + 1e-9)).mean()
    return {"dist": dist_col, "kpi": kpi, "n_pares": len(dif), "dif_media_pp": float(dif.mean()), "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)), "p_pareado": float(p), "smd_medio_post": float(smd),
            "dist_media_cerca": float(cerca[dist_col].mean()), "dist_media_lejos": float(lejos[dist_col].mean())}


filas = []
for nombre, sub in [("DEV", pb6[~pb6.es_lockbox]), ("LOCKBOX", pb6[pb6.es_lockbox]), ("TOTAL", pb6)]:
    for dc in ["dist_total", "dist_R", "dist_F"]:
        for kpi in KPIS:
            r = matching(sub, dc, kpi)
            if r:
                r["subset"] = nombre; r["sistema"] = "v6"; filas.append(r)
mat = pd.DataFrame(filas); mat.to_csv(SALIDA / "e8_07_matching.csv", index=False)
log("\n", mat[mat.kpi == "recuperacion_refinada_pct"].round(3).to_string())

# =============================================================================
# B) Dosis-respuesta por terciles (v6): KPI refinado, f_dross, f_polvo
# =============================================================================
log("=== B) dosis-respuesta por terciles (v6) ===")


def dosis(sub: pd.DataFrame, dist_col: str, metric: str) -> dict | None:
    d = sub[[dist_col, metric] + COV].dropna()
    if len(d) < 30:
        return None
    ter = pd.qcut(d[dist_col], 3, labels=["cerca", "medio", "lejos"])
    m = d.groupby(ter, observed=True)[metric].mean()
    X = sm.add_constant(((d[[dist_col] + COV] - d[[dist_col] + COV].mean()) / d[[dist_col] + COV].std()))
    res = sm.OLS(d[metric], X).fit(cov_type="HC3")
    return {"dist": dist_col, "metric": metric, "n": len(d), "cerca": float(m.iloc[0]), "medio": float(m.iloc[1]),
            "lejos": float(m.iloc[2]), "d_cerca_lejos": float(m.iloc[0] - m.iloc[2]),
            "ols_coef_por_sd": float(res.params[dist_col]), "ols_p": float(res.pvalues[dist_col])}


filas = []
for nombre, sub in [("DEV", pb6[~pb6.es_lockbox]), ("LOCKBOX", pb6[pb6.es_lockbox]), ("TOTAL", pb6)]:
    for dc in ["dist_total", "dist_R", "dist_F"]:
        for metric in ["recuperacion_refinada_pct", "f_dross", "f_polvo"]:
            r = dosis(sub, dc, metric)
            if r:
                r["subset"] = nombre; r["sistema"] = "v6"; filas.append(r)
dr = pd.DataFrame(filas); dr.to_csv(SALIDA / "e8_07_dosis_respuesta.csv", index=False)
log("\n", dr[dr.metric == "recuperacion_refinada_pct"].round(3).to_string())
log("\n", dr[dr.metric.isin(["f_dross", "f_polvo"])].round(4).to_string())

# =============================================================================
# C) Cadena theta -> agregado -> KPI (v6 y v5, misma muestra) con controles de E8-03
# =============================================================================
log("=== C) cadena theta -> agregado -> KPI (v6 y v5, controles E8-03) ===")


def pendientes(d: pd.DataFrame, cols: list[str], kpi: str, controles: list[str]) -> np.ndarray:
    X = sm.add_constant(pd.concat([d[cols], (d[controles] - d[controles].mean()) / d[controles].std()], axis=1))
    return sm.OLS(d[kpi], X).fit().params[cols].to_numpy()


SISTEMAS = {
    "v6": dict(pb=pb6, cols=["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret"], cols_d=["R_d_sn_dep_pred", "R_d_feo_ret_pred"]),
    "v5": dict(pb=pb5, cols=["R_sum_ln_sn_dep", "R_sum_ln_feo_ret"], cols_d=["R_d_sn_dep_pred", "R_d_feo_ret_pred"]),
}
cadena = []
for sistema, cfgs in SISTEMAS.items():
    pb, cols, cols_d = cfgs["pb"], cfgs["cols"], cfgs["cols_d"]
    for nombre, sub in [("DEV", pb[~pb.es_lockbox]), ("TOTAL", pb)]:
        for kpi in ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_refinada_pct_media2"]:
            d = sub[cols + [kpi] + CONTROLES_E803].dropna()
            b0 = pendientes(d, cols, kpi, CONTROLES_E803)
            bs = np.array([pendientes(d.iloc[rng.integers(0, len(d), len(d))], cols, kpi, CONTROLES_E803) for _ in range(1000)])
            dd = pb[["Batch", "es_lockbox"] + cols_d].dropna()
            for nom2, s2 in [("DEV", dd[~dd.es_lockbox]), ("LOCKBOX", dd[dd.es_lockbox]), ("TOTAL", dd)]:
                dx = s2[cols_d].to_numpy()
                up = dx @ b0                                   # pp de KPI esperados por batch
                up_bs = dx.mean(0) @ bs.T                      # incertidumbre de las pendientes sobre el cambio medio
                cadena.append({"sistema": sistema, "pendientes_de": nombre, "kpi": kpi, "politica_en": nom2, "n_batches": len(s2),
                               "b_sn_dep_pp_por_unidad": b0[0], "b_sn_lo": np.percentile(bs[:, 0], 2.5), "b_sn_hi": np.percentile(bs[:, 0], 97.5),
                               "b_feo_ret_pp_por_unidad": b0[1], "b_feo_lo": np.percentile(bs[:, 1], 2.5), "b_feo_hi": np.percentile(bs[:, 1], 97.5),
                               "w_implicito": b0[1] / b0[0] if abs(b0[0]) > 1e-9 else np.nan,
                               "d_sn_dep_medio": dx[:, 0].mean(), "d_feo_ret_medio": dx[:, 1].mean(),
                               "uplift_kpi_pp_medio": float(up.mean()), "uplift_kpi_pp_mediana": float(np.median(up)),
                               "uplift_ci_lo": float(np.percentile(up_bs, 2.5)), "uplift_ci_hi": float(np.percentile(up_bs, 97.5)),
                               "uplift_kg_sn_medio": float(up.mean() * mp6.KG_SN_POR_PP_KPI),
                               "uplift_kg_sn_ci_lo": float(np.percentile(up_bs, 2.5) * mp6.KG_SN_POR_PP_KPI),
                               "uplift_kg_sn_ci_hi": float(np.percentile(up_bs, 97.5) * mp6.KG_SN_POR_PP_KPI),
                               "pct_batches_uplift_pos": float((up > 0).mean() * 100)})
cad = pd.DataFrame(cadena); cad.to_csv(SALIDA / "e8_07_cadena.csv", index=False)
log("\n", cad[(cad.kpi == "recuperacion_refinada_pct") & (cad.pendientes_de == "DEV")]
    [["sistema", "politica_en", "n_batches", "w_implicito", "uplift_kpi_pp_medio", "uplift_ci_lo", "uplift_ci_hi",
      "uplift_kg_sn_medio", "pct_batches_uplift_pos"]].round(3).to_string())

# =============================================================================
# D) Comparacion directa v5 vs v6 por batch
# =============================================================================
log("=== D) comparacion directa v5 vs v6 ===")
dir_cols = sorted(set(c for c in pb6.columns if c.startswith("dir_")) & set(c for c in pb5.columns if c.startswith("dir_")))
metricas_cont = ["dist_total", "dist_R", "dist_F", "uplift_fisico_total_kg", "uplift_fisico_fusion_kg", "uplift_fisico_reduccion_kg"]
comp = pb6[["Batch"] + metricas_cont + dir_cols].merge(pb5[["Batch"] + metricas_cont + dir_cols], on="Batch", suffixes=("_v6", "_v5"))

filas = []
for metric in metricas_cont:
    x, y = comp[f"{metric}_v5"], comp[f"{metric}_v6"]
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 10:
        continue
    pear = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1]); spear = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    filas.append({"metric": metric, "tipo": "continuo", "n": len(d), "pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
                  "spearman_rho": float(spear[0]), "spearman_p": float(spear[1]), "media_v5": float(d.iloc[:, 0].mean()),
                  "media_v6": float(d.iloc[:, 1].mean()), "pct_mismo_signo": np.nan})
for c in dir_cols:
    x, y = comp[f"{c}_v5"], comp[f"{c}_v6"]
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 10:
        continue
    pear = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1]); spear = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    pct_signo = float((np.sign(d.iloc[:, 0]) == np.sign(d.iloc[:, 1])).mean() * 100)
    filas.append({"metric": c, "tipo": "direccion_palanca", "n": len(d), "pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
                  "spearman_rho": float(spear[0]), "spearman_p": float(spear[1]), "media_v5": float(d.iloc[:, 0].mean()),
                  "media_v6": float(d.iloc[:, 1].mean()), "pct_mismo_signo": pct_signo})
comp_tab = pd.DataFrame(filas); comp_tab.to_csv(SALIDA / "e8_07_comparacion_v5_v6.csv", index=False)
log("\n", comp_tab.round(3).to_string())

# =============================================================================
# E) Politica v6 por tramo de avance (Reduccion) y por orden de escalon (Fusion)
# =============================================================================
log("=== E) politica v6 por tramo/orden ===")
esc6 = pd.read_csv(SALIDA / "e8_05_recomendaciones_v6_escalones.csv")
ACC = mp6.ACCIONES_OPTIMIZABLES_V6
UNIDADES = {"tasa_feed_Carbon_kg_min": "kg/min", "tasa_gn_nm3_min": "Nm3/min", "tasa_o2_nm3_min": "Nm3/min", "tasa_aire_nm3_min": "Nm3/min"}

dfb = mp6._preparar_base(df)
dev_b, _ = mp.split_dev_lockbox(df)
sd = {}
for fase in mp6.FASES:
    s = dfb[(dfb.fase_proceso == fase) & dfb.Batch.isin(dev_b)]
    for a in ACC[fase]:
        sd[(fase, a)] = float(s[a].std())

filas = []
er = esc6[esc6.fase == "Reducción"].copy()
bins = [-np.inf, 0.84, 0.96, 0.975, np.inf]
labels = ["<0.84", "0.84-0.96", "0.96-0.975", ">0.975"]
er["tramo"] = pd.cut(er["m6_avance_prev"], bins=bins, labels=labels)
for tramo, g in er.groupby("tramo", observed=True):
    for a in ACC["Reducción"]:
        d = (g[f"rec__{a}"] - g[f"hist__{a}"]).dropna()
        if len(d) == 0:
            continue
        filas.append({"segmento": "Reduccion_avance", "tramo": str(tramo), "palanca": a, "unidad": UNIDADES[a],
                      "n": int(len(d)), "mediana_fisica": float(d.median()), "mediana_sd": float((d / sd[("Reducción", a)]).median())})

ef = esc6[esc6.fase == "Fusión"].copy()
for orden, g in ef.groupby("orden_escalon_fase"):
    for a in ACC["Fusión"]:
        d = (g[f"rec__{a}"] - g[f"hist__{a}"]).dropna()
        if len(d) == 0:
            continue
        filas.append({"segmento": "Fusion_orden", "tramo": str(int(orden)), "palanca": a, "unidad": UNIDADES[a],
                      "n": int(len(d)), "mediana_fisica": float(d.median()), "mediana_sd": float((d / sd[("Fusión", a)]).median())})

pol = pd.DataFrame(filas); pol.to_csv(SALIDA / "e8_07_politica_por_tramo.csv", index=False)
log("\n", pol.round(3).to_string())

# =============================================================================
# Veredicto: adherencia vs KPI (E8-05/E7-05, sin recalcular) para el memo
# =============================================================================
ev6 = pd.read_csv(SALIDA / "e8_05_evidencia_politica_v6.csv")
ev5 = pd.read_csv(V5_DIR / "e7_05_evidencia_politica_v5.csv")
ver = pd.concat([ev6.assign(sistema="v6"), ev5.assign(sistema="v5")], ignore_index=True)
ver = ver[ver.variable.isin(["dist_F", "dist_R", "dist_total"]) & ver.kpi.isin(["recuperacion_refinada_pct", "f_dross", "f_polvo"])]
ver = ver[["sistema", "kpi", "variable", "subset", "n", "coef_por_sd", "p_hc3", "spearman"]]

# =============================================================================
# Memo
# =============================================================================
with open(SALIDA / "e8_07_resultados.md", "w", encoding="utf-8") as f:
    f.write("# E8-07 -- evidencia extendida de la politica v6 (KPI refinado), comparada con v5\n\n")
    f.write(f"Muestra: {len(pb6)} batches, identica para v5 y v6 (misma union DEV/lockbox, 63 lockbox / "
            f"{len(pb6) - int(pb6.es_lockbox.sum())} DEV). Replica la metodologia de E7-06 sobre la politica v6.\n\n")
    f.write("## A) Emparejamiento por estado -- v6 (cerca vs lejos de la politica; dif > 0 favorece seguirla)\n\n")
    f.write(mat.round(3).to_string(index=False) + "\n\n")
    f.write("## B) Dosis-respuesta por terciles de adherencia -- v6 (KPI refinado, f_dross, f_polvo)\n\n")
    f.write(dr.round(4).to_string(index=False) + "\n\n")
    f.write("## C) Cadena theta -> agregado -> KPI -- v6 vs v5 (misma muestra, controles E8-03)\n\n")
    f.write(cad.round(3).to_string(index=False) + "\n\n")
    f.write("## D) Comparacion directa v5 vs v6 por batch\n\n")
    f.write(comp_tab.round(3).to_string(index=False) + "\n\n")
    f.write("## E) Politica v6 por tramo de avance (Reduccion) y orden de escalon (Fusion)\n\n")
    f.write(pol.round(3).to_string(index=False) + "\n\n")
    f.write("## Adherencia vs KPI (E8-05 / E7-05, referencia para el veredicto)\n\n")
    f.write(ver.round(4).to_string(index=False) + "\n")
log("LISTO")
