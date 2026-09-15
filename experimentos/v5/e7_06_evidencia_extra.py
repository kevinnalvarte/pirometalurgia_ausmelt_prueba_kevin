"""E7-06: evidencia extendida de que seguir la politica v5 sube el KPI refinado del batch.
Entradas: e7_05_por_batch_v5.csv (politica cross-fitted, una fila por batch), e7_05_efectos_control_v5.csv.
Tres capas complementarias a e7_05_evidencia_politica_v5.csv (OLS de adherencia con controles):
  A) Emparejamiento por estado: batches "cerca" de la politica (tercil inferior de dist) vs "lejos" (tercil superior),
     emparejados 1:1 (vecino mas cercano con reemplazo) sobre covariables de estado/contexto; diferencia de KPI con IC bootstrap.
  B) Dosis-respuesta: KPI medio por tercil de adherencia (DEV / lockbox / total), tendencia con controles.
  C) Cadena theta -> agregado -> KPI: el cambio predicho de las componentes agregadas por batch (sum pred_rec - pred_hist)
     multiplicado por las pendientes de batch del KPI refinado sobre las componentes (OLS con controles, bootstrap por batch)
     = uplift de KPI esperado por batch, con IC. Se compara con el uplift "fisico" en kg que reporta el sistema.
Salidas: e7_06_matching.csv, e7_06_dosis_respuesta.csv, e7_06_cadena.csv, e7_06_resultados.md (tablas)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_lib as v5
import modelo_predictivo_v5 as mp5
SALIDA = Path(__file__).resolve().parent
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

pb = pd.read_csv(SALIDA / f"e7_05_por_batch_v5{SUF}.csv")
df = v5.cargar_df(con_termicas=False)
batch = v5.construir_batch_v5(df).reset_index()
pb = pb.merge(batch[["Batch", "F_lnirf_F0", "R_sum_ln_sn_dep", "R_sum_ln_feo_ret", "R_lnirf_twmean", "T_media_R"]], on="Batch", how="left")
KPIS = ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_refinada_pct_next", "recuperacion_refinada_pct_media2"]
COV = ["ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm", "sn_cargado_batch_kg", "temperatura_media_fusion",
       "frac_carga_secundaria_media_fusion", "feed_dross_Fe_total_t", "F_lnirf_F0", "idx_cronologico"]
rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------- A) matching
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
    # balance de covariables tras el emparejamiento (SMD medio)
    smd = np.abs((cerca[COV].to_numpy()[idx[:, 0]].mean(0) - lejos[COV].to_numpy().mean(0)) / (d[COV].std().to_numpy() + 1e-9)).mean()
    return {"dist": dist_col, "kpi": kpi, "n_pares": len(dif), "dif_media_pp": float(dif.mean()), "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)), "p_pareado": float(p), "smd_medio_post": float(smd),
            "dist_media_cerca": float(cerca[dist_col].mean()), "dist_media_lejos": float(lejos[dist_col].mean())}

filas = []
for nombre, sub in [("DEV", pb[~pb.es_lockbox]), ("LOCKBOX", pb[pb.es_lockbox]), ("TOTAL", pb)]:
    for dc in ["dist_total", "dist_R", "dist_F"]:
        for kpi in KPIS:
            r = matching(sub, dc, kpi)
            if r:
                r["subset"] = nombre; filas.append(r)
mat = pd.DataFrame(filas); mat.to_csv(SALIDA / f"e7_06_matching{SUF}.csv", index=False)
log("\n", mat[mat.kpi == "recuperacion_refinada_pct"].round(3).to_string())

# ----------------------------------------------------------------------------- B) dosis-respuesta por terciles
filas = []
for nombre, sub in [("DEV", pb[~pb.es_lockbox]), ("LOCKBOX", pb[pb.es_lockbox]), ("TOTAL", pb)]:
    for dc in ["dist_total", "dist_R", "dist_F"]:
        d = sub[[dc, "recuperacion_refinada_pct", "rendimiento_proxy_batch"] + COV].dropna()
        if len(d) < 30:
            continue
        ter = pd.qcut(d[dc], 3, labels=["cerca", "medio", "lejos"])
        m = d.groupby(ter)[["recuperacion_refinada_pct", "rendimiento_proxy_batch"]].mean()
        X = sm.add_constant(((d[[dc] + COV] - d[[dc] + COV].mean()) / d[[dc] + COV].std()))
        res = sm.OLS(d["recuperacion_refinada_pct"], X).fit(cov_type="HC3")
        filas.append({"subset": nombre, "dist": dc, "n": len(d), "kpi_cerca": m.iloc[0, 0], "kpi_medio": m.iloc[1, 0], "kpi_lejos": m.iloc[2, 0],
                      "d_cerca_lejos_pp": m.iloc[0, 0] - m.iloc[2, 0], "proxy_cerca": m.iloc[0, 1], "proxy_lejos": m.iloc[2, 1],
                      "ols_pp_por_sd": float(res.params[dc]), "ols_p": float(res.pvalues[dc])})
dr = pd.DataFrame(filas); dr.to_csv(SALIDA / f"e7_06_dosis_respuesta{SUF}.csv", index=False)
log("\n", dr.round(3).to_string())

# ----------------------------------------------------------------------------- C) cadena theta -> agregado -> KPI
# pendientes de batch del KPI refinado sobre las componentes agregadas (unidades naturales), bootstrap por batch (DEV)
def pendientes(d: pd.DataFrame, cols: list[str], kpi: str) -> np.ndarray:
    X = sm.add_constant(pd.concat([d[cols], (d[COV] - d[COV].mean()) / d[COV].std()], axis=1))
    return sm.OLS(d[kpi], X).fit().params[cols].to_numpy()

cadena = []
for nombre, sub in [("DEV", pb[~pb.es_lockbox]), ("TOTAL", pb)]:
    for kpi in ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_refinada_pct_media2"]:
        d = sub[["R_sum_ln_sn_dep", "R_sum_ln_feo_ret", kpi] + COV].dropna()
        cols = ["R_sum_ln_sn_dep", "R_sum_ln_feo_ret"]
        b0 = pendientes(d, cols, kpi)
        bs = np.array([pendientes(d.iloc[rng.integers(0, len(d), len(d))], cols, kpi) for _ in range(1000)])
        # cambio predicho por la politica (por batch, cross-fitted): sum(pred_rec - pred_hist) de cada componente
        dd = pb[["Batch", "es_lockbox", "R_d_sn_dep_pred", "R_d_feo_ret_pred"]].dropna()
        for nom2, s2 in [("DEV", dd[~dd.es_lockbox]), ("LOCKBOX", dd[dd.es_lockbox]), ("TOTAL", dd)]:
            dx = s2[["R_d_sn_dep_pred", "R_d_feo_ret_pred"]].to_numpy()
            up = dx @ b0                                   # pp de KPI esperados por batch
            up_bs = dx.mean(0) @ bs.T                      # incertidumbre de las pendientes sobre el cambio medio
            cadena.append({"pendientes_de": nombre, "kpi": kpi, "politica_en": nom2, "n_batches": len(s2),
                           "b_sn_dep_pp_por_unidad": b0[0], "b_sn_lo": np.percentile(bs[:, 0], 2.5), "b_sn_hi": np.percentile(bs[:, 0], 97.5),
                           "b_feo_ret_pp_por_unidad": b0[1], "b_feo_lo": np.percentile(bs[:, 1], 2.5), "b_feo_hi": np.percentile(bs[:, 1], 97.5),
                           "w_implicito": b0[1] / b0[0] if abs(b0[0]) > 1e-9 else np.nan,
                           "d_sn_dep_medio": dx[:, 0].mean(), "d_feo_ret_medio": dx[:, 1].mean(),
                           "uplift_kpi_pp_medio": float(up.mean()), "uplift_kpi_pp_mediana": float(np.median(up)),
                           "uplift_ci_lo": float(np.percentile(up_bs, 2.5)), "uplift_ci_hi": float(np.percentile(up_bs, 97.5)),
                           "uplift_kg_sn_medio": float(up.mean() * mp5.KG_SN_POR_PP_KPI),
                           "pct_batches_uplift_pos": float((up > 0).mean() * 100)})
cad = pd.DataFrame(cadena); cad.to_csv(SALIDA / f"e7_06_cadena{SUF}.csv", index=False)
log("\n", cad[(cad.kpi == "recuperacion_refinada_pct")].round(3).to_string())

# ----------------------------------------------------------------------------- memo
with open(SALIDA / f"e7_06_resultados{SUF}.md", "w", encoding="utf-8") as f:
    f.write("# E7-06 — evidencia extendida de la politica v5 (KPI refinado)\n\n")
    f.write("## A) Emparejamiento por estado (cerca vs lejos de la politica; dif > 0 favorece seguir la politica)\n\n")
    f.write(mat.round(3).to_string(index=False) + "\n\n")
    f.write("## B) Dosis-respuesta por terciles de adherencia\n\n" + dr.round(3).to_string(index=False) + "\n\n")
    f.write("## C) Cadena theta -> agregado -> KPI\n\n" + cad.round(3).to_string(index=False) + "\n")
log("LISTO")
