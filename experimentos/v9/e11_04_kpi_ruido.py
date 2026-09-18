"""E11-04 (H4): existe una medida de rendimiento del batch con mejor senal/ruido para evaluar decisiones que
`recuperacion_refinada_pct`? Ver experimentos/v9/ITERACION_11_diseno.md y experimentos/v9/e11_04_resultados.md.

Salidas: e11_04_autocorr.csv, e11_04_mod3.csv, e11_04_crosscorr.csv, e11_04_kpis_batch.csv, e11_04_kpi_stats.csv,
e11_04_prueba_anidada.csv, e11_04_placebo_hac.csv.
"""
import sys, time
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import periodogram
import statsmodels.api as sm

import v9_lib as L
import v8_lib as L8
import feature_engineering as fe

OUT = L.SALIDA
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- datos
df = L8.cargar_df()
res = L.residuos_escalon(df)
t = L.tabla_batch(df, res)
bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
t = t.join(bal[["sn_escoria_final_balance_kg", "feed_sn_total_kg", "cierre_sn_frac"]], how="left")

# exposiciones de Reduccion completa (R0-R3): media de z__gn, z__carbon en Rt+Rl
subR = res[res["grupo"].isin(["Rt", "Rl"])]
t["x_R_gn"] = subR.groupby("Batch")["z__gn"].mean()
t["x_R_carbon"] = subR.groupby("Batch")["z__carbon"].mean()

t = t.sort_values("idx_cronologico").copy()
log("tabla", t.shape, "DEV", int(t.es_dev.sum()), "lockbox", int((~t.es_dev).sum()))

M = bal.reindex(t.index)["sn_metal_kg"]; D = bal.reindex(t.index)["sn_dross_kg"]
P = bal.reindex(t.index)["sn_polvo_kg"]; E = bal.reindex(t.index)["sn_escoria_final_balance_kg"]
feed = bal.reindex(t.index)["feed_sn_total_kg"]
t["M_kg"], t["D_kg"], t["P_kg"], t["E_kg"] = M.to_numpy(), D.to_numpy(), P.to_numpy(), E.to_numpy()

# ============================================================================ 1) anatomia del ruido
def acf(x: pd.Series, lag: int) -> tuple[float, int]:
    a, b = x.to_numpy()[:-lag] if lag else x.to_numpy(), x.to_numpy()[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    r, _ = stats.pearsonr(a[m], b[m])
    return float(r), int(m.sum())


canales = {"M_kg": t["M_kg"], "D_kg": t["D_kg"], "P_kg": t["P_kg"], "E_kg": t["E_kg"],
           "f_metal": t["f_metal"], "f_dross": t["f_dross"], "f_polvo": t["f_polvo"],
           "sn_perdido_escoria_frac": t["sn_perdido_escoria_frac"], "cierre_sn_frac": t["cierre_sn_frac"],
           "recuperacion_refinada_pct": t["recuperacion_refinada_pct"], "recuperacion_real_pct": t["recuperacion_real_pct"],
           "rendimiento_proxy_batch": t["rendimiento_proxy_batch"]}

filas_ac = []
for nombre, serie in canales.items():
    for conjunto, mask in (("DEV", t.es_dev), ("total", pd.Series(True, index=t.index))):
        s = serie[mask].reset_index(drop=True)
        for lag in range(1, 9):
            r, n = acf(s, lag)
            filas_ac.append({"canal": nombre, "conjunto": conjunto, "lag": lag, "acf": r, "n": n})
ac_tab = pd.DataFrame(filas_ac)
ac_tab.to_csv(OUT / "e11_04_autocorr.csv", index=False)
log("autocorr ok")

# periodograma simple (total, detrend lineal, dropna)
filas_per = []
for nombre, serie in canales.items():
    s = serie.dropna()
    if len(s) < 20:
        continue
    x = sm.add_constant(np.arange(len(s)))
    resid = s.to_numpy() - sm.OLS(s.to_numpy(), x).fit().fittedvalues
    f, Pxx = periodogram(resid)
    per = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    top = np.argsort(Pxx)[::-1][:5]
    for k in top:
        if per[k] < 2 or not np.isfinite(per[k]):
            continue
        filas_per.append({"canal": nombre, "periodo_batches": float(per[k]), "potencia": float(Pxx[k]),
                          "potencia_rel": float(Pxx[k] / Pxx.sum())})
per_tab = pd.DataFrame(filas_per).sort_values(["canal", "potencia_rel"], ascending=[True, False])
per_tab.to_csv(OUT / "e11_04_periodograma.csv", index=False)
log("periodograma ok")

# patron por posicion en ciclos de 3 (idx mod 3)
t["idx_mod3"] = t["idx_cronologico"].astype(int) % 3
filas_m3 = []
for nombre, serie in canales.items():
    g = serie.groupby(t["idx_mod3"])
    kw = stats.kruskal(*[v.dropna() for _, v in g])
    for k, v in g:
        filas_m3.append({"canal": nombre, "idx_mod3": int(k), "media": float(v.mean()), "sd": float(v.std()),
                         "n": int(v.notna().sum()), "kruskal_p": float(kw.pvalue)})
m3_tab = pd.DataFrame(filas_m3)
m3_tab.to_csv(OUT / "e11_04_mod3.csv", index=False)
log("mod3 ok")

# correlaciones cruzadas entre canales a lag 0..3 (compensacion entre batches vecinos)
pares = [("f_polvo", "f_polvo"), ("f_dross", "f_dross"), ("f_metal", "f_metal"), ("M_kg", "D_kg"), ("M_kg", "P_kg"),
         ("D_kg", "P_kg"), ("f_metal", "f_dross"), ("f_metal", "f_polvo"), ("cierre_sn_frac", "cierre_sn_frac"),
         ("E_kg", "E_kg")]
filas_cc = []
for a, b in pares:
    xa, xb = canales[a].reset_index(drop=True), canales[b].reset_index(drop=True)
    for lag in range(0, 4):
        if lag == 0 and a == b:
            continue
        va = xa.to_numpy()[:-lag] if lag else xa.to_numpy()
        vb = xb.to_numpy()[lag:]
        m = np.isfinite(va) & np.isfinite(vb)
        r, p = stats.pearsonr(va[m], vb[m]) if m.sum() > 10 else (np.nan, np.nan)
        filas_cc.append({"canal_a": a, "canal_b": f"{b}_lag{lag}", "lag": lag, "r": float(r), "p": float(p), "n": int(m.sum())})
cc_tab = pd.DataFrame(filas_cc)
cc_tab.to_csv(OUT / "e11_04_crosscorr.csv", index=False)
log("crosscorr ok")

# ============================================================================ 2) KPIs candidatos
denom_a = t["M_kg"] + t["D_kg"] + t["P_kg"] + t["E_kg"]
t["kpi_a_refinado"] = 100 * t["M_kg"] / denom_a
t["kpi_b_sinpolvo"] = 100 * t["M_kg"] / (t["M_kg"] + t["D_kg"] + t["E_kg"])
for w in (3, 5):
    Psm = t["P_kg"].rolling(w, center=True, min_periods=1).mean()
    t[f"kpi_c{w}_polvosuav"] = 100 * t["M_kg"] / (t["M_kg"] + t["D_kg"] + Psm + t["E_kg"])
Msm3 = t["M_kg"].rolling(3, center=True, min_periods=1).mean()
Dsm3 = t["D_kg"].rolling(3, center=True, min_periods=1).mean()
Psm3 = t["P_kg"].rolling(3, center=True, min_periods=1).mean()
t["kpi_d3_todosuav"] = 100 * Msm3 / (Msm3 + Dsm3 + Psm3 + t["E_kg"])
t["kpi_e_legado"] = 0.5 * (t["kpi_a_refinado"] + t["kpi_a_refinado"].shift(-1))
t["kpi_f_real"] = t["recuperacion_real_pct"]
t["kpi_g_controlable"] = 100 * (1 - t["D_kg"] / denom_a - t["E_kg"] / denom_a)
t["kpi_g_dross_frac"] = t["D_kg"] / denom_a
t["kpi_g_escoria_frac"] = t["E_kg"] / denom_a
t["kpi_h_medido_cargado"] = 100 * (t["M_kg"] + t["D_kg"] + t["P_kg"]) / t["feed_sn_total_kg"]

KPIS_CAND = ["kpi_a_refinado", "kpi_b_sinpolvo", "kpi_c3_polvosuav", "kpi_c5_polvosuav", "kpi_d3_todosuav",
             "kpi_e_legado", "kpi_f_real", "kpi_g_controlable", "kpi_h_medido_cargado"]
KPIS_VENTANA = {"kpi_c3_polvosuav": 3, "kpi_c5_polvosuav": 5, "kpi_d3_todosuav": 3, "kpi_e_legado": 2}

t[["idx_cronologico", "es_dev", "fecha_batch"] + KPIS_CAND + ["kpi_g_dross_frac", "kpi_g_escoria_frac", "M_kg", "D_kg",
  "P_kg", "E_kg", "cierre_sn_frac", "x_R_gn", "x_R_carbon", "x_F_carbon", "x_F_exo2"]].to_csv(OUT / "e11_04_kpis_batch.csv")
log("kpis candidatos ok, sd=", {k: round(float(t[k].std()), 3) for k in KPIS_CAND})

# ============================================================================ 3) estadisticas por KPI
CONTROLES = L.CONTROLES
EXPOS = ["x_R_gn", "x_R_carbon", "x_F_carbon", "x_F_exo2"]
ESTADO_AGREG = ["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "F_lnirf_F0", "R_lnirf_fin"]


def acf1(serie: pd.Series) -> float:
    r, _ = acf(serie.reset_index(drop=True), 1)
    return r


def r2_oof_bloques(d: pd.DataFrame, y_col: str, X_cols: list[str], modelo: str, n_bloques: int = 5) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    dd = d.dropna(subset=[y_col] + X_cols).reset_index(drop=True)
    if len(dd) < 30:
        return np.nan
    bl = (np.arange(len(dd)) * n_bloques // len(dd))
    oof = np.full(len(dd), np.nan)
    for k in range(n_bloques):
        tr, te = bl != k, bl == k
        if modelo == "ridge":
            mu, sd = dd.loc[tr, X_cols].mean(), dd.loc[tr, X_cols].std() + 1e-12
            Xtr = (dd.loc[tr, X_cols] - mu) / sd; Xte = (dd.loc[te, X_cols] - mu) / sd
            m = Ridge(alpha=5.0).fit(Xtr, dd.loc[tr, y_col])
            oof[te] = m.predict(Xte)
        else:
            m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, random_state=42).fit(dd.loc[tr, X_cols], dd.loc[tr, y_col])
            oof[te] = m.predict(dd.loc[te, X_cols])
    return float(r2_score(dd[y_col], oof))


def t_hc3(y: pd.Series, X: pd.DataFrame, col: str) -> tuple[float, float]:
    d = pd.concat([y, X], axis=1).dropna()
    fit = sm.OLS(d[y.name], sm.add_constant(d[X.columns], has_constant="add")).fit(cov_type="HC3")
    return float(fit.tvalues[col]), float(fit.params[col])


filas_stats = []
for kpi in KPIS_CAND:
    fila = {"kpi": kpi}
    for conjunto, mask in (("DEV", t.es_dev), ("total", pd.Series(True, index=t.index))):
        s = t.loc[mask, kpi]
        fila[f"sd_{conjunto}"] = float(s.std())
        fila[f"acf1_{conjunto}"] = acf1(s)
        fila[f"corr_kpi_real_{conjunto}"] = float(s.corr(t.loc[mask, "recuperacion_real_pct"])) if kpi != "kpi_f_real" else 1.0
        dsub = t.loc[mask]
        fila[f"r2_ridge_{conjunto}"] = r2_oof_bloques(dsub, kpi, CONTROLES + ESTADO_AGREG, "ridge")
        fila[f"r2_hgb_{conjunto}"] = r2_oof_bloques(dsub, kpi, CONTROLES + ESTADO_AGREG, "hgb")
        Xr = dsub[EXPOS + CONTROLES]
        for e in EXPOS:
            tt, bb = t_hc3(dsub[kpi], Xr, e)
            fila[f"t_{e}_{conjunto}"] = tt; fila[f"beta_{e}_{conjunto}"] = bb
    filas_stats.append(fila)
    log(kpi, "sd_DEV", round(fila["sd_DEV"], 3), "acf1_DEV", round(fila["acf1_DEV"], 3),
        "r2_ridge_DEV", round(fila["r2_ridge_DEV"], 3), "t_gn_DEV", round(fila["t_x_R_gn_DEV"], 2),
        "t_carbonR_DEV", round(fila["t_x_R_carbon_DEV"], 2))
stats_tab = pd.DataFrame(filas_stats)
stats_tab.to_csv(OUT / "e11_04_kpi_stats.csv", index=False)
log("stats por KPI ok")

# ============================================================================ 3b) prueba anidada (gkf y crono) por KPI
filas_pa = []
dev = t[t.es_dev].copy()
for kpi in KPIS_CAND:
    for esquema in ("gkf", "crono"):
        try:
            r = L.prueba_anidada(dev, ["x_R_gn", "x_R_carbon"], kpi=kpi, esquema=esquema,
                                 signos={"x_R_gn": -1, "x_R_carbon": +1})
        except Exception as ex:
            log("prueba_anidada fallo", kpi, esquema, ex); continue
        r.pop("score")
        r["kpi"] = kpi; r["esquema"] = esquema
        filas_pa.append(r)
pa_tab = pd.DataFrame(filas_pa)
pa_tab.to_csv(OUT / "e11_04_prueba_anidada.csv", index=False)
log("prueba anidada ok")

# ============================================================================ 4) HAC + placebo para KPI con ventana
filas_hac = []
for kpi, w in KPIS_VENTANA.items():
    dd = t.dropna(subset=[kpi] + EXPOS + CONTROLES).reset_index(drop=True)
    X = sm.add_constant(dd[EXPOS + CONTROLES], has_constant="add")
    fit_hc3 = sm.OLS(dd[kpi], X).fit(cov_type="HC3")
    fit_hac = sm.OLS(dd[kpi], X).fit(cov_type="HAC", cov_kwds={"maxlags": w})
    # placebo: exposicion del batch b+3 -> KPI suavizado de b
    dd_pl = dd.copy()
    for e in EXPOS:
        dd_pl[f"{e}_lead3"] = dd_pl[e].shift(-3)
    dd_pl = dd_pl.dropna(subset=[f"{e}_lead3" for e in EXPOS])
    Xpl = sm.add_constant(dd_pl[[f"{e}_lead3" for e in EXPOS] + CONTROLES], has_constant="add")
    fit_pl = sm.OLS(dd_pl[kpi], Xpl).fit(cov_type="HAC", cov_kwds={"maxlags": w})
    for e in EXPOS:
        filas_hac.append({"kpi": kpi, "ventana": w, "expo": e, "t_hc3": float(fit_hc3.tvalues[e]),
                          "t_hac": float(fit_hac.tvalues[e]), "beta": float(fit_hac.params[e]),
                          "t_placebo_lead3_hac": float(fit_pl.tvalues[f"{e}_lead3"])})
hac_tab = pd.DataFrame(filas_hac)
hac_tab.to_csv(OUT / "e11_04_placebo_hac.csv", index=False)
log("HAC + placebo ok")

log("listo. sd relativa a=1.0:", {k: round(float(t[k].std() / t['kpi_a_refinado'].std()), 3) for k in KPIS_CAND})
