"""ST-12 (Fable 5.1) -- formalizacion del hallazgo de ST-09: la matriz de escoria DURANTE la Reduccion (IRF) replica en DEV y
lockbox. Candidato por escalon:
    lnIRF_R[t] = ln( %FeO[t] / (%SiO2[t] + %Al2O3[t] + %CaO[t]) )   al cierre de cada escalon de Reduccion (estado)
    agregacion: media ponderada por tiempo (estado intensivo) ; tambien ultimo valor y primer valor.
Descomposicion: como en Reduccion no entra carga, ln IRF[t] = ln IRF[ini R] + ln(FeO_inv[t]/FeO_inv[ini]) - ln(otros[t]/otros[ini])
    ~ heel/estado inicial + retencion de FeO (la componente del SDI) -> el target une el estado heredado y la accion del batch.
Bateria: rho por bloque + IC bootstrap, OLS con controles, batch siguiente, Monte Carlo de ruido (50 rep), PLM v4 (theta palancas),
comparacion con SDI y con la combinacion SDI + lnIRF_R. Salidas: st_12_*.csv, st_12_log.txt, figs/st_12_*.png
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parent.parent))
import st_targets as st  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
t0 = time.time(); rng = np.random.default_rng(42)
df = pd.read_pickle(AQUI / "cache_df_st.pkl").sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
batch = pd.read_csv(AQUI / "st_batch.csv", index_col=0).sort_values("idx_cronologico"); orden = list(batch.index)
KPI = "rendimiento_proxy_batch"
dev_idx = batch.index[~batch.es_lockbox]; lb_idx = batch.index[batch.es_lockbox]; ter = np.array_split(dev_idx, 3)
BL = {"dev": dev_idx, "lockbox": lb_idx, "total": batch.index, "T1": ter[0], "T2": ter[1], "T3": ter[2]}
nxt = {orden[i]: orden[i + 1] for i in range(len(orden) - 1)}
kpi_next = pd.Series({k: batch[KPI].get(v, np.nan) for k, v in nxt.items()}).reindex(orden)

def targets_escalon(d: pd.DataFrame) -> pd.DataFrame:
    R = d.fase_proceso == "Reducción"
    irf = d["ley_feo_escoria_pct"] / (d["ley_sio2_escoria_pct"] + d["ley_al2o3_escoria_pct"] + d["ley_cao_escoria_pct"])
    out = pd.DataFrame(index=d.index)
    out["lnirf_R"] = np.log(irf.where(irf > 0)).where(R)
    ok = R & d.sn_inv_prev.gt(20) & d.sn_inventario_escoria_est_kg.gt(20) & d.feo_inv_prev.gt(100) & d.feo_inventario_escoria_est_kg.gt(100) & ~d.es_primer_escalon_batch
    out["sdi"] = (np.log(d.sn_inv_prev / d.sn_inventario_escoria_est_kg) + st.W_SDI * np.log(d.feo_inventario_escoria_est_kg / d.feo_inv_prev)).where(ok)
    out["ln_feo_ret"] = np.log(d.feo_inventario_escoria_est_kg / d.feo_inv_prev).where(ok)
    out["dt"] = d["delta_tiempo"]; out["Batch"] = d["Batch"]; out["orden"] = d["orden_escalon_fase"]; out["R"] = R
    return out

def agregar(te: pd.DataFrame) -> pd.DataFrame:
    g = te[te.R].groupby("Batch")
    def tw(s):
        x = s.dropna(); w = te.loc[x.index, "dt"]; return float(np.average(x, weights=w)) if len(x) and w.sum() > 0 else np.nan
    A = pd.DataFrame({"lnIRF_R_twmean": g["lnirf_R"].apply(tw), "lnIRF_R_last": g["lnirf_R"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
                      "lnIRF_R_first": g["lnirf_R"].apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
                      "SDI": g["sdi"].sum(min_count=1), "ln_feo_ret": g["ln_feo_ret"].sum(min_count=1)}).reindex(orden)
    F0 = df[(df.fase_proceso == "Fusión") & (df.orden_escalon_fase == 0)].set_index("Batch")["indice_irf"]
    A["lnIRF_F0"] = np.log(F0.where(F0 > 0)).reindex(orden)
    A["lnIRF_finF"] = np.log(df[df.fase_proceso == "Fusión"].dropna(subset=["indice_irf"]).groupby("Batch")["indice_irf"].last().where(lambda s: s > 0)).reindex(orden)
    return A

te = targets_escalon(df); A = agregar(te)
def rho(s, idx, y=None):
    y = batch[KPI] if y is None else y
    d = pd.concat([s.reindex(idx), y.reindex(idx)], axis=1).dropna()
    return (spearmanr(d.iloc[:, 0], d.iloc[:, 1]) + (len(d),)) if len(d) > 8 else (np.nan, np.nan, len(d))
def boot(s, idx, n=2000):
    d = pd.concat([s.reindex(idx), batch[KPI].reindex(idx)], axis=1).dropna().to_numpy(); v = []
    for _ in range(n):
        i = rng.integers(0, len(d), len(d)); v.append(spearmanr(d[i, 0], d[i, 1])[0])
    return np.percentile(v, 2.5), np.percentile(v, 97.5)
def ols(s, idx, y=None):
    y = batch[KPI] if y is None else y
    d = pd.concat([s.rename("x").reindex(idx), y.rename("y").reindex(idx), batch.loc[idx, st.CONTROLES_BATCH + ["idx_cronologico"]]], axis=1).dropna()
    regs = [c for c in d.columns if c != "y" and d[c].std() > 1e-9]; X = (d[regs] - d[regs].mean()) / d[regs].std()
    m = sm.OLS(d.y, sm.add_constant(X)).fit(cov_type="HC3"); return m.params["x"], m.pvalues["x"], m.rsquared_adj

# --- descomposicion: lnIRF_R_twmean ~ lnIRF_F0 (heel) + ln_feo_ret (accion)
d = pd.concat([A[["lnIRF_R_twmean", "lnIRF_F0", "ln_feo_ret", "lnIRF_finF"]]], axis=1).dropna()
m = sm.OLS(d.lnIRF_R_twmean, sm.add_constant(d[["lnIRF_finF", "ln_feo_ret"]])).fit()
log(f"descomposicion lnIRF_R_twmean ~ lnIRF_finF + ln_feo_ret: R2 {m.rsquared:.3f}; coef {m.params.round(3).to_dict()}")
log("rho lnIRF_R_twmean vs lnIRF_F0 %.3f, vs lnIRF_finF %.3f, vs ln_feo_ret %.3f, vs SDI %.3f" % tuple(spearmanr(A[a].fillna(A[a].median()), A[b].fillna(A[b].median()))[0] for a, b in [("lnIRF_R_twmean", "lnIRF_F0"), ("lnIRF_R_twmean", "lnIRF_finF"), ("lnIRF_R_twmean", "ln_feo_ret"), ("lnIRF_R_twmean", "SDI")]))

# --- metricas
filas = []
comb = {"lnIRF_R_twmean": A.lnIRF_R_twmean, "lnIRF_R_last": A.lnIRF_R_last, "lnIRF_R_first": A.lnIRF_R_first, "SDI": A.SDI, "ln_feo_ret": A.ln_feo_ret,
        "lnIRF_finF": A.lnIRF_finF, "lnIRF_F0 (heel)": A.lnIRF_F0}
zd = lambda s: (s - s.reindex(dev_idx).mean()) / s.reindex(dev_idx).std()
comb["z(SDI)+z(lnIRF_R_twmean)"] = zd(A.SDI) + zd(A.lnIRF_R_twmean)
comb["z(lnIRF_R_twmean)+z(lnIRF_finF)"] = zd(A.lnIRF_R_twmean) + zd(A.lnIRF_finF)
log("\n=== rho Spearman vs rendimiento por bloque; OLS por sd con controles; batch siguiente ===")
for k, s in comb.items():
    f = {"target": k}
    for b, idx in BL.items():
        r, p, n = rho(s, idx); f[f"rho_{b}"] = r; f[f"p_{b}"] = p; f[f"n_{b}"] = n
    f["ci_lo_dev"], f["ci_hi_dev"] = boot(s, dev_idx); f["ci_lo_lockbox"], f["ci_hi_lockbox"] = boot(s, lb_idx)
    for b in ["dev", "lockbox", "total"]:
        c, p, r2 = ols(s, BL[b]); f[f"ols_{b}"] = c; f[f"ols_p_{b}"] = p; f[f"ols_r2adj_{b}"] = r2
        r, p2, n = rho(s, BL[b], kpi_next); f[f"rho_next_{b}"] = r; f[f"p_next_{b}"] = p2
    for kk in ["f_dross", "f_polvo", "recuperacion_real_pct"]:
        f[f"rho_dev_{kk}"] = rho(s, dev_idx, batch[kk])[0]
    dd = pd.concat([s.rename("x").reindex(dev_idx), batch.loc[dev_idx, KPI]], axis=1).dropna(); q = pd.qcut(dd.x, 5, labels=False, duplicates="drop"); gq = dd.groupby(q)[KPI].mean()
    f["monotonia"] = int((np.diff(gq.values) > 0).sum()); f["dQ5Q1"] = gq.iloc[-1] - gq.iloc[0]
    filas.append(f)
    log(f"{k:32s} DEV {f['rho_dev']:+.3f} [{f['ci_lo_dev']:+.2f},{f['ci_hi_dev']:+.2f}] p={f['p_dev']:.1e} | T1/T2/T3 {f['rho_T1']:+.2f}/{f['rho_T2']:+.2f}/{f['rho_T3']:+.2f} | lockbox {f['rho_lockbox']:+.3f} (p={f['p_lockbox']:.3f}) | total {f['rho_total']:+.3f} (p={f['p_total']:.1e}) | OLS dev {f['ols_dev']:+.2f} (p={f['ols_p_dev']:.3f}) lb {f['ols_lockbox']:+.2f} (p={f['ols_p_lockbox']:.3f}) | next lb {f['rho_next_lockbox']:+.2f} | dross {f['rho_dev_f_dross']:+.2f} polvo {f['rho_dev_f_polvo']:+.2f} | mono {f['monotonia']} dQ {f['dQ5Q1']:+.1f}")
met = pd.DataFrame(filas); met.to_csv(AQUI / "st_12_metricas.csv", index=False)

# --- Monte Carlo del ruido de ensayo (50 rep) para lnIRF_R_twmean y SDI
log("\n=== Monte Carlo ruido de ensayo (50 rep) ===")
res = {"lnIRF_R_twmean": [], "SDI": []}
for _ in range(50):
    d2 = df.copy()
    d2["ley_cao_escoria_pct"] = d2["ley_cao_escoria_pct"] * (1 + rng.normal(0, 0.02, len(d2)))
    d2["ley_sn_escoria_pct"] = (d2["ley_sn_escoria_pct"] + rng.normal(0, 0.05, len(d2))).clip(lower=0)
    d2["ley_feo_escoria_pct"] = (d2["ley_feo_escoria_pct"] + rng.normal(0, 0.3, len(d2))).clip(lower=0)
    d2["ley_sio2_escoria_pct"] = (d2["ley_sio2_escoria_pct"] + rng.normal(0, 0.3, len(d2))).clip(lower=0)
    d2["ley_al2o3_escoria_pct"] = (d2["ley_al2o3_escoria_pct"] + rng.normal(0, 0.1, len(d2))).clip(lower=0)
    cum = d2.groupby("Batch")["feed_CaO_kgh"].cumsum(); masa = cum / (d2["ley_cao_escoria_pct"].where(d2["ley_cao_escoria_pct"] > 0.5) / 100)
    d2["sn_inventario_escoria_est_kg"] = masa * d2["ley_sn_escoria_pct"] / 100; d2["feo_inventario_escoria_est_kg"] = masa * d2["ley_feo_escoria_pct"] / 100
    d2["sn_inv_prev"] = d2.groupby("Batch")["sn_inventario_escoria_est_kg"].shift(1); d2["feo_inv_prev"] = d2.groupby("Batch")["feo_inventario_escoria_est_kg"].shift(1)
    A2 = agregar(targets_escalon(d2))
    res["lnIRF_R_twmean"].append(rho(A2.lnIRF_R_twmean, dev_idx)[0]); res["SDI"].append(rho(A2.SDI, dev_idx)[0])
for k, v in res.items():
    log(f"{k}: rho_dev bajo ruido media {np.mean(v):.3f} sd {np.std(v):.3f} P5 {np.percentile(v,5):.3f} P95 {np.percentile(v,95):.3f}")

# --- PLM v4: efecto de las palancas sobre lnirf_R por escalon (estado curado de Reduccion)
log("\n=== PLM v4 sobre lnIRF_R (escalon) ===")
d3 = mp4.agregar_columnas_v4(df.copy()); d3["lnirf_R"] = te["lnirf_R"]
S = mp4.ESTADO_V4[("Reducción", "sn_kg")]; PRIM = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"]
dd = d3[(d3.fase_proceso == "Reducción") & (~d3.es_primer_escalon_batch) & d3.Batch.isin(dev_idx)].dropna(subset=S + PRIM + ["lnirf_R"]).reset_index(drop=True)
mplm = mp4.ModeloPLM(estado=S, palancas=PRIM, target="lnirf_R").fit(dd, n_boot=200)
tt = mplm.tabla_theta(); tt["theta_1sd_rel"] = tt.theta_1sd / dd.lnirf_R.std(); tt.to_csv(AQUI / "st_12_theta_lnirf.csv")
log(f"R2 OOF interno {mplm.r2_oof_interno:.3f}; sd target {dd.lnirf_R.std():.3f}"); log(tt[["theta_1sd", "ci_lo_1sd", "ci_hi_1sd", "theta_1sd_rel", "significativo"]].round(4).to_string())

# --- figura
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (k, s) in zip(axes, [("lnIRF_R_twmean", A.lnIRF_R_twmean), ("SDI", A.SDI)]):
    for idx, col, lab in [(dev_idx, "#1f77b4", "DEV"), (lb_idx, "#d62728", "lockbox")]:
        d = pd.concat([s.reindex(idx), batch.loc[idx, KPI]], axis=1).dropna(); ax.scatter(d.iloc[:, 0], d.iloc[:, 1], s=14, alpha=0.6, color=col, label=f"{lab} ρ={spearmanr(d.iloc[:,0], d.iloc[:,1])[0]:+.2f}")
    ax.set_xlabel(k); ax.set_ylabel("rendimiento proxy [%]"); ax.legend(); ax.set_title(k)
plt.tight_layout(); fig.savefig(AQUI / "figs" / "st_12_lnirf_vs_sdi.png", dpi=120)
te[["Batch", "orden", "lnirf_R", "sdi"]].to_csv(AQUI / "st_12_targets_escalon.csv", index=False)
log(f"tiempo {time.time()-t0:.0f}s"); (AQUI / "st_12_log.txt").write_text("\n".join(LOG), encoding="utf-8")
