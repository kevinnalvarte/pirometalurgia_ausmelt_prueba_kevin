"""ST-13 (Fable 5.1) -- aplicar la recomendacion: cambiar el KPI. Diagnostico de la atribucion de dross (D) y polvo (P) por batch.

Hipotesis: D y P se atribuyen con retardo o por periodo (autocorrelacion anomala a lag 3, ST-11). Si D(b) refleja el proceso
del batch b-k, la correlacion de D con senales de proceso (FeO reducido, Sn cargado, Fe cargado) sera maxima a lag k, no a 0.
Con el lag identificado se construye un KPI realineado y se compara con: recuperacion real, KPI de ventana movil (sumas).
Salidas: st_13_crosscorr.csv, st_13_kpis_alternativos.csv, st_13_log.txt, figs/st_13_*.png
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parent.parent))
import st_targets as st  # noqa: E402
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
t0 = time.time()
df = st.cargar_df(); b = st.construir_batch(df).sort_values("idx_cronologico"); orden = list(b.index)
g = df.groupby("Batch")
M = g["sn_en_metal_crudo_batch_t"].first().reindex(orden); D = g["sn_en_dross_fe_batch_t"].first().reindex(orden); P = g["sn_en_polvo_fundicion_batch_t"].first().reindex(orden)
feed = g["feed_Sn_kgf"].sum().reindex(orden) / 1000
R = df[df.fase_proceso == "Reducción"]
feo_R = -b["st_R02_feo_ret"]                       # kg FeO reducido en Reduccion (orientacion original)
fe_feed = (g["feed_Fe_kgh"].sum() + g["feed_dross_Fe_kgh"].sum()).reindex(orden) / 1000
dross_feed = g["feed_dross_Fe_kgh"].sum().reindex(orden) / 1000
pellets_feed = g["feed_pellets_t7_kgh"].sum().reindex(orden) / 1000
sn_dross_feed = g["sn_dross_Fe_t5_kgf"].sum().reindex(orden) / 1000
sn_pellets_feed = g["sn_pellets_t7_kgf"].sum().reindex(orden) / 1000
Tgas = df.groupby("Batch")["temperatura_gas_pre_bhf_celsius"].mean().reindex(orden)
gas_total = (g["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum() + g["volumen_o2_inyectado_lanza_escalon_nm3"].sum() + g["volumen_aire_inyectado_lanza_escalon_nm3"].sum()).reindex(orden)
Tbano = df.groupby("Batch")["temperatura_horno_celsius"].mean().reindex(orden)
sdi = b["st_R22_sdi"]; lnirf = b["st_R23_lnirf_R"]; irfF = b["st_F10_irf_next"]
lb = b.es_lockbox.values

def xcorr(y: pd.Series, x: pd.Series, lags=range(-6, 7)):
    """rho(y[b], x[b - k]) para k en lags: k>0 => la salida y del batch b responde al proceso de k batches ANTES."""
    out = {}
    for k in lags:
        xs = x.shift(k); d = pd.concat([y, xs], axis=1).dropna()
        out[k] = spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) > 20 else np.nan
    return out

log("=== 1) correlacion cruzada rho(salida[b], senal[b-k]) ; k>0 = la salida responde a batches anteriores ===")
filas = []
for yname, y in [("D_dross_t", D), ("P_polvo_t", P), ("M_metal_t", M)]:
    for xname, x in [("Sn_cargado_t", feed), ("FeO_reducido_R_kg", feo_R), ("Fe_cargado_t", fe_feed), ("dross_Fe_cargado_t", dross_feed), ("pellets_cargados_t", pellets_feed),
                     ("Sn_en_dross_cargado_t", sn_dross_feed), ("Sn_en_pellets_cargado_t", sn_pellets_feed), ("SDI", sdi), ("lnIRF_R", lnirf), ("IRF_finF", irfF), ("T_gas_bhf", Tgas), ("gas_total", gas_total), ("T_bano", Tbano), ("M_metal_t", M), ("D_dross_t", D), ("P_polvo_t", P)]:
        xc = xcorr(y, x); kmax = max(xc, key=lambda k: abs(xc[k]) if not np.isnan(xc[k]) else -1)
        filas.append(dict(salida=yname, senal=xname, **{f"lag{k}": v for k, v in xc.items()}, lag_max=kmax, rho_max=xc[kmax], rho_lag0=xc[0]))
        log(f"{yname:10s} vs {xname:24s} " + " ".join(f"{k:+d}:{xc[k]:+.2f}" for k in range(-3, 6)) + f"   | max en lag {kmax:+d} ({xc[kmax]:+.2f})")
xc_df = pd.DataFrame(filas); xc_df.to_csv(AQUI / "st_13_crosscorr.csv", index=False)

# ----------------------------------------------------------------------------- 2) KPIs alternativos
log("\n=== 2) KPIs alternativos y correlacion con los targets teoricos ===")
def kpi(Mv, Dv, Pv):
    tot = Mv + Dv + Pv; return (100 * Mv / tot).where(tot > 0)
K = {"K0_proxy": kpi(M, D, P), "K1_recuperacion_real": 100 * M / feed}
snfin = R.dropna(subset=["sn_inventario_escoria_est_kg"]).groupby("Batch")["sn_inventario_escoria_est_kg"].last().reindex(orden) / 1000
K["K2_recup_corr_escoria"] = 100 * M / (feed - snfin.fillna(snfin.median()))
for w in [3, 5]:
    K[f"K3_ventana{w}_centrada"] = kpi(M.rolling(w, center=True, min_periods=w).sum(), D.rolling(w, center=True, min_periods=w).sum(), P.rolling(w, center=True, min_periods=w).sum())
for k in [1, 2, 3]:
    K[f"K5_DP_desplazados_{k}"] = kpi(M, D.shift(-k), P.shift(-k))
K["K6_solo_dross"] = 100 * (M + P) / (M + D + P); K["K7_M_sobre_MD"] = 100 * M / (M + D)
K["K8_M_por_Sn_menos_D"] = 100 * (M) / (feed - D)   # metal sobre Sn que no fue a dross: ignora la atribucion de polvo
Kdf = pd.DataFrame(K); Kdf.to_csv(AQUI / "st_13_kpis_alternativos.csv")
targets = {"SDI": sdi, "lnIRF_R": lnirf, "IRF_finF": irfF, "z(SDI)+z(lnIRF_R)": (sdi - sdi[~lb].mean()) / sdi[~lb].std() + (lnirf - lnirf[~lb].mean()) / lnirf[~lb].std()}
filas = []
for kn, kv in K.items():
    f = {"kpi": kn, "n": int(kv.notna().sum()), "sd": kv.std(), "r_con_K0": pearsonr(*pd.concat([kv, K["K0_proxy"]], axis=1).dropna().T.values)[0], "r_con_K1": pearsonr(*pd.concat([kv, K["K1_recuperacion_real"]], axis=1).dropna().T.values)[0]}
    for tn, tv in targets.items():
        for mn, mask in [("dev", ~lb), ("lockbox", lb), ("total", np.ones(len(lb), bool))]:
            d = pd.concat([tv[mask], kv[mask]], axis=1).dropna(); r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1]); f[f"{tn}_{mn}"] = r; f[f"{tn}_{mn}_p"] = p
    filas.append(f)
    log(f"{kn:24s} sd {f['sd']:5.2f} r(K0) {f['r_con_K0']:+.2f} r(K1) {f['r_con_K1']:+.2f} | " + " | ".join(f"{tn}: {f[tn+'_dev']:+.2f}/{f[tn+'_lockbox']:+.2f}/{f[tn+'_total']:+.2f}" for tn in targets))
res = pd.DataFrame(filas); res.to_csv(AQUI / "st_13_targets_vs_kpis.csv", index=False)
# autocorrelacion de los KPIs (para saber cuanto es estructura)
log("\nautocorrelacion lag1..4 de cada KPI:")
for kn, kv in K.items():
    v = kv.dropna(); log(f"  {kn:24s} " + " ".join(f"{k}:{pearsonr(v.iloc[k:].values, v.iloc[:-k].values)[0]:+.2f}" for k in [1, 2, 3, 4]))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, (yname, y) in zip(axes, [("D", D), ("P", P), ("M", M)]):
    for xname, x in [("Sn cargado", feed), ("FeO reducido R", feo_R), ("SDI", sdi), ("lnIRF_R", lnirf)]:
        xc = xcorr(y, x); ax.plot(list(xc.keys()), list(xc.values()), "o-", label=xname)
    ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="grey", ls=":"); ax.set_title(f"rho({yname}[b], senal[b-k])"); ax.set_xlabel("k"); ax.legend(fontsize=7)
plt.tight_layout(); fig.savefig(AQUI / "figs" / "st_13_crosscorr.png", dpi=120)
log(f"tiempo {time.time()-t0:.0f}s"); (AQUI / "st_13_log.txt").write_text("\n".join(LOG), encoding="utf-8")
