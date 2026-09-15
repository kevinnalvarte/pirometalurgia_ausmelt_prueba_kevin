"""ST-05 (Fable 5.1) -- hipotesis de segunda ronda tras ST-01.

H1  Reduccion: en DEV manda la sobre-reduccion (FeO retenido, rho +0.31) y en el lockbox el agotamiento de Sn (k_app +0.35);
    ambos son fracciones de inventario y estan en trade-off (rho -0.38). Un "indice de agotamiento selectivo" en
    log-inventarios (la masa de escoria se cancela):
        SDI_w[t] = ln(Sn_inv[t-1]/Sn_inv[t]) + w * ln(FeO_inv[t]/FeO_inv[t-1])        (por escalon)
        batch    = sum_t SDI_w[t] = ln(Sn_ini/Sn_fin) + w * ln(FeO_fin/FeO_ini)          (telescopica)
    w = 0 -> k_app*dt (R07); w -> inf -> ln(FeO retenido) (R08); w = 1 -> -d log(Sn/FeO) (R15).
    Pregunta: existe w con rho > 0 en los 3 tercios de DEV y en el lockbox? Cual es w* elegido SOLO con DEV?
H1b Lo mismo en masa normalizada por Sn cargado: (Sn_ext - lam * FeO_ext) / Sn_cargado_batch (R12 - lam * R20).
H2  Fusion: el IRF al fin de Fusion (F10) podria ser un artefacto del PLAN de carga (mas mineral de Fe / menos cal ->
    IRF alto) y no de la operacion. Test: OLS HC3 con controles del plan (Fe mineral t4, dross Fe t5, cal/carga,
    C/Sn, GN/carga, feed total) y descomposicion IRF_fin = IRF_F0 + sum(dIRF): cual parte correlaciona?
H3  Trade-off entre fases: un IRF alto al fin de Fusion, deja mas FeO para sobre-reducir en R? (rho F10 vs R08).
Salidas: st_05_sdi_sweep.csv, st_05_irf_controles.csv, st_05_log.txt, figs/st_05_*.png
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parent.parent))
import st_targets as st  # noqa: E402

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

t0 = time.time()
df = pd.read_pickle(AQUI / "cache_df_st.pkl")
batch = pd.read_csv(AQUI / "st_batch.csv", index_col=0).sort_values("idx_cronologico")
dev = batch[~batch.es_lockbox]; lb = batch[batch.es_lockbox]
tercios = np.array_split(dev.index, 3)
bloques = {"dev_T1": tercios[0], "dev_T2": tercios[1], "dev_T3": tercios[2], "lockbox": lb.index, "dev": dev.index, "total": batch.index}

def rho_bloques(serie: pd.Series, kpi="rendimiento_proxy_batch") -> dict:
    out = {}
    for k, idx in bloques.items():
        d = pd.concat([serie.reindex(idx), batch.loc[idx, kpi]], axis=1).dropna()
        r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1]) if len(d) > 5 else (np.nan, np.nan)
        out[f"rho_{k}"] = r; out[f"p_{k}"] = p; out[f"n_{k}"] = len(d)
    return out

# ----------------------------------------------------------------------------- H1: SDI_w
R = df[(df.fase_proceso == "Reducción") & (~df.es_primer_escalon_batch)].copy()
ok = R.sn_inv_prev.gt(20) & R.sn_inventario_escoria_est_kg.gt(20) & R.feo_inv_prev.gt(100) & R.feo_inventario_escoria_est_kg.gt(100)
R["ln_sn_dep"] = np.log(R.sn_inv_prev / R.sn_inventario_escoria_est_kg).where(ok)
R["ln_feo_ret"] = np.log(R.feo_inventario_escoria_est_kg / R.feo_inv_prev).where(ok)
agg = R.groupby("Batch")[["ln_sn_dep", "ln_feo_ret"]].sum(min_count=1)
filas = []
log("=== H1: SDI_w = ln(Sn_ini/Sn_fin) + w*ln(FeO_fin/FeO_ini) (Reduccion, suma telescopica) ===")
for w in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 50]:
    s = agg.ln_sn_dep + w * agg.ln_feo_ret
    r = rho_bloques(s); r.update(w=w, variante="SDI_log"); filas.append(r)
    log(f"w={w:5.2f}  T1 {r['rho_dev_T1']:+.3f}  T2 {r['rho_dev_T2']:+.3f}  T3 {r['rho_dev_T3']:+.3f}  lockbox {r['rho_lockbox']:+.3f}  | DEV {r['rho_dev']:+.3f} (p={r['p_dev']:.2g})  total {r['rho_total']:+.3f}")
# solo la parte FeO (w->inf) y solo Sn
log("componentes: ln_sn_dep", {k: round(v, 3) for k, v in rho_bloques(agg.ln_sn_dep).items() if k.startswith("rho")})
log("componentes: ln_feo_ret", {k: round(v, 3) for k, v in rho_bloques(agg.ln_feo_ret).items() if k.startswith("rho")})

# H1b: masa normalizada por Sn cargado
log("=== H1b: (Sn_ext - lam*FeO_ext)/Sn_cargado (Reduccion) ===")
sn_cargado = df.groupby("Batch").feed_Sn_kgf.sum()
Rk = R.groupby("Batch")[["sn_ext", "feo_ext"]].sum(min_count=1)
for lam in [0, 0.3, 0.6, 1, 1.5, 2, 3, 5, 10]:
    s = (Rk.sn_ext - lam * Rk.feo_ext.clip(lower=0)) / sn_cargado.reindex(Rk.index)
    r = rho_bloques(s); r.update(w=lam, variante="masa_norm"); filas.append(r)
    log(f"lam={lam:5.2f}  T1 {r['rho_dev_T1']:+.3f}  T2 {r['rho_dev_T2']:+.3f}  T3 {r['rho_dev_T3']:+.3f}  lockbox {r['rho_lockbox']:+.3f}  | DEV {r['rho_dev']:+.3f} (p={r['p_dev']:.2g})")
pd.DataFrame(filas).to_csv(AQUI / "st_05_sdi_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
sw = pd.DataFrame(filas); s1 = sw[sw.variante == "SDI_log"]
for k in ["rho_dev_T1", "rho_dev_T2", "rho_dev_T3", "rho_lockbox", "rho_dev"]:
    ax.plot(s1.w, s1[k], "o-", label=k, lw=2 if k == "rho_dev" else 1)
ax.axhline(0, color="k", lw=0.8); ax.set_xscale("symlog"); ax.set_xlabel("w (peso del FeO retenido en log)"); ax.set_ylabel("Spearman vs rendimiento")
ax.set_title("SDI_w = ln(Sn_ini/Sn_fin) + w·ln(FeO_fin/FeO_ini): rho por bloque"); ax.legend(fontsize=8); plt.tight_layout()
fig.savefig(AQUI / "figs" / "st_05_sdi_sweep.png", dpi=120); plt.close(fig)

# ----------------------------------------------------------------------------- H2: IRF y plan de carga
log("=== H2: IRF fin Fusion vs controles del plan de carga ===")
F = df[df.fase_proceso == "Fusión"]
plan = F.groupby("Batch").agg(fe_mineral_t=("feed_Fe_kgh", "sum"), dross_fe_t=("feed_dross_Fe_kgh", "sum"), cal_t=("feed_CaO_kgh", "sum"),
                              carbon_F_t=("feed_Carbon_kgh", "sum"), feed_total_t=("feed_total_kgh", "sum"), feed_sn_F_t=("feed_Sn_kgf", "sum"),
                              gn_F=("volumen_gas_natural_inyectado_lanza_escalon_nm3", "sum"), o2_F=("volumen_o2_inyectado_lanza_escalon_nm3", "sum"),
                              aire_F=("volumen_aire_inyectado_lanza_escalon_nm3", "sum"), T_F=("temperatura_horno_celsius", "mean"),
                              irf_F0=("indice_irf", "first"), irf_fin=("indice_irf", "last"), sio2_fin=("ley_sio2_escoria_pct", "last"),
                              cao_fin=("ley_cao_escoria_pct", "last"), feo_fin=("ley_feo_escoria_pct", "last"), al2o3_fin=("ley_al2o3_escoria_pct", "last"))
for c in ["fe_mineral_t", "dross_fe_t", "cal_t", "carbon_F_t", "feed_total_t", "feed_sn_F_t"]:
    plan[c] = plan[c] / 1000
plan["cal_por_carga"] = plan.cal_t / plan.feed_total_t
plan["fe_por_carga"] = (plan.fe_mineral_t + plan.dross_fe_t) / plan.feed_total_t
plan["c_por_sn"] = plan.carbon_F_t / plan.feed_sn_F_t
plan["gn_por_carga"] = plan.gn_F / plan.feed_total_t
plan["exceso_o2_F"] = 100 * (plan.o2_F + 0.21 * plan.aire_F - 2 * plan.gn_F) / (2 * plan.gn_F)
plan["d_irf"] = plan.irf_fin - plan.irf_F0
B = batch.join(plan, how="left")
B["irf_fin_st"] = B["st_F10_irf_next"]

def ols_z(d, dep, regs, idx):
    dd = d.loc[idx, [dep] + regs].dropna()
    dd = dd.loc[:, dd.std() > 1e-9]
    regs2 = [r for r in regs if r in dd.columns]
    X = (dd[regs2] - dd[regs2].mean()) / dd[regs2].std()
    m = sm.OLS(dd[dep], sm.add_constant(X)).fit(cov_type="HC3")
    return m, len(dd)

filas2 = []
CONTROLES = st.CONTROLES_BATCH
PLAN = ["fe_por_carga", "cal_por_carga", "c_por_sn", "gn_por_carga", "exceso_o2_F", "T_F"]
for nombre, regs in [("A_solo_irf", ["irf_fin_st"]), ("B_irf+controles", ["irf_fin_st"] + CONTROLES),
                     ("C_irf+controles+plan", ["irf_fin_st"] + CONTROLES + PLAN), ("D_plan_sin_irf", CONTROLES + PLAN),
                     ("E_irf_F0_y_dIRF", ["irf_F0", "d_irf"] + CONTROLES + PLAN)]:
    for muestra in ["dev", "lockbox", "total"]:
        m, n = ols_z(B, "rendimiento_proxy_batch", regs, bloques[muestra])
        for v in m.params.index:
            if v == "const":
                continue
            filas2.append(dict(modelo=nombre, muestra=muestra, variable=v, coef_por_sd=m.params[v], ci_lo=m.conf_int().loc[v, 0], ci_hi=m.conf_int().loc[v, 1], p=m.pvalues[v], r2_adj=m.rsquared_adj, n=n))
        vs = [v for v in m.params.index if v.startswith("irf") or v == "d_irf"]
        log(f"{nombre:22s} {muestra:8s} n={n:3d} R2adj={m.rsquared_adj:.3f} " + "  ".join(f"{v}={m.params[v]:+.2f} (p={m.pvalues[v]:.3f})" for v in vs))
pd.DataFrame(filas2).to_csv(AQUI / "st_05_irf_controles.csv", index=False)
# que explica el IRF fin? (determinantes de plan vs operacion)
m, n = ols_z(B, "irf_fin_st", PLAN + CONTROLES, bloques["total"])
log("Determinantes del IRF fin Fusion (OLS z, total): R2adj=%.3f" % m.rsquared_adj)
log(pd.DataFrame({"coef_sd": m.params, "p": m.pvalues}).drop("const").round(3).sort_values("p").to_string())
# componentes del IRF: numerador vs denominador
log("rho vs rendimiento (DEV) de componentes del IRF fin: " + "  ".join(f"{c}={spearmanr(B.loc[dev.index, c], B.loc[dev.index, 'rendimiento_proxy_batch'], nan_policy='omit')[0]:+.3f}" for c in ["feo_fin", "sio2_fin", "cao_fin", "al2o3_fin", "irf_F0", "d_irf"]))
log("rho vs rendimiento (lockbox): " + "  ".join(f"{c}={spearmanr(B.loc[lb.index, c], B.loc[lb.index, 'rendimiento_proxy_batch'], nan_policy='omit')[0]:+.3f}" for c in ["feo_fin", "sio2_fin", "cao_fin", "al2o3_fin", "irf_F0", "d_irf"]))
# H3 trade-off entre fases
for k in ["dev", "lockbox", "total"]:
    idx = bloques[k]
    log(f"H3 {k}: rho(IRF fin F, FeO retenido R08) = {spearmanr(B.loc[idx,'st_F10_irf_next'], B.loc[idx,'st_R08_feo_ret_frac'], nan_policy='omit')[0]:+.3f}; "
        f"rho(IRF fin F, FeO reducido kg) = {spearmanr(B.loc[idx,'st_F10_irf_next'], -B.loc[idx,'st_R02_feo_ret'], nan_policy='omit')[0]:+.3f}; "
        f"rho(IRF fin F, f_polvo) = {spearmanr(B.loc[idx,'st_F10_irf_next'], B.loc[idx,'f_polvo'], nan_policy='omit')[0]:+.3f}; "
        f"rho(IRF fin F, f_dross) = {spearmanr(B.loc[idx,'st_F10_irf_next'], B.loc[idx,'f_dross'], nan_policy='omit')[0]:+.3f}")
# IRF por escalon de Fusion: correlacion del IRF al cierre de cada escalon con el KPI (donde se decide?)
log("IRF al cierre de cada escalon de Fusion vs rendimiento (DEV / lockbox):")
for o in range(7):
    s = F[F.orden_escalon_fase == o].set_index("Batch")["indice_irf"]
    r = rho_bloques(s)
    log(f"  F{o}: dev {r['rho_dev']:+.3f} (p={r['p_dev']:.3f})  lockbox {r['rho_lockbox']:+.3f} (p={r['p_lockbox']:.3f})")
log(f"tiempo {time.time()-t0:.1f}s")
(AQUI / "st_05_log.txt").write_text("\n".join(LOG), encoding="utf-8")
