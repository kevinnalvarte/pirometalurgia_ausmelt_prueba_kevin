"""ST-06 (Fable 5.1) -- targets ganadores por escalon y su defensa cuantitativa.

Ganadores (decision en README.md de la carpeta):
  REDUCCION  sdi_w[t] = ln(Sn_inv[t-1]/Sn_inv[t]) + w * ln(FeO_inv[t]/FeO_inv[t-1])
             "indice de agotamiento selectivo": agotamiento logaritmico del Sn de la escoria mas w veces la retencion
             logaritmica del FeO (la masa de escoria se cancela en cada log-cociente: solo dependen de las leyes y
             del cociente de masas del trazador). Agregacion: suma telescopica = ln(Sn_ini/Sn_fin) + w*ln(FeO_fin/FeO_ini).
             w se elige SOLO con DEV: el menor w cuyo rho(DEV) >= 0.95 * max_w rho(DEV) (parsimonia: la componente de
             agotamiento de Sn evita el objetivo degenerado "no reducir nada").
             Componente dominante: ln_feo_ret = ln(FeO_inv[t]/FeO_inv[t-1]) (fraccion log de FeO retenida).
  FUSION     irf_next[t] = %FeO[t] / (%SiO2[t] + %Al2O3[t] + %CaO[t]) al cierre del escalon (indice IRF de planta);
             agregacion = ultimo valor (estado terminal de la fase). Objetivo operativo: conservar la matriz FeO-rica
             (no reducir FeO ni diluir con formadores de red durante la Fusion).
Salidas: st_06_metricas_ganadores.csv, st_06_w_seleccion.csv, st_06_efecto_siguiente_batch.csv, st_06_targets_escalon.csv,
         st_06_log.txt, figs/st_06_*.png
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parent.parent))
import st_targets as st  # noqa: E402

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

SEED = 42
t0 = time.time()
df = pd.read_pickle(AQUI / "cache_df_st.pkl").sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
batch = pd.read_csv(AQUI / "st_batch.csv", index_col=0).sort_values("idx_cronologico")
orden = list(batch.index)
dev_idx = batch.index[~batch.es_lockbox]; lb_idx = batch.index[batch.es_lockbox]
tercios = np.array_split(dev_idx, 3)
BLOQUES = {"dev": dev_idx, "lockbox": lb_idx, "total": batch.index, "dev_T1": tercios[0], "dev_T2": tercios[1], "dev_T3": tercios[2]}
KPI = "rendimiento_proxy_batch"

# ----------------------------------------------------------------------------- 1) targets por escalon
R = df["fase_proceso"] == "Reducción"
ok = R & df.sn_inv_prev.gt(20) & df.sn_inventario_escoria_est_kg.gt(20) & df.feo_inv_prev.gt(100) & df.feo_inventario_escoria_est_kg.gt(100) & ~df.es_primer_escalon_batch
df["ln_sn_dep"] = np.log(df.sn_inv_prev / df.sn_inventario_escoria_est_kg).where(ok)
df["ln_feo_ret"] = np.log(df.feo_inventario_escoria_est_kg / df.feo_inv_prev).where(ok)
F = df["fase_proceso"] == "Fusión"
df["irf_next"] = df["indice_irf"].where(F)

agg = df[R].groupby("Batch")[["ln_sn_dep", "ln_feo_ret"]].sum(min_count=1).reindex(orden)
irf_fin = df[F].dropna(subset=["irf_next"]).groupby("Batch")["irf_next"].last().reindex(orden)
irf_F0 = df[F & (df.orden_escalon_fase == 0)].set_index("Batch")["indice_irf"].reindex(orden)

def rho(s: pd.Series, idx, kpi=KPI):
    d = pd.concat([s.reindex(idx), batch.loc[idx, kpi]], axis=1).dropna()
    if len(d) < 8:
        return np.nan, np.nan, len(d)
    r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1]); return r, p, len(d)

# ----------------------------------------------------------------------------- 2) seleccion de w (solo DEV)
ws = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50, 100]
filas = []
for w in ws:
    s = agg.ln_sn_dep + w * agg.ln_feo_ret
    fila = {"w": w}
    for k, idx in BLOQUES.items():
        r, p, n = rho(s, idx); fila[f"rho_{k}"] = r; fila[f"p_{k}"] = p
    filas.append(fila)
wsel = pd.DataFrame(filas)
rmax = wsel.rho_dev.max()
w_star = float(wsel.loc[wsel.rho_dev >= 0.95 * rmax, "w"].min())
wsel["elegido"] = wsel.w == w_star
wsel.to_csv(AQUI / "st_06_w_seleccion.csv", index=False)
log(f"=== seleccion de w (solo DEV): max rho_dev = {rmax:.3f}; regla 'menor w con rho_dev >= 0.95*max' -> w* = {w_star} ===")
log(wsel[["w", "rho_dev", "p_dev", "rho_dev_T1", "rho_dev_T2", "rho_dev_T3", "rho_lockbox", "p_lockbox", "rho_total"]].round(4).to_string(index=False))
df["sdi"] = df["ln_sn_dep"] + w_star * df["ln_feo_ret"]
sdi = agg.ln_sn_dep + w_star * agg.ln_feo_ret

GANADORES = {
    "R_sdi": (f"Reducción: SDI_w (w={w_star:g}) = ln(Sn_ini/Sn_fin) + w·ln(FeO_fin/FeO_ini)", sdi),
    "R_ln_feo_ret": ("Reducción: componente FeO, ln(FeO_fin/FeO_ini)", agg.ln_feo_ret),
    "R_ln_sn_dep": ("Reducción: componente Sn, ln(Sn_ini/Sn_fin)", agg.ln_sn_dep),
    "F_irf_fin": ("Fusión: IRF al fin de Fusión (último escalón)", irf_fin),
    "F_irf_F0": ("Fusión: IRF al cierre de F0 (estado inicial, referencia)", irf_F0),
    "ref_v4_R": ("referencia v4: Sn_ext − 0.6·FeO_ext (kg, Reducción)", batch["st_R04_sn_net_l06"]),
    "ref_v4_F": ("referencia v4: Sn extraído en Fusión (kg, signo −)", batch["st_F01_sn_ext"]),
}

# ----------------------------------------------------------------------------- 3) metricas por bloque + bootstrap + OLS
rng = np.random.default_rng(SEED)
def boot_rho(s, idx, n=2000):
    d = pd.concat([s.reindex(idx), batch.loc[idx, KPI]], axis=1).dropna().to_numpy()
    if len(d) < 8:
        return np.nan, np.nan
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(d), len(d)); vals.append(spearmanr(d[i, 0], d[i, 1])[0])
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)

def ols_ctrl(s, idx, extra=None):
    d = pd.concat([s.rename("x").reindex(idx), batch.loc[idx, [KPI] + st.CONTROLES_BATCH + ["idx_cronologico"]]], axis=1)
    if extra is not None:
        d = pd.concat([d, extra.rename("extra").reindex(idx)], axis=1)
    d = d.dropna()
    regs = [c for c in d.columns if c != KPI and d[c].std() > 1e-9]
    X = (d[regs] - d[regs].mean()) / d[regs].std()
    m = sm.OLS(d[KPI], sm.add_constant(X)).fit(cov_type="HC3")
    return m.params["x"], m.conf_int().loc["x", 0], m.conf_int().loc["x", 1], m.pvalues["x"], m.rsquared_adj, len(d)

# BH: p de los 36 candidatos (st_01) + ganadores, por fase, en DEV
corr01 = pd.read_csv(AQUI / "st_01_correlaciones.csv")
rank01 = corr01[(corr01.kpi == KPI) & (corr01.muestra == "dev")][["id", "p_spearman"]].rename(columns={"p_spearman": "p_dev"})
filas = []
for k, (desc, s) in GANADORES.items():
    fila = {"target": k, "descripcion": desc}
    for b, idx in BLOQUES.items():
        r, p, n = rho(s, idx); fila[f"rho_{b}"] = r; fila[f"p_{b}"] = p; fila[f"n_{b}"] = n
        if b in ("dev", "lockbox", "total"):
            lo, hi = boot_rho(s, idx); fila[f"ci_lo_{b}"] = lo; fila[f"ci_hi_{b}"] = hi
            c, lo2, hi2, p2, r2, n2 = ols_ctrl(s, idx); fila[f"ols_coef_sd_{b}"] = c; fila[f"ols_ci_lo_{b}"] = lo2; fila[f"ols_ci_hi_{b}"] = hi2; fila[f"ols_p_{b}"] = p2; fila[f"ols_r2adj_{b}"] = r2
    for kpi2 in ["f_dross", "f_polvo", "f_metal", "recuperacion_real_pct"]:
        r, p, n = rho(s, dev_idx, kpi2); fila[f"rho_dev_{kpi2}"] = r; fila[f"p_dev_{kpi2}"] = p
    # dosis-respuesta por quintiles (DEV)
    d = pd.concat([s.rename("x").reindex(dev_idx), batch.loc[dev_idx, KPI]], axis=1).dropna()
    q = pd.qcut(d.x, 5, labels=False, duplicates="drop"); g = d.groupby(q)[KPI].mean()
    fila["quintiles_dev"] = " / ".join(f"{v:.2f}" for v in g.values); fila["monotonia_dev"] = int((np.diff(g.values) > 0).sum())
    fila["delta_Q5_Q1_pp"] = float(g.iloc[-1] - g.iloc[0])
    filas.append(fila)
met = pd.DataFrame(filas)
# BH conjunto por fase: p_dev de los 36 candidatos + el ganador de esa fase (37 tests)
for fase, gan in [("Reducción", "R_sdi"), ("Fusión", "F_irf_fin")]:
    ps = list(rank01.loc[rank01.id.str.startswith(fase[0]), "p_dev"]) + [float(met.loc[met.target == gan, "p_dev"].iloc[0])]
    rej, p_bh, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")
    met.loc[met.target == gan, "p_bh_dev_37tests"] = p_bh[-1]
met.to_csv(AQUI / "st_06_metricas_ganadores.csv", index=False)
log("\n=== metricas de los ganadores (Spearman vs rendimiento_proxy_batch; OLS HC3 por sd con controles + tendencia) ===")
cols = ["target", "rho_dev", "p_dev", "ci_lo_dev", "ci_hi_dev", "p_bh_dev_37tests", "rho_dev_T1", "rho_dev_T2", "rho_dev_T3", "rho_lockbox", "p_lockbox", "rho_total", "p_total",
        "ols_coef_sd_dev", "ols_p_dev", "ols_coef_sd_lockbox", "ols_p_lockbox", "ols_coef_sd_total", "ols_p_total", "rho_dev_f_dross", "rho_dev_f_polvo", "monotonia_dev", "delta_Q5_Q1_pp"]
log(met[cols].round(4).to_string(index=False))

# ----------------------------------------------------------------------------- 4) efecto sobre el batch SIGUIENTE (heel de escoria)
nxt = {orden[i]: orden[i + 1] for i in range(len(orden) - 1)}
kpi_next = pd.Series({k: batch[KPI].get(v, np.nan) for k, v in nxt.items()}).reindex(orden)
irf_F0_next = pd.Series({k: irf_F0.get(v, np.nan) for k, v in nxt.items()}).reindex(orden)
filas = []
log("\n=== efecto del target de Reduccion sobre el batch SIGUIENTE (heel quimico de escoria) ===")
for k in ["R_sdi", "R_ln_feo_ret", "R_ln_sn_dep", "F_irf_fin"]:
    s = GANADORES[k][1]
    fila = {"target": k}
    for b, idx in [("dev", dev_idx), ("lockbox", lb_idx), ("total", batch.index)]:
        d = pd.concat([s.reindex(idx), kpi_next.reindex(idx)], axis=1).dropna()
        r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1]); fila[f"rho_kpi_next_{b}"] = r; fila[f"p_kpi_next_{b}"] = p; fila[f"n_{b}"] = len(d)
        d2 = pd.concat([s.reindex(idx), irf_F0_next.reindex(idx)], axis=1).dropna()
        fila[f"rho_irfF0_next_{b}"] = spearmanr(d2.iloc[:, 0], d2.iloc[:, 1])[0]
        # OLS: kpi_next ~ z(target_b) + controles del batch siguiente + kpi propio (para descartar autocorrelacion)
        ctrl_next = pd.DataFrame({c: pd.Series({kk: batch[c].get(v, np.nan) for kk, v in nxt.items()}) for c in st.CONTROLES_BATCH}).reindex(idx)
        dd = pd.concat([s.rename("x").reindex(idx), kpi_next.rename("y").reindex(idx), batch.loc[idx, KPI].rename("kpi_propio"), ctrl_next], axis=1).dropna()
        regs = [c for c in dd.columns if c != "y" and dd[c].std() > 1e-9]
        X = (dd[regs] - dd[regs].mean()) / dd[regs].std(); m = sm.OLS(dd.y, sm.add_constant(X)).fit(cov_type="HC3")
        fila[f"ols_next_coef_sd_{b}"] = m.params["x"]; fila[f"ols_next_p_{b}"] = m.pvalues["x"]
    filas.append(fila)
    log(f"{k:14s} " + "  ".join(f"{b}: rho={fila[f'rho_kpi_next_{b}']:+.3f} (p={fila[f'p_kpi_next_{b}']:.3g}) ols={fila[f'ols_next_coef_sd_{b}']:+.2f} (p={fila[f'ols_next_p_{b}']:.3f})" for b in ["dev", "lockbox", "total"]))
pd.DataFrame(filas).to_csv(AQUI / "st_06_efecto_siguiente_batch.csv", index=False)
# efecto conjunto propio + siguiente: KPI de los dos batches (media) vs target
s = GANADORES["R_sdi"][1]; kpi2 = (batch[KPI] + kpi_next) / 2
for b, idx in [("dev", dev_idx), ("lockbox", lb_idx), ("total", batch.index)]:
    d = pd.concat([s.reindex(idx), kpi2.reindex(idx)], axis=1).dropna(); r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    log(f"SDI vs media(KPI propio, KPI siguiente) {b}: rho={r:+.3f} p={p:.3g} n={len(d)}")

# ----------------------------------------------------------------------------- 5) tabla por escalon (para modelado) + figuras
df[["Batch", "fase_proceso", "orden_escalon_fase", "fecha_inicio", "ln_sn_dep", "ln_feo_ret", "sdi", "irf_next"]].to_csv(AQUI / "st_06_targets_escalon.csv", index=False)
log(f"\nescalones con target: sdi {df.sdi.notna().sum()}, irf_next {df.irf_next.notna().sum()}; sd(sdi) escalon = {df.sdi.std():.3f}, sd(ln_feo_ret) = {df.ln_feo_ret.std():.3f}, sd(irf_next) = {df.irf_next.std():.4f}")
log("correlacion Spearman escalon a escalon (Reduccion) entre ln_sn_dep y ln_feo_ret: %.3f" % spearmanr(df.loc[ok, 'ln_sn_dep'], df.loc[ok, 'ln_feo_ret'])[0])
log("por escalon de Reduccion: media ln_sn_dep / ln_feo_ret / sdi:")
log(df[ok].groupby("orden_escalon_fase")[["ln_sn_dep", "ln_feo_ret", "sdi"]].mean().round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axes[0]
for k in ["rho_dev", "rho_lockbox", "rho_dev_T1", "rho_dev_T2", "rho_dev_T3"]:
    ax.plot(wsel.w, wsel[k], "o-", label=k, lw=2.2 if k == "rho_dev" else 1)
ax.axvline(w_star, color="r", ls="--", label=f"w* = {w_star:g}"); ax.axhline(0, color="k", lw=0.7); ax.set_xscale("symlog")
ax.set_xlabel("w"); ax.set_ylabel("Spearman vs rendimiento"); ax.set_title("SDI_w: rho por bloque y w* (regla sólo DEV)"); ax.legend(fontsize=7)
for ax, (k, titulo) in zip(axes[1:], [("R_sdi", f"Reducción: SDI (w={w_star:g})"), ("F_irf_fin", "Fusión: IRF fin de Fusión")]):
    s = GANADORES[k][1]
    for idx, col, lab in [(dev_idx, "#1f77b4", "DEV"), (lb_idx, "#d62728", "lockbox")]:
        d = pd.concat([s.reindex(idx), batch.loc[idx, KPI]], axis=1).dropna(); ax.scatter(d.iloc[:, 0], d.iloc[:, 1], s=14, alpha=0.6, color=col, label=f"{lab} ρ={spearmanr(d.iloc[:,0], d.iloc[:,1])[0]:+.2f}")
    d = pd.concat([s.reindex(dev_idx), batch.loc[dev_idx, KPI]], axis=1).dropna(); q = pd.qcut(d.iloc[:, 0], 5, labels=False, duplicates="drop")
    g = d.groupby(q).agg(x=(d.columns[0], "mean"), y=(KPI, "mean")); ax.plot(g.x, g.y, "ks-", lw=2, label="media por quintil (DEV)")
    ax.set_xlabel(k); ax.set_ylabel("rendimiento proxy batch [%]"); ax.set_title(titulo); ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(AQUI / "figs" / "st_06_ganadores.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
xs = np.arange(len(met));
for j, b in enumerate(["dev", "lockbox", "total"]):
    ax.bar(xs + (j - 1) * 0.27, met[f"rho_{b}"], width=0.27, label=b, yerr=[met[f"rho_{b}"] - met[f"ci_lo_{b}"], met[f"ci_hi_{b}"] - met[f"rho_{b}"]], capsize=2)
ax.set_xticks(xs); ax.set_xticklabels(met.target, rotation=20, ha="right", fontsize=8); ax.axhline(0, color="k", lw=0.7); ax.set_ylabel("Spearman vs rendimiento (IC95% bootstrap)")
ax.set_title("Ganadores vs referencias v4"); ax.legend(); plt.tight_layout(); fig.savefig(AQUI / "figs" / "st_06_rho_ganadores.png", dpi=120); plt.close(fig)
log(f"tiempo {time.time()-t0:.1f}s")
(AQUI / "st_06_log.txt").write_text("\n".join(LOG), encoding="utf-8")
