"""E12-01 tarea 4: placebos. (a) zD del batch siguiente (lead 1, lead 3) -> KPI actual (debe ser nulo). (b) zD ~
pre-tratamiento (ley_sn_conc_batch_pct, feed_sn_total_t, F_lnirf_F0, KPI del batch anterior): confusion inversa /
exogeneidad. (c) correlacion cruda zD_R3 vs %Sn al cierre de R2 (= ley_sn_escoria_pct_prev de R3): debe ser ~0 por
construccion del residuo (ya esta en el estado condicionado)."""
import sys
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import e11_07_lib as E
import v9_lib as L

SALIDA = L.RAIZ / "experimentos" / "v10"
CTRL = E.C_SIN_ESPESOR

exp = pd.read_csv(SALIDA / "e12_01_exposiciones.csv", index_col=0)
t = E.cargar_tabla().join(exp[["zD_R3", "zD_F6", "R3_ley_sn_escoria_pct_prev"]], how="left").sort_values("idx_cronologico").reset_index()
t = t.rename(columns={"index": "Batch"}) if "Batch" not in t.columns else t
dev_mask = t["es_dev"]

# --- (a) leads: zD_R3(t+k), zD_F6(t+k) -> KPI(t)
for k in (1, 3):
    t[f"zD_R3_lead{k}"] = t["zD_R3"].shift(-k)
    t[f"zD_F6_lead{k}"] = t["zD_F6"].shift(-k)

filas_lead = []
for k in (1, 3):
    Xk = [f"zD_R3_lead{k}", f"zD_F6_lead{k}"]
    for nombre, sub in (("DEV", t[dev_mask]), ("TOTAL", t)):
        d = sub.dropna(subset=[E.KPI] + Xk + CTRL)
        fit = L.ols_hc3(d[E.KPI], d[Xk + CTRL])
        for v in Xk:
            filas_lead.append({"lead": k, "muestra": nombre, "n": len(d), "var": v, "coef": fit.params[v], "t": fit.tvalues[v], "p": fit.pvalues[v]})
lead_tab = pd.DataFrame(filas_lead)
print("--- placebo (a): KPI_t ~ zD_{t+lead} + controles ---")
print(lead_tab.round(4).to_string())
lead_tab.to_csv(SALIDA / "e12_01_placebo_leads.csv", index=False)

# --- (b) zD ~ pre-tratamiento
t["KPI_prev"] = t[E.KPI].shift(1)
pretrat = ["ley_sn_conc_batch_pct", "feed_sn_total_t", "F_lnirf_F0", "KPI_prev"]
filas_pre = []
for zvar in ("zD_R3", "zD_F6"):
    for nombre, sub in (("DEV", t[dev_mask]), ("TOTAL", t)):
        d = sub.dropna(subset=[zvar] + pretrat)
        fit = L.ols_hc3(d[zvar], d[pretrat])
        for v in pretrat:
            filas_pre.append({"zvar": zvar, "muestra": nombre, "n": len(d), "var": v, "coef": fit.params[v], "t": fit.tvalues[v], "p": fit.pvalues[v]})
        filas_pre.append({"zvar": zvar, "muestra": nombre, "n": len(d), "var": "F_conjunto", "coef": np.nan,
                          "t": np.nan, "p": fit.f_pvalue})
pre_tab = pd.DataFrame(filas_pre)
print("\n--- placebo (b): zD ~ pre-tratamiento (ley_sn_conc, feed_sn_total, F_lnirf_F0, KPI_prev) ---")
print(pre_tab.round(4).to_string())
pre_tab.to_csv(SALIDA / "e12_01_placebo_pretratamiento.csv", index=False)

# --- (c) confusion inversa: zD_R3 vs %Sn al cierre de R2 (= ley_sn_escoria_pct_prev al inicio de R3)
d = t.dropna(subset=["zD_R3", "R3_ley_sn_escoria_pct_prev"])
r_p, p_p = stats.pearsonr(d["zD_R3"], d["R3_ley_sn_escoria_pct_prev"])
r_s, p_s = stats.spearmanr(d["zD_R3"], d["R3_ley_sn_escoria_pct_prev"])
print(f"\n--- confusion inversa: corr(zD_R3, %Sn al cierre de R2) --- pearson={r_p:.4f} (p={p_p:.4f})  spearman={r_s:.4f} (p={p_s:.4f})  n={len(d)}")
pd.DataFrame([{"pearson": r_p, "p_pearson": p_p, "spearman": r_s, "p_spearman": p_s, "n": len(d)}]).to_csv(
    SALIDA / "e12_01_placebo_confusion_inversa.csv", index=False)
