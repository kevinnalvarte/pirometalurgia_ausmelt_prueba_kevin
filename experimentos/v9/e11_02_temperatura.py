"""E11-02 tarea 4: interaccion de z_gn con temperatura ademas de con Sn, en TOTAL (362); solapamiento DEV/lockbox."""
import sys
from pathlib import Path
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd
import statsmodels.api as sm

import v9_lib as L
import v8_lib as L8
import modelo_predictivo_v8 as mp8

OUT = Path("experimentos/v9")
PRIMARIO = "ley_sn_escoria_pct_prev"
LN_PRIMARIO = True

df = L8.cargar_df()
res = L.residuos_escalon(df)
base = mp8._preparar_base(df)
t_batch = L.tabla_batch(df, res)
dev_b, lockbox_b = L8.mp.split_dev_lockbox(df)

MODS = [PRIMARIO, "temperatura_horno_celsius_prev"]
R = res[res.fase_proceso == "Reducción"].merge(
    base[["Batch", "fase_proceso", "orden_escalon_fase"] + MODS].drop_duplicates(["Batch", "fase_proceso", "orden_escalon_fase"]),
    on=["Batch", "fase_proceso", "orden_escalon_fase"], how="left")
R["es_dev"] = R.Batch.isin(dev_b)


def controles_ok(d, nm="DEV", controles=None):
    controles = controles or L.CONTROLES
    keep = list(controles)
    if nm != "DEV" and "espesor_ladrillo_norm_mm" in keep and "idx_cronologico" in d.columns:
        dd = d[["espesor_ladrillo_norm_mm", "idx_cronologico"]].dropna()
        if len(dd) > 5 and abs(dd.corr().iloc[0, 1]) > 0.9:
            keep = [c for c in keep if c != "espesor_ladrillo_norm_mm"]
    return keep


def ols_fit(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")


def zmod(mod, ln):
    x = np.log(R[mod]) if ln else R[mod].astype(float)
    mu, sd = float(x[R.es_dev].mean()), float(x[R.es_dev].std())
    return (x - mu) / sd, mu, sd


m_sn, mu_sn, sd_sn = zmod(PRIMARIO, LN_PRIMARIO)
m_t, mu_t, sd_t = zmod("temperatura_horno_celsius_prev", False)
print("corr(m_sn, m_temp) en DEV:", round(pd.Series(m_sn[R.es_dev]).corr(pd.Series(m_t[R.es_dev])), 3))
print("corr(m_sn, m_temp) en TOTAL:", round(pd.Series(m_sn).corr(pd.Series(m_t)), 3))

out = {}
for p in ("gn", "carbon"):
    z = R[f"z__{p}"]
    out[f"x_R_{p}"] = z.groupby(R.Batch).mean()
    out[f"x_R_{p}_x_sn"] = (z * m_sn).groupby(R.Batch).mean()
    out[f"x_R_{p}_x_temp"] = (z * m_t).groupby(R.Batch).mean()
t4 = t_batch.join(pd.DataFrame(out))

print("\n=== TAREA 4: TOTAL (362), z_gn x Sn y z_gn x temperatura ===")
filas = []
especs = {
    "solo_sn": ["x_R_gn", "x_R_gn_x_sn", "x_R_carbon", "x_R_carbon_x_sn"],
    "solo_temp": ["x_R_gn", "x_R_gn_x_temp", "x_R_carbon", "x_R_carbon_x_temp"],
    "ambos": ["x_R_gn", "x_R_gn_x_sn", "x_R_gn_x_temp", "x_R_carbon", "x_R_carbon_x_sn", "x_R_carbon_x_temp"],
}
for nm, mask in (("DEV", t4.es_dev), ("TOTAL", pd.Series(True, index=t4.index))):
    d = t4[mask]
    ctr = controles_ok(d, nm)
    for nombre, X in especs.items():
        dd = d.dropna(subset=X + ctr + [L.KPI])
        f = ols_fit(dd[L.KPI], dd[X + ctr])
        row = {"subset": nm, "espec": nombre, "n": len(dd)}
        for c in X:
            row[f"b_{c}"] = f.params[c]; row[f"p_{c}"] = f.pvalues[c]
        filas.append(row)
tabla4 = pd.DataFrame(filas)
tabla4.to_csv(OUT / "e11_02_temperatura_total.csv", index=False)
print(tabla4.round(3).to_string())

# --------------------------------------------------------------- solapamiento de temperatura DEV vs lockbox
print("\n--- solapamiento temperatura (escalon, Reduccion) ---")
t_dev = R.loc[R.es_dev, "temperatura_horno_celsius_prev"].dropna()
t_lb = R.loc[~R.es_dev, "temperatura_horno_celsius_prev"].dropna()
print("DEV: n", len(t_dev), "media", t_dev.mean().round(1), "mediana", t_dev.median().round(1), "rango", t_dev.min().round(1), t_dev.max().round(1))
print("LB : n", len(t_lb), "media", t_lb.mean().round(1), "mediana", t_lb.median().round(1), "rango", t_lb.min().round(1), t_lb.max().round(1))
frac_dev_bajo_max_lb = float((t_dev <= t_lb.max()).mean())
frac_lb_sobre_p50_dev = float((t_lb >= t_dev.median()).mean())
print(f"% de escalones DEV con T <= max(LB) [{t_lb.max():.1f}]: {frac_dev_bajo_max_lb*100:.1f}%")
print(f"% de escalones LB con T >= mediana(DEV) [{t_dev.median():.1f}]: {frac_lb_sobre_p50_dev*100:.1f}%")
# coeficiente de solapamiento (OVL) por histograma comun
bins = np.linspace(min(t_dev.min(), t_lb.min()), max(t_dev.max(), t_lb.max()), 40)
h_dev, _ = np.histogram(t_dev, bins=bins, density=True)
h_lb, _ = np.histogram(t_lb, bins=bins, density=True)
width = np.diff(bins)
ovl = float(np.sum(np.minimum(h_dev, h_lb) * width))
print("OVL (coef. de solapamiento de densidades):", round(ovl, 3))
pd.DataFrame({"subset": ["DEV", "LB"], "n": [len(t_dev), len(t_lb)], "media": [t_dev.mean(), t_lb.mean()],
              "mediana": [t_dev.median(), t_lb.median()], "min": [t_dev.min(), t_lb.min()], "max": [t_dev.max(), t_lb.max()],
              "OVL": [ovl, ovl]}).to_csv(OUT / "e11_02_solapamiento_temp.csv", index=False)

print("\nscript temperatura completo")
