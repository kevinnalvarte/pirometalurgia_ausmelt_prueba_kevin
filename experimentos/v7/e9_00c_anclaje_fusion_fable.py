"""E9-00c (Fable) -- anclaje del objetivo de Fusion como NIVEL terminal (E9-04): KPI ~ ln(Sn_inv al cierre de F6) con controles.

Relacion con el target de escalon: ln Sn_inv_F6 = ln Sn_disp_F6 - ln_sn_dep_F6 y, hacia atras, cada ln_sn_dep_t entra con peso
w_t = prod_{tau>t} Sn_inv_{tau-1}/(Sn_inv_{tau-1}+feed_tau) (dilucion por la carga posterior). Aqui: (1) beta por unidad de ln
Sn_inv_F6 (pp de KPI) con controles, heel y ln(Sn alimentado total) para separar escala de conservacion; (2) pesos w_t medios por
orden; (3) beta implicito por unidad de ln_sn_dep_t = -beta * w_t; (4) comparacion con el anclaje de suma (E7-02/E8-03).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_lib as L  # noqa: E402

OUT = Path(__file__).resolve().parent
df = L.cargar_df().sort_values(["Batch", "fecha_inicio"])
batch = L.construir_batch(df)
F = df[df["fase_proceso"] == "Fusión"]
fin = F.groupby("Batch").agg(sn_inv_F6=("m6_sn_inv_kg", "last"), masa_F6=("m6_masa_kg", "last"), ley_sn_F6=("ley_sn_escoria_pct", "last"),
                             feed_sn_F=("feed_Sn_kgf", "sum"), feo_inv_F6=("m6_feo_inv_kg", "last"))
batch = batch.join(fin)
batch["ln_sn_inv_F6"] = np.log(batch["sn_inv_F6"]); batch["ln_feed_sn_F"] = np.log(batch["feed_sn_F"])
batch["ln_frac_sn_conservado_F"] = batch["ln_sn_inv_F6"] - batch["ln_feed_sn_F"]   # fraccion del Sn alimentado que sigue en la escoria
batch["ln_ley_sn_F6"] = np.log(batch["ley_sn_F6"])
K = L.KPI_PRINCIPAL; C = L.CONTROLES_BATCH

def ols(y, cols, sub, ctrl=C):
    dd = sub[[y] + cols + ctrl].dropna()
    m = sm.OLS(dd[y], sm.add_constant(dd[cols + ctrl])).fit(cov_type="HC3")
    return m, len(dd)

filas = []
for nombre, sub in [("dev", batch[batch["es_dev"]]), ("total", batch)]:
    for etiqueta, cols in [
        ("nivel", ["ln_sn_inv_F6"]),
        ("nivel+ln_feed", ["ln_sn_inv_F6", "ln_feed_sn_F"]),
        ("frac_conservada", ["ln_frac_sn_conservado_F"]),
        ("frac_conservada+heel", ["ln_frac_sn_conservado_F", "F_lnirf_F0"]),
        ("ley_F6", ["ln_ley_sn_F6"]),
        ("ley_F6+heel", ["ln_ley_sn_F6", "F_lnirf_F0"]),
        ("suma_ln_sn_dep(ref)", ["F_sum_m6_ln_sn_dep"]),
        ("suma+heel(ref)", ["F_sum_m6_ln_sn_dep", "F_lnirf_F0"]),
        ("nivel+R", ["ln_frac_sn_conservado_F", "R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "F_lnirf_F0"]),
    ]:
        for y in [K, "f_dross", "f_polvo", "sn_perdido_escoria_frac"]:
            m, n = ols(y, cols, sub)
            for c in cols:
                filas.append(dict(muestra=nombre, spec=etiqueta, y=y, variable=c, coef=m.params[c], lo=m.conf_int().loc[c, 0], hi=m.conf_int().loc[c, 1],
                                  p=m.pvalues[c], n=n, r2adj=m.rsquared_adj, sd_x=float(sub[c].std())))
res = pd.DataFrame(filas)
pd.set_option("display.width", 230)
print(res[res["y"] == K].round(4).to_string())
print("\n-- canales (dev) --")
print(res[(res["muestra"] == "dev") & (res["y"] != K) & (res["variable"].isin(["ln_frac_sn_conservado_F", "F_sum_m6_ln_sn_dep"]))].round(4).to_string())
res.to_csv(OUT / "e9_00c_anclaje_fusion.csv", index=False)

# pesos w_t por orden: derivada de ln Sn_inv_F6 respecto de ln_sn_dep_t
rows = []
for b, g in F.groupby("Batch", sort=False):
    g = g.reset_index(drop=True)
    inv = g["m6_sn_inv_kg"].to_numpy(); feed = g["feed_Sn_kgf"].fillna(0).to_numpy()
    for t in range(1, len(g)):
        w = 1.0
        for tau in range(t + 1, len(g)):
            den = inv[tau - 1] + feed[tau]
            w *= inv[tau - 1] / den if den > 0 else np.nan
        rows.append(dict(Batch=b, orden=int(g.loc[t, "orden_escalon_fase"]), w=w))
pesos = pd.DataFrame(rows).groupby("orden")["w"].agg(["mean", "median", "std"]).round(3)
print("\n-- peso w_t de ln_sn_dep_t sobre ln Sn_inv_F6 por orden --\n", pesos)
pesos.to_csv(OUT / "e9_00c_pesos_orden.csv")
print("sd ln_sn_inv_F6", batch["ln_sn_inv_F6"].std().round(3), "sd ln_frac_cons", batch["ln_frac_sn_conservado_F"].std().round(3),
      "media frac conservada", np.exp(batch["ln_frac_sn_conservado_F"]).mean().round(3))
print("FIN")
