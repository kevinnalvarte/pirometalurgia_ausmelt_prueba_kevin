"""E18-04 (Fable): coeficientes FISICOS de conversion a nivel batch (relaciones de masa, no de conducta):
  Sn en dross [kg]  ~ lambda * FeO reducido en Reduccion [kg]      (hardhead: kg de Sn atrapado por kg de FeO metalizado)
  Sn a metal  [kg]  ~ mu * Sn agotado de la escoria en Reduccion [kg]
y balance neto de Sn a metal de +1 Nm3/min de GN por ORDEN con incertidumbre conjunta (theta intra-batch de E18-02 x lambda, mu)."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import asesor_receta_v10 as A, v9_lib as L
r, bt = A.escalones_con_receta(); r = r.sort_values(["Batch", "orden_escalon_fase"])
g = r.groupby("Batch")
B = pd.DataFrame({"feo_ini": g.m6_feo_inv_kg_prev.first(), "feo_fin": g.m6_feo_inv_kg.last(), "sn_ini": g.m6_sn_inv_kg_prev.first(), "sn_fin": g.m6_sn_inv_kg.last(), "n": g.size()})
B["feo_red_kg"] = B.feo_ini - B.feo_fin; B["sn_dep_kg"] = B.sn_ini - B.sn_fin
t = bt.join(B); t["sn_dross_kg"] = 1000 * t.sn_dross_t; t["sn_metal_kg"] = 1000 * t.sn_metal_t; t["sn_polvo_kg"] = 1000 * t.sn_polvo_t; t["feed_kg"] = 1000 * t.feed_sn_total_t
t = t[(t.n == 4)].replace([np.inf, -np.inf], np.nan).dropna(subset=["feo_red_kg", "sn_dep_kg", "sn_dross_kg", "sn_metal_kg", "sn_polvo_kg", "feed_kg", "ley_sn_conc_batch_pct", "frac_carga_secundaria_F", "feed_dross_Fe_total_t"]); print("n", len(t)); print(t[["feo_red_kg", "sn_dep_kg", "sn_dross_kg", "sn_metal_kg", "sn_polvo_kg"]].describe().round(0).T[["mean", "std"]].to_string())
C = ["feed_kg", "ley_sn_conc_batch_pct", "frac_carga_secundaria_F", "feed_dross_Fe_total_t", "idx_cronologico"]
for y in ("sn_dross_kg", "sn_metal_kg", "sn_polvo_kg"):
    for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
        f = L.ols_hc3(tt[y], tt[["feo_red_kg", "sn_dep_kg"] + C]); print(y, nm, len(tt), " ".join(f"{c}:{f.params[c]:+.3f}[{f.conf_int().loc[c,0]:+.3f},{f.conf_int().loc[c,1]:+.3f}]" for c in ["feo_red_kg", "sn_dep_kg"]))
f = L.ols_hc3(t.sn_dross_kg, t[["feo_red_kg", "sn_dep_kg"] + C]); lam, lam_se = f.params.feo_red_kg, f.bse.feo_red_kg
f2 = L.ols_hc3(t.sn_metal_kg, t[["feo_red_kg", "sn_dep_kg"] + C]); mu, mu_se, lam_m, lam_m_se = f2.params.sn_dep_kg, f2.bse.sn_dep_kg, f2.params.feo_red_kg, f2.bse.feo_red_kg
print("lambda (Sn a dross por kg FeO reducido) %.3f +- %.3f | mu (Sn a metal por kg Sn agotado) %.3f +- %.3f | efecto directo FeO red -> metal %.3f +- %.3f" % (lam, lam_se, mu, mu_se, lam_m, lam_m_se))
# balance por orden con theta intra-batch por orden

