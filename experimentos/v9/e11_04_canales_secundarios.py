"""E11-04 (H4), complemento: KPI secundario por canal (f_dross para GN_R, Sn en escoria para carbon_R) -- ver diseno
item 5. Reusa la tabla de e11_04_kpi_ruido.py."""
import sys
sys.path.insert(0, "experimentos/v9")
import pandas as pd
import statsmodels.api as sm
import v9_lib as L
import v8_lib as L8
import feature_engineering as fe

df = L8.cargar_df()
res = L.residuos_escalon(df)
t = L.tabla_batch(df, res)
bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
t = t.join(bal[["sn_escoria_final_balance_kg", "feed_sn_total_kg"]], how="left")
subR = res[res["grupo"].isin(["Rt", "Rl"])]
t["x_R_gn"] = subR.groupby("Batch")["z__gn"].mean()
t["x_R_carbon"] = subR.groupby("Batch")["z__carbon"].mean()
denom = t["sn_metal_t"] * 1000 + t["sn_dross_t"] * 1000 + t["sn_polvo_t"] * 1000 + t["sn_escoria_final_balance_kg"]
t["canal_dross_frac"] = t["sn_dross_t"] * 1000 / denom
t["canal_escoria_frac"] = t["sn_escoria_final_balance_kg"] / denom
CONTROLES = L.CONTROLES

filas = []
for conjunto, mask in (("DEV", t.es_dev), ("total", pd.Series(True, index=t.index))):
    d = t.loc[mask]
    for canal, expo in (("canal_dross_frac", "x_R_gn"), ("canal_escoria_frac", "x_R_carbon")):
        dd = d[[canal, expo] + CONTROLES].dropna()
        fit = sm.OLS(dd[canal], sm.add_constant(dd[[expo] + CONTROLES], has_constant="add")).fit(cov_type="HC3")
        filas.append({"conjunto": conjunto, "canal": canal, "expo": expo, "n": len(dd),
                      "beta": float(fit.params[expo]), "t": float(fit.tvalues[expo]), "p": float(fit.pvalues[expo]),
                      "sd_canal": float(dd[canal].std())})
tab = pd.DataFrame(filas)
tab.to_csv(L.SALIDA / "e11_04_canales_secundarios.csv", index=False)
print(tab.round(4).to_string(index=False))
