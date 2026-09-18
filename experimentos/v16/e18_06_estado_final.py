"""E18-06 (Fable): el Sn agotado en un escalon puede ser solo ADELANTO (lo que no se agota en R0 se agota despues). Lo que importa para el batch es el ESTADO
FINAL de la Reduccion: Sn que queda en la escoria final (perdida irreversible) y FeO total reducido (fuente del Fe metalico -> dross). Ambos medidos por ensayo
(mucho menos ruido que el KPI de pesaje). Regresion por batch sobre el desvio de GN respecto de la receta POR ORDEN (Nm3/min), con controles de estado inicial."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import asesor_receta_v10 as A, v9_lib as L
r, bt = A.escalones_con_receta(); r = r.sort_values(["Batch", "orden_escalon_fase"]); g = r.groupby("Batch")
B = pd.DataFrame({"feo_ini": g.m6_feo_inv_kg_prev.first(), "feo_fin": g.m6_feo_inv_kg.last(), "sn_ini": g.m6_sn_inv_kg_prev.first(), "sn_fin": g.m6_sn_inv_kg.last(), "ley_sn_fin": g.ley_sn_escoria_pct.last(),
                  "T_ini": g.temperatura_horno_celsius_prev.first(), "n": g.size()})
for o in range(4):
    B[f"gn{o}"] = r[r.orden_escalon_fase == o].set_index("Batch").dev__gn; B[f"c{o}"] = r[r.orden_escalon_fase == o].set_index("Batch").dev__carbon
B["feo_red"] = B.feo_ini - B.feo_fin; B["ln_sn_fin"] = np.log(B.sn_fin.clip(lower=50)); B["gn_tard"] = B[["gn2", "gn3"]].mean(axis=1); B["gn_temp"] = B[["gn0", "gn1"]].mean(axis=1)
B["c_tard"] = B[["c2", "c3"]].mean(axis=1); B["c_temp"] = B[["c0", "c1"]].mean(axis=1)
t = bt.join(B).replace([np.inf, -np.inf], np.nan); t = t[t.n == 4]; t["dross_pp"] = 100 * t.f_dross; t["escoria_pp"] = 100 * t.sn_perdido_escoria_frac
C = ["sn_ini", "feo_ini", "T_ini", "feed_sn_total_t", "ley_sn_conc_batch_pct", "frac_carga_secundaria_F", "feed_dross_Fe_total_t", "idx_cronologico"]
pd.set_option("display.width", 220); filas = []
for Xs in (["gn0", "gn1", "gn2", "gn3"], ["gn_temp", "gn_tard", "c_temp", "c_tard"]):
    for y, esc in (("sn_fin", 1), ("ley_sn_fin", 1), ("feo_red", 1), ("escoria_pp", 1), ("dross_pp", 1), (L.KPI, 1)):
        for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
            dd = tt.dropna(subset=Xs + C + [y]); f = L.ols_hc3(dd[y], dd[Xs + C])
            filas.append({"X": "+".join(Xs)[:12], "y": y, "muestra": nm, "n": len(dd), **{x: f"{f.params[x]:+.3g} (t{f.tvalues[x]:+.1f})" for x in Xs}})
out = pd.DataFrame(filas); out.to_csv("experimentos/v16/e18_06_estado_final.csv", index=False)
for X_, g_ in out.groupby("X", sort=False):
    print(g_.dropna(axis=1, how="all").drop(columns="X").to_string(index=False))
print(t[["sn_fin", "ley_sn_fin", "feo_red"]].describe().round(1).T[["mean", "std"]].to_string())
