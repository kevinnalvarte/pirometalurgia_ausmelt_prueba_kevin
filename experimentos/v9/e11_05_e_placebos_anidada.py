"""E11-05 (e) -- Placebos (exposicion del batch SIGUIENTE -> KPI actual; exposicion -> pre-tratamiento) y
validacion anidada (L.prueba_anidada) con (a) solo exo2, (b) exo2+carbon, (c) + Reduccion."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v9")
import v9_lib as L
import numpy as np, pandas as pd

OUT = L.SALIDA
res = L.residuos_escalon()
t = L.tabla_batch(res=res).sort_values("idx_cronologico").copy()
t["x_F_exo2_next"] = t["x_F_exo2"].shift(-1)
t["x_F_carbon_next"] = t["x_F_carbon"].shift(-1)
t["kpi_prev"] = t[L.KPI].shift(1)

print("=== Placebo (a): exposicion de Fusion del batch SIGUIENTE -> KPI actual ===")
filas = []
for x in ["x_F_exo2_next", "x_F_carbon_next"]:
    for nombre, d in [("DEV", t[t.es_dev]), ("total", t)]:
        dd = d.dropna(subset=[x, L.KPI] + L.CONTROLES)
        f = L.ols_hc3(dd[L.KPI], dd[[x] + L.CONTROLES])
        filas.append(dict(x=x, regimen=nombre, n=len(dd), beta=f.params[x], p=f.pvalues[x]))
pb1 = pd.DataFrame(filas)
print(pb1.round(4).to_string())

print()
print("=== Placebo (b): exposicion de Fusion -> variables pre-tratamiento ===")
filas2 = []
for x in ["x_F_exo2", "x_F_carbon"]:
    for y in ["ley_sn_conc_batch_pct", "feed_sn_total_t", "F_lnirf_F0", "kpi_prev"]:
        for nombre, d in [("DEV", t[t.es_dev]), ("total", t)]:
            ctr = [c for c in L.CONTROLES if c != y]
            dd = d.dropna(subset=[x, y] + ctr)
            f = L.ols_hc3(dd[y], dd[[x] + ctr])
            filas2.append(dict(x=x, y=y, regimen=nombre, n=len(dd), beta=f.params[x], p=f.pvalues[x]))
pb2 = pd.DataFrame(filas2)
pb2.to_csv(OUT / "e11_05_placebos.csv", index=False)
pd.concat([pb1.assign(tipo="siguiente_a_kpi"), pb2.assign(tipo="expo_a_pretrat")], ignore_index=True).to_csv(
    OUT / "e11_05_placebos.csv", index=False)
print(pb2.round(4).to_string())

# ---------- validacion anidada ----------
print()
print("=== Validacion anidada (L.prueba_anidada), DEV ===")
dev = t[t.es_dev]
specs = {
    "a_solo_exo2": ["x_F_exo2"],
    "b_exo2_carbon": ["x_F_exo2", "x_F_carbon"],
    "c_mas_reduccion": ["x_F_exo2", "x_F_carbon", "x_Rt_carbon", "x_Rt_gn", "x_Rl_carbon", "x_Rl_gn"],
}
filas3 = []
for nombre, X in specs.items():
    for esq in ("gkf", "crono"):
        r = L.prueba_anidada(dev, X, esquema=esq)
        r.pop("score")
        beta = r.pop("beta_medio"); r.pop("beta_sd_folds", None)
        print(nombre, esq, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, "beta_medio", beta)
        row = dict(spec=nombre, **{k: v for k, v in r.items() if not isinstance(v, dict)})
        filas3.append(row)
nest = pd.DataFrame(filas3)
nest.to_csv(OUT / "e11_05_anidada.csv", index=False)
