"""E18-01 tarea 2 -- primera etapa: ejecutado ~ receta + controles + tendencia flexible (lineal / cubica / FE
bloques de 40). F del instrumento (debe ser > 10) para GN y para carbon, univariado y conjunto."""
import sys
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
import e18_01_lib as E

t = E.tabla_batch()
SAL = E.RAIZ / "experimentos" / "v16"
TENDENCIAS = ["lineal", "cubica", "bloque40"]
filas = []

for regimen, tt in (("TOTAL", t), ("DEV", t[t.es_dev]), ("LB", t[~t.es_dev])):
    for tend in TENDENCIAS:
        Xexog = E.controles_tendencia(tt, tend)
        cols_needed = ["ejec_gn", "ejec_carbon", "plan_gn", "plan_carbon"] + list(Xexog.columns)
        d = tt[cols_needed].join(Xexog.loc[:, ~Xexog.columns.duplicated()], lsuffix="_x") if False else None
        base = pd.concat([tt[["ejec_gn", "ejec_carbon", "plan_gn", "plan_carbon"]], Xexog], axis=1).dropna()
        if len(base) < 30:
            continue
        Xexog_np = np.column_stack([np.ones(len(base)), base[Xexog.columns].to_numpy(float)])
        # univariado: propio instrumento solo
        for p in ("gn", "carbon"):
            r_own = E.primera_etapa_F(base[f"ejec_{p}"].to_numpy(float), base[f"plan_{p}"].to_numpy(float), Xexog_np[:, 1:], None, cov="HC3")
            filas.append({"regimen": regimen, "tendencia": tend, "modelo": "univariado", "palanca": p, "n": r_own["n"],
                          "F": r_own["F"], "p": r_own["p"], "coef_propio": r_own["coef_own"][0], "r2": r_own["r2"]})
        # conjunto: los dos instrumentos presentes, F de exclusion del propio controlando por el del otro
        for p, otro in (("gn", "carbon"), ("carbon", "gn")):
            r_j = E.primera_etapa_F(base[f"ejec_{p}"].to_numpy(float), base[f"plan_{p}"].to_numpy(float), Xexog_np[:, 1:],
                                     base[f"plan_{otro}"].to_numpy(float)[:, None], cov="HC3")
            filas.append({"regimen": regimen, "tendencia": tend, "modelo": "conjunto", "palanca": p, "n": r_j["n"],
                          "F": r_j["F"], "p": r_j["p"], "coef_propio": r_j["coef_own"][0], "r2": r_j["r2"]})

res = pd.DataFrame(filas)
res.to_csv(SAL / "e18_01_02_primera_etapa.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\n>> Resumen F minimo por palanca/regimen (deben ser > 10):")
print(res.groupby(["regimen", "palanca", "modelo"])["F"].min().round(1).to_string())
