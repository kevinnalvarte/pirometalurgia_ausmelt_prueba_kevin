"""E11-01 tarea 6: valor de robustez a confusion no observada (Cinelli & Hazlett 2020) para x_R_gn y x_R_carbon,
sobre la regresion conjunta DEV (spec6, KPI ~ exposiciones + controles). Benchmark = R2 parcial del control mas fuerte."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "experimentos/v9")
import v9_lib as L  # noqa: E402
import e11_01_common as C  # noqa: E402

OUT = Path("experimentos/v9")
exp6 = [f"x_{g}_{p}" for g in ("F", "R") for p in C.PALANCAS3]
FOCO = ["x_R_gn", "x_R_carbon"]


def main():
    res = L.residuos_escalon()
    t = L.tabla_batch(res=res).join(C.exposiciones_R_combinada(res))
    dev = t[t.es_dev].dropna(subset=exp6 + L.CONTROLES + [L.KPI])
    fit = L.ols_hc3(dev[L.KPI], dev[exp6 + L.CONTROLES])
    dof = int(fit.df_resid)

    # RV para las exposiciones foco
    filas = []
    for e in FOCO:
        rv = C.rv_sensemakr(float(fit.tvalues[e]), dof)
        rv["variable"] = e; rv["rol"] = "exposicion_foco"
        filas.append(rv)
    # benchmark: R2 parcial de cada control y de las otras exposiciones (candidatos a "confusor mas fuerte observado")
    for e in [c for c in exp6 if c not in FOCO] + L.CONTROLES:
        rv = C.rv_sensemakr(float(fit.tvalues[e]), dof)
        rv["variable"] = e; rv["rol"] = "benchmark_candidato"
        filas.append(rv)
    tabla = pd.DataFrame(filas)
    bench_max = tabla[tabla.rol == "benchmark_candidato"].sort_values("r2_partial", ascending=False)
    tabla["benchmark_r2_max"] = float(bench_max.iloc[0]["r2_partial"])
    tabla["benchmark_var_max"] = bench_max.iloc[0]["variable"]
    for e in FOCO:
        m = tabla.variable == e
        tabla.loc[m, "RV_q1_supera_benchmark"] = tabla.loc[m, "RV_q1"].to_numpy()[0] > tabla["benchmark_r2_max"].to_numpy()[0]
    tabla.to_csv(OUT / "e11_01_sensemakr.csv", index=False)
    print("dof =", dof)
    print(tabla[["variable", "rol", "t", "r2_partial", "RV_q1", "RV_q1_alpha05"]].sort_values("r2_partial", ascending=False).to_string(index=False))
    print("benchmark (control/exposicion no-foco mas fuerte):", bench_max.iloc[0]["variable"], round(float(bench_max.iloc[0]["r2_partial"]), 4))


if __name__ == "__main__":
    main()
