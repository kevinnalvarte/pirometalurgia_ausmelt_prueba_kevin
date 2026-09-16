"""E9-00b (Fable) -- termino de legado: el estado de la escoria al final de la Reduccion (IRF, FeO retenido) se hereda por el batch
siguiente (heel de matriz, iteracion 5). ¿Debe el objetivo anclar w y K sobre KPI_actual + KPI_siguiente?

Pruebas:
 1. KPI_next ~ R_sum_ln_feo_ret + R_sum_ln_sn_dep + controles (E8-03 'next'): coef FeO ~10.6 pp/unidad. Aqui: ¿lo media R_lnirf_fin
    (heel)? Regresion con y sin R_lnirf_fin; y F_lnirf_F0 del batch siguiente como mediador explicito (KPI_next ~ F_lnirf_F0_next).
 2. Placebo temporal: KPI_prev ~ R_sum_ln_feo_ret (el batch anterior no puede depender de lo que hago hoy). Si sale significativo,
    la senal 'next' es confusion de campana.
 3. Anclaje sobre KPI_actual + KPI_next (suma): K_Sn, K_FeO, w, con IC bootstrap por batch; compararlo con el anclaje actual.
 4. Cuanto del efecto next pasa por el heel: producto de coeficientes (R_sum_feo -> F_lnirf_F0_next) x (F_lnirf_F0 -> KPI).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_lib as L  # noqa: E402
import statsmodels.api as sm  # noqa: E402

OUT = Path(__file__).resolve().parent
df = L.cargar_df()
batch = L.construir_batch(df).sort_values("fecha_batch")
K = L.KPI_PRINCIPAL
batch["KPI_prev"] = batch[K].shift(1)
batch["KPI_next"] = batch[K].shift(-1)
batch["F_lnirf_F0_next"] = batch["F_lnirf_F0"].shift(-1)
batch["R_lnirf_fin_prev"] = batch["R_lnirf_fin"].shift(1)
batch["KPI_sum2"] = batch[K] + batch["KPI_next"]
C = L.CONTROLES_BATCH
SN, FE = "R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret"


def ols(y, cols, sub, controles=C):
    dd = sub[[y] + cols + controles].dropna()
    m = sm.OLS(dd[y], sm.add_constant(dd[cols + controles])).fit(cov_type="HC3")
    return m, len(dd)


filas = []
for nombre, sub in [("dev", batch[batch["es_dev"]]), ("total", batch)]:
    specs = [
        ("next_base", "KPI_next", [SN, FE]),
        ("next_con_heel_fin", "KPI_next", [SN, FE, "R_lnirf_fin"]),
        ("next_mediador_F0next", "KPI_next", [SN, FE, "F_lnirf_F0_next"]),
        ("mediador_F0next_sobre_feo", "F_lnirf_F0_next", [SN, FE]),
        ("placebo_prev", "KPI_prev", [SN, FE]),
        ("placebo_prev_heel", "KPI_prev", [SN, FE, "R_lnirf_fin"]),
        ("actual_base", K, [SN, FE]),
        ("actual_con_heel_propio", K, [SN, FE, "F_lnirf_F0"]),
        ("suma2", "KPI_sum2", [SN, FE]),
        ("suma2_con_heel_propio", "KPI_sum2", [SN, FE, "F_lnirf_F0"]),
    ]
    for etiqueta, y, cols in specs:
        m, n = ols(y, cols, sub)
        for c in cols:
            filas.append(dict(muestra=nombre, spec=etiqueta, y=y, variable=c, coef=m.params[c], lo=m.conf_int().loc[c, 0],
                              hi=m.conf_int().loc[c, 1], p=m.pvalues[c], n=n, r2adj=m.rsquared_adj))
res = pd.DataFrame(filas)
pd.set_option("display.width", 220)
print(res.round(4).to_string())
res.to_csv(OUT / "e9_00b_legado.csv", index=False)

# 3) anclaje sobre suma de dos batches con bootstrap por batch
rng = np.random.default_rng(0)
for nombre, sub in [("dev", batch[batch["es_dev"]]), ("total", batch)]:
    for y in [K, "KPI_sum2"]:
        dd = sub[[y, SN, FE] + C].dropna().reset_index(drop=True)
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, len(dd), len(dd))
            b = dd.iloc[idx]
            p = sm.OLS(b[y], sm.add_constant(b[[SN, FE] + C])).fit().params
            boots.append((p[SN], p[FE], p[FE] / p[SN] if p[SN] > 0 else np.nan))
        boots = np.array(boots)
        m = sm.OLS(dd[y], sm.add_constant(dd[[SN, FE] + C])).fit(cov_type="HC3")
        print(f"\n[{nombre}] y={y}: K_Sn {m.params[SN]:.3f}, K_FeO {m.params[FE]:.3f}, w {m.params[FE]/m.params[SN]:.2f} "
              f"IC w [{np.nanpercentile(boots[:,2],2.5):.2f}, {np.nanpercentile(boots[:,2],97.5):.2f}] "
              f"frac ambos>0 {np.mean((boots[:,0]>0)&(boots[:,1]>0)):.3f} n={len(dd)}")
print("FIN")
