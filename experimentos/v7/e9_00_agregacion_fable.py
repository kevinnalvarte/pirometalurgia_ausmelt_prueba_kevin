"""E9-00 (Fable) -- estructura de agregacion del objetivo: escalon -> grupo -> fase -> batch, y anclaje por grupos en el KPI.

Preguntas:
 1. Identidad telescopica: sum_t m6_ln_sn_dep (R) = ln(Sn_ini_R / Sn_fin_R); sum_t m6_ln_feo_ret = ln(FeO_fin/FeO_ini). Error cuando faltan escalones.
 2. Anclaje del KPI refinado por GRUPO de escalones (R0, R1, R2-R3; F1-F3, F4-F6): OLS HC3 con controles. Si los coeficientes por grupo
    son homogeneos, un unico K y w por fase es defendible (agregacion aditiva); si no, el objetivo debe ponderar por grupo.
 3. Descomposicion de la varianza del J_batch historico por grupo y por componente (Sn vs FeO): que escalones "deciden" el batch.
 4. Consistencia J_batch (targets reales, K y w de v6) vs KPI: Spearman DEV/lockbox/total.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_lib as L  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

OUT = Path(__file__).resolve().parent
df = L.cargar_df()
dev, lockbox = L.split(df)
d = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
R = d[d["fase_proceso"] == "Reducción"]; F = d[d["fase_proceso"] == "Fusión"]

# 1) identidad telescopica -----------------------------------------------------------------
rows = []
for b, g in R.groupby("Batch", sort=False):
    ok = g["m6_ln_sn_dep"].notna() & g["m6_ln_feo_ret"].notna()
    sn_ini = g["m6_sn_inv_kg_prev"].iloc[0]; sn_fin = g["m6_sn_inv_kg"].iloc[-1]
    feo_ini = g["m6_feo_inv_kg_prev"].iloc[0]; feo_fin = g["m6_feo_inv_kg"].iloc[-1]
    feed = g["feed_Sn_kgf"].fillna(0).sum()
    rows.append(dict(Batch=b, n_validos=int(ok.sum()), sum_sn=g["m6_ln_sn_dep"].sum(min_count=1), sum_feo=g["m6_ln_feo_ret"].sum(min_count=1),
                     tele_sn=np.log((sn_ini + feed) / sn_fin) if sn_fin > 0 else np.nan, tele_feo=np.log(feo_fin / feo_ini) if feo_ini > 0 else np.nan,
                     feed_sn_R=feed))
tele = pd.DataFrame(rows)
tele["err_sn"] = tele["sum_sn"] - tele["tele_sn"]; tele["err_feo"] = tele["sum_feo"] - tele["tele_feo"]
res1 = tele.groupby("n_validos")[["err_sn", "err_feo"]].agg(["count", "mean", "std"]).round(4)
print("== identidad telescopica por n escalones validos ==\n", res1)
print("feed Sn en Reduccion (kg) mediana", tele["feed_sn_R"].median(), "p90", tele["feed_sn_R"].quantile(0.9))
tele.to_csv(OUT / "e9_00_telescopica.csv", index=False)

# 2) agregados por grupo -------------------------------------------------------------------
def grupo_R(o):
    return {0: "R0", 1: "R1", 2: "R23", 3: "R23"}[int(o)]

def grupo_F(o):
    return "F123" if o <= 3 else "F456"

R = R.assign(grupo=R["orden_escalon_fase"].map(grupo_R)); F = F.assign(grupo=F["orden_escalon_fase"].map(grupo_F))
aggR = R.pivot_table(index="Batch", columns="grupo", values=["m6_ln_sn_dep", "m6_ln_feo_ret"], aggfunc="sum")
aggR.columns = [f"{a}__{b}" for a, b in aggR.columns]
aggF = F.pivot_table(index="Batch", columns="grupo", values=["m6_ln_sn_dep"], aggfunc="sum")
aggF.columns = [f"F_{a}__{b}" for a, b in aggF.columns]
batch = L.construir_batch(df).join(aggR).join(aggF)
batch["J_R_v6"] = L.CONFIG["pp_kpi_por_unidad_sdi"] * (batch["R_sum_m6_ln_sn_dep"] + L.CONFIG["w_sdi"] * batch["R_sum_m6_ln_feo_ret"])
batch["J_F_v6"] = L.CONFIG["pp_kpi_por_unidad_sn_dep_F"] * batch["F_sum_m6_ln_sn_dep"]
batch["J_batch_v6"] = batch["J_R_v6"] + batch["J_F_v6"]

def ols(cols, sub, kpi=L.KPI_PRINCIPAL, controles=L.CONTROLES_BATCH, std=False):
    dd = sub[[kpi] + cols + controles].dropna()
    X = dd[cols + controles].copy()
    if std:
        X = (X - X.mean()) / X.std()
    m = sm.OLS(dd[kpi], sm.add_constant(X)).fit(cov_type="HC3")
    return m, len(dd)

filas = []
for nombre, sub in [("dev", batch[batch["es_dev"]]), ("total", batch)]:
    for etiqueta, cols in [
        ("fase", ["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret"]),
        ("grupos_R", ["m6_ln_sn_dep__R0", "m6_ln_sn_dep__R1", "m6_ln_sn_dep__R23", "m6_ln_feo_ret__R0", "m6_ln_feo_ret__R1", "m6_ln_feo_ret__R23"]),
        ("fase+F", ["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "F_sum_m6_ln_sn_dep"]),
        ("fase+F_grupos", ["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "F_m6_ln_sn_dep__F123", "F_m6_ln_sn_dep__F456"]),
        ("fase+F+heel", ["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "F_sum_m6_ln_sn_dep", "F_lnirf_F0"]),
    ]:
        m, n = ols(cols, sub)
        for c in cols:
            filas.append(dict(muestra=nombre, spec=etiqueta, variable=c, coef=m.params[c], lo=m.conf_int().loc[c, 0], hi=m.conf_int().loc[c, 1],
                              p=m.pvalues[c], n=n, r2adj=m.rsquared_adj))
        # test de homogeneidad por grupo (Wald): coef R0 = R1 = R23 para Sn y para FeO
        if etiqueta == "grupos_R":
            for comp in ["m6_ln_sn_dep", "m6_ln_feo_ret"]:
                names = [f"{comp}__R0", f"{comp}__R1", f"{comp}__R23"]
                Rm = np.zeros((2, len(m.params))); idx = [list(m.params.index).index(n_) for n_ in names]
                Rm[0, idx[0]] = 1; Rm[0, idx[1]] = -1; Rm[1, idx[1]] = 1; Rm[1, idx[2]] = -1
                w = m.wald_test(Rm, scalar=True)
                filas.append(dict(muestra=nombre, spec="wald_homogeneidad", variable=comp, coef=float(w.statistic), p=float(w.pvalue), n=n))
anc = pd.DataFrame(filas)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("\n== anclaje por fase / grupo (OLS HC3 con controles, unidades: pp KPI por unidad de ln) ==")
print(anc.round(4).to_string())
anc.to_csv(OUT / "e9_00_anclaje_grupos.csv", index=False)

# 3) varianza de J_batch por grupo y componente ------------------------------------------------
K, w = L.CONFIG["pp_kpi_por_unidad_sdi"], L.CONFIG["w_sdi"]
comp = pd.DataFrame({
    "Sn_R0": K * batch["m6_ln_sn_dep__R0"], "Sn_R1": K * batch["m6_ln_sn_dep__R1"], "Sn_R23": K * batch["m6_ln_sn_dep__R23"],
    "FeO_R0": K * w * batch["m6_ln_feo_ret__R0"], "FeO_R1": K * w * batch["m6_ln_feo_ret__R1"], "FeO_R23": K * w * batch["m6_ln_feo_ret__R23"],
}).dropna()
J = comp.sum(axis=1)
cov = comp.apply(lambda c: np.cov(c, J)[0, 1] / J.var())
print("\n== descomposicion de la varianza de J_R (fraccion cov(comp, J)/var(J)) ==\n", cov.round(3), "\nmedias pp:", comp.mean().round(3).to_dict())
cov.rename("frac_var").to_frame().assign(media_pp=comp.mean()).to_csv(OUT / "e9_00_varianza_J.csv")

# 4) consistencia J_batch vs KPI ---------------------------------------------------------------
filas = []
for nombre, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
    for col in ["J_R_v6", "J_F_v6", "J_batch_v6", "R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "F_sum_m6_ln_sn_dep"]:
        for kpi in [L.KPI_PRINCIPAL, "recuperacion_refinada_pct_next", "f_dross", "f_polvo"]:
            dd = sub[[col, kpi]].dropna()
            rho, p = stats.spearmanr(dd[col], dd[kpi])
            filas.append(dict(muestra=nombre, variable=col, kpi=kpi, rho=rho, p=p, n=len(dd)))
cons = pd.DataFrame(filas)
print("\n== Spearman J vs KPI ==\n", cons[cons["kpi"] == L.KPI_PRINCIPAL].round(3).to_string())
cons.to_csv(OUT / "e9_00_consistencia_J_kpi.csv", index=False)
print("FIN")
