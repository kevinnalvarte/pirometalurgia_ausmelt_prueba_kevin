"""E12-04 (1,2) -- estados de E[a|S]: seguridad, NaN, R2 OOF por palanca/orden, correlacion de residuos con S1,
exposiciones xG/xC por batch, OLS HC3 conjunta KPI+canales (DEV/total, controles sin espesor)."""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v10")
import numpy as np
import pandas as pd

import e12_04_lib as L

t0 = time.time()
pd.set_option("display.width", 220)

ver = L.verificar_estados()
ver.to_csv(L.SALIDA / "e12_04_01_seguridad_estados.csv", index=False)
print(ver.to_string())
assert ver["seguro"].all(), "hay estados inseguros"

base = L.cargar_base()
d = L.filas_reduccion(base)
print(f"[{time.time()-t0:5.0f}s] filas_reduccion: {len(d)} filas, {d['Batch'].nunique()} batches")

nan_rep = L.reporte_nan(base)
nan_rep.to_csv(L.SALIDA / "e12_04_01_nan_por_estado.csv", index=False)

RESID, R2 = {}, []
for nm, S in L.ESTADOS.items():
    r = L.residuos_estado(d, S, n_splits=5, seed=L.SEED)
    RESID[nm] = r
    for p in L.PALANCAS:
        info = r[p]
        fila = {"estado": nm, "palanca": p, "n_ok": info["n_ok"], "n_total": info["n_total"], "r2_global": info["r2_global"]}
        for o, v in info["r2_orden"].items():
            fila[f"r2_orden_{o}"] = v
        R2.append(fila)
    print(f"[{time.time()-t0:5.0f}s] estado {nm} listo")

r2_tab = pd.DataFrame(R2)
r2_tab.to_csv(L.SALIDA / "e12_04_01_r2_oof_por_estado.csv", index=False)
print(r2_tab.round(3).to_string())

# --------------------------------------------------------------------------- correlacion de residuos z con S1
filas_corr = []
for nm in L.ESTADOS:
    if nm == "S1_v9":
        continue
    for p in L.PALANCAS:
        z1 = RESID["S1_v9"][p]["z"]; z2 = RESID[nm][p]["z"]
        ok = np.isfinite(z1) & np.isfinite(z2)
        if ok.sum() > 10:
            rho = pd.Series(z1[ok]).corr(pd.Series(z2[ok]), method="spearman")
            r = float(np.corrcoef(z1[ok], z2[ok])[0, 1])
        else:
            rho = r = np.nan
        filas_corr.append({"estado": nm, "palanca": p, "n": int(ok.sum()), "corr_pearson_z_vs_S1": r, "corr_spearman_z_vs_S1": rho})
corr_tab = pd.DataFrame(filas_corr)
corr_tab.to_csv(L.SALIDA / "e12_04_01_correlacion_residuos_vs_S1.csv", index=False)
print(corr_tab.round(3).to_string())

# --------------------------------------------------------------------------- exposiciones por batch y OLS HC3 conjunta
import statsmodels.api as sm

bt = L.L8.construir_batch(L.L8.cargar_df())  # tabla batch cruda (KPI, canales, controles)
if "Batch" in bt.columns:
    bt = bt.set_index("Batch")

filas_ols = []
for nm in L.ESTADOS:
    expo = L.exposiciones_batch(d, RESID[nm])
    expo = expo.rename(columns={f"x_{p}": f"x_{p}_{nm}" for p in L.PALANCAS})
    bt = bt.join(expo, how="left")
    Xcols = [f"x_{p}_{nm}" for p in L.PALANCAS]
    for y in [L.KPI] + L.CANALES:
        for muestra, dd in (("DEV", bt[bt.es_dev]), ("total", bt)):
            dd2 = dd.dropna(subset=Xcols + L.CONTROLES_SIN_ESPESOR + [y])
            if len(dd2) < 30:
                continue
            fit = sm.OLS(dd2[y], sm.add_constant(dd2[Xcols + L.CONTROLES_SIN_ESPESOR])).fit(cov_type="HC3")
            for x in Xcols:
                filas_ols.append({"estado": nm, "y": y, "muestra": muestra, "n": len(dd2), "x": x,
                                  "beta": float(fit.params[x]), "se": float(fit.bse[x]), "t": float(fit.tvalues[x]),
                                  "p": float(fit.pvalues[x])})
ols_tab = pd.DataFrame(filas_ols)
ols_tab.to_csv(L.SALIDA / "e12_04_01_ols_conjunta.csv", index=False)
print(ols_tab[ols_tab.y == L.KPI].round(4).to_string())

bt.to_csv(L.SALIDA / "e12_04_01_tabla_batch_exposiciones.csv")
print(f"[{time.time()-t0:5.0f}s] listo")
