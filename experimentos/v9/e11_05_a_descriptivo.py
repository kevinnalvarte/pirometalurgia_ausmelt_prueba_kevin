"""E11-05 (a) -- Descriptivo de exo2/O2/aire/GN en Fusion: serie temporal, autocorrelacion, ICC, y auditoria de confusion
temporal (poly3 / bloques de 30 batches / des-tendenciado por media movil +-10 batches)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v9")
import v9_lib as L
import numpy as np, pandas as pd

OUT = L.SALIDA
res = L.residuos_escalon()
t = L.tabla_batch(res=res)
dev = t[t.es_dev].copy()

F = res[res.grupo == "F"].copy()
PAL = ["exo2", "o2", "aire", "gn", "carbon"]

# ---------- 1) descriptivo por batch (nivel bruto a__p) ----------
filas = []
for p in PAL:
    g = F.groupby("Batch")[f"a__{p}"]
    bm = g.mean().rename(f"a_{p}")
    between = g.mean().var(ddof=1)
    within = g.var(ddof=1).mean()
    icc = between / (between + within)
    tb = t.join(bm)
    tb = tb.sort_values("idx_cronologico")
    serie = tb[f"a_{p}"].dropna()
    ac1 = serie.autocorr(lag=1)
    corr_idx = tb[[f"a_{p}", "idx_cronologico"]].corr().iloc[0, 1]
    corr_esp = tb[[f"a_{p}", "espesor_ladrillo_norm_mm"]].corr().iloc[0, 1]
    # residual z (lo que entra a la regresion) tambien
    zg = F.groupby("Batch")[f"z__{p}"]
    zb = zg.mean().rename(f"z_{p}")
    tz = t.join(zb).sort_values("idx_cronologico")
    ac1_z = tz[f"z_{p}"].dropna().autocorr(lag=1)
    corr_idx_z = tz[[f"z_{p}", "idx_cronologico"]].corr().iloc[0, 1]
    filas.append(dict(palanca=p, icc_between_within=icc, autocorr1_nivel=ac1, corr_idx_nivel=corr_idx,
                       corr_espesor_nivel=corr_esp, autocorr1_residual=ac1_z, corr_idx_residual=corr_idx_z,
                       media=serie.mean(), sd=serie.std()))
desc = pd.DataFrame(filas)
desc.to_csv(OUT / "e11_05_descriptivo_exposiciones.csv", index=False)
print(desc.round(3).to_string())

# ---------- 2) exo2: exposicion residual (x_F_exo2) vs tiempo, robustez con controles temporales flexibles ----------
tb = t.sort_values("idx_cronologico").copy()
tb["bloque30"] = (np.arange(len(tb)) // 30)
idx = tb["idx_cronologico"].to_numpy(float)
idx_c = (idx - idx.mean()) / idx.std()
tb["idx_p1"], tb["idx_p2"], tb["idx_p3"] = idx_c, idx_c ** 2, idx_c ** 3
# des-tendenciado: residuo de x_F_exo2 respecto de su media movil +-10 batches (ordenado cronologicamente)
tb["x_F_exo2_mm21"] = tb["x_F_exo2"].rolling(21, center=True, min_periods=8).mean()
tb["x_F_exo2_detrend"] = tb["x_F_exo2"] - tb["x_F_exo2_mm21"]

resumen = []
for nombre, ctr_extra in [("controles v9 (idx lineal)", []),
                          ("+ poly3 idx_cronologico", ["idx_p1", "idx_p2", "idx_p3"]),
                          ("+ FE bloques de 30", [c for c in pd.get_dummies(tb["bloque30"], prefix="blq", drop_first=True).columns])]:
    d = tb.copy()
    if "blq" in str(ctr_extra):
        dummies = pd.get_dummies(d["bloque30"], prefix="blq", drop_first=True).astype(float)
        d = pd.concat([d, dummies], axis=1)
    ctrl = [c for c in L.CONTROLES if c != "idx_cronologico"] + (["idx_cronologico"] if "poly3" not in nombre and "bloques" not in nombre else []) + ctr_extra
    dd = d[d.es_dev].dropna(subset=["x_F_exo2", L.KPI] + ctrl)
    f = L.ols_hc3(dd[L.KPI], dd[["x_F_exo2"] + ctrl])
    resumen.append(dict(spec=nombre, n=len(dd), beta_exo2=f.params["x_F_exo2"], p_exo2=f.pvalues["x_F_exo2"]))
# con exposicion des-tendenciada, controles v9 estandar
dd = tb[tb.es_dev].dropna(subset=["x_F_exo2_detrend", L.KPI] + L.CONTROLES)
f = L.ols_hc3(dd[L.KPI], dd[["x_F_exo2_detrend"] + L.CONTROLES])
resumen.append(dict(spec="exo2 des-tendenciado (residuo vs media movil +-10 batches), controles v9",
                     n=len(dd), beta_exo2=f.params["x_F_exo2_detrend"], p_exo2=f.pvalues["x_F_exo2_detrend"]))
rdf = pd.DataFrame(resumen)
rdf.to_csv(OUT / "e11_05_robustez_temporal_exo2.csv", index=False)
print()
print(rdf.round(4).to_string())

# correlacion cruzada entre las 3 exposiciones de Fusion (exo2 viene de o2, aire, gn: cual primitivo domina)
print()
print("corr(x_F_o2, x_F_aire, x_F_gn, x_F_exo2), DEV:")
print(dev[["x_F_o2", "x_F_aire", "x_F_gn", "x_F_exo2"]].corr().round(2).to_string())
