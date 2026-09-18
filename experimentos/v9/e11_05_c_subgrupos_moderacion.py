"""E11-05 (c) -- Subgrupos F1-F3 vs F4-F6 y moderacion por estado (T previa, %Sn y %FeO previos de escoria);
carbon de Fusion: neto por regimen/estado."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v9")
import v9_lib as L
import numpy as np, pandas as pd

OUT = L.SALIDA
res = L.residuos_escalon()
t = L.tabla_batch(res=res)

sys.path.insert(0, "experimentos/v8")
import v8_lib as L8, modelo_predictivo_v8 as mp8
df = L8.cargar_df()
base = mp8._preparar_base(df)

# ---------- subgrupos F1-F3 / F4-F6 ----------
F = res[res.grupo == "F"].copy()
sub13 = F[F.orden_escalon_fase <= 3]
sub46 = F[F.orden_escalon_fase >= 4]
expo = {}
for p in ["carbon", "gn", "exo2"]:
    expo[f"x_F13_{p}"] = sub13.groupby("Batch")[f"z__{p}"].mean()
    expo[f"x_F46_{p}"] = sub46.groupby("Batch")[f"z__{p}"].mean()
tsub = t.join(pd.DataFrame(expo))
dev = tsub[tsub.es_dev]
X = list(expo.keys())
filas = []
for nombre, d in [("DEV", dev), ("total", tsub)]:
    for y in [L.KPI, "f_metal", "f_dross", "f_polvo", "sn_perdido_escoria_frac"]:
        esc = 100.0 if (y.startswith("f_") or y.startswith("sn_")) else 1.0
        dd = d.dropna(subset=X + L.CONTROLES + [y])
        f = L.ols_hc3(dd[y] * esc, dd[X + L.CONTROLES])
        for x in X:
            filas.append(dict(regimen=nombre, y=y, x=x, n=len(dd), beta=f.params[x], p=f.pvalues[x]))
sg = pd.DataFrame(filas)
sg.to_csv(OUT / "e11_05_subgrupos_F13_F46.csv", index=False)
print("=== Subgrupos F1-F3 vs F4-F6 ===")
print(sg.round(4).to_string())

# ---------- moderacion por estado previo (batch = estado en F1: primer escalon de Fusion) ----------
key = ["Batch"]
F1 = base[(base.fase_proceso == "Fusión") & (base.orden_escalon_fase == 1)][
    key + ["temperatura_horno_celsius_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev"]
].drop_duplicates("Batch").set_index("Batch")
F1.columns = ["T_prev_F1", "sn_prev_F1", "feo_prev_F1"]
tm = t.join(F1)
filas_m = []
for mod in ["T_prev_F1", "sn_prev_F1", "feo_prev_F1"]:
    d = tm.dropna(subset=["x_F_exo2", "x_F_carbon", mod, L.KPI] + L.CONTROLES).copy()
    d["mod_c"] = (d[mod] - d[mod].mean()) / d[mod].std()
    d["exo2_x_mod"] = d["x_F_exo2"] * d["mod_c"]
    d["carbon_x_mod"] = d["x_F_carbon"] * d["mod_c"]
    for nombre, dd in [("DEV", d[d.es_dev]), ("total", d)]:
        Xm = ["x_F_exo2", "x_F_carbon", "mod_c", "exo2_x_mod", "carbon_x_mod"]
        f = L.ols_hc3(dd[L.KPI], dd[Xm + L.CONTROLES])
        for x in ["exo2_x_mod", "carbon_x_mod"]:
            filas_m.append(dict(moderador=mod, regimen=nombre, x=x, n=len(dd), beta=f.params[x], p=f.pvalues[x]))
mm = pd.DataFrame(filas_m)
mm.to_csv(OUT / "e11_05_moderacion.csv", index=False)
print()
print("=== Moderacion por estado previo (interaccion con exo2/carbon de Fusion) ===")
print(mm.round(4).to_string())

# ---------- carbon de Fusion: neto por regimen de ley de Sn y temperatura ----------
filas_c = []
for mod, dd0 in [("ley_sn_conc_batch_pct", t), ("T_media_F", t)]:
    d = dd0.dropna(subset=["x_F_carbon", mod, L.KPI] + L.CONTROLES).copy()
    med = d[mod].median()
    for nombre_reg, cond in [("bajo", d[mod] <= med), ("alto", d[mod] > med)]:
        dsub = d[cond]
        for nm, ds in [("DEV", dsub[dsub.es_dev]), ("total", dsub)]:
            dds = ds.dropna(subset=["x_F_carbon"] + L.CONTROLES + [L.KPI])
            if len(dds) < 30:
                continue
            f = L.ols_hc3(dds[L.KPI], dds[["x_F_carbon"] + L.CONTROLES])
            filas_c.append(dict(moderador=mod, nivel=nombre_reg, regimen=nm, n=len(dds),
                                 beta_carbon=f.params["x_F_carbon"], p=f.pvalues["x_F_carbon"]))
cc = pd.DataFrame(filas_c)
cc.to_csv(OUT / "e11_05_carbon_neto_regimen.csv", index=False)
print()
print("=== Carbon de Fusion: neto por regimen (ley Sn conc, T media F) ===")
print(cc.round(4).to_string())
