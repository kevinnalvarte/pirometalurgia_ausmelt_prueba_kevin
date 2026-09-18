"""E17-02 (Fable): las revisiones de la receta (niveles discretos por tramos) son quiebres conocidos ex-ante. (1) cuando ocurren; (2) beta del desvio de GN por
version de receta; (3) asesor con ventana reiniciada en cada revision (solo batches de la version vigente, min 30; si no, abstenerse) vs W=100."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import asesor_receta_v10 as A, v9_lib as L, e11_07_lib as W7
r, bt = A.escalones_con_receta(); t = pd.read_csv("experimentos/v15/e17_01_tabla.csv", index_col=0).sort_values("idx_cronologico")
pl = r.pivot_table(index="Batch", columns="orden_escalon_fase", values="plan__gn").reindex(t.index); pc = r.pivot_table(index="Batch", columns="orden_escalon_fase", values="plan__carbon").reindex(t.index)
firma = pl.round(1).fillna(-1).apply(lambda s: "|".join(f"{v:.1f}" for v in s), axis=1); cambio = (firma != firma.shift(1)); ver = cambio.cumsum(); t["version_gn"] = ver.to_numpy()
runs = t.groupby("version_gn").agg(ini=("idx_cronologico", "min"), n=("idx_cronologico", "size"), T=("T_media_R", "mean")); runs["receta_GN_R0..R3"] = [firma[t.version_gn == v].iloc[0] for v in runs.index]
print("n versiones de receta GN:", len(runs)); print(runs[runs.n >= 8].round(0).to_string())
# version 'gruesa': media de la receta GN del batch, suavizada: nivel de fuego planificado
t["plan_gn_medio"] = pl.mean(axis=1).to_numpy(); t["plan_c_medio"] = pc.mean(axis=1).to_numpy()
print(t.groupby(pd.cut(t.idx_cronologico, [0, 100, 200, 300, 362]))[["plan_gn_medio", "plan_c_medio", "T_media_R", "xG"]].mean().round(2).to_string())
C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]
# beta del desvio por nivel de receta (terciles del nivel planificado de GN)
t["nivel"] = pd.qcut(t.plan_gn_medio, 3, labels=["receta GN baja", "media", "alta"])
for nv, g in t.groupby("nivel", observed=True):
    dd = g.dropna(subset=["xG", "xC", L.KPI] + C); f = L.ols_hc3(dd[L.KPI], dd[["xG", "xC"] + C]); print(nv, len(dd), "idx medio %.0f" % dd.idx_cronologico.mean(), "bG %.2f (t %.1f) bC %.2f (t %.1f)" % (f.params.xG, f.tvalues.xG, f.params.xC, f.tvalues.xC))
# interaccion continua del desvio con el nivel de la receta (conocido ex-ante)
t["zP"] = (t.plan_gn_medio - t.plan_gn_medio[t.es_dev].mean()) / t.plan_gn_medio[t.es_dev].std(); t["xG_zP"] = t.xG * t.zP
for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
    dd = tt.dropna(subset=["xG", "xG_zP", "xC", "zP", L.KPI] + C); f = L.ols_hc3(dd[L.KPI], dd[["xG", "xG_zP", "xC", "zP"] + C]); print(nm, len(dd), " ".join(f"{c}:{f.params[c]:+.2f}(t{f.tvalues[c]:+.1f})" for c in ["xG", "xG_zP", "xC", "zP"]), "| zP medio %.2f" % dd.zP.mean())
d1 = t[t.es_dev].dropna(subset=["xG", "xG_zP", "xC", "zP", L.KPI] + C); f = L.ols_hc3(d1[L.KPI], d1[["xG", "xG_zP", "xC", "zP"] + C]); d2 = t[~t.es_dev].copy(); d2["score"] = d2[["xG", "xG_zP", "xC"]].to_numpy() @ f.params[["xG", "xG_zP", "xC"]].to_numpy()
q = L.ols_hc3(d2[L.KPI], d2[["score"] + C]); print(">> HOLDOUT DEV->LB con interaccion por nivel de receta: pendiente %.2f t %.2f" % (q.params.score, q.tvalues.score))
