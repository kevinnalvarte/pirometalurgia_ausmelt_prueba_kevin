"""E18-01 tarea 1 -- caracteriza las revisiones de receta (GN y carbon): fechas/idx, tamano del cambio por
escalon, duracion de cada version; regresion del CAMBIO de receta sobre variables PRE-revision (KPI medio de
los 10 batches previos, T horno previa, ley del concentrado, dross previo) para saber si el planificador
reacciona a algo observable (politica reactiva) -> si es asi, hay que controlar esa variable en el resto del
experimento.
"""
import sys
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
import e18_01_lib as E

r, bt = E.A.escalones_con_receta()
t = E.tabla_batch()

SAL = E.RAIZ / "experimentos" / "v16"


def firma_y_version(col_plan: str) -> tuple[pd.Series, pd.Series]:
    piv = r.pivot_table(index="Batch", columns="orden_escalon_fase", values=col_plan).reindex(t.index)
    firma = piv.round(2).fillna(-1).apply(lambda s: "|".join(f"{v:.2f}" for v in s), axis=1)
    version = (firma != firma.shift(1)).cumsum()
    return firma, version


firma_gn, ver_gn = firma_y_version("plan__gn")
firma_c, ver_c = firma_y_version("plan__carbon")
t["version_gn"], t["version_carbon"] = ver_gn.to_numpy(), ver_c.to_numpy()


def tabla_runs(version: pd.Series, plan_media_col: str) -> pd.DataFrame:
    tt = t.assign(version=version.to_numpy())
    runs = tt.groupby("version").agg(ini_idx=("idx_cronologico", "min"), fin_idx=("idx_cronologico", "max"),
                                      n=("idx_cronologico", "size"), plan_medio=(plan_media_col, "mean"),
                                      fecha_ini=("idx_cronologico", "min"))
    runs["delta_plan"] = runs["plan_medio"].diff()
    return runs


runs_gn = tabla_runs(ver_gn, "plan_gn")
runs_c = tabla_runs(ver_c, "plan_carbon")
print(f"n versiones GN: {len(runs_gn)} (n>=8 batches: {(runs_gn.n >= 8).sum()})")
print(f"n versiones carbon: {len(runs_c)} (n>=8 batches: {(runs_c.n >= 8).sum()})")
print("\n-- versiones GN (n>=8) --")
print(runs_gn[runs_gn.n >= 8].round(2).to_string())
print("\n-- versiones carbon (n>=8) --")
print(runs_c[runs_c.n >= 8].round(2).to_string())
runs_gn.to_csv(SAL / "e18_01_01_versiones_gn.csv")
runs_c.to_csv(SAL / "e18_01_01_versiones_carbon.csv")

# ---------------------------------------------------------------- regresion del cambio de receta sobre estado pre-revision
def eventos_prerevision(version: pd.Series, plan_media_col: str, k: int = 10) -> pd.DataFrame:
    tt = t.assign(version=version.to_numpy()).sort_values("idx_cronologico").reset_index()
    filas = []
    vers = sorted(tt.version.unique())
    for v in vers[1:]:
        cur = tt[tt.version == v]
        prev = tt[tt.version == v - 1] if (v - 1) in tt.version.values else tt[tt.version < v]
        if len(prev) < 3 or len(cur) < 1:
            continue
        pre = prev.tail(k)
        delta = cur[plan_media_col].mean() - prev[plan_media_col].mean()
        filas.append({"version": v, "idx_inicio": cur.idx_cronologico.min(), "n_pre": len(pre),
                      "delta_plan": delta, "kpi_prev10": pre[E.KPI].mean(), "T_prev": pre["T_media_R"].mean(),
                      "ley_prev": pre["ley_sn_conc_batch_pct"].mean(), "dross_prev10": pre["f_dross"].mean(),
                      "feed_sn_prev": pre["feed_sn_total_t"].mean()})
    return pd.DataFrame(filas)


ev_gn = eventos_prerevision(ver_gn, "plan_gn")
ev_c = eventos_prerevision(ver_c, "plan_carbon")
ev_gn.to_csv(SAL / "e18_01_01_eventos_prerevision_gn.csv", index=False)
ev_c.to_csv(SAL / "e18_01_01_eventos_prerevision_carbon.csv", index=False)

print(f"\nn eventos de revision GN (con >=3 batches previos): {len(ev_gn)}")
print(f"n eventos de revision carbon: {len(ev_c)}")

for nm, ev in (("GN", ev_gn), ("carbon", ev_c)):
    d = ev.dropna(subset=["delta_plan", "kpi_prev10", "T_prev", "ley_prev", "dross_prev10"])
    if len(d) < 8:
        print(f"{nm}: muy pocos eventos ({len(d)}) para regresion confiable")
        continue
    X = d[["kpi_prev10", "T_prev", "ley_prev", "dross_prev10"]].copy()
    X = (X - X.mean()) / X.std()   # estandarizado: delta por sd de cada regresor
    f = E.ols_robusto(d["delta_plan"], X, cov="HC3")
    tabla = pd.DataFrame({"coef": f.params, "se": f.bse, "t": f.tvalues, "p": f.pvalues})
    print(f"\n>> delta_plan_{nm} ~ estado PRE-revision (n={len(d)}, estandarizado), R2={f.rsquared:.3f}:")
    print(tabla.round(3).to_string())
    tabla.to_csv(SAL / f"e18_01_01_regresion_prerevision_{nm}.csv")
