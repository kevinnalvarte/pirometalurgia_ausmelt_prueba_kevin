"""E18-01 tarea 4 -- estudio de eventos: revisiones de receta GN con |delta| >= 0.4 Nm3/min (media R0-R3) y
>= 6 batches sin otra revision grande a cada lado. KPI/canales residualizados por controles; comparacion de
medias de residuos en k batches posteriores vs anteriores (k=6,8,10); pendiente pp por Nm3/min (una obs por
evento) con bootstrap por evento."""
import sys
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
import e18_01_lib as E

t = E.tabla_batch().reset_index()
SAL = E.RAIZ / "experimentos" / "v16"
UMBRAL = 0.4
KS = (6, 8, 10)

# firma/version igual que en la tarea 1 (recalculada aqui para no depender de ese script)
r, bt = E.A.escalones_con_receta()
piv = r.pivot_table(index="Batch", columns="orden_escalon_fase", values="plan__gn").reindex(t.Batch)
firma = piv.round(2).fillna(-1).apply(lambda s: "|".join(f"{v:.2f}" for v in s), axis=1)
version = (firma != firma.shift(1)).cumsum()
t["version_gn"] = version.to_numpy()
runs = t.groupby("version_gn").agg(ini_idx=("idx_cronologico", "min"), fin_idx=("idx_cronologico", "max"), n=("idx_cronologico", "size"), plan_medio=("plan_gn", "mean"))
runs["delta_plan"] = runs["plan_medio"].diff()

# residuales de KPI y f_dross ~ controles (incl. idx_cronologico) sobre toda la tabla
def residuos(y_col: str) -> pd.Series:
    d = t.dropna(subset=[y_col] + E.CONTROLES)
    f = E.ols_robusto(d[y_col], d[E.CONTROLES], cov="HC3")
    resid = pd.Series(np.nan, index=t.index)
    resid.loc[d.index] = f.resid
    return resid


t["res_kpi"] = residuos(E.KPI)
t["res_dross"] = residuos("f_dross") * 100.0

eventos = []
grandes = runs[(runs.delta_plan.abs() >= UMBRAL)]
for v in grandes.index:
    if v - 1 not in runs.index or v + 1 not in runs.index:
        pass
    n_prev = runs.loc[v - 1, "n"] if (v - 1) in runs.index else 0
    n_cur = runs.loc[v, "n"]
    if n_prev < 6 or n_cur < 1:
        continue
    idx0 = int(runs.loc[v, "ini_idx"])   # primer idx de la version nueva
    pos0 = t.index[t.idx_cronologico == idx0][0]
    for k in KS:
        pre = t.iloc[max(0, pos0 - k):pos0]
        post = t.iloc[pos0:pos0 + k]
        if len(pre) < k or len(post) < min(k, n_cur):
            continue
        eventos.append({"version": v, "idx_evento": idx0, "k": k, "delta_plan_gn": runs.loc[v, "delta_plan"],
                         "n_pre_disp": len(pre), "n_post_disp": len(post),
                         "kpi_res_pre": pre.res_kpi.mean(), "kpi_res_post": post.res_kpi.mean(),
                         "dross_res_pre": pre.res_dross.mean(), "dross_res_post": post.res_dross.mean()})

ev = pd.DataFrame(eventos)
ev["d_kpi"] = ev.kpi_res_post - ev.kpi_res_pre
ev["d_dross"] = ev.dross_res_post - ev.dross_res_pre
ev.to_csv(SAL / "e18_01_04_eventos.csv", index=False)
print(f"Umbral |delta receta GN| >= {UMBRAL} Nm3/min, >=6 batches previos sin otra revision grande")
print(f"n eventos candidatos (versiones): {len(grandes)}; n filas evento x k: {len(ev)}")
pd.set_option("display.width", 200)
print(ev.round(3).to_string(index=False))

rng = np.random.default_rng(0)
for k in KS:
    d = ev[ev.k == k].dropna(subset=["delta_plan_gn", "d_kpi"])
    if len(d) < 5:
        print(f"\nk={k}: solo {len(d)} eventos, insuficiente")
        continue
    f = E.ols_robusto(d.d_kpi, d[["delta_plan_gn"]], cov="HC3")
    b = f.params["delta_plan_gn"]
    n = len(d)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        dd = d.iloc[idx]
        if dd.delta_plan_gn.std() < 1e-9:
            continue
        try:
            bb = E.ols_robusto(dd.d_kpi, dd[["delta_plan_gn"]], cov="HC3").params["delta_plan_gn"]
            boots.append(bb)
        except Exception:
            continue
    boots = np.array(boots)
    print(f"\nk={k} (n eventos={n}): pendiente dKPI/dReceta_GN = {b:+.3f} pp/(Nm3/min), t={f.tvalues['delta_plan_gn']:.2f}, "
          f"IC95 bootstrap [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}], P(<0)={np.mean(boots < 0):.3f}")
    fd = E.ols_robusto(d.reindex(ev[ev.k == k].dropna(subset=['delta_plan_gn','d_dross']).index).d_dross if False else
                        ev[ev.k == k].dropna(subset=["delta_plan_gn", "d_dross"]).d_dross,
                        ev[ev.k == k].dropna(subset=["delta_plan_gn", "d_dross"])[["delta_plan_gn"]], cov="HC3")
    print(f"        pendiente dDROSS/dReceta_GN = {fd.params['delta_plan_gn']:+.3f} pp/(Nm3/min), t={fd.tvalues['delta_plan_gn']:.2f}")
