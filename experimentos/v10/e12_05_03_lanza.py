"""E12-05 tarea 3: lanza, forma no monotona.
(a) lineal+cuadratico del residuo (batch-level, KPI~xG+xC+ctrl+z+z^2).
(b) nivel absoluto: KPI residualizado vs posicion MEDIA de lanza en Reduccion, quintiles + curva cuadratica (vertice
    con IC bootstrap por batch).
(c) mecanismo de escalon: residuo-sobre-residuo de m6_ln_sn_dep, m8_ln_feo_ret_dross50, dT sobre z__lanza (lineal+cuad), n~1400.
(d) dinamica intra-batch: es un rastro de desgaste (tendencia con orden/gas acumulado) o de decision.
"""
import sys; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
from scipy import stats
import e12_05_lib as E

t = pd.read_csv(E.SALIDA / "e12_05_tabla_batch.csv", index_col=0)
res = E.residuos_escalon_e12()
CTRL = E.C_SIN_ESPESOR; BASE = ["xG", "xC"]; dev = t[t.es_dev]

# ---------------------------------------------------------------- (a) lineal + cuadratico, DEV/lockbox/total
filas = []
for g in ("R", "F"):
    col, colsq = f"x_{g}_lanza", f"x_{g}_lanza_sq"
    X = BASE + CTRL + [col, colsq]
    for muestra, tt in (("DEV", t[t.es_dev]), ("LOCKBOX", t[~t.es_dev]), ("TOTAL", t)):
        d = tt.dropna(subset=X + [E.KPI])
        if len(d) < 30:
            continue
        f = E.ols_hc3(d[E.KPI], d[X])
        b1, b2 = float(f.params[col]), float(f.params[colsq])
        # vertice en unidades de z (residuo estandarizado): -b1/(2 b2), solo si b2<0 (concava, maximo interior)
        vertice_z = -b1 / (2 * b2) if abs(b2) > 1e-9 else np.nan
        filas.append({"grupo": g, "muestra": muestra, "n": len(d), "lineal": b1, "p_lin": float(f.pvalues[col]),
                      "cuadratico": b2, "p_cuad": float(f.pvalues[colsq]), "forma": "concava(max interior)" if b2 < 0 else "convexa(min interior)",
                      "vertice_z": vertice_z})
tab_a = pd.DataFrame(filas); tab_a.to_csv(E.SALIDA / "e12_05_lanza_cuadratico.csv", index=False)
print("== (a) lineal + cuadratico del residuo de lanza, KPI =="); print(tab_a.round(4).to_string(index=False))

# ---------------------------------------------------------------- (b) nivel absoluto: posicion media R, quintiles + spline
# posicion media (cruda) de lanza en Reduccion, por batch
pos_media = res[res.grupo == "R"].groupby("Batch")["a__lanza"].mean().rename("pos_media_R")
tb = t.join(pos_media, how="left")
d = tb.dropna(subset=["pos_media_R", E.KPI] + BASE + CTRL).copy()
# KPI residualizado (fuera de la posicion) por xG+xC+controles
import statsmodels.api as sm
f0 = E.ols_hc3(d[E.KPI], d[BASE + CTRL])
d["kpi_resid"] = d[E.KPI] - f0.predict(sm.add_constant(d[BASE + CTRL], has_constant="add"))
d["quintil"] = pd.qcut(d["pos_media_R"], 5, labels=False, duplicates="drop")
qtab = d.groupby("quintil").agg(pos_media=("pos_media_R", "mean"), kpi_resid_media=("kpi_resid", "mean"),
                                 kpi_resid_se=("kpi_resid", lambda x: x.std() / np.sqrt(len(x))), n=("kpi_resid", "size"))
qtab.to_csv(E.SALIDA / "e12_05_lanza_quintiles.csv")
print("\n== (b) KPI residualizado por quintil de posicion media de lanza (Reduccion), DEV+LB =="); print(qtab.round(3).to_string())

# curva cuadratica (nivel, no residuo) + vertice con IC bootstrap por batch
x = d["pos_media_R"].to_numpy(); y = d["kpi_resid"].to_numpy(); batches = d.index.to_numpy()


def ajustar_vertice(xx, yy):
    xm = xx.mean(); xs = xx.std()
    xz = (xx - xm) / xs
    c = np.polyfit(xz, yy, 2)  # c[0] z^2 + c[1] z + c[2]
    if abs(c[0]) < 1e-9:
        return np.nan, c
    vz = -c[1] / (2 * c[0])
    return vz * xs + xm, c


v_obs, c_obs = ajustar_vertice(x, y)
rng = np.random.default_rng(E.SEED)
uniq = np.unique(batches)
vs = []
for _ in range(1000):
    samp = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([np.where(batches == b)[0] for b in samp])
    vv, _ = ajustar_vertice(x[idx], y[idx])
    if np.isfinite(vv) and d["pos_media_R"].min() - 500 < vv < d["pos_media_R"].max() + 500:
        vs.append(vv)
vs = np.array(vs)
print(f"\n== (b) vertice (nivel, curva cuadratica) = {v_obs:.0f} mm | signo cuadratico obs = {'concava(max)' if c_obs[0]<0 else 'convexa(min)'} "
      f"| IC95 bootstrap (batch, n_validos={len(vs)}/1000) = [{np.percentile(vs,2.5):.0f}, {np.percentile(vs,97.5):.0f}] "
      f"| mediana historica pos_media_R = {d['pos_media_R'].median():.0f}")
pd.DataFrame({"vertice_boot": vs}).to_csv(E.SALIDA / "e12_05_lanza_vertice_boot.csv", index=False)

# ---------------------------------------------------------------- (c) mecanismo de escalon (n~1400, grupo R)
res_extra = E.residuos_escalon_e12(recalcular=True, cache=None, estados={"Reducción": E.ESTADOS["Reducción"]},
                                    palancas={"sn_dep": "m6_ln_sn_dep", "feo_ret": "m8_ln_feo_ret_dross50", "dT": "grad_temperatura_horno_celsius"})
key = ["Batch", "orden_escalon_fase"]
rc = res[res.grupo == "R"][key + ["z__lanza"]].merge(
    res_extra[res_extra.grupo == "R"][key + ["res__sn_dep", "res__feo_ret", "res__dT", "es_dev"]], on=key, how="inner")
rc["z_lanza_sq"] = rc["z__lanza"] ** 2
print(f"\n== (c) mecanismo de escalon, grupo R, n={len(rc)} ==")
filas_c = []
for target in ("res__sn_dep", "res__feo_ret", "res__dT"):
    dd = rc.dropna(subset=[target, "z__lanza", "z_lanza_sq"])
    if len(dd) < 50:
        continue
    import statsmodels.api as sm
    X = sm.add_constant(dd[["z__lanza", "z_lanza_sq"]], has_constant="add")
    fit = sm.OLS(dd[target], X).fit(cov_type="cluster", cov_kwds={"groups": dd["Batch"]})
    filas_c.append({"target": target, "n": len(dd), "lineal": float(fit.params["z__lanza"]), "p_lin": float(fit.pvalues["z__lanza"]),
                    "cuadratico": float(fit.params["z_lanza_sq"]), "p_cuad": float(fit.pvalues["z_lanza_sq"])})
tab_c = pd.DataFrame(filas_c); tab_c.to_csv(E.SALIDA / "e12_05_lanza_mecanismo.csv", index=False)
print(tab_c.round(4).to_string(index=False))

# ---------------------------------------------------------------- (d) dinamica intra-batch: desgaste vs decision
print("\n== (d) dinamica intra-batch de la lanza ==")
filas_d = []
for g in ("R", "F"):
    sub = res[res.grupo == g].dropna(subset=["z__lanza", "orden_escalon_fase"]).copy()
    import statsmodels.api as sm
    X = sm.add_constant(sub[["orden_escalon_fase"]], has_constant="add")
    fit = sm.OLS(sub["z__lanza"], X).fit(cov_type="cluster", cov_kwds={"groups": sub["Batch"]})
    filas_d.append({"grupo": g, "chequeo": "z__lanza ~ orden (residual, deberia ser ~0: ya en S)",
                    "coef": float(fit.params["orden_escalon_fase"]), "p": float(fit.pvalues["orden_escalon_fase"]), "n": len(sub)})
# correlacion con proxies de desgaste (gas acumulado) vs decision (leyes/inventario, ya en S -> deberian ser ~0 tras residualizar)
dfc = E.preparar_df()
import modelo_predictivo_v8 as mp8
base = mp8._preparar_base(dfc)
key = ["Batch", "fase_proceso", "orden_escalon_fase"]
extra = base[key + ["cum_o2_nm3_prev", "cum_gn_nm3_prev", "cum_aire_nm3_prev", "tiempo_fase", "d_posicion_vertical_lanza_mm"]].drop_duplicates(key)
rw = res.merge(extra, left_on=["Batch", "fase_proceso", "orden_escalon_fase"], right_on=key, how="left")
for g in ("R", "F"):
    sub = rw[rw.grupo == g]
    for proxy in ("cum_o2_nm3_prev", "cum_gn_nm3_prev", "cum_aire_nm3_prev", "tiempo_fase"):
        d2 = sub.dropna(subset=["z__lanza", proxy])
        if len(d2) < 50:
            continue
        rho, pv = stats.spearmanr(d2["z__lanza"], d2[proxy])
        filas_d.append({"grupo": g, "chequeo": f"z__lanza vs {proxy} (proxy desgaste, Spearman)", "coef": round(rho, 3), "p": round(pv, 4), "n": len(d2)})
tab_d = pd.DataFrame(filas_d); tab_d.to_csv(E.SALIDA / "e12_05_lanza_intra_batch.csv", index=False)
print(tab_d.round(4).to_string(index=False))

# distribucion de d_posicion_vertical_lanza_mm (cambio escalon a escalon crudo): drift suave vs saltos discretos
dd = rw.dropna(subset=["d_posicion_vertical_lanza_mm"])
for g in ("R", "F"):
    sub = dd[dd.grupo == g]["d_posicion_vertical_lanza_mm"]
    print(f"grupo {g}: d_posicion median={sub.median():.0f} | P10={sub.quantile(.10):.0f} P90={sub.quantile(.90):.0f} "
          f"| frac |d|<50mm = {(sub.abs()<50).mean():.2f} | frac |d|>300mm = {(sub.abs()>300).mean():.2f}")
