"""E18-03 B -- regla unilateral "GN nunca por encima de la receta" (disciplina, sin modelo).
Exposicion en unidades FISICAS (Nm3/min de desvio) -- no requiere un beta estimado para ejecutarse."""
import sys, time
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import e18_03_lib as E
import e11_07_lib as W7
import e13_03_lib as L13

t0 = time.time()
SAL = "experimentos/v16"
r, bt = E.cargar()
reglas = E.reglas_alcance(r, bt)
A1 = reglas["A1_T1000"]

# 1) % de escalones con GN > receta y exceso medio por orden
desc = r.groupby("orden_escalon_fase").apply(lambda g: pd.Series({
    "pct_gn_sobre_receta": 100 * (g["dev__gn"] > 0).mean(),
    "exceso_medio_Nm3min": g.loc[g["dev__gn"] > 0, "dev__gn"].mean(),
    "defecto_medio_Nm3min": g.loc[g["dev__gn"] < 0, "dev__gn"].mean(),
    "n": len(g)}), include_groups=False)
desc.to_csv(f"{SAL}/e18_03_B_descriptivo_por_orden.csv")
print("== GN ejecutado vs receta, por orden ==")
print(desc.round(3).to_string())
print(f"\nTotal: {100*(r['dev__gn']>0).mean():.1f}% de escalones con GN > receta; exceso medio {r.loc[r.dev__gn>0,'dev__gn'].mean():.2f} Nm3/min")

# 2) x+ = media(max(dev_gn,0)); x- = media(min(dev_gn,0)); carbon = media(dev__carbon) -- fisicas, por batch
xplus = r["dev__gn"].clip(lower=0).groupby(r["Batch"]).mean().rename("x_plus")
xminus = r["dev__gn"].clip(upper=0).groupby(r["Batch"]).mean().rename("x_minus")
xcarb = r["dev__carbon"].groupby(r["Batch"]).mean().rename("x_carbon")
expo = pd.concat([xplus, xminus, xcarb], axis=1)

CONTROLES = E.CONTROLES
for y_col, esc in ((E.KPI, 1.0), ("f_dross", 100.0), ("sn_perdido_escoria_frac", 100.0)):
    d = bt.join(expo).dropna(subset=["x_plus", "x_minus", "x_carbon"] + CONTROLES + [y_col])
    f = sm.OLS(d[y_col] * esc, sm.add_constant(d[["x_plus", "x_minus", "x_carbon"] + CONTROLES])).fit(cov_type="HC3")
    bp, bm = f.params["x_plus"], f.params["x_minus"]
    # test de asimetria: beta+ == beta- (Wald)
    R = np.zeros(len(f.params)); R[list(f.params.index).index("x_plus")] = 1; R[list(f.params.index).index("x_minus")] = -1
    wald = f.t_test(R)
    wald_t = float(np.asarray(wald.tvalue).ravel()[0]); wald_p = float(np.asarray(wald.pvalue).ravel()[0])
    print(f"\n[{y_col}] n={len(d)}  x_plus:{bp:+.3f}(t{f.tvalues['x_plus']:+.2f})  x_minus:{bm:+.3f}(t{f.tvalues['x_minus']:+.2f})  "
          f"x_carbon:{f.params['x_carbon']:+.3f}(t{f.tvalues['x_carbon']:+.2f})  |  H0 beta+=beta-: t={wald_t:.2f} p={wald_p:.4f}")
    pd.DataFrame({"param": f.params.index, "beta": f.params.values, "se": f.bse.values, "t": f.tvalues.values}).assign(
        y=y_col, wald_asimetria_t=wald_t, wald_asimetria_p=wald_p).to_csv(
        f"{SAL}/e18_03_B_asimetria_{y_col.split('_')[0]}.csv", index=False)

# 3) ganancia esperada = -beta_plus * E[x_plus], bootstrap POR BLOQUES cronologicos de 10 (DEV, total, dentro de A1)
xplus_A1 = r["dev__gn"].clip(lower=0).where(A1).groupby(r["Batch"]).mean().fillna(0.0).rename("x_plus_A1")
d_kpi = bt.join(expo).join(xplus_A1).dropna(subset=["x_plus", "x_minus", "x_carbon", "x_plus_A1"] + CONTROLES + [E.KPI]).sort_values("idx_cronologico").reset_index(drop=True)


def boot_bloques(dd_base: pd.DataFrame, xcol: str, reps: int, seed: int) -> np.ndarray:
    rng_ = np.random.default_rng(seed); n_ = len(dd_base); nb_ = max(n_ // 10, 1)
    out = []
    for _ in range(reps):
        starts = rng_.integers(0, nb_, nb_)
        take = np.concatenate([np.arange(b * 10, min(b * 10 + 10, n_)) for b in starts])
        dd = dd_base.iloc[take]
        ff = sm.OLS(dd[E.KPI], sm.add_constant(dd[[xcol, "x_minus", "x_carbon"] + CONTROLES])).fit()
        out.append((ff.params[xcol], -ff.params[xcol] * dd[xcol].mean()))
    return np.array(out)


filas_g = []
for nombre, sub, xcol in (("total", d_kpi, "x_plus"), ("DEV", d_kpi[d_kpi["es_dev"]].reset_index(drop=True), "x_plus"),
                           ("dentro_A1(T>=1000)", d_kpi, "x_plus_A1")):
    boots = boot_bloques(sub, xcol, 2000, 7)
    filas_g.append({"segmento": nombre, "n": len(sub), "beta_plus_P50": np.median(boots[:, 0]),
                     "ganancia_P50": np.median(boots[:, 1]), "IC95_lo": np.percentile(boots[:, 1], 2.5),
                     "IC95_hi": np.percentile(boots[:, 1], 97.5), "P_mayor_0": np.mean(boots[:, 1] > 0)})
gan = pd.DataFrame(filas_g); gan.to_csv(f"{SAL}/e18_03_B_ganancia_boot.csv", index=False)
print("\n== ganancia esperada de 'GN nunca sobre receta' (-beta+ * E[x+]), bootstrap por bloques ==")
print(gan.round(4).to_string())

# 4) prueba prospectiva de x+ solo
t_tab = bt.join(expo[["x_plus"]]).fillna({"x_plus": 0.0}).sort_values("idx_cronologico")
P = W7.preparar(t_tab, ["x_plus"], y_col=E.KPI, controles=CONTROLES)
sc = L13.wf(P["Xexp"], P["Xctrl"], P["y"], 100, paso=10, modo="js")
obs = W7.stats_final(sc, P["y"], P["Xctrl"])
bl = L13.bloques(P["n"], 20, seed=0)
rngp = np.random.default_rng(E.SEED)
nul = np.array([W7.stats_final(L13.wf(L13.permutar_bloques(P["Xexp"], bl, rngp), P["Xctrl"], P["y"], 100, modo="js"), P["y"], P["Xctrl"])["t"] for _ in range(1000)])
p_perm = L13.p_perm(obs["t"], nul)
print(f"\n== prospectivo x+ solo == n={obs['n']} pendiente={obs['pendiente']:.3f} t={obs['t']:.2f} p_perm={p_perm:.4f}")
pd.DataFrame([{"n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"], "p_perm": p_perm}]).to_csv(f"{SAL}/e18_03_B_prospectivo_xplus.csv", index=False)

# 5) seguridad termica: efecto de x+ y x- sobre dT del escalon y T horno al cierre de Reduccion
r2 = r.copy()
r2["dev_gn_pos"] = r2["dev__gn"].clip(lower=0); r2["dev_gn_neg"] = r2["dev__gn"].clip(upper=0)
d_esc = r2.dropna(subset=["d_temperatura_horno_celsius", "dev_gn_pos", "dev_gn_neg", "temperatura_horno_celsius_prev", "orden_escalon_fase"])
f_dt = sm.OLS(d_esc["d_temperatura_horno_celsius"], sm.add_constant(d_esc[["dev_gn_pos", "dev_gn_neg", "temperatura_horno_celsius_prev", "orden_escalon_fase"]])
              ).fit(cov_type="cluster", cov_kwds={"groups": d_esc["Batch"]})
print("\n== dT del escalon ~ x+ (escalon) + x- (escalon) + T_prev + orden (cluster por Batch) ==")
print(f_dt.summary2().tables[1].round(3).to_string())

T_cierre = r[r["orden_escalon_fase"] == 3].set_index("Batch")["temperatura_horno_celsius"].rename("T_cierre_R")
d_b = bt.join(expo).join(T_cierre).dropna(subset=["x_plus", "x_minus", "T_cierre_R"] + CONTROLES)
f_tc = sm.OLS(d_b["T_cierre_R"], sm.add_constant(d_b[["x_plus", "x_minus", "x_carbon"] + CONTROLES])).fit(cov_type="HC3")
print("\n== T horno al cierre de Reduccion ~ x+ + x- + x_carbon + controles ==")
print(f_tc.summary2().tables[1].round(3).to_string())
f_dt.summary2().tables[1].assign(y="dT_escalon").to_csv(f"{SAL}/e18_03_B_seguridad_dT.csv")
f_tc.summary2().tables[1].assign(y="T_cierre_R").to_csv(f"{SAL}/e18_03_B_seguridad_Tcierre.csv")
print(f"\n[{time.time()-t0:.0f}s] listo")
