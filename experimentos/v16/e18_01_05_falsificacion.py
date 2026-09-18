"""E18-01 tarea 5 -- falsificacion: (a) revisiones placebo en ubicaciones aleatorias -> nula del estudio de
eventos; (b) la receta FUTURA (version siguiente) no debe predecir el KPI actual; (c) receta de aire (solo 2
valores en toda la campana) como instrumento placebo."""
import sys
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
import e18_01_lib as E

t = E.tabla_batch().reset_index()
SAL = E.RAIZ / "experimentos" / "v16"
rng = np.random.default_rng(0)

r, bt = E.A.escalones_con_receta()
piv = r.pivot_table(index="Batch", columns="orden_escalon_fase", values="plan__gn").reindex(t.Batch)
firma = piv.round(2).fillna(-1).apply(lambda s: "|".join(f"{v:.2f}" for v in s), axis=1)
version = (firma != firma.shift(1)).cumsum()
t["version_gn"] = version.to_numpy()
runs = t.groupby("version_gn").agg(ini_idx=("idx_cronologico", "min"), n=("idx_cronologico", "size"), plan_medio=("plan_gn", "mean"))
runs["delta_plan"] = runs["plan_medio"].diff()
runs["plan_futuro"] = runs["plan_medio"].shift(-1)   # media de la receta de la SIGUIENTE version

t["plan_gn_futuro"] = t["version_gn"].map(runs["plan_futuro"])


def residuos(y_col: str) -> pd.Series:
    d = t.dropna(subset=[y_col] + E.CONTROLES)
    f = E.ols_robusto(d[y_col], d[E.CONTROLES], cov="HC3")
    resid = pd.Series(np.nan, index=t.index)
    resid.loc[d.index] = f.resid
    return resid


t["res_kpi"] = residuos(E.KPI)

# --------------------------------------------------------------------------- (a) revisiones placebo
K = 6
deltas_reales = runs.loc[runs.delta_plan.abs() >= 0.4, "delta_plan"].dropna().to_numpy()
n_ev = 15  # igual que en e18_01_04 (k=6)
slope_real = 0.552  # de e18_01_04, k=6

n_rep = 3000
slopes_null = []
n_t = len(t)
for _ in range(n_rep):
    pos = rng.integers(K, n_t - K, n_ev)
    deltas = rng.choice(deltas_reales, n_ev, replace=True)
    d_kpi = np.array([t.res_kpi.iloc[p:p + K].mean() - t.res_kpi.iloc[p - K:p].mean() for p in pos])
    ok = ~np.isnan(d_kpi)
    if ok.sum() < 5 or np.std(deltas[ok]) < 1e-9:
        continue
    b = np.polyfit(deltas[ok], d_kpi[ok], 1)[0]
    slopes_null.append(b)
slopes_null = np.array(slopes_null)
p_val = np.mean(np.abs(slopes_null) >= abs(slope_real))
print(f"(a) Placebo eventos aleatorios (k={K}, {len(slopes_null)} rep): nula de la pendiente dKPI/dReceta -> "
      f"media {slopes_null.mean():+.3f}, sd {slopes_null.std():.3f}, P(|nula|>=|real={slope_real}|) = {p_val:.3f}")
pd.DataFrame({"slope_null": slopes_null}).to_csv(SAL / "e18_01_05a_placebo_eventos.csv", index=False)

# --------------------------------------------------------------------------- (b) receta futura no debe predecir KPI actual
d = t.dropna(subset=["plan_gn_futuro", "plan_gn", E.KPI] + E.CONTROLES)
X = d[["plan_gn_futuro", "plan_gn"] + E.CONTROLES]
f = E.ols_robusto(d[E.KPI], X, cov="HC3")
print(f"\n(b) KPI actual ~ receta GN FUTURA (siguiente version) + receta GN actual + controles (n={len(d)}):")
tabla_b = pd.DataFrame({"coef": f.params, "se": f.bse, "t": f.tvalues, "p": f.pvalues})
print(tabla_b.round(3).to_string())
tabla_b.to_csv(SAL / "e18_01_05b_receta_futura.csv")

# --------------------------------------------------------------------------- (c) aire como instrumento placebo
Xexog = E.controles_tendencia(t, "lineal")
base = pd.concat([t[["ejec_aire", "plan_aire", E.KPI]], Xexog], axis=1).dropna()
print(f"\n(c) receta de AIRE: valores distintos en toda la campana = {t.plan_aire.round(2).nunique()}")
r_f = E.primera_etapa_F(base["ejec_aire"].to_numpy(float), base["plan_aire"].to_numpy(float),
                        np.column_stack([np.ones(len(base)), base[Xexog.columns].to_numpy(float)])[:, 1:], None, cov="HC3")
print(f"    primera etapa (ejec_aire ~ plan_aire + controles): F={r_f['F']:.2f}  p={r_f['p']:.3f}  coef={r_f['coef_own'][0]:.3f}  R2={r_f['r2']:.3f}")
xend = base[["ejec_aire"]].to_numpy(float); zexcl = base[["plan_aire"]].to_numpy(float)
xexog_c = np.column_stack([np.ones(len(base)), base[Xexog.columns].to_numpy(float)])
iv = E.iv2sls(base[E.KPI].to_numpy(float), xend, xexog_c, zexcl, cov="HC1", nombres_end=["ejec_aire"], nombres_exog=["const"] + list(Xexog.columns))
print(f"    2SLS KPI ~ ejec_aire (instrumentado por plan_aire): b={iv['beta']['ejec_aire']:+.3f}  se={iv['se']['ejec_aire']:.3f}  t={iv['t']['ejec_aire']:.2f}")
pd.DataFrame([{"F_primera_etapa": r_f["F"], "p_primera_etapa": r_f["p"], "b_iv": iv["beta"]["ejec_aire"],
               "se_iv": iv["se"]["ejec_aire"], "t_iv": iv["t"]["ejec_aire"], "n_valores_aire": int(t.plan_aire.round(2).nunique())}]
             ).to_csv(SAL / "e18_01_05c_placebo_aire.csv", index=False)
