"""E11-05 (d) -- Mecanismo de escalon en Fusion: PLM residual-sobre-residual (z de la accion vs residuo HGB cross-fitted
del target), y si exo2_F predice el estado de entrada a Reduccion (FeO/Sn/T al cierre de F6)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v9")
import v9_lib as L
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

OUT = L.SALIDA
sys.path.insert(0, "experimentos/v8")
import v8_lib as L8, modelo_predictivo_v8 as mp8

df = L8.cargar_df()
base = mp8._preparar_base(df)
res = L.residuos_escalon()
dev_batches, lockbox_batches = L8.mp.split_dev_lockbox(df)

F = base[(base["fase_proceso"] == "Fusión") & (base["orden_escalon_fase"] >= 1)].copy()
F["es_dev"] = F["Batch"].isin(dev_batches)
S = L.ESTADO_F
L8.asegurar_seguras([s for s in S if s not in ("tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min")])

TARGETS = {"m6_sn_dep": "m6_ln_sn_dep", "d_lnirf": "v5_d_lnirf", "dT": "d_temperatura_horno_celsius"}
seed = 42
for nombre, col in TARGETS.items():
    ok = F[col].notna().to_numpy()
    yhat = np.full(len(F), np.nan)
    idx_dev = np.where(F["es_dev"].to_numpy() & ok)[0]
    for tr, te in GroupKFold(5).split(idx_dev, groups=F["Batch"].to_numpy()[idx_dev]):
        m = L8.hgb_nuisance(seed).fit(F[S].iloc[idx_dev[tr]], F[col].iloc[idx_dev[tr]])
        yhat[idx_dev[te]] = m.predict(F[S].iloc[idx_dev[te]])
    F[f"yhat__{nombre}"] = yhat
    F[f"yres__{nombre}"] = F[col] - yhat

key = ["Batch", "fase_proceso", "orden_escalon_fase"]
Fj = F.merge(res[key + ["z__exo2", "z__o2", "z__carbon", "z__gn"]], on=key, how="left")
Fdev = Fj[Fj.es_dev]

print("=== R2 OOF de E[target|estado] (DEV) ===")
from sklearn.metrics import r2_score
for nombre, col in TARGETS.items():
    d = Fdev.dropna(subset=[col, f"yhat__{nombre}"])
    print(nombre, col, "R2 =", round(r2_score(d[col], d[f"yhat__{nombre}"]), 3), "n =", len(d))

print()
print("=== PLM residual-sobre-residual: yres__target ~ z__exo2 / z__o2 / z__carbon (+ dummies de orden) ===")
filas = []
for nombre in TARGETS:
    for expo in ["z__exo2", "z__o2", "z__carbon"]:
        d = Fdev.dropna(subset=[f"yres__{nombre}", expo, "orden_escalon_fase"]).copy()
        dum = pd.get_dummies(d["orden_escalon_fase"].astype(int), prefix="ord", drop_first=True).astype(float)
        X = pd.concat([d[[expo]], dum], axis=1)
        f = L.ols_hc3(d[f"yres__{nombre}"], X)
        filas.append(dict(target=nombre, expo=expo, n=len(d), beta=f.params[expo], p=f.pvalues[expo]))
mec = pd.DataFrame(filas)
mec.to_csv(OUT / "e11_05_mecanismo_escalon.csv", index=False)
print(mec.round(4).to_string())

# ---------- exo2_F (batch) predice el estado de entrada a Reduccion ----------
R0 = base[(base["fase_proceso"] == "Reducción") & (base["orden_escalon_fase"] == 0)][
    ["Batch", "temperatura_horno_celsius_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "m6_feo_inv_kg_prev"]
].drop_duplicates("Batch").set_index("Batch")
R0.columns = ["T_entrada_R", "sn_entrada_R", "feo_entrada_R", "feo_inv_entrada_R"]
t = L.tabla_batch(res=res).join(R0)
print()
print("=== x_F_exo2 (batch) -> estado de entrada a Reduccion (OLS HC3 + controles) ===")
filas2 = []
for y in ["T_entrada_R", "sn_entrada_R", "feo_entrada_R", "feo_inv_entrada_R"]:
    for nombre, d in [("DEV", t[t.es_dev]), ("total", t)]:
        dd = d.dropna(subset=["x_F_exo2", "x_F_carbon", y] + L.CONTROLES)
        f = L.ols_hc3(dd[y], dd[["x_F_exo2", "x_F_carbon"] + L.CONTROLES])
        filas2.append(dict(y=y, regimen=nombre, n=len(dd), beta_exo2=f.params["x_F_exo2"], p_exo2=f.pvalues["x_F_exo2"],
                            beta_carbon=f.params["x_F_carbon"], p_carbon=f.pvalues["x_F_carbon"]))
est = pd.DataFrame(filas2)
est.to_csv(OUT / "e11_05_estado_entrada_reduccion.csv", index=False)
print(est.round(4).to_string())
