"""E12-01 tarea 2: mecanismo de escalon en R3. Residuo-sobre-residuo (estado L.mp8.ESTADO_R_V8 = ESTADO_R_SIN_ESPESOR,
HGB cross-fitted por batch) de m6_ln_sn_dep, m8_ln_feo_ret_dross50, d_temperatura_horno_celsius y de los deltas en kg
(Sn extraido, FeO reducido) sobre el residuo de la duracion de R3 (res_dur, zD_R3) y sobre duracion x tasas de C/GN.
Dosis-respuesta por quintiles de zD_R3. Selectividad marginal del minuto extra (kg Sn / kg FeO) y su dependencia del
%Sn de la escoria al inicio de R3.
"""
import sys
sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v8")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy import stats
import v9_lib as L, v8_lib as L8
import modelo_predictivo_v8 as mp8

SALIDA = L.RAIZ / "experimentos" / "v10"
SEED = 42
ESTADO = list(mp8.ESTADO_R_V8)

df = L8.cargar_df()
base = mp8._preparar_base(df)
dev, lockbox = L8.mp.split_dev_lockbox(base)

r3 = base[(base["fase_proceso"] == "Reducción") & (base["orden_escalon_fase"] == 3)].copy().reset_index(drop=True)
r3["dur_min"] = (pd.to_datetime(r3["fecha_final"]) - pd.to_datetime(r3["fecha_inicio"])).dt.total_seconds() / 60.0
r3["d_sn_kg"] = r3["m6_sn_inv_kg_prev"] - r3["m6_sn_inv_kg"]         # Sn extraido de la escoria en R3 (kg)
r3["d_feo_kg"] = r3["m6_feo_inv_kg_prev"] - r3["m6_feo_inv_kg"]      # FeO reducido en R3 (kg)
r3["es_dev"] = r3["Batch"].isin(dev)

# residuo de la duracion (reutiliza la misma receta que e12_01_construir, para no depender del csv)
es_dev = r3["es_dev"].to_numpy()
ok = r3[ESTADO].notna().all(axis=1).to_numpy() & r3["dur_min"].notna().to_numpy()
dur_hat = np.full(len(r3), np.nan)
idx_dev = np.where(es_dev & ok)[0]
for tr, te in GroupKFold(5).split(idx_dev, groups=r3["Batch"].to_numpy()[idx_dev]):
    m = L8.hgb_nuisance(SEED).fit(r3[ESTADO].iloc[idx_dev[tr]], r3["dur_min"].iloc[idx_dev[tr]])
    dur_hat[idx_dev[te]] = m.predict(r3[ESTADO].iloc[idx_dev[te]])
idx_lb = np.where(~es_dev & ok)[0]
if len(idx_lb):
    m = L8.hgb_nuisance(SEED).fit(r3[ESTADO].iloc[idx_dev], r3["dur_min"].iloc[idx_dev])
    dur_hat[idx_lb] = m.predict(r3[ESTADO].iloc[idx_lb])
r3["res_dur"] = r3["dur_min"] - dur_hat
sd_dur = float(r3.loc[es_dev, "res_dur"].std())
r3["zD_R3"] = r3["res_dur"] / sd_dur


def residuo_oof(d: pd.DataFrame, target: str, estado: list[str], es_dev: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, float, float]:
    ok = d[estado].notna().all(axis=1).to_numpy() & d[target].notna().to_numpy()
    yhat = np.full(len(d), np.nan)
    idx_dev = np.where(es_dev & ok)[0]
    for tr, te in GroupKFold(5).split(idx_dev, groups=d["Batch"].to_numpy()[idx_dev]):
        m = L8.hgb_nuisance(seed).fit(d[estado].iloc[idx_dev[tr]], d[target].iloc[idx_dev[tr]])
        yhat[idx_dev[te]] = m.predict(d[estado].iloc[idx_dev[te]])
    idx_lb = np.where(~es_dev & ok)[0]
    r2_lb = np.nan
    if len(idx_lb):
        m = L8.hgb_nuisance(seed).fit(d[estado].iloc[idx_dev], d[target].iloc[idx_dev])
        yhat[idx_lb] = m.predict(d[estado].iloc[idx_lb])
        r2_lb = r2_score(d[target].iloc[idx_lb], yhat[idx_lb])
    r2_dev = r2_score(d[target].iloc[idx_dev], yhat[idx_dev])
    return d[target].to_numpy() - yhat, r2_dev, r2_lb


targets = ["m6_ln_sn_dep", "m8_ln_feo_ret_dross50", "d_temperatura_horno_celsius", "d_sn_kg", "d_feo_kg"]
r2s = []
for tgt in targets:
    res, r2d, r2l = residuo_oof(r3, tgt, ESTADO, es_dev)
    r3[f"resid__{tgt}"] = res
    r2s.append({"target": tgt, "r2_oof_dev": r2d, "r2_lockbox": r2l})
r2tab = pd.DataFrame(r2s)
print("--- R2 OOF de los targets ~ ESTADO_R_V8 (para contexto del residuo) ---")
print(r2tab.round(3).to_string())
r2tab.to_csv(SALIDA / "e12_01_escalon_r2.csv", index=False)

# --- dosis extra por duracion x tasas (kg de carbon / Nm3 de GN "extra" atribuibles al residuo de duracion)
r3["extra_C_kg"] = r3["res_dur"] * r3["tasa_feed_Carbon_kg_min"]
r3["extra_GN_nm3"] = r3["res_dur"] * r3["tasa_gn_nm3_min"]
for c in ("extra_C_kg", "extra_GN_nm3"):
    mu, sd = r3.loc[es_dev, c].mean(), r3.loc[es_dev, c].std()
    r3[f"{c}_z"] = (r3[c] - mu) / sd


def ols_reporte(d: pd.DataFrame, y: str, X: list[str]) -> pd.DataFrame:
    dd = d.dropna(subset=[y] + X)
    fit = L.ols_hc3(dd[y], dd[X])
    out = pd.DataFrame({"coef": fit.params, "se": fit.bse, "t": fit.tvalues, "p": fit.pvalues})
    out["n"] = len(dd)
    return out.loc[["const"] + X]


print("\n--- residuo del target ~ zD_R3 + extra_C_kg_z + extra_GN_nm3_z (DEV, HC3) ---")
filas_mec = []
for tgt in ("m6_ln_sn_dep", "m8_ln_feo_ret_dross50", "d_temperatura_horno_celsius"):
    tab = ols_reporte(r3[es_dev], f"resid__{tgt}", ["zD_R3", "extra_C_kg_z", "extra_GN_nm3_z"])
    tab["target"] = tgt
    print(tgt); print(tab.round(4).to_string())
    filas_mec.append(tab.reset_index().rename(columns={"index": "var"}))
mec = pd.concat(filas_mec, ignore_index=True)
mec.to_csv(SALIDA / "e12_01_escalon_mecanismo.csv", index=False)

# --- solo zD_R3 (simple) para comparar
print("\n--- residuo del target ~ zD_R3 (simple, DEV) ---")
filas_simple = []
for tgt in ("m6_ln_sn_dep", "m8_ln_feo_ret_dross50", "d_temperatura_horno_celsius"):
    tab = ols_reporte(r3[es_dev], f"resid__{tgt}", ["zD_R3"])
    tab["target"] = tgt
    print(tgt, dict(coef=round(tab.loc["zD_R3", "coef"], 4), t=round(tab.loc["zD_R3", "t"], 2), p=round(tab.loc["zD_R3", "p"], 4)))
    filas_simple.append(tab.reset_index().rename(columns={"index": "var"}))
pd.concat(filas_simple, ignore_index=True).to_csv(SALIDA / "e12_01_escalon_mecanismo_simple.csv", index=False)

# --- dosis-respuesta por quintiles de zD_R3 (DEV)
d_dev = r3[es_dev].copy()
d_dev["q"] = pd.qcut(d_dev["zD_R3"], 5, labels=False, duplicates="drop")
quint = d_dev.groupby("q").agg(n=("zD_R3", "size"), zD_R3_media=("zD_R3", "mean"), dur_media=("dur_min", "mean"),
                                resid_sn_dep=("resid__m6_ln_sn_dep", "mean"), resid_feo_ret=("resid__m8_ln_feo_ret_dross50", "mean"),
                                resid_dT=("resid__d_temperatura_horno_celsius", "mean"), resid_dSn_kg=("resid__d_sn_kg", "mean"),
                                resid_dFeO_kg=("resid__d_feo_kg", "mean"))
print("\n--- dosis-respuesta por quintiles de zD_R3 (DEV) ---")
print(quint.round(3).to_string())
quint.to_csv(SALIDA / "e12_01_escalon_quintiles.csv")

# --- selectividad marginal del minuto extra: kg Sn / kg FeO por minuto adicional (coef de resid_dX_kg ~ res_dur, kg/min)
sel = {}
for tgt, nombre in (("d_sn_kg", "Sn_kg_por_min"), ("d_feo_kg", "FeO_kg_por_min")):
    tab = ols_reporte(r3[es_dev], f"resid__{tgt}", ["res_dur"])
    sel[nombre] = {"coef": tab.loc["res_dur", "coef"], "se": tab.loc["res_dur", "se"], "t": tab.loc["res_dur", "t"], "p": tab.loc["res_dur", "p"]}
print("\n--- selectividad marginal del minuto extra (DEV) ---")
for k, v in sel.items():
    print(k, {kk: round(vv, 4) for kk, vv in v.items()})
ratio = sel["Sn_kg_por_min"]["coef"] / sel["FeO_kg_por_min"]["coef"] if sel["FeO_kg_por_min"]["coef"] != 0 else np.nan
print("ratio Sn_kg/FeO_kg por minuto extra =", round(ratio, 3))

# --- dependencia de la selectividad del %Sn de escoria al inicio de R3 (ley_sn_escoria_pct_prev)
r3["z_leySn"] = (r3["ley_sn_escoria_pct_prev"] - r3.loc[es_dev, "ley_sn_escoria_pct_prev"].mean()) / r3.loc[es_dev, "ley_sn_escoria_pct_prev"].std()
r3["res_dur_x_leySn"] = r3["res_dur"] * r3["z_leySn"]
filas_dep = []
for tgt in ("d_sn_kg", "d_feo_kg"):
    tab = ols_reporte(r3[es_dev], f"resid__{tgt}", ["res_dur", "res_dur_x_leySn"])
    tab["target"] = tgt
    print(f"\nresid_{tgt} ~ res_dur + res_dur:z_leySn (DEV)")
    print(tab.round(4).to_string())
    filas_dep.append(tab.reset_index().rename(columns={"index": "var"}))
pd.concat(filas_dep, ignore_index=True).to_csv(SALIDA / "e12_01_escalon_selectividad_leySn.csv", index=False)

# terciles de ley_sn_escoria_pct_prev: selectividad por tercil
d_dev["tercil_leySn"] = pd.qcut(d_dev["ley_sn_escoria_pct_prev"], 3, labels=["bajo", "medio", "alto"])
filas_terc = []
for terc, sub in d_dev.groupby("tercil_leySn", observed=True):
    row = {"tercil_leySn": terc, "n": len(sub), "leySn_media": sub["ley_sn_escoria_pct_prev"].mean()}
    for tgt, nombre in (("d_sn_kg", "Sn_kg_por_min"), ("d_feo_kg", "FeO_kg_por_min")):
        tab = ols_reporte(sub, f"resid__{tgt}", ["res_dur"])
        row[nombre] = tab.loc["res_dur", "coef"]
        row[f"{nombre}_t"] = tab.loc["res_dur", "t"]
    filas_terc.append(row)
terc_tab = pd.DataFrame(filas_terc)
terc_tab["ratio_Sn_FeO"] = terc_tab["Sn_kg_por_min"] / terc_tab["FeO_kg_por_min"]
print("\n--- selectividad por tercil de %Sn de escoria al inicio de R3 (DEV) ---")
print(terc_tab.round(3).to_string())
terc_tab.to_csv(SALIDA / "e12_01_escalon_selectividad_terciles.csv", index=False)

r3.to_pickle(SALIDA / "e12_01_r3_mecanismo.pkl")
