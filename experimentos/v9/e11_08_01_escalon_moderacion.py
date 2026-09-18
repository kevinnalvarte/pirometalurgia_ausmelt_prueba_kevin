"""E11-08 tareas 1-3: moderacion termica del ESCALON de Reduccion (n ~ 1300-1450, ordenes 0-3).
- Tarea 1: theta por terciles/cuartiles de T_prev (global) para Cx_sn_v6 / carbon crudo / GN sobre Sn, y GN sobre FeO.
  Interaccion continua A_res x zT.
- Tarea 2: separa T de tiempo de campana: A_res x zT + A_res x z(idx) en el mismo modelo; y moderacion por T DENTRO
  de tercios cronologicos.
- Tarea 3: proxies de polvo/arrastre en linea (tiro, gases, presion de punta) -> A_res(GN) x zT.

Salidas: e11_08_tabla1_terciles.csv, e11_08_tabla1_cuartiles.csv, e11_08_tabla1_continua.csv,
e11_08_tabla2_idx_vs_T.csv, e11_08_tabla2_dentro_tercios.csv, e11_08_tabla3_proxies.csv.
"""
import sys
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd
from scipy import stats

import v9_lib as L
import v8_lib as L8
import modelo_predictivo_v8 as mp8

SEED = 42
N_BOOT = 500
rng_global = np.random.default_rng(SEED)

df = L8.cargar_df()
base = mp8._preparar_base(df)
b = L8.construir_batch(df)  # index Batch, tiene idx_cronologico
dev_b, lb_b = L8.mp.split_dev_lockbox(df)

r = base[base["fase_proceso"] == "Reducción"].copy()
r["grupo"] = [L.grupo_de("Reducción", int(o)) for o in r["orden_escalon_fase"]]
r = r[r["grupo"].notna()].reset_index(drop=True)
r["idx_cronologico"] = r["Batch"].map(b["idx_cronologico"])

S = list(dict.fromkeys(L.ESTADO_R))
T_COL = "temperatura_horno_celsius_prev"

SPECS = {
    "sn_dep__Cx_sn": dict(target="m6_ln_sn_dep",
                          A=["Cx_sn_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                          interes=["Cx_sn_v6", "tasa_gn_nm3_min"]),
    "sn_dep__C_crudo": dict(target="m6_ln_sn_dep",
                            A=["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                            interes=["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min"]),
    "feo_ret__GN": dict(target="m8_ln_feo_ret_dross50",
                        A=["m8_exceso_C_pos", "Cx_av_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                        interes=["tasa_gn_nm3_min"]),
}


def preparar(nombre_muestra, spec):
    cols = list(dict.fromkeys(S + spec["A"] + [spec["target"], T_COL, "idx_cronologico", "Batch"]))
    d = r.dropna(subset=cols).reset_index(drop=True)
    if nombre_muestra == "DEV":
        d = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
    return d


def ajustar_residuos(d, spec):
    L8.asegurar_seguras(S + spec["A"])
    t_tab, ex = L8.theta_dual(d, S, spec["A"], spec["target"], signos=None, n_boot=200, seed=SEED)
    A = spec["A"]
    dd = pd.DataFrame({"y_res": ex["y_res"], "Batch": d["Batch"].to_numpy(), "T_prev": d[T_COL].to_numpy(),
                       "idx": d["idx_cronologico"].to_numpy()})
    for j, a in enumerate(A):
        dd[f"ares__{a}"] = ex["A_res"][:, j]
    return dd


def hc3_fit(y, X):
    return L.ols_hc3(y, X)


def boot_cluster(dd, ycol, Xcols, batches_col="Batch", n_boot=N_BOOT, seed=SEED):
    """Bootstrap por batch (cluster). Devuelve matriz (n_boot, len(Xcols)) de coeficientes OLS con intercepto."""
    grupos = dd[batches_col].to_numpy()
    uniq = np.unique(grupos)
    idx_by = {u: np.where(grupos == u)[0] for u in uniq}
    y = dd[ycol].to_numpy(float)
    X = np.column_stack([np.ones(len(dd))] + [dd[c].to_numpy(float) for c in Xcols])
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, len(Xcols)))
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        out[i] = beta[1:]
    return out


def modelo_regimenes(dd, palanca, controles, col_regimen, devolver_boots=False):
    """y_res ~ 0 + ap:D_r (r en regimenes de col_regimen) + controles (main effect)."""
    regs = sorted(dd[col_regimen].dropna().unique())
    Xcols = []
    for reg in regs:
        c = f"__int_{reg}"
        dd[c] = dd[f"ares__{palanca}"] * (dd[col_regimen] == reg)
        Xcols.append(c)
    Xcols_ctrl = [f"ares__{c}" for c in controles]
    fit = hc3_fit(dd["y_res"], dd[Xcols + Xcols_ctrl])
    boots = boot_cluster(dd, "y_res", Xcols + Xcols_ctrl)
    filas = []
    for i, reg in enumerate(regs):
        n_reg = int((dd[col_regimen] == reg).sum())
        filas.append({"regimen": reg, "n": n_reg, "T_media": float(dd.loc[dd[col_regimen] == reg, "T_prev"].mean()),
                     "theta": float(fit.params[Xcols[i]]), "se_hc3": float(fit.bse[Xcols[i]]), "p_hc3": float(fit.pvalues[Xcols[i]]),
                     "ci_lo_boot": float(np.percentile(boots[:, i], 2.5)), "ci_hi_boot": float(np.percentile(boots[:, i], 97.5))})
    for c in ["__int_" + str(r) for r in regs]:
        del dd[c]
    tabla = pd.DataFrame(filas)
    if devolver_boots:
        return tabla, regs, boots[:, :len(regs)]
    return tabla


def modelo_continuo(dd, palanca, controles, moderadores):
    """y_res ~ ap + sum_m ap:zm + controles (main). moderadores: lista de columnas z ya en dd."""
    dd["__ap"] = dd[f"ares__{palanca}"]
    Xcols = ["__ap"]
    for m in moderadores:
        c = f"__ap_x_{m}"
        dd[c] = dd["__ap"] * dd[m]
        Xcols.append(c)
    Xcols_ctrl = [f"ares__{c}" for c in controles]
    fit = hc3_fit(dd["y_res"], dd[Xcols + Xcols_ctrl])
    boots = boot_cluster(dd, "y_res", Xcols + Xcols_ctrl)
    filas = []
    for i, c in enumerate(Xcols):
        filas.append({"termino": c, "coef": float(fit.params[c]), "se_hc3": float(fit.bse[c]), "t": float(fit.tvalues[c]),
                     "p_hc3": float(fit.pvalues[c]), "ci_lo_boot": float(np.percentile(boots[:, i], 2.5)),
                     "ci_hi_boot": float(np.percentile(boots[:, i], 97.5))})
    dd.drop(columns=Xcols, inplace=True)
    return pd.DataFrame(filas)


tab1_terc, tab1_cuart, tab1_cont, tab2_idxT, tab2_dentro = [], [], [], [], []

for muestra in ("TOTAL", "DEV"):
    for nombre_spec, spec in SPECS.items():
        d = preparar(muestra, spec)
        dd = ajustar_residuos(d, spec)
        dd["zT"] = (dd["T_prev"] - dd["T_prev"].mean()) / dd["T_prev"].std()
        dd["zidx"] = (dd["idx"] - dd["idx"].mean()) / dd["idx"].std()
        dd["terc_T"] = pd.qcut(dd["T_prev"], 3, labels=["frio", "medio", "caliente"])
        dd["cuart_T"] = pd.qcut(dd["T_prev"], 4, labels=["q1_frio", "q2", "q3", "q4_caliente"])
        for p in spec["interes"]:
            controles = [a for a in spec["A"] if a != p]
            # --- Tarea 1: terciles / cuartiles / continuo ---
            rt = modelo_regimenes(dd, p, controles, "terc_T"); rt["muestra"] = muestra; rt["spec"] = nombre_spec; rt["palanca"] = p
            tab1_terc.append(rt)
            rc = modelo_regimenes(dd, p, controles, "cuart_T"); rc["muestra"] = muestra; rc["spec"] = nombre_spec; rc["palanca"] = p
            tab1_cuart.append(rc)
            rco = modelo_continuo(dd, p, controles, ["zT"]); rco["muestra"] = muestra; rco["spec"] = nombre_spec; rco["palanca"] = p
            tab1_cont.append(rco)
            # --- Tarea 2a: T vs idx en el mismo modelo ---
            r2 = modelo_continuo(dd, p, controles, ["zT", "zidx"]); r2["muestra"] = muestra; r2["spec"] = nombre_spec; r2["palanca"] = p
            tab2_idxT.append(r2)
            # --- Tarea 2b: moderacion por T DENTRO de tercios cronologicos ---
            dd["terc_idx"] = pd.qcut(dd["idx"], 3, labels=["temprano", "medio_camp", "tardio"])
            for terc in ["temprano", "medio_camp", "tardio"]:
                sub = dd[dd["terc_idx"] == terc].copy()
                if len(sub) < 60:
                    continue
                sub["zT_local"] = (sub["T_prev"] - sub["T_prev"].mean()) / sub["T_prev"].std()
                rr = modelo_continuo(sub, p, controles, ["zT_local"])
                rr["muestra"] = muestra; rr["spec"] = nombre_spec; rr["palanca"] = p; rr["tercio_campana"] = terc
                rr["n"] = len(sub); rr["T_media_tercio"] = float(sub["T_prev"].mean())
                tab2_dentro.append(rr)

tab1_terc = pd.concat(tab1_terc, ignore_index=True)
tab1_cuart = pd.concat(tab1_cuart, ignore_index=True)
tab1_cont = pd.concat(tab1_cont, ignore_index=True)
tab2_idxT = pd.concat(tab2_idxT, ignore_index=True)
tab2_dentro = pd.concat(tab2_dentro, ignore_index=True)

tab1_terc.to_csv(L.SALIDA / "e11_08_tabla1_terciles.csv", index=False)
tab1_cuart.to_csv(L.SALIDA / "e11_08_tabla1_cuartiles.csv", index=False)
tab1_cont.to_csv(L.SALIDA / "e11_08_tabla1_continua.csv", index=False)
tab2_idxT.to_csv(L.SALIDA / "e11_08_tabla2_idx_vs_T.csv", index=False)
tab2_dentro.to_csv(L.SALIDA / "e11_08_tabla2_dentro_tercios.csv", index=False)

pd.set_option("display.width", 220)
print("=== Tabla 1: terciles de T (global) ===")
print(tab1_terc.round(3).to_string())
print("\n=== Tabla 1: interaccion continua A_res x zT ===")
print(tab1_cont[tab1_cont.termino.str.contains("zT")].round(4).to_string())
print("\n=== Tabla 2: A_res x zT + A_res x zidx (mismo modelo) ===")
print(tab2_idxT.round(4).to_string())
print("\n=== Tabla 2: moderacion por T DENTRO de tercios cronologicos ===")
print(tab2_dentro[tab2_dentro.termino.str.contains("zT_local")].round(4).to_string())

# ------------------------------------------------------------------
# Tarea 3: proxies de polvo/arrastre en linea (tiro, gases, presion)
# ------------------------------------------------------------------
PROXIES = ["tiro_horno_pct", "temperatura_gas_pre_bhf_celsius", "temperatura_gas_pre_cuchilla_celsius",
          "delta_T_gas_pre", "presion_punta_lanza_kpa", "d_tiro_horno_pct", "d_presion_punta_lanza_kpa"]
A_PROXY = ["tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"]

tab3 = []
for muestra in ("TOTAL", "DEV"):
    for proxy in PROXIES:
        if proxy not in r.columns:
            tab3.append({"proxy": proxy, "muestra": muestra, "estado": "MISSING"}); continue
        cols = list(dict.fromkeys(S + A_PROXY + [proxy, T_COL, "idx_cronologico", "Batch"]))
        d = r.dropna(subset=cols).reset_index(drop=True)
        if muestra == "DEV":
            d = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
        if len(d) < 100:
            tab3.append({"proxy": proxy, "muestra": muestra, "estado": f"n_insuf={len(d)}"}); continue
        L8.asegurar_seguras(S + A_PROXY)
        t_tab, ex = L8.theta_dual(d, S, A_PROXY, proxy, signos=None, n_boot=200, seed=SEED)
        dd = pd.DataFrame({"y_res": ex["y_res"], "Batch": d["Batch"].to_numpy(), "T_prev": d[T_COL].to_numpy()})
        for j, a in enumerate(A_PROXY):
            dd[f"ares__{a}"] = ex["A_res"][:, j]
        dd["zT"] = (dd["T_prev"] - dd["T_prev"].mean()) / dd["T_prev"].std()
        controles = [a for a in A_PROXY if a != "tasa_gn_nm3_min"]
        rr = modelo_continuo(dd, "tasa_gn_nm3_min", controles, ["zT"])
        rr = rr[rr.termino.isin(["__ap", "__ap_x_zT"])].copy()
        rr["proxy"] = proxy; rr["muestra"] = muestra; rr["n"] = len(d)
        rr["r2_oof_proxy"] = float(t_tab["r2_oof_plm_interno"].iloc[0]) if "r2_oof_plm_interno" in t_tab else np.nan
        tab3.append(rr)

tab3_rows = [x for x in tab3 if isinstance(x, pd.DataFrame)]
tab3_skip = [x for x in tab3 if isinstance(x, dict)]
tab3_df = pd.concat(tab3_rows, ignore_index=True) if tab3_rows else pd.DataFrame()
if tab3_skip:
    tab3_df = pd.concat([tab3_df, pd.DataFrame(tab3_skip)], ignore_index=True)
tab3_df.to_csv(L.SALIDA / "e11_08_tabla3_proxies.csv", index=False)
print("\n=== Tabla 3: proxies de polvo/arrastre, GN_res x zT ===")
print(tab3_df.round(4).to_string())

# ------------------------------------------------------------------
# Diferencia frio-caliente (bootstrap) para los pares clave: GN->FeO, GN->Sn
# ------------------------------------------------------------------
filas_dif = []
for muestra in ("TOTAL", "DEV"):
    for nombre_spec, p in (("feo_ret__GN", "tasa_gn_nm3_min"), ("sn_dep__Cx_sn", "tasa_gn_nm3_min"), ("sn_dep__Cx_sn", "Cx_sn_v6")):
        spec = SPECS[nombre_spec]
        d = preparar(muestra, spec)
        dd = ajustar_residuos(d, spec)
        dd["terc_T"] = pd.qcut(dd["T_prev"], 3, labels=["frio", "medio", "caliente"])
        controles = [a for a in spec["A"] if a != p]
        _, regs, boots = modelo_regimenes(dd, p, controles, "terc_T", devolver_boots=True)
        i_frio, i_cal = regs.index("frio"), regs.index("caliente")
        dif = boots[:, i_frio] - boots[:, i_cal]
        filas_dif.append({"spec": nombre_spec, "palanca": p, "muestra": muestra,
                          "theta_frio_menos_caliente": float(np.mean(dif)),
                          "ci_lo": float(np.percentile(dif, 2.5)), "ci_hi": float(np.percentile(dif, 97.5)),
                          "p_boot_dosCaras": float(2 * min((dif > 0).mean(), (dif < 0).mean()))})
tab_dif = pd.DataFrame(filas_dif)
tab_dif.to_csv(L.SALIDA / "e11_08_tabla1_diferencia_regimenes.csv", index=False)
print("\n=== Diferencia bootstrap theta(frio) - theta(caliente) ===")
print(tab_dif.round(4).to_string())

print("\nOK e11_08_01")
