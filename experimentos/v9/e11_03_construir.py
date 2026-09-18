"""E11-03 (H3) -- paso 1-2: theta cross-fitted por fold externo de batches (sin fuga) + D_Sn/D_FeO por batch
(version LOG y KG, variante 'identificadas' y 'todas'). Guarda experimentos/v9/e11_03_tabla_batch.csv/.pkl y
experimentos/v9/e11_03_theta_full_dev.csv (theta de referencia ajustado en todo DEV, para lockbox y para
diagnostico). .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3, primer plano, sin joblib.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v9", RAIZ / "experimentos" / "v8"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v9_lib as L          # noqa: E402
import v8_lib as L8         # noqa: E402
import modelo_predictivo_v8 as mp8  # noqa: E402

SALIDA = RAIZ / "experimentos" / "v9"
CACHE = RAIZ / "experimentos" / "cache" / "e11_03_residuos.pkl"
SEED = 42
N_OUTER, N_INNER, N_BOOT = 5, 5, 300

TARGET_SN, TARGET_FEO = "m6_ln_sn_dep", "m8_ln_feo_ret_dross50"
A_SN = ["Cx_sn_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
A_FEO = ["m8_exceso_C_pos", "Cx_av_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
SIGNOS_SN = {a: mp8.SIGNO_TEORICO_V8[("Reducción", "sn_dep")].get(a, 0) for a in A_SN}
SIGNOS_FEO = {a: mp8.SIGNO_TEORICO_V8[("Reducción", "feo_ret")].get(a, 0) for a in A_FEO}
IDENT_SN, IDENT_FEO = ["Cx_sn_v6", "tasa_gn_nm3_min"], ["tasa_gn_nm3_min"]
S_R = list(mp8.ESTADO_R_V8)


def preparar_d() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = L8.cargar_df()
    base = mp8._preparar_base(df)
    d = base[base["fase_proceso"] == "Reducción"].copy()
    dev_b, _ = L8.mp.split_dev_lockbox(df)
    d["es_dev"] = d["Batch"].isin(dev_b)
    need = list(dict.fromkeys(S_R + A_SN + A_FEO + [TARGET_SN, TARGET_FEO, "m6_sn_inv_kg", "m6_feo_inv_kg", "Batch",
                                                     "orden_escalon_fase", "fecha_inicio"]))
    d = d.dropna(subset=need).reset_index(drop=True)
    return d[d["es_dev"]].reset_index(drop=True), d[~d["es_dev"]].reset_index(drop=True)


def crossfit(d_dev: pd.DataFrame, d_lb: pd.DataFrame, S: list[str], A: list[str], target: str, signos: dict,
             seed: int = SEED, n_outer: int = N_OUTER, n_inner: int = N_INNER, n_boot: int = N_BOOT):
    """theta LIBRE cross-fitted por fold externo de Batch (theta que anota un batch nunca lo incluye en su ajuste).
    Devuelve (a_res, y_res, theta_fila) para d_dev + d_lb concatenados (mismo orden), y theta_full (ajustado en
    todo DEV, usado para anotar lockbox) con su tabla completa (IC libre/acotado, identificado)."""
    n_dev = len(d_dev)
    a_res = np.full((n_dev + len(d_lb), len(A)), np.nan)
    y_res = np.full(n_dev + len(d_lb), np.nan)
    theta_fila = np.full((n_dev + len(d_lb), len(A)), np.nan)
    groups = d_dev["Batch"].to_numpy()
    fold_id = np.full(n_dev, -1)
    for k, (tr, te) in enumerate(GroupKFold(n_outer).split(d_dev[S], d_dev[target], groups)):
        t0 = time.time()
        _, extras = L8.theta_dual(d_dev.iloc[tr].reset_index(drop=True), S, A, target, signos,
                                  n_splits=n_inner, n_boot=n_boot, seed=seed)
        th = extras["theta_libre"]
        for j, c in enumerate([target] + A):
            mdl = L8.hgb_nuisance(seed).fit(d_dev[S].iloc[tr], d_dev[c].iloc[tr])
            pred = mdl.predict(d_dev[S].iloc[te])
            if c == target:
                y_res[te] = d_dev[target].iloc[te].to_numpy() - pred
            else:
                a_res[te, j - 1] = d_dev[c].iloc[te].to_numpy() - pred
        theta_fila[te, :] = th
        fold_id[te] = k
        print(f"    [{target}] fold externo {k}: n_tr={len(tr)} n_te={len(te)} theta_libre={th.round(4)} ({time.time()-t0:.0f}s)", flush=True)
    # theta ajustado en TODO DEV (para anotar lockbox y como referencia)
    tabla_full, extras_full = L8.theta_dual(d_dev, S, A, target, signos, n_splits=n_inner, n_boot=n_boot, seed=seed)
    theta_full = extras_full["theta_libre"]
    if len(d_lb):
        for j, c in enumerate([target] + A):
            mdl = L8.hgb_nuisance(seed).fit(d_dev[S], d_dev[c])
            pred = mdl.predict(d_lb[S])
            if c == target:
                y_res[n_dev:] = d_lb[target].to_numpy() - pred
            else:
                a_res[n_dev:, j - 1] = d_lb[c].to_numpy() - pred
        theta_fila[n_dev:, :] = theta_full
    return a_res, y_res, theta_fila, theta_full, tabla_full, fold_id


def main():
    t0 = time.time()
    d_dev, d_lb = preparar_d()
    print(f"d_dev={len(d_dev)} d_lb={len(d_lb)}", flush=True)
    d = pd.concat([d_dev, d_lb], ignore_index=True)

    print("== theta Sn (agotamiento) ==", flush=True)
    a_res_sn, y_res_sn, th_fila_sn, th_full_sn, tabla_full_sn, fold_sn = crossfit(d_dev, d_lb, S_R, A_SN, TARGET_SN, SIGNOS_SN)
    print("== theta FeO (retencion neta) ==", flush=True)
    a_res_feo, y_res_feo, th_fila_feo, th_full_feo, tabla_full_feo, fold_feo = crossfit(d_dev, d_lb, S_R, A_FEO, TARGET_FEO, SIGNOS_FEO)

    idx_sn = {a: j for j, a in enumerate(A_SN)}
    idx_feo = {a: j for j, a in enumerate(A_FEO)}

    def suma_theta(theta_fila, idx, a_res, cols):
        out = np.zeros(len(d))
        for c in cols:
            out += theta_fila[:, idx[c]] * a_res[:, idx[c]]
        return out

    d["d_sn_ident"] = suma_theta(th_fila_sn, idx_sn, a_res_sn, IDENT_SN)
    d["d_sn_all"] = suma_theta(th_fila_sn, idx_sn, a_res_sn, A_SN)
    d["d_feo_ident"] = suma_theta(th_fila_feo, idx_feo, a_res_feo, IDENT_FEO)
    d["d_feo_all"] = suma_theta(th_fila_feo, idx_feo, a_res_feo, A_FEO)
    # paso 7: componentes aislados por palanca (para el valor marginal de GN/carbon por escalon)
    d["shift_sn_from_GN"] = th_fila_sn[:, idx_sn["tasa_gn_nm3_min"]] * a_res_sn[:, idx_sn["tasa_gn_nm3_min"]]
    d["shift_sn_from_CxSn"] = th_fila_sn[:, idx_sn["Cx_sn_v6"]] * a_res_sn[:, idx_sn["Cx_sn_v6"]]
    d["shift_feo_from_GN"] = th_fila_feo[:, idx_feo["tasa_gn_nm3_min"]] * a_res_feo[:, idx_feo["tasa_gn_nm3_min"]]
    d["a_res_gn_sn"] = a_res_sn[:, idx_sn["tasa_gn_nm3_min"]]
    d["theta_gn_sn"] = th_fila_sn[:, idx_sn["tasa_gn_nm3_min"]]
    d["theta_gn_feo"] = th_fila_feo[:, idx_feo["tasa_gn_nm3_min"]]
    d["theta_cxsn_sn"] = th_fila_sn[:, idx_sn["Cx_sn_v6"]]

    # version KG: Delta Sn_kg = Sn_inv_final * (1 - e^-d_sn) ; Delta FeO_kg = FeO_inv_final * (1 - e^d_feo)
    for suf in ("ident", "all"):
        d[f"dsn_kg_{suf}"] = d["m6_sn_inv_kg"] * (1.0 - np.exp(-d[f"d_sn_{suf}"]))
        d[f"dfeo_kg_{suf}"] = d["m6_feo_inv_kg"] * (1.0 - np.exp(d[f"d_feo_{suf}"]))

    # exposiciones crudas para sobre-identificacion (paso 5): media de z sobre R0-R3 (Reduccion completa)
    res = L.residuos_escalon()
    resR = res[(res["fase_proceso"] == "Reducción") & res["grupo"].notna()]
    x_gn = resR.groupby("Batch")["z__gn"].mean()
    x_carbon = resR.groupby("Batch")["z__carbon"].mean()

    agg = d.groupby("Batch").agg(
        agg_sn=(TARGET_SN, "sum"), agg_feo=(TARGET_FEO, "sum"),
        D_sn_log_ident=("d_sn_ident", "sum"), D_sn_log_all=("d_sn_all", "sum"),
        D_feo_log_ident=("d_feo_ident", "sum"), D_feo_log_all=("d_feo_all", "sum"),
        D_sn_kg_ident=("dsn_kg_ident", "sum"), D_sn_kg_all=("dsn_kg_all", "sum"),
        D_feo_kg_ident=("dfeo_kg_ident", "sum"), D_feo_kg_all=("dfeo_kg_all", "sum"),
    )
    agg["x_R_gn"] = x_gn
    agg["x_R_carbon"] = x_carbon

    b = L8.construir_batch(mp8.cargar_df())
    tabla = b.join(agg, how="left")
    tabla.to_csv(SALIDA / "e11_03_tabla_batch.csv")
    tabla.to_pickle(SALIDA / "e11_03_tabla_batch.pkl")

    for nombre, tab in (("sn", tabla_full_sn), ("feo", tabla_full_feo)):
        tab.to_csv(SALIDA / f"e11_03_theta_full_dev_{nombre}.csv", index=False)

    cols_escalon = ["Batch", "orden_escalon_fase", "es_dev", "m6_sn_inv_kg", "m6_feo_inv_kg", "m6_sn_inv_kg_prev",
                    "m6_feo_inv_kg_prev", "d_sn_ident", "d_sn_all", "d_feo_ident", "d_feo_all", "dsn_kg_ident",
                    "dsn_kg_all", "dfeo_kg_ident", "dfeo_kg_all", "shift_sn_from_GN", "shift_sn_from_CxSn",
                    "shift_feo_from_GN", "a_res_gn_sn", "theta_gn_sn", "theta_gn_feo", "theta_cxsn_sn"]
    d[cols_escalon].to_pickle(SALIDA / "e11_03_escalon.pkl")

    print(f"listo en {time.time()-t0:.0f}s -> {SALIDA/'e11_03_tabla_batch.csv'}", flush=True)
    print(tabla[[c for c in tabla.columns if c.startswith(("D_", "agg_", "x_R"))]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
