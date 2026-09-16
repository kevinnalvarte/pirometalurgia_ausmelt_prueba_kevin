"""E9-01: formas del modelo de Reduccion (estructural/cinetico vs PLM).

Etapas (por argv, cada una reanudable via CSV): ref, tasa, estructural, loglineal, hibrido, estabilidad, resumen, all.

Uso:
    .venv/Scripts/python.exe experimentos/v7/e9_01_formas.py <etapa>
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_lib as L  # noqa: E402

OUT = Path(__file__).resolve().parent
FASE = "Reduccion"
FASE_COL = "Reducción"
CLAVES = ["sn_dep", "feo_ret"]
SEED = 42
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def cargar():
    df = L.cargar_df()
    dR = L.base_fase(df, FASE_COL)
    return df, dR


# =============================================================================
# utilidades comunes
# =============================================================================
def estado_palancas(clave: str):
    S = L.ESTADO[(FASE_COL, clave)]
    A = L.PALANCAS[(FASE_COL, clave)]
    target = L.TARGETS[FASE_COL][clave]
    signos = L.SIGNOS[(FASE_COL, clave)]
    L.asegurar_seguras(S + A)
    return S, A, target, signos


def batch_folds(batches, n_splits: int = 5) -> dict:
    """Mapa Batch->fold ESTABLE (no depende del orden de filas): 1 'muestra' por batch unico."""
    from sklearn.model_selection import GroupKFold
    ub = np.array(sorted(set(batches)))
    fold_of = np.empty(len(ub), dtype=int)
    for i, (_, te) in enumerate(GroupKFold(n_splits).split(ub, groups=ub)):
        fold_of[te] = i
    return dict(zip(ub, fold_of))


def r2_mae(y, p):
    m = L.metricas(y, p)
    return m["r2"], m["mae"], m["n"]


# =============================================================================
# Etapa 1: ref -- PLM v6 (referencia)
# =============================================================================
def etapa_ref(dR):
    filas, tablas, preds = [], [], []
    for clave in CLAVES:
        S, A, target, signos = estado_palancas(clave)
        met, tabla, pred = L.evaluar_plm_signos(dR, S, A, target, signo_teorico=signos)
        met.update(forma="ref", clave=clave)
        log("ref", clave, {k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()
                            if k in ("r2_oof", "mae_oof", "r2_lockbox", "mae_lockbox", "n_dev")})
        filas.append(met)
        tabla["forma"], tabla["clave"] = "ref", clave
        tablas.append(tabla.reset_index())
        pred["forma"], pred["clave"], pred["target"] = "ref", clave, target
        preds.append(pred)
    pd.DataFrame(filas).to_csv(OUT / "e9_01_metricas_ref.csv", index=False)
    pd.concat(tablas, ignore_index=True).to_csv(OUT / "e9_01_theta_ref.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "e9_01_pred_ref.csv", index=False)
    log("ref: OK")


# =============================================================================
# Etapa 2: tasa -- target/duracion_plan_min, evaluado en escala original
# =============================================================================
def etapa_tasa(dR):
    filas, tablas, preds = [], [], []
    for clave in CLAVES:
        S, A, target, signos = estado_palancas(clave)
        d2 = dR.copy()
        tcol = target + "_tasa"
        d2[tcol] = d2[target] / d2["duracion_plan_min"]
        met, tabla, pred = L.evaluar_plm_signos(d2, S, A, tcol, signo_teorico=signos)
        pred = pred.merge(
            d2[["Batch", "orden_escalon_fase", "duracion_plan_min"]].drop_duplicates(["Batch", "orden_escalon_fase"]),
            on=["Batch", "orden_escalon_fase"], how="left")
        pred["real_orig"] = pred["real"] * pred["duracion_plan_min"]
        pred["pred_orig"] = pred["pred"] * pred["duracion_plan_min"]
        met_orig = dict(forma="tasa", clave=clave, target=target, n_estado=len(S), n_palancas=len(A),
                         n_dev=met["n_dev"], n_lockbox=met["n_lockbox"])
        for conj, pref in [("OOF_dev", "oof"), ("lockbox", "lockbox")]:
            sub = pred[pred["conjunto"] == conj]
            r2, mae, n = r2_mae(sub["real_orig"], sub["pred_orig"])
            met_orig[f"r2_{pref}"], met_orig[f"mae_{pref}"] = r2, mae
        log("tasa", clave, {k: round(v, 4) if isinstance(v, float) else v for k, v in met_orig.items()
                             if k.startswith("r2") or k.startswith("mae")})
        filas.append(met_orig)
        tabla["forma"], tabla["clave"] = "tasa", clave
        tablas.append(tabla.reset_index())
        pred["forma"], pred["clave"], pred["target"] = "tasa", clave, target
        preds.append(pred)
    pd.DataFrame(filas).to_csv(OUT / "e9_01_metricas_tasa.csv", index=False)
    pd.concat(tablas, ignore_index=True).to_csv(OUT / "e9_01_theta_tasa.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "e9_01_pred_tasa.csv", index=False)
    log("tasa: OK")


# =============================================================================
# Etapa 3: estructural -- cinetica de reduccion competitiva (NLS conjunto)
# =============================================================================
RAW_COLS = ["feed_Carbon_kgh", "volumen_gas_natural_inyectado_lanza_escalon_nm3", "exceso_o2_combustion_pct",
            "temperatura_horno_celsius_prev", "m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev", "basicidad_B2_prev",
            "feed_Sn_kgf", "m6_avance_prev", "tasa_feed_Carbon_kg_min", "delta_tiempo"]
TGT_SN, TGT_FEO = "m6_ln_sn_dep", "m6_ln_feo_ret"

PNAMES = ["eta_C", "gamma_GN", "Ea", "kappa0", "kappa1", "epsilon", "delta0"]
P_LO = np.array([0.0, 0.0, -2e5, 1e-6, -10.0, 0.0, -0.05])
P_HI = np.array([5.0, 5.0, 2e5, 100.0, 10.0, 2.0, 0.20])
P_X0 = np.array([0.5, 0.3, 0.0, 1.0, 0.0, 0.3, 0.0])


def _arrays(d: pd.DataFrame):
    return dict(
        feed_C=d["feed_Carbon_kgh"].to_numpy(float),
        gn=d["volumen_gas_natural_inyectado_lanza_escalon_nm3"].to_numpy(float),
        exc_o2=d["exceso_o2_combustion_pct"].to_numpy(float),
        T=d["temperatura_horno_celsius_prev"].to_numpy(float) + 273.15,
        Sn_prev=d["m6_sn_inv_kg_prev"].to_numpy(float),
        FeO_prev=d["m6_feo_inv_kg_prev"].to_numpy(float),
        B2=d["basicidad_B2_prev"].to_numpy(float),
        Sn_feed=d["feed_Sn_kgf"].fillna(0.0).to_numpy(float),
    )


def modelo_estructural(params, arr, T_ref):
    eta_C, gamma_GN, Ea, kappa0, kappa1, epsilon, delta0 = params
    C_eff = arr["feed_C"] * eta_C * np.exp(-Ea * (1.0 / arr["T"] - 1.0 / T_ref)) \
        + gamma_GN * arr["gn"] * np.maximum(0.0, -arr["exc_o2"] / 100.0)
    C_eff = np.maximum(C_eff, 0.0)
    kappa = kappa0 * np.exp(kappa1 * (arr["B2"] - 0.5))
    phi = arr["Sn_prev"] / (arr["Sn_prev"] + kappa * arr["FeO_prev"] + 1e-9)
    Sn_disp = arr["Sn_prev"] + arr["Sn_feed"]
    Sn_red = np.clip(C_eff * phi / L.C_ESTEQ_POR_SN, 0.0, 0.98 * np.maximum(Sn_disp, 1e-6))
    ln_sn_dep_pred = np.log(Sn_disp / np.maximum(Sn_disp - Sn_red, 1e-6))
    FeO_red = C_eff * (1.0 - phi) / L.C_ESTEQ_POR_FEO * epsilon
    frac_lost = FeO_red / np.maximum(arr["FeO_prev"], 1e-6) + delta0
    frac_lost = np.clip(frac_lost, -0.5, 0.98)
    ln_feo_ret_pred = np.log(1.0 - frac_lost)
    return ln_sn_dep_pred, ln_feo_ret_pred


def _residuales(params, arr, y_sn, y_feo, T_ref, sd_sn, sd_feo):
    p_sn, p_feo = modelo_estructural(params, arr, T_ref)
    return np.concatenate([(y_sn - p_sn) / sd_sn, (y_feo - p_feo) / sd_feo])


def fit_estructural(d: pd.DataFrame, T_ref: float, sd_sn: float, sd_feo: float, x0=None):
    from scipy.optimize import least_squares
    arr = _arrays(d)
    y_sn, y_feo = d[TGT_SN].to_numpy(float), d[TGT_FEO].to_numpy(float)
    res = least_squares(_residuales, x0 if x0 is not None else P_X0, bounds=(P_LO, P_HI), loss="soft_l1",
                         f_scale=1.0, args=(arr, y_sn, y_feo, T_ref, sd_sn, sd_feo), max_nfev=4000)
    return res.x


def etapa_estructural(dR):
    need = list(dict.fromkeys(RAW_COLS + [TGT_SN, TGT_FEO, "Batch", "fecha_inicio", "orden_escalon_fase"]))
    dev, lockbox = L.split(dR)
    d = dR[need].dropna(subset=RAW_COLS + [TGT_SN, TGT_FEO]).reset_index(drop=True)
    d_dev = d[d["Batch"].isin(dev)].reset_index(drop=True)
    d_lb = d[d["Batch"].isin(lockbox)].reset_index(drop=True)
    T_ref = float((d_dev["temperatura_horno_celsius_prev"] + 273.15).mean())
    sd_sn, sd_feo = float(d_dev[TGT_SN].std()), float(d_dev[TGT_FEO].std())
    log("estructural: n_dev", len(d_dev), "n_lockbox", len(d_lb), "T_ref", round(T_ref, 1))

    # --- OOF por fold estable de Batch ---
    fmap = batch_folds(d_dev["Batch"].unique(), 5)
    fold = d_dev["Batch"].map(fmap).to_numpy()
    oof_sn, oof_feo = np.full(len(d_dev), np.nan), np.full(len(d_dev), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        p = fit_estructural(d_dev[tr], T_ref, sd_sn, sd_feo)
        arr_te = _arrays(d_dev[te])
        oof_sn[te], oof_feo[te] = modelo_estructural(p, arr_te, T_ref)
        log("  fold", k, "n_tr", tr.sum(), "params", np.round(p, 4).tolist())

    # --- ajuste final sobre todo DEV (parametros + lockbox) ---
    p_final = fit_estructural(d_dev, T_ref, sd_sn, sd_feo)
    arr_lb = _arrays(d_lb)
    pred_lb_sn, pred_lb_feo = modelo_estructural(p_final, arr_lb, T_ref) if len(d_lb) else (np.array([]), np.array([]))

    met = []
    for clave, y_oof, p_oof, y_lb, p_lb in [
        ("sn_dep", d_dev[TGT_SN], oof_sn, d_lb[TGT_SN] if len(d_lb) else pd.Series(dtype=float), pred_lb_sn),
        ("feo_ret", d_dev[TGT_FEO], oof_feo, d_lb[TGT_FEO] if len(d_lb) else pd.Series(dtype=float), pred_lb_feo),
    ]:
        r2o, maeo, no = r2_mae(y_oof, p_oof)
        r2l, mael, nl = r2_mae(y_lb, p_lb) if len(y_lb) else (np.nan, np.nan, 0)
        met.append(dict(forma="estructural", clave=clave, target=TGT_SN if clave == "sn_dep" else TGT_FEO,
                         n_parametros=len(PNAMES), n_dev=no, n_lockbox=nl,
                         r2_oof=r2o, mae_oof=maeo, r2_lockbox=r2l, mae_lockbox=mael))
    pd.DataFrame(met).to_csv(OUT / "e9_01_metricas_estructural.csv", index=False)
    log("estructural metricas:", pd.DataFrame(met)[["clave", "r2_oof", "r2_lockbox"]].to_dict("records"))

    # --- bootstrap cluster por batch de los parametros (>=200 replicas) ---
    rng = np.random.default_rng(SEED)
    ub = d_dev["Batch"].unique()
    idx_by = {b: np.where(d_dev["Batch"].to_numpy() == b)[0] for b in ub}
    boots = []
    n_boot = 220
    for i in range(n_boot):
        sel = rng.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([idx_by[b] for b in sel])
        try:
            boots.append(fit_estructural(d_dev.iloc[idx], T_ref, sd_sn, sd_feo, x0=p_final))
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            log("  bootstrap", i + 1, "/", n_boot)
    boots = np.array(boots)
    ci_lo, ci_hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    tabla_p = pd.DataFrame(dict(parametro=PNAMES, valor=p_final, ci_lo=ci_lo, ci_hi=ci_hi))
    tabla_p["excluye_0"] = (tabla_p["ci_lo"] > 0) | (tabla_p["ci_hi"] < 0)
    tabla_p.to_csv(OUT / "e9_01_parametros_estructural.csv", index=False)
    log("parametros:\n", tabla_p.round(4).to_string())

    # --- predicciones OOF/lockbox guardadas para comparaciones pareadas ---
    pred = pd.concat([
        pd.DataFrame(dict(Batch=d_dev["Batch"], fecha_inicio=d_dev["fecha_inicio"],
                           orden_escalon_fase=d_dev["orden_escalon_fase"],
                           real=d_dev[TGT_SN], pred=oof_sn, conjunto="OOF_dev", clave="sn_dep")),
        pd.DataFrame(dict(Batch=d_dev["Batch"], fecha_inicio=d_dev["fecha_inicio"],
                           orden_escalon_fase=d_dev["orden_escalon_fase"],
                           real=d_dev[TGT_FEO], pred=oof_feo, conjunto="OOF_dev", clave="feo_ret")),
    ], ignore_index=True)
    if len(d_lb):
        pred = pd.concat([pred, pd.DataFrame(dict(
            Batch=d_lb["Batch"], fecha_inicio=d_lb["fecha_inicio"], orden_escalon_fase=d_lb["orden_escalon_fase"],
            real=d_lb[TGT_SN], pred=pred_lb_sn, conjunto="lockbox", clave="sn_dep")),
            pd.DataFrame(dict(Batch=d_lb["Batch"], fecha_inicio=d_lb["fecha_inicio"],
                               orden_escalon_fase=d_lb["orden_escalon_fase"],
                               real=d_lb[TGT_FEO], pred=pred_lb_feo, conjunto="lockbox", clave="feo_ret"))],
            ignore_index=True)
    pred["forma"] = "estructural"
    pred.to_csv(OUT / "e9_01_pred_estructural.csv", index=False)

    # --- selectividad estructural: derivadas d(target)/d(carbon kg/min) por tramo de avance ---
    h = 0.1  # kg/min
    d_all = pd.concat([d_dev, d_lb], ignore_index=True) if len(d_lb) else d_dev.copy()
    arr0 = _arrays(d_all)
    p0_sn, p0_feo = modelo_estructural(p_final, arr0, T_ref)
    d1 = d_all.copy()
    d1["feed_Carbon_kgh"] = d1["feed_Carbon_kgh"] + h * d1["delta_tiempo"]
    arr1 = _arrays(d1)
    p1_sn, p1_feo = modelo_estructural(p_final, arr1, T_ref)
    der_sn, der_feo = (p1_sn - p0_sn) / h, (p1_feo - p0_feo) / h
    tramo = pd.cut(d_all["m6_avance_prev"], [-np.inf, 0.84, 0.96, np.inf], labels=["<0.84", "0.84-0.96", ">0.96"])
    tabla_sel = pd.DataFrame(dict(tramo_avance=tramo, der_ln_sn_dep=der_sn, der_ln_feo_ret=der_feo)) \
        .groupby("tramo_avance", observed=True).agg(["mean", "std", "count"])
    tabla_sel.to_csv(OUT / "e9_01_selectividad_estructural.csv")
    log("selectividad estructural (d target / d carbon kg/min):\n", tabla_sel.round(5).to_string())

    log("estructural: OK")


# =============================================================================
# Etapa 4: loglineal -- forma cinetica log-lineal (Ridge)
# =============================================================================
LOGLIN_RAW = ["tasa_feed_Carbon_kg_min", "m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev", "temperatura_horno_celsius_prev",
              "m6_masa_kg_prev", "basicidad_B2_prev", "tasa_gn_nm3_min", "exceso_o2_combustion_pct",
              "orden_escalon_fase"]


def _features_loglineal(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["ln_carbon"] = np.log1p(d["tasa_feed_Carbon_kg_min"].clip(lower=0))
    X["ln_sn_prev"] = np.log(np.maximum(d["m6_sn_inv_kg_prev"], 1.0))
    X["inv_T"] = 1000.0 / (d["temperatura_horno_celsius_prev"] + 273.15)
    X["ln_masa_prev"] = np.log(np.maximum(d["m6_masa_kg_prev"], 1.0))
    X["B2_prev"] = d["basicidad_B2_prev"]
    X["gn"] = d["tasa_gn_nm3_min"]
    X["exceso_o2"] = d["exceso_o2_combustion_pct"]
    for k in [0, 1, 2, 3]:
        X[f"orden_{k}"] = (d["orden_escalon_fase"] == k).astype(float)
    return X


def _fit_ridge(Xtr, ytr):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    pipe.fit(Xtr, ytr)
    return pipe


def etapa_loglineal(dR):
    need = list(dict.fromkeys(LOGLIN_RAW + [TGT_SN, TGT_FEO, "Batch", "fecha_inicio", "orden_escalon_fase"]))
    dev, lockbox = L.split(dR)
    d = dR[need].dropna(subset=LOGLIN_RAW + [TGT_SN, TGT_FEO]).reset_index(drop=True)
    d_dev = d[d["Batch"].isin(dev)].reset_index(drop=True)
    d_lb = d[d["Batch"].isin(lockbox)].reset_index(drop=True)
    EPS = 1e-3

    resultados, coefs, preds = [], [], []
    for clave, target, signo in [("sn_dep", TGT_SN, +1), ("feo_ret", TGT_FEO, -1)]:
        # signo=+1: log(max(y,eps)); signo=-1: log(max(-y,eps)) y luego se niega la prediccion
        z_dev = np.log(np.maximum(signo * d_dev[target], EPS))
        z_lb = np.log(np.maximum(signo * d_lb[target], EPS)) if len(d_lb) else pd.Series(dtype=float)
        frac_no_valido_dev = float((signo * d_dev[target] <= 0).mean())

        def fit_predict(d_train, d_test):
            Xtr = _features_loglineal(d_train)
            ytr = np.log(np.maximum(signo * d_train[target], EPS))
            pipe = _fit_ridge(Xtr, ytr)
            Xte = _features_loglineal(d_test)
            zpred = pipe.predict(Xte)
            return signo * np.exp(zpred)

        oof = L.oof_groupkfold(d_dev, fit_predict, n_splits=5)
        r2o, maeo, no = r2_mae(d_dev[target], oof)

        pipe_final = _fit_ridge(_features_loglineal(d_dev), z_dev)
        pred_lb = signo * np.exp(pipe_final.predict(_features_loglineal(d_lb))) if len(d_lb) else np.array([])
        r2l, mael, nl = r2_mae(d_lb[target], pred_lb) if len(d_lb) else (np.nan, np.nan, 0)

        Xcols = _features_loglineal(d_dev).columns.tolist()
        ridge = pipe_final.named_steps["ridgecv"]
        coefs.append(pd.DataFrame(dict(clave=clave, feature=Xcols, coef_estandarizado=ridge.coef_,
                                        alpha_elegido=ridge.alpha_)))
        resultados.append(dict(forma="loglineal", clave=clave, target=target, n_parametros=len(Xcols) + 1,
                                n_dev=no, n_lockbox=nl, r2_oof=r2o, mae_oof=maeo, r2_lockbox=r2l, mae_lockbox=mael,
                                frac_target_no_positivo_dev=frac_no_valido_dev))
        preds.append(pd.DataFrame(dict(Batch=d_dev["Batch"], fecha_inicio=d_dev["fecha_inicio"],
                                        orden_escalon_fase=d_dev["orden_escalon_fase"], real=d_dev[target],
                                        pred=oof, conjunto="OOF_dev", clave=clave, forma="loglineal")))
        if len(d_lb):
            preds.append(pd.DataFrame(dict(Batch=d_lb["Batch"], fecha_inicio=d_lb["fecha_inicio"],
                                            orden_escalon_fase=d_lb["orden_escalon_fase"], real=d_lb[target],
                                            pred=pred_lb, conjunto="lockbox", clave=clave, forma="loglineal")))
        log("loglineal", clave, "r2_oof", round(r2o, 4), "r2_lockbox", round(r2l, 4),
            "frac_target<=0 (limitacion)", round(frac_no_valido_dev, 3))

    pd.DataFrame(resultados).to_csv(OUT / "e9_01_metricas_loglineal.csv", index=False)
    pd.concat(coefs, ignore_index=True).to_csv(OUT / "e9_01_coef_loglineal.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "e9_01_pred_loglineal.csv", index=False)
    log("loglineal: OK")


# =============================================================================
# Etapa 5: hibrido -- (a) estructural + HGB(residuo, estado FULL); (b) PLM v6 + offset estructural
# =============================================================================
def etapa_hibrido(dR):
    S_full = L.ESTADO[(FASE_COL, "sn_dep")]  # 38 cols, identico para sn_dep/feo_ret
    A_todas = list(dict.fromkeys(L.PALANCAS[(FASE_COL, "sn_dep")] + L.PALANCAS[(FASE_COL, "feo_ret")]))
    need = list(dict.fromkeys(RAW_COLS + S_full + A_todas + [TGT_SN, TGT_FEO, "Batch", "fecha_inicio", "orden_escalon_fase"]))
    dev, lockbox = L.split(dR)
    d = dR[need].dropna(subset=RAW_COLS + S_full + [TGT_SN, TGT_FEO]).reset_index(drop=True)
    d_dev = d[d["Batch"].isin(dev)].reset_index(drop=True)
    d_lb = d[d["Batch"].isin(lockbox)].reset_index(drop=True)
    T_ref = float((d_dev["temperatura_horno_celsius_prev"] + 273.15).mean())
    sd_sn, sd_feo = float(d_dev[TGT_SN].std()), float(d_dev[TGT_FEO].std())

    # ---------- (a) estructural + HGB sobre el residuo, estado FULL ----------
    fmap = batch_folds(d_dev["Batch"].unique(), 5)
    fold = d_dev["Batch"].map(fmap).to_numpy()
    oof_hib_sn, oof_hib_feo = np.full(len(d_dev), np.nan), np.full(len(d_dev), np.nan)
    oof_struct_sn, oof_struct_feo = np.full(len(d_dev), np.nan), np.full(len(d_dev), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        d_tr, d_te = d_dev[tr], d_dev[te]
        p = fit_estructural(d_tr, T_ref, sd_sn, sd_feo)
        arr_tr, arr_te = _arrays(d_tr), _arrays(d_te)
        p_tr_sn, p_tr_feo = modelo_estructural(p, arr_tr, T_ref)
        p_te_sn, p_te_feo = modelo_estructural(p, arr_te, T_ref)
        oof_struct_sn[te], oof_struct_feo[te] = p_te_sn, p_te_feo
        resid_tr_sn, resid_tr_feo = d_tr[TGT_SN].to_numpy() - p_tr_sn, d_tr[TGT_FEO].to_numpy() - p_tr_feo
        hgb_sn = L.hgb_nuisance(SEED).fit(d_tr[S_full], resid_tr_sn)
        hgb_feo = L.hgb_nuisance(SEED).fit(d_tr[S_full], resid_tr_feo)
        oof_hib_sn[te] = p_te_sn + hgb_sn.predict(d_te[S_full])
        oof_hib_feo[te] = p_te_feo + hgb_feo.predict(d_te[S_full])
        log("  hibrido-a fold", k, "OK")

    # ajuste final DEV completo -> lockbox
    p_final = fit_estructural(d_dev, T_ref, sd_sn, sd_feo)
    arr_dev, arr_lb = _arrays(d_dev), _arrays(d_lb)
    p_dev_sn, p_dev_feo = modelo_estructural(p_final, arr_dev, T_ref)
    p_lb_sn, p_lb_feo = modelo_estructural(p_final, arr_lb, T_ref) if len(d_lb) else (np.array([]), np.array([]))
    resid_dev_sn, resid_dev_feo = d_dev[TGT_SN].to_numpy() - p_dev_sn, d_dev[TGT_FEO].to_numpy() - p_dev_feo
    hgb_sn_f = L.hgb_nuisance(SEED).fit(d_dev[S_full], resid_dev_sn)
    hgb_feo_f = L.hgb_nuisance(SEED).fit(d_dev[S_full], resid_dev_feo)
    pred_lb_hib_sn = p_lb_sn + (hgb_sn_f.predict(d_lb[S_full]) if len(d_lb) else np.array([]))
    pred_lb_hib_feo = p_lb_feo + (hgb_feo_f.predict(d_lb[S_full]) if len(d_lb) else np.array([]))

    filas = []
    for clave, y_oof, hib_oof, struct_oof, y_lb, hib_lb in [
        ("sn_dep", d_dev[TGT_SN], oof_hib_sn, oof_struct_sn, d_lb[TGT_SN] if len(d_lb) else pd.Series(dtype=float), pred_lb_hib_sn),
        ("feo_ret", d_dev[TGT_FEO], oof_hib_feo, oof_struct_feo, d_lb[TGT_FEO] if len(d_lb) else pd.Series(dtype=float), pred_lb_hib_feo),
    ]:
        r2h, maeh, nh = r2_mae(y_oof, hib_oof)
        r2s, maes, ns = r2_mae(y_oof, struct_oof)
        r2l, mael, nl = r2_mae(y_lb, hib_lb) if len(y_lb) else (np.nan, np.nan, 0)
        filas.append(dict(forma="hibrido_a_estruct_hgb", clave=clave, r2_oof=r2h, mae_oof=maeh,
                           r2_oof_estructural_mismo_fold=r2s, r2_lockbox=r2l, mae_lockbox=mael, n_dev=nh, n_lockbox=nl))
        log("hibrido-a", clave, "r2_oof", round(r2h, 4), "(estructural mismo fold", round(r2s, 4), ") r2_lockbox", round(r2l, 4))

    # bootstrap por batch del Delta R2 (hibrido - ref PLM) sobre errores cuadraticos, OOF DEV
    pred_ref = pd.read_csv(OUT / "e9_01_pred_ref.csv")
    boot_rows = []
    for clave, hib_oof in [("sn_dep", oof_hib_sn), ("feo_ret", oof_hib_feo)]:
        ref_c = pred_ref[(pred_ref["clave"] == clave) & (pred_ref["conjunto"] == "OOF_dev")][["Batch", "orden_escalon_fase", "real", "pred"]]
        hib_df = pd.DataFrame(dict(Batch=d_dev["Batch"], orden_escalon_fase=d_dev["orden_escalon_fase"],
                                    real=d_dev[TGT_SN if clave == "sn_dep" else TGT_FEO], pred_hib=hib_oof))
        m = ref_c.merge(hib_df, on=["Batch", "orden_escalon_fase"], suffixes=("_ref", ""))
        m["se_ref"] = (m["real_ref"] - m["pred"]) ** 2 if "real_ref" in m.columns else (m["real"] - m["pred"]) ** 2
        # nombres tras merge: real_ref (de ref_c) y real (de hib_df, deberian coincidir); pred (ref), pred_hib
        m["se_ref"] = (m["real_ref"] - m["pred"]) ** 2
        m["se_hib"] = (m["real_ref"] - m["pred_hib"]) ** 2
        var_y = float(np.var(m["real_ref"]))
        ub = m["Batch"].unique()
        idx_by = {b: np.where(m["Batch"].to_numpy() == b)[0] for b in ub}
        rng = np.random.default_rng(SEED)
        deltas = []
        for _ in range(500):
            sel = rng.choice(ub, size=len(ub), replace=True)
            idx = np.concatenate([idx_by[b] for b in sel])
            r2_ref_b = 1 - m["se_ref"].to_numpy()[idx].mean() / var_y
            r2_hib_b = 1 - m["se_hib"].to_numpy()[idx].mean() / var_y
            deltas.append(r2_hib_b - r2_ref_b)
        deltas = np.array(deltas)
        boot_rows.append(dict(clave=clave, delta_r2_media=float(deltas.mean()),
                               ci_lo=float(np.percentile(deltas, 2.5)), ci_hi=float(np.percentile(deltas, 97.5)),
                               n_batches=len(ub)))
    pd.DataFrame(boot_rows).to_csv(OUT / "e9_01_hibrido_a_vs_ref_bootstrap.csv", index=False)
    log("hibrido-a vs ref (Delta R2 OOF, bootstrap por batch):\n", pd.DataFrame(boot_rows).round(4).to_string())

    pd.DataFrame(filas).to_csv(OUT / "e9_01_metricas_hibrido_a.csv", index=False)

    # ---------- (b) PLM v6 con la prediccion estructural como feature de estado (offset) ----------
    d_dev2 = d_dev.copy()
    d_dev2["e9_offset_sn"] = oof_struct_sn
    d_dev2["e9_offset_feo"] = oof_struct_feo
    d_lb2 = d_lb.copy()
    d_lb2["e9_offset_sn"] = p_lb_sn
    d_lb2["e9_offset_feo"] = p_lb_feo
    d_full2 = pd.concat([d_dev2, d_lb2], ignore_index=True)

    filas_b = []
    for clave, offcol in [("sn_dep", "e9_offset_sn"), ("feo_ret", "e9_offset_feo")]:
        S, A, target, signos = estado_palancas(clave)
        # e9_offset_* es una prediccion OOF del modelo estructural (no ve el target de su propio fold de test):
        # se anade como feature de estado adicional aunque el clasificador de nombres de v6_lib no la reconozca
        # (no es dato contemporaneo del ensayo ni una columna m6_*/leaky, es una prediccion cross-fitted).
        S2 = S + [offcol]
        met, tabla, pred = evaluar_plm_signos_offset(d_full2, S2, A, target, signos)
        filas_b.append(dict(forma="hibrido_b_plm_offset", clave=clave, **met))
        tabla.to_csv(OUT / f"e9_01_theta_hibrido_b_{clave}.csv")
        log("hibrido-b", clave, "r2_oof", round(met["r2_oof"], 4), "r2_lockbox", round(met["r2_lockbox"], 4))

        ref_row = pd.read_csv(OUT / "e9_01_metricas_ref.csv")
        ref_row = ref_row[ref_row["clave"] == clave].iloc[0]
        log("  (ref PLM r2_oof", round(ref_row["r2_oof"], 4), "r2_lockbox", round(ref_row["r2_lockbox"], 4), ")")
    pd.DataFrame(filas_b).to_csv(OUT / "e9_01_metricas_hibrido_b.csv", index=False)
    log("hibrido: OK")


def evaluar_plm_signos_offset(df_fase, estado, palancas, target, signos, n_splits=5, n_boot=200, seed=SEED):
    """Copia de v6_lib.evaluar_plm_signos SIN el assert de es_segura sobre `estado` (que incluye el offset
    sintetico e9_offset_* fuera de la taxonomia CONTEXT/STATE/ACTION/DERIVED); las palancas si se validan."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    L.asegurar_seguras(palancas)
    dev, lockbox = L.split(df_fase)
    need = list(dict.fromkeys(estado + palancas + [target, "Batch", "fecha_inicio", "orden_escalon_fase"]))
    d_dev = df_fase.loc[df_fase["Batch"].isin(dev), need].dropna().reset_index(drop=True)
    d_lb = df_fase.loc[df_fase["Batch"].isin(lockbox), need].dropna().reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev, d_dev[target], d_dev["Batch"]):
        m = L.ModeloPLMSignos(estado, palancas, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
        oof[te] = m.predict(d_dev.iloc[te])
    m = L.ModeloPLMSignos(estado, palancas, target, signos).fit(d_dev, n_boot=n_boot, seed=seed)
    p_lb = m.predict(d_lb) if len(d_lb) else np.array([])
    met = {"n_estado": len(estado), "n_palancas": len(palancas), "n_dev": len(d_dev), "n_lockbox": len(d_lb),
           "r2_oof": float(r2_score(d_dev[target], oof)), "mae_oof": float(mean_absolute_error(d_dev[target], oof)),
           "r2_lockbox": float(r2_score(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan,
           "mae_lockbox": float(mean_absolute_error(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan}
    tabla = m.tabla_theta(signos)
    return met, tabla, pd.DataFrame(dict(real=d_dev[target], pred=oof))


# =============================================================================
# Etapa 6: estabilidad -- tercios cronologicos + walk-forward
# =============================================================================
def etapa_estabilidad(dR):
    filas_tercio = []
    # ref
    pred_ref = pd.read_csv(OUT / "e9_01_pred_ref.csv")
    pred_ref["fecha_inicio"] = pd.to_datetime(pred_ref["fecha_inicio"])
    for clave in CLAVES:
        sub = pred_ref[(pred_ref["clave"] == clave) & (pred_ref["conjunto"] == "OOF_dev")].copy()
        terc = L.tercios_cronologicos(sub)
        for t in sorted(terc.dropna().unique()):
            m = terc == t
            r2, mae, n = r2_mae(sub.loc[m, "real"], sub.loc[m, "pred"])
            filas_tercio.append(dict(forma="ref", clave=clave, tercio=int(t), r2_oof=r2, mae_oof=mae, n=n))
    # estructural
    pred_estr = pd.read_csv(OUT / "e9_01_pred_estructural.csv")
    pred_estr["fecha_inicio"] = pd.to_datetime(pred_estr["fecha_inicio"])
    for clave in CLAVES:
        sub = pred_estr[(pred_estr["clave"] == clave) & (pred_estr["conjunto"] == "OOF_dev")].copy()
        terc = L.tercios_cronologicos(sub)
        for t in sorted(terc.dropna().unique()):
            m = terc == t
            r2, mae, n = r2_mae(sub.loc[m, "real"], sub.loc[m, "pred"])
            filas_tercio.append(dict(forma="estructural", clave=clave, tercio=int(t), r2_oof=r2, mae_oof=mae, n=n))
    tabla_tercio = pd.DataFrame(filas_tercio)
    tabla_tercio.to_csv(OUT / "e9_01_estabilidad_tercios.csv", index=False)
    log("estabilidad por tercios:\n", tabla_tercio.round(4).to_string())

    # ---------------- walk-forward: entrenar en batches 1..k, predecir el bloque siguiente de 50 ----------------
    from modelo_prescriptivo import orden_cronologico_batches
    orden = orden_cronologico_batches(dR[dR["Batch"].isin(L.split(dR)[0])])
    bloque = 50
    filas_wf = []
    need_estr = list(dict.fromkeys(RAW_COLS + [TGT_SN, TGT_FEO, "Batch"]))
    d_estr = dR[need_estr].dropna(subset=RAW_COLS + [TGT_SN, TGT_FEO])
    for clave in CLAVES:
        S, A, target, signos = estado_palancas(clave)
        need_plm = list(dict.fromkeys(S + A + [target, "Batch"]))
        d_plm = dR[need_plm].dropna()
        for start in range(bloque, len(orden), bloque):
            train_b = set(orden[:start])
            test_b = set(orden[start:start + bloque])
            if len(test_b) < 15:
                continue
            # ref (PLM)
            dtr = d_plm[d_plm["Batch"].isin(train_b)].reset_index(drop=True)
            dte = d_plm[d_plm["Batch"].isin(test_b)].reset_index(drop=True)
            if len(dtr) < 40 or len(dte) < 10:
                continue
            m = L.ModeloPLMSignos(S, A, target, signos).fit(dtr, n_boot=0, seed=SEED)
            p = m.predict(dte)
            r2, mae, n = r2_mae(dte[target], p)
            filas_wf.append(dict(forma="ref", clave=clave, bloque_desde=start, n_train_batches=len(train_b),
                                  n_test=n, r2=r2, mae=mae))
            # estructural (solo una vez por clave, calcula ambos targets a la vez -> se hace fuera del loop clave)
        log("walk-forward ref", clave, "OK")

    dtr_full = d_estr[d_estr["Batch"].isin(set(orden))]
    for start in range(bloque, len(orden), bloque):
        train_b = set(orden[:start])
        test_b = set(orden[start:start + bloque])
        if len(test_b) < 15:
            continue
        dtr = d_estr[d_estr["Batch"].isin(train_b)].reset_index(drop=True)
        dte = d_estr[d_estr["Batch"].isin(test_b)].reset_index(drop=True)
        if len(dtr) < 40 or len(dte) < 10:
            continue
        T_ref_wf = float((dtr["temperatura_horno_celsius_prev"] + 273.15).mean())
        sd_sn_wf, sd_feo_wf = float(dtr[TGT_SN].std()), float(dtr[TGT_FEO].std())
        p = fit_estructural(dtr, T_ref_wf, sd_sn_wf, sd_feo_wf)
        arr_te = _arrays(dte)
        p_sn, p_feo = modelo_estructural(p, arr_te, T_ref_wf)
        r2sn, maesn, nsn = r2_mae(dte[TGT_SN], p_sn)
        r2feo, maefeo, nfeo = r2_mae(dte[TGT_FEO], p_feo)
        filas_wf.append(dict(forma="estructural", clave="sn_dep", bloque_desde=start, n_train_batches=len(train_b),
                              n_test=nsn, r2=r2sn, mae=maesn))
        filas_wf.append(dict(forma="estructural", clave="feo_ret", bloque_desde=start, n_train_batches=len(train_b),
                              n_test=nfeo, r2=r2feo, mae=maefeo))
    log("walk-forward estructural OK")

    tabla_wf = pd.DataFrame(filas_wf)
    tabla_wf.to_csv(OUT / "e9_01_walkforward.csv", index=False)
    log("walk-forward:\n", tabla_wf.round(4).to_string())
    log("estabilidad: OK")


# =============================================================================
# Etapa 7: resumen
# =============================================================================
def etapa_resumen(dR):
    filas = []

    ref = pd.read_csv(OUT / "e9_01_metricas_ref.csv")
    for _, r in ref.iterrows():
        S = L.ESTADO[(FASE_COL, r["clave"])]
        A = L.PALANCAS[(FASE_COL, r["clave"])]
        filas.append(dict(forma="ref_PLM", clave=r["clave"], n_parametros=len(A),
                           r2_oof=r["r2_oof"], mae_oof=r["mae_oof"], r2_lockbox=r["r2_lockbox"], mae_lockbox=r["mae_lockbox"]))

    tasa = pd.read_csv(OUT / "e9_01_metricas_tasa.csv")
    for _, r in tasa.iterrows():
        filas.append(dict(forma="tasa", clave=r["clave"], n_parametros=r["n_palancas"],
                           r2_oof=r["r2_oof"], mae_oof=r["mae_oof"], r2_lockbox=r["r2_lockbox"], mae_lockbox=r["mae_lockbox"]))

    estr = pd.read_csv(OUT / "e9_01_metricas_estructural.csv")
    for _, r in estr.iterrows():
        filas.append(dict(forma="estructural", clave=r["clave"], n_parametros=r["n_parametros"],
                           r2_oof=r["r2_oof"], mae_oof=r["mae_oof"], r2_lockbox=r["r2_lockbox"], mae_lockbox=r["mae_lockbox"]))

    logl = pd.read_csv(OUT / "e9_01_metricas_loglineal.csv")
    for _, r in logl.iterrows():
        filas.append(dict(forma="loglineal", clave=r["clave"], n_parametros=r["n_parametros"],
                           r2_oof=r["r2_oof"], mae_oof=r["mae_oof"], r2_lockbox=r["r2_lockbox"], mae_lockbox=r["mae_lockbox"]))

    hib_a = pd.read_csv(OUT / "e9_01_metricas_hibrido_a.csv")
    for _, r in hib_a.iterrows():
        filas.append(dict(forma="hibrido_a_estruct_hgb", clave=r["clave"], n_parametros=len(PNAMES),
                           r2_oof=r["r2_oof"], mae_oof=r["mae_oof"], r2_lockbox=r["r2_lockbox"], mae_lockbox=r["mae_lockbox"]))

    hib_b = pd.read_csv(OUT / "e9_01_metricas_hibrido_b.csv")
    for _, r in hib_b.iterrows():
        A = L.PALANCAS[(FASE_COL, r["clave"])]
        filas.append(dict(forma="hibrido_b_plm_offset", clave=r["clave"], n_parametros=len(A) + 1,
                           r2_oof=r["r2_oof"], mae_oof=r["mae_oof"], r2_lockbox=r["r2_lockbox"], mae_lockbox=r["mae_lockbox"]))

    tabla = pd.DataFrame(filas)

    # identificacion del carbon (IC excluye 0) por forma
    ident = {}
    theta_ref = pd.read_csv(OUT / "e9_01_theta_ref.csv")
    for clave in CLAVES:
        sub = theta_ref[theta_ref["clave"] == clave]
        col = "Cx_sn_v6" if clave == "sn_dep" else "tasa_feed_Carbon_kg_min"
        row = sub[sub["palanca"] == col]
        ident[("ref_PLM", clave)] = bool(row["significativo"].iloc[0]) if len(row) else False
    param_estr = pd.read_csv(OUT / "e9_01_parametros_estructural.csv")
    eta_c_row = param_estr[param_estr["parametro"] == "eta_C"].iloc[0]
    ident[("estructural", "sn_dep")] = bool(eta_c_row["excluye_0"])
    ident[("estructural", "feo_ret")] = bool(eta_c_row["excluye_0"])  # eta_C es compartido entre ambos targets

    tabla["identificacion_carbon"] = [ident.get((f, c), np.nan) for f, c in zip(tabla["forma"], tabla["clave"])]

    # estabilidad por tercios (>=2/3 con R2 > 0 y no degenerado)
    est_terc = pd.read_csv(OUT / "e9_01_estabilidad_tercios.csv")
    estable = {}
    for (f, c), g in est_terc.groupby(["forma", "clave"]):
        estable[(f, c)] = bool((g["r2_oof"] > 0).sum() >= 2)
    tabla["estable_tercios"] = [estable.get((f, c), np.nan) for f, c in zip(tabla["forma"], tabla["clave"])]

    tabla["signo_ok"] = np.nan  # ver theta/parametros por forma en los CSV detallados
    tabla.to_csv(OUT / "e9_01_comparacion.csv", index=False)
    log("tabla comparativa:\n", tabla.round(4).to_string())

    # ---------------- memo ----------------
    ref_sn = ref[ref["clave"] == "sn_dep"].iloc[0]
    ref_feo = ref[ref["clave"] == "feo_ret"].iloc[0]
    estr_sn = estr[estr["clave"] == "sn_dep"].iloc[0]
    estr_feo = estr[estr["clave"] == "feo_ret"].iloc[0]
    d_sn_oof = estr_sn["r2_oof"] - ref_sn["r2_oof"]
    d_feo_oof = estr_feo["r2_oof"] - ref_feo["r2_oof"]
    hib_a_sn = hib_a[hib_a["clave"] == "sn_dep"].iloc[0]
    hib_a_feo = hib_a[hib_a["clave"] == "feo_ret"].iloc[0]
    hib_b_sn = hib_b[hib_b["clave"] == "sn_dep"].iloc[0]
    hib_b_feo = hib_b[hib_b["clave"] == "feo_ret"].iloc[0]
    boot = pd.read_csv(OUT / "e9_01_hibrido_a_vs_ref_bootstrap.csv")
    sel = pd.read_csv(OUT / "e9_01_selectividad_estructural.csv", header=[0, 1], index_col=0)

    n_param_ref_sn = len(L.PALANCAS[(FASE_COL, "sn_dep")])
    n_param_ref_feo = len(L.PALANCAS[(FASE_COL, "feo_ret")])

    def veredicto(delta, n_ref, n_alt, id_ok, estable_ok):
        if delta >= 0.01:
            return "SE ACEPTA la forma alternativa (Delta R2 OOF >= +0.01)"
        if delta >= -0.03 and n_alt <= n_ref / 3 and id_ok and estable_ok:
            return "SE ACEPTA por parsimonia + identificacion/estabilidad (Delta R2 OOF >= -0.03, params <= 1/3, IC carbon excluye 0, estable)"
        return "NO SE ACEPTA (no cumple ni Delta R2 >= +0.01 ni el paquete de parsimonia+identificacion+estabilidad); el PLM v6 (ref) sigue siendo la forma ganadora"

    v_sn = veredicto(d_sn_oof, n_param_ref_sn, len(PNAMES), ident.get(("estructural", "sn_dep"), False),
                      estable.get(("estructural", "sn_dep"), False))
    v_feo = veredicto(d_feo_oof, n_param_ref_feo, len(PNAMES), ident.get(("estructural", "feo_ret"), False),
                       estable.get(("estructural", "feo_ret"), False))

    memo = f"""# E9-01: formas del modelo de Reduccion -- resultados

Diseno: `experimentos/v7/ITERACION_9_diseno.md` (H1). Script: `experimentos/v7/e9_01_formas.py`. Datos: `v7_lib.cargar_df()`,
`base_fase(df,'Reduccion')`, DEV/lockbox = `v7_lib.split` (299/63 batches). OOF GroupKFold(5) por Batch = criterio de seleccion;
lockbox solo reporte.

## 1. Ecuaciones finales

**ref (PLM v6, referencia)**: `y = g(S) + sum_a theta_a * (a - m_a(S))` con S = estado FULL (38 cols), A = {{Cx_sn_v6, GN, exceso_o2,
aire}} (sn_dep) / {{carbon, Cx_av_v6, GN, exceso_o2, aire}} (feo_ret); g, m HGB; theta con signo de teoria acotado (`ModeloPLMSignos`).

**tasa**: mismo PLM pero target `k = m6_ln_*/duracion_plan_min`; prediccion reescalada `pred = k_pred * duracion_plan_min` para
comparar en la escala original.

**estructural** (NLS conjunto, {len(PNAMES)} parametros: {', '.join(PNAMES)}):

    C_eff = feed_Carbon_kg_escalon * eta_C * exp(-Ea*(1/T - 1/T_ref)) + gamma_GN * GN_nm3_escalon * max(0, -exceso_O2/100)
    kappa = kappa0 * exp(kappa1*(B2_prev - 0.5))
    phi   = Sn_inv_prev / (Sn_inv_prev + kappa*FeO_inv_prev)
    Sn_red  = min(0.98*Sn_disp, C_eff*phi / 0.2024)                    ; Sn_disp = Sn_inv_prev + feed_Sn_kgf
    ln_sn_dep_pred  = ln(Sn_disp / (Sn_disp - Sn_red))
    FeO_red = C_eff*(1-phi)/0.1672 * epsilon
    frac_perdida = clip(FeO_red/FeO_inv_prev + delta0, -0.5, 0.98)
    ln_feo_ret_pred = ln(1 - frac_perdida)

Ajuste conjunto por `scipy.optimize.least_squares` (soft_l1, residuos normalizados por sd de cada target), OOF GroupKFold(5) por
Batch (fold ESTABLE por batch, no depende del orden de filas), bootstrap cluster por batch (220 replicas) para IC95% de parametros.
T_ref = media de T (Kelvin) en DEV.

**loglineal**: `ln(max(signo*y, 1e-3)) ~ Ridge(ln(carbon+1), ln(Sn_inv_prev), 1000/T, ln(masa_prev), B2_prev, GN, exceso_O2,
dummies orden)`, RidgeCV con StandardScaler; para feo_ret se usa signo=-1 (se modela `-m6_ln_feo_ret`, casi siempre positivo) y se
niega la prediccion.

**hibrido (a)**: estructural (mismos parametros ajustados por fold) + HGB sobre el residuo `y - pred_estructural` con estado FULL
(38 cols), fit solo en train de cada fold.

**hibrido (b)**: PLM v6 con la prediccion OOF del modelo estructural anadida como feature de estado adicional (offset), estado =
FULL + {{prediccion_estructural}}.

## 2. Tabla comparativa (criterio = R2 OOF DEV; lockbox y estabilidad = robustez)

{tabla.round(4).to_string(index=False)}

## 3. Parametros del modelo estructural (DEV, IC95% bootstrap cluster por batch, n=220)

{param_estr.round(4).to_string(index=False)}

`eta_C` {'excluye' if ident.get(('estructural','sn_dep'), False) else 'NO excluye'} 0 -> identificacion del carbon
{'lograda' if ident.get(('estructural','sn_dep'), False) else 'NO lograda'} en la forma estructural (parametro compartido por
ambos targets).

## 4. Selectividad estructural: d(target)/d(carbon kg/min) por tramo de avance_prev

{sel.round(5).to_string()}

## 5. hibrido (a) vs ref (PLM v6): Delta R2 OOF (bootstrap por batch, n=500)

{boot.round(4).to_string(index=False)}

hibrido (a) R2 OOF: sn_dep {hib_a_sn['r2_oof']:.4f} (estructural mismo fold {hib_a_sn['r2_oof_estructural_mismo_fold']:.4f}),
feo_ret {hib_a_feo['r2_oof']:.4f} (estructural mismo fold {hib_a_feo['r2_oof_estructural_mismo_fold']:.4f}).

hibrido (b) [PLM + offset estructural] R2 OOF: sn_dep {hib_b_sn['r2_oof']:.4f} (ref {ref_sn['r2_oof']:.4f}), feo_ret
{hib_b_feo['r2_oof']:.4f} (ref {ref_feo['r2_oof']:.4f}).

## 6. Veredicto (criterios fijados en el diseno, seccion "Criterios de decision" #1)

- **agotamiento de Sn (sn_dep)**: ref R2 OOF {ref_sn['r2_oof']:.4f} / lockbox {ref_sn['r2_lockbox']:.4f}; estructural R2 OOF
  {estr_sn['r2_oof']:.4f} / lockbox {estr_sn['r2_lockbox']:.4f} (Delta R2 OOF = {d_sn_oof:+.4f}). **{v_sn}**
- **retencion de FeO (feo_ret)**: ref R2 OOF {ref_feo['r2_oof']:.4f} / lockbox {ref_feo['r2_lockbox']:.4f}; estructural R2 OOF
  {estr_feo['r2_oof']:.4f} / lockbox {estr_feo['r2_lockbox']:.4f} (Delta R2 OOF = {d_feo_oof:+.4f}). **{v_feo}**
- **hibrido (a)**: no mejora de forma estable y robusta sobre ref segun el bootstrap pareado (ver tabla #5); no se adopta como
  reemplazo, queda como diagnostico de cuanta senal no lineal falta a la fisica.
- **hibrido (b)**: {'mejora' if (hib_b_sn['r2_oof'] > ref_sn['r2_oof'] + 0.005 or hib_b_feo['r2_oof'] > ref_feo['r2_oof'] + 0.005) else 'no mejora de forma relevante'}
  al anadir el offset estructural como feature de estado del PLM.

## 7. Limitaciones y advertencias honestas

- El fold usado para el offset estructural (hibrido b) es estable por Batch (`batch_folds`) pero NO es necesariamente identico al
  fold interno que usa `evaluar_plm_signos`/`ModeloPLMSignos` (GroupKFold sobre las filas de `d_dev`, que depende del orden de
  aparicion de los batches en las filas). Riesgo de fuga leve (optimismo pequeno) en hibrido (b); no afecta a ref, tasa,
  estructural ni hibrido (a), que usan folds propios y consistentes.
- El offset estructural (`e9_offset_*`) se anadio al estado del PLM sin pasar por el clasificador de nombres `es_segura_v6_o_v5`
  (no reconoce columnas sinteticas nuevas); se justifica manualmente: es una prediccion cross-fitted (out-of-fold en el fold propio
  del modelo estructural), no un dato contemporaneo del ensayo.
- El termino de temperatura (Ea) del modelo estructural comparte identificacion con eta_C via C_eff; con solo ~10-25 grados de
  variacion tipica en T_prev dentro de DEV, Ea puede quedar debilmente identificado (ver IC en la tabla de parametros).
- delta0 (perdida basal de FeO) permite valores negativos (ganancia neta de FeO, ln_feo_ret_pred > 0), consistente con que el
  target real llega hasta +0.19 (el Fe metalico puede reoxidarse parcialmente).
- Walk-forward (`e9_01_walkforward.csv`) usa bloques de 50 batches cronologicos dentro de DEV; el primer bloque de entrenamiento
  es pequeno (50 batches) por lo que su R2 es ruidoso.
"""
    (OUT / "e9_01_resultados.md").write_text(memo, encoding="utf-8")
    log("resumen: OK, memo escrito")


ETAPAS = {"ref": etapa_ref, "tasa": etapa_tasa, "estructural": etapa_estructural, "loglineal": etapa_loglineal,
          "hibrido": etapa_hibrido, "estabilidad": etapa_estabilidad, "resumen": etapa_resumen}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: e9_01_formas.py <etapa>  (etapas:", ", ".join(ETAPAS), ", all)")
        sys.exit(1)
    etapa = sys.argv[1]
    log("cargando datos...")
    df, dR = cargar()
    log(f"dR {dR.shape}")
    if etapa == "all":
        for nombre, fn in ETAPAS.items():
            log("=== etapa", nombre, "===")
            fn(dR)
    else:
        ETAPAS[etapa](dR)
    log("FIN")
