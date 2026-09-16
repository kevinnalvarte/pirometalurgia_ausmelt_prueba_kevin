"""E9-04 -- Fusion: reformulacion (predictibilidad robusta a la campana y objetivo defendible).

Diseno: experimentos/v7/ITERACION_9_diseno.md (H4). Libreria: experimentos/v7/v7_lib.py (L).
Punto de partida: PLM v6 de Fusion (`candidate_win_model_v6.md`, `candidate_win_model.md` SS3/4/6/8,
`experimentos/v5/e7_02_resultados.md`, `experimentos/v5/e7_09_diag_fusion_lockbox.py`): sn_dep R2 OOF
0.161 / lockbox -0.47 (sesgo +0.08 en F6, varianza del target 25% menor en lockbox); dT 0.50/0.04.

Etapas (CLI):
    diagnostico      -- reproduce PLM v6 Fusion (sn_dep, dT); R2/sesgo/sd por orden de escalon
                        (OOF y lockbox); recalibracion de intercepto (global y por orden) con los
                        primeros 20 batches del lockbox.
    reentrenamiento  -- walk-forward cronologico sobre TODO (DEV+lockbox): bloques de 30 batches,
                        ventana creciente y ventana movil de 150; reentrenamiento progresivo cada 10
                        batches dentro del lockbox. R2/MAE/correlacion/theta_GN por bloque.
    targets          -- 8 formulaciones candidatas del target de Fusion (mismo estado ESTADO_F_CURADO_V6
                        y mismas palancas); R2 OOF/lockbox, theta con IC, correlacion del agregado de
                        batch con el KPI.
    estado_final     -- estado al cierre de Fusion (fila R0 de cada batch) como predictor de
                        R_sum_m6_ln_feo_ret, R_sum_m6_ln_sn_dep, J_R_batch y el KPI: OLS HC3
                        estandarizada + controles, HGB OOF, importancia por permutacion.
    polvo            -- canal polvo: OLS HC3 de f_polvo y KPI sobre gas de lanza de Fusion por t de
                        carga y T media; PLM de proxies de escalon (d_temperatura_gas_pre_bhf_celsius,
                        d_tiro_horno_pct) sobre estado+palancas de Fusion.
    resumen          -- ensambla e9_04_resumen.csv desde las salidas de las etapas anteriores.
    todas            -- corre todas en orden.

Entorno: .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3; sin joblib/loky.
Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v7/e9_04_fusion.py <etapa>
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent            # experimentos/v7
sys.path.insert(0, str(HERE))
import v7_lib as L                                 # noqa: E402
import modelo_prescriptivo as mp                   # noqa: E402
import modelo_predictivo_v5 as mp5                 # noqa: E402

SEED = L.SEED
BLOQUE_WF = 30
VENTANA_MOVIL = 150
BLOQUE_LB = 10
N_CALIB_LOCKBOX = 20

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# =============================================================================
# 1) diagnostico
# =============================================================================

def etapa_diagnostico() -> None:
    df = L.cargar_df()
    dF = L.base_fase(df, "Fusión")
    filas_metricas, filas_orden, filas_recal = [], [], []
    for clave in ("sn_dep", "dT"):
        target = L.TARGETS["Fusión"][clave]
        S, A = L.ESTADO[("Fusión", clave)], L.PALANCAS[("Fusión", clave)]
        signo = L.SIGNOS.get(("Fusión", clave))
        L.asegurar_seguras(S + A)
        met, tabla, pred = L.evaluar_plm_signos(dF, S, A, target, signo_teorico=signo, n_boot=300)
        met["clave"] = clave
        filas_metricas.append(met)
        tabla.to_csv(HERE / f"e9_04_theta_{clave}.csv", index=False)
        log(f"{clave}: r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f} "
            f"n_dev={met['n_dev']} n_lockbox={met['n_lockbox']}")

        # --- R2 / sesgo / sd por orden de escalon (F1..F6), OOF y lockbox
        for conjunto, g0 in pred.groupby("conjunto"):
            for orden, g in g0.groupby("orden_escalon_fase"):
                if len(g) < 5:
                    continue
                filas_orden.append({
                    "clave": clave, "conjunto": conjunto, "orden_escalon_fase": int(orden), "n": len(g),
                    "r2": r2_score(g.real, g.pred) if g.real.std() > 0 else np.nan,
                    "mae": mean_absolute_error(g.real, g.pred),
                    "sesgo_pred_menos_real": float((g.pred - g.real).mean()),
                    "sd_real": float(g.real.std()), "media_real": float(g.real.mean()),
                    "pearson": pearsonr(g.real, g.pred)[0] if g.real.std() > 0 and g.pred.std() > 0 else np.nan,
                })

        # --- recalibracion de intercepto (simulacion de "ajuste de campana minimo") con los
        # primeros 20 batches del lockbox (cronologico); evaluacion sobre el resto del lockbox
        lb = pred[pred.conjunto == "lockbox"].copy()
        orden_lb_batches = lb.groupby("Batch")["fecha_inicio"].min().sort_values().index.tolist()
        calib_b, eval_b = set(orden_lb_batches[:N_CALIB_LOCKBOX]), set(orden_lb_batches[N_CALIB_LOCKBOX:])
        calib, evalset = lb[lb.Batch.isin(calib_b)], lb[lb.Batch.isin(eval_b)].copy()
        bias_global = float((calib.pred - calib.real).mean())
        bias_por_orden = (calib.assign(err=calib.pred - calib.real).groupby("orden_escalon_fase")["err"].mean())
        evalset["pred_sin_recalibrar"] = evalset["pred"]
        evalset["pred_intercepto_global"] = evalset["pred"] - bias_global
        evalset["pred_intercepto_por_orden"] = evalset["pred"] - evalset["orden_escalon_fase"].map(bias_por_orden).fillna(bias_global)
        for politica in ("sin_recalibrar", "intercepto_global", "intercepto_por_orden"):
            pc = evalset[f"pred_{politica}"]
            filas_recal.append({
                "clave": clave, "politica": politica, "n_calib_batches": len(calib_b), "n_eval_batches": len(eval_b),
                "n_eval_filas": len(evalset), "r2": r2_score(evalset.real, pc) if evalset.real.std() > 0 else np.nan,
                "mae": mean_absolute_error(evalset.real, pc),
                "pearson": pearsonr(evalset.real, pc)[0] if evalset.real.std() > 0 else np.nan,
                "bias_global_calib": bias_global,
            })
        log(f"  recalibracion {clave}: bias_global(calib)={bias_global:+.4f}  "
            f"bias_por_orden={bias_por_orden.round(4).to_dict()}")

    pd.DataFrame(filas_metricas).to_csv(HERE / "e9_04_diag_metricas.csv", index=False)
    pd.DataFrame(filas_orden).to_csv(HERE / "e9_04_diag_por_orden.csv", index=False)
    pd.DataFrame(filas_recal).to_csv(HERE / "e9_04_diag_recalibracion.csv", index=False)
    log("FIN diagnostico")


# =============================================================================
# 2) reentrenamiento (walk-forward TODO + progresivo dentro del lockbox)
# =============================================================================

def _fit_eval(d_tr: pd.DataFrame, d_te: pd.DataFrame, S: list, A: list, target: str, signo: dict, seed: int = SEED):
    m = L.ModeloPLMSignos(S, A, target, signo).fit(d_tr, n_boot=0, seed=seed)
    p = m.predict(d_te)
    r2 = r2_score(d_te[target], p) if d_te[target].std() > 0 else np.nan
    mae = mean_absolute_error(d_te[target], p)
    corr = pearsonr(d_te[target], p)[0] if d_te[target].std() > 0 and np.std(p) > 0 else np.nan
    theta_gn_1sd = np.nan
    if "tasa_gn_nm3_min" in A:
        theta_gn_1sd = float(m.theta[A.index("tasa_gn_nm3_min")]) * float(d_tr["tasa_gn_nm3_min"].std())
    return m, p, r2, mae, corr, theta_gn_1sd


def etapa_reentrenamiento() -> None:
    df = L.cargar_df()
    dF = L.base_fase(df, "Fusión")
    orden = mp.orden_cronologico_batches(dF)
    n = len(orden)
    filas = []
    for clave in ("sn_dep", "dT"):
        target = L.TARGETS["Fusión"][clave]
        S, A = L.ESTADO[("Fusión", clave)], L.PALANCAS[("Fusión", clave)]
        signo = L.SIGNOS.get(("Fusión", clave))
        need = list(dict.fromkeys(S + A + [target, "Batch"]))

        # (a)/(b) walk-forward sobre TODO: bloques de 30, ventana creciente y ventana movil 150
        for politica, ventana in [("expandido_todo", None), (f"ventana_movil_{VENTANA_MOVIL}", VENTANA_MOVIL)]:
            reales_all, preds_all = [], []
            for i in range(BLOQUE_WF, n, BLOQUE_WF):
                test_b = set(orden[i:i + BLOQUE_WF])
                train_b = set(orden[:i]) if ventana is None else set(orden[max(0, i - ventana):i])
                d_tr = dF.loc[dF.Batch.isin(train_b), need].dropna().reset_index(drop=True)
                d_te = dF.loc[dF.Batch.isin(test_b), need].dropna().reset_index(drop=True)
                if len(d_tr) < 50 or len(d_te) < 5:
                    continue
                t0 = time.time()
                _, p, r2, mae, corr, theta_gn = _fit_eval(d_tr, d_te, S, A, target, signo)
                filas.append({"clave": clave, "politica": politica, "bloque_ini": i, "n_train": len(d_tr),
                              "n_test": len(d_te), "r2": r2, "mae": mae, "pearson": corr, "theta_gn_1sd": theta_gn})
                reales_all.append(d_te[target].to_numpy()); preds_all.append(p)
                log(f"  {clave}/{politica} bloque_ini={i:3d} n_tr={len(d_tr):4d} n_te={len(d_te):3d} "
                    f"r2={r2:+.3f} ({time.time()-t0:.1f}s)")
            if reales_all:
                ra, pa = np.concatenate(reales_all), np.concatenate(preds_all)
                filas.append({"clave": clave, "politica": politica, "bloque_ini": "GLOBAL", "n_train": np.nan,
                              "n_test": len(ra), "r2": r2_score(ra, pa), "mae": mean_absolute_error(ra, pa),
                              "pearson": pearsonr(ra, pa)[0], "theta_gn_1sd": np.nan})
                log(f"  {clave}/{politica} GLOBAL: r2={r2_score(ra, pa):+.3f} n={len(ra)}")

        # (c) reentrenamiento progresivo cada 10 batches DENTRO del lockbox (parte de DEV completo)
        dev, lockbox = mp.split_dev_lockbox(dF)
        orden_dev = [b for b in orden if b in dev]
        orden_lb = [b for b in orden if b in lockbox]
        d_dev_full = dF.loc[dF.Batch.isin(set(orden_dev)), need].dropna().reset_index(drop=True)
        bloques_lb = [orden_lb[i:i + BLOQUE_LB] for i in range(0, len(orden_lb), BLOQUE_LB)]
        vistos: list = []
        reales_p, preds_p = [], []
        for bl in bloques_lb:
            d_vistos = dF.loc[dF.Batch.isin(set(vistos)), need].dropna() if vistos else d_dev_full.iloc[:0]
            d_tr = pd.concat([d_dev_full, d_vistos], ignore_index=True) if len(d_vistos) else d_dev_full
            d_te = dF.loc[dF.Batch.isin(set(bl)), need].dropna().reset_index(drop=True)
            if len(d_te) == 0:
                continue
            t0 = time.time()
            _, p, r2, mae, corr, theta_gn = _fit_eval(d_tr, d_te, S, A, target, signo)
            filas.append({"clave": clave, "politica": "progresivo_lockbox_10", "bloque_ini": bl[0], "n_train": len(d_tr),
                          "n_test": len(d_te), "r2": r2, "mae": mae, "pearson": corr, "theta_gn_1sd": theta_gn})
            reales_p.append(d_te[target].to_numpy()); preds_p.append(p); vistos.extend(bl)
            log(f"  {clave}/progresivo bloque={bl[0]} n_tr={len(d_tr):4d} n_te={len(d_te):3d} r2={r2:+.3f} ({time.time()-t0:.1f}s)")
        if reales_p:
            ra, pa = np.concatenate(reales_p), np.concatenate(preds_p)
            filas.append({"clave": clave, "politica": "progresivo_lockbox_10", "bloque_ini": "GLOBAL", "n_train": np.nan,
                          "n_test": len(ra), "r2": r2_score(ra, pa), "mae": mean_absolute_error(ra, pa),
                          "pearson": pearsonr(ra, pa)[0], "theta_gn_1sd": np.nan})
            log(f"  {clave}/progresivo GLOBAL: r2={r2_score(ra, pa):+.3f} n={len(ra)}")

        # (d) referencia: fijo DEV -> lockbox completo (mismo enfoque v6)
        d_lb_full = dF.loc[dF.Batch.isin(set(orden_lb)), need].dropna().reset_index(drop=True)
        if len(d_dev_full) and len(d_lb_full):
            _, _, r2, mae, corr, theta_gn = _fit_eval(d_dev_full, d_lb_full, S, A, target, signo)
            filas.append({"clave": clave, "politica": "fijo_DEV", "bloque_ini": "GLOBAL", "n_train": len(d_dev_full),
                          "n_test": len(d_lb_full), "r2": r2, "mae": mae, "pearson": corr, "theta_gn_1sd": theta_gn})
            log(f"  {clave}/fijo_DEV: r2={r2:+.3f}")

    pd.DataFrame(filas).to_csv(HERE / "e9_04_reentrenamiento.csv", index=False)
    log("FIN reentrenamiento")


# =============================================================================
# 3) targets robustos a la deriva
# =============================================================================

def _construir_targets_escalon(dF: pd.DataFrame) -> pd.DataFrame:
    dF = dF.copy()
    dF["_t_frac_ret"] = np.exp(-dF["m6_ln_sn_dep"])
    dF["_t_sn_ext_norm"] = dF["m6_sn_extraido_kg"] / dF["feed_Sn_kgf"].where(dF["feed_Sn_kgf"] > 20)
    dF["_t_tasa_sn_dep"] = dF["m6_ln_sn_dep"] / dF["duracion_plan_min"].where(dF["duracion_plan_min"] > 5)
    return dF


def _agregados_batch_targets(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for b, g in df.groupby("Batch"):
        gf = g[g["fase_proceso"] == "Fusión"]
        filas.append({
            "Batch": b,
            "F_sum_d_ley_sn": float(gf["d_ley_sn_escoria_pct"].sum(min_count=1)),
            "F_sum_d_ley_feo": float(gf["d_ley_feo_escoria_pct"].sum(min_count=1)),
            "F_feed_Sn_total_kg": float(gf["feed_Sn_kgf"].sum(min_count=1)),
            "F_duracion_total_min": float(gf["duracion_plan_min"].sum(min_count=1)),
        })
    agg = pd.DataFrame(filas).set_index("Batch")
    out = batch.join(agg, how="left")
    out["F_frac_retenida_batch"] = np.exp(-out["F_sum_m6_ln_sn_dep"])
    out["F_sn_ext_norm_batch"] = out["F_m6_sn_ext_kg"] / out["F_feed_Sn_total_kg"]
    out["F_tasa_sn_dep_batch"] = out["F_sum_m6_ln_sn_dep"] / out["F_duracion_total_min"]
    return out


def _candidatos(signo_sn: dict) -> list[dict]:
    signo_ret = {k: -v for k, v in signo_sn.items()}
    signo_libre = {k: 0 for k in signo_sn}
    signo_lnirf = mp5.SIGNO_TEORICO_V5[("Fusión", "d_lnirf")]
    return [
        dict(id="a_ln_sn_dep", desc="agotamiento log (referencia v6)", col="m6_ln_sn_dep", signo=signo_sn, batch_col="F_sum_m6_ln_sn_dep"),
        dict(id="b_frac_retenida", desc="fraccion de Sn disponible retenida (nivel, no log)", col="_t_frac_ret", signo=signo_ret, batch_col="F_frac_retenida_batch"),
        dict(id="c_sn_ext_norm", desc="Sn extraido [kg] / Sn alimentado del escalon", col="_t_sn_ext_norm", signo=signo_sn, batch_col="F_sn_ext_norm_batch"),
        dict(id="d_dley_sn", desc="Delta %Sn escoria (target v1)", col="d_ley_sn_escoria_pct", signo=signo_libre, batch_col="F_sum_d_ley_sn"),
        dict(id="e1_lnirf_nivel", desc="ln IRF al cierre del escalon (nivel)", col="v5_lnirf", signo=signo_lnirf, batch_col="F_lnirf_fin"),
        dict(id="e2_d_lnirf", desc="Delta ln IRF del escalon", col="v5_d_lnirf", signo=signo_lnirf, batch_col="F_sum_d_lnirf"),
        dict(id="f_dley_feo", desc="Delta %FeO escoria", col="d_ley_feo_escoria_pct", signo=signo_libre, batch_col="F_sum_d_ley_feo"),
        dict(id="g_tasa_sn_dep", desc="agotamiento log por minuto de escalon (forma en tasa)", col="_t_tasa_sn_dep", signo=signo_sn, batch_col="F_tasa_sn_dep_batch"),
    ]


def etapa_targets() -> None:
    df = L.cargar_df()
    dF = _construir_targets_escalon(L.base_fase(df, "Fusión"))
    batch = _agregados_batch_targets(df, L.construir_batch(df))
    S, A = L.ESTADO[("Fusión", "sn_dep")], L.PALANCAS[("Fusión", "sn_dep")]
    L.asegurar_seguras(S + A)
    filas = []
    for cand in _candidatos(L.SIGNOS[("Fusión", "sn_dep")]):
        col, signo = cand["col"], cand["signo"]
        need = list(dict.fromkeys(S + A + [col, "Batch"]))
        n_validos = dF[need].dropna().shape[0]
        if n_validos < 100:
            log(f"{cand['id']}: insuficiente n={n_validos}, se omite"); continue
        t0 = time.time()
        met, tabla, _ = L.evaluar_plm_signos(dF, S, A, col, signo_teorico=signo, n_boot=200)
        fila = {"id": cand["id"], "descripcion": cand["desc"], "columna": col,
                "r2_oof": met["r2_oof"], "mae_oof": met["mae_oof"], "r2_lockbox": met["r2_lockbox"],
                "mae_lockbox": met["mae_lockbox"], "n_dev": met["n_dev"], "n_lockbox": met["n_lockbox"],
                "sd_dev": met["sd_target_dev"]}
        tabla = tabla.set_index("palanca")
        for p in A:
            fila[f"theta_{p}_1sd"] = float(tabla.loc[p, "theta_1sd"])
            fila[f"theta_{p}_lo"] = float(tabla.loc[p, "ci_lo_1sd"])
            fila[f"theta_{p}_hi"] = float(tabla.loc[p, "ci_hi_1sd"])
            fila[f"theta_{p}_significativo"] = bool(tabla.loc[p, "significativo"])
        bcol = cand["batch_col"]
        if bcol in batch.columns:
            r = L.bateria_batch(batch, bcol)
            fila["rho_dev_kpi"] = r.get("rho_dev"); fila["p_dev_kpi"] = r.get("p_dev")
            fila["rho_lockbox_kpi"] = r.get("rho_lockbox"); fila["rho_total_kpi"] = r.get("rho_total")
            fila["ols_dev_kpi_pp_sd"] = r.get("ols_dev"); fila["ols_p_dev_kpi"] = r.get("ols_p_dev")
        filas.append(fila)
        log(f"{cand['id']:20s} r2_oof={met['r2_oof']:+.3f} r2_lockbox={met['r2_lockbox']:+.3f} "
            f"rho_kpi_dev={fila.get('rho_dev_kpi')}  ({time.time()-t0:.1f}s)")
    pd.DataFrame(filas).to_csv(HERE / "e9_04_targets.csv", index=False)
    log("FIN targets")


# =============================================================================
# 4) valor del estado al final de Fusion (fila R0 de cada batch)
# =============================================================================

ESTADO_FINAL_F = ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "basicidad_B2_prev", "indice_irf_prev",
                  "temperatura_horno_celsius_prev", "m6_masa_kg_prev", "m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev",
                  "m6_avance_prev", "m6_resto_frac_prev", "cum_feed_Carbon_kg_prev", "relacion_C_cum_Sn_cum_prev"]
DEP_VARS_ESTADO_FINAL = ["R_sum_m6_ln_feo_ret", "R_sum_m6_ln_sn_dep", "J_R_batch", L.KPI_PRINCIPAL]


def _hgb_ef(seed=SEED):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=15,
                                         l2_regularization=1.0, random_state=seed)


def etapa_estado_final() -> None:
    df = L.cargar_df()
    dR = L.base_fase(df, "Reducción")
    L.asegurar_seguras(ESTADO_FINAL_F)
    batch = L.construir_batch(df)
    r0 = dR.loc[dR["orden_escalon_fase"] == 0, ["Batch"] + ESTADO_FINAL_F].drop_duplicates("Batch").set_index("Batch")
    batch = batch.join(r0, how="left")
    batch["J_R_batch"] = L.CONFIG["pp_kpi_por_unidad_sdi"] * (
        batch["R_sum_m6_ln_sn_dep"] + L.CONFIG["w_sdi"] * batch["R_sum_m6_ln_feo_ret"])

    # m6_resto_frac_prev = 1 - 1.270*ley_sn_escoria_pct_prev/100 - ley_feo_escoria_pct_prev/100 (formula de cierre
    # fisico v6, candidate_win_model_v6.md SS1): es una combinacion lineal EXACTA de las dos leyes -> VIF ~1e15 si
    # se incluyen las tres juntas en OLS. Se excluye de la regresion OLS (queda en HGB/importancia, donde la
    # colinealidad no rompe el ajuste, solo reparte la importancia entre las tres).
    ESTADO_FINAL_OLS = [c for c in ESTADO_FINAL_F if c != "m6_resto_frac_prev"]
    filas_ols, filas_hgb, filas_imp = [], [], []
    regresores = ESTADO_FINAL_F + L.CONTROLES_BATCH
    regresores_ols = ESTADO_FINAL_OLS + L.CONTROLES_BATCH
    for dep in DEP_VARS_ESTADO_FINAL:
        need = regresores + [dep, "es_dev", "es_lockbox"]
        d = batch[need].dropna()
        log(f"=== {dep}: n_total={len(d)} ===")
        for muestra, sub in [("dev", d[d.es_dev]), ("total", d)]:
            if len(sub) < 30:
                continue
            X = sub[regresores_ols]
            Xz = (X - X.mean()) / X.std()
            import statsmodels.api as sm
            res = sm.OLS(sub[dep].to_numpy(), sm.add_constant(Xz)).fit(cov_type="HC3")
            ci = res.conf_int()
            for v in ESTADO_FINAL_OLS:
                filas_ols.append({"dep": dep, "muestra": muestra, "variable": v,
                                  "coef_1sd": float(res.params[v]), "p": float(res.pvalues[v]),
                                  "ci_lo": float(ci.loc[v, 0]), "ci_hi": float(ci.loc[v, 1]),
                                  "n": len(sub), "r2adj": float(res.rsquared_adj)})
            log(f"  OLS {muestra}: n={len(sub)} r2adj={res.rsquared_adj:.3f}")

        d_dev = d[d.es_dev].reset_index(drop=True)
        d_lb = d[d.es_lockbox].reset_index(drop=True)
        Xd = d_dev[regresores]
        kf = KFold(5, shuffle=True, random_state=SEED)
        oof = np.full(len(d_dev), np.nan)
        for tr, te in kf.split(Xd):
            h = _hgb_ef().fit(Xd.iloc[tr], d_dev[dep].iloc[tr])
            oof[te] = h.predict(Xd.iloc[te])
        r2_oof = r2_score(d_dev[dep], oof) if d_dev[dep].std() > 0 else np.nan
        h_full = _hgb_ef().fit(Xd, d_dev[dep])
        r2_lb = np.nan
        if len(d_lb) > 10:
            r2_lb = r2_score(d_lb[dep], h_full.predict(d_lb[regresores])) if d_lb[dep].std() > 0 else np.nan
        filas_hgb.append({"dep": dep, "r2_oof": r2_oof, "r2_lockbox": r2_lb, "n_dev": len(d_dev), "n_lockbox": len(d_lb)})
        log(f"  HGB: r2_oof={r2_oof:.3f} r2_lockbox={r2_lb}")

        Xtr, Xte, ytr, yte = train_test_split(Xd, d_dev[dep], test_size=0.25, random_state=SEED)
        h2 = _hgb_ef().fit(Xtr, ytr)
        imp = permutation_importance(h2, Xte, yte, n_repeats=20, random_state=SEED, n_jobs=1)
        for v, m_, s_ in zip(Xd.columns, imp.importances_mean, imp.importances_std):
            filas_imp.append({"dep": dep, "variable": v, "importancia_media": float(m_), "importancia_sd": float(s_),
                              "es_estado_final_F": v in ESTADO_FINAL_F})

    pd.DataFrame(filas_ols).to_csv(HERE / "e9_04_valor_estado_final_ols.csv", index=False)
    pd.DataFrame(filas_hgb).to_csv(HERE / "e9_04_valor_estado_final_hgb.csv", index=False)
    pd.DataFrame(filas_imp).to_csv(HERE / "e9_04_valor_estado_final_importancia.csv", index=False)

    v = pd.DataFrame(filas_ols)
    vkpi = v[(v.dep == L.KPI_PRINCIPAL) & (v.muestra == "dev")][["variable", "coef_1sd", "p", "ci_lo", "ci_hi"]]
    vkpi = vkpi.rename(columns={"coef_1sd": "pp_kpi_por_sd"})
    vkpi.to_csv(HERE / "e9_04_valor_estado_final.csv", index=False)
    log("FIN estado_final")


# =============================================================================
# 5) canal polvo
# =============================================================================

def _agregados_polvo(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for b, g in df.groupby("Batch"):
        gf = g[g["fase_proceso"] == "Fusión"]
        gas_gn = float(gf["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum(min_count=1))
        gas_o2 = float(gf["volumen_o2_inyectado_lanza_escalon_nm3"].sum(min_count=1))
        gas_aire = float(gf["volumen_aire_inyectado_lanza_escalon_nm3"].sum(min_count=1))
        carga_kg = float(gf["feed_total_kgh"].sum(min_count=1))
        filas.append({"Batch": b, "F_gn_total_nm3": gas_gn, "F_o2_total_nm3": gas_o2, "F_aire_total_nm3": gas_aire,
                      "F_gas_total_nm3": gas_gn + gas_o2 + gas_aire, "F_carga_total_t": carga_kg / 1000})
    agg = pd.DataFrame(filas).set_index("Batch")
    out = batch.join(agg, how="left")
    for col in ("F_gn_total_nm3", "F_o2_total_nm3", "F_aire_total_nm3", "F_gas_total_nm3"):
        out[f"{col}_por_t"] = out[col] / out["F_carga_total_t"]
    return out


def etapa_polvo() -> None:
    df = L.cargar_df()
    batch = _agregados_polvo(df, L.construir_batch(df))
    filas = []
    for kpi in ("f_polvo", L.KPI_PRINCIPAL):
        for col in ("F_gn_total_nm3_por_t", "F_o2_total_nm3_por_t", "F_aire_total_nm3_por_t",
                    "F_gas_total_nm3_por_t", "T_media_F"):
            r = L.bateria_batch(batch, col, kpi=kpi)
            r["kpi"] = kpi
            filas.append(r)
            log(f"{kpi} ~ {col}: ols_dev={r.get('ols_dev')} p={r.get('ols_p_dev')} rho_dev={r.get('rho_dev')}")
    pd.DataFrame(filas).to_csv(HERE / "e9_04_polvo_batch.csv", index=False)

    dF = L.base_fase(df, "Fusión")
    S, A = L.ESTADO[("Fusión", "sn_dep")], L.PALANCAS[("Fusión", "dT")]   # incluye tasa_o2_nm3_min
    L.asegurar_seguras(S + A)
    filas2 = []
    for proxy in ("d_temperatura_gas_pre_bhf_celsius", "d_tiro_horno_pct"):
        if proxy not in dF.columns:
            log(f"{proxy}: no existe en dF, se omite"); continue
        need = list(dict.fromkeys(S + A + [proxy, "Batch"]))
        n_validos = dF[need].dropna().shape[0]
        if n_validos < 100:
            log(f"{proxy}: insuficiente n={n_validos}, se omite"); continue
        signo = {a: 0 for a in A}   # sin restriccion de signo: se busca evidencia, no se impone
        t0 = time.time()
        met, tabla, _ = L.evaluar_plm_signos(dF, S, A, proxy, signo_teorico=signo, n_boot=200)
        tabla = tabla.reset_index()
        tabla["proxy"] = proxy; tabla["r2_oof"] = met["r2_oof"]; tabla["r2_lockbox"] = met["r2_lockbox"]
        filas2.append(tabla)
        log(f"{proxy}: r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f} ({time.time()-t0:.1f}s)")
    if filas2:
        pd.concat(filas2, ignore_index=True).to_csv(HERE / "e9_04_polvo_escalon.csv", index=False)
    log("FIN polvo")


# =============================================================================
# 6) resumen
# =============================================================================

def etapa_resumen() -> None:
    filas = []

    def _leer(nombre):
        p = HERE / nombre
        return pd.read_csv(p) if p.exists() else None

    diag = _leer("e9_04_diag_metricas.csv")
    if diag is not None:
        for _, r in diag.iterrows():
            filas.append({"bloque": "diagnostico_PLM_v6", "item": r["clave"], "r2_oof": r["r2_oof"],
                          "r2_lockbox": r["r2_lockbox"], "n_dev": r["n_dev"], "n_lockbox": r["n_lockbox"]})

    recal = _leer("e9_04_diag_recalibracion.csv")
    if recal is not None:
        for _, r in recal.iterrows():
            filas.append({"bloque": "recalibracion_intercepto_lockbox", "item": f"{r['clave']}/{r['politica']}",
                          "r2_oof": np.nan, "r2_lockbox": r["r2"], "n_dev": np.nan, "n_lockbox": r["n_eval_filas"]})

    reen = _leer("e9_04_reentrenamiento.csv")
    if reen is not None:
        glob = reen[reen.bloque_ini == "GLOBAL"]
        for _, r in glob.iterrows():
            filas.append({"bloque": "reentrenamiento_walkforward", "item": f"{r['clave']}/{r['politica']}",
                          "r2_oof": np.nan, "r2_lockbox": r["r2"], "n_dev": np.nan, "n_lockbox": r["n_test"]})

    tgt = _leer("e9_04_targets.csv")
    if tgt is not None:
        for _, r in tgt.iterrows():
            filas.append({"bloque": "targets_candidatos", "item": r["id"], "r2_oof": r["r2_oof"],
                          "r2_lockbox": r["r2_lockbox"], "n_dev": r["n_dev"], "n_lockbox": r["n_lockbox"]})

    ef = _leer("e9_04_valor_estado_final_hgb.csv")
    if ef is not None:
        for _, r in ef.iterrows():
            filas.append({"bloque": "estado_final_F_predice", "item": r["dep"], "r2_oof": r["r2_oof"],
                          "r2_lockbox": r["r2_lockbox"], "n_dev": r["n_dev"], "n_lockbox": r["n_lockbox"]})

    pd.DataFrame(filas).to_csv(HERE / "e9_04_resumen.csv", index=False)
    log(f"FIN resumen ({len(filas)} filas)")


# =============================================================================
# main
# =============================================================================

ETAPAS = {"diagnostico": etapa_diagnostico, "reentrenamiento": etapa_reentrenamiento, "targets": etapa_targets,
          "estado_final": etapa_estado_final, "polvo": etapa_polvo, "resumen": etapa_resumen}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=list(ETAPAS) + ["todas"])
    args = ap.parse_args()
    if args.etapa == "todas":
        for nombre, fn in ETAPAS.items():
            log(f"##### etapa {nombre} #####")
            fn()
    else:
        ETAPAS[args.etapa]()
    log(f"TOTAL {time.time() - T0:.1f}s")
