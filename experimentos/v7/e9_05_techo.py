"""E9-05 (iteracion 9, 2026-09-16): techo de ruido, curva de aprendizaje y bake-off de algoritmos para
los targets v6 de Reduccion (m6_ln_sn_dep, m6_ln_feo_ret) y Fusion (m6_ln_sn_dep).

Objetivo: poder afirmar con evidencia "no hay modelo mejor con estos datos" o cuantificar el margen
restante (H5 de ITERACION_9_diseno.md). Cinco etapas independientes (uso: python e9_05_techo.py <etapa>):

  techo     Monte Carlo del ruido de ensayo (>=200 replicas): perturba leyes, recalcula la masa v6
            (masa_v6.agregar_masa_v6) y los targets; var_ruido = E[(y_pert - y)^2]; R2_max = 1 -
            var_ruido/var_total; sensibilidad a niveles de ruido; autocorrelacion lag-1 de residuos OOF.
  curva     Curva de aprendizaje (n_batches in {50,100,150,200,250,299}, 3 repeticiones) del PLM v6 y de
            HGB estado+palancas, ambos targets de Reduccion; ajuste de saturacion R2 = a - b/n.
  bakeoff   Mismas filas/features (estado+palancas), OOF GroupKFold(5) DEV + lockbox, para los 3 targets:
            HGB, HGB con grid de hiperparametros, XGBoost, RandomForest, ExtraTrees, Ridge/Lasso con
            cuadraticos, kNN, MLP, stacking (promedio y meta-Ridge), PLM v6; HGB y PLM con 5 semillas.
  desglose  R2/MAE OOF y lockbox del PLM v6 por orden de escalon, tercio cronologico de DEV y cuartil de
            m6_avance_prev; diagnostico de residuos (normalidad, heterocedasticidad, estructura en palancas).
  resumen   Tabla y memo de veredicto: techo vs alcanzado, mejor algoritmo (IC bootstrap del delta R2),
            saturacion de la curva, frase de veredicto por target.

Reglas de entorno: `.venv/Scripts/python.exe`, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3, sin joblib/loky
(salvo xgboost n_jobs=3); sklearn con n_jobs=1. OOF GroupKFold(5) por Batch sobre DEV = criterio; lockbox
solo reporte.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_V7 = Path(__file__).resolve().parent
if str(RAIZ_V7) not in sys.path:
    sys.path.insert(0, str(RAIZ_V7))
import v7_lib as L  # noqa: E402
import masa_v6  # noqa: E402

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
SEEDS5 = [42, 1, 2, 3, 4]
N_SPLITS = 5

# (fase, clave, target, etiqueta corta)
TARGET_DEFS = [
    ("Reducción", "sn_dep", "m6_ln_sn_dep", "R_sn_dep"),
    ("Reducción", "feo_ret", "m6_ln_feo_ret", "R_feo_ret"),
    ("Fusión", "sn_dep", "m6_ln_sn_dep", "F_sn_dep"),
]

CSV = {
    "techo": RAIZ_V7 / "e9_05_techo.csv",
    "curva": RAIZ_V7 / "e9_05_curva.csv",
    "bakeoff": RAIZ_V7 / "e9_05_bakeoff.csv",
    "bakeoff_grid": RAIZ_V7 / "e9_05_bakeoff_grid_hgb.csv",
    "desglose": RAIZ_V7 / "e9_05_desglose.csv",
    "resumen": RAIZ_V7 / "e9_05_resumen.csv",
}


def log(t0: float, *a) -> None:
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def cargar():
    df = L.cargar_df()
    dR = L.base_fase(df, "Reducción")
    dF = L.base_fase(df, "Fusión")
    dev, lb = L.split(df)
    return df, dR, dF, dev, lb


def base_de_fase(fase: str, dR: pd.DataFrame, dF: pd.DataFrame) -> pd.DataFrame:
    return dR if fase == "Reducción" else dF


# =============================================================================
# ETAPA 1: techo de ruido (Monte Carlo)
# =============================================================================
LEY_COLS_G = ["ley_cao_escoria_pct", "ley_sio2_escoria_pct", "ley_al2o3_escoria_pct", "ley_mgo_escoria_pct"]


def _cols_m6(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("m6_")] + [c for c in ("Cx_sn_v6", "Cx_av_v6") if c in df.columns]


def preparar_base_sin_m6(df: pd.DataFrame) -> pd.DataFrame:
    """df ordenado (Batch, fecha_inicio) SIN columnas m6_*/Cx_*_v6: necesario porque
    masa_v6.agregar_masa_v6 hace pd.concat + drop_duplicates(keep='first') y, si el df ya trae m6_*,
    CONGELA los valores viejos en vez de recalcularlos (verificado empiricamente)."""
    return df.drop(columns=_cols_m6(df)).sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)


def perturbar_leyes(df_base: pd.DataFrame, rng: np.random.Generator, sigma_sn: float, sigma_feo: float,
                    sigma_g: float = 0.028) -> pd.DataFrame:
    """Ruido de ensayo log-normal multiplicativo, independiente por fila y por columna."""
    d = df_base.copy()
    n = len(d)
    d["ley_sn_escoria_pct"] = d["ley_sn_escoria_pct"] * np.exp(sigma_sn * rng.standard_normal(n))
    d["ley_feo_escoria_pct"] = d["ley_feo_escoria_pct"] * np.exp(sigma_feo * rng.standard_normal(n))
    for c in LEY_COLS_G:
        d[c] = d[c] * np.exp(sigma_g * rng.standard_normal(n))
    return d


def montecarlo_ruido(df_base: pd.DataFrame, sigma_sn: float, sigma_feo: float, n_rep: int, seed: int = SEED,
                     sigma_g: float = 0.028) -> tuple[pd.DataFrame, dict]:
    """Devuelve (referencia de columnas fijas, {col_target: var_ruido por fila})."""
    rng = np.random.default_rng(seed)
    out0 = masa_v6.agregar_masa_v6(df_base.copy(), masa_v6.PARAMS_DEFAULT)
    y0 = {c: out0[c].to_numpy() for c in ("m6_ln_sn_dep", "m6_ln_feo_ret")}
    n = len(out0)
    sumsq = {c: np.zeros(n) for c in y0}
    cnt = {c: np.zeros(n) for c in y0}
    for _ in range(n_rep):
        dp = perturbar_leyes(df_base, rng, sigma_sn, sigma_feo, sigma_g)
        out = masa_v6.agregar_masa_v6(dp, masa_v6.PARAMS_DEFAULT)
        for c in y0:
            yp = out[c].to_numpy()
            m = np.isfinite(yp) & np.isfinite(y0[c])
            diff = np.where(m, yp - y0[c], 0.0)
            sumsq[c] += diff ** 2
            cnt[c] += m.astype(float)
    var_ruido = {c: np.where(cnt[c] > 0, sumsq[c] / np.maximum(cnt[c], 1), np.nan) for c in y0}
    ref = out0[["Batch", "fecha_inicio", "fase_proceso", "orden_escalon_fase"]].copy()
    for c in y0:
        ref[c] = y0[c]
    return ref, var_ruido


def _plm_residuos(df_fase: pd.DataFrame, S: list[str], A: list[str], target: str, signos: dict | None) -> tuple[dict, pd.DataFrame]:
    met, _, pred = L.evaluar_plm_signos(df_fase, S, A, target, n_boot=0, seed=SEED, signo_teorico=signos)
    return met, pred


def autocorr_lag1_residuos(pred: pd.DataFrame) -> tuple[float, int]:
    """Autocorrelacion lag-1 (Pearson) de los residuos OOF-dev dentro de cada batch, ordenados por
    fecha_inicio. Un ruido de ensayo puro en la masa encadenada produce autocorr ~ -0.5 en Delta."""
    d = pred.loc[pred["conjunto"] == "OOF_dev"].copy()
    d["res"] = d["real"] - d["pred"]
    d = d.sort_values(["Batch", "fecha_inicio"])
    d["res_prev"] = d.groupby("Batch")["res"].shift(1)
    dd = d.dropna(subset=["res", "res_prev"])
    if len(dd) < 20:
        return float("nan"), len(dd)
    return float(np.corrcoef(dd["res"], dd["res_prev"])[0, 1]), len(dd)


def etapa_techo() -> None:
    t0 = time.time()
    df, dR, dF, dev, lb = cargar()
    log(t0, f"df {df.shape}, dev={len(dev)} lb={len(lb)}")
    df_base = preparar_base_sin_m6(df)
    log(t0, f"df_base sin m6_* : {df_base.shape}")

    niveles_sn = [0.01, 0.02, 0.03, 0.04]
    niveles_feo = [0.02, 0.028, 0.04]
    combos = [(s, 0.028) for s in niveles_sn] + [(0.02, f) for f in niveles_feo if f != 0.028]
    N_REP = 200

    filas = []
    # --- PLM OOF (una vez, no depende del nivel de ruido) + autocorrelacion ---
    plm_info = {}
    for fase, clave, target, etq in TARGET_DEFS:
        S, A = L.ESTADO[(fase, clave)], L.PALANCAS[(fase, clave)]
        df_fase = base_de_fase(fase, dR, dF)
        met, pred = _plm_residuos(df_fase, S, A, target, L.SIGNOS.get((fase, clave)))
        ac, n_ac = autocorr_lag1_residuos(pred)
        plm_info[(fase, clave)] = dict(r2_oof=met["r2_oof"], r2_lockbox=met["r2_lockbox"], autocorr=ac, n_autocorr=n_ac)
        log(t0, f"PLM {etq}: r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f} "
                f"autocorr_lag1_res={ac:.3f} (n={n_ac})")

    for sigma_sn, sigma_feo in combos:
        log(t0, f"MC sigma_sn={sigma_sn} sigma_feo={sigma_feo} n_rep={N_REP} ...")
        ref, var_ruido = montecarlo_ruido(df_base, sigma_sn, sigma_feo, N_REP, seed=SEED)
        es_base = np.isclose(sigma_sn, 0.02) and np.isclose(sigma_feo, 0.028)
        for fase, clave, target, etq in TARGET_DEFS:
            col = target
            mask_fase = (ref["fase_proceso"] == fase).to_numpy()
            mask_dev = ref["Batch"].isin(dev).to_numpy()
            y0 = ref[col].to_numpy()
            valido = mask_fase & mask_dev & np.isfinite(y0) & np.isfinite(var_ruido[col])
            if valido.sum() < 20:
                continue
            var_t = float(np.var(y0[valido]))
            var_r_global = float(np.mean(var_ruido[col][valido]))
            r2max_global = 1 - var_r_global / var_t if var_t > 0 else np.nan
            info = plm_info[(fase, clave)]
            filas.append(dict(target=etq, fase=fase, clave=clave, orden="todos", sigma_sn=sigma_sn, sigma_feo=sigma_feo,
                              n=int(valido.sum()), var_total=var_t, var_ruido=var_r_global, r2_max=r2max_global,
                              r2_oof_plm=info["r2_oof"], r2_lockbox_plm=info["r2_lockbox"],
                              r2_relativa_oof=info["r2_oof"] / r2max_global if r2max_global > 0 else np.nan,
                              autocorr_lag1_res_oof=info["autocorr"] if es_base else np.nan,
                              n_autocorr=info["n_autocorr"] if es_base else np.nan))
            if es_base:
                for orden, sub in ref.loc[valido].groupby("orden_escalon_fase"):
                    idx = sub.index.to_numpy()
                    yv = y0[idx]
                    vt = float(np.var(yv))
                    vr = float(np.mean(var_ruido[col][idx]))
                    r2m = 1 - vr / vt if vt > 0 else np.nan
                    filas.append(dict(target=etq, fase=fase, clave=clave, orden=int(orden), sigma_sn=sigma_sn,
                                      sigma_feo=sigma_feo, n=len(idx), var_total=vt, var_ruido=vr, r2_max=r2m,
                                      r2_oof_plm=np.nan, r2_lockbox_plm=np.nan, r2_relativa_oof=np.nan,
                                      autocorr_lag1_res_oof=np.nan, n_autocorr=np.nan))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CSV["techo"], index=False)
    log(t0, f"guardado {CSV['techo']} ({len(tabla)} filas)")
    print("FIN techo", flush=True)


# =============================================================================
# ETAPA 2: curva de aprendizaje
# =============================================================================

def _plm_oof_lockbox(d_dev: pd.DataFrame, d_lb: pd.DataFrame, S: list[str], A: list[str], target: str,
                     signos: dict | None, n_splits: int = N_SPLITS, seed: int = SEED) -> dict:
    n_splits_ef = min(n_splits, d_dev["Batch"].nunique())
    oof = np.full(len(d_dev), np.nan)
    if n_splits_ef >= 2:
        for tr, te in GroupKFold(n_splits_ef).split(d_dev, d_dev[target], d_dev["Batch"]):
            m = L.ModeloPLMSignos(S, A, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
            oof[te] = m.predict(d_dev.iloc[te])
    m_full = L.ModeloPLMSignos(S, A, target, signos).fit(d_dev, n_boot=0, seed=seed)
    p_lb = m_full.predict(d_lb) if len(d_lb) else np.array([])
    return dict(r2_oof=r2_score(d_dev[target], oof) if np.isfinite(oof).sum() > 10 else np.nan,
               mae_oof=mean_absolute_error(d_dev[target], oof) if np.isfinite(oof).sum() > 10 else np.nan,
               r2_lockbox=r2_score(d_lb[target], p_lb) if len(d_lb) > 10 else np.nan,
               mae_lockbox=mean_absolute_error(d_lb[target], p_lb) if len(d_lb) > 10 else np.nan)


def _hgb_oof_lockbox(d_dev: pd.DataFrame, d_lb: pd.DataFrame, X: list[str], target: str,
                     n_splits: int = N_SPLITS, seed: int = SEED, max_iter: int = 300) -> dict:
    n_splits_ef = min(n_splits, d_dev["Batch"].nunique())
    oof = np.full(len(d_dev), np.nan)
    if n_splits_ef >= 2:
        for tr, te in GroupKFold(n_splits_ef).split(d_dev[X], d_dev[target], d_dev["Batch"]):
            h = L.hgb_nuisance(seed, max_iter).fit(d_dev[X].iloc[tr], d_dev[target].iloc[tr])
            oof[te] = h.predict(d_dev[X].iloc[te])
    h_full = L.hgb_nuisance(seed, max_iter).fit(d_dev[X], d_dev[target])
    p_lb = h_full.predict(d_lb[X]) if len(d_lb) else np.array([])
    return dict(r2_oof=r2_score(d_dev[target], oof) if np.isfinite(oof).sum() > 10 else np.nan,
               mae_oof=mean_absolute_error(d_dev[target], oof) if np.isfinite(oof).sum() > 10 else np.nan,
               r2_lockbox=r2_score(d_lb[target], p_lb) if len(d_lb) > 10 else np.nan,
               mae_lockbox=mean_absolute_error(d_lb[target], p_lb) if len(d_lb) > 10 else np.nan)


def etapa_curva() -> None:
    t0 = time.time()
    df, dR, dF, dev, lb = cargar()
    tamanos = [50, 100, 150, 200, 250, 299]
    filas = []
    for fase, clave, target, etq in TARGET_DEFS:
        if fase != "Reducción":
            continue
        S, A = L.ESTADO[(fase, clave)], L.PALANCAS[(fase, clave)]
        X = S + A
        signos = L.SIGNOS.get((fase, clave))
        d_dev_full, d_lb = L.preparar(dR, X, target)
        batches_pool = sorted(d_dev_full["Batch"].unique())
        n_total = len(batches_pool)
        log(t0, f"{etq}: batches DEV disponibles={n_total}, n_dev_full={len(d_dev_full)}, n_lb={len(d_lb)}")
        for n_b in tamanos:
            n_b_ef = min(n_b, n_total)
            reps = 1 if n_b_ef >= n_total else 3
            for rep in range(reps):
                rng = np.random.default_rng(1000 * n_b + rep)
                muestra = set(rng.choice(batches_pool, size=n_b_ef, replace=False))
                d = d_dev_full[d_dev_full["Batch"].isin(muestra)].reset_index(drop=True)
                if d["Batch"].nunique() < 5 or len(d) < 30:
                    continue
                t1 = time.time()
                r_plm = _plm_oof_lockbox(d, d_lb, S, A, target, signos)
                dt_plm = time.time() - t1
                t1 = time.time()
                r_hgb = _hgb_oof_lockbox(d, d_lb, X, target)
                dt_hgb = time.time() - t1
                filas.append(dict(target=etq, n_batches=n_b_ef, rep=rep, modelo="PLM_v6", **r_plm, tiempo_s=dt_plm))
                filas.append(dict(target=etq, n_batches=n_b_ef, rep=rep, modelo="HGB_estado_palancas", **r_hgb, tiempo_s=dt_hgb))
                log(t0, f"{etq} n={n_b_ef} rep={rep}: PLM r2_oof={r_plm['r2_oof']:.3f} r2_lb={r_plm['r2_lockbox']:.3f} "
                        f"({dt_plm:.1f}s) | HGB r2_oof={r_hgb['r2_oof']:.3f} r2_lb={r_hgb['r2_lockbox']:.3f} ({dt_hgb:.1f}s)")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CSV["curva"], index=False)
    log(t0, f"guardado {CSV['curva']} ({len(tabla)} filas)")

    # ajuste de saturacion R2 = a - b/n (minimos cuadrados sobre el promedio por n_batches)
    filas_ajuste = []
    for (etq, modelo), g in tabla.groupby(["target", "modelo"]):
        prom = g.groupby("n_batches")["r2_oof"].mean().dropna()
        if len(prom) < 3:
            continue
        n_arr, y_arr = prom.index.to_numpy(float), prom.to_numpy(float)
        Xd = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
        coef, *_ = np.linalg.lstsq(Xd, y_arr, rcond=None)
        a, b = coef
        n_max = int(prom.index.max())
        r2_n_max = float(prom.loc[n_max])
        filas_ajuste.append(dict(target=etq, modelo=modelo, a_asintota=float(a), b=float(b), n_max=n_max,
                                 r2_oof_n_max=r2_n_max, gap_asintota=float(a - r2_n_max)))
    ajuste = pd.DataFrame(filas_ajuste)
    ajuste.to_csv(RAIZ_V7 / "e9_05_curva_ajuste.csv", index=False)
    log(t0, "ajuste de saturacion:\n" + ajuste.to_string())
    print("FIN curva", flush=True)


# =============================================================================
# ETAPA 3: bake-off de algoritmos
# =============================================================================

def oof_lockbox_generic(build_model, d_dev: pd.DataFrame, d_lb: pd.DataFrame, X: list[str], target: str,
                        n_splits: int = N_SPLITS) -> dict:
    n_splits_ef = min(n_splits, d_dev["Batch"].nunique())
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits_ef).split(d_dev[X], d_dev[target], d_dev["Batch"]):
        m = build_model()
        m.fit(d_dev[X].iloc[tr], d_dev[target].iloc[tr])
        oof[te] = m.predict(d_dev[X].iloc[te])
    m_full = build_model()
    m_full.fit(d_dev[X], d_dev[target])
    p_lb = m_full.predict(d_lb[X]) if len(d_lb) else np.array([])
    return dict(r2_oof=float(r2_score(d_dev[target], oof)), mae_oof=float(mean_absolute_error(d_dev[target], oof)),
               r2_lockbox=float(r2_score(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan,
               mae_lockbox=float(mean_absolute_error(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan,
               oof=oof, pred_lb=p_lb, modelo_full=m_full)


def etapa_bakeoff() -> None:
    t0 = time.time()
    df, dR, dF, dev, lb = cargar()
    filas = []
    filas_grid = []
    for fase, clave, target, etq in TARGET_DEFS:
        S, A = L.ESTADO[(fase, clave)], L.PALANCAS[(fase, clave)]
        X = S + A
        signos = L.SIGNOS.get((fase, clave))
        df_fase = base_de_fase(fase, dR, dF)
        d_dev, d_lb = L.preparar(df_fase, X, target)
        log(t0, f"=== {etq}: n_dev={len(d_dev)} n_lb={len(d_lb)} n_features={len(X)} ===")

        def fila(algoritmo, config, r, tiempo, extra=None):
            filas.append(dict(fase=fase, target=etq, algoritmo=algoritmo, config=config,
                              n_dev=len(d_dev), n_lockbox=len(d_lb),
                              r2_oof=r["r2_oof"], mae_oof=r["mae_oof"], r2_lockbox=r["r2_lockbox"],
                              mae_lockbox=r["mae_lockbox"], tiempo_s=tiempo, **(extra or {})))

        guardados = {}

        # 1) HGB referencia, 5 semillas
        for seed in SEEDS5:
            t1 = time.time()
            r = oof_lockbox_generic(lambda seed=seed: L.hgb_nuisance(seed, 300), d_dev, d_lb, X, target)
            fila(f"HGB_ref", f"seed={seed}", r, time.time() - t1)
            if seed == SEED:
                guardados["hgb"] = r
        log(t0, f"{etq}: HGB_ref listo (5 semillas)")

        # 2) HGB grid pequeno (sesgo optimista: seleccion por el mismo OOF que se reporta)
        t1 = time.time()
        mejor = None
        mejor_cfg = None
        for md in (3, 4, 6):
            for lrr in (0.03, 0.05, 0.1):
                for mi in (150, 300, 600):
                    for msl in (10, 20, 40):
                        r = oof_lockbox_generic(
                            lambda md=md, lrr=lrr, mi=mi, msl=msl: HistGradientBoostingRegressor(
                                max_depth=md, learning_rate=lrr, max_iter=mi, min_samples_leaf=msl, random_state=SEED),
                            d_dev, d_lb, X, target)
                        filas_grid.append(dict(fase=fase, target=etq, max_depth=md, learning_rate=lrr, max_iter=mi,
                                               min_samples_leaf=msl, r2_oof=r["r2_oof"], r2_lockbox=r["r2_lockbox"]))
                        if mejor is None or r["r2_oof"] > mejor["r2_oof"]:
                            mejor, mejor_cfg = r, dict(max_depth=md, learning_rate=lrr, max_iter=mi, min_samples_leaf=msl)
        fila("HGB_grid_mejor(OPTIMISTA:seleccion=OOF)", str(mejor_cfg), mejor, time.time() - t1)
        log(t0, f"{etq}: HGB grid (81 cfgs) listo, mejor={mejor_cfg} r2_oof={mejor['r2_oof']:.3f} ({time.time()-t1:.0f}s)")

        # 3) XGBoost, 3 configs
        import xgboost as xgb
        xgb_cfgs = [dict(max_depth=3, n_estimators=300, learning_rate=0.05),
                    dict(max_depth=4, n_estimators=600, learning_rate=0.03),
                    dict(max_depth=6, n_estimators=200, learning_rate=0.1)]
        for i, cfg in enumerate(xgb_cfgs):
            t1 = time.time()
            r = oof_lockbox_generic(lambda cfg=cfg: xgb.XGBRegressor(n_jobs=3, random_state=SEED, **cfg),
                                    d_dev, d_lb, X, target)
            fila(f"XGBoost_cfg{i + 1}", str(cfg), r, time.time() - t1)
            if i == 0:
                guardados["xgb"] = r
        log(t0, f"{etq}: XGBoost listo")

        # 4) RandomForest / ExtraTrees
        t1 = time.time()
        r = oof_lockbox_generic(lambda: RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1), d_dev, d_lb, X, target)
        fila("RandomForest", "n_estimators=500", r, time.time() - t1)
        t1 = time.time()
        r = oof_lockbox_generic(lambda: ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=1), d_dev, d_lb, X, target)
        fila("ExtraTrees", "n_estimators=500", r, time.time() - t1)
        log(t0, f"{etq}: RF/ET listos")

        # 5) Ridge / Lasso con cuadraticos de las 8 features mas importantes (permutation importance sobre HGB)
        from sklearn.inspection import permutation_importance
        hgb_full = L.hgb_nuisance(SEED, 300).fit(d_dev[X], d_dev[target])
        imp = permutation_importance(hgb_full, d_dev[X], d_dev[target], n_repeats=5, random_state=SEED, n_jobs=1)
        importancias = pd.Series(imp.importances_mean, index=X).sort_values(ascending=False)
        top8 = list(importancias.index[:8])
        med = d_dev[top8].median()
        d_dev_q, d_lb_q = d_dev.copy(), d_lb.copy()
        for f in top8:
            d_dev_q[f"{f}__sq"] = (d_dev_q[f] - med[f]) ** 2
            d_lb_q[f"{f}__sq"] = (d_lb_q[f] - med[f]) ** 2
        Xq = X + [f"{f}__sq" for f in top8]
        t1 = time.time()
        r = oof_lockbox_generic(lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))), d_dev_q, d_lb_q, Xq, target)
        fila("Ridge+cuadraticos_top8", f"top8={top8}", r, time.time() - t1)
        guardados["ridge"] = r
        t1 = time.time()
        r = oof_lockbox_generic(lambda: make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 1, 13), max_iter=5000)), d_dev_q, d_lb_q, Xq, target)
        fila("Lasso+cuadraticos_top8", f"top8={top8}", r, time.time() - t1)
        log(t0, f"{etq}: Ridge/Lasso listos (top8={top8})")

        # 6) kNN (grid de k, sesgo optimista igual que HGB grid)
        t1 = time.time()
        mejor_knn, mejor_k = None, None
        for k in (10, 15, 20, 25, 30):
            r = oof_lockbox_generic(lambda k=k: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)), d_dev, d_lb, X, target)
            if mejor_knn is None or r["r2_oof"] > mejor_knn["r2_oof"]:
                mejor_knn, mejor_k = r, k
        fila("kNN_mejor(OPTIMISTA:seleccion=OOF)", f"k={mejor_k}", mejor_knn, time.time() - t1)
        log(t0, f"{etq}: kNN listo (mejor k={mejor_k})")

        # 7) MLP pequeno
        t1 = time.time()
        r = oof_lockbox_generic(lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=2000, random_state=SEED)), d_dev, d_lb, X, target)
        fila("MLP_64x64", "early_stopping", r, time.time() - t1)
        log(t0, f"{etq}: MLP listo")

        # 8) Stacking (promedio y meta-Ridge) sobre HGB_ref(seed=42) + XGBoost_cfg1 + Ridge+cuadraticos
        t1 = time.time()
        oof_h, lb_h = guardados["hgb"]["oof"], guardados["hgb"]["pred_lb"]
        oof_x, lb_x = guardados["xgb"]["oof"], guardados["xgb"]["pred_lb"]
        oof_r, lb_r = guardados["ridge"]["oof"], guardados["ridge"]["pred_lb"]
        prom_oof = (oof_h + oof_x + oof_r) / 3
        prom_lb = (lb_h + lb_x + lb_r) / 3
        r = dict(r2_oof=float(r2_score(d_dev[target], prom_oof)), mae_oof=float(mean_absolute_error(d_dev[target], prom_oof)),
                r2_lockbox=float(r2_score(d_lb[target], prom_lb)) if len(d_lb) > 10 else np.nan,
                mae_lockbox=float(mean_absolute_error(d_lb[target], prom_lb)) if len(d_lb) > 10 else np.nan)
        fila("Stacking_promedio(HGB+XGB+Ridge)", "media simple", r, time.time() - t1)

        t1 = time.time()
        M = np.column_stack([oof_h, oof_x, oof_r])
        meta_oof = np.full(len(d_dev), np.nan)
        n_splits_ef = min(N_SPLITS, d_dev["Batch"].nunique())
        y_arr = d_dev[target].to_numpy()
        for tr, te in GroupKFold(n_splits_ef).split(M, y_arr, d_dev["Batch"]):
            lrm = LinearRegression().fit(M[tr], y_arr[tr])
            meta_oof[te] = lrm.predict(M[te])
        lrm_full = LinearRegression().fit(M, y_arr)
        Mlb = np.column_stack([lb_h, lb_x, lb_r])
        meta_lb = lrm_full.predict(Mlb) if len(d_lb) else np.array([])
        r = dict(r2_oof=float(r2_score(y_arr, meta_oof)), mae_oof=float(mean_absolute_error(y_arr, meta_oof)),
                r2_lockbox=float(r2_score(d_lb[target], meta_lb)) if len(d_lb) > 10 else np.nan,
                mae_lockbox=float(mean_absolute_error(d_lb[target], meta_lb)) if len(d_lb) > 10 else np.nan)
        fila("Stacking_meta-Ridge(lineal sobre OOF)", "LinearRegression sobre [HGB,XGB,Ridge]", r, time.time() - t1)
        log(t0, f"{etq}: stacking listo")

        # 9) PLM v6, 5 semillas
        for seed in SEEDS5:
            t1 = time.time()
            met, _, _ = L.evaluar_plm_signos(df_fase, S, A, target, n_boot=0, seed=seed, signo_teorico=signos)
            r = dict(r2_oof=met["r2_oof"], mae_oof=met["mae_oof"], r2_lockbox=met["r2_lockbox"], mae_lockbox=met["mae_lockbox"])
            fila("PLM_v6", f"seed={seed}", r, time.time() - t1)
        log(t0, f"{etq}: PLM_v6 listo (5 semillas)")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(CSV["bakeoff"], index=False)
    pd.DataFrame(filas_grid).to_csv(CSV["bakeoff_grid"], index=False)
    log(t0, f"guardado {CSV['bakeoff']} ({len(tabla)} filas) y {CSV['bakeoff_grid']}")
    print("FIN bakeoff", flush=True)


# =============================================================================
# ETAPA 4: desglose (orden, tercio, cuartil de avance, residuos)
# =============================================================================

def _r2_mae(y, p) -> tuple[float, float, int]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())
    return float(r2_score(y[m], p[m])), float(mean_absolute_error(y[m], p[m])), int(m.sum())


def etapa_desglose() -> None:
    from scipy import stats as sps
    t0 = time.time()
    df, dR, dF, dev, lb = cargar()
    filas = []
    for fase, clave, target, etq in TARGET_DEFS:
        S, A = L.ESTADO[(fase, clave)], L.PALANCAS[(fase, clave)]
        signos = L.SIGNOS.get((fase, clave))
        df_fase = base_de_fase(fase, dR, dF)
        met, tabla_theta, pred = L.evaluar_plm_signos(df_fase, S, A, target, n_boot=0, seed=SEED, signo_teorico=signos)
        log(t0, f"{etq}: r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f}")

        # por orden de escalon (DEV-OOF y lockbox)
        for conjunto in ("OOF_dev", "lockbox"):
            sub = pred.loc[pred["conjunto"] == conjunto]
            for orden, g in sub.groupby("orden_escalon_fase"):
                r2, mae, n = _r2_mae(g["real"], g["pred"])
                filas.append(dict(target=etq, tipo="orden", grupo=f"R{int(orden)}" if fase == "Reducción" else f"F{int(orden)}",
                                  conjunto=conjunto, r2=r2, mae=mae, n=n))

        # por tercio cronologico de DEV (solo OOF_dev)
        sub = pred.loc[pred["conjunto"] == "OOF_dev"].copy()
        base_fase_dev = df_fase.loc[df_fase["Batch"].isin(dev)]
        terc = L.tercios_cronologicos(base_fase_dev.reset_index(drop=True))
        mapa_terc = dict(zip(base_fase_dev.reset_index(drop=True)["Batch"], terc))
        sub["tercio"] = sub["Batch"].map(mapa_terc)
        for tercio, g in sub.groupby("tercio"):
            r2, mae, n = _r2_mae(g["real"], g["pred"])
            filas.append(dict(target=etq, tipo="tercio_cronologico_DEV", grupo=f"T{int(tercio)}",
                              conjunto="OOF_dev", r2=r2, mae=mae, n=n))

        # por cuartil de m6_avance_prev (si esta en el estado; ambas fases lo tienen en R, Fusion puede no)
        if "m6_avance_prev" in df_fase.columns:
            av = df_fase[["Batch", "fecha_inicio", "m6_avance_prev"]].copy()
            sub2 = sub.merge(av, on=["Batch", "fecha_inicio"], how="left")
            sub2 = sub2.dropna(subset=["m6_avance_prev"])
            if len(sub2) > 40:
                sub2["cuartil_avance"] = pd.qcut(sub2["m6_avance_prev"], 4, labels=False, duplicates="drop")
                for q, g in sub2.groupby("cuartil_avance"):
                    r2, mae, n = _r2_mae(g["real"], g["pred"])
                    filas.append(dict(target=etq, tipo="cuartil_avance_prev_DEV", grupo=f"Q{int(q)}",
                                      conjunto="OOF_dev", r2=r2, mae=mae, n=n))

        # --- residuos OOF-dev: normalidad, heterocedasticidad, estructura en palancas ---
        d_oof = pred.loc[pred["conjunto"] == "OOF_dev"].copy()
        d_oof["res"] = d_oof["real"] - d_oof["pred"]
        res = d_oof["res"].to_numpy()
        skew, kurt = float(sps.skew(res)), float(sps.kurtosis(res))
        n_sw = min(len(res), 4990)
        sw_stat, sw_p = sps.shapiro(res[:n_sw]) if n_sw >= 20 else (np.nan, np.nan)
        filas.append(dict(target=etq, tipo="residuo_normalidad", grupo="skew_kurt_shapiro",
                          conjunto="OOF_dev", r2=skew, mae=kurt, n=len(res), extra=f"shapiro_p={sw_p:.4f}"))

        rho_het, p_het = sps.spearmanr(np.abs(res), d_oof["pred"])
        filas.append(dict(target=etq, tipo="heterocedasticidad", grupo="abs_res_vs_pred",
                          conjunto="OOF_dev", r2=float(rho_het), mae=float(p_het), n=len(res)))

        need_pal = list(dict.fromkeys(A + ["Batch", "fecha_inicio"]))
        pal = df_fase[need_pal].copy()
        d_oof2 = d_oof.merge(pal, on=["Batch", "fecha_inicio"], how="left")
        for a in A:
            dd = d_oof2.dropna(subset=[a, "res"])
            if len(dd) > 30:
                rho, p = sps.spearmanr(dd["res"], dd[a])
                filas.append(dict(target=etq, tipo="residuo_vs_palanca", grupo=a, conjunto="OOF_dev",
                                  r2=float(rho), mae=float(p), n=len(dd)))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CSV["desglose"], index=False)
    log(t0, f"guardado {CSV['desglose']} ({len(tabla)} filas)")
    print("FIN desglose", flush=True)


# =============================================================================
# ETAPA 5: resumen y veredicto
# =============================================================================

def _bootstrap_delta_r2_por_batch(d_dev: pd.DataFrame, target: str, oof_a: np.ndarray, oof_b: np.ndarray,
                                  n_boot: int = 500, seed: int = SEED) -> tuple[float, float, float]:
    """IC bootstrap (por batch) de delta R2 = R2(a) - R2(b) sobre las MISMAS filas OOF."""
    rng = np.random.default_rng(seed)
    y = d_dev[target].to_numpy()
    batches = d_dev["Batch"].to_numpy()
    uniq = np.unique(batches)
    idx_by = {b: np.where(batches == b)[0] for b in uniq}
    delta_obs = r2_score(y, oof_a) - r2_score(y, oof_b)
    deltas = []
    for _ in range(n_boot):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[b] for b in sel])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(r2_score(y[idx], oof_a[idx]) - r2_score(y[idx], oof_b[idx]))
    if not deltas:
        return delta_obs, np.nan, np.nan
    return delta_obs, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def etapa_resumen() -> None:
    t0 = time.time()
    df, dR, dF, dev, lb = cargar()
    techo = pd.read_csv(CSV["techo"]) if CSV["techo"].exists() else pd.DataFrame()
    curva = pd.read_csv(CSV["curva"]) if CSV["curva"].exists() else pd.DataFrame()
    ajuste = pd.read_csv(RAIZ_V7 / "e9_05_curva_ajuste.csv") if (RAIZ_V7 / "e9_05_curva_ajuste.csv").exists() else pd.DataFrame()
    bakeoff = pd.read_csv(CSV["bakeoff"]) if CSV["bakeoff"].exists() else pd.DataFrame()

    filas = []
    for fase, clave, target, etq in TARGET_DEFS:
        S, A = L.ESTADO[(fase, clave)], L.PALANCAS[(fase, clave)]
        signos = L.SIGNOS.get((fase, clave))
        df_fase = base_de_fase(fase, dR, dF)
        d_dev, d_lb = L.preparar(df_fase, S + A, target)

        fila = dict(target=etq, fase=fase, clave=clave)
        # techo (fila base sigma_sn=0.02, sigma_feo=0.028, orden='todos')
        if not techo.empty:
            t_row = techo[(techo["target"] == etq) & (techo["orden"] == "todos") &
                         np.isclose(techo["sigma_sn"], 0.02) & np.isclose(techo["sigma_feo"], 0.028)]
            if len(t_row):
                fila["r2_max_ruido"] = float(t_row["r2_max"].iloc[0])
                fila["r2_oof_plm_techo"] = float(t_row["r2_oof_plm"].iloc[0])
                fila["r2_relativa_oof"] = float(t_row["r2_relativa_oof"].iloc[0])
                fila["autocorr_lag1_res_oof"] = float(t_row["autocorr_lag1_res_oof"].iloc[0])

        # bakeoff: PLM_v6 (media 5 semillas) vs mejor algoritmo no-PLM
        if not bakeoff.empty:
            sub = bakeoff[bakeoff["target"] == etq]
            plm_rows = sub[sub["algoritmo"] == "PLM_v6"]
            r2_oof_plm = float(plm_rows["r2_oof"].mean())
            r2_lb_plm = float(plm_rows["r2_lockbox"].mean())
            sd_r2_oof_plm = float(plm_rows["r2_oof"].std())
            no_plm = sub[sub["algoritmo"] != "PLM_v6"]
            fila_best_bruto = no_plm.loc[no_plm["r2_oof"].idxmax()]
            no_plm_honesto = no_plm[~no_plm["algoritmo"].str.contains("OPTIMISTA")]
            fila_best = no_plm_honesto.loc[no_plm_honesto["r2_oof"].idxmax()] if len(no_plm_honesto) else fila_best_bruto
            fila.update(r2_oof_plm=r2_oof_plm, r2_lockbox_plm=r2_lb_plm, sd_r2_oof_plm_5seeds=sd_r2_oof_plm,
                       mejor_algoritmo=fila_best["algoritmo"], mejor_config=fila_best["config"],
                       r2_oof_mejor=float(fila_best["r2_oof"]), r2_lockbox_mejor=float(fila_best["r2_lockbox"]),
                       mejor_algoritmo_bruto_optimista=fila_best_bruto["algoritmo"],
                       r2_oof_mejor_bruto_optimista=float(fila_best_bruto["r2_oof"]))

            # recomputar OOF de PLM(seed=42) y del mejor algoritmo (honesto, sin seleccion-por-grid) sobre las
            # MISMAS filas para el bootstrap del delta
            oof_plm = np.full(len(d_dev), np.nan)
            n_splits_ef = min(N_SPLITS, d_dev["Batch"].nunique())
            for tr, te in GroupKFold(n_splits_ef).split(d_dev, d_dev[target], d_dev["Batch"]):
                m = L.ModeloPLMSignos(S, A, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=SEED)
                oof_plm[te] = m.predict(d_dev.iloc[te])

            import re
            import xgboost as xgb

            def _xgb_builder(cfg_idx: int):
                cfgs = [dict(max_depth=3, n_estimators=300, learning_rate=0.05),
                        dict(max_depth=4, n_estimators=600, learning_rate=0.03),
                        dict(max_depth=6, n_estimators=200, learning_rate=0.1)]
                cfg = cfgs[cfg_idx - 1]
                return lambda: xgb.XGBRegressor(n_jobs=3, random_state=SEED, **cfg)

            def _oof_X(builder, X, dd=None):
                dd = d_dev if dd is None else dd
                oof = np.full(len(dd), np.nan)
                for tr, te in GroupKFold(n_splits_ef).split(dd[X], dd[target], dd["Batch"]):
                    mdl = builder()
                    mdl.fit(dd[X].iloc[tr], dd[target].iloc[tr])
                    oof[te] = mdl.predict(dd[X].iloc[te])
                return oof

            nombre_mejor = fila_best["algoritmo"]
            oof_best = None
            if nombre_mejor.startswith("HGB_ref"):
                oof_best = _oof_X(lambda: L.hgb_nuisance(SEED, 300), S + A)
            elif nombre_mejor.startswith("XGBoost_cfg"):
                idx = int(re.search(r"cfg(\d+)", nombre_mejor).group(1))
                oof_best = _oof_X(_xgb_builder(idx), S + A)
            elif nombre_mejor.startswith("RandomForest"):
                oof_best = _oof_X(lambda: RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1), S + A)
            elif nombre_mejor.startswith("ExtraTrees"):
                oof_best = _oof_X(lambda: ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=1), S + A)
            elif nombre_mejor.startswith("Stacking"):
                from sklearn.inspection import permutation_importance
                hgb_full = L.hgb_nuisance(SEED, 300).fit(d_dev[S + A], d_dev[target])
                imp = permutation_importance(hgb_full, d_dev[S + A], d_dev[target], n_repeats=5, random_state=SEED, n_jobs=1)
                top8 = list(pd.Series(imp.importances_mean, index=S + A).sort_values(ascending=False).index[:8])
                med = d_dev[top8].median()
                d_dev_q = d_dev.copy()
                for f in top8:
                    d_dev_q[f"{f}__sq"] = (d_dev_q[f] - med[f]) ** 2
                Xq = S + A + [f"{f}__sq" for f in top8]
                oof_h = _oof_X(lambda: L.hgb_nuisance(SEED, 300), S + A)
                oof_x = _oof_X(_xgb_builder(1), S + A)
                oof_r = _oof_X(lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))), Xq, dd=d_dev_q)
                if "meta" in nombre_mejor:
                    M = np.column_stack([oof_h, oof_x, oof_r])
                    y_arr = d_dev[target].to_numpy()
                    oof_best = np.full(len(d_dev), np.nan)
                    for tr, te in GroupKFold(n_splits_ef).split(M, y_arr, d_dev["Batch"]):
                        lrm = LinearRegression().fit(M[tr], y_arr[tr])
                        oof_best[te] = lrm.predict(M[te])
                else:
                    oof_best = (oof_h + oof_x + oof_r) / 3
            if oof_best is not None:
                delta, ci_lo, ci_hi = _bootstrap_delta_r2_por_batch(d_dev, target, oof_best, oof_plm)
                fila.update(delta_r2_mejor_vs_plm=delta, delta_ci_lo=ci_lo, delta_ci_hi=ci_hi,
                           supera_umbral_0_01=bool(delta >= 0.01 and ci_lo > 0))
            else:
                fila.update(delta_r2_mejor_vs_plm=float(fila_best["r2_oof"]) - r2_oof_plm, delta_ci_lo=np.nan,
                           delta_ci_hi=np.nan, supera_umbral_0_01=np.nan,
                           nota_delta="mejor algoritmo (Ridge/Lasso/kNN/MLP) sin refit en resumen; delta sin IC bootstrap")

        # curva de aprendizaje (saturacion)
        if not ajuste.empty:
            a_row = ajuste[(ajuste["target"] == etq) & (ajuste["modelo"] == "PLM_v6")]
            if len(a_row):
                fila["curva_asintota_a"] = float(a_row["a_asintota"].iloc[0])
                fila["curva_n_max"] = float(a_row["n_max"].iloc[0])
                fila["curva_r2_n_max"] = float(a_row["r2_oof_n_max"].iloc[0])
                fila["curva_gap_asintota"] = float(a_row["gap_asintota"].iloc[0])
        filas.append(fila)
        log(t0, f"{etq}: {fila}")

    resumen = pd.DataFrame(filas)

    def veredicto(row) -> str:
        partes = []
        r2max = row.get("r2_max_ruido", np.nan)
        r2plm = row.get("r2_oof_plm", row.get("r2_oof_plm_techo", np.nan))
        if pd.notna(r2max) and pd.notna(r2plm):
            partes.append(f"techo de ruido R2_max={r2max:.2f}, PLM v6 alcanza {r2plm:.2f} "
                          f"(R2 relativa {r2plm / r2max:.0%})" if r2max > 0 else "techo de ruido no identificado (R2_max<=0)")
        delta = row.get("delta_r2_mejor_vs_plm", np.nan)
        ci_lo = row.get("delta_ci_lo", np.nan)
        if pd.notna(delta):
            if delta >= 0.01 and pd.notna(ci_lo) and ci_lo > 0:
                partes.append(f"{row.get('mejor_algoritmo')} supera al PLM en {delta:+.3f} OOF (IC>0): queda margen algoritmico")
            else:
                partes.append(f"ningun algoritmo supera al PLM en >=0.01 OOF con IC solido (mejor delta {delta:+.3f}); "
                              f"no hay modelo mejor con estos datos y este estado")
        gap = row.get("curva_gap_asintota", np.nan)
        if pd.notna(gap):
            partes.append(f"curva de aprendizaje: gap a la asintota estimada {gap:+.3f} R2 (saturada si <=0.02)")
        return "; ".join(partes) if partes else "datos insuficientes para veredicto"

    resumen["veredicto"] = resumen.apply(veredicto, axis=1)
    resumen.to_csv(CSV["resumen"], index=False)
    log(t0, f"guardado {CSV['resumen']}")
    print(resumen.to_string())
    print("FIN resumen", flush=True)


ETAPAS = {"techo": etapa_techo, "curva": etapa_curva, "bakeoff": etapa_bakeoff, "desglose": etapa_desglose,
         "resumen": etapa_resumen}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ETAPAS:
        print(f"uso: python e9_05_techo.py <{'|'.join(ETAPAS)}>")
        sys.exit(1)
    ETAPAS[sys.argv[1]]()
