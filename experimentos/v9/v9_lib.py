"""v9_lib -- librería compartida de la iteración 11 (2026-09-16/17): anclaje del prescriptor en el RESULTADO DEL BATCH por el canal de
la acción.

Punto de partida: v8 (`modelo_predictivo_v8.py`, `candidate_win_model_v8.md`, hallazgos §22). Brecha que ataca: el objetivo J de v8
(K, w anclados regresando el KPI sobre los targets agregados, que mezclan varianza de estado y de acción) NO predice el KPI por la
vía de la acción (`uplift_batch_pp` vs KPI nulo; HGB no ve ganancia), mientras que el residuo de la acción (a − E[a|S]) agregado por
batch SÍ se asocia al KPI (GN −0.46 pp/sd; carbón tardío +0.81 pp/sd). La iteración 11 convierte esa pista en el eje del sistema:

1) `residuos_escalon`: a − E[a|S] por escalón, E[a|S] = HGB cross-fitted por batch (DEV: GroupKFold; lockbox: modelo entrenado con todo
   DEV). NUNCA usa el KPI ni ningún dato posterior al inicio del escalón: S = estado conocido al inicio + orden (la receta es por orden).
   Estandarizado por la sd del residuo dentro de (fase, orden) en DEV.
2) `exposiciones_batch`: media por batch y grupo (F = Fusión F1-F6, Rt = Reducción R0-R1, Rl = Reducción R2-R3) del residuo z.
3) `prueba_anidada`: validación cruzada ANIDADA por batch del vínculo acción → KPI: en cada fold β (pp de KPI por sd de desviación
   respecto de la práctica habitual dado el estado) se estima sin los batches de prueba; el puntaje del batch de prueba = Σ β̂·x es el
   "valor atribuido por la política a lo que el operador hizo"; se contrasta KPI ~ puntaje + controles sobre los OOF (HC3, permutación).
   Es la demostración exigida: batches NO vistos cuya operación se pareció más a la recomendada rinden más.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v8"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v8_lib as L8  # noqa: E402
import modelo_predictivo_v8 as mp8  # noqa: E402

SALIDA = RAIZ / "experimentos" / "v9"
CACHE_RES = RAIZ / "experimentos" / "cache" / "v9_residuos_escalon.pkl"
SEED = 42
KPI = L8.KPI_PRINCIPAL                       # recuperacion_refinada_pct
CONTROLES = list(L8.CONTROLES_BATCH)         # feed Sn, ley conc, espesor (reloj), carga secundaria, dross Fe, idx cronológico
CANALES = ["f_metal", "f_dross", "f_polvo", "sn_perdido_escoria_frac"]

PALANCAS = {"carbon": "tasa_feed_Carbon_kg_min", "gn": "tasa_gn_nm3_min", "o2": "tasa_o2_nm3_min", "aire": "tasa_aire_nm3_min",
            "exo2": "exceso_o2_combustion_pct"}
# Estado de la política de comportamiento E[a|S]: lo que el operador conoce al INICIO del escalón.
ESTADO_R = list(mp8.ESTADO_R_V8) + ["orden_escalon_fase", "temperatura_horno_celsius_prev"]
ESTADO_F = ["m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev", "m6_masa_kg_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev",
            "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev", "temperatura_horno_celsius_prev", "termocupla_media_celsius_prev",
            "grad_temperatura_horno_celsius_prev", "posicion_vertical_lanza_mm_prev", "orden_escalon_fase", "cum_feed_Sn_kg_prev",
            "cum_feed_Carbon_kg_prev", "tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min"]   # la carga del escalón es PLAN (contexto)
ESTADOS = {"Reducción": ESTADO_R, "Fusión": ESTADO_F}


def grupo_de(fase: str, orden: int) -> str | None:
    if fase == "Fusión":
        return "F" if orden >= 1 else None        # F0 = arranque (talón), sin decisión modelable
    if fase == "Reducción":
        return "Rt" if orden <= 1 else ("Rl" if orden <= 3 else None)
    return None


def residuos_escalon(df: pd.DataFrame | None = None, n_splits: int = 5, seed: int = SEED, recalcular: bool = False,
                     estados: dict | None = None, cache: Path | None = CACHE_RES) -> pd.DataFrame:
    """Tabla por escalón: Batch, fase, orden, grupo, dur_min, y por palanca `a__p`, `ahat__p` (E[a|S] fuera de muestra), `res__p`,
    `z__p` (residuo / sd del residuo en DEV dentro de (fase, orden))."""
    from sklearn.model_selection import GroupKFold
    if cache is not None and cache.exists() and not recalcular and estados is None:
        return pd.read_pickle(cache)
    if df is None:
        df = L8.cargar_df()
    estados = estados or ESTADOS
    base = mp8._preparar_base(df)
    dev, lockbox = L8.mp.split_dev_lockbox(df)
    out = []
    for fase, S in estados.items():
        L8.asegurar_seguras([s for s in S if s not in ("tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min")])
        d = base[base["fase_proceso"] == fase].copy()
        d["grupo"] = [grupo_de(fase, int(o)) for o in d["orden_escalon_fase"]]
        d = d[d["grupo"].notna()].reset_index(drop=True)
        es_dev = d["Batch"].isin(dev).to_numpy()
        t = d[["Batch", "fase_proceso", "orden_escalon_fase", "grupo", "fecha_inicio"]].copy()
        t["dur_min"] = (pd.to_datetime(d["fecha_final"]) - pd.to_datetime(d["fecha_inicio"])).dt.total_seconds() / 60.0
        t["es_dev"] = es_dev
        for p, col in PALANCAS.items():
            ok = d[col].notna().to_numpy()
            ahat = np.full(len(d), np.nan)
            idx_dev = np.where(es_dev & ok)[0]
            for tr, te in GroupKFold(n_splits).split(idx_dev, groups=d["Batch"].to_numpy()[idx_dev]):
                m = L8.hgb_nuisance(seed).fit(d[S].iloc[idx_dev[tr]], d[col].iloc[idx_dev[tr]])
                ahat[idx_dev[te]] = m.predict(d[S].iloc[idx_dev[te]])
            idx_lb = np.where(~es_dev & ok)[0]
            if len(idx_lb):
                m = L8.hgb_nuisance(seed).fit(d[S].iloc[idx_dev], d[col].iloc[idx_dev])
                ahat[idx_lb] = m.predict(d[S].iloc[idx_lb])
            t[f"a__{p}"] = d[col].to_numpy(); t[f"ahat__{p}"] = ahat; t[f"res__{p}"] = t[f"a__{p}"] - ahat
            sd = t.loc[t["es_dev"]].groupby("orden_escalon_fase")[f"res__{p}"].std()
            t[f"z__{p}"] = t[f"res__{p}"] / t["orden_escalon_fase"].map(sd)
        out.append(t)
    res = pd.concat(out, ignore_index=True)
    if cache is not None and estados is ESTADOS:
        cache.parent.mkdir(exist_ok=True); res.to_pickle(cache)
    return res


def exposiciones_batch(res: pd.DataFrame, palancas: list[str] | None = None, grupos: tuple[str, ...] = ("F", "Rt", "Rl"),
                       columna: str = "z", ponderar_dur: bool = False) -> pd.DataFrame:
    """Una fila por batch: x_{grupo}_{palanca} = media (opcionalmente ponderada por duración) del residuo z del grupo."""
    palancas = palancas or list(PALANCAS)
    filas = {}
    for g in grupos:
        sub = res[res["grupo"] == g]
        for p in palancas:
            v = sub[f"{columna}__{p}"]
            if ponderar_dur:
                w = sub["dur_min"].clip(lower=1)
                s = (v * w).groupby(sub["Batch"]).sum() / w.where(v.notna()).groupby(sub["Batch"]).sum()
            else:
                s = v.groupby(sub["Batch"]).mean()
            filas[f"x_{g}_{p}"] = s
    return pd.DataFrame(filas)


def tabla_batch(df: pd.DataFrame | None = None, res: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    """Batch × (KPI, canales, controles, exposiciones). Índice = Batch, orden cronológico."""
    if df is None:
        df = L8.cargar_df()
    if res is None:
        res = residuos_escalon(df)
    b = L8.construir_batch(df)
    if "Batch" in b.columns:
        b = b.set_index("Batch")
    t = b.join(exposiciones_batch(res, **kw), how="left")
    return t.sort_values("idx_cronologico")


def ols_hc3(y: pd.Series, X: pd.DataFrame):
    import statsmodels.api as sm
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")


def _folds(t: pd.DataFrame, esquema: str, n: int, seed: int) -> np.ndarray:
    k = np.zeros(len(t), int)
    if esquema == "crono":
        k = (np.arange(len(t)) * n // len(t)).astype(int)
    else:
        rng = np.random.default_rng(seed); k = rng.permutation(np.arange(len(t)) % n)
    return k


def prueba_anidada(t: pd.DataFrame, exposiciones: list[str], kpi: str = KPI, controles: list[str] | None = None, esquema: str = "gkf",
                   n_folds: int = 5, n_rep: int = 20, n_perm: int = 2000, seed: int = SEED, ridge: float = 0.0,
                   signos: dict | None = None) -> dict:
    """CV anidada por batch. Devuelve el puntaje OOF (promedio de `n_rep` particiones si esquema='gkf'), la pendiente del KPI sobre el
    puntaje con controles (HC3; ideal = 1: calibración), p unilateral, Spearman parcial y p de permutación (el puntaje se permuta
    entre batches y se recalcula el estadístico t de la pendiente). `signos`: {exposición: +1/−1} → β acotado por signo (lsq_linear)."""
    from scipy import stats
    from scipy.optimize import lsq_linear
    controles = CONTROLES if controles is None else controles
    d = t[[kpi] + exposiciones + controles].dropna().copy()
    Xc = d[controles].to_numpy(float); Xe = d[exposiciones].to_numpy(float); y = d[kpi].to_numpy(float)
    reps = n_rep if esquema == "gkf" else 1
    score = np.zeros(len(d)); betas = []
    for r in range(reps):
        k = _folds(d, esquema, n_folds, seed + r)
        for f in range(n_folds):
            tr, te = k != f, k == f
            mu_c, sd_c = Xc[tr].mean(0), Xc[tr].std(0) + 1e-12
            Z = np.column_stack([np.ones(tr.sum()), (Xc[tr] - mu_c) / sd_c, Xe[tr]])
            lo = np.full(Z.shape[1], -np.inf); hi = np.full(Z.shape[1], np.inf)
            for j, e in enumerate(exposiciones):
                s = (signos or {}).get(e, 0)
                if s > 0: lo[1 + Xc.shape[1] + j] = 0.0
                if s < 0: hi[1 + Xc.shape[1] + j] = 0.0
            if ridge > 0:
                pen = np.zeros(Z.shape[1]); pen[1 + Xc.shape[1]:] = np.sqrt(ridge)
                Zr = np.vstack([Z, np.diag(pen)]); yr = np.concatenate([y[tr], np.zeros(Z.shape[1])])
            else:
                Zr, yr = Z, y[tr]
            beta = lsq_linear(Zr, yr, bounds=(lo, hi), lsmr_tol="auto").x if signos else np.linalg.lstsq(Zr, yr, rcond=None)[0]
            be = beta[1 + Xc.shape[1]:]; betas.append(be)
            score[te] += Xe[te] @ be / reps
    d["score"] = score
    fit = ols_hc3(d[kpi], d[["score"] + controles])
    tval = float(fit.tvalues["score"]); pend = float(fit.params["score"])
    # Spearman parcial (residuos de KPI y score respecto de los controles)
    ry = ols_hc3(d[kpi], d[controles]).resid; rs = ols_hc3(d["score"], d[controles]).resid
    rho, p_rho = stats.spearmanr(ry, rs)
    rng = np.random.default_rng(seed); tp = []
    for _ in range(n_perm):
        rs_p = rng.permutation(rs.to_numpy())
        tp.append(np.corrcoef(rs_p, ry)[0, 1])
    r_obs = float(np.corrcoef(rs, ry)[0, 1])
    p_perm = float((np.sum(np.array(tp) >= r_obs) + 1) / (n_perm + 1))
    q = pd.qcut(d["score"], 3, labels=False, duplicates="drop")
    ter = ry.groupby(q).mean()
    return {"n": len(d), "esquema": esquema, "pendiente": pend, "se": float(fit.bse["score"]), "t": tval,
            "p_unilateral": float(stats.norm.sf(tval)), "spearman_parcial": float(rho), "p_spearman": float(p_rho),
            "r_parcial": r_obs, "p_perm_unilateral": p_perm, "sd_score_pp": float(d["score"].std()),
            "kpi_tercio_bajo": float(ter.iloc[0]), "kpi_tercio_alto": float(ter.iloc[-1]),
            "beta_medio": dict(zip(exposiciones, np.mean(betas, 0).round(4))), "beta_sd_folds": dict(zip(exposiciones, np.std(betas, 0).round(4))),
            "score": d["score"]}


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = L8.cargar_df(); res = residuos_escalon(df, recalcular=True)
    print(res.shape, f"{time.time()-t0:.0f}s")
    from sklearn.metrics import r2_score
    for (f, g), s in res[res.es_dev].groupby(["fase_proceso", "grupo"]):
        print(f, g, {p: round(r2_score(s[f"a__{p}"][s[f"ahat__{p}"].notna() & s[f"a__{p}"].notna()], s[f"ahat__{p}"][s[f"ahat__{p}"].notna() & s[f"a__{p}"].notna()]), 3) for p in PALANCAS})
    t = tabla_batch(df, res); print(t.shape)
    dev = t[t.es_dev]
    X = [c for c in t.columns if c.startswith("x_") and c.split("_")[2] in ("carbon", "gn")]
    f = ols_hc3(dev.dropna(subset=X + CONTROLES + [KPI])[KPI], dev.dropna(subset=X + CONTROLES + [KPI])[X + CONTROLES])
    print(f.summary().tables[1])
    for esq in ("gkf", "crono"):
        r = prueba_anidada(dev, X, esquema=esq); r.pop("score"); print(esq, r)
