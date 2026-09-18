"""E18-03 -- libreria compartida: tres replanteamientos evaluables sobre el asesor v10 (receta).
Reutiliza asesor_receta_v10 (escalones_con_receta, replay_prospectivo), e11_07_lib (W7, walk-forward numpy)
y e13_03_lib (contraccion JS, nulas por bloques/circular). Ejecutar SIEMPRE desde la raiz del repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v9", RAIZ / "experimentos" / "v11", RAIZ / "experimentos" / "v8"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import asesor_receta_v10 as A  # noqa: E402
import e11_07_lib as W7  # noqa: E402
import e13_03_lib as L13  # noqa: E402

KPI = A.KPI
CONTROLES = A.CONTROLES               # sin espesor (ya excluido en asesor_receta_v10)
SEED = 42
REPS = 1000


# --------------------------------------------------------------------------- carga
def cargar():
    """r: escalon de Reduccion (1448 x ...) + idx_cronologico/es_dev del batch. bt: 362 batches, orden cronologico."""
    r, bt = A.escalones_con_receta()
    r = r.merge(bt[["idx_cronologico", "es_dev"]], left_on="Batch", right_index=True, how="left")
    r = r.sort_values(["idx_cronologico", "orden_escalon_fase"]).reset_index(drop=True)
    return r, bt


def z_dev(r: pd.DataFrame, palanca: str) -> pd.Series:
    """Desvio ejecutado-receta estandarizado por la media/sd DEL ORDEN, calculadas SOLO en DEV (E15-01)."""
    col = f"dev__{palanca}"
    esdev = r["es_dev"].astype(bool)
    mu = r.loc[esdev].groupby("orden_escalon_fase")[col].mean()
    sd = r.loc[esdev].groupby("orden_escalon_fase")[col].std()
    return (r[col] - r["orden_escalon_fase"].map(mu)) / r["orden_escalon_fase"].map(sd)


# --------------------------------------------------------------------------- reglas de alcance (A)
def reglas_alcance(r: pd.DataFrame, bt: pd.DataFrame) -> dict[str, pd.Series]:
    """4 reglas ex-ante, booleanas a nivel de ESCALON (index de r). Ninguna mira el KPI."""
    out = {}
    out["A1_T1000"] = r["temperatura_horno_celsius_prev"] >= 1000.0
    out["A2_T1020"] = r["temperatura_horno_celsius_prev"] >= 1020.0
    # A3: espesor >= P20 expansivo de "lo visto hasta la fecha" (vida de campana, min 20 batches previos)
    esp_b = bt.sort_values("idx_cronologico")["espesor_ladrillo_norm_mm"]
    p20_exp = esp_b.expanding(min_periods=20).quantile(0.20).shift(1)  # SOLO batches estrictamente anteriores
    umbral_por_batch = pd.Series(p20_exp.values, index=esp_b.index)   # index = Batch
    r_umbral = r["Batch"].map(umbral_por_batch)
    out["A3_espesorP20"] = (r["espesor_ladrillo_norm_mm"] >= r_umbral) & r_umbral.notna()
    # A4: receta de GN de R0 <= 28.3 Nm3/min (nivel de batch)
    p_gn_r0 = r.loc[r["orden_escalon_fase"] == 0].set_index("Batch")["plan__gn"]
    out["A4_recetaR0"] = r["Batch"].map(p_gn_r0) <= 28.3
    return out


def exposicion_alcance(r: pd.DataFrame, mask: pd.Series, palancas=("gn", "carbon"), dentro: bool = True) -> pd.DataFrame:
    """Media por batch del z-score SOLO en escalones donde mask (dentro=True) o ~mask (dentro=False); 0 si ninguno."""
    m = mask if dentro else ~mask
    out = {}
    for p in palancas:
        z = z_dev(r, p).where(m)
        out[f"x_{p}"] = z.groupby(r["Batch"]).mean()
    res = pd.DataFrame(out)
    return res.fillna(0.0)


def pct_alcance_por_tramo(r: pd.DataFrame, mask: pd.Series, bins=(0, 100, 200, 300, 400)) -> pd.Series:
    tramo = pd.cut(r["idx_cronologico"], bins=bins, right=True)
    return mask.groupby(tramo).mean() * 100


# --------------------------------------------------------------------------- OLS conjunta HC3
def ols_conjunta(bt: pd.DataFrame, expo: pd.DataFrame, y_cols=(KPI,), controles=CONTROLES) -> pd.DataFrame:
    d = bt.join(expo)
    filas = []
    xcols = list(expo.columns)
    for y in y_cols:
        dd = d.dropna(subset=xcols + controles + [y])
        esc = 100.0 if y != KPI else 1.0
        f = sm.OLS(dd[y] * esc, sm.add_constant(dd[xcols + controles])).fit(cov_type="HC3")
        for x in xcols:
            filas.append({"y": y, "x": x, "n": len(dd), "beta": float(f.params[x]), "se": float(f.bse[x]), "t": float(f.tvalues[x])})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- prospectivo + multiplicidad (max-stat)
def preparar_batch(bt: pd.DataFrame, expo: pd.DataFrame, controles=CONTROLES, y=KPI):
    t = bt.join(expo).sort_values("idx_cronologico")
    P = W7.preparar(t, list(expo.columns), y_col=y, controles=controles)
    return P


def prospectivo(P: dict, W=100, paso=10, modo="js") -> dict:
    sc = L13.wf(P["Xexp"], P["Xctrl"], P["y"], W, paso=paso, modo=modo)
    return W7.stats_final(sc, P["y"], P["Xctrl"])


def maxstat_multiplicidad(Ps: dict[str, dict], W=100, paso=10, modo="js", reps=REPS, tam_bloque=20, seed=SEED) -> pd.DataFrame:
    """Corrige por multiplicidad sobre varias reglas: en cada replica se aplica el MISMO reordenamiento de
    posiciones (por bloques cronologicos) a la exposicion de cada regla, y se toma el maximo |t| entre reglas.
    Requiere que todas las reglas comparten el mismo n y el mismo orden temporal (btabla base identica)."""
    nombres = list(Ps.keys())
    n = Ps[nombres[0]]["n"]
    for k in nombres:
        assert Ps[k]["n"] == n, f"n distinto en {k}"
    bloques = L13.bloques(n, tam_bloque, seed=0)
    obs_t = {k: prospectivo(Ps[k], W, paso, modo)["t"] for k in nombres}
    rng = np.random.default_rng(seed)
    nul_t = {k: [] for k in nombres}
    for _ in range(reps):
        perm_idx = np.arange(n)
        for b in bloques:
            perm_idx[b] = b[rng.permutation(len(b))]
        for k in nombres:
            Xp = Ps[k]["Xexp"][perm_idx]
            sc = L13.wf(Xp, Ps[k]["Xctrl"], Ps[k]["y"], W, paso=paso, modo=modo)
            nul_t[k].append(W7.stats_final(sc, Ps[k]["y"], Ps[k]["Xctrl"])["t"])
    nul_t = {k: np.nan_to_num(np.array(v), nan=-np.inf) for k, v in nul_t.items()}
    t_max_null = np.max(np.column_stack([nul_t[k] for k in nombres]), axis=1)
    filas = []
    for k in nombres:
        if not np.isfinite(obs_t[k]):
            filas.append({"regla": k, "t_obs": np.nan, "p_perm_simple": np.nan, "p_corregida_maxstat_4reglas": np.nan})
            continue
        p_simple = L13.p_perm(obs_t[k], nul_t[k])
        p_corr = L13.p_perm(obs_t[k], t_max_null)
        filas.append({"regla": k, "t_obs": obs_t[k], "p_perm_simple": p_simple, "p_corregida_maxstat_4reglas": p_corr})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- jackknife y bootstrap por bloques
def jackknife_quitar_k(P: dict, k=10, reps=300, W=100, modo="js", seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = P["n"]; ts, sl = [], []
    for _ in range(reps):
        keep = np.sort(rng.choice(n, n - k, replace=False))
        sc = L13.wf(P["Xexp"][keep], P["Xctrl"][keep], P["y"][keep], W, modo=modo)
        st = W7.stats_final(sc, P["y"][keep], P["Xctrl"][keep])
        ts.append(st["t"]); sl.append(st["pendiente"])
    ts, sl = np.array(ts), np.array(sl)
    return {"t_P5": np.percentile(ts, 5), "t_P50": np.percentile(ts, 50), "t_P95": np.percentile(ts, 95),
            "pct_t_gt_1.645": 100 * float(np.mean(ts > 1.645)), "pend_P50": float(np.percentile(sl, 50))}


def calibracion_boot(P: dict, reps=1000, W=100, modo="js", seed=1) -> dict:
    """Bootstrap por bloques de 10 batches (con reemplazo) del walk-forward completo -> IC95 de la pendiente."""
    rng = np.random.default_rng(seed); n = P["n"]; nb = n // 10
    sls = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(b * 10, min(b * 10 + 10, n)) for b in rng.integers(0, nb, nb)])
        sc = L13.wf(P["Xexp"][idx], P["Xctrl"][idx], P["y"][idx], W, modo=modo)
        st = W7.stats_final(sc, P["y"][idx], P["Xctrl"][idx])
        sls.append(st["pendiente"])
    sls = np.array(sls)
    return {"pendiente_P50": float(np.nanmedian(sls)), "IC95_lo": float(np.nanpercentile(sls, 2.5)),
            "IC95_hi": float(np.nanpercentile(sls, 97.5)), "P_mayor_0": float(np.mean(sls > 0))}


# --------------------------------------------------------------------------- BH (FDR)
def bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        val = min(prev, p[idx] * n / rank)
        q[idx] = val; prev = val
    return q.tolist()
