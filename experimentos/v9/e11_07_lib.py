"""E11-07 -- libreria: walk-forward de ventana movil reimplementado en numpy puro (sin statsmodels en el loop caliente),
nulas por permutacion en bloques + desplazamiento circular, placebos, alternativas de re-anclaje, bootstrap por bloques.

Carga la tabla ya construida por e11_06b (`e11_06b_tabla_batch.csv`, 362 batches, orden idx_cronologico) para no repetir el
pipeline pesado de v9_lib (HGB cross-fitted). Todo lo que sigue opera sobre arrays numpy en orden cronologico puro.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experimentos/v9")
import v9_lib as L  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent.parent
SALIDA = RAIZ / "experimentos" / "v9"
KPI = "recuperacion_refinada_pct"
CONTROLES = ["feed_sn_total_t", "ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm", "frac_carga_secundaria_F",
             "feed_dross_Fe_total_t", "idx_cronologico"]
C_SIN_ESPESOR = [c for c in CONTROLES if c != "espesor_ladrillo_norm_mm"]
X_PRINCIPAL = ["xG", "xC"]
WS_MULTIPLICIDAD = [60, 80, 100, 120, 150, None]           # None = expansiva
PASO = 10
SEED = 42


def cargar_tabla(extra_cols: list[str] | None = None) -> pd.DataFrame:
    t = pd.read_csv(SALIDA / "e11_06b_tabla_batch.csv", index_col=0).sort_values("idx_cronologico")
    if extra_cols:
        falt = [c for c in extra_cols if c not in t.columns]
        if falt:
            tb = L.tabla_batch()[falt]
            t = t.join(tb, how="left")
    return t


def preparar(t: pd.DataFrame, x_cols: list[str], y_col: str = KPI, controles: list[str] | None = None) -> dict:
    """Devuelve arrays numpy en orden cronologico (sin NaN) + metadatos."""
    controles = C_SIN_ESPESOR if controles is None else controles
    cols = x_cols + controles + [y_col]
    d = t.dropna(subset=cols).copy()
    return {"Xexp": d[x_cols].to_numpy(float), "Xctrl": d[controles].to_numpy(float), "y": d[y_col].to_numpy(float),
            "es_dev": d["es_dev"].to_numpy(bool), "idx": d.index.to_numpy(), "T_media_R": d.get("T_media_R", pd.Series(np.nan, index=d.index)).to_numpy(float),
            "espesor": d.get("espesor_ladrillo_norm_mm", pd.Series(np.nan, index=d.index)).to_numpy(float), "n": len(d)}


# --------------------------------------------------------------------------- OLS numpy puro (rapido, para loops)
def _lstsq_beta(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return beta


def ols_classic(Z: np.ndarray, y: np.ndarray) -> dict:
    """OLS homocedastico rapido: beta, se, t. Z SIN columna de unos (se agrega aqui)."""
    Zc = np.column_stack([np.ones(len(y)), Z])
    beta = _lstsq_beta(Zc, y)
    resid = y - Zc @ beta
    dof = max(len(y) - Zc.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(Zc.T @ Zc)
    se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return {"beta": beta, "se": se, "t": t, "resid": resid}


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else np.nan


# --------------------------------------------------------------------------- walk-forward nucleo
def walk_forward_score(Xexp: np.ndarray, Xctrl: np.ndarray, y: np.ndarray, W: int | None, paso: int = PASO,
                        start: int | None = None, tmin: float = 0.0, halflife: float | None = None,
                        ridge_lambda: float | None = None, shrink_expansiva: bool = False) -> np.ndarray:
    """Reproduce e11_06d: beta_t estimado SOLO con el pasado (ventana dura W, o expansiva si W=None), aplicado
    prospectivamente a los `paso` batches siguientes. `halflife`: pondera exponencialmente TODA la historia pasada
    (ignora W, usa lo=0) en vez de una ventana dura. `ridge_lambda`: penaliza SOLO los coeficientes de exposicion
    (grados de libertad efectivos objetivo). `shrink_expansiva`: si True, el beta aplicado es el promedio del beta
    de ventana W y el beta expansivo (alternativa iii)."""
    n = len(y); ne = Xexp.shape[1]; nc = Xctrl.shape[1]
    start = start if start is not None else (W if W is not None else 100)
    score = np.full(n, np.nan)
    s = start
    while s < n:
        lo = 0 if (W is None or halflife is not None) else max(0, s - W)
        hi_te = min(s + paso, n)
        Xtr_c, Xtr_e, ytr = Xctrl[lo:s], Xexp[lo:s], y[lo:s]
        Z = np.column_stack([np.ones(s - lo), Xtr_c, Xtr_e])
        if halflife is not None:
            edad = (s - 1) - np.arange(lo, s)
            w = 0.5 ** (edad / halflife); sw = np.sqrt(w)
            Zw, yw = Z * sw[:, None], ytr * sw
            beta = _lstsq_beta(Zw, yw)
        elif ridge_lambda is not None:
            beta = _ridge_beta(Z, ytr, nc, ridge_lambda)
        else:
            beta = _lstsq_beta(Z, ytr)
        be = beta[1 + nc:]
        if tmin > 0:
            resid = ytr - Z @ beta; dof = max(len(ytr) - Z.shape[1], 1)
            sigma2 = float(resid @ resid) / dof
            se = np.sqrt(np.clip(np.diag(np.linalg.pinv(Z.T @ Z)) * sigma2, 0, None))[1 + nc:]
            be = np.where(np.abs(np.divide(be, se, out=np.full_like(be, np.inf), where=se > 0)) > tmin, be, 0.0)
        if shrink_expansiva:
            Zexp = np.column_stack([np.ones(s), Xctrl[:s], Xexp[:s]])
            beta_exp = _lstsq_beta(Zexp, y[:s])
            be = 0.5 * (be + beta_exp[1 + nc:])
        score[s:hi_te] = Xexp[s:hi_te] @ be
        s += paso
    return score


def _ridge_beta(Z: np.ndarray, y: np.ndarray, nc: int, lam_target_dof: float) -> np.ndarray:
    """Ridge SOLO sobre las columnas de exposicion (const+controles sin penalizar), con lambda elegido para que los
    grados de libertad efectivos de esas columnas (tras partir out const+controles) sean ~= lam_target_dof."""
    Zc, Ze = Z[:, :1 + nc], Z[:, 1 + nc:]
    # residualizar Ze y y respecto de Zc
    beta_c_e = _lstsq_beta(Zc, Ze) if Ze.shape[1] else Ze
    Ze_tilde = Ze - Zc @ beta_c_e if Ze.shape[1] else Ze
    beta_c_y = _lstsq_beta(Zc, y)
    y_tilde = y - Zc @ beta_c_y
    if Ze_tilde.shape[1] == 0 or Ze_tilde.shape[0] < 3:
        beta_full = _lstsq_beta(Z, y); return beta_full
    U, s_sv, Vt = np.linalg.svd(Ze_tilde, full_matrices=False)
    def dof(lam):
        return float(np.sum(s_sv ** 2 / (s_sv ** 2 + lam)))
    lo_l, hi_l = 1e-8, 1e8
    if dof(lo_l) <= lam_target_dof:
        lam = lo_l
    elif dof(hi_l) >= lam_target_dof:
        lam = hi_l
    else:
        for _ in range(60):
            mid = np.sqrt(lo_l * hi_l)
            if dof(mid) > lam_target_dof:
                lo_l = mid
            else:
                hi_l = mid
        lam = np.sqrt(lo_l * hi_l)
    D = s_sv / (s_sv ** 2 + lam)
    beta_e_tilde = Vt.T @ (D * (U.T @ y_tilde))
    beta_c = beta_c_y - beta_c_e @ beta_e_tilde if Ze.shape[1] else beta_c_y
    return np.concatenate([beta_c, beta_e_tilde])


# --------------------------------------------------------------------------- estadistico final (pooled)
def stats_final(score: np.ndarray, y: np.ndarray, Xctrl: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = ~np.isnan(score)
    else:
        mask = mask & ~np.isnan(score)
    s, yy, C = score[mask], y[mask], Xctrl[mask]
    if mask.sum() < 10 or np.nanstd(s) < 1e-10:
        return {"n": int(mask.sum()), "pendiente": np.nan, "t": np.nan, "p_uni": np.nan, "spearman": np.nan}
    r = ols_classic(np.column_stack([s, C]), yy)
    t = float(r["t"][1])
    from scipy import stats as sst
    ry = ols_classic(C, yy)["resid"]; rs = ols_classic(C, s)["resid"]
    rho = spearman_rho(rs, ry)
    return {"n": int(mask.sum()), "pendiente": float(r["beta"][1]), "se": float(r["se"][1]), "t": t,
            "p_uni": float(sst.norm.sf(t)), "spearman": rho, "sd_score": float(np.std(s))}


def stats_final_hc3(score: np.ndarray, y: np.ndarray, t_df: pd.DataFrame, controles: list[str], mask: np.ndarray, kpi_col: str) -> dict:
    """Version HC3 (statsmodels) solo para las tablas reportadas (no usar dentro de loops de permutacion)."""
    idx = t_df.index[mask]
    dd = t_df.loc[idx].copy(); dd["score"] = score[mask]
    fit = L.ols_hc3(dd[kpi_col], dd[["score"] + controles])
    from scipy import stats as sst
    tval = float(fit.tvalues["score"])
    return {"n": len(dd), "pendiente": float(fit.params["score"]), "t": tval, "p_uni": float(sst.norm.sf(tval)),
            "spearman": spearman_rho(dd["score"].to_numpy(), dd[kpi_col].to_numpy())}


# --------------------------------------------------------------------------- nulas
def bloques_cronologicos(n: int, tam: int = 20, seed: int = SEED) -> list[np.ndarray]:
    return [np.arange(i, min(i + tam, n)) for i in range(0, n, tam)]


def permutar_bloques(Xexp: np.ndarray, bloques: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Permuta las FILAS de Xexp (xG,xC conjuntas) dentro de cada bloque cronologico."""
    Xp = Xexp.copy()
    for b in bloques:
        Xp[b] = Xexp[b[rng.permutation(len(b))]]
    return Xp


def circular_shift(Xexp: np.ndarray, k: int) -> np.ndarray:
    return np.roll(Xexp, k, axis=0)
