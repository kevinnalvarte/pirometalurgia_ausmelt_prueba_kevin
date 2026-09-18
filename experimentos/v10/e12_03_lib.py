"""E12-03 -- libreria: filtro de Kalman (beta_t = beta_{t-1} + eta_t, paseo aleatorio) para la regresion por batch
KPI ~ controles + xG + xC, con prediccion un-paso-adelante (beta_{b|b-1}, no usa el KPI del batch b).

Reutiliza `experimentos/v9/e11_07_lib.py` para cargar la tabla, los controles, las nulas por bloques y las
utilidades OLS/HC3 (mismo diseno metodologico que E11-07 / v9, ver ITERACION_12_diseno.md, "Metodo comun").

Estado Z_i = [1, controles..., xG_i, xC_i] (dim p = 1+nc+2). Solo se reportan/anotan los coeficientes de xG,xC
(ultimas 2 columnas del estado); controles+intercepto pueden ser "fijos" (Q=0 -> equivalente a OLS recursivo /
expansivo dentro del mismo filtro) o "paseo lento" (Q pequeno pero >0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v10")
import e11_07_lib as E  # noqa: E402
import v9_lib as L  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent.parent
SALIDA = RAIZ / "experimentos" / "v10"
KPI = E.KPI
CONTROLES = E.C_SIN_ESPESOR
X_PRINCIPAL = E.X_PRINCIPAL
PASO = 10
START = 100          # igual que v9 (W=100): primer batch anotado
BURNIN = 100          # ventana usada para fijar hiperparametros por verosimilitud
NINIT = 40            # batches usados para inicializar beta0,P0 por OLS (evita saltos del prior difuso)
SEED = 42
P0_SCALE = 1.0e4       # prior difuso (solo si no se usa init_ols)


# --------------------------------------------------------------------------- filtro nucleo
def _predict(beta: np.ndarray, P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return beta, P + Q  # transicion identidad


def _update(beta_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray, y_i: float, R: float):
    yhat = z @ beta_pred
    v = y_i - yhat
    Pz = P_pred @ z
    Fv = float(z @ Pz + R)
    K = Pz / Fv
    beta = beta_pred + K * v
    P = P_pred - np.outer(K, Pz)
    P = 0.5 * (P + P.T)
    return beta, P, v, Fv


def init_ols(Z: np.ndarray, y: np.ndarray, ninit: int, inflar: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Inicializacion 'diffuse-to-proper': OLS clasico en los primeros `ninit` batches da beta0 y P0 = Cov(OLS)
    inflada (factor `inflar`), en vez de un prior difuso (evita saltos espurios de beta en los primeros pasos por
    Z'Z casi singular con pocas observaciones)."""
    Zi, yi = Z[:ninit], y[:ninit]
    beta0, *_ = np.linalg.lstsq(Zi, yi, rcond=None)
    resid = yi - Zi @ beta0
    dof = max(ninit - Zi.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(Zi.T @ Zi)
    P0 = XtX_inv * sigma2 * inflar
    return beta0, P0


def kalman_run(Z: np.ndarray, y: np.ndarray, Q_diag: np.ndarray, R: float, beta0: np.ndarray | None = None,
               P0_scale: float = P0_SCALE, P0: np.ndarray | None = None, i0: int = 0) -> dict:
    """Corre el filtro secuencialmente desde `i0` (beta0,P0 = prior en i0, tipicamente de `init_ols` sobre los
    batches [0,i0) para evitar saltos de un prior difuso) hasta el final. Para i<i0 se rellena con beta0/P0 (no
    filtrado, solo para mantener arreglos de largo n). Devuelve, para cada i>=i0: beta_{i|i-1} (prediccion
    un-paso-adelante, usa solo el pasado), P_{i|i-1} (matriz completa), beta_{i|i} (filtrado, tras absorber y_i),
    innovacion v_i y su varianza F_i."""
    n, p = Z.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    P = (np.eye(p) * P0_scale) if P0 is None else P0.copy()
    Q = np.diag(Q_diag)
    beta_pred = np.zeros((n, p)); P_pred_diag = np.zeros((n, p))
    beta_filt = np.zeros((n, p)); P_filt_diag = np.zeros((n, p))
    P_pred_full = np.zeros((n, p, p)); P_filt_full = np.zeros((n, p, p))
    innov = np.full(n, np.nan); Fvar = np.full(n, np.nan)
    if i0 > 0:
        beta_pred[:i0] = beta; P_pred_diag[:i0] = np.diag(P); P_pred_full[:i0] = P
        beta_filt[:i0] = beta; P_filt_diag[:i0] = np.diag(P); P_filt_full[:i0] = P
    for i in range(i0, n):
        bp, Pp = _predict(beta, P, Q)
        beta_pred[i] = bp; P_pred_diag[i] = np.diag(Pp); P_pred_full[i] = Pp
        beta, P, v, Fv = _update(bp, Pp, Z[i], y[i], R)
        beta_filt[i] = beta; P_filt_diag[i] = np.diag(P); P_filt_full[i] = P
        innov[i] = v; Fvar[i] = Fv
    return {"beta_pred": beta_pred, "P_pred_diag": P_pred_diag, "beta_filt": beta_filt, "P_filt_diag": P_filt_diag,
            "P_pred_full": P_pred_full, "P_filt_full": P_filt_full, "innov": innov, "Fvar": Fvar, "Q": Q, "R": R,
            "i0": i0}


def score_continuous(Z: np.ndarray, y: np.ndarray, Q_diag: np.ndarray, R: float, start: int = START,
                      ne: int = 2, ninit: int = NINIT) -> tuple[np.ndarray, dict]:
    """Puntaje prospectivo actualizado batch a batch: score_i = beta_pred[i,-ne:] @ Z[i,-ne:], i>=start.
    beta0,P0 se inicializan por OLS en los primeros `ninit` batches (init_ols); el filtro corre desde ahi."""
    beta0, P0 = init_ols(Z, y, ninit)
    res = kalman_run(Z, y, Q_diag, R, beta0=beta0, P0=P0, i0=ninit)
    n = len(y)
    score = np.full(n, np.nan)
    for i in range(start, n):
        score[i] = Z[i, -ne:] @ res["beta_pred"][i, -ne:]
    return score, res


def _absorb_range(Z, y, lo, hi, beta, P, Q, R):
    for i in range(lo, hi):
        bp, Pp = _predict(beta, P, Q)
        beta, P, _, _ = _update(bp, Pp, Z[i], y[i], R)
    return beta, P


def score_periodic(Z: np.ndarray, y: np.ndarray, Q_diag: np.ndarray, R: float, start: int = START, paso: int = PASO,
                    ne: int = 2, ninit: int = NINIT) -> np.ndarray:
    """Variante de latencia igual a v9: beta se congela por bloques de `paso` batches (se predice con
    beta_{s|s-1}, constante para todo el bloque), y solo se actualiza (absorbe el bloque) al terminarlo."""
    n, p = Z.shape
    Q = np.diag(Q_diag)
    beta, P = init_ols(Z, y, ninit)
    beta, P = _absorb_range(Z, y, ninit, start, beta, P, Q, R)
    score = np.full(n, np.nan)
    s = start
    while s < n:
        hi = min(s + paso, n)
        score[s:hi] = Z[s:hi, -ne:] @ beta[-ne:]
        beta, P = _absorb_range(Z, y, s, hi, beta, P, Q, R)
        s = hi
    return score


# --------------------------------------------------------------------------- seleccion de hiperparametros
def loglik_predictiva(Z, y, Q_diag, R, beta0, P0, i0, lo, hi) -> float:
    res = kalman_run(Z, y, Q_diag, R, beta0=beta0, P0=P0, i0=i0)
    v, F = res["innov"][lo:hi], res["Fvar"][lo:hi]
    return float(np.sum(-0.5 * np.log(2 * np.pi * F) - 0.5 * v ** 2 / F))


def elegir_hiperparametros(Z, y, nc: int, grid_q: list[float], burnin: int = BURNIN, r_fijo: float | None = None,
                            q_ctrl: float = 0.0, ninit: int = NINIT) -> dict:
    """Rejilla (q_G, q_C) por verosimilitud predictiva un-paso-adelante SOLO en los primeros `burnin` batches:
    beta0,P0 se inicializan por OLS en los primeros `ninit` batches (init_ols) y la verosimilitud se evalua en
    [ninit, burnin) -- estrictamente pasado respecto de `start`=burnin. R se fija por la varianza residual de un
    OLS clasico en los `burnin` batches (o se pasa fijo)."""
    Zb, yb = Z[:burnin], y[:burnin]
    if r_fijo is None:
        r = E.ols_classic(Zb[:, 1:], yb)  # con columna de unos agregada dentro
        R = float(r["resid"] @ r["resid"]) / max(len(yb) - Zb.shape[1], 1)
    else:
        R = r_fijo
    beta0, P0 = init_ols(Z, y, ninit)
    mejor = None
    filas = []
    for qg in grid_q:
        for qc in grid_q:
            Qd = np.array([q_ctrl] * (1 + nc) + [qg, qc])
            ll = loglik_predictiva(Z, y, Qd, R, beta0, P0, ninit, ninit, burnin)
            filas.append({"q_G": qg, "q_C": qc, "R": R, "loglik_burnin": ll})
            if mejor is None or ll > mejor[0]:
                mejor = (ll, qg, qc)
    tabla = pd.DataFrame(filas).sort_values("loglik_burnin", ascending=False)
    return {"q_G": mejor[1], "q_C": mejor[2], "R": R, "loglik": mejor[0], "tabla": tabla}


def kalman_adaptive_score(Z, y, nc: int, grid_q: list[float], first_select: int = BURNIN, reestim_step: int = 20,
                           ninit: int = NINIT, q_ctrl: float = 0.0, r_fijo: float | None = None,
                           start: int = START, ne: int = 2) -> tuple[np.ndarray, dict]:
    """Version adaptativa: re-selecciona (q_G,q_C) cada `reestim_step` batches (desde que hay >= first_select
    observaciones) por verosimilitud predictiva evaluada en TODO el pasado ya filtrado desde `ninit` (ventana
    expansiva, estrictamente causal: en el checkpoint s solo usa datos < s). R se fija UNA vez con OLS en los
    primeros `first_select` batches (no se re-estima, por costo)."""
    n, p = Z.shape
    Zb, yb = Z[:first_select], y[:first_select]
    if r_fijo is None:
        r = E.ols_classic(Zb[:, 1:], yb)
        R = float(r["resid"] @ r["resid"]) / max(len(yb) - Zb.shape[1], 1)
    else:
        R = r_fijo
    beta0, P0 = init_ols(Z, y, ninit)
    beta, P = beta0.copy(), P0.copy()
    beta_pred_hist = np.zeros((n, p))
    qg_hist = np.full(n, np.nan); qc_hist = np.full(n, np.nan)
    Qcur = np.array([q_ctrl] * (1 + nc) + [0.0, 0.0])
    for i in range(ninit, first_select):
        bp, Pp = _predict(beta, P, np.diag(Qcur))
        beta_pred_hist[i] = bp
        beta, P, _, _ = _update(bp, Pp, Z[i], y[i], R)
    s = first_select
    while s < n:
        mejor = None
        for qg in grid_q:
            for qc in grid_q:
                Qd = [q_ctrl] * (1 + nc) + [qg, qc]
                ll = loglik_predictiva(Z[:s], y[:s], Qd, R, beta0, P0, ninit, ninit, s)
                if mejor is None or ll > mejor[0]:
                    mejor = (ll, qg, qc)
        Qcur = np.array([q_ctrl] * (1 + nc) + [mejor[1], mejor[2]])
        hi = min(s + reestim_step, n)
        for i in range(s, hi):
            bp, Pp = _predict(beta, P, np.diag(Qcur))
            beta_pred_hist[i] = bp; qg_hist[i] = mejor[1]; qc_hist[i] = mejor[2]
            beta, P, _, _ = _update(bp, Pp, Z[i], y[i], R)
        s = hi
    score = np.full(n, np.nan)
    for i in range(start, n):
        score[i] = Z[i, -ne:] @ beta_pred_hist[i, -ne:]
    return score, {"beta_pred": beta_pred_hist, "qg_hist": qg_hist, "qc_hist": qc_hist, "R": R}


# --------------------------------------------------------------------------- suavizado (RTS), descriptivo
def rts_smoother(res: dict) -> dict:
    beta_pred, P_pred_full = res["beta_pred"], res["P_pred_full"]
    beta_filt, P_filt_full = res["beta_filt"], res["P_filt_full"]
    n, p = beta_filt.shape
    beta_s = beta_filt.copy(); P_s = P_filt_full.copy()
    for i in range(n - 2, -1, -1):
        P_pred_next = P_pred_full[i + 1]
        J = P_filt_full[i] @ np.linalg.pinv(P_pred_next)
        beta_s[i] = beta_filt[i] + J @ (beta_s[i + 1] - beta_pred[i + 1])
        P_s[i] = P_filt_full[i] + J @ (P_s[i + 1] - P_pred_next) @ J.T
    return {"beta_smooth": beta_s, "P_smooth_diag": np.diagonal(P_s, axis1=1, axis2=2)}


# --------------------------------------------------------------------------- preparar Z con controles+exposiciones
def construir_Z(D: dict) -> np.ndarray:
    n = D["Xctrl"].shape[0]
    return np.column_stack([np.ones(n), D["Xctrl"], D["Xexp"]])
