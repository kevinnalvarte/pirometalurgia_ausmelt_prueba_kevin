"""E13-03 -- libreria compartida para la auditoria adversarial de E13-01 (contraccion James-Stein).
Generaliza wf() a una FAMILIA de contracciones y agrega generadores de nulas alternativas
(bloques de distinto tamano, desplazamiento circular completo, permutacion de signo por bloque).
Todo vectorizado con numpy, reutiliza e11_07_lib (W7) tal como e13_01.
"""
import sys
sys.path.insert(0, "experimentos/v9")
import numpy as np
import e11_07_lib as W7

MODOS = ["ols", "js", "hard1", "hard1.5", "hard2", "lineal", "eb", "js_joint"]


def contraer(b, se, modo, hist_b=None, hist_se=None):
    t_ = np.divide(b, se, out=np.full_like(b, np.inf), where=se > 0)
    if modo == "ols":
        return b
    if modo == "js":
        return b * np.clip(1 - (se / np.where(b == 0, 1e-9, b)) ** 2, 0, None)
    if modo == "hard1":
        return np.where(np.abs(t_) >= 1.0, b, 0.0)
    if modo == "hard1.5":
        return np.where(np.abs(t_) >= 1.5, b, 0.0)
    if modo == "hard2":
        return np.where(np.abs(t_) >= 2.0, b, 0.0)
    if modo == "lineal":
        return b * np.clip(np.abs(t_) / 2.0, 0, 1)
    if modo == "eb":
        if hist_b is not None and len(hist_b) >= 2:
            hb = np.array(hist_b); hse = np.array(hist_se)
            tau2 = np.clip(np.mean(hb ** 2 - hse ** 2, axis=0), 0, None)
            c = np.divide(tau2, tau2 + se ** 2, out=np.ones_like(se), where=(tau2 + se ** 2) > 0)
            return c * b
        return b  # sin historia suficiente -> sin contraccion (conservador)
    if modo == "js_joint":
        S = float(np.sum(t_ ** 2)); k = len(b)
        c = max(0.0, 1 - k / S) if S > 0 else 0.0
        return c * b
    raise ValueError(modo)


def wf(Xe, Cc, yy, W=100, paso=10, modo="js", start=None):
    """Igual que e13_01.wf pero con la familia completa de contracciones (contraer())."""
    nn = len(yy); sc = np.full(nn, np.nan); s = start or (W or 100)
    hist_b, hist_se = [], []
    while s < nn:
        lo = 0 if W is None else max(0, s - W)
        Z = np.column_stack([Cc[lo:s], Xe[lo:s]])
        r = W7.ols_classic(Z, yy[lo:s])
        ne = Xe.shape[1]
        b, se = r["beta"][-ne:], r["se"][-ne:]
        bt = contraer(b, se, modo, hist_b, hist_se)
        if modo == "eb":
            hist_b.append(b); hist_se.append(se)
        sc[s:s + paso] = Xe[s:s + paso] @ bt
        s += paso
    return sc


# --------------------------------------------------------------------------- nulas alternativas
def bloques(n, tam, seed=0):
    return W7.bloques_cronologicos(n, tam, seed=seed)


def permutar_bloques(Xexp, bl, rng):
    return W7.permutar_bloques(Xexp, bl, rng)


def signo_bloques(Xexp, bl, rng):
    Xp = Xexp.copy()
    for b in bl:
        s = rng.choice([-1.0, 1.0])
        Xp[b] = Xexp[b] * s
    return Xp


def circular_shifts_todos(Xexp, margen=15):
    n = len(Xexp)
    return [W7.circular_shift(Xexp, k) for k in range(margen, n - margen)]


def p_perm(obs_t, nul_ts):
    nul_ts = np.asarray(nul_ts)
    return (np.sum(nul_ts >= obs_t) + 1) / (len(nul_ts) + 1)
