"""E12-02 -- libreria: valoracion prospectiva POR CANAL de perdida (H2, ITERACION_12_diseno.md).

Reusa `experimentos/v9/e11_07_lib.py` (OLS numpy puro, walk-forward de ventana movil, nulas por permutacion en
bloques) y agrega solo lo que ese modulo no cubre:
  - `beta_se_series`: serie beta(t), se(t) por canal, estimada SOLO con el pasado (igual esquema que
    `walk_forward_score` pero devolviendo el coeficiente y su error estandar en vez de aplicarlo ya).
  - `apply_beta_series`: aplica una serie de beta ya calculada (posiblemente modificada: contraida o con signo
    restringido) a las exposiciones, reproduciendo el mismo puntaje prospectivo que `walk_forward_score`.
  - `score_esquema`: construye el puntaje KPI de cada uno de los 6 esquemas A-F a partir de las exposiciones
    (posiblemente permutadas) y las variables canal, TODO estimado solo con el pasado.
  - clasificacion estable/inestable por canal: fija a priori (teoria) y prospectiva sin fuga (regla de 2 ventanas
    disjuntas mas reciente vs beta expansivo, decidida en cada paso SOLO con datos anteriores a ese paso).

Identidad exacta verificada (ver `e12_02_01_identidad_estabilidad.py`):
    recuperacion_refinada_pct = 100 * f_metal * (1 - sn_perdido_escoria_frac)
    con f_metal = 1 - f_dross - f_polvo (suma exacta a 1, verificado a 1e-16).
Esto NO es una suma lineal de los 3 canales de perdida; el esquema aditivo -100*Sum(beta_canal) que usa H2 es la
aproximacion de primer orden (ya usada y validada en `e11_07_04_canales.py`: el "compuesto" reconstruido reproduce
la pendiente/t del KPI directo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v10")
import e11_07_lib as E  # noqa: E402

RAIZ = E.RAIZ
SALIDA = RAIZ / "experimentos" / "v10"
KPI = E.KPI
CANALES = ["f_dross", "f_polvo", "sn_perdido_escoria_frac"]
CANAL_ESTABLE_TEORIA = {"f_dross": True, "sn_perdido_escoria_frac": True, "f_polvo": False}
# pares (canal, palanca) con signo de teoria inequivoco (indice en X_PRINCIPAL = ["xG","xC"]: 0=xG, 1=xC)
SIGNO_TEORIA = {("f_dross", 0): +1, ("sn_perdido_escoria_frac", 1): -1}
X_PRINCIPAL = E.X_PRINCIPAL
PASO = E.PASO
START = 100
SEED = E.SEED
KG_SN_POR_PP = 512.0


def cargar(extra_cols: list[str] | None = None) -> pd.DataFrame:
    t = E.cargar_tabla(extra_cols)
    for c in CANALES:
        t[c + "_pp"] = t[c] * 100.0
    return t


def preparar_canal(t: pd.DataFrame, canal_pp: str) -> dict:
    return E.preparar(t, X_PRINCIPAL, y_col=canal_pp)


def preparar_todo(t: pd.DataFrame, controles: list[str] | None = None) -> dict:
    """Una sola fila de dropna sobre KPI + los 3 canales + controles + xG,xC, para que todos los esquemas
    operen exactamente sobre el mismo conjunto de batches (evita desalineos por NaN especificos de un canal)."""
    controles = E.C_SIN_ESPESOR if controles is None else controles
    canales_pp = [c + "_pp" for c in CANALES]
    cols = X_PRINCIPAL + controles + [KPI] + canales_pp
    d = t.dropna(subset=cols).copy()
    return {"Xexp": d[X_PRINCIPAL].to_numpy(float), "Xctrl": d[controles].to_numpy(float),
            "y_kpi": d[KPI].to_numpy(float), "ycan": {c: d[c].to_numpy(float) for c in canales_pp},
            "es_dev": d["es_dev"].to_numpy(bool), "idx": d.index.to_numpy(), "n": len(d), "controles": controles,
            "t_df": d[[KPI] + controles]}


# --------------------------------------------------------------------------- beta/se por canal, solo con el pasado
def beta_se_series(Xexp: np.ndarray, Xctrl: np.ndarray, y: np.ndarray, W: int | None,
                    paso: int = PASO, start: int = START) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s_grid, beta(n_steps,ne), se(n_steps,ne). Mismo esquema temporal que `E.walk_forward_score`
    (ventana dura W o expansiva si W=None), pero devuelve el coeficiente en vez de aplicarlo."""
    n = len(y); nc = Xctrl.shape[1]; ne = Xexp.shape[1]
    s_grid = np.arange(start, n, paso)
    betas = np.full((len(s_grid), ne), np.nan)
    ses = np.full((len(s_grid), ne), np.nan)
    for i, s in enumerate(s_grid):
        lo = 0 if W is None else max(0, s - W)
        Z = np.column_stack([Xctrl[lo:s], Xexp[lo:s]])
        r = E.ols_classic(Z, y[lo:s])
        betas[i] = r["beta"][1 + nc:]
        ses[i] = r["se"][1 + nc:]
    return s_grid, betas, ses


def apply_beta_series(Xexp: np.ndarray, s_grid: np.ndarray, betas: np.ndarray, n: int, paso: int = PASO) -> np.ndarray:
    score = np.full(n, np.nan)
    for i, s in enumerate(s_grid):
        hi = min(s + paso, n)
        score[s:hi] = Xexp[s:hi] @ betas[i]
    return score


# --------------------------------------------------------------------------- esquemas A-F
def score_esquema(nombre: str, Xexp: np.ndarray, Xctrl: np.ndarray, y_kpi: np.ndarray, ycan: dict, n: int) -> np.ndarray:
    """Puntaje prospectivo del KPI (pp), positivo = mejora prevista. `ycan`: dict canal_pp -> array (mismo largo n)."""
    kw = dict(paso=PASO, tmin=0.0, start=START)
    if nombre == "A_directo":
        return E.walk_forward_score(Xexp, Xctrl, y_kpi, 100, **kw)
    if nombre == "B_canales_W100":
        tot = sum(E.walk_forward_score(Xexp, Xctrl, ycan[c + "_pp"], 100, **kw) for c in CANALES)
        return -tot
    if nombre in ("C_60", "C_100"):
        w_polvo = 60 if nombre == "C_60" else 100
        s_dross = E.walk_forward_score(Xexp, Xctrl, ycan["f_dross_pp"], None, **kw)
        s_esc = E.walk_forward_score(Xexp, Xctrl, ycan["sn_perdido_escoria_frac_pp"], None, **kw)
        s_polvo = E.walk_forward_score(Xexp, Xctrl, ycan["f_polvo_pp"], w_polvo, **kw)
        return -(s_dross + s_esc + s_polvo)
    if nombre == "D_estables":
        s_dross = E.walk_forward_score(Xexp, Xctrl, ycan["f_dross_pp"], None, **kw)
        s_esc = E.walk_forward_score(Xexp, Xctrl, ycan["sn_perdido_escoria_frac_pp"], None, **kw)
        return -(s_dross + s_esc)
    if nombre == "E_eb":
        tot = np.zeros(n)
        for c in CANALES:
            sg, b100, se100 = beta_se_series(Xexp, Xctrl, ycan[c + "_pp"], 100)
            _, bexp, seexp = beta_se_series(Xexp, Xctrl, ycan[c + "_pp"], None)
            w1 = 1.0 / np.clip(se100, 1e-6, None) ** 2
            w2 = 1.0 / np.clip(seexp, 1e-6, None) ** 2
            bshrink = (w1 * b100 + w2 * bexp) / (w1 + w2)
            tot += apply_beta_series(Xexp, sg, bshrink, n)
        return -tot
    if nombre == "F_signo":
        tot = np.zeros(n)
        for c in CANALES:
            sg, b, _ = beta_se_series(Xexp, Xctrl, ycan[c + "_pp"], 100)
            bC = b.copy()
            for (cc, j), signo in SIGNO_TEORIA.items():
                if cc == c:
                    bC[:, j] = np.clip(bC[:, j], 0, None) if signo > 0 else np.clip(bC[:, j], None, 0)
            tot += apply_beta_series(Xexp, sg, bC, n)
        return -tot
    raise ValueError(nombre)


ESQUEMAS = ["A_directo", "B_canales_W100", "C_60", "C_100", "D_estables", "E_eb", "F_signo"]


# --------------------------------------------------------------------------- beta_KPI,j(t) y su precision por esquema
def beta_kpi_series_por_esquema(nombre: str, Xexp: np.ndarray, Xctrl: np.ndarray, y_kpi: np.ndarray, ycan: dict):
    """s_grid, beta_KPI(n_steps,2), se_KPI(n_steps,2) [se combinada por canal asumiendo independencia condicional
    en X -- aproximacion razonable ya que son residuos de canales de perdida distintos, no exacta]. Para A: KPI
    directo (sin negar). Para B/C/D/E/F: beta_KPI = -Sum_canal beta_canal (con las mismas ventanas que score_esquema)."""
    if nombre == "A_directo":
        return beta_se_series(Xexp, Xctrl, y_kpi, 100)
    if nombre == "B_canales_W100":
        canales = CANALES
        partes = [beta_se_series(Xexp, Xctrl, ycan[c + "_pp"], 100) for c in canales]
    elif nombre in ("C_60", "C_100"):
        w_polvo = 60 if nombre == "C_60" else 100
        partes = [beta_se_series(Xexp, Xctrl, ycan["f_dross_pp"], None),
                  beta_se_series(Xexp, Xctrl, ycan["sn_perdido_escoria_frac_pp"], None),
                  beta_se_series(Xexp, Xctrl, ycan["f_polvo_pp"], w_polvo)]
    elif nombre == "D_estables":
        partes = [beta_se_series(Xexp, Xctrl, ycan["f_dross_pp"], None),
                  beta_se_series(Xexp, Xctrl, ycan["sn_perdido_escoria_frac_pp"], None)]
    elif nombre == "E_eb":
        partes = []
        for c in CANALES:
            sg, b100, se100 = beta_se_series(Xexp, Xctrl, ycan[c + "_pp"], 100)
            _, bexp, seexp = beta_se_series(Xexp, Xctrl, ycan[c + "_pp"], None)
            w1 = 1.0 / np.clip(se100, 1e-6, None) ** 2
            w2 = 1.0 / np.clip(seexp, 1e-6, None) ** 2
            bshrink = (w1 * b100 + w2 * bexp) / (w1 + w2)
            se_shrink = np.sqrt(1.0 / (w1 + w2))
            partes.append((sg, bshrink, se_shrink))
    elif nombre == "F_signo":
        partes = []
        for c in CANALES:
            sg, b, se = beta_se_series(Xexp, Xctrl, ycan[c + "_pp"], 100)
            bC = b.copy()
            for (cc, j), signo in SIGNO_TEORIA.items():
                if cc == c:
                    bC[:, j] = np.clip(bC[:, j], 0, None) if signo > 0 else np.clip(bC[:, j], None, 0)
            partes.append((sg, bC, se))
    else:
        raise ValueError(nombre)
    s_grid = partes[0][0]
    beta_kpi = -sum(p[1] for p in partes)
    se_kpi = np.sqrt(sum(p[2] ** 2 for p in partes))
    return s_grid, beta_kpi, se_kpi


# --------------------------------------------------------------------------- clasificacion estable/inestable sin fuga
def clasif_prospectiva(Xexp: np.ndarray, Xctrl: np.ndarray, ycan: dict, canal_pp: str, j: int,
                        bloque: int = 80, start: int = 2 * 80 + START) -> pd.DataFrame:
    """Para el par (canal, palanca j): en cada paso s (>= 2*bloque + START), estable=True si el signo del beta
    EXPANSIVO (0:s) coincide con el signo en las 2 ventanas disjuntas mas recientes [s-2*bloque:s-bloque) y
    [s-bloque:s). Todo con datos < s (sin fuga)."""
    y = ycan[canal_pp]
    n = len(y); nc = Xctrl.shape[1]
    filas = []
    s_grid = np.arange(start, n, PASO)
    for s in s_grid:
        b_exp = E.ols_classic(np.column_stack([Xctrl[:s], Xexp[:s]]), y[:s])["beta"][1 + nc + j]
        b_w1 = E.ols_classic(np.column_stack([Xctrl[s - 2 * bloque:s - bloque], Xexp[s - 2 * bloque:s - bloque]]),
                              y[s - 2 * bloque:s - bloque])["beta"][1 + nc + j]
        b_w2 = E.ols_classic(np.column_stack([Xctrl[s - bloque:s], Xexp[s - bloque:s]]),
                              y[s - bloque:s])["beta"][1 + nc + j]
        estable = (np.sign(b_exp) == np.sign(b_w1)) and (np.sign(b_exp) == np.sign(b_w2)) and np.sign(b_exp) != 0
        filas.append({"s": int(s), "beta_expansivo": b_exp, "beta_ventana1": b_w1, "beta_ventana2": b_w2, "estable": bool(estable)})
    return pd.DataFrame(filas)
