"""E15-02 -- libreria compartida de la auditoria adversarial de E15-01 (receta ex-ante de Alimentacion).
Reutiliza e11_07_lib (W7, walk-forward numpy puro) y e13_03_lib (contraccion JS/familia, nulas alternativas)
tal como hizo E13-03 con E13-01. Agrega: (a) re-estandarizacion ESTRICTA (solo pasado) del desvio ejecutado
vs receta por escalon, con variantes sin centrar y en unidades fisicas crudas; (b) construccion de
exposiciones por subconjunto de ordenes (R0-R1 vs R2-R3); (c) chequeos de "receta ex-post" (niveles
discretos, escalones por fecha, receta explicada por el estado pre-Reduccion).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v11")
import e11_07_lib as W7  # noqa: E402
import e13_03_lib as L13  # noqa: E402
import v9_lib as L  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent.parent
SAL = RAIZ / "experimentos" / "v13"
KPI = W7.KPI
CTRL = W7.C_SIN_ESPESOR
PALANCAS = ("gn", "carbon", "o2", "aire")
SEED = 42


def cargar():
    r = pd.read_pickle(SAL / "e15_01_escalones_receta.pkl")
    t = pd.read_csv(SAL / "e15_01_tabla_batch.csv", index_col=0)
    r = r.join(t[["idx_cronologico", "es_dev"]], on="Batch")
    r = r.sort_values(["idx_cronologico", "orden_escalon_fase"]).reset_index(drop=True)
    return r, t


# --------------------------------------------------------------------------- 1) re-estandarizacion estricta
def exposiciones_estrictas(r: pd.DataFrame, palancas=("gn", "carbon"), min_hist: int = 40) -> pd.DataFrame:
    """Para cada escalon (Batch, orden), media/sd de dev__p por ORDEN calculadas SOLO con batches
    estrictamente anteriores (expansivo, orden cronologico por idx_cronologico), exigiendo >= min_hist
    observaciones previas (si no, NaN = calentamiento). Devuelve batch x {d_{p}_exp, d_{p}_exp_sinctr,
    d_{p}_crudo} (media por batch de las 4 ordenes, ignorando NaN de calentamiento)."""
    out = {}
    for p in palancas:
        col = f"dev__{p}"
        d = r[["Batch", "idx_cronologico", "orden_escalon_fase", col]].copy()
        d = d.sort_values(["orden_escalon_fase", "idx_cronologico"])
        g = d.groupby("orden_escalon_fase")[col]
        # expanding shift(1): estadisticos de TODO lo anterior a la fila actual, dentro del mismo orden
        cnt = g.cumcount()
        mu = g.apply(lambda s: s.expanding().mean().shift(1)).reset_index(level=0, drop=True)
        sd = g.apply(lambda s: s.expanding().std().shift(1)).reset_index(level=0, drop=True)
        mu = mu.where(cnt >= min_hist); sd = sd.where(cnt >= min_hist)
        d["z_exp"] = (d[col] - mu) / sd
        d["z_sinctr"] = d[col] / sd
        d["crudo"] = d[col]
        d = d.sort_values(["idx_cronologico", "orden_escalon_fase"])
        b = d.groupby("Batch")[["z_exp", "z_sinctr", "crudo"]].mean()
        b.columns = [f"d_{p}_exp", f"d_{p}_sinctr", f"d_{p}_crudo"]
        out[p] = b
    res = pd.concat(out.values(), axis=1)
    return res


# --------------------------------------------------------------------------- exposiciones por subconjunto de ordenes
def exposicion_ordenes(r: pd.DataFrame, t: pd.DataFrame, palanca: str, ordenes: list[int]) -> pd.Series:
    """Reproduce EXACTO el estandarizado original de E15-01 (media/sd por orden en DEV, constante) pero
    promedia el z-score solo sobre el subconjunto `ordenes` (para dividir R0-R1 vs R2-R3)."""
    col = f"dev__{palanca}"
    esdev = r["es_dev"]
    mu = r.loc[esdev].groupby("orden_escalon_fase")[col].mean()
    sd = r.loc[esdev].groupby("orden_escalon_fase")[col].std()
    z = (r[col] - r.orden_escalon_fase.map(mu)) / r.orden_escalon_fase.map(sd)
    sub = r.loc[r.orden_escalon_fase.isin(ordenes)]
    return z.loc[sub.index].groupby(sub.Batch).mean()


# --------------------------------------------------------------------------- test prospectivo principal (envoltura)
def test_principal(t: pd.DataFrame, xcols: list[str], controles=None, W=100, paso=10, modo="js", start=None,
                    reps=1000, tam_bloque=20, seed=SEED):
    P = W7.preparar(t, xcols, controles=controles)
    X, C, y, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]
    sc = L13.wf(X, C, y, W, paso=paso, modo=modo, start=start)
    obs = W7.stats_final(sc, y, C)
    bl = L13.bloques(n, tam_bloque, seed=0)
    rng = np.random.default_rng(seed)
    nts = np.array([W7.stats_final(L13.wf(L13.permutar_bloques(X, bl, rng), C, y, W, paso=paso, modo=modo, start=start), y, C)["t"]
                     for _ in range(reps)])
    p = L13.p_perm(obs["t"], nts)
    return {"P": P, "obs": obs, "nul_t": nts, "p_perm": p}


def dev_mask(t_idx: pd.Index, tabla: pd.DataFrame, P: dict) -> np.ndarray:
    """No usado directamente: P['es_dev'] ya trae la mascara alineada."""
    return P["es_dev"]
