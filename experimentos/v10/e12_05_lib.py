"""E12-05 -- libreria: residuos a - E[a|S] para las palancas NO evaluadas por el canal directo accion -> KPI en E11-01
(H5, `ITERACION_12_diseno.md`): posicion vertical de lanza, O2 de lanza, enriquecimiento O2, aire, estequiometria de
planta, caudal total de gas de lanza (GN+O2+aire) y presion de punta.

Reutiliza el "Metodo comun": E[a|S] = HGB cross-fitted por batch (GroupKFold en DEV, lockbox con modelo entrenado en
todo DEV), S = ESTADO_R / ESTADO_F de v9_lib (incluye `posicion_vertical_lanza_mm_prev`), residuo estandarizado por
la sd dentro de (fase, orden) en DEV. A diferencia de v9_lib.residuos_escalon (que separa Reduccion en Rt=R0-R1 /
Rl=R2-R3), aqui el grupo es SIMPLE por fase: "R" = R0-R3, "F" = F1-F6 (F0 = talon, sin decision modelable), tal como
pide el diseno E12-05 ("para cada candidata y fase").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v9", RAIZ / "experimentos" / "v8"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v9_lib as L9      # noqa: E402
import v8_lib as L8      # noqa: E402
import modelo_predictivo_v8 as mp8  # noqa: E402
import e11_07_lib as E7  # noqa: E402

SALIDA = RAIZ / "experimentos" / "v10"
CACHE_RES = RAIZ / "experimentos" / "cache" / "e12_05_residuos.pkl"
SEED = 42
KPI = L9.KPI
CONTROLES = L9.CONTROLES               # sin espesor la version C_SIN_ESPESOR (abajo)
C_SIN_ESPESOR = E7.C_SIN_ESPESOR
CANALES = ["f_metal", "f_dross", "f_polvo", "sn_perdido_escoria_frac"]
ESTADOS = L9.ESTADOS

# candidatas E12-05 + referencia (gn, carbon -- ya evaluadas en E11-01, r~0.8 con o2 esperado por diseno del quemador)
PALANCAS = {
    "lanza": "posicion_vertical_lanza_mm",
    "o2": "tasa_o2_nm3_min",
    "enriq": "enriquecimiento_o2_lanza_planta_pct",
    "aire": "tasa_aire_nm3_min",
    "esteq": "estequiometria_lanza_planta_pct",
    "caudal": "tasa_caudal_lanza_total_nm3_min",
    "ppunta": "presion_punta_lanza_kpa",
}
PALANCAS_REF = {"gn": "tasa_gn_nm3_min", "carbon": "tasa_feed_Carbon_kg_min"}
TODAS = {**PALANCAS, **PALANCAS_REF}


def preparar_df() -> pd.DataFrame:
    df = L8.cargar_df()
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    df["tasa_caudal_lanza_total_nm3_min"] = (df["tasa_gn_nm3_min"].fillna(0) + df["tasa_o2_nm3_min"].fillna(0)
                                              + df["tasa_aire_nm3_min"].fillna(0))
    return df


def grupo_simple(fase: str, orden: int) -> str | None:
    if fase == "Fusión":
        return "F" if orden >= 1 else None
    if fase == "Reducción":
        return "R" if 0 <= orden <= 3 else None
    return None


def residuos_escalon_e12(df: pd.DataFrame | None = None, n_splits: int = 5, seed: int = SEED,
                          recalcular: bool = False, cache: Path | None = CACHE_RES,
                          palancas: dict | None = None, estados: dict | None = None) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold
    if cache is not None and cache.exists() and not recalcular:
        return pd.read_pickle(cache)
    if df is None:
        df = preparar_df()
    palancas = palancas or TODAS
    estados = estados or ESTADOS
    base = mp8._preparar_base(df)
    dev, lockbox = L8.mp.split_dev_lockbox(df)
    out = []
    for fase, S in estados.items():
        L8.asegurar_seguras([s for s in S if s not in ("tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min")])
        d = base[base["fase_proceso"] == fase].copy()
        d["grupo"] = [grupo_simple(fase, int(o)) for o in d["orden_escalon_fase"]]
        d = d[d["grupo"].notna()].reset_index(drop=True)
        es_dev = d["Batch"].isin(dev).to_numpy()
        t = d[["Batch", "fase_proceso", "orden_escalon_fase", "grupo", "fecha_inicio"]].copy()
        t["dur_min"] = (pd.to_datetime(d["fecha_final"]) - pd.to_datetime(d["fecha_inicio"])).dt.total_seconds() / 60.0
        t["es_dev"] = es_dev
        for p, col in palancas.items():
            if col not in d.columns:
                continue
            ok = d[col].notna().to_numpy()
            ahat = np.full(len(d), np.nan)
            idx_dev = np.where(es_dev & ok)[0]
            if len(idx_dev) < n_splits:
                t[f"a__{p}"] = d[col].to_numpy(); t[f"ahat__{p}"] = ahat; t[f"res__{p}"] = np.nan; t[f"z__{p}"] = np.nan
                continue
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
    if cache is not None:
        cache.parent.mkdir(exist_ok=True); res.to_pickle(cache)
    return res


def exposiciones_batch_e12(res: pd.DataFrame, palancas: list[str] | None = None, grupos=("F", "R"),
                            cuadratico: tuple[str, ...] = ("lanza",)) -> pd.DataFrame:
    """x_{grupo}_{palanca} = media(z) por batch; para `cuadratico` ademas x_{grupo}_{palanca}_sq = media(z^2)
    (desviacion respecto de la posicion/tasa habitual, sin importar signo)."""
    palancas = palancas or list(PALANCAS) + list(PALANCAS_REF)
    filas = {}
    for g in grupos:
        sub = res[res["grupo"] == g]
        for p in palancas:
            col = f"z__{p}"
            if col not in sub.columns:
                continue
            v = sub[col]
            filas[f"x_{g}_{p}"] = v.groupby(sub["Batch"]).mean()
            if p in cuadratico:
                filas[f"x_{g}_{p}_sq"] = (v ** 2).groupby(sub["Batch"]).mean()
    return pd.DataFrame(filas)


def tabla_batch_e12(res: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    """e11_06b_tabla_batch.csv (KPI, canales, controles, xG, xC, es_dev, idx_cronologico) + exposiciones E12-05."""
    t = pd.read_csv(SALIDA.parent / "v9" / "e11_06b_tabla_batch.csv", index_col=0)
    if res is None:
        res = residuos_escalon_e12()
    exp = exposiciones_batch_e12(res, **kw)
    out = t.join(exp, how="left").sort_values("idx_cronologico")
    return out


def ols_hc3(y, X):
    return L9.ols_hc3(y, X)


def bh(pvals: pd.Series) -> pd.Series:
    """q-values Benjamini-Hochberg."""
    p = pvals.dropna().sort_values()
    n = len(p)
    q = (p * n / (np.arange(n) + 1)).values
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = pd.Series(np.nan, index=pvals.index)
    out.loc[p.index] = np.minimum(q, 1.0)
    return out


def bootstrap_batch(t: pd.DataFrame, y_col: str, x_col: str, controles: list[str], n_boot: int = 1000,
                     seed: int = SEED) -> dict:
    d = t[[y_col, x_col] + controles].dropna()
    batches = d.index.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(batches)
    coefs = []
    for _ in range(n_boot):
        samp = rng.choice(batches, size=n, replace=True)
        dd = d.loc[samp]
        try:
            f = ols_hc3(dd[y_col], dd[[x_col] + controles])
            coefs.append(f.params[x_col])
        except Exception:
            continue
    coefs = np.array(coefs)
    return {"n_boot": len(coefs), "ci_lo": float(np.percentile(coefs, 2.5)), "ci_hi": float(np.percentile(coefs, 97.5)),
            "excluye_0": bool(np.percentile(coefs, 2.5) > 0 or np.percentile(coefs, 97.5) < 0)}


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = preparar_df()
    res = residuos_escalon_e12(df, recalcular=True)
    print(res.shape, f"{time.time()-t0:.0f}s")
    from sklearn.metrics import r2_score
    for (f, g), s in res[res.es_dev].groupby(["fase_proceso", "grupo"]):
        r2 = {}
        for p in TODAS:
            m = s[f"ahat__{p}"].notna() & s[f"a__{p}"].notna() if f"ahat__{p}" in s.columns else None
            if m is not None and m.sum() > 5:
                r2[p] = round(r2_score(s[f"a__{p}"][m], s[f"ahat__{p}"][m]), 3)
        print(f, g, r2)
    t = tabla_batch_e12(res)
    print(t.shape)
    t.to_csv(SALIDA / "e12_05_tabla_batch.csv")
