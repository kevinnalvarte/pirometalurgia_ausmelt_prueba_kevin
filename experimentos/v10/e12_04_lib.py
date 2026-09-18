"""E12-04 -- libreria: estados ejecutables para E[a|S] en Reduccion (H4, ITERACION_12_diseno.md).

Cuatro estados, todos verificados con `L8.asegurar_seguras`:
  S1  ("v9")      = mp8.ESTADO_R_V8 (k15 sin espesor) + orden + T_horno_prev   -- ensayo de t-1, referencia NO ejecutable al inicio.
  S2  ("t-2")     = L8.ESTADO_R_ONLINE -- leyes/inventarios _prev2 + en linea de t-1 + orden.  Para R0 equivale al cierre de F5,
                    para R1 al cierre de F6 (el _prev2 de R1 es F6).
  S3p ("plan")    = L8.ESTADO_R_PLAN, con las 8 columnas *_F6 sustituidas por *_F6p: para R0 (donde el cierre de F6 NO esta
                    disponible al inicio) usa el _prev2 de esa fila (= cierre de F5); para R1-R3 usa el _F6 tal cual (ya es el
                    cierre de F6, disponible con margen).
  S4  ("online")  = L8.ESTADO_R_ONLINE sin los 8 campos de ensayo (leyes/ratio/inventarios _prev2) -- solo en linea + orden.

Las columnas *_F6p se registran como STATE en la clasificacion anti-fuga (son la misma cantidad fisica que *_F6/ *_prev2, solo
resueltas por orden) para que `L8.asegurar_seguras` las acepte.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (_RAIZ, _RAIZ / "experimentos" / "v8", _RAIZ / "experimentos" / "v9"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_engineering as fe  # noqa: E402
import v8_lib as L8  # noqa: E402
import modelo_predictivo_v8 as mp8  # noqa: E402
import modelo_predictivo_v9 as mp9  # noqa: E402

SALIDA = _RAIZ / "experimentos" / "v10"
SEED = 42
KPI = L8.KPI_PRINCIPAL
PALANCAS = dict(mp9.PALANCAS_V9)              # {"gn": "tasa_gn_nm3_min", "carbon": "tasa_feed_Carbon_kg_min"}
CONTROLES_SIN_ESPESOR = [c for c in L8.CONTROLES_BATCH if c != "espesor_ladrillo_norm_mm"]
CANALES = ["f_dross", "sn_perdido_escoria_frac"]     # dross y Sn en escoria (v9_lib.CANALES)

# --------------------------------------------------------------------------- estado "plan ejecutable" (S3')
F6_PARES = {
    "ley_sn_escoria_pct_F6": "ley_sn_escoria_pct_prev2",
    "ley_feo_escoria_pct_F6": "ley_feo_escoria_pct_prev2",
    "ley_sio2_escoria_pct_F6": "ley_sio2_escoria_pct_prev2",
    "ley_cao_escoria_pct_F6": "ley_cao_escoria_pct_prev2",
    "ratio_sn_feo_F6": "ratio_sn_feo_prev2",
    "m6_sn_inv_kg_F6": "m6_sn_inv_kg_prev2",
    "m6_feo_inv_kg_F6": "m6_feo_inv_kg_prev2",
    "m6_resto_frac_F6": "m6_resto_frac_prev2",
}


def _registrar_seguras_planp() -> None:
    for c in F6_PARES:
        nuevo = c + "p"
        L8.CLASIFICACION_M8.setdefault(nuevo, "STATE")
        fe.CLASIFICACION_FEATURES.setdefault(nuevo, "STATE")


def con_estado_planp(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las 8 columnas *_F6p: para R0 usa el _prev2 de la fila (= cierre de F5); para R1-R3, el *_F6 (cierre de F6)."""
    df = df.copy()
    es_r0 = df["fase_proceso"].eq("Reducción") & df["orden_escalon_fase"].eq(0)
    for c_f6, c_p2 in F6_PARES.items():
        df[c_f6 + "p"] = np.where(es_r0, df[c_p2], df[c_f6])
    return df


_registrar_seguras_planp()


ESTADO_S1_V9 = list(mp9.ESTADO_R_V9)
ESTADO_S2_T2 = list(L8.ESTADO_R_ONLINE)
ESTADO_S3P_PLAN = [(c + "p" if c in F6_PARES else c) for c in L8.ESTADO_R_PLAN]
_CAMPOS_ENSAYO_ONLINE = ["ley_sn_escoria_pct_prev2", "ley_feo_escoria_pct_prev2", "ley_sio2_escoria_pct_prev2",
                         "ley_cao_escoria_pct_prev2", "ratio_sn_feo_prev2", "m6_sn_inv_kg_prev2", "m6_feo_inv_kg_prev2",
                         "m6_resto_frac_prev2"]
ESTADO_S4_ONLINE = [c for c in L8.ESTADO_R_ONLINE if c not in _CAMPOS_ENSAYO_ONLINE]

ESTADOS = {"S1_v9": ESTADO_S1_V9, "S2_t2": ESTADO_S2_T2, "S3p_plan": ESTADO_S3P_PLAN, "S4_online": ESTADO_S4_ONLINE}


def verificar_estados() -> pd.DataFrame:
    filas = []
    for nm, S in ESTADOS.items():
        try:
            L8.asegurar_seguras(S)
            ok, msg = True, ""
        except AssertionError as e:
            ok, msg = False, str(e)
        filas.append({"estado": nm, "n_cols": len(S), "seguro": ok, "detalle": msg})
    return pd.DataFrame(filas)


def cargar_base() -> pd.DataFrame:
    """df v8 + columnas *_F6p, preparado (filas sin primer escalon del batch excluidas) -- listo para filas_reduccion."""
    df = L8.cargar_df()
    df = con_estado_planp(df)
    base = mp8._preparar_base(df)
    return base


def filas_reduccion(base: pd.DataFrame) -> pd.DataFrame:
    return mp9.filas_reduccion(base)


# --------------------------------------------------------------------------- E[a|S]: residuos cross-fitted + R2 OOF
def residuos_estado(d: pd.DataFrame, S: list[str], n_splits: int = 5, seed: int = SEED) -> dict:
    """Para cada palanca: ahat OOF (GroupKFold por Batch), residuo, z (residuo / sd por orden), R2 OOF global y por orden."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    out = {}
    for p, col in PALANCAS.items():
        ok = d[col].notna().to_numpy() & d[S].notna().all(axis=1).to_numpy()
        ah = np.full(len(d), np.nan)
        idx = np.where(ok)[0]
        for tr, te in GroupKFold(n_splits).split(idx, groups=d["Batch"].to_numpy()[idx]):
            m = L8.hgb_nuisance(seed).fit(d[S].iloc[idx[tr]], d[col].iloc[idx[tr]])
            ah[idx[te]] = m.predict(d[S].iloc[idx[te]])
        res = d[col].to_numpy() - ah
        sd_orden = pd.Series(res).groupby(d["orden_escalon_fase"].to_numpy()).std()
        z = res / d["orden_escalon_fase"].map(sd_orden).to_numpy()
        r2_glob = r2_score(d[col].to_numpy()[ok], ah[ok]) if ok.sum() > 10 else np.nan
        r2_orden = {}
        for o in sorted(d["orden_escalon_fase"].unique()):
            m = ok & (d["orden_escalon_fase"].to_numpy() == o)
            r2_orden[int(o)] = float(r2_score(d[col].to_numpy()[m], ah[m])) if m.sum() > 10 else np.nan
        out[p] = {"ahat": ah, "res": res, "z": z, "ok": ok, "r2_global": float(r2_glob), "r2_orden": r2_orden,
                  "n_ok": int(ok.sum()), "n_total": len(d), "sd_orden": sd_orden.to_dict()}
    return out


def exposiciones_batch(d: pd.DataFrame, resid: dict) -> pd.DataFrame:
    filas = {}
    for p in PALANCAS:
        z = pd.Series(resid[p]["z"], index=d.index)
        filas[f"x_{p}"] = z.groupby(d["Batch"]).mean()
    return pd.DataFrame(filas)


def reporte_nan(base: pd.DataFrame) -> pd.DataFrame:
    d = filas_reduccion(base)
    filas = []
    for nm, S in ESTADOS.items():
        nan_por_col = d[S].isna().mean().sort_values(ascending=False)
        completas = d[S].notna().all(axis=1)
        filas.append({"estado": nm, "n_cols": len(S), "n_filas": len(d), "n_completas": int(completas.sum()),
                      "pct_completas": float(completas.mean()), "peor_columna": nan_por_col.index[0] if len(nan_por_col) else None,
                      "peor_columna_pct_nan": float(nan_por_col.iloc[0]) if len(nan_por_col) else np.nan})
        for o, g in d.groupby("orden_escalon_fase"):
            comp_o = g[S].notna().all(axis=1)
            filas.append({"estado": nm, "orden": int(o), "n_filas_orden": len(g), "n_completas_orden": int(comp_o.sum()),
                          "pct_completas_orden": float(comp_o.mean())})
    return pd.DataFrame(filas)
