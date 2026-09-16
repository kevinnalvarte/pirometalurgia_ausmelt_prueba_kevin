"""v7_lib -- librería compartida de la iteración 9 (2026-09-16): búsqueda del mejor enfoque predictivo/prescriptivo.

Punto de partida: prescriptor v6 (`modelo_predictivo_v6.py`, `experimentos/v6/v6_lib.py`, `candidate_win_model_v6.md`,
hallazgos §19). Esta librería NO redefine nada: sólo carga rápido la caché `experimentos/cache/df_v6.pkl` (df completo
v3+v4+b6+v5+m6, 3982 filas × 281 columnas) y re-exporta lo que los experimentos E9-* necesitan con nombres cortos.

Convenciones innegociables (ver `metodologia` en memoria y hallazgos §14-19):
- OOF GroupKFold(5) por Batch sobre DEV (299 batches) = criterio de selección. LOCKBOX = últimos 17.5 % de batches
  cronológicos (63, agosto 2026, fin de campaña de refractario) = sólo reporte, NUNCA para elegir.
- Anti-fuga: toda feature de estado/palanca debe pasar `es_segura` (clasificación CONTEXT/STATE/ACTION/DERIVED); los
  targets `m6_*`/`v5_*`/`d_*` y todo lo contemporáneo (leyes de t, m6_masa_kg sin _prev) son LEAKAGE como feature.
- Targets v6 por escalón (log, adimensionales, telescópicos por batch):
      m6_ln_sn_dep  = ln( (Sn_inv[t-1] + Sn alimentado[t]) / Sn_inv[t] )   agotamiento del Sn de la escoria
      m6_ln_feo_ret = ln( FeO_inv[t] / FeO_inv[t-1] )                      retención de FeO (sólo Reducción)
  Σ_R m6_ln_sn_dep = ln(Sn_ini_R/Sn_fin_R); Σ_R m6_ln_feo_ret = ln(FeO_fin/FeO_ini).
- Objetivo v6 (anclado en el KPI refinado `recuperacion_refinada_pct`, E8-03): J_R = K·(ln_sn_dep + w·ln_feo_ret) − costos −
  ventana T − IC − soporte, w = 6.09, K = 2.893 pp/unidad (× 512 kg Sn/pp); Fusión β_F = −0.58 pp/unidad de Σ_F ln_sn_dep.
- Python: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=3`; sin joblib/loky (se cuelga en Windows);
  LightGBM no disponible (usar HGB/XGBoost).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "balances", RAIZ / "experimentos" / "v5", RAIZ / "experimentos" / "v6"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import modelo_predictivo_v6 as mp6  # noqa: E402
import v5_lib  # noqa: E402
import v6_lib  # noqa: E402
import masa_v6  # noqa: E402

warnings.filterwarnings("ignore")

CACHE_V6 = RAIZ / "experimentos" / "cache" / "df_v6.pkl"
FASES = ["Fusión", "Reducción"]
KPI_PRINCIPAL = v5_lib.KPI_PRINCIPAL          # recuperacion_refinada_pct
KPIS = v5_lib.KPIS
CONTROLES_BATCH = v5_lib.CONTROLES_BATCH      # controles de las regresiones de batch (E7-01/E8-03)
SEED = 42
KG_SN_POR_PP_KPI = mp5.KG_SN_POR_PP_KPI       # ~512 kg Sn por pp de KPI
C_ESTEQ_POR_SN = mp5.C_ESTEQ_POR_SN           # 0.2024 kg C / kg Sn (SnO2 + 2C)
C_ESTEQ_POR_FEO = 12.011 / 71.844             # 0.1672 kg C / kg FeO (FeO + C -> Fe + CO)

# --- Estado / palancas / targets / signos v6 (referencia de la iteración 8) ---
TARGETS = mp6.TARGETS_V6                      # {fase: {clave: columna}}
ESTADO = mp6.ESTADO_V6                        # {(fase, clave): [cols]}  (Reducción: 38 = FULL v3 + m6_resto_frac_prev)
PALANCAS = mp6.PALANCAS_V6                    # {(fase, clave): [cols]}
SIGNOS = mp6.SIGNO_TEORICO_V6                 # {(fase, clave): {palanca: +1/-1/0}}
CONFIG = dict(mp6.CONFIG_V6)                  # w_sdi 6.09, pp_kpi_por_unidad_sdi 2.893, pp_kpi_por_unidad_sn_dep_F -0.58
ESTADO_R_FULL = v6_lib.ESTADO_R_FULL_V6
ESTADO_R_CURADO = v6_lib.ESTADO_R_CURADO_V6
ESTADO_F_CURADO = v6_lib.ESTADO_F_CURADO_V6
PALANCAS_PRIMITIVAS = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"]
PLAN_FUSION = ["feed_Sn_kgf", "tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min", "duracion_plan_min"]

ModeloPLMSignos = mp5.ModeloPLMSignos
hgb_nuisance = mp4._hgb_nuisance

# --------------------------------------------------------------------------- anti-fuga
es_segura = v6_lib.es_segura_v6_o_v5


def filtrar_seguras(nombres: list[str]) -> list[str]:
    return [n for n in nombres if es_segura(n)]


def asegurar_seguras(nombres: list[str]) -> None:
    inseg = [n for n in nombres if not es_segura(n)]
    if inseg:
        raise AssertionError(f"features inseguras (LEAKAGE/TARGET): {inseg}")


# --------------------------------------------------------------------------- datos
def cargar_df() -> pd.DataFrame:
    """df completo (3982 × 281). Regenerar la caché con `experimentos/v6/_cache_df_v6.py` si no existe."""
    if CACHE_V6.exists():
        return pd.read_pickle(CACHE_V6)
    df = v6_lib.cargar_df(con_termicas=True)
    df.to_pickle(CACHE_V6)
    return df


def base_fase(df: pd.DataFrame, fase: str) -> pd.DataFrame:
    """Escalones modelables de la fase (excluye el primer escalón del batch, sin estado _prev)."""
    return v5_lib.base_fase(df, fase)


def split(df: pd.DataFrame) -> tuple[set, set]:
    """(batches_dev, batches_lockbox) — lockbox = últimos 17.5 % cronológicos."""
    return mp.split_dev_lockbox(df)


def construir_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: KPIs, controles, agregados v5 y v6 (R_sum_m6_ln_sn_dep, R_sum_m6_ln_feo_ret, F_sum_m6_ln_sn_dep, ...),
    es_dev/es_lockbox, idx_cronologico y KPI del batch siguiente (`*_next`)."""
    return v6_lib.construir_batch_v6(df)


bateria_batch = v5_lib.bateria_batch          # Spearman IC + OLS HC3 con controles (DEV/lockbox/total) de una columna vs KPI
evaluar_plm_signos = v6_lib.evaluar_plm_signos  # (met, tabla_theta, pred) del PLM con signos, OOF DEV + lockbox


# --------------------------------------------------------------------------- métricas
def metricas(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    if m.sum() < 10:
        return {"r2": np.nan, "mae": np.nan, "n": int(m.sum())}
    return {"r2": float(r2_score(y[m], pred[m])), "mae": float(mean_absolute_error(y[m], pred[m])), "n": int(m.sum())}


def oof_groupkfold(d: pd.DataFrame, fit_predict, n_splits: int = 5) -> np.ndarray:
    """OOF genérico: `fit_predict(d_train, d_test) -> pred_test`, folds GroupKFold por Batch."""
    from sklearn.model_selection import GroupKFold
    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits).split(d, groups=d["Batch"]):
        oof[te] = fit_predict(d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True))
    return oof


def preparar(df_fase: pd.DataFrame, cols: list[str], target: str, extra: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(d_dev, d_lockbox) con filas completas en cols + target (+ extra), índices reseteados."""
    dev, lockbox = split(df_fase)
    keep = list(dict.fromkeys(cols + [target, "Batch", "fecha_inicio", "orden_escalon_fase", "duracion_plan_min"] + (extra or [])))
    d = df_fase[keep].dropna(subset=cols + [target])
    return (d[d["Batch"].isin(dev)].reset_index(drop=True), d[d["Batch"].isin(lockbox)].reset_index(drop=True))


def tercios_cronologicos(d: pd.DataFrame, n: int = 3) -> pd.Series:
    """Etiqueta 0..n-1 por tercio cronológico de batches (para estabilidad temporal)."""
    orden = d.groupby("Batch")["fecha_inicio"].min().sort_values()
    corte = pd.qcut(np.arange(len(orden)), n, labels=False)
    return d["Batch"].map(dict(zip(orden.index, corte)))
