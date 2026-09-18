"""v8_lib -- librería compartida de la iteración 10 (2026-09-16): prescriptor v8 = fase 0 del dictamen del comité (v7.1
conservador) + mejoras de modelo con los datos existentes (fase 6 del dictamen) que sean refutables.

Punto de partida: v7 (`modelo_predictivo_v7.py`, `candidate_win_model_v7.md`, hallazgos §20) y el dictamen del comité
(`dictamen_comite_v7.md`, auditorías A-D en `experimentos/v7/`, hallazgos §21). Esta librería re-exporta `v7_lib` (datos,
split, OOF, PLM con signos) y añade SOLO lo nuevo de la iteración 10:

1) Columnas m8_* (`agregar_columnas_v8`), todas clasificadas anti-fuga en `CLASIFICACION_M8`:
   - Target de FeO con descuento del Fe metálico alimentado (dross de Fe en R0; auditoría A §1):
         m8_ln_feo_ret_dross{40,50,60} = ln( FeO_inv[t] / (FeO_inv[t-1] + 1.2865 · fFe · feed_dross_Fe_kgh[t]) )
     con fFe = fracción de Fe metálico del dross (0.40 / 0.50 / 0.60; planta no la reporta; el dross tiene ~39 % Sn).
   - Agotamiento de Sn con piso de equilibrio C* (auditoría A §2): m8_ln_sn_dep_cstar = ln((Sn_disp − C*·M_prev)/(Sn_inv − C*·M_prev)),
     C* = 0.7 % Sn de la escoria (sólo donde ambos numerador y denominador > 20 kg).
   - Formas del carbón (dosis específica, saturante; comité §4.6 / auditoría A §3): m8_dosis = C_kg/Sn_disp (= m6_dosis_C_sn),
     m8_ln1p_dosis = ln(1 + m8_dosis), m8_dosis_sq = m8_dosis², m8_ln1p_dosis_x_av = m8_ln1p_dosis · avance_prev,
     m8_exceso_C_pos (= m6_exceso_C_pos, carbón sobrante sobre el estequiométrico de Sn, que sólo puede ir a FeO + C).
   - θ heterogéneo (auditoría A §3): m8_C_x_zT = tasa_C · z(T_horno_prev), m8_C_x_zB2 = tasa_C · z(B2_prev), m8_gn_sq = (GN − media)².
     (z con media/sd de DEV, guardadas en `CENTROS_V8`).
   - Estado ejecutable / latencia (auditoría C §1): leyes y inventarios con un escalón extra de retardo (`*_prev2` = cierre de t−2),
     y `ley_al2o3_escoria_pct_prev`, `es_F1`.
2) Conjuntos de estado candidatos (`ESTADOS_R`): k16 (v7), k15 sin espesor, k17 (+T horno, +Al2O3), online (sin ensayo de t−1).
3) Validación adicional: `oof_cronologico` (5 bloques contiguos de batches), `walk_forward` (ventana creciente, bloques de 50 batches)
   y `theta_dual` (θ LIBRE y ACOTADO con IC bootstrap por batch y fracción de réplicas en la cota: comité §4.1).

Convenciones (innegociables): OOF GroupKFold(5) por Batch en DEV = criterio; lockbox (63 últimos batches) = sólo reporte y
declarado "gastado" por el comité → para FeO reportar además OOF cronológico y walk-forward. Python `.venv/Scripts/python.exe`,
`PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=3`, sin joblib.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
for _p in (RAIZ, RAIZ / "experimentos" / "v7", RAIZ / "experimentos" / "v6", RAIZ / "experimentos" / "v5", RAIZ / "experimentos" / "balances"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v7_lib as L7  # noqa: E402
import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import modelo_predictivo_v6 as mp6  # noqa: E402
import modelo_predictivo_v7 as mp7  # noqa: E402

# re-exports
cargar_df_v6 = L7.cargar_df
split = L7.split
base_fase = L7.base_fase
preparar = L7.preparar
metricas = L7.metricas
oof_groupkfold = L7.oof_groupkfold
tercios_cronologicos = L7.tercios_cronologicos
construir_batch = L7.construir_batch
bateria_batch = L7.bateria_batch
ModeloPLMSignos = mp5.ModeloPLMSignos
hgb_nuisance = mp4._hgb_nuisance
FASES = L7.FASES
SEED = 42
KPI_PRINCIPAL = L7.KPI_PRINCIPAL
CONTROLES_BATCH = L7.CONTROLES_BATCH
KG_SN_POR_PP_KPI = L7.KG_SN_POR_PP_KPI
C_ESTEQ_POR_SN = L7.C_ESTEQ_POR_SN
FEO_POR_FE = 71.844 / 55.845          # 1.2865 kg FeO por kg Fe
CSTAR_PCT = 0.7                       # % Sn de equilibrio en la escoria (auditoría A: 0.7-1 %)
CACHE_V8 = RAIZ / "experimentos" / "cache" / "df_v8.pkl"

ESTADO_R_V7 = list(mp7.ESTADO_R_V7)
ESTADO_R_SIN_ESPESOR = [c for c in ESTADO_R_V7 if c != "espesor_ladrillo_norm_mm"]
ESTADO_R_K17 = ESTADO_R_SIN_ESPESOR + ["temperatura_horno_celsius_prev", "ley_al2o3_escoria_pct_prev"]
# Estado EJECUTABLE: sin ningún ensayo del cierre de t-1 (latencia ≥ 15 min); leyes e inventarios de t-2, variables en línea de t-1.
ESTADO_R_ONLINE = ["ley_sn_escoria_pct_prev2", "ley_feo_escoria_pct_prev2", "ley_sio2_escoria_pct_prev2", "ley_cao_escoria_pct_prev2",
                   "ratio_sn_feo_prev2", "m6_sn_inv_kg_prev2", "m6_feo_inv_kg_prev2", "m6_resto_frac_prev2",
                   "grad_temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "temperatura_horno_celsius_prev",
                   "posicion_vertical_lanza_mm_prev", "tiro_horno_pct_prev", "presion_punta_lanza_kpa_prev",
                   "cum_feed_Carbon_kg_prev", "cum_aire_nm3_prev", "cum_gn_nm3_prev", "cum_o2_nm3_prev", "relacion_C_cum_Sn_cum_prev",
                   "cum_feed_Sn_kg_prev", "orden_escalon_fase", "tiempo_fase"]
ESTADOS_R = {"k16_v7": ESTADO_R_V7, "k15_sin_espesor": ESTADO_R_SIN_ESPESOR, "k17": ESTADO_R_K17, "online": ESTADO_R_ONLINE}  # + "plan": ESTADO_R_PLAN (abajo)

PALANCAS_R_SN_V7 = list(mp6.PALANCAS_V6[("Reducción", "sn_dep")])      # Cx_sn_v6, GN, exceso O2, aire
PALANCAS_R_FEO_V7 = list(mp6.PALANCAS_V6[("Reducción", "feo_ret")])    # carbon, Cx_av_v6, GN, exceso O2, aire
SIGNOS_R_SN_V7 = dict(mp6.SIGNO_TEORICO_V6[("Reducción", "sn_dep")])
SIGNOS_R_FEO_V7 = dict(mp6.SIGNO_TEORICO_V6[("Reducción", "feo_ret")])
SIGNOS_V8 = {  # signos de teoría de las formas nuevas (+1: θ ≥ 0; −1: θ ≤ 0; 0: libre)
    "m8_dosis": +1, "m8_ln1p_dosis": +1, "m8_dosis_sq": -1, "m8_ln1p_dosis_x_av": -1, "m8_exceso_C_pos": -1,
    "m8_C_x_zT": 0, "m8_C_x_zB2": 0, "m8_gn_sq": -1,
}

CENTROS_V8: dict[str, tuple[float, float]] = {}   # {col: (media_DEV, sd_DEV)} para z-scores (se llena en agregar_columnas_v8)

CLASIFICACION_M8: dict[str, str] = {
    "m8_ln_feo_ret_dross40": "TARGET", "m8_ln_feo_ret_dross50": "TARGET", "m8_ln_feo_ret_dross60": "TARGET",
    "m8_ln_sn_dep_cstar": "TARGET", "m8_feo_eq_in_kg50": "DERIVED (A_t x CONTEXT)",
    "m8_dosis": "DERIVED (A_t x S_prev)", "m8_ln1p_dosis": "DERIVED (A_t x S_prev)", "m8_dosis_sq": "DERIVED (A_t x S_prev)",
    "m8_ln1p_dosis_x_av": "DERIVED (A_t x S_prev)", "m8_exceso_C_pos": "DERIVED (A_t x S_prev)",
    "m8_C_x_zT": "DERIVED (A_t x S_prev)", "m8_C_x_zB2": "DERIVED (A_t x S_prev)", "m8_gn_sq": "DERIVED (A_t)",
    "ley_sn_escoria_pct_prev2": "STATE", "ley_feo_escoria_pct_prev2": "STATE", "ley_sio2_escoria_pct_prev2": "STATE",
    "ley_cao_escoria_pct_prev2": "STATE", "ratio_sn_feo_prev2": "STATE", "m6_sn_inv_kg_prev2": "STATE", "m6_feo_inv_kg_prev2": "STATE",
    "m6_resto_frac_prev2": "STATE", "ley_al2o3_escoria_pct_prev": "STATE", "es_F1": "CONTEXT",
    "ley_sn_escoria_pct_F6": "STATE", "ley_feo_escoria_pct_F6": "STATE", "ley_sio2_escoria_pct_F6": "STATE", "ley_cao_escoria_pct_F6": "STATE",
    "ratio_sn_feo_F6": "STATE", "m6_sn_inv_kg_F6": "STATE", "m6_feo_inv_kg_F6": "STATE", "m6_resto_frac_F6": "STATE", "avance_F6": "DERIVED-STATE",
    "Cx_sn_F6": "DERIVED (A_t x S_prev)", "Cx_av_F6": "DERIVED (A_t x S_prev)",
}
# Estado PLAN (E10-03, C_plan): cierre de F6 + variables en linea de t-1; conserva 92 % del R2 de Sn e identifica carbon y GN.
ESTADO_R_PLAN = ["ley_sn_escoria_pct_F6", "ley_feo_escoria_pct_F6", "ley_sio2_escoria_pct_F6", "ley_cao_escoria_pct_F6", "ratio_sn_feo_F6",
                 "m6_sn_inv_kg_F6", "m6_feo_inv_kg_F6", "m6_resto_frac_F6", "orden_escalon_fase", "termocupla_media_celsius_prev",
                 "temperatura_horno_celsius_prev", "grad_temperatura_horno_celsius_prev", "posicion_vertical_lanza_mm_prev",
                 "cum_feed_Carbon_kg_prev", "cum_gn_nm3_prev", "cum_o2_nm3_prev", "cum_aire_nm3_prev", "relacion_C_cum_Sn_cum_prev",
                 "cum_feed_Sn_kg_prev", "tiempo_fase"]


def es_segura_v8(nombre: str) -> bool:
    rol = CLASIFICACION_M8.get(nombre)
    if rol is not None:
        return not ("LEAKAGE" in rol or rol in ("TARGET", "DIAG"))
    return L7.es_segura(nombre)


def asegurar_seguras(nombres: list[str]) -> None:
    inseg = [n for n in nombres if not es_segura_v8(n)]
    if inseg:
        raise AssertionError(f"features inseguras (LEAKAGE/TARGET): {inseg}")


# --------------------------------------------------------------------------- columnas v8
def agregar_columnas_v8(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    B = df["Batch"]
    dev, _ = mp.split_dev_lockbox(df)
    es_dev = B.isin(dev)
    primero = df["es_primer_escalon_batch"].astype(bool)
    segundo = primero.groupby(B).shift(1).fillna(False).astype(bool)
    n = {}
    # --- target FeO con descuento del Fe metálico alimentado (dross de Fe, tolva 5)
    feo_prev, feo = df["m6_feo_inv_kg_prev"], df["m6_feo_inv_kg"]
    red = df["fase_proceso"].eq("Reducción")
    ok = df["m6_valido"].astype(bool) & red
    for f in (0.40, 0.50, 0.60):
        eq = FEO_POR_FE * f * df["feed_dross_Fe_kgh"].fillna(0)
        n[f"m8_ln_feo_ret_dross{int(f*100)}"] = np.log(feo / (feo_prev + eq)).where(ok)
        if f == 0.50:
            n["m8_feo_eq_in_kg50"] = eq
    # --- agotamiento con piso C*
    sn_eq = CSTAR_PCT / 100 * df["m6_masa_kg_prev"]
    num, den = df["m6_sn_disp"] - sn_eq, df["m6_sn_inv_kg"] - sn_eq
    n["m8_ln_sn_dep_cstar"] = np.log(num / den).where(df["m6_valido"].astype(bool) & num.gt(20) & den.gt(20))
    # --- formas del carbón
    c_kg = df["feed_Carbon_kgh"].fillna(0)
    dosis = (c_kg / df["m6_sn_disp"]).where(df["m6_sn_disp"].gt(200))
    n["m8_dosis"] = dosis
    n["m8_ln1p_dosis"] = np.log1p(dosis)
    n["m8_dosis_sq"] = dosis ** 2
    n["m8_ln1p_dosis_x_av"] = n["m8_ln1p_dosis"] * df["m6_avance_prev"]
    n["m8_exceso_C_pos"] = (c_kg - C_ESTEQ_POR_SN * df["m6_sn_disp"]).clip(lower=0)
    # --- heterogeneidad (z con DEV)
    tasa_c = df["tasa_feed_Carbon_kg_min"]
    for col, nombre in (("temperatura_horno_celsius_prev", "m8_C_x_zT"), ("basicidad_B2_prev", "m8_C_x_zB2")):
        mu, sd = float(df.loc[es_dev, col].mean()), float(df.loc[es_dev, col].std())
        CENTROS_V8[col] = (mu, sd)
        n[nombre] = tasa_c * (df[col] - mu) / sd
    mu_gn = float(df.loc[es_dev & red, "tasa_gn_nm3_min"].mean())
    CENTROS_V8["tasa_gn_nm3_min"] = (mu_gn, float(df.loc[es_dev & red, "tasa_gn_nm3_min"].std()))
    n["m8_gn_sq"] = (df["tasa_gn_nm3_min"] - mu_gn) ** 2
    # --- retardo extra (t-2) y Al2O3 prev
    for c in ("ley_sn_escoria_pct", "ley_feo_escoria_pct", "ley_sio2_escoria_pct", "ley_cao_escoria_pct", "m6_sn_inv_kg", "m6_feo_inv_kg", "m6_resto_frac"):
        n[f"{c}_prev2"] = df[c].groupby(B).shift(2).where(~primero & ~segundo)
    n["ratio_sn_feo_prev2"] = n["ley_sn_escoria_pct_prev2"] / n["ley_feo_escoria_pct_prev2"].where(n["ley_feo_escoria_pct_prev2"] > 0)
    n["ley_al2o3_escoria_pct_prev"] = df["ley_al2o3_escoria_pct"].groupby(B).shift(1).where(~primero)
    n["es_F1"] = (df["fase_proceso"].eq("Fusión") & df["orden_escalon_fase"].eq(1)).astype(float)
    # --- estado PLAN (E10-03): leyes/inventarios congelados al cierre de F6 (= *_prev de la fila R0) para todos los escalones de
    #     Reduccion del batch; conocidos antes de R1 en cualquier caso (sin fuga). Palancas cineticas con ese inventario.
    esR0 = red & df["orden_escalon_fase"].eq(0)
    for c in ("ley_sn_escoria_pct", "ley_feo_escoria_pct", "ley_sio2_escoria_pct", "ley_cao_escoria_pct", "m6_sn_inv_kg", "m6_feo_inv_kg", "m6_resto_frac"):
        v = df[c + "_prev"].where(esR0)
        n[c + "_F6"] = v.groupby(B).transform("max").where(red)
    n["ratio_sn_feo_F6"] = n["ley_sn_escoria_pct_F6"] / n["ley_feo_escoria_pct_F6"].where(n["ley_feo_escoria_pct_F6"] > 0)
    n["avance_F6"] = 1.0 - fe.division_segura(n["m6_sn_inv_kg_F6"], df["cum_feed_Sn_kg_prev"], umbral=500.0)
    n["Cx_sn_F6"] = tasa_c * n["m6_sn_inv_kg_F6"] / 1000.0
    n["Cx_av_F6"] = tasa_c * n["avance_F6"]
    out = pd.concat([df.drop(columns=[c for c in n if c in df.columns]), pd.DataFrame(n, index=df.index)], axis=1)
    fe.CLASIFICACION_FEATURES.update({k: v for k, v in CLASIFICACION_M8.items() if k not in fe.CLASIFICACION_FEATURES})
    return out


def cargar_df() -> pd.DataFrame:
    """df v6 (3982 × 281) + columnas m8_* (caché `experimentos/cache/df_v8.pkl`)."""
    if CACHE_V8.exists():
        df = pd.read_pickle(CACHE_V8)
        if "Cx_sn_F6" in df.columns:
            if not CENTROS_V8:  # recomputar centros (baratos) para el simulador
                agregar_columnas_v8(cargar_df_v6())
            return df
    df = agregar_columnas_v8(cargar_df_v6())
    tmp = CACHE_V8.with_suffix(".tmp.pkl"); df.to_pickle(tmp); tmp.replace(CACHE_V8)
    return df


# --------------------------------------------------------------------------- validación adicional
def bloques_cronologicos(d: pd.DataFrame, n: int = 5) -> pd.Series:
    """Etiqueta 0..n-1 por bloque cronológico contiguo de batches (para OOF cronológico)."""
    return tercios_cronologicos(d, n)


def oof_cronologico(d: pd.DataFrame, fit_predict, n: int = 5) -> np.ndarray:
    """OOF con bloques CONTIGUOS de batches (cada bloque se predice con el resto): mide la generalización entre regímenes."""
    bl = bloques_cronologicos(d, n).to_numpy()
    oof = np.full(len(d), np.nan)
    for k in range(n):
        te = np.where(bl == k)[0]; tr = np.where(bl != k)[0]
        oof[te] = fit_predict(d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True))
    return oof


def walk_forward(d: pd.DataFrame, fit_predict, target: str, bloque: int = 50, min_train: int = 100) -> tuple[np.ndarray, pd.DataFrame]:
    """Ventana creciente: predice cada bloque de `bloque` batches con todo lo anterior (≥ min_train batches).
    Devuelve (pred con NaN donde no hay predicción, tabla R² por bloque)."""
    orden = d.groupby("Batch")["fecha_inicio"].min().sort_values()
    rank = d["Batch"].map({b: i for i, b in enumerate(orden.index)}).to_numpy()
    pred = np.full(len(d), np.nan); filas = []
    start = min_train
    while start < len(orden):
        te = np.where((rank >= start) & (rank < start + bloque))[0]; tr = np.where(rank < start)[0]
        if len(te) >= 10:
            pred[te] = fit_predict(d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True))
            filas.append({"bloque_inicio": start, "n_test": len(te), **metricas(d.iloc[te][target], pred[te])})
        start += bloque
    return pred, pd.DataFrame(filas)


def fit_predict_plm(S: list[str], A: list[str], target: str, signos: dict | None, seed: int = SEED):
    """Fábrica de fit_predict para `oof_groupkfold`/`oof_cronologico`/`walk_forward`."""
    def fp(d_tr: pd.DataFrame, d_te: pd.DataFrame) -> np.ndarray:
        return ModeloPLMSignos(S, A, target, signos).fit(d_tr, n_boot=0, seed=seed).predict(d_te)
    return fp


def fit_predict_hgb(X: list[str], target: str, seed: int = SEED, max_iter: int = 300):
    def fp(d_tr: pd.DataFrame, d_te: pd.DataFrame) -> np.ndarray:
        return hgb_nuisance(seed, max_iter).fit(d_tr[X], d_tr[target]).predict(d_te[X])
    return fp


def theta_dual(d: pd.DataFrame, S: list[str], A: list[str], target: str, signos: dict | None, n_splits: int = 5,
               n_boot: int = 500, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """θ LIBRE (lstsq) y ACOTADO por signo (lsq_linear) sobre los MISMOS residuos cross-fitted (GroupKFold por batch), con IC95 %
    bootstrap cluster por batch para ambos y fracción de réplicas acotadas que caen en la cota (comité §4.1 / auditoría B §1).
    Devuelve (tabla, extras) con extras = {'y_res', 'A_res', 'oof_a' (E[a|S] OOF), 'r2_balance': R² OOF de E[a|S]}."""
    from scipy.optimize import lsq_linear
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    signos = signos or {}
    grupos = d["Batch"].to_numpy()
    cols = [target] + A
    oof = pd.DataFrame(np.nan, index=d.index, columns=cols)
    for tr, te in GroupKFold(n_splits).split(d[S], d[target], grupos):
        for c in cols:
            mdl = hgb_nuisance(seed).fit(d[S].iloc[tr], d[c].iloc[tr])
            oof.iloc[te, oof.columns.get_loc(c)] = mdl.predict(d[S].iloc[te])
    y_res = d[target].to_numpy() - oof[target].to_numpy()
    A_res = np.column_stack([d[a].to_numpy() - oof[a].to_numpy() for a in A])
    lo = np.array([0.0 if signos.get(a, 0) > 0 else -np.inf for a in A]); hi = np.array([0.0 if signos.get(a, 0) < 0 else np.inf for a in A])

    def solve(Ar, yr):
        libre = np.linalg.lstsq(Ar, yr, rcond=None)[0]
        acot = lsq_linear(Ar, yr, bounds=(lo, hi), lsmr_tol="auto").x if not (np.all(np.isinf(lo)) and np.all(np.isinf(hi))) else libre
        return libre, acot
    th_l, th_a = solve(A_res, y_res)
    rng = np.random.default_rng(seed); uniq = np.unique(grupos); idx_by = {b: np.where(grupos == b)[0] for b in uniq}
    BL, BA = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
        l, a = solve(A_res[idx], y_res[idx]); BL.append(l); BA.append(a)
    BL, BA = np.array(BL), np.array(BA)
    sd = d[A].std().to_numpy()
    t = pd.DataFrame({"palanca": A, "signo_teorico": [signos.get(a, 0) for a in A], "sd_palanca": sd,
                      "theta_libre_1sd": th_l * sd, "ci_lo_libre_1sd": np.percentile(BL, 2.5, 0) * sd, "ci_hi_libre_1sd": np.percentile(BL, 97.5, 0) * sd,
                      "theta_acot_1sd": th_a * sd, "ci_lo_acot_1sd": np.percentile(BA, 2.5, 0) * sd, "ci_hi_acot_1sd": np.percentile(BA, 97.5, 0) * sd,
                      "frac_replicas_en_cota": (np.abs(BA) < 1e-9).mean(0),
                      "r2_balance_oof": [float(r2_score(d[a], oof[a])) for a in A]})
    t["identificado_libre"] = (t["ci_lo_libre_1sd"] > 0) | (t["ci_hi_libre_1sd"] < 0)
    t["interior"] = t["frac_replicas_en_cota"] < 0.05
    t["r2_oof_plm_interno"] = float(r2_score(d[target], d[target].to_numpy() - (y_res - A_res @ th_a)))
    return t, {"y_res": y_res, "A_res": A_res, "oof_a": oof[A], "theta_libre": th_l, "theta_acot": th_a}


def r2_por_conjuntos(d_dev: pd.DataFrame, d_lb: pd.DataFrame, S: list[str], A: list[str], target: str, signos: dict | None,
                     seed: int = SEED, con_wf: bool = True) -> dict:
    """R² OOF aleatorio (GroupKFold), OOF cronológico (5 bloques), walk-forward (bloques de 50, ≥100 batches) y lockbox."""
    fp = fit_predict_plm(S, A, target, signos, seed)
    oof = oof_groupkfold(d_dev, fp); ooc = oof_cronologico(d_dev, fp)
    out = {"n_dev": len(d_dev), "n_lockbox": len(d_lb), "r2_oof": metricas(d_dev[target], oof)["r2"], "mae_oof": metricas(d_dev[target], oof)["mae"],
           "r2_oof_cronologico": metricas(d_dev[target], ooc)["r2"]}
    if con_wf:
        pw, _ = walk_forward(d_dev, fp, target); m = np.isfinite(pw)
        out["r2_walk_forward"] = metricas(d_dev[target][m], pw[m])["r2"]; out["n_walk_forward"] = int(m.sum())
    if len(d_lb) > 10:
        p_lb = ModeloPLMSignos(S, A, target, signos).fit(d_dev, n_boot=0, seed=seed).predict(d_lb)
        out["r2_lockbox"] = metricas(d_lb[target], p_lb)["r2"]; out["mae_lockbox"] = metricas(d_lb[target], p_lb)["mae"]
    return out


if __name__ == "__main__":
    df = cargar_df()
    print(df.shape, [c for c in df.columns if c.startswith("m8_") or c.endswith("_prev2")])
    r = base_fase(df, "Reducción")
    print(r[["m6_ln_feo_ret", "m8_ln_feo_ret_dross50", "m6_ln_sn_dep", "m8_ln_sn_dep_cstar", "m8_dosis", "m8_ln1p_dosis"]].describe().round(3).T.to_string())
    for k, S in ESTADOS_R.items():
        asegurar_seguras(S); print(k, len(S), "NaN filas R:", int(r[S].isna().any(axis=1).sum()))
