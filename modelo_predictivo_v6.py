"""Modelo prescriptivo v6 de Lingo Smelter (iteracion 8, 2026-09-16): misma arquitectura que v5 (PLM por fase con
signos de teoria, objetivo anclado en el KPI refinado, optimizador con soporte relativo, politica cross-fitted y
evidencia off-policy) sobre una NUEVA INFERENCIA DE MASA DE ESCORIA (`experimentos/v6/masa_v6.py`).

Que cambia respecto de v5 y por que (auditoria del trazador CaO, hallazgos §19):
1. Masa de escoria por escalon (inferible en linea):
     Fusion    M_t = (pG·Σcal_h + gG·Σcarga_h) / G_t,  G = %CaO+%SiO2+%Al2O3+%MgO   (trazador de matriz, 4 ensayos)
     Reduccion M_t·(1 − K·s_t − f_t) = M_{t−1}·(1 − K·s_{t−1} − f_{t−1}) + resto_in_t  (cierre fisico: solo salen SnO2 y FeO)
   con pG, gG coeficientes efectivos (leyes de cal, ganga y ceniza desconocidas) calibrados en DEV contra la masa final del
   balance global de batch (metal + dross + polvo). El trazador v3 (cal/%CaO) suponia cal pura y unica fuente de CaO; su
   ruido de ensayo dominaba la senal de masa en Reduccion y el PLM de `v5_ln_feo_ret` aprendia la reversion de ese ruido.
2. Inventarios, estado, targets log y palancas cineticas recalculados con esa masa (prefijo m6_, `Cx_sn_v6`, `Cx_av_v6`).
3. Anclajes del objetivo (w, K, beta_F) re-estimados con los targets v6 (E8-03) -> CONFIG_V6.
Todo lo demas (ModeloPLMSignos, soporte relativo, ventana termica, costos, cross-fitting, evidencia) se reutiliza de v5.

Uso tipico:
    import modelo_predictivo_v6 as mp6
    df = mp6.construir_dataset_modelo_v6("Datos Lingo smelter fase II.xlsx")   # o mp6.agregar_columnas_v6(df_v3)
    sistema = mp6.entrenar_sistema_v6(df)
    print(mp6.tabla_validacion_v6(df)); print(mp6.tabla_efectos_control_v6(sistema))
    rep = mp6.reporte_recomendaciones_batch_v6(sistema, df, "AP0350")
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import feature_engineering as fe
import modelo_prescriptivo as mp
import modelo_predictivo_v3 as mp3
import modelo_predictivo_v4 as mp4
import modelo_predictivo_v5 as mp5

_RAIZ = Path(__file__).resolve().parent
if str(_RAIZ / "experimentos" / "v6") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "experimentos" / "v6"))
import masa_v6  # noqa: E402

FASES = mp.FASES
SEED = mp5.SEED
KG_SN_POR_PP_KPI = mp5.KG_SN_POR_PP_KPI
C_ESTEQ_POR_SN = mp5.C_ESTEQ_POR_SN
ModeloPLMSignos = mp5.ModeloPLMSignos
SistemaPrescriptivoV6 = mp5.SistemaPrescriptivoV5   # misma estructura de datos

# Parametros del estimador de masa (E8-01); se sobreescriben desde experimentos si cambian
PARAMS_MASA_V6: dict = dict(masa_v6.PARAMS_DEFAULT)

# =============================================================================
# 0) Mapa de columnas v5 -> v6 y roles
# =============================================================================
MAPA_V6: dict[str, str] = {
    "masa_escoria_est_kg_prev": "m6_masa_kg_prev", "sn_inventario_escoria_est_kg_prev": "m6_sn_inv_kg_prev",
    "feo_inventario_escoria_est_kg_prev": "m6_feo_inv_kg_prev", "avance_reduccion_sn_prev": "m6_avance_prev",
    "Cx_sn_v4": "Cx_sn_v6", "Cx_av_v4": "Cx_av_v6",
    "v5_ln_sn_dep": "m6_ln_sn_dep", "v5_ln_feo_ret": "m6_ln_feo_ret",
    "v5_dosis_C_sn": "m6_dosis_C_sn", "v5_exceso_C_pos": "m6_exceso_C_pos", "v5_C_x_avance": "m6_C_x_avance",
    "v5_gn_esp_nm3_t": "m6_gn_esp_nm3_t", "v5_sn_disp": "m6_sn_disp",
}
ESTADO_EXTRA_V6 = ["m6_resto_frac_prev"]   # fraccion no reducible al cierre de t-1 (estado fisico nuevo)


def _m(cols: list[str]) -> list[str]:
    return [MAPA_V6.get(c, c) for c in cols]


TARGETS_V6: dict[str, dict[str, str]] = {f: {k: MAPA_V6.get(v, v) for k, v in d.items()} for f, d in mp5.TARGETS_V5.items()}
ESTADO_V6: dict[tuple[str, str], list[str]] = {k: _m(v) for k, v in mp5.ESTADO_V5.items()}
for _k in list(ESTADO_V6):
    if _k[1] in ("sn_dep", "feo_ret"):
        ESTADO_V6[_k] = list(dict.fromkeys(ESTADO_V6[_k] + ESTADO_EXTRA_V6))
PALANCAS_V6: dict[tuple[str, str], list[str]] = {k: _m(v) for k, v in mp5.PALANCAS_V5.items()}
SIGNO_TEORICO_V6: dict[tuple[str, str], dict[str, int]] = {k: {MAPA_V6.get(a, a): s for a, s in v.items()} for k, v in mp5.SIGNO_TEORICO_V5.items()}
ACCIONES_OPTIMIZABLES_V6 = {f: list(v) for f, v in mp5.ACCIONES_OPTIMIZABLES_V5.items()}
ACCIONES_PRIMITIVAS_V6 = {f: list(v) for f, v in mp5.ACCIONES_PRIMITIVAS_V5.items()}
FEATURES_SOPORTE_V6 = {f: _m(v) for f, v in mp5.FEATURES_SOPORTE_V5.items()}
PERCENTIL_SOPORTE_MINIMO = mp5.PERCENTIL_SOPORTE_MINIMO
CENTROS_CUADRATICOS: dict[str, float] = {}

# Anclajes del objetivo re-estimados con los targets v6 definitivos (E8-03 + re-anclaje con PARAMS cierre/cierre, delta 0.01):
# OLS HC3 del KPI refinado por batch (DEV n=285, mismos controles y tendencia que E7-01): coef ln_sn_dep = +2.893 pp/unidad
# (p 5e-6), coef ln_feo_ret = +17.63 (p 1e-7) -> w = 6.09 [4.05, 9.99] (v5: 6.04 [3.63, 14.33]; IC ~50 % mas angosto por el
# menor ruido del target de FeO). Fusion: conservacion del Sn -0.580 pp/unidad (DEV p 0.17; TOTAL -0.909 p 0.014): candidato
# debil de signo teorico, misma decision que v5.
CONFIG_V6: dict = dict(mp5.CONFIG_V5, w_sdi=6.09, pp_kpi_por_unidad_sdi=2.893, pp_kpi_por_unidad_sn_dep_F=-0.580)
PESOS_V6_DEFAULT = dict(mp5.PESOS_V5_DEFAULT)

CLASIFICACION_V6 = dict(masa_v6.CLASIFICACION_M6)


def es_segura_v6(nombre: str) -> bool:
    return masa_v6.es_segura_v6(nombre) or mp5.es_segura_v5(nombre)


def _verificar_seguridad() -> None:
    for k, feats in list(ESTADO_V6.items()) + list(PALANCAS_V6.items()):
        inseg = [f for f in feats if not es_segura_v6(f)]
        if inseg:
            raise AssertionError(f"v6 {k}: features inseguras {inseg}")


_verificar_seguridad()


def kg_sn_por_unidad_sdi() -> float:
    return CONFIG_V6["pp_kpi_por_unidad_sdi"] * KG_SN_POR_PP_KPI


# =============================================================================
# 1) Columnas v6 (dataset y por fila para el simulador)
# =============================================================================

def agregar_columnas_v6(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """v5 (v4 + targets/palancas v5) + masa v6 e inventarios/targets/palancas m6_*."""
    if "v5_ln_sn_dep" not in df.columns:
        df = mp5.agregar_columnas_v5(df)
    if "m6_masa_kg" not in df.columns:
        df = masa_v6.agregar_masa_v6(df, dict(PARAMS_MASA_V6, **(params or {})))
    return agregar_cuadraticos_v6(df)


def agregar_cuadraticos_v6(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a, c in CENTROS_CUADRATICOS.items():
        if a in df.columns:
            df[f"{a}__sq"] = (df[a] - c) ** 2
    return df


def construir_dataset_modelo_v6(ruta_excel: str, hoja: str = "Datos") -> pd.DataFrame:
    return agregar_columnas_v6(mp5.construir_dataset_modelo_v5(ruta_excel, hoja))


def _agregar_columnas_v6_fila(fila: dict) -> dict:
    """Version por fila (simulador): palancas cineticas y derivadas v6 desde el estado m6_* y la accion."""
    c_kg = float(fila.get("feed_Carbon_kgh", 0.0) or 0.0)
    tasa_c = float(fila.get("tasa_feed_Carbon_kg_min", np.nan))
    sn_prev = float(fila.get("m6_sn_inv_kg_prev", np.nan))
    avance = float(fila.get("m6_avance_prev", np.nan))
    feed_sn = float(fila.get("feed_Sn_kgf", 0.0) or 0.0)
    sn_disp = sn_prev + feed_sn
    fila["m6_sn_disp"] = sn_disp
    fila["Cx_sn_v6"] = tasa_c * sn_prev / 1000.0
    fila["Cx_av_v6"] = tasa_c * avance
    dosis = c_kg / sn_disp if sn_disp > 200.0 else np.nan
    fila["m6_dosis_C_sn"] = dosis
    fila["m6_exceso_C_pos"] = max(c_kg - C_ESTEQ_POR_SN * sn_disp, 0.0) if np.isfinite(sn_disp) else np.nan
    fila["m6_C_x_avance"] = c_kg * avance
    masa_prev = float(fila.get("m6_masa_kg_prev", np.nan))
    gn_nm3 = float(fila.get("volumen_gas_natural_inyectado_lanza_escalon_nm3", 0.0) or 0.0)
    fila["m6_gn_esp_nm3_t"] = gn_nm3 / (masa_prev / 1000) if masa_prev > 1000 else np.nan
    for a, c in CENTROS_CUADRATICOS.items():
        v = fila.get(a, np.nan)
        fila[f"{a}__sq"] = (v - c) ** 2 if pd.notna(v) else np.nan
    return fila


def fila_a_state_action_context(fase: str, row: pd.Series) -> tuple[dict, dict, dict]:
    state, action, context = mp5.fila_a_state_action_context(fase, row)
    for extra in ("m6_masa_kg_prev", "m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev", "m6_avance_prev", "m6_resto_frac_prev", "m6_G_pct_prev"):
        if extra in row.index:
            state[extra] = row.get(extra, np.nan)
    return state, action, context


# =============================================================================
# 2) Sistema v6
# =============================================================================

def _nuevo_plm(fase: str, clave: str, S: list[str], A: list[str], target: str):
    return ModeloPLMSignos(S, A, target, SIGNO_TEORICO_V6.get((fase, clave)))


def _preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    if "m6_ln_sn_dep" not in df.columns or "m6_dosis_C_sn" not in df.columns:
        df = agregar_columnas_v6(df)
    df_base = mp.dataset_base_modelo(df)
    return agregar_cuadraticos_v6(df_base)


def entrenar_sistema_v6(df: pd.DataFrame, pct_lockbox: float = mp.N_LOCKBOX_PCT, n_boot: int = 200, seed: int = SEED,
                        config: dict | None = None) -> SistemaPrescriptivoV6:
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df, pct_lockbox)
    sistema = SistemaPrescriptivoV6(batches_dev=batches_dev, batches_lockbox=batches_lockbox, config=dict(CONFIG_V6, **(config or {})))
    for fase in FASES:
        sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        for clave, target in TARGETS_V6[fase].items():
            S, A = ESTADO_V6[(fase, clave)], PALANCAS_V6[(fase, clave)]
            d = sub.dropna(subset=list(dict.fromkeys(S + A + [target]))).reset_index(drop=True)
            sistema.modelos[(fase, clave)] = _nuevo_plm(fase, clave, S, A, target).fit(d, n_boot=n_boot, seed=seed)
        feats_sop = FEATURES_SOPORTE_V6[fase]
        sub_sop = sub.dropna(subset=feats_sop)
        sistema.soporte[fase] = mp._construir_soporte_historico(sub_sop, feats_sop)
        scores = mp3._support_scores_lote(sistema.soporte[fase], sub_sop)
        sistema.soporte_minimo[fase] = float(np.quantile(scores, PERCENTIL_SOPORTE_MINIMO))
        acciones = sub[ACCIONES_PRIMITIVAS_V6[fase]].dropna()
        sistema.limites_accion[fase] = {a: (float(acciones[a].quantile(0.01)), float(acciones[a].quantile(0.99)))
                                        for a in ACCIONES_PRIMITIVAS_V6[fase]}
        t = sub["temperatura_horno_celsius"].dropna()
        sistema.ventana_temperatura[fase] = (float(t.quantile(0.05)), float(t.quantile(0.95)))
    return sistema


def tabla_efectos_control_v6(sistema: SistemaPrescriptivoV6) -> pd.DataFrame:
    filas = []
    for (fase, clave), m in sistema.modelos.items():
        t = m.tabla_theta(SIGNO_TEORICO_V6.get((fase, clave))).reset_index()
        t.insert(0, "target", TARGETS_V6[fase][clave]); t.insert(0, "clave", clave); t.insert(0, "fase", fase)
        filas.append(t)
    return pd.concat(filas, ignore_index=True)


# =============================================================================
# 3) Validacion OOF / lockbox
# =============================================================================

def _oof_plm(d_dev: pd.DataFrame, S: list[str], A: list[str], target: str, n_splits: int, seed: int, signos: dict | None = None) -> np.ndarray:
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev[S], d_dev[target], d_dev["Batch"]):
        m = ModeloPLMSignos(S, A, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
        oof[te] = m.predict(d_dev.iloc[te])
    return oof


def tabla_validacion_v6(df: pd.DataFrame, n_splits: int = mp.N_SPLITS_OOF, seed: int = SEED, con_hgb_ref: bool = True,
                        con_v5_ref: bool = True) -> pd.DataFrame:
    """R2/MAE OOF (DEV) y lockbox por (fase, target) del PLM v6; referencias HGB y PLM v5 (mismas filas cuando es posible)."""
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    filas = []
    for fase in FASES:
        sub = df_base.loc[df_base["fase_proceso"] == fase]
        for clave, target in TARGETS_V6[fase].items():
            S, A = ESTADO_V6[(fase, clave)], PALANCAS_V6[(fase, clave)]
            need = list(dict.fromkeys(S + A + [target, "Batch"]))
            d_dev = sub.loc[sub["Batch"].isin(batches_dev), need].dropna().reset_index(drop=True)
            d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), need].dropna().reset_index(drop=True)
            oof = _oof_plm(d_dev, S, A, target, n_splits, seed, SIGNO_TEORICO_V6.get((fase, clave)))
            m = _nuevo_plm(fase, clave, S, A, target).fit(d_dev, n_boot=0, seed=seed)
            p_lb = m.predict(d_lb)
            base = {"fase": fase, "target": clave, "n_estado": len(S), "n_palancas": len(A), "n_dev": len(d_dev), "n_lockbox": len(d_lb)}
            filas.append({**base, "columna": target, "forma": "PLM_v6",
                          "r2_oof": r2_score(d_dev[target], oof), "mae_oof": mean_absolute_error(d_dev[target], oof),
                          "r2_lockbox": r2_score(d_lb[target], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target], p_lb),
                          "sd_target_dev": float(d_dev[target].std()), "sd_target_lockbox": float(d_lb[target].std())})
            if con_hgb_ref:
                X = S + A
                oofh = np.full(len(d_dev), np.nan)
                for tr, te in GroupKFold(n_splits).split(d_dev[X], d_dev[target], d_dev["Batch"]):
                    h = mp4._hgb_nuisance(seed, 300).fit(d_dev[X].iloc[tr], d_dev[target].iloc[tr])
                    oofh[te] = h.predict(d_dev[X].iloc[te])
                h = mp4._hgb_nuisance(seed, 300).fit(d_dev[X], d_dev[target])
                filas.append({**base, "columna": target, "forma": "HGB_estado+palancas_ref",
                              "r2_oof": r2_score(d_dev[target], oofh), "mae_oof": mean_absolute_error(d_dev[target], oofh),
                              "r2_lockbox": r2_score(d_lb[target], h.predict(d_lb[X])), "mae_lockbox": mean_absolute_error(d_lb[target], h.predict(d_lb[X])),
                              "sd_target_dev": float(d_dev[target].std()), "sd_target_lockbox": float(d_lb[target].std())})
            if con_v5_ref and (fase, clave) in mp5.ESTADO_V5:
                S5, A5, t5 = mp5.ESTADO_V5[(fase, clave)], mp5.PALANCAS_V5[(fase, clave)], mp5.TARGETS_V5[fase][clave]
                need5 = list(dict.fromkeys(S5 + A5 + [t5, "Batch"]))
                d5 = sub.loc[sub["Batch"].isin(batches_dev), need5].dropna().reset_index(drop=True)
                l5 = sub.loc[sub["Batch"].isin(batches_lockbox), need5].dropna().reset_index(drop=True)
                oof5 = _oof_plm(d5, S5, A5, t5, n_splits, seed, mp5.SIGNO_TEORICO_V5.get((fase, clave)))
                m5 = ModeloPLMSignos(S5, A5, t5, mp5.SIGNO_TEORICO_V5.get((fase, clave))).fit(d5, n_boot=0, seed=seed)
                filas.append({**base, "columna": t5, "forma": "PLM_v5_ref", "n_estado": len(S5), "n_palancas": len(A5), "n_dev": len(d5), "n_lockbox": len(l5),
                              "r2_oof": r2_score(d5[t5], oof5), "mae_oof": mean_absolute_error(d5[t5], oof5),
                              "r2_lockbox": r2_score(l5[t5], m5.predict(l5)), "mae_lockbox": mean_absolute_error(l5[t5], m5.predict(l5)),
                              "sd_target_dev": float(d5[t5].std()), "sd_target_lockbox": float(l5[t5].std())})
    return pd.DataFrame(filas)


def predicciones_oof_y_lockbox_v6(df: pd.DataFrame, fase: str, clave: str, n_splits: int = mp.N_SPLITS_OOF, seed: int = SEED) -> pd.DataFrame:
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    target = TARGETS_V6[fase][clave]
    S, A = ESTADO_V6[(fase, clave)], PALANCAS_V6[(fase, clave)]
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    keep = ["Batch", "fecha_inicio", "orden_escalon_fase", "m6_avance_prev"]
    need = list(dict.fromkeys(S + A + [target] + keep))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), need].dropna(subset=S + A + [target]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), need].dropna(subset=S + A + [target]).reset_index(drop=True)
    oof = _oof_plm(d_dev, S, A, target, n_splits, seed, SIGNO_TEORICO_V6.get((fase, clave)))
    m = _nuevo_plm(fase, clave, S, A, target).fit(d_dev, n_boot=0, seed=seed)
    return pd.concat([d_dev[keep].assign(real=d_dev[target], pred=oof, conjunto="OOF_dev"),
                      d_lb[keep].assign(real=d_lb[target], pred=m.predict(d_lb), conjunto="lockbox")], ignore_index=True)


# =============================================================================
# 4) Simulador y objetivo
# =============================================================================

def _todas_las_columnas(sistema: SistemaPrescriptivoV6, fase: str) -> list[str]:
    cols = set(sistema.soporte[fase]["features"])
    for (f, k), m in sistema.modelos.items():
        if f == fase:
            cols |= set(m.estado) | set(m.palancas)
    return sorted(cols)


DERIVADAS_REPORTE = ("exceso_o2_combustion_pct", "relacion_C_Sn_carga", "m6_dosis_C_sn", "m6_exceso_C_pos")


def simular_acciones_lote_v6(sistema: SistemaPrescriptivoV6, fase: str, state: dict, acciones: list[dict],
                             context: dict | None = None, alpha: float = 0.20) -> pd.DataFrame:
    filas = [_agregar_columnas_v6_fila(mp5._agregar_palancas_v5_fila(mp4._agregar_columnas_v4_fila(
        mp3.recalcular_derivadas_v3(fase, state, a, context)))) for a in acciones]
    cols = _todas_las_columnas(sistema, fase)
    X = pd.DataFrame([[f.get(c, np.nan) for c in cols] for f in filas], columns=cols)
    out = pd.DataFrame(index=range(len(acciones)))
    for clave in TARGETS_V6[fase]:
        if clave == "dT":
            continue
        p, lo, hi = sistema.modelos[(fase, clave)].predict_intervalo(X, alpha)
        out[f"pred_{clave}"], out[f"{clave}_lo"], out[f"{clave}_hi"] = p, lo, hi
    dT, dlo, dhi = sistema.modelos[(fase, "dT")].predict_intervalo(X, alpha)
    t_prev = float(state.get("temperatura_horno_celsius_prev", np.nan))
    out["pred_dT"], out["dT_lo"], out["dT_hi"] = dT, dlo, dhi
    out["pred_temperatura"] = t_prev + dT if pd.notna(t_prev) else np.nan
    out["support_score"] = mp3._support_scores_lote(sistema.soporte[fase], X)
    for k in DERIVADAS_REPORTE:
        out[f"derivada__{k}"] = [f.get(k, np.nan) for f in filas]
    return out


def objetivo_lote_v6(sistema: SistemaPrescriptivoV6, fase: str, state: dict, acciones: list[dict], context: dict | None = None,
                     pesos: dict | None = None, soporte_ref: float | None = None) -> pd.DataFrame:
    """J [kg de Sn a metal equivalente] por accion candidata (misma forma que v5, anclajes CONFIG_V6)."""
    p = dict(PESOS_V6_DEFAULT, **(pesos or {}))
    cfg = sistema.config
    K = cfg["pp_kpi_por_unidad_sdi"] * KG_SN_POR_PP_KPI
    sim = simular_acciones_lote_v6(sistema, fase, state, acciones, context)
    A = pd.DataFrame(acciones)
    dt = A["duracion_plan_min"].to_numpy(dtype=float)
    costo = (p["costo_carbon_kgSn_por_kg"] * A["tasa_feed_Carbon_kg_min"].to_numpy(dtype=float) * dt
             + p["costo_gn_kgSn_por_nm3"] * A["tasa_gn_nm3_min"].to_numpy(dtype=float) * dt
             + p["costo_o2_kgSn_por_nm3"] * A["tasa_o2_nm3_min"].to_numpy(dtype=float) * dt)
    t_min, t_max = sistema.ventana_temperatura[fase]
    t_pred = sim["pred_temperatura"].to_numpy(dtype=float)
    fuera = np.where(np.isnan(t_pred), 0.0, np.maximum(0.0, t_min - t_pred) + np.maximum(0.0, t_pred - t_max))
    sim["J_costo"] = -costo
    sim["J_temperatura"] = -p["w_temp"] * fuera
    if soporte_ref is None:
        sim["J_soporte"] = -p["w_soporte"] * (1 - sim["support_score"])
    else:
        sim["J_soporte"] = -p["w_soporte"] * np.maximum(0.0, soporte_ref - sim["support_score"])
    if fase == "Reducción":
        w = cfg["w_sdi"]
        sim["J_sn"] = K * sim["pred_sn_dep"]
        sim["J_feo"] = K * w * sim["pred_feo_ret"]
        ancho = (sim["sn_dep_hi"] - sim["sn_dep_lo"]) + w * (sim["feo_ret_hi"] - sim["feo_ret_lo"])
        sim["J_incertidumbre"] = -p["w_incertidumbre"] * K * ancho
    else:
        bf = cfg["beta_fusion"]; nF = cfg["n_escalones_fusion"]
        z_c = (sim["derivada__relacion_C_Sn_carga"].to_numpy(dtype=float) - bf["C_por_Sn"]["media_batch"]) / bf["C_por_Sn"]["sd_batch"]
        z_t = (t_pred - bf["T_media"]["media_batch"]) / bf["T_media"]["sd_batch"]
        sim["J_sn"] = (KG_SN_POR_PP_KPI / nF) * (bf["C_por_Sn"]["pp_por_sd"] * np.nan_to_num(z_c) - bf["T_media"]["pp_por_sd"] * np.nan_to_num(z_t))
        d_irf = sim["pred_d_lnirf"] if "pred_d_lnirf" in sim.columns else 0.0
        d_irf_ancho = (sim["d_lnirf_hi"] - sim["d_lnirf_lo"]) if "pred_d_lnirf" in sim.columns else 0.0
        sim["J_feo"] = KG_SN_POR_PP_KPI * (cfg["pp_kpi_por_unidad_sn_dep_F"] * sim["pred_sn_dep"] + cfg["pp_kpi_por_unidad_d_lnirf_F"] * d_irf)
        ancho = (abs(cfg["pp_kpi_por_unidad_sn_dep_F"]) * (sim["sn_dep_hi"] - sim["sn_dep_lo"])
                 + abs(cfg["pp_kpi_por_unidad_d_lnirf_F"]) * d_irf_ancho
                 + bf["T_media"]["pp_por_sd"] / nF * (sim["dT_hi"] - sim["dT_lo"]) / bf["T_media"]["sd_batch"])
        sim["J_incertidumbre"] = -p["w_incertidumbre"] * KG_SN_POR_PP_KPI * ancho
    sim["J"] = sim[["J_sn", "J_feo", "J_costo", "J_temperatura", "J_incertidumbre", "J_soporte"]].sum(axis=1)
    return sim


def optimizar_accion_v6(sistema: SistemaPrescriptivoV6, fase: str, state: dict, accion_base: dict, context: dict | None = None,
                        acciones_a_optimizar: list[str] | None = None, n_candidatos: int = 300, n_refinamiento: int = 60,
                        semilla: int = 0, pesos: dict | None = None) -> tuple[dict, float, pd.Series]:
    optimizables = acciones_a_optimizar or ACCIONES_OPTIMIZABLES_V6[fase]
    limites = {a: sistema.limites_accion[fase][a] for a in optimizables}
    rng = np.random.default_rng(semilla)
    candidatos = [dict(accion_base)]
    for _ in range(n_candidatos):
        cand = dict(accion_base)
        for a, (lo, hi) in limites.items():
            cand[a] = float(rng.uniform(lo, hi))
        candidatos.append(cand)
    s_ref = float(simular_acciones_lote_v6(sistema, fase, state, [accion_base], context)["support_score"].iloc[0])
    sim = objetivo_lote_v6(sistema, fase, state, candidatos, context, pesos, soporte_ref=s_ref)
    umbral = sistema.soporte_minimo.get(fase, 0.0)
    J_adm = sim["J"].where(sim["support_score"] >= umbral)
    J_adm.iloc[0] = sim["J"].iloc[0]
    i_best = int(J_adm.idxmax())
    mejor_accion, mejor_J, mejor_fila = candidatos[i_best], float(sim.loc[i_best, "J"]), sim.loc[i_best]
    if n_refinamiento > 0:
        refin = []
        for _ in range(n_refinamiento):
            cand = dict(mejor_accion)
            for a, (lo, hi) in limites.items():
                cand[a] = float(np.clip(cand[a] + rng.normal(0, 0.08 * (hi - lo)), lo, hi))
            refin.append(cand)
        sim_r = objetivo_lote_v6(sistema, fase, state, refin, context, pesos, soporte_ref=s_ref)
        J_r = sim_r["J"].where(sim_r["support_score"] >= umbral)
        if J_r.notna().any():
            j = int(J_r.idxmax())
            if float(sim_r.loc[j, "J"]) > mejor_J:
                mejor_accion, mejor_J, mejor_fila = refin[j], float(sim_r.loc[j, "J"]), sim_r.loc[j]
    return mejor_accion, mejor_J, mejor_fila


def curva_respuesta_v6(sistema: SistemaPrescriptivoV6, fase: str, state: dict, action: dict, context: dict | None,
                       palanca: str, n: int = 25, pesos: dict | None = None) -> pd.DataFrame:
    lo, hi = sistema.limites_accion[fase][palanca]
    grid = np.linspace(lo, hi, n)
    acciones = []
    for v in grid:
        a = dict(action); a[palanca] = float(v); acciones.append(a)
    s_ref = float(simular_acciones_lote_v6(sistema, fase, state, [action], context)["support_score"].iloc[0])
    sim = objetivo_lote_v6(sistema, fase, state, acciones, context, pesos, soporte_ref=s_ref)
    sim.insert(0, "valor", grid); sim.insert(0, "palanca", palanca)
    return sim


# =============================================================================
# 5) Reportes por batch, politica cross-fitted y evidencia
# =============================================================================

def reporte_recomendaciones_batch_v6(sistema: SistemaPrescriptivoV6, df: pd.DataFrame, batch_id: str, n_candidatos: int = 200,
                                     semilla: int = 0, pesos: dict | None = None) -> pd.DataFrame:
    df_base = _preparar_base(df)
    sub = df_base.loc[df_base["Batch"] == batch_id].sort_values("fecha_inicio")
    filas = []
    for _, row in sub.iterrows():
        fase = row["fase_proceso"]
        state, accion_hist, context = fila_a_state_action_context(fase, row)
        if any(pd.isna(v) for v in accion_hist.values()) or pd.isna(state.get("ley_sn_escoria_pct_prev")) or pd.isna(state.get("m6_sn_inv_kg_prev")):
            continue
        try:
            s_h = float(simular_acciones_lote_v6(sistema, fase, state, [accion_hist], context)["support_score"].iloc[0])
            sim_h = objetivo_lote_v6(sistema, fase, state, [accion_hist], context, pesos, soporte_ref=s_h).iloc[0]
            accion_opt, J_o, sim_o = optimizar_accion_v6(sistema, fase, state, accion_hist, context, None, n_candidatos, semilla=semilla, pesos=pesos)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{batch_id} {fase} {row['orden_escalon_fase']}: {type(e).__name__}: {e}")
            continue
        fila = {"Batch": batch_id, "fase": fase, "orden_escalon_fase": row["orden_escalon_fase"],
                "m6_avance_prev": state.get("m6_avance_prev"), "ley_sn_escoria_pct_prev": state["ley_sn_escoria_pct_prev"],
                "real_sn_dep": row.get("m6_ln_sn_dep"), "real_feo_ret": row.get("m6_ln_feo_ret"), "real_d_lnirf": row.get("v5_d_lnirf"),
                "temperatura_real": row.get("temperatura_horno_celsius"),
                "pred_T_historico": sim_h["pred_temperatura"], "pred_T_recomendado": sim_o["pred_temperatura"],
                "support_historico": sim_h["support_score"], "support_recomendado": sim_o["support_score"],
                "J_historico": float(sim_h["J"]), "J_recomendado": J_o, "uplift_J_kgSn": J_o - float(sim_h["J"])}
        for k in [c for c in sim_h.index if c.startswith("pred_") and c not in ("pred_temperatura",)]:
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for k in ("J_sn", "J_feo", "J_costo", "J_temperatura", "J_incertidumbre", "J_soporte"):
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for a in ACCIONES_PRIMITIVAS_V6[fase]:
            fila[f"hist__{a}"] = accion_hist[a]; fila[f"rec__{a}"] = accion_opt[a]
        for k in DERIVADAS_REPORTE:
            fila[f"hist_derivada__{k}"] = float(sim_h[f"derivada__{k}"]); fila[f"rec_derivada__{k}"] = float(sim_o[f"derivada__{k}"])
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_reporte_batch_v6(rep: pd.DataFrame) -> dict:
    if rep.empty:
        return {}
    fus, red = rep["fase"] == "Fusión", rep["fase"] == "Reducción"
    fis = sum(rep[f"{k}_rec"] - rep[f"{k}_hist"] for k in ("J_sn", "J_feo", "J_costo", "J_temperatura"))
    out = {"n_escalones": int(len(rep)),
           "uplift_fisico_fusion_kg": float(fis[fus].sum()), "uplift_fisico_reduccion_kg": float(fis[red].sum()),
           "uplift_fisico_total_kg": float(fis.sum()), "uplift_J_total_kg": float(rep["uplift_J_kgSn"].sum()),
           "pct_escalones_con_mejora": float((rep["uplift_J_kgSn"] > 1.0).mean() * 100),
           "support_medio_historico": float(rep["support_historico"].mean()),
           "support_medio_recomendado": float(rep["support_recomendado"].mean())}
    w = CONFIG_V6["w_sdi"]
    if red.any():
        out["R_sdi_pred_hist"] = float((rep.loc[red, "pred_sn_dep_hist"] + w * rep.loc[red, "pred_feo_ret_hist"]).sum())
        out["R_sdi_pred_rec"] = float((rep.loc[red, "pred_sn_dep_rec"] + w * rep.loc[red, "pred_feo_ret_rec"]).sum())
        out["R_d_sn_dep_pred"] = float((rep.loc[red, "pred_sn_dep_rec"] - rep.loc[red, "pred_sn_dep_hist"]).sum())
        out["R_d_feo_ret_pred"] = float((rep.loc[red, "pred_feo_ret_rec"] - rep.loc[red, "pred_feo_ret_hist"]).sum())
    if fus.any():
        out["F_d_sn_dep_pred"] = float((rep.loc[fus, "pred_sn_dep_rec"] - rep.loc[fus, "pred_sn_dep_hist"]).sum())
    return out


def recomendaciones_cross_fitted_v6(df: pd.DataFrame, etiqueta: str, n_folds: int = 5, n_candidatos: int = 150, semilla: int = 0,
                                    log=print, pesos: dict | None = None, config: dict | None = None) -> pd.DataFrame:
    """'0'..'n_folds-1' = fold de DEV (sistema entrenado SIN esos batches); 'lockbox' = sistema entrenado con todo DEV."""
    dev, lockbox, folds = mp4._folds_dev(df, n_folds)
    if etiqueta == "lockbox":
        sistema = entrenar_sistema_v6(df, n_boot=0, config=config)
        batches, fold, es_lb = sorted(lockbox), -1, True
    else:
        k = int(etiqueta)
        sistema = entrenar_sistema_v6(df[df["Batch"].isin(dev - set(folds[k]))], pct_lockbox=0.0, n_boot=0, config=config)
        batches, fold, es_lb = folds[k], k, False
    reps = []
    for i, b in enumerate(batches):
        rep = reporte_recomendaciones_batch_v6(sistema, df, b, n_candidatos=n_candidatos, semilla=semilla, pesos=pesos)
        if not rep.empty:
            rep["fold"], rep["es_lockbox"] = fold, es_lb
            reps.append(rep)
        if (i + 1) % 10 == 0:
            log(f"{etiqueta}: {i+1}/{len(batches)}")
    return pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()


def metricas_por_batch_v6(df: pd.DataFrame, escalones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: uplift, adherencia por palanca, KPIs y controles (misma estructura que v5)."""
    dev, _ = mp.split_dev_lockbox(df)
    df_base = _preparar_base(df)
    bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
    sd = {}
    for fase in FASES:
        s = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev)]
        for a in ACCIONES_OPTIMIZABLES_V6[fase]:
            sd[(fase, a)] = float(s[a].std())
    filas = []
    for b, rep in escalones.groupby("Batch"):
        fila = resumen_reporte_batch_v6(rep); fila["Batch"] = b
        fila["es_lockbox"] = bool(rep["es_lockbox"].iloc[0]); fila["fold"] = int(rep["fold"].iloc[0])
        dists = []
        for fase in FASES:
            r = rep.loc[rep["fase"] == fase]
            if r.empty:
                continue
            d_f = []
            for a in ACCIONES_OPTIMIZABLES_V6[fase]:
                d = ((r[f"hist__{a}"] - r[f"rec__{a}"]).abs() / sd[(fase, a)]).mean()
                fila[f"dist_{fase}_{a}"] = float(d); fila[f"dir_{fase}_{a}"] = float(((r[f"rec__{a}"] - r[f"hist__{a}"]) / sd[(fase, a)]).mean())
                d_f.append(d)
            fila[f"dist_{'F' if fase == 'Fusión' else 'R'}"] = float(np.mean(d_f)); dists += d_f
        fila["dist_total"] = float(np.mean(dists))
        g = df.loc[df["Batch"] == b]; gF = g.loc[g["fase_proceso"] == "Fusión"]
        m_, d_, p_ = (float(g[c].iloc[0]) for c in ("sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"))
        tot = m_ + d_ + p_
        feed_sn = float(g["feed_Sn_kgf"].sum())
        fila.update({"rendimiento_proxy_batch": float(g["rendimiento_proxy_batch"].iloc[0]),
                     "recuperacion_refinada_pct": float(bal.loc[b, "recuperacion_refinada_pct"]) if b in bal.index else np.nan,
                     "recuperacion_real_pct": 100 * m_ * 1000 / feed_sn if feed_sn > 0 else np.nan,
                     "f_metal": m_ / tot, "f_dross": d_ / tot, "f_polvo": p_ / tot,
                     "ley_sn_conc_batch_pct": float(g["ley_sn_conc_batch_pct"].iloc[0]), "espesor_ladrillo_norm_mm": float(g["espesor_ladrillo_norm_mm"].mean()),
                     "sn_cargado_batch_kg": feed_sn, "temperatura_media_fusion": float(gF["temperatura_horno_celsius"].mean()),
                     "frac_carga_secundaria_media_fusion": float(gF["frac_carga_secundaria"].mean()),
                     "feed_dross_Fe_total_t": float(g["feed_dross_Fe_kgh"].sum()) / 1000, "fecha_inicio_batch": g["fecha_inicio"].min()})
        filas.append(fila)
    out = pd.DataFrame(filas).sort_values("fecha_inicio_batch").reset_index(drop=True)
    out["idx_cronologico"] = np.arange(1, len(out) + 1)
    for k in ("recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_real_pct"):
        out[f"{k}_next"] = out[k].shift(-1)
        out[f"{k}_media2"] = (out[k] + out[f"{k}_next"]) / 2
    return out


CONTROLES_KPI_V6 = list(mp5.CONTROLES_KPI_V5)
evidencia_politica_v6 = mp5.evidencia_politica_v5   # opera solo sobre la tabla por batch (misma estructura)
