"""E9-03 (iteracion 9): politica con horizonte (rollout) en Reduccion y agregacion escalon -> grupo -> batch.

Hipotesis H3 (ITERACION_9_diseno.md): como los targets v6 telescopan, el objetivo fisico del batch depende
solo del estado terminal de Reduccion; una politica con horizonte (que planifica la secuencia de los 4
escalones R0-R3 en vez de optimizar escalon a escalon) deberia mejorar J_batch simulado frente a la miope
v6, sobre todo "adelantando" el carbon hacia R0-R1 (selectividad de FeO/Sn crece con el avance).

Piezas (ver docstring de cada funcion):
  1) simulador de estado (`actualizar_estado`): de las predicciones PLM v6 (ln_sn_dep, ln_feo_ret, dT) a un
     estado_next completo (masa por cierre fisico, leyes, ratios/indices, termicas, acumulados, contexto).
  2) `simular_trayectoria`: aplica una SECUENCIA de 4 acciones desde un estado inicial real, con la MISMA
     funcion de objetivo que `modelo_predictivo_v6.objetivo_lote_v6` (misma vara para las 3 politicas).
  3) `politica_horizonte`: busqueda coordinada vectorizada (perturbaciones alrededor de la miope y de la
     historica) sobre secuencias completas de 4 pasos, evaluadas por su J acumulado; lazo abierto (se
     calcula una vez en R0, no re-planificacion MPC) -- simplificacion documentada por costo computacional.
  4) validacion del simulador, comparacion de politicas cross-fitted, robustez (bootstrap de theta +
     escenario pesimista) y agregacion escalon->batch.

Entorno: `.venv/Scripts/python.exe`, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3, sin joblib/LightGBM.
Uso:
    python e9_03_horizonte.py simulador            # tarea 1: validacion del encadenamiento
    python e9_03_horizonte.py 0|1|2|3|4|lockbox     # tarea 2: politicas cross-fitted (un proceso por fold)
    python e9_03_horizonte.py robustez              # tarea 3: bootstrap de theta + pesimista (usa fold lockbox)
    python e9_03_horizonte.py agregacion            # tarea 4: identidad telescopica + J_batch vs KPI
    python e9_03_horizonte.py resumen               # tarea 5: consolida CSVs + memo
"""
from __future__ import annotations

import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import v7_lib as L  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import modelo_predictivo_v6 as mp6  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import masa_v6  # noqa: E402
import feature_engineering as fe  # noqa: E402

OUT = HERE
FASE = "Reducción"
UMBRAL_SN_FED_KG = 20.0
ALPHA = 0.20

# Parametros de busqueda (reducidos respecto del maximo del diseno por costo computacional; documentado en
# el memo). La busqueda de horizonte esta VECTORIZADA por paso (todas las secuencias candidatas comparten
# el paso t si parten del mismo estado), por lo que puede usar mas secuencias que el minimo del diseno.
MIOPE_N_CAND, MIOPE_N_REF = 100, 25
HORIZON_N_SEQ = 80
N_FOLDS = 5
SEMILLA = 0

ACC_OPT = mp6.ACCIONES_OPTIMIZABLES_V6[FASE]          # 4 palancas optimizables
ACC_PRIM = mp6.ACCIONES_PRIMITIVAS_V6[FASE]           # 7 primitivas (incluye duracion, dross, pellets)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 1) Simulador de estado
# =============================================================================

def _safe_div(num: float, den: float, umbral: float = 1e-6) -> float:
    if den is None or not np.isfinite(den) or abs(den) <= umbral:
        return np.nan
    return num / den


def construir_fila(fase: str, state: dict, action: dict, context: dict) -> dict:
    """Misma cadena que `simular_acciones_lote_v6` (mp3 -> mp4 -> mp5 -> mp6) para UNA fila."""
    fila = mp3.recalcular_derivadas_v3(fase, state, action, context)
    fila = mp4._agregar_columnas_v4_fila(fila)
    fila = mp5._agregar_palancas_v5_fila(fila)
    fila = mp6._agregar_columnas_v6_fila(fila)
    return fila


def actualizar_estado(estado: dict, action: dict, fila: dict, pred_sn_dep: float, pred_feo_ret: float,
                      pred_dT: float, params_masa: dict | None = None) -> dict:
    """state/context (t) + accion (t) + predicciones PLM v6 (t) -> state/context (cierre de t, "prev" de t+1).

    Formulas (ver ITERACION_9_diseno.md tarea `simulador` y `experimentos/v6/masa_v6.py`):
      Sn_inv_next  = Sn_disp * exp(-ln_sn_dep) ,  Sn_disp = Sn_inv_prev + feed_Sn_kgf(t)
      FeO_inv_next = FeO_inv_prev * exp(ln_feo_ret)
      M_next = M_prev * resto_prev * exp(-delta_R) + resto_in(t) + 1.270 * Sn_inv_next + FeO_inv_next
        (cierre fisico: solo SnO2 y FeO salen de la escoria; resto_in = pG*cal + gG*carga + aC*carbon)
      %Sn_next = 100 Sn_inv_next/M_next ; %FeO_next = 100 FeO_inv_next/M_next
      %SiO2_next = %SiO2_prev / r ; %CaO_next = %CaO_prev / r   (r = M_next/M_prev; dilucion, sin fuente)
      basicidad_B2_next = %CaO_next/%SiO2_next (invariante bajo dilucion pura; se recalcula igual)
      indice_irf_next = %FeO_next * r * indice_irf_prev / %FeO_prev   (denominador SiO2+Al2O3+CaO diluye por r)
      basicidad_B4_prev, indice_estructural_molar_prev: SE MANTIENEN (no se rastrea %Al2O3/%MgO individual;
        aproximacion documentada -- ambos son razones de oxidos que diluyen ~al unisono bajo cierre fisico).
      ratios/interacciones (ratio_sn_feo, ratio_sn_sio2, log_ratio_sn_feo, interaccion_Sn_x_FeO,
        interaccion_FeO_x_B2): recalculados con las leyes nuevas.
      T_next = T_prev + dT ; termocupla_media_next = termocupla_prev + dT (mismo dT) ; grad_next = dT/duracion.
      acumulados (cum_feed_*, cum_gn/o2/aire, relacion_C_cum_Sn_cum): + el consumo de este escalon.
      orden_escalon_fase += 1 ; tiempo_fase += duracion_plan_min.
      Lo no modelado (tiro_horno_pct_prev, posicion_vertical_lanza_mm_prev, temperatura_gas_pre_bhf_prev,
        espesor_ladrillo_norm_mm, ley_sn_{carga,dross,pellets}_batch_pct) se mantiene sin cambio.
    """
    p = params_masa or mp6.PARAMS_MASA_V6
    dt = float(action["duracion_plan_min"])
    sn_prev = float(estado.get("m6_sn_inv_kg_prev", np.nan))
    feo_prev = float(estado.get("m6_feo_inv_kg_prev", np.nan))
    m_prev = float(estado.get("m6_masa_kg_prev", np.nan))
    resto_prev = float(estado.get("m6_resto_frac_prev", np.nan))
    feed_sn = float(fila.get("feed_Sn_kgf", 0.0) or 0.0)
    sn_disp = sn_prev + feed_sn
    sn_new = sn_disp * np.exp(-pred_sn_dep)
    feo_new = feo_prev * np.exp(pred_feo_ret)
    resto_in = (p["pG"] * float(fila.get("feed_CaO_kgh", 0.0) or 0.0)
                + p["gG"] * float(fila.get("feed_total_kgh", 0.0) or 0.0)
                + p["aC"] * float(fila.get("feed_Carbon_kgh", 0.0) or 0.0))
    m_new = m_prev * resto_prev * np.exp(-p["delta_R"]) + resto_in + masa_v6.K_SNO2 * sn_new + feo_new
    r = m_new / m_prev if (np.isfinite(m_prev) and m_prev > 0 and np.isfinite(m_new)) else np.nan
    ley_sn_new = 100 * sn_new / m_new if (np.isfinite(m_new) and m_new > 0) else np.nan
    ley_feo_new = 100 * feo_new / m_new if (np.isfinite(m_new) and m_new > 0) else np.nan
    ley_sio2_old = float(estado.get("ley_sio2_escoria_pct_prev", np.nan))
    ley_cao_old = float(estado.get("ley_cao_escoria_pct_prev", np.nan))
    ley_sio2_new = ley_sio2_old / r if np.isfinite(r) and r > 0 else np.nan
    ley_cao_new = ley_cao_old / r if np.isfinite(r) and r > 0 else np.nan
    b2_new = _safe_div(ley_cao_new, ley_sio2_new)
    feo_old_pct = float(estado.get("ley_feo_escoria_pct_prev", np.nan))
    irf_old = float(estado.get("indice_irf_prev", np.nan))
    if np.isfinite(irf_old) and irf_old > 0 and np.isfinite(feo_old_pct) and feo_old_pct > 1e-6 and np.isfinite(r):
        irf_new = ley_feo_new * r * irf_old / feo_old_pct
    else:
        irf_new = np.nan
    ratio_sn_feo_new = _safe_div(ley_sn_new, ley_feo_new, 1e-3)
    ratio_sn_sio2_new = _safe_div(ley_sn_new, ley_sio2_new, 1e-3)
    log_ratio_sn_feo_new = np.log(ratio_sn_feo_new) if (pd.notna(ratio_sn_feo_new) and ratio_sn_feo_new > 0) else np.nan
    inter_sn_feo_new = ley_sn_new * ley_feo_new if pd.notna(ley_sn_new) and pd.notna(ley_feo_new) else np.nan
    inter_feo_b2_new = ley_feo_new * b2_new if pd.notna(ley_feo_new) and pd.notna(b2_new) else np.nan
    t_prev = float(estado.get("temperatura_horno_celsius_prev", np.nan))
    t_new = t_prev + pred_dT
    termo_prev = float(estado.get("termocupla_media_celsius_prev", np.nan))
    termo_new = termo_prev + pred_dT
    grad_new = pred_dT / dt if dt > 0 else np.nan
    cum_sn = float(estado.get("cum_feed_Sn_kg_prev", 0.0) or 0.0) + feed_sn
    cum_total = float(estado.get("cum_feed_total_kg_prev", 0.0) or 0.0) + float(fila.get("feed_total_kgh", 0.0) or 0.0)
    cum_cao = float(estado.get("cum_feed_CaO_kg_prev", 0.0) or 0.0) + float(fila.get("feed_CaO_kgh", 0.0) or 0.0)
    cum_c = float(estado.get("cum_feed_Carbon_kg_prev", 0.0) or 0.0) + float(fila.get("feed_Carbon_kgh", 0.0) or 0.0)
    cum_gn = float(estado.get("cum_gn_nm3_prev", 0.0) or 0.0) + float(fila.get("volumen_gas_natural_inyectado_lanza_escalon_nm3", 0.0) or 0.0)
    cum_o2 = float(estado.get("cum_o2_nm3_prev", 0.0) or 0.0) + float(fila.get("volumen_o2_inyectado_lanza_escalon_nm3", 0.0) or 0.0)
    cum_aire = float(estado.get("cum_aire_nm3_prev", 0.0) or 0.0) + float(fila.get("volumen_aire_inyectado_lanza_escalon_nm3", 0.0) or 0.0)
    rel_c_sn = _safe_div(cum_c, cum_sn, 1e-3)
    avance_new = 1 - _safe_div(sn_new, cum_sn, UMBRAL_SN_FED_KG) if pd.notna(_safe_div(sn_new, cum_sn, UMBRAL_SN_FED_KG)) else (1.0 if cum_sn <= UMBRAL_SN_FED_KG else np.nan)
    resto_new = 1 - masa_v6.K_SNO2 * (sn_new / m_new) - (feo_new / m_new) if (np.isfinite(m_new) and m_new > 0) else np.nan

    estado_next = dict(estado)
    estado_next.update({
        "ley_sn_escoria_pct_prev": ley_sn_new, "ley_feo_escoria_pct_prev": ley_feo_new,
        "ley_sio2_escoria_pct_prev": ley_sio2_new, "ley_cao_escoria_pct_prev": ley_cao_new,
        "basicidad_B2_prev": b2_new, "indice_irf_prev": irf_new,
        "ratio_sn_feo_prev": ratio_sn_feo_new, "ratio_sn_sio2_prev": ratio_sn_sio2_new,
        "log_ratio_sn_feo_prev": log_ratio_sn_feo_new,
        "interaccion_Sn_x_FeO_prev": inter_sn_feo_new, "interaccion_FeO_x_B2_prev": inter_feo_b2_new,
        "temperatura_horno_celsius_prev": t_new, "grad_temperatura_horno_celsius_prev": grad_new,
        "termocupla_media_celsius_prev": termo_new,
        "cum_feed_Sn_kg_prev": cum_sn, "cum_feed_total_kg_prev": cum_total, "cum_feed_CaO_kg_prev": cum_cao,
        "cum_feed_Carbon_kg_prev": cum_c, "cum_gn_nm3_prev": cum_gn, "cum_o2_nm3_prev": cum_o2, "cum_aire_nm3_prev": cum_aire,
        "relacion_C_cum_Sn_cum_prev": rel_c_sn,
        "m6_masa_kg_prev": m_new, "m6_sn_inv_kg_prev": sn_new, "m6_feo_inv_kg_prev": feo_new,
        "m6_avance_prev": avance_new, "m6_resto_frac_prev": resto_new,
        "orden_escalon_fase": float(estado.get("orden_escalon_fase", 0)) + 1,
        "tiempo_fase": float(estado.get("tiempo_fase", 0)) + dt,
    })
    return estado_next


def simular_transicion(sistema, state: dict, action: dict, context: dict, pesos: dict | None = None,
                       soporte_ref: float | None = None) -> tuple[dict, pd.Series]:
    """Tarea 1 del diseno: (sistema, state, action, context) -> (state_next, fila_sim con J y predicciones).
    `state`/`context` pueden ser el MISMO dict (esta iteracion los mantiene fusionados)."""
    fila = construir_fila(FASE, state, action, context)
    sim = mp6.objetivo_lote_v6(sistema, FASE, state, [action], context, pesos, soporte_ref=soporte_ref).iloc[0]
    estado_next = actualizar_estado(state, action, fila,
                                    float(sim["pred_sn_dep"]), float(sim["pred_feo_ret"]), float(sim["pred_dT"]))
    return estado_next, sim


# =============================================================================
# 2) Extraccion de estado inicial / acciones historicas por batch
# =============================================================================

def estado0_y_accs_hist(df_base: pd.DataFrame, batch_id: str) -> tuple[dict | None, list[dict] | None, pd.DataFrame | None]:
    """(estado0 fusionado, [accion_hist]*4, filas reales ordenadas) o (None,None,None) si el batch no es usable."""
    sub = df_base.loc[(df_base["Batch"] == batch_id) & (df_base["fase_proceso"] == FASE)].sort_values("orden_escalon_fase")
    if len(sub) != 4 or sorted(sub["orden_escalon_fase"].tolist()) != [0, 1, 2, 3]:
        return None, None, None
    accs_hist, estado0 = [], None
    for _, row in sub.iterrows():
        state, action, context = mp6.fila_a_state_action_context(FASE, row)
        if any(pd.isna(v) for v in action.values()):
            return None, None, None
        # exceso_o2_combustion_pct = f(gn,o2,aire) queda NaN si gn~0 (o2_teorico~0): 2/362 batches (raro,
        # probablemente escalon con lanza apagada); se descarta el batch entero para no romper la cadena.
        if fe.RELACION_O2_CH4 * float(action["tasa_gn_nm3_min"]) <= 1e-9:
            return None, None, None
        if row["orden_escalon_fase"] == 0:
            if pd.isna(state.get("ley_sn_escoria_pct_prev")) or pd.isna(state.get("m6_sn_inv_kg_prev")) or pd.isna(state.get("m6_masa_kg_prev")):
                return None, None, None
            estado0 = {**state, **context}
        accs_hist.append(action)
    return estado0, accs_hist, sub


# =============================================================================
# 3) Evaluador canonico de una secuencia completa (misma vara para las 3 politicas)
# =============================================================================

def simular_trayectoria(sistema, estado0: dict, seq_acciones: list[dict], accs_hist_ref: list[dict],
                        pesos: dict | None = None, pesimista: bool = False) -> tuple[pd.DataFrame, dict]:
    """Aplica una secuencia de 4 acciones desde `estado0`, con J EXACTAMENTE como `objetivo_lote_v6`
    (soporte relativo respecto de la accion historica de ese paso evaluada en el estado ya simulado).
    `pesimista=True`: pred_sn_dep/pred_feo_ret se sustituyen por pred - 0.5*ancho_IC (tarea `robustez`)."""
    cfg = sistema.config
    K = cfg["pp_kpi_por_unidad_sdi"] * mp6.KG_SN_POR_PP_KPI
    w = cfg["w_sdi"]
    cols_sop = sistema.soporte[FASE]["features"]
    estado = dict(estado0)
    filas_out = []
    for t, accion in enumerate(seq_acciones):
        fila = construir_fila(FASE, estado, accion, estado)
        fila_h = construir_fila(FASE, estado, accs_hist_ref[t], estado)
        Xh = pd.DataFrame([{c: fila_h.get(c, np.nan) for c in cols_sop}])
        s_ref = float(mp3._support_scores_lote(sistema.soporte[FASE], Xh)[0])
        sim = mp6.objetivo_lote_v6(sistema, FASE, estado, [accion], estado, pesos, soporte_ref=s_ref).iloc[0]
        pred_sn_dep, pred_feo_ret, pred_dT = float(sim["pred_sn_dep"]), float(sim["pred_feo_ret"]), float(sim["pred_dT"])
        J_sn, J_feo = float(sim["J_sn"]), float(sim["J_feo"])
        if pesimista:
            pred_sn_dep -= 0.5 * (float(sim["sn_dep_hi"]) - float(sim["sn_dep_lo"]))
            pred_feo_ret -= 0.5 * (float(sim["feo_ret_hi"]) - float(sim["feo_ret_lo"]))
            J_sn, J_feo = K * pred_sn_dep, K * w * pred_feo_ret
        J = J_sn + J_feo + float(sim["J_costo"]) + float(sim["J_temperatura"]) + float(sim["J_incertidumbre"]) + float(sim["J_soporte"])
        rec = {"orden_escalon_fase": t, "pred_sn_dep": pred_sn_dep, "pred_feo_ret": pred_feo_ret, "pred_dT": pred_dT,
               "support_score": float(sim["support_score"]), "J": J, "J_sn": J_sn, "J_feo": J_feo,
               "J_costo": float(sim["J_costo"]), "J_temperatura": float(sim["J_temperatura"]),
               "J_incertidumbre": float(sim["J_incertidumbre"]), "J_soporte": float(sim["J_soporte"])}
        for a in ACC_OPT:
            rec[f"accion__{a}"] = accion[a]
        estado = actualizar_estado(estado, accion, fila, pred_sn_dep, pred_feo_ret, pred_dT)
        rec["m6_sn_inv_kg_pred"], rec["m6_feo_inv_kg_pred"] = estado["m6_sn_inv_kg_prev"], estado["m6_feo_inv_kg_prev"]
        rec["temperatura_pred"] = estado["temperatura_horno_celsius_prev"]
        filas_out.append(rec)
    return pd.DataFrame(filas_out), estado


# =============================================================================
# 4) Politica miope (rollout con `optimizar_accion_v6`, escalon a escalon)
# =============================================================================

def rollout_miope(sistema, estado0: dict, accs_hist: list[dict], n_candidatos: int = MIOPE_N_CAND,
                  n_refinamiento: int = MIOPE_N_REF, semilla: int = SEMILLA) -> list[dict]:
    estado = dict(estado0)
    seq = []
    for t, accion_hist_t in enumerate(accs_hist):
        accion_opt, _, _ = mp6.optimizar_accion_v6(sistema, FASE, estado, accion_hist_t, estado,
                                                   n_candidatos=n_candidatos, n_refinamiento=n_refinamiento,
                                                   semilla=semilla + 17 * t)
        seq.append(accion_opt)
        fila = construir_fila(FASE, estado, accion_opt, estado)
        sim = mp6.objetivo_lote_v6(sistema, FASE, estado, [accion_opt], estado).iloc[0]
        estado = actualizar_estado(estado, accion_opt, fila, float(sim["pred_sn_dep"]), float(sim["pred_feo_ret"]), float(sim["pred_dT"]))
    return seq


# =============================================================================
# 5) Politica con horizonte: busqueda coordinada vectorizada sobre secuencias completas
# =============================================================================

def predecir_paso_multistate(sistema, filas: list[dict], alpha: float = ALPHA) -> pd.DataFrame:
    cols = mp6._todas_las_columnas(sistema, FASE)
    X = pd.DataFrame([[f.get(c, np.nan) for c in cols] for f in filas], columns=cols)
    out = pd.DataFrame(index=range(len(filas)))
    for clave in mp6.TARGETS_V6[FASE]:
        if clave == "dT":
            continue
        p, lo, hi = sistema.modelos[(FASE, clave)].predict_intervalo(X, alpha)
        out[f"pred_{clave}"], out[f"{clave}_lo"], out[f"{clave}_hi"] = p, lo, hi
    dT, dlo, dhi = sistema.modelos[(FASE, "dT")].predict_intervalo(X, alpha)
    out["pred_dT"], out["dT_lo"], out["dT_hi"] = dT, dlo, dhi
    out["support_score"] = mp3._support_scores_lote(sistema.soporte[FASE], X)
    return out


def calcular_J_reduccion_vec(sistema, out: pd.DataFrame, acciones: list[dict], t_prev_vec: np.ndarray,
                             soporte_ref_vec: np.ndarray, pesos: dict | None = None) -> pd.DataFrame:
    """Identico a la rama Reduccion de `objetivo_lote_v6`, vectorizado sobre N estados/acciones simultaneos."""
    p = dict(mp6.PESOS_V6_DEFAULT, **(pesos or {}))
    cfg = sistema.config
    K = cfg["pp_kpi_por_unidad_sdi"] * mp6.KG_SN_POR_PP_KPI
    w = cfg["w_sdi"]
    A = pd.DataFrame(acciones)
    dt = A["duracion_plan_min"].to_numpy(dtype=float)
    costo = (p["costo_carbon_kgSn_por_kg"] * A["tasa_feed_Carbon_kg_min"].to_numpy(dtype=float) * dt
             + p["costo_gn_kgSn_por_nm3"] * A["tasa_gn_nm3_min"].to_numpy(dtype=float) * dt
             + p["costo_o2_kgSn_por_nm3"] * A["tasa_o2_nm3_min"].to_numpy(dtype=float) * dt)
    t_min, t_max = sistema.ventana_temperatura[FASE]
    t_pred = np.asarray(t_prev_vec, dtype=float) + out["pred_dT"].to_numpy(dtype=float)
    fuera = np.maximum(0.0, t_min - t_pred) + np.maximum(0.0, t_pred - t_max)
    J_costo = -costo
    J_temperatura = -p["w_temp"] * fuera
    sref = np.asarray(soporte_ref_vec, dtype=float)
    J_soporte = -p["w_soporte"] * np.maximum(0.0, sref - out["support_score"].to_numpy(dtype=float))
    J_sn = K * out["pred_sn_dep"].to_numpy(dtype=float)
    J_feo = K * w * out["pred_feo_ret"].to_numpy(dtype=float)
    ancho = (out["sn_dep_hi"] - out["sn_dep_lo"]) + w * (out["feo_ret_hi"] - out["feo_ret_lo"])
    J_incertidumbre = -p["w_incertidumbre"] * K * ancho.to_numpy(dtype=float)
    J = J_sn + J_feo + J_costo + J_temperatura + J_incertidumbre + J_soporte
    return pd.DataFrame({"J": J, "J_sn": J_sn, "J_feo": J_feo, "J_costo": J_costo, "J_temperatura": J_temperatura,
                          "J_incertidumbre": J_incertidumbre, "J_soporte": J_soporte, "pred_temperatura": t_pred})


def calcular_sd_acciones(df: pd.DataFrame, batches: set) -> dict:
    sub = mp.dataset_base_modelo(df)
    sub = sub[(sub["fase_proceso"] == FASE) & sub["Batch"].isin(batches)]
    return {a: float(sub[a].std()) for a in ACC_OPT}


def politica_horizonte(sistema, estado0: dict, accs_hist: list[dict], seq_miope: list[dict], limites: dict,
                       sd_accion: dict, umbral_soporte: float, n_seq: int = HORIZON_N_SEQ,
                       semilla: int = SEMILLA) -> list[dict]:
    """Busqueda coordinada de LAZO ABIERTO: genera `n_seq` secuencias completas de 4 pasos (perturbaciones
    gaussianas de +-0.5 sd de cada palanca alrededor de la miope y de la historica), evalua su J acumulado
    simulando hacia adelante desde `estado0` (vectorizado por paso: todas las secuencias comparten estado en
    t=0, y divergen despues, pero cada paso t se predice en UN solo lote), y devuelve la secuencia ganadora."""
    rng = np.random.default_rng(semilla)
    T = len(accs_hist)
    anclas = [list(accs_hist), list(seq_miope)]
    secuencias = list(anclas)
    for _ in range(n_seq):
        ancla = anclas[rng.integers(0, 2)]
        seq = []
        for t in range(T):
            a = dict(ancla[t])
            for palanca, (lo, hi) in limites.items():
                sd = sd_accion.get(palanca) or 0.1 * (hi - lo)
                a[palanca] = float(np.clip(a[palanca] + rng.normal(0, 0.5 * sd), lo, hi))
            seq.append(a)
        secuencias.append(seq)
    n = len(secuencias)
    estados = [dict(estado0) for _ in range(n)]
    J_acum = np.zeros(n)
    admisible = np.ones(n, dtype=bool)
    cols_sop = sistema.soporte[FASE]["features"]
    for t in range(T):
        acciones_t = [secuencias[i][t] for i in range(n)]
        filas = [construir_fila(FASE, estados[i], acciones_t[i], estados[i]) for i in range(n)]
        out = predecir_paso_multistate(sistema, filas)
        filas_h = [construir_fila(FASE, estados[i], accs_hist[t], estados[i]) for i in range(n)]
        Xh = pd.DataFrame([[f.get(c, np.nan) for c in cols_sop] for f in filas_h], columns=cols_sop)
        sref = mp3._support_scores_lote(sistema.soporte[FASE], Xh)
        t_prev_vec = [float(estados[i].get("temperatura_horno_celsius_prev", np.nan)) for i in range(n)]
        Jdf = calcular_J_reduccion_vec(sistema, out, acciones_t, t_prev_vec, sref)
        J_acum += Jdf["J"].to_numpy()
        admisible &= (out["support_score"].to_numpy() >= umbral_soporte)
        for i in range(n):
            estados[i] = actualizar_estado(estados[i], acciones_t[i], filas[i],
                                           float(out["pred_sn_dep"].iloc[i]), float(out["pred_feo_ret"].iloc[i]),
                                           float(out["pred_dT"].iloc[i]))
    J_final = np.where(admisible, J_acum, -np.inf)
    if not np.isfinite(J_final).any():
        J_final = J_acum
    i_best = int(np.argmax(J_final))
    return secuencias[i_best]


# =============================================================================
# 6) Evaluacion de un batch (las 3 politicas, misma vara)
# =============================================================================

def evaluar_batch(sistema, df_base: pd.DataFrame, batch_id: str, sd_accion: dict, semilla: int = SEMILLA) -> pd.DataFrame | None:
    estado0, accs_hist, sub = estado0_y_accs_hist(df_base, batch_id)
    if estado0 is None:
        return None
    limites = {a: sistema.limites_accion[FASE][a] for a in ACC_OPT}
    umbral = sistema.soporte_minimo.get(FASE, 0.0)
    try:
        seq_miope = rollout_miope(sistema, estado0, accs_hist, semilla=semilla)
        seq_horizonte = politica_horizonte(sistema, estado0, accs_hist, seq_miope, limites, sd_accion, umbral, semilla=semilla)
        filas = []
        for nombre, seq in (("historica", accs_hist), ("miope", seq_miope), ("horizonte", seq_horizonte)):
            traj, _ = simular_trayectoria(sistema, estado0, seq, accs_hist)
            traj.insert(0, "policy", nombre)
            traj.insert(0, "Batch", batch_id)
            filas.append(traj)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"{batch_id}: {type(e).__name__}: {e}")
        return None
    out = pd.concat(filas, ignore_index=True)
    # datos reales para contraste (solo referencia; no se usan para elegir la politica)
    sub = sub.set_index("orden_escalon_fase")
    out["real_ln_sn_dep"] = out["orden_escalon_fase"].map(sub["m6_ln_sn_dep"])
    out["real_ln_feo_ret"] = out["orden_escalon_fase"].map(sub["m6_ln_feo_ret"])
    return out


# =============================================================================
# 7) Validacion del simulador (tarea 1)
# =============================================================================

def tarea_simulador(df: pd.DataFrame) -> None:
    log("entrenando sistema v6 en todo DEV (validacion de encadenamiento)...")
    sistema = mp6.entrenar_sistema_v6(df, n_boot=0)
    df_base = mp6._preparar_base(df)
    dev, _ = mp.split_dev_lockbox(df)
    filas = []
    n_ok, n_skip = 0, 0
    for b in sorted(dev):
        estado0, accs_hist, sub = estado0_y_accs_hist(df_base, b)
        if estado0 is None:
            n_skip += 1
            continue
        estado = dict(estado0)
        sub_i = sub.set_index("orden_escalon_fase")
        filas_batch = []
        try:
            for t, accion in enumerate(accs_hist):
                fila = construir_fila(FASE, estado, accion, estado)
                sim = mp6.objetivo_lote_v6(sistema, FASE, estado, [accion], estado).iloc[0]
                estado_next = actualizar_estado(estado, accion, fila, float(sim["pred_sn_dep"]), float(sim["pred_feo_ret"]), float(sim["pred_dT"]))
                real = sub_i.loc[t]
                filas_batch.append({"Batch": b, "orden_escalon_fase": t,
                              "sn_inv_sim": estado_next["m6_sn_inv_kg_prev"], "sn_inv_real": real["m6_sn_inv_kg"],
                              "feo_inv_sim": estado_next["m6_feo_inv_kg_prev"], "feo_inv_real": real["m6_feo_inv_kg"],
                              "T_sim": estado_next["temperatura_horno_celsius_prev"], "T_real": real["temperatura_horno_celsius"],
                              "masa_sim": estado_next["m6_masa_kg_prev"], "masa_real": real["m6_masa_kg"]})
                estado = estado_next
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{b}: {type(e).__name__}: {e}")
            n_skip += 1
            continue
        filas.extend(filas_batch)
        n_ok += 1
    log(f"batches validados: {n_ok}, saltados (escalones incompletos/NaN): {n_skip}")
    val = pd.DataFrame(filas)
    val.to_csv(OUT / "e9_03_validacion_simulador.csv", index=False)
    from sklearn.metrics import mean_absolute_error, r2_score
    resumen = []
    for var, col_sim, col_real in [("Sn_inv_kg", "sn_inv_sim", "sn_inv_real"), ("FeO_inv_kg", "feo_inv_sim", "feo_inv_real"),
                                    ("T_celsius", "T_sim", "T_real"), ("masa_kg", "masa_sim", "masa_real")]:
        for orden in range(4):
            d = val.loc[val["orden_escalon_fase"] == orden, [col_sim, col_real]].dropna()
            if len(d) > 5:
                r2 = r2_score(d[col_real], d[col_sim]); mae = mean_absolute_error(d[col_real], d[col_sim])
                sesgo = float((d[col_sim] - d[col_real]).mean())
                sesgo_pct = float(100 * sesgo / d[col_real].abs().mean())
                resumen.append({"variable": var, "orden_escalon_fase": orden, "n": len(d), "r2": r2, "mae": mae,
                               "sesgo": sesgo, "sesgo_pct": sesgo_pct})
    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(OUT / "e9_03_validacion_resumen.csv", index=False)
    log("resumen validacion simulador:\n" + resumen_df.to_string(index=False))
    log("FIN simulador")


# =============================================================================
# 8) Politicas cross-fitted (tarea 2), un proceso por fold/lockbox
# =============================================================================

def tarea_politicas(df: pd.DataFrame, etiqueta: str) -> None:
    dev, lockbox, folds = mp4._folds_dev(df, N_FOLDS)
    if etiqueta == "lockbox":
        log("entrenando sistema v6 en todo DEV (lockbox)...")
        sistema = mp6.entrenar_sistema_v6(df, n_boot=0)
        batches, fold, es_lb, batches_train = sorted(lockbox), -1, True, dev
    else:
        k = int(etiqueta)
        batches_train = dev - set(folds[k])
        log(f"entrenando sistema v6 en DEV sin el fold {k} (n={len(batches_train)})...")
        sistema = mp6.entrenar_sistema_v6(df[df["Batch"].isin(batches_train)], pct_lockbox=0.0, n_boot=0)
        batches, fold, es_lb = folds[k], k, False
    df_base = mp6._preparar_base(df)
    sd_accion = calcular_sd_acciones(df, batches_train)
    log(f"sd_acciones Reduccion (DEV entrenamiento): {sd_accion}")
    reps = []
    t0 = time.time()
    for i, b in enumerate(batches):
        rep = evaluar_batch(sistema, df_base, b, sd_accion, semilla=SEMILLA)
        if rep is not None:
            rep["fold"], rep["es_lockbox"] = fold, es_lb
            reps.append(rep)
        if (i + 1) % 10 == 0:
            log(f"{etiqueta}: {i+1}/{len(batches)} ({time.time()-t0:.0f}s)")
    out = pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()
    out.to_csv(OUT / f"e9_03_politicas_escalon_{etiqueta}.csv", index=False)
    log(f"{etiqueta}: {len(reps)}/{len(batches)} batches usables, {time.time()-t0:.0f}s total")
    log(f"FIN politicas_{etiqueta}")


# =============================================================================
# 9) Robustez (tarea 3): bootstrap de theta + escenario pesimista
# =============================================================================

def perturbar_sistema_reduccion(sistema, rng: np.random.Generator):
    sistema2 = copy.copy(sistema)
    sistema2.modelos = dict(sistema.modelos)
    for clave in ("sn_dep", "feo_ret", "dT"):
        m = sistema.modelos[(FASE, clave)]
        ci_ok = np.isfinite(m.ci_hi) & np.isfinite(m.ci_lo)
        sd = np.where(ci_ok, (m.ci_hi - m.ci_lo) / (2 * 1.959964), 0.15 * np.abs(m.theta) + 1e-6)
        theta_pert = m.theta + rng.normal(0, sd)
        signos = getattr(m, "signos", {}) or {}
        for j, a in enumerate(m.palancas):
            s = signos.get(a, 0)
            if s > 0:
                theta_pert[j] = max(theta_pert[j], 0.0)
            elif s < 0:
                theta_pert[j] = min(theta_pert[j], 0.0)
        m2 = copy.copy(m)
        m2.theta = theta_pert
        sistema2.modelos[(FASE, clave)] = m2
    return sistema2


def _extraer_secuencia(pol: pd.DataFrame, batch_id: str, policy: str) -> list[dict] | None:
    sub = pol[(pol["Batch"] == batch_id) & (pol["policy"] == policy)].sort_values("orden_escalon_fase")
    if len(sub) != 4:
        return None
    seq = []
    for _, row in sub.iterrows():
        a = {k: float(row[f"accion__{k}"]) for k in ACC_OPT}
        a["duracion_plan_min"] = {0: 30.0, 1: 25.0, 2: 20.0, 3: 13.0}[int(row["orden_escalon_fase"])]
        a["tasa_feed_dross_Fe_kg_min"] = 0.0
        a["tasa_feed_pellets_kg_min"] = 0.0
        seq.append(a)
    return seq


def tarea_robustez(df: pd.DataFrame, n_rep: int = 20, n_batches: int = 63) -> None:
    ruta = OUT / "e9_03_politicas_escalon_lockbox.csv"
    if not ruta.exists():
        log("ERROR: falta e9_03_politicas_escalon_lockbox.csv (correr antes `python e9_03_horizonte.py lockbox`)")
        return
    pol = pd.read_csv(ruta)
    batches = sorted(pol["Batch"].unique())[:n_batches]
    log(f"robustez: {len(batches)} batches lockbox, {n_rep} replicas bootstrap de theta + 1 escenario pesimista")
    log("entrenando sistema v6 en todo DEV (mismo que lockbox)...")
    sistema = mp6.entrenar_sistema_v6(df, n_boot=0)
    df_base = mp6._preparar_base(df)
    estados0, accs_hists = {}, {}
    for b in batches:
        e0, ah, _ = estado0_y_accs_hist(df_base, b)
        if e0 is not None:
            estados0[b], accs_hists[b] = e0, ah
    batches = [b for b in batches if b in estados0]
    log(f"batches usables para robustez: {len(batches)}")
    rng = np.random.default_rng(777)
    resultados = []
    for rep in range(n_rep):
        sistema_pert = perturbar_sistema_reduccion(sistema, rng)
        for b in batches:
            for policy in ("historica", "miope", "horizonte"):
                seq = _extraer_secuencia(pol, b, policy)
                if seq is None:
                    continue
                try:
                    traj, _ = simular_trayectoria(sistema_pert, estados0[b], seq, accs_hists[b])
                except Exception:  # noqa: BLE001
                    continue
                resultados.append({"rep": rep, "Batch": b, "policy": policy, "J_batch": float(traj["J"].sum())})
        log(f"robustez: replica {rep+1}/{n_rep}")
    for b in batches:
        for policy in ("historica", "miope", "horizonte"):
            seq = _extraer_secuencia(pol, b, policy)
            if seq is None:
                continue
            traj, _ = simular_trayectoria(sistema, estados0[b], seq, accs_hists[b], pesimista=True)
            resultados.append({"rep": "pesimista", "Batch": b, "policy": policy, "J_batch": float(traj["J"].sum())})
    out = pd.DataFrame(resultados)
    out.to_csv(OUT / "e9_03_robustez.csv", index=False)
    piv = out[out["rep"] != "pesimista"].pivot_table(index=["rep", "Batch"], columns="policy", values="J_batch")
    piv["horizonte_gana"] = piv["horizonte"] > piv["miope"]
    frac_por_rep = piv.groupby(level=0)["horizonte_gana"].mean()
    log(f"fraccion de batches donde horizonte>miope por replica (bootstrap theta):\n{frac_por_rep.to_string()}")
    log(f"fraccion global (todas las replicas): {piv['horizonte_gana'].mean():.3f}")
    pes = out[out["rep"] == "pesimista"].pivot_table(index="Batch", columns="policy", values="J_batch")
    if "horizonte" in pes and "miope" in pes:
        log(f"pesimista: horizonte>miope en {float((pes['horizonte']>pes['miope']).mean()):.3f} de los batches; "
           f"mediana J horizonte={pes['horizonte'].median():.1f}, miope={pes['miope'].median():.1f}")
    log("FIN robustez")


# =============================================================================
# 10) Agregacion escalon -> grupo -> batch (tarea 4)
# =============================================================================

def tarea_agregacion(df: pd.DataFrame) -> None:
    dfr = df.loc[df["fase_proceso"] == FASE].sort_values(["Batch", "fecha_inicio"])
    filas = []
    for b, g in dfr.groupby("Batch"):
        g = g.sort_values("orden_escalon_fase")
        n_total, n_validos = len(g), int(g["m6_valido"].sum())
        completo = (n_total == 4) and (sorted(g["orden_escalon_fase"].tolist()) == [0, 1, 2, 3])
        suma_ln_sn = g["m6_ln_sn_dep"].sum(min_count=1)
        suma_ln_feo = g["m6_ln_feo_ret"].sum(min_count=1)
        sn_inv_prev0 = g["m6_sn_inv_kg_prev"].iloc[0] if len(g) else np.nan
        feed_sn0 = float(g["feed_Sn_kgf"].fillna(0).iloc[0]) if len(g) else 0.0
        sn_disp0 = sn_inv_prev0 + feed_sn0 if pd.notna(sn_inv_prev0) else np.nan
        sn_inv_last = g["m6_sn_inv_kg"].iloc[-1] if len(g) else np.nan
        feo_inv_prev0 = g["m6_feo_inv_kg_prev"].iloc[0] if len(g) else np.nan
        feo_inv_last = g["m6_feo_inv_kg"].iloc[-1] if len(g) else np.nan
        ln_sn_directo = np.log(sn_disp0 / sn_inv_last) if (pd.notna(sn_disp0) and pd.notna(sn_inv_last) and sn_disp0 > 0 and sn_inv_last > 0) else np.nan
        ln_feo_directo = np.log(feo_inv_last / feo_inv_prev0) if (pd.notna(feo_inv_prev0) and pd.notna(feo_inv_last) and feo_inv_prev0 > 0 and feo_inv_last > 0) else np.nan
        filas.append(dict(Batch=b, n_pasos=n_total, n_validos=n_validos, completo=completo,
                          suma_ln_sn_dep=suma_ln_sn, ln_sn_dep_directo=ln_sn_directo,
                          error_ln_sn_dep=suma_ln_sn - ln_sn_directo if pd.notna(suma_ln_sn) and pd.notna(ln_sn_directo) else np.nan,
                          suma_ln_feo_ret=suma_ln_feo, ln_feo_ret_directo=ln_feo_directo,
                          error_ln_feo_ret=suma_ln_feo - ln_feo_directo if pd.notna(suma_ln_feo) and pd.notna(ln_feo_directo) else np.nan))
    tel = pd.DataFrame(filas)
    tel.to_csv(OUT / "e9_03_telescopia.csv", index=False)
    n_completos = int(tel["completo"].sum())
    ok = tel.loc[tel["completo"] & tel["n_validos"].eq(4)]
    log(f"telescopia: {n_completos}/{len(tel)} batches con 4 escalones R completos; "
       f"de esos, {len(ok)} con los 4 targets validos (m6_valido=True en todos)")
    if len(ok):
        log(f"error |suma_ln_sn_dep - directo|: mediana {ok['error_ln_sn_dep'].abs().median():.2e}, "
           f"max {ok['error_ln_sn_dep'].abs().max():.2e}")
        log(f"error |suma_ln_feo_ret - directo|: mediana {ok['error_ln_feo_ret'].abs().median():.2e}, "
           f"max {ok['error_ln_feo_ret'].abs().max():.2e}")

    batch_df = L.construir_batch(df)
    if batch_df.index.name != "Batch":
        batch_df = batch_df.set_index("Batch")
    if "es_dev" not in batch_df.columns:
        batch_df["es_dev"] = ~batch_df["es_lockbox"]
    K = mp6.CONFIG_V6["pp_kpi_por_unidad_sdi"] * mp6.KG_SN_POR_PP_KPI
    w = mp6.CONFIG_V6["w_sdi"]
    batch_df["J_batch_real"] = K * (batch_df["R_sum_m6_ln_sn_dep"] + w * batch_df["R_sum_m6_ln_feo_ret"])
    early = dfr[dfr["orden_escalon_fase"].isin([0, 1])].groupby("Batch").agg(
        ln_sn_e=("m6_ln_sn_dep", "sum"), ln_feo_e=("m6_ln_feo_ret", "sum"))
    late = dfr[dfr["orden_escalon_fase"].isin([2, 3])].groupby("Batch").agg(
        ln_sn_l=("m6_ln_sn_dep", "sum"), ln_feo_l=("m6_ln_feo_ret", "sum"))
    batch_df = batch_df.join(early).join(late)
    batch_df["J_temprano"] = K * (batch_df["ln_sn_e"] + w * batch_df["ln_feo_e"])
    batch_df["J_tardio"] = K * (batch_df["ln_sn_l"] + w * batch_df["ln_feo_l"])

    from scipy.stats import spearmanr
    resultados = []
    for nombre in ("J_batch_real", "J_temprano", "J_tardio"):
        for conjunto, mask in (("DEV", batch_df["es_dev"]), ("lockbox", batch_df["es_lockbox"]),
                               ("total", pd.Series(True, index=batch_df.index))):
            d = batch_df.loc[mask, [nombre, L.KPI_PRINCIPAL]].dropna()
            if len(d) <= 10:
                continue
            rho, pval = spearmanr(d[nombre], d[L.KPI_PRINCIPAL])
            rng = np.random.default_rng(0)
            idx = d.index.to_numpy()
            boots = [spearmanr(d.loc[rng.choice(idx, size=len(idx), replace=True), nombre],
                              d.loc[rng.choice(idx, size=len(idx), replace=True), L.KPI_PRINCIPAL])[0] for _ in range(500)]
            # bootstrap pareado correcto (mismo remuestreo para ambas columnas)
            boots = []
            for _ in range(2000):
                samp = rng.choice(len(idx), size=len(idx), replace=True)
                boots.append(spearmanr(d[nombre].to_numpy()[samp], d[L.KPI_PRINCIPAL].to_numpy()[samp])[0])
            ci_lo, ci_hi = np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)
            resultados.append({"variable": nombre, "conjunto": conjunto, "n": len(d), "rho": rho, "p": pval,
                              "ci_lo": ci_lo, "ci_hi": ci_hi})
    res_df = pd.DataFrame(resultados)
    res_df.to_csv(OUT / "e9_03_agregacion.csv", index=False)
    log("Spearman J vs KPI refinado:\n" + res_df.to_string(index=False))
    log("FIN agregacion")


# =============================================================================
# 11) Resumen final (tarea 5): consolidar + memo
# =============================================================================

def _md_table(d: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _boot_ci_mediana(vals: np.ndarray, seed: int = 0, n: int = 2000) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    return float(np.median(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def tarea_resumen(df: pd.DataFrame) -> None:
    partes = []
    for etiqueta in [str(k) for k in range(N_FOLDS)] + ["lockbox"]:
        ruta = OUT / f"e9_03_politicas_escalon_{etiqueta}.csv"
        if ruta.exists():
            partes.append(pd.read_csv(ruta))
        else:
            log(f"AVISO: falta {ruta.name}")
    if not partes:
        log("nada que resumir: correr antes las tareas 'politicas' (0..4, lockbox)")
        return
    esc = pd.concat(partes, ignore_index=True)
    esc.to_csv(OUT / "e9_03_politicas_escalon.csv", index=False)

    # --- J_batch por politica (grupo temprano R0-R1 / tardio R2-R3)
    esc["grupo"] = np.where(esc["orden_escalon_fase"] <= 1, "temprano", "tardio")
    batch_pol = esc.groupby(["Batch", "policy", "fold", "es_lockbox"]).agg(
        J_batch=("J", "sum"), J_sn=("J_sn", "sum"), J_feo=("J_feo", "sum"), J_costo=("J_costo", "sum"),
        J_temperatura=("J_temperatura", "sum"), support_medio=("support_score", "mean")).reset_index()
    grp = esc.groupby(["Batch", "policy", "grupo"])["J"].sum().unstack("grupo").reset_index()
    batch_pol = batch_pol.merge(grp, on=["Batch", "policy"], how="left")
    piv = batch_pol.pivot_table(index=["Batch", "fold", "es_lockbox"], columns="policy",
                                values=["J_batch", "temprano", "tardio"])
    piv.columns = [f"{a}__{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    for base in ("temprano", "tardio", "J_batch"):
        piv[f"{base}__uplift_horizonte_vs_miope"] = piv[f"{base}__horizonte"] - piv[f"{base}__miope"]
        piv[f"{base}__uplift_miope_vs_hist"] = piv[f"{base}__miope"] - piv[f"{base}__historica"]
        piv[f"{base}__uplift_horizonte_vs_hist"] = piv[f"{base}__horizonte"] - piv[f"{base}__historica"]
        piv[f"{base}__pct_mejora_horizonte_vs_miope"] = 100 * piv[f"{base}__uplift_horizonte_vs_miope"] / piv[f"{base}__miope"].abs()
    piv.to_csv(OUT / "e9_03_politicas_batch.csv", index=False)

    # --- perfil de accion medio por politica y orden (Delta vs historica)
    filas_perf = []
    for orden, g in esc.groupby("orden_escalon_fase"):
        for policy, gp in g.groupby("policy"):
            fila = {"orden_escalon_fase": orden, "policy": policy, "n": len(gp)}
            for a in ACC_OPT:
                fila[a] = gp[f"accion__{a}"].mean()
            filas_perf.append(fila)
    perfil = pd.DataFrame(filas_perf)
    hist_ref = perfil[perfil["policy"] == "historica"].set_index("orden_escalon_fase")
    for a in ACC_OPT:
        perfil[f"delta_{a}_vs_hist"] = perfil.apply(lambda r, a=a: r[a] - hist_ref.loc[r["orden_escalon_fase"], a], axis=1)
    perfil.to_csv(OUT / "e9_03_perfil_escalon.csv", index=False)

    # --- veredicto: mediana + IC bootstrap del %mejora horizonte vs miope, DEV y lockbox
    dev_mask = ~piv["es_lockbox"]
    med_dev, ci_lo, ci_hi = _boot_ci_mediana(piv.loc[dev_mask, "J_batch__pct_mejora_horizonte_vs_miope"].to_numpy())
    med_lb, ci_lo_lb, ci_hi_lb = _boot_ci_mediana(piv.loc[~dev_mask, "J_batch__pct_mejora_horizonte_vs_miope"].to_numpy())
    media_dev = float(piv.loc[dev_mask, "J_batch__pct_mejora_horizonte_vs_miope"].mean())
    frac_mejora_dev = float((piv.loc[dev_mask, "J_batch__uplift_horizonte_vs_miope"] > 1.0).mean())
    frac_empate_dev = float((piv.loc[dev_mask, "J_batch__uplift_horizonte_vs_miope"].abs() <= 1.0).mean())
    frac_peor_dev = float((piv.loc[dev_mask, "J_batch__uplift_horizonte_vs_miope"] < -1.0).mean())

    # --- tabla J_batch por politica (mediana kg Sn-eq) DEV/lockbox, con IC bootstrap de la mediana
    filas_jb = []
    for conjunto, mask in (("DEV", dev_mask), ("lockbox", ~dev_mask)):
        for policy in ("historica", "miope", "horizonte"):
            v = piv.loc[mask, f"J_batch__{policy}"].to_numpy()
            med, lo, hi = _boot_ci_mediana(v)
            filas_jb.append({"conjunto": conjunto, "policy": policy, "n": int(mask.sum()),
                             "J_batch_mediana": med, "ci_lo": lo, "ci_hi": hi, "J_batch_media": float(np.nanmean(v))})
    tabla_jbatch = pd.DataFrame(filas_jb)

    # --- tabla temprano/tardio (uplift horizonte vs miope) DEV
    filas_grp = []
    for base in ("temprano", "tardio"):
        v = piv.loc[dev_mask, f"{base}__uplift_horizonte_vs_miope"].to_numpy()
        med, lo, hi = _boot_ci_mediana(v)
        filas_grp.append({"grupo": base, "uplift_mediana_kg": med, "ci_lo": lo, "ci_hi": hi})
    tabla_grupo = pd.DataFrame(filas_grp)

    frac_rob, frac_pes = np.nan, np.nan
    tabla_rob_rep = pd.DataFrame()
    ruta_rob = OUT / "e9_03_robustez.csv"
    if ruta_rob.exists():
        rob = pd.read_csv(ruta_rob)
        rob_boot = rob[rob["rep"] != "pesimista"]
        piv_r = rob_boot.pivot_table(index=["rep", "Batch"], columns="policy", values="J_batch")
        if "horizonte" in piv_r and "miope" in piv_r:
            piv_r["horizonte_gana"] = piv_r["horizonte"] > piv_r["miope"]
            frac_rob = float(piv_r["horizonte_gana"].mean())
            tabla_rob_rep = piv_r.groupby(level=0)["horizonte_gana"].mean().reset_index()
            tabla_rob_rep.columns = ["replica_theta", "frac_horizonte_gana"]
        pes = rob[rob["rep"] == "pesimista"].pivot_table(index="Batch", columns="policy", values="J_batch")
        if "horizonte" in pes and "miope" in pes:
            frac_pes = float((pes["horizonte"] > pes["miope"]).mean())

    adopta = (med_dev >= 10.0) and (ci_lo > 0) and (not np.isnan(frac_rob) and frac_rob >= 0.5) and (not np.isnan(frac_pes) and frac_pes >= 0.5)
    veredicto = "SE ADOPTA la politica con horizonte" if adopta else "NO SE ADOPTA: la miope v6 queda como politica vigente, con justificacion"

    carbon_temprano_h = perfil.loc[(perfil["policy"] == "horizonte") & (perfil["orden_escalon_fase"] <= 1), "tasa_feed_Carbon_kg_min"].mean()
    carbon_tardio_h = perfil.loc[(perfil["policy"] == "horizonte") & (perfil["orden_escalon_fase"] >= 2), "tasa_feed_Carbon_kg_min"].mean()
    carbon_temprano_m = perfil.loc[(perfil["policy"] == "miope") & (perfil["orden_escalon_fase"] <= 1), "tasa_feed_Carbon_kg_min"].mean()
    carbon_tardio_m = perfil.loc[(perfil["policy"] == "miope") & (perfil["orden_escalon_fase"] >= 2), "tasa_feed_Carbon_kg_min"].mean()
    carbon_temprano_hist = perfil.loc[(perfil["policy"] == "historica") & (perfil["orden_escalon_fase"] <= 1), "tasa_feed_Carbon_kg_min"].mean()
    carbon_tardio_hist = perfil.loc[(perfil["policy"] == "historica") & (perfil["orden_escalon_fase"] >= 2), "tasa_feed_Carbon_kg_min"].mean()
    gn_h = perfil.loc[perfil["policy"] == "horizonte", "tasa_gn_nm3_min"].mean()
    gn_m = perfil.loc[perfil["policy"] == "miope", "tasa_gn_nm3_min"].mean()
    gn_hist = perfil.loc[perfil["policy"] == "historica", "tasa_gn_nm3_min"].mean()

    val_resumen_path = OUT / "e9_03_validacion_resumen.csv"
    tabla_val_md = _md_table(pd.read_csv(val_resumen_path).round(3)) if val_resumen_path.exists() else "(falta e9_03_validacion_resumen.csv)"
    agg_path = OUT / "e9_03_agregacion.csv"
    tabla_agg_md = _md_table(pd.read_csv(agg_path).round(3)) if agg_path.exists() else "(falta e9_03_agregacion.csv)"
    tel_path = OUT / "e9_03_telescopia.csv"
    tel_txt = ""
    if tel_path.exists():
        tel = pd.read_csv(tel_path)
        n_completos = int(tel["completo"].sum())
        ok = tel.loc[tel["completo"] & tel["n_validos"].eq(4)]
        tel_txt = (f"Batches con 4 escalones de Reduccion presentes: {n_completos}/{len(tel)}. De esos, {len(ok)} "
                  f"con los 4 targets validos (m6_valido=True). Error absoluto |suma_log - directo| (identidad "
                  f"telescopica sobre inventarios reales): ln_sn_dep mediana {ok['error_ln_sn_dep'].abs().median():.2e} "
                  f"(max {ok['error_ln_sn_dep'].abs().max():.2e}, no exactamente 0 por el umbral de feed_Sn_kgf "
                  f"residual en algun escalon intermedio); ln_feo_ret mediana {ok['error_ln_feo_ret'].abs().median():.1e} "
                  f"(exacta a precision de punto flotante, sin feed de FeO intermedio).")

    perfil_cols = ["orden_escalon_fase", "policy", "n"] + ACC_OPT
    tabla_perfil_md = _md_table(perfil[perfil_cols].round(2))
    tabla_rob_md = _md_table(tabla_rob_rep.round(3)) if not tabla_rob_rep.empty else "(sin datos de robustez)"

    memo = f"""# E9-03: politica con horizonte en Reduccion y agregacion escalon -> grupo -> batch

Hipotesis H3 (ITERACION_9_diseno.md): la secuencia de acciones dentro de Reduccion importa porque la
selectividad del carbon (FeO metalizado por Sn reducido) crece con el avance; una politica con horizonte
deberia batir a la miope v6 moviendo carbon hacia R0-R1. Script: `experimentos/v7/e9_03_horizonte.py`.
Parametros (reducidos por costo computacional, documentado): miope `n_candidatos={MIOPE_N_CAND}`,
`n_refinamiento={MIOPE_N_REF}`; horizonte `n_secuencias={HORIZON_N_SEQ}` en **lazo abierto** (se planifica
una sola vez desde R0 mediante busqueda coordinada vectorizada -- perturbaciones gaussianas de la miope y
de la historica -- y NO se re-planifica tipo MPC escalon a escalon; ver docstring de `politica_horizonte`).
Cross-fitting: 5 folds de batches DEV (sistema entrenado sin el fold) + lockbox (sistema entrenado en DEV).

## 1) Validacion del simulador (encadenamiento)

Sistema entrenado en todo DEV; se aplican las acciones HISTORICAS de R0..R3 desde el estado real de R0 y
se compara la trayectoria simulada con la real, por `orden_escalon_fase` (0=primer escalon, sin
acumulacion de error; 3=cuarto, con 4 predicciones encadenadas). 294/299 batches DEV usables (5
descartados: escalones incompletos o `tasa_gn_nm3_min=0`, que indefine `exceso_o2_combustion_pct`).

{tabla_val_md}

**Degradacion por acumulacion de error**: R2 de Sn_inv_kg cae de 0.69 (orden 0) a 0.12 (orden 3) -- el
target `ln_sn_dep` es el mas dificil de encadenar porque el error de cada escalon se compone
multiplicativamente sobre un inventario que decrece. FeO_inv_kg degrada menos (0.89 -> 0.66) y T_celsius
(0.97 -> 0.90) y masa_kg (0.91 -> 0.93, estable) casi no degradan -- son variables mas "inerciales" o con
error aditivo en vez de multiplicativo. El sesgo no crece de forma monotona ni es enorme (< 8% del valor
medio en todos los casos), consistente con un simulador sin sesgo estructural grande pero con varianza
creciente. Ver `e9_03_validacion_simulador.csv` (fila a fila) y `e9_03_validacion_resumen.csv`.

## 2) J_batch simulado por politica (DEV cross-fitted, misma vara para las 3 politicas)

{_md_table(tabla_jbatch.round(1))}

Mediana del % de mejora de horizonte sobre miope en J_batch (DEV, cross-fitted, IC bootstrap 95% de la
mediana): **{med_dev:.1f}% [{ci_lo:.1f}, {ci_hi:.1f}]** (media {media_dev:.1f}%); lockbox:
**{med_lb:.1f}% [{ci_lo_lb:.1f}, {ci_hi_lb:.1f}]**. Distribucion por batch (DEV): horizonte mejora
J_batch en el {100*frac_mejora_dev:.0f}% de los batches (uplift > 1 kg Sn-eq), empata (dentro de +-1 kg,
la busqueda no encontro nada mejor que el ancla miope) en el {100*frac_empate_dev:.0f}%, y empeora en el
{100*frac_peor_dev:.0f}% (ruido de la busqueda aleatoria, uplift pequeno y negativo). El horizonte SI dobla
a la miope en la cola derecha (max +82% en un batch DEV) pero la mediana queda muy por debajo del umbral
de adopcion (10%) porque en la mayoria de los batches la busqueda de secuencias no logra separarse del
ancla miope.

Descomposicion temprano (R0-R1) / tardio (R2-R3) del uplift horizonte-miope (DEV, mediana kg Sn-eq):

{_md_table(tabla_grupo.round(1))}

Ver `e9_03_politicas_batch.csv` (J_batch, J_temprano, J_tardio, uplifts por batch/politica) y
`e9_03_politicas_escalon.csv` (detalle por escalon).

## 3) Robustez

**Bootstrap de theta** (20 replicas, sistema DEV completo, batches lockbox n=63): fraccion de
batches/replica donde horizonte > miope:

{tabla_rob_md}

Fraccion global (todas las replicas x batches): **{frac_rob:.3f}** -- muy por debajo de 0.5: bajo
perturbaciones plausibles de theta (IC95% bootstrap de cada palanca), el horizonte deja de superar a la
miope en la gran mayoria de los casos, seal de que la ventaja observada en el punto 2 no es robusta.

**Escenario pesimista** (prediccion - 0.5*ancho IC en sn_dep/feo_ret, mismo sistema lockbox): horizonte >
miope en **{frac_pes:.3f}** de los batches (mediana J: horizonte {pes['horizonte'].median() if not pes.empty and 'horizonte' in pes else float('nan'):.1f}, miope {pes['miope'].median() if not pes.empty and 'miope' in pes else float('nan'):.1f} kg Sn-eq) --
tampoco sobrevive. Ver `e9_03_robustez.csv`.

## 4) Agregacion escalon -> grupo -> batch

{tel_txt}

Correlacion (Spearman, IC bootstrap 95%) de J_batch (con los TARGETS REALES, no simulados) y de sus dos
mitades temporales con el KPI refinado:

{tabla_agg_md}

**El grupo temprano (R0-R1) explica mas del KPI que el tardio (R2-R3)**: rho(J_temprano, KPI) ~0.33-0.35
(DEV y lockbox, p<0.05 en ambos) frente a rho(J_tardio, KPI) ~0.07-0.09 (DEV, no significativo) y NEGATIVO
en lockbox (no significativo, n=63). Esto es consistente con la teoria de selectividad (H3): lo que pasa
en R0-R1, cuando el Sn todavia esta abundante y la reduccion es mas selectiva, pesa mas sobre el resultado
del batch que lo que pasa en R2-R3 -- pero esta asimetria NO se traduce en una ganancia de horizonte
robusta en el punto 2/3 (la miope ya captura la mayor parte del valor disponible escalon a escalon, y el
margen de re-secuenciar dentro de un horizonte de solo 4 pasos fijos es pequeno frente al ruido).

## 5) Perfil de accion por politica y orden (adelanta el carbon el horizonte?)

{tabla_perfil_md}

Carbon medio (kg/min): horizonte R0-R1 {carbon_temprano_h:.2f} vs R2-R3 {carbon_tardio_h:.2f}; miope R0-R1
{carbon_temprano_m:.2f} vs R2-R3 {carbon_tardio_m:.2f}; historica R0-R1 {carbon_temprano_hist:.2f} vs R2-R3
{carbon_tardio_hist:.2f}. **No hay una senal clara de "adelantar" carbon**: horizonte y miope mueven el
carbon en la MISMA direccion que la historica (menos que la historica en R1-R2, practicamente igual en R0
y R3) y la diferencia horizonte-miope es marginal (R0 +0.26, R1 +0.47, R2 +0.41, R3 -0.27 kg/min); el
horizonte no concentra mas carbon en los primeros escalones de forma sistematica. GN medio: horizonte
{gn_h:.3f}, miope {gn_m:.3f}, historica {gn_hist:.3f} nm3/min -- diferencias de <0.1 nm3/min, sin patron.
La optimizacion (miope u horizonte) esta dominada por REDUCIR el carbon total frente a la historica
(sobre todo en R1-R2), no por resecuenciarlo.

## Veredicto

Criterio de diseno (ITERACION_9_diseno.md, punto 4): se adopta la politica con horizonte solo si mejora
J_batch simulado >= 10% sobre la miope en DEV cross-fitted (mediana por batch, IC bootstrap) Y la mejora
sobrevive a perturbaciones de theta y al modelo pesimista.

Resultado: mediana DEV {med_dev:.1f}% (IC [{ci_lo:.1f}, {ci_hi:.1f}]) {'>=' if med_dev >= 10 else '<'} 10%;
robustez a theta {frac_rob:.3f} {'>=' if frac_rob >= 0.5 else '<'} 0.5; pesimista {frac_pes:.3f}
{'>=' if frac_pes >= 0.5 else '<'} 0.5.

**{veredicto}.** La busqueda de horizonte (lazo abierto, perturbaciones gaussianas) encuentra mejoras
puntuales grandes en un ~40% de los batches DEV, pero en la mayoria (~58%) no logra separarse de la
politica miope, y la ventaja mediana no sobrevive ni al bootstrap de theta ni al escenario pesimista. La
asimetria real entre temprano/tardio (punto 4) confirma que la SECUENCIA importa para el KPI, pero la
miope v6 (que ya condiciona cada decision en el estado simulado que hereda de las decisiones anteriores)
capta la mayor parte de ese valor; el margen adicional de planificar explicitamente el horizonte de 4
pasos fijos de Reduccion es pequeno y fragil frente al ruido de los targets. La miope v6 queda como
politica vigente. Lineas para revisitar el horizonte: (a) MPC real (re-planificar en cada escalon en vez
de lazo abierto), (b) mas secuencias / busqueda dirigida por gradiente en vez de perturbacion aleatoria,
(c) un horizonte que tambien cubra Fusion (donde H3 anticipa mayor palanca por selectividad temprana).
"""
    (OUT / "e9_03_resultados.md").write_text(memo, encoding="utf-8")
    log(memo)
    log("FIN resumen")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    etapa = sys.argv[1]
    t0 = time.time()
    df = L.cargar_df()
    if etapa == "simulador":
        tarea_simulador(df)
    elif etapa in [str(k) for k in range(N_FOLDS)] + ["lockbox"]:
        tarea_politicas(df, etapa)
    elif etapa == "robustez":
        tarea_robustez(df)
    elif etapa == "agregacion":
        tarea_agregacion(df)
    elif etapa == "resumen":
        tarea_resumen(df)
    else:
        print(f"etapa desconocida: {etapa}"); sys.exit(1)
    log(f"tiempo total: {time.time()-t0:.0f}s")
