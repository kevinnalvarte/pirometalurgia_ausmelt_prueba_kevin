"""E8-02 -- balance de energia por escalon con la masa de escoria v6 (Iteracion 8).

Ver `experimentos/v6/ITERACION_8_diseno.md`, `experimentos/v6/masa_v6.py` (docstring, estimador
`agregar_masa_v6`) y `experimentos/balances/bal_features.py` (funcion `agregar_balance_energia`,
`SUPUESTOS`). NO se edita `bal_features.py`: aqui se copia una version PARAMETRIZADA de
`agregar_balance_energia` (misma fisica, mismas constantes leidas de `bal.SUPUESTOS`/`bal.S()`) que
acepta la columna de masa/inventario previo como argumento, para generar dos juegos de columnas:

  b6_*  -- masa v3 (trazador CaO re-anclado, `masa_escoria_est_kg_prev` / `sn_inventario_escoria_est_kg_prev`)
  e6_*  -- masa v6 (trazador de matriz G + cierre fisico en Reduccion, `m6_masa_kg_prev` / `m6_sn_inv_kg_prev`)

Formulas (identicas a `bal_features.agregar_balance_energia`, ver su docstring para el detalle
termoquimico; aqui solo se resume el uso de la columna de masa/inventario parametrizada):

  q_combustion(t)   = f(GN, O2, aire; PCI_GN)                              -- no depende de la masa
  q_carbon_lanza(t) = f(carbon, O2 libre)                                  -- no depende de la masa
  q_carga(t)        = H_CARGA*feed_total + H_CARBON*carbon                 -- no depende de la masa
  q_gases(t)        = f(v_gas, T_horno_prev)                               -- no depende de la masa
  q_perdidas(t)     = PERDIDAS_MJ_MIN[fase] * delta_tiempo                 -- no depende de la masa
  q_neto(t)         = q_combustion + q_carbon_lanza - q_carga - q_gases - q_perdidas
  cap_bano(t)       = MASA_PREV(t) * CP_ESCORIA_KJ_KG_K / 1000             [MJ/K]   <- AQUI difieren b6/e6
  dT_teorico(t)     = q_neto(t) / cap_bano(t)                              [K]      <- AQUI difieren b6/e6
  q_reduccion_max   = Q_SN*(SN_INV_PREV(t) + feed_Sn) + Q_FE_HEMATITA*feed_Fe*FE_MINERAL   <- AQUI difieren
  margen_termico    = q_neto - q_reduccion_max                                             <- AQUI difieren
  (post-hoc) q_reaccion_aparente = q_neto - cap_bano*dT_medido
  (post-hoc) q_reaccion_trazador = Q_SN*SN_EXTRAIDO(t) + Q_FEO*FEO_EXTRAIDO(t) + Q_FE_HEMATITA*feed_Fe*FE_MINERAL  <- SN/FEO_EXTRAIDO difieren

Ademas se agregan features de ENERGIA ESPECIFICA (normalizadas por masa o por Sn disponible, para
que los coeficientes sean comparables entre escalones de distinto tamano de carga):
  e6_gn_nm3_por_t_escoria    = GN_nm3(t) / (m6_masa_kg_prev(t)/1000)         [Nm3 GN / t escoria]
                               (idem a `masa_v6.m6_gn_esp_nm3_t`, se re-expone con el prefijo e6_
                               para que quede junto al resto de las features termicas de este script)
  e6_q_neto_MJ_por_t         = e6_q_neto_disponible_MJ(t) / (m6_masa_kg_prev(t)/1000)  [MJ / t escoria]
  e6_q_neto_MJ_por_kg_sn_disp = e6_q_neto_disponible_MJ(t) / m6_sn_disp(t)             [MJ / kg Sn disponible]

Anti-fuga: todas las columnas de esta lista son DERIVED (A_t x S_prev) -- combinan una accion del
escalon t (GN/O2/aire/carbon/feed) con un estado_prev (masa, inventario de Sn) -- exactamente la
misma logica que `Cx_sn_v6`/`m6_gn_esp_nm3_t` en `masa_v6.py` o `b6_q_neto_por_min_MJ` en
`bal_features.py` (ambos clasificados DERIVED (A_t x S_prev), seguros como PALANCA). Se verifica con
`masa_v6.es_segura_v6` (mas una clasificacion local `CLASIFICACION_E802` para las nuevas) que ninguna
building-block sea LEAKAGE. OJO (instruccion de diseno): estas columnas mezclan accion_t y no pueden
usarse como ESTADO g(S) del PLM de OTRO target (contaminarian la nuisance model de la palanca que se
quiere evaluar y sesgarian el Double ML); en este script solo se usan como PALANCA lineal del propio
modelo de dT o de m6_ln_sn_dep/m6_ln_feo_ret (targets del propio escalon t), nunca dentro de una lista
`estado`.

Modelos: maquinaria PLM v5 (`modelo_predictivo_v5.ModeloPLMSignos`, estado `_S_DT`/`ESTADO_R_CURADO`,
`PALANCAS_V5`, `SIGNO_TEORICO_V5`), OOF GroupKFold(5) por Batch en DEV, lockbox solo reporte (imita
`modelo_predictivo_v5.tabla_validacion_v5`).

NOTA DE ENTORNO (2026-09-16): todo el trabajo (secciones 0-4) esta envuelto en `main()` bajo
`if __name__ == "__main__":`. HistGradientBoostingRegressor invoca joblib internamente; sin este
guard, el backend de multiprocessing de Windows (spawn) re-importa este script como si fuera el
modulo "__main__" en cada worker, re-ejecutando TODO desde cero (incluida la carga de datos y el
ajuste de modelos) de forma recursiva -- eso es lo que producia los procesos duplicados corriendo
este mismo archivo con el Python del sistema y una cascada de crashes. Con el guard, los workers
importan el modulo sin ejecutar nada a nivel superior.

Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/v6/e8_02_energia_v6.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RAIZ / "experimentos" / "balances"))

import bal_features as bal          # noqa: E402  (solo se leen SUPUESTOS/S()/constantes; no se edita)
import masa_v6                      # noqa: E402
import modelo_prescriptivo as mp    # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402

FASES = ["Fusión", "Reducción"]


def log(*a):
    print(*a, flush=True)


def csv_out(df: pd.DataFrame, nombre: str):
    p = HERE / nombre
    df.to_csv(p, index=False)
    log(f"  -> {p.name} ({df.shape[0]}x{df.shape[1]})")


def agregar_balance_energia_v6(df: pd.DataFrame, prefijo: str, col_masa_prev: str, col_sn_inv_prev: str,
                               col_sn_ext: str | None = None, col_feo_ext: str | None = None) -> pd.DataFrame:
    """Copia parametrizada de `bal_features.agregar_balance_energia` (misma fisica; NO se edita
    bal_features.py). `col_masa_prev`/`col_sn_inv_prev` seleccionan el inventario previo (masa v3 o
    v6) usado en la capacidad termica del bano y en la demanda endotermica maxima; `col_sn_ext`/
    `col_feo_ext` (opcionales) seleccionan la extraccion real usada solo en el diagnostico post-hoc
    del calor de reaccion segun trazador."""
    df = df.copy()
    n = {}
    gn = df["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0)
    o2 = df["volumen_o2_inyectado_lanza_escalon_nm3"].fillna(0) * bal.S("PUREZA_O2")
    aire = df["volumen_aire_inyectado_lanza_escalon_nm3"].fillna(0)
    o2_disp = o2 + 0.21 * aire
    x = (4.0 - 2.0 * o2_disp / gn.where(gn > 1)).clip(0, 1).fillna(0)
    n[f"{prefijo}frac_gas_a_co"] = x
    kmol_gn = gn / bal.VM_NM3_KMOL
    q_comb = kmol_gn * (-bal.DH_CH4_CO2 - (bal.DH_CH4_CO - bal.DH_CH4_CO2) * x) * (bal.S("PCI_GN_MJ_NM3") / (-bal.DH_CH4_CO2 / bal.VM_NM3_KMOL))
    n[f"{prefijo}q_combustion_MJ"] = q_comb
    exceso_nm3 = o2_disp - 2.0 * gn
    c_coal = df["feed_Carbon_kgh"].fillna(0) * bal.S("CARBONO_FIJO")
    c_quemado = np.minimum(exceso_nm3.clip(lower=0) / bal.VM_NM3_KMOL * 2 * bal.M_C, c_coal)
    n[f"{prefijo}c_quemado_o2_libre_kg"] = c_quemado
    n[f"{prefijo}q_carbon_lanza_MJ"] = c_quemado / bal.M_C * (-bal.DH_C_CO)
    n[f"{prefijo}q_carga_MJ"] = df["feed_total_kgh"].fillna(0) * bal.S("H_CARGA_MJ_KG") + df["feed_Carbon_kgh"].fillna(0) * bal.S("H_CARBON_MJ_KG")
    v_gas = 0.79 * aire + 3.0 * gn + exceso_nm3.clip(lower=0) + c_coal / bal.M_C * bal.VM_NM3_KMOL
    t_prev = df["temperatura_horno_celsius_prev"]
    n[f"{prefijo}v_gas_salida_nm3"] = v_gas
    n[f"{prefijo}q_gases_MJ"] = v_gas * bal.S("CP_GAS_KJ_NM3_K") * (t_prev - bal.S("T_REF_C")) / 1000
    perd = df["fase_proceso"].map(bal.S("PERDIDAS_MJ_MIN")).astype(float)
    n[f"{prefijo}q_perdidas_MJ"] = perd * df["delta_tiempo"]
    q_neto = q_comb + n[f"{prefijo}q_carbon_lanza_MJ"] - n[f"{prefijo}q_carga_MJ"] - n[f"{prefijo}q_gases_MJ"] - n[f"{prefijo}q_perdidas_MJ"]
    n[f"{prefijo}q_neto_disponible_MJ"] = q_neto
    n[f"{prefijo}q_neto_por_min_MJ"] = q_neto / df["delta_tiempo"]
    cap_bano = df[col_masa_prev] * bal.S("CP_ESCORIA_KJ_KG_K") / 1000
    n[f"{prefijo}capacidad_termica_bano_MJ_K"] = cap_bano
    n[f"{prefijo}dT_teorico_sin_reaccion"] = q_neto / cap_bano.where(cap_bano > 1)
    sn_disp = df[col_sn_inv_prev].fillna(0) + df["feed_Sn_kgf"].fillna(0)
    n[f"{prefijo}q_reduccion_max_MJ"] = bal.Q_SN_MJ_KG * sn_disp + bal.Q_FE_HEMATITA_MJ_KG * df["feed_Fe_kgh"].fillna(0) * bal.S("FE_MINERAL")
    n[f"{prefijo}margen_termico_MJ"] = q_neto - n[f"{prefijo}q_reduccion_max_MJ"]
    dT = df["d_temperatura_horno_celsius"]
    n[f"{prefijo}q_reaccion_aparente_MJ"] = q_neto - cap_bano * dT
    if col_sn_ext and col_feo_ext:
        n[f"{prefijo}q_reaccion_trazador_MJ"] = (bal.Q_SN_MJ_KG * df[col_sn_ext].clip(lower=0).fillna(0)
                                                 + bal.Q_FEO_MJ_KG * df[col_feo_ext].clip(lower=0).fillna(0)
                                                 + bal.Q_FE_HEMATITA_MJ_KG * df["feed_Fe_kgh"].fillna(0) * bal.S("FE_MINERAL"))
    return pd.concat([df, pd.DataFrame(n, index=df.index)], axis=1)


CLASIFICACION_E802: dict[str, str] = {
    **{f"b6_{c}": r for c, r in {
        "frac_gas_a_co": "DERIVED-ACTION", "q_combustion_MJ": "DERIVED-ACTION", "c_quemado_o2_libre_kg": "DERIVED-ACTION",
        "q_carbon_lanza_MJ": "DERIVED-ACTION", "q_carga_MJ": "DERIVED-ACTION", "v_gas_salida_nm3": "DERIVED-ACTION",
        "q_gases_MJ": "DERIVED (A_t x S_prev)", "q_perdidas_MJ": "DERIVED-ACTION",
        "q_neto_disponible_MJ": "DERIVED (A_t x S_prev)", "q_neto_por_min_MJ": "DERIVED (A_t x S_prev)",
        "capacidad_termica_bano_MJ_K": "STATE", "dT_teorico_sin_reaccion": "DERIVED (A_t x S_prev)",
        "q_reduccion_max_MJ": "DERIVED (A_t x S_prev)", "margen_termico_MJ": "DERIVED (A_t x S_prev)",
        "q_reaccion_aparente_MJ": "LEAKAGE", "q_reaccion_trazador_MJ": "LEAKAGE",
    }.items()},
    **{f"e6_{c}": r for c, r in {
        "frac_gas_a_co": "DERIVED-ACTION", "q_combustion_MJ": "DERIVED-ACTION", "c_quemado_o2_libre_kg": "DERIVED-ACTION",
        "q_carbon_lanza_MJ": "DERIVED-ACTION", "q_carga_MJ": "DERIVED-ACTION", "v_gas_salida_nm3": "DERIVED-ACTION",
        "q_gases_MJ": "DERIVED (A_t x S_prev)", "q_perdidas_MJ": "DERIVED-ACTION",
        "q_neto_disponible_MJ": "DERIVED (A_t x S_prev)", "q_neto_por_min_MJ": "DERIVED (A_t x S_prev)",
        "capacidad_termica_bano_MJ_K": "STATE", "dT_teorico_sin_reaccion": "DERIVED (A_t x S_prev)",
        "q_reduccion_max_MJ": "DERIVED (A_t x S_prev)", "margen_termico_MJ": "DERIVED (A_t x S_prev)",
        "q_reaccion_aparente_MJ": "LEAKAGE", "q_reaccion_trazador_MJ": "LEAKAGE",
        "gn_nm3_por_t_escoria": "DERIVED (A_t x S_prev)", "q_neto_MJ_por_t": "DERIVED (A_t x S_prev)",
        "q_neto_MJ_por_kg_sn_disp": "DERIVED (A_t x S_prev)",
    }.items()},
}


def es_segura_e802(nombre: str) -> bool:
    rol = CLASIFICACION_E802.get(nombre)
    if rol is not None:
        return not (rol.startswith("LEAKAGE") or "LEAKAGE" in rol)
    return masa_v6.es_segura_v6(nombre)


SEGURAS_E802 = [c for c, r in CLASIFICACION_E802.items() if not (r.startswith("LEAKAGE") or "LEAKAGE" in r)]

MAP_MASA_V6 = {"masa_escoria_est_kg_prev": "m6_masa_kg_prev"}
MAP_INV_V6 = {"masa_escoria_est_kg_prev": "m6_masa_kg_prev", "sn_inventario_escoria_est_kg_prev": "m6_sn_inv_kg_prev",
             "feo_inventario_escoria_est_kg_prev": "m6_feo_inv_kg_prev", "avance_reduccion_sn_prev": "m6_avance_prev"}


def sustituir(lista: list[str], mapa: dict[str, str]) -> list[str]:
    return [mapa.get(c, c) for c in lista]


FEATS_TERMICAS_E6 = ["e6_q_neto_por_min_MJ", "e6_dT_teorico_sin_reaccion", "e6_margen_termico_MJ"]
SIGNO_TERMICAS_E6 = {"e6_dT_teorico_sin_reaccion": +1, "e6_q_neto_por_min_MJ": +1, "e6_margen_termico_MJ": +1}
# Nota de diseno (ver docstring): e6_q_neto_por_min_MJ/e6_dT_teorico_sin_reaccion/e6_margen_termico_MJ
# combinan la accion del escalon t (GN, O2, aire, carbon) con el estado_prev (masa, T, Sn disponible).
# NO se agregan a la lista `estado` (serian A_t x S_prev dentro de g(S), lo que sesgaria el Double ML de
# la propia dT si compitieran con las palancas primitivas de la config (a)); se agregan solo como PALANCA
# LINEAL adicional del PLM de dT (que ya usa palancas GN/O2/aire/carbon primitivas en (a)/(b)).

FEATS_ENERGIA_ESP = ["e6_gn_nm3_por_t_escoria", "e6_q_neto_MJ_por_t", "e6_q_neto_MJ_por_kg_sn_disp"]
TARGET_DT = "d_temperatura_horno_celsius"
PARES = [("capacidad_termica_bano_MJ_K", "MJ/K"), ("dT_teorico_sin_reaccion", "K"), ("margen_termico_MJ", "MJ")]
# dT: mismo signo teorico +1 que tasa_gn_nm3_min (mas calor neto especifico -> mas subida de T, relacion
# directa e inequivoca). sn_dep/feo_ret: las features de energia especifica son una MEZCLA algebraica de
# GN + O2 + aire + carbon (q_neto ya suma/resta las cuatro), sin signo teorico univoco frente al agotamiento
# de Sn o la retencion de FeO -> se dejan libres (0) y se documenta como exploratorio.
SIGNO_ESPECIFICA_DT = {"e6_gn_nm3_por_t_escoria": +1, "e6_q_neto_MJ_por_t": +1, "e6_q_neto_MJ_por_kg_sn_disp": +1}


def _hgb(seed=SEED, max_iter=150):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=max_iter, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=seed)


def main():
    t_inicio = time.time()

    # =========================================================================
    # 0) DATOS: v3 base -> v4 -> v5 (targets/palancas/estado) -> masa v6 (m6_*)
    # =========================================================================
    log("=" * 90)
    log("0) CARGA DE DATOS")
    df = masa_v6.cargar_df_base()
    df = mp4.agregar_columnas_v4(df)
    df = mp5.agregar_columnas_v5(df)
    bal_batch = masa_v6.balance_batch(df)
    df6 = masa_v6.agregar_masa_v6(df, None, bal_batch)
    dev_b, lock_b = mp.split_dev_lockbox(df6)
    df6["set"] = np.where(df6["Batch"].isin(lock_b), "lockbox", "dev")
    log(f"df6: {df6.shape}, DEV batches={len(dev_b)}, lockbox batches={len(lock_b)}")

    # =========================================================================
    # 1) BALANCE DE ENERGIA PARAMETRIZADO (copia de bal_features.agregar_balance_energia)
    # =========================================================================
    log("=" * 90)
    log("1) FEATURES TERMICAS b6_* (masa v3) y e6_* (masa v6)")
    df6 = agregar_balance_energia_v6(df6, "b6_", "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev",
                                     "sn_extraido_est_kg", "feo_extraido_est_kg")
    df6 = agregar_balance_energia_v6(df6, "e6_", "m6_masa_kg_prev", "m6_sn_inv_kg_prev",
                                     "m6_sn_extraido_kg", "m6_feo_extraido_kg")

    # energia especifica (normalizada por masa / por Sn disponible)
    df6["e6_gn_nm3_por_t_escoria"] = df6["m6_gn_esp_nm3_t"]  # identico a masa_v6 (misma formula, mismo umbral)
    masa_t = df6["m6_masa_kg_prev"] / 1000.0
    df6["e6_q_neto_MJ_por_t"] = (df6["e6_q_neto_disponible_MJ"] / masa_t).where(masa_t.gt(0.5))
    df6["e6_q_neto_MJ_por_kg_sn_disp"] = (df6["e6_q_neto_disponible_MJ"] / df6["m6_sn_disp"]).where(df6["m6_sn_disp"].gt(200.0))

    # verificacion anti-fuga de las features termicas seguras (todas las no-LEAKAGE de ambos juegos)
    assert all(es_segura_e802(c) for c in SEGURAS_E802), "hay features termicas mal clasificadas"
    log(f"features termicas seguras (DERIVED/STATE): {len(SEGURAS_E802)}; LEAKAGE (solo diagnostico): "
        f"{len(CLASIFICACION_E802) - len(SEGURAS_E802)}")

    df_base = mp.dataset_base_modelo(df6)  # excluye primer escalon de batch (sin _prev)
    log(f"df_base (sin primer escalon de batch): {df_base.shape}")

    # ---- descriptiva comparada b6 vs e6 (capacidad termica, dT teorico, margen termico) ----
    filas_desc = []
    for fase in FASES:
        sub = df_base.loc[df_base["fase_proceso"] == fase]
        for base, unidad in PARES:
            for pref in ("b6_", "e6_"):
                col = f"{pref}{base}"
                s = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
                filas_desc.append({"fase": fase, "feature_base": base, "unidad": unidad, "version": pref.strip("_"),
                                   "columna": col, "n": int(len(s)), "media": float(s.mean()), "sd": float(s.std()),
                                   "p5": float(s.quantile(.05)), "mediana": float(s.median()), "p95": float(s.quantile(.95))})
            b = sub[f"b6_{base}"].replace([np.inf, -np.inf], np.nan)
            e = sub[f"e6_{base}"].replace([np.inf, -np.inf], np.nan)
            m = b.notna() & e.notna()
            rho = float(b[m].corr(e[m], method="spearman")) if m.sum() > 10 else np.nan
            r = float(b[m].corr(e[m])) if m.sum() > 10 else np.nan
            filas_desc.append({"fase": fase, "feature_base": base, "unidad": unidad, "version": "b6_vs_e6_corr",
                               "columna": f"b6_{base}~e6_{base}", "n": int(m.sum()), "media": np.nan, "sd": np.nan,
                               "p5": rho, "mediana": r, "p95": np.nan})
    tabla_desc_termica = pd.DataFrame(filas_desc)
    csv_out(tabla_desc_termica, "e8_02_features_termicas.csv")
    log(tabla_desc_termica.round(3).to_string())
    log(f"[{time.time()-t_inicio:.0f}s] seccion 1 lista")

    # =========================================================================
    # 2) MODELOS DE dT: (a) v5 tal cual, (b) v5 + masa v6 en estado, (c) b + termicas e6 como palanca,
    #    (d) referencia HGB estado+palancas de (c)
    # =========================================================================
    log("=" * 90)
    log("2) MODELOS DE dT (PLM v5, 4 configuraciones)")

    assert all(es_segura_e802(c) for c in FEATS_TERMICAS_E6)

    dev_b_, lock_b_ = dev_b, lock_b  # closures locales para evaluar_plm/evaluar_hgb

    def evaluar_plm(df_in, fase, estado, palancas, target, signos, n_splits=5, n_boot=150, seed=SEED):
        sub = df_in.loc[df_in["fase_proceso"] == fase]
        need = list(dict.fromkeys(estado + palancas + [target, "Batch"]))
        d_dev = sub.loc[sub["Batch"].isin(dev_b_), need].dropna().reset_index(drop=True)
        d_lb = sub.loc[sub["Batch"].isin(lock_b_), need].dropna().reset_index(drop=True)
        oof = np.full(len(d_dev), np.nan)
        for tr, te in GroupKFold(n_splits).split(d_dev[estado], d_dev[target], d_dev["Batch"]):
            mdl = mp5.ModeloPLMSignos(estado, palancas, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
            oof[te] = mdl.predict(d_dev.iloc[te])
        m_full = mp5.ModeloPLMSignos(estado, palancas, target, signos).fit(d_dev, n_boot=n_boot, seed=seed)
        p_lb = m_full.predict(d_lb) if len(d_lb) else np.array([])
        res = dict(n_estado=len(estado), n_palancas=len(palancas), n_dev=len(d_dev), n_lockbox=len(d_lb),
                  r2_oof=float(r2_score(d_dev[target], oof)), mae_oof=float(mean_absolute_error(d_dev[target], oof)),
                  r2_lockbox=float(r2_score(d_lb[target], p_lb)) if len(d_lb) else np.nan,
                  mae_lockbox=float(mean_absolute_error(d_lb[target], p_lb)) if len(d_lb) else np.nan,
                  sd_target_dev=float(d_dev[target].std()), sd_target_lockbox=float(d_lb[target].std()) if len(d_lb) else np.nan)
        return res, m_full

    def evaluar_hgb(df_in, fase, features, target, n_splits=5, seed=SEED):
        sub = df_in.loc[df_in["fase_proceso"] == fase]
        need = list(dict.fromkeys(features + [target, "Batch"]))
        d_dev = sub.loc[sub["Batch"].isin(dev_b_), need].dropna().reset_index(drop=True)
        d_lb = sub.loc[sub["Batch"].isin(lock_b_), need].dropna().reset_index(drop=True)
        oof = np.full(len(d_dev), np.nan)
        for tr, te in GroupKFold(n_splits).split(d_dev[features], d_dev[target], d_dev["Batch"]):
            mdl = _hgb(seed).fit(d_dev[features].iloc[tr], d_dev[target].iloc[tr])
            oof[te] = mdl.predict(d_dev[features].iloc[te])
        mdl_full = _hgb(seed).fit(d_dev[features], d_dev[target])
        p_lb = mdl_full.predict(d_lb[features]) if len(d_lb) else np.array([])
        return dict(n_dev=len(d_dev), n_lockbox=len(d_lb), r2_oof=float(r2_score(d_dev[target], oof)),
                   mae_oof=float(mean_absolute_error(d_dev[target], oof)),
                   r2_lockbox=float(r2_score(d_lb[target], p_lb)) if len(d_lb) else np.nan,
                   mae_lockbox=float(mean_absolute_error(d_lb[target], p_lb)) if len(d_lb) else np.nan)

    filas_dT, filas_theta = [], []
    for fase in FASES:
        estado_a = mp5.ESTADO_V5[(fase, "dT")]
        palancas_a = mp5.PALANCAS_V5[(fase, "dT")]
        signos_a = mp5.SIGNO_TEORICO_V5[(fase, "dT")]
        estado_b = sustituir(estado_a, MAP_MASA_V6)
        palancas_c = palancas_a + FEATS_TERMICAS_E6
        signos_c = dict(signos_a, **SIGNO_TERMICAS_E6)

        res_a, m_a = evaluar_plm(df_base, fase, estado_a, palancas_a, TARGET_DT, signos_a)
        filas_dT.append({"fase": fase, "config": "a_v5_tal_cual", **res_a})
        t_a = m_a.tabla_theta(signos_a).reset_index(); t_a.insert(0, "config", "a_v5_tal_cual"); t_a.insert(0, "fase", fase)
        filas_theta.append(t_a)

        res_b, m_b = evaluar_plm(df_base, fase, estado_b, palancas_a, TARGET_DT, signos_a)
        filas_dT.append({"fase": fase, "config": "b_masa_v6_en_estado", **res_b})
        t_b = m_b.tabla_theta(signos_a).reset_index(); t_b.insert(0, "config", "b_masa_v6_en_estado"); t_b.insert(0, "fase", fase)
        filas_theta.append(t_b)

        res_c, m_c = evaluar_plm(df_base, fase, estado_b, palancas_c, TARGET_DT, signos_c)
        filas_dT.append({"fase": fase, "config": "c_b_mas_termicas_e6", **res_c})
        t_c = m_c.tabla_theta(signos_c).reset_index(); t_c.insert(0, "config", "c_b_mas_termicas_e6"); t_c.insert(0, "fase", fase)
        filas_theta.append(t_c)

        feats_d = list(dict.fromkeys(estado_b + palancas_c))
        res_d = evaluar_hgb(df_base, fase, feats_d, TARGET_DT)
        filas_dT.append({"fase": fase, "config": "d_HGB_estado_palancas_c", "n_estado": len(estado_b),
                         "n_palancas": len(palancas_c), **res_d})

        log(f"  {fase}: a={res_a['r2_oof']:.3f}/{res_a['r2_lockbox']:.3f}  b={res_b['r2_oof']:.3f}/{res_b['r2_lockbox']:.3f}  "
            f"c={res_c['r2_oof']:.3f}/{res_c['r2_lockbox']:.3f}  d={res_d['r2_oof']:.3f}/{res_d['r2_lockbox']:.3f}")

    tabla_dT = pd.DataFrame(filas_dT)
    tabla_theta_dT = pd.concat(filas_theta, ignore_index=True)
    csv_out(tabla_dT, "e8_02_modelos_dT.csv")
    csv_out(tabla_theta_dT, "e8_02_theta_dT.csv")
    log(f"[{time.time()-t_inicio:.0f}s] seccion 2 lista")

    # =========================================================================
    # 3) ENERGIA ESPECIFICA COMO PALANCA (en vez de tasas absolutas): dT y m6_ln_sn_dep/m6_ln_feo_ret (R)
    # =========================================================================
    log("=" * 90)
    log("3) ENERGIA ESPECIFICA COMO PALANCA")

    estado_r_curado_v6 = sustituir(mp5.ESTADO_R_CURADO, MAP_INV_V6)
    assert all(es_segura_e802(c) if c.startswith(("e6_", "b6_", "m6_")) else masa_v6.es_segura_v6(c) for c in estado_r_curado_v6)
    assert all(es_segura_e802(c) for c in FEATS_ENERGIA_ESP)

    # variantes de palanca por target: sustituyen "tasa_gn_nm3_min" (unica tasa absoluta de GN) por cada
    # feature de energia especifica; el resto de la palanca (O2, aire, carbon) se deja igual (son ya tasas,
    # no dependen de la masa).
    configs_target_r = {
        "dT": dict(estado=sustituir(mp5._S_DT, MAP_MASA_V6), target=TARGET_DT,
                  base=mp5.PALANCAS_V5[("Reducción", "dT")],
                  signo_base=mp5.SIGNO_TEORICO_V5[("Reducción", "dT")], signo_especifica=0),
        "m6_ln_sn_dep": dict(estado=estado_r_curado_v6, target="m6_ln_sn_dep",
                            base=["Cx_sn_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                            signo_base={"Cx_sn_v6": 1, "tasa_gn_nm3_min": 1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": 0},
                            signo_especifica=0),
        "m6_ln_feo_ret": dict(estado=estado_r_curado_v6, target="m6_ln_feo_ret",
                             base=["tasa_feed_Carbon_kg_min", "Cx_av_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                             signo_base={"tasa_feed_Carbon_kg_min": -1, "Cx_av_v6": -1, "tasa_gn_nm3_min": -1,
                                        "exceso_o2_combustion_pct": 1, "tasa_aire_nm3_min": 1},
                             signo_especifica=0),
    }

    filas_esp = []
    for clave, cfg in configs_target_r.items():
        fase = "Reducción"
        estado, target, base, signo_base = cfg["estado"], cfg["target"], cfg["base"], cfg["signo_base"]
        res_base, _ = evaluar_plm(df_base, fase, estado, base, target, signo_base, n_boot=0)
        filas_esp.append({"target": clave, "variante": "base_tasa_absoluta", "palancas": ",".join(base), **res_base})
        for feat_esp in FEATS_ENERGIA_ESP:
            palancas_v = [feat_esp if a == "tasa_gn_nm3_min" else a for a in base]
            signo_v = dict(signo_base)
            signo_v.pop("tasa_gn_nm3_min", None)
            signo_v[feat_esp] = SIGNO_ESPECIFICA_DT[feat_esp] if clave == "dT" else cfg["signo_especifica"]
            res_v, _ = evaluar_plm(df_base, fase, estado, palancas_v, target, signo_v, n_boot=0)
            filas_esp.append({"target": clave, "variante": f"gn_a_{feat_esp}", "palancas": ",".join(palancas_v), **res_v})
        log(f"  {clave}: " + "  ".join(f"{r['variante']}={r['r2_oof']:.3f}/{r['r2_lockbox']:.3f}"
                                       for r in [f for f in filas_esp if f["target"] == clave]))

    tabla_esp = pd.DataFrame(filas_esp)
    csv_out(tabla_esp, "e8_02_energia_especifica.csv")
    log(f"[{time.time()-t_inicio:.0f}s] seccion 3 lista")

    # =========================================================================
    # 4) CALOR DE REACCION APARENTE vs POR INVENTARIOS (v3 vs v6, post-hoc, diagnostico)
    # =========================================================================
    log("=" * 90)
    log("4) CIERRE DE ENERGIA: aparente vs trazador (v3 y v6)")

    from scipy import stats

    filas_cierre = []
    for pref, etiqueta in [("b6_", "masa_v3"), ("e6_", "masa_v6")]:
        ap_col, tr_col = f"{pref}q_reaccion_aparente_MJ", f"{pref}q_reaccion_trazador_MJ"
        for fase in FASES:
            sub = df_base.loc[(df_base["fase_proceso"] == fase) & (df_base["set"] == "dev"),
                              [ap_col, tr_col, "delta_tiempo"]].replace([np.inf, -np.inf], np.nan).dropna()
            ap, tr, dt = sub[ap_col].to_numpy(), sub[tr_col].to_numpy(), sub["delta_tiempo"].to_numpy()
            bias = ap - tr
            rho, p_rho = stats.spearmanr(ap, tr)
            r, p_r = stats.pearsonr(ap, tr)
            # correccion de perdidas (MJ/min) que anula el sesgo medio: nuevo_sesgo = sesgo - extra_perdida*dt
            extra_perdida = float(np.mean(bias) / np.mean(dt)) if np.mean(dt) > 0 else np.nan
            perd_actual = bal.S("PERDIDAS_MJ_MIN")[fase]
            filas_cierre.append({"version": etiqueta, "fase": fase, "n": int(len(sub)),
                                 "spearman_aparente_trazador": float(rho), "p_spearman": float(p_rho),
                                 "pearson_aparente_trazador": float(r), "p_pearson": float(p_r),
                                 "sesgo_medio_MJ": float(np.mean(bias)), "sesgo_mediana_MJ": float(np.median(bias)),
                                 "sesgo_sd_MJ": float(np.std(bias)),
                                 "perdidas_actuales_MJ_min": float(perd_actual),
                                 "extra_perdida_MJ_min_para_sesgo_cero": extra_perdida,
                                 "perdidas_ajustadas_MJ_min": float(perd_actual + extra_perdida)})
    tabla_cierre = pd.DataFrame(filas_cierre)
    csv_out(tabla_cierre, "e8_02_cierre_energia.csv")
    log(tabla_cierre.round(4).to_string())
    log(f"[{time.time()-t_inicio:.0f}s] seccion 4 lista")

    log("=" * 90)
    log(f"TODO listo, total {time.time()-t_inicio:.0f}s")


if __name__ == "__main__":
    main()
