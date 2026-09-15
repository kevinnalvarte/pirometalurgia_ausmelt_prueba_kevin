"""Modelo prescriptivo v5 de Lingo Smelter (iteracion 7, 2026-09-14/15): objetivo por escalon en
ESCALA LOGARITMICA anclado en los targets ganadores (iteracion 5) y en el KPI refinado (iteracion 6).

Extiende v4 (`modelo_predictivo_v4.py`: PLM parcialmente lineal por fase, optimizador con soporte
historico, politica cross-fitted) sin reemplazarlo. Que cambia y por que:

1. **Targets por escalon en escala log (adimensionales; la masa del trazador CaO se cancela):**
       v5_ln_sn_dep  = ln(Sn_disp / Sn_inv[t])     agotamiento del Sn de la escoria (cinetica de 1er orden)
                                                    Sn_disp = Sn_inv[t-1] + Sn alimentado en t
       v5_ln_feo_ret = ln(FeO_inv[t] / FeO_inv[t-1]) retencion de FeO (<= 0 si se metaliza Fe; solo Reduccion)
       v5_d_lnirf    = ln IRF[t] - ln IRF[t-1]       cambio de la matriz de escoria, IRF = %FeO/(%SiO2+%Al2O3+%CaO)
   La suma por batch es telescopica: sum ln_sn_dep = ln(Sn_ini/Sn_fin), sum ln_feo_ret = ln(FeO_fin/FeO_ini).
   El objetivo v4 en kg (Sn - 0.6 FeO) NO correlaciona con el rendimiento del batch (rho 0.035); el SDI
   = ln_sn_dep + w ln_feo_ret si (rho 0.32 DEV) y sus componentes son las palancas de la matriz de escoria
   (ln IRF en Reduccion, rho 0.30/0.38 lockbox) (experimentos/search_targets, hallazgos 16).

2. **Objetivo de Reduccion anclado en el KPI:**
       J_R = K_SDI * [ ln_sn_dep_pred + w * ln_feo_ret_pred ] - costos - w_T*(T fuera de ventana)
             - w_U * ancho_IC - w_S * (1 - soporte)
   w se ancla en la regresion de batch del KPI refinado sobre las dos componentes agregadas (E7-01) y
   K_SDI convierte unidades de SDI a kg de Sn a metal (pp de KPI por unidad de SDI x kg de Sn de salida
   por pp), para que costos y soporte queden en las mismas unidades que v4.

3. **Palancas con estructura cinetica en escala log (E7-01/E7-03):**
       v5_dosis_C_sn  = C_kg / Sn_disp            kg C por kg Sn disponible (estequiometrico 0.2024)
       v5_exceso_C_pos = max(C_kg - 0.2024 Sn_disp, 0)  carbon sobrante para FeO + C -> Fe + CO
       v5_C_x_avance  = C_kg * avance_prev        selectividad decreciente con el avance
   mas GN, exceso de O2 de la lanza y aire. Forma funcional por palanca segun E7-03.

4. **Fusion (E7-02):** objetivo anclado en regresiones de batch con el KPI refinado (carbon por Sn cargado,
   temperatura) mas conservacion de la escoria (Sn y matriz) via PLM de escalon.

5. **Evidencia de valor (E7-05/06):** politica cross-fitted para los 362 batches y adherencia vs KPI refinado
   (DEV / lockbox / total / batch siguiente), placebo, emparejamiento por estado y cadena theta -> agregado -> KPI.

La CONFIGURACION (CONFIG_V5) se fija con los resultados de experimentos/v5/e7_01..04 y queda documentada en
candidate_win_model.md.

Uso tipico:
    import modelo_predictivo_v5 as mp5
    df = mp5.construir_dataset_modelo_v5("Datos Lingo smelter fase II.xlsx")
    sistema = mp5.entrenar_sistema_v5(df)
    print(mp5.tabla_validacion_v5(df)); print(mp5.tabla_efectos_control_v5(sistema))
    rep = mp5.reporte_recomendaciones_batch_v5(sistema, df, "AP0350")
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import feature_engineering as fe
import modelo_prescriptivo as mp
import modelo_predictivo_v3 as mp3
import modelo_predictivo_v4 as mp4

FASES = mp.FASES
SEED = 42
C_ESTEQ_POR_SN = 2 * 12.011 / 118.71   # 0.2024 kg C / kg Sn (SnO2 + 2C -> Sn + 2CO)
UMBRAL_SN_KG, UMBRAL_FEO_KG, UMBRAL_SN_DISP_DOSIS = 20.0, 100.0, 200.0
SN_SALIDA_BATCH_KG = mp4.SN_SALIDA_BATCH_KG      # ~51 250 kg de Sn de salida por batch (metal + dross + polvo)
KG_SN_POR_PP_KPI = SN_SALIDA_BATCH_KG / 100.0   # 1 pp de recuperacion = ~512 kg de Sn a metal


# =============================================================================
# 0) Targets y palancas v5 (misma definicion que experimentos/v5/v5_lib.py)
# =============================================================================

def agregar_targets_v5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    sn_prev, sn = df["sn_inventario_escoria_est_kg_prev"], df["sn_inventario_escoria_est_kg"]
    feo_prev, feo = df["feo_inventario_escoria_est_kg_prev"], df["feo_inventario_escoria_est_kg"]
    sn_disp = sn_prev + df["feed_Sn_kgf"].fillna(0)
    ok = sn_prev.gt(UMBRAL_SN_KG) & sn.gt(UMBRAL_SN_KG) & feo_prev.gt(UMBRAL_FEO_KG) & feo.gt(UMBRAL_FEO_KG) & ~df["es_primer_escalon_batch"]
    red = df["fase_proceso"].eq("Reducción")
    n = {"v5_sn_disp": sn_disp, "v5_valido": ok}
    n["v5_ln_sn_dep"] = np.log(sn_disp / sn).where(ok & sn_disp.gt(UMBRAL_SN_KG))
    n["v5_ln_feo_ret"] = np.log(feo / feo_prev).where(ok & red)
    n["v5_sdi_w10"] = n["v5_ln_sn_dep"] + 10.0 * n["v5_ln_feo_ret"]   # SDI de la iteracion 5 (referencia)
    irf = df["ley_feo_escoria_pct"] / (df["ley_sio2_escoria_pct"] + df["ley_al2o3_escoria_pct"] + df["ley_cao_escoria_pct"])
    lnirf = np.log(irf.where(irf > 0))
    n["v5_lnirf"] = lnirf
    lnirf_prev = lnirf.groupby(df["Batch"]).shift(1)
    n["v5_lnirf_prev"] = lnirf_prev.where(~df["es_primer_escalon_batch"])
    n["v5_d_lnirf"] = (lnirf - lnirf_prev).where(~df["es_primer_escalon_batch"])
    return pd.concat([df, pd.DataFrame(n, index=df.index)], axis=1)


def agregar_palancas_v5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c_kg = df["feed_Carbon_kgh"].fillna(0)
    sn_disp = df["v5_sn_disp"]
    dosis = (c_kg / sn_disp).where(sn_disp.gt(UMBRAL_SN_DISP_DOSIS))
    df["v5_dosis_C_sn"] = dosis
    df["v5_ln_dosis_C"] = np.log(dosis.where(dosis > 0))
    df["v5_exceso_C_kg"] = c_kg - C_ESTEQ_POR_SN * sn_disp
    df["v5_exceso_C_pos"] = df["v5_exceso_C_kg"].clip(lower=0)
    df["v5_C_x_avance"] = c_kg * df["avance_reduccion_sn_prev"]
    masa_prev = df["masa_escoria_est_kg_prev"]
    df["v5_gn_esp_nm3_t"] = (df["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0) / (masa_prev / 1000)).where(masa_prev.gt(1000))
    return df


def _agregar_palancas_v5_fila(fila: dict) -> dict:
    """Version por fila para el simulador (misma formula que agregar_palancas_v5)."""
    c_kg = float(fila.get("feed_Carbon_kgh", 0.0) or 0.0)
    sn_prev = float(fila.get("sn_inventario_escoria_est_kg_prev", np.nan))
    feed_sn = float(fila.get("feed_Sn_kgf", 0.0) or 0.0)
    sn_disp = sn_prev + feed_sn
    fila["v5_sn_disp"] = sn_disp
    dosis = c_kg / sn_disp if sn_disp > UMBRAL_SN_DISP_DOSIS else np.nan
    fila["v5_dosis_C_sn"] = dosis
    fila["v5_ln_dosis_C"] = np.log(dosis) if (pd.notna(dosis) and dosis > 0) else np.nan
    fila["v5_exceso_C_kg"] = c_kg - C_ESTEQ_POR_SN * sn_disp
    fila["v5_exceso_C_pos"] = max(fila["v5_exceso_C_kg"], 0.0)
    fila["v5_C_x_avance"] = c_kg * float(fila.get("avance_reduccion_sn_prev", np.nan))
    masa_prev = float(fila.get("masa_escoria_est_kg_prev", np.nan))
    gn_nm3 = float(fila.get("volumen_gas_natural_inyectado_lanza_escalon_nm3", 0.0) or 0.0)
    fila["v5_gn_esp_nm3_t"] = gn_nm3 / (masa_prev / 1000) if masa_prev > 1000 else np.nan
    # cuadraticos centrados (forma funcional segun E7-03); se centran en CENTROS_CUADRATICOS
    for a, c in CENTROS_CUADRATICOS.items():
        v = fila.get(a, np.nan)
        fila[f"{a}__sq"] = (v - c) ** 2 if pd.notna(v) else np.nan
    return fila


def agregar_cuadraticos_v5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a, c in CENTROS_CUADRATICOS.items():
        if a in df.columns:
            df[f"{a}__sq"] = (df[a] - c) ** 2
    return df


def construir_dataset_modelo_v5(ruta_excel: str, hoja: str = "Datos") -> pd.DataFrame:
    """Dataset v4 (v3 + Cx) + targets y palancas v5 (+ cuadraticos centrados)."""
    return agregar_columnas_v5(mp4.construir_dataset_modelo_v4(ruta_excel, hoja))


def agregar_columnas_v5(df: pd.DataFrame) -> pd.DataFrame:
    if "Cx_sn_v4" not in df.columns:
        df = mp4.agregar_columnas_v4(df)
    return agregar_cuadraticos_v5(agregar_palancas_v5(agregar_targets_v5(df)))


CLASIFICACION_V5: dict[str, str] = {
    "v5_sn_disp": "DERIVED (A_t x S_prev)", "v5_valido": "DIAG",
    "v5_ln_sn_dep": "TARGET", "v5_ln_feo_ret": "TARGET", "v5_sdi_w10": "TARGET", "v5_lnirf": "TARGET", "v5_d_lnirf": "TARGET", "v5_lnirf_prev": "STATE",
    "v5_dosis_C_sn": "DERIVED (A_t x S_prev)", "v5_ln_dosis_C": "DERIVED (A_t x S_prev)", "v5_exceso_C_kg": "DERIVED (A_t x S_prev)",
    "v5_exceso_C_pos": "DERIVED (A_t x S_prev)", "v5_C_x_avance": "DERIVED (A_t x S_prev)", "v5_gn_esp_nm3_t": "DERIVED (A_t x S_prev)",
}


def es_segura_v5(nombre: str) -> bool:
    if fe.es_feature_segura_para_prescripcion(nombre) or nombre in ("Cx_sn_v4", "Cx_av_v4"):
        return True
    if nombre.endswith("__sq"):
        return es_segura_v5(nombre[:-4])
    return CLASIFICACION_V5.get(nombre) in ("STATE", "DERIVED (A_t x S_prev)")


# =============================================================================
# 1) CONFIGURACION v5 (se fija con experimentos/v5/e7_01..04; ver candidate_win_model.md)
# =============================================================================

# Centros de los terminos cuadraticos (media DEV de la palanca); vacio = sin cuadraticos
CENTROS_CUADRATICOS: dict[str, float] = {}

ESTADO_R_CURADO = ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev",
                   "indice_irf_prev", "basicidad_B2_prev", "log_ratio_sn_feo_prev",
                   "temperatura_horno_celsius_prev", "grad_temperatura_horno_celsius_prev", "termocupla_media_celsius_prev",
                   "posicion_vertical_lanza_mm_prev", "tiro_horno_pct_prev",
                   "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev", "feo_inventario_escoria_est_kg_prev",
                   "avance_reduccion_sn_prev", "orden_escalon_fase", "tiempo_fase", "espesor_ladrillo_norm_mm",
                   "ley_sn_carga_batch_pct"]
ESTADO_F_CURADO = ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev",
                   "indice_irf_prev", "basicidad_B2_prev",
                   "temperatura_horno_celsius_prev", "grad_temperatura_horno_celsius_prev", "termocupla_media_celsius_prev",
                   "posicion_vertical_lanza_mm_prev", "tiro_horno_pct_prev",
                   "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev", "feo_inventario_escoria_est_kg_prev",
                   "cum_feed_Sn_kg_prev", "cum_feed_CaO_kg_prev", "orden_escalon_fase", "tiempo_fase", "espesor_ladrillo_norm_mm",
                   "ley_sn_conc_batch_pct", "ley_sn_carga_batch_pct",
                   "feed_Sn_kgf", "tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min"]
_S_DT = ["temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "masa_escoria_est_kg_prev", "orden_escalon_fase",
         "grad_temperatura_horno_celsius_prev", "espesor_ladrillo_norm_mm"]

ESTADO_R_FULL = list(dict.fromkeys(mp3.STATE_V3["Reducción"] + mp3.CONTEXT_V3["Reducción"]))   # 37: identifica el carbon (E7-01)

TARGETS_V5: dict[str, dict[str, str]] = {
    "Reducción": {"sn_dep": "v5_ln_sn_dep", "feo_ret": "v5_ln_feo_ret", "dT": "d_temperatura_horno_celsius"},
    "Fusión": {"sn_dep": "v5_ln_sn_dep", "dT": "d_temperatura_horno_celsius"},
}
ESTADO_V5: dict[tuple[str, str], list[str]] = {
    ("Reducción", "sn_dep"): ESTADO_R_FULL, ("Reducción", "feo_ret"): ESTADO_R_FULL, ("Reducción", "dT"): _S_DT,
    ("Fusión", "sn_dep"): ESTADO_F_CURADO, ("Fusión", "d_lnirf"): ESTADO_F_CURADO,
    ("Fusión", "dT"): _S_DT + ["tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min"],
}
# Parametrizacion elegida (E7-01, solo DEV): estado FULL; Sn = cinetica v4 (Cx_sn = C x Sn disponible: +0.18 sd/sd, IC [0.01, 0.28]);
# FeO = carbon + carbon x avance (selectividad, teoria) + GN (-0.021*, unico robusto) + exceso O2 + aire
PALANCAS_V5: dict[tuple[str, str], list[str]] = {
    ("Reducción", "sn_dep"): ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    ("Reducción", "feo_ret"): ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    ("Reducción", "dT"): ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"],
    ("Fusión", "sn_dep"): ["relacion_C_Sn_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    ("Fusión", "d_lnirf"): ["relacion_C_Sn_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    ("Fusión", "dT"): ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"],
}
SIGNO_TEORICO_V5: dict[tuple[str, str], dict[str, int]] = {
    ("Reducción", "sn_dep"): {"Cx_sn_v4": +1, "v5_dosis_C_sn": +1, "tasa_feed_Carbon_kg_min": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": 0},
    ("Reducción", "feo_ret"): {"tasa_feed_Carbon_kg_min": -1, "Cx_av_v4": -1, "v5_exceso_C_pos": -1, "v5_C_x_avance": -1, "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1},
    ("Reducción", "dT"): {"tasa_gn_nm3_min": +1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": -1, "tasa_feed_Carbon_kg_min": 0},
    ("Fusión", "sn_dep"): {"relacion_C_Sn_carga": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": 0},
    ("Fusión", "d_lnirf"): {"relacion_C_Sn_carga": 0, "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": 0},
    ("Fusión", "dT"): {"tasa_gn_nm3_min": +1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": -1, "tasa_feed_Carbon_kg_min": 0},
}
ACCIONES_OPTIMIZABLES_V5 = {f: list(v) for f, v in mp4.ACCIONES_OPTIMIZABLES_V4.items()}
ACCIONES_PRIMITIVAS_V5 = {f: list(v) for f, v in mp4.ACCIONES_PRIMITIVAS_V4.items()}

# Objetivo (anclajes; se fijan con E7-01/E7-02)
CONFIG_V5: dict = dict(
    # --- Reduccion (E7-01, OLS HC3 de recuperacion_refinada_pct sobre sum ln_sn_dep y sum ln_feo_ret, DEV n=285, controles + tendencia):
    #     coef ln_sn_dep = +2.124 pp/unidad (p 0.001), coef ln_feo_ret = +12.83 pp/unidad (p < 0.001) -> w = 6.04 [IC boot 3.8, 13.6]
    w_sdi=6.04,                    # peso de ln_feo_ret frente a ln_sn_dep (anclado en el KPI refinado)
    pp_kpi_por_unidad_sdi=2.124,   # pp de recuperacion refinada por unidad de ln_sn_dep (escala de J en kg: x 512 kg/pp)
    # --- Fusion (E7-02): ningun termino de C/Sn ni de T es defendible con el KPI refinado (dross y polvo se cancelan);
    #     conservacion del Sn: KPI ~ -0.45 pp por unidad de sum ln_sn_dep de Fusion (DEV p 0.08; total -0.59 p 0.014), controlando el heel
    beta_fusion={"C_por_Sn": {"pp_por_sd": 0.0, "sd_batch": 0.018, "media_batch": 0.272},
                 "T_media": {"pp_por_sd": 0.0, "sd_batch": 75.1, "media_batch": 1068.0}},
    pp_kpi_por_unidad_sn_dep_F=-0.45,  # conservacion del Sn en Fusion (negativo: extraer Sn en Fusion baja la recuperacion)
    pp_kpi_por_unidad_d_lnirf_F=0.0,   # Delta ln IRF en Fusion: nulo controlando el heel (E7-02) -> desactivado
    n_escalones_fusion=7,
)
PESOS_V5_DEFAULT = dict(
    costo_carbon_kgSn_por_kg=0.05, costo_gn_kgSn_por_nm3=0.03, costo_o2_kgSn_por_nm3=0.01,
    w_temp=2.0, w_incertidumbre=0.25, w_soporte=2000.0,
)
PERCENTIL_SOPORTE_MINIMO = 0.10
FEATURES_SOPORTE_V5 = {
    "Fusión": ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev",
               "basicidad_B2_prev", "temperatura_horno_celsius_prev", "orden_escalon_fase", "feed_Sn_kgf", "tasa_feed_total_kg_min",
               "tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
    "Reducción": ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev",
                  "avance_reduccion_sn_prev", "temperatura_horno_celsius_prev", "orden_escalon_fase",
                  "tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
}


def _verificar_seguridad() -> None:
    for k, feats in list(ESTADO_V5.items()) + list(PALANCAS_V5.items()):
        inseg = [f for f in feats if not es_segura_v5(f)]
        if inseg:
            raise AssertionError(f"v5 {k}: features inseguras {inseg}")


_verificar_seguridad()


def kg_sn_por_unidad_sdi() -> float:
    """Conversion SDI (adimensional) -> kg de Sn a metal: pp de KPI por unidad x kg por pp."""
    return CONFIG_V5["pp_kpi_por_unidad_sdi"] * KG_SN_POR_PP_KPI


# =============================================================================
# 1b) PLM con restricciones de signo por teoria (E7-03/E7-05)
# =============================================================================

class ModeloPLMSignos(mp4.ModeloPLM):
    """PLM v4 cuyos theta se estiman por minimos cuadrados ACOTADOS con el signo que dicta la teoria
    (SIGNO_TEORICO_V5: +1 -> theta >= 0, -1 -> theta <= 0, 0 -> libre). Es la unica monotonia impuesta
    en v5 y solo sobre el efecto lineal de las palancas de CONTROL: un reductor (carbon, GN) no puede
    aumentar la retencion de FeO ni reducir el agotamiento de Sn; una lanza mas oxidante (exceso de O2)
    no puede aumentar el agotamiento de Sn ni reducir la retencion de FeO. Si el dato contradice el signo
    (IC que cruza 0), el theta colapsa a 0 = "sin efecto identificado" en vez de un efecto espurio de signo
    fisicamente imposible (p. ej. carbon +0.0004 sobre ln_feo_ret, E7-05 provisional). El estado g(S)
    sigue sin restricciones. IC95 % por bootstrap cluster (misma restriccion en cada replica)."""

    def __init__(self, estado, palancas, target, signos: dict | None = None):
        super().__init__(estado, palancas, target)
        self.signos = dict(signos or {})

    def _bounds(self):
        lo = np.array([0.0 if self.signos.get(a, 0) > 0 else -np.inf for a in self.palancas])
        hi = np.array([0.0 if self.signos.get(a, 0) < 0 else np.inf for a in self.palancas])
        return lo, hi

    def _lstsq(self, A, y):
        from scipy.optimize import lsq_linear
        lo, hi = self._bounds()
        if np.all(np.isinf(lo)) and np.all(np.isinf(hi)):
            return np.linalg.lstsq(A, y, rcond=None)[0]
        return lsq_linear(A, y, bounds=(lo, hi), lsmr_tol="auto").x

    def fit(self, d: pd.DataFrame, n_splits: int = 5, n_boot: int = 200, seed: int = SEED):
        S, A, y, grupos = self.estado, self.palancas, self.target, d["Batch"].to_numpy()
        cols = [y] + A
        oof = pd.DataFrame(np.nan, index=d.index, columns=cols)
        for tr, te in GroupKFold(n_splits).split(d[S], d[y], grupos):
            for c in cols:
                mdl = mp4._hgb_nuisance(seed).fit(d[S].iloc[tr], d[c].iloc[tr])
                oof.iloc[te, oof.columns.get_loc(c)] = mdl.predict(d[S].iloc[te])
        y_res = d[y].to_numpy() - oof[y].to_numpy()
        A_res = np.column_stack([d[a].to_numpy() - oof[a].to_numpy() for a in A])
        self.theta = self._lstsq(A_res, y_res)
        self.res_oof = y_res - A_res @ self.theta
        self.r2_oof_interno = float(r2_score(d[y], d[y].to_numpy() - self.res_oof))
        rng = np.random.default_rng(seed)
        uniq = np.unique(grupos)
        idx_by = {b: np.where(grupos == b)[0] for b in uniq}
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
            boots.append(self._lstsq(A_res[idx], y_res[idx]))
        if boots:
            boots = np.array(boots)
            self.ci_lo, self.ci_hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        else:
            self.ci_lo, self.ci_hi = np.full(len(A), np.nan), np.full(len(A), np.nan)
        self.sd_palancas = d[A].std().to_numpy()
        self.g = mp4._hgb_nuisance(seed).fit(d[S], d[y])
        self.m = {a: mp4._hgb_nuisance(seed).fit(d[S], d[a]) for a in A}
        self.n_train = len(d)
        return self


    def tabla_theta(self, signo_teorico: dict | None = None) -> pd.DataFrame:
        t = super().tabla_theta(signo_teorico)
        tol = 1e-9
        t["en_cota"] = (np.abs(self.theta) < tol)
        t["significativo"] = (t["ci_lo"] > tol) | (t["ci_hi"] < -tol)
        if "concuerda" in t.columns:
            t["concuerda"] = t["concuerda"].astype(object)
            t.loc[t["en_cota"], "concuerda"] = np.nan
        return t


def _nuevo_plm(fase: str, clave: str, S: list[str], A: list[str], target: str):
    return ModeloPLMSignos(S, A, target, SIGNO_TEORICO_V5.get((fase, clave)))


# =============================================================================
# 2) Sistema v5
# =============================================================================

@dataclass
class SistemaPrescriptivoV5:
    modelos: dict = field(default_factory=dict)          # (fase, clave) -> mp4.ModeloPLM
    soporte: dict = field(default_factory=dict)
    soporte_minimo: dict = field(default_factory=dict)
    limites_accion: dict = field(default_factory=dict)
    ventana_temperatura: dict = field(default_factory=dict)
    batches_dev: set = field(default_factory=set)
    batches_lockbox: set = field(default_factory=set)
    config: dict = field(default_factory=lambda: dict(CONFIG_V5))


def _preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df_base = mp.dataset_base_modelo(df)
    if "v5_ln_sn_dep" not in df_base.columns or "v5_dosis_C_sn" not in df_base.columns:
        df_base = agregar_columnas_v5(df_base)
    df_base = agregar_cuadraticos_v5(df_base)
    return df_base


def entrenar_sistema_v5(df: pd.DataFrame, pct_lockbox: float = mp.N_LOCKBOX_PCT, n_boot: int = 200, seed: int = SEED,
                        config: dict | None = None) -> SistemaPrescriptivoV5:
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df, pct_lockbox)
    sistema = SistemaPrescriptivoV5(batches_dev=batches_dev, batches_lockbox=batches_lockbox, config=dict(CONFIG_V5, **(config or {})))
    for fase in FASES:
        sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        for clave, target in TARGETS_V5[fase].items():
            S, A = ESTADO_V5[(fase, clave)], PALANCAS_V5[(fase, clave)]
            d = sub.dropna(subset=list(dict.fromkeys(S + A + [target]))).reset_index(drop=True)
            sistema.modelos[(fase, clave)] = _nuevo_plm(fase, clave, S, A, target).fit(d, n_boot=n_boot, seed=seed)
        feats_sop = FEATURES_SOPORTE_V5[fase]
        sub_sop = sub.dropna(subset=feats_sop)
        sistema.soporte[fase] = mp._construir_soporte_historico(sub_sop, feats_sop)
        scores = mp3._support_scores_lote(sistema.soporte[fase], sub_sop)
        sistema.soporte_minimo[fase] = float(np.quantile(scores, PERCENTIL_SOPORTE_MINIMO))
        acciones = sub[ACCIONES_PRIMITIVAS_V5[fase]].dropna()
        sistema.limites_accion[fase] = {a: (float(acciones[a].quantile(0.01)), float(acciones[a].quantile(0.99)))
                                        for a in ACCIONES_PRIMITIVAS_V5[fase]}
        t = sub["temperatura_horno_celsius"].dropna()
        sistema.ventana_temperatura[fase] = (float(t.quantile(0.05)), float(t.quantile(0.95)))
    return sistema


def tabla_efectos_control_v5(sistema: SistemaPrescriptivoV5) -> pd.DataFrame:
    filas = []
    for (fase, clave), m in sistema.modelos.items():
        t = m.tabla_theta(SIGNO_TEORICO_V5.get((fase, clave))).reset_index()
        t.insert(0, "target", TARGETS_V5[fase][clave]); t.insert(0, "clave", clave); t.insert(0, "fase", fase)
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


def tabla_validacion_v5(df: pd.DataFrame, n_splits: int = mp.N_SPLITS_OOF, seed: int = SEED, con_hgb_ref: bool = True) -> pd.DataFrame:
    """R2/MAE OOF (DEV) y lockbox por (fase, target); referencia HGB con estado + palancas (mismas filas)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    filas = []
    for fase in FASES:
        sub = df_base.loc[df_base["fase_proceso"] == fase]
        for clave, target in TARGETS_V5[fase].items():
            S, A = ESTADO_V5[(fase, clave)], PALANCAS_V5[(fase, clave)]
            need = list(dict.fromkeys(S + A + [target, "Batch"]))
            d_dev = sub.loc[sub["Batch"].isin(batches_dev), need].dropna().reset_index(drop=True)
            d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), need].dropna().reset_index(drop=True)
            oof = _oof_plm(d_dev, S, A, target, n_splits, seed, SIGNO_TEORICO_V5.get((fase, clave)))
            m = _nuevo_plm(fase, clave, S, A, target).fit(d_dev, n_boot=0, seed=seed)
            p_lb = m.predict(d_lb)
            filas.append({"fase": fase, "target": clave, "columna": target, "forma": "PLM_v5", "n_estado": len(S), "n_palancas": len(A),
                          "n_dev": len(d_dev), "n_lockbox": len(d_lb),
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
                filas.append({"fase": fase, "target": clave, "columna": target, "forma": "HGB_estado+palancas_ref", "n_estado": len(S),
                              "n_palancas": len(A), "n_dev": len(d_dev), "n_lockbox": len(d_lb),
                              "r2_oof": r2_score(d_dev[target], oofh), "mae_oof": mean_absolute_error(d_dev[target], oofh),
                              "r2_lockbox": r2_score(d_lb[target], h.predict(d_lb[X])), "mae_lockbox": mean_absolute_error(d_lb[target], h.predict(d_lb[X])),
                              "sd_target_dev": float(d_dev[target].std()), "sd_target_lockbox": float(d_lb[target].std())})
    return pd.DataFrame(filas)


def predicciones_oof_y_lockbox_v5(df: pd.DataFrame, fase: str, clave: str, n_splits: int = mp.N_SPLITS_OOF, seed: int = SEED) -> pd.DataFrame:
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    target = TARGETS_V5[fase][clave]
    S, A = ESTADO_V5[(fase, clave)], PALANCAS_V5[(fase, clave)]
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    need = list(dict.fromkeys(S + A + [target, "Batch", "fecha_inicio", "orden_escalon_fase", "avance_reduccion_sn_prev"]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), need].dropna(subset=S + A + [target]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), need].dropna(subset=S + A + [target]).reset_index(drop=True)
    oof = _oof_plm(d_dev, S, A, target, n_splits, seed, SIGNO_TEORICO_V5.get((fase, clave)))
    m = _nuevo_plm(fase, clave, S, A, target).fit(d_dev, n_boot=0, seed=seed)
    keep = ["Batch", "fecha_inicio", "orden_escalon_fase", "avance_reduccion_sn_prev"]
    return pd.concat([d_dev[keep].assign(real=d_dev[target], pred=oof, conjunto="OOF_dev"),
                      d_lb[keep].assign(real=d_lb[target], pred=m.predict(d_lb), conjunto="lockbox")], ignore_index=True)


# =============================================================================
# 4) Simulador y objetivo
# =============================================================================

def fila_a_state_action_context(fase: str, row: pd.Series) -> tuple[dict, dict, dict]:
    state, action, context = mp3.fila_a_state_action_context(fase, row)
    for a in ACCIONES_PRIMITIVAS_V5[fase]:
        if a not in action:
            action[a] = row.get(a, np.nan)
    # el estado v5 incluye columnas fuera de STATE_V3 (log_ratio, lnirf_prev): se anaden si existen
    for extra in ("log_ratio_sn_feo_prev", "v5_lnirf_prev", "ley_sn_conc_batch_pct", "ley_sn_carga_batch_pct"):
        if extra in row.index and extra not in state and extra not in context:
            state[extra] = row.get(extra, np.nan)
    return state, action, context


def _todas_las_columnas(sistema: SistemaPrescriptivoV5, fase: str) -> list[str]:
    cols = set(sistema.soporte[fase]["features"])
    for (f, k), m in sistema.modelos.items():
        if f == fase:
            cols |= set(m.estado) | set(m.palancas)
    return sorted(cols)


def simular_acciones_lote_v5(sistema: SistemaPrescriptivoV5, fase: str, state: dict, acciones: list[dict],
                             context: dict | None = None, alpha: float = 0.20) -> pd.DataFrame:
    """state + lote de acciones -> componentes log predichas (con intervalo conformal), T predicha, soporte, derivadas."""
    filas = [_agregar_palancas_v5_fila(mp4._agregar_columnas_v4_fila(mp3.recalcular_derivadas_v3(fase, state, a, context))) for a in acciones]
    cols = _todas_las_columnas(sistema, fase)
    X = pd.DataFrame([[f.get(c, np.nan) for c in cols] for f in filas], columns=cols)
    out = pd.DataFrame(index=range(len(acciones)))
    for clave in TARGETS_V5[fase]:
        if clave == "dT":
            continue
        p, lo, hi = sistema.modelos[(fase, clave)].predict_intervalo(X, alpha)
        out[f"pred_{clave}"], out[f"{clave}_lo"], out[f"{clave}_hi"] = p, lo, hi
    dT, dlo, dhi = sistema.modelos[(fase, "dT")].predict_intervalo(X, alpha)
    t_prev = float(state.get("temperatura_horno_celsius_prev", np.nan))
    out["pred_dT"], out["dT_lo"], out["dT_hi"] = dT, dlo, dhi
    out["pred_temperatura"] = t_prev + dT if pd.notna(t_prev) else np.nan
    out["support_score"] = mp3._support_scores_lote(sistema.soporte[fase], X)
    for k in ("exceso_o2_combustion_pct", "relacion_C_Sn_carga", "v5_dosis_C_sn", "v5_exceso_C_pos"):
        out[f"derivada__{k}"] = [f.get(k, np.nan) for f in filas]
    return out


def objetivo_lote_v5(sistema: SistemaPrescriptivoV5, fase: str, state: dict, acciones: list[dict], context: dict | None = None,
                     pesos: dict | None = None, soporte_ref: float | None = None) -> pd.DataFrame:
    """J [kg de Sn a metal, equivalente] por accion candidata.
    Reduccion: J = K_SDI*(ln_sn_dep + w*ln_feo_ret) - costos - w_T*fuera - w_U*ancho_IC(K_SDI*(..)) - w_S*(1-soporte)
    Fusion:    J = K_pp*[beta_C*z(C/Sn) - beta_T*z(T_pred)]/n_F - K_pp*b_S*ln_sn_dep + K_pp*b_IRF*d_lnirf - costos - ... """
    p = dict(PESOS_V5_DEFAULT, **(pesos or {}))
    cfg = sistema.config
    K = cfg["pp_kpi_por_unidad_sdi"] * KG_SN_POR_PP_KPI
    sim = simular_acciones_lote_v5(sistema, fase, state, acciones, context)
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
    # Soporte RELATIVO (v5): solo se penaliza perder densidad historica respecto de la accion de referencia (la historica);
    # ganar densidad no se premia (en v4 el termino absoluto -w_S*(1-s) dominaba a la fisica y empujaba hacia la moda historica).
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


def optimizar_accion_v5(sistema: SistemaPrescriptivoV5, fase: str, state: dict, accion_base: dict, context: dict | None = None,
                        acciones_a_optimizar: list[str] | None = None, n_candidatos: int = 300, n_refinamiento: int = 60,
                        semilla: int = 0, pesos: dict | None = None) -> tuple[dict, float, pd.Series]:
    """Busqueda aleatoria en P1-P99 + refinamiento local con soporte minimo duro (P10 DEV); la accion base siempre es admisible."""
    optimizables = acciones_a_optimizar or ACCIONES_OPTIMIZABLES_V5[fase]
    limites = {a: sistema.limites_accion[fase][a] for a in optimizables}
    rng = np.random.default_rng(semilla)
    candidatos = [dict(accion_base)]
    for _ in range(n_candidatos):
        cand = dict(accion_base)
        for a, (lo, hi) in limites.items():
            cand[a] = float(rng.uniform(lo, hi))
        candidatos.append(cand)
    s_ref = float(simular_acciones_lote_v5(sistema, fase, state, [accion_base], context)["support_score"].iloc[0])
    sim = objetivo_lote_v5(sistema, fase, state, candidatos, context, pesos, soporte_ref=s_ref)
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
        sim_r = objetivo_lote_v5(sistema, fase, state, refin, context, pesos, soporte_ref=s_ref)
        J_r = sim_r["J"].where(sim_r["support_score"] >= umbral)
        if J_r.notna().any():
            j = int(J_r.idxmax())
            if float(sim_r.loc[j, "J"]) > mejor_J:
                mejor_accion, mejor_J, mejor_fila = refin[j], float(sim_r.loc[j, "J"]), sim_r.loc[j]
    return mejor_accion, mejor_J, mejor_fila


def curva_respuesta_v5(sistema: SistemaPrescriptivoV5, fase: str, state: dict, action: dict, context: dict | None,
                       palanca: str, n: int = 25, pesos: dict | None = None) -> pd.DataFrame:
    lo, hi = sistema.limites_accion[fase][palanca]
    grid = np.linspace(lo, hi, n)
    acciones = []
    for v in grid:
        a = dict(action); a[palanca] = float(v); acciones.append(a)
    s_ref = float(simular_acciones_lote_v5(sistema, fase, state, [action], context)["support_score"].iloc[0])
    sim = objetivo_lote_v5(sistema, fase, state, acciones, context, pesos, soporte_ref=s_ref)
    sim.insert(0, "valor", grid); sim.insert(0, "palanca", palanca)
    return sim


# =============================================================================
# 5) Reportes por batch, politica cross-fitted y evidencia (KPI refinado)
# =============================================================================

def reporte_recomendaciones_batch_v5(sistema: SistemaPrescriptivoV5, df: pd.DataFrame, batch_id: str, n_candidatos: int = 200,
                                     semilla: int = 0, pesos: dict | None = None) -> pd.DataFrame:
    df_base = _preparar_base(df)
    sub = df_base.loc[df_base["Batch"] == batch_id].sort_values("fecha_inicio")
    filas = []
    for _, row in sub.iterrows():
        fase = row["fase_proceso"]
        state, accion_hist, context = fila_a_state_action_context(fase, row)
        if any(pd.isna(v) for v in accion_hist.values()) or pd.isna(state.get("ley_sn_escoria_pct_prev")):
            continue
        try:
            s_h = float(simular_acciones_lote_v5(sistema, fase, state, [accion_hist], context)["support_score"].iloc[0])
            sim_h = objetivo_lote_v5(sistema, fase, state, [accion_hist], context, pesos, soporte_ref=s_h).iloc[0]
            accion_opt, J_o, sim_o = optimizar_accion_v5(sistema, fase, state, accion_hist, context, None, n_candidatos, semilla=semilla, pesos=pesos)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{batch_id} {fase} {row['orden_escalon_fase']}: {type(e).__name__}: {e}")
            continue
        fila = {"Batch": batch_id, "fase": fase, "orden_escalon_fase": row["orden_escalon_fase"],
                "avance_reduccion_sn_prev": state.get("avance_reduccion_sn_prev"), "ley_sn_escoria_pct_prev": state["ley_sn_escoria_pct_prev"],
                "real_sn_dep": row.get("v5_ln_sn_dep"), "real_feo_ret": row.get("v5_ln_feo_ret"), "real_d_lnirf": row.get("v5_d_lnirf"),
                "temperatura_real": row.get("temperatura_horno_celsius"),
                "pred_T_historico": sim_h["pred_temperatura"], "pred_T_recomendado": sim_o["pred_temperatura"],
                "support_historico": sim_h["support_score"], "support_recomendado": sim_o["support_score"],
                "J_historico": float(sim_h["J"]), "J_recomendado": J_o, "uplift_J_kgSn": J_o - float(sim_h["J"])}
        for k in [c for c in sim_h.index if c.startswith("pred_") and c not in ("pred_temperatura",)]:
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for k in ("J_sn", "J_feo", "J_costo", "J_temperatura", "J_incertidumbre", "J_soporte"):
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for a in ACCIONES_PRIMITIVAS_V5[fase]:
            fila[f"hist__{a}"] = accion_hist[a]; fila[f"rec__{a}"] = accion_opt[a]
        for k in ("exceso_o2_combustion_pct", "relacion_C_Sn_carga", "v5_dosis_C_sn", "v5_exceso_C_pos"):
            fila[f"hist_derivada__{k}"] = float(sim_h[f"derivada__{k}"]); fila[f"rec_derivada__{k}"] = float(sim_o[f"derivada__{k}"])
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_reporte_batch_v5(rep: pd.DataFrame) -> dict:
    if rep.empty:
        return {}
    fus, red = rep["fase"] == "Fusión", rep["fase"] == "Reducción"
    fis = sum(rep[f"{k}_rec"] - rep[f"{k}_hist"] for k in ("J_sn", "J_feo", "J_costo", "J_temperatura"))
    out = {"n_escalones": int(len(rep)),
           "uplift_fisico_fusion_kg": float(fis[fus].sum()), "uplift_fisico_reduccion_kg": float(fis[red].sum()),
           "uplift_fisico_total_kg": float(fis.sum()),
           "uplift_J_total_kg": float(rep["uplift_J_kgSn"].sum()),
           "pct_escalones_con_mejora": float((rep["uplift_J_kgSn"] > 1.0).mean() * 100),
           "support_medio_historico": float(rep["support_historico"].mean()),
           "support_medio_recomendado": float(rep["support_recomendado"].mean())}
    if red.any():
        out["R_sdi_pred_hist"] = float((rep.loc[red, "pred_sn_dep_hist"] + CONFIG_V5["w_sdi"] * rep.loc[red, "pred_feo_ret_hist"]).sum())
        out["R_sdi_pred_rec"] = float((rep.loc[red, "pred_sn_dep_rec"] + CONFIG_V5["w_sdi"] * rep.loc[red, "pred_feo_ret_rec"]).sum())
        out["R_d_sn_dep_pred"] = float((rep.loc[red, "pred_sn_dep_rec"] - rep.loc[red, "pred_sn_dep_hist"]).sum())
        out["R_d_feo_ret_pred"] = float((rep.loc[red, "pred_feo_ret_rec"] - rep.loc[red, "pred_feo_ret_hist"]).sum())
    if fus.any():
        out["F_d_sn_dep_pred"] = float((rep.loc[fus, "pred_sn_dep_rec"] - rep.loc[fus, "pred_sn_dep_hist"]).sum())
        if "pred_d_lnirf_rec" in rep.columns:
            out["F_d_lnirf_pred"] = float((rep.loc[fus, "pred_d_lnirf_rec"] - rep.loc[fus, "pred_d_lnirf_hist"]).sum())
    return out


def _folds_dev(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> tuple[set, set, dict]:
    return mp4._folds_dev(df, n_folds, seed)


def recomendaciones_cross_fitted_v5(df: pd.DataFrame, etiqueta: str, n_folds: int = 5, n_candidatos: int = 150, semilla: int = 0,
                                    log=print, pesos: dict | None = None, config: dict | None = None) -> pd.DataFrame:
    """'0'..'n_folds-1' = fold de DEV (sistema entrenado SIN esos batches); 'lockbox' = sistema entrenado con todo DEV."""
    dev, lockbox, folds = _folds_dev(df, n_folds)
    if etiqueta == "lockbox":
        sistema = entrenar_sistema_v5(df, n_boot=0, config=config)
        batches, fold, es_lb = sorted(lockbox), -1, True
    else:
        k = int(etiqueta)
        sistema = entrenar_sistema_v5(df[df["Batch"].isin(dev - set(folds[k]))], pct_lockbox=0.0, n_boot=0, config=config)
        batches, fold, es_lb = folds[k], k, False
    reps = []
    for i, b in enumerate(batches):
        rep = reporte_recomendaciones_batch_v5(sistema, df, b, n_candidatos=n_candidatos, semilla=semilla, pesos=pesos)
        if not rep.empty:
            rep["fold"], rep["es_lockbox"] = fold, es_lb
            reps.append(rep)
        if (i + 1) % 10 == 0:
            log(f"{etiqueta}: {i+1}/{len(batches)}")
    return pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()


def metricas_por_batch_v5(df: pd.DataFrame, escalones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: uplift estimado, adherencia (|hist-rec|/sd DEV por palanca y fase), direccion, KPIs
    (refinado, proxy, real, canales), KPI del batch siguiente y controles."""
    dev, _ = mp.split_dev_lockbox(df)
    df_base = _preparar_base(df)
    bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
    sd = {}
    for fase in FASES:
        s = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev)]
        for a in ACCIONES_OPTIMIZABLES_V5[fase]:
            sd[(fase, a)] = float(s[a].std())
    filas = []
    for b, rep in escalones.groupby("Batch"):
        fila = resumen_reporte_batch_v5(rep); fila["Batch"] = b
        fila["es_lockbox"] = bool(rep["es_lockbox"].iloc[0]); fila["fold"] = int(rep["fold"].iloc[0])
        dists = []
        for fase in FASES:
            r = rep.loc[rep["fase"] == fase]
            if r.empty:
                continue
            d_f = []
            for a in ACCIONES_OPTIMIZABLES_V5[fase]:
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


CONTROLES_KPI_V5 = ["ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm", "sn_cargado_batch_kg", "temperatura_media_fusion",
                    "frac_carga_secundaria_media_fusion", "feed_dross_Fe_total_t", "idx_cronologico"]


def evidencia_politica_v5(por_batch: pd.DataFrame, variables: list[str] | None = None, kpis: list[str] | None = None,
                          n_perm: int = 1000, seed: int = 0, controles: list[str] | None = None) -> pd.DataFrame:
    """Adherencia/direccion de la politica vs KPI del batch: OLS HC3 con controles (coef por +1 sd), Spearman, permutacion;
    por subconjunto DEV/LOCKBOX/TOTAL. Hipotesis: dist_* con coeficiente NEGATIVO (mas lejos de la politica, peor KPI)."""
    import statsmodels.api as sm
    from scipy import stats
    variables = variables or ["dist_total", "dist_F", "dist_R", "uplift_fisico_total_kg"]
    kpis = kpis or ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_refinada_pct_next", "recuperacion_refinada_pct_media2", "f_dross", "f_polvo"]
    controles = controles or CONTROLES_KPI_V5
    rng = np.random.default_rng(seed)
    filas = []
    for nombre, sub in [("DEV", por_batch[~por_batch["es_lockbox"]]), ("LOCKBOX", por_batch[por_batch["es_lockbox"]]), ("TOTAL", por_batch)]:
        for kpi in kpis:
            for v in variables:
                d = sub[[kpi, v] + controles].dropna()
                if len(d) < 20:
                    continue
                X = d[[v] + controles].copy()
                X = (X - X.mean()) / X.std().replace(0, 1)
                X = sm.add_constant(X)
                res = sm.OLS(d[kpi], X).fit(cov_type="HC3")
                rho, p_rho = stats.spearmanr(d[kpi], d[v])
                coef = float(res.params[v])
                if n_perm and nombre != "LOCKBOX":
                    nulos = []
                    for _ in range(n_perm):
                        Xp = X.copy(); Xp[v] = rng.permutation(Xp[v].to_numpy())
                        nulos.append(float(sm.OLS(d[kpi], Xp).fit().params[v]))
                    p_perm = float(np.mean(np.abs(nulos) >= abs(coef)))
                else:
                    p_perm = np.nan
                filas.append({"subset": nombre, "kpi": kpi, "variable": v, "n": len(d), "coef_por_sd": coef,
                              "ci_lo": float(res.conf_int().loc[v, 0]), "ci_hi": float(res.conf_int().loc[v, 1]),
                              "p_hc3": float(res.pvalues[v]), "p_perm": p_perm, "spearman": float(rho), "p_spearman": float(p_rho),
                              "r2_adj": float(res.rsquared_adj)})
    return pd.DataFrame(filas)

# =============================================================================
# 7) Variante: posicion de lanza como palanca (lineal + cuadratica, signo libre)
# =============================================================================

LANZA = "posicion_vertical_lanza_mm"
VARIANTE_ACTIVA = "base"


def activar_variante_lanza(centro_mm: float = 2650.0) -> None:
    """Promueve `posicion_vertical_lanza_mm` (ACTION_primitiva del escalon) a palanca de CONTROL en los cinco modelos,
    con termino lineal y cuadratico centrado, ambos de SIGNO LIBRE: la convencion de la medida (mm hacia arriba o hacia
    abajo) no esta confirmada con planta, asi que la direccion y el eventual optimo interior se aprenden de los datos.
    La posicion previa sigue en el estado (punto de partida). Muta los diccionarios de configuracion del modulo."""
    global VARIANTE_ACTIVA, ACCIONES_PRIMITIVAS_V5, ACCIONES_OPTIMIZABLES_V5, FEATURES_SOPORTE_V5, PALANCAS_V5, SIGNO_TEORICO_V5
    CENTROS_CUADRATICOS[LANZA] = float(centro_mm)
    sq = f"{LANZA}__sq"
    PALANCAS_V5 = {k: [a for a in v if a not in (LANZA, sq)] + [LANZA, sq] for k, v in PALANCAS_V5.items()}
    SIGNO_TEORICO_V5 = {k: dict(v, **{LANZA: 0, sq: 0}) for k, v in SIGNO_TEORICO_V5.items()}
    ACCIONES_PRIMITIVAS_V5 = {f: [a for a in v if a != LANZA] + [LANZA] for f, v in ACCIONES_PRIMITIVAS_V5.items()}
    ACCIONES_OPTIMIZABLES_V5 = {f: [a for a in v if a != LANZA] + [LANZA] for f, v in ACCIONES_OPTIMIZABLES_V5.items()}
    FEATURES_SOPORTE_V5 = {f: [a for a in v if a != LANZA] + [LANZA] for f, v in FEATURES_SOPORTE_V5.items()}
    VARIANTE_ACTIVA = "lanza"
    _verificar_seguridad()

