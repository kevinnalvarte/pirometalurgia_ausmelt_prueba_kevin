"""Modelo prescriptivo v8 de Lingo Smelter (iteracion 10, 2026-09-16): el nucleo predictivo de v7 (masa de escoria por cierre fisico,
targets log telescopicos, PLM parcialmente lineal con cross-fitting por batch, estado parsimonioso) con las CORRECCIONES DE FASE 0 del
dictamen del comite (`dictamen_comite_v7.md`, hallazgos §21) y las mejoras de modelo que sobrevivieron a la iteracion 10
(`experimentos/v8/ITERACION_10_diseno.md`, memos e10_0N_resultados.md, hallazgos §22).

Que cambia respecto de v7 y por que:
1. IDENTIFICACION HONESTA: `ModeloPLMDual` estima theta LIBRE y ACOTADO por signo sobre los mismos residuos cross-fitted, con bootstrap
   por batch para ambos; una palanca cuyo IC libre no excluye 0 se declara NO IDENTIFICADA y su theta de uso es 0 (no entra al objetivo).
   La cota de signo ya no fabrica significancia (auditoria B §1). Las replicas bootstrap de theta se conservan para que el termino de
   incertidumbre del objetivo dependa de la accion (auditoria C §2: en v7 era una constante).
2. FORMA DEL CARBON con optimo interior por teoria (E10-01): dosis especifica saturante (`m8_ln1p_dosis` = ln(1 + C/Sn_disp), concava:
   rendimientos decrecientes del reductor) sobre el agotamiento de Sn, y carbon sobrante sobre el estequiometrico de Sn
   (`m8_exceso_C_pos`, que solo puede reducir FeO) sobre la retencion de FeO. CONFIG_V8["formas"] fija la forma elegida.
3. TARGET DE FeO y ESTADO (E10-02): retencion de FeO con descuento del FeO equivalente del Fe metalico alimentado (dross de Fe en R0) si
   mejora la generalizacion; estado sin `espesor_ladrillo_norm_mm` (reloj de campana, auditoria D §1c).
4. OBJETIVO (E10-04a): base SIN legado (K, w anclados en el KPI del batch actual, con IC) y contraste con un objetivo LINEAL EN KG
   (Delta Sn_kg − lambda·Delta FeO_kg); la politica debe conservar el signo bajo ambos y bajo w en su IC (`OBJETIVO_V8`).
5. OPTIMIZADOR CONSERVADOR (dictamen §4.2): limites P5-P95 POR ORDEN de escalon, |Delta| <= 1 sd del orden, carbon de R0 <= P90 del orden,
   aire y O2 CONGELADOS al plan, sin recomendacion si el soporte historico del estado < P10 o faltan leyes, ventana termica DURA
   (T_pred − MAE >= P10 del orden), IC dependiente de la accion, F1 hereda la receta (sin recomendacion).
6. FUSION: beta_F = 0 (conservar Sn en Fusion descartado como principio); J_F = −costos − ventana termica dura − termino de polvo
   solo si sobrevive q-BH (E10-06) − IC; recorte de carbon acotado y acompanado de −O2 proporcional (lambda de lanza constante).
7. EVIDENCIA: adherencia CON SIGNO por palanca, uplift con IC propagado (K, w, theta) y evaluacion CRUZADA (las recomendaciones del fold k
   se evaluan con el modelo de otro fold y con un HGB directo: correccion de la maldicion del optimizador).

Uso tipico:
    import modelo_predictivo_v8 as mp8
    df = mp8.cargar_df()                       # experimentos/cache/df_v8.pkl
    print(mp8.tabla_validacion_v8(df))
    sistema = mp8.entrenar_sistema_v8(df)
    print(mp8.tabla_efectos_control_v8(sistema))
    rep = mp8.reporte_recomendaciones_batch_v8(sistema, df, "AP0350")
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

_RAIZ = Path(__file__).resolve().parent
for _p in (_RAIZ / "experimentos" / "v8", _RAIZ / "experimentos" / "v7", _RAIZ / "experimentos" / "v6", _RAIZ / "experimentos" / "v5",
           _RAIZ / "experimentos" / "balances"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import modelo_predictivo_v6 as mp6  # noqa: E402
import v8_lib as L8  # noqa: E402

warnings.filterwarnings("ignore")
FASES = ["Fusión", "Reducción"]
SEED = 42
KG_SN_POR_PP_KPI = mp5.KG_SN_POR_PP_KPI       # ~512 kg Sn a metal por pp de KPI
C_ESTEQ_POR_SN = mp5.C_ESTEQ_POR_SN           # 0.2024 kg C / kg Sn
O2_POR_KG_C_A_CO = 22.414 / 12.011 / 2        # 0.933 Nm3 O2 por kg C (C + 1/2 O2 -> CO)
CACHE_V8 = L8.CACHE_V8

# =============================================================================
# 0) Configuracion v8 (se fija con los veredictos de E10-01..E10-06; ver candidate_win_model_v8.md)
# =============================================================================

TARGETS_V8: dict[str, dict[str, str]] = {
    # E10-02: retencion de FeO con descuento del FeO equivalente del Fe metalico alimentado (dross de Fe, 50 % Fe; sensibilidad 40/60 %):
    #   m8_ln_feo_ret_dross50 = ln( FeO_inv[t] / (FeO_inv[t-1] + 1.2865 * 0.50 * feed_dross_Fe_kgh[t]) )
    # R2 OOF 0.33 -> 0.43-0.45 (R0: 0.05 -> 0.12; R1-R3 identicos), OOF cronologico 0.30 -> 0.40, walk-forward 0.25 -> 0.32; GN identificado con
    # IC libre; anclaje K/w invariante (E10-04b). Piso C* de equilibrio rechazado (R2 0.55).
    "Reducción": {"sn_dep": "m6_ln_sn_dep", "feo_ret": "m8_ln_feo_ret_dross50", "dT": "d_temperatura_horno_celsius"},
    "Fusión": {"dT": "d_temperatura_horno_celsius"},                                                            # beta_F = 0: sin sn_dep
}
ESTADO_R_V8: list[str] = list(L8.ESTADO_R_SIN_ESPESOR)                                                          # <- E10-02
_ESTADO_R_K15 = list(ESTADO_R_V8)
_S_DT_R = ["temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "m6_masa_kg_prev", "orden_escalon_fase",
           "grad_temperatura_horno_celsius_prev"]
_S_DT_F = _S_DT_R + ["tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min"]   # E10-06: dT de Fusion SIN F1 (R2 OOF 0.50 vs 0.40); es_F1 no aporta
ESTADO_V8: dict[tuple[str, str], list[str]] = {
    ("Reducción", "sn_dep"): ESTADO_R_V8, ("Reducción", "feo_ret"): ESTADO_R_V8, ("Reducción", "dT"): _S_DT_R,
    ("Fusión", "dT"): _S_DT_F,
}
# Palancas de CONTROL (theta lineal) por (fase, target). Aire y O2 se mantienen en el modelo (para predecir bajo el plan) pero NO se
# optimizan (dictamen §3: aire = setpoint de proteccion de lanza; O2 congelado en el piloto).
PALANCAS_V8: dict[tuple[str, str], list[str]] = {
    # E10-01: 8 formas del carbon sobre el agotamiento de Sn dan el mismo R2 (0.763-0.767); SOLO Cx_sn_v6 (carbon x Sn disponible, cinetica
    # bimolecular) queda identificada con IC LIBRE (+0.179 [0.033, 0.292]); la dosis saturante ln(1 + C/Sn) no se identifica (+0.023 [-0.006,
    # 0.056]) y la curvatura (dosis^2 -0.034 [-0.130, +0.006]) tiene el signo de la teoria pero no excluye 0 -> monotona en el rango historico.
    ("Reducción", "sn_dep"): ["Cx_sn_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    # FeO: los dos canales teoricos del carbon (sobrante sobre el estequiometrico de Sn, y carbon x avance) se mantienen en el modelo para
    # reportarlos, pero NO estan identificados (IC libre +-0.006 / +-0.04): theta de uso 0. GN si (-0.010 [-0.020, -0.0025]).
    ("Reducción", "feo_ret"): ["m8_exceso_C_pos", "Cx_av_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    ("Reducción", "dT"): ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"],
    ("Fusión", "dT"): ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"],
}
SIGNO_TEORICO_V8: dict[tuple[str, str], dict[str, int]] = {
    ("Reducción", "sn_dep"): {"m8_ln1p_dosis": +1, "m8_dosis": +1, "Cx_sn_v6": +1, "tasa_feed_Carbon_kg_min": +1, "m8_dosis_sq": -1,
                              "m8_ln1p_dosis_x_av": -1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": 0},
    ("Reducción", "feo_ret"): {"m8_exceso_C_pos": -1, "tasa_feed_Carbon_kg_min": -1, "Cx_av_v6": -1, "m8_ln1p_dosis": -1,
                               "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1},
    ("Reducción", "dT"): {"tasa_gn_nm3_min": +1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": -1, "tasa_feed_Carbon_kg_min": 0},
    ("Fusión", "dT"): {"tasa_gn_nm3_min": +1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": -1, "tasa_feed_Carbon_kg_min": 0},
}
ACCIONES_PRIMITIVAS_V8 = {f: list(v) for f, v in mp3.ACCIONES_PRIMITIVAS_V3.items()}
ACCIONES_OPTIMIZABLES_V8 = {"Reducción": ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min"], "Fusión": ["tasa_feed_Carbon_kg_min"]}
ACCIONES_CONGELADAS_V8 = ["tasa_o2_nm3_min", "tasa_aire_nm3_min"]
FEATURES_SOPORTE_V8 = {
    "Fusión": ["m6_sn_inv_kg_prev", "m6_masa_kg_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "basicidad_B2_prev",
               "temperatura_horno_celsius_prev", "orden_escalon_fase", "feed_Sn_kgf", "tasa_feed_total_kg_min",
               "tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
    "Reducción": ["m6_sn_inv_kg_prev", "m6_masa_kg_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "m6_avance_prev",
                  "temperatura_horno_celsius_prev", "orden_escalon_fase", "tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min",
                  "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
}

CONFIG_V8: dict = dict(
    # --- objetivo log (base, SIN legado; anclaje E10-04a). K en pp de KPI por unidad de ln_sn_dep; w = K_FeO / K_Sn.
    # E10-04a/b: OLS HC3 del KPI refinado (DEV n 297) sobre Σ_R ln_sn_dep + Σ_R ln_feo_ret(dross50) + controles: K_Sn 2.781 [1.60, 3.96] pp/unidad,
    # K_FeO 16.49 [10.3, 22.7] -> w 5.93 [3.96, 9.62]; sin legado (sensibilidad: legado K 3.52 / w 7.86; EIV w 7.4; lockbox w ~ 0.8).
    w_sdi=5.93, pp_kpi_por_unidad_sdi=2.781,
    w_sdi_ic=(3.96, 9.62),                             # IC95 % bootstrap de w (robustez de la politica)
    # --- objetivo lineal en kg (contraste, auditoria A §4): J = pp_por_kg_sn · (Delta Sn_kg − lambda_kg · Delta FeO_kg)
    lambda_kg=3.0, pp_kpi_por_kg_sn=1.0 / KG_SN_POR_PP_KPI,   # E10-04a: lambda_kg 4.7 DEV / 2.8 total [1, 4.7] -> 3.0 como contraste
    # --- Fusion: beta_F = 0; termino de polvo (pp de KPI por kg de carbon de Fusion) solo si E10-06 lo sostiene
    pp_kpi_por_kg_carbon_F=0.0,                        # <- E10-06
    # --- politica
    solo_identificados=True,       # theta de uso = acotado si el IC LIBRE excluye 0; si no, 0 (no identificado)
    delta_max_sd=1.0,              # |rec − hist| <= delta_max_sd · sd del orden
    techo_carbon_R0_pct=0.90,      # carbon de R0 <= P90 del orden
    percentil_limites=(0.05, 0.95),
    percentil_soporte_minimo=0.10,
    ventana_T_percentiles=(0.10, 0.90),
    ordenes_sin_recomendacion={"Fusión": [1]},   # F1 hereda la receta (E10-06: dT de F1 R2 0.05; no se modela)
    fusion_modo="receta",          # E10-06: sin termino de KPI ni palanca termica identificada en Fusion -> "receta" = mantener (nivel 0 del
                                   # dictamen); "economica" = recorte de carbon por costo (solo con precios reales), <= 1 sd, >= P10, -O2 proporcional
    congelar_fusion_fin_campana=True,             # sin recorte de carbon en Fusion con espesor < P10 (fin de campana) o T previa < P10
    lambda_lanza_constante=True,   # al mover carbon en Fusion, O2 acompana: Delta O2 = 0.933 · Delta C (kg/min -> Nm3/min)
    exigir_ganancia_fisica=True,   # E10-05: sin ganancia fisica (K·Δ(ln_sn_dep + w·ln_feo_ret) <= 0) no se recomienda por costo (precios placeholder)
    robustez_w=(3.96, 9.62),       # E10-05 sensibilidad: el GN cambia de signo con w bajo; se recomienda mover GN solo si la direccion se conserva
                                   # en los extremos del IC de w (el carbon es robusto: <= 2 % de signos opuestos dentro del IC)
)
OBJETIVO_V8 = "log"    # "log" (K·(ln_sn_dep + w·ln_feo_ret)) | "lineal_kg" (Delta Sn_kg − lambda·Delta FeO_kg)
PESOS_V8_DEFAULT = dict(costo_carbon_kgSn_por_kg=0.05, costo_gn_kgSn_por_nm3=0.03, costo_o2_kgSn_por_nm3=0.01,
                        w_incertidumbre=0.5, w_soporte=2000.0)   # precios placeholder (dictamen: pedir precios reales a planta)


def es_segura_v8(nombre: str) -> bool:
    return L8.es_segura_v8(nombre)


def _verificar_seguridad() -> None:
    for k, feats in list(ESTADO_V8.items()) + list(PALANCAS_V8.items()):
        inseg = [f for f in feats if not es_segura_v8(f)]
        if inseg:
            raise AssertionError(f"v8 {k}: features inseguras {inseg}")
    for fase, feats in FEATURES_SOPORTE_V8.items():
        inseg = [f for f in feats if not es_segura_v8(f)]
        if inseg:
            raise AssertionError(f"v8 soporte {fase}: features inseguras {inseg}")


_verificar_seguridad()


MODO_ESTADO_V8 = "k15"   # "k15" (ensayo del cierre de t-1; nivel 2 del dictamen) | "plan" (cierre de F6 para todo R; nivel 1, ejecutable sin latencia)
_PALANCAS_K15 = {k: list(v) for k, v in PALANCAS_V8.items() if k[0] == "Reducción" and k[1] in ("sn_dep", "feo_ret")}


def activar_modo_plan(activar: bool = True) -> None:
    """E10-03: estado PLAN = leyes/inventarios congelados al cierre de F6 + variables en linea de t-1 (conserva 92 % del R2 de Sn, 78 % del
    de FeO; carbon +0.118 [0.028, 0.209] y GN identificados). Permite planificar R0-R3 al inicio de la Reduccion sin esperar al laboratorio."""
    global MODO_ESTADO_V8
    if activar:
        configurar_v8(estado_R=L8.ESTADO_R_PLAN,
                      palancas={("Reducción", "sn_dep"): ["Cx_sn_F6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                                ("Reducción", "feo_ret"): ["m8_exceso_C_pos", "Cx_av_F6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]})
        SIGNO_TEORICO_V8[("Reducción", "sn_dep")]["Cx_sn_F6"] = +1; SIGNO_TEORICO_V8[("Reducción", "feo_ret")]["Cx_av_F6"] = -1
        MODO_ESTADO_V8 = "plan"
    else:
        configurar_v8(estado_R=_ESTADO_R_K15, palancas=_PALANCAS_K15); MODO_ESTADO_V8 = "k15"


def configurar_v8(targets: dict | None = None, estado_R: list[str] | None = None, palancas: dict | None = None,
                  config: dict | None = None, objetivo: str | None = None) -> None:
    """Reconfigura el modulo (para experimentos y para fijar los veredictos de la iteracion 10)."""
    global OBJETIVO_V8
    if targets:
        for f, d in targets.items():
            TARGETS_V8[f].update(d)
    if estado_R is not None:
        ESTADO_R_V8[:] = list(estado_R)
        ESTADO_V8[("Reducción", "sn_dep")] = ESTADO_R_V8; ESTADO_V8[("Reducción", "feo_ret")] = ESTADO_R_V8
    if palancas:
        PALANCAS_V8.update(palancas)
    if config:
        CONFIG_V8.update(config)
    if objetivo:
        OBJETIVO_V8 = objetivo
    _verificar_seguridad()


# =============================================================================
# 1) Datos y columnas por fila (simulador)
# =============================================================================

def cargar_df() -> pd.DataFrame:
    return L8.cargar_df()


def _preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    if "m8_ln1p_dosis" not in df.columns:
        df = L8.agregar_columnas_v8(df)
    df_base = mp.dataset_base_modelo(df)
    g = "grad_temperatura_horno_celsius_prev"
    m = (df_base["fase_proceso"] == "Fusión") & (df_base["orden_escalon_fase"] == 1) & df_base[g].isna()
    df_base.loc[m, g] = 0.0   # F1: gradiente estructuralmente indefinido (v7)
    return df_base


def _fila_v8(fila: dict) -> dict:
    """Derivadas v8 por fila (misma formula que `v8_lib.agregar_columnas_v8`) a partir del estado m6_* y la accion."""
    fila = mp6._agregar_columnas_v6_fila(mp5._agregar_palancas_v5_fila(mp4._agregar_columnas_v4_fila(fila)))
    c_kg = float(fila.get("feed_Carbon_kgh", 0.0) or 0.0)
    tasa_c = float(fila.get("tasa_feed_Carbon_kg_min", np.nan))
    sn_disp = float(fila.get("m6_sn_disp", np.nan))
    dosis = c_kg / sn_disp if (np.isfinite(sn_disp) and sn_disp > 200.0) else np.nan
    fila["m8_dosis"] = dosis
    fila["m8_ln1p_dosis"] = np.log1p(dosis) if np.isfinite(dosis) else np.nan
    fila["m8_dosis_sq"] = dosis ** 2 if np.isfinite(dosis) else np.nan
    fila["m8_ln1p_dosis_x_av"] = fila["m8_ln1p_dosis"] * float(fila.get("m6_avance_prev", np.nan))
    fila["m8_exceso_C_pos"] = max(c_kg - C_ESTEQ_POR_SN * sn_disp, 0.0) if np.isfinite(sn_disp) else np.nan
    # estado PLAN (E10-03): palancas cineticas con el inventario congelado al cierre de F6
    sn_f6 = float(fila.get("m6_sn_inv_kg_F6", np.nan)); av_f6 = float(fila.get("avance_F6", np.nan))
    fila["Cx_sn_F6"] = tasa_c * sn_f6 / 1000.0 if np.isfinite(sn_f6) else np.nan
    fila["Cx_av_F6"] = tasa_c * av_f6 if np.isfinite(av_f6) else np.nan
    for col, nombre in (("temperatura_horno_celsius_prev", "m8_C_x_zT"), ("basicidad_B2_prev", "m8_C_x_zB2")):
        mu, sd = L8.CENTROS_V8.get(col, (np.nan, np.nan))
        fila[nombre] = tasa_c * (float(fila.get(col, np.nan)) - mu) / sd
    mu_gn, _ = L8.CENTROS_V8.get("tasa_gn_nm3_min", (np.nan, np.nan))
    fila["m8_gn_sq"] = (float(fila.get("tasa_gn_nm3_min", np.nan)) - mu_gn) ** 2
    return fila


def fila_a_state_action_context(fase: str, row: pd.Series) -> tuple[dict, dict, dict]:
    state, action, context = mp6.fila_a_state_action_context(fase, row)
    usados = set(FEATURES_SOPORTE_V8[fase])
    for (f, k) in ESTADO_V8:
        if f == fase:
            usados |= set(ESTADO_V8[(f, k)]) | set(PALANCAS_V8[(f, k)])
    for c in ("es_F1", "espesor_ladrillo_norm_mm", "orden_escalon_fase", "tiempo_fase", "m6_sn_disp", "ley_al2o3_escoria_pct_prev",
              "presion_punta_lanza_kpa_prev", "tiro_horno_pct_prev", "cum_gn_nm3_prev", "cum_o2_nm3_prev", "cum_feed_Sn_kg_prev",
              "avance_F6", *L8.ESTADO_R_PLAN):
        usados.add(c)
    for c in usados:
        if c not in state and c not in action and c not in context and c in row.index:
            state[c] = row.get(c, np.nan)
    return state, action, context


# =============================================================================
# 2) PLM dual: theta libre y acotado, replicas bootstrap, identificacion honesta
# =============================================================================

class ModeloPLMDual(mp5.ModeloPLMSignos):
    """PLM v5/v7 (`y = g(S) + theta·(a − E[a|S])`, nuisances HGB cross-fitted por batch) que ademas:
    - estima theta LIBRE y theta ACOTADO por el signo de teoria sobre los MISMOS residuos y guarda el IC bootstrap de ambos;
    - declara `identificado[j]` = el IC95 % LIBRE excluye 0 (la cota no fabrica significancia);
    - theta de USO = acotado · identificado (si `solo_identificados`), es decir, 0 = "sin efecto identificado";
    - conserva las replicas bootstrap del theta de uso (`boots_uso`) para intervalos DEPENDIENTES DE LA ACCION."""

    def __init__(self, estado, palancas, target, signos=None, solo_identificados: bool = True):
        super().__init__(estado, palancas, target, signos)
        self.solo_identificados = solo_identificados

    def fit(self, d: pd.DataFrame, n_splits: int = 5, n_boot: int = 300, seed: int = SEED):
        from scipy.optimize import lsq_linear
        S, A, y, grupos = self.estado, self.palancas, self.target, d["Batch"].to_numpy()
        cols = [y] + A
        oof = pd.DataFrame(np.nan, index=d.index, columns=cols)
        for tr, te in GroupKFold(n_splits).split(d[S], d[y], grupos):
            for c in cols:
                mdl = mp4._hgb_nuisance(seed).fit(d[S].iloc[tr], d[c].iloc[tr])
                oof.iloc[te, oof.columns.get_loc(c)] = mdl.predict(d[S].iloc[te])
        y_res = d[y].to_numpy() - oof[y].to_numpy()
        A_res = np.column_stack([d[a].to_numpy() - oof[a].to_numpy() for a in A])
        lo, hi = self._bounds()
        acotar = not (np.all(np.isinf(lo)) and np.all(np.isinf(hi)))

        def solve(Ar, yr):
            libre = np.linalg.lstsq(Ar, yr, rcond=None)[0]
            acot = lsq_linear(Ar, yr, bounds=(lo, hi), lsmr_tol="auto").x if acotar else libre
            return libre, acot
        self.theta_libre, self.theta_acot = solve(A_res, y_res)
        rng = np.random.default_rng(seed)
        uniq = np.unique(grupos); idx_by = {b: np.where(grupos == b)[0] for b in uniq}
        BL, BA = [], []
        for _ in range(n_boot):
            idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
            l_, a_ = solve(A_res[idx], y_res[idx]); BL.append(l_); BA.append(a_)
        if n_boot:
            BL, BA = np.array(BL), np.array(BA)
            self.ci_lo_libre, self.ci_hi_libre = np.percentile(BL, 2.5, 0), np.percentile(BL, 97.5, 0)
            self.ci_lo, self.ci_hi = np.percentile(BA, 2.5, 0), np.percentile(BA, 97.5, 0)
            self.frac_en_cota = (np.abs(BA) < 1e-9).mean(0)
            self.identificado = (self.ci_lo_libre > 0) | (self.ci_hi_libre < 0)
        else:
            k = len(A)
            self.ci_lo_libre = self.ci_hi_libre = self.ci_lo = self.ci_hi = np.full(k, np.nan)
            self.frac_en_cota = np.full(k, np.nan)
            self.identificado = np.abs(self.theta_acot) > 1e-9   # sin bootstrap: se usa el acotado tal cual (para folds rapidos)
            BA = np.array([self.theta_acot])
        self.theta = np.where(self.identificado, self.theta_acot, 0.0) if self.solo_identificados else self.theta_acot
        self.boots_uso = (BA * self.identificado[None, :]) if self.solo_identificados else BA
        self.res_oof = y_res - A_res @ self.theta
        self.r2_oof_interno = float(r2_score(d[y], d[y].to_numpy() - self.res_oof))
        self.sd_palancas = d[A].std().to_numpy()
        self.g = mp4._hgb_nuisance(seed).fit(d[S], d[y])
        self.m = {a: mp4._hgb_nuisance(seed).fit(d[S], d[a]) for a in A}
        self.n_train = len(d)
        self.mae_oof = float(np.mean(np.abs(self.res_oof)))
        return self

    def predict_replicas(self, X: pd.DataFrame, n_max: int = 200) -> np.ndarray:
        """Predicciones bajo las replicas bootstrap del theta de uso (g y m fijos): matriz (n_replicas, n_filas)."""
        base = self.g.predict(X[self.estado]).astype(float)
        A_res = np.column_stack([X[a].to_numpy(dtype=float) - self.m[a].predict(X[self.estado]) for a in self.palancas])
        B = self.boots_uso[: n_max]
        return base[None, :] + (A_res @ B.T).T

    def tabla_theta(self, signo_teorico: dict | None = None) -> pd.DataFrame:
        sd = self.sd_palancas
        t = pd.DataFrame({"palanca": self.palancas, "signo_teorico": [self.signos.get(a, 0) for a in self.palancas], "sd_palanca": sd,
                          "theta_libre_1sd": self.theta_libre * sd, "ci_lo_libre_1sd": self.ci_lo_libre * sd, "ci_hi_libre_1sd": self.ci_hi_libre * sd,
                          "theta_acot_1sd": self.theta_acot * sd, "ci_lo_acot_1sd": self.ci_lo * sd, "ci_hi_acot_1sd": self.ci_hi * sd,
                          "frac_replicas_en_cota": self.frac_en_cota, "identificado": self.identificado,
                          "theta_uso_1sd": self.theta * sd, "theta_uso_unidad": self.theta})
        t["concuerda_signo"] = [(np.sign(th) == s) if (s != 0 and abs(th) > 1e-12) else np.nan for th, s in zip(self.theta_libre, t["signo_teorico"])]
        return t.set_index("palanca")


# =============================================================================
# 3) Sistema v8
# =============================================================================

@dataclass
class SistemaPrescriptivoV8:
    modelos: dict = field(default_factory=dict)          # (fase, clave) -> ModeloPLMDual
    soporte: dict = field(default_factory=dict)
    soporte_minimo: dict = field(default_factory=dict)
    limites_orden: dict = field(default_factory=dict)    # (fase, orden) -> {primitiva: (p5, p95, sd, p90)}
    ventana_T_orden: dict = field(default_factory=dict)  # (fase, orden) -> (p10, p90)
    umbral_fin_campana: float = np.nan                   # P10 de espesor en DEV
    batches_dev: set = field(default_factory=set)
    batches_lockbox: set = field(default_factory=set)
    config: dict = field(default_factory=lambda: dict(CONFIG_V8))
    objetivo: str = OBJETIVO_V8


def _filas_modelo(sub: pd.DataFrame, fase: str, clave: str) -> pd.DataFrame:
    """Filas que entran a cada modelo: en Fusion se excluye F1 (E10-06: gradiente previo indefinido, dT no predecible, sin recomendacion)."""
    if fase == "Fusión":
        return sub.loc[sub["orden_escalon_fase"] != 1]
    return sub


def _nuevo_plm(fase: str, clave: str, S: list[str], A: list[str], target: str, config: dict):
    return ModeloPLMDual(S, A, target, SIGNO_TEORICO_V8.get((fase, clave)), solo_identificados=config.get("solo_identificados", True))


def entrenar_sistema_v8(df: pd.DataFrame, pct_lockbox: float = mp.N_LOCKBOX_PCT, n_boot: int = 300, seed: int = SEED,
                        config: dict | None = None, objetivo: str | None = None) -> SistemaPrescriptivoV8:
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df, pct_lockbox)
    cfg = dict(CONFIG_V8, **(config or {}))
    sistema = SistemaPrescriptivoV8(batches_dev=batches_dev, batches_lockbox=batches_lockbox, config=cfg, objetivo=objetivo or OBJETIVO_V8)
    plo, phi = cfg["percentil_limites"]
    tlo, thi = cfg["ventana_T_percentiles"]
    for fase in FASES:
        sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        for clave, target in TARGETS_V8[fase].items():
            S, A = ESTADO_V8[(fase, clave)], PALANCAS_V8[(fase, clave)]
            d = _filas_modelo(sub, fase, clave).dropna(subset=list(dict.fromkeys(S + A + [target]))).reset_index(drop=True)
            sistema.modelos[(fase, clave)] = _nuevo_plm(fase, clave, S, A, target, cfg).fit(d, n_boot=n_boot, seed=seed)
        feats_sop = FEATURES_SOPORTE_V8[fase]
        sub_sop = sub.dropna(subset=feats_sop)
        sistema.soporte[fase] = mp._construir_soporte_historico(sub_sop, feats_sop)
        scores = mp3._support_scores_lote(sistema.soporte[fase], sub_sop)
        sistema.soporte_minimo[fase] = float(np.quantile(scores, cfg["percentil_soporte_minimo"]))
        for orden, g in sub.groupby("orden_escalon_fase"):
            lim = {}
            for a in ACCIONES_PRIMITIVAS_V8[fase]:
                x = g[a].dropna()
                if len(x) < 10:
                    continue
                lim[a] = (float(x.quantile(plo)), float(x.quantile(phi)), float(x.std()), float(x.quantile(cfg["techo_carbon_R0_pct"])))
            sistema.limites_orden[(fase, int(orden))] = lim
            t = g["temperatura_horno_celsius"].dropna()
            sistema.ventana_T_orden[(fase, int(orden))] = (float(t.quantile(tlo)), float(t.quantile(thi)))
    esp = df.loc[df["Batch"].isin(batches_dev), "espesor_ladrillo_norm_mm"].dropna()
    sistema.umbral_fin_campana = float(esp.quantile(0.10)) if len(esp) else np.nan
    return sistema


def tabla_efectos_control_v8(sistema: SistemaPrescriptivoV8) -> pd.DataFrame:
    filas = []
    for (fase, clave), m in sistema.modelos.items():
        t = m.tabla_theta().reset_index()
        t.insert(0, "target", TARGETS_V8[fase][clave]); t.insert(0, "clave", clave); t.insert(0, "fase", fase)
        filas.append(t)
    return pd.concat(filas, ignore_index=True)


# =============================================================================
# 4) Validacion OOF (aleatorio y cronologico) / walk-forward / lockbox
# =============================================================================

def _fp(fase, clave, S, A, target, cfg, seed):
    def fp(d_tr, d_te):
        return _nuevo_plm(fase, clave, S, A, target, cfg).fit(d_tr, n_boot=0, seed=seed).predict(d_te)
    return fp


def tabla_validacion_v8(df: pd.DataFrame, n_splits: int = mp.N_SPLITS_OOF, seed: int = SEED, con_hgb_ref: bool = True,
                        con_wf: bool = True) -> pd.DataFrame:
    """R2/MAE OOF aleatorio (criterio), OOF cronologico, walk-forward y lockbox por (fase, target); referencia HGB directo."""
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    cfg = dict(CONFIG_V8, solo_identificados=False)   # en la validacion predictiva el theta acotado completo (como v7)
    filas = []
    for fase in FASES:
        sub = df_base.loc[df_base["fase_proceso"] == fase]
        for clave, target in TARGETS_V8[fase].items():
            S, A = ESTADO_V8[(fase, clave)], PALANCAS_V8[(fase, clave)]
            need = list(dict.fromkeys(S + A + [target, "Batch", "fecha_inicio", "orden_escalon_fase"]))
            subm = _filas_modelo(sub, fase, clave)
            d_dev = subm.loc[subm["Batch"].isin(batches_dev), need].dropna(subset=S + A + [target]).reset_index(drop=True)
            d_lb = subm.loc[subm["Batch"].isin(batches_lockbox), need].dropna(subset=S + A + [target]).reset_index(drop=True)
            fp = _fp(fase, clave, S, A, target, cfg, seed)
            oof = L8.oof_groupkfold(d_dev, fp, n_splits); ooc = L8.oof_cronologico(d_dev, fp)
            fila = {"fase": fase, "target": clave, "columna": target, "forma": "PLM_v8", "n_estado": len(S), "n_palancas": len(A),
                    "n_dev": len(d_dev), "n_lockbox": len(d_lb),
                    "r2_oof": r2_score(d_dev[target], oof), "mae_oof": mean_absolute_error(d_dev[target], oof),
                    "r2_oof_cronologico": r2_score(d_dev[target], ooc)}
            if con_wf:
                pw, _ = L8.walk_forward(d_dev, fp, target); m = np.isfinite(pw)
                fila["r2_walk_forward"] = r2_score(d_dev[target][m], pw[m]); fila["n_walk_forward"] = int(m.sum())
            mdl = _nuevo_plm(fase, clave, S, A, target, cfg).fit(d_dev, n_boot=0, seed=seed)
            fila["r2_lockbox"] = r2_score(d_lb[target], mdl.predict(d_lb)); fila["mae_lockbox"] = mean_absolute_error(d_lb[target], mdl.predict(d_lb))
            fila["sd_target_dev"] = float(d_dev[target].std()); fila["sd_target_lockbox"] = float(d_lb[target].std())
            filas.append(fila)
            if con_hgb_ref:
                X = S + A
                fph = L8.fit_predict_hgb(X, target, seed)
                oofh = L8.oof_groupkfold(d_dev, fph, n_splits); ooch = L8.oof_cronologico(d_dev, fph)
                h = mp4._hgb_nuisance(seed, 300).fit(d_dev[X], d_dev[target])
                filas.append({**fila, "forma": "HGB_estado+palancas_ref", "r2_oof": r2_score(d_dev[target], oofh),
                              "mae_oof": mean_absolute_error(d_dev[target], oofh), "r2_oof_cronologico": r2_score(d_dev[target], ooch),
                              "r2_walk_forward": np.nan, "r2_lockbox": r2_score(d_lb[target], h.predict(d_lb[X])),
                              "mae_lockbox": mean_absolute_error(d_lb[target], h.predict(d_lb[X]))})
    return pd.DataFrame(filas)


def predicciones_oof_y_lockbox_v8(df: pd.DataFrame, fase: str, clave: str, n_splits: int = mp.N_SPLITS_OOF, seed: int = SEED) -> pd.DataFrame:
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    target = TARGETS_V8[fase][clave]
    S, A = ESTADO_V8[(fase, clave)], PALANCAS_V8[(fase, clave)]
    cfg = dict(CONFIG_V8, solo_identificados=False)
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    keep = ["Batch", "fecha_inicio", "orden_escalon_fase", "m6_avance_prev"]
    need = list(dict.fromkeys(S + A + [target] + keep))
    sub = _filas_modelo(sub, fase, clave)
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), need].dropna(subset=S + A + [target]).reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), need].dropna(subset=S + A + [target]).reset_index(drop=True)
    fp = _fp(fase, clave, S, A, target, cfg, seed)
    oof = L8.oof_groupkfold(d_dev, fp, n_splits)
    m = _nuevo_plm(fase, clave, S, A, target, cfg).fit(d_dev, n_boot=0, seed=seed)
    return pd.concat([d_dev[keep].assign(real=d_dev[target], pred=oof, conjunto="OOF_dev"),
                      d_lb[keep].assign(real=d_lb[target], pred=m.predict(d_lb), conjunto="lockbox")], ignore_index=True)


# =============================================================================
# 5) Simulador, objetivo y optimizador conservador
# =============================================================================

def _todas_las_columnas(sistema: SistemaPrescriptivoV8, fase: str) -> list[str]:
    cols = set(sistema.soporte[fase]["features"])
    for (f, k), m in sistema.modelos.items():
        if f == fase:
            cols |= set(m.estado) | set(m.palancas)
    return sorted(cols)


DERIVADAS_REPORTE = ("exceso_o2_combustion_pct", "m8_dosis", "m8_exceso_C_pos", "Cx_sn_v6", "Cx_sn_F6")


def simular_acciones_lote_v8(sistema: SistemaPrescriptivoV8, fase: str, state: dict, acciones: list[dict],
                             context: dict | None = None, n_rep: int = 200) -> pd.DataFrame:
    """state + lote de acciones -> predicciones de cada target, IC dependiente de la accion (replicas bootstrap del theta de uso),
    T predicha, soporte y derivadas."""
    filas = [_fila_v8(mp3.recalcular_derivadas_v3(fase, state, a, context)) for a in acciones]
    cols = _todas_las_columnas(sistema, fase)
    X = pd.DataFrame([[f.get(c, np.nan) for c in cols] for f in filas], columns=cols)
    out = pd.DataFrame(index=range(len(acciones)))
    for clave in TARGETS_V8[fase]:
        m = sistema.modelos[(fase, clave)]
        rep = m.predict_replicas(X, n_rep)
        out[f"pred_{clave}"] = m.predict(X)
        out[f"{clave}_lo"], out[f"{clave}_hi"] = np.percentile(rep, 2.5, 0), np.percentile(rep, 97.5, 0)
        out[f"{clave}_sd_rep"] = rep.std(0)
    t_prev = float(state.get("temperatura_horno_celsius_prev", np.nan))
    out["pred_temperatura"] = t_prev + out["pred_dT"] if pd.notna(t_prev) else np.nan
    out["mae_dT"] = sistema.modelos[(fase, "dT")].mae_oof
    out["support_score"] = mp3._support_scores_lote(sistema.soporte[fase], X)
    for k in DERIVADAS_REPORTE:
        out[f"derivada__{k}"] = [f.get(k, np.nan) for f in filas]
    out["sn_disp"] = [f.get("m6_sn_disp", np.nan) for f in filas]
    out["feo_inv_prev"] = float(state.get("m6_feo_inv_kg_prev", np.nan))
    return out


def objetivo_lote_v8(sistema: SistemaPrescriptivoV8, fase: str, state: dict, acciones: list[dict], context: dict | None = None,
                     pesos: dict | None = None, soporte_ref: float | None = None, orden: int | None = None,
                     objetivo: str | None = None, config: dict | None = None) -> pd.DataFrame:
    """J [kg de Sn a metal equivalente] por accion candidata.
    Reduccion, objetivo "log":       J = K·(ln_sn_dep + w·ln_feo_ret) − costos − IC(accion) − soporte relativo, con K = pp/unidad × 512 kg/pp
    Reduccion, objetivo "lineal_kg": J = pp_por_kg·512·(ΔSn_kg − λ·ΔFeO_kg) − …, ΔSn_kg = Sn_disp·(1 − e^{−ln_sn_dep}), ΔFeO_kg = FeO_prev·(1 − e^{ln_feo_ret})
    Fusion:                          J = −pp_polvo·512·C_kg − costos − IC − soporte (beta_F = 0)
    La ventana termica es una RESTRICCION dura (columna `admisible_T`), no una penalizacion."""
    p = dict(PESOS_V8_DEFAULT, **(pesos or {}))
    cfg = dict(sistema.config, **(config or {}))
    obj = objetivo or sistema.objetivo
    sim = simular_acciones_lote_v8(sistema, fase, state, acciones, context)
    A = pd.DataFrame(acciones)
    dt = A["duracion_plan_min"].to_numpy(dtype=float)
    C = A["tasa_feed_Carbon_kg_min"].to_numpy(dtype=float)
    costo = (p["costo_carbon_kgSn_por_kg"] * C * dt + p["costo_gn_kgSn_por_nm3"] * A["tasa_gn_nm3_min"].to_numpy(dtype=float) * dt
             + p["costo_o2_kgSn_por_nm3"] * A["tasa_o2_nm3_min"].to_numpy(dtype=float) * dt)
    sim["J_costo"] = -costo
    if soporte_ref is None:
        sim["J_soporte"] = -p["w_soporte"] * (1 - sim["support_score"])
    else:
        sim["J_soporte"] = -p["w_soporte"] * np.maximum(0.0, soporte_ref - sim["support_score"])
    # ventana termica dura por orden: T_pred − MAE >= P10 y T_pred + MAE <= P90 (orden desconocido -> sin restriccion)
    # Restriccion DURA relativa a la accion de referencia (la primera del lote = historica): un candidato es admisible si su T
    # predicha (con margen MAE) queda dentro de la ventana P10-P90 del orden, o si NO empeora la T respecto de la referencia en la
    # direccion en que esta se sale (no enfria un horno ya frio, no calienta uno ya caliente). Asi la ventana nunca bloquea mejoras.
    if orden is not None and (fase, int(orden)) in sistema.ventana_T_orden:
        t_lo, t_hi = sistema.ventana_T_orden[(fase, int(orden))]
        tp, mae = sim["pred_temperatura"].to_numpy(dtype=float), sim["mae_dT"].to_numpy(dtype=float)
        t_ref = tp[0]
        ok_lo = (tp - mae >= t_lo) | (tp >= t_ref - 1e-9)
        ok_hi = (tp + mae <= t_hi) | (tp <= t_ref + 1e-9)
        sim["admisible_T"] = np.isnan(tp) | (ok_lo & ok_hi)
    else:
        sim["admisible_T"] = True
    if fase == "Reducción":
        Kpp = cfg["pp_kpi_por_unidad_sdi"]; w = cfg["w_sdi"]
        if obj == "log":
            sim["J_sn"] = Kpp * KG_SN_POR_PP_KPI * sim["pred_sn_dep"]
            sim["J_feo"] = Kpp * KG_SN_POR_PP_KPI * w * sim["pred_feo_ret"]
            sd_J = Kpp * KG_SN_POR_PP_KPI * np.sqrt(sim["sn_dep_sd_rep"] ** 2 + (w * sim["feo_ret_sd_rep"]) ** 2)
        else:  # lineal en kg
            kk = cfg["pp_kpi_por_kg_sn"] * KG_SN_POR_PP_KPI   # kg Sn a metal por kg de Sn agotado (≈ 1 si pp_por_kg = 1/512)
            d_sn = sim["sn_disp"] * (1 - np.exp(-sim["pred_sn_dep"]))
            d_feo = sim["feo_inv_prev"] * (1 - np.exp(sim["pred_feo_ret"]))
            sim["J_sn"] = kk * d_sn
            sim["J_feo"] = -kk * cfg["lambda_kg"] * d_feo
            sd_J = kk * np.sqrt((sim["sn_disp"] * np.exp(-sim["pred_sn_dep"]) * sim["sn_dep_sd_rep"]) ** 2
                                + (cfg["lambda_kg"] * sim["feo_inv_prev"] * np.exp(sim["pred_feo_ret"]) * sim["feo_ret_sd_rep"]) ** 2)
        sim["J_incertidumbre"] = -p["w_incertidumbre"] * sd_J
    else:
        sim["J_sn"] = 0.0
        sim["J_feo"] = -cfg["pp_kpi_por_kg_carbon_F"] * KG_SN_POR_PP_KPI * C * dt     # termino de polvo (0 si E10-06 no lo sostiene)
        sim["J_incertidumbre"] = 0.0
    sim["J"] = sim[["J_sn", "J_feo", "J_costo", "J_incertidumbre", "J_soporte"]].sum(axis=1)
    return sim


def _caja_accion(sistema: SistemaPrescriptivoV8, fase: str, orden: int, accion_base: dict, palanca: str) -> tuple[float, float]:
    """Caja admisible de una palanca: [P5, P95] del orden ∩ [hist − sd, hist + sd] del orden; carbon de R0 <= P90 del orden."""
    cfg = sistema.config
    lim = sistema.limites_orden.get((fase, int(orden)), {}).get(palanca)
    if lim is None:
        return (np.nan, np.nan)
    p5, p95, sd, p90 = lim
    h = float(accion_base[palanca])
    lo, hi = max(p5, h - cfg["delta_max_sd"] * sd), min(p95, h + cfg["delta_max_sd"] * sd)
    if fase == "Reducción" and int(orden) == 0 and palanca == "tasa_feed_Carbon_kg_min":
        hi = min(hi, p90)
    lo, hi = min(lo, h), max(hi, h)   # la accion historica siempre es admisible
    return (lo, hi)


def optimizar_accion_v8(sistema: SistemaPrescriptivoV8, fase: str, orden: int, state: dict, accion_base: dict, context: dict | None = None,
                        n_grid_c: int = 21, n_grid_gn: int = 7, pesos: dict | None = None, objetivo: str | None = None,
                        config: dict | None = None) -> tuple[dict, float, pd.Series, str]:
    """Busqueda en grilla dentro de la caja por orden (carbon × GN en Reduccion; carbon en Fusion con O2 acompanando si
    lambda_lanza_constante). Devuelve (accion, J, fila_sim, estado_recomendacion) con estado_recomendacion en
    {'recomendar', 'mantener', 'sin_recomendacion:soporte', 'sin_recomendacion:orden', 'sin_recomendacion:fin_campana'}."""
    cfg = dict(sistema.config, **(config or {}))
    s_ref = float(simular_acciones_lote_v8(sistema, fase, state, [accion_base], context, n_rep=1)["support_score"].iloc[0])
    if int(orden) in cfg["ordenes_sin_recomendacion"].get(fase, []):
        sim0 = objetivo_lote_v8(sistema, fase, state, [accion_base], context, pesos, s_ref, orden, objetivo, cfg).iloc[0]
        return dict(accion_base), float(sim0["J"]), sim0, "sin_recomendacion:orden"
    umbral = sistema.soporte_minimo.get(fase, 0.0)
    sim0 = objetivo_lote_v8(sistema, fase, state, [accion_base], context, pesos, s_ref, orden, objetivo, cfg).iloc[0]
    if s_ref < umbral:
        return dict(accion_base), float(sim0["J"]), sim0, "sin_recomendacion:soporte"
    if fase == "Fusión" and cfg.get("fusion_modo", "receta") == "receta":
        return dict(accion_base), float(sim0["J"]), sim0, "sin_recomendacion:receta"
    if fase == "Fusión" and cfg["congelar_fusion_fin_campana"]:
        esp = float(state.get("espesor_ladrillo_norm_mm", np.nan)); t_prev = float(state.get("temperatura_horno_celsius_prev", np.nan))
        t_lo = sistema.ventana_T_orden.get((fase, int(orden)), (np.nan, np.nan))[0]
        if (np.isfinite(esp) and esp < sistema.umbral_fin_campana) or (np.isfinite(t_prev) and np.isfinite(t_lo) and t_prev < t_lo):
            return dict(accion_base), float(sim0["J"]), sim0, "sin_recomendacion:fin_campana"
    opt = ACCIONES_OPTIMIZABLES_V8[fase]
    cajas = {a: _caja_accion(sistema, fase, orden, accion_base, a) for a in opt}
    if any(np.isnan(v[0]) for v in cajas.values()):
        return dict(accion_base), float(sim0["J"]), sim0, "sin_recomendacion:orden"
    grids = {"tasa_feed_Carbon_kg_min": np.linspace(*cajas["tasa_feed_Carbon_kg_min"], n_grid_c)}
    if "tasa_gn_nm3_min" in opt:
        grids["tasa_gn_nm3_min"] = np.linspace(*cajas["tasa_gn_nm3_min"], n_grid_gn)
    candidatos = [dict(accion_base)]
    import itertools
    for vals in itertools.product(*grids.values()):
        cand = dict(accion_base)
        for a, v in zip(grids.keys(), vals):
            cand[a] = float(v)
        if fase == "Fusión" and cfg["lambda_lanza_constante"]:
            d_c = cand["tasa_feed_Carbon_kg_min"] - float(accion_base["tasa_feed_Carbon_kg_min"])
            cand["tasa_o2_nm3_min"] = max(0.0, float(accion_base["tasa_o2_nm3_min"]) + O2_POR_KG_C_A_CO * d_c)
        candidatos.append(cand)
    sim = objetivo_lote_v8(sistema, fase, state, candidatos, context, pesos, s_ref, orden, objetivo, cfg)
    adm = (sim["support_score"] >= umbral) & sim["admisible_T"].astype(bool)
    adm.iloc[0] = True
    J_adm = sim["J"].where(adm)
    i_best = int(J_adm.idxmax())
    accion, J, fila = candidatos[i_best], float(sim.loc[i_best, "J"]), sim.loc[i_best]
    # Robustez del GN al peso w (E10-05 sensibilidad): la direccion del GN debe conservarse en los extremos del IC de w; si no, GN = historico
    rw = cfg.get("robustez_w")
    if rw and fase == "Reducción" and "tasa_gn_nm3_min" in opt and abs(accion["tasa_gn_nm3_min"] - accion_base["tasa_gn_nm3_min"]) > 1e-6:
        signo = np.sign(accion["tasa_gn_nm3_min"] - accion_base["tasa_gn_nm3_min"]); robusto = True
        for w_alt in rw:
            cands = []
            for v in np.linspace(*cajas["tasa_gn_nm3_min"], n_grid_gn):
                c = dict(accion); c["tasa_gn_nm3_min"] = float(v); cands.append(c)
            sim_w = objetivo_lote_v8(sistema, fase, state, [accion_base] + cands, context, pesos, s_ref, orden, objetivo, dict(cfg, w_sdi=w_alt))
            adm_w = (sim_w["support_score"] >= umbral) & sim_w["admisible_T"].astype(bool); adm_w.iloc[0] = True
            j = int(sim_w["J"].where(adm_w).idxmax())
            gn_w = float(accion_base["tasa_gn_nm3_min"]) if j == 0 else cands[j - 1]["tasa_gn_nm3_min"]
            if np.sign(gn_w - accion_base["tasa_gn_nm3_min"]) != signo:
                robusto = False; break
        if not robusto:
            accion = dict(accion); accion["tasa_gn_nm3_min"] = float(accion_base["tasa_gn_nm3_min"])
            sim2 = objetivo_lote_v8(sistema, fase, state, [accion_base, accion], context, pesos, s_ref, orden, objetivo, cfg)
            J, fila = float(sim2.loc[1, "J"]), sim2.loc[1]; sim = sim2; i_best = 1
    cambia = any(abs(accion[a] - accion_base[a]) > 1e-6 for a in opt)
    if cambia and cfg.get("exigir_ganancia_fisica", True):
        # Salvaguarda (dictamen §2/§6: los precios son placeholders): una recomendacion cuyo unico motor es el costo (la parte fisica
        # K·Δ(ln_sn_dep + w·ln_feo_ret) no mejora) se degrada a "mantener" y se reporta el motivo.
        fis = float(fila["J_sn"] + fila["J_feo"] - sim.loc[0, "J_sn"] - sim.loc[0, "J_feo"])
        if fis <= 1e-6:
            return dict(accion_base), float(sim.loc[0, "J"]), sim.loc[0], "mantener:solo_costo"
    return accion, J, fila, ("recomendar" if cambia else "mantener")


def curva_respuesta_v8(sistema: SistemaPrescriptivoV8, fase: str, orden: int, state: dict, action: dict, context: dict | None,
                       palanca: str, n: int = 25, pesos: dict | None = None, caja: bool = False, objetivo: str | None = None) -> pd.DataFrame:
    """J y predicciones a lo largo de una palanca (P5-P95 del orden, o la caja |Δ| ≤ 1 sd si `caja`)."""
    lim = sistema.limites_orden[(fase, int(orden))][palanca]
    lo, hi = (_caja_accion(sistema, fase, orden, action, palanca) if caja else (lim[0], lim[1]))
    grid = np.linspace(lo, hi, n)
    acciones = []
    for v in grid:
        a = dict(action); a[palanca] = float(v)
        if fase == "Fusión" and palanca == "tasa_feed_Carbon_kg_min" and sistema.config["lambda_lanza_constante"]:
            a["tasa_o2_nm3_min"] = max(0.0, float(action["tasa_o2_nm3_min"]) + O2_POR_KG_C_A_CO * (a[palanca] - float(action[palanca])))
        acciones.append(a)
    s_ref = float(simular_acciones_lote_v8(sistema, fase, state, [action], context, n_rep=1)["support_score"].iloc[0])
    sim = objetivo_lote_v8(sistema, fase, state, acciones, context, pesos, s_ref, orden, objetivo)
    sim.insert(0, "valor", grid); sim.insert(0, "palanca", palanca)
    return sim


# =============================================================================
# 6) Reportes por batch, politica cross-fitted, agregacion y evidencia
# =============================================================================

GRUPOS_R = {0: "R_temprano", 1: "R_temprano", 2: "R_tardio", 3: "R_tardio"}


def reporte_recomendaciones_batch_v8(sistema: SistemaPrescriptivoV8, df: pd.DataFrame, batch_id: str, pesos: dict | None = None,
                                     objetivo: str | None = None, config: dict | None = None) -> pd.DataFrame:
    df_base = _preparar_base(df)
    sub = df_base.loc[df_base["Batch"] == batch_id].sort_values("fecha_inicio")
    filas = []
    for _, row in sub.iterrows():
        fase = row["fase_proceso"]; orden = int(row["orden_escalon_fase"])
        state, accion_hist, context = fila_a_state_action_context(fase, row)
        estado_ok = not (any(pd.isna(accion_hist[a]) for a in ACCIONES_PRIMITIVAS_V8[fase]) or pd.isna(state.get("ley_sn_escoria_pct_prev"))
                         or pd.isna(state.get("m6_sn_inv_kg_prev")))
        if not estado_ok:
            filas.append({"Batch": batch_id, "fase": fase, "orden_escalon_fase": orden, "estado_recomendacion": "sin_recomendacion:estado"})
            continue
        try:
            s_h = float(simular_acciones_lote_v8(sistema, fase, state, [accion_hist], context, n_rep=1)["support_score"].iloc[0])
            sim_h = objetivo_lote_v8(sistema, fase, state, [accion_hist], context, pesos, s_h, orden, objetivo, config).iloc[0]
            accion_opt, J_o, sim_o, est = optimizar_accion_v8(sistema, fase, orden, state, accion_hist, context, pesos=pesos, objetivo=objetivo, config=config)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{batch_id} {fase} {orden}: {type(e).__name__}: {e}")
            continue
        fila = {"Batch": batch_id, "fase": fase, "orden_escalon_fase": orden, "estado_recomendacion": est,
                "m6_avance_prev": state.get("m6_avance_prev"), "ley_sn_escoria_pct_prev": state["ley_sn_escoria_pct_prev"],
                "real_sn_dep": row.get(TARGETS_V8["Reducción"]["sn_dep"]), "real_feo_ret": row.get(TARGETS_V8["Reducción"]["feo_ret"]),
                "temperatura_real": row.get("temperatura_horno_celsius"),
                "pred_T_historico": sim_h["pred_temperatura"], "pred_T_recomendado": sim_o["pred_temperatura"],
                "support_historico": sim_h["support_score"], "support_recomendado": sim_o["support_score"],
                "J_historico": float(sim_h["J"]), "J_recomendado": J_o, "uplift_J_kgSn": J_o - float(sim_h["J"])}
        for k in [c for c in sim_h.index if c.startswith("pred_") and c != "pred_temperatura"]:
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for k in [c for c in sim_h.index if c.endswith("_sd_rep")]:
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for k in ("J_sn", "J_feo", "J_costo", "J_incertidumbre", "J_soporte"):
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for a in ACCIONES_PRIMITIVAS_V8[fase]:
            fila[f"hist__{a}"] = accion_hist[a]; fila[f"rec__{a}"] = accion_opt[a]
        for k in DERIVADAS_REPORTE:
            fila[f"hist_derivada__{k}"] = float(sim_h[f"derivada__{k}"]); fila[f"rec_derivada__{k}"] = float(sim_o[f"derivada__{k}"])
        lim = sistema.limites_orden.get((fase, orden), {})
        for a in ACCIONES_OPTIMIZABLES_V8[fase]:
            if a in lim:
                fila[f"sd_orden__{a}"] = lim[a][2]
        filas.append(fila)
    return pd.DataFrame(filas)


def agregar_objetivo(rep: pd.DataFrame, config: dict | None = None) -> dict:
    """Descompone el objetivo fisico por escalon -> grupo (R temprano / tardio) -> fase -> batch en pp de KPI, accion historica y
    recomendada. Identidad telescopica: J_R(fase) = K·[ln(Sn_ini/Sn_fin) + w·ln(FeO_fin/FeO_ini)] = Σ_t J_t."""
    cfg = config or CONFIG_V8
    K, w = cfg["pp_kpi_por_unidad_sdi"], cfg["w_sdi"]
    out = {}
    red = rep["fase"] == "Reducción"
    rr = rep[red & rep["pred_sn_dep_hist"].notna()] if "pred_sn_dep_hist" in rep else rep.iloc[0:0]
    for suf in ("hist", "rec"):
        if len(rr):
            jR = K * (rr[f"pred_sn_dep_{suf}"] + w * rr[f"pred_feo_ret_{suf}"])
            grp = rr["orden_escalon_fase"].map(GRUPOS_R)
            for g in ("R_temprano", "R_tardio"):
                out[f"J_{g}_{suf}_pp"] = float(jR[grp == g].sum())
            out[f"J_R_{suf}_pp"] = float(jR.sum())
            out[f"dSn_{suf}"] = float(rr[f"pred_sn_dep_{suf}"].sum()); out[f"dFeO_{suf}"] = float(rr[f"pred_feo_ret_{suf}"].sum())
        else:
            out[f"J_R_{suf}_pp"] = np.nan
        fus = rep[(rep["fase"] == "Fusión") & rep.get("hist__tasa_feed_Carbon_kg_min", pd.Series(dtype=float)).notna()] if "hist__tasa_feed_Carbon_kg_min" in rep else rep.iloc[0:0]
        out[f"J_F_{suf}_pp"] = float(-cfg["pp_kpi_por_kg_carbon_F"] * (fus[f"{suf}__tasa_feed_Carbon_kg_min"] * fus["hist__duracion_plan_min"]).sum()) if len(fus) else 0.0
        out[f"J_batch_{suf}_pp"] = out[f"J_R_{suf}_pp"] + out[f"J_F_{suf}_pp"]
    out["uplift_batch_pp"] = out["J_batch_rec_pp"] - out["J_batch_hist_pp"]
    out["uplift_batch_kg"] = out["uplift_batch_pp"] * KG_SN_POR_PP_KPI
    return out


def recomendaciones_cross_fitted_v8(df: pd.DataFrame, etiqueta: str, n_folds: int = 5, log=print, pesos: dict | None = None,
                                    config: dict | None = None, objetivo: str | None = None, n_boot: int = 100) -> pd.DataFrame:
    """'0'..'n_folds-1' = fold de DEV (sistema entrenado SIN esos batches); 'lockbox' = sistema entrenado con todo DEV.
    n_boot > 0 para que el termino de incertidumbre dependa de la accion (dictamen §3)."""
    dev, lockbox, folds = mp4._folds_dev(df, n_folds)
    if etiqueta == "lockbox":
        sistema = entrenar_sistema_v8(df, n_boot=n_boot, config=config, objetivo=objetivo)
        batches, fold, es_lb = sorted(lockbox), -1, True
    else:
        k = int(etiqueta)
        sistema = entrenar_sistema_v8(df[df["Batch"].isin(dev - set(folds[k]))], pct_lockbox=0.0, n_boot=n_boot, config=config, objetivo=objetivo)
        batches, fold, es_lb = folds[k], k, False
    reps = []
    for i, b in enumerate(batches):
        rep = reporte_recomendaciones_batch_v8(sistema, df, b, pesos=pesos, objetivo=objetivo)
        if not rep.empty:
            rep["fold"], rep["es_lockbox"] = fold, es_lb
            reps.append(rep)
        if (i + 1) % 10 == 0:
            log(f"{etiqueta}: {i+1}/{len(batches)}")
    return pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()


def filtrar_solo_costo(escalones: pd.DataFrame) -> pd.DataFrame:
    """Aplica a posteriori la salvaguarda `exigir_ganancia_fisica` sobre una tabla de escalones ya calculada: las filas 'recomendar' sin
    ganancia fisica pasan a 'mantener:solo_costo' y su accion recomendada vuelve a la historica."""
    esc = escalones.copy()
    if "J_sn_rec" not in esc.columns:
        return esc
    fis = (esc["J_sn_rec"] + esc["J_feo_rec"]) - (esc["J_sn_hist"] + esc["J_feo_hist"])
    m = (esc["estado_recomendacion"] == "recomendar") & (fis <= 1e-6)
    esc.loc[m, "estado_recomendacion"] = "mantener:solo_costo"
    for c in [c for c in esc.columns if c.startswith("rec__")]:
        esc.loc[m, c] = esc.loc[m, "hist" + c[3:]]
    for k in ("pred_sn_dep", "pred_feo_ret", "pred_dT", "sn_dep_sd_rep", "feo_ret_sd_rep", "dT_sd_rep", "J_sn", "J_feo", "J_costo", "J_incertidumbre", "J_soporte"):
        if f"{k}_rec" in esc.columns:
            esc.loc[m, f"{k}_rec"] = esc.loc[m, f"{k}_hist"]
    if "J_recomendado" in esc.columns:
        esc.loc[m, "J_recomendado"] = esc.loc[m, "J_historico"]; esc.loc[m, "uplift_J_kgSn"] = 0.0
    if "pred_T_recomendado" in esc.columns:
        esc.loc[m, "pred_T_recomendado"] = esc.loc[m, "pred_T_historico"]; esc.loc[m, "support_recomendado"] = esc.loc[m, "support_historico"]
    return esc


def metricas_por_batch_v8(df: pd.DataFrame, escalones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: uplift, adherencia (distancia |Δ|/sd) y DIRECCION (Δ/sd, con signo) por palanca y fase, KPIs y controles."""
    esc = escalones[escalones["estado_recomendacion"].isin(["recomendar", "mantener", "mantener:solo_costo"])].copy()
    esc["fold"] = escalones["fold"]; esc["es_lockbox"] = escalones["es_lockbox"]
    dev, _ = mp.split_dev_lockbox(df)
    df_base = _preparar_base(df)
    bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
    filas = []
    for b, rep in esc.groupby("Batch"):
        fila = {"Batch": b, "n_escalones": int(len(rep)), "uplift_J_total_kg": float(rep["uplift_J_kgSn"].sum()),
                "pct_escalones_con_mejora": float((rep["uplift_J_kgSn"] > 1.0).mean() * 100),
                "es_lockbox": bool(rep["es_lockbox"].iloc[0]), "fold": int(rep["fold"].iloc[0])}
        fila.update(agregar_objetivo(rep))
        dists = []
        for fase in FASES:
            r = rep[rep["fase"] == fase]
            if r.empty:
                continue
            d_f = []
            for a in ACCIONES_OPTIMIZABLES_V8[fase]:
                sd = r[f"sd_orden__{a}"].replace(0, np.nan)
                dd = ((r[f"hist__{a}"] - r[f"rec__{a}"]).abs() / sd)
                fila[f"dist_{fase}_{a}"] = float(dd.mean()); fila[f"dir_{fase}_{a}"] = float(((r[f"rec__{a}"] - r[f"hist__{a}"]) / sd).mean())
                for g, ordenes in (("temprano", [0, 1]), ("tardio", [2, 3])):
                    rg = r[r["orden_escalon_fase"].isin(ordenes)]
                    if fase == "Reducción" and len(rg):
                        fila[f"dir_R_{g}_{a}"] = float(((rg[f"rec__{a}"] - rg[f"hist__{a}"]) / rg[f"sd_orden__{a}"].replace(0, np.nan)).mean())
                d_f.append(dd.mean())
            fila[f"dist_{'F' if fase == 'Fusión' else 'R'}"] = float(np.nanmean(d_f)); dists += d_f
        fila["dist_total"] = float(np.nanmean(dists))
        g = df.loc[df["Batch"] == b]; gF = g.loc[g["fase_proceso"] == "Fusión"]
        m_, d_, p_ = (float(g[c].iloc[0]) for c in ("sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"))
        tot = m_ + d_ + p_; feed_sn = float(g["feed_Sn_kgf"].sum())
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


CONTROLES_KPI_V8 = list(mp5.CONTROLES_KPI_V5)
evidencia_politica_v8 = mp5.evidencia_politica_v5   # OLS HC3 + controles + permutacion sobre la tabla por batch (dist_* y dir_*)


def uplift_propagado(por_batch: pd.DataFrame, muestras_Kw: pd.DataFrame | None = None, factor_theta: tuple[float, float] = (0.6, 1.4),
                     n: int = 2000, seed: int = 0) -> dict:
    """Uplift por batch (pp de KPI) con IC que propaga (K, w) (muestras bootstrap del anclaje: columnas K_sn, K_feo) y un factor de
    escala de theta (uniforme en `factor_theta`, aproximacion del IC relativo de los theta identificados). Devuelve mediana e IC."""
    rng = np.random.default_rng(seed)
    dSn = (por_batch["dSn_rec"] - por_batch["dSn_hist"]).to_numpy(dtype=float)
    dFeO = (por_batch["dFeO_rec"] - por_batch["dFeO_hist"]).to_numpy(dtype=float)
    ok = np.isfinite(dSn) & np.isfinite(dFeO); dSn, dFeO = dSn[ok], dFeO[ok]
    if muestras_Kw is None:
        Ks = np.full(n, CONFIG_V8["pp_kpi_por_unidad_sdi"]); Kf = Ks * CONFIG_V8["w_sdi"]
    else:
        idx = rng.integers(0, len(muestras_Kw), n)
        Ks, Kf = muestras_Kw["K_sn"].to_numpy()[idx], muestras_Kw["K_feo"].to_numpy()[idx]
    f = rng.uniform(*factor_theta, n)
    ups = []
    for i in range(n):
        ib = rng.integers(0, len(dSn), len(dSn))
        ups.append(np.mean(f[i] * (Ks[i] * dSn[ib] + Kf[i] * dFeO[ib])))
    ups = np.array(ups)
    return {"n_batches": int(len(dSn)), "uplift_medio_pp_puntual": float(np.mean(CONFIG_V8["pp_kpi_por_unidad_sdi"] * (dSn + CONFIG_V8["w_sdi"] * dFeO))),
            "uplift_medio_pp_mediana": float(np.median(ups)), "ic_lo": float(np.percentile(ups, 2.5)), "ic_hi": float(np.percentile(ups, 97.5)),
            "prob_positivo": float((ups > 0).mean())}


def evaluacion_cruzada(df: pd.DataFrame, escalones: pd.DataFrame, n_folds: int = 5, seed: int = SEED, log=print) -> pd.DataFrame:
    """Anti-maldicion del optimizador: la accion recomendada para el fold k (elegida con el modelo entrenado sin k) se evalua con
    (a) el PLM entrenado sin el fold k' != k (que SI vio el fold k) y (b) un HGB directo estado+palancas entrenado sin el fold k.
    Reporta, por escalon de Reduccion, Δ predicho (rec − hist) segun el optimizador, el evaluador cruzado y el HGB."""
    dev, lockbox, folds = mp4._folds_dev(df, n_folds)
    df_base = _preparar_base(df)
    esc = escalones[(escalones["fase"] == "Reducción") & (escalones["estado_recomendacion"] == "recomendar")].copy()
    out = []
    fase = "Reducción"
    for k in sorted(esc["fold"].unique()):
        if k < 0:
            continue
        k2 = (k + 1) % n_folds
        sis2 = entrenar_sistema_v8(df[df["Batch"].isin(dev - set(folds[k2]))], pct_lockbox=0.0, n_boot=0)
        sub = df_base[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev - set(folds[k]))]
        hgb = {}
        for clave in ("sn_dep", "feo_ret"):
            S, A, tg = ESTADO_V8[(fase, clave)], PALANCAS_V8[(fase, clave)], TARGETS_V8[fase][clave]
            d = sub.dropna(subset=S + A + [tg])
            hgb[clave] = (mp4._hgb_nuisance(seed, 300).fit(d[S + A], d[tg]), S + A)
        ek = esc[esc["fold"] == k]
        for b, rep in ek.groupby("Batch"):
            filas_b = df_base[df_base["Batch"] == b]
            for _, e in rep.iterrows():
                row = filas_b[(filas_b["fase_proceso"] == fase) & (filas_b["orden_escalon_fase"] == e["orden_escalon_fase"])]
                if row.empty:
                    continue
                state, a_h, ctx = fila_a_state_action_context(fase, row.iloc[0])
                a_r = dict(a_h); a_r.update({a: e[f"rec__{a}"] for a in ACCIONES_PRIMITIVAS_V8[fase]})
                sim = simular_acciones_lote_v8(sis2, fase, state, [a_h, a_r], ctx, n_rep=1)
                filas = [_fila_v8(mp3.recalcular_derivadas_v3(fase, state, a, ctx)) for a in (a_h, a_r)]
                fila = {"Batch": b, "fold": k, "orden_escalon_fase": e["orden_escalon_fase"],
                        "d_sn_dep_opt": e["pred_sn_dep_rec"] - e["pred_sn_dep_hist"], "d_feo_ret_opt": e["pred_feo_ret_rec"] - e["pred_feo_ret_hist"],
                        "d_sn_dep_cruzado": float(sim["pred_sn_dep"].iloc[1] - sim["pred_sn_dep"].iloc[0]),
                        "d_feo_ret_cruzado": float(sim["pred_feo_ret"].iloc[1] - sim["pred_feo_ret"].iloc[0])}
                for clave, (h, X) in hgb.items():
                    Xd = pd.DataFrame([[f.get(c, np.nan) for c in X] for f in filas], columns=X)
                    p = h.predict(Xd)
                    fila[f"d_{clave}_hgb"] = float(p[1] - p[0])
                out.append(fila)
        log(f"evaluacion cruzada fold {k}: {len(out)} escalones acumulados")
    return pd.DataFrame(out)
