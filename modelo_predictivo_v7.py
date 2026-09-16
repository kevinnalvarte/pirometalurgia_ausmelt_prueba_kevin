"""Modelo prescriptivo v7 de Lingo Smelter (iteracion 9, 2026-09-16): el enfoque v6 (PLM por fase con signos de teoria sobre
targets log telescopicos y masa de escoria por cierre fisico) con las tres mejoras que sobrevivieron a la iteracion 9 y sin las
que no (ver `experimentos/v7/ITERACION_9_diseno.md`, memos E9-00..E9-06 y hallazgos §20):

1. ESTADO PARSIMONIOSO de Reduccion (E9-06, `libre_k16`): 16 features en vez de 38 con el mismo R2 OOF (agotamiento 0.767 vs
   0.765; retencion de FeO 0.333 vs 0.329), lockbox equivalente, carbon identificado (Cx_sn +0.18 [0.05, 0.29]) y estable en
   2 de 3 tercios (FULL: 1 de 3), mejor walk-forward. Fusion conserva el estado curado v6 (25) y se corrige el defecto de que F1
   nunca se modelaba (gradiente de T indefinido en el segundo escalon del batch -> se imputa 0).
2. OBJETIVO CON LEGADO (E9-00b): la retencion de FeO de un batch tambien sube el KPI del batch SIGUIENTE (+10.4 pp/unidad, p 2e-4,
   placebo temporal nulo, mediado por el IRF terminal = talon quimico). Anclaje sobre KPI_actual + KPI_siguiente (OLS HC3, DEV
   n=297): K = 3.522 pp/unidad de ln_sn_dep, K_FeO = 27.68 -> w = 7.86 [5.14, 13.04]. Un solo (K, w) por fase: la agregacion
   escalon -> grupo -> fase es exactamente telescopica (E9-00 §1) y la heterogeneidad por grupo es atenuacion por ruido (§2).
3. POLITICA: miope por escalon (v5/v6) o con horizonte (E9-03, rollout sobre los escalones restantes de Reduccion) segun el
   veredicto de E9-03 (`POLITICA_V7`); ambas usan la misma J.

Lo que NO cambia (y por que): la forma PLM (E9-01: estructural/cinetico 0.34 vs 0.77; E9-05: ningun algoritmo supera al PLM),
las palancas (carbon, GN, exceso O2, aire; la cal NO es palanca: E9-02, el efecto de E7-02 era artefacto del trazador), el
anclaje de Fusion (conservacion de Sn beta_F = -0.58, debil; E9-00c y E9-04) y el KPI refinado.

Implementacion: capa fina sobre `modelo_predictivo_v6` que RECONFIGURA sus diccionarios en el proceso (`activar_v7()`, llamado
al importar): las funciones de v6 leen ESTADO_V6/CONFIG_V6/_preparar_base por nombre en tiempo de ejecucion, asi que tras
importar este modulo `mp6.*` se comporta como v7. Para reproducir v6 puro hay que usar un proceso sin importar v7.

Uso tipico:
    import modelo_predictivo_v7 as mp7
    df = mp7.cargar_df()                          # cache experimentos/cache/df_v6.pkl
    print(mp7.tabla_validacion_v7(df))
    sistema = mp7.entrenar_sistema_v7(df)
    print(mp7.tabla_efectos_control_v7(sistema))
    rep = mp7.reporte_recomendaciones_batch_v7(sistema, df, "AP0350")
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent
for _p in (_RAIZ / "experimentos" / "v6", _RAIZ / "experimentos" / "v5", _RAIZ / "experimentos" / "balances"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import modelo_predictivo_v6 as mp6  # noqa: E402

FASES = mp6.FASES
SEED = mp6.SEED
KG_SN_POR_PP_KPI = mp6.KG_SN_POR_PP_KPI
CACHE_V6 = _RAIZ / "experimentos" / "cache" / "df_v6.pkl"

# ----------------------------------------------------------------------------- 1) estado parsimonioso de Reduccion (E9-06 libre_k16)
ESTADO_R_V7: list[str] = [
    "ley_sn_escoria_pct_prev",          # %Sn de la escoria al cierre de t-1: fuerza motriz de SnO2 + 2C (importancia 0.32)
    "ley_feo_escoria_pct_prev",         # %FeO: competidor por el reductor (FeO + C)
    "ley_sio2_escoria_pct_prev",        # matriz silicatada (viscosidad / actividad de SnO)
    "ley_cao_escoria_pct_prev",         # basicidad (con SiO2)
    "ratio_sn_feo_prev",                # selectividad termodinamica disponible Sn/FeO
    "interaccion_Sn_x_FeO_prev",        # cinetica bimolecular (producto de concentraciones)
    "grad_temperatura_horno_celsius_prev",  # tendencia termica del bano
    "termocupla_media_celsius_prev",    # temperatura del bano (proxy mas estable que la del horno)
    "posicion_vertical_lanza_mm_prev",  # agitacion / inmersion de la lanza
    "cum_feed_Carbon_kg_prev",          # historia de dosificacion de carbon (confusion del operador)
    "cum_aire_nm3_prev",                # historia de aire de lanza (oxidacion acumulada)
    "relacion_C_cum_Sn_cum_prev",       # C/Sn acumulado: cuanto reductor ya se ha gastado por Sn cargado
    "m6_sn_inv_kg_prev",                # inventario de Sn en la escoria (kg, cierre fisico)
    "m6_feo_inv_kg_prev",               # inventario de FeO (kg)
    "espesor_ladrillo_norm_mm",         # desgaste del refractario = campana (CONTEXT)
    "m6_resto_frac_prev",               # fraccion no reducible de la escoria (cierre fisico)
]

# ----------------------------------------------------------------------------- 2) objetivo con legado (E9-00b)
CONFIG_V7: dict = dict(mp6.CONFIG_V6,
                       w_sdi=7.86,                    # K_FeO / K_Sn sobre KPI_actual + KPI_siguiente (DEV n=297; IC [5.14, 13.04])
                       pp_kpi_por_unidad_sdi=3.522,   # pp de KPI (dos batches) por unidad de ln_sn_dep de Reduccion
                       pp_kpi_por_unidad_sn_dep_F=-0.58)   # conservacion de Sn en Fusion (v6; E9-00c: nivel terminal equivalente)
CONFIG_V7_SIN_LEGADO: dict = dict(mp6.CONFIG_V6)      # variante: solo el batch actual (w 6.09, K 2.893) para sensibilidad

# ----------------------------------------------------------------------------- 3) politica
POLITICA_V7 = "miope"   # "miope" (v5/v6) | "horizonte" (E9-03); se fija tras el veredicto de E9-03

_ESTADO_V6_ORIGINAL = {k: list(v) for k, v in mp6.ESTADO_V6.items()}
_CONFIG_V6_ORIGINAL = dict(mp6.CONFIG_V6)
_PREPARAR_BASE_V6 = mp6._preparar_base


def _preparar_base_v7(df: pd.DataFrame) -> pd.DataFrame:
    """Como v6 pero imputa 0 al gradiente de temperatura previo en el segundo escalon del batch (F1), que es
    estructuralmente NaN (no hay dos escalones anteriores) y dejaba a F1 fuera de todos los modelos de Fusion (E9-04 §1)."""
    df_base = _PREPARAR_BASE_V6(df)
    g = "grad_temperatura_horno_celsius_prev"
    if g in df_base.columns:
        m = (df_base["fase_proceso"] == "Fusión") & (df_base["orden_escalon_fase"] == 1) & df_base[g].isna()
        df_base.loc[m, g] = 0.0
    return df_base


def activar_v7(politica: str = POLITICA_V7, con_legado: bool = True, estado_parsimonioso: bool = True) -> None:
    """Reconfigura `modelo_predictivo_v6` en el proceso: estado de Reduccion, anclajes del objetivo, base con F1."""
    global POLITICA_V7
    for k in list(mp6.ESTADO_V6):
        mp6.ESTADO_V6[k] = list(_ESTADO_V6_ORIGINAL[k])
    if estado_parsimonioso:
        for clave in ("sn_dep", "feo_ret"):
            mp6.ESTADO_V6[("Reducción", clave)] = list(ESTADO_R_V7)
    mp6.CONFIG_V6.clear()
    mp6.CONFIG_V6.update(CONFIG_V7 if con_legado else CONFIG_V7_SIN_LEGADO)
    mp6._preparar_base = _preparar_base_v7
    mp6._verificar_seguridad()
    POLITICA_V7 = politica


activar_v7()


def desactivar_v7() -> None:
    """Restaura v6 (para comparaciones en el mismo proceso)."""
    for k in list(mp6.ESTADO_V6):
        mp6.ESTADO_V6[k] = list(_ESTADO_V6_ORIGINAL[k])
    mp6.CONFIG_V6.clear(); mp6.CONFIG_V6.update(_CONFIG_V6_ORIGINAL)
    mp6._preparar_base = _PREPARAR_BASE_V6


# ----------------------------------------------------------------------------- datos
def cargar_df() -> pd.DataFrame:
    if CACHE_V6.exists():
        return pd.read_pickle(CACHE_V6)
    return mp6.agregar_columnas_v6(pd.read_pickle(_RAIZ / "experimentos" / "cache" / "df_v3.pkl"))


# ----------------------------------------------------------------------------- re-exports (se comportan como v7 tras activar_v7)
ESTADO_V7 = mp6.ESTADO_V6
PALANCAS_V7 = mp6.PALANCAS_V6
TARGETS_V7 = mp6.TARGETS_V6
SIGNO_TEORICO_V7 = mp6.SIGNO_TEORICO_V6
ACCIONES_OPTIMIZABLES_V7 = mp6.ACCIONES_OPTIMIZABLES_V6
ACCIONES_PRIMITIVAS_V7 = mp6.ACCIONES_PRIMITIVAS_V6
PESOS_V7_DEFAULT = mp6.PESOS_V6_DEFAULT
SistemaPrescriptivoV7 = mp6.SistemaPrescriptivoV6
es_segura_v7 = mp6.es_segura_v6
fila_a_state_action_context = mp6.fila_a_state_action_context
entrenar_sistema_v7 = mp6.entrenar_sistema_v6
tabla_efectos_control_v7 = mp6.tabla_efectos_control_v6
tabla_validacion_v7 = mp6.tabla_validacion_v6
predicciones_oof_y_lockbox_v7 = mp6.predicciones_oof_y_lockbox_v6
simular_acciones_lote_v7 = mp6.simular_acciones_lote_v6
objetivo_lote_v7 = mp6.objetivo_lote_v6
optimizar_accion_v7 = mp6.optimizar_accion_v6
curva_respuesta_v7 = mp6.curva_respuesta_v6
reporte_recomendaciones_batch_v7 = mp6.reporte_recomendaciones_batch_v6
resumen_reporte_batch_v7 = mp6.resumen_reporte_batch_v6
recomendaciones_cross_fitted_v7 = mp6.recomendaciones_cross_fitted_v6
metricas_por_batch_v7 = mp6.metricas_por_batch_v6
evidencia_politica_v7 = mp6.evidencia_politica_v6
CONTROLES_KPI_V7 = mp6.CONTROLES_KPI_V6


def kg_sn_por_unidad_sdi() -> float:
    return mp6.CONFIG_V6["pp_kpi_por_unidad_sdi"] * KG_SN_POR_PP_KPI


# ----------------------------------------------------------------------------- agregacion escalon -> grupo -> fase -> batch
GRUPOS_R = {0: "R_temprano", 1: "R_temprano", 2: "R_tardio", 3: "R_tardio"}
GRUPOS_F = {1: "F_inicial", 2: "F_inicial", 3: "F_inicial", 4: "F_final", 5: "F_final", 6: "F_final"}


def agregar_objetivo(rep: pd.DataFrame, config: dict | None = None) -> dict:
    """Descompone el objetivo fisico de un reporte por escalon (`reporte_recomendaciones_batch_v7`) en escalon -> grupo -> fase ->
    batch, en pp de KPI (dos batches: actual + siguiente) y kg de Sn equivalente, para la accion historica y la recomendada.
    Identidad: J_R(fase) = K·[ln(Sn_ini/Sn_fin) + w·ln(FeO_fin/FeO_ini)] = sum de los J_t de sus escalones (telescopica)."""
    cfg = config or mp6.CONFIG_V6
    K, w, bF = cfg["pp_kpi_por_unidad_sdi"], cfg["w_sdi"], cfg["pp_kpi_por_unidad_sn_dep_F"]
    out = {}
    for suf in ("hist", "rec"):
        red = rep["fase"] == "Reducción"; fus = rep["fase"] == "Fusión"
        jR = K * (rep.loc[red, f"pred_sn_dep_{suf}"] + w * rep.loc[red, f"pred_feo_ret_{suf}"])
        jF = bF * rep.loc[fus, f"pred_sn_dep_{suf}"]
        grp = rep.loc[red, "orden_escalon_fase"].map(GRUPOS_R)
        for gname in ("R_temprano", "R_tardio"):
            out[f"J_{gname}_{suf}_pp"] = float(jR[grp == gname].sum())
        grpF = rep.loc[fus, "orden_escalon_fase"].map(GRUPOS_F)
        for gname in ("F_inicial", "F_final"):
            out[f"J_{gname}_{suf}_pp"] = float(jF[grpF == gname].sum())
        out[f"J_R_{suf}_pp"] = float(jR.sum()); out[f"J_F_{suf}_pp"] = float(jF.sum())
        out[f"J_batch_{suf}_pp"] = out[f"J_R_{suf}_pp"] + out[f"J_F_{suf}_pp"]
        out[f"J_batch_{suf}_kg"] = out[f"J_batch_{suf}_pp"] * KG_SN_POR_PP_KPI
    out["uplift_batch_pp"] = out["J_batch_rec_pp"] - out["J_batch_hist_pp"]
    out["uplift_batch_kg"] = out["uplift_batch_pp"] * KG_SN_POR_PP_KPI
    return out
