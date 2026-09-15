"""Construccion del dataset analitico de Lingo Smelter."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RUTA_CONFIG_LIMPIEZA_DEFECTO = Path(__file__).parent / "config_limpieza.yaml"

RENOMBRES = {
    "Fase del proceso": "fase_proceso",
    "Fecha inicio real": "fecha_inicio",
    "Fecha final real": "fecha_final",
    "Escalón": "escalon_idx",
    "duracion plan": "duracion_plan_min",
    # --- Alimentacion agregada (Sn fino y masa humeda) -------------------------------
    # TMF Sn total = suma EXACTA de los tmf de Sn de las tolvas 1,3,4,5,7 (verificado
    # 3982/3982 filas); TMH total = suma EXACTA de las tmh de esas mismas tolvas.
    "TMF Sn total F+ R (Kg)": "feed_Sn_kgf",
    "TMH tolva: 3+4+5+7+1 (Kg)": "feed_total_kgh",
    # --- Alimentacion POR TOLVA (masa humeda, kg del escalon) -------------------------
    "tmh tolva 1 concentrado": "feed_conc_t1_kgh",          # Concentrado de Sn (carga principal)
    "tmh tolva 3 alimentacin (Kg)": "feed_reciclo_t3_kgh",  # Alimentacion secundaria Sn/Fe (reciclo)
    "tmh tolva 4 Hierro WF (Kg)": "feed_Fe_kgh",            # Mineral de hierro WF (tolva 4)
    "tmh tolva 5 Dross Fierro": "feed_dross_Fe_kgh",        # Dross de fierro (tolva 5)
    "tmh tolva 7 alimentador": "feed_pellets_t7_kgh",       # Pellets / humos (tolva 7)
    # feed_CaO_kgh = tmh tolva 6 (cal, CaO). ATENCION: la columna "CaO total (kg)" del
    # Excel es EXACTAMENTE tolva 6 + tolva 3 (verificado 3982/3982 filas), es decir
    # suma el reciclo Sn/Fe a la cal; las sesiones anteriores a 2026-09-13 (noche)
    # usaban esa columna como feed_CaO_kgh. Se conserva como
    # feed_CaO_total_excel_kg solo para trazabilidad, NO como fundente.
    "Kgmh tolva 6 Oxido de Calcio": "feed_CaO_kgh",
    "CaO total (kg)": "feed_CaO_total_excel_kg",
    # Carbon: total F+R (= tolva 2 fusion + tolva 2 reduccion, salvo redondeos) y su
    # desglose por etapa de dosificacion.
    "Carbon total F+R (Kg)": "feed_Carbon_kgh",
    "tmh tolva 2 Carbon": "feed_Carbon_fusion_kgh",
    "tmh carbón reducción tolva 2 (kg/h)": "feed_Carbon_reduccion_kgh",
    # --- Sn fino POR TOLVA (kg de Sn del escalon). Verificado: tmf_i = tmh_i * ley_i, con
    # ley_i CONSTANTE dentro de cada batch (std intra-batch = 0) y variable entre batches:
    # es el ensayo de Sn de cada corriente asignado al batch, no un ensayo por escalon.
    "tmf Sn tolva 1 concentrado": "sn_conc_t1_kgf",
    "tmf Sn tolva 3 alimentacion (Kg)": "sn_reciclo_t3_kgf",
    "tmf Sn tolva 4 Hierro WF (Kg)": "sn_Fe_t4_kgf",
    "tmf Sn tolva 5 Dross Fierro": "sn_dross_Fe_t5_kgf",
    "tmf Sn tolva 7 alimentador": "sn_pellets_t7_kgf",
    # --- Salidas del batch (unicas por batch, repetidas en todos sus escalones) --------
    "Sn al Metal Crudo (t)": "sn_en_metal_crudo_batch_t",
    "Sn al Dross Fe (t)": "sn_en_dross_fe_batch_t",
    "Sn al Polvo de Fundición (t)": "sn_en_polvo_fundicion_batch_t",
    "Generación de Sn Crudo (t)": "sn_crudo_generado_batch_t",
    "Rendimiento HA": "rendimiento_ha_batch",
    # --- Contexto de campana / estado termico del refractario ------------------------
    "Espesor de ladrillo normalizado": "espesor_ladrillo_norm_mm",
    "H1-Termocupla 100 mm - NO": "termocupla_100mm_NO_celsius",
    "H1-Termocupla 50 mm - N": "termocupla_50mm_N_celsius",
    "H1-Termocupla 100 mm - E": "termocupla_100mm_E_celsius",
    "H1-Termocupla 50 mm - O": "termocupla_50mm_O_celsius",
    "H1-Termocupla 75 mm - SE": "termocupla_75mm_SE_celsius",
    "H1-Termocupla 150 mm - NE": "termocupla_150mm_NE_celsius",
    "H1-Termocupla 100 mm - SO": "termocupla_100mm_SO_celsius",
    "H1-Termocupla 0 mm - SO": "termocupla_0mm_SO_celsius",
    # --- Lectura de planta de la estequiometria/enriquecimiento de la lanza -----------
    # Verificado: correlacion 1.000 con exceso_o2_combustion_pct (= estequiometria - 100)
    # y con oxygen_enrichment_pct de feature_engineering.py; se conservan como
    # validacion cruzada, no como features adicionales.
    "HA-01 Estequiometría (%)": "estequiometria_lanza_planta_pct",
    "HA-01 Enriquecimiento de O2": "enriquecimiento_o2_lanza_planta_pct",
    "%Sn": "ley_sn_escoria_pct",
    "%Fe": "ley_fe_total_escoria_pct",
    "%SiO2": "ley_sio2_escoria_pct",
    "%CaO": "ley_cao_escoria_pct",
    "Alumina": "ley_al2o3_escoria_pct",
    "%MgO": "ley_mgo_escoria_pct",
    "%As": "ley_as_escoria_pct",
    "%Sb": "ley_sb_escoria_pct",
    "Cromo": "ley_cr_escoria_pct",
    "Gas Natural total (Nm3)": "volumen_gas_natural_inyectado_lanza_escalon_nm3",
    "O2 total (Nm3)": "volumen_o2_inyectado_lanza_escalon_nm3",
    "Aire total (Nm3)": "volumen_aire_inyectado_lanza_escalon_nm3",
    "Tiro de horno (%)": "tiro_horno_pct",
    "Presión ingreso gas natural lanza Ausmelt (kPa)": "presion_suministro_gn_lanza_kpa",
    "Presión oxígeno Lanza Ausmelt (kPa)": "presion_suministro_o2_lanza_kpa",
    "H1 - Tip Pressure (kPa)": "presion_punta_lanza_kpa",
    "Gas Back Pressure (kPa)": "contrapresion_gn_lanza_kpa",
    "Lance Air Back Pressure (kPa)": "contrapresion_aire_lanza_kpa",
    "Horno 1 Lance Height (Posición Lanza) (mm)": "posicion_vertical_lanza_mm",
    "T °C cara interior del horno": "temperatura_horno_celsius",
    "Temperatura entrada gases BHF (antes bag house) °C": "temperatura_gas_pre_bhf_celsius",
    "Temperatura entrada Gases BHF (antes cuchilla) °C": "temperatura_gas_pre_cuchilla_celsius",
}

COLUMNAS_TERMOCUPLAS = [
    "termocupla_100mm_NO_celsius", "termocupla_50mm_N_celsius", "termocupla_100mm_E_celsius",
    "termocupla_50mm_O_celsius", "termocupla_75mm_SE_celsius", "termocupla_150mm_NE_celsius",
    "termocupla_100mm_SO_celsius", "termocupla_0mm_SO_celsius",
]

# Columnas de alimentacion por tolva (masa humeda) y su Sn fino asociado, en el mismo orden.
COLUMNAS_FEED_POR_TOLVA = {
    "feed_conc_t1_kgh": "sn_conc_t1_kgf",
    "feed_reciclo_t3_kgh": "sn_reciclo_t3_kgf",
    "feed_Fe_kgh": "sn_Fe_t4_kgf",
    "feed_dross_Fe_kgh": "sn_dross_Fe_t5_kgf",
    "feed_pellets_t7_kgh": "sn_pellets_t7_kgf",
}

CAMPOS_BASE = [
    "Batch", "fase_proceso", "fecha_inicio", "fecha_final", "escalon_idx", "duracion_plan_min",
    "feed_Sn_kgf", "feed_Fe_kgh", "feed_dross_Fe_kgh", "feed_CaO_kgh", "feed_CaO_total_excel_kg",
    "feed_Carbon_kgh", "feed_Carbon_fusion_kgh", "feed_Carbon_reduccion_kgh", "feed_total_kgh",
    "feed_conc_t1_kgh", "feed_reciclo_t3_kgh", "feed_pellets_t7_kgh",
    "sn_conc_t1_kgf", "sn_reciclo_t3_kgf", "sn_Fe_t4_kgf", "sn_dross_Fe_t5_kgf", "sn_pellets_t7_kgf",
    "espesor_ladrillo_norm_mm", *COLUMNAS_TERMOCUPLAS,
    "estequiometria_lanza_planta_pct", "enriquecimiento_o2_lanza_planta_pct",
    "temperatura_horno_celsius", "tiro_horno_pct",
    "volumen_gas_natural_inyectado_lanza_escalon_nm3",
    "volumen_o2_inyectado_lanza_escalon_nm3",
    "volumen_aire_inyectado_lanza_escalon_nm3",
    "presion_suministro_gn_lanza_kpa", "presion_suministro_o2_lanza_kpa",
    "contrapresion_gn_lanza_kpa", "contrapresion_aire_lanza_kpa",
    "presion_punta_lanza_kpa", "posicion_vertical_lanza_mm",
    "ley_sn_escoria_pct", "ley_fe_total_escoria_pct",
    "ley_sio2_escoria_pct", "ley_cao_escoria_pct", "ley_al2o3_escoria_pct",
    "ley_mgo_escoria_pct", "ley_sb_escoria_pct", "ley_cr_escoria_pct", "ley_as_escoria_pct",
    "temperatura_gas_pre_bhf_celsius", "temperatura_gas_pre_cuchilla_celsius",
    "sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t",
    "sn_crudo_generado_batch_t", "rendimiento_ha_batch",
]


def _duracion_plan_a_minutos(serie: pd.Series) -> pd.Series:
    """Convierte 'duracion plan' (texto 'HH:MM:SS', timedelta o datetime.time) a minutos."""
    def _uno(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, pd.Timedelta):
            return v.total_seconds() / 60
        if hasattr(v, "hour") and hasattr(v, "minute") and not hasattr(v, "year"):
            return v.hour * 60 + v.minute + v.second / 60
        try:
            return pd.to_timedelta(str(v)).total_seconds() / 60
        except (ValueError, TypeError):
            return np.nan
    return serie.map(_uno).astype(float)



# =============================================================================
# Limpieza de datos (rangos fisicos gobernados por config_limpieza.yaml)
# =============================================================================
# Funcion independiente de construir_dataset: se invoca sobre cualquier DataFrame
# con columnas equivalentes, y se puede omitir su llamada para dejar los datos
# sin limpiar (no queda "enganchada" al resto del pipeline).

def limpiar_dataset(
    df: pd.DataFrame,
    ruta_config: str | Path = RUTA_CONFIG_LIMPIEZA_DEFECTO,
    devolver_reporte: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Reemplaza por NaN los valores fuera de los rangos fisicos definidos en `config_limpieza.yaml`.

    Nunca elimina filas. Es independiente de `construir_dataset`, por lo que se puede
    invocar sobre cualquier DataFrame con columnas equivalentes, o dejar de llamarla
    para trabajar con los datos sin limpiar.
    """
    with open(Path(ruta_config), "r", encoding="utf-8") as archivo:
        rangos_fisicos = yaml.safe_load(archivo).get("rangos_fisicos") or {}

    df = df.copy()
    registros = []
    for columna, limites in rangos_fisicos.items():
        if columna not in df.columns:
            continue
        minimo, maximo = limites.get("min"), limites.get("max")
        fuera_rango = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera_rango |= df[columna] < minimo
        if maximo is not None:
            fuera_rango |= df[columna] > maximo
        n_afectados = int(fuera_rango.sum())
        if n_afectados:
            df.loc[fuera_rango, columna] = np.nan
            registros.append({"columna": columna, "rango_valido": f"[{minimo}, {maximo}]", "n_valores_a_nan": n_afectados})

    reporte = pd.DataFrame(registros, columns=["columna", "rango_valido", "n_valores_a_nan"])
    if not reporte.empty:
        print(f"limpiar_dataset: {reporte['n_valores_a_nan'].sum()} valor(es) marcados como NaN "
              f"en {len(reporte)} columna(s) (detalle en el reporte devuelto).")

    return (df, reporte) if devolver_reporte else df

def construir_dataset(
    ruta_excel: str | Path,
    hoja: str = "Datos",
    clean_mode: bool = False,
    ruta_config_limpieza: str | Path = RUTA_CONFIG_LIMPIEZA_DEFECTO,
) -> pd.DataFrame:
    """Lee, ordena y renombra los datos base para el analisis.

    Si `clean_mode` es True, aplica `limpiar_dataset` (rangos fisicos de
    `config_limpieza.yaml`) antes de devolver el resultado; si es False, devuelve
    los datos sin limpiar.
    """
    df = pd.read_excel(Path(ruta_excel), sheet_name=hoja)
    df = df.rename(columns=RENOMBRES).sort_values(["Batch", "fecha_inicio"]).copy()
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"])
    df["fecha_final"] = pd.to_datetime(df["fecha_final"])
    df["duracion_plan_min"] = _duracion_plan_a_minutos(df["duracion_plan_min"])
    df["escalon_idx"] = pd.to_numeric(df["escalon_idx"], errors="coerce").astype("Int64")
    df = df[CAMPOS_BASE].copy()

    if clean_mode:
        df = limpiar_dataset(df, ruta_config=ruta_config_limpieza)

    return df

