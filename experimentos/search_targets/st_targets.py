"""search_targets -- libreria compartida de TARGETS CANDIDATOS por escalon (Fusion / Reduccion)
y de su agregacion escalon -> batch, para la familia de experimentos `experimentos/search_targets/`.

Pregunta de la familia: que variable objetivo por escalon, maximizada por un recomendador, apunta a
maximizar `rendimiento_proxy_batch` (Sn a metal / (metal + dross + polvo))?

Condiciones que debe cumplir cada candidato (enunciado del /goal 2026-09-14):
  1. calculable desde teoria pirometalurgica (formula explicita, unidades),
  2. coherente con "mejor eficiencia de Fusion / Reduccion" (direccion declarada: +1 = maximizar),
  3. su agregacion escalon -> batch debe ser coherente matematica y pirometalurgicamente
     (masas: suma / telescopica; fracciones de primer orden: supervivencia; cocientes: cociente de
     sumas, nunca media de cocientes (Guia_Ausmelt_Sn.md 10.8); intensivas: media ponderada por tiempo;
     estados terminales: ultimo valor de la fase),
  4. el agregado debe correlacionar positiva y significativamente con el KPI de batch.

Uso:
    import st_targets as st
    df = st.cargar_df()                       # df_v3 cacheado + columnas auxiliares + st_<id> por escalon
    batch = st.construir_batch(df)            # una fila por batch: agregados + KPIs + controles + es_lockbox
    st.REGISTRO                               # metadatos de cada candidato (fase, direccion, agregacion, teoria)

Convenciones:
  - Columnas de escalon: `st_<id>`. Orientadas de forma que MAYOR = MEJOR segun la teoria (direccion ya
    aplicada), salvo los candidatos con `direccion = 0` (exploratorios sin signo a priori) que se dejan crudos.
  - El agregado de batch se llama igual (`st_<id>`) en la tabla de batch.
  - F0 (primer escalon del batch) no tiene estado previo valido: los targets de escalon quedan NaN ahi (igual
    que en v3/v4) y las sumas de Fusion cubren F1..F6 (telescopicas desde el estado al cierre de F0).
  - El ultimo escalon de Reduccion (R3) no tiene ensayo en 105/362 batches: "ultimo valor" = ultimo disponible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp  # noqa: E402

CACHE = RAIZ / "experimentos" / "cache" / "df_v3.pkl"
FASES = ["Fusión", "Reducción"]
CONTROLES_BATCH = ["feed_sn_total_t", "ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm",
                   "frac_carga_secundaria_F", "feed_dross_Fe_total_t"]
KPIS = ["rendimiento_proxy_batch", "f_metal", "f_dross", "f_polvo", "recuperacion_real_pct"]

# constantes fisicas
M_FEO, M_FE, M_C, M_SN = 71.844, 55.845, 12.011, 118.71
C_ESTEQ_POR_SN = 2 * M_C / M_SN          # kg C por kg Sn (SnO2 + 2C -> Sn + 2CO) = 0.2024
B2_OPTIMA = 1.4                          # feature_engineering.BASICIDAD_OPTIMA_FUSION
LAMBDA_FE_DROSS = 0.30                   # kg Sn a dross por kg FeO reducido (experimentos/15)
LAMBDA_FE_TOTAL = 0.60                   # dross + polvo sobre f_metal (experimentos/19)


# =============================================================================
# 1) Columnas auxiliares por escalon
# =============================================================================

def _aux(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    g = df.groupby("Batch")
    sn_inv, feo_inv, masa = df["sn_inventario_escoria_est_kg"], df["feo_inventario_escoria_est_kg"], df["masa_escoria_est_kg"]
    df["sn_inv_prev"] = df["sn_inventario_escoria_est_kg_prev"]
    df["feo_inv_prev"] = df["feo_inventario_escoria_est_kg_prev"]
    df["sn_disp"] = df["sn_inv_prev"] + df["feed_Sn_kgf"].fillna(0)        # Sn disponible en el escalon
    df["sn_ext"] = df["sn_extraido_est_kg"]
    df["feo_ext"] = df["feo_extraido_est_kg"]
    df["dt"] = df["delta_tiempo"]
    df["c_kg"] = df["feed_Carbon_kgh"].fillna(0)
    df["fe_feed_kg"] = df["feed_Fe_kgh"].fillna(0) + df["feed_dross_Fe_kgh"].fillna(0)  # corrientes ferriferas (t4 + t5)
    df["d_feo_inv"] = feo_inv - df["feo_inv_prev"]
    df["d_sn_inv"] = sn_inv - df["sn_inv_prev"]
    df["log_sn_feo"] = np.log(df["ley_sn_escoria_pct"].where(df["ley_sn_escoria_pct"] > 0)) - \
        np.log(df["ley_feo_escoria_pct"].where(df["ley_feo_escoria_pct"] > 0))
    df["log_sn_feo_prev"] = g["log_sn_feo"].shift(1)
    df["d_log_sn_feo"] = df["log_sn_feo"] - df["log_sn_feo_prev"]
    df["T"] = df["temperatura_horno_celsius"]
    df["T_prev"] = df["temperatura_horno_celsius_prev"]
    df["dT"] = df["d_temperatura_horno_celsius"]
    df["T_gas"] = df["temperatura_gas_pre_bhf_celsius"]
    df["B2"] = df["basicidad_B2"]
    df["irf"] = df["indice_irf"]
    df["cum_feed_sn_incl"] = g["feed_Sn_kgf"].cumsum()
    return df


# =============================================================================
# 2) Registro de candidatos
#   fase, unidad, direccion (+1 maximizar / -1 minimizar / 0 exploratorio), formula (df -> Series cruda,
#   ANTES de aplicar direccion), agregacion ('sum' | 'ratio' (num, den) | 'survival' | 'product' |
#   'kapp' | 'twmean' | 'last' | 'mean'), teoria (una frase), fuente.
# =============================================================================

def _ratio_pos(num: pd.Series, den: pd.Series, umbral: float) -> pd.Series:
    ok = num.gt(0) & den.gt(umbral)
    return (num / den).where(ok)


REGISTRO: dict[str, dict] = {}


def _reg(id_: str, fase: str, unidad: str, direccion: int, formula, agg: str, teoria: str, fuente: str,
         num: str | None = None, den: str | None = None, familia: str = "", nota: str = ""):
    REGISTRO[id_] = dict(id=id_, fase=fase, unidad=unidad, direccion=direccion, formula=formula, agg=agg,
                         num=num, den=den, teoria=teoria, fuente=fuente, familia=familia, nota=nota)


# ---------------------------------------------------------------- REDUCCION
_reg("R01_sn_ext", "Reducción", "kg Sn", +1, lambda d: d["sn_ext"], "sum",
     "Sn que deja la escoria (SnO2 + 2C -> Sn): objetivo v4; el agregado es la extraccion total de la fase.",
     "hallazgos 14.2; v4", familia="extraccion")
_reg("R02_feo_ret", "Reducción", "kg FeO", -1, lambda d: d["feo_ext"], "sum",
     "FeO reducido a Fe0 (FeO + C -> Fe + CO): sobre-reduccion -> hardhead/dross que arrastra Sn. Minimizar.",
     "Guia 5.3, 11.3; exp 15 lambda_Fe", familia="sobre-reduccion")
_reg("R03_sn_net_l03", "Reducción", "kg Sn eq", +1, lambda d: d["sn_ext"] - LAMBDA_FE_DROSS * d["feo_ext"].clip(lower=0), "sum",
     "Sn extraido menos el Sn que el FeO reducido enviara a dross (lambda 0.30, canal dross).",
     "exp 15", familia="metal-equivalente")
_reg("R04_sn_net_l06", "Reducción", "kg Sn eq", +1, lambda d: d["sn_ext"] - LAMBDA_FE_TOTAL * d["feo_ext"].clip(lower=0), "sum",
     "Sn extraido menos Sn perdido a dross+polvo por FeO reducido (lambda 0.60): objetivo J_R de v4.",
     "exp 19; v4", familia="metal-equivalente")
_reg("R05_sn_net_l10", "Reducción", "kg Sn eq", +1, lambda d: d["sn_ext"] - 1.0 * d["feo_ext"].clip(lower=0), "sum",
     "Metal equivalente con lambda 1.0 (limite superior del IC de lambda_Fe).", "exp 19 IC", familia="metal-equivalente")
_reg("R06_frac_sn", "Reducción", "fraccion", +1, lambda d: d["frac_sn_extraido_escalon"], "survival",
     "Fraccion del Sn disponible extraida (cinetica de 1er orden): batch = 1 - prod(1 - f_t) = fraccion total de la fase.",
     "Guia 10.7", familia="extraccion")
_reg("R07_kapp_sn", "Reducción", "1/min", +1,
     lambda d: -np.log(_ratio_pos(d["sn_inventario_escoria_est_kg"], d["sn_inv_prev"], 20.0)) / d["dt"], "kapp",
     "Constante aparente de agotamiento de Sn en masa, k = -ln(inv_t/inv_t-1)/dt; batch = -ln(inv_fin/inv_ini)/sum dt.",
     "Guia 10.7", familia="extraccion")
_reg("R08_feo_ret_frac", "Reducción", "fraccion", +1,
     lambda d: _ratio_pos(d["feo_inventario_escoria_est_kg"], d["feo_inv_prev"], 100.0).clip(upper=1.5), "product",
     "Fraccion del FeO que permanece en la escoria (no metalizado); batch = prod = FeO_fin/FeO_ini.",
     "Guia 5.2-5.3", familia="sobre-reduccion")
_reg("R09_selectividad", "Reducción", "kg Sn/kg FeO", +1, lambda d: _ratio_pos(d["sn_ext"], d["feo_ext"], 1.0), "ratio",
     "Selectividad en masa Sn/FeO (SRE): kg de Sn extraido por kg de FeO reducido; batch = cociente de sumas.",
     "consideraciones 32, 72", num="sn_ext", den="feo_ext", familia="selectividad")
_reg("R10_log_selectividad", "Reducción", "log(kg/kg)", +1, lambda d: np.log(_ratio_pos(d["sn_ext"], d["feo_ext"], 1.0)), "ratio_log",
     "log de la selectividad (escala simetrica, multiplicativa); batch = log(sum Sn / sum FeO).",
     "consideraciones 26.1, 32", num="sn_ext", den="feo_ext", familia="selectividad")
_reg("R11_util_carbon", "Reducción", "kg Sn/kg Sn-capacidad C", +1,
     lambda d: _ratio_pos(d["sn_ext"], d["c_kg"] / C_ESTEQ_POR_SN, 50.0), "ratio",
     "Utilizacion del carbon: Sn extraido / Sn que el carbon podria reducir estequiometricamente (C que no fue a Sn fue a Fe, CO2 o inquemado).",
     "Guia 14", num="sn_ext", den="c_cap_sn", familia="selectividad")
_reg("R12_d_avance", "Reducción", "fraccion del Sn cargado", +1, lambda d: d["sn_ext"] / d["cum_feed_sn_incl"], "sum",
     "Incremento del avance de reduccion (Sn extraido / Sn cargado acumulado); suma telescopica = avance_fin - avance_ini.",
     "consideraciones 44.1", familia="extraccion")
_reg("R13_sn_residual", "Reducción", "kg Sn", -1, lambda d: d["sn_inventario_escoria_est_kg"], "last",
     "Sn que queda en la escoria al cierre del escalon (restriccion Sn_slag_final <= limite); batch = inventario final.",
     "consideraciones 38", familia="extraccion")
_reg("R14_log_sn_feo_next", "Reducción", "log(%Sn/%FeO)", -1, lambda d: d["log_sn_feo"], "last",
     "log-ratio Sn/FeO de la escoria al cierre: bajo = se agoto el Sn sin agotar el FeO (limpieza selectiva).",
     "consideraciones 26.1", familia="selectividad")
_reg("R15_d_log_sn_feo", "Reducción", "d log(%Sn/%FeO)", -1, lambda d: d["d_log_sn_feo"], "sum",
     "Caida del log-ratio Sn/FeO en el escalon (composicion cerrada corregida por FeO); suma telescopica.",
     "consideraciones 26.1, 32", familia="selectividad")
_reg("R16_feo_por_sn_disp", "Reducción", "kg FeO/kg Sn disp", -1, lambda d: _ratio_pos(d["feo_ext"], d["sn_disp"], 20.0), "ratio",
     "FeO reducido por kg de Sn aun disponible: intensidad de sobre-reduccion relativa a la fuerza impulsora restante.",
     "Guia 3.3, 11.3", num="feo_ext", den="sn_disp", familia="sobre-reduccion")
_reg("R17_dT", "Reducción", "C", 0, lambda d: d["dT"], "sum",
     "Cambio de temperatura del bano (suma = T_fin - T_ini). Exploratorio: sin signo a priori.", "Guia 19", familia="termico")
_reg("R18_T_ventana", "Reducción", "-|T - P50|", +1, lambda d: -(d["T"] - d["T"].median()).abs(), "twmean",
     "Cercania a la ventana termica (P50 de la fase): lejos de ella sube volatilizacion o baja cinetica/viscosidad.",
     "Guia 8, 19; consideraciones 44.6", familia="termico")
_reg("R19_T_gas_bhf", "Reducción", "C", -1, lambda d: d["T_gas"], "twmean",
     "Temperatura de gases pre-BHF: proxy de arrastre/fume (canal polvo). Minimizar.", "Guia 22", familia="polvo")
_reg("R20_fe_metalizado_por_sn", "Reducción", "kg Fe/kg Sn cargado", -1,
     lambda d: (d["feo_ext"].clip(lower=0) * M_FE / M_FEO) / d["cum_feed_sn_incl"], "sum",
     "Fe metalizado por kg de Sn cargado: dross de Fe especifico (normaliza el FeO reducido por el tamano del batch).",
     "Guia 6, 7.4", familia="sobre-reduccion")

# ---------------------------------------------------------------- FUSION
_reg("F01_sn_ext", "Fusión", "kg Sn", -1, lambda d: d["sn_ext"], "sum",
     "Sn extraido prematuramente en Fusion (lambda_F: -0.13 kg metal/kg): conservar el Sn como SnO2 en la escoria. Minimizar.",
     "hallazgos 14.6", familia="conservacion")
_reg("F02_frac_sn_ext", "Fusión", "fraccion del Sn cargado", -1, lambda d: _ratio_pos(d["sn_ext"].clip(lower=0), d["feed_Sn_kgf"], 20.0), "ratio",
     "Fraccion del Sn cargado en el escalon que sale de la escoria; batch = sum Sn extraido / sum Sn cargado (~ avance al fin de Fusion). Minimizar.",
     "hallazgos 14.6 (F_avance_final)", num="sn_ext", den="feed_Sn_kgf", familia="conservacion")
_reg("F03_feo_formado", "Fusión", "kg FeO", +1, lambda d: d["d_feo_inv"], "sum",
     "FeO incorporado a la escoria (Fe de mineral/dross que se oxida a FeO y no se metaliza); suma = FeO al fin de Fusion.",
     "Guia 7.4, 9.5", familia="matriz")
_reg("F04_feo_ret_por_fe", "Fusión", "kg FeO/kg corriente Fe", +1, lambda d: _ratio_pos(d["d_feo_inv"], d["fe_feed_kg"], 50.0), "ratio",
     "FeO retenido en escoria por kg de corriente ferrifera (t4 mineral + t5 dross) cargada: rendimiento de escorificacion del Fe.",
     "Guia 7.4", num="d_feo_inv", den="fe_feed_kg", familia="matriz")
_reg("F05_ley_feo_next", "Fusión", "%FeO", +1, lambda d: d["ley_feo_escoria_pct"], "last",
     "%FeO de la escoria al cierre: matriz fayalitica fluida y reserva de FeO que protege de la sobre-reduccion; ultimo = fin de Fusion.",
     "Guia 7.4; hallazgos 14.6", familia="matriz")
_reg("F06_B2_ventana", "Fusión", "-|B2 - 1.4|", +1, lambda d: -(d["B2"] - B2_OPTIMA).abs(), "last",
     "Cercania de la basicidad a su optimo interior (viscosidad/liquidus) al fin de Fusion.", "Guia 7.7, 8", familia="matriz")
_reg("F07_T_media", "Fusión", "C", -1, lambda d: d["T"], "twmean",
     "Temperatura del bano: mas T -> mas volatilizacion de SnO y polvo (beta_T de v4). Minimizar dentro de rango.",
     "exp 19; Guia 4.3", familia="termico")
_reg("F08_T_ventana", "Fusión", "-|T - P50|", +1, lambda d: -(d["T"] - d["T"].median()).abs(), "twmean",
     "Cercania a la ventana termica de Fusion.", "Guia 8, 19", familia="termico")
_reg("F09_T_gas_bhf", "Fusión", "C", -1, lambda d: d["T_gas"], "twmean",
     "Temperatura de gases pre-BHF: proxy de arrastre/fume. Minimizar.", "Guia 22", familia="polvo")
_reg("F10_irf_next", "Fusión", "IRF", +1, lambda d: d["irf"], "last",
     "Indice IRF de planta (FeO / (SiO2+Al2O3+CaO)) al fin de Fusion (+0.22 con rendimiento en 14.6).",
     "hallazgos 14.6", familia="matriz")
_reg("F11_log_sn_feo_next", "Fusión", "log(%Sn/%FeO)", +1, lambda d: d["log_sn_feo"], "last",
     "Reserva de SnO2 relativa al FeO al fin de Fusion: alta = la Reduccion arranca con fuerza impulsora de Sn y ventana selectiva amplia.",
     "Guia 3.3", familia="conservacion")
_reg("F12_d_log_sn_feo", "Fusión", "d log(%Sn/%FeO)", +1, lambda d: d["d_log_sn_feo"], "sum",
     "Enriquecimiento del log-ratio Sn/FeO durante el escalon (suma telescopica).", "consideraciones 26.1", familia="conservacion")
_reg("F13_masa_por_carga", "Fusión", "kg escoria/kg carga", 0, lambda d: _ratio_pos(d["d_masa_escoria_est_kg"], d["feed_total_kgh"], 100.0), "ratio",
     "Escoria formada por kg de carga (dilucion/volumen de escoria). Exploratorio.", "Guia 7.1",
     num="d_masa_escoria_est_kg", den="feed_total_kgh", familia="matriz")
_reg("F14_sn_ret_por_c", "Fusión", "kg Sn ret/kg C", +1,
     lambda d: (d["feed_Sn_kgf"] - d["sn_ext"]) / d["c_kg"].where(d["c_kg"] > 50), "ratio",
     "Sn retenido en escoria por kg de carbon: el carbon de Fusion debe fundir/reducir SnO2 en exceso sin extraer Sn (C/Sn baja dross, exp 19).",
     "exp 19", num="sn_ret", den="c_kg", familia="conservacion")
_reg("F15_dT", "Fusión", "C", 0, lambda d: d["dT"], "sum", "Cambio de temperatura (exploratorio).", "Guia 19", familia="termico")
_reg("F16_sn_ret_frac", "Fusión", "fraccion", +1,
     lambda d: _ratio_pos(d["sn_inventario_escoria_est_kg"], d["sn_disp"], 20.0).clip(upper=1.5), "ratio",
     "Fraccion del Sn disponible que permanece en la escoria al cierre; batch = Sn fin Fusion / Sn cargado (cociente de sumas).",
     "hallazgos 14.6", num="sn_inv_fin", den="feed_sn_F", familia="conservacion")

# ---------------------------------------------------------------- GANADORES (anadidos tras ST-05/06; ver README.md)
W_SDI = 10.0   # peso del FeO retenido en el SDI, elegido solo con DEV (st_06_w_seleccion.csv)


def _ln_sn_dep(d: pd.DataFrame) -> pd.Series:
    ok = d["sn_inv_prev"].gt(20) & d["sn_inventario_escoria_est_kg"].gt(20) & d["feo_inv_prev"].gt(100) & d["feo_inventario_escoria_est_kg"].gt(100)
    return np.log(d["sn_inv_prev"] / d["sn_inventario_escoria_est_kg"]).where(ok)


def _ln_feo_ret(d: pd.DataFrame) -> pd.Series:
    ok = d["sn_inv_prev"].gt(20) & d["sn_inventario_escoria_est_kg"].gt(20) & d["feo_inv_prev"].gt(100) & d["feo_inventario_escoria_est_kg"].gt(100)
    return np.log(d["feo_inventario_escoria_est_kg"] / d["feo_inv_prev"]).where(ok)


_reg("R21_ln_feo_ret", "Reducción", "ln(kg/kg)", +1, _ln_feo_ret, "sum",
     "Retencion logaritmica de FeO en la escoria, ln(FeO_inv[t]/FeO_inv[t-1]) (la masa del trazador se cancela); suma = ln(FeO_fin/FeO_ini).",
     "Guia 5.2-5.3, 7.4; ST-05/06", familia="sobre-reduccion")
_reg("R22_sdi", "Reducción", "adimensional", +1, lambda d: _ln_sn_dep(d) + W_SDI * _ln_feo_ret(d), "sum",
     "GANADOR Reduccion: indice de agotamiento selectivo SDI = ln(Sn_inv[t-1]/Sn_inv[t]) + w*ln(FeO_inv[t]/FeO_inv[t-1]), w=10 (DEV); "
     "suma telescopica = ln(Sn_ini/Sn_fin) + w*ln(FeO_fin/FeO_ini). Agotar Sn (cinetica 1er orden) sin metalizar Fe.",
     "Guia 3.3, 11.3; ST-05/06", familia="selectividad")
# GANADOR Fusion = F10_irf_next (IRF al cierre del escalon, ultimo valor de la fase), ya registrado arriba.


def _lnirf(d: pd.DataFrame) -> pd.Series:
    irf = d["ley_feo_escoria_pct"] / (d["ley_sio2_escoria_pct"] + d["ley_al2o3_escoria_pct"] + d["ley_cao_escoria_pct"])
    return np.log(irf.where(irf > 0))


_reg("R23_lnirf_R", "Reducción", "ln(IRF)", +1, _lnirf, "twmean",
     "RONDA 2 (ST-09/ST-12): ln IRF = ln(%FeO/(%SiO2+%Al2O3+%CaO)) al cierre de cada escalon de Reduccion (matriz de escoria "
     "durante la Reduccion = estado heredado x retencion de FeO); agregacion = media ponderada por tiempo (estado intensivo). "
     "Replica en DEV y lockbox; responde a GN (-) y O2 (+).", "Guia 7.4, 5.5; ST-12", familia="matriz")


# derivados usados por 'ratio'
def _cols_ratio(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["c_cap_sn"] = d["c_kg"] / C_ESTEQ_POR_SN
    d["sn_ret"] = d["feed_Sn_kgf"] - d["sn_ext"]
    return d


# =============================================================================
# 3) Construccion del df de escalones con st_<id>
# =============================================================================

def cargar_df() -> pd.DataFrame:
    df = pd.read_pickle(CACHE)
    return construir_df_targets(df)


def construir_df_targets(df: pd.DataFrame) -> pd.DataFrame:
    d = _cols_ratio(_aux(df))
    nuevas = {}
    for id_, r in REGISTRO.items():
        s = r["formula"](d).astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.where(d["fase_proceso"] == r["fase"])
        s = s.where(~d["es_primer_escalon_batch"])          # F0: sin estado previo valido
        nuevas["st_" + id_] = s * (r["direccion"] if r["direccion"] != 0 else 1)
    return pd.concat([d, pd.DataFrame(nuevas, index=d.index)], axis=1)


def columnas_targets(fase: str | None = None) -> list[str]:
    return ["st_" + k for k, r in REGISTRO.items() if fase is None or r["fase"] == fase]


# =============================================================================
# 4) Agregacion escalon -> batch
# =============================================================================

def _agregar_candidato(g: pd.DataFrame, r: dict) -> float:
    """g = escalones de UNA fase de UN batch, ordenados. Devuelve el agregado (ya con direccion)."""
    dirn = r["direccion"] if r["direccion"] != 0 else 1
    col = "st_" + r["id"]
    x = g[col]
    agg = r["agg"]
    if agg == "sum":
        return float(x.sum(min_count=1)) if x.notna().any() else np.nan
    if agg == "twmean":
        m = x.notna() & g["dt"].gt(0)
        return float(np.average(x[m], weights=g.loc[m, "dt"])) if m.any() else np.nan
    if agg == "mean":
        return float(x.mean()) if x.notna().any() else np.nan
    if agg == "last":
        v = x.dropna()
        return float(v.iloc[-1]) if len(v) else np.nan
    if agg == "survival":                       # 1 - prod(1 - f)
        f = (x * dirn).dropna().clip(-5, 1)
        return float(dirn * (1 - np.prod(1 - f))) if len(f) else np.nan
    if agg == "product":                        # prod de fracciones retenidas = fin/ini
        f = (x * dirn).dropna()
        return float(dirn * np.prod(f)) if len(f) else np.nan
    if agg == "kapp":                           # -ln(inv_fin / inv_ini) / sum dt sobre escalones validos
        m = x.notna()
        if not m.any():
            return np.nan
        gi = g.loc[m]
        inv_ini, inv_fin, dt = gi["sn_inv_prev"].iloc[0], gi["sn_inventario_escoria_est_kg"].iloc[-1], gi["dt"].sum()
        if not (inv_ini > 0 and inv_fin > 0 and dt > 0):
            return np.nan
        return float(dirn * (-np.log(inv_fin / inv_ini) / dt))
    if agg in ("ratio", "ratio_log"):           # cociente de sumas (num, den) sobre escalones donde el candidato es valido
        m = x.notna()
        num, den = r["num"], r["den"]
        if r["id"] == "F16_sn_ret_frac":        # definicion explicita: Sn al fin de Fusion / Sn cargado en la fase
            inv = g["sn_inventario_escoria_est_kg"].dropna()
            feed = g["feed_Sn_kgf"].sum()
            if not len(inv) or feed <= 0:
                return np.nan
            return float(dirn * inv.iloc[-1] / feed)
        if not m.any():
            return np.nan
        sn, sd = g.loc[m, num].sum(), g.loc[m, den].sum()
        if not (sd > 0):
            return np.nan
        v = sn / sd
        if agg == "ratio_log":
            return float(dirn * np.log(v)) if v > 0 else np.nan
        return float(dirn * v)
    raise ValueError(agg)


def construir_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: agregados st_<id> (direccion aplicada: mayor = mejor), KPIs, controles, es_lockbox."""
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    filas = []
    for b, g in df.groupby("Batch", sort=False):
        g = g.sort_values("fecha_inicio")
        gf, gr = g[g["fase_proceso"] == "Fusión"], g[g["fase_proceso"] == "Reducción"]
        fila = {"Batch": b, "fecha_batch": g["fecha_inicio"].min(), "es_lockbox": b in batches_lockbox}
        for id_, r in REGISTRO.items():
            sub = gf if r["fase"] == "Fusión" else gr
            fila["st_" + id_] = _agregar_candidato(sub, r) if len(sub) else np.nan
            fila["n_" + id_] = int(sub["st_" + id_].notna().sum()) if len(sub) else 0
        m, dr, po = (g[c].iloc[0] for c in ["sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"])
        tot = m + dr + po
        feed_sn_t = g["feed_Sn_kgf"].sum(min_count=1) / 1000
        fila.update(rendimiento_proxy_batch=g["rendimiento_proxy_batch"].iloc[0],
                    f_metal=m / tot if tot else np.nan, f_dross=dr / tot if tot else np.nan, f_polvo=po / tot if tot else np.nan,
                    recuperacion_real_pct=100 * m / feed_sn_t if feed_sn_t else np.nan,
                    sn_metal_t=m, sn_dross_t=dr, sn_polvo_t=po,
                    feed_sn_total_t=feed_sn_t, ley_sn_conc_batch_pct=g["ley_sn_conc_batch_pct"].iloc[0],
                    espesor_ladrillo_norm_mm=g["espesor_ladrillo_norm_mm"].mean(),
                    frac_carga_secundaria_F=gf["frac_carga_secundaria"].mean() if len(gf) else np.nan,
                    feed_dross_Fe_total_t=g["feed_dross_Fe_kgh"].sum(min_count=1) / 1000,
                    T_media_F=gf["T"].mean() if len(gf) else np.nan)
        filas.append(fila)
    out = pd.DataFrame(filas).set_index("Batch").replace([np.inf, -np.inf], np.nan)
    out["idx_cronologico"] = out["fecha_batch"].rank(method="first").astype(int)
    return out


def tabla_registro() -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "formula"} for r in REGISTRO.values()]).set_index("id")


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = cargar_df()
    batch = construir_batch(df)
    print(f"df {df.shape}, batch {batch.shape}, {time.time() - t0:.1f}s")
    print(tabla_registro()[["fase", "unidad", "direccion", "agg", "familia"]].to_string())
    pd.set_option("display.width", 250)
    print(batch[columnas_targets()].describe().T.round(3).to_string())
    print(df.groupby("fase_proceso")[columnas_targets()].count().T.to_string())
