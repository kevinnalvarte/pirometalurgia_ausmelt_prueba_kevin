"""Ingenieria de variables (features derivadas) del dataset de Lingo Smelter.

Todas las columnas que crea este modulo son DERIVADAS: ninguna viene directamente
del Excel fuente ni de `dataset_lingo_smelter.construir_dataset()`. Se documentan
aqui (formula, unidad, racional metalurgico y caveats) en vez de en
`diccionario_datos.json`, que se reserva solo para las variables originales.

Se espera recibir como entrada el DataFrame ya construido con
`dataset_lingo_smelter.construir_dataset()` y, opcionalmente, ya pasado por
`dataset_lingo_smelter.limpiar_dataset()` (recomendado: limpiar primero, para que
ninguna columna derivada aqui herede valores fuera de rango fisico, en particular
`ley_feo_escoria_pct`, que depende de `ley_fe_total_escoria_pct`).

Uso tipico:
    import dataset_lingo_smelter as dls
    import feature_engineering as fe

    df = dls.construir_dataset(ruta_excel)
    df = dls.limpiar_dataset(df)
    df = fe.construir_features(df)            # features base (EDA, Pasos 0-4)
    df = fe.construir_features_modelado(df)   # features de modelado (Pasos 10-32)
    df = fe.construir_features_v3(df)         # o directamente: base + modelado + v3 (carga por tolva,
                                              # inventario/masa de escoria por trazador CaO, targets en masa;
                                              # sesion 2026-09-13 noche, ver bloque v3 y hallazgos.md 14)

El notebook (analisis_lingo_smelter.ipynb) aplica estas mismas funciones paso a
paso (no de una sola vez), intercalando diagnosticos/graficos entre cada una,
para preservar la narrativa del analisis; este modulo es la fuente de verdad
de cada formula.

Targets candidatos (nunca features de entrada; llamar aparte segun el target
que se quiera modelar, ninguno se agrega dentro de construir_features_modelado):
    fe.agregar_target_tasa_relativa_sn(df)              # T3: fraccion relativa de Sn removido
    fe.agregar_target_extraccion_normalizada_sio2(df)   # Sn/SiO2: robusto a composicion cerrada
    fe.agregar_indice_selectividad_sn_feo(df)           # diagnostico de sobre-reduccion Sn vs Fe
    fe.agregar_k_app_sn(df)                             # constante cinetica aparente [1/min]
`d_ley_feo_escoria_pct`/`grad_ley_feo_escoria_pct` (target de %FeO, senal de
sobre-reduccion) ya quedan disponibles automaticamente via
agregar_tasas_y_gradientes, sin necesidad de una funcion aparte.

Features a nivel de Batch (para un modelo terminal, no de escalon):
    fe.construir_resumen_batch_masa(df)         # recuperacion_real_batch_pct
    fe.construir_resumen_batch_eda(df)          # agregados EDA por fase (lineales, debiles)
    fe.construir_features_transicion_fase(df)   # delta_FR_* (salto Fusion->Reduccion)
    fe.construir_features_trayectoria_batch(df) # perfil/dispersion/zonas de riesgo (ver modelo_predictivo_v2.py)

Disciplina anti-fuga para modelado prescriptivo: `CLASIFICACION_FEATURES`
etiqueta cada columna que este modulo puede producir como CONTEXT / STATE /
ACTION_primitiva / DERIVED-* / TARGET / LEAKAGE; usar
`fe.filtrar_features_seguras(lista_de_features)` antes de entrenar un modelo
cuyas predicciones se usaran para recomendar una accion (no solo para
describir un escalon ya cerrado) — ver la seccion correspondiente al final de
este archivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# Constantes fisico-quimicas
# =============================================================================

# Factor estequiometrico para convertir Fe elemental ensayado a su equivalente
# como oxido FeO (peso molecular FeO / peso molecular Fe). Es el mismo calculo
# que reportaba la columna "FEO" del Excel fuente (dataset_lingo_smelter.py la
# ignora deliberadamente para dejar esta dependencia explicita en el codigo).
PESO_MOLECULAR_FEO = 71.844
PESO_MOLECULAR_FE = 55.845
FACTOR_ESTEQUIOMETRICO_FEO_FE = PESO_MOLECULAR_FEO / PESO_MOLECULAR_FE

# Masas molares (g/mol) de los principales oxidos de la escoria, usadas por
# agregar_indice_estructural_molar para comparar modificadores de red (CaO, MgO, FeO
# equivalente) contra formadores de red (SiO2, Al2O3) en base molar, no masica (una
# razon de %masa no equivale a una razon de moles porque los pesos moleculares difieren,
# ver Guia_Ausmelt_Sn.md seccion 5.3). Fuente: pesos atomicos estandar IUPAC.
MASA_MOLAR_CAO = 56.077
MASA_MOLAR_MGO = 40.304
MASA_MOLAR_SIO2 = 60.084
MASA_MOLAR_AL2O3 = 101.961

# Fraccion volumetrica de O2 en aire, y relacion estequiometrica O2:CH4 para
# combustion completa de gas natural asumido como metano puro:
# CH4 + 2 O2 -> CO2 + 2 H2O
FRACCION_O2_EN_AIRE = 0.21
RELACION_O2_CH4 = 2.0

# variable base -> nombre de la tasa (reactivo consumido/inyectado por minuto
# de escalon). Todas son columnas "flujo durante el escalon" (ver
# _meta.hallazgo_feed_y_volumen_no_acumulados en diccionario_datos.json), por lo
# que basta normalizar por delta_tiempo; NO se les aplica diff().
VARIABLES_FEED_VOLUMEN_A_TASA = {
    "feed_Sn_kgf": "tasa_feed_Sn_kg_min",
    "feed_Fe_kgh": "tasa_feed_Fe_kg_min",
    "feed_dross_Fe_kgh": "tasa_feed_dross_Fe_kg_min",
    "feed_CaO_kgh": "tasa_feed_CaO_kg_min",
    "feed_Carbon_kgh": "tasa_feed_Carbon_kg_min",
    "feed_total_kgh": "tasa_feed_total_kg_min",
    "volumen_gas_natural_inyectado_lanza_escalon_nm3": "tasa_gn_nm3_min",
    "volumen_o2_inyectado_lanza_escalon_nm3": "tasa_o2_nm3_min",
    "volumen_aire_inyectado_lanza_escalon_nm3": "tasa_aire_nm3_min",
}

# Variables de ESTADO (medicion puntual al cierre del escalon: leyes de
# escoria, temperaturas, presiones): a diferencia de feed_*/volumen_*, estas SI
# se diferencian (diff) entre escalones consecutivos del mismo batch, para
# obtener el efecto neto de las reacciones ocurridas con los reactivos
# consumidos en ese escalon.
VARIABLES_ESTADO_PARA_GRADIENTE = [
    "temperatura_horno_celsius", "tiro_horno_pct",
    "presion_suministro_gn_lanza_kpa", "presion_suministro_o2_lanza_kpa",
    "contrapresion_gn_lanza_kpa", "contrapresion_aire_lanza_kpa",
    "presion_punta_lanza_kpa", "posicion_vertical_lanza_mm",
    "ley_sn_escoria_pct", "ley_fe_total_escoria_pct", "ley_feo_escoria_pct",
    "ley_sio2_escoria_pct", "ley_cao_escoria_pct", "ley_al2o3_escoria_pct",
    "ley_mgo_escoria_pct", "ley_sb_escoria_pct", "ley_cr_escoria_pct", "ley_as_escoria_pct",
    "temperatura_gas_pre_bhf_celsius", "temperatura_gas_pre_cuchilla_celsius",
]


def division_segura(numerador: pd.Series, denominador: pd.Series, umbral: float = 1e-6) -> pd.Series:
    """Division elemento a elemento devolviendo NaN cuando el denominador es ~0.

    Evita ratios que se disparan a valores sin sentido cuando el denominador
    fisico esta ausente (p.ej. relacion_C_carga en Reduccion, o basicidad
    cuando %SiO2 ronda 0).
    """
    denominador_valido = denominador.where(denominador.abs() > umbral)
    return numerador / denominador_valido


# =============================================================================
# ley_feo_escoria_pct
# =============================================================================

def agregar_ley_feo(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `ley_feo_escoria_pct`: contenido de oxido ferroso en la escoria (%).

    Formula: ley_feo_escoria_pct = ley_fe_total_escoria_pct * FACTOR_ESTEQUIOMETRICO_FEO_FE

    NO es un ensayo de laboratorio independiente ni una columna del Excel fuente:
    dataset_lingo_smelter.py ignora deliberadamente la columna "FEO" del Excel
    (no esta en RENOMBRES ni en CAMPOS_BASE) porque esa columna reportaba
    exactamente este mismo calculo estequiometrico a partir de %Fe. Se recalcula
    aqui de forma explicita para dejar visible la dependencia con
    ley_fe_total_escoria_pct (son, en esencia, la misma senal reescalada).

    Verificado empiricamente contra el Excel fuente: pendiente de regresion
    FEO~%Fe = 1.2865, correlacion 0.96 (3849 filas). Implicacion: es
    practicamente colineal con ley_fe_total_escoria_pct y no aporta ley quimica
    independiente; tratar con cautela cualquier correlacion, ratio o
    interaccion que use ambas leyes juntas (ver hallazgos.md seccion 5).

    Requiere que `df` ya haya pasado por `dataset_lingo_smelter.limpiar_dataset()`
    para no heredar valores de ley_fe_total_escoria_pct fuera de rango fisico.
    """
    df = df.copy()
    df["ley_feo_escoria_pct"] = df["ley_fe_total_escoria_pct"] * FACTOR_ESTEQUIOMETRICO_FEO_FE
    return df


# =============================================================================
# Tiempos
# =============================================================================

def agregar_tiempos(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `delta_tiempo`, `tiempo` y `tiempo_fase`, en minutos.

    - delta_tiempo = fecha_final - fecha_inicio (duracion del escalon). Se usa
      para normalizar feed_*/volumen_* a tasas por minuto y para convertir
      deltas de variables de estado en gradientes por minuto.
    - tiempo = fecha_final - min(fecha_inicio) del batch (minutos transcurridos
      desde el inicio del batch hasta el cierre de este escalon).
    - tiempo_fase = fecha_final - min(fecha_inicio) de la fase actual del mismo
      batch (minutos transcurridos desde el inicio de la fase -Fusion o
      Reduccion- hasta el cierre de este escalon). A diferencia de
      `orden_escalon_fase` (posicion discreta 0-6/0-3), `tiempo_fase` es
      continuo y refleja que la duracion del escalon varia bastante (Fusion
      25-188 min, Reduccion 5-125 min segun descripcion_fases.md): dos
      escalones con el mismo `orden_escalon_fase` pueden representar un avance
      real de proceso muy distinto si sus duraciones acumuladas difieren.
      Recomendado por guia_fenomenologica_ausmelt_sn.md seccion 12.5
      (`tiempo\\_fase_i=t_i-t_{inicio\\,fase}`).
    """
    df = df.copy()
    df["delta_tiempo"] = (
        df["fecha_final"] - df["fecha_inicio"]
    ).dt.total_seconds() / 60
    df["tiempo"] = (
        df["fecha_final"]
        - df.groupby("Batch")["fecha_inicio"].transform("min")
    ).dt.total_seconds() / 60
    df["tiempo_fase"] = (
        df["fecha_final"]
        - df.groupby(["Batch", "fase_proceso"])["fecha_inicio"].transform("min")
    ).dt.total_seconds() / 60
    return df


# =============================================================================
# Variable objetivo
# =============================================================================

def agregar_rendimiento_proxy_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `rendimiento_proxy_batch` (%): proxy de rendimiento metalurgico del batch.

    Formula:
        100 * sn_en_metal_crudo_batch_t
            / (sn_en_metal_crudo_batch_t + sn_en_dross_fe_batch_t + sn_en_polvo_fundicion_batch_t)

    Variable objetivo a maximizar. Unica por batch: queda repetida en todos los
    escalones de ese batch (sn_en_*_batch_t tambien lo son). Si el total de
    salidas de Sn es 0, el resultado queda en NaN (no hay una particion valida
    que calcular).

    Nota: los agregados lineales simples de features por batch (sumas/promedios
    por fase) correlacionan debil con esta variable (|Spearman|<=0.13); requiere
    modelar a nivel de escalon, ver hallazgos.md seccion 6.
    """
    df = df.copy()
    sn_total_salidas_batch_t = (
        df["sn_en_metal_crudo_batch_t"]
        + df["sn_en_dross_fe_batch_t"]
        + df["sn_en_polvo_fundicion_batch_t"]
    )
    df["rendimiento_proxy_batch"] = (
        100 * df["sn_en_metal_crudo_batch_t"] / sn_total_salidas_batch_t
    ).where(sn_total_salidas_batch_t.ne(0))
    return df


# =============================================================================
# Tasas (reactivos por minuto) y gradientes (respuesta del bano por minuto)
# =============================================================================

def agregar_tasas_y_gradientes(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las familias `tasa_<variable>` y `d_<variable>`/`grad_<variable>`.

    Requiere `delta_tiempo` ya creado (ver `agregar_tiempos`) y que `df` este
    ordenado (se reordena aqui por las dudas) por ["Batch", "fecha_inicio"].

    tasa_<variable> (una por cada entrada de VARIABLES_FEED_VOLUMEN_A_TASA):
        tasa_x = x_escalon / delta_tiempo   [kg/min o Nm3/min]
    Tasa de consumo/inyeccion del reactivo durante el escalon, normalizada por
    su duracion; permite comparar escalones de distinta duracion. NO se le
    aplica diff(): feed_*/volumen_* ya representan lo consumido/inyectado
    DURANTE el escalon, no un acumulado (ver
    dataset_lingo_smelter / diccionario_datos.json _meta.hallazgo_feed_y_volumen_no_acumulados).

    d_<variable> y grad_<variable> (una por cada variable en
    VARIABLES_ESTADO_PARA_GRADIENTE, todas variables de ESTADO: leyes de
    escoria, temperaturas, presiones, medidas puntualmente al cierre del
    escalon):
        d_y[i]    = y[i] - y[i-1]     (dentro del mismo Batch, ordenado por fecha_inicio)
        grad_y[i] = d_y[i] / delta_tiempo[i]
    Representan el cambio (d_) y la velocidad de cambio por minuto (grad_) de
    la variable de estado entre el escalon actual y el anterior del mismo
    batch: el efecto neto de las reacciones ocurridas con los reactivos
    consumidos en ese escalon. El primer escalon de cada batch queda en NaN en
    ambas (no hay escalon previo de ese batch). El diff NO se resetea en la
    transicion Fusion->Reduccion; esa transicion suele ser la senal mas
    informativa.
    """
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()

    nuevas_cols: dict[str, pd.Series] = {}
    for col_origen, col_tasa in VARIABLES_FEED_VOLUMEN_A_TASA.items():
        nuevas_cols[col_tasa] = df[col_origen] / df["delta_tiempo"]

    por_batch = df.groupby("Batch")
    for var in VARIABLES_ESTADO_PARA_GRADIENTE:
        d = por_batch[var].diff()
        nuevas_cols[f"d_{var}"] = d
        nuevas_cols[f"grad_{var}"] = d / df["delta_tiempo"]

    return pd.concat([df, pd.DataFrame(nuevas_cols, index=df.index)], axis=1)


# =============================================================================
# Posicion relativa del escalon
# =============================================================================

def agregar_orden_escalon(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `orden_escalon_fase`, `es_primer_escalon_fase`, `es_primer_escalon_batch`
    y `es_transicion_FR`.

    - orden_escalon_fase: posicion (0-indexada) del escalon dentro de su fase y
      batch (0-6 en Fusion, 0-3 en Reduccion). Permite alinear el "mismo
      momento relativo del proceso" entre distintos batches, dado que la
      estructura de 7+4 escalones es fija.
    - es_primer_escalon_fase: True si orden_escalon_fase == 0.
    - es_primer_escalon_batch: True si es el primer escalon del batch (no tiene
      escalon previo de ese batch, por lo que sus d_<variable>/grad_<variable>
      son NaN).
    - es_transicion_FR: True si es el primer escalon de Reduccion del batch
      (fase_proceso=='Reducción' y orden_escalon_fase==0). Marca explicitamente
      la fila donde ocurre el evento mas fuerte y mas universal de todo el
      batch: el salto de tasa de carbon a su maximo y la mayor caida de
      `ley_sn_escoria_pct` (mediana ~-12 puntos, ~100% de los batches
      evaluables, ver hallazgos.md seccion 3/hallazgos.md seccion 4). Sin este
      indicador, un modelo unico Fusion+Reduccion puede confundir esta
      transicion deliberada de regimen con dinamica ordinaria de escalon
      (recomendado explicitamente por guia_fenomenologica_ausmelt_sn.md
      seccion 12.2). El `d_<variable>`/`grad_<variable>` de esta fila NO se
      resetea (agregar_tasas_y_gradientes seccion superior) precisamente
      porque es la senal mas informativa del batch.
    """
    df = df.copy()
    df["orden_escalon_fase"] = df.groupby(["Batch", "fase_proceso"]).cumcount()
    df["es_primer_escalon_fase"] = df["orden_escalon_fase"].eq(0)
    df["es_primer_escalon_batch"] = df.groupby("Batch").cumcount().eq(0)
    df["es_transicion_FR"] = df["es_primer_escalon_fase"] & df["fase_proceso"].eq("Reducción")
    return df


# =============================================================================
# Features compuestas con fundamento pirometalurgico
# =============================================================================

def agregar_features_pirometalurgicas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea basicidades, balance de combustion de la lanza y relaciones carga/reactivo.

    a) Basicidad de la escoria (controla viscosidad, capacidad de fundente y la
       particion de Sn/Fe entre metal y escoria; en un sistema fayalitico
       FeO-SiO2-CaO, la basicidad regula cuanto SnO2 queda atrapado en la
       escoria):
           basicidad_B2 = %CaO / %SiO2
           basicidad_B4 = (%CaO + %MgO) / (%SiO2 + %Al2O3)
       basicidad_B4 puede dispararse (outlier) cuando %SiO2 esta muy cerca de
       0; recortar percentiles 1%-99% solo para graficar.

    b) Balance estequiometrico de combustion en la lanza Ausmelt (define si la
       atmosfera generada es oxidante o reductora; clave porque el proceso
       alterna Fusion -oxidante, favorece formar SnO2/FeO- con Reduccion
       -reductora, favorece recuperar Sn metalico). Asumiendo combustion de
       gas natural como metano puro (CH4 + 2 O2 -> CO2 + 2 H2O):
           o2_teorico_combustion_nm3 = RELACION_O2_CH4 * volumen_gas_natural_inyectado_lanza_escalon_nm3
           o2_disponible_nm3 = volumen_o2_inyectado_lanza_escalon_nm3
                                + FRACCION_O2_EN_AIRE * volumen_aire_inyectado_lanza_escalon_nm3
           exceso_o2_combustion_pct = 100 * (o2_disponible_nm3 - o2_teorico_combustion_nm3)
                                          / o2_teorico_combustion_nm3
       Positivo = atmosfera oxidante (excedente de O2 disponible para oxidar el
       bano); negativo = atmosfera reductora (deficit, genera CO/H2 reductores
       que reducen SnO2/FeO del bano). Validado empiricamente: mediana
       ~+18.9% en Fusion, ~-4.8% en Reduccion.

    c) Relacion reductor/carga y fundente/carga (estequiometria de reduccion de
       casiterita, SnO2 + 2C -> Sn + 2CO2, y de fusion con mineral de Fe/dross;
       se usan las MASAS del escalon, no las tasas, por ser un ratio
       adimensional reactivo/mineral):
           relacion_C_carga = feed_Carbon_kgh / (feed_Sn_kgf + feed_Fe_kgh + feed_dross_Fe_kgh)
           relacion_CaO_carga_Fe = feed_CaO_kgh / (feed_Fe_kgh + feed_dross_Fe_kgh)
       relacion_C_carga solo es interpretable en Fusion (mediana ~0.17, rango
       fisico razonable). En Reduccion casi no se alimenta mineral/dross
       fresco (denominador ~0) y el ratio se dispara a valores sin sentido; ahi
       usar directamente tasa_feed_Carbon_kg_min.

    d) Indice estructural molar de la escoria (complementa basicidad_B2/B4, que
       son razones de MASA y por tanto mezclan oxidos de peso molecular muy
       distinto -CaO=56.08, SiO2=60.08, MgO=40.30, Al2O3=101.96 g/mol- con el
       mismo peso que si fueran comparables mol a mol). Compara modificadores
       de red (CaO, MgO, FeO equivalente) contra formadores de red (SiO2,
       Al2O3) en base molar (Guia_Ausmelt_Sn.md seccion 5.3):
           indice_estructural_molar =
               (%CaO/M_CaO + %MgO/M_MgO + %FeO_eq/M_FeO)
               / (%SiO2/M_SiO2 + %Al2O3/M_Al2O3)
       Es un indice estructural SIMPLIFICADO bajo supuestos explicitos (asume
       modificadores divalentes, ignora especiacion Fe2+/Fe3+, SnOx disuelto y
       compensacion de carga del Al tetraedrico): no es NBO/T medido ni una
       basicidad termodinamica validada, solo una version molar de B2/B4. Mismo
       riesgo de outlier que B2/B4 si %SiO2+%Al2O3 esta cerca de 0.

    Todos los ratios usan `division_segura` para devolver NaN en vez de valores
    absurdos cuando el denominador fisico esta ausente (~0).
    """
    df = df.copy()

    # a) Basicidad de la escoria
    df["basicidad_B2"] = division_segura(df["ley_cao_escoria_pct"], df["ley_sio2_escoria_pct"])
    df["basicidad_B4"] = division_segura(
        df["ley_cao_escoria_pct"] + df["ley_mgo_escoria_pct"],
        df["ley_sio2_escoria_pct"] + df["ley_al2o3_escoria_pct"],
    )

    # d) Indice estructural molar (I_mod). Requiere ley_feo_escoria_pct (agregar_ley_feo).
    df["indice_estructural_molar"] = division_segura(
        df["ley_cao_escoria_pct"] / MASA_MOLAR_CAO
        + df["ley_mgo_escoria_pct"] / MASA_MOLAR_MGO
        + df["ley_feo_escoria_pct"] / PESO_MOLECULAR_FEO,
        df["ley_sio2_escoria_pct"] / MASA_MOLAR_SIO2
        + df["ley_al2o3_escoria_pct"] / MASA_MOLAR_AL2O3,
    )

    # b) Balance estequiometrico de combustion (lanza Ausmelt)
    df["o2_teorico_combustion_nm3"] = RELACION_O2_CH4 * df["volumen_gas_natural_inyectado_lanza_escalon_nm3"]
    df["o2_disponible_nm3"] = (
        df["volumen_o2_inyectado_lanza_escalon_nm3"]
        + FRACCION_O2_EN_AIRE * df["volumen_aire_inyectado_lanza_escalon_nm3"]
    )
    df["exceso_o2_combustion_pct"] = 100 * division_segura(
        df["o2_disponible_nm3"] - df["o2_teorico_combustion_nm3"],
        df["o2_teorico_combustion_nm3"],
    )

    # c) Relaciones reductor/carga y fundente/carga
    carga_oxidos_fe_sn = df["feed_Sn_kgf"] + df["feed_Fe_kgh"] + df["feed_dross_Fe_kgh"]
    carga_fe = df["feed_Fe_kgh"] + df["feed_dross_Fe_kgh"]
    df["relacion_C_carga"] = division_segura(df["feed_Carbon_kgh"], carga_oxidos_fe_sn)
    df["relacion_CaO_carga_Fe"] = division_segura(df["feed_CaO_kgh"], carga_fe)

    return df


# =============================================================================
# Orquestador (features base: EDA y quimica del proceso)
# =============================================================================

def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica, en orden, la ingenieria de variables BASE (EDA / Pasos 0-4 del notebook).

    Orden de dependencias: ley_feo_escoria_pct y tiempos primero (los usan
    tasas/gradientes y basicidad/o2 despues); rendimiento_proxy_batch y orden
    de escalon son independientes del resto.

    Para las features adicionales de modelado (lags, acumulados, interacciones;
    Pasos 10 en adelante del notebook), ver `construir_features_modelado` mas
    abajo, que se aplica DESPUES de esta funcion.
    """
    df = agregar_ley_feo(df)
    df = agregar_tiempos(df)
    df = agregar_rendimiento_proxy_batch(df)
    df = agregar_tasas_y_gradientes(df)
    df = agregar_orden_escalon(df)
    df = agregar_features_pirometalurgicas(df)
    return df


# =============================================================================
# Features de modelado: estado previo (_prev), acumulados y ratios (Paso 10)
# =============================================================================

# Regla anti-fuga (aplicada a toda esta seccion, Pasos 10, 17 y 20): toda
# feature nueva debe ser calculable con informacion disponible ANTES o al
# inicio del escalon que se esta prediciendo (estado previo `_prev`, acumulado
# hasta el escalon anterior, o insumos del escalon actual) — nunca con las
# leyes/estado contemporaneo del escalon actual (eso seria fuga de informacion,
# ya que el target usual es el delta de ese mismo escalon).

# Variables de estado a las que se les crea su version "_prev" (valor al cierre
# del escalon anterior del mismo batch). NaN solo en el primer escalon de cada
# batch (no hay escalon previo de ESE batch).
VARIABLES_LAG_ESTADO = [
    "ley_sn_escoria_pct", "ley_feo_escoria_pct", "ley_fe_total_escoria_pct",
    "ley_sio2_escoria_pct", "ley_cao_escoria_pct",
    "basicidad_B2", "basicidad_B4", "indice_estructural_molar",
    "temperatura_horno_celsius", "posicion_vertical_lanza_mm",
    "presion_punta_lanza_kpa", "tiro_horno_pct", "temperatura_gas_pre_bhf_celsius",
]


def agregar_estado_previo(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `<variable>_prev` para cada variable en VARIABLES_LAG_ESTADO (Paso 10a).

    <variable>_prev[i] = <variable>[i-1] (dentro del mismo Batch, ordenado por
    fecha_inicio). Informacion disponible ANTES de la reaccion del escalon
    actual (no es fuga: el target sigue siendo el delta/estado del escalon
    actual, nunca se usa la ley contemporanea como feature). Requiere que
    basicidad_B2, basicidad_B4 e indice_estructural_molar ya existan (ver
    `agregar_features_pirometalurgicas`).
    """
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    por_batch = df.groupby("Batch")
    for var in VARIABLES_LAG_ESTADO:
        df[f"{var}_prev"] = por_batch[var].shift(1)
    return df


# variable base (masa/volumen del escalon) -> nombre del acumulado "_prev"
VARIABLES_CUM_A_PREV = {
    "feed_Sn_kgf": "cum_feed_Sn_kg",
    "feed_Fe_kgh": "cum_feed_Fe_kg",
    "feed_dross_Fe_kgh": "cum_feed_dross_Fe_kg",
    "feed_CaO_kgh": "cum_feed_CaO_kg",
    "feed_Carbon_kgh": "cum_feed_Carbon_kg",
    "volumen_gas_natural_inyectado_lanza_escalon_nm3": "cum_gn_nm3",
    "volumen_o2_inyectado_lanza_escalon_nm3": "cum_o2_nm3",
    "volumen_aire_inyectado_lanza_escalon_nm3": "cum_aire_nm3",
}


def agregar_acumulados_previos(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `<cum>_prev` (inventario acumulado antes del escalon) y `relacion_C_cum_Sn_cum_prev` (Paso 10b).

    Para cada variable base x en VARIABLES_CUM_A_PREV, el acumulado hasta el
    escalon ANTERIOR (excluye el consumo del escalon actual) es:
        M_x_prev[i] = sum(x[k] para k < i, mismo Batch) = cumsum(x)[i] - x[i]
    Representa cuanto de ese reactivo ya se alimento al horno antes de que
    empiece el escalon actual (el "inventario construido" hasta ese punto, en
    unidades fisicas kg/Nm3 -- a diferencia de orden_escalon_fase, que solo da
    la posicion, no la magnitud).

    relacion_C_cum_Sn_cum_prev = cum_feed_Carbon_kg_prev / cum_feed_Sn_kg_prev:
    severidad reductora acumulada (cuanto carbon se ha dosificado por cada kg
    de Sn fino alimentado hasta el momento), util sobre todo en Reduccion
    (donde ya no se alimenta Sn fresco y lo relevante es cuanto reductor se
    acumulo respecto al Sn ya cargado en Fusion).
    """
    df = df.copy()
    por_batch = df.groupby("Batch")
    for col_origen, col_cum in VARIABLES_CUM_A_PREV.items():
        cum_incluyente = por_batch[col_origen].cumsum()
        df[f"{col_cum}_prev"] = cum_incluyente - df[col_origen]

    df["relacion_C_cum_Sn_cum_prev"] = division_segura(
        df["cum_feed_Carbon_kg_prev"], df["cum_feed_Sn_kg_prev"], umbral=1e-3
    )
    return df


def agregar_ratios_composicionales_previos(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `ratio_sn_feo_prev`, `ratio_feo_sio2_prev`, `log_ratio_sn_feo_prev`,
    `ratio_sn_sio2_prev` y `log_ratio_sn_sio2_prev` (Paso 10c).

    Requiere `agregar_estado_previo` ya aplicado. Las leyes de escoria suman
    ~100% (composicion cerrada, ver nota_composicion_cerrada en
    diccionario_datos.json), por lo que un % aislado mezcla "cuanto hay
    realmente" con "cuanto subio/bajo el resto". Las razones son mas robustas
    a ese artefacto y tienen lectura pirometalurgica directa, usando el
    ESTADO PREVIO (no contemporaneo, para no filtrar informacion del propio
    escalon objetivo):
        ratio_sn_feo_prev    = %Sn_prev / %FeO_prev
        ratio_feo_sio2_prev  = %FeO_prev / %SiO2_prev
        ratio_sn_sio2_prev   = %Sn_prev / %SiO2_prev
    ratio_sn_feo_prev aproxima que tan "rica" en Sn esta la escoria relativa a
    su principal competidor redox (FeO_escoria + Sn_metal <-> SnO_escoria +
    Fe_metal); ratio_feo_sio2_prev es un proxy adicional de estructura de la
    escoria (FeO tambien actua como modificador de red). ratio_sn_sio2_prev usa
    SiO2 -el formador de red, que no se reduce ni se forma dentro del horno-
    como referencia mas inerte que FeO frente al Sn (Guia_Ausmelt_Sn.md
    secciones 7.2/7.3/10.5): es la version SEGURA (sin fuga) del `ratio_sn_sio2`
    contemporaneo usado como target en `agregar_target_extraccion_normalizada_sio2`,
    y sirve tambien como feature de entrada (nivel de partida del escalon en
    esa escala mas robusta a dilucion).
    log_ratio_sn_feo_prev / log_ratio_sn_sio2_prev = log(ratio) (solo donde es
    > 0): razones de composicion suelen tener una escala mas natural en log
    (cambios multiplicativos, no aditivos).
    """
    df = df.copy()
    df["ratio_sn_feo_prev"] = division_segura(
        df["ley_sn_escoria_pct_prev"], df["ley_feo_escoria_pct_prev"], umbral=1e-3
    )
    df["ratio_feo_sio2_prev"] = division_segura(
        df["ley_feo_escoria_pct_prev"], df["ley_sio2_escoria_pct_prev"], umbral=1e-3
    )
    df["log_ratio_sn_feo_prev"] = np.log(df["ratio_sn_feo_prev"].where(df["ratio_sn_feo_prev"] > 0))
    df["ratio_sn_sio2_prev"] = division_segura(
        df["ley_sn_escoria_pct_prev"], df["ley_sio2_escoria_pct_prev"], umbral=1e-3
    )
    df["log_ratio_sn_sio2_prev"] = np.log(df["ratio_sn_sio2_prev"].where(df["ratio_sn_sio2_prev"] > 0))
    return df


def agregar_features_lanza_redox_extra(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features adicionales de lanza/redox (Paso 10d).

    Requiere las tasas de `agregar_tasas_y_gradientes` ya aplicadas.

    - oxygen_enrichment_pct: fraccion de O2 puro en el gas OXIDANTE total (no
      en el exceso respecto al GN, a diferencia de exceso_o2_combustion_pct):
          100 * (tasa_o2 + FRACCION_O2_EN_AIRE * tasa_aire) / (tasa_o2 + tasa_aire)
      Describe solo la composicion del gas oxidante entregado por la lanza
      (aire puro ~21%, O2 puro ~100%); relevante porque el enriquecimiento con
      O2 cambia la hidrodinamica (menos N2 inerte).
    - total_process_gas_nm3_min: caudal total de gas de proceso de la lanza
      (GN + O2 + aire), tasa_gn + tasa_o2 + tasa_aire.
    - lanza_pos_x_gas_total: posicion_vertical_lanza_mm * total_process_gas_nm3_min.
      La hidrodinamica (penetracion, splashing, mezcla) depende de la
      combinacion de caudal de gas y profundidad de inmersion, no de cada uno
      por separado.
    - interaccion_lanza_x_gn: posicion_vertical_lanza_mm * tasa_gn_nm3_min.
      Version mas especifica de la interaccion anterior, usando solo el gas
      natural (en vez del caudal total de la lanza). Justificada por evidencia
      cuantitativa directa (no solo teorica): SHAP interaction values muestran
      que el 64.5% del efecto SHAP total de `posicion_vertical_lanza_mm` sobre
      `d_ley_sn_escoria_pct` en Reduccion proviene de su interaccion con
      `tasa_gn_nm3_min`, no de un efecto propio aislado (hallazgos.md seccion
      12, "Pasos 46-47"); consistente con consideraciones seccion 16 ("deben
      modelarse interacciones posicion x flujo de gas, no posicion aislada").
      NOTA: `posicion_vertical_lanza_mm` tiene un optimo interior (valle
      ~3300-4200mm de la variable cruda, no un efecto monotono) y su signo
      cambio entre multiples especificaciones del analisis -- tratar cualquier
      feature derivada de ella como candidata a validar con planta antes de
      uso prescriptivo, no como palanca de signo estable.
    - indice_severidad_reductora: tasa_feed_Carbon_kg_min / (tasa_o2 + FRACCION_O2_EN_AIRE * tasa_aire).
      Compara el reductor solido contra el oxidante disponible de la lanza (en
      vez de contra la carga solida como relacion_C_carga, que en Reduccion es
      ~0).
    - margen_presion_gn: presion_suministro_gn_lanza_kpa - contrapresion_gn_lanza_kpa.
      Proxy simple de cuanta presion "sobra" para vencer restricciones/inmersion.
    """
    df = df.copy()
    df["oxygen_enrichment_pct"] = 100 * division_segura(
        df["tasa_o2_nm3_min"] + FRACCION_O2_EN_AIRE * df["tasa_aire_nm3_min"],
        df["tasa_o2_nm3_min"] + df["tasa_aire_nm3_min"],
    )
    df["total_process_gas_nm3_min"] = df["tasa_gn_nm3_min"] + df["tasa_o2_nm3_min"] + df["tasa_aire_nm3_min"]
    df["lanza_pos_x_gas_total"] = df["posicion_vertical_lanza_mm"] * df["total_process_gas_nm3_min"]
    df["interaccion_lanza_x_gn"] = df["posicion_vertical_lanza_mm"] * df["tasa_gn_nm3_min"]
    df["indice_severidad_reductora"] = division_segura(
        df["tasa_feed_Carbon_kg_min"],
        df["tasa_o2_nm3_min"] + FRACCION_O2_EN_AIRE * df["tasa_aire_nm3_min"],
        umbral=1e-3,
    )
    df["margen_presion_gn"] = df["presion_suministro_gn_lanza_kpa"] - df["contrapresion_gn_lanza_kpa"]
    return df


def agregar_delta_temperatura_gases(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `delta_T_gas_pre` (Paso 10e): proxy simple del tren de gases aguas abajo.

        delta_T_gas_pre = temperatura_gas_pre_cuchilla_celsius - temperatura_gas_pre_bhf_celsius

    No se le atribuye causalidad directa sobre la quimica del bano (son
    sensores downstream, con retardo), pero puede ayudar a distinguir
    regimenes de operacion del tren de gases que covarian con la intensidad de
    combustion de la lanza.
    """
    df = df.copy()
    df["delta_T_gas_pre"] = df["temperatura_gas_pre_cuchilla_celsius"] - df["temperatura_gas_pre_bhf_celsius"]
    return df


# =============================================================================
# Candidato de target normalizado (Paso 11)
# =============================================================================

# Por debajo de este nivel de %Sn en escoria del escalon anterior, T3 se
# descarta (division inestable: un mismo delta absoluto se dispara a una
# fraccion relativa sin sentido si el nivel de partida es casi 0).
UMBRAL_LEY_SN_PREV = 0.5  # %


def agregar_target_tasa_relativa_sn(df: pd.DataFrame, umbral_ley_sn_prev: float = UMBRAL_LEY_SN_PREV) -> pd.DataFrame:
    """Crea `tasa_relativa_sn_escoria` (Paso 11, target candidato T3).

    Requiere `agregar_estado_previo` (ley_sn_escoria_pct_prev) y
    `agregar_tasas_y_gradientes` (d_ley_sn_escoria_pct) ya aplicados.

        T3 = tasa_relativa_sn_escoria = d_ley_sn_escoria_pct / ley_sn_escoria_pct_prev
             (NaN si ley_sn_escoria_pct_prev < umbral_ley_sn_prev)

    Fraccion del %Sn de la escoria removida en el escalon, relativa al nivel
    con el que el escalon arranca (el Sn ya presente en la escoria antes de la
    reaccion de este escalon, disponible para ser reducido). Fundamento
    cinetico: en una reaccion de tipo primer-orden (d[Sn]/dt ~ -[Sn]), la
    fraccion removida por escalon es mas comparable entre escalones con
    distinto nivel de partida que el delta absoluto (T1) o el delta por minuto
    (T2 = grad_ley_sn_escoria_pct, ya existente).
    """
    df = df.copy()
    ley_sn_prev_protegida = df["ley_sn_escoria_pct_prev"].where(
        df["ley_sn_escoria_pct_prev"] >= umbral_ley_sn_prev
    )
    df["tasa_relativa_sn_escoria"] = df["d_ley_sn_escoria_pct"] / ley_sn_prev_protegida
    return df


def agregar_target_extraccion_normalizada_sio2(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `ratio_sn_sio2`, `d_ratio_sn_sio2` y `d_log_ratio_sn_sio2` (targets candidatos).

    Requiere `agregar_estado_previo` ya aplicado (usa `ley_sn_escoria_pct_prev`
    y `ley_sio2_escoria_pct_prev`).

    `ratio_sn_sio2` es CONTEMPORANEO (usa las leyes del escalon actual): uso
    exclusivo como target/diagnostico, NUNCA como feature de entrada -- para
    features de entrada usar `ratio_sn_sio2_prev`/`log_ratio_sn_sio2_prev`
    (ver `agregar_ratios_composicionales_previos`).

        ratio_sn_sio2 = ley_sn_escoria_pct / ley_sio2_escoria_pct
        d_ratio_sn_sio2 = ratio_sn_sio2 - ratio_sn_sio2_prev
        d_log_ratio_sn_sio2 = log(ratio_sn_sio2) - log(ratio_sn_sio2_prev)
            (NaN si algun ratio no es > 0)

    Fundamento: SiO2 es el formador de red de la escoria y, a diferencia de Sn
    o FeO, no se reduce ni se forma dentro del horno (Guia_Ausmelt_Sn.md
    secciones 7.2, 7.3, 10.5). Por composicion cerrada (diccionario_datos.json
    nota_composicion_cerrada), un cambio de `ley_sn_escoria_pct` mezcla
    extraccion real de Sn con simple dilucion/concentracion cuando cambia la
    masa total de escoria; usar Sn/SiO2 en vez de %Sn puro atenua (no elimina:
    Guia_Ausmelt_Sn.md seccion 7.3 advierte que SiO2 tampoco es un trazador
    perfecto, puede entrar/salir por ganga/CaO) ese artefacto. `d_log_ratio_sn_sio2`
    tiene ademas la escala mas natural para cambios multiplicativos
    (consideraciones seccion 26.1).

    Candidatos de TARGET alternativos a `d_ley_sn_escoria_pct` /
    `tasa_relativa_sn_escoria`: mas robustos al efecto de composicion cerrada,
    pero heredan el ruido de DOS ensayos de laboratorio en vez de uno, y siguen
    sin ser una masa de Sn real (no hay masa de escoria por escalon, ver
    hallazgos.md seccion 6b/9.5).
    """
    df = df.copy()
    df["ratio_sn_sio2"] = division_segura(df["ley_sn_escoria_pct"], df["ley_sio2_escoria_pct"], umbral=1e-3)
    ratio_prev = division_segura(
        df["ley_sn_escoria_pct_prev"], df["ley_sio2_escoria_pct_prev"], umbral=1e-3
    )
    df["d_ratio_sn_sio2"] = df["ratio_sn_sio2"] - ratio_prev
    log_actual = np.log(df["ratio_sn_sio2"].where(df["ratio_sn_sio2"] > 0))
    log_prev = np.log(ratio_prev.where(ratio_prev > 0))
    df["d_log_ratio_sn_sio2"] = log_actual - log_prev
    return df


def agregar_indice_selectividad_sn_feo(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `indice_selectividad_sn_feo` (target/diagnostico de sobre-reduccion).

    Requiere `agregar_tasas_y_gradientes` ya aplicado (usa `d_ley_sn_escoria_pct`
    y `d_ley_feo_escoria_pct`).

        indice_selectividad_sn_feo = (-d_ley_sn_escoria_pct) / (-d_ley_feo_escoria_pct)
                                    = (caida de %Sn) / (caida de %FeO) en el escalon

    Definido SOLO cuando ambas variaciones tienen el signo esperable de una
    reduccion en curso (d_ley_sn_escoria_pct <= 0 y d_ley_feo_escoria_pct <= 0);
    en cualquier otro caso queda NaN en vez de un signo/magnitud sin sentido
    fisico -- exactamente la prevencion que pide consideracioness_pirometalurgia_ausmelt.md
    seccion 32 ("debe verse como feature empirico... con tratamiento robusto de
    signos", no como una eficiencia termodinamica).

    Fundamento: operacionaliza cuantitativamente la nocion de "selectividad
    Sn/Fe" de consideraciones seccion 32 (indice SRE) y seccion 72
    (Selectivity_Sn/Fe). Un valor alto significa que el escalon removio Sn de
    la escoria mucho mas rapido de lo que ataco el FeO (deseable); un valor
    bajo (mucha caida de FeO por poca caida de Sn) es la firma operacional de
    sobre-reduccion descrita en consideraciones secciones 5.3 y 11.3-11.4 (Fe
    hacia el metal, hardhead, dross de Fe en aumento). Es un indice EMPIRICO
    sobre leyes (%), no un balance de masa real (no hay masa de escoria por
    escalon en el dataset): dos escorias con igual ley pero distinta masa
    total pueden tener distinta selectividad real (ver
    guia_fenomenologica_ausmelt_sn.md seccion 11.8, "Por que un cambio de ley
    no es un cambio de masa").

    Uso recomendado: TARGET secundario/diagnostico (junto a
    `d_ley_feo_escoria_pct`) para identificar el punto donde seguir
    intensificando la reduccion deja de ser rentable -- el mecanismo que la
    funcion objetivo de un optimizador restringido (J_R, ver
    consideracioness_pirometalurgia_ausmelt.md seccion 38 y Guia_Ausmelt_Sn.md
    seccion 11.3) necesita para no perseguir el minimo de %Sn a cualquier costo
    de Fe reducido.
    """
    df = df.copy()
    d_sn = df["d_ley_sn_escoria_pct"]
    d_feo = df["d_ley_feo_escoria_pct"]
    ambos_bajan = d_sn.le(0) & d_feo.le(0)
    df["indice_selectividad_sn_feo"] = division_segura(
        (-d_sn).where(ambos_bajan), (-d_feo).where(ambos_bajan), umbral=1e-3
    )
    return df


def agregar_k_app_sn(df: pd.DataFrame, umbral_ley_sn_prev: float = UMBRAL_LEY_SN_PREV) -> pd.DataFrame:
    """Crea `k_app_sn` (target candidato): constante cinetica aparente [1/min].

    Requiere `agregar_estado_previo` y `agregar_tiempos` ya aplicados.

        k_app_sn = -ln(ley_sn_escoria_pct / ley_sn_escoria_pct_prev) / delta_tiempo
        (NaN si ley_sn_escoria_pct_prev < umbral_ley_sn_prev, o si alguna de las
        dos leyes no es estrictamente positiva)

    Fundamento (Guia_Ausmelt_Sn.md seccion 10.7): si el agotamiento de Sn en
    escoria siguiera una cinetica de primer orden respecto al Sn disponible
    (d[Sn]/dt ~ -k[Sn] -la hipotesis de trabajo detras de
    `tasa_relativa_sn_escoria`/T3, y validada de forma indirecta por el signo y
    la forma limpia y monotona de `interaccion_C_x_Sn_prev` documentada en
    hallazgos.md seccion 10.3-), k_app_sn seria la constante de velocidad
    aparente de esa cinetica. Se llama "aparente" porque combina reaccion
    quimica real, transferencia de masa, mezcla y, al no existir masa de
    escoria por escalon, tambien dilucion/concentracion (mismo caveat que
    aplica a T3 y a `ratio_sn_sio2`). Guia_Ausmelt_Sn.md seccion 10.7 advierte
    ademas sobre el ruido de diferencias/logaritmos en escalones cortos y el
    riesgo de regresion a la media: preferir esta feature como candidato de
    target adicional, no como sustituto automatico de T1/T2/T3 ya validados.

    Reutiliza el mismo umbral que T3 (UMBRAL_LEY_SN_PREV) para evitar razones
    inestables cuando el %Sn de partida ya esta muy cerca de 0 (tipico de
    escalones finales de Reduccion).
    """
    df = df.copy()
    ley_sn_prev_protegida = df["ley_sn_escoria_pct_prev"].where(
        df["ley_sn_escoria_pct_prev"] >= umbral_ley_sn_prev
    )
    ratio = df["ley_sn_escoria_pct"] / ley_sn_prev_protegida
    log_ratio = np.log(ratio.where(ratio > 0))
    df["k_app_sn"] = -log_ratio / df["delta_tiempo"]
    return df


# =============================================================================
# Codificacion de fase y KPIs de batch alternativos
# =============================================================================

def agregar_es_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `es_fusion` (Paso 16): dummy binaria 1=Fusion, 0=Reduccion.

    Permite entrenar un unico modelo combinado (Fusion+Reduccion) que pueda
    distinguir el regimen, en vez de dos modelos separados por fase.
    """
    df = df.copy()
    df["es_fusion"] = (df["fase_proceso"] == "Fusión").astype(int)
    return df


def construir_resumen_batch_masa(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla a nivel de Batch con `recuperacion_real_batch_pct` (Paso 14).

    KPI de batch alternativo a rendimiento_proxy_batch, anclado en balance de
    masa entrada/salida en vez de reparto entre salidas:
        recuperacion_real_batch_pct = 100 * sn_en_metal_crudo_batch_t
                                           / (sum(feed_Sn_kgf del batch) / 1000)
    A diferencia de rendimiento_proxy_batch (que reparte el Sn contabilizado
    entre 3 salidas sin compararlo contra cuanto Sn entro), esta metrica
    responde directamente "cuanto del Sn que entro termino como metal crudo".

    Limitacion a declarar: el denominador no incluye Sn residual/heel de
    batches anteriores que pudiera quedar en el horno, por lo que valores
    >100% o inconsistentes en algunos batches son posibles y no implican error
    de calculo, sino de los supuestos de "batch aislado".
    """
    resumen = df.groupby("Batch").agg(
        feed_sn_total_kg=("feed_Sn_kgf", "sum"),
        sn_en_metal_crudo_batch_t=("sn_en_metal_crudo_batch_t", "first"),
        rendimiento_proxy_batch=("rendimiento_proxy_batch", "first"),
    ).reset_index()
    resumen["feed_sn_total_t"] = resumen["feed_sn_total_kg"] / 1000
    resumen["recuperacion_real_batch_pct"] = 100 * division_segura(
        resumen["sn_en_metal_crudo_batch_t"], resumen["feed_sn_total_t"]
    )
    return resumen


# =============================================================================
# Bloque v6 (iteracion 6, 2026-09-14 noche): balance global de masa por batch y KPI refinado
# con el Sn que queda en la escoria final. Inspirado en el balance de masa del modelo
# teorico de los metalurgistas (`func-teorethical-model`, Rust): carga por tolva ->
# metal, dross, humos, escoria, gases. Evaluado en experimentos/balances/b04 (ADOPTAR).
# =============================================================================

# Pesos molares (IUPAC) usados por el balance global.
MASA_MOLAR_SN, MASA_MOLAR_FE, MASA_MOLAR_O = 118.71, 55.845, 15.999
MASA_MOLAR_SNO2 = 150.71

# Supuestos de planta del balance global (parametros; sensibilidad en b04_resultados.md:
# la masa absoluta de escoria varia 2-14 % con ellos, el KPI refinado < 0.2 pp).
SUPUESTOS_BALANCE_GLOBAL: dict[str, float] = {
    "LEY_SN_METAL": 0.96,    # fraccion de Sn en el metal crudo (Ausmelt ~0.95-0.98)
    "LEY_SN_DROSS": 0.65,    # fraccion de Sn en el dross de Fe (modelo Rust: factor_dross_fusion 65 %)
    "LEY_SN_POLVO": 0.55,    # fraccion de Sn en el polvo de fundicion (SnO2 + FeO + otros)
    "FE_MET_DROSS": 0.30,    # kg Fe metalico / kg dross de Fe (tolva 5 y dross de salida)
    "HUMEDAD_CARGA": 0.06,   # humedad de la carga humeda (tmh) distinta de la cal
    "CENIZA_CARBON": 0.10,   # fraccion de ceniza del carbon (va a escoria)
}


def construir_resumen_batch_balance_global(df: pd.DataFrame, supuestos: dict | None = None) -> pd.DataFrame:
    """Tabla a nivel de Batch con el BALANCE GLOBAL DE MASA y el KPI refinado con Sn en escoria.

    Estructura del balance del modelo teorico Rust aplicada a los datos del Excel. Por batch,
    con las salidas de Sn del batch M (metal), D (dross), P (polvo) [kg] y la carga por tolva
    (F+R):
        carga_solida = sum(feed_total_kgh) + sum(feed_CaO_kgh)
        masa_escoria_balance_kg = carga_solida
            - HUMEDAD_CARGA * sum(feed_total_kgh)                (humedad evaporada; cal seca)
            - (M + D + P) * M_SnO2/M_Sn                          (el Sn salio como SnO2: su O se fue en CO)
            - Fe_metalizado * M_FeO/M_Fe                         (FeO reducido a Fe que salio en metal/dross)
            + Fe_met_dross_in * M_O/M_Fe                         (el Fe metalico del dross cargado se escorifica)
            - (P/LEY_SN_POLVO - P)                               (polvo no-Sn arrastrado)
            + CENIZA_CARBON * sum(feed_Carbon_kgh)
        Fe_metalizado = 0.8 * M/LEY_SN_METAL * (1 - LEY_SN_METAL) + D/LEY_SN_DROSS * FE_MET_DROSS
        sn_escoria_final_balance_kg = masa_escoria_balance_kg * ley_sn_escoria_pct[ultimo escalon de
                                      Reduccion con ensayo] / 100
        recuperacion_refinada_pct   = 100 * M / (M + D + P + sn_escoria_final_balance_kg)
        sn_perdido_escoria_frac     = sn_escoria_final_balance_kg / (M + D + P + sn_escoria_final_balance_kg)
        cierre_sn_frac              = (sum(feed_Sn_kgf) - (M + D + P) - sn_escoria_final) / sum(feed_Sn_kgf)

    Que aporta (experimentos/balances/b04): la masa final de escoria por balance (mediana 57.9 t)
    supera en ~9 % a la del trazador CaO (53.5 t) y NO depende de la mezcla de cal (el "factor de
    escala" trazador->balance esta confundido con la fraccion de cal, rho -0.35: es el artefacto
    contable de exp 18, por eso NO se usa un factor multiplicativo). El Sn en la escoria final es
    ~1.3 % del Sn de salida (IQR 1.0-1.7 %); el KPI refinado correlaciona 0.99 con
    rendimiento_proxy_batch pero reclasifica el 9.4 % de los batches a otro cuartil, es robusto a
    los supuestos (rango < 0.2 pp) y su techo de predictibilidad iguala o supera al del proxy en
    lockbox. Con el KPI refinado, en lockbox replican los targets de matriz de escoria (ln IRF en
    Reduccion, IRF fin de Fusion) y no el SDI.

    Limitaciones: `cierre_sn_frac` tiene sesgo medio -1.9 % (p 4e-8, sd 6.6 %): perdidas no
    modeladas (volatilizacion, splash) o sesgo conjunto de supuestos; las leyes de metal/dross/polvo y
    la humedad no estan medidas en el dataset (ver SUPUESTOS_BALANCE_GLOBAL). No modela talon de
    escoria entre batches (entra y sale la misma cantidad en estado estacionario).
    """
    p = dict(SUPUESTOS_BALANCE_GLOBAL)
    if supuestos:
        p.update(supuestos)
    filas = []
    for b, g in df.sort_values(["Batch", "fecha_inicio"]).groupby("Batch", sort=False):
        gr = g[g["fase_proceso"] == "Reducción"].dropna(subset=["masa_escoria_est_kg"]) if "masa_escoria_est_kg" in g else g[g["fase_proceso"] == "Reducción"]
        tot_h = g["feed_total_kgh"].fillna(0).sum()
        cal = g["feed_CaO_kgh"].fillna(0).sum()
        carbon = g["feed_Carbon_kgh"].fillna(0).sum()
        dross_in = g["feed_dross_Fe_kgh"].fillna(0).sum()
        feed_sn = g["feed_Sn_kgf"].fillna(0).sum()
        M, D, P = (g[c].iloc[0] * 1000 for c in ["sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"])
        sn_out = M + D + P
        fe_metal = 0.8 * M / p["LEY_SN_METAL"] * (1 - p["LEY_SN_METAL"]) + D / p["LEY_SN_DROSS"] * p["FE_MET_DROSS"]
        masa = (tot_h + cal - tot_h * p["HUMEDAD_CARGA"] - sn_out * MASA_MOLAR_SNO2 / MASA_MOLAR_SN
                - fe_metal * PESO_MOLECULAR_FEO / MASA_MOLAR_FE + dross_in * p["FE_MET_DROSS"] * MASA_MOLAR_O / MASA_MOLAR_FE
                - (P / p["LEY_SN_POLVO"] - P) + carbon * p["CENIZA_CARBON"])
        ley_fin = gr["ley_sn_escoria_pct"].dropna().iloc[-1] if gr["ley_sn_escoria_pct"].notna().any() else np.nan
        sn_esc = masa * ley_fin / 100
        filas.append(dict(
            Batch=b, carga_solida_kg=tot_h + cal, feed_sn_total_kg=feed_sn,
            sn_metal_kg=M, sn_dross_kg=D, sn_polvo_kg=P,
            masa_escoria_balance_kg=masa, ley_sn_escoria_final_pct=ley_fin, sn_escoria_final_balance_kg=sn_esc,
            rendimiento_proxy_batch=g["rendimiento_proxy_batch"].iloc[0] if "rendimiento_proxy_batch" in g else np.nan,
            recuperacion_refinada_pct=100 * M / (sn_out + sn_esc) if np.isfinite(sn_esc) and sn_out + sn_esc > 0 else np.nan,
            sn_perdido_escoria_frac=sn_esc / (sn_out + sn_esc) if np.isfinite(sn_esc) and sn_out + sn_esc > 0 else np.nan,
            cierre_sn_frac=(feed_sn - sn_out - sn_esc) / feed_sn if feed_sn > 0 and np.isfinite(sn_esc) else np.nan,
        ))
    return pd.DataFrame(filas).replace([np.inf, -np.inf], np.nan)


def construir_features_transicion_fase(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla a nivel de Batch con las variables `delta_FR_<var>` (salto Fusion->Reduccion).

    Requiere que `df` ya tenga `fase_proceso`, `orden_escalon_fase` (ver
    `agregar_orden_escalon`) y las variables de estado de interes ya calculadas
    (`ley_feo_escoria_pct` via `agregar_ley_feo`, `exceso_o2_combustion_pct` via
    `agregar_features_pirometalurgicas`).

    Para cada Batch, toma el ULTIMO escalon de Fusion (orden_escalon_fase maximo
    dentro de fase_proceso=='Fusión') y el PRIMER escalon de Reduccion
    (orden_escalon_fase==0, fase_proceso=='Reducción'), y calcula:

        delta_FR_y = y(R0) - y(F_ultimo)

    para y en {ley_sn_escoria_pct, ley_feo_escoria_pct, temperatura_horno_celsius,
    presion_punta_lanza_kpa, exceso_o2_combustion_pct} (se omite silenciosamente
    cualquier columna no presente en `df`).

    Fundamento (hallazgos.md seccion 3; guia_fenomenologica_ausmelt_sn.md
    seccion 16.6): la transicion Fusion->Reduccion es, con diferencia, el
    evento mas fuerte y mas universal de todo el batch -- la tasa de carbon
    salta a su maximo del batch y la caida de `ley_sn_escoria_pct` del primer
    escalon de Reduccion es la mayor de todo el batch (mediana ~-12 puntos,
    presente en ~100% de los batches evaluables). `es_transicion_FR` (ver
    `agregar_orden_escalon`) marca la fila donde ocurre este evento, pero no
    cuantifica su MAGNITUD; estas columnas `delta_FR_*` sí la cuantifican, una
    vez por batch, y sirven como features de entrada para un modelo a nivel de
    BATCH (Guia_Ausmelt_Sn.md seccion 12.4) o como diagnostico de que tan
    abrupta fue la transicion en cada corrida.

    Advertencia de uso: NO usar `delta_FR_*` como feature de un modelo de
    ESCALON dentro de Fusion -- para esos escalones el estado de Reduccion
    todavia no existe (seria fuga hacia informacion futura del propio batch).
    Es valida como feature de un modelo a nivel de Batch (evaluado al cierre) o
    como feature de ESTADO PREVIO para escalones de Reduccion posteriores al
    primero (R1 en adelante), ya que ahi el salto F->R ya ocurrio antes de esos
    escalones.
    """
    variables = [
        "ley_sn_escoria_pct", "ley_feo_escoria_pct", "temperatura_horno_celsius",
        "presion_punta_lanza_kpa", "exceso_o2_combustion_pct",
    ]
    variables_presentes = [v for v in variables if v in df.columns]

    ultimo_fusion = (
        df[df["fase_proceso"] == "Fusión"]
        .sort_values("orden_escalon_fase")
        .groupby("Batch")[variables_presentes]
        .last()
    )
    primero_reduccion = (
        df[(df["fase_proceso"] == "Reducción") & (df["orden_escalon_fase"] == 0)]
        .groupby("Batch")[variables_presentes]
        .first()
    )
    delta = primero_reduccion.subtract(ultimo_fusion).add_prefix("delta_FR_")
    return delta.reset_index()


def construir_resumen_batch_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla exploratoria a nivel de Batch con `relacion_C_reduccion_vs_fusion` (Paso 5).

    Agregados simples por batch (sumas/promedios por fase) usados para
    contrastar contra rendimiento_proxy_batch en el EDA. Incluye:
        relacion_C_reduccion_vs_fusion = carbon_total_reduccion_kg / carbon_total_fusion_kg
    Nota: estos agregados lineales simples correlacionan debil (|Spearman|<=0.13)
    con rendimiento_proxy_batch (ver hallazgos.md seccion 6); de ahi que el
    modelado se haga a nivel de escalon (VARIABLES_LAG_ESTADO, etc.) y no con
    esta tabla.
    """
    filas_batch = []
    for batch_id, g in df.groupby("Batch"):
        g = g.sort_values("fecha_inicio")
        filas_batch.append({
            "Batch": batch_id,
            "rendimiento_proxy_batch": g["rendimiento_proxy_batch"].iloc[0],
            "carbon_total_fusion_kg": g.loc[g["fase_proceso"] == "Fusión", "feed_Carbon_kgh"].sum(),
            "carbon_total_reduccion_kg": g.loc[g["fase_proceso"] == "Reducción", "feed_Carbon_kgh"].sum(),
            "exceso_o2_medio_fusion_pct": g.loc[g["fase_proceso"] == "Fusión", "exceso_o2_combustion_pct"].mean(),
            "exceso_o2_medio_reduccion_pct": g.loc[g["fase_proceso"] == "Reducción", "exceso_o2_combustion_pct"].mean(),
            "basicidad_media_fusion": g.loc[g["fase_proceso"] == "Fusión", "basicidad_B2"].mean(),
            "ley_sn_escoria_final_pct": g["ley_sn_escoria_pct"].iloc[-1],
            "ley_sn_escoria_max_pct": g["ley_sn_escoria_pct"].max(),
            "caida_ley_sn_reduccion_pct": g.loc[g["fase_proceso"] == "Reducción", "d_ley_sn_escoria_pct"].sum(),
            "temperatura_media_horno": g["temperatura_horno_celsius"].mean(),
        })
    df_batch = pd.DataFrame(filas_batch)
    df_batch["relacion_C_reduccion_vs_fusion"] = division_segura(
        df_batch["carbon_total_reduccion_kg"], df_batch["carbon_total_fusion_kg"]
    )
    return df_batch


# Zona de "valle" de posicion_vertical_lanza_mm hallada empiricamente en Reduccion
# (hallazgos.md seccion 12, Pasos 46-47): tramo de la variable cruda donde el modelo
# de escalon predice simultaneamente la mayor caida de %Sn (deseable) y de %FeO
# (riesgo de hardhead) -- NO es un optimo en el sentido de "mejor resultado", es la
# region de mayor intensidad de agitacion/poder reductor. Se usa aqui solo para
# construir un INDICADOR descriptivo de cuanto tiempo pasa un batch en esa region,
# no para imponer una restriccion monotonica (seria pirometalurgicamente incorrecto
# sobre una relacion con optimo interior, ver consideraciones seccion 84).
VALLE_LANZA_MM = (3300, 4200)
# Basicidad "objetivo" ya usada como parametro de negocio explicito en el optimizador
# (modelo_prescriptivo.PESOS_J_F_DEFAULT["basicidad_optima"]); se reutiliza aqui solo
# para describir que fraccion de la Fusion opero cerca de ese objetivo, no se re-deriva.
BASICIDAD_OPTIMA_FUSION = 1.4


def construir_features_trayectoria_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla a nivel de Batch con features de TRAYECTORIA (perfil de dosificacion,
    dispersion, fracciones de tiempo en zonas de riesgo), no solo sumas/promedios
    simples como `construir_resumen_batch_eda` (ya documentada como debil: hallazgos.md
    seccion 6, |Spearman|<=0.13 con `rendimiento_proxy_batch`).

    Motivacion (hallazgos.md seccion 6b/9.5/11.7; consideraciones seccion 93,
    "Outcome = f6(trajectory)"): si el vinculo escalon->batch es debil con agregados
    lineales, la siguiente hipotesis razonable no es "mas features lineales" sino
    "features que resuman la FORMA de la trayectoria" (perfil de carbon, cuanto
    tiempo la lanza pasa en la zona de riesgo ya identificada, etc.), evaluadas con
    un modelo NO lineal (ver `modelo_predictivo_v2.entrenar_modelo_batch`).

    Requiere que `df` ya tenga aplicado `construir_features` + `construir_features_modelado`
    (usa `tasa_feed_Carbon_kg_min`, `basicidad_B2`, `exceso_o2_combustion_pct`,
    `interaccion_C_x_Sn_prev`, `d_ley_feo_escoria_pct`, etc.). Excluye batches sin
    ambas fases presentes (dataset incompleto para ese batch).

    IMPORTANTE (misma disciplina anti-fuga que el resto del modulo): esta tabla
    resume el batch COMPLETO ya cerrado -- valida como feature de entrada de un
    modelo que predice/explica el KPI FINAL de un batch ya terminado (o para
    comparar politicas agregadas entre batches), NUNCA como feature de un modelo
    de escalon que recomienda la siguiente accion dentro de un batch en curso
    (en ese momento la trayectoria futura del batch todavia no existe).
    """
    filas = []
    for batch_id, g in df.groupby("Batch"):
        g = g.sort_values("fecha_inicio")
        gf = g.loc[g["fase_proceso"] == "Fusión"]
        gr = g.loc[g["fase_proceso"] == "Reducción"]
        if gf.empty or gr.empty:
            continue

        fila = {"Batch": batch_id, "rendimiento_proxy_batch": g["rendimiento_proxy_batch"].iloc[0]}

        fila["fusion_carbon_total_kg"] = gf["feed_Carbon_kgh"].sum()
        fila["fusion_carbon_rate_mean"] = gf["tasa_feed_Carbon_kg_min"].mean()
        fila["fusion_carbon_rate_cv"] = division_segura(
            pd.Series([gf["tasa_feed_Carbon_kg_min"].std()]), pd.Series([gf["tasa_feed_Carbon_kg_min"].mean()])
        ).iloc[0]
        fila["fusion_b2_mean"] = gf["basicidad_B2"].mean()
        fila["fusion_frac_b2_optimo"] = gf["basicidad_B2"].sub(BASICIDAD_OPTIMA_FUSION).abs().lt(0.3).mean()
        fila["fusion_o2_exceso_mean"] = gf["exceso_o2_combustion_pct"].mean()
        fila["fusion_o2_exceso_std"] = gf["exceso_o2_combustion_pct"].std()
        fila["fusion_duracion_total_min"] = gf["delta_tiempo"].sum()
        fila["fusion_ley_sn_pico"] = gf["ley_sn_escoria_pct"].max()
        fila["fusion_ley_sn_final"] = gf["ley_sn_escoria_pct"].iloc[-1]
        fila["fusion_temp_mean"] = gf["temperatura_horno_celsius"].mean()
        fila["fusion_temp_trend"] = _pendiente_lineal(gf["temperatura_horno_celsius"].to_numpy())
        fila["fusion_cao_total_kg"] = gf["feed_CaO_kgh"].sum()
        fila["fusion_interaccion_C_x_Sn_prev_mean"] = gf["interaccion_C_x_Sn_prev"].mean() if "interaccion_C_x_Sn_prev" in gf else np.nan

        ultimo_f, primero_r = gf.iloc[-1], gr.iloc[0]
        for var in ["ley_sn_escoria_pct", "ley_feo_escoria_pct", "temperatura_horno_celsius",
                    "presion_punta_lanza_kpa", "exceso_o2_combustion_pct"]:
            if var in g.columns:
                fila[f"delta_FR_{var}"] = primero_r.get(var, np.nan) - ultimo_f.get(var, np.nan)

        tasa_c = gr["tasa_feed_Carbon_kg_min"].to_numpy()
        fila["reduccion_carbon_total_kg"] = gr["feed_Carbon_kgh"].sum()
        fila["reduccion_carbon_rate_first"] = tasa_c[0] if len(tasa_c) else np.nan
        fila["reduccion_carbon_decay_ratio"] = (tasa_c[-1] / tasa_c[0]) if len(tasa_c) and tasa_c[0] > 1e-6 else np.nan
        fila["reduccion_gn_total_nm3"] = gr["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum()
        fila["reduccion_o2_total_nm3"] = gr["volumen_o2_inyectado_lanza_escalon_nm3"].sum()
        fila["reduccion_aire_total_nm3"] = gr["volumen_aire_inyectado_lanza_escalon_nm3"].sum()
        fila["reduccion_o2_exceso_mean"] = gr["exceso_o2_combustion_pct"].mean()
        fila["reduccion_lanza_pos_mean"] = gr["posicion_vertical_lanza_mm"].mean()
        fila["reduccion_lanza_pos_std"] = gr["posicion_vertical_lanza_mm"].std()
        fila["reduccion_frac_en_valle_lanza"] = gr["posicion_vertical_lanza_mm"].between(*VALLE_LANZA_MM).mean()
        selectividad = division_segura(-gr["d_ley_sn_escoria_pct"], -gr["d_ley_feo_escoria_pct"])
        fila["reduccion_selectividad_media"] = selectividad.replace([np.inf, -np.inf], np.nan).mean()
        fila["reduccion_duracion_total_min"] = gr["delta_tiempo"].sum()
        fila["reduccion_ley_sn_inicio"] = gr["ley_sn_escoria_pct_prev"].iloc[0] if "ley_sn_escoria_pct_prev" in gr else np.nan
        fila["reduccion_ley_sn_final"] = gr["ley_sn_escoria_pct"].iloc[-1]
        fila["reduccion_ley_feo_final"] = gr["ley_feo_escoria_pct"].iloc[-1] if "ley_feo_escoria_pct" in gr else np.nan
        fila["reduccion_feo_caida_total"] = gr["d_ley_feo_escoria_pct"].sum() if "d_ley_feo_escoria_pct" in gr else np.nan
        fila["reduccion_sn_caida_total"] = gr["d_ley_sn_escoria_pct"].sum()
        fila["n_escalones_reduccion"] = len(gr)
        filas.append(fila)

    df_batch = pd.DataFrame(filas).set_index("Batch")
    df_batch = df_batch.join(
        construir_resumen_batch_masa(df).set_index("Batch")[["recuperacion_real_batch_pct"]], how="left"
    )
    return df_batch


def _pendiente_lineal(y: np.ndarray) -> float:
    """Pendiente de una regresion lineal simple de `y` contra su indice (0..n-1);
    NaN si hay menos de 2 puntos validos. Usada para resumir tendencias dentro
    de una fase (ej. `fusion_temp_trend`) sin depender de statsmodels."""
    y = np.asarray(y, dtype=float)
    idx = np.arange(len(y))
    valido = ~np.isnan(y)
    if valido.sum() < 2:
        return np.nan
    return float(np.polyfit(idx[valido], y[valido], 1)[0])


# =============================================================================
# Interacciones y estequiometria adicional (Pasos 17 y 20)
# =============================================================================

MASA_MOLAR_C = 12.011    # g/mol
MASA_MOLAR_SN = 118.71   # g/mol
# SnO2 + 2C -> Sn + 2CO: kg de C requeridos por kg de Sn fino, estequiometrico.
FACTOR_ESTEQ_C_SN = 2 * MASA_MOLAR_C / MASA_MOLAR_SN


def agregar_interacciones_carbon_y_estequiometria(
    df: pd.DataFrame, umbral_ley_sn_prev: float = UMBRAL_LEY_SN_PREV
) -> pd.DataFrame:
    """Crea interacciones de carbon y features estequiometricas/termicas (Paso 17).

    Requiere `agregar_estado_previo`, `agregar_tasas_y_gradientes` y
    `agregar_features_pirometalurgicas` ya aplicados.

    Interacciones carbon x estado previo/redox (productos, no razones -> sin
    riesgo nuevo de division inestable):
        interaccion_C_x_Sn_prev    = tasa_feed_Carbon_kg_min * ley_sn_escoria_pct_prev
        interaccion_C_x_FeO_prev   = tasa_feed_Carbon_kg_min * ley_feo_escoria_pct_prev
        interaccion_C_x_temp_prev  = tasa_feed_Carbon_kg_min * temperatura_horno_celsius_prev
        interaccion_C_x_exceso_o2  = tasa_feed_Carbon_kg_min * exceso_o2_combustion_pct

    Reductor por inventario de escoria (reutiliza UMBRAL_LEY_SN_PREV del Paso 11):
        carbon_sobre_inventario_sn_prev = tasa_feed_Carbon_kg_min / ley_sn_escoria_pct_prev
        (NaN si ley_sn_escoria_pct_prev < umbral_ley_sn_prev)

    Estequiometria exacta y proxies de carga/termico (SOLO validas como
    candidatas en Fusion: denominador ~0 por diseno de proceso en Reduccion,
    que casi no alimenta Sn/mineral fresco; se calculan para todo el df por
    simplicidad, pero solo deben usarse en el pool de features de Fusion):
        exceso_C_estequiometrico_pct = 100 * (feed_Carbon_kgh - FACTOR_ESTEQ_C_SN*feed_Sn_kgf)
                                            / (FACTOR_ESTEQ_C_SN*feed_Sn_kgf)
        oxidante_sobre_carga_total   = o2_disponible_nm3 / feed_total_kgh
        carga_termica_relativa       = volumen_gas_natural_inyectado_lanza_escalon_nm3 / feed_total_kgh
    """
    df = df.copy()
    df["interaccion_C_x_Sn_prev"] = df["tasa_feed_Carbon_kg_min"] * df["ley_sn_escoria_pct_prev"]
    df["interaccion_C_x_FeO_prev"] = df["tasa_feed_Carbon_kg_min"] * df["ley_feo_escoria_pct_prev"]
    df["interaccion_C_x_temp_prev"] = df["tasa_feed_Carbon_kg_min"] * df["temperatura_horno_celsius_prev"]
    df["interaccion_C_x_exceso_o2"] = df["tasa_feed_Carbon_kg_min"] * df["exceso_o2_combustion_pct"]

    df["carbon_sobre_inventario_sn_prev"] = division_segura(
        df["tasa_feed_Carbon_kg_min"],
        df["ley_sn_escoria_pct_prev"].where(df["ley_sn_escoria_pct_prev"] >= umbral_ley_sn_prev),
    )

    df["exceso_C_estequiometrico_pct"] = 100 * division_segura(
        df["feed_Carbon_kgh"] - FACTOR_ESTEQ_C_SN * df["feed_Sn_kgf"],
        FACTOR_ESTEQ_C_SN * df["feed_Sn_kgf"],
    )
    df["oxidante_sobre_carga_total"] = division_segura(df["o2_disponible_nm3"], df["feed_total_kgh"])
    df["carga_termica_relativa"] = division_segura(
        df["volumen_gas_natural_inyectado_lanza_escalon_nm3"], df["feed_total_kgh"]
    )
    return df


def agregar_interacciones_feo_basicidad_temp(df: pd.DataFrame) -> pd.DataFrame:
    """Crea interacciones FeO/basicidad/temperatura previas (Paso 20).

    Requiere `agregar_estado_previo` ya aplicado (usa las versiones `_prev`).
    Productos (no ratios) para evitar el riesgo de division inestable de sus
    equivalentes en razon (p.ej. ratio_sn_feo_prev cuando %FeO es bajo):
        interaccion_FeO_x_B2_prev  = ley_feo_escoria_pct_prev * basicidad_B2_prev
            Riqueza conjunta de los dos principales modificadores de red de la
            escoria al iniciar el escalon.
        interaccion_B2_x_temp_prev = basicidad_B2_prev * temperatura_horno_celsius_prev
            La basicidad solo se traduce en fluidez real si hay suficiente
            temperatura; el producto aproxima ese efecto conjunto.
        interaccion_Sn_x_FeO_prev  = ley_sn_escoria_pct_prev * ley_feo_escoria_pct_prev
            Alternativa mas robusta a ratio_sn_feo_prev: inventario conjunto de
            ambos oxidos competidores (redox Sn-Fe) sin dividir.
    """
    df = df.copy()
    df["interaccion_FeO_x_B2_prev"] = df["ley_feo_escoria_pct_prev"] * df["basicidad_B2_prev"]
    df["interaccion_B2_x_temp_prev"] = df["basicidad_B2_prev"] * df["temperatura_horno_celsius_prev"]
    df["interaccion_Sn_x_FeO_prev"] = df["ley_sn_escoria_pct_prev"] * df["ley_feo_escoria_pct_prev"]
    return df


def agregar_gradiente_temperatura_previo(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `grad_temperatura_horno_celsius_prev` (Paso 32): version sin fuga del gradiente termico.

    grad_temperatura_horno_celsius (Paso 1) usa la temperatura CONTEMPORANEA
    del escalon (t vs t-1), por lo que no es valida como feature para predecir
    el resultado de ese mismo escalon t (fuga de informacion). Esta version
    desplaza toda la serie un escalon hacia atras dentro de cada Batch:
        grad_temperatura_horno_celsius_prev[i] = grad_temperatura_horno_celsius[i-1]
    es decir, la tendencia termica ya conocida ANTES del escalon t (calculada
    con temperaturas hasta t-1 vs t-2).
    """
    df = df.copy()
    df["grad_temperatura_horno_celsius_prev"] = df.groupby("Batch")["grad_temperatura_horno_celsius"].shift(1)
    return df


# =============================================================================
# Orquestador (features de modelado: Pasos 10-32)
# =============================================================================

def construir_features_modelado(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica, en orden, la ingenieria de variables de MODELADO (Pasos 10-32 del notebook).

    Requiere que `df` ya haya pasado por `construir_features` (necesita
    basicidad_B2, tasas, gradientes y delta_tiempo). No incluye
    `agregar_target_tasa_relativa_sn` por ser un candidato de TARGET, no una
    feature predictora (se deja disponible para llamarse aparte cuando se
    necesite ese target).
    """
    df = agregar_estado_previo(df)
    df = agregar_acumulados_previos(df)
    df = agregar_ratios_composicionales_previos(df)
    df = agregar_features_lanza_redox_extra(df)
    df = agregar_delta_temperatura_gases(df)
    df = agregar_es_fusion(df)
    df = agregar_interacciones_carbon_y_estequiometria(df)
    df = agregar_interacciones_feo_basicidad_temp(df)
    df = agregar_gradiente_temperatura_previo(df)
    return df


# =============================================================================
# v3 (sesion 2026-09-13 noche): carga POR TOLVA, plan de duracion, estado termico
# del refractario, inventario de masa por trazador CaO y targets EN MASA
# =============================================================================
#
# Motivacion (pedido explicito del usuario + hallazgos.md 6b/9.5/11.7/13.7): el
# modelo de escalon solo veia la carga como dos totales casi colineales
# (feed_Sn_kgf ~ feed_total_kgh, r=0.9988) y no existia masa de escoria por
# escalon, el cuello de botella declarado para conectar escalon -> batch. El
# Excel fuente si trae (a) la masa humeda y el Sn fino POR TOLVA (concentrado,
# reciclo Sn/Fe, mineral de Fe WF, dross de Fe, pellets/humos), (b) la duracion
# PLANIFICADA del escalon, (c) el estado termico del refractario (8 termocuplas
# de carcasa + espesor de ladrillo) y (d) con la cal (tolva 6) como trazador
# inerte, una estimacion de la masa de escoria por escalon.
#
# Verificaciones sobre el Excel que fundamentan este bloque (3982 filas):
#   - TMF Sn total = suma exacta de tmf tolvas 1,3,4,5,7; TMH total = suma exacta
#     de tmh de esas tolvas.
#   - tmf_i = tmh_i * ley_i con ley_i CONSTANTE dentro de cada batch (std
#     intra-batch = 0.000 en las 5 tolvas): el "Sn fino por tolva" es un ensayo
#     por corriente asignado al batch, no un ensayo por escalon. Por eso aqui se
#     separa en (i) MEZCLA de carga por escalon (fracciones/tasas, ACTION) y (ii)
#     ENSAYO por corriente del batch (ley_sn_<corriente>_batch_pct, CONTEXT).
#   - "CaO total (kg)" del Excel = tolva 6 + tolva 3 exactamente; la cal pura es
#     tolva 6 (feed_CaO_kgh desde esta sesion, ver dataset_lingo_smelter.py).
#   - El %Sn del primer escalon de un batch NO correlaciona con el %Sn final del
#     batch anterior (r=-0.005, 362 batches): no hay heel de escoria detectable,
#     los batches se tratan como independientes.

TOLVAS_CARGA: dict[str, tuple[str, str, str]] = {
    # corriente -> (masa humeda del escalon, Sn fino del escalon, nombre de la tasa kg/min)
    "conc": ("feed_conc_t1_kgh", "sn_conc_t1_kgf", "tasa_feed_conc_kg_min"),
    "reciclo": ("feed_reciclo_t3_kgh", "sn_reciclo_t3_kgf", "tasa_feed_reciclo_kg_min"),
    "Fe": ("feed_Fe_kgh", "sn_Fe_t4_kgf", "tasa_feed_Fe_kg_min"),                 # ya existe (VARIABLES_FEED_VOLUMEN_A_TASA)
    "dross": ("feed_dross_Fe_kgh", "sn_dross_Fe_t5_kgf", "tasa_feed_dross_Fe_kg_min"),  # ya existe
    "pellets": ("feed_pellets_t7_kgh", "sn_pellets_t7_kgf", "tasa_feed_pellets_kg_min"),
}
CORRIENTES_SECUNDARIAS = ["reciclo", "dross", "pellets"]  # carga que no es concentrado ni mineral de Fe
COLUMNAS_TERMOCUPLAS_CARCASA = [
    "termocupla_100mm_NO_celsius", "termocupla_50mm_N_celsius", "termocupla_100mm_E_celsius",
    "termocupla_50mm_O_celsius", "termocupla_75mm_SE_celsius", "termocupla_150mm_NE_celsius",
    "termocupla_100mm_SO_celsius", "termocupla_0mm_SO_celsius",
]
# Por debajo de esta masa el escalon se considera "sin carga" (fracciones de mezcla = NaN,
# tipico de Reduccion, donde ~75% de los escalones no alimentan solidos).
UMBRAL_CARGA_KG = 50.0
# Por debajo de este Sn fino alimentado, las fracciones "por kg de Sn cargado" quedan NaN.
UMBRAL_SN_FED_KG = 20.0


def agregar_features_carga_por_tolva(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la familia de features de CARGA POR TOLVA (entrada, pirometalurgicamente interpretable).

    Requiere `delta_tiempo` (agregar_tiempos). Todas usan solo insumos del propio
    escalon t o ensayos del batch conocidos antes de empezar -> seguras para prescripcion.

    a) Tasas por corriente [kg/min]  (ACTION derivada):
           tasa_feed_conc_kg_min, tasa_feed_reciclo_kg_min, tasa_feed_pellets_kg_min
       (tasa_feed_Fe_kg_min y tasa_feed_dross_Fe_kg_min ya existen).
    b) Mezcla de carga del escalon (ACTION derivada, adimensional, NaN si feed_total < UMBRAL_CARGA_KG):
           frac_<corriente>_carga = feed_<corriente> / feed_total_kgh
           frac_carga_secundaria  = (reciclo + dross + pellets) / feed_total_kgh
       La proporcion de carga secundaria (reciclos ricos en Sn pero con Fe metalico/
       hardhead y finos de humos) cambia la demanda de O2, la carga de Fe hacia la
       escoria y la fraccion de finos que vuelve a fumar (Guia_Ausmelt_Sn.md 4.x).
    c) Ley de Sn de la mezcla cargada [%] (ACTION derivada; combina la mezcla con el
       ensayo del batch): ley_sn_carga_pct = 100*feed_Sn_kgf/feed_total_kgh.
    d) Ensayo de Sn por corriente del BATCH [%] (CONTEXT, constante dentro del batch,
       conocido al inicio):  ley_sn_<corriente>_batch_pct = 100*sum(tmf_i)/sum(tmh_i)
       y ley_sn_carga_batch_pct = 100*sum(feed_Sn_kgf)/sum(feed_total_kgh).
    e) Practica de fundente y reductor (ACTION derivada):
           relacion_CaO_carga  = feed_CaO_kgh / feed_total_kgh      [kg cal / kg carga]
           relacion_C_Sn_carga = feed_Carbon_kgh / feed_Sn_kgf       [kg C / kg Sn cargado]
               (estequiometrico SnO2+2C->Sn+2CO: FACTOR_ESTEQ_C_SN ~ 0.2024)
           tasa_feed_Carbon_fusion_kg_min / tasa_feed_Carbon_reduccion_kg_min: desglose de
           la tolva 2 por etapa de dosificacion.
    f) Duracion planificada vs real: `duracion_plan_min` (columna original, ACTION
       primitiva: se decide ANTES del escalon, 77.5% de los escalones la cumplen
       exactamente) y desvio_duracion_min = delta_tiempo - duracion_plan_min
       (diagnostico; solo conocido al cierre -> LEAKAGE como feature).
    g) Estado termico del refractario: termocupla_media_celsius (media de las 8
       termocuplas de carcasa, contemporanea -> LEAKAGE) y su version
       termocupla_media_celsius_prev (STATE). Junto con espesor_ladrillo_norm_mm
       (CONTEXT, decrece monotonamente 300->218 mm a lo largo de la campana)
       describen las perdidas de calor por las paredes (consideraciones sec. 8).
    """
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    nuevas: dict[str, pd.Series] = {}
    por_batch = df.groupby("Batch")
    carga_valida = df["feed_total_kgh"].where(df["feed_total_kgh"] > UMBRAL_CARGA_KG)

    for corriente, (col_h, col_f, col_tasa) in TOLVAS_CARGA.items():
        if col_tasa not in df.columns:
            nuevas[col_tasa] = df[col_h] / df["delta_tiempo"]
        nuevas[f"frac_{corriente}_carga"] = df[col_h] / carga_valida
        suma_h = por_batch[col_h].transform("sum")
        suma_f = por_batch[col_f].transform("sum")
        nuevas[f"ley_sn_{corriente}_batch_pct"] = 100 * division_segura(suma_f, suma_h, umbral=UMBRAL_CARGA_KG)

    secundaria = sum(df[TOLVAS_CARGA[c][0]] for c in CORRIENTES_SECUNDARIAS)
    nuevas["feed_carga_secundaria_kgh"] = secundaria
    nuevas["frac_carga_secundaria"] = secundaria / carga_valida
    nuevas["ley_sn_carga_pct"] = 100 * df["feed_Sn_kgf"] / carga_valida
    nuevas["ley_sn_carga_batch_pct"] = 100 * division_segura(
        por_batch["feed_Sn_kgf"].transform("sum"), por_batch["feed_total_kgh"].transform("sum"), umbral=UMBRAL_CARGA_KG
    )
    nuevas["relacion_CaO_carga"] = df["feed_CaO_kgh"] / carga_valida
    nuevas["relacion_C_Sn_carga"] = df["feed_Carbon_kgh"] / df["feed_Sn_kgf"].where(df["feed_Sn_kgf"] > UMBRAL_SN_FED_KG)
    nuevas["tasa_feed_Carbon_fusion_kg_min"] = df["feed_Carbon_fusion_kgh"] / df["delta_tiempo"]
    nuevas["tasa_feed_Carbon_reduccion_kg_min"] = df["feed_Carbon_reduccion_kgh"] / df["delta_tiempo"]

    nuevas["desvio_duracion_min"] = df["delta_tiempo"] - df["duracion_plan_min"]

    termocuplas = [c for c in COLUMNAS_TERMOCUPLAS_CARCASA if c in df.columns]
    tc_media = df[termocuplas].mean(axis=1) if termocuplas else pd.Series(np.nan, index=df.index)
    nuevas["termocupla_media_celsius"] = tc_media
    nuevas["termocupla_media_celsius_prev"] = tc_media.groupby(df["Batch"]).shift(1)

    return pd.concat([df, pd.DataFrame(nuevas, index=df.index)], axis=1)


def agregar_inventario_masa_escoria(df: pd.DataFrame) -> pd.DataFrame:
    """Crea el INVENTARIO de masa del horno y la masa de escoria estimada por trazador CaO.

    Requiere `agregar_ley_feo`, `agregar_tiempos`, `agregar_acumulados_previos`
    (cum_feed_Sn_kg_prev, cum_feed_CaO_kg_prev) y `agregar_features_carga_por_tolva`.

    1) Acumulados de carga hasta el escalon anterior (STATE, kg):
           cum_feed_total_kg_prev, cum_feed_<corriente>_kg_prev (conc, reciclo, pellets),
           cum_feed_carga_secundaria_kg_prev
       (cum_feed_Fe_kg_prev, cum_feed_dross_Fe_kg_prev, cum_feed_Sn_kg_prev,
       cum_feed_CaO_kg_prev, cum_feed_Carbon_kg_prev ya existen).

    2) Masa de escoria estimada por TRAZADOR CaO (metodo clasico de balance por
       componente inerte, Guia_Ausmelt_Sn.md 7.3): la cal (tolva 6) es el unico aporte
       de CaO cuantificado en el dataset y el CaO no se reduce ni se volatiliza, por
       lo que TODO el CaO alimentado hasta el cierre del escalon t debe estar en la
       escoria de ese momento:
           masa_escoria_est_kg[t] = sum_{k<=t} feed_CaO_kgh[k] / (ley_cao_escoria_pct[t]/100)
       De ahi el inventario de Sn y FeO en la escoria (kg):
           sn_inventario_escoria_est_kg  = masa_escoria_est_kg * ley_sn_escoria_pct/100
           feo_inventario_escoria_est_kg = masa_escoria_est_kg * ley_feo_escoria_pct/100
       Las versiones contemporaneas son LEAKAGE (usan las leyes del escalon t) y solo
       sirven para construir TARGETS; las versiones `_prev` (valor al cierre de t-1)
       son STATE validas para prescripcion.
       Supuestos declarados: (i) el CaO de la ganga del concentrado/reciclos no esta
       cuantificado -> la masa absoluta queda SUBESTIMADA en un factor ~constante por
       batch (no afecta las comparaciones relativas ni los deltas dentro del batch mas
       que por ese factor); (ii) el ruido del ensayo de %CaO (~2% relativo, estimado
       de diferencias entre escalones consecutivos de Reduccion sin carga) se traslada
       proporcionalmente a la masa; (iii) sin heel de escoria entre batches (verificado,
       ver cabecera del bloque v3).

    3) Avance de la reduccion (DERIVED-STATE, adimensional):
           avance_reduccion_sn_prev = 1 - sn_inventario_escoria_est_kg_prev / cum_feed_Sn_kg_prev
       fraccion del Sn ya cargado que YA salio de la fase escoria (a metal, o a polvo)
       al inicio del escalon t. Es la version en masa del "progreso" del batch.

    4) TARGETS en masa (nunca features de entrada):
           d_masa_escoria_est_kg         = masa_escoria_est_kg[t] - masa_escoria_est_kg[t-1]
           d_sn_inventario_escoria_est_kg = sn_inv[t] - sn_inv[t-1]
           sn_extraido_est_kg            = feed_Sn_kgf[t] - d_sn_inventario_escoria_est_kg
               Sn que NO quedo en la escoria durante el escalon (fue a metal crudo o a
               polvo). En Reduccion (feed ~0) coincide con -d_sn_inventario.
           frac_sn_extraido_escalon      = sn_extraido_est_kg / (sn_inv[t-1] + feed_Sn_kgf[t])
               fraccion del Sn DISPONIBLE (inventario previo + cargado) extraida de la
               escoria en el escalon: version en masa, valida en AMBAS fases, de la
               fraccion relativa T3 (que en Fusion no tenia sentido porque el Sn
               entrante inflaba el %Sn). Cinetica de primer orden -> comparable entre
               escalones con distinto nivel de partida.
           tasa_extraccion_sn_kg_min     = sn_extraido_est_kg / delta_tiempo
           feo_extraido_est_kg           = -(feo_inv[t] - feo_inv[t-1]) (solo interpretable
               en Reduccion: en Fusion entra FeO con la carga y no esta cuantificado)
           selectividad_masa_sn_feo      = sn_extraido_est_kg / feo_extraido_est_kg
               (solo si ambos > 0; version en masa de indice_selectividad_sn_feo).
    """
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    nuevas: dict[str, pd.Series] = {}
    por_batch = df.groupby("Batch")

    cum_total_incl = por_batch["feed_total_kgh"].cumsum()
    nuevas["cum_feed_total_kg_prev"] = cum_total_incl - df["feed_total_kgh"]
    for corriente in ["conc", "reciclo", "pellets"]:
        col_h = TOLVAS_CARGA[corriente][0]
        nuevas[f"cum_feed_{corriente}_kg_prev"] = por_batch[col_h].cumsum() - df[col_h]
    cum_sec_incl = por_batch["feed_carga_secundaria_kgh"].cumsum()
    nuevas["cum_feed_carga_secundaria_kg_prev"] = cum_sec_incl - df["feed_carga_secundaria_kgh"]

    cum_cao_incl = por_batch["feed_CaO_kgh"].cumsum()
    masa = cum_cao_incl / (df["ley_cao_escoria_pct"].where(df["ley_cao_escoria_pct"] > 0.5) / 100)
    sn_inv = masa * df["ley_sn_escoria_pct"] / 100
    feo_inv = masa * df["ley_feo_escoria_pct"] / 100
    nuevas["masa_escoria_est_kg"] = masa
    nuevas["sn_inventario_escoria_est_kg"] = sn_inv
    nuevas["feo_inventario_escoria_est_kg"] = feo_inv
    masa_prev = masa.groupby(df["Batch"]).shift(1)
    sn_inv_prev = sn_inv.groupby(df["Batch"]).shift(1)
    feo_inv_prev = feo_inv.groupby(df["Batch"]).shift(1)
    nuevas["masa_escoria_est_kg_prev"] = masa_prev
    nuevas["sn_inventario_escoria_est_kg_prev"] = sn_inv_prev
    nuevas["feo_inventario_escoria_est_kg_prev"] = feo_inv_prev
    nuevas["avance_reduccion_sn_prev"] = 1 - division_segura(sn_inv_prev, df["cum_feed_Sn_kg_prev"], umbral=UMBRAL_SN_FED_KG)

    d_masa = masa - masa_prev
    d_sn_inv = sn_inv - sn_inv_prev
    sn_extraido = df["feed_Sn_kgf"] - d_sn_inv
    disponible = sn_inv_prev + df["feed_Sn_kgf"]
    nuevas["d_masa_escoria_est_kg"] = d_masa
    nuevas["d_sn_inventario_escoria_est_kg"] = d_sn_inv
    nuevas["sn_extraido_est_kg"] = sn_extraido
    nuevas["frac_sn_extraido_escalon"] = division_segura(sn_extraido, disponible, umbral=UMBRAL_SN_FED_KG)
    nuevas["tasa_extraccion_sn_kg_min"] = sn_extraido / df["delta_tiempo"]
    feo_extraido = -(feo_inv - feo_inv_prev)
    nuevas["feo_extraido_est_kg"] = feo_extraido
    ambos_positivos = sn_extraido.gt(0) & feo_extraido.gt(0)
    nuevas["selectividad_masa_sn_feo"] = division_segura(
        sn_extraido.where(ambos_positivos), feo_extraido.where(ambos_positivos), umbral=1.0
    )
    return pd.concat([df, pd.DataFrame(nuevas, index=df.index)], axis=1)


def agregar_indice_irf(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `indice_irf` (contemporaneo, LEAKAGE) e `indice_irf_prev` (STATE).

        indice_irf = %FeO / (%SiO2 + %Al2O3 + %CaO)

    Indice de la propia planta (columna "IRF" del Excel, que aqui se recalcula a partir
    de las leyes base en vez de importarse): razon entre el principal modificador de
    red en escorias fayaliticas (FeO) y el conjunto de formadores/otros modificadores.
    Complementa ratio_feo_sio2_prev (ratio fayalitico) y basicidad_B2/B4.
    """
    df = df.copy()
    df["indice_irf"] = division_segura(
        df["ley_feo_escoria_pct"],
        df["ley_sio2_escoria_pct"] + df["ley_al2o3_escoria_pct"] + df["ley_cao_escoria_pct"],
    )
    df["indice_irf_prev"] = df.groupby("Batch")["indice_irf"].shift(1)
    return df


def construir_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo v3: features base + modelado + carga por tolva + inventario/masa.

    Equivale a `construir_features` -> `construir_features_modelado` ->
    `agregar_features_carga_por_tolva` -> `agregar_inventario_masa_escoria` ->
    `agregar_indice_irf` -> targets escalares ya existentes (T3, k_app, selectividad
    en leyes). Requiere el DataFrame de `dataset_lingo_smelter.construir_dataset`
    (version con columnas por tolva, 2026-09-13 noche) ya limpiado.
    """
    df = construir_features(df)
    df = construir_features_modelado(df)
    df = agregar_features_carga_por_tolva(df)
    df = agregar_inventario_masa_escoria(df)
    df = agregar_indice_irf(df)
    df = agregar_target_tasa_relativa_sn(df)
    df = agregar_k_app_sn(df)
    df = agregar_indice_selectividad_sn_feo(df)
    return df


# =============================================================================
# Clasificacion STATE / ACTION / CONTEXT / DERIVED / TARGET / LEAKAGE
# =============================================================================
#
# Registro de a que categoria pertenece cada columna que este modulo puede
# producir (mas las primitivas relevantes de dataset_lingo_smelter.py), pensado
# para que un modelo de escalon sea utilizable de forma PRESCRIPTIVA (no solo
# descriptiva): una feature con buen poder predictivo pero clasificada como
# LEAKAGE no esta disponible al momento de decidir la accion del escalon t, sin
# importar que tan alto sea su SHAP en un modelo puramente descriptivo.
#
# Convencion (igual a la usada en analisis_lingo_smelter.ipynb, Paso 31):
#   - CONTEXT: identificadores, fase, tiempo y posicion -- no son estado fisico
#     ni accion manipulable.
#   - STATE: ensayo/inventario conocido AL CIERRE del escalon t-1 (toda
#     columna `_prev`/`cum_..._prev`), disponible ANTES de decidir la accion
#     del escalon t.
#   - ACTION_primitiva: lo que el operador fija/decide para el escalon t
#     (alimentacion, gases de lanza, posicion, duracion).
#   - DERIVED-ACTION: funcion solo de columnas ACTION del propio escalon t
#     (ej. una tasa = masa/tiempo) -- tan segura como sus insumos.
#   - DERIVED-STATE: funcion solo de columnas STATE (`_prev`) -- tan segura
#     como sus insumos.
#   - "DERIVED (A_t x S_prev)": accion del escalon t aplicada sobre un estado
#     de partida ya conocido -- valida para prescripcion, es exactamente lo
#     que necesita un modelo de transicion estado+accion->resultado.
#   - LEAKAGE: medicion hecha AL CIERRE del escalon t (contemporanea al
#     target) -- nunca disponible al momento de decidir A_t; existe una
#     version seguras "_prev" para la mayoria (usar esa en su lugar).
#   - TARGET: candidato de variable objetivo -- nunca debe usarse como feature
#     de entrada de un modelo que prediga otro target (o si mismo).
#
# Nota importante sobre `delta_tiempo`: se clasifica aqui como ACTION_primitiva
# (el operador decide cuanto sostener una combinacion de alimentacion/lanza
# antes de re-muestrear escoria), pero el caso es ambiguo si en la operacion
# real el escalon termina por una "parada endogena" (p.ej. cuando el ensayo de
# laboratorio da el resultado esperado) en vez de por una decision previa --
# ver analisis_lingo_smelter.ipynb Paso 31 y hallazgos.md seccion 11 para la
# discusion completa. Tratar su uso prescriptivo con la misma cautela que
# `posicion_vertical_lanza_mm` hasta confirmar el protocolo real de cierre de
# escalon con planta.


def _construir_clasificacion_features() -> dict[str, str]:
    """Construye `CLASIFICACION_FEATURES` a partir de las listas ya definidas en
    este modulo (VARIABLES_FEED_VOLUMEN_A_TASA, VARIABLES_ESTADO_PARA_GRADIENTE,
    VARIABLES_LAG_ESTADO, VARIABLES_CUM_A_PREV), para que el registro se derive
    automaticamente de la definicion de cada familia de features en vez de
    mantenerse duplicado a mano (y quedar desincronizado si esas listas
    cambian).
    """
    clasificacion: dict[str, str] = {}

    for col in [
        "Batch", "fase_proceso", "fecha_inicio", "fecha_final",
        "orden_escalon_fase", "es_primer_escalon_fase", "es_primer_escalon_batch",
        "es_transicion_FR", "tiempo", "tiempo_fase", "es_fusion",
    ]:
        clasificacion[col] = "CONTEXT"

    for col in [
        "feed_Sn_kgf", "feed_Fe_kgh", "feed_dross_Fe_kgh", "feed_CaO_kgh",
        "feed_Carbon_kgh", "feed_total_kgh",
        "volumen_gas_natural_inyectado_lanza_escalon_nm3",
        "volumen_o2_inyectado_lanza_escalon_nm3",
        "volumen_aire_inyectado_lanza_escalon_nm3",
        "posicion_vertical_lanza_mm", "delta_tiempo",
        "presion_suministro_gn_lanza_kpa", "presion_suministro_o2_lanza_kpa",
    ]:
        clasificacion[col] = "ACTION_primitiva"

    for col in VARIABLES_FEED_VOLUMEN_A_TASA.values():
        clasificacion[col] = "DERIVED-ACTION"
    for col in [
        "exceso_o2_combustion_pct", "oxygen_enrichment_pct", "indice_severidad_reductora",
        "total_process_gas_nm3_min", "lanza_pos_x_gas_total", "interaccion_lanza_x_gn",
        "margen_presion_gn", "o2_teorico_combustion_nm3", "o2_disponible_nm3",
        "relacion_C_carga", "relacion_CaO_carga_Fe", "exceso_C_estequiometrico_pct",
        "oxidante_sobre_carga_total", "carga_termica_relativa",
        "interaccion_C_x_exceso_o2",
    ]:
        clasificacion[col] = "DERIVED-ACTION"

    for col in VARIABLES_LAG_ESTADO:
        clasificacion[f"{col}_prev"] = "STATE"
    for col in VARIABLES_CUM_A_PREV.values():
        clasificacion[f"{col}_prev"] = "STATE"
    clasificacion["relacion_C_cum_Sn_cum_prev"] = "STATE"

    for col in [
        "ratio_sn_feo_prev", "ratio_feo_sio2_prev", "log_ratio_sn_feo_prev",
        "ratio_sn_sio2_prev", "log_ratio_sn_sio2_prev",
        "interaccion_Sn_x_FeO_prev", "interaccion_FeO_x_B2_prev", "interaccion_B2_x_temp_prev",
        "grad_temperatura_horno_celsius_prev",
    ]:
        clasificacion[col] = "DERIVED-STATE"

    for col in [
        "interaccion_C_x_Sn_prev", "interaccion_C_x_FeO_prev", "interaccion_C_x_temp_prev",
        "carbon_sobre_inventario_sn_prev",
    ]:
        clasificacion[col] = "DERIVED (A_t x S_prev)"

    # VARIABLES_ESTADO_PARA_GRADIENTE mezcla, por diseno, dos naturalezas distintas
    # (ver el comentario junto a su definicion): la mayoria son STATE genuino (leyes,
    # temperatura, tiro, presion de punta, contrapresiones -mediciones/consecuencias al
    # cierre del escalon, nunca fijadas directamente por el operador), pero 3 son en
    # realidad ACTION_primitiva (posicion de lanza y las 2 presiones de SUMINISTRO, que
    # son basicamente el setpoint que el operador fija -- Paso 31 del notebook las separa
    # explicitamente de STATE por el mismo motivo). Para esas 3, el valor contemporaneo NO
    # es fuga (ya se clasifico arriba como ACTION_primitiva) y su d_/grad_ es una
    # DERIVED-ACTION valida (cuanto cambio esa accion respecto al escalon anterior), no un
    # candidato de TARGET.
    variables_accion_con_gradiente = {
        "posicion_vertical_lanza_mm",
        "presion_suministro_gn_lanza_kpa",
        "presion_suministro_o2_lanza_kpa",
    }
    for col in VARIABLES_ESTADO_PARA_GRADIENTE:
        if col in variables_accion_con_gradiente:
            clasificacion[f"d_{col}"] = "DERIVED-ACTION"
            clasificacion[f"grad_{col}"] = "DERIVED-ACTION"
            continue
        clasificacion[col] = "LEAKAGE (usar version _prev)"
        clasificacion[f"d_{col}"] = "TARGET"
        clasificacion[f"grad_{col}"] = "TARGET"
    for col in ["basicidad_B2", "basicidad_B4", "indice_estructural_molar", "ratio_sn_sio2"]:
        clasificacion[col] = "LEAKAGE (usar version _prev)"
    # delta_T_gas_pre usa temperatura_gas_pre_cuchilla_celsius contemporanea, que no tiene
    # version _prev en VARIABLES_LAG_ESTADO -> tambien es fuga si se usa tal cual.
    clasificacion["delta_T_gas_pre"] = "LEAKAGE (usa temperatura de gas contemporanea)"

    for col in [
        "tasa_relativa_sn_escoria", "d_ratio_sn_sio2", "d_log_ratio_sn_sio2",
        "indice_selectividad_sn_feo", "k_app_sn",
        "rendimiento_proxy_batch", "recuperacion_real_batch_pct",
        # sn_en_*_batch_t: componentes crudos de rendimiento_proxy_batch (unicos por
        # batch). Auxiliar-TARGET valido (Guia_Ausmelt_Sn.md 8.4: "prediccion separada de
        # M,D,P"), pero NUNCA feature de entrada de un modelo de rendimiento_proxy_batch
        # (el modelo reconstruiria trivialmente su propia formula).
        "sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t",
    ]:
        clasificacion[col] = "TARGET"

    # delta_FR_*: features a nivel de BATCH producidas por construir_features_transicion_fase
    # (salto Fusion->Reduccion). Validas como feature de un modelo de BATCH o de escalones de
    # Reduccion desde R1 en adelante; NUNCA dentro de un modelo de escalon de Fusion (ver el
    # docstring de esa funcion para el detalle de por que).
    for col in [
        "ley_sn_escoria_pct", "ley_feo_escoria_pct", "temperatura_horno_celsius",
        "presion_punta_lanza_kpa", "exceso_o2_combustion_pct",
    ]:
        clasificacion[f"delta_FR_{col}"] = "DERIVED-STATE (solo batch o Reduccion desde R1)"

    # Agregados EDA retrospectivos de construir_resumen_batch_eda/construir_resumen_batch_masa:
    # se calculan sobre el batch YA CERRADO completo (ej. temperatura_media_horno promedia
    # escalones que, para cualquier decision intermedia real, todavia estaban en el futuro).
    # Documentados en esas funciones como exploratorios y debilmente correlacionados con el
    # target (hallazgos.md secciones 4 y 6) -- utiles para EDA/explicacion retrospectiva,
    # nunca como feature de un modelo que recomienda una accion dentro del batch.
    for col in [
        "carbon_total_fusion_kg", "carbon_total_reduccion_kg",
        "exceso_o2_medio_fusion_pct", "exceso_o2_medio_reduccion_pct",
        "basicidad_media_fusion", "ley_sn_escoria_final_pct", "ley_sn_escoria_max_pct",
        "caida_ley_sn_reduccion_pct", "temperatura_media_horno",
        "relacion_C_reduccion_vs_fusion", "feed_sn_total_kg", "feed_sn_total_t",
    ]:
        clasificacion[col] = "LEAKAGE (agregado retrospectivo de batch completo)"

    # ------------------------------------------------------------------ v3 (carga por
    # tolva, plan de duracion, refractario, inventario por trazador CaO, targets en masa)
    for col in ["escalon_idx", "espesor_ladrillo_norm_mm", "ley_sn_carga_batch_pct"]:
        clasificacion[col] = "CONTEXT"
    for corriente in TOLVAS_CARGA:
        clasificacion[f"ley_sn_{corriente}_batch_pct"] = "CONTEXT"      # ensayo del batch, conocido al inicio
    for col in [
        "duracion_plan_min", "feed_conc_t1_kgh", "feed_reciclo_t3_kgh", "feed_pellets_t7_kgh",
        "feed_Carbon_fusion_kgh", "feed_Carbon_reduccion_kgh",
    ]:
        clasificacion[col] = "ACTION_primitiva"
    # Sn fino por tolva = tmh (accion) x ley del batch (contexto): derivada de accion.
    for corriente, (_, col_f, col_tasa) in TOLVAS_CARGA.items():
        clasificacion[col_f] = "DERIVED-ACTION"
        clasificacion[col_tasa] = "DERIVED-ACTION"
        clasificacion[f"frac_{corriente}_carga"] = "DERIVED-ACTION"
        clasificacion[f"cum_feed_{corriente}_kg_prev"] = "STATE"
    for col in [
        "feed_carga_secundaria_kgh", "frac_carga_secundaria", "ley_sn_carga_pct",
        "relacion_CaO_carga", "relacion_C_Sn_carga",
        "tasa_feed_Carbon_fusion_kg_min", "tasa_feed_Carbon_reduccion_kg_min",
        # lecturas de planta equivalentes (r=1.000) a exceso_o2_combustion_pct+100 y oxygen_enrichment_pct
        "estequiometria_lanza_planta_pct", "enriquecimiento_o2_lanza_planta_pct",
    ]:
        clasificacion[col] = "DERIVED-ACTION"
    for col in [
        "cum_feed_total_kg_prev", "cum_feed_carga_secundaria_kg_prev",
        "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev", "feo_inventario_escoria_est_kg_prev",
        "termocupla_media_celsius_prev", "indice_irf_prev",
    ]:
        clasificacion[col] = "STATE"
    clasificacion["avance_reduccion_sn_prev"] = "DERIVED-STATE"
    for col in COLUMNAS_TERMOCUPLAS_CARCASA + [
        "termocupla_media_celsius", "masa_escoria_est_kg", "sn_inventario_escoria_est_kg",
        "feo_inventario_escoria_est_kg", "indice_irf",
    ]:
        clasificacion[col] = "LEAKAGE (usar version _prev)"
    clasificacion["desvio_duracion_min"] = "LEAKAGE (solo se conoce al cierre del escalon)"
    # "CaO total (kg)" del Excel = tolva 6 + tolva 3: no es fundente puro, se conserva solo
    # para trazabilidad de resultados anteriores a 2026-09-13 (noche).
    clasificacion["feed_CaO_total_excel_kg"] = "DEPRECATED (tolva 6 + tolva 3; usar feed_CaO_kgh)"
    for col in [
        "d_masa_escoria_est_kg", "d_sn_inventario_escoria_est_kg", "sn_extraido_est_kg",
        "frac_sn_extraido_escalon", "tasa_extraccion_sn_kg_min", "feo_extraido_est_kg",
        "selectividad_masa_sn_feo", "rendimiento_ha_batch", "sn_crudo_generado_batch_t",
    ]:
        clasificacion[col] = "TARGET"

    return clasificacion


CLASIFICACION_FEATURES: dict[str, str] = _construir_clasificacion_features()


def es_feature_segura_para_prescripcion(nombre: str) -> bool:
    """True si `nombre` es segura como feature de entrada de un modelo prescriptivo.

    "Segura" = clasificada como CONTEXT, STATE, ACTION_primitiva o cualquier
    DERIVED (ver `CLASIFICACION_FEATURES`); False si es LEAKAGE, TARGET, o si
    el nombre no esta en el registro (columna no reconocida: por seguridad se
    asume que NO es segura en vez de asumir lo contrario).
    """
    rol = CLASIFICACION_FEATURES.get(nombre)
    if rol is None:
        return False
    return not (rol.startswith("LEAKAGE") or rol == "TARGET")


def filtrar_features_seguras(nombres: list[str]) -> list[str]:
    """Filtra `nombres` dejando solo las reconocidas como seguras para prescripcion.

    Ver `es_feature_segura_para_prescripcion`. Preserva el orden de entrada; no
    reordena ni deduplica.
    """
    return [n for n in nombres if es_feature_segura_para_prescripcion(n)]
