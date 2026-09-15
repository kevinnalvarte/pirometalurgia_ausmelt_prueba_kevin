"""balances -- bloque v6 de features/targets inspirado en el modelo teorico de los metalurgistas
(`func-teorethical-model`, balance de masa y energia por batch en Rust) llevado al nivel de ESCALON.

Uso:
    import bal_features as bal
    df = bal.cargar_df()                       # df_v3 cacheado + columnas b6_* por escalon
    batch = bal.construir_batch_balance(df)    # una fila por batch: balance global de masa + KPIs
    bal.CLASIFICACION_V6                       # rol de cada columna b6_* (anti-fuga)
    bal.SUPUESTOS                              # parametros fisicos con su fuente (sensibilidad)

Todas las columnas nuevas llevan prefijo `b6_`. Cuatro familias:

  A) CIERRE MULTI-TRAZADOR (Reduccion). En Reduccion no entra carga (feed_total ~ 0), asi que los
     oxidos inertes de la escoria (CaO, SiO2, Al2O3, MgO) conservan su MASA; solo salen SnO (reducido a
     Sn) y FeO (reducido a Fe). Entre dos escalones consecutivos, para cada inerte X:
           m[t]/m[t-1] = %X[t-1] / %X[t]
     Con 4 inertes se obtiene un estimador del cociente de masas con menos ruido que el trazador CaO
     solo (media ponderada de ln-ratios, pesos = 1/sd^2 relativo del ensayo). La masa absoluta se ancla
     en el ultimo escalon de Fusion (trazador CaO, como v3): la escala sigue siendo la de v3, pero los
     DELTAS dentro de Reduccion usan los 4 trazadores. Es la logica de "Dump 1 / Dump 2" del modelo Rust
     (SiO2, Al2O3, CaO, MgO no reaccionan) aplicada como estimador.
       b6_ln_ratio_masa_multi   ln(m[t]/m[t-1]) multi-inerte           (DIAG / componente de targets)
       b6_dispersion_trazadores sd ponderada de los ln-ratios entre inertes (ruido de ensayo / inerte sesgado)
       b6_masa_escoria_multi_kg, b6_sn_inv_multi_kg, b6_feo_inv_multi_kg  (LEAKAGE; `_prev` = STATE)
       b6_sn_extraido_multi_kg, b6_feo_extraido_multi_kg, b6_sdi_multi    (TARGET)
       b6_residuo_masa_kg  = d_masa_multi + 1.135*Sn_ext + FeO_ext  (masa que salio sin explicarse por
                             reduccion: splash, arrastre de finos, ruido) (DIAG)

  B) BALANCE REDUCTOR / OXIDANTE (kg de carbono equivalente, a tiempo de decision). Traduce a un solo
     numero la logica del modelo Rust "carbon libre = carbon − reducciones − carbon quemado por O2 libre":
       oferta reductora  = C fijo del carbon
                          + C-eq del gas de lanza (CO por deficit de O2; negativo si hay exceso de O2,
                            que quema carbon a CO: C + 1/2 O2 -> CO, 2 mol C por mol O2)
                          + C-eq del Fe metalico del dross de Fe (Fe + SnO -> FeO + Sn: 1 mol C por mol Fe)
       demanda reductora = 2 C por Sn disponible (SnO2 + 2C -> Sn + 2CO) + 1 C por Fe2O3 del mineral (t4)
       b6_reductor_neto_kg_ceq = oferta − demanda   (>0: sobra reductor para FeO -> Fe, sobre-reduccion)
       b6_ratio_reductor_demanda = oferta / demanda
       b6_capacidad_sobre_reduccion_feo_kg = max(neto,0) * M_FeO/M_C, acotada por el FeO en inventario
       b6_cum_reductor_neto_kg_prev = suma del neto hasta t-1 (banco de reductor no consumido) (STATE)
     Diagnosticos post-hoc: b6_c_consumido_teorico_kg (0.2024 Sn_ext + 0.167 FeO_ext) y
     b6_utilizacion_reductor (consumido / oferta) (LEAKAGE).

  C) BALANCE DE ENERGIA POR ESCALON (MJ). Version por escalon del balance del horno del modelo Rust:
       q_combustion   = GN * (802.3 − 283.0*x)/22.414 kJ/mol, x = fraccion del C del gas que solo llega
                        a CO cuando el O2 disponible es sub-estequiometrico (x = 4 − 2*O2disp/GN, en [0,1])
       q_carbon_lanza = calor de C + 1/2 O2 -> CO con el O2 en exceso (acotado por el carbon alimentado)
       q_carga        = calentar y fundir la carga (H_CARGA_MJ_KG * feed_total + H_CARBON_MJ_KG * carbon)
       q_gases        = sensible de los gases de salida a T_prev (N2 del aire, CO2/H2O del gas, O2 sobrante,
                        CO del carbon)
       q_perdidas     = PERDIDAS_MJ_MIN[fase] * dt (paredes + agua de paneles del modelo Rust)
       b6_q_neto_disponible_MJ = q_combustion + q_carbon_lanza − q_carga − q_gases − q_perdidas
       b6_dT_teorico_sin_reaccion = q_neto / (masa_prev * CP_ESCORIA) [K]: subida de T si no hubiera
                                    reacciones endotermicas (feature del modelo de dT)
     Post-hoc: b6_q_reaccion_aparente_MJ = q_neto − masa_prev*CP*dT_medido (calor que absorbieron las
     reacciones) y b6_q_reaccion_trazador_MJ = 3.004*Sn_ext + 2.248*FeO_ext + 1.519*Fe(t4) (mismo calor
     segun el trazador). Si el balance de energia "ve" la extraccion, ambos correlacionan.

  D) BALANCE GLOBAL DE MASA POR BATCH (`construir_batch_balance`): estructura del modelo Rust con los
     datos del Excel: carga por tolva (F+R) − humedad − C del carbon (a gas) − Sn de salida como SnO2
     (metal+dross+polvo) − FeO del Fe metalizado + O ganado por el Fe metalico del dross + ceniza =
     masa final de escoria. Se compara con el trazador (factor de escala), se calcula el Sn en la escoria
     final y el KPI refinado M/(M+D+P+Sn_escoria). Supuestos (leyes de metal, dross, polvo, humedad) en
     SUPUESTOS; cada uno es un parametro para sensibilidad.

Convenciones anti-fuga: `_prev`/`cum_*_prev` = STATE; funciones de acciones del escalon t = DERIVED-ACTION;
accion x estado previo = DERIVED (A_t x S_prev); todo lo que use leyes/temperatura del cierre de t = LEAKAGE
o TARGET; DIAG = diagnostico post-hoc (nunca feature).
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

# =============================================================================
# Constantes fisicas (IUPAC / NIST-JANAF, 25 C) y supuestos de planta (parametrizables)
# =============================================================================
M_SN, M_FE, M_C, M_O = 118.71, 55.845, 12.011, 15.999
M_FEO, M_FE2O3, M_SNO2, M_SNO = 71.844, 159.69, 150.71, 134.71
M_CAO, M_SIO2, M_AL2O3, M_MGO = 56.077, 60.084, 101.961, 40.304
VM_NM3_KMOL = 22.414
C_POR_SN = 2 * M_C / M_SN            # 0.2024 kg C / kg Sn   (SnO2 + 2C -> Sn + 2CO)
C_POR_FEO = M_C / M_FEO              # 0.1672 kg C / kg FeO  (FeO + C -> Fe + CO)
C_POR_FE_HEMATITA = M_C / (2 * M_FE) # 0.1075 kg C / kg Fe   (Fe2O3 + C -> 2FeO + CO)
CEQ_POR_FE_METAL = M_C / M_FE        # 0.2151 kg C-eq / kg Fe (Fe + SnO -> FeO + Sn, 1 Fe ~ 1 C)
CEQ_POR_NM3_O2 = 2 * M_C / VM_NM3_KMOL  # 1.0717 kg C-eq por Nm3 de O2 (2 mol C por mol O2, base CO)

# Entalpias de reaccion (kJ/mol, 25 C; NIST-JANAF: SnO2 -577.6, SnO -280.7, FeO -272.0, Fe2O3 -824.2,
# CO -110.5, CO2 -393.5, H2O(g) -241.8, CH4 -74.9)
DH_CH4_CO2 = -802.3     # CH4 + 2 O2 -> CO2 + 2 H2O(g)
DH_CH4_CO = -519.3      # CH4 + 1.5 O2 -> CO + 2 H2O(g)
DH_C_CO = -110.5        # C + 1/2 O2 -> CO
DH_SNO2_C = +356.6      # SnO2 + 2C -> Sn + 2CO  (por mol Sn)
DH_FEO_C = +161.5       # FeO + C -> Fe + CO
DH_FE2O3_C = +169.7     # Fe2O3 + C -> 2FeO + CO (por mol Fe2O3)
Q_SN_MJ_KG = DH_SNO2_C / M_SN            # 3.004 MJ por kg de Sn reducido (con C)
Q_FEO_MJ_KG = DH_FEO_C / M_FEO           # 2.248 MJ por kg de FeO reducido
Q_FE_HEMATITA_MJ_KG = DH_FE2O3_C / (2 * M_FE)  # 1.519 MJ por kg de Fe del mineral (Fe2O3 -> FeO)

SUPUESTOS: dict[str, dict] = {
    # --- combustion y reductores
    "PCI_GN_MJ_NM3": dict(valor=35.8, fuente="LHV metano; gas de Camisea 36-37; el modelo Rust lo lee de combustible_parametros"),
    "CARBONO_FIJO": dict(valor=0.75, fuente="fraccion de C fijo del carbon (carbon_alimento del modelo Rust no esta en el repo); rango 0.6-0.85"),
    "FE_MINERAL": dict(valor=0.55, fuente="kg Fe / kg de mineral WF (tolva 4) como hematita; rango 0.45-0.65"),
    "FE_MET_DROSS": dict(valor=0.30, fuente="kg Fe metalico / kg dross de Fe (tolva 5); modelo Rust: dross 65-71.5 % Sn, resto Fe+otros"),
    "PUREZA_O2": dict(valor=0.98, fuente="modelo Rust: pureza_oxigeno = 98"),
    # --- energia
    "H_CARGA_MJ_KG": dict(valor=1.7, fuente="sensible+fusion de solidos oxidicos 25->1100 C (~1.2-1.5) + 6 % humedad evaporada; rango 1.3-2.2"),
    "H_CARBON_MJ_KG": dict(valor=1.5, fuente="sensible del carbon 25->1100 C"),
    "CP_GAS_KJ_NM3_K": dict(valor=1.45, fuente="media N2/CO2/H2O/CO a ~1000 C (1.35-2.1)"),
    "CP_ESCORIA_KJ_KG_K": dict(valor=1.2, fuente="escoria fayalitica liquida 1.1-1.3"),
    "T_REF_C": dict(valor=25.0, fuente="referencia de entalpias"),
    "PERDIDAS_MJ_MIN": dict(valor={"Fusión": 117.0, "Reducción": 158.0},
                           fuente="modelo Rust JSON_DATA: perdidas_fusion 248.6 Mcal/h + agua 358 m3/h x 4 K; reduccion 468.9 Mcal/h + agua x 5 K"),
    # --- balance global por batch
    "LEY_SN_METAL": dict(valor=0.96, fuente="metal crudo Ausmelt ~95-98 % Sn"),
    "LEY_SN_DROSS": dict(valor=0.65, fuente="modelo Rust factor_dross_fusion 65 %"),
    "LEY_SN_POLVO": dict(valor=0.55, fuente="humos de Sn (SnO2 + FeO + otros) ~50-60 % Sn"),
    "HUMEDAD_CARGA": dict(valor=0.06, fuente="tmh = masa humeda; humedad tipica de concentrado/reciclos 4-8 %"),
    "CENIZA_CARBON": dict(valor=0.10, fuente="ceniza del carbon (a escoria)"),
    "UMBRAL_SIN_CARGA_KG": dict(valor=200.0, fuente="feed_total por debajo del cual el escalon se considera sin carga (cierre multi-inerte valido)"),
}
# precision relativa del ensayo por inerte (sd relativa): pesos 1/sd^2 para el cierre multi-inerte
SD_REL_ENSAYO = {"cao": 0.02, "sio2": 0.015, "al2o3": 0.03, "mgo": 0.05}
COL_INERTE = {"cao": "ley_cao_escoria_pct", "sio2": "ley_sio2_escoria_pct",
              "al2o3": "ley_al2o3_escoria_pct", "mgo": "ley_mgo_escoria_pct"}


def S(nombre: str):
    return SUPUESTOS[nombre]["valor"]


# =============================================================================
# A) Cierre multi-trazador (Reduccion)
# =============================================================================

def agregar_multitrazador(df: pd.DataFrame, inertes: tuple[str, ...] = ("cao", "sio2", "al2o3", "mgo"),
                          w_sdi: float = 10.0) -> pd.DataFrame:
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    g = df.groupby("Batch")
    n = {}
    sin_carga = df["feed_total_kgh"].fillna(0) < S("UMBRAL_SIN_CARGA_KG")
    es_red = df["fase_proceso"] == "Reducción"
    valido = sin_carga & es_red & ~df["es_primer_escalon_fase"].astype(bool)

    lnr = {}
    for k in inertes:
        x = df[COL_INERTE[k]].where(df[COL_INERTE[k]] > 0.3)
        xp = g[COL_INERTE[k]].shift(1).where(lambda s: s > 0.3)
        lnr[k] = np.log(xp / x)                    # ln(m[t]/m[t-1]) segun el inerte k
    L = pd.DataFrame(lnr)
    w = pd.Series({k: 1 / SD_REL_ENSAYO[k] ** 2 for k in inertes})
    W = L.notna() * w
    ln_multi = (L.fillna(0) * W).sum(axis=1) / W.sum(axis=1).replace(0, np.nan)
    disp = np.sqrt(((L.sub(ln_multi, axis=0)) ** 2 * W).sum(axis=1) / W.sum(axis=1).replace(0, np.nan))
    n["b6_ln_ratio_masa_multi"] = ln_multi.where(valido)
    n["b6_ln_ratio_masa_cao"] = (L["cao"] if "cao" in L.columns else pd.Series(np.nan, index=df.index)).where(valido)
    n["b6_dispersion_trazadores"] = disp.where(valido)
    n["b6_n_inertes_validos"] = L.notna().sum(axis=1).where(valido)

    # masa multi: ancla = masa trazador CaO del escalon anterior al primer escalon valido de Reduccion;
    # luego encadena exp(ln_multi). Donde no es valido (Fusion, R0 con carga) usa la masa v3.
    masa_v3 = df["masa_escoria_est_kg"]
    masa_multi = masa_v3.copy()
    for _, idx in g.indices.items():
        prev = np.nan
        for i in idx:
            if bool(valido.iat[i]) and np.isfinite(ln_multi.iat[i]) and np.isfinite(prev):
                masa_multi.iat[i] = prev * np.exp(ln_multi.iat[i])
            else:
                masa_multi.iat[i] = masa_v3.iat[i]
            prev = masa_multi.iat[i] if np.isfinite(masa_multi.iat[i]) else prev
    sn_inv = masa_multi * df["ley_sn_escoria_pct"] / 100
    feo_inv = masa_multi * df["ley_feo_escoria_pct"] / 100
    n["b6_masa_escoria_multi_kg"] = masa_multi
    n["b6_sn_inv_multi_kg"] = sn_inv
    n["b6_feo_inv_multi_kg"] = feo_inv
    masa_prev = masa_multi.groupby(df["Batch"]).shift(1)
    sn_prev = sn_inv.groupby(df["Batch"]).shift(1)
    feo_prev = feo_inv.groupby(df["Batch"]).shift(1)
    n["b6_masa_escoria_multi_kg_prev"] = masa_prev
    n["b6_sn_inv_multi_kg_prev"] = sn_prev
    n["b6_feo_inv_multi_kg_prev"] = feo_prev
    sn_ext = df["feed_Sn_kgf"].fillna(0) - (sn_inv - sn_prev)
    feo_ext = -(feo_inv - feo_prev)
    n["b6_sn_extraido_multi_kg"] = sn_ext.where(valido)
    n["b6_feo_extraido_multi_kg"] = feo_ext.where(valido)
    ok = sn_prev.gt(20) & sn_inv.gt(20) & feo_prev.gt(100) & feo_inv.gt(100) & valido
    n["b6_ln_sn_dep_multi"] = np.log(sn_prev / sn_inv).where(ok)
    n["b6_ln_feo_ret_multi"] = np.log(feo_inv / feo_prev).where(ok)
    n["b6_sdi_multi"] = (n["b6_ln_sn_dep_multi"] + w_sdi * n["b6_ln_feo_ret_multi"])
    # residuo de masa: la escoria pierde SnO (1.135 kg por kg Sn) y FeO (1 kg por kg) al reducir; lo demas es
    # masa que salio (splash / arrastre) o ruido
    d_masa = masa_multi - masa_prev
    n["b6_residuo_masa_kg"] = (d_masa + sn_ext * M_SNO / M_SN + feo_ext.clip(lower=0)).where(valido)
    return pd.concat([df, pd.DataFrame(n, index=df.index)], axis=1)


# =============================================================================
# B) Balance reductor / oxidante (kg C-eq)
# =============================================================================

def agregar_balance_reductor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = {}
    gn = df["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0)
    o2 = df["volumen_o2_inyectado_lanza_escalon_nm3"].fillna(0) * S("PUREZA_O2")
    aire = df["volumen_aire_inyectado_lanza_escalon_nm3"].fillna(0)
    o2_disp = o2 + 0.21 * aire
    o2_esteq = 2.0 * gn
    exceso_nm3 = o2_disp - o2_esteq                     # >0 oxidante, <0 CO reductor
    n["b6_o2_exceso_nm3"] = exceso_nm3
    n["b6_ceq_lanza_kg"] = -exceso_nm3 * CEQ_POR_NM3_O2   # C-eq aportado (+) o consumido (-) por la lanza
    c_coal = df["feed_Carbon_kgh"].fillna(0) * S("CARBONO_FIJO")
    r_fe = df["feed_dross_Fe_kgh"].fillna(0) * S("FE_MET_DROSS") * CEQ_POR_FE_METAL
    n["b6_c_fijo_kg"] = c_coal
    n["b6_ceq_fe_dross_kg"] = r_fe
    oferta = c_coal + r_fe + n["b6_ceq_lanza_kg"]
    sn_disp = df["sn_inventario_escoria_est_kg_prev"].fillna(0) + df["feed_Sn_kgf"].fillna(0)
    d_sn = C_POR_SN * sn_disp
    d_fe = df["feed_Fe_kgh"].fillna(0) * S("FE_MINERAL") * C_POR_FE_HEMATITA
    n["b6_demanda_sn_kg_ceq"] = d_sn
    n["b6_demanda_fe2o3_kg_ceq"] = d_fe
    n["b6_oferta_reductora_kg_ceq"] = oferta
    n["b6_reductor_neto_kg_ceq"] = oferta - d_sn - d_fe
    n["b6_ratio_reductor_demanda"] = oferta / (d_sn + d_fe).where((d_sn + d_fe) > 5)
    cap = (oferta - d_sn - d_fe).clip(lower=0) / C_POR_FEO
    n["b6_capacidad_sobre_reduccion_feo_kg"] = np.minimum(cap, df["feo_inventario_escoria_est_kg_prev"].fillna(np.inf))
    n["b6_reductor_neto_por_min"] = n["b6_reductor_neto_kg_ceq"] / df["delta_tiempo"]
    neto = pd.Series(n["b6_reductor_neto_kg_ceq"], index=df.index)
    cum = neto.groupby(df["Batch"]).cumsum()
    n["b6_cum_reductor_neto_kg_prev"] = cum - neto
    cum_of = pd.Series(oferta, index=df.index).groupby(df["Batch"]).cumsum()
    n["b6_cum_oferta_reductora_kg_prev"] = cum_of - oferta
    # post-hoc
    sn_ext = df["sn_extraido_est_kg"]
    feo_ext = df["feo_extraido_est_kg"].clip(lower=0)
    cons = C_POR_SN * sn_ext.clip(lower=0) + C_POR_FEO * feo_ext.fillna(0) + d_fe
    n["b6_c_consumido_teorico_kg"] = cons
    n["b6_utilizacion_reductor"] = cons / oferta.where(oferta > 20)
    return pd.concat([df, pd.DataFrame(n, index=df.index)], axis=1)


# =============================================================================
# C) Balance de energia por escalon (MJ)
# =============================================================================

def agregar_balance_energia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = {}
    gn = df["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0)
    o2 = df["volumen_o2_inyectado_lanza_escalon_nm3"].fillna(0) * S("PUREZA_O2")
    aire = df["volumen_aire_inyectado_lanza_escalon_nm3"].fillna(0)
    o2_disp = o2 + 0.21 * aire
    x = (4.0 - 2.0 * o2_disp / gn.where(gn > 1)).clip(0, 1).fillna(0)   # fraccion de C del gas a CO
    n["b6_frac_gas_a_co"] = x
    kmol_gn = gn / VM_NM3_KMOL
    # MJ (kJ/mol * kmol = MJ); escalado por PCI/35.79 para que SUPUESTOS["PCI_GN_MJ_NM3"] sea efectivo
    # (35.79 = 802.3 kJ/mol / 22.414 Nm3/kmol es el PCI implicito del metano puro)
    q_comb = kmol_gn * (-DH_CH4_CO2 - (DH_CH4_CO - DH_CH4_CO2) * x) * (S("PCI_GN_MJ_NM3") / (-DH_CH4_CO2 / VM_NM3_KMOL))
    n["b6_q_combustion_MJ"] = q_comb
    exceso_nm3 = o2_disp - 2.0 * gn
    c_coal = df["feed_Carbon_kgh"].fillna(0) * S("CARBONO_FIJO")
    c_quemado = np.minimum(exceso_nm3.clip(lower=0) / VM_NM3_KMOL * 2 * M_C, c_coal)   # kg C a CO con el O2 sobrante
    n["b6_c_quemado_o2_libre_kg"] = c_quemado
    n["b6_q_carbon_lanza_MJ"] = c_quemado / M_C * (-DH_C_CO)
    n["b6_q_carga_MJ"] = df["feed_total_kgh"].fillna(0) * S("H_CARGA_MJ_KG") + df["feed_Carbon_kgh"].fillna(0) * S("H_CARBON_MJ_KG")
    # gases de salida: N2 del aire + (CO2/CO + 2 H2O) del gas + O2 sobrante + CO del carbon (todo el C fijo, cota superior)
    v_gas = 0.79 * aire + 3.0 * gn + exceso_nm3.clip(lower=0) + c_coal / M_C * VM_NM3_KMOL
    t_prev = df["temperatura_horno_celsius_prev"]
    n["b6_v_gas_salida_nm3"] = v_gas
    n["b6_q_gases_MJ"] = v_gas * S("CP_GAS_KJ_NM3_K") * (t_prev - S("T_REF_C")) / 1000
    perd = df["fase_proceso"].map(S("PERDIDAS_MJ_MIN")).astype(float)
    n["b6_q_perdidas_MJ"] = perd * df["delta_tiempo"]
    q_neto = q_comb + n["b6_q_carbon_lanza_MJ"] - n["b6_q_carga_MJ"] - n["b6_q_gases_MJ"] - n["b6_q_perdidas_MJ"]
    n["b6_q_neto_disponible_MJ"] = q_neto
    n["b6_q_neto_por_min_MJ"] = q_neto / df["delta_tiempo"]
    cap_bano = df["masa_escoria_est_kg_prev"] * S("CP_ESCORIA_KJ_KG_K") / 1000   # MJ/K
    n["b6_capacidad_termica_bano_MJ_K"] = cap_bano
    n["b6_dT_teorico_sin_reaccion"] = q_neto / cap_bano.where(cap_bano > 1)
    # demanda endotermica maxima si todo el Sn disponible y el Fe del mineral se redujeran (a tiempo de decision)
    sn_disp = df["sn_inventario_escoria_est_kg_prev"].fillna(0) + df["feed_Sn_kgf"].fillna(0)
    n["b6_q_reduccion_max_MJ"] = Q_SN_MJ_KG * sn_disp + Q_FE_HEMATITA_MJ_KG * df["feed_Fe_kgh"].fillna(0) * S("FE_MINERAL")
    n["b6_margen_termico_MJ"] = q_neto - n["b6_q_reduccion_max_MJ"]
    # post-hoc
    dT = df["d_temperatura_horno_celsius"]
    n["b6_q_reaccion_aparente_MJ"] = q_neto - cap_bano * dT
    n["b6_q_reaccion_trazador_MJ"] = (Q_SN_MJ_KG * df["sn_extraido_est_kg"].clip(lower=0)
                                      + Q_FEO_MJ_KG * df["feo_extraido_est_kg"].clip(lower=0).fillna(0)
                                      + Q_FE_HEMATITA_MJ_KG * df["feed_Fe_kgh"].fillna(0) * S("FE_MINERAL"))
    n["b6_sn_extraido_energia_kg"] = (n["b6_q_reaccion_aparente_MJ"]
                                      - Q_FEO_MJ_KG * df["feo_extraido_est_kg"].clip(lower=0).fillna(0)
                                      - Q_FE_HEMATITA_MJ_KG * df["feed_Fe_kgh"].fillna(0) * S("FE_MINERAL")) / Q_SN_MJ_KG
    return pd.concat([df, pd.DataFrame(n, index=df.index)], axis=1)


# =============================================================================
# D) Balance global de masa por batch
# =============================================================================

def construir_batch_balance(df: pd.DataFrame, supuestos: dict | None = None) -> pd.DataFrame:
    """Una fila por batch: balance global de masa (estructura del modelo Rust), masa final de escoria por
    balance vs trazador, Sn en escoria final y KPIs (proxy, real, refinado). `supuestos` sobreescribe SUPUESTOS."""
    p = {k: v["valor"] for k, v in SUPUESTOS.items()}
    if supuestos:
        p.update(supuestos)
    dev, lock = mp.split_dev_lockbox(df)
    filas = []
    for b, g in df.groupby("Batch", sort=False):
        g = g.sort_values("fecha_inicio")
        gr = g[g["fase_proceso"] == "Reducción"]
        gf = g[g["fase_proceso"] == "Fusión"]
        tol = {c: g[c].fillna(0).sum() for c in ["feed_conc_t1_kgh", "feed_reciclo_t3_kgh", "feed_Fe_kgh", "feed_dross_Fe_kgh",
                                                  "feed_CaO_kgh", "feed_pellets_t7_kgh", "feed_Carbon_kgh", "feed_total_kgh", "feed_Sn_kgf"]}
        M, D, P = (g[c].iloc[0] * 1000 for c in ["sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"])
        carga_solida = tol["feed_total_kgh"] + tol["feed_CaO_kgh"]
        humedad = (tol["feed_total_kgh"]) * p["HUMEDAD_CARGA"]           # la cal se asume seca
        sn_out = M + D + P
        fe_metal = M / p["LEY_SN_METAL"] * (1 - p["LEY_SN_METAL"]) * 0.8 + D / p["LEY_SN_DROSS"] * p["FE_MET_DROSS"]
        fe_dross_in = tol["feed_dross_Fe_kgh"] * p["FE_MET_DROSS"]
        ceniza = tol["feed_Carbon_kgh"] * p["CENIZA_CARBON"]
        polvo_total = P / p["LEY_SN_POLVO"]
        # escoria final = carga solida seca − Sn que salio (como SnO2, su O se fue en CO) − FeO del Fe metalizado
        #                 + O que gano el Fe metalico del dross al escorificarse − polvo no-Sn arrastrado + ceniza
        masa_esc_bal = (carga_solida - humedad - sn_out * M_SNO2 / M_SN - fe_metal * M_FEO / M_FE
                        + fe_dross_in * M_O / M_FE - (polvo_total - P) + ceniza)
        ult = gr.dropna(subset=["masa_escoria_est_kg"])
        masa_traz = ult["masa_escoria_est_kg"].iloc[-1] if len(ult) else np.nan
        ley_sn_fin = ult["ley_sn_escoria_pct"].iloc[-1] if len(ult) else np.nan
        ley_feo_fin = ult["ley_feo_escoria_pct"].iloc[-1] if len(ult) else np.nan
        sn_esc_bal = masa_esc_bal * ley_sn_fin / 100
        sn_esc_traz = masa_traz * ley_sn_fin / 100
        tot = M + D + P
        fila = dict(Batch=b, fecha_batch=g["fecha_inicio"].min(), es_lockbox=b in lock,
                    carga_solida_kg=carga_solida, feed_sn_kg=tol["feed_Sn_kgf"], carbon_kg=tol["feed_Carbon_kgh"],
                    cal_kg=tol["feed_CaO_kgh"], dross_fe_in_kg=tol["feed_dross_Fe_kgh"], mineral_fe_kg=tol["feed_Fe_kgh"],
                    pellets_kg=tol["feed_pellets_t7_kgh"],
                    sn_metal_kg=M, sn_dross_kg=D, sn_polvo_kg=P,
                    masa_escoria_balance_kg=masa_esc_bal, masa_escoria_trazador_kg=masa_traz,
                    factor_escala_trazador=masa_esc_bal / masa_traz if np.isfinite(masa_traz) and masa_traz > 5000 else np.nan,
                    ley_sn_escoria_final_pct=ley_sn_fin, ley_feo_escoria_final_pct=ley_feo_fin,
                    sn_escoria_final_balance_kg=sn_esc_bal, sn_escoria_final_trazador_kg=sn_esc_traz,
                    cierre_sn_kg=tol["feed_Sn_kgf"] - sn_out - sn_esc_bal,
                    cierre_sn_frac=(tol["feed_Sn_kgf"] - sn_out - sn_esc_bal) / tol["feed_Sn_kgf"] if tol["feed_Sn_kgf"] > 0 else np.nan,
                    rendimiento_proxy_batch=g["rendimiento_proxy_batch"].iloc[0],
                    f_metal=M / tot, f_dross=D / tot, f_polvo=P / tot,
                    recuperacion_real_pct=100 * M / tol["feed_Sn_kgf"] if tol["feed_Sn_kgf"] > 0 else np.nan,
                    recuperacion_refinada_pct=100 * M / (tot + sn_esc_bal) if np.isfinite(sn_esc_bal) else np.nan,
                    sn_perdido_escoria_frac=sn_esc_bal / (tot + sn_esc_bal) if np.isfinite(sn_esc_bal) else np.nan,
                    ley_sn_conc_batch_pct=g["ley_sn_conc_batch_pct"].iloc[0],
                    espesor_ladrillo_norm_mm=g["espesor_ladrillo_norm_mm"].mean(),
                    frac_carga_secundaria_F=gf["frac_carga_secundaria"].mean() if len(gf) else np.nan,
                    feed_dross_Fe_total_t=tol["feed_dross_Fe_kgh"] / 1000, feed_sn_total_t=tol["feed_Sn_kgf"] / 1000,
                    n_escalones_R_con_ensayo=int(len(ult)))
        filas.append(fila)
    out = pd.DataFrame(filas).set_index("Batch").replace([np.inf, -np.inf], np.nan)
    out["idx_cronologico"] = out["fecha_batch"].rank(method="first").astype(int)
    return out


# =============================================================================
# Orquestador y clasificacion
# =============================================================================

def construir_features_v6(df: pd.DataFrame) -> pd.DataFrame:
    df = agregar_multitrazador(df)
    df = agregar_balance_reductor(df)
    df = agregar_balance_energia(df)
    return df


def cargar_df() -> pd.DataFrame:
    return construir_features_v6(pd.read_pickle(CACHE))


CLASIFICACION_V6: dict[str, str] = {
    # A) multi-trazador
    "b6_ln_ratio_masa_multi": "DIAG", "b6_ln_ratio_masa_cao": "DIAG", "b6_dispersion_trazadores": "DIAG",
    "b6_n_inertes_validos": "DIAG",
    "b6_masa_escoria_multi_kg": "LEAKAGE", "b6_sn_inv_multi_kg": "LEAKAGE", "b6_feo_inv_multi_kg": "LEAKAGE",
    "b6_masa_escoria_multi_kg_prev": "STATE", "b6_sn_inv_multi_kg_prev": "STATE", "b6_feo_inv_multi_kg_prev": "STATE",
    "b6_sn_extraido_multi_kg": "TARGET", "b6_feo_extraido_multi_kg": "TARGET", "b6_ln_sn_dep_multi": "TARGET",
    "b6_ln_feo_ret_multi": "TARGET", "b6_sdi_multi": "TARGET", "b6_residuo_masa_kg": "DIAG",
    # B) balance reductor
    "b6_o2_exceso_nm3": "DERIVED-ACTION", "b6_ceq_lanza_kg": "DERIVED-ACTION", "b6_c_fijo_kg": "DERIVED-ACTION",
    "b6_ceq_fe_dross_kg": "DERIVED-ACTION", "b6_demanda_sn_kg_ceq": "DERIVED (A_t x S_prev)",
    "b6_demanda_fe2o3_kg_ceq": "DERIVED-ACTION", "b6_oferta_reductora_kg_ceq": "DERIVED-ACTION",
    "b6_reductor_neto_kg_ceq": "DERIVED (A_t x S_prev)", "b6_ratio_reductor_demanda": "DERIVED (A_t x S_prev)",
    "b6_capacidad_sobre_reduccion_feo_kg": "DERIVED (A_t x S_prev)", "b6_reductor_neto_por_min": "DERIVED (A_t x S_prev)",
    "b6_cum_reductor_neto_kg_prev": "STATE", "b6_cum_oferta_reductora_kg_prev": "STATE",
    "b6_c_consumido_teorico_kg": "LEAKAGE", "b6_utilizacion_reductor": "LEAKAGE",
    # C) energia
    "b6_frac_gas_a_co": "DERIVED-ACTION", "b6_q_combustion_MJ": "DERIVED-ACTION", "b6_c_quemado_o2_libre_kg": "DERIVED-ACTION",
    "b6_q_carbon_lanza_MJ": "DERIVED-ACTION", "b6_q_carga_MJ": "DERIVED-ACTION", "b6_v_gas_salida_nm3": "DERIVED-ACTION",
    "b6_q_gases_MJ": "DERIVED (A_t x S_prev)", "b6_q_perdidas_MJ": "DERIVED-ACTION",
    "b6_q_neto_disponible_MJ": "DERIVED (A_t x S_prev)", "b6_q_neto_por_min_MJ": "DERIVED (A_t x S_prev)",
    "b6_capacidad_termica_bano_MJ_K": "STATE", "b6_dT_teorico_sin_reaccion": "DERIVED (A_t x S_prev)",
    "b6_q_reduccion_max_MJ": "DERIVED (A_t x S_prev)", "b6_margen_termico_MJ": "DERIVED (A_t x S_prev)",
    "b6_q_reaccion_aparente_MJ": "LEAKAGE", "b6_q_reaccion_trazador_MJ": "LEAKAGE", "b6_sn_extraido_energia_kg": "LEAKAGE",
}
SEGURAS_V6 = [c for c, r in CLASIFICACION_V6.items() if r in ("STATE", "DERIVED-ACTION", "DERIVED (A_t x S_prev)")]


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = cargar_df()
    print("df", d.shape, f"{time.time() - t0:.1f}s")
    pd.set_option("display.width", 250)
    cols = [c for c in d.columns if c.startswith("b6_")]
    for f in FASES:
        print("=====", f)
        print(d.loc[d["fase_proceso"] == f, cols].describe(percentiles=[.05, .5, .95]).T[["count", "mean", "5%", "50%", "95%"]].round(3).to_string())
    b = construir_batch_balance(d)
    print("batch", b.shape)
    print(b[["masa_escoria_balance_kg", "masa_escoria_trazador_kg", "factor_escala_trazador", "sn_escoria_final_balance_kg",
             "cierre_sn_frac", "recuperacion_refinada_pct", "rendimiento_proxy_batch"]].describe().T.round(3).to_string())
    faltan = [c for c in cols if c not in CLASIFICACION_V6]
    print("sin clasificar:", faltan)
