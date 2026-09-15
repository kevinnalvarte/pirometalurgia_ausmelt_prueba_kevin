"""ST-09 -- FAMILIA 2 de targets candidatos por escalon (Fusion/Reduccion), ronda 2.

Ver `experimentos/search_targets/SEARCH_TARGETS_ronda2.md` (pregunta ST-09) y `README.md` (ronda 1:
ganadores SDI [Reduccion] e IRF al cierre [Fusion]). NO modifica `st_targets.py`; reutiliza
`st_targets._aux`, `st_targets._agregar_candidato` y el mecanismo de registro (copiado a un dict
local `REGISTRO2`) como patron.

30 candidatos nuevos, ninguno definido con `sn_en_*_batch_t` ni `rendimiento_proxy_batch` (para no ser
una identidad con el KPI):
  FE (a, balance de Fe en Fusion, 4): FE01-04
  SB (b, balance de Sn sobre el "metal implicito", lambda=0.6/1.5, ambas fases, 4): SB01-04
  MX (c, matriz/viscosidad -- NBO/T, FeO/SiO2, Al2O3/(FeO+CaO), IRF twmean/inicio, indice fluidez, 8): MX01-08
  TH (d, sobrecalentamiento vs proxy de liquidus por regresion de la composicion, 3): TH01-03
  PV (e, canal polvo -- T pre-cuchilla, delta T gases, gas total/carga, intensidad de fume, tiro, 8): PV01-08
  CP (f, compuestos teoricos -- SDI extendido a Fusion, SDI+beta*IRF, SDI de dos batches/heel, 3): CP01-03

Bateria identica a ST-01/04: Spearman+Pearson con IC bootstrap (1000) en DEV/lockbox/total, BH por fase
(DEV vs rendimiento_proxy_batch), OLS HC3 con controles (`st_targets.CONTROLES_BATCH` + idx_cronologico),
tercios cronologicos de DEV, Monte Carlo de ruido de ensayo (50 replicas, perturbando %CaO/%Sn/%FeO y
recalculando TODA la cadena de inventarios -- perturbar() es una copia literal de st_04_robustez.py).

Ejecutar:
  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/search_targets/st_09_familia2.py

Salidas: st_09_registro.csv, st_09_batch.csv, st_09_correlaciones.csv, st_09_ranking.csv, st_09_log.txt,
figs/st_09_*.png.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import st_targets as st  # noqa: E402 -- reutiliza _aux, _agregar_candidato, constantes, CONTROLES_BATCH

FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)
N_BOOT = 1000
N_MC = 50

KPIS = st.KPIS                      # rendimiento_proxy_batch, f_metal, f_dross, f_polvo, recuperacion_real_pct
CONTROLES = st.CONTROLES_BATCH
NEG_KPIS = {"f_dross", "f_polvo"}
FASES = st.FASES

# constantes molares adicionales (Guia 5.4, base 100 kg / M_i en kg/kmol)
M_SIO2, M_AL2O3, M_CAO, M_MGO = 60.084, 101.961, 56.077, 40.304
M_FE, M_FEO = st.M_FE, st.M_FEO
FE_FEO = M_FE / M_FEO                # 0.7772 kg Fe / kg FeO
W_SDI = 10.0                         # mismo peso que el ganador R22 (ST-05/06), reutilizado tal cual
BETA_IRF = 1.0                       # peso declarado (no ajustado) del bono de IRF en CP03
K_FLUIDEZ = 1.0                      # peso declarado (no ajustado) del Al2O3 (fraccion) en el "indice de fluidez"
E_R_FUME = 30000.0                   # K, E/R de Arrhenius declarado para la "intensidad de fume" (forma funcional, no calibrado a especie)

LOG: list[str] = []


def log(msg: object = "") -> None:
    s = str(msg)
    print(s)
    LOG.append(s)


def _div(num: pd.Series, den: pd.Series, umbral: float) -> pd.Series:
    ok = den.abs().gt(umbral)
    return (num / den).where(ok)


# =============================================================================
# 1) REGISTRO2 -- 30 candidatos (mismo patron que st_targets._reg / REGISTRO)
# =============================================================================
REGISTRO2: dict[str, dict] = {}


def _reg2(id_: str, fase: str, unidad: str, direccion: int, formula, agg: str, teoria: str, fuente: str,
          num: str | None = None, den: str | None = None, familia: str = "", nota: str = "") -> None:
    REGISTRO2[id_] = dict(id=id_, fase=fase, unidad=unidad, direccion=direccion, formula=formula, agg=agg,
                          num=num, den=den, teoria=teoria, fuente=fuente, familia=familia, nota=nota)


# ---------------------------------------------------------------- FE: balance de Fe en Fusion (a)
_reg2("FE01_frac_fe_retenido_oxido", "Fusión", "fraccion", +1, lambda d: d["feo_ganado_fe_eq"], "ratio",
      "Fraccion del Fe cargado (mineral t4 + dross t5, kg de corriente como proxy de Fe -- sin ley de Fe del feed) "
      "que queda retenido en la escoria como FeO (Fe cargado = Fe-oxido + Fe-metalizado): Fe_FeO = d(FeO_inv)*M_Fe/M_FeO, "
      "num/den = cociente de sumas sobre la fase.",
      "Guia 7.4; consideraciones 5.1-5.2; ST-09", num="feo_ganado_fe_eq", den="fe_feed_kg", familia="balance_Fe")
_reg2("FE02_frac_fe_metalizado_implicito", "Fusión", "fraccion", -1, lambda d: d["fe_metal_implicito_kg"], "ratio",
      "Complemento de FE01 (misma partida de masa, canal complementario): fraccion del Fe cargado que NO quedo como "
      "FeO (metalizado implicitamente -> hardhead/dross). Por construccion FE02 = 1 - FE01 en la muestra total.",
      "Guia 7.4; ST-09", num="fe_metal_implicito_kg", den="fe_feed_kg", familia="balance_Fe")
_reg2("FE03_fe_metalizado_implicito_kg", "Fusión", "kg Fe", -1, lambda d: d["fe_metal_implicito_kg"], "sum",
      "Version en kg absolutos (sin normalizar por tamano de batch) de FE02: Fe cargado menos FeO ganado.",
      "Guia 7.4; ST-09", familia="balance_Fe")
_reg2("FE04_pureza_feo_fe_total", "Reducción", "fraccion", 0, lambda d: _div(d["feo_fe_inv_kg"], d["fe_total_inv_kg"], 50.0), "last",
      "Fraccion del Fe total de la escoria (ensayo ley_fe_total_escoria_pct) explicada por el FeO medido, al cierre de "
      "Reduccion: <1 sugiere Fe no-FeO (Fe3+, u oclusiones/finos metalicos entrampados). Sin signo teorico claro a priori "
      "(exploratorio: podria senalar tanto matriz mas oxidada -bien- como arrastre metalico -mal-).",
      "consideraciones 5.1; Guia 7.4; ST-09", familia="balance_Fe")

# ---------------------------------------------------------------- SB: balance de Sn / "metal implicito" (b)
_reg2("SB01_sn_metal_eq_l06_R", "Reducción", "fraccion del Sn cargado", +1, lambda d: d["_sb_l06"], "sb_custom",
      "Sn extraido de la escoria menos lambda=0.6 veces el FeO extraido (si FeO crece, feo_ext<0 y el termino SUMA -- "
      "matriz protectora --; si FeO se reduce, feo_ext>0 y el termino RESTA -- perdida a dross --), normalizado por el "
      "Sn total cargado al cierre de la fase (cum_feed_sn_incl, ultimo valor: el feed de Sn en Reduccion es ~0 por "
      "escalon, por eso se usa el acumulado y no la suma de feed_Sn_kgf de la fase).",
      "exp 15/19 (lambda_Fe=0.30/0.60); Guia 3.3; ST-09", num="_sb_l06", den="cum_feed_sn_incl", familia="balance_Sn")
_reg2("SB02_sn_metal_eq_l15_R", "Reducción", "fraccion del Sn cargado", +1, lambda d: d["_sb_l15"], "sb_custom",
      "Igual que SB01 con lambda=1.5 (limite superior exploratorio, por encima del IC de lambda_Fe de exp 19).",
      "exp 19 IC; Guia 3.3; ST-09", num="_sb_l15", den="cum_feed_sn_incl", familia="balance_Sn")
_reg2("SB03_sn_metal_eq_l06_F", "Fusión", "fraccion del Sn cargado", +1, lambda d: d["_sb_l06"], "sb_custom",
      "Igual que SB01 pero evaluado en Fusion: alli feo_ext es tipicamente negativo (el FeO se esta formando, no "
      "reduciendo), asi que el termino -lambda*feo_ext es tipicamente positivo -- premia formar FeO sin extraer Sn.",
      "exp 15/19; Guia 3.3; ST-09", num="_sb_l06", den="cum_feed_sn_incl", familia="balance_Sn")
_reg2("SB04_sn_metal_eq_l15_F", "Fusión", "fraccion del Sn cargado", +1, lambda d: d["_sb_l15"], "sb_custom",
      "Igual que SB03 con lambda=1.5.", "exp 19 IC; Guia 3.3; ST-09", num="_sb_l15", den="cum_feed_sn_incl", familia="balance_Sn")

# ---------------------------------------------------------------- MX: matriz / viscosidad (c)
_reg2("MX01_nbo_t_F", "Fusión", "adimensional", +1, lambda d: d["nbo_t"], "last",
      "NBO/T aproximado (Guia 5.4, idealizacion con Si/Al tetraedricos y modificadores divalentes): "
      "2*(n_CaO+n_MgO+n_FeO - n_Al2O3) / (n_SiO2 + 2*n_Al2O3), usando %FeO MEDIDO (no FeO-equivalente de Fe total). "
      "Mas alto = escoria mas despolimerizada (mas fluida) dentro del rango historico; fuera de rango puede favorecer "
      "solidos (5.5-5.6) -- indice bajo supuestos, no NBO/T termodinamico medido.",
      "Guia 5.4; ST-09", familia="matriz")
_reg2("MX02_nbo_t_R", "Reducción", "adimensional", +1, lambda d: d["nbo_t"], "twmean",
      "NBO/T aproximado (igual formula que MX01), media ponderada por tiempo durante toda la Reduccion.",
      "Guia 5.4; ST-09", familia="matriz")
_reg2("MX03_feo_sio2_F", "Fusión", "kg/kg", +1, lambda d: d["feo_sio2"], "last",
      "Razon fayalitica B_f = %FeO/%SiO2 (Guia 5.6) al cierre de Fusion: alta = matriz fayalitica fluida (2FeO+SiO2 <-> Fe2SiO4).",
      "Guia 5.6; ST-09", familia="matriz")
_reg2("MX04_feo_sio2_R", "Reducción", "kg/kg", +1, lambda d: d["feo_sio2"], "twmean",
      "Razon fayalitica FeO/SiO2, media ponderada por tiempo en Reduccion.", "Guia 5.6; ST-09", familia="matriz")
_reg2("MX05_al2o3_sobre_mod_R", "Reducción", "kg/kg", -1, lambda d: d["al2o3_sobre_mod"], "twmean",
      "Al2O3 / (FeO+CaO): formador/compensador de red (Al2O3, Guia 5.6/7.5) sobre modificadores; alto = mas viscoso "
      "y mayor liquidus. Minimizar (media ponderada por tiempo en Reduccion).",
      "Guia 5.6, 7.5; ST-09", familia="matriz")
_reg2("MX06_irf_twmean_R", "Reducción", "IRF", +1, lambda d: d["irf"], "twmean",
      "IRF medio ponderado por tiempo durante TODA la Reduccion (contraste con el ganador F10, que es el 'ultimo valor' "
      "al cierre de Fusion): mide si sostener un IRF alto durante la reduccion misma aporta señal adicional.",
      "hallazgos 14.6; ST-05; ST-09", familia="matriz")
_reg2("MX07_irf_inicio_R", "Reducción", "IRF", +1, lambda d: d["irf"], "first",
      "IRF al primer escalon VALIDO de Reduccion (estado heredado del cierre de Fusion): separa el estado inicial del "
      "cambio durante la fase, en paralelo al hallazgo 'heel de matriz de escoria' de ST-05 (IRF de F0 <- IRF final de "
      "la Reduccion anterior) pero mirado desde el lado de Reduccion.",
      "ST-05 hallazgo heel; ST-09", familia="matriz")
_reg2("MX08_indice_fluidez_R", "Reducción", "adimensional", +1, lambda d: d["indice_fluidez"], "twmean",
      f"Indice de fluidez = IRF - {K_FLUIDEZ}*(%Al2O3/100): valora el IRF y penaliza el formador de red Al2O3 a la vez. "
      "Peso K ilustrativo (no ajustado por regresion), declarado.",
      "Guia 5.6, 7.5; ST-09", familia="matriz")

# ---------------------------------------------------------------- TH: sobrecalentamiento / liquidus proxy (d)
_reg2("TH01_sobrecal_resid_R", "Reducción", "C", 0, lambda d: d["sobrecal_resid"], "twmean",
      "Sobrecalentamiento proxy: T_bano - T_esperada segun composicion, con T_esperada de un OLS T ~ B2 + %FeO + %SiO2 "
      "+ %Al2O3 + %CaO AJUSTADO SOLO EN DEV, por fase (evita fuga hacia lockbox). No es liquidus real (no hay dato "
      "termodinamico), es la T tipica historica para esa matriz; residuo alto = mas caliente de lo usual para esa "
      "composicion. Sin signo a priori: Guia 5.6 describe una VENTANA de sobrecalentamiento, no un maximo/minimo.",
      "Guia 5.6; ST-09", familia="termico")
_reg2("TH02_sobrecal_resid_F", "Fusión", "C", 0, lambda d: d["sobrecal_resid"], "twmean",
      "Igual que TH01, con el OLS ajustado en los escalones de Fusion de DEV.", "Guia 5.6; ST-09", familia="termico")
_reg2("TH03_sobrecal_ventana_R", "Reducción", "-C", +1, lambda d: d["sobrecal_ventana"], "twmean",
      "Version 'ventana' de TH01: -|residuo|, cercania al sobrecalentamiento tipico de esa composicion (coherente con "
      "la lectura de ventana de Guia 5.6, en vez de forzar mas o menos T).",
      "Guia 5.6; ST-09", familia="termico")

# ---------------------------------------------------------------- PV: canal polvo (e)
_reg2("PV01_T_precuchilla_R", "Reducción", "C", -1, lambda d: d["T_precuchilla"], "twmean",
      "Temperatura de gases pre-cuchilla: sensor downstream, proxy de carga termica / estado del tren de gases "
      "(consid. 22.3). Minimizar (media ponderada por tiempo).", "consideraciones 22.3; ST-09", familia="polvo")
_reg2("PV02_T_precuchilla_F", "Fusión", "C", -1, lambda d: d["T_precuchilla"], "twmean",
      "Igual que PV01 en Fusion.", "consideraciones 22.3; ST-09", familia="polvo")
_reg2("PV03_delta_T_gas_R", "Reducción", "C", 0, lambda d: d["delta_T_gas"], "twmean",
      "Delta T entre pre-BHF y pre-cuchilla (T_bhf - T_cuchilla): mayor delta = mas enfriamiento/reaccion en el tramo "
      "del tren de gases. Signo practico ambiguo (mas enfriamiento puede ser deposito/incrustacion -malo- o intercambio "
      "termico normal -neutro-, consid. 22.3-22.4): exploratorio.", "consideraciones 22.3-22.4; ST-09", familia="polvo")
_reg2("PV04_gas_total_por_carga_R", "Reducción", "Nm3/kg", -1, lambda d: d["gas_total_nm3"], "ratio",
      "Gas total inyectado por la lanza (GN+O2+aire, Nm3 del escalon) por kg de carga total del escalon (misma "
      "convencion que F13 de st_targets.py: feed_total_kgh como proxy de kg de carga). Mas gas por unidad de carga "
      "= mas intensidad/arrastre mecanico (consid. 22.2). Minimizar.",
      "consideraciones 22.2; Guia 22; ST-09", num="gas_total_nm3", den="feed_total_kgh", familia="polvo")
_reg2("PV05_gas_total_por_carga_F", "Fusión", "Nm3/kg", -1, lambda d: d["gas_total_nm3"], "ratio",
      "Igual que PV04 en Fusion.", "consideraciones 22.2; ST-09", num="gas_total_nm3", den="feed_total_kgh", familia="polvo")
_reg2("PV06_fume_intensidad_R", "Reducción", "Nm3*exp(-E/RT)", -1, lambda d: d["fume_intensidad"], "sum",
      f"Intensidad de fume = gas_total * exp(-E/(R*T_K)), E/R = {E_R_FUME:.0f} K (declarado, forma funcional de "
      "Arrhenius para presion de vapor / cinetica de volatilizacion, Guia 3.5 y 4.3; NO calibrado a una especie "
      "concreta de Sn). Suma = exposicion acumulada de la fase. Minimizar.",
      "Guia 3.5, 4.3; ST-09", familia="polvo")
_reg2("PV07_fume_intensidad_F", "Fusión", "Nm3*exp(-E/RT)", -1, lambda d: d["fume_intensidad"], "sum",
      "Igual que PV06 en Fusion.", "Guia 3.5, 4.3; ST-09", familia="polvo")
_reg2("PV08_tiro_ventana_R", "Reducción", "-pct", +1, lambda d: d["tiro_ventana"], "twmean",
      "Cercania del tiro de horno a su valor tipico (mediana de DEV en la fase): tanto muy poco como demasiado tiro "
      "son indeseables (consid. 21.1-21.2) y no hay un optimo unico documentado -- version 'ventana', no monotonica "
      "en el propio tiro.", "consideraciones 21.1-21.3; ST-09", familia="polvo")

# ---------------------------------------------------------------- CP: compuestos teoricos (f)
_reg2("CP01_sdi_heel2b", "Reducción", "adimensional", +1, None, "batch_especial",
      "SDI de dos batches: promedio del SDI del propio batch (st_R22_sdi, ganador ST-05/06, tomado de st_batch.csv) y "
      "el SDI del batch CRONOLOGICAMENTE ANTERIOR (heel de escoria que la Reduccion previa hereda hacia la Fusion "
      "siguiente, hallazgo ST-05). No es una agregacion escalon->batch: es una combinacion directa de dos valores ya "
      "agregados por batch (shift(1) sobre idx_cronologico); no se recalcula bajo ruido Monte Carlo (hereda la "
      "sensibilidad al ruido de R22, ya caracterizada en st_04/README).",
      "ST-05 hallazgo heel; ST-09", familia="compuesto")
_reg2("CP02_sdi_fusion_log", "Fusión", "ln(kg/kg)", +1, lambda d: d["feo_ganado_fe_eq"], "ratio_log",
      "SDI extendido a Fusion: ln( FeO ganado en base Fe / Fe cargado ) = ln(cociente de sumas) -- retencion "
      "logaritmica de FeO respecto al Fe cargado, analogo del componente de FeO del SDI de Reduccion (ln_feo_ret) "
      "pero aplicado a la fase de FORMACION de escoria.",
      "ST-05/06 SDI; Guia 7.4; ST-09", num="feo_ganado_fe_eq", den="fe_feed_kg", familia="compuesto")
_reg2("CP03_sdi_beta_irf", "Reducción", "adimensional", +1, lambda d: d["sdi_beta_irf_term"], "sum",
      f"SDI (ln(Sn_prev/Sn) + {W_SDI:.0f}*ln(FeO/FeO_prev), mismo w que el ganador R22) + beta*IRF anadido SOLO en el "
      f"ULTIMO escalon valido de la fase, beta={BETA_IRF:.1f} (declarado, sin ajustar por barrido): premia terminar la "
      "Reduccion agotando Sn selectivamente Y con una matriz FeO-rica (heel de matriz para la Fusion siguiente).",
      "ST-05/06 SDI; hallazgos 14.6; ST-09", familia="compuesto")

N_TOTAL = len(REGISTRO2)
REGISTRO2_ESCALON = {k: v for k, v in REGISTRO2.items() if v["agg"] != "batch_especial"}
CAND_IDS = list(REGISTRO2.keys())
FASE_DE = {k: v["fase"] for k, v in REGISTRO2.items()}
FAMILIA_DE = {k: v["familia"] for k, v in REGISTRO2.items()}
AGG_DE = {k: v["agg"] for k, v in REGISTRO2.items()}


# =============================================================================
# 2) columnas auxiliares de familia 2 (sobre un df ya pasado por st._aux)
# =============================================================================

def agregar_aux_familia2(d: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    d = d.copy()
    d["feo_ganado_fe_eq"] = d["d_feo_inv"] * FE_FEO
    d["fe_metal_implicito_kg"] = d["fe_feed_kg"] - d["feo_ganado_fe_eq"]
    d["fe_total_inv_kg"] = d["masa_escoria_est_kg"] * d["ley_fe_total_escoria_pct"] / 100.0
    d["feo_fe_inv_kg"] = d["feo_inventario_escoria_est_kg"] * FE_FEO

    n_sio2 = d["ley_sio2_escoria_pct"] / M_SIO2
    n_al2o3 = d["ley_al2o3_escoria_pct"] / M_AL2O3
    n_cao = d["ley_cao_escoria_pct"] / M_CAO
    n_mgo = d["ley_mgo_escoria_pct"] / M_MGO
    n_feo = d["ley_feo_escoria_pct"] / M_FEO
    n_mo = n_cao + n_mgo + n_feo
    denom_nbo = n_sio2 + 2 * n_al2o3
    nbo = (2 * (n_mo - n_al2o3) / denom_nbo).where(denom_nbo.abs() > 1e-6)
    d["nbo_t"] = nbo.clip(-20, 20)
    d["feo_sio2"] = _div(d["ley_feo_escoria_pct"], d["ley_sio2_escoria_pct"], 0.5).clip(0, 50)
    d["al2o3_sobre_mod"] = _div(d["ley_al2o3_escoria_pct"], d["ley_feo_escoria_pct"] + d["ley_cao_escoria_pct"], 0.5).clip(0, 10)
    d["indice_fluidez"] = d["irf"] - K_FLUIDEZ * (d["ley_al2o3_escoria_pct"] / 100.0)

    d["T_precuchilla"] = d["temperatura_gas_pre_cuchilla_celsius"]
    d["gas_total_nm3"] = (d["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0)
                           + d["volumen_o2_inyectado_lanza_escalon_nm3"].fillna(0)
                           + d["volumen_aire_inyectado_lanza_escalon_nm3"].fillna(0))
    T_K = d["T"] + 273.15
    d["fume_intensidad"] = d["gas_total_nm3"] * np.exp(-E_R_FUME / T_K.where(T_K > 200))
    d["delta_T_gas"] = d["temperatura_gas_pre_bhf_celsius"] - d["temperatura_gas_pre_cuchilla_celsius"]

    d["tiro_ventana"] = np.nan
    for fase in FASES:
        mfase = d["fase_proceso"] == fase
        med = d.loc[mfase & (~d["es_lockbox"].astype(bool)), "tiro_horno_pct"].median()
        d.loc[mfase, "tiro_ventana"] = -(d.loc[mfase, "tiro_horno_pct"] - med).abs()

    d["sobrecal_resid"] = np.nan
    regresores = ["B2", "ley_feo_escoria_pct", "ley_sio2_escoria_pct", "ley_al2o3_escoria_pct", "ley_cao_escoria_pct"]
    for fase in FASES:
        mfase = d["fase_proceso"] == fase
        base_ok = d[["T"] + regresores].notna().all(axis=1)
        mfit = mfase & (~d["es_lockbox"].astype(bool)) & base_ok
        mpred = mfase & base_ok
        if mfit.sum() < 30:
            continue
        X_fit = sm.add_constant(d.loc[mfit, regresores])
        modelo = sm.OLS(d.loc[mfit, "T"], X_fit).fit()
        Xp = sm.add_constant(d.loc[mpred, regresores], has_constant="add")
        d.loc[mpred, "sobrecal_resid"] = d.loc[mpred, "T"] - modelo.predict(Xp)
        if verbose:
            log(f"  regresion sobrecalentamiento {fase}: n_fit={int(mfit.sum())}, R2={modelo.rsquared:.3f}, "
                f"coefs={modelo.params.round(3).to_dict()}")
    d["sobrecal_ventana"] = -d["sobrecal_resid"].abs()

    d["_sb_l06"] = d["sn_ext"] - 0.6 * d["feo_ext"]
    d["_sb_l15"] = d["sn_ext"] - 1.5 * d["feo_ext"]

    sdi_component = st._ln_sn_dep(d) + W_SDI * st._ln_feo_ret(d)
    d["sdi_component"] = sdi_component
    d["_es_ultimo_valido_R"] = False
    mR = (d["fase_proceso"] == "Reducción") & d["sdi_component"].notna()
    if mR.any():
        idx_last = d[mR].groupby("Batch").tail(1).index
        d.loc[idx_last, "_es_ultimo_valido_R"] = True
    d["sdi_beta_irf_term"] = np.where(d["_es_ultimo_valido_R"], sdi_component + BETA_IRF * d["irf"], sdi_component)
    return d


def construir_df_targets2(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    nuevas = {}
    for id_, r in REGISTRO2_ESCALON.items():
        s = r["formula"](d).astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.where(d["fase_proceso"] == r["fase"])
        s = s.where(~d["es_primer_escalon_batch"])
        nuevas["st_" + id_] = s * (r["direccion"] if r["direccion"] != 0 else 1)
    return pd.concat([d, pd.DataFrame(nuevas, index=d.index)], axis=1)


def _agregar_candidato2(g: pd.DataFrame, r: dict) -> float:
    """Extiende st_targets._agregar_candidato con 'first' y 'sb_custom' (denominador = ultimo valor, no suma)."""
    dirn = r["direccion"] if r["direccion"] != 0 else 1
    agg = r["agg"]
    col = "st_" + r["id"]
    if agg == "first":
        v = g[col].dropna()
        return float(v.iloc[0]) if len(v) else np.nan
    if agg == "sb_custom":
        x = g[col]
        m = x.notna()
        if not m.any():
            return np.nan
        num_raw = g.loc[m, r["num"]].sum()
        den_series = g[r["den"]].dropna()
        if not len(den_series):
            return np.nan
        den = den_series.iloc[-1]
        if not (den > 50.0):
            return np.nan
        return float(dirn * num_raw / den)
    return st._agregar_candidato(g, r)


def construir_batch2(d2: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for b, g in d2.groupby("Batch", sort=False):
        g = g.sort_values("fecha_inicio")
        gf, gr = g[g["fase_proceso"] == "Fusión"], g[g["fase_proceso"] == "Reducción"]
        fila = {"Batch": b}
        for id_, r in REGISTRO2_ESCALON.items():
            sub = gf if r["fase"] == "Fusión" else gr
            fila["st_" + id_] = _agregar_candidato2(sub, r) if len(sub) else np.nan
            fila["n_" + id_] = int(sub["st_" + id_].notna().sum()) if len(sub) else 0
        filas.append(fila)
    return pd.DataFrame(filas).set_index("Batch")


# =============================================================================
# 3) datos base y pipeline principal
# =============================================================================

def main() -> None:
    t0 = time.time()
    df_raw = pd.read_pickle(HERE / "cache_df_st.pkl")
    BASE_COLS = [c for c in df_raw.columns if not c.startswith("st_")]
    df0 = df_raw[BASE_COLS].sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()

    # st_batch.csv en disco quedo desactualizado (36 candidatos originales, SIN los ganadores R21/R22 anadidos
    # despues por ST-05/06 -- ver st_targets.py REGISTRO actual). Se regenera fresco con las funciones oficiales
    # de st_targets.py (mismos KPIs/controles/es_lockbox/idx_cronologico, pero con R21_ln_feo_ret y R22_sdi
    # disponibles para CP01, el compuesto "SDI de dos batches").
    batch_ref = st.construir_batch(st.construir_df_targets(df0.copy())).reset_index()
    if "st_R22_sdi" not in batch_ref.columns:
        raise RuntimeError("st_targets.REGISTRO no contiene R22_sdi -- revisar version de st_targets.py")
    batch_ref = batch_ref.set_index("Batch")
    log(f"ST-09 familia2 -- carga {time.time()-t0:.1f}s | escalones {df0.shape}, batches {batch_ref.shape} "
        f"(batch_ref regenerado en vivo desde st_targets.py, no desde el st_batch.csv obsoleto en disco), "
        f"candidatos={N_TOTAL} ({len(REGISTRO2_ESCALON)} por escalon + 1 compuesto de batch)")

    d = st._aux(df0.copy())
    d["es_lockbox"] = d["Batch"].map(batch_ref["es_lockbox"]).astype(bool)
    d = agregar_aux_familia2(d, verbose=True)

    d2 = construir_df_targets2(d)
    batch2 = construir_batch2(d2)
    batch = batch_ref.join(batch2, how="left")
    batch = batch.sort_values("idx_cronologico")
    batch["st_CP01_sdi_heel2b"] = 0.5 * (batch["st_R22_sdi"] + batch["st_R22_sdi"].shift(1))
    batch["n_CP01_sdi_heel2b"] = np.where(batch["st_R22_sdi"].notna() & batch["st_R22_sdi"].shift(1).notna(), 2, 0)

    dev = batch[~batch["es_lockbox"]]
    lockbox = batch[batch["es_lockbox"]]
    total = batch
    muestras = {"dev": dev, "lockbox": lockbox, "total": total}
    log(f"batch construido {time.time()-t0:.1f}s | dev n={len(dev)}, lockbox n={len(lockbox)}, "
        f"escalones familia2 (Reduccion/Fusion)={dict(d2.groupby('fase_proceso').size())}")

    batch.reset_index().to_csv(HERE / "st_09_batch.csv", index=False, encoding="utf-8")
    tabla_reg = pd.DataFrame([{k: v for k, v in r.items() if k != "formula"} for r in REGISTRO2.values()]).set_index("id")
    tabla_reg.to_csv(HERE / "st_09_registro.csv", encoding="utf-8")
    log(f"[datos] st_09_batch.csv y st_09_registro.csv escritos ({N_TOTAL} candidatos).")

    # -------------------------------------------------------------------
    # 4) correlaciones (Spearman + Pearson, IC bootstrap) x KPI x muestra
    # -------------------------------------------------------------------
    def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
        n = len(x)
        if n < 8:
            return np.nan, np.nan
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        idx = RNG.integers(0, n, size=(n_boot, n))
        rxb, ryb = rx[idx], ry[idx]
        mx, my = rxb.mean(axis=1, keepdims=True), ryb.mean(axis=1, keepdims=True)
        dx, dy = rxb - mx, ryb - my
        num = (dx * dy).sum(axis=1)
        den = np.sqrt((dx ** 2).sum(axis=1) * (dy ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / den
        r = r[np.isfinite(r)]
        if len(r) < n_boot * 0.5:
            return np.nan, np.nan
        return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

    def corr_pair(x: pd.Series, y: pd.Series) -> dict:
        sub = pd.concat([x, y], axis=1).dropna()
        n = len(sub)
        out = dict(n=n, rho=np.nan, p_rho=np.nan, r=np.nan, p_r=np.nan, ci_low=np.nan, ci_high=np.nan)
        if n < 8:
            return out
        xv, yv = sub.iloc[:, 0].to_numpy(float), sub.iloc[:, 1].to_numpy(float)
        if np.std(xv) == 0 or np.std(yv) == 0:
            return out
        rho, p_rho = stats.spearmanr(xv, yv)
        r, p_r = stats.pearsonr(xv, yv)
        ci_low, ci_high = bootstrap_ci_spearman(xv, yv)
        out.update(rho=rho, p_rho=p_rho, r=r, p_r=p_r, ci_low=ci_low, ci_high=ci_high)
        return out

    st_col = {c: f"st_{c}" for c in CAND_IDS}
    rows = []
    for cid in CAND_IDS:
        col = st_col[cid]
        for kpi in KPIS:
            for mname, mdf in muestras.items():
                r = corr_pair(mdf[col], mdf[kpi])
                row = dict(id=cid, fase=FASE_DE[cid], familia=FAMILIA_DE[cid], agg=AGG_DE[cid], kpi=kpi, muestra=mname,
                           n=r["n"], rho_spearman=r["rho"], p_spearman=r["p_rho"], r_pearson=r["r"], p_pearson=r["p_r"],
                           ci_low=r["ci_low"], ci_high=r["ci_high"])
                row["signo_esperado"] = -1 if kpi in NEG_KPIS else np.nan
                row["signo_ok"] = (r["rho"] < 0) if (kpi in NEG_KPIS and pd.notna(r["rho"])) else np.nan
                rows.append(row)
    corr = pd.DataFrame(rows)

    dir_ok_map = {}
    for cid in CAND_IDS:
        m = (corr["id"] == cid) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")
        rho_dev = corr.loc[m, "rho_spearman"]
        dir_ok_map[cid] = bool(rho_dev.iloc[0] > 0) if len(rho_dev) and pd.notna(rho_dev.iloc[0]) else False
    corr["dir_ok"] = corr["id"].map(dir_ok_map)

    # BH por fase, DEV, vs rendimiento_proxy_batch
    corr["p_bh"] = np.nan
    corr["significativo_bh"] = pd.Series([np.nan] * len(corr), dtype=object)
    mask_base = (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")
    for fase in FASES:
        m = mask_base & (corr["fase"] == fase)
        pvals = corr.loc[m, "p_spearman"].to_numpy()
        valid = ~np.isnan(pvals)
        p_bh = np.full(len(pvals), np.nan)
        sig_bh = np.full(len(pvals), np.nan, dtype=object)
        if valid.sum() > 0:
            rej_bh, pb, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
            p_bh[valid] = pb
            sig_bh[valid] = rej_bh
        corr.loc[m, "p_bh"] = p_bh
        corr.loc[m, "significativo_bh"] = sig_bh

    corr.to_csv(HERE / "st_09_correlaciones.csv", index=False, encoding="utf-8")
    log(f"[1] st_09_correlaciones.csv escrito ({len(corr)} filas).")

    # -------------------------------------------------------------------
    # 5) OLS HC3 con controles (+ idx_cronologico), estandarizado con TOTAL
    # -------------------------------------------------------------------
    z_vars = [st_col[c] for c in CAND_IDS] + CONTROLES + ["idx_cronologico"]
    means_total = {v: total[v].mean() for v in z_vars}
    stds_total = {v: total[v].std(ddof=1) for v in z_vars}
    batch_z = batch.copy()
    for v in z_vars:
        s = stds_total[v]
        batch_z["z_" + v] = (batch[v] - means_total[v]) / s if s and s > 1e-12 else np.nan
    ctrl_z = ["z_" + c for c in CONTROLES] + ["z_idx_cronologico"]

    def fit_ols(sub: pd.DataFrame, regressors: list[str], kpi: str):
        used = [c for c in regressors if sub[c].std(ddof=1) >= 1e-8]
        if not used:
            return None, used
        X = sm.add_constant(sub[used], has_constant="add")
        try:
            model = sm.OLS(sub[kpi], X).fit(cov_type="HC3")
        except Exception:
            return None, used
        return model, used

    ols_rows = []
    for cid in CAND_IDS:
        cz = "z_" + st_col[cid]
        for mname, mdf in muestras.items():
            sub = batch_z.loc[mdf.index, [cz] + ctrl_z + ["rendimiento_proxy_batch"]].dropna()
            n = len(sub)
            if n < 15 or sub[cz].std(ddof=1) < 1e-8:
                ols_rows.append(dict(id=cid, muestra=mname, n=n, coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, r2_adj=np.nan))
                continue
            model, used = fit_ols(sub, [cz] + ctrl_z, "rendimiento_proxy_batch")
            if model is None or cz not in model.params.index:
                ols_rows.append(dict(id=cid, muestra=mname, n=n, coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, r2_adj=np.nan))
                continue
            ci = model.conf_int().loc[cz]
            ols_rows.append(dict(id=cid, muestra=mname, n=n, coef=model.params[cz], ci_low=ci[0], ci_high=ci[1],
                                  p=model.pvalues[cz], r2_adj=model.rsquared_adj))
    ols_df = pd.DataFrame(ols_rows)
    log(f"[2] OLS HC3 con controles calculado ({len(ols_df)} filas).")

    # -------------------------------------------------------------------
    # 6) tercios cronologicos de DEV
    # -------------------------------------------------------------------
    dev_sorted = dev.sort_values("idx_cronologico")
    terciles = np.array_split(np.array(dev_sorted.index), 3)
    bloques = {"tercio1": terciles[0], "tercio2": terciles[1], "tercio3": terciles[2]}

    def spearman_np(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            return np.nan, np.nan, int(m.sum())
        rho, p = stats.spearmanr(x[m], y[m])
        return float(rho), float(p), int(m.sum())

    terc_rows = []
    for cid in CAND_IDS:
        col = batch[st_col[cid]]
        rhos = {}
        for nombre, idxb in bloques.items():
            rho, p, n = spearman_np(col.loc[idxb], batch.loc[idxb, "rendimiento_proxy_batch"])
            rhos[nombre] = rho
        vals = [v for v in rhos.values() if pd.notna(v)]
        terc_rows.append(dict(id=cid, rho_tercio1=rhos["tercio1"], rho_tercio2=rhos["tercio2"], rho_tercio3=rhos["tercio3"],
                               n_tercios_pos=sum(1 for v in vals if v > 0), n_tercios_validos=len(vals)))
    terc_df = pd.DataFrame(terc_rows).set_index("id")
    log(f"[3] tercios cronologicos de DEV calculados.")

    # -------------------------------------------------------------------
    # 7) Monte Carlo de ruido de ensayo (50 replicas, copia de st_04_robustez.py)
    # -------------------------------------------------------------------
    def perturbar(d0: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        d = d0.copy()
        n = len(d)
        cao0 = d["ley_cao_escoria_pct"]
        sn0 = d["ley_sn_escoria_pct"]
        feo0 = d["ley_feo_escoria_pct"]
        cao = cao0 * (1 + rng.normal(0, 0.02, n))
        sn = (sn0 + rng.normal(0, 0.05, n)).clip(lower=0)
        feo = (feo0 + rng.normal(0, 0.3, n)).clip(lower=0)
        d["ley_cao_escoria_pct"] = cao
        d["ley_sn_escoria_pct"] = sn
        d["ley_feo_escoria_pct"] = feo

        cum_cao_incl = d.groupby("Batch")["feed_CaO_kgh"].cumsum()
        masa = cum_cao_incl / (cao.where(cao > 0.5) / 100)
        sn_inv = masa * sn / 100
        feo_inv = masa * feo / 100
        d["masa_escoria_est_kg"] = masa
        d["sn_inventario_escoria_est_kg"] = sn_inv
        d["feo_inventario_escoria_est_kg"] = feo_inv
        masa_prev = masa.groupby(d["Batch"]).shift(1)
        sn_inv_prev = sn_inv.groupby(d["Batch"]).shift(1)
        feo_inv_prev = feo_inv.groupby(d["Batch"]).shift(1)
        d["masa_escoria_est_kg_prev"] = masa_prev
        d["sn_inventario_escoria_est_kg_prev"] = sn_inv_prev
        d["feo_inventario_escoria_est_kg_prev"] = feo_inv_prev

        d_masa = masa - masa_prev
        d_sn_inv = sn_inv - sn_inv_prev
        sn_extraido = d["feed_Sn_kgf"] - d_sn_inv
        disponible = sn_inv_prev + d["feed_Sn_kgf"]
        d["d_masa_escoria_est_kg"] = d_masa
        d["sn_extraido_est_kg"] = sn_extraido
        d["frac_sn_extraido_escalon"] = sn_extraido / disponible.where(disponible.abs() > 20.0)
        feo_extraido = -(feo_inv - feo_inv_prev)
        d["feo_extraido_est_kg"] = feo_extraido

        sio2 = d["ley_sio2_escoria_pct"]
        d["basicidad_B2"] = cao / sio2.where(sio2.abs() > 1e-6)
        denom_irf = sio2 + d["ley_al2o3_escoria_pct"] + cao
        d["indice_irf"] = feo / denom_irf.where(denom_irf.abs() > 1e-6)
        return d

    t_mc0 = time.time()
    rng_mc = np.random.default_rng(SEED)
    kpi_dev = batch.loc[dev.index, "rendimiento_proxy_batch"]
    rho_mat = np.full((N_MC, len(CAND_IDS)), np.nan)
    id_pos = {cid: j for j, cid in enumerate(CAND_IDS)}
    for rep in range(N_MC):
        d_p = perturbar(df0, rng_mc)
        d_p = st._aux(d_p)
        d_p["es_lockbox"] = d_p["Batch"].map(batch_ref["es_lockbox"]).astype(bool)
        d_p = agregar_aux_familia2(d_p, verbose=False)
        d_p2 = construir_df_targets2(d_p)
        b_p = construir_batch2(d_p2)
        b_p_dev = b_p.loc[b_p.index.intersection(dev.index)]
        for cid in CAND_IDS:
            if AGG_DE[cid] == "batch_especial":
                continue
            rho, p, n = spearman_np(b_p_dev[st_col[cid]], kpi_dev.loc[b_p_dev.index])
            rho_mat[rep, id_pos[cid]] = rho
        if (rep + 1) % 10 == 0 or rep == N_MC - 1:
            log(f"  MC replica {rep+1}/{N_MC} ({time.time()-t_mc0:.0f}s transcurridos)")

    mc_rows = []
    for cid in CAND_IDS:
        j = id_pos[cid]
        rho_col = rho_mat[:, j]
        ok = np.isfinite(rho_col)
        mc_rows.append(dict(id=cid, n_replicas=int(ok.sum()),
                             rho_mc_media=float(np.nanmean(rho_col)) if ok.any() else np.nan,
                             rho_mc_sd=float(np.nanstd(rho_col)) if ok.any() else np.nan,
                             frac_rho_positivo=float(np.nanmean(rho_col > 0)) if ok.any() else np.nan))
    mc_df = pd.DataFrame(mc_rows).set_index("id")
    log(f"[4] Monte Carlo de ruido de ensayo ({N_MC} replicas, {time.time()-t_mc0:.0f}s). CP01 no se recalcula "
        f"bajo ruido (hereda la de R22, ver README/st_04).")

    # -------------------------------------------------------------------
    # 8) ranking final
    # -------------------------------------------------------------------
    rank_rows = []
    for cid in CAND_IDS:
        row = dict(id=cid, fase=FASE_DE[cid], familia=FAMILIA_DE[cid], agg=AGG_DE[cid])
        dev_r = corr[(corr["id"] == cid) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")].iloc[0]
        lb_r = corr[(corr["id"] == cid) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "lockbox")].iloc[0]
        tot_r = corr[(corr["id"] == cid) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "total")].iloc[0]
        fdross_r = corr[(corr["id"] == cid) & (corr["kpi"] == "f_dross") & (corr["muestra"] == "dev")].iloc[0]
        fpolvo_r = corr[(corr["id"] == cid) & (corr["kpi"] == "f_polvo") & (corr["muestra"] == "dev")].iloc[0]
        recu_r = corr[(corr["id"] == cid) & (corr["kpi"] == "recuperacion_real_pct") & (corr["muestra"] == "dev")].iloc[0]
        row.update(n_dev=dev_r["n"], rho_dev=dev_r["rho_spearman"], p_spearman_dev=dev_r["p_spearman"], p_bh_dev=dev_r["p_bh"],
                   ci_dev=f"[{dev_r['ci_low']:.3f}, {dev_r['ci_high']:.3f}]" if pd.notna(dev_r["ci_low"]) else "",
                   rho_lockbox=lb_r["rho_spearman"], p_lockbox=lb_r["p_spearman"], n_lockbox=lb_r["n"],
                   rho_total=tot_r["rho_spearman"], p_total=tot_r["p_spearman"],
                   rho_f_dross_dev=fdross_r["rho_spearman"], rho_f_polvo_dev=fpolvo_r["rho_spearman"],
                   rho_recuperacion_dev=recu_r["rho_spearman"])
        ols_dev = ols_df[(ols_df["id"] == cid) & (ols_df["muestra"] == "dev")]
        ols_lb = ols_df[(ols_df["id"] == cid) & (ols_df["muestra"] == "lockbox")]
        ols_tot = ols_df[(ols_df["id"] == cid) & (ols_df["muestra"] == "total")]
        row["coef_ols_dev"] = ols_dev["coef"].iloc[0] if len(ols_dev) else np.nan
        row["p_ols_dev"] = ols_dev["p"].iloc[0] if len(ols_dev) else np.nan
        row["coef_ols_lockbox"] = ols_lb["coef"].iloc[0] if len(ols_lb) else np.nan
        row["p_ols_lockbox"] = ols_lb["p"].iloc[0] if len(ols_lb) else np.nan
        row["coef_ols_total"] = ols_tot["coef"].iloc[0] if len(ols_tot) else np.nan
        row["p_ols_total"] = ols_tot["p"].iloc[0] if len(ols_tot) else np.nan
        if cid in terc_df.index:
            row.update(rho_tercio1=terc_df.loc[cid, "rho_tercio1"], rho_tercio2=terc_df.loc[cid, "rho_tercio2"],
                       rho_tercio3=terc_df.loc[cid, "rho_tercio3"], n_tercios_pos=terc_df.loc[cid, "n_tercios_pos"])
        if cid in mc_df.index:
            row.update(rho_mc_media=mc_df.loc[cid, "rho_mc_media"], rho_mc_sd=mc_df.loc[cid, "rho_mc_sd"],
                       frac_rho_mc_positivo=mc_df.loc[cid, "frac_rho_positivo"])
        row["dir_ok"] = dir_ok_map[cid]
        rank_rows.append(row)
    rank_df = pd.DataFrame(rank_rows).sort_values(["fase", "rho_dev"], ascending=[True, False])
    rank_df.to_csv(HERE / "st_09_ranking.csv", index=False, encoding="utf-8")
    log(f"[5] st_09_ranking.csv escrito ({len(rank_df)} filas).")

    # -------------------------------------------------------------------
    # 9) figuras
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharex=True)
    for ax, fase in zip(axes, FASES):
        sub_dev = rank_df[rank_df["fase"] == fase].sort_values("rho_dev", ascending=True)
        ids_order = sub_dev["id"].tolist()
        y = np.arange(len(ids_order))
        colors = ["tab:blue" if s is True else "lightsteelblue" for s in sub_dev["p_bh_dev"] < 0.05]
        ax.barh(y + 0.2, sub_dev["rho_dev"], height=0.35, color=colors, label="dev")
        lb_vals = sub_dev.set_index("id")["rho_lockbox"]
        ax.barh(y - 0.2, lb_vals, height=0.35, color="tab:orange", alpha=0.6, label="lockbox")
        ax.set_yticks(y)
        ax.set_yticklabels(ids_order, fontsize=6.5)
        ax.axvline(0, color="grey", lw=0.7)
        ax.set_title(f"{fase} (azul oscuro = p_bh_dev<0.05)")
        ax.set_xlabel("rho Spearman vs rendimiento_proxy_batch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "st_09_rho_barras.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, fase in zip(axes, FASES):
        top8 = rank_df[(rank_df["fase"] == fase) & (rank_df["agg"] != "batch_especial")].sort_values("rho_dev", ascending=False).head(8)
        data = [rho_mat[:, id_pos[cid]][np.isfinite(rho_mat[:, id_pos[cid]])] for cid in top8["id"]]
        ax.boxplot(data, tick_labels=list(top8["id"]), showmeans=True)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.set_title(fase)
        ax.set_ylabel("rho Spearman (MC, ruido trazador)")
    fig.suptitle(f"Sensibilidad al ruido de ensayo: rho bajo {N_MC} perturbaciones (top 8 por fase)")
    fig.tight_layout()
    fig.savefig(FIGS / "st_09_boxplot_ruido.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    bloque_orden = ["rho_tercio1", "rho_tercio2", "rho_tercio3", "rho_lockbox"]
    for ax, fase in zip(axes, FASES):
        top6 = rank_df[rank_df["fase"] == fase].sort_values("rho_dev", ascending=False).head(6)
        for _, rr in top6.iterrows():
            ys = [rr[b] for b in bloque_orden]
            ax.plot(["t1", "t2", "t3", "lockbox"], ys, marker="o", label=rr["id"])
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(fase)
        ax.legend(fontsize=6.5, loc="best")
        ax.set_ylabel("rho Spearman vs KPI")
    fig.suptitle("Estabilidad temporal: rho por tercio de DEV + lockbox (6 mejores por fase)")
    fig.tight_layout()
    fig.savefig(FIGS / "st_09_estabilidad_temporal.png", dpi=130)
    plt.close(fig)
    log("[6] figuras escritas: st_09_rho_barras.png, st_09_boxplot_ruido.png, st_09_estabilidad_temporal.png")

    # -------------------------------------------------------------------
    # 10) resumen final
    # -------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("RESUMEN FINAL")
    log("=" * 78)
    for fase in FASES:
        log(f"\n-- top-8 {fase} (rho dev) --")
        top8 = rank_df[rank_df["fase"] == fase].sort_values("rho_dev", ascending=False).head(8)
        for _, rr in top8.iterrows():
            log(f"  {rr['id']:<28} rho_dev={rr['rho_dev']:.4f} p_bh_dev={rr['p_bh_dev']:.4g} "
                f"rho_lockbox={rr['rho_lockbox']:.4f} coef_ols_dev={rr['coef_ols_dev']:.4f} p_ols_dev={rr['p_ols_dev']:.4g} "
                f"tercios(+/{rr['n_tercios_pos'] if pd.notna(rr['n_tercios_pos']) else '-'}) "
                f"rho_mc={rr['rho_mc_media']:.4f}+-{rr['rho_mc_sd']:.4f} familia={rr['familia']}")

    fuertes = rank_df[rank_df["rho_dev"].abs() > 0.5]
    log(f"\nCandidatos con |rho_dev| > 0.5: {len(fuertes)}")
    for _, rr in fuertes.iterrows():
        log(f"  {rr['id']}: rho_dev={rr['rho_dev']:.4f} -- revisar que no sea identidad con KPI (formula no usa "
            f"sn_en_*_batch_t ni rendimiento_proxy_batch por construccion; ver st_09_registro.csv).")

    debiles = rank_df[rank_df["rho_dev"].abs() < 0.2]
    log(f"\nHonestidad: {len(debiles)}/{len(rank_df)} candidatos tienen |rho_dev| < 0.2. "
        f"Maximo |rho_dev|: {rank_df['rho_dev'].abs().max():.4f} ({rank_df.loc[rank_df['rho_dev'].abs().idxmax(), 'id']}).")

    log(f"\nTiempo total: {(time.time()-t0)/60:.1f} min")
    with open(HERE / "st_09_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
