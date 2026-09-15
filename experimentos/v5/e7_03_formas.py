"""E7-03 -- Formas funcionales de las palancas y optimos interiores (teoria vs datos).

Iteracion 7 (prescriptor v5). Ver experimentos/v5/ITERACION_7_diseno.md (tabla teorica de
formas esperadas al final del documento) y experimentos/v5/v5_lib.py (libreria compartida,
usada tal cual: cargar_df, base_fase, evaluar_plm, es_segura, SIGNO_TEORICO_V5).

Enfoque adaptado de experimentos/16_formas_pd_teoria.py (PD/ALE/SHAP vs teoria) pero con
targets v5 (componentes log del SDI en Reduccion, d_lnIRF/ln_sn_dep/dT en Fusion) y con dos
piezas nuevas que el experimento 16 no tenia: (a) test de Robinson de terminos cuadraticos
cross-fitted dentro del PLM v4 (Sec. 2) y (b) theta por tramo de avance/orden de escalon
(Sec. 3).

Preguntas que responde (condicion de aceptacion: "no asumir monotonia; los optimos interiores
deben basarse en teoria piro"):
  1. Existen terminos cuadraticos significativos en las palancas del PLM? Donde cae el vertice?
  2. La selectividad marginal FeO/Sn del carbon cambia por tramo de avance, como predice la
     teoria (hallazgo exp15)? La reproduce el termino de interaccion v5_C_x_avance del PLM global?
  3. Las formas PD/ALE del HGB (no lineal, sin restricciones) concuerdan con la tabla teorica?
  4. Donde cae el optimo de J = ln_sn_dep + 10*ln_feo_ret en funcion del carbon/GN, por nivel
     de avance -- es interior o de frontera?

Salidas (experimentos/v5/):
  e7_03_cuadraticos.csv, e7_03_theta_por_tramo.csv, e7_03_pd_curvas.csv,
  e7_03_formas_vs_teoria.csv, e7_03_curva_J.csv, figs/e7_03_curva_J.png,
  figs/e7_03_pd_ale__<target>__<variable>.png (paneles PD|ALE|SHAP), e7_03_resultados.md.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos" / "v5"))
import v5_lib as v5  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import partial_dependence  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
import shap  # noqa: E402

SALIDA = Path(__file__).resolve().parent
FIGS = SALIDA / "figs"
FIGS.mkdir(exist_ok=True, parents=True)
T0 = time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)


def t():
    return f"[{time.time() - T0:6.1f}s]"


def log(*a):
    print(t(), *a, flush=True)


# =============================================================================
# 0) Datos
# =============================================================================
log("cargando dataset (cache v3 + v4 + b6 + v5)...")
DF = v5.cargar_df()
dR = v5.base_fase(DF, "Reducción")
dF = v5.base_fase(DF, "Fusión")
log(f"dR {dR.shape}, dF {dF.shape}")

PALANCAS_R_SN = ["v5_dosis_C_sn", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
PALANCAS_R_FEO = ["v5_exceso_C_pos", "v5_C_x_avance", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
PALANCAS_F = ["relacion_C_Sn_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]

# =============================================================================
# 1) Helpers compartidos: cuadraticos cross-fitted (Robinson) + bootstrap del vertice
# =============================================================================


def agregar_cuadratico(df_fase: pd.DataFrame, estado: list, palancas: list, target: str, sufijo: str = "_c2"):
    """Anade a^2 (centrada en la media DEV de a) para cada palanca de `palancas`. Registra las
    columnas nuevas como DERIVED en v5.CLASIFICACION_V5 (no se edita v5_lib.py; se muta el dict
    en memoria desde este script, tal como indica el enunciado)."""
    dev, _ = mp.split_dev_lockbox(df_fase)
    need = list(dict.fromkeys(estado + palancas + [target, "Batch"]))
    d_dev = df_fase.loc[df_fase["Batch"].isin(dev), need].dropna()
    medias = d_dev[palancas].mean()
    df2 = df_fase.copy()
    cuad_cols = []
    for a in palancas:
        col = f"{a}{sufijo}"
        df2[col] = (df2[a] - medias[a]) ** 2
        cuad_cols.append(col)
        v5.CLASIFICACION_V5[col] = "DERIVED (A_t x S_prev)"
    return df2, cuad_cols, medias


def residualizar(d: pd.DataFrame, estado: list, palancas: list, target: str, n_splits: int = 5, seed: int = 42):
    """Replica el cross-fitting de mp4.ModeloPLM.fit para obtener y_res, A_res (sin el ultimo
    paso de OLS): necesario para bootstrapear conjuntamente (theta_1, theta_2) y de ahi el
    vertice, que ModeloPLM no expone (solo guarda percentiles marginales de theta)."""
    S, A, y = estado, palancas, target
    cols = [y] + A
    grupos = d["Batch"].to_numpy()
    oof = pd.DataFrame(np.nan, index=d.index, columns=cols)
    for tr, te in GroupKFold(n_splits).split(d[S], d[y], grupos):
        for c in cols:
            mdl = mp4._hgb_nuisance(seed).fit(d[S].iloc[tr], d[c].iloc[tr])
            oof.iloc[te, oof.columns.get_loc(c)] = mdl.predict(d[S].iloc[te])
    y_res = d[y].to_numpy() - oof[y].to_numpy()
    A_res = np.column_stack([d[a].to_numpy() - oof[a].to_numpy() for a in A])
    return y_res, A_res, grupos


def bootstrap_vertice(y_res, A_res, grupos, i1: int, i2: int, mean_a: float, n_boot: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(grupos)
    idx_by = {b: np.where(grupos == b)[0] for b in uniq}
    vs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
        theta = np.linalg.lstsq(A_res[idx], y_res[idx], rcond=None)[0]
        if abs(theta[i2]) > 1e-9:
            vs.append(mean_a - theta[i1] / (2 * theta[i2]))
    return np.array(vs)


# =============================================================================
# 2) Terminos cuadraticos cross-fitted (test de Robinson) por target
# =============================================================================
log("\n===== 2) cuadraticos cross-fitted =====")
ESPECIFICACIONES_CUAD = [
    dict(fase="Reducción", df=dR, target="v5_ln_sn_dep", estado=v5.ESTADO_R_CURADO, palancas=PALANCAS_R_SN),
    dict(fase="Reducción", df=dR, target="v5_ln_feo_ret", estado=v5.ESTADO_R_CURADO, palancas=PALANCAS_R_FEO),
    dict(fase="Fusión", df=dF, target="v5_d_lnirf", estado=v5.ESTADO_F_CURADO, palancas=PALANCAS_F),
    dict(fase="Fusión", df=dF, target="v5_ln_sn_dep", estado=v5.ESTADO_F_CURADO, palancas=PALANCAS_F),
    dict(fase="Fusión", df=dF, target="d_temperatura_horno_celsius", estado=v5.ESTADO_F_CURADO, palancas=PALANCAS_F),
]

filas_cuad = []
MODELOS_J = {}  # target Reduccion -> dict(modelo, tipo, cuad_cols, medias, palancas, estado) para el paso 5

for spec in ESPECIFICACIONES_CUAD:
    fase, df_fase, target, estado, palancas = spec["fase"], spec["df"], spec["target"], spec["estado"], spec["palancas"]
    log(f"  {fase} / {target}: palancas base = {palancas}")
    df2, cuad_cols, medias = agregar_cuadratico(df_fase, estado, palancas, target)
    signo = v5.SIGNO_TEORICO_V5.get(target, {})

    met_lin, tabla_lin, _ = v5.evaluar_plm(df2, estado, palancas, target, n_boot=300, signo_teorico=signo)
    met_quad, tabla_quad, _ = v5.evaluar_plm(df2, estado, palancas + cuad_cols, target, n_boot=300, signo_teorico=signo)
    tabla_lin, tabla_quad = tabla_lin.set_index("palanca"), tabla_quad.set_index("palanca")
    log(f"    R2_oof lineal={met_lin['r2_oof']:.4f} (lockbox {met_lin['r2_lockbox']:.4f})  "
        f"lineal+cuad={met_quad['r2_oof']:.4f} (lockbox {met_quad['r2_lockbox']:.4f})  n_dev={met_lin['n_dev']}")

    dev, _ = mp.split_dev_lockbox(df2)
    need = list(dict.fromkeys(estado + palancas + cuad_cols + [target, "Batch"]))
    d_dev = df2.loc[df2["Batch"].isin(dev), need].dropna().reset_index(drop=True)
    p5_p95 = {a: (float(d_dev[a].quantile(0.05)), float(d_dev[a].quantile(0.95))) for a in palancas}

    palancas_full = palancas + cuad_cols
    for a in palancas:
        a2 = f"{a}_c2"
        th1, th1_lo, th1_hi = tabla_quad.loc[a, ["theta_unidad", "ci_lo", "ci_hi"]]
        th2, th2_lo, th2_hi = tabla_quad.loc[a2, ["theta_unidad", "ci_lo", "ci_hi"]]
        sig2 = bool(tabla_quad.loc[a2, "significativo"])
        vertice_pt = vertice_lo = vertice_hi = np.nan
        dentro_rango = np.nan
        if sig2:
            i1, i2 = palancas_full.index(a), palancas_full.index(a2)
            y_res, A_res, grupos = residualizar(d_dev, estado, palancas_full, target, seed=42)
            theta_full = np.linalg.lstsq(A_res, y_res, rcond=None)[0]
            vertice_pt = float(medias[a] - theta_full[i1] / (2 * theta_full[i2]))
            vs = bootstrap_vertice(y_res, A_res, grupos, i1, i2, float(medias[a]), n_boot=300, seed=42)
            if len(vs):
                vertice_lo, vertice_hi = float(np.nanpercentile(vs, 2.5)), float(np.nanpercentile(vs, 97.5))
            dentro_rango = bool(p5_p95[a][0] <= vertice_pt <= p5_p95[a][1])
            log(f"    {a}: cuadratico SIGNIFICATIVO (theta2={th2:.4g} [{th2_lo:.4g},{th2_hi:.4g}]) "
                f"vertice={vertice_pt:.4g} IC[{vertice_lo:.4g},{vertice_hi:.4g}] "
                f"P5-P95={p5_p95[a]} dentro={dentro_rango}")
        filas_cuad.append(dict(
            fase=fase, target=target, palanca=a,
            theta_lineal=float(tabla_lin.loc[a, "theta_unidad"]), ci_lo_lineal=float(tabla_lin.loc[a, "ci_lo"]),
            ci_hi_lineal=float(tabla_lin.loc[a, "ci_hi"]),
            theta_lineal_en_modelo_cuad=float(th1), ci_lo_lin_cuad=float(th1_lo), ci_hi_lin_cuad=float(th1_hi),
            theta_cuadratico=float(th2), ci_lo_cuad=float(th2_lo), ci_hi_cuad=float(th2_hi),
            cuadratico_significativo=sig2,
            r2_oof_lineal=met_lin["r2_oof"], r2_oof_lineal_cuad=met_quad["r2_oof"],
            r2_lockbox_lineal=met_lin["r2_lockbox"], r2_lockbox_lineal_cuad=met_quad["r2_lockbox"],
            media_dev=float(medias[a]), p5_dev=p5_p95[a][0], p95_dev=p5_p95[a][1],
            vertice=vertice_pt, vertice_ci_lo=vertice_lo, vertice_ci_hi=vertice_hi,
            vertice_dentro_p5_p95=dentro_rango, n_dev=met_lin["n_dev"],
        ))

    if fase == "Reducción":
        cualquier_sig = tabla_quad.loc[cuad_cols, "significativo"].any()
        if cualquier_sig:
            modelo_final = mp4.ModeloPLM(estado, palancas_full, target).fit(d_dev, n_boot=0, seed=42)
            MODELOS_J[target] = dict(modelo=modelo_final, tipo="lineal+cuadratico", cuad_cols=cuad_cols,
                                      medias=medias, palancas=palancas, estado=estado)
        else:
            modelo_final = mp4.ModeloPLM(estado, palancas, target).fit(d_dev.drop(columns=cuad_cols) if False else d_dev, n_boot=0, seed=42)
            MODELOS_J[target] = dict(modelo=modelo_final, tipo="lineal", cuad_cols=[], medias=medias,
                                      palancas=palancas, estado=estado)

cuad_df = pd.DataFrame(filas_cuad)
cuad_df.to_csv(SALIDA / "e7_03_cuadraticos.csv", index=False)
log(f"e7_03_cuadraticos.csv -> {cuad_df.shape}; significativos: "
    f"{cuad_df.loc[cuad_df.cuadratico_significativo, ['fase', 'target', 'palanca']].to_dict('records')}")

# =============================================================================
# 3) Theta por tramo de avance / orden de escalon (Reduccion, parametrizacion P3 = dosis/sobrante)
# =============================================================================
log("\n===== 3) theta por tramo de avance / orden de escalon =====")
BINS_AVANCE = [-np.inf, 0.84, 0.96, 0.975, np.inf]
LABELS_AVANCE = ["<0.84", "0.84-0.96", "0.96-0.975", ">0.975"]
dR = dR.copy()
dR["tramo_avance"] = pd.cut(dR["avance_reduccion_sn_prev"], bins=BINS_AVANCE, labels=LABELS_AVANCE)
ORDENES = sorted(pd.unique(dR["orden_escalon_fase"].dropna()))[:4]

TARGETS_TRAMO = {"v5_ln_sn_dep": PALANCAS_R_SN, "v5_ln_feo_ret": PALANCAS_R_FEO}
CARBON_POR_TARGET = {"v5_ln_sn_dep": "v5_dosis_C_sn", "v5_ln_feo_ret": "v5_exceso_C_pos"}

filas_tramo = []


def plm_tramo(sub: pd.DataFrame, tipo_tramo: str, tramo_lbl, target: str, palancas: list, n_boot: int = 150):
    dev, _ = mp.split_dev_lockbox(sub)
    n_batches = sub.loc[sub["Batch"].isin(dev), "Batch"].nunique()
    n_splits = int(min(5, max(2, n_batches)))
    need = list(dict.fromkeys(v5.ESTADO_R_CURADO + palancas + [target, "Batch"]))
    if sub.loc[sub["Batch"].isin(dev), need].dropna().shape[0] < 40:
        log(f"    [omitido] {tipo_tramo}={tramo_lbl} target={target}: muestra insuficiente")
        return None
    try:
        met, tabla, _ = v5.evaluar_plm(sub, v5.ESTADO_R_CURADO, palancas, target, n_splits=n_splits, n_boot=n_boot,
                                       signo_teorico=v5.SIGNO_TEORICO_V5.get(target, {}))
    except Exception as exc:  # noqa: BLE001
        log(f"    [error] {tipo_tramo}={tramo_lbl} target={target}: {exc}")
        return None
    tabla = tabla.reset_index()
    tabla.insert(0, "tramo", tramo_lbl)
    tabla.insert(0, "tipo_tramo", tipo_tramo)
    tabla["r2_oof"] = met["r2_oof"]
    tabla["n_dev"] = met["n_dev"]
    tabla["n_lockbox"] = met["n_lockbox"]
    return tabla


for target, palancas in TARGETS_TRAMO.items():
    for lbl in LABELS_AVANCE:
        sub = dR[dR["tramo_avance"] == lbl]
        tabla = plm_tramo(sub, "avance", lbl, target, palancas)
        if tabla is not None:
            filas_tramo.append(tabla)
    for o in ORDENES:
        sub = dR[dR["orden_escalon_fase"] == o]
        tabla = plm_tramo(sub, "orden_escalon_fase", int(o), target, palancas)
        if tabla is not None:
            filas_tramo.append(tabla)

tramo_df = pd.concat(filas_tramo, ignore_index=True) if filas_tramo else pd.DataFrame()
if "target" not in tramo_df.columns: tramo_df.insert(2, "target", "")
# el insert de 'target' arriba crea la columna vacia; se rellena abajo con el valor real por bloque
# (evaluar_plm no incluye 'target' en tabla_theta; se agrega al concatenar cada bloque)
filas_tramo2 = []
idx = 0
for target, palancas in TARGETS_TRAMO.items():
    for lbl in LABELS_AVANCE:
        sub = dR[dR["tramo_avance"] == lbl]
        tabla = plm_tramo(sub, "avance", lbl, target, palancas, n_boot=0)  # ya calculado arriba; se recalcula solo el target aqui abajo
        break
    break
# (nota: el bloque anterior de doble calculo se evita: reconstruimos directamente con 'target' desde el inicio)

filas_tramo = []
for target, palancas in TARGETS_TRAMO.items():
    for lbl in LABELS_AVANCE:
        sub = dR[dR["tramo_avance"] == lbl]
        tabla = plm_tramo(sub, "avance", lbl, target, palancas)
        if tabla is not None:
            tabla = tabla.drop(columns=[c for c in ["target"] if c in tabla.columns]); tabla.insert(2, "target", target)
            filas_tramo.append(tabla)
    for o in ORDENES:
        sub = dR[dR["orden_escalon_fase"] == o]
        tabla = plm_tramo(sub, "orden_escalon_fase", int(o), target, palancas)
        if tabla is not None:
            tabla = tabla.drop(columns=[c for c in ["target"] if c in tabla.columns]); tabla.insert(2, "target", target)
            filas_tramo.append(tabla)

tramo_df = pd.concat(filas_tramo, ignore_index=True) if filas_tramo else pd.DataFrame()
tramo_df.to_csv(SALIDA / "e7_03_theta_por_tramo.csv", index=False)
log(f"e7_03_theta_por_tramo.csv -> {tramo_df.shape}")

# selectividad marginal FeO/Sn del carbon por tramo (en unidades por 1sd, parametrizacion P3:
# dosis C/Sn para sn_dep, exceso de C sobre la demanda para feo_ret -- no son la misma unidad
# fisica que en exp15 (kg C/min); se reporta la tendencia, no el nivel absoluto)
log("\n  -- selectividad marginal (P3, en unidades por 1sd) por tramo --")
sel_filas = []
for tipo_tramo in ["avance", "orden_escalon_fase"]:
    tramos_u = tramo_df.loc[tramo_df.tipo_tramo == tipo_tramo, "tramo"].unique()
    for tr in tramos_u:
        row_sn = tramo_df[(tramo_df.tipo_tramo == tipo_tramo) & (tramo_df.tramo == tr) &
                          (tramo_df.target == "v5_ln_sn_dep") & (tramo_df.palanca == CARBON_POR_TARGET["v5_ln_sn_dep"])]
        row_feo = tramo_df[(tramo_df.tipo_tramo == tipo_tramo) & (tramo_df.tramo == tr) &
                           (tramo_df.target == "v5_ln_feo_ret") & (tramo_df.palanca == CARBON_POR_TARGET["v5_ln_feo_ret"])]
        if len(row_sn) and len(row_feo):
            th_sn, th_feo = float(row_sn["theta_1sd"].iloc[0]), float(row_feo["theta_1sd"].iloc[0])
            sel = -th_feo / th_sn if abs(th_sn) > 1e-9 else np.nan
            sel_filas.append(dict(tipo_tramo=tipo_tramo, tramo=tr, theta_1sd_carbon_sn_dep=th_sn,
                                   theta_1sd_carbon_feo_ret=th_feo, selectividad_feo_sn=sel))
sel_df = pd.DataFrame(sel_filas)
log(sel_df.to_string(index=False))
sel_df.to_csv(SALIDA / "e7_03_selectividad_por_tramo.csv", index=False)

# contraste: el termino de interaccion v5_C_x_avance en el modelo GLOBAL de ln_feo_ret (paso 2)
theta_cxav_global = cuad_df.loc[(cuad_df.target == "v5_ln_feo_ret") & (cuad_df.palanca == "v5_C_x_avance"),
                                ["theta_lineal", "ci_lo_lineal", "ci_hi_lineal"]]
log(f"  v5_C_x_avance en el PLM global de ln_feo_ret: theta={theta_cxav_global.to_dict('records')}")

# =============================================================================
# 4) PD / ALE / SHAP del HGB estado+palancas, por target, vs teoria
# =============================================================================
log("\n===== 4) PD / ALE / SHAP vs teoria =====")

PALANCAS_HGB_R = list(dict.fromkeys(v5.PALANCAS_PRIMITIVAS + ["v5_dosis_C_sn", "v5_exceso_C_pos", "v5_C_x_avance",
                                                              "Cx_sn_v4", "Cx_av_v4", "exceso_o2_combustion_pct",
                                                              "v5_gn_esp_nm3_t"]))
PALANCAS_HGB_F = list(dict.fromkeys(v5.PALANCAS_PRIMITIVAS + ["relacion_C_Sn_carga", "exceso_o2_combustion_pct"]))
ESTADOS_CLAVE = ["temperatura_horno_celsius_prev", "basicidad_B2_prev", "posicion_vertical_lanza_mm_prev",
                 "avance_reduccion_sn_prev", "ley_feo_escoria_pct_prev"]

# --- tabla teorica local: combina v5.SIGNO_TEORICO_V5 (signos) con la tabla teorica de
# ITERACION_7_diseno.md (formas concava/optimo interior/valle que un signo no captura)
def _signo_a_tipo(s):
    return {1: "mono+", -1: "mono-", 0: "sin_restriccion"}.get(s, "no_cubierta")


TEORIA_V5: dict[str, dict[str, tuple[str, str]]] = {}
for tgt in ["v5_ln_sn_dep", "v5_ln_feo_ret", "v5_d_lnirf", "d_temperatura_horno_celsius"]:
    TEORIA_V5[tgt] = {}
    for feat, s in v5.SIGNO_TEORICO_V5.get(tgt, {}).items():
        TEORIA_V5[tgt][feat] = (f"signo esperado {s:+d} (v5.SIGNO_TEORICO_V5)", _signo_a_tipo(s))

# overrides Reduccion (tabla teorica del diseno, filas 1-9)
TEORIA_V5["v5_ln_sn_dep"].update({
    "v5_dosis_C_sn": ("+ (SnO2+2C; 1er orden en C mientras hay SnOx), concava/saturante", "mono+_sat"),
    "v5_exceso_C_pos": ("~0 (exceso ya no limita al Sn)", "sin_restriccion"),
    "tasa_gn_nm3_min": ("+ (calor + CO/H2 reductor)", "mono+"),
    "exceso_o2_combustion_pct": ("- (atmosfera oxidante frena la reduccion / reoxida Sn)", "mono-"),
    "tasa_aire_nm3_min": ("~0 / - (enfria, diluye)", "mono-_debil"),
    "temperatura_horno_celsius_prev": ("+ con meseta (cinetica, viscosidad; ventana de volatilizacion/refractario)", "mono+_sat"),
    "basicidad_B2_prev": ("no monotona, optimo ~1.4 (viscosidad/liquidus)", "optimo_interior"),
    "posicion_vertical_lanza_mm_prev": ("no monotona, valle 3300-4200mm", "valle"),
    "avance_reduccion_sn_prev": ("+ en escala log (queda poco Sn, cada kg pesa mas)", "mono+"),
    "ley_feo_escoria_pct_prev": ("no cubierta explicitamente", "no_cubierta"),
})
TEORIA_V5["v5_ln_feo_ret"].update({
    "v5_dosis_C_sn": ("aprox 0 con SnOx abundante (marginal, sin condicionar por avance)", "sin_restriccion"),
    "v5_exceso_C_pos": ("- (reductor disponible para FeO+C->Fe+CO)", "mono-"),
    "v5_C_x_avance": ("- (crece con el avance: selectividad FeO/Sn del carbon, exp15)", "mono-"),
    "tasa_gn_nm3_min": ("- (reductor no selectivo; exp19: GN total de Reduccion sube dross)", "mono-"),
    "exceso_o2_combustion_pct": ("+ (retiene FeO; lanza oxidante)", "mono+"),
    "tasa_aire_nm3_min": ("+ debil (aporta O2)", "mono+_debil"),
    "temperatura_horno_celsius_prev": ("- (mas reduccion de FeO a T alta)", "mono-"),
    "basicidad_B2_prev": ("aprox 0", "sin_restriccion"),
    "posicion_vertical_lanza_mm_prev": ("no monotona, sin confirmar", "valle"),
    "avance_reduccion_sn_prev": ("- (menos SnOx que compita)", "mono-"),
    "ley_feo_escoria_pct_prev": ("no cubierta explicitamente", "no_cubierta"),
})
# Fusion: sin tabla detallada -> se usa el signo de v5.SIGNO_TEORICO_V5 (ya cargado arriba);
# override puntual: relacion_C_Sn_carga sobre d_lnirf se declara 0 explicitamente en v5_lib
TEORIA_V5["v5_ln_sn_dep"].setdefault("relacion_C_Sn_carga", ("+ (extraer Sn en Fusion; efecto propio fragil)", "mono+_debil"))
for tgt in ["v5_d_lnirf", "d_temperatura_horno_celsius"]:
    TEORIA_V5[tgt].setdefault("relacion_C_Sn_carga", ("sin prediccion fuerte (interaccion 2do orden)", "no_cubierta"))


def clasificar_forma(x: np.ndarray, y: np.ndarray, sd_target: float):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    orden = np.argsort(x)
    x, y = x[orden], y[orden]
    rango = float(np.nanmax(y) - np.nanmin(y))
    rango_rel_sd = rango / sd_target if sd_target > 1e-9 else np.nan
    if not np.isfinite(rango_rel_sd) or rango_rel_sd < 0.05:
        return "plana", float(x[len(x) // 2]), rango_rel_sd
    # regla explicita: pendiente por tercios del grid (signo de la variacion en cada tercio)
    n = len(x)
    k = max(1, n // 3)
    d1 = y[min(k, n - 1)] - y[0]
    d2 = y[min(2 * k, n - 1)] - y[min(k, n - 1)]
    d3 = y[-1] - y[min(2 * k, n - 1)]
    signos = [np.sign(d) for d in (d1, d2, d3) if abs(d) > 0.02 * rango]
    if signos and all(s == signos[0] for s in signos):
        forma = "monótona +" if signos[0] > 0 else "monótona -"
        x_ext = float(x[int(np.argmax(y))]) if signos[0] > 0 else float(x[int(np.argmin(y))])
        return forma, x_ext, rango_rel_sd
    rho = spearmanr(x, y).statistic
    if np.isfinite(rho) and abs(rho) >= 0.9:
        forma = "monótona +" if rho > 0 else "monótona -"
        x_ext = float(x[int(np.argmax(y))]) if rho > 0 else float(x[int(np.argmin(y))])
        return forma, x_ext, rango_rel_sd
    candidatos = []
    for kind, idx in (("máximo", int(np.argmax(y))), ("mínimo", int(np.argmin(y)))):
        pos_frac = idx / (n - 1)
        if 0.15 <= pos_frac <= 0.85:
            left_diff, right_diff = abs(y[idx] - y[0]), abs(y[idx] - y[-1])
            if left_diff > 0.15 * rango and right_diff > 0.15 * rango:
                candidatos.append((kind, idx, min(left_diff, right_diff)))
    if candidatos:
        candidatos.sort(key=lambda tt: -tt[2])
        kind, idx, _ = candidatos[0]
        return ("óptimo interior" if kind == "máximo" else "valle"), float(x[idx]), rango_rel_sd
    idx_ext = int(np.argmax(np.abs(y - y.mean())))
    return "no monótona compleja", float(x[idx_ext]), rango_rel_sd


def evaluar_concuerda(tipo_teorico: str, forma: str):
    if tipo_teorico in ("no_cubierta", "sin_restriccion"):
        return "n/a", "sin prediccion falsable"
    mono_pos, mono_neg = forma == "monótona +", forma == "monótona -"
    opt_max, opt_min, plana = forma == "óptimo interior", forma == "valle", forma == "plana"
    if tipo_teorico in ("mono+", "mono+_sat", "mono+_debil"):
        if mono_pos or (tipo_teorico == "mono+_sat" and opt_max):
            return "sí", "forma + como se esperaba"
        if plana:
            return "no" if tipo_teorico != "mono+_debil" else "parcial", "sin efecto detectado"
        if mono_neg or opt_min:
            return "no", "signo opuesto"
        return "parcial", "forma compleja"
    if tipo_teorico == "mono-":
        if mono_neg:
            return "sí", "forma - como se esperaba"
        if plana:
            return "no", "sin efecto detectado pese a teoria -"
        if mono_pos or opt_max:
            return "no", "signo opuesto"
        return "parcial", "forma compleja"
    if tipo_teorico == "mono-_debil":
        if mono_neg or opt_min:
            return "sí", "efecto - presente (esperado debil)"
        if plana:
            return "parcial", "nulo, consistente con debil"
        return "no", "signo opuesto"
    if tipo_teorico == "optimo_interior":
        if opt_max:
            return "sí", "optimo interior (maximo) detectado"
        if opt_min:
            return "parcial", "extremo interior pero de signo opuesto"
        return "no" if not plana else "no", "no se detecto optimo interior"
    if tipo_teorico == "valle":
        if opt_min:
            return "sí", "valle detectado"
        if opt_max:
            return "parcial", "extremo interior pero maximo, no valle"
        return "no" if not plana else "no", "no se detecto valle"
    return "n/a", ""


def calcular_pd(model, d, feats, feature, n_grid=25):
    try:
        res = partial_dependence(model, d[feats], [feature], kind="average", grid_resolution=n_grid,
                                 percentiles=(0.02, 0.98))
    except ValueError:
        return None
    x, y = np.asarray(res["grid_values"][0], float), np.asarray(res["average"][0], float)
    return (x, y) if len(x) >= 3 else None


def calcular_ale(model, d, feats, feature, n_bins=25):
    x_full = d[feature].to_numpy(dtype=float)
    qs = np.unique(np.quantile(x_full, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 4:
        return None
    qs_ext = qs.copy()
    qs_ext[0] -= 1e-9
    qs_ext[-1] += 1e-9
    bin_idx = np.clip(np.digitize(x_full, qs_ext, right=True) - 1, 0, len(qs_ext) - 2)
    Xlo, Xhi = d[feats].copy(), d[feats].copy()
    Xlo[feature] = qs_ext[bin_idx]
    Xhi[feature] = qs_ext[bin_idx + 1]
    delta = model.predict(Xhi[feats]) - model.predict(Xlo[feats])
    nb = len(qs_ext) - 1
    sums, counts = np.zeros(nb), np.zeros(nb)
    np.add.at(sums, bin_idx, delta)
    np.add.at(counts, bin_idx, 1)
    bin_mean = np.divide(sums, counts, out=np.zeros(nb), where=counts > 0)
    edges_val = np.concatenate([[0.0], np.cumsum(bin_mean)])
    mid_val = (edges_val[:-1] + edges_val[1:]) / 2.0
    w_mean = np.average(mid_val, weights=counts) if counts.sum() > 0 else 0.0
    mid_x = (qs[:-1] + qs[1:]) / 2.0
    return mid_x, mid_val - w_mean


ESPECIFICACIONES_HGB = [
    dict(fase="Reducción", df=dR, target="v5_ln_sn_dep", estado=v5.ESTADO_R_CURADO, palancas=PALANCAS_HGB_R),
    dict(fase="Reducción", df=dR, target="v5_ln_feo_ret", estado=v5.ESTADO_R_CURADO, palancas=PALANCAS_HGB_R),
    dict(fase="Fusión", df=dF, target="v5_d_lnirf", estado=v5.ESTADO_F_CURADO, palancas=PALANCAS_HGB_F),
    dict(fase="Fusión", df=dF, target="v5_ln_sn_dep", estado=v5.ESTADO_F_CURADO, palancas=PALANCAS_HGB_F),
    dict(fase="Fusión", df=dF, target="d_temperatura_horno_celsius", estado=v5.ESTADO_F_CURADO, palancas=PALANCAS_HGB_F),
]

filas_formas, filas_curvas, filas_shap = [], [], []

for spec in ESPECIFICACIONES_HGB:
    fase, df_fase, target, estado, palancas_hgb = spec["fase"], spec["df"], spec["target"], spec["estado"], spec["palancas"]
    feats = list(dict.fromkeys(estado + palancas_hgb))
    inseg = [f for f in feats if not v5.es_segura(f)]
    assert not inseg, f"features inseguras en HGB {target}: {inseg}"
    dev, _ = mp.split_dev_lockbox(df_fase)
    d = df_fase.loc[df_fase["Batch"].isin(dev), feats + [target]].dropna().reset_index(drop=True)
    log(f"  HGB {fase}/{target}: n={len(d)}, {len(feats)} features")
    model = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05, min_samples_leaf=20,
                                          random_state=42).fit(d[feats], d[target])
    sd_target = float(d[target].std())

    estados_disp = [e for e in ESTADOS_CLAVE if e in estado]
    variables = list(dict.fromkeys(palancas_hgb + estados_disp))

    explainer = shap.TreeExplainer(model)
    n_shap = min(800, len(d))
    muestra = d.sample(n_shap, random_state=0).reset_index(drop=True)
    shap_vals = explainer.shap_values(muestra[feats])

    for var in variables:
        pd_res = calcular_pd(model, d, feats, var)
        ale_res = calcular_ale(model, d, feats, var)
        if pd_res is not None:
            for xx, yy in zip(*pd_res):
                filas_curvas.append(dict(target=target, variable=var, fuente="pd", x=xx, pd=yy, ale=np.nan))
        if ale_res is not None:
            for xx, yy in zip(*ale_res):
                filas_curvas.append(dict(target=target, variable=var, fuente="ale", x=xx, pd=np.nan, ale=yy))
        base = ale_res if ale_res is not None else pd_res
        if base is None:
            continue
        forma, x_ext, rango_rel_sd = clasificar_forma(base[0], base[1], sd_target)
        rol = "palanca" if var in palancas_hgb else "estado"
        teoria_txt, tipo_teorico = TEORIA_V5.get(target, {}).get(var, ("no cubierta", "no_cubierta"))
        concuerda, com = evaluar_concuerda(tipo_teorico, forma)
        filas_formas.append(dict(fase=fase, target=target, variable=var, rol=rol, n=len(d),
                                  sd_target=round(sd_target, 4), rango_rel_sd=round(rango_rel_sd, 4)
                                  if np.isfinite(rango_rel_sd) else np.nan,
                                  forma=forma, x_extremo=round(x_ext, 4), teoria=teoria_txt,
                                  tipo_teorico=tipo_teorico, concuerda=concuerda, comentario=com))

        # panel PD | ALE
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        if pd_res is not None:
            axes[0].plot(pd_res[0], pd_res[1], "o-", ms=3, color="#1f77b4")
            axes[0].set_title("PD")
        if ale_res is not None:
            axes[1].plot(ale_res[0], ale_res[1], "o-", ms=3, color="#ff7f0e")
            axes[1].axhline(0, color="gray", lw=0.6)
            axes[1].set_title("ALE")
        for ax in axes:
            ax.set_xlabel(var, fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{target} :: {var}  [{forma}, {concuerda}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / f"e7_03_pd_ale__{target}__{var}.png", dpi=100)
        plt.close(fig)

    # --- SHAP: pendiente global + interaccion con avance (solo Reduccion, solo palancas)
    for var in palancas_hgb:
        j = feats.index(var)
        xv = muestra[var].to_numpy(dtype=float)
        sv = shap_vals[:, j]
        ok = np.isfinite(xv) & np.isfinite(sv)
        slope_global = float(np.polyfit(xv[ok], sv[ok], 1)[0]) if ok.sum() > 5 and np.std(xv[ok]) > 0 else np.nan
        slope_lo = slope_mid = slope_hi = np.nan
        if fase == "Reducción" and "avance_reduccion_sn_prev" in muestra.columns:
            av = muestra["avance_reduccion_sn_prev"].to_numpy(dtype=float)
            terciles = pd.qcut(av, 3, labels=False, duplicates="drop")
            for lab, name in zip([0, 1, 2], ["slope_lo", "slope_mid", "slope_hi"]):
                m = ok & (terciles == lab)
                if m.sum() > 5 and np.std(xv[m]) > 0:
                    val = float(np.polyfit(xv[m], sv[m], 1)[0])
                    if name == "slope_lo":
                        slope_lo = val
                    elif name == "slope_mid":
                        slope_mid = val
                    else:
                        slope_hi = val
        filas_shap.append(dict(target=target, variable=var, shap_slope_global=slope_global,
                                shap_slope_avance_bajo=slope_lo, shap_slope_avance_medio=slope_mid,
                                shap_slope_avance_alto=slope_hi))

formas_df = pd.DataFrame(filas_formas)
curvas_df = pd.DataFrame(filas_curvas)
shap_df = pd.DataFrame(filas_shap)
formas_df = formas_df.merge(shap_df, on=["target", "variable"], how="left")
curvas_df.to_csv(SALIDA / "e7_03_pd_curvas.csv", index=False)
formas_df.to_csv(SALIDA / "e7_03_formas_vs_teoria.csv", index=False)
log(f"e7_03_pd_curvas.csv -> {curvas_df.shape}; e7_03_formas_vs_teoria.csv -> {formas_df.shape}")
log("\n resumen concuerda por target:\n" + formas_df.groupby(["target", "concuerda"]).size().unstack(fill_value=0).to_string())

# =============================================================================
# 5) Curva de J = ln_sn_dep + 10*ln_feo_ret vs carbon/GN, por nivel de avance
# =============================================================================
log("\n===== 5) curva de J vs carbon / GN =====")
W = v5.W_SDI_ITER5


def predecir_componente(target: str, X: pd.DataFrame) -> np.ndarray:
    info = MODELOS_J[target]
    Xc = X.copy()
    if info["tipo"] == "lineal+cuadratico":
        for a in info["palancas"]:
            Xc[f"{a}_c2"] = (Xc[a] - info["medias"][a]) ** 2
    return info["modelo"].predict(Xc)


dev_R, _ = mp.split_dev_lockbox(dR)
need_J = list(dict.fromkeys(v5.ESTADO_R_CURADO + PALANCAS_R_SN + PALANCAS_R_FEO +
                            ["v5_sn_disp", "avance_reduccion_sn_prev", "tasa_feed_Carbon_kg_min", "Batch"]))
d_dev_R = dR.loc[dR["Batch"].isin(dev_R), need_J].dropna().reset_index(drop=True)

AVANCES_OBJETIVO = [0.75, 0.90, 0.965, 0.985]
base_rows = []
for av in AVANCES_OBJETIVO:
    idx = (d_dev_R["avance_reduccion_sn_prev"] - av).abs().idxmin()
    base_rows.append(d_dev_R.loc[idx])

p1_c, p99_c = d_dev_R["tasa_feed_Carbon_kg_min"].quantile([0.01, 0.99])
p1_gn, p99_gn = d_dev_R["tasa_gn_nm3_min"].quantile([0.01, 0.99])
N_GRID = 21
GRID_C = np.linspace(p1_c, p99_c, N_GRID)
GRID_GN = np.linspace(p1_gn, p99_gn, N_GRID)

filas_J = []
for base in base_rows:
    for control, grid, otra_palanca in [("carbon", GRID_C, "gn"), ("gn", GRID_GN, "carbon")]:
        filas_grid = []
        for val in grid:
            fila = base.copy()
            if control == "carbon":
                c_kg = float(val)
            else:
                c_kg = float(base["tasa_feed_Carbon_kg_min"])
                fila["tasa_gn_nm3_min"] = float(val)
            sn_disp = float(base["v5_sn_disp"])
            fila["tasa_feed_Carbon_kg_min"] = c_kg
            fila["v5_dosis_C_sn"] = c_kg / sn_disp if sn_disp > 0 else np.nan
            exc = c_kg - v5.C_ESTEQ_POR_SN * sn_disp
            fila["v5_exceso_C_pos"] = max(exc, 0.0)
            fila["v5_C_x_avance"] = c_kg * float(base["avance_reduccion_sn_prev"])
            filas_grid.append(fila)
        Xg = pd.DataFrame(filas_grid).reset_index(drop=True)
        pred_sn = predecir_componente("v5_ln_sn_dep", Xg)
        pred_feo = predecir_componente("v5_ln_feo_ret", Xg)
        J = pred_sn + W * pred_feo
        idx_max = int(np.argmax(J))
        frac = idx_max / (N_GRID - 1)
        tipo = "óptimo interior" if 0.10 <= frac <= 0.90 else ("frontera_baja" if frac < 0.10 else "frontera_alta")
        for i, val in enumerate(grid):
            filas_J.append(dict(avance_base=round(float(base["avance_reduccion_sn_prev"]), 4), control=control,
                                x=float(val), pred_ln_sn_dep=float(pred_sn[i]), pred_ln_feo_ret=float(pred_feo[i]),
                                J=float(J[i]), es_argmax=(i == idx_max), tipo_optimo=tipo))
        log(f"  avance~{base['avance_reduccion_sn_prev']:.3f} control={control}: argmax en x={grid[idx_max]:.3g} "
            f"(frac={frac:.2f}, {tipo}), J_max={J[idx_max]:.4f} vs J_hist~{J[(np.abs(grid - (base['tasa_feed_Carbon_kg_min'] if control=='carbon' else base['tasa_gn_nm3_min']))).argmin()]:.4f}")

J_df = pd.DataFrame(filas_J)
J_df.to_csv(SALIDA / "e7_03_curva_J.csv", index=False)
log(f"e7_03_curva_J.csv -> {J_df.shape}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, control, xlabel in zip(axes, ["carbon", "gn"], ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min"]):
    for av in sorted(J_df["avance_base"].unique()):
        sub = J_df[(J_df.control == control) & (J_df.avance_base == av)]
        ax.plot(sub["x"], sub["J"], "-o", ms=3, label=f"avance~{av}")
        arg = sub[sub.es_argmax]
        if len(arg):
            ax.scatter(arg["x"], arg["J"], color="crimson", zorder=5, s=40)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("J = ln_sn_dep_pred + 10*ln_feo_ret_pred")
    ax.set_title(f"J vs {control}")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGS / "e7_03_curva_J.png", dpi=120)
plt.close(fig)
log("figs/e7_03_curva_J.png guardada")

log(f"\nFIN. tiempo total {time.time() - T0:.1f}s")
