"""B-02 -- Balance reductor/oxidante en kg de carbono equivalente como feature estructurada.

Iteracion 6, bloque de balances (`experimentos/balances/bal_features.py`, NO modificado). Este
script responde a las 5 tareas del enunciado (ver `ITERACION_6_diseno.md` y el mensaje de tarea):

  1. Descriptiva y coherencia fisica de `b6_reductor_neto_kg_ceq` y afines; Spearman con targets
     de FeO/Sn, global y por tramos de avance.
  2. PLM v4 (ModeloPLM) con y sin el balance reductor como palanca, en Reduccion (5 targets) y
     Fusion (1 target): R2 OOF DEV / lockbox, tabla theta con IC95%, concordancia de signo.
  3. Utilizacion del reductor por batch/fase vs el 19.5%/48% "perdido al tiro" del modelo Rust;
     regresion OLS HC3 (DEV) sobre proxies de arrastre.
  4. Sensibilidad de la variante (b) a CARBONO_FIJO, FE_MET_DROSS, FE_MINERAL (uno a la vez) y a
     quitar el termino de la lanza ("sin lanza").
  5. Curvas de respuesta de la mejor variante: barrido de `b6_reductor_neto_kg_ceq` en P5..P95 por
     tramo de avance, optimo interior de Sn - 0.6*FeO y del SDI.

Semilla 42 en todo (GroupKFold, bootstrap, HGB). Anti-fuga: todas las palancas/estado nuevas se
verifican contra `bal.CLASIFICACION_V6` (solo STATE / DERIVED-ACTION / DERIVED (A_t x S_prev)).
OOF GroupKFold(5) por Batch en DEV es el criterio; lockbox solo se reporta (nunca se usa para
elegir). Salidas: `b02_*.csv` + este log (`b02_log.txt`, generado por el runner).
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
sys.path.insert(0, str(RAIZ / "experimentos" / "balances"))
sys.path.insert(0, str(RAIZ / "experimentos" / "search_targets"))

import bal_features as bal        # noqa: E402
import st_targets as st           # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
from scipy import stats           # noqa: E402
import statsmodels.api as sm      # noqa: E402

SEED = 42
OUT = Path(__file__).resolve().parent
N_BOOT = 150   # <=200, dominante es el costo de HGB, no el bootstrap (medido: ~9-10s por fit sea n_boot 0 o 200)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def log(*a):
    print(*a, flush=True)


# =============================================================================
# 0) Dataset combinado: b6_* (bal_features) + st_* (search_targets), sobre el MISMO orden de filas
# =============================================================================

def cargar_df_combinado(supuestos: dict | None = None) -> pd.DataFrame:
    raw = pd.read_pickle(bal.CACHE)
    d = bal.construir_features_v6(raw)          # aplica bal.SUPUESTOS vigente (para sensibilidad, mutar antes)
    d = st.construir_df_targets(d)               # agrega st_* (re-ordena por Batch/fecha, consistente)
    return d


df0 = cargar_df_combinado()
BATCHES_DEV, BATCHES_LOCKBOX = mp.split_dev_lockbox(df0)
log(f"df0 {df0.shape}, DEV batches={len(BATCHES_DEV)}, lockbox batches={len(BATCHES_LOCKBOX)}")

# columnas b02_* derivadas locales (no tocan bal_features.py)
df0["b02_reductor_neto_sin_lanza_kg_ceq"] = (df0["b6_c_fijo_kg"] + df0["b6_ceq_fe_dross_kg"]
                                             - df0["b6_demanda_sn_kg_ceq"] - df0["b6_demanda_fe2o3_kg_ceq"])

# anti-fuga: todas las columnas b6_/b02_ que usaremos como feature deben ser SEGURAS_V6 (o derivadas
# triviales de columnas seguras, como b02_reductor_neto_sin_lanza_kg_ceq = misma familia que b6_reductor_neto_kg_ceq)
FEATS_B6_USADAS = ["b6_reductor_neto_kg_ceq", "b6_ratio_reductor_demanda", "b6_capacidad_sobre_reduccion_feo_kg",
                   "b6_cum_reductor_neto_kg_prev", "b6_cum_oferta_reductora_kg_prev"]
_no_seguras = [c for c in FEATS_B6_USADAS if c not in bal.SEGURAS_V6]
assert not _no_seguras, f"features b6_ inseguras: {_no_seguras}"
log("anti-fuga OK:", {c: bal.CLASIFICACION_V6[c] for c in FEATS_B6_USADAS})


def _preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df_base = mp.dataset_base_modelo(df)          # excluye es_primer_escalon_batch
    df_base = mp4.agregar_columnas_v4(df_base)     # Cx_sn_v4, Cx_av_v4
    return df_base


DF_BASE0 = _preparar_base(df0)

# =============================================================================
# 1) Descriptiva y coherencia fisica
# =============================================================================
log("\n" + "=" * 90)
log("SECCION 1 -- Descriptiva y coherencia fisica")
log("=" * 90)

COLS_DESC = ["b6_reductor_neto_kg_ceq", "b6_ratio_reductor_demanda", "b6_ceq_lanza_kg",
            "b6_capacidad_sobre_reduccion_feo_kg", "b6_cum_reductor_neto_kg_prev"]


def _iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)


filas_desc = []
for fase in bal.FASES:
    sub = df0.loc[df0["fase_proceso"] == fase]
    for orden, g in sub.groupby("orden_escalon_fase"):
        for c in COLS_DESC:
            filas_desc.append({"fase": fase, "orden_escalon_fase": int(orden), "columna": c, "n": int(g[c].notna().sum()),
                               "mediana": float(g[c].median()), "iqr": float(_iqr(g[c])),
                               "p5": float(g[c].quantile(0.05)), "p95": float(g[c].quantile(0.95))})
tabla_desc = pd.DataFrame(filas_desc)
tabla_desc.to_csv(OUT / "b02_descriptiva_por_escalon.csv", index=False)
log(tabla_desc.pivot_table(index=["fase", "orden_escalon_fase"], columns="columna", values="mediana").round(1).to_string())

# --- Spearman con targets, global y por tramos de avance (solo Reduccion) ---
# filtro: ~es_primer_escalon_batch (mismo criterio que dataset_base_modelo / _preparar_base), NO
# es_primer_escalon_fase -- R0 SI tiene estado previo valido (hereda el cierre de F6).
TRAMOS = [(-np.inf, 0.84, "<0.84"), (0.84, 0.96, "0.84-0.96"), (0.96, np.inf, ">0.96")]
sub_r = df0.loc[(df0["fase_proceso"] == "Reducción") & (~df0["es_primer_escalon_batch"].astype(bool))].copy()

targets_corr = ["feo_extraido_est_kg", "b6_feo_extraido_multi_kg", "sn_extraido_est_kg"]
predictores_corr = ["b6_reductor_neto_kg_ceq", "b6_capacidad_sobre_reduccion_feo_kg"]

filas_corr = []
for pred in predictores_corr:
    for tgt in targets_corr:
        d = sub_r[[pred, tgt, "avance_reduccion_sn_prev"]].dropna()
        if len(d) >= 20:
            rho, p = stats.spearmanr(d[pred], d[tgt])
            filas_corr.append({"predictor": pred, "target": tgt, "tramo": "global", "n": len(d), "rho": rho, "p": p})
        for lo, hi, nombre in TRAMOS:
            dt = d.loc[d["avance_reduccion_sn_prev"].between(lo, hi, inclusive="right" if lo != -np.inf else "both")]
            if len(dt) >= 15:
                rho, p = stats.spearmanr(dt[pred], dt[tgt])
                filas_corr.append({"predictor": pred, "target": tgt, "tramo": nombre, "n": len(dt), "rho": rho, "p": p})
tabla_corr = pd.DataFrame(filas_corr)
tabla_corr.to_csv(OUT / "b02_spearman_por_tramo.csv", index=False)
log(tabla_corr.to_string(index=False))

# --- utilizacion / smoke-test rapido por fase ---
log("\nb6_utilizacion_reductor por fase (mediana, P25, P75):")
for fase in bal.FASES:
    s = df0.loc[df0["fase_proceso"] == fase, "b6_utilizacion_reductor"].dropna()
    log(f"  {fase}: mediana={s.median():.3f} P25={s.quantile(.25):.3f} P75={s.quantile(.75):.3f} n={len(s)}")

# =============================================================================
# 2) PLM v4 con y sin balance reductor
# =============================================================================
log("\n" + "=" * 90)
log("SECCION 2 -- PLM v4: variantes de palancas (Reduccion y Fusion)")
log("=" * 90)

S_RED = mp4.ESTADO_V4[("Reducción", "feo_kg")]                 # == ESTADO_V4[("Reducción","sn_kg")]
S_RED_EXT = S_RED + ["b6_cum_reductor_neto_kg_prev", "b6_cum_oferta_reductora_kg_prev"]   # variantes (d)/(e)

# Palancas por "rol" del target: sn (cinetica Sn), feo (cinetica FeO/sobre-reduccion), sdi (selectividad)
PALANCAS_RED = {
    "sn": {
        "a": ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
        "b": ["b6_reductor_neto_kg_ceq", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
        "c": ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "b6_capacidad_sobre_reduccion_feo_kg"],
        "d": ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
        "e": ["b6_reductor_neto_kg_ceq", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
    },
    "feo": {
        "a": ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "b": ["b6_reductor_neto_kg_ceq", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "c": ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min",
              "b6_capacidad_sobre_reduccion_feo_kg"],
        "d": ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
        "e": ["b6_reductor_neto_kg_ceq", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    },
}
PALANCAS_RED["sdi"] = PALANCAS_RED["feo"]   # SDI: misma familia de palancas que FeO (selectividad Sn/FeO)

ESTADO_RED = {"a": S_RED, "b": S_RED, "c": S_RED, "d": S_RED_EXT, "e": S_RED_EXT}

# Signo teorico. Para sn/feo: literal del enunciado (reductor_neto + sobre FeO y Sn; capacidad + sobre FeO).
# Para SDI: interpretacion propia (documentada en b02_resultados.md): el SDI premia agotar Sn SIN metalizar
# Fe, por lo que el CANAL DE SOBRA de reductor (una vez satisfecha la demanda de Sn) es el mismo canal de
# sobre-reduccion que penaliza el SDI -> signo NEGATIVO para b6_reductor_neto_kg_ceq y Cx_av_v4/capacidad;
# el carbon TOTAL (dominado por el canal de Sn, exp 19) se mantiene positivo.
SIGNO_RED = {
    "sn": {
        "a": {"Cx_sn_v4": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1},
        "b": {"b6_reductor_neto_kg_ceq": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1},
        "c": {"Cx_sn_v4": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "b6_capacidad_sobre_reduccion_feo_kg": 0},
        "d": {"Cx_sn_v4": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1},
        "e": {"b6_reductor_neto_kg_ceq": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1},
    },
    "feo": {
        "a": {"tasa_feed_Carbon_kg_min": +1, "Cx_av_v4": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": -1},
        "b": {"b6_reductor_neto_kg_ceq": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": -1},
        "c": {"tasa_feed_Carbon_kg_min": +1, "Cx_av_v4": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": -1,
              "b6_capacidad_sobre_reduccion_feo_kg": +1},
        "d": {"tasa_feed_Carbon_kg_min": +1, "Cx_av_v4": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": -1},
        "e": {"b6_reductor_neto_kg_ceq": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": -1},
    },
    "sdi": {
        "a": {"tasa_feed_Carbon_kg_min": +1, "Cx_av_v4": -1, "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1},
        "b": {"b6_reductor_neto_kg_ceq": -1, "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1},
        "c": {"tasa_feed_Carbon_kg_min": +1, "Cx_av_v4": -1, "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1,
              "b6_capacidad_sobre_reduccion_feo_kg": -1},
        "d": {"tasa_feed_Carbon_kg_min": +1, "Cx_av_v4": -1, "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1},
        "e": {"b6_reductor_neto_kg_ceq": -1, "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1},
    },
}

TARGETS_RED = {
    "feo_extraido_est_kg": "feo",
    "b6_feo_extraido_multi_kg": "feo",
    "sn_extraido_est_kg": "sn",
    "st_R22_sdi": "sdi",
    "b6_sdi_multi": "sdi",
}
VARIANTES = ["a", "b", "c", "d", "e"]

# verificacion anti-fuga de TODAS las combinaciones estado/palanca usadas
import feature_engineering as fe  # noqa: E402
for rol, dic in PALANCAS_RED.items():
    for v, feats in dic.items():
        for f in feats:
            if f.startswith("b6_"):
                assert f in bal.SEGURAS_V6, f"{f} no es segura"
            elif not f.endswith("_v4"):
                assert f in fe.filtrar_features_seguras([f]), f"{f} no es segura"
for v, S in ESTADO_RED.items():
    for f in S:
        if f.startswith("b6_"):
            assert f in bal.SEGURAS_V6, f"{f} no es segura"


def fit_variant(df_base: pd.DataFrame, fase: str, target: str, estado: list[str], palancas: list[str],
                signo: dict, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[mp4.ModeloPLM, dict, pd.DataFrame]:
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    need = list(dict.fromkeys(estado + palancas + [target, "Batch"]))
    d_dev = sub.loc[sub["Batch"].isin(BATCHES_DEV), need].dropna().reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(BATCHES_LOCKBOX), need].dropna().reset_index(drop=True)
    m = mp4.ModeloPLM(estado, palancas, target).fit(d_dev, n_boot=n_boot, seed=seed)
    p_lb = m.predict(d_lb)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2_lb = r2_score(d_lb[target], p_lb) if len(d_lb) > 5 else np.nan
    resumen = dict(fase=fase, target=target, n_estado=len(estado), n_palancas=len(palancas),
                   n_dev=len(d_dev), n_lockbox=len(d_lb), r2_oof_dev=m.r2_oof_interno, r2_lockbox=r2_lb,
                   mae_lockbox=mean_absolute_error(d_lb[target], p_lb) if len(d_lb) > 5 else np.nan,
                   std_target_lockbox=float(d_lb[target].std()) if len(d_lb) else np.nan)
    theta = m.tabla_theta(signo).reset_index()
    return m, resumen, theta


t0 = time.time()
resumenes, thetas, modelos = [], [], {}
for target, rol in TARGETS_RED.items():
    for v in VARIANTES:
        estado, palancas = ESTADO_RED[v], PALANCAS_RED[rol][v]
        signo = SIGNO_RED[rol][v]
        m, resumen, theta = fit_variant(DF_BASE0, "Reducción", target, estado, palancas, signo)
        resumen["variante"] = v; resumen["rol"] = rol
        theta.insert(0, "variante", v); theta.insert(0, "target", target)
        resumenes.append(resumen); thetas.append(theta)
        modelos[("Reducción", target, v)] = m
        log(f"  R {target:28s} [{v}] r2_oof={resumen['r2_oof_dev']:+.3f} r2_lb={resumen['r2_lockbox']:+.3f} "
            f"n_dev={resumen['n_dev']} ({time.time()-t0:.0f}s acum)")

TABLA_RESUMEN_R = pd.DataFrame(resumenes)
TABLA_THETA_R = pd.concat(thetas, ignore_index=True)
TABLA_RESUMEN_R.to_csv(OUT / "b02_plm_reduccion_resumen.csv", index=False)
TABLA_THETA_R.to_csv(OUT / "b02_plm_reduccion_theta.csv", index=False)

# --- Fusion: sn_extraido_est_kg, (a) v4 vs (b) reductor ---
S_FUS = mp4.ESTADO_V4[("Fusión", "sn_kg")]
PALANCAS_FUS = {
    "a": ["relacion_C_Sn_carga", "relacion_CaO_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
    "b": ["b6_ratio_reductor_demanda", "b6_reductor_neto_kg_ceq", "relacion_CaO_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
}
SIGNO_FUS = {
    "a": {"relacion_C_Sn_carga": +1, "relacion_CaO_carga": 0, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": 0},
    "b": {"b6_ratio_reductor_demanda": +1, "b6_reductor_neto_kg_ceq": +1, "relacion_CaO_carga": 0, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": 0},
}
for v, feats in PALANCAS_FUS.items():
    for f in feats:
        if f.startswith("b6_"):
            assert f in bal.SEGURAS_V6

resumenes_f, thetas_f = [], []
for v in ["a", "b"]:
    m, resumen, theta = fit_variant(DF_BASE0, "Fusión", "sn_extraido_est_kg", S_FUS, PALANCAS_FUS[v], SIGNO_FUS[v])
    resumen["variante"] = v
    theta.insert(0, "variante", v); theta.insert(0, "target", "sn_extraido_est_kg")
    resumenes_f.append(resumen); thetas_f.append(theta)
    modelos[("Fusión", "sn_extraido_est_kg", v)] = m
    log(f"  F sn_extraido_est_kg [{v}] r2_oof={resumen['r2_oof_dev']:+.3f} r2_lb={resumen['r2_lockbox']:+.3f} n_dev={resumen['n_dev']}")

TABLA_RESUMEN_F = pd.DataFrame(resumenes_f)
TABLA_THETA_F = pd.concat(thetas_f, ignore_index=True)
TABLA_RESUMEN_F.to_csv(OUT / "b02_plm_fusion_resumen.csv", index=False)
TABLA_THETA_F.to_csv(OUT / "b02_plm_fusion_theta.csv", index=False)
log(f"seccion 2 completa en {time.time()-t0:.0f}s")

# =============================================================================
# 3) Utilizacion del reductor
# =============================================================================
log("\n" + "=" * 90)
log("SECCION 3 -- Utilizacion del reductor por batch/fase")
log("=" * 90)

util_batch = []
for (b, fase), g in df0.groupby(["Batch", "fase_proceso"]):
    cons = g["b6_c_consumido_teorico_kg"].sum(min_count=1)
    ofer = g["b6_oferta_reductora_kg_ceq"].sum(min_count=1)
    util_batch.append({"Batch": b, "fase_proceso": fase, "c_consumido_kg": cons, "oferta_kg": ofer,
                       "utilizacion": cons / ofer if ofer and ofer > 0 else np.nan})
tabla_util = pd.DataFrame(util_batch)
tabla_util.to_csv(OUT / "b02_utilizacion_por_batch.csv", index=False)

log("Utilizacion por batch (sum consumido / sum oferta), mediana / IQR, vs supuesto Rust (0.80 F / 0.52 R):")
for fase in bal.FASES:
    s = tabla_util.loc[tabla_util["fase_proceso"] == fase, "utilizacion"].dropna()
    log(f"  {fase}: mediana={s.median():.3f} IQR=[{s.quantile(.25):.3f},{s.quantile(.75):.3f}] n={len(s)}")

# --- Regresion OLS HC3 (DEV) de la utilizacion de Reduccion sobre proxies de arrastre ---
prox_r = df0.loc[df0["fase_proceso"] == "Reducción"].groupby("Batch").agg(
    tiro_horno_pct_medio=("tiro_horno_pct", "mean"),
    gas_total_medio=("total_process_gas_nm3_min", "mean"),
    pos_lanza_media=("posicion_vertical_lanza_mm", "mean"))
prox_f = df0.loc[df0["fase_proceso"] == "Fusión"].groupby("Batch").agg(frac_pellets_F=("frac_pellets_carga", "mean"))
batch_st = st.construir_batch(df0)  # f_metal, f_dross, f_polvo, es_lockbox, etc.

util_r = tabla_util.loc[tabla_util["fase_proceso"] == "Reducción"].set_index("Batch")
reg_df = util_r[["utilizacion"]].join(prox_r).join(prox_f).join(batch_st[["f_polvo", "f_dross", "es_lockbox"]])
reg_df = reg_df.dropna()
reg_dev = reg_df.loc[~reg_df["es_lockbox"]]
log(f"\nRegresion utilizacion_R ~ proxies de arrastre (OLS HC3, DEV, n={len(reg_dev)}):")

X = reg_dev[["tiro_horno_pct_medio", "gas_total_medio", "pos_lanza_media", "frac_pellets_F", "f_polvo", "f_dross"]].copy()
X = (X - X.mean()) / X.std()
X = sm.add_constant(X)
res_util = sm.OLS(reg_dev["utilizacion"], X).fit(cov_type="HC3")
log(res_util.summary().as_text())
tabla_ols_util = pd.DataFrame({"coef_por_sd": res_util.params, "ci_lo": res_util.conf_int()[0],
                               "ci_hi": res_util.conf_int()[1], "p": res_util.pvalues})
tabla_ols_util.to_csv(OUT / "b02_ols_utilizacion_reduccion.csv")

# =============================================================================
# 4) Sensibilidad
# =============================================================================
log("\n" + "=" * 90)
log("SECCION 4 -- Sensibilidad a supuestos")
log("=" * 90)

BASE_VALS = {k: bal.SUPUESTOS[k]["valor"] for k in ("CARBONO_FIJO", "FE_MET_DROSS", "FE_MINERAL")}
GRID = {"CARBONO_FIJO": [0.60, 0.75, 0.85], "FE_MET_DROSS": [0.20, 0.30, 0.40], "FE_MINERAL": [0.45, 0.55, 0.65]}
TARGETS_SENS = {"feo_extraido_est_kg": "feo", "st_R22_sdi": "sdi"}

filas_sens = []
t0 = time.time()
for param, valores in GRID.items():
    for val in valores:
        bal.SUPUESTOS[param]["valor"] = val
        dfx = cargar_df_combinado()
        dfx["b02_reductor_neto_sin_lanza_kg_ceq"] = (dfx["b6_c_fijo_kg"] + dfx["b6_ceq_fe_dross_kg"]
                                                     - dfx["b6_demanda_sn_kg_ceq"] - dfx["b6_demanda_fe2o3_kg_ceq"])
        dfx_base = _preparar_base(dfx)
        for target, rol in TARGETS_SENS.items():
            palancas = PALANCAS_RED[rol]["b"]
            signo = SIGNO_RED[rol]["b"]
            m, resumen, theta = fit_variant(dfx_base, "Reducción", target, S_RED, palancas, signo, n_boot=N_BOOT)
            resumen.update(param=param, valor=val, es_baseline=(val == BASE_VALS[param]))
            theta_neto = theta.set_index("palanca").loc["b6_reductor_neto_kg_ceq"] if "b6_reductor_neto_kg_ceq" in theta["palanca"].values else None
            if theta_neto is not None:
                resumen["theta_1sd_reductor_neto"] = theta_neto["theta_1sd"]
                resumen["ci_lo_1sd_reductor_neto"] = theta_neto["ci_lo_1sd"]
                resumen["ci_hi_1sd_reductor_neto"] = theta_neto["ci_hi_1sd"]
            filas_sens.append(resumen)
        bal.SUPUESTOS[param]["valor"] = BASE_VALS[param]   # restaurar
        log(f"  {param}={val}: {time.time()-t0:.0f}s acum")

tabla_sens = pd.DataFrame(filas_sens)
tabla_sens.to_csv(OUT / "b02_sensibilidad_supuestos.csv", index=False)
log(tabla_sens[["param", "valor", "target", "r2_oof_dev", "r2_lockbox", "theta_1sd_reductor_neto",
               "ci_lo_1sd_reductor_neto", "ci_hi_1sd_reductor_neto"]].to_string(index=False))

# --- variante "sin lanza" ---
log("\nVariante 'sin lanza' (b6_c_fijo_kg + b6_ceq_fe_dross_kg - demanda, sin CO de la lanza):")
filas_sin_lanza = []
for target, rol in TARGETS_SENS.items():
    palancas = ["b02_reductor_neto_sin_lanza_kg_ceq"] + [p for p in PALANCAS_RED[rol]["b"] if p != "b6_reductor_neto_kg_ceq"]
    signo = {**{k: v for k, v in SIGNO_RED[rol]["b"].items() if k != "b6_reductor_neto_kg_ceq"},
             "b02_reductor_neto_sin_lanza_kg_ceq": SIGNO_RED[rol]["b"]["b6_reductor_neto_kg_ceq"]}
    m, resumen, theta = fit_variant(DF_BASE0, "Reducción", target, S_RED, palancas, signo, n_boot=N_BOOT)
    resumen.update(param="sin_lanza", valor=np.nan, es_baseline=False)
    filas_sens_sl = resumen
    filas_sin_lanza.append(resumen)
    theta.insert(0, "variante", "b_sin_lanza"); theta.insert(0, "target", target)
    theta.to_csv(OUT / f"b02_theta_sin_lanza_{target.replace('.', '_')}.csv", index=False)
    log(f"  {target}: r2_oof={resumen['r2_oof_dev']:+.3f} r2_lb={resumen['r2_lockbox']:+.3f}")
    log(theta.to_string(index=False))
tabla_sin_lanza = pd.DataFrame(filas_sin_lanza)
tabla_sin_lanza.to_csv(OUT / "b02_sin_lanza_resumen.csv", index=False)

# =============================================================================
# 5) Curvas de respuesta (mejor variante)
# =============================================================================
log("\n" + "=" * 90)
log("SECCION 5 -- Curvas de respuesta de la mejor variante")
log("=" * 90)

# elegir "mejor variante" por target oficial (feo_extraido_est_kg, sn_extraido_est_kg): mejor r2_oof DEV
# entre (a)-(e) que no empeore el lockbox respecto a (a)
mejor_var = {}
for target in ["feo_extraido_est_kg", "sn_extraido_est_kg"]:
    tr = TABLA_RESUMEN_R.loc[TABLA_RESUMEN_R["target"] == target].set_index("variante")
    base = tr.loc["a"]
    candidatas = tr.loc[tr["r2_lockbox"] >= base["r2_lockbox"] - 0.01]
    mejor = candidatas["r2_oof_dev"].idxmax() if len(candidatas) else "a"
    mejor_var[target] = mejor
    log(f"mejor variante para {target}: {mejor} (r2_oof={tr.loc[mejor,'r2_oof_dev']:.3f} vs a={base['r2_oof_dev']:.3f}; "
        f"r2_lb={tr.loc[mejor,'r2_lockbox']:.3f} vs a={base['r2_lockbox']:.3f})")

# La variante (b) es la unica que usa b6_reductor_neto_kg_ceq como palanca en TODOS los targets
# (feo, sn, sdi) con el MISMO estado (S_RED): se usa para las 3 curvas conjuntas, sea o no la
# variante que gana en R2 OOF (eso se reporta arriba, en `mejor_var`, como diagnostico aparte).
m_feo = modelos[("Reducción", "feo_extraido_est_kg", "b")]
m_sn = modelos[("Reducción", "sn_extraido_est_kg", "b")]
m_sdi = modelos[("Reducción", "st_R22_sdi", "b")]
log(f"(nota: la curva usa variante 'b' en los 3 targets; 'mejor variante' por R2 fue {mejor_var} para feo/sn)")

palanca_barrido = "b6_reductor_neto_kg_ceq"
sub_r_dev = DF_BASE0.loc[(DF_BASE0["fase_proceso"] == "Reducción") & DF_BASE0["Batch"].isin(BATCHES_DEV)]
p5, p95 = sub_r_dev[palanca_barrido].quantile([0.05, 0.95])
grid = np.linspace(p5, p95, 41)

cols_necesarias = list(dict.fromkeys(m_feo.estado + m_feo.palancas + m_sn.estado + m_sn.palancas + m_sdi.estado + m_sdi.palancas))
filas_curva = []
for lo, hi, nombre in TRAMOS:
    d_tr = sub_r_dev.loc[sub_r_dev["avance_reduccion_sn_prev"].between(lo, hi, inclusive="right" if lo != -np.inf else "both"), cols_necesarias]
    d_tr = d_tr.dropna()
    if len(d_tr) < 15:
        continue
    fila_mediana = d_tr.median(numeric_only=True)
    for val in grid:
        fila = fila_mediana.copy()
        fila[palanca_barrido] = val
        X = pd.DataFrame([fila])
        pred_feo = float(m_feo.predict(X[m_feo.estado + m_feo.palancas])[0])
        pred_sn = float(m_sn.predict(X[m_sn.estado + m_sn.palancas])[0])
        pred_sdi = float(m_sdi.predict(X[m_sdi.estado + m_sdi.palancas])[0])
        J = pred_sn - 0.6 * max(pred_feo, 0.0)
        filas_curva.append({"tramo_avance": nombre, "b6_reductor_neto_kg_ceq": val, "pred_feo_extraido_kg": pred_feo,
                            "pred_sn_extraido_kg": pred_sn, "pred_sdi": pred_sdi, "J_sn_menos_06feo": J})

tabla_curva = pd.DataFrame(filas_curva)
tabla_curva.to_csv(OUT / "b02_curvas_respuesta.csv", index=False)

log("\nOptimo interior de J = Sn - 0.6*FeO por tramo de avance (si existe, valor de reductor_neto en el maximo):")
for nombre, g in tabla_curva.groupby("tramo_avance"):
    i = g["J_sn_menos_06feo"].idxmax()
    borde = "BORDE" if g.loc[i, "b6_reductor_neto_kg_ceq"] in (g["b6_reductor_neto_kg_ceq"].min(), g["b6_reductor_neto_kg_ceq"].max()) else "INTERIOR"
    log(f"  avance {nombre}: optimo J en reductor_neto={g.loc[i,'b6_reductor_neto_kg_ceq']:.0f} kg_ceq "
        f"(J={g.loc[i,'J_sn_menos_06feo']:.1f}) [{borde}]")
    if g["pred_sdi"].notna().any():
        j = g["pred_sdi"].idxmax()
        borde2 = "BORDE" if g.loc[j, "b6_reductor_neto_kg_ceq"] in (g["b6_reductor_neto_kg_ceq"].min(), g["b6_reductor_neto_kg_ceq"].max()) else "INTERIOR"
        log(f"    optimo SDI en reductor_neto={g.loc[j,'b6_reductor_neto_kg_ceq']:.0f} kg_ceq (SDI={g.loc[j,'pred_sdi']:.3f}) [{borde2}]")

log("\n" + "=" * 90)
log("FIN b02_balance_reductor.py")
log("=" * 90)
