"""Experimento 15 -- mejor senal de sobre-reduccion de FeO en Reduccion y re-estimacion de lambda_Fe.

Ver experimentos/ITERACION_4_diseno.md. Cinco secciones:
  1) Techo de ruido del trazador CaO/FeO (subgrupo carbon-bajo+duracion-corta; pares R2-R3 con
     carbon bajo; propagacion analitica del ruido de ensayo a kg de FeO).
  2) Targets alternativos de sobre-reduccion en Reduccion (T0 crudo, T1 fraccion, T2 k aparente,
     T3 trazador suavizado, T4 perdida robusta, T5 cierre de composicion corregido), mismo HGB,
     OOF GroupKFold(5) por Batch en DEV + lockbox, metricas nativas y convertidas a kg.
  3) Agregado por fase (batch-level): FeO_reducido_total_R_kg vs modelo de escalon.
  4) Re-estimacion de lambda_Fe (OLS entre batches, HC3, con/sin controles, crudo/suavizado,
     R0-R1 vs R2-R3 separados).
  5) Selectividad marginal empirica por bins de avance_reduccion_sn_prev.

Salidas: 15_techo_ruido.csv, 15_senales_feo.csv, 15_agregado_batch.csv, 15_lambda_fe.csv,
15_selectividad_marginal.csv, figs_15/15_selectividad_marginal.png, 15_log.txt.
Ejecutar con .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=4.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import feature_engineering as fe  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

OUT = RAIZ / "experimentos"
FIGDIR = OUT / "figs_15"
FIGDIR.mkdir(exist_ok=True, parents=True)
LOG_PATH = OUT / "15_log.txt"
_LOG_LINES: list[str] = []

SEED = 42
N_SPLITS = 5


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _LOG_LINES.append(line)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")


# =============================================================================
# Modelos
# =============================================================================

def make_hgb(loss: str = "squared_error") -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=15, l2_regularization=1.0,
                                          random_state=SEED, loss=loss)


def hgb_pequeno() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=100,
                                          min_samples_leaf=10, l2_regularization=1.0, random_state=SEED)


# =============================================================================
# 1) Techo de ruido del trazador
# =============================================================================

def seccion1(red: pd.DataFrame, batches_dev: set) -> pd.DataFrame:
    log("=" * 100)
    log("1) Techo de ruido del trazador CaO/FeO")
    log("=" * 100)
    red_dev = red[red["Batch"].isin(batches_dev)].copy()

    p10_carbon = red_dev["tasa_feed_Carbon_kg_min"].quantile(0.10)
    p25_dur = red_dev["duracion_plan_min"].quantile(0.25)
    log(f"P10 tasa_feed_Carbon_kg_min (Reduccion DEV) = {p10_carbon:.2f} kg/min")
    log(f"P25 duracion_plan_min (Reduccion DEV) = {p25_dur:.2f} min")

    mask_bajo = (red_dev["tasa_feed_Carbon_kg_min"] < p10_carbon) & (red_dev["duracion_plan_min"] <= p25_dur)
    sub = red_dev.loc[mask_bajo, ["feo_extraido_est_kg", "sn_extraido_est_kg"]].dropna()
    log(f"Escalones con carbon bajo + duracion corta: n={len(sub)} (de {len(red_dev)} en DEV)")

    var_ruido_feo = sub["feo_extraido_est_kg"].var(ddof=1)
    var_ruido_sn = sub["sn_extraido_est_kg"].var(ddof=1)
    var_total_feo = red_dev["feo_extraido_est_kg"].var(ddof=1)
    var_total_sn = red_dev["sn_extraido_est_kg"].var(ddof=1)
    r2max_feo = 1 - var_ruido_feo / var_total_feo
    r2max_sn = 1 - var_ruido_sn / var_total_sn
    log(f"FeO: sd_ruido={var_ruido_feo ** 0.5:.0f} kg (var={var_ruido_feo:.0f}) sd_total={var_total_feo ** 0.5:.0f} "
        f"-> R2_max = 1 - var_ruido/var_total = {r2max_feo:.3f}")
    log(f"Sn : sd_ruido={var_ruido_sn ** 0.5:.0f} kg (var={var_ruido_sn:.0f}) sd_total={var_total_sn ** 0.5:.0f} "
        f"-> R2_max = {r2max_sn:.3f}")

    # Pares consecutivos R2-R3 con carbon bajo en AMBOS escalones (mismo batch)
    cols = ["Batch", "ley_cao_escoria_pct", "ley_feo_escoria_pct", "tasa_feed_Carbon_kg_min"]
    r2df = red_dev.loc[red_dev.orden_escalon_fase == 2, cols].rename(
        columns={c: f"{c}_r2" for c in cols if c != "Batch"})
    r3df = red_dev.loc[red_dev.orden_escalon_fase == 3, cols].rename(
        columns={c: f"{c}_r3" for c in cols if c != "Batch"})
    pares = r2df.merge(r3df, on="Batch")
    pares_bajo = pares[(pares.tasa_feed_Carbon_kg_min_r2 < p10_carbon) & (pares.tasa_feed_Carbon_kg_min_r3 < p10_carbon)].dropna(
        subset=["ley_cao_escoria_pct_r2", "ley_cao_escoria_pct_r3", "ley_feo_escoria_pct_r2", "ley_feo_escoria_pct_r3"])
    log(f"Pares R2-R3 con ambos carbon bajo (< P10): n={len(pares_bajo)}")

    d_cao = pares_bajo["ley_cao_escoria_pct_r3"] - pares_bajo["ley_cao_escoria_pct_r2"]
    d_feo = pares_bajo["ley_feo_escoria_pct_r3"] - pares_bajo["ley_feo_escoria_pct_r2"]
    # Si el %CaO/%FeO "verdadero" no cambia (sin carga de cal, avance ya casi agotado en R2-R3),
    # la diferencia observada es ruido(t)-ruido(t-1) de ensayos independientes: sd_ruido = sd(diff)/sqrt(2)
    sd_ruido_cao_pct = d_cao.std(ddof=1) / np.sqrt(2)
    sd_ruido_feo_pct = d_feo.std(ddof=1) / np.sqrt(2)
    media_cao = red_dev["ley_cao_escoria_pct"].mean()
    media_feo = red_dev["ley_feo_escoria_pct"].mean()
    rel_ruido_cao = sd_ruido_cao_pct / media_cao
    rel_ruido_feo = sd_ruido_feo_pct / media_feo
    log(f"Ruido ensayo %CaO (pares R2-R3, carbon bajo): sd={sd_ruido_cao_pct:.3f} pts "
        f"({rel_ruido_cao * 100:.2f}% relativo sobre media %CaO={media_cao:.2f}%)")
    log(f"Ruido ensayo %FeO (pares R2-R3, carbon bajo): sd={sd_ruido_feo_pct:.3f} pts "
        f"({rel_ruido_feo * 100:.2f}% relativo sobre media %FeO={media_feo:.2f}%)")

    # Propagacion analitica: escoria ~55 t, %FeO ~18-30%. masa = CaO_acum/(%CaO/100) -> ruido
    # relativo de masa ~= ruido relativo de %CaO (trazador). feo_inv = masa*%FeO/100 combina ambos
    # ruidos relativos (independientes) en cuadratura. feo_extraido = feo_inv[t]-feo_inv[t-1] ->
    # var(feo_extraido) = 2*var(feo_inv) si los ruidos de ensayo son independientes entre escalones.
    masa_tip = red_dev["masa_escoria_est_kg"].median()
    feo_tip_pct = red_dev["ley_feo_escoria_pct"].median()
    feo_inv_tip = masa_tip * feo_tip_pct / 100
    rel_ruido_masa = rel_ruido_cao
    rel_ruido_feo_inv = float(np.sqrt(rel_ruido_masa ** 2 + rel_ruido_feo ** 2))
    sd_feo_inv = feo_inv_tip * rel_ruido_feo_inv
    sd_feo_extraido_analitico = float(np.sqrt(2) * sd_feo_inv)
    log(f"Propagacion analitica: masa tipica={masa_tip:.0f} kg, %FeO tipico={feo_tip_pct:.1f}%, "
        f"feo_inventario tipico={feo_inv_tip:.0f} kg")
    log(f"  ruido relativo masa(=%CaO)={rel_ruido_masa * 100:.2f}%  ruido relativo %FeO={rel_ruido_feo * 100:.2f}% "
        f"-> ruido relativo feo_inventario={rel_ruido_feo_inv * 100:.2f}%")
    log(f"  sd(feo_inventario) = {sd_feo_inv:.0f} kg  ->  sd(feo_extraido) analitico = sqrt(2)*sd(feo_inv) "
        f"= {sd_feo_extraido_analitico:.0f} kg")
    log(f"  Comparar con: sd empirico subgrupo carbon-bajo+duracion-corta = {var_ruido_feo ** 0.5:.0f} kg, "
        f"y con MAE del modelo feo_kg reportado en hallazgos.md 14.5 = 515 kg (std total 814 kg)")

    tabla = pd.DataFrame([
        dict(metodo="subgrupo_carbon_bajo_duracion_corta", n=len(sub),
             sd_ruido_feo_kg=var_ruido_feo ** 0.5, sd_ruido_sn_kg=var_ruido_sn ** 0.5,
             sd_total_feo_kg=var_total_feo ** 0.5, sd_total_sn_kg=var_total_sn ** 0.5,
             r2_max_feo=r2max_feo, r2_max_sn=r2max_sn),
        dict(metodo="pares_R2R3_carbon_bajo_ensayo", n=len(pares_bajo),
             sd_ruido_cao_pct=sd_ruido_cao_pct, rel_ruido_cao_pct=rel_ruido_cao * 100,
             sd_ruido_feo_pct=sd_ruido_feo_pct, rel_ruido_feo_pct=rel_ruido_feo * 100),
        dict(metodo="propagacion_analitica_a_kg", masa_tipica_kg=masa_tip, feo_pct_tipico=feo_tip_pct,
             feo_inventario_tipico_kg=feo_inv_tip, rel_ruido_feo_inventario_pct=rel_ruido_feo_inv * 100,
             sd_feo_inventario_kg=sd_feo_inv, sd_feo_extraido_kg_analitico=sd_feo_extraido_analitico,
             mae_modelo_feo_kg_referencia_hallazgos=515.0, std_total_referencia_hallazgos=814.0),
    ])
    tabla.to_csv(OUT / "15_techo_ruido.csv", index=False)
    log("Guardado 15_techo_ruido.csv")
    return tabla


# =============================================================================
# 2) Targets alternativos
# =============================================================================

def agregar_feo_suavizado(df: pd.DataFrame, red: pd.DataFrame) -> pd.DataFrame:
    """Recalcula la masa de escoria con un %CaO suavizado (mediana movil de 3) sobre la cadena
    [ultimo escalon de Fusion, R0, R1, R2, R3] de cada batch, y de ahi `feo_extraido_suav_kg`."""
    fus6 = df.loc[(df.fase_proceso == "Fusión") & (df.orden_escalon_fase == 6),
                  ["Batch", "ley_cao_escoria_pct", "ley_feo_escoria_pct", "masa_escoria_est_kg"]].copy()
    fus6["fase_seq"] = -1
    red_chain = df.loc[df.fase_proceso == "Reducción",
                        ["Batch", "orden_escalon_fase", "ley_cao_escoria_pct", "ley_feo_escoria_pct",
                         "masa_escoria_est_kg"]].copy()
    red_chain["fase_seq"] = red_chain["orden_escalon_fase"]
    cadena = pd.concat([fus6, red_chain.drop(columns="orden_escalon_fase")], ignore_index=True)
    cadena = cadena.sort_values(["Batch", "fase_seq"]).reset_index(drop=True)

    cadena["cum_cao_incl_kg"] = cadena["masa_escoria_est_kg"] * cadena["ley_cao_escoria_pct"] / 100
    cadena["ley_cao_suav_pct"] = cadena.groupby("Batch")["ley_cao_escoria_pct"].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).median())
    cao_suav_prot = cadena["ley_cao_suav_pct"].where(cadena["ley_cao_suav_pct"] > 0.5)
    cadena["masa_suav_kg"] = cadena["cum_cao_incl_kg"] / (cao_suav_prot / 100)
    cadena["feo_inv_suav_kg"] = cadena["masa_suav_kg"] * cadena["ley_feo_escoria_pct"] / 100
    cadena["feo_extraido_suav_kg"] = -(cadena.groupby("Batch")["feo_inv_suav_kg"].diff())

    suav_red = cadena.loc[cadena.fase_seq >= 0, ["Batch", "fase_seq", "feo_extraido_suav_kg", "masa_suav_kg"]].rename(
        columns={"fase_seq": "orden_escalon_fase"})
    red = red.merge(suav_red, on=["Batch", "orden_escalon_fase"], how="left", validate="one_to_one")
    n_ok = red["feo_extraido_suav_kg"].notna().sum()
    log(f"feo_extraido_suav_kg calculado para {n_ok} / {len(red)} filas de Reduccion")
    return red


def construir_targets_alternativos(df: pd.DataFrame, red: pd.DataFrame) -> pd.DataFrame:
    red = red.copy()
    inv_prev = red["feo_inventario_escoria_est_kg_prev"]
    frac_raw = (red["feo_extraido_est_kg"] / inv_prev).where(inv_prev.abs() > 500)
    q_lo, q_hi = frac_raw.quantile([0.005, 0.995])
    red["frac_feo_extraido"] = frac_raw.clip(q_lo, q_hi)
    log(f"T1 frac_feo_extraido: recorte a cuantiles [{q_lo:.3f}, {q_hi:.3f}] (raw n validas={frac_raw.notna().sum()})")

    frac_k = frac_raw.clip(0.005, 0.95)
    red["k_app_feo"] = -np.log(1 - frac_k) / red["duracion_plan_min"]
    log(f"T2 k_app_feo: n validas={red['k_app_feo'].notna().sum()}")

    masa_prev = red["masa_escoria_est_kg_prev"]
    masa_t = red["masa_escoria_est_kg"]
    red["delta_feo_corr"] = red["ley_feo_escoria_pct"] - red["ley_feo_escoria_pct_prev"] * (masa_prev / masa_t)
    check_kg = -masa_t / 100 * red["delta_feo_corr"]
    diff = (check_kg - red["feo_extraido_est_kg"]).abs()
    corr_check = check_kg.corr(red["feo_extraido_est_kg"])
    log(f"T5 verificacion algebraica vs T0: diff abs media={diff.mean():.4f} kg (max={diff.max():.4f}), "
        f"correlacion={corr_check:.6f} (deberia ser identidad exacta salvo redondeo/NaN)")

    red = agregar_feo_suavizado(df, red)
    return red


FEATURES_BASE = list(mp3.FEATURES_V3_POR_DEFECTO["Reducción"]["feo_kg"])
FEATURES_PLUS = FEATURES_BASE + ["avance_reduccion_sn_prev", "interaccion_Sn_x_FeO_prev"]
FEATURE_SETS = {"feo_kg_set": FEATURES_BASE, "feo_kg_set_plus": FEATURES_PLUS}


def oof_lockbox(red: pd.DataFrame, features: list[str], target_col: str, batches_dev: set, batches_lockbox: set,
                 loss: str = "squared_error", n_splits: int = N_SPLITS):
    d = red.dropna(subset=features + [target_col]).copy()
    d_dev = d[d["Batch"].isin(batches_dev)].reset_index(drop=True)
    d_lb = d[d["Batch"].isin(batches_lockbox)].reset_index(drop=True)
    if len(d_dev) == 0:
        return d_dev, d_lb
    Xd, yd, gd = d_dev[features].to_numpy(), d_dev[target_col].to_numpy(), d_dev["Batch"].to_numpy()
    n_batches = d_dev["Batch"].nunique()
    splits = min(n_splits, n_batches)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits=splits).split(Xd, yd, gd):
        m = make_hgb(loss)
        m.fit(Xd[tr], yd[tr])
        oof[te] = m.predict(Xd[te])
    d_dev["_pred"] = oof
    modelo_full = make_hgb(loss)
    modelo_full.fit(Xd, yd)
    if len(d_lb):
        d_lb["_pred"] = modelo_full.predict(d_lb[features].to_numpy())
    return d_dev, d_lb


def evaluar_senal(nombre: str, target_col: str, to_kg, red: pd.DataFrame, features: list[str],
                   batches_dev: set, batches_lockbox: set, loss: str = "squared_error") -> dict:
    d_dev, d_lb = oof_lockbox(red, features, target_col, batches_dev, batches_lockbox, loss=loss)
    if len(d_dev) == 0:
        return dict(target=nombre, n_dev=0)
    r2_oof = r2_score(d_dev[target_col], d_dev["_pred"])
    mae_oof = mean_absolute_error(d_dev[target_col], d_dev["_pred"])
    r2_lb = mae_lb = np.nan
    if len(d_lb):
        r2_lb = r2_score(d_lb[target_col], d_lb["_pred"])
        mae_lb = mean_absolute_error(d_lb[target_col], d_lb["_pred"])

    pred_kg_dev = np.asarray(to_kg(d_dev["_pred"].to_numpy(), d_dev), dtype=float)
    actual_kg_dev = d_dev["feo_extraido_est_kg"].to_numpy(dtype=float)
    mask = np.isfinite(pred_kg_dev) & np.isfinite(actual_kg_dev)
    r2_oof_kg = r2_score(actual_kg_dev[mask], pred_kg_dev[mask]) if mask.sum() > 5 else np.nan
    mae_oof_kg = mean_absolute_error(actual_kg_dev[mask], pred_kg_dev[mask]) if mask.sum() > 5 else np.nan

    r2_lb_kg = mae_lb_kg = np.nan
    if len(d_lb):
        pred_kg_lb = np.asarray(to_kg(d_lb["_pred"].to_numpy(), d_lb), dtype=float)
        actual_kg_lb = d_lb["feo_extraido_est_kg"].to_numpy(dtype=float)
        mask_lb = np.isfinite(pred_kg_lb) & np.isfinite(actual_kg_lb)
        if mask_lb.sum() > 5:
            r2_lb_kg = r2_score(actual_kg_lb[mask_lb], pred_kg_lb[mask_lb])
            mae_lb_kg = mean_absolute_error(actual_kg_lb[mask_lb], pred_kg_lb[mask_lb])

    return dict(target=nombre, n_dev=len(d_dev), n_lockbox=len(d_lb),
                r2_oof=r2_oof, mae_oof=mae_oof, r2_lockbox=r2_lb, mae_lockbox=mae_lb,
                r2_oof_en_kg=r2_oof_kg, mae_oof_en_kg=mae_oof_kg,
                r2_lockbox_en_kg=r2_lb_kg, mae_lockbox_en_kg=mae_lb_kg)


def seccion2(red: pd.DataFrame, batches_dev: set, batches_lockbox: set) -> pd.DataFrame:
    log("=" * 100)
    log("2) Targets alternativos de sobre-reduccion (mismo HGB, OOF GroupKFold(5) + lockbox)")
    log("=" * 100)

    signals = [
        ("T0_feo_extraido_kg", "feo_extraido_est_kg", (lambda p, d: p), "squared_error"),
        ("T1_frac_feo_extraido", "frac_feo_extraido",
         (lambda p, d: p * d["feo_inventario_escoria_est_kg_prev"].to_numpy()), "squared_error"),
        ("T2_k_app_feo", "k_app_feo",
         (lambda p, d: (1 - np.exp(-p * d["duracion_plan_min"].to_numpy())) *
                       d["feo_inventario_escoria_est_kg_prev"].to_numpy()), "squared_error"),
        ("T3_feo_extraido_suav_kg", "feo_extraido_suav_kg", (lambda p, d: p), "squared_error"),
        ("T4_feo_extraido_kg_abs_error", "feo_extraido_est_kg", (lambda p, d: p), "absolute_error"),
        ("T5_delta_feo_corr_pts", "delta_feo_corr",
         (lambda p, d: -p * d["masa_escoria_est_kg"].to_numpy() / 100), "squared_error"),
    ]

    filas = []
    for set_name, feats in FEATURE_SETS.items():
        feats_seguras = fe.filtrar_features_seguras(feats)
        if feats_seguras != feats:
            log(f"AVISO: features inseguras descartadas en {set_name}: {set(feats) - set(feats_seguras)}")
        for nombre, target_col, to_kg, loss in signals:
            try:
                r = evaluar_senal(nombre, target_col, to_kg, red, feats, batches_dev, batches_lockbox, loss=loss)
            except Exception as exc:  # pragma: no cover
                log(f"ERROR en {nombre}/{set_name}: {exc}")
                continue
            r["set"] = set_name
            filas.append(r)
            if r.get("n_dev", 0) > 0:
                log(f"{nombre:30s} set={set_name:16s} n_dev={r['n_dev']:4d} n_lb={r['n_lockbox']:3d} | "
                    f"nativo: R2_oof={r['r2_oof']:+.3f} MAE_oof={r['mae_oof']:.2f} "
                    f"R2_lb={r['r2_lockbox']:+.3f} MAE_lb={r['mae_lockbox']:.2f} || "
                    f"en kg: R2_oof={r['r2_oof_en_kg']:+.3f} MAE_oof={r['mae_oof_en_kg']:.1f} "
                    f"R2_lb={r['r2_lockbox_en_kg']:+.3f} MAE_lb={r['mae_lockbox_en_kg']:.1f}")

    tabla2 = pd.DataFrame(filas)
    tabla2.to_csv(OUT / "15_senales_feo.csv", index=False)
    log("Guardado 15_senales_feo.csv")
    return tabla2


# =============================================================================
# 3) Agregado por fase (batch-level)
# =============================================================================

def seccion3(df: pd.DataFrame, red: pd.DataFrame, batches_dev: set, batches_lockbox: set):
    log("=" * 100)
    log("3) Agregado por fase (batch-level): FeO_reducido_total_R_kg vs modelo de escalon")
    log("=" * 100)

    fus_last = df.loc[(df.fase_proceso == "Fusión") & (df.orden_escalon_fase == 6)].set_index("Batch")
    g_r = red.groupby("Batch")

    batch = pd.DataFrame(index=pd.Index(sorted(red["Batch"].unique()), name="Batch"))
    batch["FeO_reducido_total_R_kg"] = g_r["feo_extraido_est_kg"].sum(min_count=1)
    batch["FeO_reducido_total_R_suav_kg"] = g_r["feo_extraido_suav_kg"].sum(min_count=1)
    batch["sn_inv_fin_fusion_kg"] = fus_last["sn_inventario_escoria_est_kg"]
    batch["feo_inv_fin_fusion_kg"] = fus_last["feo_inventario_escoria_est_kg"]
    batch["masa_escoria_fin_fusion_kg"] = fus_last["masa_escoria_est_kg"]
    batch["feo_pct_fin_fusion"] = fus_last["ley_feo_escoria_pct"]
    batch["sio2_pct_fin_fusion"] = fus_last["ley_sio2_escoria_pct"]
    batch["cao_pct_fin_fusion"] = fus_last["ley_cao_escoria_pct"]
    batch["temp_fin_fusion_c"] = fus_last["temperatura_horno_celsius"]
    batch["carbon_total_R_kg"] = g_r["feed_Carbon_kgh"].sum(min_count=1)
    gn_step = red["tasa_gn_nm3_min"] * red["duracion_plan_min"]
    batch["gn_total_R_nm3"] = gn_step.groupby(red["Batch"]).sum(min_count=1)
    batch["exceso_o2_medio_R_pct"] = g_r["exceso_o2_combustion_pct"].mean()
    batch["duracion_total_R_min"] = g_r["duracion_plan_min"].sum(min_count=1)
    batch = batch.reset_index()
    batch["set_"] = np.where(batch["Batch"].isin(batches_dev), "DEV", "LOCKBOX")

    feats_batch = ["sn_inv_fin_fusion_kg", "feo_inv_fin_fusion_kg", "masa_escoria_fin_fusion_kg",
                   "feo_pct_fin_fusion", "sio2_pct_fin_fusion", "cao_pct_fin_fusion", "temp_fin_fusion_c",
                   "carbon_total_R_kg", "gn_total_R_nm3", "exceso_o2_medio_R_pct", "duracion_total_R_min"]

    filas = []
    for target_batch, etiqueta in [("FeO_reducido_total_R_kg", "crudo"), ("FeO_reducido_total_R_suav_kg", "suavizado")]:
        d = batch.dropna(subset=feats_batch + [target_batch]).copy()
        d_dev = d[d["set_"] == "DEV"].reset_index(drop=True)
        d_lb = d[d["set_"] == "LOCKBOX"].reset_index(drop=True)
        X_dev, y_dev = d_dev[feats_batch].to_numpy(), d_dev[target_batch].to_numpy()
        X_lb, y_lb = d_lb[feats_batch].to_numpy(), d_lb[target_batch].to_numpy()

        kf = KFold(n_splits=min(N_SPLITS, len(d_dev)), shuffle=True, random_state=SEED)
        oof_hgb = np.full(len(d_dev), np.nan)
        oof_ridge = np.full(len(d_dev), np.nan)
        for tr, te in kf.split(X_dev):
            m = hgb_pequeno()
            m.fit(X_dev[tr], y_dev[tr])
            oof_hgb[te] = m.predict(X_dev[te])
            sc = StandardScaler().fit(X_dev[tr])
            r = Ridge(alpha=1.0).fit(sc.transform(X_dev[tr]), y_dev[tr])
            oof_ridge[te] = r.predict(sc.transform(X_dev[te]))

        modelo_hgb_full = hgb_pequeno().fit(X_dev, y_dev)
        sc_full = StandardScaler().fit(X_dev)
        modelo_ridge_full = Ridge(alpha=1.0).fit(sc_full.transform(X_dev), y_dev)

        for modelo_nombre, oof_pred, lb_pred in [
            ("HGB_pequeno", oof_hgb, modelo_hgb_full.predict(X_lb) if len(X_lb) else np.array([])),
            ("Ridge", oof_ridge, modelo_ridge_full.predict(sc_full.transform(X_lb)) if len(X_lb) else np.array([])),
        ]:
            r2_oof = r2_score(y_dev, oof_pred)
            mae_oof = mean_absolute_error(y_dev, oof_pred)
            r2_lb = np.nan
            mae_lb = np.nan
            if len(y_lb):
                r2_lb = r2_score(y_lb, lb_pred)
                mae_lb = mean_absolute_error(y_lb, lb_pred)
            filas.append(dict(target=etiqueta, modelo=modelo_nombre, n_dev=len(d_dev), n_lockbox=len(d_lb),
                               r2_oof=r2_oof, mae_oof=mae_oof, r2_lockbox=r2_lb, mae_lockbox=mae_lb))
            log(f"batch-agregado {etiqueta:10s} {modelo_nombre:12s} n_dev={len(d_dev):3d} n_lb={len(d_lb):3d} "
                f"R2_oof={r2_oof:+.3f} MAE_oof={mae_oof:.1f} R2_lb={r2_lb:+.3f} MAE_lb={mae_lb:.1f}")

    # Correlacion a nivel batch con sn_en_dross_fe_batch_t (crudo vs suavizado)
    dross_batch = df.groupby("Batch")["sn_en_dross_fe_batch_t"].first()
    corr_df = batch.set_index("Batch")[["FeO_reducido_total_R_kg", "FeO_reducido_total_R_suav_kg"]].join(dross_batch).dropna()
    p_crudo = corr_df["FeO_reducido_total_R_kg"].corr(corr_df["sn_en_dross_fe_batch_t"])
    s_crudo = corr_df["FeO_reducido_total_R_kg"].corr(corr_df["sn_en_dross_fe_batch_t"], method="spearman")
    p_suav = corr_df["FeO_reducido_total_R_suav_kg"].corr(corr_df["sn_en_dross_fe_batch_t"])
    s_suav = corr_df["FeO_reducido_total_R_suav_kg"].corr(corr_df["sn_en_dross_fe_batch_t"], method="spearman")
    log(f"Correlacion con sn_en_dross_fe_batch_t (n={len(corr_df)}): "
        f"crudo pearson={p_crudo:.3f} spearman={s_crudo:.3f} | suavizado pearson={p_suav:.3f} spearman={s_suav:.3f}")
    filas.append(dict(target="crudo", modelo="corr_con_sn_dross_pearson", n_dev=len(corr_df), r2_oof=p_crudo))
    filas.append(dict(target="crudo", modelo="corr_con_sn_dross_spearman", n_dev=len(corr_df), r2_oof=s_crudo))
    filas.append(dict(target="suavizado", modelo="corr_con_sn_dross_pearson", n_dev=len(corr_df), r2_oof=p_suav))
    filas.append(dict(target="suavizado", modelo="corr_con_sn_dross_spearman", n_dev=len(corr_df), r2_oof=s_suav))

    tabla3 = pd.DataFrame(filas)
    tabla3.to_csv(OUT / "15_agregado_batch.csv", index=False)
    log("Guardado 15_agregado_batch.csv")
    return tabla3, batch


# =============================================================================
# 4) Re-estimacion de lambda_Fe
# =============================================================================

def ajustar_ols(nombre: str, data: pd.DataFrame, formula: str, filas: list, contrast: str | None = None):
    d = data.dropna()
    if len(d) < 20:
        log(f"[{nombre}] n insuficiente ({len(d)}), se omite")
        return None
    m = smf.ols(formula, data=d).fit(cov_type="HC3")
    ci = m.conf_int()
    for var in m.params.index:
        if var == "Intercept":
            continue
        filas.append(dict(variante=nombre, n=int(m.nobs), variable=var, coef=m.params[var], se=m.bse[var],
                           ci_lo=ci.loc[var].iloc[0], ci_hi=ci.loc[var].iloc[1], pvalue=m.pvalues[var],
                           r2=m.rsquared))
    if contrast:
        t = m.t_test(contrast)
        ci_t = np.asarray(t.conf_int()).ravel()
        filas.append(dict(variante=nombre, n=int(m.nobs), variable=contrast, coef=float(np.asarray(t.effect).ravel()[0]),
                           se=float(np.asarray(t.sd).ravel()[0]), ci_lo=float(ci_t[0]), ci_hi=float(ci_t[1]),
                           pvalue=float(np.asarray(t.pvalue).ravel()[0]), r2=m.rsquared))
    log(f"[{nombre}] n={int(m.nobs)} R2={m.rsquared:.3f}")
    for line in str(m.summary().tables[1]).splitlines():
        log("    " + line)
    if contrast:
        log(f"    contraste {contrast}: {t}")
    return m


def seccion4(df: pd.DataFrame, red: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    log("=" * 100)
    log("4) Re-estimacion de lambda_Fe (OLS entre batches, HC3)")
    log("=" * 100)
    g = df.groupby("Batch")
    b = pd.DataFrame({
        "sn_dross_t": g["sn_en_dross_fe_batch_t"].first(),
        "sn_metal_t": g["sn_en_metal_crudo_batch_t"].first(),
        "sn_polvo_t": g["sn_en_polvo_fundicion_batch_t"].first(),
        "sn_cargado_total_t": g["feed_Sn_kgf"].sum() / 1000,
        "ley_sn_conc_batch_pct": g["ley_sn_conc_batch_pct"].first(),
        "espesor_ladrillo_norm_mm": g["espesor_ladrillo_norm_mm"].first(),
        "dross_fe_alimentado_total_t": g["feed_dross_Fe_kgh"].sum() / 1000,
        "rendimiento_proxy_batch": g["rendimiento_proxy_batch"].first(),
    })
    b["temp_media_R_c"] = df.loc[df.fase_proceso == "Reducción"].groupby("Batch")["temperatura_horno_celsius"].mean()
    b = b.join(batch.set_index("Batch")[["FeO_reducido_total_R_kg", "FeO_reducido_total_R_suav_kg"]])
    b["FeO_R01_kg"] = red.loc[red.orden_escalon_fase.isin([0, 1])].groupby("Batch")["feo_extraido_est_kg"].sum(min_count=1)
    b["FeO_R23_kg"] = red.loc[red.orden_escalon_fase.isin([2, 3])].groupby("Batch")["feo_extraido_est_kg"].sum(min_count=1)
    b["frac_sn_dross"] = b["sn_dross_t"] / (b["sn_metal_t"] + b["sn_dross_t"] + b["sn_polvo_t"])

    b["FeO_total_R_t"] = b["FeO_reducido_total_R_kg"] / 1000
    b["FeO_total_R_suav_t"] = b["FeO_reducido_total_R_suav_kg"] / 1000
    b["FeO_R01_t"] = b["FeO_R01_kg"] / 1000
    b["FeO_R23_t"] = b["FeO_R23_kg"] / 1000

    log(f"Batches con FeO_total_R y sn_dross validos: {b[['sn_dross_t', 'FeO_total_R_t']].dropna().shape[0]} de {len(b)}")

    trim_mask = b["FeO_total_R_t"].between(*b["FeO_total_R_t"].quantile([0.02, 0.98]))
    bt = b[trim_mask].copy()
    log(f"Trim [.02,.98] sobre FeO_total_R_t: n={len(bt)} de {len(b)}")

    filas: list = []
    controles = "sn_cargado_total_t + ley_sn_conc_batch_pct + espesor_ladrillo_norm_mm + dross_fe_alimentado_total_t + temp_media_R_c"
    ajustar_ols("a_crudo_simple", bt, "sn_dross_t ~ FeO_total_R_t", filas)
    ajustar_ols("a_crudo_controles", bt, f"sn_dross_t ~ FeO_total_R_t + {controles}", filas)
    ajustar_ols("b_suavizado_simple", bt, "sn_dross_t ~ FeO_total_R_suav_t", filas)
    ajustar_ols("b_suavizado_controles", bt, f"sn_dross_t ~ FeO_total_R_suav_t + {controles}", filas)
    ajustar_ols("c_R01_R23_simple", bt, "sn_dross_t ~ FeO_R01_t + FeO_R23_t", filas,
                contrast="FeO_R23_t - FeO_R01_t = 0")
    ajustar_ols("c_R01_R23_controles", bt, f"sn_dross_t ~ FeO_R01_t + FeO_R23_t + {controles}", filas,
                contrast="FeO_R23_t - FeO_R01_t = 0")
    ajustar_ols("d_frac_sn_dross_vs_FeO", bt, "frac_sn_dross ~ FeO_total_R_t", filas)
    ajustar_ols("e_rendimiento_vs_FeO", bt, "rendimiento_proxy_batch ~ FeO_total_R_t", filas)

    for yy in ["frac_sn_dross", "rendimiento_proxy_batch"]:
        dd = bt[[yy, "FeO_total_R_t"]].dropna()
        rho_p = dd[yy].corr(dd["FeO_total_R_t"])
        rho_s = dd[yy].corr(dd["FeO_total_R_t"], method="spearman")
        log(f"{yy} vs FeO_total_R_t: pearson={rho_p:.3f} spearman={rho_s:.3f} (n={len(dd)})")

    tabla4 = pd.DataFrame(filas)
    tabla4.to_csv(OUT / "15_lambda_fe.csv", index=False)
    log("Guardado 15_lambda_fe.csv")
    return b


# =============================================================================
# 5) Selectividad marginal empirica
# =============================================================================

def seccion5(red: pd.DataFrame) -> pd.DataFrame:
    log("=" * 100)
    log("5) Selectividad marginal empirica por avance de la reduccion")
    log("=" * 100)
    d = red[["avance_reduccion_sn_prev", "feo_extraido_est_kg", "sn_extraido_est_kg"]].dropna().copy()
    try:
        d["bin"] = pd.qcut(d["avance_reduccion_sn_prev"], 5, labels=False, duplicates="drop")
    except Exception as exc:
        log(f"qcut fallo ({exc}), usando cut uniforme")
        d["bin"] = pd.cut(d["avance_reduccion_sn_prev"], 5, labels=False)

    filas = []
    for b_, sub in d.groupby("bin"):
        if len(sub) < 20:
            log(f"bin {b_} con n={len(sub)} < 20, se omite")
            continue
        m = smf.ols("feo_extraido_est_kg ~ sn_extraido_est_kg", data=sub).fit(cov_type="HC3")
        ci = m.conf_int().loc["sn_extraido_est_kg"]
        fila = dict(bin=int(b_), n=len(sub), avance_min=sub["avance_reduccion_sn_prev"].min(),
                    avance_max=sub["avance_reduccion_sn_prev"].max(), avance_medio=sub["avance_reduccion_sn_prev"].mean(),
                    pendiente_feo_por_sn=m.params["sn_extraido_est_kg"], ci_lo=ci.iloc[0], ci_hi=ci.iloc[1],
                    pvalue=m.pvalues["sn_extraido_est_kg"], r2=m.rsquared)
        filas.append(fila)
        log(f"bin {int(b_)} avance[{fila['avance_min']:.2f},{fila['avance_max']:.2f}] n={len(sub)} "
            f"pendiente(dFeO/dSn)={fila['pendiente_feo_por_sn']:.3f} [{fila['ci_lo']:.3f},{fila['ci_hi']:.3f}] "
            f"p={fila['pvalue']:.4f} R2={fila['r2']:.3f}")

    tabla5 = pd.DataFrame(filas)
    tabla5.to_csv(OUT / "15_selectividad_marginal.csv", index=False)
    log("Guardado 15_selectividad_marginal.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        yerr = np.vstack([tabla5["pendiente_feo_por_sn"] - tabla5["ci_lo"], tabla5["ci_hi"] - tabla5["pendiente_feo_por_sn"]])
        ax.errorbar(tabla5["avance_medio"], tabla5["pendiente_feo_por_sn"], yerr=yerr, fmt="o-", capsize=4, color="#c0392b")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("avance_reduccion_sn_prev (valor medio del bin)")
        ax.set_ylabel("pendiente local dFeO/dSn extraidos [kg FeO / kg Sn] (IC95% HC3)")
        ax.set_title("Selectividad marginal Sn/FeO por avance de la reduccion")
        plt.tight_layout()
        fig.savefig(FIGDIR / "15_selectividad_marginal.png", dpi=130)
        plt.close(fig)
        log("Guardada figura figs_15/15_selectividad_marginal.png")
    except Exception as exc:  # pragma: no cover
        log(f"No se pudo generar la figura: {exc}")

    return tabla5


# =============================================================================
# main
# =============================================================================

def main() -> None:
    t0 = time.time()
    log("#" * 100)
    log("Experimento 15 -- senal de sobre-reduccion de FeO en Reduccion y re-estimacion de lambda_Fe")
    log("#" * 100)

    df = pd.read_pickle(RAIZ / "experimentos" / "cache" / "df_v3.pkl")
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
    base = mp.dataset_base_modelo(df)
    red = base[base["fase_proceso"] == "Reducción"].copy()
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    log(f"Dataset: {df.shape}, {df['Batch'].nunique()} batches. Reduccion: {len(red)} filas, "
        f"{red['Batch'].nunique()} batches ({len(batches_dev)} DEV / {len(batches_lockbox)} lockbox)")
    flush_log()

    seccion1(red, batches_dev)
    flush_log()

    red = construir_targets_alternativos(df, red)
    seccion2(red, batches_dev, batches_lockbox)
    flush_log()

    tabla3, batch = seccion3(df, red, batches_dev, batches_lockbox)
    flush_log()

    seccion4(df, red, batch)
    flush_log()

    seccion5(red)
    flush_log()

    log(f"TOTAL tiempo: {time.time() - t0:.1f}s")
    flush_log()


if __name__ == "__main__":
    main()
