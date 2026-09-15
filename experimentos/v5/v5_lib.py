"""v5_lib — librería compartida de la iteración 7 (prescriptor v5, 2026-09-14/15).

Punto de partida: v4 (`modelo_predictivo_v4.py`, PLM por fase), targets ganadores de la iteración 5
(`experimentos/search_targets`: SDI, ln IRF en Reducción, IRF fin de Fusión) y KPI refinado de la
iteración 6 (`feature_engineering.construir_resumen_batch_balance_global`).

Qué provee (todo con clasificación anti-fuga explícita en CLASIFICACION_V5):

1) `cargar_df()` -> df de escalones = cache v3 + columnas v4 (Cx_sn_v4, Cx_av_v4) + bloque térmico b6 (solo
   como DERIVED, nunca como STATE) + targets v5 + palancas v5.

2) Targets v5 por escalón (log, adimensionales; la masa del trazador CaO se cancela en cada cociente):
       v5_ln_sn_dep  = ln(Sn_disp / Sn_inv[t])          Sn_disp = Sn_inv[t-1] + Sn alimentado en t
                       (Reducción: Sn alimentado ≈ 0 => ln(Sn_inv[t-1]/Sn_inv[t]) = componente Sn del SDI)
                       agotamiento logarítmico del Sn de la escoria (cinética de 1er orden: = k·dt)
       v5_ln_feo_ret = ln(FeO_inv[t] / FeO_inv[t-1])   (solo Reducción: sin carga de Fe; <= 0 si se reduce FeO)
                       retención logarítmica de FeO (componente FeO del SDI; -ln_feo_ret = fracción de FeO metalizada)
       v5_sdi_w10    = v5_ln_sn_dep + 10·v5_ln_feo_ret  (SDI, ganador iteración 5; w=10 elegido en DEV)
       v5_lnirf      = ln(%FeO / (%SiO2 + %Al2O3 + %CaO)) al cierre del escalón (ambas fases)
       v5_d_lnirf    = v5_lnirf[t] - v5_lnirf[t-1]      (cambio de la matriz durante el escalón; en Reducción
                       ≈ v5_ln_feo_ret + término de cierre composicional)
   Validez: inventarios > umbral (Sn 20 kg, FeO 100 kg) y no primer escalón del batch. Todas TARGET.

3) Palancas v5 (acción del escalón × estado previo, seguras a tiempo de decisión):
       v5_dosis_C_sn   = feed_Carbon_kgh / Sn_disp            kg C por kg Sn disponible (dosis específica de reductor;
                         estequiométrico SnO2+2C: 0.2024 kg C/kg Sn); NaN si Sn_disp < 200 kg
       v5_ln_dosis_C   = ln(v5_dosis_C_sn)
       v5_exceso_C_kg  = feed_Carbon_kgh - 0.2024·Sn_disp     carbón por encima de la demanda estequiométrica del Sn
       v5_exceso_C_pos = max(v5_exceso_C_kg, 0)               (reductor "sobrante" disponible para FeO + C -> Fe + CO)
       v5_C_x_avance   = feed_Carbon_kgh · avance_reduccion_sn_prev  (= Cx_av_v4 en kg del escalón)
       v5_gn_esp_nm3_t = volumen GN del escalón / masa de escoria previa [Nm3/t] (intensidad térmica específica)
   Además se reutilizan `Cx_sn_v4`, `Cx_av_v4`, `exceso_o2_combustion_pct`, `relacion_C_Sn_carga` y las primitivas
   `tasa_feed_Carbon_kg_min`, `tasa_gn_nm3_min`, `tasa_o2_nm3_min`, `tasa_aire_nm3_min`.

4) `construir_batch_v5(df)` -> una fila por batch con KPIs (`recuperacion_refinada_pct` = KPI principal de esta
   iteración, `rendimiento_proxy_batch`, `recuperacion_real_pct`, f_metal/f_dross/f_polvo), controles de contexto,
   agregados coherentes de los targets (sumas telescópicas de las componentes log, media ponderada por tiempo del
   ln IRF de Reducción, ln IRF fin de Fusión, ln IRF al cierre de F0 = heel) y el KPI del batch siguiente.

5) `bateria_batch(batch, col, kpi)` -> Spearman con IC bootstrap en DEV/lockbox/total, OLS HC3 por sd con controles
   y tendencia, monotonía por quintiles. `evaluar_plm(...)` -> R2 OOF (GroupKFold por batch en DEV) y lockbox +
   tabla de theta con IC bootstrap cluster (usa `modelo_predictivo_v4.ModeloPLM`).

Convenciones: OOF GroupKFold(5) por Batch sobre DEV = criterio de selección; lockbox (últimos 17.5 % de batches)
solo reporte. Python: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4`, sin joblib.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos" / "balances"))
import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402

warnings.filterwarnings("ignore")
CACHE = RAIZ / "experimentos" / "cache" / "df_v3.pkl"
FASES = ["Fusión", "Reducción"]
KPI_PRINCIPAL = "recuperacion_refinada_pct"
KPIS = ["recuperacion_refinada_pct", "rendimiento_proxy_batch", "recuperacion_real_pct", "f_metal", "f_dross", "f_polvo"]
CONTROLES_BATCH = ["feed_sn_total_t", "ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm", "frac_carga_secundaria_F",
                   "feed_dross_Fe_total_t", "idx_cronologico"]
C_ESTEQ_POR_SN = 2 * 12.011 / 118.71   # 0.2024 kg C / kg Sn
UMBRAL_SN_KG, UMBRAL_FEO_KG, UMBRAL_SN_DISP_DOSIS = 20.0, 100.0, 200.0
W_SDI_ITER5 = 10.0
PALANCAS_PRIMITIVAS = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"]

# --------------------------------------------------------------------------- estado (STATE + CONTEXT)
ESTADO_R_FULL = list(dict.fromkeys(mp3.STATE_V3["Reducción"] + mp3.CONTEXT_V3["Reducción"]))
ESTADO_F_FULL = list(dict.fromkeys(mp3.STATE_V3["Fusión"] + mp3.CONTEXT_V3["Fusión"]))
# curado (teoría): composición y matriz, térmico, lanza previa, inventarios/avance, posición en la fase, campaña, ley
ESTADO_R_CURADO = ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev",
                   "indice_irf_prev", "basicidad_B2_prev", "log_ratio_sn_feo_prev",
                   "temperatura_horno_celsius_prev", "grad_temperatura_horno_celsius_prev", "termocupla_media_celsius_prev",
                   "posicion_vertical_lanza_mm_prev", "tiro_horno_pct_prev",
                   "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev", "feo_inventario_escoria_est_kg_prev",
                   "avance_reduccion_sn_prev", "orden_escalon_fase", "tiempo_fase", "espesor_ladrillo_norm_mm",
                   "ley_sn_carga_batch_pct"]
ESTADO_F_CURADO = ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev",
                   "indice_irf_prev", "basicidad_B2_prev",
                   "temperatura_horno_celsius_prev", "grad_temperatura_horno_celsius_prev", "termocupla_media_celsius_prev",
                   "posicion_vertical_lanza_mm_prev", "tiro_horno_pct_prev",
                   "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev", "feo_inventario_escoria_est_kg_prev",
                   "cum_feed_Sn_kg_prev", "cum_feed_CaO_kg_prev", "orden_escalon_fase", "tiempo_fase", "espesor_ladrillo_norm_mm",
                   "ley_sn_conc_batch_pct", "ley_sn_carga_batch_pct",
                   # plan de carga del escalón (control NO optimizable, entra en g(S) como en v4)
                   "feed_Sn_kgf", "tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min"]

CLASIFICACION_V5: dict[str, str] = {
    "v5_sn_disp": "DERIVED (A_t x S_prev)",
    "v5_ln_sn_dep": "TARGET", "v5_ln_feo_ret": "TARGET", "v5_sdi_w10": "TARGET", "v5_lnirf": "TARGET", "v5_d_lnirf": "TARGET",
    "v5_lnirf_prev": "STATE",
    "v5_dosis_C_sn": "DERIVED (A_t x S_prev)", "v5_ln_dosis_C": "DERIVED (A_t x S_prev)", "v5_exceso_C_kg": "DERIVED (A_t x S_prev)",
    "v5_exceso_C_pos": "DERIVED (A_t x S_prev)", "v5_C_x_avance": "DERIVED (A_t x S_prev)", "v5_gn_esp_nm3_t": "DERIVED (A_t x S_prev)",
    "v5_valido": "DIAG",
}
PALANCAS_V5 = [c for c, r in CLASIFICACION_V5.items() if r == "DERIVED (A_t x S_prev)" and c != "v5_sn_disp"]
TARGETS_V5 = [c for c, r in CLASIFICACION_V5.items() if r == "TARGET"]


def es_segura(nombre: str) -> bool:
    """Anti-fuga: seguras = las de feature_engineering + v4 (Cx_*) + b6 seguras + STATE/DERIVED de v5."""
    if fe.es_feature_segura_para_prescripcion(nombre):
        return True
    if nombre in ("Cx_sn_v4", "Cx_av_v4"):
        return True
    rol = CLASIFICACION_V5.get(nombre)
    if rol is not None:
        return rol in ("STATE", "DERIVED (A_t x S_prev)")
    try:
        import bal_features as bal  # noqa: WPS433
        return nombre in bal.SEGURAS_V6
    except Exception:  # noqa: BLE001
        return False


def filtrar_seguras(nombres: list[str]) -> list[str]:
    return [n for n in nombres if es_segura(n)]


# --------------------------------------------------------------------------- construcción del df
def agregar_targets_v5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    sn_prev, sn = df["sn_inventario_escoria_est_kg_prev"], df["sn_inventario_escoria_est_kg"]
    feo_prev, feo = df["feo_inventario_escoria_est_kg_prev"], df["feo_inventario_escoria_est_kg"]
    sn_disp = sn_prev + df["feed_Sn_kgf"].fillna(0)
    ok = sn_prev.gt(UMBRAL_SN_KG) & sn.gt(UMBRAL_SN_KG) & feo_prev.gt(UMBRAL_FEO_KG) & feo.gt(UMBRAL_FEO_KG) & ~df["es_primer_escalon_batch"]
    red = df["fase_proceso"].eq("Reducción")
    n = {}
    n["v5_sn_disp"] = sn_disp
    n["v5_valido"] = ok
    n["v5_ln_sn_dep"] = np.log(sn_disp / sn).where(ok & sn_disp.gt(UMBRAL_SN_KG))
    n["v5_ln_feo_ret"] = np.log(feo / feo_prev).where(ok & red)
    n["v5_sdi_w10"] = n["v5_ln_sn_dep"] + W_SDI_ITER5 * n["v5_ln_feo_ret"]
    irf = df["ley_feo_escoria_pct"] / (df["ley_sio2_escoria_pct"] + df["ley_al2o3_escoria_pct"] + df["ley_cao_escoria_pct"])
    lnirf = np.log(irf.where(irf > 0))
    n["v5_lnirf"] = lnirf
    lnirf_prev = lnirf.groupby(df["Batch"]).shift(1)
    n["v5_lnirf_prev"] = lnirf_prev.where(~df["es_primer_escalon_batch"])
    n["v5_d_lnirf"] = (lnirf - lnirf_prev).where(~df["es_primer_escalon_batch"])
    return pd.concat([df, pd.DataFrame(n, index=df.index)], axis=1)


def agregar_palancas_v5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c_kg = df["feed_Carbon_kgh"].fillna(0)
    sn_disp = df["v5_sn_disp"]
    dosis = (c_kg / sn_disp).where(sn_disp.gt(UMBRAL_SN_DISP_DOSIS))
    df["v5_dosis_C_sn"] = dosis
    df["v5_ln_dosis_C"] = np.log(dosis.where(dosis > 0))
    df["v5_exceso_C_kg"] = c_kg - C_ESTEQ_POR_SN * sn_disp
    df["v5_exceso_C_pos"] = df["v5_exceso_C_kg"].clip(lower=0)
    df["v5_C_x_avance"] = c_kg * df["avance_reduccion_sn_prev"]
    masa_prev = df["masa_escoria_est_kg_prev"]
    df["v5_gn_esp_nm3_t"] = (df["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0) / (masa_prev / 1000)).where(masa_prev.gt(1000))
    return df


def cargar_df(con_termicas: bool = True) -> pd.DataFrame:
    df = pd.read_pickle(CACHE)
    df = mp4.agregar_columnas_v4(df)
    if con_termicas:
        import bal_features as bal
        df = bal.agregar_balance_energia(df)
    df = agregar_targets_v5(df)
    df = agregar_palancas_v5(df)
    return df


def base_fase(df: pd.DataFrame, fase: str) -> pd.DataFrame:
    """Escalones modelables de la fase (sin el primer escalón del batch)."""
    d = mp.dataset_base_modelo(df)
    return d.loc[d["fase_proceso"] == fase].reset_index(drop=True)


# --------------------------------------------------------------------------- batch
def _twmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.gt(0)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def construir_batch_v5(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: KPIs (refinado, proxy, real, canales), controles, agregados v5 y KPI del batch siguiente."""
    dev, lockbox = mp.split_dev_lockbox(df)
    bal = fe.construir_resumen_batch_balance_global(df).set_index("Batch")
    filas = []
    for b, g in df.sort_values(["Batch", "fecha_inicio"]).groupby("Batch", sort=False):
        gf, gr = g[g["fase_proceso"] == "Fusión"], g[g["fase_proceso"] == "Reducción"]
        m_, d_, p_ = (float(g[c].iloc[0]) for c in ["sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"])
        tot = m_ + d_ + p_
        feed_sn_t = g["feed_Sn_kgf"].sum(min_count=1) / 1000
        dt_r = gr["delta_tiempo"]
        fila = dict(
            Batch=b, fecha_batch=g["fecha_inicio"].min(), es_lockbox=b in lockbox,
            rendimiento_proxy_batch=float(g["rendimiento_proxy_batch"].iloc[0]),
            recuperacion_refinada_pct=float(bal.loc[b, "recuperacion_refinada_pct"]) if b in bal.index else np.nan,
            sn_perdido_escoria_frac=float(bal.loc[b, "sn_perdido_escoria_frac"]) if b in bal.index else np.nan,
            recuperacion_real_pct=100 * m_ / feed_sn_t if feed_sn_t else np.nan,
            f_metal=m_ / tot if tot else np.nan, f_dross=d_ / tot if tot else np.nan, f_polvo=p_ / tot if tot else np.nan,
            sn_metal_t=m_, sn_dross_t=d_, sn_polvo_t=p_,
            # controles
            feed_sn_total_t=feed_sn_t, ley_sn_conc_batch_pct=float(g["ley_sn_conc_batch_pct"].iloc[0]),
            espesor_ladrillo_norm_mm=float(g["espesor_ladrillo_norm_mm"].mean()),
            frac_carga_secundaria_F=float(gf["frac_carga_secundaria"].mean()) if len(gf) else np.nan,
            feed_dross_Fe_total_t=g["feed_dross_Fe_kgh"].sum(min_count=1) / 1000,
            T_media_F=float(gf["temperatura_horno_celsius"].mean()) if len(gf) else np.nan,
            T_media_R=float(gr["temperatura_horno_celsius"].mean()) if len(gr) else np.nan,
            # agregados v5 de Reducción (telescópicos / intensivos)
            R_sum_ln_sn_dep=float(gr["v5_ln_sn_dep"].sum(min_count=1)) if gr["v5_ln_sn_dep"].notna().any() else np.nan,
            R_sum_ln_feo_ret=float(gr["v5_ln_feo_ret"].sum(min_count=1)) if gr["v5_ln_feo_ret"].notna().any() else np.nan,
            R_n_validos=int(gr["v5_sdi_w10"].notna().sum()),
            R_lnirf_twmean=_twmean(gr["v5_lnirf"], dt_r) if len(gr) else np.nan,
            R_lnirf_fin=float(gr["v5_lnirf"].dropna().iloc[-1]) if gr["v5_lnirf"].notna().any() else np.nan,
            # agregados v5 de Fusión
            F_lnirf_F0=float(gf["v5_lnirf"].dropna().iloc[0]) if gf["v5_lnirf"].notna().any() else np.nan,
            F_lnirf_fin=float(gf["v5_lnirf"].dropna().iloc[-1]) if gf["v5_lnirf"].notna().any() else np.nan,
            F_sum_d_lnirf=float(gf["v5_d_lnirf"].sum(min_count=1)) if gf["v5_d_lnirf"].notna().any() else np.nan,
            F_sum_ln_sn_dep=float(gf["v5_ln_sn_dep"].sum(min_count=1)) if gf["v5_ln_sn_dep"].notna().any() else np.nan,
            F_sn_ext_kg=float(gf["sn_extraido_est_kg"].sum(min_count=1)) if gf["sn_extraido_est_kg"].notna().any() else np.nan,
        )
        fila["R_sdi_w10"] = fila["R_sum_ln_sn_dep"] + W_SDI_ITER5 * fila["R_sum_ln_feo_ret"]
        filas.append(fila)
    out = pd.DataFrame(filas).set_index("Batch").replace([np.inf, -np.inf], np.nan)
    out = out.sort_values("fecha_batch")
    out["idx_cronologico"] = np.arange(1, len(out) + 1)
    for k in KPIS:
        out[f"{k}_next"] = out[k].shift(-1)
    out["es_dev"] = ~out["es_lockbox"]
    return out


# --------------------------------------------------------------------------- batería de batch
def _spearman_ic(x: np.ndarray, y: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float, float]:
    from scipy import stats
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bs.append(stats.spearmanr(x[idx], y[idx])[0])
    return float(rho), float(p), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def bateria_batch(batch: pd.DataFrame, col: str, kpi: str = KPI_PRINCIPAL, controles: list[str] | None = None,
                  n_boot: int = 500) -> dict:
    """Spearman (IC bootstrap) en DEV/lockbox/total, OLS HC3 por +1 sd con controles + tendencia (DEV, total), quintiles DEV."""
    import statsmodels.api as sm
    controles = controles or CONTROLES_BATCH
    out = {"variable": col, "kpi": kpi}
    for nombre, sub in [("dev", batch[batch["es_dev"]]), ("lockbox", batch[batch["es_lockbox"]]), ("total", batch)]:
        d = sub[[col, kpi]].dropna()
        if len(d) < 15:
            continue
        rho, p, lo, hi = _spearman_ic(d[col].to_numpy(), d[kpi].to_numpy(), n_boot)
        out.update({f"rho_{nombre}": rho, f"p_{nombre}": p, f"ci_lo_{nombre}": lo, f"ci_hi_{nombre}": hi, f"n_{nombre}": len(d)})
        if nombre in ("dev", "total"):
            dd = sub[[col, kpi] + controles].dropna()
            X = dd[[col] + controles]
            X = (X - X.mean()) / X.std()
            res = sm.OLS(dd[kpi], sm.add_constant(X)).fit(cov_type="HC3")
            out.update({f"ols_{nombre}": float(res.params[col]), f"ols_p_{nombre}": float(res.pvalues[col]),
                        f"ols_lo_{nombre}": float(res.conf_int().loc[col, 0]), f"ols_hi_{nombre}": float(res.conf_int().loc[col, 1]),
                        f"ols_r2adj_{nombre}": float(res.rsquared_adj)})
    d = batch.loc[batch["es_dev"], [col, kpi]].dropna()
    if len(d) >= 50:
        q = pd.qcut(d[col], 5, labels=False, duplicates="drop")
        med = d.groupby(q)[kpi].mean()
        out["quintiles_crecientes"] = int((np.diff(med.to_numpy()) > 0).sum())
        out["dQ5Q1"] = float(med.iloc[-1] - med.iloc[0])
    return out


# --------------------------------------------------------------------------- PLM
def evaluar_plm(df_fase: pd.DataFrame, estado: list[str], palancas: list[str], target: str, n_splits: int = 5,
                n_boot: int = 300, seed: int = 42, signo_teorico: dict | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """PLM v4 (estado HGB + palancas lineales, cross-fitted): R2/MAE OOF en DEV, lockbox (entrenado en DEV),
    theta con IC bootstrap cluster. Devuelve (metricas, tabla_theta, predicciones)."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    inseg = [c for c in estado + palancas if not es_segura(c)]
    if inseg:
        raise AssertionError(f"features inseguras: {inseg}")
    dev, lockbox = mp.split_dev_lockbox(df_fase)
    need = list(dict.fromkeys(estado + palancas + [target, "Batch", "fecha_inicio", "orden_escalon_fase"]))
    d_dev = df_fase.loc[df_fase["Batch"].isin(dev), need].dropna().reset_index(drop=True)
    d_lb = df_fase.loc[df_fase["Batch"].isin(lockbox), need].dropna().reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev, d_dev[target], d_dev["Batch"]):
        m = mp4.ModeloPLM(estado, palancas, target).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
        oof[te] = m.predict(d_dev.iloc[te])
    m = mp4.ModeloPLM(estado, palancas, target).fit(d_dev, n_boot=n_boot, seed=seed)
    p_lb = m.predict(d_lb) if len(d_lb) else np.array([])
    met = {"target": target, "n_estado": len(estado), "n_palancas": len(palancas), "n_dev": len(d_dev), "n_lockbox": len(d_lb),
           "r2_oof": float(r2_score(d_dev[target], oof)), "mae_oof": float(mean_absolute_error(d_dev[target], oof)),
           "r2_lockbox": float(r2_score(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan,
           "mae_lockbox": float(mean_absolute_error(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan,
           "sd_target_dev": float(d_dev[target].std())}
    tabla = m.tabla_theta(signo_teorico).reset_index()
    tabla.insert(0, "target", target)
    tabla["theta_1sd_rel"] = tabla["theta_1sd"] / met["sd_target_dev"]
    tabla["ancho_ic_1sd_rel"] = (tabla["ci_hi_1sd"] - tabla["ci_lo_1sd"]) / met["sd_target_dev"]
    pred = pd.concat([d_dev[["Batch", "fecha_inicio", "orden_escalon_fase"]].assign(real=d_dev[target], pred=oof, conjunto="OOF_dev"),
                      d_lb[["Batch", "fecha_inicio", "orden_escalon_fase"]].assign(real=d_lb[target], pred=p_lb, conjunto="lockbox")],
                     ignore_index=True)
    return met, tabla, pred


# --------------------------------------------------------------------------- tabla teórica (para auditar signos/formas)
SIGNO_TEORICO_V5: dict[str, dict[str, int]] = {
    # +1: sube el target; -1: baja; 0: sin expectativa
    "v5_ln_sn_dep": {"tasa_feed_Carbon_kg_min": +1, "v5_dosis_C_sn": +1, "v5_ln_dosis_C": +1, "Cx_sn_v4": +1,
                     "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": 0,
                     "v5_gn_esp_nm3_t": +1},
    "v5_ln_feo_ret": {"tasa_feed_Carbon_kg_min": -1, "v5_exceso_C_pos": -1, "v5_exceso_C_kg": -1, "Cx_av_v4": -1, "v5_C_x_avance": -1,
                      "tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_o2_nm3_min": +1, "tasa_aire_nm3_min": +1,
                      "v5_gn_esp_nm3_t": -1},
    "v5_sdi_w10": {"tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_o2_nm3_min": +1, "tasa_aire_nm3_min": +1},
    "v5_lnirf": {"tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_o2_nm3_min": +1, "tasa_feed_Carbon_kg_min": -1},
    "v5_d_lnirf": {"tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_o2_nm3_min": +1, "tasa_feed_Carbon_kg_min": -1,
                   "relacion_C_Sn_carga": 0},
    "sn_extraido_est_kg": {"tasa_feed_Carbon_kg_min": +1, "relacion_C_Sn_carga": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1},
    "d_temperatura_horno_celsius": {"tasa_gn_nm3_min": +1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": -1, "tasa_feed_Carbon_kg_min": 0},
}


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = cargar_df()
    print("df", df.shape, f"{time.time() - t0:.1f}s")
    for f in FASES:
        d = base_fase(df, f)
        print(f, d.shape, d[TARGETS_V5].describe().round(3).T.to_string())
    b = construir_batch_v5(df)
    print("batch", b.shape)
    print(b[["recuperacion_refinada_pct", "rendimiento_proxy_batch", "R_sdi_w10", "R_lnirf_twmean", "F_lnirf_fin"]].describe().round(3).T.to_string())
    for col in ["R_sdi_w10", "R_sum_ln_sn_dep", "R_sum_ln_feo_ret", "R_lnirf_twmean", "F_lnirf_fin", "F_lnirf_F0", "F_sum_d_lnirf", "F_sum_ln_sn_dep"]:
        r = bateria_batch(b, col)
        print(col, {k: round(v, 3) for k, v in r.items() if k.startswith(("rho_", "p_", "ols_dev", "ols_p_dev"))})
    print(f"{time.time() - t0:.1f}s")
