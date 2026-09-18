"""E10-06 -- Fusion con beta_F = 0: que terminos del objetivo son defendibles y como acotar el recorte de carbon.

Diseno: experimentos/v8/ITERACION_10_diseno.md (H6). Libreria: experimentos/v8/v8_lib.py (L, re-exporta v7_lib).
Punto de partida: dictamen del comite (beta_F descartado, experimentos/v7/dictamen_comite_v7.md); auditoria A
S:2/3/6 (carbon de Fusion = reductor + combustible; exceso_o2_combustion_pct NO incluye carbon; recorte sin bajar
O2 desplaza oxigeno al bano); auditoria C S3-4 (recorte de carbon de Fusion <= 1 sd del orden y >= P10, con -O2
proporcional, prohibido en fin de campana o T previa en decil inferior); e9_04_resultados.md (Fusion: sn_dep no
generaliza, dT si con F1 imputado; canal polvo solo T_media_F).

Etapas (CLI):
    dT          -- ESTADO_V6[('Fusion','dT')] en 4 variantes: (a) sin F1 (grad NaN dropea F1), (b) F1 imputado a 0
                   (v7), (c) F1 imputado + es_F1 en el estado, (d) variante b sin temperatura_horno_celsius_prev
                   (para exponer la confusion de control reactivo del O2). R2 por conjuntos (OOF/cronologico/WF/
                   lockbox), theta libre/acotado (GN,O2,aire,carbon; n_boot 500), R2 por orden F1..F6.
                   -> e10_06_dT.csv, e10_06_dT_theta.csv, e10_06_dT_por_orden.csv
    familia     -- familia pre-registrada de 3 KPI x 7 terminos de batch de Fusion (21 pruebas), OLS HC3 con
                   CONTROLES_BATCH en DEV/total + Spearman en lockbox, q-BH (Benjamini-Hochberg) dentro de la
                   familia DEV. -> e10_06_familia_batch.csv
    recorte     -- lambda extendida (incluye el O2 teorico del carbon a CO, 0.933 Nm3/kg) por orden de Fusion;
                   efecto termico (ModeloPLMSignos/theta_dual de dT) de recortar carbon -1 sd del orden (i) sin
                   tocar O2 y (ii) bajando O2 en 0.933*deltaC (lambda constante); mismo calculo sobre
                   d_temperatura_gas_pre_bhf_celsius si es modelable. -> e10_06_recorte.csv
    envolvente  -- P5/P10/P50/P90/P95/sd de carbon, GN, O2, aire por orden F1-F6 en DEV; % de escalones en fin de
                   campana (espesor < P10 global) por orden. -> e10_06_envolvente.csv
    todas       -- corre todas en orden.

Entorno: .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=3; sin joblib/loky; LightGBM no disponible.
Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v8/e10_06_fusion.py <etapa>
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent          # experimentos/v8
sys.path.insert(0, str(HERE))
import v8_lib as L                               # noqa: E402
import modelo_predictivo_v6 as mp6               # noqa: E402

SEED = L.SEED
N_BOOT_THETA = 500
FRACCION_O2_EN_AIRE = 0.21
O2_TEORICO_POR_KG_C_CO = 22.414 / 12.011 / 2      # 0.933 Nm3 O2 / kg C (C + 1/2 O2 -> CO)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# =============================================================================
# datos comunes
# =============================================================================

CLAVE_DT = "dT"
TARGET_DT = mp6.TARGETS_V6["Fusión"][CLAVE_DT]          # d_temperatura_horno_celsius
S_DT = list(mp6.ESTADO_V6[("Fusión", CLAVE_DT)])        # 8 features (incl. grad_temperatura_horno_celsius_prev)
A_DT = list(mp6.PALANCAS_V6[("Fusión", CLAVE_DT)])      # [tasa_gn_nm3_min, tasa_o2_nm3_min, tasa_aire_nm3_min, tasa_feed_Carbon_kg_min]
SIGNOS_DT = dict(mp6.SIGNO_TEORICO_V6[("Fusión", CLAVE_DT)])
GRAD = "grad_temperatura_horno_celsius_prev"
T_PREV = "temperatura_horno_celsius_prev"


def _cargar_fusion() -> pd.DataFrame:
    df = L.cargar_df()
    L.asegurar_seguras(S_DT + A_DT)
    return L.base_fase(df, "Fusión")


def _imputar_f1(dF: pd.DataFrame) -> pd.DataFrame:
    """Replica _preparar_base_v7: en Fusion, orden_escalon_fase==1 (F1), grad NaN -> 0.0."""
    out = dF.copy()
    m = out["orden_escalon_fase"].eq(1) & out[GRAD].isna()
    out.loc[m, GRAD] = 0.0
    return out


VARIANTES_DT = ["a_sin_F1", "b_F1_imputado_v7", "c_F1_imputado_mas_esF1", "d_b_sin_temperatura_prev"]


def _estado_variante(v: str) -> list[str]:
    if v == "c_F1_imputado_mas_esF1":
        return S_DT + ["es_F1"]
    if v == "d_b_sin_temperatura_prev":
        return [c for c in S_DT if c != T_PREV]
    return list(S_DT)


def _datos_variante(dF: pd.DataFrame, dF_imp: pd.DataFrame, v: str) -> pd.DataFrame:
    return dF if v == "a_sin_F1" else dF_imp


# =============================================================================
# 1) dT: F1 y confusion de O2
# =============================================================================

def etapa_dT() -> None:
    dF = _cargar_fusion()
    dF_imp = _imputar_f1(dF)

    # --- confusion de control reactivo: corr(O2, T_prev) y corr(GN, T_prev) en Fusion DEV
    dev_b, _ = L.split(dF)
    ddev = dF[dF["Batch"].isin(dev_b)][["tasa_o2_nm3_min", "tasa_gn_nm3_min", T_PREV]].dropna()
    corr_o2_t = float(ddev["tasa_o2_nm3_min"].corr(ddev[T_PREV]))
    corr_gn_t = float(ddev["tasa_gn_nm3_min"].corr(ddev[T_PREV]))
    log(f"confusion reactiva DEV: corr(O2,T_prev)={corr_o2_t:+.3f}  corr(GN,T_prev)={corr_gn_t:+.3f}  n={len(ddev)}")

    filas_r2, filas_theta, filas_orden = [], [], []
    for v in VARIANTES_DT:
        S = _estado_variante(v)
        d = _datos_variante(dF, dF_imp, v)
        L.asegurar_seguras(S + A_DT)
        d_dev, d_lb = L.preparar(d, S + A_DT, TARGET_DT)
        log(f"{v}: n_dev={len(d_dev)} n_lockbox={len(d_lb)} (n_F1_dev={(d_dev.orden_escalon_fase == 1).sum()})")

        r2 = L.r2_por_conjuntos(d_dev, d_lb, S, A_DT, TARGET_DT, SIGNOS_DT, seed=SEED)
        r2["variante"] = v; r2["n_estado"] = len(S)
        filas_r2.append(r2)
        log(f"  r2_oof={r2['r2_oof']:.3f}  r2_oof_cronologico={r2['r2_oof_cronologico']:.3f}  "
            f"r2_wf={r2.get('r2_walk_forward', np.nan):.3f}  r2_lockbox={r2.get('r2_lockbox', np.nan):.3f}")

        t0 = time.time()
        tabla, _ = L.theta_dual(d_dev, S, A_DT, TARGET_DT, SIGNOS_DT, n_boot=N_BOOT_THETA, seed=SEED)
        tabla.insert(0, "variante", v)
        filas_theta.append(tabla)
        log(f"  theta ({time.time() - t0:.1f}s): " +
            "; ".join(f"{r.palanca}={r.theta_libre_1sd:+.3f}[{r.ci_lo_libre_1sd:+.3f},{r.ci_hi_libre_1sd:+.3f}]"
                      for r in tabla.itertuples()))

        # R2 por orden F1..F6 (OOF cross-fitted, mismo criterio que r2_por_conjuntos)
        fp = L.fit_predict_plm(S, A_DT, TARGET_DT, SIGNOS_DT, seed=SEED)
        oof = L.oof_groupkfold(d_dev, fp)
        for orden, g in d_dev.assign(pred=oof).groupby("orden_escalon_fase"):
            if len(g) < 5:
                continue
            r2o = L.metricas(g[TARGET_DT], g["pred"])
            filas_orden.append({"variante": v, "orden_escalon_fase": int(orden), "n": len(g),
                                "r2_oof": r2o["r2"], "mae_oof": r2o["mae"]})

    r2_out = pd.DataFrame(filas_r2)
    r2_out["corr_o2_temperatura_prev_dev"] = corr_o2_t
    r2_out["corr_gn_temperatura_prev_dev"] = corr_gn_t
    r2_out.to_csv(HERE / "e10_06_dT.csv", index=False)
    pd.concat(filas_theta, ignore_index=True).to_csv(HERE / "e10_06_dT_theta.csv", index=False)
    pd.DataFrame(filas_orden).to_csv(HERE / "e10_06_dT_por_orden.csv", index=False)
    log("FIN dT")


# =============================================================================
# 2) familia de terminos de batch para J_F (q-BH)
# =============================================================================

def _agregados_familia(df: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for b, g in df.groupby("Batch"):
        gf = g[g["fase_proceso"] == "Fusión"]
        c_kg = float(gf["feed_Carbon_kgh"].sum(min_count=1))
        sn_kg = float(gf["feed_Sn_kgf"].sum(min_count=1))
        filas.append({
            "Batch": b,
            "F_carbon_total_kg": c_kg,
            "F_c_sn_ratio": c_kg / sn_kg if sn_kg and sn_kg > 0 else np.nan,
            "F_gn_total_nm3": float(gf["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum(min_count=1)),
            "F_exceso_o2_media": float(gf["exceso_o2_combustion_pct"].mean()) if len(gf) else np.nan,
            "F_tgas_prebhf_media": float(gf["temperatura_gas_pre_bhf_celsius"].mean()) if len(gf) else np.nan,
        })
    agg = pd.DataFrame(filas).set_index("Batch")
    return batch.join(agg, how="left")


FAMILIA_X = ["F_carbon_total_kg", "F_c_sn_ratio", "T_media_F", "F_gn_total_nm3", "F_exceso_o2_media",
             "F_tgas_prebhf_media", "F_sum_m6_ln_sn_dep"]
FAMILIA_Y = [L.KPI_PRINCIPAL, "f_polvo", "f_dross"]


def etapa_familia() -> None:
    from statsmodels.stats.multitest import multipletests
    df = L.cargar_df()
    batch = _agregados_familia(df, L.construir_batch(df))
    filas = []
    for y in FAMILIA_Y:
        for x in FAMILIA_X:
            r = L.bateria_batch(batch, x, kpi=y)
            r["y"] = y; r["x"] = x
            filas.append(r)
            log(f"{y} ~ {x}: ols_dev={r.get('ols_dev')} p_dev={r.get('ols_p_dev')} "
                f"ols_total={r.get('ols_total')} rho_lockbox={r.get('rho_lockbox')}")
    fam = pd.DataFrame(filas)
    fam["p_dev_para_bh"] = fam["ols_p_dev"]
    ok = fam["p_dev_para_bh"].notna()
    q = np.full(len(fam), np.nan)
    if ok.any():
        _, qvals, _, _ = multipletests(fam.loc[ok, "p_dev_para_bh"], method="fdr_bh")
        q[ok.to_numpy()] = qvals
    fam["q_bh_dev"] = q
    fam["signo_dev"] = np.sign(fam["ols_dev"])
    fam["signo_total"] = np.sign(fam["ols_total"])
    fam["signo_lockbox_rho"] = np.sign(fam["rho_lockbox"])
    fam["replica_signo_total"] = fam["signo_dev"] == fam["signo_total"]
    fam["replica_signo_lockbox"] = fam["signo_dev"] == fam["signo_lockbox_rho"]
    fam["defendible"] = (fam["ols_p_dev"] < 0.05) & (fam["q_bh_dev"] < 0.10) & fam["replica_signo_total"]
    fam.to_csv(HERE / "e10_06_familia_batch.csv", index=False)
    log(f"FIN familia: {int(fam['defendible'].sum())}/{len(fam)} terminos defendibles (p<0.05, q-BH<0.10, signo replica en total)")


# =============================================================================
# 3) recorte de carbon con lambda constante
# =============================================================================

VARIANTE_ADOPTADA_RECORTE = "b_F1_imputado_v7"   # F1 imputado (v7), sin dummy es_F1 (mas simple, mismo estado que v6/v7)


def etapa_recorte() -> None:
    dF = _cargar_fusion()
    dF_imp = _imputar_f1(dF)
    dev_b, _ = L.split(dF)
    dev = dF[dF["Batch"].isin(dev_b)].copy()
    dev["lambda_ext"] = ((dev["tasa_o2_nm3_min"] + FRACCION_O2_EN_AIRE * dev["tasa_aire_nm3_min"])
                          / (2 * dev["tasa_gn_nm3_min"] + O2_TEORICO_POR_KG_C_CO * dev["tasa_feed_Carbon_kg_min"]))
    filas_lambda = []
    for orden, g in dev.groupby("orden_escalon_fase"):
        s = g["lambda_ext"].dropna()
        if len(s) < 5:
            continue
        filas_lambda.append({"orden_escalon_fase": int(orden), "n": len(s), "media": float(s.mean()), "sd": float(s.std()),
                             "P5": float(s.quantile(.05)), "P10": float(s.quantile(.10)), "P50": float(s.quantile(.50)),
                             "P90": float(s.quantile(.90)), "P95": float(s.quantile(.95))})
    tabla_lambda = pd.DataFrame(filas_lambda)
    log("lambda extendida por orden (DEV):\n" + tabla_lambda.round(3).to_string(index=False))

    # --- theta libre de dT y de d_temperatura_gas_pre_bhf_celsius (mismo estado/palancas, variante adoptada)
    S = _estado_variante(VARIANTE_ADOPTADA_RECORTE)
    d = _datos_variante(dF, dF_imp, VARIANTE_ADOPTADA_RECORTE)
    filas_efecto = []
    for target, etiqueta in [(TARGET_DT, "d_temperatura_horno_celsius"), ("d_temperatura_gas_pre_bhf_celsius", "d_temperatura_gas_pre_bhf_celsius")]:
        L.asegurar_seguras(S + A_DT)
        d_dev, d_lb = L.preparar(d, S + A_DT, target)
        signos = SIGNOS_DT if target == TARGET_DT else None
        t0 = time.time()
        tabla, _ = L.theta_dual(d_dev, S, A_DT, target, signos, n_boot=300, seed=SEED)
        r2_modelo = float(tabla["r2_oof_plm_interno"].iloc[0])
        log(f"{etiqueta}: r2_oof_plm_interno={r2_modelo:.3f} n_dev={len(d_dev)} ({time.time()-t0:.1f}s)")
        tabla = tabla.set_index("palanca")
        theta_c_raw = float(tabla.loc["tasa_feed_Carbon_kg_min", "theta_libre_1sd"]) / float(tabla.loc["tasa_feed_Carbon_kg_min", "sd_palanca"])
        theta_o2_raw = float(tabla.loc["tasa_o2_nm3_min", "theta_libre_1sd"]) / float(tabla.loc["tasa_o2_nm3_min", "sd_palanca"])
        ci_c = (float(tabla.loc["tasa_feed_Carbon_kg_min", "ci_lo_libre_1sd"]) / float(tabla.loc["tasa_feed_Carbon_kg_min", "sd_palanca"]),
                float(tabla.loc["tasa_feed_Carbon_kg_min", "ci_hi_libre_1sd"]) / float(tabla.loc["tasa_feed_Carbon_kg_min", "sd_palanca"]))
        for orden, g in dev.groupby("orden_escalon_fase"):
            sd_c_orden = float(g["tasa_feed_Carbon_kg_min"].std())
            if not np.isfinite(sd_c_orden) or sd_c_orden <= 0:
                continue
            delta_c = -sd_c_orden                              # recorte de -1 sd del orden
            delta_o2_proporcional = -O2_TEORICO_POR_KG_C_CO * sd_c_orden   # -O2 = 0.933*deltaC (lambda constante)
            efecto_sin_o2 = theta_c_raw * delta_c
            efecto_con_o2 = theta_c_raw * delta_c + theta_o2_raw * delta_o2_proporcional
            filas_efecto.append({
                "target": etiqueta, "r2_oof_plm_interno": r2_modelo, "orden_escalon_fase": int(orden),
                "sd_carbon_orden_kg_min": sd_c_orden, "theta_carbon_raw_por_kg_min": theta_c_raw,
                "theta_carbon_raw_ci_lo": ci_c[0], "theta_carbon_raw_ci_hi": ci_c[1],
                "theta_o2_raw_por_nm3_min": theta_o2_raw, "delta_o2_proporcional_nm3_min": delta_o2_proporcional,
                "efecto_recorte_sin_bajar_o2": efecto_sin_o2, "efecto_recorte_con_o2_proporcional": efecto_con_o2,
            })
    efecto = pd.DataFrame(filas_efecto)
    tabla_lambda["target"] = "lambda_extendida"
    out = pd.concat([tabla_lambda, efecto], ignore_index=True, sort=False)
    out.to_csv(HERE / "e10_06_recorte.csv", index=False)
    log("FIN recorte")


# =============================================================================
# 4) envolvente por orden (limites del optimizador v8)
# =============================================================================

COLS_ENVOLVENTE = {"tasa_feed_Carbon_kg_min": "carbon_kg_min", "tasa_gn_nm3_min": "gn_nm3_min",
                   "tasa_o2_nm3_min": "o2_nm3_min", "tasa_aire_nm3_min": "aire_nm3_min"}


def etapa_envolvente() -> None:
    dF = _cargar_fusion()
    dev_b, _ = L.split(dF)
    dev = dF[dF["Batch"].isin(dev_b)].copy()
    p10_espesor_global = float(dev["espesor_ladrillo_norm_mm"].quantile(.10))
    log(f"P10 global de espesor_ladrillo_norm_mm (DEV, fin de campana): {p10_espesor_global:.1f} mm")

    filas = []
    for orden, g in dev.groupby("orden_escalon_fase"):
        fin_campana = g["espesor_ladrillo_norm_mm"] < p10_espesor_global
        for col, etiqueta in COLS_ENVOLVENTE.items():
            s = g[col].dropna()
            if len(s) < 5:
                continue
            filas.append({
                "orden_escalon_fase": int(orden), "variable": etiqueta, "n": len(s), "media": float(s.mean()),
                "sd": float(s.std()), "P5": float(s.quantile(.05)), "P10": float(s.quantile(.10)),
                "P50": float(s.quantile(.50)), "P90": float(s.quantile(.90)), "P95": float(s.quantile(.95)),
                "pct_fin_campana": float(fin_campana.mean() * 100), "n_fin_campana": int(fin_campana.sum()),
                "p10_espesor_global_mm": p10_espesor_global,
            })
    pd.DataFrame(filas).to_csv(HERE / "e10_06_envolvente.csv", index=False)
    log("FIN envolvente")


# =============================================================================
# main
# =============================================================================

ETAPAS = {"dT": etapa_dT, "familia": etapa_familia, "recorte": etapa_recorte, "envolvente": etapa_envolvente}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=list(ETAPAS) + ["todas"])
    args = ap.parse_args()
    if args.etapa == "todas":
        for nombre, fn in ETAPAS.items():
            log(f"##### etapa {nombre} #####")
            fn()
    else:
        ETAPAS[args.etapa]()
    log(f"TOTAL {time.time() - T0:.1f}s")
