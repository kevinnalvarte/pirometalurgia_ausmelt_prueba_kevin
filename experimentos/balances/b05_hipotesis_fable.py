"""B-05a (Fable) -- dos hipotesis propias derivadas del modelo Rust que no cubren B-01..B-04.

H1  Cementacion en Fusion (ec. 6 del modelo Rust: Fe + SnO -> FeO + Sn con TODO el Fe metalico disponible).
    El dross de Fe cargado (tolva 5, Fe metalico/hardhead) deberia extraer Sn de la escoria en Fusion por
    intercambio, sin necesidad de carbon. Si es asi: (a) el Sn extraido en Fusion responde a b6_ceq_fe_dross_kg
    (C-eq del Fe metalico) con theta > 0 e IC que no cruza 0, mas que al carbon; (b) a nivel batch, mas dross
    de Fe cargado en Fusion -> mas Sn a dross (hardhead) y menos f_metal, controlando por Sn cargado.
H2  Monitor de Fe: el escalon de Reduccion en que el FeO empieza a caer (inicio de sobre-reduccion) y el %Sn
    de la escoria en ese momento. Prediccion: cuanto mas alto el %Sn al inicio de la perdida de FeO, peor
    la selectividad del batch (mas dross). Se calcula con el multi-trazador (b6_feo_inv_multi_kg).

Ejecutar: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/balances/b05_hipotesis_fable.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bal_features as bal  # noqa: E402
sys.path.insert(0, str(bal.RAIZ / "experimentos" / "search_targets"))
import st_targets as st  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


def main():
    df = st.construir_df_targets(bal.cargar_df())
    df = mp4.agregar_columnas_v4(df)
    dev, lock = mp.split_dev_lockbox(df)
    batch = st.construir_batch(df)

    # ------------------------------------------------------------------ H1 escalon
    log("=== H1 Cementacion en Fusion: PLM del Sn extraido con Fe metalico del dross como palanca")
    fus = df[(df["fase_proceso"] == "Fusión") & (~df["es_primer_escalon_batch"].astype(bool))
             & df["sn_extraido_est_kg"].notna() & df["Batch"].isin(dev)].copy()
    est = mp4.ESTADO_V4[("Fusión", "sn_kg")]
    fus["tasa_ceq_fe_dross"] = fus["b6_ceq_fe_dross_kg"] / fus["delta_tiempo"]
    fus["tasa_c_fijo"] = fus["b6_c_fijo_kg"] / fus["delta_tiempo"]
    fus["tasa_ceq_lanza"] = fus["b6_ceq_lanza_kg"] / fus["delta_tiempo"]
    variantes = {
        "v4": mp4.PALANCAS_V4[("Fusión", "sn_kg")],
        "fisica": ["tasa_c_fijo", "tasa_ceq_fe_dross", "tasa_ceq_lanza", "tasa_gn_nm3_min"],
        "fisica+cal": ["tasa_c_fijo", "tasa_ceq_fe_dross", "tasa_ceq_lanza", "tasa_gn_nm3_min", "relacion_CaO_carga"],
    }
    filas = []
    for nombre, pal in variantes.items():
        d = fus.dropna(subset=est + pal + ["sn_extraido_est_kg"])
        m = mp4.ModeloPLM(est, pal, "sn_extraido_est_kg").fit(d, n_splits=5, n_boot=200)
        t = m.tabla_theta()
        log(f"--- variante {nombre}: n={len(d)}, R2 OOF interno={m.r2_oof_interno:.3f}")
        log(t[["theta_1sd", "ci_lo_1sd", "ci_hi_1sd", "significativo"]].round(1).to_string())
        for p, r in t.iterrows():
            filas.append(dict(variante=nombre, palanca=p, theta_1sd=r["theta_1sd"], ci_lo=r["ci_lo_1sd"],
                              ci_hi=r["ci_hi_1sd"], signif=r["significativo"], r2_oof=m.r2_oof_interno, n=len(d)))
    pd.DataFrame(filas).to_csv(HERE / "b05_h1_plm_fusion.csv", index=False)

    # ------------------------------------------------------------------ H1 batch
    log("\n=== H1 batch: dross de Fe cargado en Fusion vs canales de perdida (OLS HC3, DEV)")
    gF = df[df["fase_proceso"] == "Fusión"].groupby("Batch")
    bb = batch.copy()
    bb["dross_fe_F_t"] = gF["feed_dross_Fe_kgh"].sum() / 1000
    bb["fe_mineral_F_t"] = gF["feed_Fe_kgh"].sum() / 1000
    bb["carbon_F_t"] = gF["feed_Carbon_kgh"].sum() / 1000
    bb["sn_ext_F_t"] = gF["sn_extraido_est_kg"].sum() / 1000
    bb["ceq_fe_dross_F"] = gF["b6_ceq_fe_dross_kg"].sum()
    dv = bb[~bb["es_lockbox"]]
    filas = []
    for kpi in ["f_dross", "f_metal", "f_polvo", "rendimiento_proxy_batch", "sn_ext_F_t"]:
        X = dv[["dross_fe_F_t", "fe_mineral_F_t", "carbon_F_t", "feed_sn_total_t", "ley_sn_conc_batch_pct",
                "espesor_ladrillo_norm_mm", "idx_cronologico"]].copy()
        y = dv[kpi]
        ok = X.notna().all(axis=1) & y.notna()
        Xz = (X[ok] - X[ok].mean()) / X[ok].std()
        res = sm.OLS(y[ok], sm.add_constant(Xz)).fit(cov_type="HC3")
        for v in ["dross_fe_F_t", "fe_mineral_F_t", "carbon_F_t"]:
            filas.append(dict(kpi=kpi, var=v, beta_por_sd=res.params[v], p=res.pvalues[v], n=int(ok.sum()), r2=res.rsquared))
        log(f"{kpi}: dross_fe {res.params['dross_fe_F_t']:+.4f} (p={res.pvalues['dross_fe_F_t']:.3f}) | "
            f"mineral_fe {res.params['fe_mineral_F_t']:+.4f} (p={res.pvalues['fe_mineral_F_t']:.3f}) | "
            f"carbon {res.params['carbon_F_t']:+.4f} (p={res.pvalues['carbon_F_t']:.3f}) | R2 {res.rsquared:.3f}")
    pd.DataFrame(filas).to_csv(HERE / "b05_h1_batch_ols.csv", index=False)

    # ------------------------------------------------------------------ H2 monitor de Fe
    log("\n=== H2 Monitor de Fe: inicio de la perdida de FeO en Reduccion")
    red = df[df["fase_proceso"] == "Reducción"].sort_values(["Batch", "fecha_inicio"])
    filas = []
    for b, g in red.groupby("Batch"):
        feo = g["b6_feo_inv_multi_kg"].to_numpy()
        feo_prev = g["b6_feo_inv_multi_kg_prev"].to_numpy()
        sn_pct = g["ley_sn_escoria_pct_prev"].to_numpy()
        av = g["avance_reduccion_sn_prev"].to_numpy()
        orden = g["orden_escalon_fase"].to_numpy()
        # primer escalon con perdida de FeO > 3 % del inventario previo (umbral > ruido ~2 %)
        perdida = (feo_prev - feo) / feo_prev
        idx = np.where(np.isfinite(perdida) & (perdida > 0.03))[0]
        if len(idx):
            i = idx[0]
            filas.append(dict(Batch=b, orden_inicio=int(orden[i]), sn_pct_inicio=sn_pct[i], avance_inicio=av[i],
                              perdida_feo_total=np.nansum(np.clip(perdida, 0, None)), hubo_perdida=1))
        else:
            filas.append(dict(Batch=b, orden_inicio=np.nan, sn_pct_inicio=np.nan, avance_inicio=np.nan,
                              perdida_feo_total=np.nansum(np.clip(perdida, 0, None)), hubo_perdida=0))
    h2 = pd.DataFrame(filas).set_index("Batch").join(batch[["rendimiento_proxy_batch", "f_dross", "f_metal", "f_polvo",
                                                             "es_lockbox", "st_R22_sdi"]])
    log(f"batches con perdida de FeO > 3 %: {int(h2['hubo_perdida'].sum())}/{len(h2)}; orden de inicio: "
        f"{h2['orden_inicio'].value_counts().sort_index().to_dict()}")
    dv = h2[~h2["es_lockbox"] & (h2["hubo_perdida"] == 1)]
    for x in ["sn_pct_inicio", "avance_inicio", "orden_inicio", "perdida_feo_total"]:
        for kpi in ["f_dross", "rendimiento_proxy_batch", "st_R22_sdi"]:
            s = dv[[x, kpi]].dropna()
            rho, p = stats.spearmanr(s[x], s[kpi])
            log(f"  DEV rho({x}, {kpi}) = {rho:+.3f} (p={p:.3g}, n={len(s)})")
    lk = h2[h2["es_lockbox"] & (h2["hubo_perdida"] == 1)]
    for x in ["sn_pct_inicio", "avance_inicio"]:
        s = lk[[x, "f_dross"]].dropna()
        rho, p = stats.spearmanr(s[x], s["f_dross"])
        log(f"  LOCKBOX rho({x}, f_dross) = {rho:+.3f} (p={p:.3g}, n={len(s)})")
    h2.to_csv(HERE / "b05_h2_monitor_fe.csv")
    (HERE / "b05_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
