"""E9-07: validacion del sistema v7 (modelo_predictivo_v7) y politica cross-fitted (espejo de E8-05).
Etapas: validacion | 0..4 | lockbox | evidencia
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp
import modelo_predictivo_v7 as mp7
import modelo_predictivo_v6 as mp6
SALIDA = RAIZ / "experimentos" / "v7"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 300)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

import os
SUF = ""
if os.environ.get("V7_SIN_LEGADO") == "1":
    mp7.activar_v7(con_legado=False); SUF = "_sinlegado"
df = mp7.cargar_df()
etapa = sys.argv[1] if len(sys.argv) > 1 else "validacion"
log("CONFIG:", mp6.CONFIG_V6, "ESTADO_R:", len(mp6.ESTADO_V6[("Reducción", "sn_dep")]))

if etapa == "validacion":
    tabla = mp7.tabla_validacion_v7(df); tabla.to_csv(SALIDA / "e9_07_tabla_validacion_v7.csv", index=False)
    log("\n", tabla.round(3).to_string())
    sistema = mp7.entrenar_sistema_v7(df, n_boot=300)
    ef = mp7.tabla_efectos_control_v7(sistema); ef.to_csv(SALIDA / "e9_07_efectos_control_v7.csv", index=False)
    log("\n", ef.round(4).to_string())
    for fase in mp7.FASES:
        for clave in mp7.TARGETS_V7[fase]:
            p = mp7.predicciones_oof_y_lockbox_v7(df, fase, clave)
            p.to_csv(SALIDA / f"e9_07_pred_{fase}_{clave}.csv", index=False)
    log("predicciones guardadas")
    df_base = mp6._preparar_base(df); dev = sistema.batches_dev
    curvas = []
    red = df_base[(df_base.fase_proceso == "Reducción") & df_base.Batch.isin(dev)]
    for av in (0.75, 0.90, 0.965, 0.985):
        r = red.iloc[(red["m6_avance_prev"] - av).abs().argsort()[:1]].iloc[0]
        st, ac, cx = mp7.fila_a_state_action_context("Reducción", r)
        for pal in mp7.ACCIONES_OPTIMIZABLES_V7["Reducción"]:
            c = mp7.curva_respuesta_v7(sistema, "Reducción", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["m6_avance_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Reducción")
            curvas.append(c)
    fus = df_base[(df_base.fase_proceso == "Fusión") & df_base.Batch.isin(dev)]
    for orden in (1, 3, 5):
        r = fus[fus.orden_escalon_fase == orden].sample(1, random_state=1).iloc[0]
        st, ac, cx = mp7.fila_a_state_action_context("Fusión", r)
        for pal in mp7.ACCIONES_OPTIMIZABLES_V7["Fusión"]:
            c = mp7.curva_respuesta_v7(sistema, "Fusión", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["m6_avance_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Fusión")
            curvas.append(c)
    cur = pd.concat(curvas, ignore_index=True); cur.to_csv(SALIDA / "e9_07_curvas_respuesta_v7.csv", index=False)
    log("curvas guardadas")

elif etapa == "evidencia":
    partes = [pd.read_csv(SALIDA / f"e9_07_parcial_{k}{SUF}.csv") for k in ["0", "1", "2", "3", "4", "lockbox"] if (SALIDA / f"e9_07_parcial_{k}{SUF}.csv").exists()]
    esc = pd.concat(partes, ignore_index=True); esc.to_csv(SALIDA / f"e9_07_recomendaciones_v7_escalones{SUF}.csv", index=False)
    log("escalones", esc.shape, "batches", esc.Batch.nunique())
    pb = mp7.metricas_por_batch_v7(df, esc)
    # agregacion escalon -> grupo -> batch (pp de KPI, dos batches) por batch
    agg = []
    for b, rep in esc.groupby("Batch"):
        a = mp7.agregar_objetivo(rep); a["Batch"] = b; agg.append(a)
    agg = pd.DataFrame(agg); pb = pb.merge(agg, on="Batch", how="left")
    pb.to_csv(SALIDA / f"e9_07_por_batch_v7{SUF}.csv", index=False)
    ev = mp7.evidencia_politica_v7(pb, variables=["dist_total", "dist_F", "dist_R", "uplift_fisico_total_kg", "uplift_batch_pp"])
    ev.to_csv(SALIDA / f"e9_07_evidencia_politica_v7{SUF}.csv", index=False)
    log("\n", ev.round(4).to_string())
    # placebo: politica aleatoria (permutar recomendaciones entre batches)
    rng = np.random.default_rng(0); pbp = pb.copy()
    for c in [c for c in pb.columns if c.startswith("dist_")]:
        pbp[c] = rng.permutation(pb[c].to_numpy())
    evp = mp7.evidencia_politica_v7(pbp, variables=["dist_total", "dist_F", "dist_R"], n_perm=0)
    evp.to_csv(SALIDA / f"e9_07_evidencia_placebo_v7{SUF}.csv", index=False)
    log("placebo\n", evp[evp.kpi == "recuperacion_refinada_pct"].round(4).to_string())
    log("uplift por batch (pp, dos batches): mediana", pb["uplift_batch_pp"].median(), "media", pb["uplift_batch_pp"].mean(),
        "kg mediana", pb["uplift_batch_kg"].median())
    log("\n", pb.groupby("es_lockbox")[["uplift_batch_pp", "J_R_temprano_hist_pp", "J_R_temprano_rec_pp", "J_R_tardio_hist_pp", "J_R_tardio_rec_pp", "J_F_hist_pp", "J_F_rec_pp"]].mean().round(3).to_string())
    perfil = esc.groupby(["fase", "orden_escalon_fase"])[[c for c in esc.columns if c.startswith("rec__") or c.startswith("hist__")]].mean()
    perfil.to_csv(SALIDA / f"e9_07_perfil_politica_v7{SUF}.csv"); log("\n", perfil.round(2).to_string())

else:
    esc = mp7.recomendaciones_cross_fitted_v7(df, etapa, n_folds=5, n_candidatos=150, semilla=0, log=log)
    esc.to_csv(SALIDA / f"e9_07_parcial_{etapa}{SUF}.csv", index=False)
    log("guardado", etapa, esc.shape)
log("FIN")
