"""E7-05: validacion del sistema v5 (modelo_predictivo_v5) y politica cross-fitted.
Etapas (argumento CLI):
  validacion      -> e7_05_tabla_validacion_v5.csv, e7_05_efectos_control_v5.csv, e7_05_pred_<fase>_<clave>.csv,
                     e7_05_curvas_respuesta_v5.csv
  0..4 | lockbox  -> e7_05_parcial_<etiqueta>.csv (recomendaciones cross-fitted por escalon)
  evidencia       -> e7_05_recomendaciones_v5_escalones.csv, e7_05_por_batch_v5.csv, e7_05_evidencia_politica_v5.csv (+ placebo)
Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v5/e7_05_validacion_v5.py <etapa>
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp
import modelo_predictivo_v5 as mp5
SALIDA = RAIZ / "experimentos" / "v5"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 300)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

df = mp5.agregar_columnas_v5(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
etapa = sys.argv[1] if len(sys.argv) > 1 else "validacion"
W_SOP = float(sys.argv[2]) if len(sys.argv) > 2 else None
SUF = "" if W_SOP is None else f"_ws{int(W_SOP)}"
PESOS = None if W_SOP is None else {"w_soporte": W_SOP}

if etapa == "validacion":
    tabla = mp5.tabla_validacion_v5(df); tabla.to_csv(SALIDA / "e7_05_tabla_validacion_v5.csv", index=False)
    log("\n", tabla.round(3).to_string())
    sistema = mp5.entrenar_sistema_v5(df, n_boot=300)
    ef = mp5.tabla_efectos_control_v5(sistema); ef.to_csv(SALIDA / "e7_05_efectos_control_v5.csv", index=False)
    log("\n", ef.round(4).to_string())
    for fase in mp5.FASES:
        for clave in mp5.TARGETS_V5[fase]:
            p = mp5.predicciones_oof_y_lockbox_v5(df, fase, clave)
            p.to_csv(SALIDA / f"e7_05_pred_{fase}_{clave}.csv", index=False)
    log("predicciones guardadas")
    df_base = mp5._preparar_base(df); dev = sistema.batches_dev
    curvas = []
    red = df_base[(df_base.fase_proceso == "Reducción") & df_base.Batch.isin(dev)]
    for av in (0.75, 0.90, 0.965, 0.985):
        r = red.iloc[(red["avance_reduccion_sn_prev"] - av).abs().argsort()[:1]].iloc[0]
        st, ac, cx = mp5.fila_a_state_action_context("Reducción", r)
        for pal in mp5.ACCIONES_OPTIMIZABLES_V5["Reducción"]:
            c = mp5.curva_respuesta_v5(sistema, "Reducción", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["avance_reduccion_sn_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Reducción")
            curvas.append(c)
    fus = df_base[(df_base.fase_proceso == "Fusión") & df_base.Batch.isin(dev)]
    for orden in (1, 3, 5):
        r = fus[fus.orden_escalon_fase == orden].sample(1, random_state=1).iloc[0]
        st, ac, cx = mp5.fila_a_state_action_context("Fusión", r)
        for pal in mp5.ACCIONES_OPTIMIZABLES_V5["Fusión"]:
            c = mp5.curva_respuesta_v5(sistema, "Fusión", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["avance_reduccion_sn_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Fusión")
            curvas.append(c)
    cur = pd.concat(curvas, ignore_index=True); cur.to_csv(SALIDA / "e7_05_curvas_respuesta_v5.csv", index=False)
    res = cur.groupby(["fase", "Batch", "palanca"]).apply(lambda g: pd.Series({"x_opt": g.loc[g.J.idxmax(), "valor"], "x_min": g.valor.min(), "x_max": g.valor.max(), "rango_J": g.J.max() - g.J.min()}))
    log("\n", res.round(2).to_string())
    log("LISTO validacion")
elif etapa == "evidencia":
    parc = [SALIDA / f"e7_05_parcial_{e}{SUF}.csv" for e in ["0", "1", "2", "3", "4", "lockbox"]]
    esc = pd.concat([pd.read_csv(p) for p in parc], ignore_index=True)
    esc.to_csv(SALIDA / f"e7_05_recomendaciones_v5_escalones{SUF}.csv", index=False)
    log("escalones", esc.shape, "batches", esc.Batch.nunique())
    pb = mp5.metricas_por_batch_v5(df, esc); pb.to_csv(SALIDA / f"e7_05_por_batch_v5{SUF}.csv", index=False)
    cols = ["uplift_fisico_total_kg", "uplift_fisico_fusion_kg", "uplift_fisico_reduccion_kg", "dist_total", "dist_F", "dist_R",
            "support_medio_historico", "support_medio_recomendado"]
    log("\n", pb[cols].describe().round(3).to_string())
    dirc = [c for c in pb.columns if c.startswith("dir_")]
    log("\n", pb[dirc].describe().T.round(3).to_string())
    ev = mp5.evidencia_politica_v5(pb, variables=["dist_total", "dist_F", "dist_R", "uplift_fisico_total_kg"] + dirc, n_perm=500)
    rng = np.random.default_rng(0); dev, _ = mp.split_dev_lockbox(df); df_base = mp5._preparar_base(df)
    esc_p = esc.copy()
    for fase in mp5.FASES:
        s = df_base[(df_base.fase_proceso == fase) & df_base.Batch.isin(dev)]
        for a in mp5.ACCIONES_OPTIMIZABLES_V5[fase]:
            lo, hi = s[a].quantile(0.01), s[a].quantile(0.99); m = esc_p.fase == fase
            esc_p.loc[m, f"rec__{a}"] = rng.uniform(lo, hi, m.sum())
    pbp = mp5.metricas_por_batch_v5(df, esc_p)
    evp = mp5.evidencia_politica_v5(pbp, variables=["dist_total", "dist_F", "dist_R"], n_perm=0); evp["variable"] = evp["variable"] + "_placebo"
    ev = pd.concat([ev, evp], ignore_index=True); ev.to_csv(SALIDA / f"e7_05_evidencia_politica_v5{SUF}.csv", index=False)
    log("\n", ev[ev.variable.isin(["dist_total", "dist_F", "dist_R", "dist_total_placebo", "dist_R_placebo", "dist_F_placebo"])]
        .pivot_table(index=["kpi", "variable"], columns="subset", values=["coef_por_sd", "p_hc3", "spearman"]).round(3).to_string())
    log("LISTO evidencia")
else:
    esc = mp5.recomendaciones_cross_fitted_v5(df, etapa, n_candidatos=150, log=log, pesos=PESOS)
    esc.to_csv(SALIDA / f"e7_05_parcial_{etapa}{SUF}.csv", index=False)
    log("LISTO", etapa, esc.shape)
