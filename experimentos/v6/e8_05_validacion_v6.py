"""E8-05: validacion del sistema v6 (modelo_predictivo_v6) y politica cross-fitted (espejo de E7-05).
Etapas (argumento CLI):
  validacion      -> e8_05_tabla_validacion_v6.csv, e8_05_efectos_control_v6.csv, e8_05_pred_<fase>_<clave>.csv,
                     e8_05_curvas_respuesta_v6.csv
  0..4 | lockbox  -> e8_05_parcial_<etiqueta>.csv (recomendaciones cross-fitted por escalon)
  evidencia       -> e8_05_recomendaciones_v6_escalones.csv, e8_05_por_batch_v6.csv, e8_05_evidencia_politica_v6.csv (+ placebo)
Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v6/e8_05_validacion_v6.py <etapa>
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp
import modelo_predictivo_v6 as mp6
SALIDA = RAIZ / "experimentos" / "v6"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 300)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

df = mp6.agregar_columnas_v6(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
etapa = sys.argv[1] if len(sys.argv) > 1 else "validacion"
W_SOP = float(sys.argv[2]) if len(sys.argv) > 2 else None
SUF = "" if W_SOP is None else f"_ws{int(W_SOP)}"
PESOS = None if W_SOP is None else {"w_soporte": W_SOP}
log("CONFIG_V6:", mp6.CONFIG_V6, "PARAMS_MASA_V6:", mp6.PARAMS_MASA_V6)

if etapa == "validacion":
    tabla = mp6.tabla_validacion_v6(df); tabla.to_csv(SALIDA / "e8_05_tabla_validacion_v6.csv", index=False)
    log("\n", tabla.round(3).to_string())
    sistema = mp6.entrenar_sistema_v6(df, n_boot=300)
    ef = mp6.tabla_efectos_control_v6(sistema); ef.to_csv(SALIDA / "e8_05_efectos_control_v6.csv", index=False)
    log("\n", ef.round(4).to_string())
    for fase in mp6.FASES:
        for clave in mp6.TARGETS_V6[fase]:
            p = mp6.predicciones_oof_y_lockbox_v6(df, fase, clave)
            p.to_csv(SALIDA / f"e8_05_pred_{fase}_{clave}.csv", index=False)
    log("predicciones guardadas")
    df_base = mp6._preparar_base(df); dev = sistema.batches_dev
    curvas = []
    red = df_base[(df_base.fase_proceso == "Reducción") & df_base.Batch.isin(dev)]
    for av in (0.75, 0.90, 0.965, 0.985):
        r = red.iloc[(red["m6_avance_prev"] - av).abs().argsort()[:1]].iloc[0]
        st, ac, cx = mp6.fila_a_state_action_context("Reducción", r)
        for pal in mp6.ACCIONES_OPTIMIZABLES_V6["Reducción"]:
            c = mp6.curva_respuesta_v6(sistema, "Reducción", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["m6_avance_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Reducción")
            curvas.append(c)
    fus = df_base[(df_base.fase_proceso == "Fusión") & df_base.Batch.isin(dev)]
    for orden in (1, 3, 5):
        r = fus[fus.orden_escalon_fase == orden].sample(1, random_state=1).iloc[0]
        st, ac, cx = mp6.fila_a_state_action_context("Fusión", r)
        for pal in mp6.ACCIONES_OPTIMIZABLES_V6["Fusión"]:
            c = mp6.curva_respuesta_v6(sistema, "Fusión", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["m6_avance_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Fusión")
            curvas.append(c)
    cur = pd.concat(curvas, ignore_index=True); cur.to_csv(SALIDA / "e8_05_curvas_respuesta_v6.csv", index=False)
    res = cur.groupby(["fase", "Batch", "palanca"]).apply(lambda g: pd.Series({"x_opt": g.loc[g.J.idxmax(), "valor"], "x_min": g.valor.min(), "x_max": g.valor.max(), "rango_J": g.J.max() - g.J.min()}))
    log("\n", res.round(2).to_string())
    log("LISTO validacion")
elif etapa == "evidencia":
    parc = [SALIDA / f"e8_05_parcial_{e}{SUF}.csv" for e in ["0", "1", "2", "3", "4", "lockbox"]]
    esc = pd.concat([pd.read_csv(p) for p in parc], ignore_index=True)
    esc.to_csv(SALIDA / f"e8_05_recomendaciones_v6_escalones{SUF}.csv", index=False)
    log("escalones", esc.shape, "batches", esc.Batch.nunique())
    pb = mp6.metricas_por_batch_v6(df, esc); pb.to_csv(SALIDA / f"e8_05_por_batch_v6{SUF}.csv", index=False)
    cols = ["uplift_fisico_total_kg", "uplift_fisico_fusion_kg", "uplift_fisico_reduccion_kg", "dist_total", "dist_F", "dist_R",
            "support_medio_historico", "support_medio_recomendado"]
    log("\n", pb[cols].describe().round(3).to_string())
    dirc = [c for c in pb.columns if c.startswith("dir_")]
    log("\n", pb[dirc].describe().T.round(3).to_string())
    ev = mp6.evidencia_politica_v6(pb, variables=["dist_total", "dist_F", "dist_R", "uplift_fisico_total_kg"] + dirc, n_perm=500)
    rng = np.random.default_rng(0); dev, _ = mp.split_dev_lockbox(df); df_base = mp6._preparar_base(df)
    esc_p = esc.copy()
    for fase in mp6.FASES:
        s = df_base[(df_base.fase_proceso == fase) & df_base.Batch.isin(dev)]
        for a in mp6.ACCIONES_OPTIMIZABLES_V6[fase]:
            lo, hi = s[a].quantile(0.01), s[a].quantile(0.99); m = esc_p.fase == fase
            esc_p.loc[m, f"rec__{a}"] = rng.uniform(lo, hi, m.sum())
    pbp = mp6.metricas_por_batch_v6(df, esc_p)
    evp = mp6.evidencia_politica_v6(pbp, variables=["dist_total", "dist_F", "dist_R"], n_perm=0); evp["variable"] = evp["variable"] + "_placebo"
    ev = pd.concat([ev, evp], ignore_index=True); ev.to_csv(SALIDA / f"e8_05_evidencia_politica_v6{SUF}.csv", index=False)
    log("\n", ev[ev.variable.isin(["dist_total", "dist_F", "dist_R", "dist_total_placebo", "dist_R_placebo", "dist_F_placebo"])]
        .pivot_table(index=["kpi", "variable"], columns="subset", values=["coef_por_sd", "p_hc3", "spearman"]).round(3).to_string())
    log("LISTO evidencia")
else:
    esc = mp6.recomendaciones_cross_fitted_v6(df, etapa, n_candidatos=150, log=log, pesos=PESOS)
    esc.to_csv(SALIDA / f"e8_05_parcial_{etapa}{SUF}.csv", index=False)
    log("LISTO", etapa, esc.shape)
