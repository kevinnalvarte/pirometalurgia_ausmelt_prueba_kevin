"""Experimento 21: validacion del sistema v4 (modelo_predictivo_v4) y evidencia de valor de su politica.
Etapas (argumento CLI):
  validacion      -> 21_tabla_validacion_v4.csv, 21_efectos_control_v4.csv, 21_pred_oof_lockbox_<fase>_<clave>.csv,
                     21_curvas_respuesta_v4.csv
  0..4 | lockbox  -> 21_parcial_<etiqueta>.csv (recomendaciones cross-fitted por escalon)
  evidencia       -> 21_recomendaciones_v4_escalones.csv, 21_por_batch_v4.csv, 21_evidencia_politica_v4.csv (+ placebo)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp
import modelo_predictivo_v4 as mp4
SALIDA = RAIZ / "experimentos"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 300)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

df = mp4.agregar_columnas_v4(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
etapa = sys.argv[1] if len(sys.argv) > 1 else "validacion"
W_SOP = float(sys.argv[2]) if len(sys.argv) > 2 else None   # variante de peso de soporte (p.ej. 500)
SUF = "" if W_SOP is None else f"_ws{int(W_SOP)}"
PESOS = None if W_SOP is None else {"w_soporte": W_SOP}

if etapa == "validacion":
    tabla = mp4.tabla_validacion_v4(df); tabla.to_csv(SALIDA / "21_tabla_validacion_v4.csv", index=False)
    log("\n", tabla.round(3).to_string())
    sistema = mp4.entrenar_sistema_v4(df, n_boot=300)
    ef = mp4.tabla_efectos_control_v4(sistema); ef.to_csv(SALIDA / "21_efectos_control_v4.csv", index=False)
    log("\n", ef.round(3).to_string())
    for fase in mp4.FASES:
        for clave in mp4.TARGETS_V4[fase]:
            p = mp4.predicciones_oof_y_lockbox_v4(df, fase, clave)
            p.to_csv(SALIDA / f"21_pred_oof_lockbox_{fase}_{clave}.csv", index=False)
    log("predicciones guardadas")
    # curvas de respuesta en estados representativos de Reduccion (por avance) y Fusion (por orden)
    df_base = mp4._preparar_base(df); dev = sistema.batches_dev
    curvas = []
    red = df_base[(df_base.fase_proceso == "Reducción") & df_base.Batch.isin(dev)]
    for av in (0.75, 0.90, 0.965, 0.985):
        r = red.iloc[(red["avance_reduccion_sn_prev"] - av).abs().argsort()[:1]].iloc[0]
        st, ac, cx = mp4.fila_a_state_action_context("Reducción", r)
        for pal in mp4.ACCIONES_OPTIMIZABLES_V4["Reducción"]:
            c = mp4.curva_respuesta_v4(sistema, "Reducción", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["avance_reduccion_sn_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Reducción")
            curvas.append(c)
    fus = df_base[(df_base.fase_proceso == "Fusión") & df_base.Batch.isin(dev)]
    for orden in (1, 3, 5):
        r = fus[fus.orden_escalon_fase == orden].sample(1, random_state=1).iloc[0]
        st, ac, cx = mp4.fila_a_state_action_context("Fusión", r)
        for pal in mp4.ACCIONES_OPTIMIZABLES_V4["Fusión"]:
            c = mp4.curva_respuesta_v4(sistema, "Fusión", st, ac, cx, pal)
            c.insert(0, "avance_prev", float(r["avance_reduccion_sn_prev"])); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Fusión")
            curvas.append(c)
    cur = pd.concat(curvas, ignore_index=True); cur.to_csv(SALIDA / "21_curvas_respuesta_v4.csv", index=False)
    res = cur.groupby(["fase", "Batch", "palanca"]).apply(lambda g: pd.Series({"x_opt": g.loc[g.J.idxmax(), "valor"], "x_min": g.valor.min(), "x_max": g.valor.max(), "rango_J": g.J.max() - g.J.min()}))
    log("\n", res.round(2).to_string())
    log("LISTO validacion")
elif etapa == "evidencia":
    parc = [SALIDA / f"21_parcial_{e}{SUF}.csv" for e in ["0", "1", "2", "3", "4", "lockbox"]]
    esc = pd.concat([pd.read_csv(p) for p in parc], ignore_index=True)
    esc.to_csv(SALIDA / f"21_recomendaciones_v4_escalones{SUF}.csv", index=False)
    log("escalones", esc.shape, "batches", esc.Batch.nunique())
    pb = mp4.metricas_por_batch_v4(df, esc); pb.to_csv(SALIDA / f"21_por_batch_v4{SUF}.csv", index=False)
    log("\n", pb[["uplift_fisico_total_kg", "uplift_fisico_fusion_kg", "uplift_fisico_reduccion_kg", "dist_total", "dist_F", "dist_R", "support_medio_historico", "support_medio_recomendado"]].describe().round(3).to_string())
    dirc = [c for c in pb.columns if c.startswith("dir_")]
    log("\n", pb[dirc].describe().T.round(3).to_string())
    ev = mp4.evidencia_politica_v4(pb, variables=["dist_total", "dist_F", "dist_R", "uplift_fisico_total_kg"] + dirc, n_perm=500)
    # placebo: politica aleatoria dentro de P1-P99 -> misma distancia
    rng = np.random.default_rng(0); dev, _ = mp.split_dev_lockbox(df); df_base = mp4._preparar_base(df)
    esc_p = esc.copy()
    for fase in mp4.FASES:
        s = df_base[(df_base.fase_proceso == fase) & df_base.Batch.isin(dev)]
        for a in mp4.ACCIONES_OPTIMIZABLES_V4[fase]:
            lo, hi = s[a].quantile(0.01), s[a].quantile(0.99); m = esc_p.fase == fase
            esc_p.loc[m, f"rec__{a}"] = rng.uniform(lo, hi, m.sum())
    pbp = mp4.metricas_por_batch_v4(df, esc_p)
    evp = mp4.evidencia_politica_v4(pbp, variables=["dist_total", "dist_F", "dist_R"], n_perm=0); evp["variable"] = evp["variable"] + "_placebo"
    ev = pd.concat([ev, evp], ignore_index=True); ev.to_csv(SALIDA / f"21_evidencia_politica_v4{SUF}.csv", index=False)
    log("\n", ev.round(4).to_string())
    log("LISTO evidencia")
else:
    esc = mp4.recomendaciones_cross_fitted_v4(df, etapa, n_candidatos=150, log=log, pesos=PESOS)
    esc.to_csv(SALIDA / f"21_parcial_{etapa}{SUF}.csv", index=False)
    log("LISTO", etapa, esc.shape)
