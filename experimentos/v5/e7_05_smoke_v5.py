"""Smoke test del modulo modelo_predictivo_v5 (esqueleto): entrena en DEV, tabla theta, recomendacion de un batch."""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ))
import modelo_predictivo_v5 as mp5
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

df = mp5.agregar_columnas_v5(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
log("df", df.shape)
sistema = mp5.entrenar_sistema_v5(df, n_boot=50)
log("sistema entrenado; soporte minimo", sistema.soporte_minimo, "ventana T", sistema.ventana_temperatura)
ef = mp5.tabla_efectos_control_v5(sistema)
log("\n", ef[["fase", "clave", "palanca", "theta_1sd", "ci_lo_1sd", "ci_hi_1sd", "significativo", "concuerda"]].round(4).to_string())
for b in ["AP0150", "AP0300"]:
    rep = mp5.reporte_recomendaciones_batch_v5(sistema, df, b, n_candidatos=120)
    log(b, "\n", rep[["fase", "orden_escalon_fase", "avance_reduccion_sn_prev", "hist__tasa_feed_Carbon_kg_min", "rec__tasa_feed_Carbon_kg_min",
                      "hist__tasa_gn_nm3_min", "rec__tasa_gn_nm3_min", "hist__tasa_o2_nm3_min", "rec__tasa_o2_nm3_min",
                      "J_historico", "J_recomendado", "support_historico", "support_recomendado"]].round(2).to_string())
    log(mp5.resumen_reporte_batch_v5(rep))
log("LISTO")
