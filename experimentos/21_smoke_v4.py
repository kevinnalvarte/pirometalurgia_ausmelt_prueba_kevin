import sys, time, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(RAIZ))
import modelo_predictivo_v4 as mp4
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
t0 = time.time()
df = mp4.agregar_columnas_v4(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
sistema = mp4.entrenar_sistema_v4(df, n_boot=100)
print(f"entrenado {time.time()-t0:.0f}s")
print(mp4.tabla_efectos_control_v4(sistema).round(3).to_string())
for fase in mp4.FASES:
    print(fase, "soporte_min", round(sistema.soporte_minimo[fase], 3), "ventana T", sistema.ventana_temperatura[fase])
t1 = time.time()
rep = mp4.reporte_recomendaciones_batch_v4(sistema, df, "AP0300", n_candidatos=150)
print(f"reporte AP0300 {time.time()-t1:.1f}s")
cols = [c for c in rep.columns if not c.startswith("hist_derivada") and not c.startswith("rec_derivada")]
print(rep[cols].round(2).T.to_string())
print(mp4.resumen_reporte_batch_v4(rep))
print(f"total {time.time()-t0:.0f}s")
