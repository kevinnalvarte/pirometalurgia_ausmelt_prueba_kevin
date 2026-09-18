import sys; sys.path.insert(0,'experimentos/v8')
import v8_lib as L, pandas as pd, numpy as np
df=L.cargar_df().sort_values(["Batch","fecha_inicio"])
df["dur"]=(pd.to_datetime(df.fecha_final)-pd.to_datetime(df.fecha_inicio)).dt.total_seconds()/60
df["gap"]=(pd.to_datetime(df.fecha_inicio)-pd.to_datetime(df.groupby("Batch").fecha_final.shift(1))).dt.total_seconds()/60
g=df.groupby(["fase_proceso","orden_escalon_fase"])
print(g.dur.describe(percentiles=[.05,.25,.5,.75,.95]).round(1)[["count","mean","std","5%","25%","50%","75%","95%"]].to_string())
print("gap entre escalones (min):",df.gap.describe(percentiles=[.05,.5,.95]).round(1).to_dict())
print([c for c in df.columns if "dur" in c.lower() or "plan" in c.lower()][:20])
