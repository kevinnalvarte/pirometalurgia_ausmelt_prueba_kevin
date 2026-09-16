"""Construye experimentos/cache/df_v6.pkl (df completo v3+v4+b6+v5+m6) para que los experimentos de la iteracion 9 lo carguen en segundos."""
import sys, time
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "experimentos" / "v6"))
import v6_lib
t0 = time.time()
df = v6_lib.cargar_df(con_termicas=True)
out = RAIZ / "experimentos" / "cache" / "df_v6.pkl"
df.to_pickle(out)
print("filas", df.shape, "seg", round(time.time() - t0, 1), "->", out)
print("batches", df["Batch"].nunique())
print("ESTADO_R_FULL_V6", v6_lib.ESTADO_R_FULL_V6)
print("ESTADO_R_CURADO_V6", v6_lib.ESTADO_R_CURADO_V6)
print("ESTADO_F_CURADO_V6", v6_lib.ESTADO_F_CURADO_V6)
import modelo_predictivo_v6 as mp6
for k, v in mp6.ESTADO_V6.items(): print("ESTADO_V6", k, len(v), v)
for k, v in mp6.PALANCAS_V6.items(): print("PALANCAS_V6", k, v)
print("TARGETS_V6", mp6.TARGETS_V6)
print("CONFIG_V6", mp6.CONFIG_V6)
print(df.groupby(["fase_proceso","orden_escalon_fase"])["duracion_plan_min"].describe()[["mean","50%","std"]])
print(df.groupby(["fase_proceso","orden_escalon_fase"])[["m6_ln_sn_dep","m6_ln_feo_ret","tasa_feed_Carbon_kg_min","tasa_feed_CaO_kg_min","tasa_gn_nm3_min"]].mean())
