import sys; sys.path.insert(0,'experimentos/v9')
import v9_lib as L, pandas as pd, numpy as np
t=L.tabla_batch(); dev=t[t.es_dev]; lb=t[~t.es_dev]
X=[c for c in t.columns if c.startswith("x_") and c.split("_")[2] in ("carbon","gn","exo2")]
for y in [L.KPI,"f_metal","f_dross","f_polvo","sn_perdido_escoria_frac","recuperacion_real_pct"]:
    for nm,d in (("DEV",dev),("TOTAL",t)):
        dd=d.dropna(subset=X+L.CONTROLES+[y]); f=L.ols_hc3(dd[y]*(100 if y.startswith("f_") or y.startswith("sn_") else 1),dd[X+L.CONTROLES])
        print(y,nm,len(dd)," ".join(f"{c[2:]}:{f.params[c]:+.2f}({f.pvalues[c]:.2f})" for c in X))
print(t[X].corr().round(2).to_string())
