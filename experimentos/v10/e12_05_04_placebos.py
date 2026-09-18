"""E12-05 tarea 4: placebos (lead1, lead3, pre-tratamiento) para las candidatas mas cercanas a pasar en la tarea 2
(ninguna cruzo p<0.05 en DEV; por diligencia se corren para las dos con menor p: F_esteq (p=.058) y F_o2 (p=.098))."""
import sys; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
import e12_05_lib as E

t = pd.read_csv(E.SALIDA / "e12_05_tabla_batch.csv", index_col=0).sort_values("idx_cronologico")
CTRL = E.C_SIN_ESPESOR; BASE = ["xG", "xC"]
CANDIDATOS = [("F", "esteq"), ("F", "o2")]

filas = []
dev = t[t.es_dev].copy()
for g, p in CANDIDATOS:
    col = f"x_{g}_{p}"
    # (i) lead: exposicion del batch siguiente (lead1) / +3 (lead3) -> KPI actual, controlando la exposicion actual
    for lead in (1, 3):
        dd = dev.sort_values("idx_cronologico").copy()
        dd[f"{col}_lead{lead}"] = dd[col].shift(-lead)
        X = BASE + CTRL + [col, f"{col}_lead{lead}"]
        d = dd.dropna(subset=X + [E.KPI])
        f = E.ols_hc3(d[E.KPI], d[X])
        filas.append({"candidata": f"{g}_{p}", "placebo": f"lead{lead}", "n": len(d),
                      "coef": float(f.params[f"{col}_lead{lead}"]), "t": float(f.tvalues[f"{col}_lead{lead}"]),
                      "p": float(f.pvalues[f"{col}_lead{lead}"])})
    # (ii) pre-tratamiento: exposicion actual -> variables previas al escalon/batch (no deberia predecirlas)
    for pretrat in ("ley_sn_conc_batch_pct", "feed_sn_total_t"):
        X = [col] + CTRL
        d = dev.dropna(subset=X + [pretrat])
        f = E.ols_hc3(d[pretrat], d[X])
        filas.append({"candidata": f"{g}_{p}", "placebo": f"pre_{pretrat}", "n": len(d),
                      "coef": float(f.params[col]), "t": float(f.tvalues[col]), "p": float(f.pvalues[col])})
    # KPI del batch ANTERIOR (pre-tratamiento temporal)
    dd = dev.sort_values("idx_cronologico").copy()
    dd["kpi_prev"] = dd[E.KPI].shift(1)
    X = [col] + CTRL
    d = dd.dropna(subset=X + ["kpi_prev"])
    f = E.ols_hc3(d["kpi_prev"], d[X])
    filas.append({"candidata": f"{g}_{p}", "placebo": "pre_kpi_prev", "n": len(d),
                  "coef": float(f.params[col]), "t": float(f.tvalues[col]), "p": float(f.pvalues[col])})

out = pd.DataFrame(filas)
out.to_csv(E.SALIDA / "e12_05_placebos.csv", index=False)
pd.set_option("display.width", 160)
print(out.round(4).to_string(index=False))
