"""E11-07 tarea 1: reproduce (numpy puro) la tabla de e11_06d (W 60/80/100/120/150/expansiva, paso 10, sin compuerta)."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as E

t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Xexp, Xctrl, y, es_dev, n = D["Xexp"], D["Xctrl"], D["y"], D["es_dev"], D["n"]
print("n =", n, "n_dev =", es_dev.sum(), "n_lb =", (~es_dev).sum())

filas = []
for W in E.WS_MULTIPLICIDAD:
    score = E.walk_forward_score(Xexp, Xctrl, y, W, paso=E.PASO, tmin=0.0)
    for nm, m in (("todo", np.ones(n, bool)), ("DEV", es_dev), ("LB", ~es_dev)):
        r = E.stats_final(score, y, Xctrl, m)
        r.update({"W": W if W is not None else "expansiva", "muestra": nm})
        filas.append(r)
out = pd.DataFrame(filas)[["W", "muestra", "n", "pendiente", "se", "t", "p_uni", "spearman", "sd_score"]]
out.to_csv(E.SALIDA / "e11_07_repro_walkforward.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string())

# comparacion contra e11_06d_ventana_movil.csv (mismo tmin=0, paso=10)
ref = pd.read_csv(E.SALIDA / "e11_06d_ventana_movil.csv")
ref = ref[(ref.tmin == 0.0) & (ref.paso == 10)].copy()
out["Wstr"] = out["W"].astype(str); ref["Wstr"] = ref["W"].astype(str)
comp = out.merge(ref, on=["Wstr", "muestra"], suffixes=("_np", "_sm"))
print("\ncomparacion pendiente numpy vs statsmodels (e11_06d):")
print(comp[["Wstr", "muestra", "n_np", "n_sm", "pendiente_np", "pendiente_sm", "t_np", "t_sm"]].round(3).to_string())
