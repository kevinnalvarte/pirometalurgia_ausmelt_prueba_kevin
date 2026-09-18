"""E11-07 tarea 3: placebos prospectivos, W=100 paso=10.
- lead1 / lead3: se sustituye la exposicion del batch actual por la del batch SIGUIENTE (t+1 / t+3) en todo el
  walk-forward (entrenamiento y prueba); si el vinculo fuera espurio por tendencia compartida, tambien apareceria aqui.
- kpi_prev: el outcome es el KPI del batch ANTERIOR (el puntaje de hoy no puede explicar lo que ya paso antes de
  decidirse la accion de hoy).
Se espera pendiente ~ 0 en los tres.
"""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as E

t = E.cargar_tabla()
D = E.preparar(t, E.X_PRINCIPAL)
Xexp, Xctrl, y, n = D["Xexp"], D["Xctrl"], D["y"], D["n"]


def correr(Xe, Xc, yy, nombre):
    score = E.walk_forward_score(Xe, Xc, yy, 100, paso=E.PASO, tmin=0.0)
    r = E.stats_final(score, yy, Xc, np.ones(len(yy), bool))
    r["placebo"] = nombre
    return r

filas = [correr(Xexp, Xctrl, y, "real (referencia W=100 todo)")]
filas.append(correr(Xexp[1:], Xctrl[1:], y[:-1], "kpi_prev (outcome = KPI del batch anterior)"))
filas.append(correr(Xexp[1:], Xctrl[:-1], y[:-1], "lead1 (exposicion del batch siguiente)"))
filas.append(correr(Xexp[3:], Xctrl[:-3], y[:-3], "lead3 (exposicion 3 batches adelante)"))

out = pd.DataFrame(filas)[["placebo", "n", "pendiente", "se", "t", "p_uni", "spearman", "sd_score"]]
out.to_csv(E.SALIDA / "e11_07_placebos.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string())
