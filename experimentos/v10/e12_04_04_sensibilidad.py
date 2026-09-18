"""E12-04 (5,6) -- valor de la informacion del ensayo de t-1: diferencia de evidencia prospectiva S1 (con ensayo) vs S3'
(sin ensayo, "plan"). Tambien arma la tabla comparativa final de los 4 estados para el veredicto."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v10")
import numpy as np
import pandas as pd

import e12_04_lib as L

pd.set_option("display.width", 220)

todo = pd.read_csv(L.SALIDA / "e12_04_02_evidencia_todos_estados.csv")
piv_t = todo.pivot(index="muestra", columns="estado", values="t")
piv_pend = todo.pivot(index="muestra", columns="estado", values="pendiente")
piv_rho = todo.pivot(index="muestra", columns="estado", values="spearman_parcial")

ref = "S1_v9"
filas = []
for muestra in piv_t.index:
    t_ref = piv_t.loc[muestra, ref]
    for estado in piv_t.columns:
        if estado == ref:
            continue
        t_e = piv_t.loc[muestra, estado]
        frac = t_e / t_ref if pd.notna(t_ref) and t_ref != 0 else np.nan
        filas.append({"muestra": muestra, "estado": estado, "t_S1": t_ref, "t_estado": t_e,
                      "frac_t_conservada": frac, "cumple_H4_80pct": bool(frac >= 0.8) if pd.notna(frac) else False,
                      "pendiente_S1": piv_pend.loc[muestra, ref], "pendiente_estado": piv_pend.loc[muestra, estado],
                      "rho_S1": piv_rho.loc[muestra, ref], "rho_estado": piv_rho.loc[muestra, estado]})
val_info = pd.DataFrame(filas)
val_info.to_csv(L.SALIDA / "e12_04_04_valor_informacion.csv", index=False)
print(val_info.round(4).to_string())

# resumen de "listo para planta" (criterio ITERACION_12_diseno.md) por estado, muestra "todo"
crit = todo[todo.muestra == "todo"].copy()
crit["p_corregida_ok"] = crit["p_unilateral"] < 0.01
crit["pendiente_gt0"] = crit["pendiente"] > 0
print()
print(crit[["estado", "n", "pendiente", "t", "p_unilateral", "p_corregida_ok", "spearman_parcial", "p_perm_bloques"]].round(4).to_string())
crit.to_csv(L.SALIDA / "e12_04_04_criterio_planta.csv", index=False)
print("listo")
