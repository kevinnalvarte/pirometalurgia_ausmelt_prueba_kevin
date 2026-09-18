"""E11-02 tarea 5: prueba anidada -- (a) 2 exposiciones sin moderar, (b) 4 terminos moderados por el mejor moderador
por teoria (ln ley_sn_escoria_pct_prev, elegido a priori: la actividad termodinamica a_SnO/a_FeO escala con la
CONCENTRACION (ley), no con el inventario absoluto en kg), (c) con signos de teoria impuestos. Sensibilidad: ln
m6_sn_inv_kg_prev y ley_sn_escoria_pct_prev sin ln.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd

import v9_lib as L
import v8_lib as L8
import modelo_predictivo_v8 as mp8

OUT = Path("experimentos/v9")

df = L8.cargar_df()
res = L.residuos_escalon(df)
base = mp8._preparar_base(df)
t_batch = L.tabla_batch(df, res)
dev_b, lockbox_b = L8.mp.split_dev_lockbox(df)

CANDIDATOS = [("ley_sn_escoria_pct_prev", True), ("m6_sn_inv_kg_prev", True), ("ley_sn_escoria_pct_prev", False)]
ETIQUETAS = {("ley_sn_escoria_pct_prev", True): "ln_ley_sn (PRIMARIO, a priori)",
             ("m6_sn_inv_kg_prev", True): "ln_m6_sn_inv_kg (sensibilidad)",
             ("ley_sn_escoria_pct_prev", False): "ley_sn nivel (sensibilidad)"}

R = res[res.fase_proceso == "Reducción"].merge(
    base[["Batch", "fase_proceso", "orden_escalon_fase", "ley_sn_escoria_pct_prev", "m6_sn_inv_kg_prev"]].drop_duplicates(
        ["Batch", "fase_proceso", "orden_escalon_fase"]),
    on=["Batch", "fase_proceso", "orden_escalon_fase"], how="left")
R["es_dev"] = R.Batch.isin(dev_b)

# exposiciones sin moderar (mean_t(z) sobre TODOS los escalones de Reduccion, R0-R3)
x_sin = {}
for p in ("gn", "carbon"):
    x_sin[f"x_R_{p}"] = R[f"z__{p}"].groupby(R.Batch).mean()
t_batch = t_batch.join(pd.DataFrame(x_sin))

filas = []
resultados_detalle = {}
for mod, ln in CANDIDATOS:
    x = np.log(R[mod]) if ln else R[mod].astype(float)
    mu, sd = float(x[R.es_dev].mean()), float(x[R.es_dev].std())
    mtil = (x - mu) / sd
    nm_m = f"m_{mod}_{'ln' if ln else 'niv'}"
    xcols = {}
    for p in ("gn", "carbon"):
        xcols[f"x_R_{p}_x_{nm_m}"] = (R[f"z__{p}"] * mtil).groupby(R.Batch).mean()
    t_batch_local = t_batch.join(pd.DataFrame(xcols))
    dev = t_batch_local[t_batch_local.es_dev]
    especs = {
        "a_sin_moderar": (["x_R_gn", "x_R_carbon"], None),
        "b_moderado_libre": (["x_R_gn", f"x_R_gn_x_{nm_m}", "x_R_carbon", f"x_R_carbon_x_{nm_m}"], None),
        "c_moderado_signos": (["x_R_gn", f"x_R_gn_x_{nm_m}", "x_R_carbon", f"x_R_carbon_x_{nm_m}"],
                              {f"x_R_gn_x_{nm_m}": +1, "x_R_carbon": +1, f"x_R_carbon_x_{nm_m}": +1}),
    }
    for nombre_espec, (X, signos) in especs.items():
        t0 = time.time()
        r_gkf = L.prueba_anidada(dev, X, esquema="gkf", signos=signos)
        r_crono_dev = L.prueba_anidada(dev, X, esquema="crono", signos=signos)
        r_crono_tot = L.prueba_anidada(t_batch_local, X, esquema="crono", n_folds=6, signos=signos)
        dt = time.time() - t0
        for esq_nombre, r in (("DEV_gkf20", r_gkf), ("DEV_crono5", r_crono_dev), ("TOTAL_crono6", r_crono_tot)):
            sc = r.pop("score")
            row = {"moderador": ETIQUETAS[(mod, ln)], "espec": nombre_espec, "esquema": esq_nombre, "tiempo_s": round(dt, 1),
                   **{k: v for k, v in r.items() if k not in ("beta_medio", "beta_sd_folds")}}
            row["beta_medio"] = str(r["beta_medio"])
            filas.append(row)
    resultados_detalle[(mod, ln)] = t_batch_local
    print(f"listo: {ETIQUETAS[(mod, ln)]}")

tabla5 = pd.DataFrame(filas)
tabla5.to_csv(OUT / "e11_02_prueba_anidada.csv", index=False)
cols_show = ["moderador", "espec", "esquema", "n", "pendiente", "t", "p_unilateral", "p_perm_unilateral",
             "kpi_tercio_bajo", "kpi_tercio_alto"]
print(tabla5[cols_show].round(4).to_string())
print("\nscript prueba anidada completo")
