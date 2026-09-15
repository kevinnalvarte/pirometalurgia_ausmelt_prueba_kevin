"""Validacion final del sistema v3 con la configuracion congelada (FEATURES_V3_POR_DEFECTO):
tabla OOF/lockbox por (fase, target), cobertura CV+ en lockbox, efectos causales conjuntos
de las palancas (DML multivariado), evaluacion de la politica recomendada sobre batches del
lockbox y un reporte detallado de un batch (con sensibilidad local por escalon).

Salidas: experimentos/10_tabla_validacion_v3.csv, 10_cobertura_cvplus_v3.csv,
         10_efectos_palancas_conjunto_v3.csv, 10_politica_lockbox_v3.csv,
         10_reporte_batch_ejemplo_v3.csv, 10_sensibilidad_local_ejemplo_v3.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402

SALIDA = RAIZ / "experimentos"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 200)
t0 = time.time()
df = mp3.construir_dataset_modelo_v3(RAIZ / "Datos Lingo smelter fase II.xlsx")
print("dataset", df.shape, f"{time.time()-t0:.0f}s", flush=True)

tabla = mp3.tabla_validacion_v3(df)
tabla.to_csv(SALIDA / "10_tabla_validacion_v3.csv")
print("\n=== tabla de validacion v3 (OOF dev / lockbox) ===\n", tabla.round(3).to_string(), f"\n{time.time()-t0:.0f}s", flush=True)

sistema = mp3.entrenar_sistema_v3(df)
print("sistema entrenado", f"{time.time()-t0:.0f}s", flush=True)

cob = []
for fase in mp3.FASES:
    for clave in mp3.TARGETS_V3[fase]:
        for alpha in (0.2, 0.1):
            cob.append(mp3.validar_cobertura_lockbox_v3(df, sistema, fase, clave, alpha))
cob = pd.DataFrame(cob); cob.to_csv(SALIDA / "10_cobertura_cvplus_v3.csv", index=False)
print("\n=== cobertura CV+ en lockbox ===\n", cob.round(3).to_string(), f"\n{time.time()-t0:.0f}s", flush=True)

efectos = []
for fase in mp3.FASES:
    for clave in (["sn_kg", "feo_kg", "dT"] if fase == "Reducción" else ["sn_kg", "sn_pts", "dT"]):
        e = mp3.efectos_marginales_palancas_v3(df, fase, mp3.TARGETS_V3[fase][clave], n_boot=150)
        e["clave"] = clave; efectos.append(e.reset_index())
        print(f"\n=== efectos conjuntos DML {fase} / {clave} ({mp3.TARGETS_V3[fase][clave]}) ===\n", e.round(4).to_string(), flush=True)
efectos = pd.concat(efectos); efectos.to_csv(SALIDA / "10_efectos_palancas_conjunto_v3.csv", index=False)
print(f"{time.time()-t0:.0f}s", flush=True)

politica = mp3.evaluar_politica_en_lockbox_v3(sistema, df, n_batches=8, n_candidatos=150)
politica.to_csv(SALIDA / "10_politica_lockbox_v3.csv")
print("\n=== politica recomendada vs historica, 8 batches del lockbox (kg Sn) ===\n", politica.round(2).to_string(), f"\n{time.time()-t0:.0f}s", flush=True)

batch = politica.index[0]
rep = mp3.reporte_recomendaciones_batch_v3(sistema, df, batch, n_candidatos=200)
rep.to_csv(SALIDA / "10_reporte_batch_ejemplo_v3.csv", index=False)
cols = [c for c in rep.columns if not c.startswith("rec_derivada")]
print(f"\n=== reporte batch {batch} ===\n", rep[cols].round(2).T.to_string(), flush=True)

df_base = mp.dataset_base_modelo(df)
row = df_base[(df_base["Batch"] == batch) & (df_base["fase_proceso"] == "Reducción")].sort_values("fecha_inicio").iloc[0]
state, action, context = mp3.fila_a_state_action_context("Reducción", row)
sens = mp3.sensibilidad_local_v3(sistema, "Reducción", state, action, context)
sens.to_csv(SALIDA / "10_sensibilidad_local_ejemplo_v3.csv")
print(f"\n=== sensibilidad local {batch} R0 ===\n", sens.round(3).to_string(), flush=True)
row = df_base[(df_base["Batch"] == batch) & (df_base["fase_proceso"] == "Fusión")].sort_values("fecha_inicio").iloc[2]
state, action, context = mp3.fila_a_state_action_context("Fusión", row)
sens = mp3.sensibilidad_local_v3(sistema, "Fusión", state, action, context)
print(f"\n=== sensibilidad local {batch} F2 ===\n", sens.round(3).to_string(), flush=True)
print(f"\nLISTO {time.time()-t0:.0f}s")
