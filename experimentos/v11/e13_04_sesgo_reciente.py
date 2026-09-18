"""E13-04 (Fable): en el replay ESTRICTO la practica habitual E[a|S] se entrena solo con el pasado -> si la receta deriva, el residuo de todo un
bloque lleva un sesgo comun (ruido confundido con la campana). Correccion ejecutable: E[a|S] + residuo medio por orden de los ultimos k batches."""
import sys, time; sys.path.insert(0, ".")
import pandas as pd
import asesor_planta_v9_1 as AP, modelo_predictivo_v9 as mp9, v8_lib as L8
df = L8.cargar_df(); t0 = time.time(); out = []
for k in (20, 40):
    esc, pb = AP.replay_planta(df, {"contraccion": "js", "sesgo_reciente_batches": k}, log=lambda *a: None)
    pb.to_csv(f"experimentos/v11/e13_04_por_batch_k{k}.csv"); esc.to_csv(f"experimentos/v11/e13_04_escalones_k{k}.csv", index=False)
    ev = mp9.evidencia_prospectiva(pb); ev.insert(0, "k_sesgo", k); out.append(ev); print(ev.round(4).to_string(), f"[{time.time()-t0:.0f}s]", flush=True)
pd.concat(out).to_csv("experimentos/v11/e13_04_evidencia.csv", index=False)
