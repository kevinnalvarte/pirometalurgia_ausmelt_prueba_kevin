"""E12-04 (3) -- prueba prospectiva identica a v9 (mp9.replay_prospectivo / evidencia_prospectiva) para cada estado,
via monkeypatch de mp9.ESTADO_R_V9. Reporta pendiente, t, Spearman parcial, p permutacion bloques, terciles;
todo / DEV / lockbox."""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v10")
import pandas as pd

import e12_04_lib as L
import modelo_predictivo_v9 as mp9

t0 = time.time()
pd.set_option("display.width", 220)

df = L.L8.cargar_df()
df = L.con_estado_planp(df)
ESTADO_ORIG = list(mp9.ESTADO_R_V9)

resultados = []
for nm, S in L.ESTADOS.items():
    mp9.ESTADO_R_V9 = list(S)
    print(f"\n=== estado {nm} ({len(S)} cols) === [{time.time()-t0:6.0f}s]", flush=True)
    esc, pb = mp9.replay_prospectivo(df, log=lambda *a: print(f"  [{time.time()-t0:6.0f}s]", *a, flush=True))
    esc.to_csv(L.SALIDA / f"e12_04_02_replay_escalones_{nm}.csv", index=False)
    pb.to_csv(L.SALIDA / f"e12_04_02_replay_por_batch_{nm}.csv")
    ev = mp9.evidencia_prospectiva(pb)
    ev.insert(0, "estado", nm)
    ev.to_csv(L.SALIDA / f"e12_04_02_evidencia_{nm}.csv", index=False)
    resultados.append(ev)
    print(ev.round(4).to_string())

mp9.ESTADO_R_V9 = ESTADO_ORIG

todo = pd.concat(resultados, ignore_index=True)
todo.to_csv(L.SALIDA / "e12_04_02_evidencia_todos_estados.csv", index=False)
print(f"\n[{time.time()-t0:6.0f}s] listo")
print(todo.round(4).to_string())
