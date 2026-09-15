"""Construye una sola vez el dataset v3 y lo guarda en experimentos/cache/df_v3.pkl (uso interno de los
experimentos 13+; el pickle NO es fuente de verdad, se regenera con este script)."""
import sys, time
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_predictivo_v3 as mp3
t0 = time.time()
df = mp3.construir_dataset_modelo_v3(RAIZ / "Datos Lingo smelter fase II.xlsx")
(RAIZ / "experimentos" / "cache").mkdir(exist_ok=True)
df.to_pickle(RAIZ / "experimentos" / "cache" / "df_v3.pkl")
print("dataset", df.shape, f"{time.time()-t0:.0f}s")
