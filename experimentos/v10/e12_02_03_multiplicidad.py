"""E12-02 tarea 4: correccion de multiplicidad. p del MEJOR esquema (mayor t observado) frente a la distribucion
nula del MAXIMO de t sobre los 7 esquemas (misma familia de permutaciones en bloques de 20 usada en e12_02_02,
para que la comparacion sea apareada por sorteo). Se hace por separado en total/DEV/lockbox."""
import sys; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
import e12_02_lib as C

obs = pd.read_csv(C.SALIDA / "e12_02_esquemas_observado.csv")
npz = np.load(C.SALIDA / "e12_02_nulas_dist.npz")
MASKS = ["total", "dev", "lockbox"]


def p_uni(obs_v, null_arr):
    return float((np.sum(null_arr >= obs_v) + 1) / (len(null_arr) + 1))


filas = []
for m in MASKS:
    sub = obs[obs.muestra == m].copy()
    esquemas = sub["esquema"].tolist()
    null_mat = np.column_stack([npz[f"t__{e}__{m}"] for e in esquemas])
    null_max = null_mat.max(axis=1)
    i_best = int(sub["t_clasico"].to_numpy().argmax())
    mejor = esquemas[i_best]
    t_best = float(sub["t_clasico"].iloc[i_best])
    p_simple = p_uni(t_best, npz[f"t__{mejor}__{m}"])          # sin corregir (ya reportado en e12_02_02)
    p_mult = p_uni(t_best, null_max)                            # corregido por buscar el mejor de 7 esquemas
    filas.append({"muestra": m, "mejor_esquema": mejor, "t_obs": t_best, "p_sin_corregir": p_simple,
                  "p_corregido_max_7_esquemas": p_mult, "n_perm": len(null_max)})
out = pd.DataFrame(filas)
out.to_csv(C.SALIDA / "e12_02_multiplicidad.csv", index=False)
pd.set_option("display.width", 200)
print("=== Correccion de multiplicidad: p del mejor esquema vs nula del maximo sobre 7 esquemas ===")
print(out.round(4).to_string())
