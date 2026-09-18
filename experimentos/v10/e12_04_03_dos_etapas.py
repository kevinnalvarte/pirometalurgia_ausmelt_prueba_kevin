"""E12-04 (4) -- esquema en dos etapas: setpoints recomendados con S3' (inicio del escalon) vs S1 (con el ensayo de t-1,
correccion). Diferencia por orden en unidades fisicas y en sd; % de escalones donde cambia la direccion de la recomendacion."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v10")
import numpy as np
import pandas as pd

import e12_04_lib as L

pd.set_option("display.width", 220)

esc1 = pd.read_csv(L.SALIDA / "e12_04_02_replay_escalones_S1_v9.csv")
esc3 = pd.read_csv(L.SALIDA / "e12_04_02_replay_escalones_S3p_plan.csv")
m = esc1.merge(esc3, on=["Batch", "orden_escalon_fase"], suffixes=("_S1", "_S3p"))
print(f"esc S1: {len(esc1)}  esc S3p: {len(esc3)}  emparejados: {len(m)}")

base = L.cargar_base()
d = L.filas_reduccion(base)
r_s1 = L.residuos_estado(d, L.ESTADO_S1_V9)
sd_orden = {"gn": r_s1["gn"]["sd_orden"], "carbon": r_s1["carbon"]["sd_orden"]}

filas = []
for p in L.PALANCAS:
    m[f"sd_{p}"] = m["orden_escalon_fase"].map(sd_orden[p])
    m[f"diff_{p}"] = m[f"rec__{p}_S3p"] - m[f"rec__{p}_S1"]
    m[f"diff_{p}_sd"] = m[f"diff_{p}"] / m[f"sd_{p}"]
    dir1 = np.sign(m[f"delta__{p}_S1"]).astype(int)
    dir3 = np.sign(m[f"delta__{p}_S3p"]).astype(int)
    m[f"dir_S1__{p}"] = dir1; m[f"dir_S3p__{p}"] = dir3
    m[f"cambia_accion__{p}"] = dir1 != dir3                       # incluye 0 <-> +-1 (recomendar vs mantener)
    ambos_activos = (dir1 != 0) & (dir3 != 0)
    m[f"revierte__{p}"] = ambos_activos & (dir1 != dir3)          # reversion real de signo (ambos recomiendan, pero opuesto)
    for o, g in m.groupby("orden_escalon_fase"):
        filas.append({"palanca": p, "orden": int(o), "n": len(g),
                      "media_abs_diff_fisica": float(g[f"diff_{p}"].abs().mean()),
                      "media_diff_fisica": float(g[f"diff_{p}"].mean()),
                      "media_abs_diff_sd": float(g[f"diff_{p}_sd"].abs().mean()),
                      "mediana_abs_diff_sd": float(g[f"diff_{p}_sd"].abs().median()),
                      "pct_cambia_accion": float(g[f"cambia_accion__{p}"].mean()),
                      "n_ambos_activos": int((ambos_activos & (m["orden_escalon_fase"] == o)).sum()),
                      "pct_revierte_dado_ambos_activos": float(g.loc[ambos_activos[g.index], f"revierte__{p}"].mean()) if ambos_activos[g.index].sum() else np.nan})
    filas.append({"palanca": p, "orden": "TODOS", "n": len(m),
                  "media_abs_diff_fisica": float(m[f"diff_{p}"].abs().mean()),
                  "media_diff_fisica": float(m[f"diff_{p}"].mean()),
                  "media_abs_diff_sd": float(m[f"diff_{p}_sd"].abs().mean()),
                  "mediana_abs_diff_sd": float(m[f"diff_{p}_sd"].abs().median()),
                  "pct_cambia_accion": float(m[f"cambia_accion__{p}"].mean()),
                  "n_ambos_activos": int(ambos_activos.sum()),
                  "pct_revierte_dado_ambos_activos": float(m.loc[ambos_activos, f"revierte__{p}"].mean()) if ambos_activos.sum() else np.nan})

tab = pd.DataFrame(filas)
tab.to_csv(L.SALIDA / "e12_04_03_dos_etapas_por_orden.csv", index=False)
print(tab.round(3).to_string())

m.to_csv(L.SALIDA / "e12_04_03_dos_etapas_escalones.csv", index=False)
print("listo")
