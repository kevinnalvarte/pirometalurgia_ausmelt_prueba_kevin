"""E11-01 tarea 3: sensibilidad de los coeficientes principales al estado de la politica de comportamiento E[a|S]
(a = libreria, b = pobre: orden + leyes previas Sn/FeO + temperatura, c = rico: a + cum_gn/cum_o2/tiempo_fase/basicidad_B2/m6_avance)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "experimentos/v9")
import v9_lib as L  # noqa: E402
import e11_01_common as C  # noqa: E402

OUT = Path("experimentos/v9")

ESTADO_R_POBRE = ["orden_escalon_fase", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "temperatura_horno_celsius_prev"]
ESTADO_F_POBRE = ["orden_escalon_fase", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "temperatura_horno_celsius_prev"]
EXTRA_RICO = ["cum_gn_nm3_prev", "cum_o2_nm3_prev", "tiempo_fase", "basicidad_B2_prev", "m6_avance_prev"]
ESTADO_R_RICO = list(L.ESTADO_R) + EXTRA_RICO
ESTADO_F_RICO = list(L.ESTADO_F) + EXTRA_RICO

VARIANTES = {
    "a_libreria": {"Reducción": L.ESTADO_R, "Fusión": L.ESTADO_F},
    "b_pobre": {"Reducción": ESTADO_R_POBRE, "Fusión": ESTADO_F_POBRE},
    "c_rico": {"Reducción": ESTADO_R_RICO, "Fusión": ESTADO_F_RICO},
}

exp6 = [f"x_{g}_{p}" for g in ("F", "R") for p in C.PALANCAS3]


def main():
    df = L.L8.cargar_df()
    filas = []
    for nombre, estados in VARIANTES.items():
        t0 = time.time()
        L.L8.asegurar_seguras([c for c in estados["Reducción"] if c not in ("tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min")])
        L.L8.asegurar_seguras([c for c in estados["Fusión"] if c not in ("tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min")])
        res = L.residuos_escalon(df, estados=estados, cache=None)
        t = L.tabla_batch(df, res=res).join(C.exposiciones_R_combinada(res))
        dev = t[t.es_dev].dropna(subset=exp6 + L.CONTROLES + [L.KPI])
        tab = C.ols_familia(dev[L.KPI], dev[exp6 + L.CONTROLES], exp6)
        tab["variante"] = nombre
        tab["n_estado_R"] = len(estados["Reducción"]); tab["n_estado_F"] = len(estados["Fusión"])
        filas.append(tab)
        print(nombre, f"{time.time()-t0:.1f}s", "n=", len(dev))
    out = pd.concat(filas, ignore_index=True)
    out.to_csv(OUT / "e11_01_estado_sensibilidad.csv", index=False)
    print(out[["variante", "exposicion", "coef", "p", "n_estado_R"]].to_string(index=False))


if __name__ == "__main__":
    main()
