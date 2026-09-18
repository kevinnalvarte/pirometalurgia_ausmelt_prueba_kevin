"""E18-02 tarea 1: modelo base by-escalon con efectos fijos de batch (+orden), exposicion exacta (desvio vs receta).
Compara: (0) sin efectos fijos, (1) within batch+orden con dev, (2) within con carbon en forma cinetica
dev_carbon x Sn_inv_prev/1000, (3) within con el NIVEL ejecutado en vez del desvio.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import e18_02_lib as L

SAL = Path(__file__).parent


def main():
    r, bt = L.cargar()
    filas = []
    for clave, target in L.TARGETS.items():
        base_X = ["dev__gn", "dev__carbon"] + L.STATE
        kin_X = ["dev__gn", "dev_carbon_x_sn"] + L.STATE
        lvl_X = ["tasa_gn_nm3_min", "tasa_feed_Carbon_kg_min"] + L.STATE

        especs = {
            "0_sin_FE": (base_X, False),
            "1_within_dev": (base_X, True),
            "2_within_carbon_cinetico": (kin_X, True),
            "3_within_nivel": (lvl_X, True),
        }
        for tag, (X, fe) in especs.items():
            tab = L.within_ols(r, target, X, extra_fe=fe)
            tab.insert(0, "target", clave); tab.insert(1, "especificacion", tag)
            for k, v in L.resumen_attrs(tab).items():
                tab[k] = v
            filas.append(tab)
    out = pd.concat(filas, ignore_index=True)
    out.to_csv(SAL / "e18_02_theta_base.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["target", "especificacion", "variable", "theta", "ci_lo", "ci_hi", "t", "p", "theta_por_sd", "n", "r2_within"]
    print(out[cols].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
