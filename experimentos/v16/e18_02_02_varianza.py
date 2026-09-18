"""E18-02 tarea 2: cuanta variacion intra-batch hay en el desvio de GN y carbon. Descomposicion entre/dentro de
batch, por orden, y correlacion entre ordenes (si el operador "sube todo junto" dentro del batch, eso limita la
potencia del diseno within)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import e18_02_lib as L

SAL = Path(__file__).parent


def descomponer(r: pd.DataFrame, col: str) -> dict:
    d = r.dropna(subset=[col])
    total = d[col].var(ddof=1)
    medias_b = d.groupby(L.BATCH)[col].transform("mean")
    entre = medias_b.drop_duplicates().var(ddof=1)  # var de las medias por batch (aprox: usa 1 por batch)
    entre = d.groupby(L.BATCH)[col].mean().var(ddof=1)
    dentro = (d[col] - medias_b).var(ddof=1)
    return {"var_total": total, "var_entre_batch": entre, "var_dentro_batch": dentro,
            "frac_dentro": dentro / total, "icc_entre": entre / total}


def main():
    r, bt = L.cargar()
    filas = []
    for col in ("dev__gn", "dev__carbon"):
        row = descomponer(r, col); row["variable"] = col; row["nivel"] = "global"
        filas.append(row)
        for o, g in r.groupby(L.ORDEN):
            row = descomponer(g, col); row["variable"] = col; row["nivel"] = f"orden_{o}"
            filas.append(row)
    varianza = pd.DataFrame(filas)[["variable", "nivel", "var_total", "var_entre_batch", "var_dentro_batch", "frac_dentro", "icc_entre"]]
    varianza.to_csv(SAL / "e18_02_varianza.csv", index=False)
    print(varianza.round(4).to_string(index=False))

    # correlacion entre ordenes (piv Batch x orden) del desvio crudo: si el operador sube/baja el escalon completo
    # junto, las columnas R0..R3 deberian correlacionar fuerte.
    filas_corr = []
    for col in ("dev__gn", "dev__carbon"):
        piv = r.pivot(index=L.BATCH, columns=L.ORDEN, values=col)
        piv.columns = [f"R{int(o)}" for o in piv.columns]
        corr = piv.corr()
        corr.index.name = "orden_i"; corr = corr.reset_index(); corr.insert(0, "variable", col)
        filas_corr.append(corr)
    corr_ordenes = pd.concat(filas_corr, ignore_index=True)
    corr_ordenes.to_csv(SAL / "e18_02_corr_ordenes.csv", index=False)
    print("\nCorrelacion del desvio crudo entre ordenes (mismo batch):")
    print(corr_ordenes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
