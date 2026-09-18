"""E11-01 tarea 7: dosis-respuesta no parametrica. KPI residualizado por controles (DEV) vs quintiles de cada
exposicion principal: media + IC bootstrap por batch, chequeo de monotonia y linealidad (R2 lineal vs "escalon")."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experimentos/v9")
import v9_lib as L  # noqa: E402
import e11_01_common as C  # noqa: E402

OUT = Path("experimentos/v9")
PRINCIPALES = ["x_R_gn", "x_R_carbon", "x_F_carbon", "x_F_exo2"]


def boot_media(x: np.ndarray, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    res = L.residuos_escalon()
    t = L.tabla_batch(res=res).join(C.exposiciones_R_combinada(res))
    dev = t[t.es_dev].dropna(subset=PRINCIPALES + L.CONTROLES + [L.KPI]).copy()
    kpi_res = L.ols_hc3(dev[L.KPI], dev[L.CONTROLES]).resid
    dev["kpi_res"] = kpi_res

    filas = []
    for e in PRINCIPALES:
        q = pd.qcut(dev[e], 5, labels=False, duplicates="drop")
        meds = dev.groupby(q)[e].median()
        for qi in sorted(q.dropna().unique()):
            sub = dev.loc[q == qi, "kpi_res"].to_numpy()
            m, lo, hi = boot_media(sub)
            filas.append({"exposicion": e, "quintil": int(qi) + 1, "mediana_x": float(meds.loc[qi]), "n": len(sub),
                          "media_kpi_res": m, "ci_lo": lo, "ci_hi": hi})
    dosis = pd.DataFrame(filas)

    # monotonia + linealidad (R2 de un ajuste lineal en la mediana_x de cada quintil vs media_kpi_res)
    resumen = []
    for e in PRINCIPALES:
        sub = dosis[dosis.exposicion == e].sort_values("quintil")
        d = np.diff(sub["media_kpi_res"].to_numpy())
        mono = "creciente" if (d >= 0).all() else ("decreciente" if (d <= 0).all() else "no_monotona")
        # R2 lineal (5 puntos) vs varianza total entre quintiles
        x = sub["mediana_x"].to_numpy(); y = sub["media_kpi_res"].to_numpy()
        b = np.polyfit(x, y, 1); yhat = np.polyval(b, x)
        ss_tot = np.sum((y - y.mean()) ** 2); ss_res = np.sum((y - yhat) ** 2)
        r2_lineal = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        resumen.append({"exposicion": e, "patron": mono, "signos_pasos": list(np.sign(d).astype(int)),
                        "r2_ajuste_lineal_quintiles": round(float(r2_lineal), 3),
                        "d_q5_q1": float(sub["media_kpi_res"].iloc[-1] - sub["media_kpi_res"].iloc[0])})
    resumen = pd.DataFrame(resumen)

    dosis.to_csv(OUT / "e11_01_dosis_respuesta.csv", index=False)
    resumen.to_csv(OUT / "e11_01_dosis_respuesta_resumen.csv", index=False)
    print(dosis.to_string(index=False))
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
