"""E11-01 tarea 5: placebos.
(i) exposicion del batch SIGUIENTE (shift -1 cronologico) -> KPI actual, controlando por la actual.
(ii) exposicion actual -> variables PRE-TRATAMIENTO (ley_sn_conc_batch_pct, feed_sn_total_t, F_lnirf_F0, KPI_prev).
(iii) permutacion en bloques cronologicos de 30 batches (500 reps) -> nula del t de GN_R y carbon_R (spec6, DEV).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experimentos/v9")
import v9_lib as L  # noqa: E402
import e11_01_common as C  # noqa: E402

OUT = Path("experimentos/v9")
exp6 = [f"x_{g}_{p}" for g in ("F", "R") for p in C.PALANCAS3]
PRINCIPALES = ["x_R_gn", "x_R_carbon", "x_F_carbon", "x_F_exo2"]


def main():
    res = L.residuos_escalon()
    t = L.tabla_batch(res=res).join(C.exposiciones_R_combinada(res)).sort_values("idx_cronologico")
    dev = t[t.es_dev].copy()

    # ---------------------------------------------------------------- (i) batch siguiente
    for e in exp6:
        t[f"{e}__next"] = t[e].shift(-1)  # orden cronologico global (continuo Batch a Batch)
    dev_i = t[t.es_dev].copy()
    filas_i = []
    for e in PRINCIPALES:
        cols = [e, f"{e}__next"] + L.CONTROLES
        d = dev_i.dropna(subset=cols + [L.KPI])
        fit = L.ols_hc3(d[L.KPI], d[cols])
        ci = fit.conf_int()
        filas_i.append({"exposicion": e, "coef_actual": float(fit.params[e]), "p_actual": float(fit.pvalues[e]),
                        "coef_siguiente": float(fit.params[f"{e}__next"]), "p_siguiente": float(fit.pvalues[f"{e}__next"]),
                        "ci_lo_siguiente": float(ci.loc[f"{e}__next", 0]), "ci_hi_siguiente": float(ci.loc[f"{e}__next", 1]), "n": int(fit.nobs)})
    placebo1 = pd.DataFrame(filas_i)
    placebo1.to_csv(OUT / "e11_01_placebo_siguiente.csv", index=False)

    # ---------------------------------------------------------------- (ii) pre-tratamiento
    dev["kpi_prev"] = dev[L.KPI].shift(1)  # orden cronologico DENTRO de DEV (fila anterior)
    pretrat = ["ley_sn_conc_batch_pct", "feed_sn_total_t", "F_lnirf_F0", "kpi_prev"]
    filas_ii = []
    for y in pretrat:
        ctl = [c for c in L.CONTROLES if c != y]
        cols = exp6 + ctl
        d = dev.dropna(subset=cols + [y])
        fit = L.ols_hc3(d[y], d[cols])
        ci = fit.conf_int()
        for e in exp6:
            filas_ii.append({"pretratamiento": y, "exposicion": e, "coef": float(fit.params[e]), "p": float(fit.pvalues[e]),
                             "ci_lo": float(ci.loc[e, 0]), "ci_hi": float(ci.loc[e, 1]), "n": int(fit.nobs)})
    placebo2 = pd.DataFrame(filas_ii)
    placebo2.to_csv(OUT / "e11_01_placebo_pretratamiento.csv", index=False)

    # ---------------------------------------------------------------- (iii) permutacion en bloques de 30
    d = dev.dropna(subset=exp6 + L.CONTROLES + [L.KPI]).sort_values("idx_cronologico").reset_index()
    n = len(d)
    bloque = (np.arange(n) // 30)
    y = d[L.KPI].to_numpy(float); Xc = d[L.CONTROLES].to_numpy(float); Xe = d[exp6].to_numpy(float)

    def t_obs(Xe_):
        Z = pd.DataFrame(np.column_stack([Xe_, Xc]), columns=exp6 + L.CONTROLES)
        fit = L.ols_hc3(pd.Series(y), Z)
        return float(fit.tvalues["x_R_gn"]), float(fit.tvalues["x_R_carbon"])

    t_gn_obs, t_c_obs = t_obs(Xe)
    rng = np.random.default_rng(42)
    n_perm = 500
    t_gn_null, t_c_null = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        Xe_p = Xe.copy()
        for b in np.unique(bloque):
            idx = np.where(bloque == b)[0]
            perm = rng.permutation(idx)
            Xe_p[idx] = Xe[perm]
        t_gn_null[i], t_c_null[i] = t_obs(Xe_p)
    p_gn = float((np.sum(np.abs(t_gn_null) >= abs(t_gn_obs)) + 1) / (n_perm + 1))
    p_c = float((np.sum(np.abs(t_c_null) >= abs(t_c_obs)) + 1) / (n_perm + 1))
    perm = pd.DataFrame({"t_gn_null": t_gn_null, "t_carbon_null": t_c_null})
    perm.to_csv(OUT / "e11_01_permutacion_bloques.csv", index=False)
    resumen = pd.DataFrame([
        {"exposicion": "x_R_gn", "t_obs": t_gn_obs, "p_perm_bilateral": p_gn, "media_null": float(t_gn_null.mean()), "sd_null": float(t_gn_null.std())},
        {"exposicion": "x_R_carbon", "t_obs": t_c_obs, "p_perm_bilateral": p_c, "media_null": float(t_c_null.mean()), "sd_null": float(t_c_null.std())},
    ])
    resumen.to_csv(OUT / "e11_01_permutacion_resumen.csv", index=False)

    print("=== placebo (i) batch siguiente ===")
    print(placebo1.to_string(index=False))
    print("=== placebo (ii) pre-tratamiento ===")
    print(placebo2[placebo2.exposicion.isin(PRINCIPALES)].to_string(index=False))
    print("=== permutacion bloques 30 (n=500) ===")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
