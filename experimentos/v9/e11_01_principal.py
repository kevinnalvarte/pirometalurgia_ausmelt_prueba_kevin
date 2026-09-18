"""E11-01 tareas 1, 2 y 4: regresion conjunta (spec9 y spec6), q-BH, estabilidad por tercios + bootstrap,
ponderacion por duracion y exposicion fisica acumulada."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "experimentos/v9")
import v9_lib as L  # noqa: E402
import e11_01_common as C  # noqa: E402

OUT = Path("experimentos/v9")


def main():
    res = L.residuos_escalon()
    t = L.tabla_batch(res=res)
    t = t.join(C.exposiciones_R_combinada(res)).join(C.exposiciones_fisicas(res))
    t_durw = L.tabla_batch(res=res, ponderar_dur=True).join(C.exposiciones_R_combinada(res, ponderar_dur=True))

    exp9 = [f"x_{g}_{p}" for g in ("F", "Rt", "Rl") for p in C.PALANCAS3]
    exp6 = [f"x_{g}_{p}" for g in ("F", "R") for p in C.PALANCAS3]
    ctrl_full = L.CONTROLES
    ctrl_lb = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]

    dev, lockbox, total = t[t.es_dev], t[~t.es_dev], t

    # ------------------------------------------------------------------ tarea 1: regresion conjunta + BH
    filas = []
    for nombre_spec, exps in (("spec9", exp9), ("spec6", exp6)):
        for nombre_c, sub, ctl in (("dev", dev, ctrl_full), ("lockbox", lockbox, ctrl_lb), ("total", total, ctrl_full)):
            tab = C.regresion_conjunta(sub, exps, ctl)
            tab["spec"] = nombre_spec; tab["conjunto"] = nombre_c
            filas.append(tab)
    ols = pd.concat(filas, ignore_index=True)
    # q-BH: familia = exposiciones x KPI, DENTRO de (spec, conjunto)
    ols["q_bh"] = np.nan
    for (spec, conj), idx in ols[(ols.y == L.KPI)].groupby(["spec", "conjunto"]).groups.items():
        ols.loc[idx, "q_bh"] = C.bh(ols.loc[idx, "p"]).to_numpy()
    ols.to_csv(OUT / "e11_01_ols_principal.csv", index=False)

    # ------------------------------------------------------------------ tarea 2a: estabilidad por tercios (DEV, spec6, KPI)
    d = dev.dropna(subset=exp6 + ctrl_full + [L.KPI]).copy()
    terc = pd.qcut(d["idx_cronologico"], 3, labels=["T1", "T2", "T3"])
    filas_t = []
    for tt in ["T1", "T2", "T3"]:
        sub = d[terc == tt]
        tab = C.ols_familia(sub[L.KPI], sub[exp6 + ctrl_full], exp6)
        tab["tercio"] = tt
        filas_t.append(tab)
    tab_full = C.ols_familia(d[L.KPI], d[exp6 + ctrl_full], exp6); tab_full["tercio"] = "DEV_completo"
    filas_t.append(tab_full)
    estab = pd.concat(filas_t, ignore_index=True)
    estab.to_csv(OUT / "e11_01_estabilidad_tercios.csv", index=False)

    # ------------------------------------------------------------------ tarea 2b: bootstrap por batch (spec6, KPI, DEV)
    boot = C.bootstrap_batch_coefs(dev, exp6, ctrl_full, L.KPI, n_boot=1000, seed=42)
    boot.to_csv(OUT / "e11_01_bootstrap_coefs.csv", index=False)

    # ------------------------------------------------------------------ tarea 4: ponderacion por duracion vs fisica vs z simple
    filas_p = []
    tab_z = C.ols_familia(dev[L.KPI], dev[exp6 + ctrl_full].dropna().pipe(lambda x: x), exp6) if False else None
    dz = dev.dropna(subset=exp6 + ctrl_full + [L.KPI])
    tab_z = C.ols_familia(dz[L.KPI], dz[exp6 + ctrl_full], exp6); tab_z["forma"] = "z_media"
    filas_p.append(tab_z)
    ddw = t_durw[t_durw.es_dev].dropna(subset=exp6 + ctrl_full + [L.KPI])
    tab_w = C.ols_familia(ddw[L.KPI], ddw[exp6 + ctrl_full], exp6); tab_w["forma"] = "z_ponderada_dur"
    filas_p.append(tab_w)
    expf = [f"xf_{g}_{p}" for g in ("F", "Rt", "Rl") for p in C.PALANCAS3]
    dff = dev.dropna(subset=expf + ctrl_full + [L.KPI])
    tab_f = C.ols_familia(dff[L.KPI], dff[expf + ctrl_full], expf); tab_f["forma"] = "fisica_acumulada"
    filas_p.append(tab_f)
    pond = pd.concat(filas_p, ignore_index=True)
    pond["t"] = pond["coef"] / ((pond["ci_hi"] - pond["ci_lo"]) / (2 * 1.96))
    pond.to_csv(OUT / "e11_01_ponderacion.csv", index=False)

    print("=== tarea1 (spec6, DEV, KPI) ===")
    print(ols[(ols.spec == "spec6") & (ols.conjunto == "dev") & (ols.y == L.KPI)].to_string(index=False))
    print("=== tarea2 estabilidad (coef por tercio, principales) ===")
    print(estab[estab.exposicion.isin(["x_R_gn", "x_R_carbon", "x_F_carbon", "x_F_exo2"])].to_string(index=False))
    print("=== tarea2 bootstrap ===")
    print(boot.to_string(index=False))
    print("=== tarea4 ponderacion |t| ===")
    print(pond[pond.exposicion.isin(["x_R_gn", "x_R_carbon", "x_F_carbon", "x_F_exo2",
                                     "xf_Rt_gn", "xf_Rl_gn", "xf_Rt_carbon", "xf_Rl_carbon", "xf_F_carbon", "xf_F_exo2"])][
        ["forma", "exposicion", "coef", "p", "t"]].to_string(index=False))


if __name__ == "__main__":
    main()
