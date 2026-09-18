"""E18-01 tarea 3 -- 2SLS: KPI (y canales) ~ ejecutado instrumentado por receta, con controles y tendencia
flexible; HC3 y Newey-West(5). Compara con OLS y con el efecto del desvio (-1.58 pp/sd) convertido a
pp/(Nm3 o kg por min). Reporta IC95 y Durbin-Wu-Hausman (funcion de control)."""
import sys
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
from scipy import stats
import e18_01_lib as E

t = E.tabla_batch()
SAL = E.RAIZ / "experimentos" / "v16"
TENDENCIAS = ["lineal", "cubica", "bloque40"]
OUTCOMES = [E.KPI, "f_dross", "f_polvo", "f_metal", "sn_perdido_escoria_frac"]
ESCALA = {E.KPI: 1.0, "f_dross": 100.0, "f_polvo": 100.0, "f_metal": 100.0, "sn_perdido_escoria_frac": 100.0}

# sd del desvio por orden (referencia v10, para convertir -1.58 pp/sd -> pp/(unidad fisica))
SD_DEV_GN = (2.175, 3.013, 2.983, 2.501)   # min..max de e15_02/e17
SD_DEV_CARBON = (4.195, 7.053)
EFECTO_DESVIO_GN_PP_SD = -1.58   # v10, DEV, t-3.9
EFECTO_DESVIO_GN_FISICO = [EFECTO_DESVIO_GN_PP_SD / s for s in SD_DEV_GN]
print(f"Referencia v10: efecto desvio GN = {EFECTO_DESVIO_GN_PP_SD} pp/sd -> {min(EFECTO_DESVIO_GN_FISICO):.3f} a {max(EFECTO_DESVIO_GN_FISICO):.3f} pp por Nm3/min\n")

filas = []
for regimen, tt in (("TOTAL", t), ("DEV", t[t.es_dev]), ("LB", t[~t.es_dev])):
    for tend in TENDENCIAS:
        Xexog = E.controles_tendencia(tt, tend)
        for p in ("gn", "carbon"):
            for y in OUTCOMES:
                cols = [f"ejec_{p}", f"plan_{p}", y] + list(Xexog.columns)
                base = pd.concat([tt[[f"ejec_{p}", f"plan_{p}", y]], Xexog], axis=1).dropna()
                if len(base) < 30:
                    continue
                yv = base[y].to_numpy(float) * ESCALA[y]
                xend = base[[f"ejec_{p}"]].to_numpy(float)
                zexcl = base[[f"plan_{p}"]].to_numpy(float)
                xexog_nc = base[Xexog.columns].to_numpy(float)
                xexog = np.column_stack([np.ones(len(base)), xexog_nc])
                # OLS
                ols = E.ols_robusto(pd.Series(yv, index=base.index), base[[f"ejec_{p}"] + list(Xexog.columns)], cov="HC3")
                b_ols, se_ols, t_ols = ols.params[f"ejec_{p}"], ols.bse[f"ejec_{p}"], ols.tvalues[f"ejec_{p}"]
                for cov in ("HC1", "HAC"):
                    iv = E.iv2sls(yv, xend, xexog, zexcl, cov=cov, nw_lags=5, nombres_end=[f"ejec_{p}"],
                                  nombres_exog=["const"] + list(Xexog.columns))
                    b, se = iv["beta"][f"ejec_{p}"], iv["se"][f"ejec_{p}"]
                    ci_lo, ci_hi = b - 1.96 * se, b + 1.96 * se
                    dwh = E.dwh_control_function(yv, base[f"ejec_{p}"].to_numpy(float), xexog_nc, base[f"plan_{p}"].to_numpy(float), cov="HC3")
                    filas.append({"regimen": regimen, "tendencia": tend, "palanca": p, "outcome": y, "n": len(base),
                                  "cov_iv": cov, "b_ols": b_ols, "t_ols": t_ols, "b_iv": b, "se_iv": se,
                                  "t_iv": b / se, "ci_lo": ci_lo, "ci_hi": ci_hi, "t_dwh": dwh["t_vhat"], "p_dwh": dwh["p_vhat"]})

res = pd.DataFrame(filas)
res.to_csv(SAL / "e18_01_03_iv_resultados.csv", index=False)
pd.set_option("display.width", 220)
# vista principal: KPI, cov=HC1, palanca gn
print(">> KPI ~ GN ejecutado (instrumentado por receta), HC1:")
print(res[(res.outcome == E.KPI) & (res.palanca == "gn") & (res.cov_iv == "HC1")]
      [["regimen", "tendencia", "n", "b_ols", "t_ols", "b_iv", "se_iv", "t_iv", "ci_lo", "ci_hi", "t_dwh", "p_dwh"]]
      .round(3).to_string(index=False))
print("\n>> KPI ~ carbon ejecutado (instrumentado por receta), HC1:")
print(res[(res.outcome == E.KPI) & (res.palanca == "carbon") & (res.cov_iv == "HC1")]
      [["regimen", "tendencia", "n", "b_ols", "t_ols", "b_iv", "se_iv", "t_iv", "ci_lo", "ci_hi", "t_dwh", "p_dwh"]]
      .round(3).to_string(index=False))
print("\n>> canales (f_dross, f_polvo, f_metal, sn_perdido_escoria_frac) ~ GN, DEV, HC1:")
print(res[(res.outcome != E.KPI) & (res.palanca == "gn") & (res.cov_iv == "HC1") & (res.regimen == "DEV")]
      [["tendencia", "outcome", "n", "b_ols", "t_ols", "b_iv", "t_iv", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
print("\n>> comparacion HAC vs HC1 (KPI, GN, DEV):")
print(res[(res.outcome == E.KPI) & (res.palanca == "gn") & (res.regimen == "DEV")]
      [["tendencia", "cov_iv", "b_iv", "se_iv", "t_iv"]].round(3).to_string(index=False))
