"""E9-08 (Fable) -- potencia del diseno observacional de evidencia off-policy (adherencia vs KPI) frente al efecto que la
propia cadena predice. Pregunta: ¿puede la regresion KPI ~ dist_R (+controles) detectar el uplift que el modelo predice?

  - Efecto esperado por +1 sd de dist_R: si el batch se aleja 1 sd de la politica, pierde (aprox.) el uplift predicho por
    unidad de distancia: pendiente esperada = -cov(uplift_pred, dist)/var(dist) ... aqui se usa la version directa: regresion del
    uplift PREDICHO por batch (pp) sobre dist_R -> pendiente "esperada" en pp/sd si el modelo fuera exacto.
  - MDE (efecto minimo detectable al 80 % de potencia, alfa 0.05, dos colas) = 2.8 * se(coef) con la se HC3 observada.
  - Numero de batches necesario para detectar el efecto esperado: n_req = n * (MDE/efecto_esperado)^2.
Salida: e9_08_potencia.csv y tabla en consola.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_lib as L  # noqa: E402
import modelo_predictivo_v7 as mp7  # noqa: E402

OUT = Path(__file__).resolve().parent
pb = pd.read_csv(OUT / "e9_07_por_batch_v7.csv")
C = mp7.CONTROLES_KPI_V7
K = "recuperacion_refinada_pct"
filas = []
for nombre, sub in [("DEV", pb[~pb["es_lockbox"]]), ("TOTAL", pb)]:
    for v, up in [("dist_R", "uplift_batch_pp"), ("dist_F", "uplift_batch_pp"), ("dist_total", "uplift_batch_pp")]:
        d = sub[[K, v, up] + C].dropna()
        X = d[[v] + C]; X = (X - X.mean()) / X.std().replace(0, 1)
        res = sm.OLS(d[K], sm.add_constant(X)).fit(cov_type="HC3")
        se = float(res.bse[v]); coef = float(res.params[v]); mde = 2.8 * se
        # efecto esperado: uplift predicho (pp) vs dist en sd
        z = (d[v] - d[v].mean()) / d[v].std()
        esp = sm.OLS(d[up], sm.add_constant(z)).fit()
        # el uplift predicho es lo que se gana al ADHERIR; alejarse 1 sd cuesta ~ pendiente del uplift sobre la distancia
        efecto_esp = -abs(float(esp.params.iloc[1])) if abs(float(esp.params.iloc[1])) > 1e-6 else -float(d[up].mean()) * 0.5
        n_req = len(d) * (mde / efecto_esp) ** 2 if efecto_esp != 0 else np.nan
        filas.append(dict(muestra=nombre, variable=v, n=len(d), coef_obs_pp_sd=coef, se_hc3=se, MDE_80pct=mde,
                          uplift_pred_medio_pp=float(d[up].mean()), pendiente_uplift_vs_dist_pp_sd=float(esp.params.iloc[1]),
                          efecto_esperado_pp_sd=efecto_esp, n_batches_requeridos=n_req, potencia_actual=float(
                              1 - __import__("scipy").stats.norm.cdf(1.96 - abs(efecto_esp) / se) + __import__("scipy").stats.norm.cdf(-1.96 - abs(efecto_esp) / se))))
t = pd.DataFrame(filas)
pd.set_option("display.width", 250)
print(t.round(4).to_string())
t.to_csv(OUT / "e9_08_potencia.csv", index=False)
print("FIN")
