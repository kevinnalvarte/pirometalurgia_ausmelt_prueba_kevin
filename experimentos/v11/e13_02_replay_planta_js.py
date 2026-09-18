"""E13-02 (Fable): replay prospectivo del asesor de PLANTA (estado ejecutable S3') con contraccion JS vs sin contraccion; evidencia, calibracion,
ganancia calibrada de la recomendacion con IC bootstrap por bloques."""
import sys, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import asesor_planta_v9_1 as AP, modelo_predictivo_v9 as mp9
import v8_lib as L8
df = L8.cargar_df(); t0 = time.time()
for modo in ("js", None):
    esc, pb = AP.replay_planta(df, {"contraccion": modo}, log=lambda *a: None)
    tag = modo or "ols"; esc.to_csv(f"experimentos/v11/e13_02_escalones_{tag}.csv", index=False); pb.to_csv(f"experimentos/v11/e13_02_por_batch_{tag}.csv")
    ev = mp9.evidencia_prospectiva(pb); ev.insert(0, "contraccion", tag); ev.to_csv(f"experimentos/v11/e13_02_evidencia_{tag}.csv", index=False)
    pd.set_option("display.width", 250); print(ev.round(4).to_string(), f"[{time.time()-t0:.0f}s]")
    # ganancia calibrada = pendiente x valor_rec medio ; bootstrap por bloques de 10 batches
    import statsmodels.api as sm
    d = pb.dropna(subset=["valor_hist_pp", "valor_rec_pp", mp9.KPI] + mp9.CONTROLES_V9).reset_index(drop=True); rng = np.random.default_rng(0); g = []
    nb = len(d) // 10
    for _ in range(1000):
        idx = np.concatenate([np.arange(b * 10, b * 10 + 10) for b in rng.integers(0, nb, nb)]); dd = d.iloc[idx]
        f = sm.OLS(dd[mp9.KPI], sm.add_constant(dd[["valor_hist_pp"] + mp9.CONTROLES_V9])).fit(); g.append((f.params["valor_hist_pp"], f.params["valor_hist_pp"] * dd["valor_rec_pp"].mean()))
    g = np.array(g); print(tag, "pendiente IC95 [%.2f, %.2f] | valor_rec medio %.2f pp | ganancia calibrada %.2f pp/batch IC95 [%.2f, %.2f] | P(>0) %.3f" % (
        *np.percentile(g[:, 0], [2.5, 97.5]), d.valor_rec_pp.mean(), np.median(g[:, 1]), *np.percentile(g[:, 1], [2.5, 97.5]), (g[:, 1] > 0).mean()))
