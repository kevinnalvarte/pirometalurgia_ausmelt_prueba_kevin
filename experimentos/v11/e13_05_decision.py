"""E13-05 (Fable): lectura de DECISION del replay estricto final (estado ejecutable + JS + sesgo reciente k=20): ganancia calibrada de seguir la
recomendacion, P(ganancia>0), peor caso plausible, por bootstrap de bloques de 10 batches; y lo mismo para el canal dross."""
import numpy as np, pandas as pd, statsmodels.api as sm
C = ["feed_sn_total_t", "ley_sn_conc_batch_pct", "frac_carga_secundaria_F", "feed_dross_Fe_total_t", "idx_cronologico"]; KPI = "recuperacion_refinada_pct"
for k in (20, 40):
    d = pd.read_csv(f"experimentos/v11/e13_04_por_batch_k{k}.csv", index_col=0).dropna(subset=["valor_hist_pp", "valor_rec_pp", KPI] + C).reset_index(drop=True)
    rng = np.random.default_rng(0); nb = len(d) // 10; G = []
    for _ in range(2000):
        idx = np.concatenate([np.arange(b * 10, b * 10 + 10) for b in rng.integers(0, nb, nb)]); dd = d.iloc[idx]
        s = sm.OLS(dd[KPI], sm.add_constant(dd[["valor_hist_pp"] + C])).fit().params["valor_hist_pp"]; G.append((s, s * dd.valor_rec_pp.mean()))
    G = np.array(G)
    print(f"k={k}: pendiente {np.median(G[:,0]):.2f} IC95 [{np.percentile(G[:,0],2.5):.2f}, {np.percentile(G[:,0],97.5):.2f}] | ganancia calibrada {np.median(G[:,1]):.2f} pp/batch "
          f"IC95 [{np.percentile(G[:,1],2.5):.2f}, {np.percentile(G[:,1],97.5):.2f}] IC90 [{np.percentile(G[:,1],5):.2f}, {np.percentile(G[:,1],95):.2f}] | P(>0) {np.mean(G[:,1]>0):.3f} | P(>0.25 pp) {np.mean(G[:,1]>0.25):.3f} | valor_rec medio {d.valor_rec_pp.mean():.2f}")
