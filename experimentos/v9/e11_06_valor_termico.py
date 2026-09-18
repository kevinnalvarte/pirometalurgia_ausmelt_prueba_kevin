"""E11-06 (Fable): valor del fuego de lanza (GN) condicionado al estado termico + carbon; especificacion fijada a priori:
T* = mediana DEV de temperatura_horno_celsius_prev en Reduccion (sin ajuste). Prueba anidada OOF (criterio 4 del diseno)."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import v9_lib as L, v8_lib as L8, modelo_predictivo_v8 as mp8
df = L8.cargar_df(); res = L.residuos_escalon(df); base = mp8._preparar_base(df)
key = ["Batch", "fase_proceso", "orden_escalon_fase"]
R = res[res.fase_proceso == "Reducción"].merge(base[key + ["temperatura_horno_celsius_prev"]].drop_duplicates(key), on=key, how="left")
T = R["temperatura_horno_celsius_prev"]; TSTAR = float(T[R.es_dev].median()); print("T* =", TSTAR, " % escalones calientes DEV/LB:", (T[R.es_dev] > TSTAR).mean().round(2), (T[~R.es_dev] > TSTAR).mean().round(2))
def expos(tstar):
    hot = (T > tstar)
    g = R.Batch
    return pd.DataFrame({"xG": R.z__gn.groupby(g).mean(), "xC": R.z__carbon.groupby(g).mean(),
                         "xG_hot": R.z__gn.where(hot, 0).groupby(g).mean(), "xG_cold": R.z__gn.where(~hot, 0).groupby(g).mean(),
                         "xC_hot": R.z__carbon.where(hot, 0).groupby(g).mean(), "xC_cold": R.z__carbon.where(~hot, 0).groupby(g).mean(),
                         "frac_hot": hot.groupby(g).mean()})
b = L8.construir_batch(df); b = b.set_index("Batch") if "Batch" in b.columns else b
t = b.join(expos(TSTAR)).sort_values("idx_cronologico"); dev = t[t.es_dev]; lb = t[~t.es_dev]
t.to_csv(L.SALIDA / "e11_06_tabla_batch.csv")
specs = {"M1_G+C": ["xG", "xC"], "M2_Ghot+C": ["xG_hot", "xC"], "M3_Ghot": ["xG_hot"], "M4_Ghot+Gcold+C": ["xG_hot", "xG_cold", "xC"], "M5_C": ["xC"],
         "M6_Ghot+Chot+Ccold": ["xG_hot", "xC_hot", "xC_cold"]}
sg = {"xG": -1, "xG_hot": -1, "xC": +1, "xC_hot": +1, "xC_cold": +1}
filas = []
for nm, X in specs.items():
    for signos in (None, sg):
        for muestra, tt, esq, ctr in (("DEV", dev, "gkf", L.CONTROLES), ("DEV", dev, "crono", L.CONTROLES), ("TOTAL", t, "crono", L.CONTROLES), ("TOTAL", t, "gkf", L.CONTROLES)):
            r = L.prueba_anidada(tt, X, esquema=esq, n_folds=6 if muestra == "TOTAL" else 5, signos={k: v for k, v in (signos or {}).items() if k in X} or None, controles=ctr)
            r.pop("score"); r.update({"spec": nm, "signos": signos is not None, "muestra": muestra}); filas.append(r)
out = pd.DataFrame(filas); out.to_csv(L.SALIDA / "e11_06_prueba_anidada.csv", index=False)
print(out[["spec", "signos", "muestra", "esquema", "n", "pendiente", "t", "p_unilateral", "p_perm_unilateral", "spearman_parcial", "kpi_tercio_bajo", "kpi_tercio_alto", "beta_medio"]].round(3).to_string())
# coeficientes por regimen (OLS HC3) y canales
for nm, tt in (("DEV", dev), ("LB", lb), ("TOTAL", t)):
    ctr = [c for c in L.CONTROLES if nm != "LB" or c != "espesor_ladrillo_norm_mm"]
    for y in (L.KPI, "f_dross", "f_metal", "f_polvo", "sn_perdido_escoria_frac"):
        X = ["xG_hot", "xG_cold", "xC"]; dd = tt.dropna(subset=X + ctr + [y]); f = L.ols_hc3(dd[y] * (1 if y == L.KPI else 100), dd[X + ctr])
        print(nm, y[:10], len(dd), " ".join(f"{c}:{f.params[c]:+.2f}[{f.conf_int().loc[c,0]:+.2f},{f.conf_int().loc[c,1]:+.2f}](p{f.pvalues[c]:.3f})" for c in X))
# sensibilidad al umbral (no se usa para elegir): t del coeficiente xG_hot en DEV
for ts in (1000, 1020, 1040, 1060, TSTAR, 1080, 1100, 1120):
    tt = b.join(expos(ts)); tt = tt[tt.es_dev]; X = ["xG_hot", "xG_cold", "xC"]; dd = tt.dropna(subset=X + L.CONTROLES + [L.KPI]); f = L.ols_hc3(dd[L.KPI], dd[X + L.CONTROLES])
    print("umbral", ts, "hot %.2f" % (T[R.es_dev] > ts).mean(), " ".join(f"{c}:{f.params[c]:+.2f}(t{f.tvalues[c]:+.2f})" for c in X))
