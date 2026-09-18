"""E11-06b (Fable): concordancia OOF de las especificaciones que corresponden a politicas desplegables.
Compuerta termica = regla de SOPORTE (no ajustada al KPI): T_prev >= P10 de DEV en Reduccion. Todas las specs probadas se reportan."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import v9_lib as L, v8_lib as L8, modelo_predictivo_v8 as mp8
df = L8.cargar_df(); res = L.residuos_escalon(df); base = mp8._preparar_base(df)
key = ["Batch", "fase_proceso", "orden_escalon_fase"]
R = res[res.fase_proceso == "Reducción"].merge(base[key + ["temperatura_horno_celsius_prev", "ley_sn_escoria_pct_prev"]].drop_duplicates(key), on=key, how="left")
T = R["temperatura_horno_celsius_prev"]; P10 = float(T[R.es_dev].quantile(.10)); print("P10 DEV T_prev:", round(P10), "| % escalones con compuerta abierta DEV/LB:", (T[R.es_dev] >= P10).mean().round(2), (T[~R.es_dev] >= P10).mean().round(2))
gate = (T >= P10); late = R.orden_escalon_fase >= 2; g = R.Batch
E = pd.DataFrame({"xG": R.z__gn.groupby(g).mean(), "xC": R.z__carbon.groupby(g).mean(),
                  "xG_gate": R.z__gn.where(gate, 0).groupby(g).mean(), "xG_late": R.z__gn.where(late, 0).groupby(g).mean(),
                  "xG_early": R.z__gn.where(~late, 0).groupby(g).mean(),
                  "xG_late_gate": R.z__gn.where(late & gate, 0).groupby(g).mean(), "xC_late": R.z__carbon.where(late, 0).groupby(g).mean(),
                  "xC_early": R.z__carbon.where(~late, 0).groupby(g).mean()})
b = L8.construir_batch(df); b = b.set_index("Batch") if "Batch" in b.columns else b
t = b.join(E).sort_values("idx_cronologico"); dev = t[t.es_dev]; lb = t[~t.es_dev]; t.to_csv(L.SALIDA / "e11_06b_tabla_batch.csv")
specs = {"P0_G+C (sin compuerta)": ["xG", "xC"], "P1_Ggate+C": ["xG_gate", "xC"], "P2_Glate+C": ["xG_late", "xC"], "P3_Glate_gate+C": ["xG_late_gate", "xC"],
         "P4_Glate_gate": ["xG_late_gate"], "P5_Ggate": ["xG_gate"], "P6_Glate_gate+Clate+Cearly": ["xG_late_gate", "xC_late", "xC_early"]}
filas = []
for nm, X in specs.items():
    for muestra, tt, esq in (("DEV", dev, "gkf"), ("DEV", dev, "crono"), ("TOTAL", t, "gkf"), ("TOTAL", t, "crono")):
        r = L.prueba_anidada(tt, X, esquema=esq, n_folds=6 if muestra == "TOTAL" else 5, n_perm=5000); r.pop("score"); r.update({"spec": nm, "muestra": muestra}); filas.append(r)
    # lockbox: beta de DEV aplicado al lockbox (sin reestimar) -> pendiente del KPI sobre el puntaje
    d = dev.dropna(subset=X + L.CONTROLES + [L.KPI]); f = L.ols_hc3(d[L.KPI], d[X + L.CONTROLES]); beta = f.params[X]
    l = lb.dropna(subset=X + [L.KPI]).copy(); l["score"] = l[X].to_numpy() @ beta.to_numpy(); ctr = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]
    fl = L.ols_hc3(l[L.KPI], l[["score"] + ctr]); from scipy import stats
    filas.append({"spec": nm, "muestra": "LOCKBOX(beta DEV)", "esquema": "holdout", "n": len(l), "pendiente": fl.params["score"], "t": fl.tvalues["score"],
                  "p_unilateral": float(stats.norm.sf(fl.tvalues["score"])), "sd_score_pp": l.score.std(), "beta_medio": beta.round(3).to_dict()})
out = pd.DataFrame(filas); out.to_csv(L.SALIDA / "e11_06b_prueba_anidada.csv", index=False)
pd.set_option("display.width", 250)
print(out[["spec", "muestra", "esquema", "n", "pendiente", "t", "p_unilateral", "p_perm_unilateral", "spearman_parcial", "kpi_tercio_bajo", "kpi_tercio_alto", "sd_score_pp", "beta_medio"]].round(3).to_string())
