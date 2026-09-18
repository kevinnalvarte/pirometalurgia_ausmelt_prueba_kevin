"""E12-08 (Fable): objetivo FISICO sin pesos ajustados: puntaje_b = D_Sn_kg - lambda*D_FeO_kg (kg de Sn-eq que la accion del operador gano
respecto de la practica habitual segun los theta de escalon cross-fitted de E11-03), lambda fijado por la composicion del dross
(hardhead ~39 % Sn -> ~0.6 kg Sn por kg FeO reducido). Ningun parametro se ajusta al KPI -> no hay adaptatividad ni multiplicidad de ventana."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import v9_lib as L
t = pd.read_csv("experimentos/v9/e11_03_tabla_batch.csv", index_col=0).sort_values("idx_cronologico")
C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]
print(t[["D_sn_kg_ident", "D_feo_kg_ident", "D_sn_kg_all", "D_feo_kg_all"]].describe().round(0).T[["mean", "std", "min", "max"]].to_string())
filas = []
for var in ("ident", "all"):
    for lam in (0.0, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, "soloFeO"):
        s = -t[f"D_feo_kg_{var}"] if lam == "soloFeO" else t[f"D_sn_kg_{var}"] - lam * t[f"D_feo_kg_{var}"]
        t["score"] = s / 512.0     # pp de KPI si 1 kg Sn-eq = 1 kg Sn a metal
        lo, hi = t.score.quantile([.01, .99]); t["score_w"] = t.score.clip(lo, hi)
        for nm, d in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
            d = d.dropna(subset=C + ["score_w", L.KPI]); f = L.ols_hc3(d[L.KPI], d[["score_w"] + C])
            ry = L.ols_hc3(d[L.KPI], d[C]).resid; rs = L.ols_hc3(d["score_w"], d[C]).resid; rho, p = stats.spearmanr(ry, rs)
            filas.append({"theta": var, "lambda": lam, "muestra": nm, "n": len(d), "sd_score_pp": d.score_w.std(), "pendiente": f.params["score_w"], "t": f.tvalues["score_w"],
                          "p_uni": stats.norm.sf(f.tvalues["score_w"]), "spearman": rho, "p_spearman": p})
out = pd.DataFrame(filas); out.to_csv("experimentos/v10/e12_08_objetivo_fisico.csv", index=False)
pd.set_option("display.width", 220); print(out.round(3).to_string())
