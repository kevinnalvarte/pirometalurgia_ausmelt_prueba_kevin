"""E7-09: por que el R2 lockbox del agotamiento de Sn en Fusion es negativo y si el modelo es espurio."""
import sys, warnings; from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ))
import modelo_predictivo_v5 as mp5, modelo_prescriptivo as mp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
pd.set_option("display.width", 220)
p = pd.read_csv(RAIZ / "experimentos/v5/e7_05_pred_Fusión_sn_dep.csv")
for conj, q in p.groupby("conjunto"):
    r = pearsonr(q.real, q.pred)[0]; rho = spearmanr(q.real, q.pred)[0]
    sesgo = (q.pred - q.real).mean(); r2c = r2_score(q.real, q.pred - sesgo)
    print(f"{conj:8s} n={len(q):5d} real media={q.real.mean():.3f} sd={q.real.std():.3f} | pred media={q.pred.mean():.3f} sd={q.pred.std():.3f} | "
          f"sesgo={sesgo:+.3f} | R2={r2_score(q.real, q.pred):.3f} R2 sin sesgo={r2c:.3f} | pearson={r:.3f} spearman={rho:.3f}")
lb = p[p.conjunto == "lockbox"]
print("\nlockbox por orden de escalon: sesgo y correlacion")
print(lb.groupby("orden_escalon_fase").apply(lambda g: pd.Series({"n": len(g), "real": g.real.mean(), "pred": g.pred.mean(), "sesgo": (g.pred - g.real).mean(), "pearson": pearsonr(g.real, g.pred)[0] if len(g) > 5 else np.nan})).round(3).to_string())
# el mismo target: DEV por tercios (nivel medio) para ver la deriva de campana
df = mp5.agregar_columnas_v5(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
d = mp5._preparar_base(df); dF = d[d.fase_proceso == "Fusión"].copy()
orden = mp.orden_cronologico_batches(df); pos = {b: i for i, b in enumerate(orden)}
dF["idx"] = dF.Batch.map(pos); dF["bloque"] = pd.cut(dF.idx, [-1, 100, 200, 299, 362], labels=["T1", "T2", "T3", "lockbox"])
print("\nnivel del target por bloque cronologico (Fusion, ln_sn_dep) y de sus drivers:")
print(dF.groupby("bloque", observed=True)[["v5_ln_sn_dep", "feed_Sn_kgf", "sn_inventario_escoria_est_kg_prev", "tasa_gn_nm3_min", "temperatura_horno_celsius_prev"]].agg(["mean", "std"]).round(3).to_string())
# theta_GN ajustando SOLO con el lockbox y con reentrenamiento progresivo
dev, lockbox = mp.split_dev_lockbox(df)
S, A, tgt = mp5.ESTADO_V5[("Fusión", "sn_dep")], mp5.PALANCAS_V5[("Fusión", "sn_dep")], "v5_ln_sn_dep"
need = list(dict.fromkeys(S + A + [tgt, "Batch"]))
d_lb = dF.loc[dF.Batch.isin(lockbox), need].dropna().reset_index(drop=True)
m_lb = mp5.ModeloPLMSignos(S, A, tgt, mp5.SIGNO_TEORICO_V5[("Fusión", "sn_dep")]).fit(d_lb, n_splits=5, n_boot=150)
print("\ntheta ajustado SOLO en lockbox (63 batches):"); print(m_lb.tabla_theta()[["theta_1sd", "ci_lo_1sd", "ci_hi_1sd", "significativo"]].round(4).to_string())
# reentrenamiento progresivo cada 10 batches sobre el lockbox
lb_ord = [b for b in orden if b in lockbox]; preds = []
d_all = dF[need].dropna().reset_index(drop=True)
for i in range(0, len(lb_ord), 10):
    vistos = set(orden[:len(orden) - len(lb_ord) + i]); test = lb_ord[i:i + 10]
    m = mp5.ModeloPLMSignos(S, A, tgt, mp5.SIGNO_TEORICO_V5[("Fusión", "sn_dep")]).fit(d_all[d_all.Batch.isin(vistos)].reset_index(drop=True), n_boot=0)
    q = d_all[d_all.Batch.isin(test)]; preds.append(pd.DataFrame({"real": q[tgt].to_numpy(), "pred": m.predict(q)}))
pr = pd.concat(preds); sesgo = (pr.pred - pr.real).mean()
print(f"\nreentrenamiento progresivo cada 10 batches: R2 lockbox={r2_score(pr.real, pr.pred):.3f}  sin sesgo={r2_score(pr.real, pr.pred - sesgo):.3f}  pearson={pearsonr(pr.real, pr.pred)[0]:.3f}  sesgo={sesgo:+.3f}")
print("theta GN del ultimo modelo progresivo:", m.tabla_theta().loc["tasa_gn_nm3_min", ["theta_1sd", "ci_lo_1sd", "ci_hi_1sd"]].round(4).to_dict())
