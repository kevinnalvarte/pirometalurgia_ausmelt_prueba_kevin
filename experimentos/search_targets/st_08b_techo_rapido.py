"""ST-08 (Fable 5.1) -- techo de predictibilidad y fiabilidad del KPI rendimiento_proxy_batch.

1) Fiabilidad: tres medidas del mismo constructo (rendimiento proxy, recuperacion real = metal/Sn cargado, rendimiento HA)
   y el cierre del balance de Sn por batch (Sn cargado vs metal + dross + polvo + Sn en escoria final): si el KPI correlaciona
   con el error de cierre, arrastra ruido contable.
2) Techo empirico: R2 OOF (KFold 5 x 5 repeticiones, y walk-forward cronologico) con TODA la informacion de escalon agregada
   por fase (media/ultimo/suma de ~150 columnas numericas x 2 fases) + contexto + heel, con Ridge / ElasticNet / HGB.
   Si ni asi se supera R2 0.5, ningun target teorico unico puede alcanzar rho/R2 0.7.
3) Cota de correlacion alcanzable: rho_max ~ sqrt(fiabilidad) (correccion por atenuacion).
Salidas: st_08b_fiabilidad.csv, st_08b_techo_modelos.csv, st_08b_log.txt, figs/st_08_*.png
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parent.parent))
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
t0 = time.time()
df = pd.read_pickle(AQUI / "cache_df_st.pkl").sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
batch = pd.read_csv(AQUI / "st_batch.csv", index_col=0).sort_values("idx_cronologico")
orden = list(batch.index)

# ----------------------------------------------------------------------------- 1) fiabilidad
g = df.groupby("Batch")
M = g["sn_en_metal_crudo_batch_t"].first().reindex(orden); D = g["sn_en_dross_fe_batch_t"].first().reindex(orden); P = g["sn_en_polvo_fundicion_batch_t"].first().reindex(orden)
HA = g["rendimiento_ha_batch"].first().reindex(orden)
feed = g["feed_Sn_kgf"].sum().reindex(orden) / 1000
R = df[df.fase_proceso == "Reducción"].dropna(subset=["sn_inventario_escoria_est_kg"]).groupby("Batch").last()
sn_slag_fin = R["sn_inventario_escoria_est_kg"].reindex(orden) / 1000
sn_slag_ini = df[(df.fase_proceso == "Fusión") & (df.orden_escalon_fase == 0)].set_index("Batch")["sn_inventario_escoria_est_kg"].reindex(orden) / 1000
kpi = batch["rendimiento_proxy_batch"]; rec = batch["recuperacion_real_pct"]
cierre = (M + D + P + sn_slag_fin) / feed          # deberia ser ~1 si el balance cierra y no hay heel
salidas = M + D + P
fila = {}
def corr(a, b, nombre):
    d = pd.concat([a, b], axis=1).dropna(); r, p = pearsonr(d.iloc[:, 0], d.iloc[:, 1]); s, ps = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    fila[nombre + "_pearson"] = r; fila[nombre + "_spearman"] = s; fila[nombre + "_n"] = len(d)
    log(f"{nombre:45s} pearson {r:+.3f} spearman {s:+.3f} (n={len(d)})")
log("=== 1) fiabilidad del KPI ===")
corr(kpi, rec, "rendimiento_proxy vs recuperacion_real")
corr(kpi, HA * 100, "rendimiento_proxy vs rendimiento_HA")
corr(rec, HA * 100, "recuperacion_real vs rendimiento_HA")
corr(kpi, cierre, "rendimiento_proxy vs cierre_balance")
corr(rec, cierre, "recuperacion_real vs cierre_balance")
corr(kpi, salidas / feed, "rendimiento_proxy vs (M+D+P)/Sn_cargado")
corr(kpi, feed, "rendimiento_proxy vs Sn_cargado")
corr(kpi, M, "rendimiento_proxy vs M"); corr(kpi, D, "rendimiento_proxy vs D"); corr(kpi, P, "rendimiento_proxy vs P")
corr(M, D, "M vs D"); corr(M, P, "M vs P"); corr(D, P, "D vs P")
corr(M, feed, "M vs Sn_cargado"); corr(D, feed, "D vs Sn_cargado"); corr(P, feed, "P vs Sn_cargado")
log(f"cierre del balance (M+D+P+Sn_escoria_fin)/Sn_cargado: media {cierre.mean():.3f} sd {cierre.std():.3f} P5 {cierre.quantile(.05):.3f} P95 {cierre.quantile(.95):.3f}")
log(f"sd del KPI {kpi.std():.3f}; sd de f_dross {batch.f_dross.std():.4f}; sd de f_polvo {batch.f_polvo.std():.4f}; sd recuperacion real {rec.std():.3f}")
# fiabilidad por acuerdo entre medidas alternativas del mismo constructo: r(proxy, recuperacion real) es una cota
# superior conservadora de la parte del KPI que es "rendimiento" y no ruido de atribucion de D y P.
r_alt = fila["rendimiento_proxy vs recuperacion_real_pearson"]
log(f"cota de correlacion alcanzable con el KPI (atenuacion, fiabilidad ~ r(proxy, recuperacion real) = {r_alt:.3f}): rho_max ~ sqrt({r_alt:.3f}) = {np.sqrt(max(r_alt,0)):.3f}")
# valores repetidos / rachas en D y P (atribucion por stock?)
for nombre, s in [("M", M), ("D", D), ("P", P)]:
    v = s.dropna(); rep = (v.round(3).diff() == 0).sum(); ac1 = pearsonr(v.iloc[1:].values, v.iloc[:-1].values)[0]
    log(f"{nombre}: unicos {v.nunique()}/{len(v)}, valores iguales al anterior {rep}, autocorr lag1 {ac1:+.3f}")
pd.DataFrame([fila]).T.to_csv(AQUI / "st_08b_fiabilidad.csv")

# ----------------------------------------------------------------------------- 2) techo con toda la informacion
log("\n=== 2) techo empirico: R2 OOF con toda la informacion de escalon agregada por fase ===")
excl = {"rendimiento_proxy_batch", "rendimiento_ha_batch", "sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t", "sn_crudo_generado_batch_t", "escalon_idx", "tiempo"}
num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in excl and not c.startswith("st_")]
feats = {}
for fase, tag in [("Fusión", "F"), ("Reducción", "R")]:
    sub = df[df.fase_proceso == fase]
    gg = sub.groupby("Batch")[num]
    feats[f"{tag}_mean"] = gg.mean().add_suffix(f"__{tag}_mean")
    feats[f"{tag}_last"] = gg.last().add_suffix(f"__{tag}_last")
    feats[f"{tag}_first"] = gg.first().add_suffix(f"__{tag}_first")
    feats[f"{tag}_sum"] = gg.sum(min_count=1).add_suffix(f"__{tag}_sum")
X = pd.concat(feats.values(), axis=1).reindex(orden)
X = X.loc[:, X.notna().mean() > 0.8]
X = X.loc[:, X.std() > 1e-9]
# heel: estado final de Reduccion del batch anterior
prev = {orden[i]: orden[i - 1] for i in range(1, len(orden))}
Rlast = X.filter(like="__R_last")
heel = Rlast.rename(columns=lambda c: c.replace("__R_last", "__heel_prev")).reindex([prev.get(b) for b in orden]); heel.index = orden
X = pd.concat([X, heel], axis=1)
X["idx_cronologico"] = batch["idx_cronologico"]
log(f"matriz X: {X.shape[1]} features x {X.shape[0]} batches")
dev_mask = ~batch.es_lockbox.values
ys = {"rendimiento": kpi, "recuperacion_real": rec, "f_dross": batch.f_dross, "f_polvo": batch.f_polvo,
      "rendimiento_2batches": (kpi + pd.Series({b: kpi.get(orden[i + 1], np.nan) for i, b in enumerate(orden[:-1])}).reindex(orden)) / 2}
def modelos():
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
        "ElasticNet": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=np.logspace(-2, 1, 25), cv=5, max_iter=5000)),
        "HGB": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=10, l2_regularization=1.0, random_state=42)),
    }
filas = []
Xd = X[dev_mask]; Xl = X[~dev_mask]
for yname, y in ys.items():
    yd = y[dev_mask]; yl = y[~dev_mask]
    ok = yd.notna().values
    for mname in ["Ridge", "HGB"]:
        r2s, rhos = [], []
        preds = np.full(ok.sum(), np.nan)
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        Xdo, ydo = Xd[ok], yd[ok]
        oof_all = np.zeros((2, len(ydo)))
        for k, (tr, te) in enumerate(rkf.split(Xdo)):
            m = modelos()[mname].fit(Xdo.iloc[tr], ydo.iloc[tr]); oof_all[k // 5, te] = m.predict(Xdo.iloc[te])
        for rep in range(2):
            r2s.append(r2_score(ydo, oof_all[rep])); rhos.append(spearmanr(ydo, oof_all[rep])[0])
        m = modelos()[mname].fit(Xdo, ydo); okl = yl.notna().values
        pl = m.predict(Xl[okl]); r2l = r2_score(yl[okl], pl); rhol = spearmanr(yl[okl], pl)[0]
        # walk-forward cronologico: 4 bloques
        idx = np.arange(len(ydo)); wf = []
        for cut in [0.5, 0.65, 0.8]:
            n = int(cut * len(idx)); mm = modelos()[mname].fit(Xdo.iloc[:n], ydo.iloc[:n]); wf.append(r2_score(ydo.iloc[n:], mm.predict(Xdo.iloc[n:])))
        filas.append(dict(kpi=yname, modelo=mname, n_dev=int(ok.sum()), r2_oof=np.mean(r2s), r2_oof_min=np.min(r2s), r2_oof_max=np.max(r2s), rho_oof=np.mean(rhos),
                          r2_lockbox=r2l, rho_lockbox=rhol, r2_walkforward=np.mean(wf)))
        log(f"{yname:22s} {mname:10s} R2oof {np.mean(r2s):+.3f} [{np.min(r2s):+.3f},{np.max(r2s):+.3f}] rho_oof {np.mean(rhos):+.3f} | lockbox R2 {r2l:+.3f} rho {rhol:+.3f} | walk-forward R2 {np.mean(wf):+.3f}")
res = pd.DataFrame(filas); res.to_csv(AQUI / "st_08b_techo_modelos.csv", index=False)
# importancia por permutacion del HGB para rendimiento (top 15)
from sklearn.inspection import permutation_importance
y = kpi[dev_mask]; ok = y.notna().values
m = modelos()["HGB"].fit(Xd[ok], y[ok])
pi = permutation_importance(m, Xd[ok], y[ok], n_repeats=5, random_state=0, scoring="r2")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
log("\nTop-15 permutation importance (HGB, rendimiento, DEV in-sample):"); log(imp.head(15).round(4).to_string())
imp.head(40).to_csv(AQUI / "st_08b_importancia_hgb.csv")
fig, ax = plt.subplots(figsize=(8, 4))
for mname in ["Ridge", "HGB"]:
    r = res[res.modelo == mname]; ax.plot(r.kpi, r.r2_oof, "o-", label=f"{mname} R2 OOF")
ax.axhline(0.7, color="r", ls="--", label="umbral 0.70"); ax.set_ylabel("R2 OOF (DEV)"); ax.legend(); ax.set_title("Techo empirico con toda la informacion de escalon")
plt.xticks(rotation=15); plt.tight_layout(); fig.savefig(AQUI / "figs" / "st_08b_techo.png", dpi=120)
log(f"tiempo {time.time()-t0:.0f}s")
(AQUI / "st_08b_log.txt").write_text("\n".join(LOG), encoding="utf-8")
