"""E16-01 (Fable): desvio respecto de la RECETA en FUSION (hoja Alimentacion: kg/h por tolva y escalon F01-F07; Tv02 = carbon). Exposiciones exactas:
dF_C = carbon ejecutado - receta (kg/min) media F1-F6; dF_feed = carga total ejecutada - receta. Conjunta con d_gn, d_carbon de Reduccion + prospectiva."""
import sys; sys.path.insert(0, "experimentos/v9"); sys.path.insert(0, "experimentos/v8")
import numpy as np, pandas as pd
import v8_lib as L8, v9_lib as L, e11_07_lib as W7
a = pd.read_excel("Datos Lingo smelter fase II.xlsx", sheet_name="Alimentación", header=1); a["Batch"] = a.Batch.astype(str); a = a[a.Batch.str.startswith("AP")].set_index("Batch")
df = L8.cargar_df(); f = df[(df.fase_proceso == "Fusión")].copy(); print("ordenes F:", sorted(f.orden_escalon_fase.unique()))
tolvas = ["Tv01", "Tv03", "Tv04", "Tv05", "Tv06", "Tv07", "Tv08"]
def plan(b, o, stem):
    c = f"{stem}-F0{int(o)+1}"; return a[c].get(b, np.nan) / 60.0 if c in a.columns else np.nan
f["plan_C"] = [plan(b, o, "Tv02") for b, o in zip(f.Batch, f.orden_escalon_fase)]
f["plan_feed"] = [np.nansum([plan(b, o, s) for s in tolvas]) for b, o in zip(f.Batch, f.orden_escalon_fase)]
f["dev_C"] = f.tasa_feed_Carbon_kg_min - f.plan_C; f["dev_feed"] = f.tasa_feed_total_kg_min - f.plan_feed
print(f.groupby("orden_escalon_fase")[["plan_C", "tasa_feed_Carbon_kg_min", "dev_C", "plan_feed", "tasa_feed_total_kg_min", "dev_feed"]].agg(["mean", "std"]).round(1).to_string())
print("corr ejecutado-receta: C %.2f feed %.2f" % (f.tasa_feed_Carbon_kg_min.corr(f.plan_C), f.tasa_feed_total_kg_min.corr(f.plan_feed)))
dev_b, _ = L8.mp.split_dev_lockbox(df); m = f.Batch.isin(dev_b); E = {}
for k in ("dev_C", "dev_feed"):
    sd = f[m].groupby("orden_escalon_fase")[k].std(); mu = f[m].groupby("orden_escalon_fase")[k].mean()
    z = (f[k] - f.orden_escalon_fase.map(mu)) / f.orden_escalon_fase.map(sd); E["F_" + k] = z[f.orden_escalon_fase >= 1].groupby(f.Batch).mean()
t = pd.read_csv("experimentos/v13/e15_01_tabla_batch.csv", index_col=0).join(pd.DataFrame(E)).sort_values("idx_cronologico"); t.to_csv("experimentos/v14/e16_01_tabla.csv")
C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]; X = ["d_gn", "d_carbon", "F_dev_C", "F_dev_feed"]
for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
    for y in (L.KPI, "f_dross", "f_polvo", "sn_perdido_escoria_frac"):
        dd = tt.dropna(subset=X + C + [y]); q = L.ols_hc3(dd[y] * (1 if y == L.KPI else 100), dd[X + C]); print(nm, y[:9], len(dd), " ".join(f"{c}:{q.params[c]:+.2f}(t{q.tvalues[c]:+.1f})" for c in X))
def wf(Xe, Cc, yy, W=100, paso=10):
    n = len(yy); sc = np.full(n, np.nan); s = W
    while s < n:
        r = W7.ols_classic(np.column_stack([Cc[s-W:s], Xe[s-W:s]]), yy[s-W:s]); k = Xe.shape[1]; sc[s:s+paso] = Xe[s:s+paso] @ r["beta"][-k:]; s += paso
    return sc
rng = np.random.default_rng(0)
for xs in (["d_gn", "d_carbon"], ["d_gn", "d_carbon", "F_dev_C"], ["d_gn", "d_carbon", "F_dev_feed"], ["d_gn", "d_carbon", "F_dev_C", "F_dev_feed"], ["F_dev_C", "F_dev_feed"]):
    P = W7.preparar(t.dropna(subset=X), xs); sc = wf(P["Xexp"], P["Xctrl"], P["y"]); o = W7.stats_final(sc, P["y"], P["Xctrl"]); blo = W7.bloques_cronologicos(P["n"], 20)
    nul = [W7.stats_final(wf(W7.permutar_bloques(P["Xexp"], blo, rng), P["Xctrl"], P["y"]), P["y"], P["Xctrl"])["t"] for _ in range(500)]
    lb = W7.stats_final(sc, P["y"], P["Xctrl"], ~P["es_dev"])
    print(xs, "n", o["n"], "pend %.2f t %.2f p_perm %.3f spearman %.3f | LB t %.2f" % (o["pendiente"], o["t"], (np.sum(np.array(nul) >= o["t"]) + 1) / 501, o["spearman"], lb["t"]))
