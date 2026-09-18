"""E18-07 (Fable): regla de disciplina 'GN nunca por encima de la receta' — variantes por tramo (todos los escalones / solo R2-R3 / solo R0-R1), en unidades
fisicas (Nm3/min de exceso medio). OLS HC3 por regimen, prueba prospectiva W=100 (permutacion por bloques y circular), ganancia esperada con bootstrap por bloques."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import asesor_receta_v10 as A, v9_lib as L, e11_07_lib as W7
r, bt = A.escalones_con_receta(); r["exc"] = r.dev__gn.clip(lower=0); r["defc"] = r.dev__gn.clip(upper=0); late = r.orden_escalon_fase >= 2
E = pd.DataFrame({"exc_todo": r.exc.groupby(r.Batch).mean(), "exc_tard": r.exc[late].groupby(r.Batch[late]).mean(), "exc_temp": r.exc[~late].groupby(r.Batch[~late]).mean(),
                  "def_todo": r.defc.groupby(r.Batch).mean(), "dC": r.dev__carbon.groupby(r.Batch).mean()})
t = bt.join(E).sort_values("idx_cronologico"); C = W7.C_SIN_ESPESOR; t["dross_pp"] = 100 * t.f_dross; t.to_csv("experimentos/v16/e18_07_tabla.csv")
print("exceso medio Nm3/min:", E[["exc_todo", "exc_tard", "exc_temp"]].mean().round(2).to_dict(), "| % escalones con exceso:", round((r.dev__gn > 0).mean() * 100), "| tardios:", round((r.dev__gn[late] > 0).mean() * 100))
for Xs in (["exc_todo", "def_todo", "dC"], ["exc_tard", "exc_temp", "def_todo", "dC"]):
    for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
        for y in (L.KPI, "dross_pp"):
            dd = tt.dropna(subset=Xs + C + [y]); f = L.ols_hc3(dd[y], dd[Xs + C]); print(nm, y[:8], len(dd), " ".join(f"{c}:{f.params[c]:+.2f}(t{f.tvalues[c]:+.1f})" for c in Xs))
rng = np.random.default_rng(0)
def wf(Xe, Cc, yy, W=100, paso=10):
    n = len(yy); sc = np.full(n, np.nan); s = W
    while s < n:
        q = W7.ols_classic(np.column_stack([Cc[s-W:s], Xe[s-W:s]]), yy[s-W:s]); k = Xe.shape[1]; sc[s:s+paso] = Xe[s:s+paso] @ q["beta"][-k:]; s += paso
    return sc
for xs in (["exc_todo"], ["exc_tard"], ["exc_temp"], ["exc_tard", "exc_temp"]):
    P = W7.preparar(t, xs); sc = wf(P["Xexp"], P["Xctrl"], P["y"]); o = W7.stats_final(sc, P["y"], P["Xctrl"]); blo = W7.bloques_cronologicos(P["n"], 20)
    nul = np.array([W7.stats_final(wf(W7.permutar_bloques(P["Xexp"], blo, rng), P["Xctrl"], P["y"]), P["y"], P["Xctrl"])["t"] for _ in range(1000)])
    circ = np.array([W7.stats_final(wf(np.roll(P["Xexp"], k, axis=0), P["Xctrl"], P["y"]), P["y"], P["Xctrl"])["t"] for k in range(15, P["n"] - 15)])
    dv = W7.stats_final(sc, P["y"], P["Xctrl"], P["es_dev"]); lb = W7.stats_final(sc, P["y"], P["Xctrl"], ~P["es_dev"])
    print(xs, "prospectivo: pend %.2f t %.2f p_perm %.4f p_circ %.4f spearman %.3f | DEV t %.2f | LB pend %.2f t %.2f" % (o["pendiente"], o["t"], (np.sum(nul >= o["t"]) + 1) / 1001, (np.sum(circ >= o["t"]) + 1) / (len(circ) + 1), o["spearman"], dv["t"], lb["pendiente"], lb["t"]))
# ganancia esperada de la regla = -beta * E[exceso], bootstrap por bloques de 10 (beta y exceso del mismo remuestreo)
import statsmodels.api as sm
for nm, tt in (("TOTAL", t), ("DEV", t[t.es_dev]), ("LB", t[~t.es_dev])):
    for xs in (["exc_todo"], ["exc_tard", "exc_temp"]):
        d = tt.dropna(subset=xs + ["def_todo", "dC", L.KPI] + C).reset_index(drop=True); nb = len(d) // 10; G = []
        for _ in range(2000):
            idx = np.concatenate([np.arange(b * 10, b * 10 + 10) for b in rng.integers(0, nb, nb)]); dd = d.iloc[idx]; f = sm.OLS(dd[L.KPI], sm.add_constant(dd[xs + ["def_todo", "dC"] + C])).fit()
            G.append([-f.params[x] * dd[x].mean() for x in xs])
        G = np.array(G); tot = G.sum(1) if len(xs) > 1 else G[:, 0]
        print(nm, xs, "ganancia de la regla %.2f pp/batch IC95 [%.2f, %.2f] P(>0) %.3f" % (np.median(tot), *np.percentile(tot, [2.5, 97.5]), (tot > 0).mean()), "| solo tardio: %.2f [%.2f, %.2f]" % (np.median(G[:, 0]), *np.percentile(G[:, 0], [2.5, 97.5])) if len(xs) > 1 else "")
