"""E15-01 (Fable): la hoja 'Alimentación' trae la RECETA por escalon de Reduccion (GN, O2, aire, carbon; R01-R04). Exposiciones exactas y ejecutables:
  dev = ejecutado - receta (por escalon, estandarizado por la sd del desvio del orden en DEV), media por batch -> dG, dC (y dO2, dAire)
  plan = receta estandarizada por orden (decision del planificador), media por batch -> pG, pC
Evaluacion: OLS HC3 conjunta (DEV/LB/total, canales) y prueba prospectiva W=100, paso 10, JS, permutacion por bloques y circular."""
import sys; sys.path.insert(0, "experimentos/v9"); sys.path.insert(0, "experimentos/v8")
import numpy as np, pandas as pd
import v8_lib as L8, v9_lib as L, e11_07_lib as W7
X = "Datos Lingo smelter fase II.xlsx"
a = pd.read_excel(X, sheet_name="Alimentación", header=1); a["Batch"] = a.Batch.astype(str); a = a[a.Batch.str.startswith("AP")].set_index("Batch")
df = L8.cargar_df(); r = df[(df.fase_proceso == "Reducción") & (df.orden_escalon_fase <= 3)].copy()
mapa = {"gn": ("GN (Nm3/h)-R0{}", "tasa_gn_nm3_min"), "carbon": ("Carbón (kg/h)-R0{}", "tasa_feed_Carbon_kg_min"), "o2": ("O2 (Nm3/h)-R0{}", "tasa_o2_nm3_min"), "aire": ("Aire (Nm3/h)-R0{}", "tasa_aire_nm3_min")}
for p, (pat, col) in mapa.items():
    r[f"plan__{p}"] = [a[pat.format(int(o) + 1)].get(b, np.nan) / 60.0 for b, o in zip(r.Batch, r.orden_escalon_fase)]
    r[f"dev__{p}"] = r[col] - r[f"plan__{p}"]
dev_b, _ = L8.mp.split_dev_lockbox(df); esdev = r.Batch.isin(dev_b)
print(r.groupby("orden_escalon_fase")[["plan__gn", "tasa_gn_nm3_min", "dev__gn", "plan__carbon", "tasa_feed_Carbon_kg_min", "dev__carbon"]].agg(["mean", "std"]).round(1).to_string())
E = {}
for p in mapa:
    for k in ("dev", "plan"):
        mu = r.loc[esdev].groupby("orden_escalon_fase")[f"{k}__{p}"].mean(); sd = r.loc[esdev].groupby("orden_escalon_fase")[f"{k}__{p}"].std()
        z = (r[f"{k}__{p}"] - r.orden_escalon_fase.map(mu)) / r.orden_escalon_fase.map(sd); E[f"{k[0]}_{p}"] = z.groupby(r.Batch).mean()
E = pd.DataFrame(E); r.to_pickle("experimentos/v13/e15_01_escalones_receta.pkl")
t = pd.read_csv("experimentos/v9/e11_06b_tabla_batch.csv", index_col=0).join(E).sort_values("idx_cronologico"); t.to_csv("experimentos/v13/e15_01_tabla_batch.csv")
print("corr con exposiciones v9:", t[["xG", "xC", "d_gn", "d_carbon", "p_gn", "p_carbon"]].corr().round(2).loc[["xG", "xC"]].to_string())
C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]
for Xs in (["d_gn", "d_carbon"], ["d_gn", "d_carbon", "p_gn", "p_carbon"], ["d_gn", "d_carbon", "d_o2", "d_aire"]):
    for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
        for y in (L.KPI, "f_dross", "sn_perdido_escoria_frac", "f_polvo"):
            dd = tt.dropna(subset=Xs + C + [y]); f = L.ols_hc3(dd[y] * (1 if y == L.KPI else 100), dd[Xs + C])
            print(nm, y[:9], len(dd), " ".join(f"{c}:{f.params[c]:+.2f}(t{f.tvalues[c]:+.1f})" for c in Xs))
def wf(Xe, Cc, yy, W=100, paso=10, js=True):
    n = len(yy); sc = np.full(n, np.nan); s = W
    while s < n:
        q = W7.ols_classic(np.column_stack([Cc[s-W:s], Xe[s-W:s]]), yy[s-W:s]); k = Xe.shape[1]; b, se = q["beta"][-k:], q["se"][-k:]
        if js: b = b * np.clip(1 - (se / np.where(b == 0, 1e-9, b)) ** 2, 0, None)
        sc[s:s+paso] = Xe[s:s+paso] @ b; s += paso
    return sc
rng = np.random.default_rng(0); filas = []
for nombre, xs in (("v9 cross-fit xG,xC (ref)", ["xG", "xC"]), ("desvio vs receta dG,dC", ["d_gn", "d_carbon"]), ("receta pG,pC", ["p_gn", "p_carbon"]), ("desvio+receta", ["d_gn", "d_carbon", "p_gn", "p_carbon"]),
                   ("ejecutado total (d+p) GN,C", None)):
    tt = t.copy()
    if xs is None:
        tt["a_gn"] = tt.d_gn + tt.p_gn; tt["a_c"] = tt.d_carbon + tt.p_carbon; xs = ["a_gn", "a_c"]
    P = W7.preparar(tt, xs); Xe, Cc, yy, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]; blo = W7.bloques_cronologicos(n, 20)
    for js in (True, False):
        sc = wf(Xe, Cc, yy, js=js); obs = W7.stats_final(sc, yy, Cc); nul = np.array([W7.stats_final(wf(W7.permutar_bloques(Xe, blo, rng), Cc, yy, js=js), yy, Cc)["t"] for _ in range(1000)])
        circ = np.array([W7.stats_final(wf(np.roll(Xe, k, axis=0), Cc, yy, js=js), yy, Cc)["t"] for k in range(15, n - 15)])
        for nm, mk in (("todo", None), ("DEV", P["es_dev"]), ("LB", ~P["es_dev"])):
            st = W7.stats_final(sc, yy, Cc, mk); st.update({"exposicion": nombre, "js": js, "muestra": nm})
            if nm == "todo": st.update({"p_perm": (np.sum(nul >= obs["t"]) + 1) / 1001, "p_circ": (np.sum(circ >= obs["t"]) + 1) / (len(circ) + 1)})
            filas.append(st)
out = pd.DataFrame(filas); out.to_csv("experimentos/v13/e15_01_prospectivo.csv", index=False); pd.set_option("display.width", 220)
print(out[["exposicion", "js", "muestra", "n", "pendiente", "t", "p_perm", "p_circ", "spearman"]].round(4).to_string())
