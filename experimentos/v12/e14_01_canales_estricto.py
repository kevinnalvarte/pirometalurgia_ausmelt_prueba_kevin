"""E14-01 (Fable): evidencia prospectiva ESTRICTA por canal de mecanismo.
Exposiciones estrictas y ejecutables: z = (a - E[a|S_ejecutable]) / sd, con E[a|S] entrenado SOLO con batches anteriores (bloques de 10, desde el
batch 40) + correccion de deriva de receta (residuo medio por orden de los 20 batches previos). Luego walk-forward W=100, paso 10, contraccion JS:
  GN  -> f_dross (pp)                    (mecanismo: fuego de lanza -> Fe metalico -> hardhead)
  C   -> Sn perdido en escoria final (pp) (mecanismo: C x Sn disponible -> agotamiento; unica perdida irreversible)
  y, como referencia, {GN, C} -> KPI refinado.
Especificacion fijada ANTES de ver resultados: 2 contrastes de canal (Bonferroni x2) + referencia."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import asesor_planta_v9_1 as AP, modelo_predictivo_v9 as mp9, v8_lib as L8, e11_07_lib as W7
df = L8.cargar_df(); base = AP.preparar_base_planta(df); S = AP.ESTADO_EJECUTABLE
bt = L8.construir_batch(df); bt = bt.set_index("Batch") if "Batch" in bt.columns else bt; bt = bt.sort_values("idx_cronologico"); orden = list(bt.index)
d = mp9.filas_reduccion(base); pos = d["Batch"].map({b: i for i, b in enumerate(orden)}).to_numpy(); o = d["orden_escalon_fase"].to_numpy()
Z = {}
for p, col in mp9.PALANCAS_V9.items():
    z = np.full(len(d), np.nan); ok = d[col].notna().to_numpy()
    for s in range(40, len(orden), 10):
        tr = np.where((pos < s) & ok)[0]; te = np.where((pos >= s) & (pos < s + 10) & ok)[0]
        if not len(te): continue
        m = L8.hgb_nuisance(42).fit(d[S].iloc[tr], d[col].iloc[tr]); r_te = d[col].to_numpy()[te] - m.predict(d[S].iloc[te])
        # sd y sesgo reciente con residuos ESTRICTOS ya calculados del pasado (si no hay aun, in-sample del pasado)
        for oo in np.unique(o[te]):
            prev = np.where((pos < s) & (pos >= s - 20) & (o == oo) & np.isfinite(z))[0]   # z previos ya estandarizados
            hist = np.where((pos < s) & (o == oo) & ok)[0]; sd = np.nanstd(d[col].to_numpy()[hist] - m.predict(d[S].iloc[hist])) * 1.15 + 1e-9
            mm = o[te] == oo; zz = r_te[mm] / sd; z[te[mm]] = zz - (np.nanmean(z[prev]) if len(prev) >= 10 else 0.0)
    Z[p] = z
E = pd.DataFrame({"xG": pd.Series(Z["gn"]).groupby(d["Batch"].to_numpy()).mean(), "xC": pd.Series(Z["carbon"]).groupby(d["Batch"].to_numpy()).mean()})
t = bt.join(E).dropna(subset=["xG", "xC"]); t["dross_pp"] = 100 * t.f_dross; t["escoria_pp"] = 100 * t.sn_perdido_escoria_frac; t["polvo_pp"] = 100 * t.f_polvo
t.to_csv("experimentos/v12/e14_01_tabla_estricta.csv"); print("batches con exposicion estricta:", len(t), "| sd xG %.2f xC %.2f" % (t.xG.std(), t.xC.std()))
def wf(Xe, Cc, yy, W=100, paso=10):
    n = len(yy); sc = np.full(n, np.nan); s = W
    while s < n:
        r = W7.ols_classic(np.column_stack([Cc[s-W:s], Xe[s-W:s]]), yy[s-W:s]); k = Xe.shape[1]; b, se = r["beta"][-k:], r["se"][-k:]
        b = b * np.clip(1 - (se / np.where(b == 0, 1e-9, b)) ** 2, 0, None); sc[s:s+paso] = Xe[s:s+paso] @ b; s += paso
    return sc
rng = np.random.default_rng(0); filas = []
for nombre, xs, y in (("GN -> dross", ["xG"], "dross_pp"), ("C -> Sn en escoria", ["xC"], "escoria_pp"), ("GN+C -> KPI", ["xG", "xC"], W7.KPI),
                      ("GN+C -> dross", ["xG", "xC"], "dross_pp"), ("GN+C -> escoria", ["xG", "xC"], "escoria_pp"), ("GN -> polvo (control negativo de estabilidad)", ["xG"], "polvo_pp")):
    P = W7.preparar(t, xs, y_col=y); X, Cc, yy, n = P["Xexp"], P["Xctrl"], P["y"], P["n"]; blo = W7.bloques_cronologicos(n, 20)
    sc = wf(X, Cc, yy); obs = W7.stats_final(sc, yy, Cc); nul = np.array([W7.stats_final(wf(W7.permutar_bloques(X, blo, rng), Cc, yy), yy, Cc)["t"] for _ in range(2000)])
    circ = np.array([W7.stats_final(wf(np.roll(X, k, axis=0), Cc, yy), yy, Cc)["t"] for k in range(15, n - 15)])
    for nm, mk in (("todo", None), ("DEV", P["es_dev"]), ("LB", ~P["es_dev"])):
        st = W7.stats_final(sc, yy, Cc, mk); st.update({"contraste": nombre, "muestra": nm})
        if nm == "todo": st.update({"p_perm_bloques": (np.sum(nul >= obs["t"]) + 1) / 2001, "p_circular": (np.sum(circ >= obs["t"]) + 1) / (len(circ) + 1)})
        filas.append(st)
out = pd.DataFrame(filas); out.to_csv("experimentos/v12/e14_01_canales_estricto.csv", index=False); pd.set_option("display.width", 220)
print(out[["contraste", "muestra", "n", "pendiente", "se", "t", "p_uni", "p_perm_bloques", "p_circular", "spearman"]].round(4).to_string())
