"""E11-07 tarea 4: walk-forward W=100 por canal (f_dross, f_polvo, f_metal, sn_perdido_escoria_frac, todos x100 pp) y
KPI compuesto reconstruido = 100 - (score_dross + score_polvo + 100*score_escoria) [deviacion de recuperacion
atribuible a la exposicion, construida bottom-up desde los canales de perdida]."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as E

t = E.cargar_tabla()
CANALES = ["f_dross", "f_polvo", "f_metal", "sn_perdido_escoria_frac"]
tt = t.copy()
for c in CANALES:
    tt[c + "_pp"] = tt[c] * 100.0
CANALES_PP = [c + "_pp" for c in CANALES]

filas = []
scores = {}
for c in CANALES_PP:
    D = E.preparar(tt, E.X_PRINCIPAL, y_col=c)
    Xexp, Xctrl, y, n = D["Xexp"], D["Xctrl"], D["y"], D["n"]
    score = E.walk_forward_score(Xexp, Xctrl, y, 100, paso=E.PASO, tmin=0.0)
    r = E.stats_final(score, y, Xctrl, np.ones(n, bool))
    r["canal"] = c
    filas.append(r)
    scores[c] = (D["idx"], score)

# KPI directo, para comparar
D0 = E.preparar(tt, E.X_PRINCIPAL, y_col="recuperacion_refinada_pct")
score0 = E.walk_forward_score(D0["Xexp"], D0["Xctrl"], D0["y"], 100, paso=E.PASO, tmin=0.0)
r0 = E.stats_final(score0, D0["y"], D0["Xctrl"], np.ones(D0["n"], bool)); r0["canal"] = "recuperacion_refinada_pct (directo)"
filas.append(r0)

out = pd.DataFrame(filas)[["canal", "n", "pendiente", "se", "t", "p_uni", "spearman", "sd_score"]]
out.to_csv(E.SALIDA / "e11_07_canales.csv", index=False)
pd.set_option("display.width", 200); print(out.round(4).to_string())

# --- KPI compuesto reconstruido: recovery_score = -(score_dross + score_polvo + score_sn_perdido), signo: mas perdida
# predicha -> menos recuperacion. Se alinean por Batch (idx original de e11_06b), solo filas con score en TODOS los canales.
idxs = [pd.Index(scores[c][0]) for c in CANALES_PP]
comunes = idxs[0]
for i in idxs[1:]:
    comunes = comunes.intersection(i)
comunes = tt.index.intersection(comunes)
sc = pd.DataFrame(index=comunes)
for c in CANALES_PP:
    idx, sco = scores[c]
    sc[c] = pd.Series(sco, index=idx).reindex(comunes)
sc = sc.dropna()
score_compuesto = -(sc["f_dross_pp"] + sc["f_polvo_pp"] + sc["sn_perdido_escoria_frac_pp"])
y_kpi = tt.loc[sc.index, "recuperacion_refinada_pct"]
Xctrl_c = tt.loc[sc.index, E.C_SIN_ESPESOR].to_numpy(float)
mask = ~score_compuesto.isna().to_numpy() & ~y_kpi.isna().to_numpy() & np.isfinite(Xctrl_c).all(1)
rC = E.stats_final(score_compuesto.to_numpy(), y_kpi.to_numpy(), Xctrl_c, mask)
rC["canal"] = "compuesto: 100 - Sum(perdidas predichas dross+polvo+escoria)"
print("\nKPI compuesto reconstruido (score = -(score_dross+score_polvo+score_escoria)):")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rC.items()})
pd.DataFrame([rC]).to_csv(E.SALIDA / "e11_07_canal_compuesto.csv", index=False)
