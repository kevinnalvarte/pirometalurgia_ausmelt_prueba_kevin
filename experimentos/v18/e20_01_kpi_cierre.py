"""E20-01 (Fable): reduccion de varianza del KPI con el CIERRE del balance de Sn como covariable de error de medida.
cierre_b = (Sn metal + Sn dross + Sn polvo + Sn escoria final) / Sn cargado  (fisicamente = 1; desvio = error de pesaje/atribucion/ensayo).
(1) cuanto KPI explica; (2) PLACEBO: el cierre no debe responder al desvio de GN ni al exceso; (3) contrastes clave con el cierre (y sus rezagos) como control."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import v9_lib as L, e11_07_lib as W7
t = pd.read_csv("experimentos/v16/e18_07_tabla.csv", index_col=0).sort_values("idx_cronologico"); K = L.KPI
sal = t.sn_metal_t + t.sn_dross_t + t.sn_polvo_t; esc = sal / (1 - t.sn_perdido_escoria_frac) - sal   # Sn en escoria final implicito en la definicion del KPI refinado
t["cierre"] = (sal + esc) / t.feed_sn_total_t; t["cierre_mdp"] = sal / t.feed_sn_total_t; t["ln_cierre"] = np.log(t.cierre)
for k in (1, 2, 3): t[f"ln_cierre_l{k}"] = t.ln_cierre.shift(k)
C = W7.C_SIN_ESPESOR; d = t.dropna(subset=[K, "ln_cierre", "exc_todo", "dC", "def_todo"] + C)
print("cierre: media %.3f sd %.3f | corr con KPI %.3f, polvo %.3f, dross %.3f, metal %.3f" % (d.cierre.mean(), d.cierre.std(), d.cierre.corr(d[K]), d.cierre.corr(d.f_polvo), d.cierre.corr(d.f_dross), d.cierre.corr(d.f_metal)))
f0 = L.ols_hc3(d[K], d[C]); f1 = L.ols_hc3(d[K], d[C + ["ln_cierre"]]); print("sd residual KPI: controles %.2f -> + cierre %.2f (R2 %.3f -> %.3f)" % (f0.resid.std(), f1.resid.std(), f0.rsquared, f1.rsquared))
# placebo: el cierre responde a la accion?
for X in (["exc_todo", "def_todo", "dC"],):
    for nm, tt in (("DEV", d[d.es_dev]), ("TOTAL", d)):
        f = L.ols_hc3(tt.ln_cierre, tt[X + C]); print("PLACEBO ln_cierre ~ accion", nm, " ".join(f"{c}:{f.params[c]:+.4f}(t{f.tvalues[c]:+.1f})" for c in X))
# contrastes con y sin cierre
for ctrl_nm, extra in (("sin cierre", []), ("+ln_cierre", ["ln_cierre"]), ("+ln_cierre y rezagos 1-3", ["ln_cierre", "ln_cierre_l1", "ln_cierre_l2", "ln_cierre_l3"])):
    for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
        dd = tt.dropna(subset=[K, "exc_todo", "def_todo", "dC"] + C + extra); f = L.ols_hc3(dd[K], dd[["exc_todo", "def_todo", "dC"] + C + extra])
        print(f"{ctrl_nm:26s} {nm:5s} n {len(dd)} exceso GN -> KPI {f.params.exc_todo:+.3f} se {f.bse.exc_todo:.3f} t {f.tvalues.exc_todo:+.2f}")
# prospectivo W=100 con cierre como control
rng = np.random.default_rng(0)
def wf(Xe, Cc, yy, W=100, paso=10):
    n = len(yy); sc = np.full(n, np.nan); s = W
    while s < n:
        q = W7.ols_classic(np.column_stack([Cc[s-W:s], Xe[s-W:s]]), yy[s-W:s]); sc[s:s+paso] = Xe[s:s+paso] @ q["beta"][-Xe.shape[1]:]; s += paso
    return sc
for ctrl_nm, extra in (("sin cierre", []), ("+ln_cierre", ["ln_cierre"])):
    P = W7.preparar(t, ["exc_todo"], controles=C + extra); sc = wf(P["Xexp"], P["Xctrl"], P["y"]); o = W7.stats_final(sc, P["y"], P["Xctrl"]); blo = W7.bloques_cronologicos(P["n"], 20)
    nul = np.array([W7.stats_final(wf(W7.permutar_bloques(P["Xexp"], blo, rng), P["Xctrl"], P["y"]), P["y"], P["Xctrl"])["t"] for _ in range(2000)])
    circ = np.array([W7.stats_final(wf(np.roll(P["Xexp"], k, axis=0), P["Xctrl"], P["y"]), P["y"], P["Xctrl"])["t"] for k in range(15, P["n"] - 15)])
    lb = W7.stats_final(sc, P["y"], P["Xctrl"], ~P["es_dev"]); dv = W7.stats_final(sc, P["y"], P["Xctrl"], P["es_dev"])
    print(f"PROSPECTIVO exceso GN {ctrl_nm}: pend {o['pendiente']:.2f} t {o['t']:.2f} p_perm {(np.sum(nul >= o['t'])+1)/2001:.4f} p_circ {(np.sum(circ >= o['t'])+1)/(len(circ)+1):.4f} | DEV t {dv['t']:.2f} | LB t {lb['t']:.2f}")
t.to_csv("experimentos/v18/e20_01_tabla.csv")
