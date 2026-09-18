"""E15-02 punto 1: fuga en la estandarizacion. E15-01 usa media/sd de DEV (constantes, calculadas con
TODO el DEV, incluido el futuro de cada batch temprano) para estandarizar el desvio ejecutado-receta.
Reestandarizamos ESTRICTO: expansivo (solo pasado, >=40 batches previos), sin centrar (solo /sd) y en
unidades fisicas crudas (Nm3/min, kg/min, sin estandarizar). Recalculamos OLS DEV/LB/TOTAL y la prueba
prospectiva principal (W=100, JS y OLS) con cada variante."""
import sys, time
sys.path.insert(0, "experimentos/v13")
import numpy as np, pandas as pd
import e15_02_lib as E
import e11_07_lib as W7
import e13_03_lib as L13

t0 = time.time()
r, t = E.cargar()
Eexp = E.exposiciones_estrictas(r, min_hist=40)
tt = t.join(Eexp)

VARIANTES = {
    "original_E15-01 (DEV cte)": ("d_gn", "d_carbon"),
    "expansivo_centrado (>=40)": ("d_gn_exp", "d_carbon_exp"),
    "expansivo_sin_centrar": ("d_gn_sinctr", "d_carbon_sinctr"),
    "crudo_fisico (sin estandarizar)": ("d_gn_crudo", "d_carbon_crudo"),
}

filas_ols = []
for nombre, (xg, xc) in VARIANTES.items():
    for nm, sub in (("DEV", tt[tt.es_dev]), ("LB", tt[~tt.es_dev]), ("TOTAL", tt)):
        cols = [xg, xc] + E.CTRL + [E.KPI]
        dd = sub.dropna(subset=cols)
        if len(dd) < 20:
            continue
        f = W7.L.ols_hc3(dd[E.KPI], dd[[xg, xc] + E.CTRL])  # KPI ya esta en puntos porcentuales
        filas_ols.append({"variante": nombre, "muestra": nm, "n": len(dd),
                           "b_gn": f.params[xg], "t_gn": f.tvalues[xg], "b_carbon": f.params[xc], "t_carbon": f.tvalues[xc]})
out_ols = pd.DataFrame(filas_ols)
out_ols.to_csv("experimentos/v13/e15_02_01_ols.csv", index=False)
print(out_ols.round(3).to_string())

filas_wf = []
for nombre, (xg, xc) in VARIANTES.items():
    for modo in ("js", "ols"):
        try:
            res = E.test_principal(tt, [xg, xc], W=100, modo=modo, reps=1000)
        except Exception as ex:
            print(nombre, modo, "ERROR", ex); continue
        obs = res["obs"]; P = res["P"]
        circ_t = np.array([W7.stats_final(L13.wf(np.roll(P["Xexp"], k, axis=0), P["Xctrl"], P["y"], 100, modo=modo),
                                            P["y"], P["Xctrl"])["t"] for k in range(15, P["n"] - 15)])
        p_circ = (np.sum(circ_t >= obs["t"]) + 1) / (len(circ_t) + 1)
        filas_wf.append({"variante": nombre, "modo": modo, "n": obs["n"], "pendiente": obs["pendiente"], "t": obs["t"],
                          "p_perm": res["p_perm"], "p_circ": p_circ, "spearman": obs["spearman"]})
        print(nombre, modo, f"n={obs['n']} pend={obs['pendiente']:.3f} t={obs['t']:.3f} p_perm={res['p_perm']:.4f} p_circ={p_circ:.4f}",
              f"[{time.time()-t0:.0f}s]")
out_wf = pd.DataFrame(filas_wf)
out_wf.to_csv("experimentos/v13/e15_02_01_prospectivo.csv", index=False)
print(out_wf.round(4).to_string())
