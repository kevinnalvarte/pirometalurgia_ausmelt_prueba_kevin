"""E12-06 (Fable): ajuste por covariables PRE-TRATAMIENTO (respecto de las acciones de Reduccion) para reducir la varianza residual del KPI
y estrechar beta. Evaluacion prospectiva identica a v9 (W=100, paso 10) + se(beta) medio en las ventanas + permutacion por bloques."""
import sys; sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e11_07_lib as W7, v9_lib as L
import v8_lib as L8, modelo_predictivo_v8 as mp8
t = W7.cargar_tabla(); df = L8.cargar_df(); base = mp8._preparar_base(df)
r0 = base[(base.fase_proceso == "Reducción") & (base.orden_escalon_fase == 0)].set_index("Batch")
pre = r0[["m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "temperatura_horno_celsius_prev", "m6_masa_kg_prev"]].add_prefix("R0_")
t = t.join(pre)
for k in (1, 2, 3):
    t[f"kpi_l{k}"] = t[W7.KPI].shift(k)
for c in ("f_dross", "f_polvo"):
    t[f"{c}_l3"] = t[c].shift(3)
base_c = W7.C_SIN_ESPESOR
sets = {"v9 (5 controles)": [],
        "+talon IRF F0": ["F_lnirf_F0"],
        "+KPI lag3": ["kpi_l3"],
        "+talon+lag3": ["F_lnirf_F0", "kpi_l3"],
        "+talon+lags1-3": ["F_lnirf_F0", "kpi_l1", "kpi_l2", "kpi_l3"],
        "+talon+lag3+estado R0 (Sn,FeO inv,T)": ["F_lnirf_F0", "kpi_l3", "R0_m6_sn_inv_kg_prev", "R0_m6_feo_inv_kg_prev", "R0_temperatura_horno_celsius_prev"],
        "+talon+IRF fin F+T_F+lag3": ["F_lnirf_F0", "F_lnirf_fin", "T_media_F", "kpi_l3"],
        "+todo": ["F_lnirf_F0", "F_lnirf_fin", "T_media_F", "kpi_l1", "kpi_l3", "f_dross_l3", "f_polvo_l3", "R0_m6_sn_inv_kg_prev", "R0_m6_feo_inv_kg_prev", "R0_temperatura_horno_celsius_prev", "F_sum_m6_ln_sn_dep"]}
allc = sorted({c for v in sets.values() for c in v}); t0 = t.dropna(subset=allc + base_c + ["xG", "xC", W7.KPI])   # MISMA muestra para todos
print("n comun:", len(t0))
filas = []; rng = np.random.default_rng(0)
for nm, extra in sets.items():
    P = W7.preparar(t0, ["xG", "xC"], controles=base_c + extra); n = P["n"]
    sc = W7.walk_forward_score(P["Xexp"], P["Xctrl"], P["y"], 100)
    # se medio de beta en las ventanas y R2 de los controles
    ses = []; r2 = []
    for s in range(100, n, 10):
        Z = np.column_stack([P["Xctrl"][s-100:s], P["Xexp"][s-100:s]]); r = W7.ols_classic(Z, P["y"][s-100:s]); ses.append(r["se"][-2:])
        rc = W7.ols_classic(P["Xctrl"][s-100:s], P["y"][s-100:s]); r2.append(1 - rc["resid"].var() / P["y"][s-100:s].var())
    Cb = W7.preparar(t0, ["xG", "xC"], controles=base_c)["Xctrl"]    # el contraste final usa SIEMPRE los controles base + los extra
    for mk, m in (("todo", None), ("DEV", P["es_dev"]), ("LB", ~P["es_dev"])):
        st = W7.stats_final(sc, P["y"], P["Xctrl"], m); st.update({"controles": nm, "k_extra": len(extra), "muestra": mk, "se_bG": np.mean(ses, 0)[0], "se_bC": np.mean(ses, 0)[1], "R2_controles_ventana": np.mean(r2)})
        filas.append(st)
    # permutacion por bloques (todo)
    obs = W7.stats_final(sc, P["y"], P["Xctrl"])["t"]; blo = W7.bloques_cronologicos(n, 20); nul = []
    for _ in range(1000):
        Xp = W7.permutar_bloques(P["Xexp"], blo, rng); nul.append(W7.stats_final(W7.walk_forward_score(Xp, P["Xctrl"], P["y"], 100), P["y"], P["Xctrl"])["t"])
    filas[-3]["p_perm"] = (np.sum(np.array(nul) >= obs) + 1) / 1001
out = pd.DataFrame(filas); out.to_csv("experimentos/v10/e12_06_covariables.csv", index=False)
pd.set_option("display.width", 250); print(out[["controles", "muestra", "n", "pendiente", "t", "p_uni", "p_perm", "spearman", "se_bG", "se_bC", "R2_controles_ventana"]].round(3).to_string())
