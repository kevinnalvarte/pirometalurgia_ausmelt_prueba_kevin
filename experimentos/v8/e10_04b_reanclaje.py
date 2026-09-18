"""E10-04b: re-anclaje de (K, w) para un target de FeO alternativo (p. ej. `m8_ln_feo_ret_dross50`, E10-02) y comparacion con el
target v6/v7 sobre los MISMOS batches: OLS HC3 del KPI refinado sobre Σ_R ln_sn_dep + Σ_R ln_feo_ret + controles (DEV / lockbox / total),
IC bootstrap por batch de w, legado (KPI_next) y Spearman de J_R(w) con el KPI.
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experimentos/v8/e10_04b_reanclaje.py [target_feo ...]
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ / "experimentos" / "v8"))
import v8_lib as L
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
T0 = time.time()
targets = sys.argv[1:] or ["m6_ln_feo_ret", "m8_ln_feo_ret_dross40", "m8_ln_feo_ret_dross50", "m8_ln_feo_ret_dross60"]
df = L.cargar_df()
b = L.construir_batch(df).reset_index()
red = df[df.fase_proceso == "Reducción"]
for t in targets:
    s = red.groupby("Batch")[t].sum(min_count=1); b[f"R_sum_{t}"] = b["Batch"].map(s)
b = b.sort_values("idx_cronologico").reset_index(drop=True)
b["KPI_next"] = b[L.KPI_PRINCIPAL].shift(-1)
ctrl = L.CONTROLES_BATCH


def anclar(d, y, feo_col, n_boot=1000, seed=0):
    cols = ["R_sum_m6_ln_sn_dep", feo_col] + ctrl
    dd = d[[y] + cols].dropna().reset_index(drop=True)
    X = sm.add_constant(dd[cols]); r = sm.OLS(dd[y], X).fit(cov_type="HC3")
    rng = np.random.default_rng(seed); ws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(dd), len(dd)); p = sm.OLS(dd[y].iloc[idx], X.iloc[idx]).fit().params
        ws.append(p[feo_col] / p["R_sum_m6_ln_sn_dep"] if p["R_sum_m6_ln_sn_dep"] > 0 else np.nan)
    ci = r.conf_int()
    return {"n": len(dd), "K_sn": r.params["R_sum_m6_ln_sn_dep"], "K_sn_lo": ci.loc["R_sum_m6_ln_sn_dep", 0], "K_sn_hi": ci.loc["R_sum_m6_ln_sn_dep", 1], "K_sn_p": r.pvalues["R_sum_m6_ln_sn_dep"],
            "K_feo": r.params[feo_col], "K_feo_lo": ci.loc[feo_col, 0], "K_feo_hi": ci.loc[feo_col, 1], "K_feo_p": r.pvalues[feo_col],
            "w": r.params[feo_col] / r.params["R_sum_m6_ln_sn_dep"], "w_boot_lo": np.nanpercentile(ws, 2.5), "w_boot_hi": np.nanpercentile(ws, 97.5), "r2adj": r.rsquared_adj}


filas = []
comun = b.dropna(subset=[f"R_sum_{t}" for t in targets] + ["R_sum_m6_ln_sn_dep", L.KPI_PRINCIPAL])
for t in targets:
    for nombre, sub in [("dev", comun[comun.es_dev]), ("lockbox", comun[comun.es_lockbox]), ("total", comun)]:
        f = anclar(sub, L.KPI_PRINCIPAL, f"R_sum_{t}"); f.update({"target": t, "muestra": nombre, "y": "KPI"}); filas.append(f)
    for nombre, sub in [("dev", comun[comun.es_dev]), ("total", comun)]:
        f = anclar(sub, "KPI_next", f"R_sum_{t}"); f.update({"target": t, "muestra": nombre, "y": "KPI_next"}); filas.append(f)
    # Spearman de la suma con el KPI y con J_R(w=6.09)
    for nombre, sub in [("dev", comun[comun.es_dev]), ("lockbox", comun[comun.es_lockbox]), ("total", comun)]:
        rho, p = stats.spearmanr(sub[f"R_sum_{t}"], sub[L.KPI_PRINCIPAL])
        J = sub["R_sum_m6_ln_sn_dep"] + 6.09 * sub[f"R_sum_{t}"]; rj, pj = stats.spearmanr(J, sub[L.KPI_PRINCIPAL])
        filas.append({"target": t, "muestra": nombre, "y": "spearman", "n": len(sub), "rho_sum_feo_kpi": rho, "p_sum": p, "rho_J_w6.09": rj, "p_J": pj,
                      "sd_sum": float(sub[f"R_sum_{t}"].std())})
out = pd.DataFrame(filas); out.to_csv(RAIZ / "experimentos" / "v8" / "e10_04b_reanclaje.csv", index=False)
print(out.round(4).to_string()); print(f"{time.time()-T0:.0f}s")
