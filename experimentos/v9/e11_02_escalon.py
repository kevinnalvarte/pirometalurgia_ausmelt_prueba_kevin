"""E11-02 tarea 6: coherencia con el escalon -- moderacion de theta_GN sobre m8_ln_feo_ret_dross50 y m6_ln_sn_dep por
Sn disponible, usando residuos cross-fitted (theta_dual, estado L.mp8.ESTADO_R_V8)."""
import sys
from pathlib import Path
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd
import statsmodels.api as sm

import v9_lib as L
import v8_lib as L8
import modelo_predictivo_v8 as mp8

OUT = Path("experimentos/v9")
SEED = 42
S = list(mp8.ESTADO_R_V8)
MODERADORES = [("ley_sn_escoria_pct_prev", True), ("m6_sn_inv_kg_prev", True)]
TARGETS = ["m8_ln_feo_ret_dross50", "m6_ln_sn_dep"]

df = L8.cargar_df()
base = mp8._preparar_base(df)
dev_b, lockbox_b = L8.mp.split_dev_lockbox(df)
Rall = base[base.fase_proceso == "Reducción"].copy()
Rdev = Rall[Rall.Batch.isin(dev_b)].reset_index(drop=True)

filas = []
for target in TARGETS:
    cols_necesarias = S + [target, "tasa_gn_nm3_min"] + [m for m, _ in MODERADORES]
    d = Rdev.dropna(subset=cols_necesarias).reset_index(drop=True)
    print(f"\n=== target {target}: n={len(d)} ===")
    t_tab, extras = L8.theta_dual(d, S, ["tasa_gn_nm3_min"], target, signos=None, n_boot=5, seed=SEED)
    y_res = extras["y_res"]
    gn_res = extras["A_res"][:, 0]
    gn_res_std = gn_res / gn_res.std()
    for mod, ln in MODERADORES:
        x = np.log(d[mod]) if ln else d[mod].astype(float)
        mtil = ((x - x.mean()) / x.std()).to_numpy()
        X = pd.DataFrame({"gn_res": gn_res_std, "gn_res_x_m": gn_res_std * mtil})
        Xc = sm.add_constant(X, has_constant="add")
        f = sm.OLS(y_res, Xc).fit(cov_type="cluster", cov_kwds={"groups": d["Batch"].to_numpy()})
        row = {"target": target, "moderador": mod, "ln": ln, "n": len(d),
               "b_gn": f.params["gn_res"], "p_gn": f.pvalues["gn_res"],
               "b_gn_x_m": f.params["gn_res_x_m"], "p_gn_x_m": f.pvalues["gn_res_x_m"],
               "r2": f.rsquared}
        filas.append(row)
        print(f"  moderador={mod} ln={ln}: b_gn={row['b_gn']:.4f}(p={row['p_gn']:.3f})  "
              f"b_gn_x_m={row['b_gn_x_m']:.4f}(p={row['p_gn_x_m']:.3f})")

tabla6 = pd.DataFrame(filas)
tabla6.to_csv(OUT / "e11_02_escalon_coherencia.csv", index=False)
print("\nscript escalon completo")
