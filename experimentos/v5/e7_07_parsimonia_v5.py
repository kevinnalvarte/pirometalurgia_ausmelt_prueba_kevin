"""E7-07: los sets parsimoniosos de E7-04 (elegidos con P3) conservan la identificacion de las palancas v5 (P2 con signos)?"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "experimentos/v5"))
import v5_lib as v5, modelo_predictivo_v5 as mp5, modelo_prescriptivo as mp
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
df = mp5.agregar_columnas_v5(pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl"))
dR = v5.base_fase(df, "Reducción")
par = pd.read_csv(RAIZ / "experimentos/v5/e7_04_parsimonia.csv")
def feats(target, k):
    r = par[(par.target == target) & (par.k == k)].iloc[0]["features"]
    return [x.strip() for x in r.split(";") if x.strip()]
dev, lb = mp.split_dev_lockbox(df)
filas = []
for clave, target, sets in [("sn_dep", "v5_ln_sn_dep", [37, 16, 10, 7]), ("feo_ret", "v5_ln_feo_ret", [37, 16, 10])]:
    A = mp5.PALANCAS_V5[("Reducción", clave)]; signos = mp5.SIGNO_TEORICO_V5[("Reducción", clave)]
    for k in sets:
        S = mp5.ESTADO_R_FULL if k == 37 else feats(target, k)
        need = list(dict.fromkeys(S + A + [target, "Batch"]))
        d_dev = dR.loc[dR.Batch.isin(dev), need].dropna().reset_index(drop=True); d_lb = dR.loc[dR.Batch.isin(lb), need].dropna().reset_index(drop=True)
        oof = np.full(len(d_dev), np.nan)
        for tr, te in GroupKFold(5).split(d_dev, d_dev[target], d_dev["Batch"]):
            m = mp5.ModeloPLMSignos(S, A, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0)
            oof[te] = m.predict(d_dev.iloc[te])
        m = mp5.ModeloPLMSignos(S, A, target, signos).fit(d_dev, n_boot=200)
        t = m.tabla_theta(signos)
        fila = {"target": target, "k": k, "r2_oof": r2_score(d_dev[target], oof), "r2_lockbox": r2_score(d_lb[target], m.predict(d_lb)), "n_dev": len(d_dev)}
        for a in A:
            fila[f"{a}_1sd"] = t.loc[a, "theta_1sd"]; fila[f"{a}_lo"] = t.loc[a, "ci_lo_1sd"]; fila[f"{a}_hi"] = t.loc[a, "ci_hi_1sd"]
        filas.append(fila); print(pd.DataFrame([fila]).round(3).to_string(), flush=True)
out = pd.DataFrame(filas); out.to_csv(RAIZ / "experimentos/v5/e7_07_parsimonia_v5.csv", index=False)
print(out.round(3).to_string())
