"""E11-05 (b) -- Regresion conjunta KPI y canales ~ controles de Fusion (carbon, gn, exo2 | o2, aire) + Reduccion + controles.
DEV / lockbox / total, tercios cronologicos, bootstrap por batch (1000), q-BH en la familia."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "experimentos/v9")
import v9_lib as L
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

OUT = L.SALIDA
res = L.residuos_escalon()
t = L.tabla_batch(res=res)
dev = t[t.es_dev].copy()
lockbox = t[~t.es_dev].copy()

CANALES = [L.KPI, "f_metal", "f_dross", "f_polvo", "sn_perdido_escoria_frac"]
REDUCCION = ["x_Rt_carbon", "x_Rt_gn", "x_Rl_carbon", "x_Rl_gn"]
SPEC_A = ["x_F_carbon", "x_F_gn", "x_F_exo2"] + REDUCCION           # exo2 como indice compuesto
SPEC_B = ["x_F_carbon", "x_F_gn", "x_F_o2", "x_F_aire"] + REDUCCION  # primitivos O2 y aire por separado


def escala(y):
    return 100.0 if (y.startswith("f_") or y.startswith("sn_")) else 1.0


def corre(d, X, y):
    dd = d.dropna(subset=X + L.CONTROLES + [y])
    f = L.ols_hc3(dd[y] * escala(y), dd[X + L.CONTROLES])
    return f, len(dd)


filas = []
for spec_nombre, X in [("A_exo2", SPEC_A), ("B_o2_aire", SPEC_B)]:
    for regimen, d in [("DEV", dev), ("lockbox", lockbox), ("total", t)]:
        for y in CANALES:
            f, n = corre(d, X, y)
            for x in X:
                filas.append(dict(spec=spec_nombre, regimen=regimen, y=y, x=x, n=n,
                                   beta=f.params[x], se=f.bse[x], p=f.pvalues[x]))
tabla = pd.DataFrame(filas)

# q-BH dentro de la familia (spec, regimen): todas las combinaciones (x en Fusion) x canal
tabla["es_fusion"] = tabla["x"].str.startswith("x_F_")
tabla["q_bh"] = np.nan
for (spec_nombre, regimen), sub in tabla[tabla.es_fusion].groupby(["spec", "regimen"]):
    q = multipletests(sub["p"].to_numpy(), method="fdr_bh")[1]
    tabla.loc[sub.index, "q_bh"] = q
tabla.to_csv(OUT / "e11_05_regresion_conjunta.csv", index=False)
print(tabla[tabla.es_fusion].round(4).to_string())

# ---------- tercios cronologicos (sobre el total, spec A) ----------
tt = t.sort_values("idx_cronologico").reset_index()
tt["tercio"] = pd.qcut(tt["idx_cronologico"], 3, labels=["T1", "T2", "T3"])
filas_t = []
for tercio, d in tt.groupby("tercio", observed=True):
    for y in CANALES:
        f, n = corre(d, SPEC_A, y)
        for x in SPEC_A:
            filas_t.append(dict(tercio=tercio, y=y, x=x, n=n, beta=f.params[x], p=f.pvalues[x]))
terc = pd.DataFrame(filas_t)
terc.to_csv(OUT / "e11_05_tercios.csv", index=False)
print()
print(terc[terc.x.str.startswith("x_F_")].round(4).to_string())

# ---------- bootstrap por batch (1000), DEV, spec A, KPI y canales ----------
rng = np.random.default_rng(42)
n_boot = 1000
batches = dev.index.to_numpy()
boot = {y: {x: [] for x in SPEC_A} for y in CANALES}
for b in range(n_boot):
    samp = rng.choice(batches, size=len(batches), replace=True)
    d = dev.loc[samp]
    for y in CANALES:
        try:
            dd = d.dropna(subset=SPEC_A + L.CONTROLES + [y])
            f = L.ols_hc3(dd[y] * escala(y), dd[SPEC_A + L.CONTROLES])
            for x in SPEC_A:
                boot[y][x].append(f.params[x])
        except Exception:
            pass
filas_b = []
for y in CANALES:
    for x in SPEC_A:
        arr = np.array(boot[y][x])
        if len(arr) < 100:
            continue
        filas_b.append(dict(y=y, x=x, n_boot=len(arr), media=arr.mean(),
                             ic_lo=np.percentile(arr, 2.5), ic_hi=np.percentile(arr, 97.5)))
bdf = pd.DataFrame(filas_b)
bdf.to_csv(OUT / "e11_05_bootstrap_batch.csv", index=False)
print()
print(bdf[bdf.x.str.startswith("x_F_")].round(4).to_string())
