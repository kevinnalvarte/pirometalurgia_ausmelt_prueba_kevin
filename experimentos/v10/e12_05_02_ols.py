"""E12-05 tarea 2: OLS HC3 KPI y canales ~ xG+xC (controles sin espesor) + candidata (una a una, y conjunta por fase),
DEV / lockbox / total; tercios; q-BH dentro de la familia (candidata x grupo -> KPI, DEV); bootstrap por batch (1000)
para las que pasen p<0.05 en DEV."""
import sys; sys.path.insert(0, "experimentos/v10")
import numpy as np, pandas as pd
from scipy import stats
import e12_05_lib as E

t = pd.read_csv(E.SALIDA / "e12_05_tabla_batch.csv", index_col=0)
CTRL = E.C_SIN_ESPESOR
BASE = ["xG", "xC"]
CAND = list(E.PALANCAS)          # 7 candidatas nuevas de H5
GRUPOS = ("R", "F")
Y_KPI = E.KPI
CANALES = E.CANALES

dev = t[t.es_dev]; lb = t[~t.es_dev]

# ---------------------------------------------------------------- 1) una a una, KPI, DEV/lockbox/total
filas = []
for g in GRUPOS:
    for p in CAND:
        col = f"x_{g}_{p}"
        if col not in t.columns:
            continue
        X = BASE + CTRL + [col]
        for muestra, tt in (("DEV", dev), ("LOCKBOX", lb), ("TOTAL", t)):
            d = tt.dropna(subset=X + [Y_KPI])
            if len(d) < 30:
                continue
            f = E.ols_hc3(d[Y_KPI], d[X])
            filas.append({"grupo": g, "candidata": p, "muestra": muestra, "n": len(d),
                          "coef": float(f.params[col]), "se": float(f.bse[col]), "t": float(f.tvalues[col]),
                          "p": float(f.pvalues[col])})
uno = pd.DataFrame(filas)
uno.to_csv(E.SALIDA / "e12_05_ols_una_a_una_kpi.csv", index=False)

# q-BH dentro de la familia DEV (candidata x grupo -> KPI)
fam = uno[uno.muestra == "DEV"].copy()
fam["q_bh"] = E.bh(fam.set_index(fam.grupo + "_" + fam.candidata)["p"]).to_numpy()
uno = uno.merge(fam[["grupo", "candidata", "q_bh"]], on=["grupo", "candidata"], how="left")
uno.to_csv(E.SALIDA / "e12_05_ols_una_a_una_kpi.csv", index=False)
pd.set_option("display.width", 200)
print("== Una a una, KPI ~ xG+xC+controles+candidata =="); print(uno.round(4).to_string(index=False))

# ---------------------------------------------------------------- 2) conjunta por grupo (todas las candidatas del grupo a la vez)
filas_c = []
for g in GRUPOS:
    cols = [f"x_{g}_{p}" for p in CAND if f"x_{g}_{p}" in t.columns]
    X = BASE + CTRL + cols
    for muestra, tt in (("DEV", dev), ("LOCKBOX", lb), ("TOTAL", t)):
        d = tt.dropna(subset=X + [Y_KPI])
        if len(d) < 30 + len(X):
            continue
        f = E.ols_hc3(d[Y_KPI], d[X])
        for c in cols:
            filas_c.append({"grupo": g, "candidata": c.split("_", 2)[-1], "muestra": muestra, "n": len(d),
                            "coef": float(f.params[c]), "se": float(f.bse[c]), "t": float(f.tvalues[c]), "p": float(f.pvalues[c])})
conj = pd.DataFrame(filas_c)
conj.to_csv(E.SALIDA / "e12_05_ols_conjunta_kpi.csv", index=False)
print("\n== Conjunta (todas las candidatas del grupo a la vez), KPI =="); print(conj.round(4).to_string(index=False))

# ---------------------------------------------------------------- 3) canales (una a una, DEV/lockbox/total)
filas_ch = []
for g in GRUPOS:
    for p in CAND:
        col = f"x_{g}_{p}"
        if col not in t.columns:
            continue
        X = BASE + CTRL + [col]
        for ycol in CANALES:
            for muestra, tt in (("DEV", dev), ("LOCKBOX", lb), ("TOTAL", t)):
                d = tt.dropna(subset=X + [ycol])
                if len(d) < 30:
                    continue
                f = E.ols_hc3(d[ycol], d[X])
                filas_ch.append({"grupo": g, "candidata": p, "canal": ycol, "muestra": muestra, "n": len(d),
                                 "coef": float(f.params[col]), "t": float(f.tvalues[col]), "p": float(f.pvalues[col])})
canal = pd.DataFrame(filas_ch)
canal.to_csv(E.SALIDA / "e12_05_ols_canales.csv", index=False)
print("\n== Canales, DEV (solo p<0.10) =="); print(canal[(canal.muestra == "DEV") & (canal.p < 0.10)].round(4).to_string(index=False))

# ---------------------------------------------------------------- 4) tercios (estabilidad del signo), DEV
filas_t = []
for g in GRUPOS:
    for p in CAND:
        col = f"x_{g}_{p}"
        if col not in t.columns:
            continue
        X = BASE + CTRL + [col]
        d = dev.dropna(subset=X + [Y_KPI]).sort_values("idx_cronologico")
        k = 3
        d["tercio"] = pd.qcut(d["idx_cronologico"], k, labels=False, duplicates="drop")
        signos = []
        for i in sorted(d["tercio"].unique()):
            dd = d[d.tercio == i]
            if len(dd) < 15:
                signos.append("."); continue
            f = E.ols_hc3(dd[Y_KPI], dd[X])
            signos.append("+" if f.params[col] > 0 else "-")
        filas_t.append({"grupo": g, "candidata": p, "signos_tercios": "".join(signos)})
terc = pd.DataFrame(filas_t)
terc.to_csv(E.SALIDA / "e12_05_tercios.csv", index=False)
print("\n== Estabilidad de signo por tercios cronologicos (DEV) =="); print(terc.to_string(index=False))

# ---------------------------------------------------------------- 5) bootstrap por batch (1000) para p<0.05 en DEV
pasan = uno[(uno.muestra == "DEV") & (uno.p < 0.05)][["grupo", "candidata"]].drop_duplicates()
filas_b = []
for _, r in pasan.iterrows():
    g, p = r["grupo"], r["candidata"]
    col = f"x_{g}_{p}"
    X = BASE + CTRL + [col]
    b = E.bootstrap_batch(dev, Y_KPI, col, [c for c in X if c != col], n_boot=1000)
    b.update({"grupo": g, "candidata": p})
    filas_b.append(b)
boot = pd.DataFrame(filas_b)
boot.to_csv(E.SALIDA / "e12_05_bootstrap.csv", index=False)
print("\n== Bootstrap por batch (1000), candidatas con p<0.05 en DEV =="); print(boot.round(4).to_string(index=False) if len(boot) else "(ninguna paso p<0.05 en DEV)")
