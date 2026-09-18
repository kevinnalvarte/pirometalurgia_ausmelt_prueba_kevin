"""E11-08 tarea 4: nivel BATCH (n=362), tabla e11_06b_tabla_batch.csv (xG, xC = exposicion residual media del batch
en GN y carbon de Reduccion, T_media_R). Modelo con interaccion xG x zT y xC x zT, y xG x z(idx), xC x z(idx):
que moderador gana (T o tiempo de campana)? Outcomes: KPI, f_dross, f_polvo, f_metal, sn_perdido_escoria_frac.
HC3 y bootstrap (no-cluster, n=1 batch/fila)."""
import sys
sys.path.insert(0, "experimentos/v9")
import numpy as np
import pandas as pd
import v9_lib as L

SEED = 42
N_BOOT = 500

t = pd.read_csv(L.SALIDA / "e11_06b_tabla_batch.csv", index_col=0).sort_values("idx_cronologico")
C = [c for c in L.CONTROLES if c != "espesor_ladrillo_norm_mm"]  # espesor colineal con idx / T (reloj de campana)
OUTCOMES = {L.KPI: 1.0, "f_dross": 100.0, "f_polvo": 100.0, "f_metal": 100.0, "sn_perdido_escoria_frac": 100.0}
X_BASE = ["xG", "xC"]


def boot_rows(d, ycol, Xcols, n_boot=N_BOOT, seed=SEED):
    y = d[ycol].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in Xcols])
    rng = np.random.default_rng(seed)
    n = len(d)
    out = np.zeros((n_boot, len(Xcols)))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        out[i] = beta[1:]
    return out


filas = []
for muestra, tt in (("DEV", t[t.es_dev]), ("TOTAL", t)):
    d = tt.dropna(subset=X_BASE + C + list(OUTCOMES) + ["T_media_R", "idx_cronologico"]).copy()
    d["zT"] = (d["T_media_R"] - d["T_media_R"].mean()) / d["T_media_R"].std()
    d["zidx"] = (d["idx_cronologico"] - d["idx_cronologico"].mean()) / d["idx_cronologico"].std()
    d["xG_x_zT"] = d["xG"] * d["zT"]; d["xC_x_zT"] = d["xC"] * d["zT"]
    d["xG_x_zidx"] = d["xG"] * d["zidx"]; d["xC_x_zidx"] = d["xC"] * d["zidx"]
    Xcols = ["xG", "xC", "xG_x_zT", "xC_x_zT", "xG_x_zidx", "xC_x_zidx"]
    for y_raw, esc in OUTCOMES.items():
        d["__y"] = d[y_raw] * esc
        fit = L.ols_hc3(d["__y"], d[Xcols + C])
        boots = boot_rows(d, "__y", Xcols + C)
        for i, c in enumerate(Xcols):
            filas.append({"muestra": muestra, "outcome": y_raw, "n": len(d), "termino": c,
                         "coef": float(fit.params[c]), "se_hc3": float(fit.bse[c]), "t": float(fit.tvalues[c]),
                         "p_hc3": float(fit.pvalues[c]),
                         "ci_lo_boot": float(np.percentile(boots[:, i], 2.5)), "ci_hi_boot": float(np.percentile(boots[:, i], 97.5))})
    del d["__y"]

tab4 = pd.DataFrame(filas)
tab4.to_csv(L.SALIDA / "e11_08_tabla4_batch_T_vs_idx.csv", index=False)

pd.set_option("display.width", 220)
print("=== Tarea 4: xG/xC x zT vs x zidx, nivel batch ===")
print(tab4.round(4).to_string())

# Resumen: para cada outcome, |t| de la interaccion con T vs con idx (gana la mayor, y si su IC excluye 0)
resumen = []
for muestra in ("DEV", "TOTAL"):
    for y in OUTCOMES:
        for palanca in ("G", "C"):
            sub = tab4[(tab4.muestra == muestra) & (tab4.outcome == y)]
            rT = sub[sub.termino == f"x{palanca}_x_zT"].iloc[0]
            rI = sub[sub.termino == f"x{palanca}_x_zidx"].iloc[0]
            gana = "T" if abs(rT["t"]) > abs(rI["t"]) else "idx"
            resumen.append({"muestra": muestra, "outcome": y, "palanca": palanca, "t_T": rT["t"], "p_T": rT["p_hc3"],
                            "t_idx": rI["t"], "p_idx": rI["p_hc3"], "gana": gana,
                            "alguno_sig_p05": bool((rT["p_hc3"] < 0.05) or (rI["p_hc3"] < 0.05))})
resumen = pd.DataFrame(resumen)
resumen.to_csv(L.SALIDA / "e11_08_tabla4_resumen_ganador.csv", index=False)
print("\n=== Resumen: gana T o idx (por |t|) ===")
print(resumen.round(4).to_string())
print("\nOK e11_08_02")
