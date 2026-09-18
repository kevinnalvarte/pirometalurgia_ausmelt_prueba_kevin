"""E12-05 tarea 5: prueba prospectiva (e11_07_lib.walk_forward_score, W=100, paso=10) comparando {xG,xC} vs
{xG,xC,candidata}, DEV/LOCKBOX/TOTAL; nula por permutacion en bloques cronologicos de 20 (>=1000 reps) sobre la
columna de la candidata."""
import sys; sys.path.insert(0, "experimentos/v10"); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e12_05_lib as E
import e11_07_lib as E7

t = pd.read_csv(E.SALIDA / "e12_05_tabla_batch.csv", index_col=0).sort_values("idx_cronologico")
CAND = list(E.PALANCAS)
GRUPOS = ("R", "F")
W = 100
N_PERM = 1200

filas = []
for g in GRUPOS:
    for p in CAND:
        col = f"x_{g}_{p}"
        if col not in t.columns:
            continue
        d = t.dropna(subset=["xG", "xC", col] + E7.C_SIN_ESPESOR + [E.KPI]).reset_index(drop=True)
        Xctrl = d[E7.C_SIN_ESPESOR].to_numpy(float); y = d[E.KPI].to_numpy(float); es_dev = d["es_dev"].to_numpy(bool)
        Xbase = d[["xG", "xC"]].to_numpy(float)
        Xcand = d[["xG", "xC", col]].to_numpy(float)
        score_base = E7.walk_forward_score(Xbase, Xctrl, y, W, paso=E7.PASO)
        score_cand = E7.walk_forward_score(Xcand, Xctrl, y, W, paso=E7.PASO)
        for nm, sc in (("base(xG+xC)", score_base), (f"cand(xG+xC+{g}_{p})", score_cand)):
            for muestra, m in (("DEV", es_dev), ("LOCKBOX", ~es_dev), ("TOTAL", np.ones(len(d), bool))):
                r = E7.stats_final(sc, y, Xctrl, m)
                filas.append({"grupo": g, "candidata": p, "spec": nm, "muestra": muestra, **{k: v for k, v in r.items()}})
        # ------- permutacion en bloques de 20 sobre la columna candidata (>=1000 reps), estadistico = t(DEV) del modelo con candidata
        r_obs = E7.stats_final(score_cand, y, Xctrl, es_dev)
        t_obs = r_obs["t"]
        bloques = E7.bloques_cronologicos(len(d), tam=20)
        rng = np.random.default_rng(E.SEED)
        xc_col = d[[col]].to_numpy(float)
        tperm = np.empty(N_PERM)
        for i in range(N_PERM):
            xp = E7.permutar_bloques(xc_col, bloques, rng)
            Xp = np.column_stack([Xbase, xp])
            sp = E7.walk_forward_score(Xp, Xctrl, y, W, paso=E7.PASO)
            rp = E7.stats_final(sp, y, Xctrl, es_dev)
            tperm[i] = rp["t"] if np.isfinite(rp["t"]) else 0.0
        p_perm = float((np.sum(np.abs(tperm) >= abs(t_obs)) + 1) / (N_PERM + 1)) if np.isfinite(t_obs) else np.nan
        filas.append({"grupo": g, "candidata": p, "spec": f"PERM_bloques20(n={N_PERM})", "muestra": "DEV",
                      "n": r_obs["n"], "pendiente": r_obs["pendiente"], "t": t_obs, "p_uni": r_obs["p_uni"],
                      "spearman": r_obs["spearman"], "sd_score": r_obs.get("sd_score"), "p_perm_bilateral": p_perm})
        print(f"{g} {p}: t_obs(DEV, con candidata)={t_obs:.2f} | t_base(DEV)={E7.stats_final(score_base, y, Xctrl, es_dev)['t']:.2f} "
              f"| p_perm_bilateral={p_perm:.3f}")

out = pd.DataFrame(filas)
out.to_csv(E.SALIDA / "e12_05_prospectiva.csv", index=False)
pd.set_option("display.width", 220)
print(out[out.spec.str.startswith("base") | out.spec.str.startswith("cand")][
    ["grupo", "candidata", "spec", "muestra", "n", "pendiente", "t", "p_uni", "spearman", "sd_score"]].round(4).to_string(index=False))
