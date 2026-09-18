"""E19-02 (Fable): el asesor v11 TAL COMO SE DESPLEGARIA (con su conmutador de regimen: fuera de regimen no actua -> exposicion 0) frente al KPI en TODA la campana.
F_b prospectivo (theta solo del pasado) x 1[escalon en regimen]; contraste con KPI, dross; permutacion por bloques (2 colas) y circular; jackknife -10/-20 batches."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v16"); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e18_02_lib as E, v9_lib as L, asesor_receta_v10 as AR
r, bt = E.cargar(); bt = bt.sort_values("idx_cronologico"); orden = list(bt.index); pos = r.Batch.map({b: i for i, b in enumerate(orden)}); y = "m8_ln_feo_ret_dross50"
for o in range(4): r[f"g{o}"] = r["dev__gn"].where(r[E.ORDEN] == o, 0.0)
X = [f"g{o}" for o in range(4)] + ["dev__carbon"] + E.STATE; ok = r[[y] + X].notna().all(axis=1)
def theta(mask):
    d = r[mask & ok].reset_index(drop=True); til = E.demean_two_way(d, [y] + X, iters=25); return np.linalg.lstsq(til[[f"{x}__tilde" for x in X]].to_numpy(), til[f"{y}__tilde"].to_numpy(), rcond=None)[0][:4]
r0 = r[r[E.ORDEN] == 0].set_index("Batch")["plan__gn"]; enreg = (r.Batch.map(r0) <= AR.RECETA_GN_R0_HORNO_FRIO) & ~(r.temperatura_horno_celsius_prev < AR.T_PREV_MIN_REGLA)
F = pd.Series(np.nan, index=r.index)
for s in range(100, len(orden), 20):
    th = theta(pos < s); blo = (pos >= s) & (pos < s + 20); F[blo] = -(np.array([th[int(o)] for o in r.loc[blo, E.ORDEN]]) * r.loc[blo, "dev__gn"].clip(lower=0) * r.loc[blo, "m6_feo_inv_kg_prev"])
Fr = F.where(enreg, 0.0).where(F.notna())
g = r.sort_values(["Batch", E.ORDEN]).groupby("Batch")
B = pd.DataFrame({"F_exceso_kg": F.groupby(r.Batch).sum(min_count=4), "F_regla_kg": Fr.groupby(r.Batch).sum(min_count=4), "frac_en_regimen": enreg.groupby(r.Batch).mean(), "feo_ini": g.m6_feo_inv_kg_prev.first(), "sn_ini": g.m6_sn_inv_kg_prev.first(), "T_ini": g.temperatura_horno_celsius_prev.first(), "n": g.size()})
t = bt.join(B).replace([np.inf, -np.inf], np.nan); t = t[t.n == 4]; t["dross_pp"] = 100 * t.f_dross
C = ["feo_ini", "sn_ini", "T_ini", "feed_sn_total_t", "ley_sn_conc_batch_pct", "frac_carga_secundaria_F", "feed_dross_Fe_total_t", "idx_cronologico"]; rng = np.random.default_rng(0)
print("batches puntuados", t.F_regla_kg.notna().sum(), "| en regimen (algun escalon):", int((t.frac_en_regimen[t.F_regla_kg.notna()] > 0).sum()), "| FeO evitable medio en regimen %.0f kg" % t.F_regla_kg[t.F_regla_kg > 0].mean())
for col in ("F_regla_kg", "F_exceso_kg"):
    for yy in (L.KPI, "dross_pp"):
        d = t.dropna(subset=[col, yy] + C).reset_index(drop=True); f = L.ols_hc3(d[yy], d[[col] + C]); ry = L.ols_hc3(d[yy], d[C]).resid.to_numpy(); rs = L.ols_hc3(d[col], d[C]).resid.to_numpy(); ro = np.corrcoef(ry, rs)[0, 1]
        blo = np.arange(len(d)) // 20; nul = np.array([np.corrcoef(ry, rs[np.concatenate([rng.permutation(np.where(blo == b)[0]) for b in np.unique(blo)])])[0, 1] for _ in range(5000)])
        circ = np.array([np.corrcoef(ry, np.roll(rs, k))[0, 1] for k in range(15, len(d) - 15)]); sg = -1 if yy == L.KPI else 1
        jk = []
        for m in (10, 20):
            ts = []
            for _ in range(300):
                keep = np.sort(rng.choice(len(d), len(d) - m, replace=False)); ts.append(L.ols_hc3(d[yy].iloc[keep], d[[col] + C].iloc[keep]).tvalues[col] * sg)
            jk.append(100 * np.mean(np.array(ts) > 1.645))
        print(f"{col:12s} -> {yy[:8]:8s} n {len(d)} pend/100kg {100*f.params[col]:+.3f} [{100*f.conf_int().loc[col,0]:+.3f}, {100*f.conf_int().loc[col,1]:+.3f}] t {f.tvalues[col]:+.2f} | p_perm 1 cola {(np.sum(sg*nul >= sg*ro)+1)/5001:.4f} 2 colas {(np.sum(np.abs(nul) >= abs(ro))+1)/5001:.4f} | p_circ {(np.sum(sg*circ >= sg*ro)+1)/(len(circ)+1):.4f} | jackknife -10/-20: {jk[0]:.0f}%/{jk[1]:.0f}%")
for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev])):
    d = tt.dropna(subset=["F_regla_kg", L.KPI] + C)
    if d.F_regla_kg.std() > 0:
        f = L.ols_hc3(d[L.KPI], d[["F_regla_kg"] + C]); print(nm, "F_regla -> KPI pend/100kg %+.3f t %+.2f | batches con regla activa %d/%d" % (100 * f.params.F_regla_kg, f.tvalues.F_regla_kg, (d.F_regla_kg > 0).sum(), len(d)))
