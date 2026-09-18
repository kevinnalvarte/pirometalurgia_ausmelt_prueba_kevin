"""E19-01 (Fable): asesor v11 = modelo FISICO de escalon (intra-batch) + valoracion por balance de Sn.
Modelo: ln_feo_ret_bt = alpha_b + gamma_o + theta_o * dev_gn_bt + estado  (efectos fijos de batch y orden; theta por orden)
Prediccion por batch del FeO EXTRA reducido por el desvio de GN:  F_b = sum_t [ -theta_o(t) * dev_gn_t * FeO_inv_prev_t ]   [kg]
Validacion PROSPECTIVA ESTRICTA: theta estimado solo con batches anteriores (ventana EXPANSIVA, re-estimado cada 20 batches desde el 100).
Contrastes: FeO total reducido en la Reduccion (ensayo), dross (pp), KPI; calibracion (pendiente ~ 1 para el FeO), permutacion por bloques, DEV y lockbox."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v16"); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e18_02_lib as E, v9_lib as L, e11_07_lib as W7
r, bt = E.cargar(); bt = bt.sort_values("idx_cronologico"); orden = list(bt.index); pos = r.Batch.map({b: i for i, b in enumerate(orden)})
y = "m8_ln_feo_ret_dross50"
for o in range(4):
    r[f"g{o}"] = r["dev__gn"].where(r[E.ORDEN] == o, 0.0)
X = [f"g{o}" for o in range(4)] + ["dev__carbon"] + E.STATE
ok = r[[y] + X].notna().all(axis=1)
def theta(mask):
    d = r[mask & ok].reset_index(drop=True); til = E.demean_two_way(d, [y] + X, iters=25)
    return np.linalg.lstsq(til[[f"{x}__tilde" for x in X]].to_numpy(), til[f"{y}__tilde"].to_numpy(), rcond=None)[0][:4]
F = pd.Series(np.nan, index=r.index); hist = []
for s in range(100, len(orden), 20):
    th = theta(pos < s); hist.append({"s": s, **{f"theta_R{o}": th[o] for o in range(4)}})
    blo = (pos >= s) & (pos < s + 20); F[blo] = -(np.array([th[int(o)] for o in r.loc[blo, E.ORDEN]]) * r.loc[blo, "dev__gn"] * r.loc[blo, "m6_feo_inv_kg_prev"])
print(pd.DataFrame(hist).round(4).to_string())
g = r.sort_values(["Batch", E.ORDEN]).groupby("Batch")
B = pd.DataFrame({"F_pred_kg": F.groupby(r.Batch).sum(min_count=4), "feo_red_kg": g.m6_feo_inv_kg_prev.first() - g.m6_feo_inv_kg.last(), "feo_ini": g.m6_feo_inv_kg_prev.first(), "sn_ini": g.m6_sn_inv_kg_prev.first(),
                  "T_ini": g.temperatura_horno_celsius_prev.first(), "n": g.size()})
t = bt.join(B).replace([np.inf, -np.inf], np.nan); t = t[(t.n == 4)]; t["dross_pp"] = 100 * t.f_dross; t["polvo_pp"] = 100 * t.f_polvo; t.to_csv("experimentos/v17/e19_01_tabla.csv")
C = ["feo_ini", "sn_ini", "T_ini", "feed_sn_total_t", "ley_sn_conc_batch_pct", "frac_carga_secundaria_F", "feed_dross_Fe_total_t", "idx_cronologico"]
print("F_pred_kg: media %.0f sd %.0f | n puntuados %d" % (t.F_pred_kg.mean(), t.F_pred_kg.std(), t.F_pred_kg.notna().sum()))
rng = np.random.default_rng(0)
for yy in ("feo_red_kg", "dross_pp", "polvo_pp", L.KPI):
    for nm, tt in (("todo", t), ("DEV", t[t.es_dev]), ("LB", t[~t.es_dev])):
        d = tt.dropna(subset=["F_pred_kg", yy] + C); f = L.ols_hc3(d[yy], d[["F_pred_kg"] + C]); line = f"{yy[:10]:10s} {nm:5s} n {len(d):3d} pendiente {f.params.F_pred_kg:+.4f} [{f.conf_int().loc['F_pred_kg',0]:+.4f}, {f.conf_int().loc['F_pred_kg',1]:+.4f}] t {f.tvalues.F_pred_kg:+.2f}"
        if nm == "todo":
            ry = L.ols_hc3(d[yy], d[C]).resid.to_numpy(); rs = L.ols_hc3(d.F_pred_kg, d[C]).resid.to_numpy(); ro = np.corrcoef(ry, rs)[0, 1]; blo = np.arange(len(d)) // 20
            nul = [np.corrcoef(ry, rs[np.concatenate([rng.permutation(np.where(blo == b)[0]) for b in np.unique(blo)])])[0, 1] for _ in range(5000)]
            p2 = (np.sum(np.abs(nul) >= abs(ro)) + 1) / 5001; line += f" | r parcial {ro:+.3f} p_perm(2 colas) {p2:.4f}"
        print(line)
