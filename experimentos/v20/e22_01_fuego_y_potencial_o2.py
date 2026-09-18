"""E22-01 (Fable): potencia de fuego y potencial de oxigeno de la lanza, como DESVIO respecto de la RECETA (exacto, ejecutable).
  O2_tot   = O2 + 0.21*aire                      [Nm3/min]
  fuego    = min(GN, O2_tot/2)                   GN quemado: potencia termica de la lanza (CH4 + 2 O2)
  o2_libre = O2_tot - 2*GN                       >0 lanza oxidante (O2 libre al bano), <0 reductora (combustible sin quemar)   [potencial de oxigeno, en caudal]
  lambda   = O2_tot / (2*GN)                     estequiometria de lanza (adimensional)
Desvio = ejecutado - receta (la receta trae GN, O2 y aire por escalon). (1) batch: KPI y canales; (2) intra-batch (EF de batch y orden): FeO ret, Sn dep, dT;
(3) prospectivo W=100 con permutacion por bloques; (4) efecto sobre el estado final (FeO total reducido, Sn final)."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experimentos/v16"); sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
import e18_02_lib as E, v9_lib as L, e11_07_lib as W7
r, bt = E.cargar(); bt = bt.sort_values("idx_cronologico")
def fis(gn, o2, aire):
    ot = o2 + 0.21 * aire; return np.minimum(gn, ot / 2), ot - 2 * gn, ot / (2 * gn)
r["fuego"], r["o2l"], r["lam"] = fis(r.tasa_gn_nm3_min, r.tasa_o2_nm3_min, r.tasa_aire_nm3_min); r["fuego_p"], r["o2l_p"], r["lam_p"] = fis(r.plan__gn, r.plan__o2, r.plan__aire)
for v in ("fuego", "o2l", "lam"): r[f"d_{v}"] = r[v] - r[f"{v}_p"]
pd.set_option("display.width", 220)
print(r.groupby(E.ORDEN)[["fuego_p", "fuego", "d_fuego", "o2l_p", "o2l", "d_o2l", "lam_p", "lam", "d_lam"]].mean().round(2).to_string())
print("%% escalones con lanza reductora (o2_libre<0): receta %.0f ejecutado %.0f | corr desvios:" % (100 * (r.o2l_p < 0).mean(), 100 * (r.o2l < 0).mean())); print(r[["dev__gn", "d_fuego", "d_o2l", "d_lam", "dev__carbon"]].corr().round(2).to_string())
g = r.groupby("Batch"); late = r[E.ORDEN] >= 2
B = pd.DataFrame({"dF": g.d_fuego.mean(), "dO": g.d_o2l.mean(), "dL": g.d_lam.mean(), "dG": g.dev__gn.mean(), "dC": g.dev__carbon.mean(), "dF_tard": r[late].groupby("Batch").d_fuego.mean(), "dO_tard": r[late].groupby("Batch").d_o2l.mean(),
                  "dF_temp": r[~late].groupby("Batch").d_fuego.mean(), "dO_temp": r[~late].groupby("Batch").d_o2l.mean()})
rs = r.sort_values(["Batch", E.ORDEN]).groupby("Batch"); B["feo_red"] = rs.m6_feo_inv_kg_prev.first() - rs.m6_feo_inv_kg.last(); B["sn_fin"] = rs.m6_sn_inv_kg.last(); B["feo_ini"] = rs.m6_feo_inv_kg_prev.first(); B["sn_ini"] = rs.m6_sn_inv_kg_prev.first()
t = bt.join(B).replace([np.inf, -np.inf], np.nan); t["dross_pp"] = 100 * t.f_dross; t["polvo_pp"] = 100 * t.f_polvo; t["escoria_pp"] = 100 * t.sn_perdido_escoria_frac; t.to_csv("experimentos/v20/e22_01_tabla.csv")
C = W7.C_SIN_ESPESOR; print("sd por batch: dF %.2f dO %.2f dL %.3f" % (t.dF.std(), t.dO.std(), t.dL.std()))
print("\n=== (1) BATCH: efecto por +1 Nm3/min (OLS HC3, conjunta con carbon)")
for Xs in (["dF", "dO", "dC"], ["dG", "dO", "dC"], ["dF_temp", "dF_tard", "dO_temp", "dO_tard", "dC"]):
    for nm, tt in (("DEV", t[t.es_dev]), ("LB", t[~t.es_dev]), ("TOTAL", t)):
        for y in (L.KPI, "dross_pp", "polvo_pp", "escoria_pp", "feo_red", "sn_fin"):
            cc = C + (["feo_ini", "sn_ini"] if y in ("feo_red", "sn_fin") else []); d = tt.dropna(subset=Xs + cc + [y]); f = L.ols_hc3(d[y], d[Xs + cc]); print(f"{nm:5s} {y[:10]:10s} n {len(d):3d} " + " ".join(f"{x}:{f.params[x]:+.3g}(t{f.tvalues[x]:+.1f})" for x in Xs))
print("\n=== (2) INTRA-BATCH (EF batch y orden; cluster por batch)")
for y in ("m8_ln_feo_ret_dross50", "m6_ln_sn_dep", "d_temperatura_horno_celsius"):
    if y not in r.columns: continue
    for Xs in (["d_fuego", "d_o2l", "dev__carbon"], ["dev__gn", "d_o2l", "dev__carbon"]):
        tab = E.within_ols(r, y, Xs + E.STATE); tb = tab[tab.variable.isin(Xs)]; print(f"{y[:22]:22s} " + " ".join(f"{v.variable}:{v.theta:+.4g}(t{v.t:+.1f})" for v in tb.itertuples()) + f" n {tab.attrs['n']}")
    for nm, m in (("DEV", r.es_dev), ("LB", ~r.es_dev)):
        tab = E.within_ols(r[m], y, ["d_fuego", "d_o2l", "dev__carbon"] + E.STATE); tb = tab[tab.variable.isin(["d_fuego", "d_o2l"])]; print(f"   {nm}: " + " ".join(f"{v.variable}:{v.theta:+.4g}(t{v.t:+.1f})" for v in tb.itertuples()))
print("\n=== (3) PROSPECTIVO W=100 (KPI), permutacion por bloques de 20 (1000)")
rng = np.random.default_rng(0)
def wf(Xe, Cc, yy, W=100, paso=10):
    n = len(yy); sc = np.full(n, np.nan); s = W
    while s < n:
        q = W7.ols_classic(np.column_stack([Cc[s-W:s], Xe[s-W:s]]), yy[s-W:s]); sc[s:s+paso] = Xe[s:s+paso] @ q["beta"][-Xe.shape[1]:]; s += paso
    return sc
t["excG"] = r.assign(e=r.dev__gn.clip(lower=0)).groupby("Batch").e.mean(); t["excF"] = r.assign(e=r.d_fuego.clip(lower=0)).groupby("Batch").e.mean()
for xs in (["dG"], ["dF"], ["dO"], ["dF", "dO"], ["dG", "dO"], ["excG"], ["excF"], ["excF", "dO"]):
    P = W7.preparar(t, xs); sc = wf(P["Xexp"], P["Xctrl"], P["y"]); o = W7.stats_final(sc, P["y"], P["Xctrl"]); blo = W7.bloques_cronologicos(P["n"], 20)
    nul = np.array([W7.stats_final(wf(W7.permutar_bloques(P["Xexp"], blo, rng), P["Xctrl"], P["y"]), P["y"], P["Xctrl"])["t"] for _ in range(1000)]); dv = W7.stats_final(sc, P["y"], P["Xctrl"], P["es_dev"]); lb = W7.stats_final(sc, P["y"], P["Xctrl"], ~P["es_dev"])
    print(f"{str(xs):18s} pend {o['pendiente']:.2f} t {o['t']:.2f} spearman {o['spearman']:.3f} p_perm {(np.sum(nul >= o['t'])+1)/1001:.4f} | DEV t {dv['t']:.2f} | LB t {lb['t']:.2f}")
