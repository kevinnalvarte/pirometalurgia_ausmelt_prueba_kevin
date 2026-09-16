"""E9-09: exporta experimentos/v7/figs/datos_v7.js (window.DATOS_V7) con los datos del artefacto de la iteracion 9."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import v7_lib as L  # noqa: E402

def r(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)

def rows(df, cols, n=3):
    return [{c: (r(v, n) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v)) for c, v in zip(cols, row)} for row in df[cols].itertuples(index=False)]

D = {}
# 1) validacion v7 vs referencias
t = pd.read_csv(HERE / "e9_07_tabla_validacion_v7.csv")
D["validacion"] = rows(t, ["fase", "target", "forma", "n_estado", "n_dev", "n_lockbox", "r2_oof", "mae_oof", "r2_lockbox", "mae_lockbox"])
# 2) theta
e = pd.read_csv(HERE / "e9_07_efectos_control_v7.csv")
D["theta"] = rows(e, ["fase", "clave", "palanca", "theta_1sd", "ci_lo_1sd", "ci_hi_1sd", "significativo", "signo_teorico", "en_cota"], 4)
# 3) real vs pred (submuestra)
rng = np.random.default_rng(0)
D["pred"] = {}
for fase, clave in [("Reducción", "sn_dep"), ("Reducción", "feo_ret"), ("Fusión", "sn_dep"), ("Reducción", "dT")]:
    p = pd.read_csv(HERE / f"e9_07_pred_{fase}_{clave}.csv")
    out = {}
    for conj, g in p.groupby("conjunto"):
        if len(g) > 700:
            g = g.sample(700, random_state=0)
        out[conj] = {"real": [r(v) for v in g["real"]], "pred": [r(v) for v in g["pred"]], "orden": [int(v) for v in g["orden_escalon_fase"]]}
    D["pred"][f"{fase}_{clave}"] = out
# 4) techo, curva, bakeoff
th = pd.read_csv(HERE / "e9_05_techo.csv")
base = th[(th.sigma_sn == 0.02) & (th.sigma_feo == 0.028) & (th.orden == "todos")]
D["techo"] = rows(base, ["target", "r2_max", "r2_oof_plm", "r2_lockbox_plm", "r2_relativa_oof"])
sens = th[(th.orden == "todos") & (th.target == "R_feo_ret")]
D["techo_sens_feo"] = rows(sens, ["sigma_sn", "sigma_feo", "r2_max"])
por_orden = th[(th.sigma_sn == 0.02) & (th.sigma_feo == 0.028) & (th.orden != "todos")]
D["techo_orden"] = rows(por_orden, ["target", "orden", "r2_max"])
cu = pd.read_csv(HERE / "e9_05_curva.csv")
cols_cu = [c for c in ["target", "modelo", "n_batches", "r2_oof", "r2_lockbox", "rep"] if c in cu.columns]
D["curva"] = rows(cu, cols_cu)
bk = pd.read_csv(HERE / "e9_05_bakeoff.csv")
cols_bk = [c for c in ["target", "algoritmo", "r2_oof", "r2_lockbox", "mae_oof"] if c in bk.columns]
D["bakeoff"] = rows(bk, cols_bk)
# 5) formas E9-01
fo = pd.read_csv(HERE / "e9_01_comparacion.csv")
D["formas"] = rows(fo, ["forma", "clave", "n_parametros", "r2_oof", "r2_lockbox"])
# 6) parsimonia E9-06
el = pd.read_csv(HERE / "e9_06_eliminacion.csv")
D["parsimonia"] = rows(el, ["ruta", "k", "sn_dep_r2_oof", "sn_dep_r2_lockbox", "Cx_sn_theta1sd", "Cx_sn_lo", "Cx_sn_hi", "feo_ret_r2_oof", "feo_ret_r2_lockbox", "GN_feo_ret_theta1sd", "GN_feo_ret_lo", "GN_feo_ret_hi"])
# 7) agregacion / anclaje / legado
an = pd.read_csv(HERE / "e9_00_anclaje_grupos.csv")
D["anclaje_grupos"] = rows(an[an.spec.isin(["grupos_R", "fase"])], ["muestra", "spec", "variable", "coef", "lo", "hi", "p"])
lg = pd.read_csv(HERE / "e9_00b_legado.csv")
D["legado"] = rows(lg[lg.muestra == "dev"], ["spec", "y", "variable", "coef", "lo", "hi", "p"])
cs = pd.read_csv(HERE / "e9_00_consistencia_J_kpi.csv")
D["consistencia"] = rows(cs[cs.kpi == "recuperacion_refinada_pct"], ["muestra", "variable", "rho", "p", "n"])
ag = pd.read_csv(HERE / "e9_03_agregacion.csv")
D["agregacion_grupos"] = rows(ag, ["variable", "conjunto", "n", "rho", "p", "ci_lo", "ci_hi"])
vj = pd.read_csv(HERE / "e9_00_varianza_J.csv")
vj.columns = ["componente"] + list(vj.columns[1:])
D["varianza_J"] = rows(vj, ["componente", "frac_var", "media_pp"])
# 8) politica: perfil por tramo y por orden; curvas de respuesta
esc = pd.read_csv(HERE / "e9_07_recomendaciones_v7_escalones.csv")
pal = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"]
red = esc[esc.fase == "Reducción"].copy()
red["tramo"] = pd.cut(red["m6_avance_prev"], [-1, 0.84, 0.96, 0.975, 2], labels=["<0.84", "0.84-0.96", "0.96-0.975", ">0.975"])
prof = []
for tr, g in red.groupby("tramo", observed=True):
    d = {"tramo": str(tr), "n": int(len(g)), "uplift_J": r(g["uplift_J_kgSn"].mean(), 1)}
    for a in pal:
        d[f"hist_{a}"] = r(g[f"hist__{a}"].mean(), 2); d[f"delta_{a}"] = r((g[f"rec__{a}"] - g[f"hist__{a}"]).mean(), 2)
    prof.append(d)
fus = esc[esc.fase == "Fusión"]
for o, g in fus.groupby("orden_escalon_fase"):
    d = {"tramo": f"F{int(o)}", "n": int(len(g)), "uplift_J": r(g["uplift_J_kgSn"].mean(), 1)}
    for a in pal:
        d[f"hist_{a}"] = r(g[f"hist__{a}"].mean(), 2); d[f"delta_{a}"] = r((g[f"rec__{a}"] - g[f"hist__{a}"]).mean(), 2)
    prof.append(d)
D["perfil_politica"] = prof
cur = pd.read_csv(HERE / "e9_07_curvas_respuesta_v7.csv")
D["curvas"] = rows(cur, ["fase", "Batch", "avance_prev", "palanca", "valor", "J", "J_sn", "J_feo", "J_costo", "J_temperatura", "J_incertidumbre", "J_soporte", "support_score"], 2)
# 9) evidencia y uplift por batch
ev = pd.read_csv(HERE / "e9_07_evidencia_politica_v7.csv")
D["evidencia"] = rows(ev[ev.kpi.isin(["recuperacion_refinada_pct", "f_polvo", "f_dross"]) & ev.variable.isin(["dist_total", "dist_F", "dist_R"])], ["subset", "kpi", "variable", "n", "coef_por_sd", "ci_lo", "ci_hi", "p_hc3", "p_perm"], 4)
pl = pd.read_csv(HERE / "e9_07_evidencia_placebo_v7.csv")
D["placebo"] = rows(pl[(pl.kpi == "recuperacion_refinada_pct")], ["subset", "variable", "coef_por_sd", "p_hc3"], 4)
pb = pd.read_csv(HERE / "e9_07_por_batch_v7.csv")
D["uplift_batch"] = {"dev": [r(v) for v in pb.loc[~pb.es_lockbox, "uplift_batch_pp"].dropna()], "lockbox": [r(v) for v in pb.loc[pb.es_lockbox, "uplift_batch_pp"].dropna()]}
D["J_grupos_batch"] = rows(pb.groupby("es_lockbox")[["J_R_temprano_hist_pp", "J_R_temprano_rec_pp", "J_R_tardio_hist_pp", "J_R_tardio_rec_pp", "J_F_hist_pp", "J_F_rec_pp", "J_batch_hist_pp", "J_batch_rec_pp"]].mean().reset_index(), ["es_lockbox", "J_R_temprano_hist_pp", "J_R_temprano_rec_pp", "J_R_tardio_hist_pp", "J_R_tardio_rec_pp", "J_F_hist_pp", "J_F_rec_pp", "J_batch_hist_pp", "J_batch_rec_pp"])
po = pd.read_csv(HERE / "e9_08_potencia.csv")
D["potencia"] = rows(po, ["muestra", "variable", "n", "coef_obs_pp_sd", "se_hc3", "MDE_80pct", "efecto_esperado_pp_sd", "n_batches_requeridos", "potencia_actual"], 3)
# 10) horizonte y cal (tablas pequenas)
hb = pd.read_csv(HERE / "e9_03_politicas_batch.csv")
cols_hb = [c for c in hb.columns if c in ("Batch", "conjunto", "policy", "J_batch", "es_lockbox", "fold")]
if "policy" in hb.columns and "J_batch" in hb.columns:
    grp = "es_lockbox" if "es_lockbox" in hb.columns else "conjunto"
    D["horizonte"] = rows(hb.groupby([grp, "policy"])["J_batch"].median().reset_index(), [grp, "policy", "J_batch"], 1)
ca = pd.read_csv(HERE / "e9_02_resumen.csv")
D["cal"] = rows(ca, list(ca.columns), 4)
# 11) ejemplo de batch: trayectoria real de %Sn y %FeO con recomendaciones (lockbox)
df = L.cargar_df()
bex = "AP0350" if "AP0350" in set(esc.Batch) else sorted(esc.Batch.unique())[-5]
g = df[df.Batch == bex].sort_values("fecha_inicio")
D["ejemplo"] = {"batch": bex, "orden": [f"{'F' if f == 'Fusión' else 'R'}{int(o)}" for f, o in zip(g.fase_proceso, g.orden_escalon_fase)],
                "sn": [r(v, 2) for v in g.ley_sn_escoria_pct], "feo": [r(v, 2) for v in g.ley_feo_escoria_pct], "T": [r(v, 0) for v in g.temperatura_horno_celsius],
                "carbon": [r(v, 1) for v in g.tasa_feed_Carbon_kg_min], "gn": [r(v, 1) for v in g.tasa_gn_nm3_min]}
ge = esc[esc.Batch == bex].sort_values(["fase", "orden_escalon_fase"], key=lambda s: s.map({"Fusión": 0, "Reducción": 1}) if s.name == "fase" else s)
D["ejemplo"]["rec"] = rows(ge, ["fase", "orden_escalon_fase", "hist__tasa_feed_Carbon_kg_min", "rec__tasa_feed_Carbon_kg_min", "hist__tasa_gn_nm3_min", "rec__tasa_gn_nm3_min", "hist__tasa_o2_nm3_min", "rec__tasa_o2_nm3_min", "hist__tasa_aire_nm3_min", "rec__tasa_aire_nm3_min", "uplift_J_kgSn", "pred_sn_dep_hist", "pred_sn_dep_rec", "pred_feo_ret_hist", "pred_feo_ret_rec"], 2)
D["config"] = {"K": 3.522, "w": 7.86, "w_lo": 5.14, "w_hi": 13.04, "beta_F": -0.58, "kg_por_pp": 512.5}
(FIGS / "datos_v7.js").write_text("window.DATOS_V7 = " + json.dumps(D, ensure_ascii=False, allow_nan=False) + ";", encoding="utf-8")
print("ok", (FIGS / "datos_v7.js").stat().st_size // 1024, "KB", list(D.keys()))
