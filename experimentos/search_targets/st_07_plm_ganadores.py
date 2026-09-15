"""ST-07 (Fable 5.1) -- efecto de las palancas de CONTROL sobre los targets ganadores (PLM v4, theta con IC95%)
y predictibilidad OOF (estado solo vs estado + control), para la defensa "el recomendador puede moverlos".

Targets: sdi (w* de st_06), ln_feo_ret, ln_sn_dep (Reduccion); irf_next y d_irf = irf_next - indice_irf_prev (Fusion).
Palancas: primitivas [carbon, GN, O2, aire] y variante cinetica [Cx_sn_v4 / relacion_C_Sn_carga, GN, exceso O2, aire].
Salidas: st_07_theta_ganadores.csv, st_07_predictibilidad_ganadores.csv, st_07_log.txt
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parent.parent))
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
t0 = time.time()
df = pd.read_pickle(AQUI / "cache_df_st.pkl").sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
df = mp4.agregar_columnas_v4(df)
tg = pd.read_csv(AQUI / "st_06_targets_escalon.csv", parse_dates=["fecha_inicio"])
for c in ["ln_sn_dep", "ln_feo_ret", "sdi", "irf_next"]:
    df[c] = tg[c].to_numpy()
df["d_irf"] = df["irf_next"] - df["indice_irf_prev"]
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
df = df[~df.es_primer_escalon_batch]

PRIM = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"]
CIN = {"Reducción": ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
       "Fusión": ["relacion_C_Sn_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]}
ESTADO = {"Reducción": mp4.ESTADO_V4[("Reducción", "sn_kg")], "Fusión": mp4.ESTADO_V4[("Fusión", "sn_kg")]}
# signo teorico esperado (mayor target = mejor): ver README
SIGNO = {
    ("sdi", "tasa_feed_Carbon_kg_min"): 0, ("sdi", "tasa_gn_nm3_min"): -1, ("sdi", "tasa_o2_nm3_min"): +1, ("sdi", "tasa_aire_nm3_min"): +1,
    ("ln_feo_ret", "tasa_feed_Carbon_kg_min"): -1, ("ln_feo_ret", "tasa_gn_nm3_min"): -1, ("ln_feo_ret", "tasa_o2_nm3_min"): +1, ("ln_feo_ret", "tasa_aire_nm3_min"): +1,
    ("ln_sn_dep", "tasa_feed_Carbon_kg_min"): +1, ("ln_sn_dep", "tasa_gn_nm3_min"): +1, ("ln_sn_dep", "tasa_o2_nm3_min"): -1, ("ln_sn_dep", "tasa_aire_nm3_min"): -1,
    ("irf_next", "tasa_feed_Carbon_kg_min"): -1, ("irf_next", "tasa_gn_nm3_min"): -1, ("irf_next", "tasa_o2_nm3_min"): +1, ("irf_next", "tasa_aire_nm3_min"): +1,
    ("d_irf", "tasa_feed_Carbon_kg_min"): -1, ("d_irf", "tasa_gn_nm3_min"): -1, ("d_irf", "tasa_o2_nm3_min"): +1, ("d_irf", "tasa_aire_nm3_min"): +1,
}
def hgb():
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=42)

filas_theta, filas_pred = [], []
for fase, targets in [("Reducción", ["sdi", "ln_feo_ret", "ln_sn_dep"]), ("Fusión", ["irf_next", "d_irf"])]:
    S = ESTADO[fase]
    for tgt in targets:
        d_all = df[(df.fase_proceso == fase)].dropna(subset=S + PRIM + CIN[fase] + [tgt]).copy()
        lo, hi = d_all.loc[d_all.Batch.isin(batches_dev), tgt].quantile([0.005, 0.995]); d_all[tgt] = d_all[tgt].clip(lo, hi)
        d_dev = d_all[d_all.Batch.isin(batches_dev)].reset_index(drop=True); d_lb = d_all[d_all.Batch.isin(batches_lockbox)]
        # predictibilidad OOF: estado solo (A) vs estado + control (B)
        for nombre, feats in [("A_estado", S), ("B_estado+control", S + PRIM + ["exceso_o2_combustion_pct", "Cx_sn_v4", "Cx_av_v4"]), ("C_control", PRIM + ["exceso_o2_combustion_pct", "Cx_sn_v4", "Cx_av_v4"])]:
            oof = np.full(len(d_dev), np.nan)
            for tr, te in GroupKFold(5).split(d_dev, d_dev[tgt], d_dev.Batch):
                oof[te] = hgb().fit(d_dev.iloc[tr][feats], d_dev.iloc[tr][tgt]).predict(d_dev.iloc[te][feats])
            m = hgb().fit(d_dev[feats], d_dev[tgt]); pl = m.predict(d_lb[feats])
            filas_pred.append(dict(fase=fase, target=tgt, set=nombre, n_dev=len(d_dev), n_lockbox=len(d_lb), sd_dev=d_dev[tgt].std(),
                                   r2_oof=r2_score(d_dev[tgt], oof), mae_oof=mean_absolute_error(d_dev[tgt], oof), r2_lockbox=r2_score(d_lb[tgt], pl), mae_lockbox=mean_absolute_error(d_lb[tgt], pl)))
            log(f"{fase} {tgt:10s} {nombre:18s} n={len(d_dev)} R2oof={filas_pred[-1]['r2_oof']:.3f} R2lb={filas_pred[-1]['r2_lockbox']:.3f} sd={d_dev[tgt].std():.3f}")
        for variante, pal in [("primitivas", PRIM), ("cinetica", CIN[fase])]:
            m = mp4.ModeloPLM(estado=S, palancas=pal, target=tgt).fit(d_dev, n_boot=300)
            t = m.tabla_theta()
            for p_, r in t.iterrows():
                s_teo = SIGNO.get((tgt, p_), np.nan)
                filas_theta.append(dict(fase=fase, target=tgt, variante=variante, palanca=p_, theta_unidad=r.theta_unidad, theta_1sd=r.theta_1sd, ci_lo_1sd=r.ci_lo_1sd, ci_hi_1sd=r.ci_hi_1sd,
                                        theta_1sd_rel=r.theta_1sd / d_dev[tgt].std(), significativo=bool(r.significativo), signo_teorico=s_teo,
                                        concuerda=(np.sign(r.theta_1sd) == s_teo) if s_teo in (-1, 1) else np.nan, r2_oof_interno=m.r2_oof_interno, n=len(d_dev)))
            log(f"  PLM {variante:10s} R2oof_int={m.r2_oof_interno:.3f}: " + "; ".join(f"{p_} {r.theta_1sd:+.3f} [{r.ci_lo_1sd:+.3f},{r.ci_hi_1sd:+.3f}]{'*' if r.significativo else ''}" for p_, r in t.iterrows()))
pd.DataFrame(filas_theta).to_csv(AQUI / "st_07_theta_ganadores.csv", index=False)
pd.DataFrame(filas_pred).to_csv(AQUI / "st_07_predictibilidad_ganadores.csv", index=False)
log(f"tiempo {time.time()-t0:.0f}s")
(AQUI / "st_07_log.txt").write_text("\n".join(LOG), encoding="utf-8")
