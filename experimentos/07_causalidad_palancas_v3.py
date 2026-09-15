"""Causalidad (Double ML cross-fitted + refutation) de las PALANCAS nuevas v3, por fase.

Misma metodologia que modelo_predictivo_v2.dml_partialling_out (Robinson/partialling-out
con cross-fitting GroupKFold por Batch + bootstrap cluster por batch + placebo/confusor/
subset), pero (a) con el STATE enriquecido v3 (inventario por trazador CaO, refractario,
ensayos del batch) como controles, (b) evaluando las palancas de carga por tolva, cal,
duracion planificada y lanza, y (c) sobre dos targets: d_ley_sn_escoria_pct (puntos) y
sn_extraido_est_kg (kg de Sn extraido de la escoria, en masa). Nuisance learner:
HistGradientBoosting (mas rapido que XGBoost, misma capacidad no lineal).

Salida: experimentos/07_efectos_causales_v3.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import dataset_lingo_smelter as dls  # noqa: E402
import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

df = dls.construir_dataset(RAIZ / "Datos Lingo smelter fase II.xlsx", clean_mode=True)
df = fe.construir_features_v3(df)
df_base = mp.dataset_base_modelo(df)
batches_dev, _ = mp.split_dev_lockbox(df)

STATE_V3 = {
    "Fusión": ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "basicidad_B2_prev", "ley_sio2_escoria_pct_prev",
               "temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "espesor_ladrillo_norm_mm",
               "cum_feed_Sn_kg_prev", "cum_feed_CaO_kg_prev", "cum_feed_Carbon_kg_prev", "cum_feed_total_kg_prev",
               "masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev", "avance_reduccion_sn_prev",
               "ley_sn_conc_batch_pct", "ley_sn_carga_batch_pct", "orden_escalon_fase", "tiempo_fase",
               "posicion_vertical_lanza_mm_prev"],
    "Reducción": ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "basicidad_B2_prev",
                  "temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "espesor_ladrillo_norm_mm",
                  "cum_feed_Sn_kg_prev", "cum_feed_Carbon_kg_prev", "masa_escoria_est_kg_prev",
                  "sn_inventario_escoria_est_kg_prev", "avance_reduccion_sn_prev", "orden_escalon_fase", "tiempo_fase",
                  "posicion_vertical_lanza_mm_prev", "grad_temperatura_horno_celsius_prev"],
}
PALANCAS = {
    "Fusión": ["tasa_feed_Carbon_kg_min", "relacion_CaO_carga", "tasa_feed_total_kg_min", "frac_carga_secundaria",
               "tasa_feed_dross_Fe_kg_min", "tasa_feed_pellets_kg_min", "duracion_plan_min", "tasa_gn_nm3_min",
               "exceso_o2_combustion_pct", "posicion_vertical_lanza_mm"],
    "Reducción": ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "oxygen_enrichment_pct",
                  "duracion_plan_min", "posicion_vertical_lanza_mm"],
}
TARGETS = ["sn_extraido_est_kg", "d_ley_sn_escoria_pct", "d_ley_feo_escoria_pct"]
# d_ley_feo solo para las palancas redox (carbon, GN, O2, lanza)
PALANCAS_FEO = {"tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "oxygen_enrichment_pct", "posicion_vertical_lanza_mm", "duracion_plan_min"}

def _nuisance():
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=120, min_samples_leaf=20, l2_regularization=1.0, random_state=0)

def _oof(X, y, groups, seed=0):
    oof = np.full(len(X), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = _nuisance(); m.set_params(random_state=seed)
        m.fit(X.iloc[tr], y.iloc[tr]); oof[te] = m.predict(X.iloc[te])
    return oof

def _boot_ci(a_res, y_res, batches, n_boot=120, seed=0):
    rng = np.random.default_rng(seed); uniq = np.unique(batches)
    idx_by = {b: np.where(batches == b)[0] for b in uniq}; th = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
        if np.var(a_res[idx]) < 1e-12: continue
        th.append(np.cov(a_res[idx], y_res[idx], bias=True)[0, 1] / np.var(a_res[idx]))
    return float(np.percentile(th, 2.5)), float(np.percentile(th, 97.5))

def dml(sub, state, accion, target, seed=0):
    d = sub[list(dict.fromkeys(state + [accion, target, "Batch"]))].dropna(subset=[accion, target]).reset_index(drop=True)
    X = d[state]; g = d["Batch"].to_numpy()
    y_res = d[target].to_numpy() - _oof(X, d[target], g, seed)
    a_res = d[accion].to_numpy() - _oof(X, d[accion], g, seed)
    theta = float(np.cov(a_res, y_res, bias=True)[0, 1] / np.var(a_res))
    lo, hi = _boot_ci(a_res, y_res, g, seed=seed)
    return theta, lo, hi, len(d), d

filas = []
for fase in mp.FASES:
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)].copy()
    sub["sn_extraido_est_kg"] = sub["sn_extraido_est_kg"].where(sub["sn_extraido_est_kg"].between(*sub["sn_extraido_est_kg"].quantile([0.005, 0.995])))
    for target in TARGETS:
        for accion in PALANCAS[fase]:
            if sub[accion].notna().mean() < 0.5 or sub[accion].std() < 1e-9:
                continue
            if target == "d_ley_feo_escoria_pct" and accion not in PALANCAS_FEO:
                continue
            state = [s for s in STATE_V3[fase] if s != accion]
            theta, lo, hi, n, d = dml(sub, state, accion, target)
            # placebo
            rng = np.random.default_rng(1); plc = []
            for r in range(3):
                d2 = d.copy(); d2[accion] = rng.permutation(d2[accion].to_numpy())
                plc.append(dml(d2, state, accion, target, seed=10 + r)[0])
            plc = np.array(plc)
            # subset por batch
            bts = d["Batch"].unique(); rng2 = np.random.default_rng(2); subs = []
            for r in range(3):
                sb = rng2.choice(bts, size=int(0.7 * len(bts)), replace=False)
                subs.append(dml(d.loc[d["Batch"].isin(sb)], state, accion, target, seed=20 + r)[0])
            subs = np.array(subs)
            sd_a = float(d[accion].std()); sd_y = float(d[target].std())
            signif = (lo > 0) or (hi < 0)
            pasa_placebo = abs(theta) > 3 * plc.std() if plc.std() > 0 else abs(theta) > 0
            mismo_signo = float(np.mean(np.sign(subs) == np.sign(theta)))
            veredicto = "ROBUSTO" if (signif and pasa_placebo and mismo_signo >= 0.8) else ("SIGNIFICATIVO_MODERADO" if signif else "NO_SIGNIFICATIVO")
            fila = dict(fase=fase, target=target, accion=accion, n=n, theta_por_unidad=theta, ci95_lo=lo, ci95_hi=hi,
                        efecto_por_1sd_accion=theta * sd_a, efecto_por_1sd_en_sd_target=theta * sd_a / sd_y,
                        placebo_std=plc.std(), pasa_placebo_3sigma=bool(pasa_placebo), frac_subset_mismo_signo=mismo_signo, veredicto=veredicto)
            filas.append(fila)
            print(f"{fase:9s} | {target:22s} | {accion:28s} | theta={theta:+.4g} [{lo:+.3g},{hi:+.3g}] | 1sd->{theta*sd_a:+.3g} ({theta*sd_a/sd_y:+.2f} sd) | {veredicto}", flush=True)

res = pd.DataFrame(filas)
res.to_csv(RAIZ / "experimentos" / "07_efectos_causales_v3.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(res.round(4).to_string())
