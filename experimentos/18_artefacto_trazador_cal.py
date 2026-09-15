"""Experimento 18 -- El efecto de la cal en Fusion: fisica o artefacto del trazador CaO?
Cuanto del R2 de los targets en kg es reversion de ruido de medicion?

Ver experimentos/ITERACION_4_diseno.md y hallazgos.md 14.4/14.6. Todo el codigo vive aqui;
no se modifica ningun modulo de la raiz. Ejecutar en primer plano, sin joblib/multiprocessing:

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/18_artefacto_trazador_cal.py

Salidas: 18_balance_cao.csv, 18_dml_rezago_cal.csv, 18_batch_cal.csv, 18_ruido_reversion.csv,
figuras en figs_18/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

FIGS = Path(__file__).resolve().parent / "figs_18"
FIGS.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def boot_ci_by_batch(values: pd.Series, batches: pd.Series, n_boot: int = 500, seed: int = 0):
    """Media + IC95% bootstrap cluster por batch de una serie (para razones/medias por tercil)."""
    df = pd.DataFrame({"v": values.to_numpy(), "b": batches.to_numpy()})
    df = df.dropna()
    uniq = df["b"].unique()
    rng = np.random.default_rng(seed)
    means = []
    grp = {b: g["v"].to_numpy() for b, g in df.groupby("b")}
    for _ in range(n_boot):
        sample_batches = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([grp[b] for b in sample_batches])
        means.append(vals.mean())
    means = np.array(means)
    return df["v"].mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), len(df)


# =============================================================================
# 0) Dataset
# =============================================================================
log("cargando dataset v3 ...")
df = pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl")
df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
dev_batches, lockbox_batches = mp.split_dev_lockbox(df)
log(f"df: {df.shape}, batches dev={len(dev_batches)} lockbox={len(lockbox_batches)}")

fus = df["fase_proceso"] == "Fusión"
red = df["fase_proceso"] == "Reducción"
# Escalones de Fusion con estado _prev valido (orden_escalon_fase >= 1, excluye primer escalon del batch)
fus_ok = fus & (df["orden_escalon_fase"] >= 1) & df["masa_escoria_est_kg_prev"].notna()

# =============================================================================
# 1) Balance de CaO nominal vs medido
# =============================================================================
log("=== Parte 1: balance CaO nominal vs medido ===")

cum_cao_t = df["cum_feed_CaO_kg_prev"] + df["feed_CaO_kgh"]
masa_esperada = df["masa_escoria_est_kg_prev"] + df["feed_total_kgh"].fillna(0.0) + df["feed_CaO_kgh"].fillna(0.0)
cao_esperado_pct = 100 * cum_cao_t / masa_esperada
d_cao_esperado = cao_esperado_pct - df["ley_cao_escoria_pct_prev"]
d_cao_medido = df["d_ley_cao_escoria_pct"]  # ya existe: ley[t]-ley[t-1]

# version corregida: resta el Sn extraido del escalon (sale de la escoria)
masa_esperada_corr = masa_esperada - df["sn_extraido_est_kg"].fillna(0.0)
cao_esperado_pct_corr = 100 * cum_cao_t / masa_esperada_corr
d_cao_esperado_corr = cao_esperado_pct_corr - df["ley_cao_escoria_pct_prev"]

sub1 = pd.DataFrame({
    "Batch": df["Batch"], "orden_escalon_fase": df["orden_escalon_fase"],
    "tasa_feed_CaO_kg_min": df["tasa_feed_CaO_kg_min"],
    "d_cao_medido": d_cao_medido, "d_cao_esperado": d_cao_esperado,
    "d_cao_esperado_corr": d_cao_esperado_corr,
}).loc[fus_ok]
sub1 = sub1.replace([np.inf, -np.inf], np.nan)
# razon valida solo si el esperado es positivo y de magnitud razonable (evita dividir por ~0)
sub1_valida = sub1.dropna(subset=["d_cao_medido", "d_cao_esperado", "tasa_feed_CaO_kg_min"])
sub1_valida = sub1_valida.loc[sub1_valida["d_cao_esperado"].abs() > 0.02]
sub1_valida["razon_medido_esperado"] = sub1_valida["d_cao_medido"] / sub1_valida["d_cao_esperado"]
sub1_valida["razon_medido_esperado_corr"] = sub1_valida["d_cao_medido"] / sub1["d_cao_esperado_corr"].reindex(sub1_valida.index)
# gana un tope robusto (evita outliers por denominador chico)
razon_clip = sub1_valida["razon_medido_esperado"].clip(-5, 5)

log(f"n valido balance CaO: {len(sub1_valida)} de {fus_ok.sum()} escalones Fusion orden>=1")

terciles = pd.qcut(sub1_valida["tasa_feed_CaO_kg_min"], 3, labels=["T1_bajo", "T2_medio", "T3_alto"])
filas_balance = []
for t in ["T1_bajo", "T2_medio", "T3_alto"]:
    m = terciles == t
    med_medido, lo_m, hi_m, n = boot_ci_by_batch(sub1_valida.loc[m, "d_cao_medido"], sub1_valida.loc[m, "Batch"])
    med_esp, lo_e, hi_e, _ = boot_ci_by_batch(sub1_valida.loc[m, "d_cao_esperado"], sub1_valida.loc[m, "Batch"])
    r_mean, r_lo, r_hi, _ = boot_ci_by_batch(razon_clip.loc[m], sub1_valida.loc[m, "Batch"])
    filas_balance.append({
        "tercil_tasa_cao": t, "n": n,
        "cal_kg_min_media": sub1_valida.loc[m, "tasa_feed_CaO_kg_min"].mean(),
        "d_cao_medido_media": med_medido, "d_cao_medido_ci_lo": lo_m, "d_cao_medido_ci_hi": hi_m,
        "d_cao_esperado_media": med_esp, "d_cao_esperado_ci_lo": lo_e, "d_cao_esperado_ci_hi": hi_e,
        "razon_medido_esperado_media": r_mean, "razon_ci_lo": r_lo, "razon_ci_hi": r_hi,
    })
tabla_balance = pd.DataFrame(filas_balance)
log(tabla_balance.to_string(index=False))

# rebote: d%CaO[t+1] vs cal[t], controlando cal[t+1] y masa previa (OLS HC3)
df_sorted = df.copy()
cal_t = df_sorted["tasa_feed_CaO_kg_min"]
cal_tp1 = df_sorted.groupby("Batch")["tasa_feed_CaO_kg_min"].shift(-1)
dcao_tp1 = df_sorted.groupby("Batch")["d_ley_cao_escoria_pct"].shift(-1)
fase_tp1 = df_sorted.groupby("Batch")["fase_proceso"].shift(-1)
rebote = pd.DataFrame({
    "Batch": df_sorted["Batch"], "cal_t": cal_t, "cal_tp1": cal_tp1, "dcao_tp1": dcao_tp1,
    "fase": df_sorted["fase_proceso"], "fase_tp1": fase_tp1,
}).loc[fus_ok]
rebote = rebote.loc[rebote["fase_tp1"] == "Fusión"].dropna(subset=["cal_t", "cal_tp1", "dcao_tp1"])
Xr = sm.add_constant(rebote[["cal_t", "cal_tp1"]])
mod_rebote = sm.OLS(rebote["dcao_tp1"], Xr).fit(cov_type="HC3")
log("Rebote d%CaO[t+1] ~ cal[t] + cal[t+1] (HC3):")
log(mod_rebote.summary().tables[1].as_text())
coef_rebote = mod_rebote.params.get("cal_t", np.nan)
ci_rebote = mod_rebote.conf_int().loc["cal_t"].tolist() if "cal_t" in mod_rebote.params.index else [np.nan, np.nan]
log(f"coef cal[t] sobre d%CaO[t+1]: {coef_rebote:.5f} IC95% [{ci_rebote[0]:.5f}, {ci_rebote[1]:.5f}] n={len(rebote)}")

tabla_balance["rebote_coef_cal_t_sobre_dcao_tp1"] = coef_rebote
tabla_balance["rebote_ci_lo"] = ci_rebote[0]
tabla_balance["rebote_ci_hi"] = ci_rebote[1]
tabla_balance["rebote_n"] = len(rebote)
tabla_balance.to_csv(Path(__file__).parent / "18_balance_cao.csv", index=False)
log("guardado 18_balance_cao.csv")

# figura
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, tabla_balance["d_cao_medido_media"], w, yerr=[
        tabla_balance["d_cao_medido_media"] - tabla_balance["d_cao_medido_ci_lo"],
        tabla_balance["d_cao_medido_ci_hi"] - tabla_balance["d_cao_medido_media"]], label="Delta%CaO medido", capsize=3)
    ax.bar(x + w / 2, tabla_balance["d_cao_esperado_media"], w, yerr=[
        tabla_balance["d_cao_esperado_media"] - tabla_balance["d_cao_esperado_ci_lo"],
        tabla_balance["d_cao_esperado_ci_hi"] - tabla_balance["d_cao_esperado_media"]], label="Delta%CaO esperado (nominal)", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(tabla_balance["tercil_tasa_cao"])
    ax.set_ylabel("puntos de %CaO")
    ax.set_title("Delta%CaO medido vs esperado por tercil de dosis de cal (Fusion)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "1_balance_cao_terciles.png", dpi=130)
    plt.close(fig)
    log("figura 1 guardada")
except Exception as e:
    log("figura 1 fallo:", repr(e))

# =============================================================================
# 2) DML con rezago (reusa mp3.efectos_marginales_palancas_v3, target_col arbitrario)
# =============================================================================
log("=== Parte 2: DML con rezago sobre la cal en Fusion ===")

df2 = df.copy()
# (b) sn_extraido[t+1], solo si el siguiente escalon tambien es Fusion (mismo batch)
sn_tp1 = df2.groupby("Batch")["sn_extraido_est_kg"].shift(-1)
fase_next = df2.groupby("Batch")["fase_proceso"].shift(-1)
sn_tp1 = sn_tp1.where(fase_next == "Fusión")
df2["sn_extraido_est_kg_tp1"] = sn_tp1
# (c) suma t + t+1
df2["sn_extraido_suma_t_tp1"] = df2["sn_extraido_est_kg"] + df2["sn_extraido_est_kg_tp1"]
# (e) masa extra no explicada por la carga nominal del escalon
df2["masa_extra_no_explicada_kg"] = df2["d_masa_escoria_est_kg"] - (
    df2["feed_total_kgh"].fillna(0.0) + df2["feed_CaO_kgh"].fillna(0.0))

resultados_dml = []
targets_dml = [
    ("sn_extraido_est_kg", "sn_kg[t]"),
    ("sn_extraido_est_kg_tp1", "sn_kg[t+1]"),
    ("sn_extraido_suma_t_tp1", "sn_kg[t]+sn_kg[t+1]"),
    ("d_ley_sn_escoria_pct", "d_ley_sn_pct[t]"),
    ("masa_extra_no_explicada_kg", "masa_extra[t]"),
]
for col, etiqueta in targets_dml:
    log(f"DML conjunto Fusion -> {etiqueta} ({col}) ...")
    try:
        out = mp3.efectos_marginales_palancas_v3(df2, "Fusión", target_col=col, n_boot=150, seed=0)
        out = out.reset_index()
        out["target_etiqueta"] = etiqueta
        resultados_dml.append(out)
        fila_cal = out.loc[out["palanca"] == "tasa_feed_CaO_kg_min"]
        if not fila_cal.empty:
            r = fila_cal.iloc[0]
            log(f"  cal: theta={r['theta_por_unidad']:.3f} efecto/1sd={r['efecto_por_1sd']:.1f} "
                f"IC95%=[{r['efecto_1sd_ci95_lo']:.1f}, {r['efecto_1sd_ci95_hi']:.1f}] n={r['n']} sig={r['significativo_ci95']}")
    except Exception as e:
        log(f"  FALLO en {etiqueta}: {repr(e)}")

tabla_dml = pd.concat(resultados_dml, ignore_index=True) if resultados_dml else pd.DataFrame()
tabla_dml.to_csv(Path(__file__).parent / "18_dml_rezago_cal.csv", index=False)
log("guardado 18_dml_rezago_cal.csv")

# figura: efecto de la cal en t, t+1 y suma
try:
    cal_rows = tabla_dml.loc[(tabla_dml["palanca"] == "tasa_feed_CaO_kg_min") &
                              (tabla_dml["target_etiqueta"].isin(["sn_kg[t]", "sn_kg[t+1]", "sn_kg[t]+sn_kg[t+1]"]))]
    if not cal_rows.empty:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = np.arange(len(cal_rows))
        vals = cal_rows["efecto_por_1sd"].to_numpy()
        lo = vals - cal_rows["efecto_1sd_ci95_lo"].to_numpy()
        hi = cal_rows["efecto_1sd_ci95_hi"].to_numpy() - vals
        ax.bar(x, vals, yerr=[lo, hi], capsize=4, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cal_rows["target_etiqueta"])
        ax.set_ylabel("kg de Sn extraido por +1 sd de tasa_feed_CaO_kg_min")
        ax.set_title("Efecto DML de la cal: escalon t, t+1 y suma (Fusion)")
        fig.tight_layout()
        fig.savefig(FIGS / "2_efecto_cal_rezago.png", dpi=130)
        plt.close(fig)
        log("figura 2 guardada")
except Exception as e:
    log("figura 2 fallo:", repr(e))

# =============================================================================
# 3) Batch-level: cal en Fusion vs KPI del batch
# =============================================================================
log("=== Parte 3: efecto de la cal a nivel de batch ===")

g_fus = df.loc[fus].groupby("Batch")
g_all = df.groupby("Batch")

batch = pd.DataFrame(index=g_all.groups.keys())
batch["cal_total_F"] = g_fus["feed_CaO_kgh"].sum()
feed_total_F = g_fus["feed_total_kgh"].sum()
batch["cal_por_carga_F"] = g_fus["feed_CaO_kgh"].sum() / feed_total_F
batch["Sn_extraido_total_F"] = g_fus["sn_extraido_est_kg"].sum(min_count=1)
feed_sn_F = g_fus["feed_Sn_kgf"].sum(min_count=1)


def _ultimo_valor(g, col):
    gg = g.sort_values("fecha_inicio")
    return gg[col].iloc[-1] if len(gg) else np.nan


sn_inv_finF = df.loc[fus].sort_values(["Batch", "fecha_inicio"]).groupby("Batch")["sn_inventario_escoria_est_kg"].last()
batch["sn_inv_fin_F"] = sn_inv_finF
batch["feed_sn_F"] = feed_sn_F
batch["F_avance_final"] = 1 - batch["sn_inv_fin_F"] / batch["feed_sn_F"]

batch["sn_metal_t"] = g_all["sn_en_metal_crudo_batch_t"].first()
batch["sn_dross_t"] = g_all["sn_en_dross_fe_batch_t"].first()
batch["sn_polvo_t"] = g_all["sn_en_polvo_fundicion_batch_t"].first()
tot_out = batch["sn_metal_t"] + batch["sn_dross_t"] + batch["sn_polvo_t"]
batch["f_metal"] = batch["sn_metal_t"] / tot_out
batch["f_dross"] = batch["sn_dross_t"] / tot_out
batch["f_polvo"] = batch["sn_polvo_t"] / tot_out
batch["rendimiento_proxy_batch"] = g_all["rendimiento_proxy_batch"].first()

batch["feed_sn_total_batch"] = g_all["feed_Sn_kgf"].sum(min_count=1)
batch["ley_sn_conc_batch_pct"] = g_all["ley_sn_conc_batch_pct"].mean()
batch["espesor_ladrillo_norm_mm"] = g_all["espesor_ladrillo_norm_mm"].mean()
batch["temp_media_F"] = g_fus["temperatura_horno_celsius"].mean()
batch["carbon_total_F"] = g_fus["feed_Carbon_kgh"].sum(min_count=1)
batch["gn_total_F"] = g_fus["volumen_gas_natural_inyectado_lanza_escalon_nm3"].sum(min_count=1)
batch["n_escalones_F"] = g_fus.size()

batch = batch.replace([np.inf, -np.inf], np.nan)
batch.index.name = "Batch"
batch = batch.reset_index()
n_batches_total = len(batch)
log(f"batches construidos: {n_batches_total}")

controles = ["feed_sn_total_batch", "ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm",
             "temp_media_F", "carbon_total_F", "gn_total_F"]


def ols_hc3(dep: str, palanca: str, extra_controles=None, tabla=None):
    ctrl = list(controles) if extra_controles is None else extra_controles
    cols = [dep, palanca] + ctrl
    d = tabla.dropna(subset=cols).copy()
    X = sm.add_constant(d[[palanca] + ctrl])
    y = d[dep]
    mod = sm.OLS(y, X).fit(cov_type="HC3")
    ci = mod.conf_int().loc[palanca].tolist()
    rho, p_rho = stats.spearmanr(d[dep], d[palanca])
    return {
        "dependiente": dep, "palanca": palanca, "n": len(d),
        "coef": mod.params[palanca], "se_hc3": mod.bse[palanca], "p_valor": mod.pvalues[palanca],
        "ci95_lo": ci[0], "ci95_hi": ci[1], "r2": mod.rsquared,
        "spearman_rho": rho, "spearman_p": p_rho,
    }


filas_ols = []
filas_ols.append(ols_hc3("Sn_extraido_total_F", "cal_total_F", tabla=batch))
filas_ols.append(ols_hc3("F_avance_final", "cal_por_carga_F", tabla=batch))
filas_ols.append(ols_hc3("f_metal", "cal_por_carga_F", tabla=batch))
filas_ols.append(ols_hc3("f_dross", "cal_por_carga_F", tabla=batch))
filas_ols.append(ols_hc3("rendimiento_proxy_batch", "cal_por_carga_F", tabla=batch))
tabla_ols_batch = pd.DataFrame(filas_ols)
log(tabla_ols_batch.to_string(index=False))
tabla_ols_batch.to_csv(Path(__file__).parent / "18_batch_cal.csv", index=False)
batch.to_csv(Path(__file__).parent / "18_batch_cal_dataset.csv", index=False)
log("guardado 18_batch_cal.csv y 18_batch_cal_dataset.csv")

try:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d = batch.dropna(subset=["cal_por_carga_F", "f_metal"])
    ax.scatter(d["cal_por_carga_F"], d["f_metal"], alpha=0.5, s=18)
    if len(d) > 2:
        b1, b0 = np.polyfit(d["cal_por_carga_F"], d["f_metal"], 1)
        xs = np.linspace(d["cal_por_carga_F"].min(), d["cal_por_carga_F"].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="firebrick", lw=2)
    ax.set_xlabel("cal / carga en Fusion (cal_por_carga_F)")
    ax.set_ylabel("fraccion de Sn a metal crudo (f_metal)")
    ax.set_title("Cal por carga en Fusion vs fraccion a metal (nivel batch)")
    fig.tight_layout()
    fig.savefig(FIGS / "3_cal_vs_fmetal_batch.png", dpi=130)
    plt.close(fig)
    log("figura 3 guardada")
except Exception as e:
    log("figura 3 fallo:", repr(e))

# =============================================================================
# 4) Reversion de ruido en targets en kg
# =============================================================================
log("=== Parte 4: reversion de ruido de medicion ===")


def oof_hgb_curado(fase: str, target_col: str, n_splits: int = 5, seed: int = 0):
    feats_key = "sn_kg" if target_col == mp3.TARGETS_V3[fase]["sn_kg"] else "feo_kg"
    feats = mp3.FEATURES_V3_POR_DEFECTO[fase][feats_key]
    df_base = mp.dataset_base_modelo(df)
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev_batches)]
    d = sub.dropna(subset=feats + [target_col]).reset_index(drop=False)
    modelo = mp3.construir_modelo_v3(feats, target_col, monotono=True)
    oof = np.full(len(d), np.nan)
    groups = d["Batch"].to_numpy()
    for tr, te in GroupKFold(n_splits).split(d[feats], d[target_col], groups):
        m = mp3.construir_modelo_v3(feats, target_col, monotono=True)
        m.fit(d.loc[tr, feats], d.loc[tr, target_col])
        oof[te] = m.predict(d.loc[te, feats])
    d["oof_pred"] = oof
    d["oof_resid"] = d[target_col] - d["oof_pred"]
    r2 = r2_score(d[target_col], d["oof_pred"])
    return d, r2


def autocorr_lag1_within_batch(d: pd.DataFrame, col: str, batch_col="Batch", orden_col="index"):
    """Correlacion de Pearson entre col[t] y col[t+1] para escalones consecutivos del mismo batch
    (usa el orden original de filas, ya viene ordenado por fecha_inicio dentro de Batch)."""
    d2 = d.sort_values([batch_col, orden_col]).reset_index(drop=True)
    nb = d2[batch_col].to_numpy()
    same_batch_next = nb[:-1] == nb[1:]
    x = d2[col].to_numpy()[:-1][same_batch_next]
    y = d2[col].to_numpy()[1:][same_batch_next]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p, int(mask.sum())


filas_ruido = []

log("OOF sn_kg Fusion (modelo curado v3) ...")
d_snF, r2_snF = oof_hgb_curado("Fusión", "sn_extraido_est_kg")
r, p, n = autocorr_lag1_within_batch(d_snF, "oof_resid")
filas_ruido.append({"item": "autocorr_resid_OOF_sn_kg_Fusion", "valor": r, "p_valor": p, "n_pares": n, "r2_oof": r2_snF})
log(f"  R2 OOF={r2_snF:.3f}, autocorr lag1 residuo OOF = {r:.3f} (p={p:.4g}, n={n})")

log("OOF sn_kg Reduccion (modelo curado v3) ...")
d_snR, r2_snR = oof_hgb_curado("Reducción", "sn_extraido_est_kg")
r, p, n = autocorr_lag1_within_batch(d_snR, "oof_resid")
filas_ruido.append({"item": "autocorr_resid_OOF_sn_kg_Reduccion", "valor": r, "p_valor": p, "n_pares": n, "r2_oof": r2_snR})
log(f"  R2 OOF={r2_snR:.3f}, autocorr lag1 residuo OOF = {r:.3f} (p={p:.4g}, n={n})")

log("OOF feo_kg Reduccion (modelo curado v3) ...")
d_feoR, r2_feoR = oof_hgb_curado("Reducción", "feo_extraido_est_kg")
r, p, n = autocorr_lag1_within_batch(d_feoR, "oof_resid")
filas_ruido.append({"item": "autocorr_resid_OOF_feo_kg_Reduccion", "valor": r, "p_valor": p, "n_pares": n, "r2_oof": r2_feoR})
log(f"  R2 OOF={r2_feoR:.3f}, autocorr lag1 residuo OOF = {r:.3f} (p={p:.4g}, n={n})")

# (b) autocorrelacion del propio target en Reduccion
d_red_all = df.loc[red & df["sn_extraido_est_kg"].notna()].copy()
r, p, n = autocorr_lag1_within_batch(d_red_all.reset_index(), "sn_extraido_est_kg", orden_col="index")
filas_ruido.append({"item": "autocorr_target_sn_extraido_kg_Reduccion", "valor": r, "p_valor": p, "n_pares": n, "r2_oof": np.nan})
log(f"  autocorr target sn_extraido_est_kg (Reduccion) = {r:.3f} (p={p:.4g}, n={n})")

d_red_feo = df.loc[red & df["feo_extraido_est_kg"].notna()].copy()
r, p, n = autocorr_lag1_within_batch(d_red_feo.reset_index(), "feo_extraido_est_kg", orden_col="index")
filas_ruido.append({"item": "autocorr_target_feo_extraido_kg_Reduccion", "valor": r, "p_valor": p, "n_pares": n, "r2_oof": np.nan})
log(f"  autocorr target feo_extraido_est_kg (Reduccion) = {r:.3f} (p={p:.4g}, n={n})")

# (c) baseline "solo inventario previo" vs regla fisica trivial vs modelo curado (Reduccion, sn_kg)
log("Comparando baseline / regla fisica trivial / modelo curado (Reduccion sn_kg) ...")
df_base = mp.dataset_base_modelo(df)
sub_r = df_base.loc[(df_base["fase_proceso"] == "Reducción") & df_base["Batch"].isin(dev_batches)]
cols_needed = ["sn_inventario_escoria_est_kg_prev", "sn_extraido_est_kg", "orden_escalon_fase", "Batch"]
d_r = sub_r.dropna(subset=cols_needed).reset_index(drop=True)
groups_r = d_r["Batch"].to_numpy()

oof_baseline = np.full(len(d_r), np.nan)
oof_fisico = np.full(len(d_r), np.nan)
for tr, te in GroupKFold(5).split(d_r, d_r["sn_extraido_est_kg"], groups_r):
    # baseline: regresion lineal simple solo con el inventario previo
    lin = LinearRegression()
    lin.fit(d_r.loc[tr, ["sn_inventario_escoria_est_kg_prev"]], d_r.loc[tr, "sn_extraido_est_kg"])
    oof_baseline[te] = lin.predict(d_r.loc[te, ["sn_inventario_escoria_est_kg_prev"]])
    # regla fisica trivial: fraccion media por orden_escalon_fase, calculada SOLO en train.
    # Se pondera por inventario (suma extraido / suma inventario) en vez de promediar el
    # cociente fila a fila: un cociente por fila explota cuando el inventario previo es chico
    # (frecuente en escalones tardios de Reduccion, ley_sn ya casi agotada).
    train = d_r.loc[tr]
    tabla_frac = (train.groupby("orden_escalon_fase")["sn_extraido_est_kg"].sum()
                  / train.groupby("orden_escalon_fase")["sn_inventario_escoria_est_kg_prev"].sum())
    frac_media_global = train["sn_extraido_est_kg"].sum() / train["sn_inventario_escoria_est_kg_prev"].sum()
    frac_pred = d_r.loc[te, "orden_escalon_fase"].map(tabla_frac).fillna(frac_media_global)
    oof_fisico[te] = d_r.loc[te, "sn_inventario_escoria_est_kg_prev"].to_numpy() * frac_pred.to_numpy()

r2_baseline = r2_score(d_r["sn_extraido_est_kg"], oof_baseline)
r2_fisico = r2_score(d_r["sn_extraido_est_kg"], oof_fisico)
r2_full_reduccion = r2_snR  # calculado arriba con el set curado de 10 features
log(f"  R2 OOF baseline (solo inventario_prev, lineal) = {r2_baseline:.3f}")
log(f"  R2 OOF regla fisica trivial (inv_prev x frac media por orden) = {r2_fisico:.3f}")
log(f"  R2 OOF modelo curado v3 (10 features) = {r2_full_reduccion:.3f}")
mejor_trivial = max(r2_baseline, r2_fisico)
r2_extra_palancas = r2_full_reduccion - mejor_trivial
log(f"  R2 adicional aportado por las palancas quimicas = {r2_extra_palancas:.4f}")

filas_ruido.append({"item": "R2_OOF_baseline_solo_inventario_prev_Reduccion", "valor": r2_baseline, "p_valor": np.nan,
                     "n_pares": len(d_r), "r2_oof": r2_baseline})
filas_ruido.append({"item": "R2_OOF_regla_fisica_trivial_Reduccion", "valor": r2_fisico, "p_valor": np.nan,
                     "n_pares": len(d_r), "r2_oof": r2_fisico})
filas_ruido.append({"item": "R2_OOF_modelo_curado_v3_Reduccion", "valor": r2_full_reduccion, "p_valor": np.nan,
                     "n_pares": len(d_r), "r2_oof": r2_full_reduccion})
filas_ruido.append({"item": "R2_extra_aportado_por_palancas_quimicas_Reduccion", "valor": r2_extra_palancas,
                     "p_valor": np.nan, "n_pares": len(d_r), "r2_oof": np.nan})

tabla_ruido = pd.DataFrame(filas_ruido)
tabla_ruido.to_csv(Path(__file__).parent / "18_ruido_reversion.csv", index=False)
log("guardado 18_ruido_reversion.csv")

log("=== FIN experimento 18 ===")
log(f"tiempo total: {time.time() - T0:.1f}s")
