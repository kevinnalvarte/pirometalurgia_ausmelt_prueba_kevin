"""Genera `win_analytics_v1.ipynb` (nbformat) con el modelo ganador v5 (iteracion 7). Ejecutar desde la raiz:
    python experimentos/build_notebook_win_v1.py
    jupyter nbconvert --to notebook --execute --inplace win_analytics_v1.ipynb --ExecutePreprocessor.timeout=3600
Las tablas pesadas (validacion, politica cross-fitted, evidencia) se leen de experimentos/v5/e7_0N_*.csv; RECALCULAR=True las reproduce.
"""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md(r"""# win_analytics_v1 — Prescriptor operacional por escalón del horno Ausmelt de Sn (modelo ganador v5)

**Objetivo.** Recomendar, escalón a escalón, la operación de las variables de CONTROL de la lanza y el reductor (tasa de carbón,
gas natural, O₂ y aire) dado el ESTADO observado al cierre del escalón anterior (leyes de escoria, inventario de Sn y FeO por
trazador CaO, matriz IRF, temperatura, avance de reducción) y el PLAN del batch (mezcla de carga, cal, duración), de modo que el
batch termine con **mayor recuperación de Sn a metal**.

**Qué es el modelo ganador (v5, `modelo_predictivo_v5.py`).** Un modelo *parcialmente lineal* (PLM) por fase y target,
`y = g(S) + Σ θ_j·(a_j − E[a_j|S])`, con el estado `S` en un gradient boosting sin restricciones y las palancas de CONTROL `a_j`
lineales con estructura cinética; los `θ_j` son el efecto causal-condicional de cada CONTROL con IC95 % (Double ML con
cross-fitting por batch). Lo nuevo de v5 frente a v4: los **targets por escalón están en escala logarítmica** (agotamiento del Sn
de la escoria y retención del FeO; adimensionales, telescópicos por batch, cancelan la masa del trazador) y el **objetivo está
anclado en el KPI refinado del batch** (recuperación con el Sn de la escoria final): `J_R = K·(ln_sn_dep + w·ln_feo_ret) − …`
con `w` y `K` estimados por regresión de batch, en vez de un λ fijado a mano.

**Cómo se llegó aquí.** v1→v3 (`hallazgos.md` §6b-§14), v4 (iteración 4, `experimentos/13-21`), búsqueda de targets (iteración 5,
`experimentos/search_targets`: SDI y ln IRF), balances y KPI refinado (iteración 6, `experimentos/balances`) e iteración 7
(`experimentos/v5`, diseño en `ITERACION_7_diseno.md`; ficha en `candidate_win_model.md`).

**Validación.** OOF GroupKFold(5) por batch sobre DEV (299 batches, abril-julio 2026) para elegir; lockbox = últimos 63 batches
cronológicos (agosto 2026, fin de campaña de refractario) sólo para reportar. Nada del lockbox se usó para elegir targets, w,
features ni pesos.""")

code(r"""import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm

import feature_engineering as fe
import modelo_prescriptivo as mp
import modelo_predictivo_v4 as mp4
import modelo_predictivo_v5 as mp5

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 120)
EXP = Path("experimentos/v5")
RECALCULAR = False   # True: recalcula tablas pesadas (validacion, cross-fitting) en vez de leer experimentos/v5/e7_05_*.csv

df = mp5.construir_dataset_modelo_v5("Datos Lingo smelter fase II.xlsx")
df_base = mp5._preparar_base(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
F, R = df_base["fase_proceso"] == "Fusión", df_base["fase_proceso"] == "Reducción"
print(f"escalones modelables {df_base.shape[0]} (Fusion {F.sum()}, Reduccion {R.sum()}) | batches DEV {len(batches_dev)} | lockbox {len(batches_lockbox)}")
print("Configuracion v5:", {k: v for k, v in mp5.CONFIG_V5.items() if k != 'beta_fusion'})""")

md(r"""## 1. Fundamento pirometalúrgico: qué se modela y por qué

**Batch = 7 escalones de Fusión + 4 de Reducción.** En Fusión entra la carga (concentrado ~40 % Sn, reciclos, mineral y dross de Fe,
pellets, cal, carbón) bajo una lanza oxidante y se forma la escoria FeO–SiO₂–CaO; ya se reduce buena parte del Sn. En Reducción
(sin carga) el carbón y la lanza sub-estequiométrica reducen el SnOₓ residual (`SnO₂ + 2C → Sn + 2CO`) compitiendo con
`FeO + C → Fe + CO`: el Fe metalizado forma hardhead/dross que arrastra Sn (pérdida 1), y la temperatura/arrastre generan polvo
(pérdida 2). La escoria final queda como *heel* químico del batch siguiente (§16.3 de `hallazgos.md`).

**Masa de escoria por trazador CaO** (la cal es inerte): `masa[t] = Σ_{k≤t} feed_CaO[k] / (%CaO[t]/100)`; `Sn_inv = masa·%Sn/100`,
`FeO_inv = masa·%FeO/100`; `avance_prev = 1 − Sn_inv[t−1]/ΣSn cargado`.

**Targets v5 por escalón (adimensionales, escala log):**

| Target | Fórmula | Sentido | Agregación a batch |
|---|---|---|---|
| `v5_ln_sn_dep` | `ln(Sn_disp / Sn_inv[t])`, `Sn_disp = Sn_inv[t−1] + Sn alimentado` | agotamiento del Sn de la escoria; cinética de 1er orden (= k·Δt) | suma telescópica = `ln(Sn_ini/Sn_fin)` |
| `v5_ln_feo_ret` | `ln(FeO_inv[t] / FeO_inv[t−1])` (Reducción) | retención del FeO (≤ 0 si se metaliza Fe: sobre-reducción) | suma telescópica = `ln(FeO_fin/FeO_ini)` |
| `v5_lnirf` | `ln(%FeO/(%SiO₂+%Al₂O₃+%CaO))` | matriz fayalítica fluida (índice IRF de planta) | media ponderada por tiempo / valor inicial (heel) |

La masa del trazador se cancela en cada cociente (sólo quedan leyes y el cociente de %CaO), por eso son menos ruidosos que los
targets en kg y aditivos por batch. El **KPI del batch** es `recuperacion_refinada_pct = 100·M/(M+D+P+Sn_escoria_final)`
(`feature_engineering.construir_resumen_batch_balance_global`), que a diferencia del proxy descuenta el Sn que queda en la escoria.

**Roles.** STATE: leyes, matriz, temperatura, lanza y tiro previos, acumulados por corriente, inventarios y avance. CONTEXT:
posición del escalón, tiempo de fase, refractario, leyes del batch. CONTROL: `tasa_feed_Carbon_kg_min`, `tasa_gn_nm3_min`,
`tasa_o2_nm3_min`, `tasa_aire_nm3_min` (→ exceso de O₂ = potencial redox de la lanza; carbón × Sn disponible = cinética
bimolecular). PLAN (no se optimiza): mezcla de carga, cal, duración (fija por protocolo en Reducción).""")

code(r"""for fase in mp5.FASES:
    print(f"=== {fase} ===")
    for clave, tgt in mp5.TARGETS_V5[fase].items():
        S, A = mp5.ESTADO_V5[(fase, clave)], mp5.PALANCAS_V5[(fase, clave)]
        print(f"  target {tgt:30s} | estado g(S): {len(S):2d} vars | palancas theta: {A}")
    print("  CONTROL optimizable:", mp5.ACCIONES_OPTIMIZABLES_V5[fase])
print()
print(df_base.loc[R, ["v5_ln_sn_dep", "v5_ln_feo_ret", "v5_lnirf"]].describe().round(3).T)
print(df_base.loc[F, ["v5_ln_sn_dep", "v5_lnirf"]].describe().round(3).T)""")

md(r"""### 1.1 Por qué estos targets: relación demostrada con el rendimiento del batch (iteraciones 5-7)

El objetivo v4 (`Sn − 0.6·FeO` en kg) no correlaciona con el KPI (ρ 0.035). Las componentes log sí, y su suma ponderada
(`SDI = Σln_sn_dep + w·Σln_feo_ret`) es el mejor correlato de un solo término del rendimiento; `w` se ancla en la regresión del
KPI refinado sobre las dos componentes (E7-01): ambas positivas y significativas.""")

code(r"""import sys; sys.path.insert(0, str(EXP))
import v5_lib as v5
batch = v5.construir_batch_v5(df).reset_index()
w = mp5.CONFIG_V5["w_sdi"]; batch["R_sdi_w"] = batch.R_sum_ln_sn_dep + w * batch.R_sum_ln_feo_ret
filas = []
for col in ["R_sum_ln_sn_dep", "R_sum_ln_feo_ret", "R_sdi_w10", "R_sdi_w", "R_lnirf_twmean", "F_lnirf_F0", "F_sum_ln_sn_dep"]:
    r = v5.bateria_batch(batch, col); filas.append({k: r.get(k) for k in ["variable", "rho_dev", "p_dev", "ci_lo_dev", "ci_hi_dev", "rho_lockbox", "p_lockbox", "rho_total", "ols_dev", "ols_p_dev", "quintiles_crecientes", "dQ5Q1"]})
tab = pd.DataFrame(filas).round(3); display(tab)
anc = pd.read_csv(EXP / "e7_01_w_anclaje.csv"); display(anc.round(3))
sw = pd.read_csv(EXP / "e7_01_w_sweep.csv")
fig, ax = plt.subplots(figsize=(8, 4))
s = sw[sw.variante == "simple"] if "variante" in sw.columns else sw
for split, col in [("dev", "#1f77b4"), ("lockbox", "#ff7f0e"), ("total", "#2ca02c")]:
    c = f"rho_{split}"
    if c in s.columns: ax.plot(s["w"], s[c], "o-", label=split, color=col)
ax.axvline(w, color="k", ls="--", lw=1, label=f"w anclado = {w:.1f}"); ax.set_xlabel("w (peso del FeO retenido)"); ax.set_ylabel("ρ Spearman con recuperación refinada")
ax.set_title("Barrido de w: el anclaje en el KPI (OLS) cae en la meseta de DEV"); ax.legend(); plt.show()""")

md(r"""**Lectura.** Con el KPI refinado ambas componentes importan (+2.1 pp por unidad de agotamiento de Sn, p 0.001; +12.8 pp por
unidad de retención de FeO, p < 0.001 → w ≈ 6, IC bootstrap [3.8, 13.6]). La componente FeO es la que manda en DEV (campaña
media) y la de Sn la que replica en el lockbox (fin de campaña, horno frío): por eso el objetivo lleva las dos. Los términos
de Fusión que sobreviven al control del heel son sólo la **conservación del Sn en Fusión** (−0.45 pp por unidad, p 0.08 DEV;
−0.59, p 0.014 total): más Sn extraído en Fusión → menos recuperación (principio "fundir oxidando, reducir después").""")

md(r"""## 2. Construcción del modelo ganador y efecto de las variables de CONTROL

`mp5.entrenar_sistema_v5` entrena, sólo con DEV, un PLM por (fase, target): `g(S)` y `E[a_j|S]` con HistGradientBoosting
(max_depth 4, 150 iteraciones, min_samples_leaf 20) y `θ` por OLS sobre residuos out-of-fold (GroupKFold por batch, 5 folds);
IC95 % por bootstrap cluster por batch (300 réplicas). Además: soporte histórico kNN sobre estado + palancas, límites P1-P99 de
cada palanca y ventana térmica P5-P95.""")

code(r"""sistema = mp5.entrenar_sistema_v5(df, n_boot=300)
efectos = mp5.tabla_efectos_control_v5(sistema)
display(efectos[["fase", "clave", "target", "palanca", "theta_unidad", "theta_1sd", "ci_lo_1sd", "ci_hi_1sd", "significativo", "signo_teorico", "concuerda"]].round(4))""")

code(r"""fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
for ax, (fase, clave, titulo) in zip(axes, [("Reducción", "sn_dep", "Reducción: agotamiento de Sn [ln]"), ("Reducción", "feo_ret", "Reducción: retención de FeO [ln]"),
                                         ("Reducción", "dT", "Reducción: ΔT [°C]"), ("Fusión", "sn_dep", "Fusión: agotamiento de Sn [ln] (conservar)")]):
    e = efectos[(efectos.fase == fase) & (efectos.clave == clave)]
    y = np.arange(len(e))
    ax.barh(y, e["theta_1sd"], xerr=[e["theta_1sd"] - e["ci_lo_1sd"], e["ci_hi_1sd"] - e["theta_1sd"]], color=np.where(e["significativo"], "#1f77b4", "#b0b0b0"), capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(e["palanca"]); ax.axvline(0, color="k", lw=0.8); ax.set_title(titulo, fontsize=10); ax.set_xlabel("efecto por +1 sd de la palanca, IC95%")
plt.tight_layout(); plt.show()""")

md(r"""**Lectura.** En Reducción el carbón × Sn disponible (`Cx_sn_v4`, cinética bimolecular) sube el agotamiento de Sn de forma
significativa y tiene efecto ≈ 0 sobre la retención de FeO: **el carbón es la palanca selectiva**, y sólo se identifica cuando el
estado incluye la historia de dosificación (acumulados de carbón y la relación C/Sn acumulada: con el estado curado su IC
cruza 0, E7-01/E7-04). El gas natural sube el agotamiento (calor + gas reductor) pero **reduce FeO** (no selectivo:
−0.021 sd/sd, IC [−0.037, −0.005]); en el objetivo su efecto neto es negativo (w = 6 pesa la pérdida de FeO). El exceso de O₂ y el
aire tienen efectos pequeños con IC que cruzan 0, del signo teórico (lanza oxidante retiene FeO). En ΔT de Reducción el GN calienta
(+2.3 °C/sd) y el O₂ enfría la cara interior (−3.7 °C/sd). En Fusión los gases de lanza (GN, exceso de O₂) extraen Sn de la
escoria (θ > 0 significativos): con un objetivo de conservación el prescriptor los recorta dentro de la ventana térmica.""")

md(r"""## 3. Métricas de predicción (OOF GroupKFold por batch en DEV; lockbox sólo reporte)""")

code(r"""tabla = mp5.tabla_validacion_v5(df) if RECALCULAR else pd.read_csv(EXP / "e7_05_tabla_validacion_v5.csv")
display(tabla[["fase", "target", "columna", "forma", "n_estado", "n_palancas", "n_dev", "n_lockbox", "r2_oof", "mae_oof", "r2_lockbox", "mae_lockbox", "sd_target_dev"]].round(3))
print("Parsimonia del estado (e7_04/e7_07): R2 y theta de los sets parsimoniosos con las palancas y signos de v5")
if (EXP / "e7_07_parsimonia_v5.csv").exists(): display(pd.read_csv(EXP / "e7_07_parsimonia_v5.csv").round(3))
print("Estabilidad temporal (experimentos/v5/e7_04): theta por tercios, walk-forward cronologico en DEV y reentrenamiento por campana")
if (EXP / "e7_04_theta_periodos_resumen.csv").exists(): display(pd.read_csv(EXP / "e7_04_theta_periodos_resumen.csv").round(3))
for f in ["e7_04_walkforward.csv", "e7_04_reentrenamiento.csv"]:
    if (EXP / f).exists(): display(pd.read_csv(EXP / f).round(3).head(40))""")

md(r"""En Reducción el agotamiento de Sn se predice bien (R² ≈ 0.77 OOF / 0.80 lockbox); la retención de FeO es intrínsecamente
ruidosa (techo de ruido del trazador R² ≈ 0.8 por Monte Carlo, `search_targets` ST-02; OOF ≈ 0.16-0.21): lo que el prescriptor
usa de ese modelo es el efecto de las palancas (θ), no el nivel. Los modelos de Fusión sólo aportan θ (el nivel del agotamiento en
Fusión depende de la carga del escalón y no generaliza al fin de campaña). El PLM iguala o supera al HGB con estado + palancas en
todos los targets.""")

md(r"""## 4. Real vs. predicho (scatter y evolutivo)""")

code(r"""def cargar_pred(fase, clave):
    return mp5.predicciones_oof_y_lockbox_v5(df, fase, clave) if RECALCULAR else pd.read_csv(EXP / f"e7_05_pred_{fase}_{clave}.csv", parse_dates=["fecha_inicio"])

combos = [("Reducción", "sn_dep"), ("Reducción", "feo_ret"), ("Reducción", "dT"), ("Fusión", "sn_dep")]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for j, (fase, clave) in enumerate(combos):
    p = cargar_pred(fase, clave)
    for i, conjunto in enumerate(["OOF_dev", "lockbox"]):
        q = p[p.conjunto == conjunto]; ax = axes[i, j]
        ax.scatter(q.real, q.pred, s=8, alpha=0.4)
        lim = [min(q.real.min(), q.pred.min()), max(q.real.max(), q.pred.max())]; ax.plot(lim, lim, "k--", lw=0.8)
        r2 = 1 - ((q.real - q.pred) ** 2).sum() / ((q.real - q.real.mean()) ** 2).sum()
        ax.set_title(f"{fase} {mp5.TARGETS_V5[fase][clave]}\n{conjunto}: R²={r2:.3f}, n={len(q)}", fontsize=9)
        ax.set_xlabel("real"); ax.set_ylabel("predicho")
plt.tight_layout(); plt.show()""")

code(r"""fig, axes = plt.subplots(3, 1, figsize=(18, 11))
for ax, (fase, clave, lab) in zip(axes, [("Reducción", "sn_dep", "Σ ln(Sn_prev/Sn) = ln(Sn_ini/Sn_fin) de Reducción"), ("Reducción", "feo_ret", "Σ ln(FeO/FeO_prev) = ln(FeO_fin/FeO_ini) de Reducción"), ("Fusión", "sn_dep", "Σ ln_sn_dep de Fusión")]):
    p = cargar_pred(fase, clave)
    b = p.groupby("Batch").agg(fecha=("fecha_inicio", "min"), real=("real", "sum"), pred=("pred", "sum"), conjunto=("conjunto", "first")).sort_values("fecha")
    x = np.arange(len(b))
    ax.plot(x, b.real, label="real (suma telescópica del batch)", lw=1); ax.plot(x, b.pred, label="predicho (OOF en DEV / lockbox)", lw=1)
    i0 = int((b.conjunto == "OOF_dev").sum()); ax.axvline(i0, color="r", ls="--", lw=1); ax.text(i0 + 1, ax.get_ylim()[1] * 0.9, "lockbox →", color="r")
    ax.set_title(f"Evolutivo por batch (orden cronológico): {lab}"); ax.legend(loc="upper left")
axes[-1].set_xlabel("batch (cronológico)"); plt.tight_layout(); plt.show()""")

code(r"""p = cargar_pred("Reducción", "sn_dep"); q = p[p.conjunto == "lockbox"].sort_values("fecha_inicio")
q = q[q.Batch.isin(sorted(q.Batch.unique())[:15])].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(18, 4))
ax.plot(q.real, "o-", ms=3, lw=1, label="real"); ax.plot(q.pred, "s-", ms=3, lw=1, label="predicho")
for i in np.where(q.orden_escalon_fase == 0)[0]:
    ax.axvline(i - 0.5, color="grey", lw=0.5, ls=":")
ax.set_title("Lockbox, escalón a escalón (15 batches): agotamiento de Sn en Reducción [ln]"); ax.legend(); plt.show()""")

md(r"""## 5. Interpretabilidad: SHAP del estado, dependencia parcial y formas aprendidas vs. teoría

El PLM separa el efecto del ESTADO (no lineal, `g(S)`, interpretado con SHAP y dependencia parcial) del efecto de las palancas
de CONTROL (lineal, `θ`, interpretado directamente). Las curvas de respuesta del prescriptor (predicción y `J` a lo largo de
cada palanca en un estado concreto) son la dependencia parcial operativa. La tabla teórica de formas está en
`experimentos/v5/ITERACION_7_diseno.md` y el contraste empírico en `e7_03_formas_vs_teoria.csv`.""")

code(r"""import shap
def shap_estado(fase, clave, n=600):
    m = sistema.modelos[(fase, clave)]
    d = df_base[(df_base.fase_proceso == fase) & df_base.Batch.isin(batches_dev)].dropna(subset=m.estado + [m.target]).sample(min(n, 5000), random_state=0)
    X = d[m.estado]; sv = shap.TreeExplainer(m.g).shap_values(X)
    return X, sv, pd.Series(np.abs(sv).mean(axis=0), index=m.estado).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, (fase, clave) in zip(axes, [("Reducción", "sn_dep"), ("Reducción", "feo_ret"), ("Fusión", "sn_dep")]):
    X, sv, imp = shap_estado(fase, clave)
    imp.head(10)[::-1].plot.barh(ax=ax); ax.set_title(f"|SHAP| medio de g(S): {fase} {mp5.TARGETS_V5[fase][clave]}"); ax.set_xlabel("ln")
plt.tight_layout(); plt.show()""")

code(r"""from statsmodels.nonparametric.smoothers_lowess import lowess
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for i, (fase, clave) in enumerate([("Reducción", "sn_dep"), ("Reducción", "feo_ret"), ("Fusión", "sn_dep")]):
    X, sv, imp = shap_estado(fase, clave)
    for j, f in enumerate(imp.index[:4]):
        ax = axes[i, j]; k = list(X.columns).index(f)
        ax.scatter(X[f], sv[:, k], s=6, alpha=0.3)
        ok = X[f].notna().to_numpy()
        lw = lowess(sv[ok, k], X[f].to_numpy()[ok], frac=0.3, return_sorted=True); ax.plot(lw[:, 0], lw[:, 1], "r-", lw=2)
        ax.set_title(f"{fase} {clave}: {f}", fontsize=9); ax.set_ylabel("SHAP [ln]")
plt.tight_layout(); plt.show()""")

code(r"""# Dependencia parcial (PD/ALE) de palancas y estados clave sobre las componentes (experimentos/v5/e7_03) y contraste con la teoria
if (EXP / "e7_03_formas_vs_teoria.csv").exists():
    formas = pd.read_csv(EXP / "e7_03_formas_vs_teoria.csv"); display(formas.round(3))
if (EXP / "e7_03_pd_curvas.csv").exists():
    pdc = pd.read_csv(EXP / "e7_03_pd_curvas.csv")
    vars_plot = [v for v in ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "avance_reduccion_sn_prev", "temperatura_horno_celsius_prev", "basicidad_B2_prev", "posicion_vertical_lanza_mm_prev", "ley_feo_escoria_pct_prev"] if v in pdc["variable"].unique()]
    tg = [t for t in ["v5_ln_sn_dep", "v5_ln_feo_ret"] if t in pdc["target"].unique()]
    fig, axes = plt.subplots(len(tg), len(vars_plot), figsize=(3.2 * len(vars_plot), 3.6 * len(tg)), squeeze=False)
    for i, t in enumerate(tg):
        for j, v in enumerate(vars_plot):
            g = pdc[(pdc.target == t) & (pdc.variable == v)].sort_values("x"); ax = axes[i, j]
            if "pd" in g.columns: ax.plot(g.x, g.pd, label="PD")
            if "ale" in g.columns: ax.plot(g.x, g.ale, label="ALE", ls="--")
            ax.set_title(f"{t}\n{v}", fontsize=8)
    axes[0, 0].legend(); plt.tight_layout(); plt.show()
if (EXP / "e7_03_theta_por_tramo.csv").exists():
    tr = pd.read_csv(EXP / "e7_03_theta_por_tramo.csv"); display(tr.round(4).head(60))""")

code(r"""cur = pd.read_csv(EXP / "e7_05_curvas_respuesta_v5.csv")
red = cur[cur.fase == "Reducción"]; K = mp5.kg_sn_por_unidad_sdi(); w = mp5.CONFIG_V5["w_sdi"]
fig, axes = plt.subplots(3, 4, figsize=(20, 11))
for j, pal in enumerate(mp5.ACCIONES_OPTIMIZABLES_V5["Reducción"]):
    for (b, av), g in red[red.palanca == pal].groupby(["Batch", "avance_prev"]):
        axes[0, j].plot(g.valor, g.pred_sn_dep, label=f"avance {av:.2f}"); axes[1, j].plot(g.valor, g.pred_feo_ret, label=f"avance {av:.2f}"); axes[2, j].plot(g.valor, g.J, label=f"avance {av:.2f}")
    axes[0, j].set_title(f"agotamiento de Sn (ln) vs {pal}", fontsize=9); axes[1, j].set_title(f"retención de FeO (ln) vs {pal}", fontsize=9); axes[2, j].set_title(f"J [kg Sn eq.] vs {pal}", fontsize=9); axes[2, j].set_xlabel(pal)
axes[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()
opt = cur.groupby(["fase", "Batch", "avance_prev", "palanca"]).apply(lambda g: pd.Series({"x_min": g.valor.min(), "x_opt": g.loc[g.J.idxmax(), "valor"], "x_max": g.valor.max(), "rango_J_kg": g.J.max() - g.J.min()})).round(2)
display(opt)""")

md(r"""**Lectura.** Las componentes responden de forma monótona a cada palanca (por construcción, con pendiente `θ`); `J` no: el
peso `w` del FeO, la ventana térmica y el soporte histórico producen máximos interiores o en el borde del soporte según el
estado (avance de reducción). Los términos cuadráticos residualizados (E7-03) no son significativos donde la teoría no los
espera, y la selectividad del carbón por tramo de avance se reporta en `e7_03_theta_por_tramo.csv`. Las recomendaciones nunca
salen del envolvente histórico (soporte mínimo duro = P10 de DEV).""")

md(r"""## 6. Capa prescriptiva: objetivo, roles y recomendación por escalón

- **Reducción:** `J_R = K·(ln_sn_dep_pred + w·ln_feo_ret_pred) − costos(C, GN, O₂) − w_T·(T fuera de P5-P95) − w_U·ancho_IC − w_S·(1−soporte)`,
  con `w = 6.04` y `K = 2.12 pp de KPI por unidad × 512 kg de Sn por pp ≈ 1 090 kg` (anclados en la regresión de batch del KPI
  refinado, E7-01). Cada unidad de `J` es un kg de Sn a metal equivalente.
- **Fusión:** `J_F = −K_S·ln_sn_dep_pred − costos − w_T·(T fuera de ventana) − w_U·ancho_IC − w_S·(1−soporte)`, con
  `K_S = 0.45 pp por unidad × 512 kg` (conservación del Sn en Fusión; E7-02). Los términos v4 de C/Sn y temperatura no son
  defendibles con el KPI refinado (bajan dross pero suben polvo, efecto neto nulo) y la matriz IRF durante la Fusión no aporta una
  vez controlado el heel.
- **Palancas:** carbón, GN, O₂, aire (búsqueda aleatoria en P1-P99 + refinamiento local; la acción histórica siempre es
  candidata; soporte mínimo duro). El resto (carga, cal, duración) es plan.""")

code(r"""batch_id = sorted(batches_lockbox)[0]
rep = mp5.reporte_recomendaciones_batch_v5(sistema, df, batch_id, n_candidatos=300)
res = mp5.resumen_reporte_batch_v5(rep)
print({k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()})
cols = ["fase", "orden_escalon_fase", "avance_reduccion_sn_prev", "pred_sn_dep_hist", "pred_sn_dep_rec", "pred_feo_ret_hist", "pred_feo_ret_rec",
        "pred_T_historico", "pred_T_recomendado", "support_historico", "support_recomendado", "uplift_J_kgSn"] + [c for c in rep.columns if (c.startswith("hist__") or c.startswith("rec__")) and ("Carbon" in c or "gn_nm3" in c or "o2_nm3" in c or "aire" in c)]
display(rep[[c for c in cols if c in rep.columns]].round(3).T)""")

code(r"""fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, pal in zip(axes, mp5.ACCIONES_OPTIMIZABLES_V5["Reducción"]):
    x = np.arange(len(rep)); ax.plot(x, rep[f"hist__{pal}"], "o-", label="histórico"); ax.plot(x, rep[f"rec__{pal}"], "s--", label="recomendado")
    ax.set_xticks(x); ax.set_xticklabels([f"{f[0]}{o}" for f, o in zip(rep.fase, rep.orden_escalon_fase)]); ax.set_title(f"{batch_id}: {pal}")
axes[0].legend(); plt.tight_layout(); plt.show()""")

code(r"""# Recomendacion media de la politica cross-fitted (cada batch recomendado por un sistema que no lo vio), por fase y tramo de avance
esc = pd.read_csv(EXP / "e7_05_recomendaciones_v5_escalones.csv")
esc["tramo"] = pd.cut(esc.avance_reduccion_sn_prev, [0, 0.84, 0.96, 0.975, 1.01], labels=["<0.84", "0.84-0.96", "0.96-0.975", ">0.975"])
for fase, key in [("Reducción", "tramo"), ("Fusión", "orden_escalon_fase")]:
    e = esc[esc.fase == fase].copy()
    for a in mp5.ACCIONES_OPTIMIZABLES_V5[fase]:
        e[f"d_{a}"] = e[f"rec__{a}"] - e[f"hist__{a}"]
    display(e.groupby(key, observed=True)[[f"d_{a}" for a in mp5.ACCIONES_OPTIMIZABLES_V5[fase]] + ["uplift_J_kgSn", "support_historico", "support_recomendado"]].agg(["mean", "median"]).round(2))""")

md(r"""## 7. Sustento: ¿aplicar las recomendaciones conduce a mayor recuperación del batch?

Cuatro capas de evidencia, todas con controles de contexto (ley del concentrado, refractario, Sn cargado, temperatura de Fusión,
carga secundaria, dross de Fe cargado, tendencia de campaña) y separando DEV (estimación) de lockbox (comprobación):

1. **Target → KPI (iteraciones 5-7):** el agregado telescópico del objetivo de escalón correlaciona con la recuperación refinada
   (sección 1.1: ρ ≈ 0.3, monótono en quintiles, replica con el batch siguiente vía heel de escoria).
2. **Palanca → target (sección 2):** θ con IC por bootstrap por batch, coherentes con la teoría.
3. **Cadena θ → agregado → KPI (E7-06):** el cambio predicho de las componentes por la política (Σ pred_rec − pred_hist) por las
   pendientes de batch del KPI = uplift esperado por batch con IC.
4. **Off-policy cross-fitted (E7-05/06):** cada batch recibe la recomendación de un sistema que nunca lo vio; la distancia entre
   la operación real y la recomendada se relaciona con el KPI observado (OLS con controles, permutación, placebo de política
   aleatoria, emparejamiento por estado y dosis-respuesta por terciles).""")

code(r"""ev = pd.read_csv(EXP / "e7_05_evidencia_politica_v5.csv"); pb = pd.read_csv(EXP / "e7_05_por_batch_v5.csv")
sel = ev[ev.variable.isin(["dist_total", "dist_F", "dist_R", "dist_total_placebo", "dist_R_placebo", "dist_F_placebo"]) & ev.kpi.isin(["recuperacion_refinada_pct", "recuperacion_refinada_pct_media2", "rendimiento_proxy_batch"])]
display(sel.pivot_table(index=["kpi", "variable"], columns="subset", values=["coef_por_sd", "p_hc3", "spearman"]).round(3))
print("Direccion de la recomendacion (rec-hist)/sd por palanca (media por batch):")
display(pb[[c for c in pb.columns if c.startswith("dir_")]].describe().T.round(3))
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, (kpi, lab) in zip(axes, [("recuperacion_refinada_pct", "recuperación refinada [%]"), ("f_dross", "fracción a dross"), ("f_polvo", "fracción a polvo")]):
    pb["q"] = pd.qcut(pb.dist_R, 5, labels=False)
    g = pb.groupby("q")[kpi].agg(["mean", "sem", "size"])
    ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="o-", capsize=4); ax.set_xlabel("quintil de distancia a la política de Reducción (0 = más cerca)"); ax.set_ylabel(lab)
    rho, p = spearmanr(pb.dist_R, pb[kpi]); ax.set_title(f"{lab}: Spearman ρ={rho:+.3f} (p={p:.3f})")
plt.tight_layout(); plt.show()""")

code(r"""for f, titulo in [("e7_06_matching.csv", "Emparejamiento por estado: KPI de batches cerca vs lejos de la politica (dif > 0 favorece seguir la politica)"),
                  ("e7_06_dosis_respuesta.csv", "Dosis-respuesta por terciles de adherencia"), ("e7_06_cadena.csv", "Cadena theta -> agregado -> KPI (uplift esperado por batch)")]:
    if (EXP / f).exists():
        print(titulo); t = pd.read_csv(EXP / f); display(t[t.kpi == "recuperacion_refinada_pct"].round(3) if "kpi" in t.columns else t.round(3))""")

code(r"""# Cuantificacion en lockbox (63 batches nunca vistos): uplift estadistico (regresion de adherencia) vs mecanistico (J y cadena)
lb = pb[pb.es_lockbox].dropna(subset=["recuperacion_refinada_pct", "dist_R"] + mp5.CONTROLES_KPI_V5).copy()
X = lb[["dist_R"] + mp5.CONTROLES_KPI_V5]; Xz = (X - X.mean()) / X.std().replace(0, 1)
m_lb = sm.OLS(lb["recuperacion_refinada_pct"], sm.add_constant(Xz)).fit(cov_type="HC3")
beta, (b_lo, b_hi), p = m_lb.params["dist_R"], m_lb.conf_int().loc["dist_R"], m_lb.pvalues["dist_R"]
lb["delta_pp"] = -beta * lb["dist_R"] / lb["dist_R"].std(); lb["delta_kg"] = lb["delta_pp"] * mp5.KG_SN_POR_PP_KPI
print(f"Lockbox (n={len(lb)}): beta(dist_R)={beta:+.3f} pp/sd [{b_lo:.3f},{b_hi:.3f}] p={p:.3f}")
print(f"Uplift estadistico medio (dist_R -> 0): {lb.delta_pp.mean():+.2f} pp = {lb.delta_kg.mean():.0f} kg Sn/batch")
print(f"Uplift mecanistico J (Reduccion): {lb.uplift_fisico_reduccion_kg.mean():.0f} kg/batch; (batch F+R): {lb.uplift_fisico_total_kg.mean():.0f} kg/batch")
cad = pd.read_csv(EXP / "e7_06_cadena.csv") if (EXP / "e7_06_cadena.csv").exists() else None
if cad is not None:
    c = cad[(cad.kpi == "recuperacion_refinada_pct") & (cad.pendientes_de == "DEV")]
    display(c[["politica_en", "n_batches", "b_sn_dep_pp_por_unidad", "b_feo_ret_pp_por_unidad", "w_implicito", "d_sn_dep_medio", "d_feo_ret_medio", "uplift_kpi_pp_medio", "uplift_ci_lo", "uplift_ci_hi", "uplift_kg_sn_medio", "pct_batches_uplift_pos"]].round(3))""")

md(r"""## 8. Conclusiones, límites y siguiente iteración

Ver `candidate_win_model.md` (ficha completa: métricas, fórmulas, hallazgos, positivo/negativo) y `hallazgos.md` §18.

- **Modelo ganador:** PLM por fase con targets logarítmicos (agotamiento de Sn, retención de FeO) y objetivo anclado en el KPI
  refinado (`w` y `K` estimados, no fijados a mano). Efectos de CONTROL con IC: carbón × Sn disponible (selectivo, +), GN
  (no selectivo: +Sn, −FeO), exceso de O₂ y aire (débiles, signo teórico).
- **Prescripción:** en Reducción más carbón mientras hay Sn disponible y menos GN (lanza menos reductora) dentro de la ventana
  térmica y del soporte histórico; en Fusión menos gas de lanza mientras la temperatura lo permita (conservar el Sn en la escoria).
- **Evidencia:** cadena target→KPI significativa (ρ ≈ 0.3, OLS con controles), θ con IC, y evidencia off-policy con placebo,
  emparejamiento y dosis-respuesta (sección 7). Es observacional: el paso siguiente es un piloto A/B por batches.
- **Límites:** ruido del trazador en el FeO (R² ≈ 0.2); carbón identificable sólo con el estado completo; Fusión con
  identificación débil; lockbox = fin de campaña (reentrenar progresivamente por campaña); KPI de batch casi ruido blanco entre
  batches (techo ρ ≈ 0.45).""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = RAIZ / "win_analytics_v1.ipynb"
nbf.write(nb, out)
print("escrito", out, len(cells), "celdas")
