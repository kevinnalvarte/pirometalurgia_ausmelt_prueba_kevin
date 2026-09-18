"""Genera `win_analytics_v2.ipynb` (nbformat) con el modelo ganador v8 (iteracion 10). Ejecutar desde la raiz:
    python experimentos/build_notebook_win_v2.py
    jupyter nbconvert --to notebook --execute --inplace win_analytics_v2.ipynb --ExecutePreprocessor.timeout=3600
Las tablas pesadas (validacion, politica cross-fitted, evidencia, sensibilidad, evaluacion cruzada) se leen de experimentos/v8/e10_05_*.csv;
RECALCULAR=True las reproduce (horas).
"""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md(r"""# win_analytics_v2 — Prescriptor operacional por escalón del horno Ausmelt de Sn (modelo ganador v8, iteración 10)

**Objetivo.** Recomendar, escalón a escalón de la **Reducción**, la operación de las variables de CONTROL del reductor y la lanza
(tasa de carbón y de gas natural; O₂ y aire congelados al plan) dado el ESTADO al cierre del escalón anterior (leyes de escoria,
inventarios de Sn y FeO por cierre físico de masa, térmico, lanza, historia de dosificación), de modo que el batch termine con
**mayor recuperación de Sn a metal**; y fijar la **receta de Fusión** (sin recomendación por escalón: ninguna palanca de Fusión
tiene efecto identificado sobre el rendimiento).

**Qué es v8** (`modelo_predictivo_v8.py`, ficha `candidate_win_model_v8.md`, hallazgos §22). Es el núcleo predictivo de v7 (masa de
escoria por cierre físico, targets log telescópicos, modelo parcialmente lineal `y = g(S) + Σ θ_j (a_j − E[a_j|S])` con cross-fitting
por batch, estado parsimonioso) con las **correcciones de la fase 0 del dictamen del comité** (`dictamen_comite_v7.md`) y las mejoras que
sobrevivieron a la iteración 10 (`experimentos/v8/`):

| Componente | v7 | v8 (por qué) |
|---|---|---|
| Identificación de θ | acotado por signo (la cota fabricaba significancia) | **θ libre y acotado sobre los mismos residuos**; θ de uso = 0 si el IC libre no excluye 0 (E10-01/02) |
| Target de FeO | `ln(FeO_t / FeO_{t−1})` | **descuenta el FeO equivalente del Fe metálico del dross alimentado** (R0): R² 0.33 → 0.43 OOF, GN identificado (E10-02) |
| Estado de Reducción | 16 con `espesor` (reloj de campaña) | **15 sin `espesor`**; variante **plan** (cierre de F6 para todo R, ejecutable sin latencia de laboratorio, E10-03) |
| Forma del carbón | `Cx_sn` | `Cx_sn = C · Sn disponible` (única forma identificada; dosis saturante y curvatura no identificadas, E10-01) |
| Objetivo | K 3.52 / w 7.86 con legado | **K 2.78 / w 5.93 [4.0, 9.6] sin legado**; sensibilidad w ∈ [2, 11], legado y objetivo lineal en kg λ = 3 (E10-04) |
| Optimizador | P1-P99 global, aire libre, IC constante | **caja por orden**: P5-P95 ∩ |Δ| ≤ 1 sd, carbón R0 ≤ P90, aire/O₂ congelados, soporte < P10 → sin recomendación, ventana térmica dura, IC dependiente de la acción |
| Fusión | conservar Sn (β_F −0.58) | **β_F = 0**; ΔT sin F1; polvo y dross se cancelan → receta (E10-06) |
| Evidencia | adherencia vs KPI (sin potencia) | dosis-respuesta residual por escalón, residuo de la acción → KPI, uplift con IC propagado, **evaluación cruzada** (anti-maldición del optimizador) |

**Validación.** OOF GroupKFold(5) por batch sobre DEV (299 batches, abril-julio 2026) para elegir; OOF cronológico y walk-forward
(criterio de generalización, por el dictamen); lockbox = últimos 63 batches (agosto 2026, fin de campaña) sólo reporte y declarado
"gastado" por el comité.""")

code(r"""import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm

import modelo_prescriptivo as mp
import modelo_predictivo_v8 as mp8
import sys; sys.path.insert(0, "experimentos/v8"); import v8_lib as L8

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 120)
EXP = Path("experimentos/v8")
RECALCULAR = False   # True: recalcula tablas pesadas (validacion, cross-fitting) en vez de leer experimentos/v8/e10_05_*.csv

df = mp8.cargar_df()                    # df v6 + columnas m8_* (cache experimentos/cache/df_v8.pkl)
df_base = mp8._preparar_base(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
F, R = df_base["fase_proceso"] == "Fusión", df_base["fase_proceso"] == "Reducción"
print(f"escalones {df_base.shape[0]} (Fusión {F.sum()}, Reducción {R.sum()}) | batches DEV {len(batches_dev)} | lockbox {len(batches_lockbox)}")
print("targets:", mp8.TARGETS_V8); print("config:", {k: v for k, v in mp8.CONFIG_V8.items() if k != "ordenes_sin_recomendacion"})""")

md(r"""## 1. Fundamento pirometalúrgico: STATE, CONTROL, targets y su agregación

**Batch** = 7 escalones de Fusión (carga, lanza oxidante, formación de la escoria FeO–SiO₂–CaO) + 4 de Reducción (sin carga:
`SnO₂ + 2C → Sn + 2CO` compite con `FeO + C → Fe + CO`; el Fe metalizado forma hardhead/dross que arrastra Sn; la temperatura y el
arrastre generan polvo).

**Masa de escoria por cierre físico** (`experimentos/v6/masa_v6.py`): `r_t = 1 − 1.270·s_t − f_t` (fracción no reducible),
`M_t·r_t = M_{t−1}·r_{t−1}·e^{−0.01} + 0.703·cal_t + 0.301·carga_t`; `Sn_inv = M·s`, `FeO_inv = M·f` (FeO ≡ 1.2865·Fe total).

**Targets por escalón** (adimensionales, log, telescópicos):

| Target | Fórmula | Sentido pirometalúrgico | Agregación |
|---|---|---|---|
| `m6_ln_sn_dep` | `ln((Sn_inv[t−1] + Sn alimentado[t]) / Sn_inv[t])` | agotamiento del Sn de la escoria (cinética aparente de 1er orden) | Σ_R = `ln(Sn_ini/Sn_fin)` (exacto) |
| `m8_ln_feo_ret_dross50` | `ln(FeO_inv[t] / (FeO_inv[t−1] + 1.2865·0.5·dross_Fe[t]))` | retención del Fe en escoria **neta del Fe metálico alimentado** (≤ 0 si se metaliza Fe: hardhead) | Σ_R = `ln(FeO_fin/(FeO_ini + Fe alimentado))` |
| `d_temperatura_horno_celsius` | ΔT del horno | ventana térmica (restricción, no objetivo) | — |

**Objetivo por escalón y agregación** (anclado en el KPI refinado `recuperacion_refinada_pct = 100·M/(M+D+P+Sn escoria final)`):
`J_t = K·(ln_sn_dep_t + w·ln_feo_ret_t) − costos_t − w_U·sd_J(acción) − w_S·max(0, s_hist − s)`, K = 2.78 pp/unidad (× 512 kg Sn/pp),
w = 5.93 [3.96, 9.62]; `J_grupo = Σ J_t` (R temprano R0-R1, R tardío R2-R3); `J_R = K·[ln(Sn_ini/Sn_fin) + w·ln(FeO_fin/FeO_ini')] − Σ costos`
(identidad telescópica exacta); `J_batch = J_R + J_F` con `J_F = −costos` (β_F = 0).

**Roles.** STATE (15): leyes previas %Sn, %FeO, %SiO₂, %CaO, `ratio_sn_feo_prev`, `interaccion_Sn_x_FeO_prev`, termocupla y gradiente de T,
lanza previa, carbón y aire acumulados, C/Sn acumulado, inventarios de Sn y FeO y fracción no reducible (cierre físico). CONTROL:
`tasa_feed_Carbon_kg_min` (vía `Cx_sn_v6 = C·Sn_inv_prev/1000`, `m8_exceso_C_pos`, `Cx_av_v6`), `tasa_gn_nm3_min`; `exceso_o2` y `aire` en el
modelo pero **congelados** (setpoint de protección de lanza; sin efecto identificado). PLAN: carga, cal, duración, O₂, aire.""")

code(r"""for fase in mp8.FASES:
    print(f"=== {fase} ===")
    for clave, tgt in mp8.TARGETS_V8[fase].items():
        S, A = mp8.ESTADO_V8[(fase, clave)], mp8.PALANCAS_V8[(fase, clave)]
        print(f"  target {tgt:28s} | estado g(S): {len(S):2d} vars | palancas theta: {A}")
    print("  CONTROL optimizable:", mp8.ACCIONES_OPTIMIZABLES_V8[fase], "| congeladas:", mp8.ACCIONES_CONGELADAS_V8)
print(df_base.loc[R, ["m6_ln_sn_dep", "m6_ln_feo_ret", "m8_ln_feo_ret_dross50", "d_temperatura_horno_celsius"]].describe().round(3).T)""")

md(r"""### 1.1 Por qué estos targets: anclaje en el rendimiento del batch (E10-04a/b)

El KPI refinado del batch regresado (OLS HC3, controles de ley, refractario, Sn cargado, carga secundaria, dross de Fe y tendencia)
sobre las sumas telescópicas de Reducción da K_Sn ≈ 2.8 pp por unidad de agotamiento y K_FeO ≈ 16.5 pp por unidad de retención →
w ≈ 5.9. El anclaje es invariante a la agregación (suma por escalón = cociente de inventarios) y al target de FeO (v7 vs corregido por
dross). En el lockbox (fin de campaña, horno frío) w cae a ≈ 0.8 (el KPI premia agotar Sn): por eso la política se exige robusta en w ∈ [2, 11].""")

code(r"""anc = pd.read_csv(EXP / "e10_04b_reanclaje.csv")
display(anc[anc.y.isin(["KPI", "KPI_next"]) & anc.target.isin(["m6_ln_feo_ret", "m8_ln_feo_ret_dross50"])][["target", "muestra", "y", "n", "K_sn", "K_sn_lo", "K_sn_hi", "K_feo", "K_feo_lo", "K_feo_hi", "K_feo_p", "w", "w_boot_lo", "w_boot_hi"]].round(3))
sw = pd.read_csv(EXP / "e10_04a_barrido_w.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
for m, c in [("dev", "#1f77b4"), ("lockbox", "#ff7f0e"), ("total", "#2ca02c")]:
    s = sw[(sw.muestra == m) & (sw.tercio == "todos")]; ax.plot(s.w, s.rho, "o-", label=m, color=c)
ax.axvline(mp8.CONFIG_V8["w_sdi"], color="k", ls="--", lw=1, label=f"w = {mp8.CONFIG_V8['w_sdi']}")
ax.set_xlabel("w (peso del FeO retenido)"); ax.set_ylabel("ρ Spearman J_R(w) vs KPI"); ax.set_title("Barrido de w: DEV en meseta desde w ≈ 4; el lockbox (fin de campaña) prefiere w ≤ 2"); ax.legend(); plt.show()""")

md(r"""## 2. Construcción del modelo y efecto de las variables de CONTROL (θ libre y acotado)

`mp8.entrenar_sistema_v8` entrena sólo con DEV un `ModeloPLMDual` por (fase, target): `g(S)` y `E[a_j|S]` con HistGradientBoosting
cross-fitted (GroupKFold por batch), θ por mínimos cuadrados **libres** y **acotados por el signo de la teoría** sobre los mismos residuos,
IC95 % bootstrap por batch (500) para ambos. **Una palanca entra al objetivo sólo si su IC libre excluye 0**; si no, θ de uso = 0 y se declara
"no identificada" (no "≈ 0"). Además: soporte histórico kNN, límites P5-P95 y ventana térmica P10-P90 **por orden de escalón**.""")

code(r"""sistema = mp8.entrenar_sistema_v8(df, n_boot=300)
efectos = mp8.tabla_efectos_control_v8(sistema)
display(efectos[["fase", "clave", "palanca", "signo_teorico", "theta_libre_1sd", "ci_lo_libre_1sd", "ci_hi_libre_1sd", "theta_acot_1sd", "frac_replicas_en_cota", "identificado", "theta_uso_1sd"]].round(4))""")

code(r"""fig, axes = plt.subplots(1, 4, figsize=(22, 4.2))
for ax, (fase, clave, titulo) in zip(axes, [("Reducción", "sn_dep", "Reducción: agotamiento de Sn [ln]"), ("Reducción", "feo_ret", "Reducción: retención de FeO neta [ln]"),
                                         ("Reducción", "dT", "Reducción: ΔT [°C]"), ("Fusión", "dT", "Fusión (F2-F6): ΔT [°C]")]):
    e = efectos[(efectos.fase == fase) & (efectos.clave == clave)]; y = np.arange(len(e))
    ax.barh(y + 0.18, e["theta_libre_1sd"], height=0.34, xerr=[e["theta_libre_1sd"] - e["ci_lo_libre_1sd"], e["ci_hi_libre_1sd"] - e["theta_libre_1sd"]], color=np.where(e["identificado"], "#1f77b4", "#b0b0b0"), capsize=3, label="θ libre (IC95 %)")
    ax.barh(y - 0.18, e["theta_uso_1sd"], height=0.34, color="#d62728", alpha=0.7, label="θ de uso (acotado × identificado)")
    ax.set_yticks(y); ax.set_yticklabels(e["palanca"], fontsize=8); ax.axvline(0, color="k", lw=0.8); ax.set_title(titulo, fontsize=10); ax.set_xlabel("efecto por +1 sd")
axes[0].legend(fontsize=8); plt.tight_layout(); plt.show()""")

md(r"""**Lectura.** En Reducción el **carbón × Sn disponible** sube el agotamiento de Sn (θ libre > 0, IC que excluye 0: la palanca se
identifica sin recurrir a la cota) y el **GN** sube el agotamiento y **baja la retención de FeO** (reductor no selectivo; con el estado sin
`espesor` y el target neto del Fe alimentado su IC libre sobre FeO excluye 0). El carbón sobre la retención de FeO **no está identificado**
(IC libre ±0.04, réplicas en la cota > 50 %): el prescriptor no le asigna efecto y por eso la recomendación de carbón es una dirección con
paso acotado, no un óptimo. Exceso de O₂ y aire: sin efecto identificado en ningún target → congelados. En Fusión ninguna palanca tiene
efecto térmico identificado (el O₂ "significativo" de v7 era cota + control reactivo).""")

md(r"""## 3. Métricas de predicción (OOF aleatorio = selección; OOF cronológico y walk-forward = generalización; lockbox = reporte)""")

code(r"""tabla = mp8.tabla_validacion_v8(df) if RECALCULAR else pd.read_csv(EXP / "e10_05_tabla_validacion_v8.csv")
display(tabla[["fase", "target", "columna", "forma", "n_estado", "n_palancas", "n_dev", "n_lockbox", "r2_oof", "mae_oof", "r2_oof_cronologico", "r2_walk_forward", "r2_lockbox", "mae_lockbox", "sd_target_dev"]].round(3))
print("Referencia v7 (candidate_win_model_v7.md): Sn 0.767/0.795, FeO 0.333/0.483 (OOF/lockbox), OOF cronológico FeO 0.296, walk-forward 0.261")
print("Comparación de targets y estados (E10-02) y del estado ejecutable (E10-03):")
for f in ["e10_02_targets_estados.csv", "e10_03_estados.csv"]:
    if (EXP / f).exists(): display(pd.read_csv(EXP / f).round(3))""")

md(r"""## 4. Real vs. predicho (scatter y evolutivo)""")

code(r"""def cargar_pred(fase, clave):
    return mp8.predicciones_oof_y_lockbox_v8(df, fase, clave) if RECALCULAR else pd.read_csv(EXP / f"e10_05_pred_{fase}_{clave}.csv", parse_dates=["fecha_inicio"])

combos = [("Reducción", "sn_dep"), ("Reducción", "feo_ret"), ("Reducción", "dT"), ("Fusión", "dT")]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for j, (fase, clave) in enumerate(combos):
    p = cargar_pred(fase, clave)
    for i, conjunto in enumerate(["OOF_dev", "lockbox"]):
        q = p[p.conjunto == conjunto]; ax = axes[i, j]
        ax.scatter(q.real, q.pred, s=8, alpha=0.4)
        lim = [min(q.real.min(), q.pred.min()), max(q.real.max(), q.pred.max())]; ax.plot(lim, lim, "k--", lw=0.8)
        r2 = 1 - ((q.real - q.pred) ** 2).sum() / ((q.real - q.real.mean()) ** 2).sum()
        ax.set_title(f"{fase} {mp8.TARGETS_V8[fase][clave]}\n{conjunto}: R²={r2:.3f}, n={len(q)}", fontsize=9); ax.set_xlabel("real"); ax.set_ylabel("predicho")
plt.tight_layout(); plt.show()""")

code(r"""fig, axes = plt.subplots(2, 1, figsize=(18, 8))
for ax, (fase, clave, lab) in zip(axes, [("Reducción", "sn_dep", "Σ_R ln_sn_dep = ln(Sn_ini/Sn_fin)"), ("Reducción", "feo_ret", "Σ_R ln_feo_ret neto = ln(FeO_fin/(FeO_ini + Fe alimentado))")]):
    p = cargar_pred(fase, clave)
    b = p.groupby("Batch").agg(fecha=("fecha_inicio", "min"), real=("real", "sum"), pred=("pred", "sum"), conjunto=("conjunto", "first")).sort_values("fecha")
    x = np.arange(len(b)); ax.plot(x, b.real, label="real (suma telescópica del batch)", lw=1); ax.plot(x, b.pred, label="predicho (OOF en DEV / lockbox)", lw=1)
    i0 = int((b.conjunto == "OOF_dev").sum()); ax.axvline(i0, color="r", ls="--", lw=1); ax.text(i0 + 1, ax.get_ylim()[1] * 0.9, "lockbox →", color="r")
    ax.set_title(f"Evolutivo por batch (cronológico): {lab}"); ax.legend(loc="upper left")
axes[-1].set_xlabel("batch (cronológico)"); plt.tight_layout(); plt.show()
p = cargar_pred("Reducción", "sn_dep"); q = p[p.conjunto == "lockbox"].sort_values("fecha_inicio"); q = q[q.Batch.isin(sorted(q.Batch.unique())[:15])].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(18, 3.6)); ax.plot(q.real, "o-", ms=3, lw=1, label="real"); ax.plot(q.pred, "s-", ms=3, lw=1, label="predicho")
for i in np.where(q.orden_escalon_fase == 0)[0]: ax.axvline(i - 0.5, color="grey", lw=0.5, ls=":")
ax.set_title("Lockbox, escalón a escalón (15 batches): agotamiento de Sn en Reducción [ln]"); ax.legend(); plt.show()""")

md(r"""## 5. Interpretabilidad: SHAP del estado, dependencia parcial y curvas de respuesta de las palancas

El PLM separa el efecto del ESTADO (no lineal, `g(S)`: SHAP y dependencia parcial) del de las palancas (lineal, θ, sección 2). Las
curvas de respuesta del prescriptor (predicciones y `J` a lo largo de cada palanca en un estado real) son la dependencia parcial operativa.
Coherencia esperada por teoría: más %Sn previo → más agotamiento (fuerza motriz); más %FeO previo → más competencia (menos selectivo);
la retención de FeO es más negativa cuando queda poco Sn (avance alto) porque el reductor pasa a metalizar Fe.""")

code(r"""import shap
def shap_estado(fase, clave, n=700):
    m = sistema.modelos[(fase, clave)]
    d = df_base[(df_base.fase_proceso == fase) & df_base.Batch.isin(batches_dev)].dropna(subset=m.estado + [m.target]).sample(min(n, 5000), random_state=0)
    X = d[m.estado]; sv = shap.TreeExplainer(m.g).shap_values(X)
    return X, sv, pd.Series(np.abs(sv).mean(axis=0), index=m.estado).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (fase, clave) in zip(axes, [("Reducción", "sn_dep"), ("Reducción", "feo_ret")]):
    X, sv, imp = shap_estado(fase, clave); imp.head(12)[::-1].plot.barh(ax=ax); ax.set_title(f"|SHAP| medio de g(S): {fase} {mp8.TARGETS_V8[fase][clave]}"); ax.set_xlabel("ln")
plt.tight_layout(); plt.show()
from statsmodels.nonparametric.smoothers_lowess import lowess
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
for i, (fase, clave) in enumerate([("Reducción", "sn_dep"), ("Reducción", "feo_ret")]):
    X, sv, imp = shap_estado(fase, clave)
    for j, f in enumerate(imp.index[:5]):
        ax = axes[i, j]; k = list(X.columns).index(f); ax.scatter(X[f], sv[:, k], s=6, alpha=0.3); ok = X[f].notna().to_numpy()
        lw = lowess(sv[ok, k], X[f].to_numpy()[ok], frac=0.3, return_sorted=True); ax.plot(lw[:, 0], lw[:, 1], "r-", lw=2); ax.set_title(f"{clave}: {f}", fontsize=9); ax.set_ylabel("SHAP [ln]")
plt.tight_layout(); plt.show()""")

code(r"""# Dependencia parcial del HGB directo sobre el carbon por orden (E10-01) y dosis-respuesta residual (E10-05): la forma aprendida vs teoria
pdc = pd.read_csv(EXP / "e10_01_pd_carbon.csv")
fig, axes = plt.subplots(1, 3, figsize=(20, 4.2))
for o, g in pdc[pdc.target == "m6_ln_sn_dep"].groupby("orden"):
    axes[0].plot(g.carbon, g.pd_media, "o-", ms=3, label=f"R{o}")
axes[0].set_title("PD del HGB: agotamiento de Sn vs carbón [kg/min] por orden (creciente y saturante)"); axes[0].set_xlabel("carbón kg/min"); axes[0].legend()
dr = pd.read_csv(EXP / "e10_05_dosis_respuesta_bins.csv")
for ax, (clave, pal, lab) in zip(axes[1:], [("sn_dep", "Cx_sn_v6", "agotamiento de Sn vs residuo de C·Sn_disp"), ("feo_ret", "tasa_gn_nm3_min", "retención de FeO vs residuo de GN")]):
    g = dr[(dr.clave == clave) & (dr.palanca == pal)]
    ax.errorbar(g.x_medio, g.y_res_medio, yerr=[g.y_res_medio - g.ic_lo, g.ic_hi - g.y_res_medio], fmt="o-", capsize=4); ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Dosis-respuesta residual (quintiles de a − E[a|S]): {lab}", fontsize=9); ax.set_xlabel("residuo de la palanca"); ax.set_ylabel("residuo del target [ln]")
plt.tight_layout(); plt.show()
dre = pd.read_csv(EXP / "e10_05_dosis_respuesta.csv"); display(dre[~dre.clave.str.startswith("batch")].round(4))""")

code(r"""cur = pd.read_csv(EXP / "e10_05_curvas_respuesta_v8.csv"); cur = cur[~cur.caja]
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
for j, pal in enumerate(mp8.ACCIONES_OPTIMIZABLES_V8["Reducción"]):
    for (o, q), g in cur[(cur.palanca == pal) & (cur.q_avance == 0.5)].groupby(["orden", "q_avance"]):
        axes[0, j].plot(g.valor, g.pred_sn_dep, label=f"R{o}"); axes[1, j].plot(g.valor, g.pred_feo_ret, label=f"R{o}"); axes[2, j].plot(g.valor, g.J, label=f"R{o}")
        axes[2, j].axvline(g["hist"].iloc[0], color="grey", lw=0.6, ls=":")
    axes[0, j].set_title(f"agotamiento de Sn (ln) vs {pal}", fontsize=9); axes[1, j].set_title(f"retención de FeO (ln) vs {pal}", fontsize=9); axes[2, j].set_title(f"J [kg Sn eq.] vs {pal} (P5-P95 del orden; punteado = histórico)", fontsize=9); axes[2, j].set_xlabel(pal)
axes[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()
oi = pd.read_csv(EXP / "e10_05_optimo_interior.csv"); print("fracción de curvas J(palanca) con máximo interior (no en la cota P5/P95):"); display(oi.groupby(["palanca", "orden"])["interior"].mean().unstack().round(2))""")

md(r"""**Lectura.** Las componentes responden de forma monótona a cada palanca dentro del rango histórico (θ lineal; la curvatura del
carbón tiene el signo de la teoría pero no se identifica, E10-01). `J` sí puede tener máximo interior por el intercambio Sn/FeO del GN
(sube agotamiento, baja retención) y por el costo; para el carbón, sin efecto identificado sobre FeO, `J` es lineal y la recomendación
es un paso acotado en la dirección del signo de `K·θ·Sn_disp − costo` (más carbón mientras el Sn disponible lo paga, menos cuando ya no).""")

md(r"""## 6. Capa prescriptiva: objetivo, salvaguardas y recomendación por escalón

- **Reducción:** `J = K·(ln_sn_dep + w·ln_feo_ret) − 0.05·C·Δt − 0.03·GN·Δt − 0.01·O₂·Δt − 0.5·sd_J(acción) − 2000·max(0, s_hist − s)`
  (kg de Sn a metal equivalente); búsqueda en grilla dentro de la **caja por orden** (P5-P95 ∩ |Δ| ≤ 1 sd; carbón R0 ≤ P90), acción histórica
  siempre admisible, soporte del estado ≥ P10 (si no: "sin recomendación"), ventana térmica dura relativa (no enfriar un horno frío ni
  calentar uno caliente), O₂ y aire congelados al plan; **sin ganancia física** (K·Δ(ln_sn_dep + w·ln_feo_ret) ≤ 0) la recomendación se degrada a
  `mantener:solo_costo` (precios placeholder); el **GN sólo se mueve si su dirección se conserva en los extremos del IC de w** [3.96, 9.62]
  (sensibilidad E10-05). Estados: `recomendar`, `mantener`, `mantener:solo_costo`, `sin_recomendacion:{soporte, orden, estado, receta}`.
- **Fusión:** receta (F1-F6 sin recomendación por escalón); el recorte económico de carbón (≤ 1 sd, ≥ P10, −O₂ 0.933 Nm³/kg C) sólo con precios reales.
- **Dos modos de estado** (E10-03): `k15` (ensayo del cierre de t−1; re-plan cuando llega el laboratorio) y `plan` (cierre de F6 + variables en
  línea: plan R0-R3 al inicio de la Reducción, ejecutable sin latencia; conserva ~92 % del R² de Sn e identifica carbón y GN).""")

code(r"""batch_id = sorted(batches_lockbox)[5]
rep = mp8.reporte_recomendaciones_batch_v8(sistema, df, batch_id)
cols = ["fase", "orden_escalon_fase", "estado_recomendacion", "m6_avance_prev", "pred_sn_dep_hist", "pred_sn_dep_rec", "pred_feo_ret_hist", "pred_feo_ret_rec",
        "pred_T_historico", "pred_T_recomendado", "support_historico", "support_recomendado", "uplift_J_kgSn", "hist__tasa_feed_Carbon_kg_min", "rec__tasa_feed_Carbon_kg_min",
        "hist__tasa_gn_nm3_min", "rec__tasa_gn_nm3_min", "hist__tasa_o2_nm3_min", "rec__tasa_o2_nm3_min"]
display(rep[[c for c in cols if c in rep.columns]].round(3).T)
print({k: round(v, 3) for k, v in mp8.agregar_objetivo(rep).items()})
r = rep[rep.fase == "Reducción"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, pal in zip(axes, mp8.ACCIONES_OPTIMIZABLES_V8["Reducción"]):
    x = np.arange(len(r)); ax.plot(x, r[f"hist__{pal}"], "o-", label="histórico"); ax.plot(x, r[f"rec__{pal}"], "s--", label="recomendado")
    ax.set_xticks(x); ax.set_xticklabels([f"R{o} ({e.split(':')[0][:5]})" for o, e in zip(r.orden_escalon_fase, r.estado_recomendacion)], fontsize=8); ax.set_title(f"{batch_id}: {pal}")
axes[0].legend(); plt.tight_layout(); plt.show()""")

code(r"""# Politica cross-fitted (cada batch recomendado por un sistema que no lo vio; 362 batches): perfil por orden y salvaguardas del comite
esc = pd.read_csv(EXP / "e10_05_recomendaciones_v8_escalones.csv")
display(pd.crosstab([esc.fase, esc.orden_escalon_fase], esc.estado_recomendacion, normalize="index").round(2))
mag = pd.read_csv(EXP / "e10_05_magnitud_cambios.csv")
display(mag[mag.fase == "Reducción"][["orden", "palanca", "n", "hist_P50", "delta_P5", "delta_P50", "delta_P95", "pct_cambia", "pct_sube", "pct_baja", "pct_abs_delta_mayor_1sd", "pct_rec_mayor_P95", "pct_rec_menor_P5", "pct_rec_en_borde_caja"] + (["pct_rec_mayor_P90"] if "pct_rec_mayor_P90" in mag.columns else [])].round(1))
e = esc[(esc.fase == "Reducción") & esc.estado_recomendacion.isin(["recomendar", "mantener", "mantener:solo_costo"])].copy()
for a in mp8.ACCIONES_OPTIMIZABLES_V8["Reducción"]: e[f"d_{a}"] = e[f"rec__{a}"] - e[f"hist__{a}"]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, a in zip(axes, mp8.ACCIONES_OPTIMIZABLES_V8["Reducción"]):
    e.boxplot(column=f"d_{a}", by="orden_escalon_fase", ax=ax, showfliers=False); ax.axhline(0, color="k", lw=0.6); ax.set_title(f"Δ recomendado − histórico: {a}"); ax.set_xlabel("orden de escalón de Reducción")
plt.suptitle(""); plt.tight_layout(); plt.show()""")

code(r"""# Robustez de la politica al objetivo (E10-05 sensibilidad): % de escalones con el mismo signo que la base bajo w in [2, 11], legado y objetivo lineal en kg
sens = pd.read_csv(EXP / "e10_05_sensibilidad_resumen.csv")
print("% de escalones con signo estrictamente opuesto a la base (carbon robusto <= 7 %; GN sensible al peso del FeO bajo el objetivo lineal en kg):")
display(sens.pivot_table(index=["palanca", "orden"], columns="variante", values="pct_signo_opuesto").round(0))
display(sens.pivot_table(index=["palanca", "orden"], columns="variante", values="delta_medio_variante").round(2))""")

md(r"""## 7. Sustento: ¿aplicar las recomendaciones conduce a mayor rendimiento del batch?

Cadena de evidencia con potencia en cada eslabón (el comité descartó la regresión de adherencia sobre el KPI por falta de potencia: se
reporta sólo como monitoreo):

1. **Palanca → target por escalón (n ≈ 1 000):** θ con IC libre (sección 2) y dosis-respuesta no paramétrica del residuo de la acción
   (sección 5): más carbón (por Sn disponible) → más agotamiento; menos GN → más retención de FeO, en el signo de la teoría.
2. **Residuo de la acción → KPI del batch, sin pasar por el modelo:** la desviación del operador respecto de lo que el estado predice
   (a − E[a|S]) agregada por batch se relaciona con el KPI en el signo que dicta la política.
3. **Target → KPI (anclaje, sección 1.1):** Σ ln_sn_dep y Σ ln_feo_ret suben la recuperación refinada (p < 1e-4), y la retención de FeO
   replica sobre el batch siguiente en el lockbox (legado).
4. **Uplift con IC propagado y evaluación cruzada:** el cambio predicho por la política se valora con (K, w) bootstrap y un factor de θ, y
   se re-evalúa con un modelo que no eligió la acción (otro fold) y con un HGB directo: corrige la maldición del optimizador.""")

code(r"""dre = pd.read_csv(EXP / "e10_05_dosis_respuesta.csv")
print("2) residuo de la accion (a - E[a|S]) agregado por batch -> KPI refinado (OLS HC3 con controles, Spearman):")
display(dre[dre.clave.str.startswith("batch")][["clave", "palanca", "grupo", "n", "spearman_residual", "p", "ols_pp_por_sd", "ols_p"]].round(4))
up = pd.read_csv(EXP / "e10_05_uplift_propagado.csv"); print("4a) uplift por batch (pp de KPI) con IC propagado:"); display(up.round(3))
cz = pd.read_csv(EXP / "e10_05_evaluacion_cruzada_resumen.csv"); print("4b) evaluacion cruzada (Δ J predicho por el optimizador vs por un modelo que no eligio la accion vs HGB directo):"); display(cz.round(4).T)""")

code(r"""pb = pd.read_csv(EXP / "e10_05_por_batch_v8.csv"); ev = pd.read_csv(EXP / "e10_05_evidencia_politica_v8.csv")
print("Monitoreo (sin potencia para probar, dictamen §3): adherencia (dist_*) y direccion (dir_*) vs KPI, OLS HC3 con controles")
sel = ev[ev.variable.isin(["dist_R", "dir_Reducción_tasa_feed_Carbon_kg_min", "dir_Reducción_tasa_gn_nm3_min", "uplift_batch_pp"]) & ev.kpi.isin(["recuperacion_refinada_pct", "recuperacion_refinada_pct_next", "f_dross", "f_polvo"])]
display(sel.pivot_table(index=["kpi", "variable"], columns="subset", values=["coef_por_sd", "p_hc3"]).round(3))
fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
for ax, (kpi, lab) in zip(axes, [("recuperacion_refinada_pct", "recuperación refinada [%]"), ("f_dross", "fracción a dross"), ("f_polvo", "fracción a polvo")]):
    d = pb.dropna(subset=["dist_R", kpi]).copy(); d["q"] = pd.qcut(d.dist_R, 4, labels=False, duplicates="drop"); g = d.groupby("q")[kpi].agg(["mean", "sem", "size"])
    ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="o-", capsize=4); rho, p = spearmanr(d.dist_R, d[kpi]); ax.set_title(f"{lab} vs cuartil de distancia a la política de R (ρ={rho:+.3f}, p={p:.2f})", fontsize=9)
plt.tight_layout(); plt.show()""")

md(r"""## 8. Conclusiones, límites y siguiente paso

Ver `candidate_win_model_v8.md` (ficha: métricas OOF/cronológico/walk-forward/lockbox, fórmulas, hallazgos, positivo/negativo) y `hallazgos.md` §22.

- **Modelo ganador (Reducción):** PLM dual por target con estado de 15 variables físicas (sin relojes de campaña), targets telescópicos
  (agotamiento de Sn; retención de Fe neta del Fe alimentado) y objetivo anclado en el KPI refinado (K 2.78, w 5.93). Palancas
  identificadas con IC libre: carbón × Sn disponible (+Sn), GN (+Sn, −FeO). No identificadas (θ de uso 0, a resolver en el piloto por
  escalón): carbón → FeO, exceso de O₂, aire.
- **Prescripción:** pasos acotados (≤ 1 sd del orden, P5-P95, R0 ≤ P90) de carbón y GN por escalón de Reducción; sin recomendación fuera
  del soporte histórico; Fusión = receta; modo `plan` para operar sin esperar al laboratorio.
- **Evidencia:** identificación por escalón con potencia, anclaje target → KPI replicado, uplift con IC propagado y evaluación cruzada.
  Es observacional: la magnitud del valor sólo la fija el piloto aleatorizado por escalón (carbón R0-R1 ±1 sd ≈ 15-30 batches por brazo;
  GN ≈ 230), como pide el dictamen.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = RAIZ / "win_analytics_v2.ipynb"
nbf.write(nb, out)
print("escrito", out, len(cells), "celdas")
