"""Genera `modelo_v3_masa_escoria_y_prescriptor.ipynb` (nbformat). Ejecutar desde la raiz:
    python experimentos/build_notebook_v3.py
    jupyter nbconvert --to notebook --execute --inplace modelo_v3_masa_escoria_y_prescriptor.ipynb
"""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md(r"""# Modelo predictivo v3: carga por tolva, masa de escoria por trazador CaO y prescriptor en kg de Sn

Continuación de `analisis_lingo_smelter.ipynb` (Pasos 0-48) y `modelo_v2_causalidad_y_prescripcion.ipynb`
(Pasos 49-55). Sesión 2026-09-13 (noche), motivada por el pedido de **volver a explorar y mejorar el modelo
predictivo para el mejor prescriptor operacional**, incorporando los datos del Excel que las rondas anteriores
no usaban: alimentación y Sn fino **por tolva** (1 concentrado, 3 reciclo Sn/Fe, 4 mineral de Fe WF, 5 dross de
Fe, 7 pellets/humos), la **duración planificada** del escalón, el estado del **refractario** y la corrección de la
cal (`CaO total` del Excel = tolva 6 + tolva 3).

Todo el código importa y ejecuta `modelo_predictivo_v3.py` (fuente de verdad de la ronda), `feature_engineering.py`
(bloque v3) y `dataset_lingo_smelter.py` (loader extendido). Los experimentos pesados viven en `experimentos/06-12_*.py`
y sus CSV; aquí se cargan sus salidas (poner `RECALCULAR = True` para reproducirlos desde cero). Resumen escrito
completo: `hallazgos.md` sección 14. Numeración: **Paso 56 en adelante**.""")

code(r"""import warnings, re
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import dataset_lingo_smelter as dls
import feature_engineering as fe
import modelo_prescriptivo as mp
import modelo_predictivo_v3 as mp3

pd.set_option("display.width", 180); pd.set_option("display.max_columns", 60)
EXP = Path("experimentos")
RECALCULAR = False   # True: re-ejecuta los experimentos pesados en vez de leer sus CSV

df = mp3.construir_dataset_modelo_v3("Datos Lingo smelter fase II.xlsx")
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
F, R = df["fase_proceso"] == "Fusión", df["fase_proceso"] == "Reducción"
print(f"df {df.shape} | columnas seguras para prescripcion: {len(fe.filtrar_features_seguras(list(df.columns)))} | dev={len(batches_dev)} lockbox={len(batches_lockbox)}")""")

md(r"""## Paso 56: verificaciones sobre el Excel que fundamentan la ronda

Antes de crear features se verificó, fila por fila, qué representan las columnas nuevas (detalle en
`diccionario_datos.json` → `_meta.actualizacion_2026_09_13_noche`).""")

code(r"""raw = pd.read_excel("Datos Lingo smelter fase II.xlsx", sheet_name="Datos").sort_values(["Batch", "Fecha inicio real"])
tmh_sum = raw[["tmh tolva 1 concentrado", "tmh tolva 3 alimentacin (Kg)", "tmh tolva 4 Hierro WF (Kg)", "tmh tolva 5 Dross Fierro", "tmh tolva 7 alimentador"]].sum(axis=1)
tmf_sum = raw[["tmf Sn tolva 1 concentrado", "tmf Sn tolva 3 alimentacion (Kg)", "tmf Sn tolva 4 Hierro WF (Kg)", "tmf Sn tolva 5 Dross Fierro", "tmf Sn tolva 7 alimentador"]].sum(axis=1)
cao_t6_t3 = raw["Kgmh tolva 6 Oxido de Calcio"].fillna(0) + raw["tmh tolva 3 alimentacin (Kg)"].fillna(0)
ley_t1 = 100 * raw["tmf Sn tolva 1 concentrado"] / raw["tmh tolva 1 concentrado"].where(raw["tmh tolva 1 concentrado"] > 50)
dur_real = (raw["Fecha final real"] - raw["Fecha inicio real"]).dt.total_seconds() / 60
dur_plan = dls._duracion_plan_a_minutos(raw["duracion plan"])
print("TMH total == suma tolvas 1,3,4,5,7 :", int((raw["TMH tolva: 3+4+5+7+1 (Kg)"] - tmh_sum).abs().lt(1e-3).sum()), "/", len(raw))
print("TMF Sn total == suma tmf tolvas   :", int((raw["TMF Sn total F+ R (Kg)"] - tmf_sum).abs().lt(1e-3).sum()), "/", len(raw))
print("'CaO total (kg)' == tolva 6 + tolva 3:", int((raw["CaO total (kg)"] - cao_t6_t3).abs().lt(1e-3).sum()), "/", len(raw))
print("std INTRA-batch de la ley Sn del concentrado (tmf/tmh), mediana entre batches:", round(ley_t1.groupby(raw["Batch"]).std().median(), 6))
print("duracion real == plan:", f"{(dur_real - dur_plan).abs().lt(0.5).mean():.1%} de los escalones; corr real vs plan {dur_real.corr(dur_plan):.3f}")
rha = raw.groupby("Batch")["Rendimiento HA"].first(); rp = raw.groupby("Batch").apply(lambda g: g["Sn al Metal Crudo (t)"].iloc[0] / (g["Sn al Metal Crudo (t)"] + g["Sn al Dross Fe (t)"] + g["Sn al Polvo de Fundición (t)"]).iloc[0])
print("corr Rendimiento HA vs rendimiento_proxy_batch:", round(rha.corr(rp), 4))
first = df.groupby("Batch").agg(t0=("fecha_inicio", "min"), sn_first=("ley_sn_escoria_pct", "first"), sn_last=("ley_sn_escoria_pct", "last")).sort_values("t0")
print("heel: corr %Sn primer escalon vs %Sn final del batch anterior:", round(first["sn_first"].corr(first["sn_last"].shift(1)), 3))""")

md(r"""## Paso 57: features v3 de ENTRADA y de ESTADO con lectura pirometalúrgica

- **Mezcla de carga por escalón** (ACTION): fracciones/tasas por corriente, `frac_carga_secundaria`, `ley_sn_carga_pct`,
  `relacion_CaO_carga` (cal/carga), `relacion_C_Sn_carga` (kg C por kg Sn cargado; estequiométrico ≈0.20).
- **Ensayo por corriente del batch** (CONTEXT): `ley_sn_<corriente>_batch_pct` (constante dentro del batch).
- **Inventario por trazador CaO** (STATE): la cal es inerte → `masa_escoria_est_kg = CaO acumulado / %CaO`;
  de ahí `sn_inventario_escoria_est_kg`, `feo_inventario_escoria_est_kg` y `avance_reduccion_sn_prev`.""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
mezcla = df.loc[F].groupby("orden_escalon_fase")[[f"frac_{c}_carga" for c in fe.TOLVAS_CARGA]].median()
mezcla.columns = list(fe.TOLVAS_CARGA); mezcla.plot.bar(stacked=True, ax=axes[0]); axes[0].set_title("Mezcla de carga por escalón de Fusión (mediana, datos observados)"); axes[0].set_ylabel("fracción de la masa cargada")
leyes = df.groupby("Batch")[[f"ley_sn_{c}_batch_pct" for c in fe.TOLVAS_CARGA]].first(); leyes.columns = list(fe.TOLVAS_CARGA)
leyes.replace(0, np.nan).boxplot(ax=axes[1]); axes[1].set_title("Ley de Sn por corriente, por batch (ensayo asignado al batch)"); axes[1].set_ylabel("% Sn")
inv = df.groupby(["fase_proceso", "orden_escalon_fase"])[["masa_escoria_est_kg", "sn_inventario_escoria_est_kg"]].median() / 1000
inv.index = [f"{'F' if f=='Fusión' else 'R'}{o}" for f, o in inv.index]
inv.plot(ax=axes[2], marker="o"); axes[2].set_title("Trazador CaO: masa de escoria y Sn en escoria (t, mediana)"); axes[2].set_ylabel("t")
plt.tight_layout(); plt.show()
print(df.groupby(["fase_proceso", "orden_escalon_fase"])[["masa_escoria_est_kg", "sn_inventario_escoria_est_kg", "avance_reduccion_sn_prev", "relacion_CaO_carga", "relacion_C_Sn_carga"]].median().round(3))""")

md(r"""## Paso 58: targets de SALIDA en masa

`sn_extraido_est_kg = Sn cargado − Δ(Sn en escoria)`: kg de Sn que dejaron la fase escoria en el escalón (a metal crudo,
o a polvo). `frac_sn_extraido_escalon` lo normaliza por el Sn disponible (inventario previo + cargado) y vale en ambas
fases (a diferencia de T3, que en Fusión se inflaba por el Sn entrante). `feo_extraido_est_kg` es la señal de
sobre-reducción **en masa**.""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
t = df_base.groupby(["fase_proceso", "orden_escalon_fase"])[["sn_extraido_est_kg", "frac_sn_extraido_escalon", "feo_extraido_est_kg"]].median()
t.index = [f"{'F' if f=='Fusión' else 'R'}{o}" for f, o in t.index]
(t["sn_extraido_est_kg"] / 1000).plot.bar(ax=axes[0], color="#4C72B0"); axes[0].set_title("Sn extraído de la escoria por escalón (t, mediana)")
t["frac_sn_extraido_escalon"].plot.bar(ax=axes[1], color="#55A868"); axes[1].set_title("Fracción del Sn disponible extraída (mediana)")
d = df_base.loc[R]; axes[2].scatter(d["d_ley_feo_escoria_pct"], d["feo_extraido_est_kg"] / 1000, s=6, alpha=0.4)
axes[2].axhline(0, color="k", lw=0.5); axes[2].axvline(0, color="k", lw=0.5)
axes[2].set_xlabel("Δ%FeO (puntos, cierre de composición)"); axes[2].set_ylabel("FeO extraído (t, masa)"); axes[2].set_title("Reducción: el %FeO SUBE aunque el FeO en masa BAJE")
plt.tight_layout(); plt.show()
print("Reducción: Spearman Δ%FeO(pts) vs FeO extraído (kg):", round(spearmanr(d["d_ley_feo_escoria_pct"], d["feo_extraido_est_kg"], nan_policy="omit")[0], 3),
      "| fracción de escalones con Δ%FeO>0:", round((d["d_ley_feo_escoria_pct"] > 0).mean(), 2), "| con FeO extraído>0:", round((d["feo_extraido_est_kg"] > 0).mean(), 2))""")

md(r"""## Paso 59: ¿qué target y qué features predicen mejor el efecto de las acciones? (experimentos/06 y 11)

Tres lecturas honestas: (1) en Fusión el target en masa da R² lockbox ≈0.70 frente a ≈0.45 del Δ%Sn, **pero** el
baseline trivial "solo Sn cargado" ya da ≈0.71 — la química agrega ~0.03 de R² OOF; (2) en Reducción el inventario
previo solo ya da 0.97-0.98 (el Sn se extrae casi por completo en R0); (3) el valor de las features v3 está en la
**respuesta a las palancas** (Paso 62), en la escala física común (kg) y en la restricción térmica, no en el R².""")

code(r"""res06 = pd.read_csv(EXP / "06_resultados_targets_features.csv")
print("R2 lockbox (escala propia) — targets x feature sets x algoritmo")
display(res06.pivot_table(index=["fase", "target"], columns=["feature_set", "algoritmo"], values="r2_lockbox").round(3))
res11 = pd.read_csv(EXP / "11_sets_curados_resultados.csv")
print("\nSets curados vs baselines triviales (OOF = criterio de selección; lockbox = solo reporte)")
display(res11[["fase", "target", "set", "n_features", "r2_oof", "r2_lockbox", "mae_lockbox"]].round(3))""")

md(r"""## Paso 60: deriva de campaña — por qué más features NO mejoran el lockbox en Fusión (experimentos/12)

Los últimos 63 batches (agosto 2026, lockbox) operan al **final de la campaña de refractario**: espesor 218-228 mm
(mínimo de DEV 235), termocuplas de carcasa 378-410 °C (P95 de DEV 384), ley de concentrado 41.5% (P95 de DEV 41.4),
baño más frío (995 vs 1077 °C). Dentro de DEV el walk-forward no cae (≈0.53 para todos los sets), pero cualquier feature
que codifique la campaña extrapola mal en el lockbox. Implicación operativa: **reentrenar/recalibrar a medida que
avanza la campaña**, y preferir sets sin contexto de campaña para el modelo de ley de Fusión.""")

code(r"""log12 = (EXP / "12_diag_deriva_fusion_log.txt").read_text(encoding="utf-8")
filas = [dict(set=m.group(1).strip(), k=int(m.group(2)), r2_oof=float(m.group(3)), walk_forward=float(m.group(4)), lockbox=float(m.group(5)))
         for m in re.finditer(r"^(.+?)\s+k=\s*(\d+) R2 oof=([-\d.]+)\s+walk-forward=([-\d.]+)\s+lockbox=([-\d.]+)", log12, re.M)]
display(pd.DataFrame(filas))
ctx = ["espesor_ladrillo_norm_mm", "termocupla_media_celsius_prev", "ley_sn_conc_batch_pct", "temperatura_horno_celsius_prev"]
q = lambda s: s.quantile([.05, .5, .95]).round(1).tolist()
print(pd.DataFrame({c: {"dev P5/P50/P95": q(df_base.loc[F & df_base.Batch.isin(batches_dev), c]), "lockbox P5/P50/P95": q(df_base.loc[F & df_base.Batch.isin(batches_lockbox), c])} for c in ctx}).T)""")

md(r"""## Paso 61: configuración final v3 y validación (OOF dev / lockbox) por fase y target

`modelo_predictivo_v3.FEATURES_V3_POR_DEFECTO` (sets curados, interpretables, con signos SHAP verificados en
experimentos/11) y `TARGETS_V3`: `sn_kg` (objetivo), `sn_pts` (continuidad v1/v2), `feo_kg` (sobre-reducción en
masa, solo Reducción), `feo_pts` (informativo) y `dT` (restricción térmica).""")

code(r"""tabla = mp3.tabla_validacion_v3(df) if RECALCULAR else pd.read_csv(EXP / "10_tabla_validacion_v3.csv").set_index(["fase", "target"])
display(tabla.round(3))
cob = pd.read_csv(EXP / "10_cobertura_cvplus_v3.csv"); print("Cobertura CV+ en lockbox (nominal 80%/90%):"); display(cob.round(3))
for fase in mp3.FASES:
    for k, v in mp3.FEATURES_V3_POR_DEFECTO[fase].items():
        print(f"{fase:9s} {k:7s} ({len(v):2d}): {v}")""")

md(r"""## Paso 62: efecto causal de las palancas dado el estado (DML cross-fitted, experimentos/07 y 10)

Efectos por 1 desviación estándar de cada palanca, controlando el estado v3 (inventario, leyes previas, temperatura,
refractario, ensayos del batch). Con el target **en masa**, en Fusión aparecen efectos ROBUSTOS que el target de ley no
identificaba: carbón (+), gas natural (+), tasa de carga (+), pellets y dross (+), cal/carga (−) y posición de lanza (−).""")

code(r"""ef = pd.read_csv(EXP / "07_efectos_causales_v3.csv")
display(ef[["fase", "target", "accion", "n", "theta_por_unidad", "ci95_lo", "ci95_hi", "efecto_por_1sd_accion", "efecto_por_1sd_en_sd_target", "veredicto"]].round(4))
sub = ef[ef.target == "sn_extraido_est_kg"]
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, fase in zip(axes, mp3.FASES):
    s = sub[sub.fase == fase].set_index("accion")
    err = np.vstack([s["efecto_por_1sd_accion"] - s["ci95_lo"] * s["efecto_por_1sd_accion"] / s["theta_por_unidad"].replace(0, np.nan),
                     s["ci95_hi"] * s["efecto_por_1sd_accion"] / s["theta_por_unidad"].replace(0, np.nan) - s["efecto_por_1sd_accion"]]).clip(0)
    colores = ["#2ca02c" if v == "ROBUSTO" else ("#ff7f0e" if v == "SIGNIFICATIVO_MODERADO" else "#bbbbbb") for v in s["veredicto"]]
    ax.barh(s.index, s["efecto_por_1sd_accion"], xerr=err, color=colores); ax.axvline(0, color="k", lw=0.7)
    ax.set_title(f"{fase}: kg de Sn extraído por +1 sd de la palanca (DML, IC95%)"); ax.set_xlabel("kg Sn / escalón")
plt.tight_layout(); plt.show()
conj = pd.read_csv(EXP / "10_efectos_palancas_conjunto_v3.csv"); print("DML CONJUNTO (todas las palancas a la vez):"); display(conj.round(3))""")

md(r"""## Paso 63: el precio físico de sobre-reducir — λ_Fe (kg Sn a dross por kg FeO reducido)

A nivel batch, el FeO extraído de la escoria en Reducción (masa, trazador CaO) se asocia al Sn que termina en dross de
Fe: OLS `Sn_dross (t) ~ FeO_extraído (t)` da pendiente ≈0.3 (IC95% 0.08-0.49, p=0.007). Es el peso `lambda_fe` del
objetivo v3, en las mismas unidades que el Sn extraído. La regresión explica poco (R² 0.03: el KPI de batch es
ruidoso) pero fija el orden de magnitud con sentido físico.""")

code(r"""import statsmodels.api as sm
b = pd.DataFrame({"dross_t": df.groupby("Batch")["sn_en_dross_fe_batch_t"].first(),
                  "feo_extr_R_t": df.loc[R].groupby("Batch")["feo_extraido_est_kg"].sum() / 1000,
                  "sel_masa": df.loc[R].groupby("Batch")["sn_extraido_est_kg"].sum() / df.loc[R].groupby("Batch")["feo_extraido_est_kg"].sum(),
                  "rha": df.groupby("Batch")["rendimiento_ha_batch"].first()}).dropna()
b = b[b.feo_extr_R_t.between(*b.feo_extr_R_t.quantile([.02, .98]))]
m = sm.OLS(b.dross_t, sm.add_constant(b[["feo_extr_R_t"]])).fit(); print(m.summary().tables[1])
fig, ax = plt.subplots(figsize=(6.5, 4)); ax.scatter(b.feo_extr_R_t, b.dross_t, s=10, alpha=0.5)
xs = np.linspace(b.feo_extr_R_t.min(), b.feo_extr_R_t.max(), 10); ax.plot(xs, m.params.iloc[0] + m.params.iloc[1] * xs, "r-")
ax.set_xlabel("FeO extraído en Reducción (t, masa)"); ax.set_ylabel("Sn a dross de Fe (t)"); ax.set_title(f"λ_Fe ≈ {m.params.iloc[1]:.2f} kg Sn / kg FeO (datos observados)"); plt.show()
print("Spearman selectividad en masa (Sn/FeO extraídos) vs rendimiento HA:", round(spearmanr(b.sel_masa, b.rha, nan_policy='omit')[0], 3))""")

md(r"""## Paso 63b: ¿conviene extraer Sn durante la Fusión? La evidencia de batch dice que NO (λ_F)

El Sn extraído de la escoria a lo largo del batch es ≈ fijo (Sn cargado − ≈1 t en la escoria final): extraer más en Fusión
solo deja menos para Reducción. Lo que sí varía entre batches es **cuándo** se extrae. La fracción del Sn cargado que ya
salió de la escoria al fin de Fusión (`F_avance_final`) **reduce** la fracción a metal (OLS −0.143, t=−3.7, controlando
ley del concentrado, refractario, temperatura y FeO reducido) y **aumenta** el dross (+0.085, p=0.004); cada t de Sn
conservada en la escoria al fin de Fusión suma ≈0.13 t de metal (p=0.001). Es el principio de dos etapas del horno de Sn:
fundir oxidando manteniendo el Sn como SnO₂ en la escoria, y reducirlo después con selectividad controlada. Por eso el
objetivo de Fusión en v3 penaliza la extracción prematura (`w_sn = −λ_F = −0.13`).""")

code(r"""gF, gR = df.loc[F].groupby("Batch"), df.loc[R].groupby("Batch"); g = df.groupby("Batch")
bb = pd.DataFrame({"feedsn_t": g["feed_Sn_kgf"].sum() / 1000, "metal": g["sn_en_metal_crudo_batch_t"].first(), "dross": g["sn_en_dross_fe_batch_t"].first(),
                   "F_avance_final": 1 - gF["sn_inventario_escoria_est_kg"].last() / gF["feed_Sn_kgf"].sum(), "F_sn_esc_final_t": gF["sn_inventario_escoria_est_kg"].last() / 1000,
                   "ley_conc": g["ley_sn_conc_batch_pct"].first(), "esp": g["espesor_ladrillo_norm_mm"].first(), "F_T_mean": gF["temperatura_horno_celsius"].mean(),
                   "R_feo_extr_t": gR["feo_extraido_est_kg"].sum() / 1000}).dropna()
bb["f_metal"] = bb.metal / bb.feedsn_t; bb["f_dross"] = bb.dross / bb.feedsn_t
print(sm.OLS(bb.f_metal, sm.add_constant(bb[["F_avance_final", "ley_conc", "esp", "F_T_mean", "R_feo_extr_t"]])).fit().summary().tables[1])
print(sm.OLS(bb.f_dross, sm.add_constant(bb[["F_avance_final", "R_feo_extr_t", "ley_conc"]])).fit().summary().tables[1])
print(sm.OLS(bb.metal, sm.add_constant(bb[["F_sn_esc_final_t", "feedsn_t"]])).fit().summary().tables[1])
bb["q"] = pd.qcut(bb.F_avance_final, 5, labels=False)
display(bb.groupby("q")[["F_avance_final", "f_metal", "f_dross", "R_feo_extr_t"]].mean().round(3).rename_axis("quintil de avance al fin de Fusión"))""")

md(r"""## Paso 64: prescriptor v3 — recomendaciones por escalón en kg de metal equivalente

`J_R = Sn extraído [kg] − λ_Fe·FeO reducido [kg] − costos (C, GN, O₂, tiempo) − w_T·(T fuera de P5-P95) − w_U·ancho CV+ − w_S·(1 − soporte)`
y `J_F = −λ_F·Sn extraído [kg] − costos − …` (Paso 63b). En Reducción el optimizador mueve carbón, GN, O₂, aire y duración
planificada; en Fusión solo carbón y cal (los gases, la carga y la duración quedan en el plan del operador porque el
modelo térmico de Fusión es débil). Restricción dura de soporte histórico (≥ P10) y penalización blanda. Cada escalón
parte de su estado real (sin encadenar); el resumen del batch está en **kg de metal equivalente** (−λ_F·ΔSn_Fusión +
ΔSn_Reducción − λ_Fe·ΔFeO_Reducción), nunca como suma de "Sn extraído".""")

code(r"""sistema = mp3.entrenar_sistema_v3(df)
pol = pd.read_csv(EXP / "10_politica_lockbox_v3.csv").set_index("Batch"); print("Política recomendada vs histórica en 8 batches del lockbox (kg Sn, estimado por el modelo):"); display(pol.round(1))
batch = pol.index[0]
rep = mp3.reporte_recomendaciones_batch_v3(sistema, df, batch, n_candidatos=150)
cols = ["fase", "orden_escalon_fase", "ley_sn_escoria_pct_prev", "sn_extraido_real_kg", "pred_sn_kg_historico", "pred_sn_kg_recomendado",
        "pred_feo_reducido_kg_historico", "pred_feo_reducido_kg_recomendado", "pred_T_recomendado", "support_recomendado", "uplift_J_kgSn",
        "hist__tasa_feed_Carbon_kg_min", "rec__tasa_feed_Carbon_kg_min", "hist__tasa_gn_nm3_min", "rec__tasa_gn_nm3_min",
        "hist__tasa_o2_nm3_min", "rec__tasa_o2_nm3_min", "hist__duracion_plan_min", "rec__duracion_plan_min"]
display(rep[[c for c in cols if c in rep.columns]].round(1)); print(mp3.resumen_reporte_batch_v3(rep))""")

code(r"""row = df_base[(df_base.Batch == batch) & (df_base.fase_proceso == "Reducción")].sort_values("fecha_inicio").iloc[0]
state, action, context = mp3.fila_a_state_action_context("Reducción", row)
print(f"Sensibilidad local del modelo en {batch} R0 (efecto de cada palanca DADO este estado):")
display(mp3.sensibilidad_local_v3(sistema, "Reducción", state, action, context).round(3))
sim = mp3.simulate_action_v3(sistema, "Reducción", state, action, context)
print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in sim.items() if k not in ("derivadas",)})""")

md(r"""## Paso 65: el rendimiento del batch y sus dos canales de pérdida (experimentos/08)

`rendimiento_ha_batch` se descompone en Sn a **dross de Fe** (12.7% ± 3.9 del Sn cargado; ρ=−0.76 con el rendimiento)
y Sn a **polvo** (18.3% ± 4.0; ρ=−0.54), independientes entre sí (ρ=0.02). El canal de dross responde a la
sobre-reducción de FeO en Reducción (caída de FeO, selectividad en masa); el de polvo no se explica con los agregados
disponibles. El KPI de batch sigue siendo poco predecible (R² lockbox ≈0, Spearman 0.25-0.40): el valor prescriptivo
está en el escalón, con el eslabón al batch dado por λ_Fe y el balance de Sn en masa.""")

code(r"""cors = pd.read_csv(EXP / "08_spearman_batch.csv", index_col=0)
display(cors.loc[cors.abs().max(axis=1) > 0.2].round(3).sort_values("frac_sn_dross"))
display(pd.read_csv(EXP / "08_resultados_batch.csv")[["target", "modelo", "r2_oof", "r2_walkforward", "r2_lockbox", "rho_lockbox"]].round(3))""")

md(r"""## Paso 66: conclusiones y limitaciones

1. **Datos**: `feed_CaO_kgh` corregido (tolva 6); el Sn por tolva es `tmh × ley del batch`; `duracion_plan_min` es la
   acción de tiempo limpia; sin heel entre batches; Rendimiento HA ≡ rendimiento_proxy_batch.
2. **Masa de escoria por trazador CaO** cierra el cuello de botella declarado desde hallazgos.md 6b: targets en kg,
   inventario de Sn/FeO, avance de reducción, y muestra que el Δ%FeO en puntos **no** mide sobre-reducción en Reducción
   (sube por cierre de composición cuando sale el Sn).
3. **Predicción**: Reducción sigue excelente (Δ%Sn lockbox ≈0.97; kg ≈0.98); Fusión mejora en kg (≈0.70) pero casi
   todo es la carga; el ΔT de Reducción pasa a ser predecible (≈0.3-0.5). Deriva de campaña en el lockbox → reentrenar.
4. **Efecto de las acciones dado el estado**: DML con el target en masa identifica palancas ROBUSTAS en Fusión (carbón +,
   GN +, cal/carga −, lanza −, carga/pellets/dross +) y confirma carbón/GN/O2 en Reducción.
5. **Dirección de Fusión**: extraer Sn prematuramente cuesta ≈0.13 kg de metal por kg (evidencia de batch, t=−3.7):
   el prescriptor de Fusión conserva el Sn en la escoria (menos carbón / más cal dentro del soporte histórico) y el de
   Reducción lo extrae con selectividad (λ_Fe ≈ 0.3 kg Sn/kg FeO). Objetivo en kg de metal equivalente, ventana
   térmica, intervalos CV+ conservadores, soporte histórico duro (P10) y blando; reporte por batch y sensibilidad local.
6. **Pendientes de planta**: sentido físico de la posición de lanza; por qué la tolva 4 (mineral de Fe) tiene ~40% Sn;
   CaO de la ganga (para calibrar la masa absoluta de escoria); precios reales para los costos del objetivo.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, RAIZ / "modelo_v3_masa_escoria_y_prescriptor.ipynb")
print("notebook escrito")
