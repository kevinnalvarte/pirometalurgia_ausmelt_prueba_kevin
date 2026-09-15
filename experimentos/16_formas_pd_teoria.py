"""Experimento 16 -- formas aprendidas (PD/ALE/SHAP) por los modelos v3 vs teoria piro-
metalurgica (monotona vs optimo interior vs plana), y forma del objetivo J por palanca.

Ver experimentos/ITERACION_4_diseno.md (tabla teorica de formas esperadas), hallazgos.md
12 (optimo interior de la lanza) y 14.5 (signos SHAP v3), modelo_predictivo_v3.py.

Salidas: 16_formas_pd.csv, 16_interacciones.csv, 16_forma_J.csv, 16_pd_curvas.csv,
         16_log.txt, figuras en figs_16/.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import feature_engineering as fe  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.inspection import partial_dependence  # noqa: E402
import shap  # noqa: E402

SALIDA = RAIZ / "experimentos"
FIGS = SALIDA / "figs_16"
FIGS.mkdir(exist_ok=True, parents=True)
LOGF = open(SALIDA / "16_log.txt", "w", encoding="utf-8")
T0 = time.time()


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    print(msg, file=LOGF, flush=True)


def t():
    return f"[{time.time()-T0:6.1f}s]"


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

# =============================================================================
# 0) Datos
# =============================================================================
log(t(), "cargando dataset cache...")
df = pd.read_pickle(RAIZ / "experimentos/cache/df_v3.pkl")
df_base = mp.dataset_base_modelo(df)
BATCHES_DEV, BATCHES_LOCKBOX = mp.split_dev_lockbox(df)
log(t(), f"df {df.shape}, base {df_base.shape}, batches dev={len(BATCHES_DEV)} lockbox={len(BATCHES_LOCKBOX)}")

# Pool amplio de features teoricamente relevantes (se filtran por seguridad y cobertura)
POOL_TEORICO = ["basicidad_B2_prev", "temperatura_horno_celsius_prev", "posicion_vertical_lanza_mm_prev",
                "duracion_plan_min", "tasa_feed_CaO_kg_min", "relacion_C_Sn_carga", "tasa_o2_nm3_min",
                "tasa_aire_nm3_min", "oxygen_enrichment_pct", "ley_feo_escoria_pct_prev", "avance_reduccion_sn_prev"]
POOL_SEGURO = fe.filtrar_features_seguras(POOL_TEORICO)
assert POOL_SEGURO == POOL_TEORICO, f"pool inseguro: {set(POOL_TEORICO) - set(POOL_SEGURO)}"

# relacion_C_Sn_carga tiene ~72% de nulos en Reduccion (no hay carga fresca de Sn en esa fase,
# ver docstring recalcular_derivadas_v3): se excluye del pool amplio de Reduccion para no
# perder el 72% de la muestra en el resto de las features auditadas.
POOL_POR_FASE = {
    "Fusión": POOL_SEGURO,
    "Reducción": [f for f in POOL_SEGURO if f != "relacion_C_Sn_carga"],
}


def features_pool_amplio(fase: str, feats_default: list[str]) -> list[str]:
    extra = [f for f in POOL_POR_FASE[fase] if f not in feats_default]
    return feats_default + extra


# =============================================================================
# 1) Especificaciones de modelo (entrenados SOLO en DEV, sin CV -- auditoria de forma)
# =============================================================================
SPECS = []
for fase, clave in [("Fusión", "sn_kg"), ("Reducción", "sn_kg"), ("Reducción", "feo_kg")]:
    feats_def = mp3.FEATURES_V3_POR_DEFECTO[fase][clave]
    target_col = mp3.TARGETS_V3[fase][clave]
    SPECS.append(dict(nombre=f"{fase}_{clave}_default", fase=fase, clave=clave, target_col=target_col,
                       features=feats_def, monotono=True))
    if clave == "sn_kg":
        SPECS.append(dict(nombre=f"{fase}_{clave}_libre", fase=fase, clave=clave, target_col=target_col,
                           features=feats_def, monotono=False))
    feats_amplio = features_pool_amplio(fase, feats_def)
    SPECS.append(dict(nombre=f"{fase}_{clave}_pool_amplio", fase=fase, clave=clave, target_col=target_col,
                       features=feats_amplio, monotono=True))

log(t(), f"{len(SPECS)} especificaciones de modelo:")
for s in SPECS:
    log("  -", s["nombre"], f"({len(s['features'])} features, monotono={s['monotono']})")

MODELOS = {}   # nombre -> dict(model, d, feats, target_col, fase, clave, sd_target)
for s in SPECS:
    fase, feats, target_col = s["fase"], s["features"], s["target_col"]
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(BATCHES_DEV)]
    d = sub.dropna(subset=feats + [target_col]).reset_index(drop=True)
    m = mp3.construir_modelo_v3(feats, target_col, s["monotono"])
    m.fit(d[feats], d[target_col])
    MODELOS[s["nombre"]] = dict(model=m, d=d, feats=feats, target_col=target_col, fase=fase, clave=s["clave"],
                                sd_target=float(d[target_col].std()), n=len(d))
    log(t(), f"  entrenado {s['nombre']}: n={len(d)}, sd_target={MODELOS[s['nombre']]['sd_target']:.1f}")

# =============================================================================
# 2) Roles STATE/CONTROL/CONTEXT por fase (para la tabla de salida)
# =============================================================================

def rol_de(fase: str, feature: str) -> str:
    if feature in mp3.ACCIONES_PRIMITIVAS_V3[fase]:
        return "CONTROL"
    if feature in mp3.STATE_V3[fase]:
        return "STATE"
    if feature in mp3.CONTEXT_V3[fase]:
        return "CONTEXT"
    if feature in ("feed_Sn_kgf", "tasa_feed_total_kg_min", "ley_sn_carga_pct"):
        return "ACTION_PLAN"
    if feature.startswith("interaccion_") or feature.startswith("relacion_") or feature in (
            "exceso_o2_combustion_pct", "oxygen_enrichment_pct", "frac_carga_secundaria"):
        return "DERIVED_CONTROL"
    return "DERIVED"


# =============================================================================
# 3) Tabla teorica: (feature, [fase/clave opcional]) -> (texto, tipo)
#    tipo en {mono+, mono+_sat, mono+_debil, mono-, concava, optimo_interior,
#             no_monotona_valle, sin_restriccion, no_cubierta}
# =============================================================================
TEORIA_GENERICA: dict[str, tuple[str, str]] = {
    "tasa_feed_Carbon_kg_min": ("+ (SnO2+2C->Sn), saturante al agotarse el Sn removible", "mono+_sat"),
    "interaccion_C_x_Sn_prev": ("familia carbon: + saturante (interaccion carbon x inventario Sn)", "mono+_sat"),
    "interaccion_C_x_FeO_prev": ("familia carbon: + debil sobre FeO (crece con avance de reduccion)", "mono+_debil"),
    "interaccion_C_x_exceso_o2": ("familia carbon: sin prediccion explicita (interaccion de 2do orden)", "no_cubierta"),
    "cum_feed_Carbon_kg_prev": ("familia carbon: + acumulado, saturante", "mono+_sat"),
    "tasa_gn_nm3_min": ("+ (calor + gas reductor) con meseta termica", "mono+_sat"),
    "tasa_feed_CaO_kg_min": ("- (dilucion de la escoria, menor T) en Fusion", "mono-"),
    "relacion_CaO_carga": ("- (dilucion de la escoria, menor actividad relativa) en Fusion", "mono-"),
    "duracion_plan_min": ("+ con rendimientos decrecientes (cinetica 1er orden): concava", "concava"),
    "basicidad_B2_prev": ("optimo interior (~1.4) en Fusion", "optimo_interior"),
    "temperatura_horno_celsius_prev": ("+ (cinetica, viscosidad) dentro de la ventana", "mono+"),
    "posicion_vertical_lanza_mm_prev": ("no monotona: valle 3300-4200mm en d%Sn (hallazgos 12) -> "
                                        "pico de kg extraidos en esa zona", "optimo_interior"),
    "sn_inventario_escoria_est_kg_prev": ("+ (fuerza impulsora de la reaccion)", "mono+"),
    "ley_sn_escoria_pct_prev": ("+ (fuerza impulsora de la reaccion)", "mono+"),
    "ley_feo_escoria_pct_prev": ("ambiguo (baja actividad de SnO pero matriz mas fluida): sin restriccion", "sin_restriccion"),
    "feed_Sn_kgf": ("+ (ACTION de plan, mas Sn cargado en el escalon)", "mono+"),
    "tasa_feed_total_kg_min": ("+ (ACTION de plan)", "mono+"),
    "exceso_o2_combustion_pct": ("- en Reduccion (atmosfera oxidante frena la reduccion)", "mono-"),
    "oxygen_enrichment_pct": ("analogo a exceso_o2_combustion_pct (no explicito en la tabla)", "mono-"),
    "avance_reduccion_sn_prev": ("no en la tabla teorica; extrapolado de hallazgos 14.5 (SHAP -0.96): "
                                 "- (queda menos Sn removible)", "mono-"),
    "relacion_C_Sn_carga": ("familia carbon (dosis relativa al Sn cargado): + saturante, solo aplica con carga fresca", "mono+_sat"),
    "tasa_o2_nm3_min": ("primitiva de la lanza; la teoria se formula sobre exceso_o2_combustion_pct (derivada)", "no_cubierta"),
    "tasa_aire_nm3_min": ("primitiva de la lanza; la teoria se formula sobre exceso_o2_combustion_pct (derivada)", "no_cubierta"),
}
# Overrides especificos por (fase, clave, feature)
TEORIA_OVERRIDE: dict[tuple[str, str, str], tuple[str, str]] = {
    ("Fusión", "sn_kg", "exceso_o2_combustion_pct"): ("~0 en Fusion (regimen oxidante ya dado, no debiera dominar)", "sin_restriccion"),
    ("Fusión", "sn_kg", "oxygen_enrichment_pct"): ("~0 en Fusion (analogo a exceso_o2)", "sin_restriccion"),
    ("Fusión", "sn_kg", "tasa_feed_Carbon_kg_min"): ("Fusion: + pero el efecto propio no se identifica separado de la "
                                                      "carga en DML conjunto (hallazgos 14.5) -- se espera + debil o plano", "mono+_debil"),
    ("Fusión", "sn_kg", "interaccion_C_x_Sn_prev"): ("Fusion: + (misma reserva que carbon; efecto propio fragil, ver DML 14.5)", "mono+_debil"),
    ("Reducción", "sn_kg", "basicidad_B2_prev"): ("Reduccion: ~0 (sin efecto claro sobre Sn)", "sin_restriccion"),
    ("Reducción", "feo_kg", "basicidad_B2_prev"): ("Reduccion: ~0 (sin efecto claro)", "sin_restriccion"),
    ("Reducción", "feo_kg", "ley_feo_escoria_pct_prev"): ("+ por analogia con inventario de Sn (mas FeO disponible, "
                                                           "mas se puede extraer); no esta en la tabla explicita", "mono+"),
    ("Reducción", "feo_kg", "sn_inventario_escoria_est_kg_prev"): ("- (el Sn compite por el poder reductor, "
                                                                    "deberia retrasar/competir con la reduccion de FeO)", "mono-"),
    ("Reducción", "feo_kg", "ley_sn_escoria_pct_prev"): ("- (compite con la reduccion de FeO)", "mono-"),
    ("Reducción", "feo_kg", "tasa_gn_nm3_min"): ("+ (GN es MENOS selectivo que el carbon: reduce Sn y FeO a la vez)", "mono+"),
    ("Reducción", "sn_kg", "tasa_feed_CaO_kg_min"): ("Reduccion: sin carga fresca de fundente activo: n/a", "no_cubierta"),
    ("Reducción", "feo_kg", "tasa_feed_CaO_kg_min"): ("Reduccion: sin carga fresca de fundente activo: n/a", "no_cubierta"),
}


def teoria_de(fase: str, clave: str, feature: str) -> tuple[str, str]:
    if (fase, clave, feature) in TEORIA_OVERRIDE:
        return TEORIA_OVERRIDE[(fase, clave, feature)]
    if feature in TEORIA_GENERICA:
        return TEORIA_GENERICA[feature]
    return ("no cubierta explicitamente por la tabla teorica de ITERACION_4_diseno.md (feature "
            "estructural/de contexto); se reporta como auditoria exploratoria", "no_cubierta")


def evaluar_concuerda(tipo_teorico: str, forma: str) -> tuple[str, str]:
    """Devuelve (concuerda, comentario_corto)."""
    if tipo_teorico in ("no_cubierta", "sin_restriccion"):
        return "n/a", "sin prediccion falsable en la tabla teorica"
    mono_pos = forma == "monótona +"
    mono_neg = forma == "monótona -"
    opt_max = forma == "óptimo interior (máximo)"
    opt_min = forma == "óptimo interior (mínimo)"
    plana = forma == "plana"
    compleja = forma == "no monótona compleja"
    if tipo_teorico in ("mono+", "mono+_sat"):
        if mono_pos:
            return "sí", "monotona + como se esperaba"
        if opt_max:
            return "parcial", "sube y luego cae (posible sobre-ajuste o efecto real de saturacion/reversion)"
        if plana:
            return "no", "el modelo no aprendio efecto (rango < 5% sd) pese a teoria +"
        if mono_neg or opt_min:
            return "no", "signo opuesto al esperado"
        return "parcial", "forma compleja, sin monotonia limpia"
    if tipo_teorico == "mono+_debil":
        if mono_pos or opt_max:
            return "sí", "efecto + presente (teoria lo esperaba debil)"
        if plana:
            return "parcial", "efecto nulo: consistente con 'debil', pero no confirma la direccion"
        if mono_neg or opt_min:
            return "no", "signo opuesto al esperado (aun siendo 'debil')"
        return "parcial", "forma compleja"
    if tipo_teorico == "mono-":
        if mono_neg:
            return "sí", "monotona - como se esperaba"
        if opt_min:
            return "parcial", "baja y luego sube (posible efecto real o ruido en el extremo)"
        if plana:
            return "no", "el modelo no aprendio efecto (rango < 5% sd) pese a teoria -"
        if mono_pos or opt_max:
            return "no", "signo opuesto al esperado"
        return "parcial", "forma compleja, sin monotonia limpia"
    if tipo_teorico == "concava":
        if opt_max:
            return "sí", "concava con maximo interior, como se esperaba"
        if mono_pos:
            return "parcial", "sigue subiendo: no se observa el quiebre de rendimientos decrecientes en el rango historico"
        if plana or mono_neg or opt_min or compleja:
            return "no" if (mono_neg or opt_min) else "parcial", "forma no coincide con concava clara"
        return "parcial", "forma compleja"
    if tipo_teorico == "optimo_interior":
        if opt_max or opt_min:
            return "sí", "optimo interior detectado, como se esperaba"
        if compleja:
            return "parcial", "no monotona pero sin un unico optimo interior limpio (posible ruido)"
        if plana:
            return "no", "el modelo no aprendio forma alguna (plana) pese a esperar un optimo interior"
        return "no", "el modelo aprendio una forma monotona simple, no un optimo interior"
    return "n/a", ""


# =============================================================================
# 4) PD, ALE y SHAP-dependence 1D
# =============================================================================

def calcular_pd(model, d: pd.DataFrame, feats: list[str], feature: str, n_grid: int = 20):
    try:
        res = partial_dependence(model, d[feats], [feature], kind="average",
                                 grid_resolution=n_grid, percentiles=(0.02, 0.98))
    except ValueError:
        return None
    x = np.asarray(res["grid_values"][0], dtype=float)
    y = np.asarray(res["average"][0], dtype=float)
    if len(x) < 3:
        return None
    return x, y


def calcular_ale(model, d: pd.DataFrame, feats: list[str], feature: str, n_bins: int = 20):
    x_full = d[feature].to_numpy(dtype=float)
    qs = np.unique(np.quantile(x_full, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 4:
        return None
    qs_ext = qs.copy()
    qs_ext[0] -= 1e-9
    qs_ext[-1] += 1e-9
    bin_idx = np.clip(np.digitize(x_full, qs_ext, right=True) - 1, 0, len(qs_ext) - 2)
    Xlo, Xhi = d[feats].copy(), d[feats].copy()
    Xlo[feature] = qs_ext[bin_idx]
    Xhi[feature] = qs_ext[bin_idx + 1]
    delta = model.predict(Xhi[feats]) - model.predict(Xlo[feats])
    nb = len(qs_ext) - 1
    sums = np.zeros(nb)
    counts = np.zeros(nb)
    np.add.at(sums, bin_idx, delta)
    np.add.at(counts, bin_idx, 1)
    bin_mean = np.divide(sums, counts, out=np.zeros(nb), where=counts > 0)
    edges_val = np.concatenate([[0.0], np.cumsum(bin_mean)])
    mid_val = (edges_val[:-1] + edges_val[1:]) / 2.0
    weighted_mean = np.average(mid_val, weights=counts) if counts.sum() > 0 else 0.0
    ale_centered = mid_val - weighted_mean
    mid_x = (qs[:-1] + qs[1:]) / 2.0
    return mid_x, ale_centered


def calcular_shap_deciles(shap_col: np.ndarray, x: np.ndarray, n_bins: int = 10):
    try:
        bins = pd.qcut(x, n_bins, duplicates="drop")
    except ValueError:
        return None
    dfm = pd.DataFrame({"bin": bins, "shap": shap_col, "x": x})
    g = dfm.groupby("bin", observed=True).agg(x_mediana=("x", "median"), shap_media=("shap", "mean"),
                                              frac_pos=("shap", lambda s: float((s > 0).mean())),
                                              n=("shap", "size")).reset_index(drop=True)
    return g["x_mediana"].to_numpy(), g["shap_media"].to_numpy(), g["frac_pos"].to_numpy()


def clasificar_forma(x: np.ndarray, y: np.ndarray, sd_target: float):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    orden = np.argsort(x)
    x, y = x[orden], y[orden]
    rango = float(np.nanmax(y) - np.nanmin(y))
    rango_rel_sd = rango / sd_target if sd_target > 1e-9 else np.nan
    if not np.isfinite(rango_rel_sd) or rango_rel_sd < 0.05:
        return "plana", float(x[len(x) // 2]), rango_rel_sd
    rho = spearmanr(x, y).statistic
    if np.isfinite(rho) and abs(rho) >= 0.9:
        forma = "monótona +" if rho > 0 else "monótona -"
        x_ext = float(x[int(np.argmax(y))]) if rho > 0 else float(x[int(np.argmin(y))])
        return forma, x_ext, rango_rel_sd
    n = len(x)
    candidatos = []
    for kind, idx in (("máximo", int(np.argmax(y))), ("mínimo", int(np.argmin(y)))):
        pos_frac = idx / (n - 1)
        if 0.15 <= pos_frac <= 0.85:
            left_diff = abs(y[idx] - y[0])
            right_diff = abs(y[idx] - y[-1])
            if left_diff > 0.15 * rango and right_diff > 0.15 * rango:
                candidatos.append((kind, idx, min(left_diff, right_diff)))
    if candidatos:
        candidatos.sort(key=lambda tt: -tt[2])
        kind, idx, _ = candidatos[0]
        return f"óptimo interior ({kind})", float(x[idx]), rango_rel_sd
    idx_ext = int(np.argmax(np.abs(y - y.mean())))
    return "no monótona compleja", float(x[idx_ext]), rango_rel_sd


# =============================================================================
# 5) Loop principal: por modelo, por feature -> PD/ALE/SHAP + clasificacion + teoria
# =============================================================================
filas_formas = []
filas_curvas = []

for nombre, info in MODELOS.items():
    model, d, feats, target_col = info["model"], info["d"], info["feats"], info["target_col"]
    fase, clave, sd_target = info["fase"], info["clave"], info["sd_target"]
    log(t(), f"PD/ALE/SHAP para {nombre} ({len(feats)} features, n={len(d)})...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(d[feats])
    n_cols_fig = 3
    for j, feature in enumerate(feats):
        pd_res = calcular_pd(model, d, feats, feature)
        ale_res = calcular_ale(model, d, feats, feature)
        shap_res = calcular_shap_deciles(shap_vals[:, j], d[feature].to_numpy())

        if pd_res is not None:
            xg, yg = pd_res
            for xx, yy in zip(xg, yg):
                filas_curvas.append(dict(modelo=nombre, feature=feature, fuente="pd", x=xx, valor=yy))
        if ale_res is not None:
            xa, ya = ale_res
            for xx, yy in zip(xa, ya):
                filas_curvas.append(dict(modelo=nombre, feature=feature, fuente="ale", x=xx, valor=yy))
        if shap_res is not None:
            xs, ys, fp = shap_res
            for xx, yy, ff in zip(xs, ys, fp):
                filas_curvas.append(dict(modelo=nombre, feature=feature, fuente="shap_decil", x=xx, valor=yy, frac_pos=ff))

        # Clasificacion principal sobre ALE (mas robusto ante correlacion, ver docstring del
        # experimento); si ALE no esta disponible (feature casi constante) se usa PD.
        base_forma = ale_res if ale_res is not None else pd_res
        if base_forma is None:
            continue
        forma, x_extremo, rango_rel_sd = clasificar_forma(base_forma[0], base_forma[1], sd_target)
        forma_pd = clasificar_forma(pd_res[0], pd_res[1], sd_target)[0] if pd_res is not None else "n/d"
        acuerdo_pd_ale = "coincide" if forma_pd == forma else f"PD dice '{forma_pd}'"

        teoria_txt, tipo_teorico = teoria_de(fase, clave, feature)
        concuerda, com_concuerda = evaluar_concuerda(tipo_teorico, forma)

        comentario = f"{com_concuerda}; ALE vs PD: {acuerdo_pd_ale}"
        if shap_res is not None:
            frac_pos_extremos = f"fracpos_shap[0,-1]=({shap_res[2][0]:.2f},{shap_res[2][-1]:.2f})"
            comentario += f"; {frac_pos_extremos}"

        filas_formas.append(dict(
            modelo=nombre, fase=fase, target=clave, columna_target=target_col, feature=feature,
            rol=rol_de(fase, feature), n=info["n"], sd_target=round(sd_target, 2),
            rango_rel_sd=round(rango_rel_sd, 4) if np.isfinite(rango_rel_sd) else np.nan,
            forma=forma, forma_pd=forma_pd, x_extremo=round(x_extremo, 4),
            teoria=teoria_txt, tipo_teorico=tipo_teorico, concuerda=concuerda, comentario=comentario,
        ))

        # --- figura panel PD | ALE | SHAP-decil
        fig, axes = plt.subplots(1, n_cols_fig, figsize=(12, 3.2))
        if pd_res is not None:
            axes[0].plot(pd_res[0], pd_res[1], "o-", color="#1f77b4", ms=3)
            axes[0].set_title("PD")
        if ale_res is not None:
            axes[1].plot(ale_res[0], ale_res[1], "o-", color="#ff7f0e", ms=3)
            axes[1].axhline(0, color="gray", lw=0.6)
            axes[1].set_title("ALE")
        if shap_res is not None:
            axes[2].plot(shap_res[0], shap_res[1], "o-", color="#2ca02c", ms=3)
            axes[2].axhline(0, color="gray", lw=0.6)
            axes[2].set_title("SHAP por decil")
        for ax in axes:
            ax.set_xlabel(feature, fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{nombre} :: {feature}  [{forma}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / f"{nombre}__{feature}.png", dpi=110)
        plt.close(fig)

formas_df = pd.DataFrame(filas_formas)
formas_df.to_csv(SALIDA / "16_formas_pd.csv", index=False)
log(t(), f"16_formas_pd.csv -> {formas_df.shape}")

curvas_df = pd.DataFrame(filas_curvas)
curvas_df.to_csv(SALIDA / "16_pd_curvas.csv", index=False)
log(t(), f"16_pd_curvas.csv -> {curvas_df.shape}")

log(t(), "\n=== resumen concuerda por modelo ===")
log(formas_df.groupby(["modelo", "concuerda"]).size().unstack(fill_value=0).to_string())

# =============================================================================
# 6) Interacciones clave (PD 2D)
# =============================================================================
INTERACCIONES = [
    dict(nombre="carbon_x_inventarioSn", modelo="Reducción_sn_kg_default",
         fx="tasa_feed_Carbon_kg_min", fy="sn_inventario_escoria_est_kg_prev",
         lectura_hint="cinetica bimolecular: el efecto del carbon debe CRECER con el inventario de Sn"),
    dict(nombre="carbon_x_leyFeO", modelo="Reducción_feo_kg_default",
         fx="tasa_feed_Carbon_kg_min", fy="ley_feo_escoria_pct_prev",
         lectura_hint="el efecto del carbon sobre FeO extraido deberia crecer con la ley de FeO (mas sustrato)"),
    dict(nombre="gn_x_temperatura", modelo="Reducción_sn_kg_default",
         fx="tasa_gn_nm3_min", fy="temperatura_horno_celsius_prev",
         lectura_hint="el efecto del GN deberia ser mayor (o mas termico) a temperaturas bajas (calienta) o "
                      "sostenerse si domina el efecto reductor"),
    dict(nombre="CaO_x_basicidad", modelo="Fusión_sn_kg_pool_amplio",
         fx="tasa_feed_CaO_kg_min", fy="basicidad_B2_prev",
         lectura_hint="el efecto de cargar mas CaO deberia depender de si la basicidad ya esta cerca del optimo (~1.4)"),
]

filas_inter = []
for spec in INTERACCIONES:
    info = MODELOS[spec["modelo"]]
    model, d, feats = info["model"], info["d"], info["feats"]
    fx, fy = spec["fx"], spec["fy"]
    if fx not in feats or fy not in feats:
        log(t(), f"  [omitido] {spec['nombre']}: {fx} o {fy} no estan en {spec['modelo']}")
        continue
    res = partial_dependence(model, d[feats], [fx, fy], kind="average", grid_resolution=10, percentiles=(0.05, 0.95))
    gx, gy = res["grid_values"][0], res["grid_values"][1]
    avg = res["average"][0]  # shape (len(gx), len(gy))
    for i, xv in enumerate(gx):
        for jx, yv in enumerate(gy):
            filas_inter.append(dict(interaccion=spec["nombre"], modelo=spec["modelo"], fx=fx, fy=fy,
                                    x_val=xv, y_val=yv, pd_valor=avg[i, jx]))
    # lectura automatica: rango del efecto de fx (max-min sobre fx) para cada nivel de fy;
    # si ese rango crece con fy -> confirma interaccion tipo "el efecto de fx crece con fy"
    rango_por_fy = avg.max(axis=0) - avg.min(axis=0)
    rho_interaccion = spearmanr(gy, rango_por_fy).statistic
    lectura = (f"rango del efecto de {fx} por nivel de {fy}: {rango_por_fy.round(1).tolist()}; "
              f"Spearman(nivel_{fy}, rango_efecto_{fx})={rho_interaccion:.2f} -> "
              f"{'CRECE con ' + fy if rho_interaccion > 0.3 else ('DECRECE con ' + fy if rho_interaccion < -0.3 else 'sin relacion clara con ' + fy)}")
    log(t(), f"  {spec['nombre']}: {lectura}")
    for r in filas_inter:
        if r["interaccion"] == spec["nombre"]:
            r["lectura"] = lectura
            r["lectura_hint_teorica"] = spec["lectura_hint"]

    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.pcolormesh(gy, gx, avg, shading="auto", cmap="RdBu_r")
    ax.set_xlabel(fy)
    ax.set_ylabel(fx)
    ax.set_title(f"PD 2D: {spec['nombre']}", fontsize=9)
    fig.colorbar(c, ax=ax, label="PD")
    fig.tight_layout()
    fig.savefig(FIGS / f"interaccion__{spec['nombre']}.png", dpi=110)
    plt.close(fig)

inter_df = pd.DataFrame(filas_inter)
inter_df.to_csv(SALIDA / "16_interacciones.csv", index=False)
log(t(), f"16_interacciones.csv -> {inter_df.shape}")

# =============================================================================
# 7) Forma del objetivo J por palanca (sistema completo, no solo el target)
# =============================================================================
log(t(), "entrenando sistema v3 completo (mp3.entrenar_sistema_v3) para la forma de J...")
sistema = mp3.entrenar_sistema_v3(df)
log(t(), "sistema entrenado.")


def muestrear_escalones(sub: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    sub = sub.dropna(subset=["orden_escalon_fase", "avance_reduccion_sn_prev"]).copy()
    sub["avance_quintil"] = pd.qcut(sub["avance_reduccion_sn_prev"], 5, duplicates="drop", labels=False)
    rng = np.random.default_rng(seed)
    grupos = sub.groupby(["orden_escalon_fase", "avance_quintil"], observed=True)
    claves = list(grupos.groups.keys())
    rng.shuffle(claves)
    iters = {k: iter(grupos.get_group(k).sample(frac=1, random_state=seed).index) for k in claves}
    activos = list(claves)
    elegidos = []
    while len(elegidos) < n and activos:
        for k in list(activos):
            try:
                elegidos.append(next(iters[k]))
                if len(elegidos) >= n:
                    break
            except StopIteration:
                activos.remove(k)
    return sub.loc[elegidos[:n]]


filas_J = []
N_POR_FASE = {"Reducción": 30, "Fusión": 20}
N_GRID_J = 21

for fase in mp3.FASES:
    sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(sistema.batches_dev)]
    muestra = muestrear_escalones(sub, N_POR_FASE[fase], seed=0)
    log(t(), f"forma de J -- {fase}: muestra {len(muestra)} escalones "
             f"(orden_escalon_fase unicos={sorted(muestra['orden_escalon_fase'].unique())})")
    curvas_J_fase = {}  # palanca -> lista de (x_grid, hist_val, J_hist_idx)
    for _, row in muestra.iterrows():
        state, accion_base, context = mp3.fila_a_state_action_context(fase, row)
        if any(pd.isna(v) for v in accion_base.values()) or pd.isna(state.get("ley_sn_escoria_pct_prev")):
            continue
        for palanca in mp3.ACCIONES_OPTIMIZABLES_V3[fase]:
            lo, hi = sistema.limites_accion[fase][palanca]
            grid = np.linspace(lo, hi, N_GRID_J)
            candidatos = []
            for val in grid:
                cand = dict(accion_base)
                cand[palanca] = float(val)
                candidatos.append(cand)
            sim = mp3.objetivo_lote_v3(sistema, fase, state, candidatos, context)
            J = sim["J"].to_numpy()
            hist_val = float(accion_base[palanca])
            idx_hist = int(np.argmin(np.abs(grid - hist_val)))
            J_hist = J[idx_hist]
            rango_J = float(J.max() - J.min())
            idx_max = int(np.argmax(J))
            pos_frac = idx_max / (N_GRID_J - 1)
            escala = max(5.0, 0.02 * abs(J_hist))
            if rango_J < escala:
                tipo = "plana"
            elif pos_frac <= 0.10:
                tipo = "frontera_baja"
            elif pos_frac >= 0.90:
                tipo = "frontera_alta"
            elif 0.15 <= pos_frac <= 0.85:
                tipo = "óptimo_interior"
            else:
                tipo = "frontera_cercana"
            filas_J.append(dict(
                fase=fase, Batch=row["Batch"], orden_escalon_fase=row["orden_escalon_fase"],
                avance_reduccion_sn_prev=row.get("avance_reduccion_sn_prev"), palanca=palanca,
                valor_historico=hist_val, J_historico=float(J_hist), x_optimo=float(grid[idx_max]),
                J_optimo=float(J[idx_max]), rango_J=rango_J, tipo_forma=tipo,
                delta_relativo_optimo_vs_hist=float((grid[idx_max] - hist_val) / (hi - lo)) if hi > lo else np.nan,
                lo_grid=lo, hi_grid=hi))
            curvas_J_fase.setdefault(palanca, []).append((grid, J, hist_val))

    for palanca, curvas in curvas_J_fase.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for grid, J, hist_val in curvas:
            ax.plot(grid, J, alpha=0.35, color="steelblue")
            idx_hist = int(np.argmin(np.abs(grid - hist_val)))
            ax.scatter([hist_val], [J[idx_hist]], color="crimson", s=10, zorder=5)
        ax.set_xlabel(palanca)
        ax.set_ylabel("J [kg Sn equivalente]")
        ax.set_title(f"J vs {palanca} -- {fase} ({len(curvas)} escalones, punto rojo=accion historica)", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / f"J__{fase}__{palanca}.png", dpi=110)
        plt.close(fig)

J_df = pd.DataFrame(filas_J)
J_df.to_csv(SALIDA / "16_forma_J.csv", index=False)
log(t(), f"16_forma_J.csv -> {J_df.shape}")
log(t(), "\n=== resumen tipo_forma de J por fase/palanca ===")
log(J_df.groupby(["fase", "palanca", "tipo_forma"]).size().unstack(fill_value=0).to_string())
log(t(), "\n=== delta_relativo_optimo_vs_hist (mediana) por fase/palanca ===")
log(J_df.groupby(["fase", "palanca"])["delta_relativo_optimo_vs_hist"].median().to_string())

log(t(), "\nFIN experimento 16.")
LOGF.close()
