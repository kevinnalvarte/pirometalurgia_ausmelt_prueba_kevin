"""v6_lib -- librería compartida de la iteración 8 (prescriptor v6, 2026-09-16).

Punto de partida: v5 (`experimentos/v5/v5_lib.py`, iteración 7) + el estimador de masa de escoria por
escalón v6 (`experimentos/v6/masa_v6.py`), que reemplaza el trazador de CaO (`masa_escoria_est_kg`, v3) por
un cierre físico sin %CaO en Reducción y un trazador de matriz G = CaO+SiO2+Al2O3+MgO en Fusión (ver
docstring de `masa_v6.py` y `experimentos/v6/ITERACION_8_diseno.md`, diagnóstico de la auditoría del
trazador CaO).

Qué provee (todo con clasificación anti-fuga explícita: `es_segura` = `masa_v6.es_segura_v6`; para paridad
con columnas v5 usar `v5_lib.es_segura`):

1) `cargar_df(con_termicas=True)` -> como `v5_lib.cargar_df` (cache v3 + columnas v4 + bloque térmico b6 +
   targets/palancas v5) más las columnas `m6_*` de `masa_v6.agregar_masa_v6` (masa, inventarios de Sn/FeO,
   estado `_prev`, targets log `m6_ln_sn_dep`/`m6_ln_feo_ret`, palancas cinéticas `Cx_sn_v6`/`Cx_av_v6` y
   dosis/sobrante `m6_dosis_C_sn`/`m6_exceso_C_pos`/`m6_C_x_avance`) calibradas contra el balance global de
   batch (`masa_v6.balance_batch`, `PARAMS_DEFAULT`). Se conservan TODAS las columnas v5 (masa v3,
   `v5_ln_sn_dep`, `v5_ln_feo_ret`, ...) para poder comparar v5 vs v6 sobre las mismas filas.
   `PARAMS` es un dict a nivel de módulo (copia de `masa_v6.PARAMS_DEFAULT`) que se puede sobreescribir
   antes de llamar a `cargar_df()` para probar variantes de calibración (p.ej. `v6_lib.PARAMS =
   dict(v6_lib.PARAMS, delta_R=0.02)`).

2) Targets v6 por escalón (heredados de `masa_v6.agregar_masa_v6`, ver también `masa_v6.CLASIFICACION_M6`):
       m6_ln_sn_dep  = ln(Sn_disp / Sn_inv[t])         (análogo a v5_ln_sn_dep, con la masa v6)
       m6_ln_feo_ret = ln(FeO_inv[t] / FeO_inv[t-1])   (solo Reducción, análogo a v5_ln_feo_ret)
   Validez: inventarios > umbral (Sn 20 kg, FeO 100 kg), no primer escalón del batch, `m6_valido`.

3) Palancas v6 (`PALANCAS_V6`, `SIGNO_TEORICO_V6`, diccionarios por (fase, clave) igual que
   `modelo_predictivo_v5.PALANCAS_V5`/`SIGNO_TEORICO_V5` pero con `Cx_sn_v4` -> `Cx_sn_v6` y
   `Cx_av_v4` -> `Cx_av_v6` (cinética con la masa/inventario v6). Los targets análogos quedan en
   `TARGETS_V6`. Las palancas de dosis/sobrante (`m6_dosis_C_sn`, `m6_exceso_C_pos`, `m6_C_x_avance`) están
   disponibles en el df pero no se fijan aquí como conjunto por defecto -- se combinan ad hoc en E8-03 igual
   que `P3_dosis_sobrante` en E7-01.

4) Estado curado con la masa v6 (`ESTADO_R_FULL_V6`, `ESTADO_R_CURADO_V6`, `ESTADO_F_CURADO_V6`): los mismos
   de v5 (`modelo_predictivo_v5.ESTADO_R_FULL/ESTADO_R_CURADO/ESTADO_F_CURADO`) sustituyendo los inventarios
   v3 (trazador CaO) por los v6 (cierre físico): `masa_escoria_est_kg_prev` -> `m6_masa_kg_prev`,
   `sn_inventario_escoria_est_kg_prev` -> `m6_sn_inv_kg_prev`, `feo_inventario_escoria_est_kg_prev` ->
   `m6_feo_inv_kg_prev`, `avance_reduccion_sn_prev` -> `m6_avance_prev`; los estados CURADO además añaden
   `m6_resto_frac_prev` y `m6_G_pct_prev` (diagnóstico de matriz/cierre, no en v5).

5) `construir_batch_v6(df)` -> como `v5_lib.construir_batch_v5(df)` (mismos KPIs, controles, agregados v5 y
   KPI del batch siguiente) más los agregados v6 por batch: `R_sum_m6_ln_sn_dep`, `R_sum_m6_ln_feo_ret`,
   `F_sum_m6_ln_sn_dep`, `F_m6_sn_ext_kg` (Sn extraído en Fusión, balance v6), `R_m6_masa_fin_kg`,
   `F_m6_masa_fin_kg` (masa final de cada fase, v6) y `m6_k_anclaje` (diagnóstico LEAKAGE offline del
   estimador, factor de anclaje al balance global -- solo para diagnóstico, nunca como feature de
   prescripción).

6) `bateria_batch` y `evaluar_plm` se REUTILIZAN de `v5_lib` sin cambios (funcionan igual sobre columnas v6,
   ya que solo dependen de nombres de columna). `evaluar_plm_signos` es el análogo de `evaluar_plm` pero con
   `modelo_predictivo_v5.ModeloPLMSignos` (theta acotado por `signo_teorico`, colapsa a 0 si el dato
   contradice la teoría en vez de un signo espurio).

Convenciones: OOF GroupKFold(5) por Batch sobre DEV = criterio de selección; lockbox (últimos 17.5 % de
batches) solo reporte. Python: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4`,
sin joblib. No se modifican `masa_v6.py`, `feature_engineering.py` ni los módulos de la raíz.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
V6_DIR = Path(__file__).resolve().parent
V5_DIR = RAIZ / "experimentos" / "v5"
for _p in (RAIZ, RAIZ / "experimentos" / "balances", V5_DIR, V6_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import v5_lib  # noqa: E402
import masa_v6  # noqa: E402

warnings.filterwarnings("ignore")

FASES = v5_lib.FASES
KPI_PRINCIPAL = v5_lib.KPI_PRINCIPAL
KPIS = v5_lib.KPIS
CONTROLES_BATCH = v5_lib.CONTROLES_BATCH

# Parametros de calibracion del estimador de masa v6 -- copia sobreescribible (modificar antes de cargar_df()).
PARAMS: dict = dict(masa_v6.PARAMS_DEFAULT)

# --------------------------------------------------------------------------- anti-fuga
es_segura = masa_v6.es_segura_v6


def es_segura_v6_o_v5(nombre: str) -> bool:
    """Union de seguridad v6 (`masa_v6.es_segura_v6`) y v5 (`v5_lib.es_segura`): util quc los PLM de
    comparacion v5-vs-v6 (mismas filas) puedan validar tanto `m6_*`/`Cx_*_v6` como `v5_*`/`Cx_*_v4`."""
    return es_segura(nombre) or v5_lib.es_segura(nombre)


def filtrar_seguras(nombres: list[str]) -> list[str]:
    return [n for n in nombres if es_segura_v6_o_v5(n)]


# --------------------------------------------------------------------------- carga del df
def cargar_df(con_termicas: bool = True) -> pd.DataFrame:
    """v5_lib.cargar_df(...) + columnas m6_* (masa/inventarios/targets/palancas v6), calibradas con
    `masa_v6.balance_batch` sobre TODO el df (la calibracion de coeficientes efectivos de `masa_v6.
    calibrar_entradas` usa internamente solo DEV; `agregar_masa_v6` con PARAMS fijos no vuelve a calibrar,
    solo aplica el estimador escalon a escalon)."""
    df = v5_lib.cargar_df(con_termicas=con_termicas)
    bal = masa_v6.balance_batch(df)
    df = masa_v6.agregar_masa_v6(df, PARAMS, bal)
    return df


def base_fase(df: pd.DataFrame, fase: str) -> pd.DataFrame:
    return v5_lib.base_fase(df, fase)


# --------------------------------------------------------------------------- estado (STATE) con masa v6
_SUST_INVENTARIOS_V6: dict[str, str] = {
    "masa_escoria_est_kg_prev": "m6_masa_kg_prev",
    "sn_inventario_escoria_est_kg_prev": "m6_sn_inv_kg_prev",
    "feo_inventario_escoria_est_kg_prev": "m6_feo_inv_kg_prev",
    "avance_reduccion_sn_prev": "m6_avance_prev",
}
_EXTRA_CURADO_V6 = ["m6_resto_frac_prev", "m6_G_pct_prev"]


def _sustituir_v6(cols: list[str], extra: list[str] | None = None) -> list[str]:
    out = [_SUST_INVENTARIOS_V6.get(c, c) for c in cols]
    for c in (extra or []):
        if c not in out:
            out.append(c)
    return out


ESTADO_R_FULL_V6 = _sustituir_v6(mp5.ESTADO_R_FULL)
ESTADO_R_CURADO_V6 = _sustituir_v6(mp5.ESTADO_R_CURADO, extra=_EXTRA_CURADO_V6)
ESTADO_F_CURADO_V6 = _sustituir_v6(mp5.ESTADO_F_CURADO, extra=_EXTRA_CURADO_V6)

# --------------------------------------------------------------------------- palancas v6 (Cx_*_v4 -> Cx_*_v6)
_SUST_PALANCA_V6: dict[str, str] = {"Cx_sn_v4": "Cx_sn_v6", "Cx_av_v4": "Cx_av_v6"}


def _sustituir_palancas_v6(lst: list[str]) -> list[str]:
    return [_SUST_PALANCA_V6.get(c, c) for c in lst]


PALANCAS_V6: dict[tuple[str, str], list[str]] = {k: _sustituir_palancas_v6(v) for k, v in mp5.PALANCAS_V5.items()}
SIGNO_TEORICO_V6: dict[tuple[str, str], dict[str, int]] = {
    k: {_SUST_PALANCA_V6.get(a, a): s for a, s in v.items()} for k, v in mp5.SIGNO_TEORICO_V5.items()
}
TARGETS_V6: dict[str, dict[str, str]] = {
    "Reducción": {"sn_dep": "m6_ln_sn_dep", "feo_ret": "m6_ln_feo_ret"},
    "Fusión": {"sn_dep": "m6_ln_sn_dep"},
}


def _verificar_seguridad_v6() -> None:
    for k, feats in list(PALANCAS_V6.items()):
        inseg = [f for f in feats if not es_segura(f)]
        if inseg:
            raise AssertionError(f"v6 PALANCAS_V6{k}: features inseguras {inseg}")
    for nombre, cols in [("ESTADO_R_FULL_V6", ESTADO_R_FULL_V6), ("ESTADO_R_CURADO_V6", ESTADO_R_CURADO_V6),
                         ("ESTADO_F_CURADO_V6", ESTADO_F_CURADO_V6)]:
        inseg = [f for f in cols if not es_segura(f)]
        if inseg:
            raise AssertionError(f"v6 {nombre}: features inseguras {inseg}")


_verificar_seguridad_v6()


# --------------------------------------------------------------------------- batch
def construir_batch_v6(df: pd.DataFrame) -> pd.DataFrame:
    """v5_lib.construir_batch_v5(df) (KPIs, controles, agregados v5, KPI del batch siguiente) + agregados
    v6 por batch: R_sum_m6_ln_sn_dep, R_sum_m6_ln_feo_ret, F_sum_m6_ln_sn_dep, F_m6_sn_ext_kg,
    R_m6_masa_fin_kg, F_m6_masa_fin_kg, m6_k_anclaje. `df` debe venir de `cargar_df()` (trae m6_* y, si
    `agregar_masa_v6` recibio `bal`, tambien `m6_k_anclaje`)."""
    out = v5_lib.construir_batch_v5(df)
    filas = []
    tiene_k = "m6_k_anclaje" in df.columns
    for b, g in df.sort_values(["Batch", "fecha_inicio"]).groupby("Batch", sort=False):
        gf, gr = g[g["fase_proceso"] == "Fusión"], g[g["fase_proceso"] == "Reducción"]
        fila = dict(
            Batch=b,
            R_sum_m6_ln_sn_dep=float(gr["m6_ln_sn_dep"].sum(min_count=1)) if gr["m6_ln_sn_dep"].notna().any() else np.nan,
            R_sum_m6_ln_feo_ret=float(gr["m6_ln_feo_ret"].sum(min_count=1)) if gr["m6_ln_feo_ret"].notna().any() else np.nan,
            F_sum_m6_ln_sn_dep=float(gf["m6_ln_sn_dep"].sum(min_count=1)) if gf["m6_ln_sn_dep"].notna().any() else np.nan,
            F_m6_sn_ext_kg=float(gf["m6_sn_extraido_kg"].sum(min_count=1)) if gf["m6_sn_extraido_kg"].notna().any() else np.nan,
            R_m6_masa_fin_kg=float(gr["m6_masa_kg"].dropna().iloc[-1]) if gr["m6_masa_kg"].notna().any() else np.nan,
            F_m6_masa_fin_kg=float(gf["m6_masa_kg"].dropna().iloc[-1]) if gf["m6_masa_kg"].notna().any() else np.nan,
            m6_k_anclaje=float(g["m6_k_anclaje"].dropna().iloc[0]) if tiene_k and g["m6_k_anclaje"].notna().any() else np.nan,
        )
        filas.append(fila)
    agg = pd.DataFrame(filas).set_index("Batch")
    out = out.join(agg, how="left")
    return out


# --------------------------------------------------------------------------- bateria y PLM (reutilizados de v5_lib)
bateria_batch = v5_lib.bateria_batch
evaluar_plm = v5_lib.evaluar_plm


def evaluar_plm_signos(df_fase: pd.DataFrame, estado: list[str], palancas: list[str], target: str, n_splits: int = 5,
                       n_boot: int = 300, seed: int = 42, signo_teorico: dict | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Como `v5_lib.evaluar_plm` (PLM cross-fitted, GroupKFold por batch, R2/MAE OOF DEV + lockbox, theta con
    IC bootstrap cluster) pero con `modelo_predictivo_v5.ModeloPLMSignos`: el theta de cada palanca queda
    acotado al signo de `signo_teorico` (colapsa a 0 si el dato lo contradice) en cada fold del cross-fitting
    y en el ajuste final -- ver docstring de `ModeloPLMSignos`."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    inseg = [c for c in estado + palancas if not es_segura_v6_o_v5(c)]
    if inseg:
        raise AssertionError(f"features inseguras: {inseg}")
    dev, lockbox = mp.split_dev_lockbox(df_fase)
    need = list(dict.fromkeys(estado + palancas + [target, "Batch", "fecha_inicio", "orden_escalon_fase"]))
    d_dev = df_fase.loc[df_fase["Batch"].isin(dev), need].dropna().reset_index(drop=True)
    d_lb = df_fase.loc[df_fase["Batch"].isin(lockbox), need].dropna().reset_index(drop=True)
    signos = signo_teorico or {}
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev, d_dev[target], d_dev["Batch"]):
        m = mp5.ModeloPLMSignos(estado, palancas, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
        oof[te] = m.predict(d_dev.iloc[te])
    m = mp5.ModeloPLMSignos(estado, palancas, target, signos).fit(d_dev, n_boot=n_boot, seed=seed)
    p_lb = m.predict(d_lb) if len(d_lb) else np.array([])
    met = {"target": target, "n_estado": len(estado), "n_palancas": len(palancas), "n_dev": len(d_dev), "n_lockbox": len(d_lb),
           "r2_oof": float(r2_score(d_dev[target], oof)), "mae_oof": float(mean_absolute_error(d_dev[target], oof)),
           "r2_lockbox": float(r2_score(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan,
           "mae_lockbox": float(mean_absolute_error(d_lb[target], p_lb)) if len(d_lb) > 10 else np.nan,
           "sd_target_dev": float(d_dev[target].std())}
    tabla = m.tabla_theta(signos).reset_index()
    tabla.insert(0, "target", target)
    tabla["theta_1sd_rel"] = tabla["theta_1sd"] / met["sd_target_dev"]
    tabla["ancho_ic_1sd_rel"] = (tabla["ci_hi_1sd"] - tabla["ci_lo_1sd"]) / met["sd_target_dev"]
    pred = pd.concat([d_dev[["Batch", "fecha_inicio", "orden_escalon_fase"]].assign(real=d_dev[target], pred=oof, conjunto="OOF_dev"),
                      d_lb[["Batch", "fecha_inicio", "orden_escalon_fase"]].assign(real=d_lb[target], pred=p_lb, conjunto="lockbox")],
                     ignore_index=True)
    return met, tabla, pred


if __name__ == "__main__":
    import time
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

    log("cargando datos (v5_lib + masa_v6)...")
    df = cargar_df()
    log(f"df {df.shape}")
    dR = base_fase(df, "Reducción")
    dF = base_fase(df, "Fusión")
    log(f"dR {dR.shape}  dF {dF.shape}  targets m6: "
        f"{dR[['m6_ln_sn_dep', 'm6_ln_feo_ret']].describe().round(3).to_dict()}")

    batch = construir_batch_v6(df)
    log(f"batch {batch.shape} ({batch['es_dev'].sum()} DEV / {batch['es_lockbox'].sum()} lockbox)")

    log("=== hechos verificados v5 (e7_01) replicados con targets v6: Spearman vs KPI, DEV/lockbox ===")
    for col in ["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret", "F_sum_m6_ln_sn_dep",
                "R_sum_ln_sn_dep", "R_sum_ln_feo_ret", "F_sum_ln_sn_dep"]:
        r = bateria_batch(batch, col)
        log(f"{col:24s} rho_dev={r.get('rho_dev', np.nan):.3f} [{r.get('ci_lo_dev', np.nan):.3f},"
            f"{r.get('ci_hi_dev', np.nan):.3f}] p_dev={r.get('p_dev', np.nan):.3f}  "
            f"rho_lockbox={r.get('rho_lockbox', np.nan):.3f}  n_dev={r.get('n_dev', 'NA')}")

    log("=== J compuesto (Sn_dep + w*FeO_ret) v6 vs v5, w=6.04 (E7-01 ancla) ===")
    for nombre, col_sn, col_feo in [("v6", "R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret"),
                                     ("v5", "R_sum_ln_sn_dep", "R_sum_ln_feo_ret")]:
        j = batch[col_sn] + 6.04 * batch[col_feo]
        batch[f"_J_{nombre}"] = j
        r = bateria_batch(batch, f"_J_{nombre}")
        log(f"J_{nombre:2s} rho_dev={r.get('rho_dev', np.nan):.3f}  rho_lockbox={r.get('rho_lockbox', np.nan):.3f}  n_dev={r.get('n_dev', 'NA')}")

    log(f"TOTAL {time.time() - t0:.1f}s")
