"""Experimento 17 -- Estabilidad temporal y deriva de campana de los modelos v3 y de
los efectos de las palancas.

Preguntas (ver experimentos/ITERACION_4_diseno.md, hallazgos.md 14.3 punto 6 y
experimentos/12_diag_deriva_fusion_log.txt): (1) cuanto se degrada el R2 al pasar de
OOF aleatorio (GroupKFold) a walk-forward cronologico dentro de DEV; (2) que politica
de reentrenamiento (todo DEV / ultimos N batches / actualizacion progresiva dentro del
lockbox) maximiza el desempeno en el lockbox; (3) si quitar/anadir variables ligadas a
la edad de campana (espesor de refractario, termocuplas de carcasa) o normalizar la
temperatura como desviacion de una mediana movil de batches anteriores ayuda; (4) si
el efecto DML de las palancas (signo y magnitud) es estable entre la primera y la
segunda mitad cronologica de DEV y el lockbox; (5) que features de estado/contexto
extrapolan mas en el lockbox.

Convenciones (ITERACION_4_diseno.md): .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8,
OMP_NUM_THREADS=4; dataset cacheado experimentos/cache/df_v3.pkl; DEV/lockbox via
modelo_prescriptivo.split_dev_lockbox; ninguna decision de features/politica usa el
lockbox para SELECCIONAR nada -- aqui el lockbox se usa EXCLUSIVAMENTE para evaluar
distintas politicas de reentrenamiento y sets de features ya propuestos en el diseno
(no se hace busqueda/tuning sobre el lockbox).

Salidas: 17_walkforward.csv, 17_reentrenamiento_ventana.csv, 17_sets_campana.csv,
17_efectos_por_periodo.csv, 17_deriva_features.csv, figs_17/*.png, 17_log.txt.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

FIGS = RAIZ / "experimentos" / "figs_17"
FIGS.mkdir(exist_ok=True)

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)

print("=" * 100)
print("Experimento 17 -- estabilidad temporal y deriva de campana")
print("=" * 100)

df = pd.read_pickle(RAIZ / "experimentos" / "cache" / "df_v3.pkl")
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
orden_todos = mp.orden_cronologico_batches(df)
orden_dev = [b for b in orden_todos if b in batches_dev]
orden_lockbox = [b for b in orden_todos if b in batches_lockbox]
print(f"Batches: {len(orden_todos)} total, {len(orden_dev)} DEV, {len(orden_lockbox)} lockbox "
      f"(primer lockbox={orden_lockbox[0]}, ultimo={orden_lockbox[-1]})")

# Los 4 targets del experimento (fase, clave de TARGETS_V3/FEATURES_V3_POR_DEFECTO)
TARGETS_17 = [("Fusión", "sn_kg"), ("Reducción", "sn_kg"), ("Reducción", "feo_kg"), ("Reducción", "dT")]


def _sub_fase(fase: str, batches: set | list | None = None) -> pd.DataFrame:
    s = df_base.loc[df_base["fase_proceso"] == fase]
    if batches is not None:
        s = s.loc[s["Batch"].isin(set(batches))]
    return s


def _r2_mae(y_true, y_pred) -> tuple[float, float]:
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        return np.nan, np.nan
    return float(r2_score(y_true, y_pred)), float(mean_absolute_error(y_true, y_pred))


def _fit_predict(feats: list[str], target_col: str, d_train: pd.DataFrame, d_test: pd.DataFrame, monotono: bool = True):
    m = mp3.construir_modelo_v3(feats, target_col, monotono)
    m.fit(d_train[feats], d_train[target_col])
    return m.predict(d_test[feats])


# =============================================================================
# 1) Walk-forward cronologico sobre DEV vs OOF aleatorio (GroupKFold)
# =============================================================================
print("\n" + "-" * 100)
print("1) Walk-forward cronologico (TimeSeriesSplit sobre batches DEV) vs OOF GroupKFold aleatorio")
print("-" * 100)

filas_wf = []
for fase, clave in TARGETS_17:
    feats = mp3.FEATURES_V3_POR_DEFECTO[fase][clave]
    target_col = mp3.TARGETS_V3[fase][clave]
    sub = _sub_fase(fase, batches_dev).dropna(subset=[target_col]).reset_index(drop=True)
    orden_batches_sub = [b for b in orden_dev if b in set(sub["Batch"])]

    # --- OOF GroupKFold aleatorio (linea base ya reportada en hallazgos.md 14.5)
    oof_rand = np.full(len(sub), np.nan)
    for tr, te in GroupKFold(n_splits=mp.N_SPLITS_OOF).split(sub[feats], sub[target_col], sub["Batch"]):
        oof_rand[te] = _fit_predict(feats, target_col, sub.iloc[tr], sub.iloc[te])
    r2_rand, mae_rand = _r2_mae(sub[target_col], oof_rand)

    # --- Walk-forward cronologico: TimeSeriesSplit(5) sobre la LISTA DE BATCHES (no sobre filas)
    pred_wf = np.full(len(sub), np.nan)
    batch_arr = sub["Batch"].to_numpy()
    for i_split, (tr_pos, te_pos) in enumerate(TimeSeriesSplit(n_splits=5).split(orden_batches_sub)):
        batches_tr = set(orden_batches_sub[i] for i in tr_pos)
        batches_te = set(orden_batches_sub[i] for i in te_pos)
        mask_tr = sub["Batch"].isin(batches_tr).to_numpy()
        mask_te = sub["Batch"].isin(batches_te).to_numpy()
        if mask_tr.sum() < 20 or mask_te.sum() < 5:
            continue
        pred_wf[mask_te] = _fit_predict(feats, target_col, sub.loc[mask_tr], sub.loc[mask_te])
        r2_b, mae_b = _r2_mae(sub.loc[mask_te, target_col], pred_wf[mask_te])
        filas_wf.append({"fase": fase, "target": clave, "bloque": i_split, "tipo": "walkforward_bloque",
                          "n_train": int(mask_tr.sum()), "n_test": int(mask_te.sum()),
                          "batch_ini_test": min(batches_te, key=lambda b: orden_todos.index(b)),
                          "batch_fin_test": max(batches_te, key=lambda b: orden_todos.index(b)),
                          "r2": r2_b, "mae": mae_b})
        print(f"  {fase:9s} {clave:6s} bloque {i_split}: n_train={mask_tr.sum():4d} n_test={mask_te.sum():4d} "
              f"R2={r2_b:+.3f} MAE={mae_b:.1f}", flush=True)

    valid = ~np.isnan(pred_wf)
    r2_wf_global, mae_wf_global = _r2_mae(sub.loc[valid, target_col], pred_wf[valid])
    filas_wf.append({"fase": fase, "target": clave, "bloque": "global_walkforward", "tipo": "walkforward_global",
                      "n_train": np.nan, "n_test": int(valid.sum()), "batch_ini_test": np.nan, "batch_fin_test": np.nan,
                      "r2": r2_wf_global, "mae": mae_wf_global})
    filas_wf.append({"fase": fase, "target": clave, "bloque": "global_oof_aleatorio", "tipo": "oof_aleatorio",
                      "n_train": np.nan, "n_test": len(sub), "batch_ini_test": np.nan, "batch_fin_test": np.nan,
                      "r2": r2_rand, "mae": mae_rand})
    filas_wf.append({"fase": fase, "target": clave, "bloque": "brecha (aleatorio - walkforward)", "tipo": "brecha",
                      "n_train": np.nan, "n_test": np.nan, "batch_ini_test": np.nan, "batch_fin_test": np.nan,
                      "r2": r2_rand - r2_wf_global, "mae": mae_wf_global - mae_rand})
    print(f"  {fase:9s} {clave:6s} GLOBAL: OOF aleatorio R2={r2_rand:+.3f} MAE={mae_rand:.1f} | "
          f"walk-forward R2={r2_wf_global:+.3f} MAE={mae_wf_global:.1f} | brecha R2={r2_rand - r2_wf_global:+.3f}",
          flush=True)

tabla_wf = pd.DataFrame(filas_wf)
tabla_wf.to_csv(RAIZ / "experimentos" / "17_walkforward.csv", index=False)
print("\nGuardado 17_walkforward.csv")


# =============================================================================
# 2) Reentrenamiento por ventana para el lockbox
# =============================================================================
print("\n" + "-" * 100)
print("2) Reentrenamiento por ventana (evaluado SOLO en lockbox; no se selecciona nada con el lockbox)")
print("-" * 100)

filas_vent = []
for fase, clave in TARGETS_17:
    feats = mp3.FEATURES_V3_POR_DEFECTO[fase][clave]
    target_col = mp3.TARGETS_V3[fase][clave]
    sub_dev = _sub_fase(fase, batches_dev).dropna(subset=[target_col]).reset_index(drop=True)
    sub_lb = _sub_fase(fase, batches_lockbox).dropna(subset=[target_col]).reset_index(drop=True)
    if sub_lb.empty:
        continue

    ventanas = {
        "todo_dev": orden_dev,
        "ultimos_150": orden_dev[-150:],
        "ultimos_100": orden_dev[-100:],
        "ultimos_60": orden_dev[-60:],
    }
    for nombre_v, batches_v in ventanas.items():
        d_tr = sub_dev.loc[sub_dev["Batch"].isin(set(batches_v))]
        if len(d_tr) < 20:
            continue
        pred = _fit_predict(feats, target_col, d_tr, sub_lb)
        r2_v, mae_v = _r2_mae(sub_lb[target_col], pred)
        filas_vent.append({"fase": fase, "target": clave, "politica": nombre_v, "n_train": len(d_tr),
                            "n_test": len(sub_lb), "r2_lockbox": r2_v, "mae_lockbox": mae_v})
        print(f"  {fase:9s} {clave:6s} politica={nombre_v:12s} n_train={len(d_tr):4d} "
              f"R2_lockbox={r2_v:+.3f} MAE_lockbox={mae_v:.1f}", flush=True)

    # (e) actualizacion progresiva dentro del lockbox: bloques cronologicos de ~10 batches,
    # reentrenando con DEV + lockbox ya visto (simula reentrenar por campana / periodicamente).
    tam_bloque = 10
    bloques = [orden_lockbox[i:i + tam_bloque] for i in range(0, len(orden_lockbox), tam_bloque)]
    pred_prog = np.full(len(sub_lb), np.nan)
    lb_batch_arr = sub_lb["Batch"].to_numpy()
    vistos: list = []
    for i_bloque, bloque in enumerate(bloques):
        d_tr = pd.concat([sub_dev, sub_lb.loc[sub_lb["Batch"].isin(set(vistos))]]) if vistos else sub_dev
        mask_te = np.isin(lb_batch_arr, bloque)
        if mask_te.sum() == 0:
            continue
        pred_bloque = _fit_predict(feats, target_col, d_tr, sub_lb.loc[mask_te])
        pred_prog[mask_te] = pred_bloque
        r2_acum, mae_acum = _r2_mae(sub_lb.loc[np.isin(lb_batch_arr, sum(bloques[:i_bloque + 1], []))][target_col],
                                     pred_prog[np.isin(lb_batch_arr, sum(bloques[:i_bloque + 1], []))])
        r2_bloque, mae_bloque = _r2_mae(sub_lb.loc[mask_te, target_col], pred_bloque)
        filas_vent.append({"fase": fase, "target": clave, "politica": f"progresivo_bloque_{i_bloque}",
                            "n_train": len(d_tr), "n_test": int(mask_te.sum()), "r2_lockbox": r2_bloque,
                            "mae_lockbox": mae_bloque, "r2_lockbox_acumulado": r2_acum, "mae_lockbox_acumulado": mae_acum})
        print(f"  {fase:9s} {clave:6s} progresivo bloque {i_bloque} (batches {bloque[0]}..{bloque[-1]}): "
              f"n_train={len(d_tr):4d} R2_bloque={r2_bloque:+.3f} R2_acumulado={r2_acum:+.3f}", flush=True)
        vistos.extend(bloque)
    valid = ~np.isnan(pred_prog)
    r2_prog_total, mae_prog_total = _r2_mae(sub_lb.loc[valid, target_col], pred_prog[valid])
    filas_vent.append({"fase": fase, "target": clave, "politica": "progresivo_total", "n_train": np.nan,
                        "n_test": int(valid.sum()), "r2_lockbox": r2_prog_total, "mae_lockbox": mae_prog_total})
    print(f"  {fase:9s} {clave:6s} progresivo TOTAL: R2={r2_prog_total:+.3f} MAE={mae_prog_total:.1f}", flush=True)

tabla_vent = pd.DataFrame(filas_vent)
tabla_vent.to_csv(RAIZ / "experimentos" / "17_reentrenamiento_ventana.csv", index=False)
print("\nGuardado 17_reentrenamiento_ventana.csv")


# =============================================================================
# 3) Sets sin/con contexto de campana (solo Fusion sn_kg y Reduccion sn_kg)
# =============================================================================
print("\n" + "-" * 100)
print("3) Sets con/sin contexto de campana + temperatura como desviacion de mediana movil de 20 batches previos")
print("-" * 100)

VARS_CAMPANA = ["espesor_ladrillo_norm_mm", "termocupla_media_celsius_prev"]

# --- feature "calculable en produccion": desviacion de temperatura_horno_celsius_prev respecto
# a la mediana movil de los ultimos 20 BATCHES anteriores (nunca usa el batch actual ni el futuro).
batch_temp_mediana = df_base.groupby("Batch")["temperatura_horno_celsius_prev"].median().reindex(orden_todos)
rolling_mediana_temp = batch_temp_mediana.rolling(window=20, min_periods=5).median().shift(1)
df_base = df_base.assign(
    temp_mediana_movil20_prev=df_base["Batch"].map(rolling_mediana_temp),
)
df_base["temp_desv_movil20_prev"] = df_base["temperatura_horno_celsius_prev"] - df_base["temp_mediana_movil20_prev"]
n_validas = df_base["temp_desv_movil20_prev"].notna().sum()
print(f"  temp_desv_movil20_prev: {n_validas}/{len(df_base)} filas validas "
      f"(NaN en los primeros ~20 batches cronologicos, sin historia previa suficiente)")

filas_sets = []
for fase, clave in [("Fusión", "sn_kg"), ("Reducción", "sn_kg")]:
    feats_default = list(mp3.FEATURES_V3_POR_DEFECTO[fase][clave])
    target_col = mp3.TARGETS_V3[fase][clave]
    presentes = [v for v in VARS_CAMPANA if v in feats_default]
    ausentes = [v for v in VARS_CAMPANA if v not in feats_default]

    variantes = {"default": feats_default}
    variantes["sin_campana"] = [f for f in feats_default if f not in VARS_CAMPANA] if presentes else list(feats_default)
    variantes["con_campana"] = feats_default + ausentes if ausentes else list(feats_default)
    if "temperatura_horno_celsius_prev" in feats_default:
        variantes["temp_desv_movil20"] = [("temp_desv_movil20_prev" if f == "temperatura_horno_celsius_prev" else f)
                                           for f in feats_default]
    print(f"  {fase} {clave}: campana presente en default={presentes or 'ninguna'} -> "
          f"sin_campana {'=default (nada que quitar)' if not presentes else 'quita ' + str(presentes)}, "
          f"con_campana {'=default (ya las incluye)' if not ausentes else 'agrega ' + str(ausentes)}")

    sub_dev_all = _sub_fase(fase, batches_dev)
    sub_lb_all = _sub_fase(fase, batches_lockbox)
    for nombre_v, feats_v in variantes.items():
        d_dev = sub_dev_all.dropna(subset=[target_col] + feats_v).reset_index(drop=True) if nombre_v == "temp_desv_movil20" else \
                sub_dev_all.dropna(subset=[target_col]).reset_index(drop=True)
        d_lb = sub_lb_all.dropna(subset=[target_col] + feats_v).reset_index(drop=True) if nombre_v == "temp_desv_movil20" else \
               sub_lb_all.dropna(subset=[target_col]).reset_index(drop=True)
        # OOF GroupKFold sobre DEV
        oof = np.full(len(d_dev), np.nan)
        for tr, te in GroupKFold(n_splits=mp.N_SPLITS_OOF).split(d_dev[feats_v], d_dev[target_col], d_dev["Batch"]):
            oof[te] = _fit_predict(feats_v, target_col, d_dev.iloc[tr], d_dev.iloc[te])
        r2_oof, mae_oof = _r2_mae(d_dev[target_col], oof)
        pred_lb = _fit_predict(feats_v, target_col, d_dev, d_lb)
        r2_lb, mae_lb = _r2_mae(d_lb[target_col], pred_lb)
        filas_sets.append({"fase": fase, "target": clave, "set": nombre_v, "n_features": len(feats_v),
                            "n_dev": len(d_dev), "n_lockbox": len(d_lb), "r2_oof": r2_oof, "mae_oof": mae_oof,
                            "r2_lockbox": r2_lb, "mae_lockbox": mae_lb})
        print(f"    set={nombre_v:18s} k={len(feats_v):2d} R2_oof={r2_oof:+.3f} R2_lockbox={r2_lb:+.3f} "
              f"MAE_lockbox={mae_lb:.1f}", flush=True)

tabla_sets = pd.DataFrame(filas_sets)
tabla_sets.to_csv(RAIZ / "experimentos" / "17_sets_campana.csv", index=False)
print("\nGuardado 17_sets_campana.csv")


# =============================================================================
# 4) Estabilidad de los efectos de las palancas (DML conjunto) por periodo
# =============================================================================
print("\n" + "-" * 100)
print("4) Estabilidad de efectos_marginales_palancas_v3 por mitades cronologicas de DEV y lockbox")
print("-" * 100)


def _split_todo_como_dev(df_in: pd.DataFrame, pct_lockbox: float = mp.N_LOCKBOX_PCT):
    """Sustituto de mp.split_dev_lockbox: TODO el df recibido se trata como DEV (sin lockbox
    propio), para poder pedirle a efectos_marginales_palancas_v3 que trabaje sobre un periodo
    ya recortado por nosotros (primera/segunda mitad de DEV, o el lockbox) sin que descarte un
    17.5% adicional de ese periodo como si fuera su propio lockbox."""
    return set(df_in["Batch"].unique()), set()


def efectos_en_periodo(df_periodo: pd.DataFrame, fase: str, target_col: str, n_boot: int = 100, seed: int = 0) -> pd.DataFrame:
    original = mp.split_dev_lockbox
    mp.split_dev_lockbox = _split_todo_como_dev
    try:
        return mp3.efectos_marginales_palancas_v3(df_periodo, fase, target_col=target_col, n_boot=n_boot, seed=seed)
    finally:
        mp.split_dev_lockbox = original


mitad = len(orden_dev) // 2
periodos = {
    "dev_primera_mitad": [b for b in orden_dev[:mitad]],
    "dev_segunda_mitad": [b for b in orden_dev[mitad:]],
    "lockbox": orden_lockbox,
}
print(f"  dev_primera_mitad: {len(periodos['dev_primera_mitad'])} batches ({periodos['dev_primera_mitad'][0]}..{periodos['dev_primera_mitad'][-1]})")
print(f"  dev_segunda_mitad: {len(periodos['dev_segunda_mitad'])} batches ({periodos['dev_segunda_mitad'][0]}..{periodos['dev_segunda_mitad'][-1]})")
print(f"  lockbox: {len(periodos['lockbox'])} batches ({periodos['lockbox'][0]}..{periodos['lockbox'][-1]})")

combos_palancas = [("Fusión", "sn_kg"), ("Reducción", "sn_kg"), ("Reducción", "feo_kg")]
filas_efectos = []
for fase, clave in combos_palancas:
    target_col = mp3.TARGETS_V3[fase][clave]
    for nombre_p, batches_p in periodos.items():
        df_p = df.loc[df["Batch"].isin(set(batches_p))]
        try:
            tabla = efectos_en_periodo(df_p, fase, target_col, n_boot=100, seed=0)
        except Exception as exc:  # deja constancia y sigue con el resto
            print(f"  [ERROR] {fase} {clave} {nombre_p}: {exc}", flush=True)
            continue
        tabla = tabla.reset_index()
        tabla["periodo"] = nombre_p
        tabla["clave_target"] = clave
        filas_efectos.append(tabla)
        for _, r in tabla.iterrows():
            print(f"  {fase:9s} {clave:6s} {nombre_p:18s} {r['palanca']:26s} theta_1sd={r['efecto_por_1sd']:+9.2f} "
                  f"[{r['efecto_1sd_ci95_lo']:+9.2f},{r['efecto_1sd_ci95_hi']:+9.2f}] n={r['n']}", flush=True)

tabla_efectos = pd.concat(filas_efectos, ignore_index=True) if filas_efectos else pd.DataFrame()


def _overlap(lo1, hi1, lo2, hi2) -> bool:
    return (lo1 <= hi2) and (lo2 <= hi1)


filas_estab = []
if not tabla_efectos.empty:
    for (fase, clave, palanca), g in tabla_efectos.groupby(["fase", "clave_target", "palanca"]):
        g = g.set_index("periodo")
        if not all(p in g.index for p in periodos):
            continue
        signos = {p: np.sign(g.loc[p, "efecto_por_1sd"]) for p in periodos}
        mismo_signo = len(set(signos.values())) == 1
        pares = [("dev_primera_mitad", "dev_segunda_mitad"), ("dev_segunda_mitad", "lockbox"),
                 ("dev_primera_mitad", "lockbox")]
        solapa = {f"{a}_vs_{b}": _overlap(g.loc[a, "efecto_1sd_ci95_lo"], g.loc[a, "efecto_1sd_ci95_hi"],
                                          g.loc[b, "efecto_1sd_ci95_lo"], g.loc[b, "efecto_1sd_ci95_hi"])
                  for a, b in pares}
        estable = bool(mismo_signo and all(solapa.values()))
        fila = {"fase": fase, "clave_target": clave, "palanca": palanca, "mismo_signo_3_periodos": mismo_signo,
                "estable": estable}
        fila.update(solapa)
        for p in periodos:
            fila[f"theta_1sd__{p}"] = float(g.loc[p, "efecto_por_1sd"])
            fila[f"ci_lo__{p}"] = float(g.loc[p, "efecto_1sd_ci95_lo"])
            fila[f"ci_hi__{p}"] = float(g.loc[p, "efecto_1sd_ci95_hi"])
        filas_estab.append(fila)

tabla_estab = pd.DataFrame(filas_estab)
# tabla larga (una fila por fase/target/palanca/periodo) + columnas de estabilidad repetidas
tabla_efectos_out = tabla_efectos.merge(
    tabla_estab[["fase", "clave_target", "palanca", "mismo_signo_3_periodos", "estable"]],
    on=["fase", "clave_target", "palanca"], how="left") if not tabla_efectos.empty else tabla_efectos
tabla_efectos_out.to_csv(RAIZ / "experimentos" / "17_efectos_por_periodo.csv", index=False)
print("\nGuardado 17_efectos_por_periodo.csv")
if not tabla_estab.empty:
    print("\nResumen de estabilidad (signo igual en los 3 periodos AND IC se solapan en las 3 comparaciones):")
    print(tabla_estab[["fase", "clave_target", "palanca", "mismo_signo_3_periodos", "estable"]].to_string(index=False))


# =============================================================================
# 5) Distribucion de features de estado/contexto por periodo (deriva)
# =============================================================================
print("\n" + "-" * 100)
print("5) Deriva de features de estado/contexto: DEV por tercios cronologicos vs lockbox")
print("-" * 100)

filas_deriva = []
for fase in mp.FASES:
    feats_usados = set()
    for clave in mp3.FEATURES_V3_POR_DEFECTO[fase]:
        if (fase, clave) in TARGETS_17:
            feats_usados |= set(mp3.FEATURES_V3_POR_DEFECTO[fase][clave])
    contexto_estado = set(mp3.STATE_V3[fase]) | set(mp3.CONTEXT_V3[fase])
    feats_diag = sorted((feats_usados & contexto_estado) | ({"espesor_ladrillo_norm_mm", "termocupla_media_celsius_prev"} & set(df_base.columns)))
    print(f"  {fase}: features de estado/contexto diagnosticadas ({len(feats_diag)}): {feats_diag}")

    sub = _sub_fase(fase)
    sub_dev = sub.loc[sub["Batch"].isin(batches_dev)]
    sub_lb = sub.loc[sub["Batch"].isin(batches_lockbox)]
    orden_dev_fase = [b for b in orden_dev if b in set(sub_dev["Batch"])]
    n = len(orden_dev_fase)
    tercios = {
        "dev_tercio1": orden_dev_fase[: n // 3],
        "dev_tercio2": orden_dev_fase[n // 3: 2 * n // 3],
        "dev_tercio3": orden_dev_fase[2 * n // 3:],
    }
    for feat in feats_diag:
        p1_dev, p99_dev = sub_dev[feat].quantile([0.01, 0.99])
        for nombre_t, batches_t in tercios.items():
            s = sub_dev.loc[sub_dev["Batch"].isin(set(batches_t)), feat].dropna()
            if len(s) == 0:
                continue
            filas_deriva.append({"fase": fase, "feature": feat, "periodo": nombre_t, "n": len(s),
                                  "mediana": float(s.median()), "p5": float(s.quantile(0.05)),
                                  "p95": float(s.quantile(0.95)), "p1_dev": float(p1_dev), "p99_dev": float(p99_dev),
                                  "pct_fuera_rango_p1_p99_dev": np.nan})
        s_lb = sub_lb[feat].dropna()
        if len(s_lb) > 0:
            pct_fuera = float(((s_lb < p1_dev) | (s_lb > p99_dev)).mean() * 100)
            filas_deriva.append({"fase": fase, "feature": feat, "periodo": "lockbox", "n": len(s_lb),
                                  "mediana": float(s_lb.median()), "p5": float(s_lb.quantile(0.05)),
                                  "p95": float(s_lb.quantile(0.95)), "p1_dev": float(p1_dev), "p99_dev": float(p99_dev),
                                  "pct_fuera_rango_p1_p99_dev": pct_fuera})
            print(f"    {feat:32s} dev[tercio1..3] med={sub_dev.loc[sub_dev['Batch'].isin(tercios['dev_tercio1']), feat].median():.1f}/"
                  f"{sub_dev.loc[sub_dev['Batch'].isin(tercios['dev_tercio2']), feat].median():.1f}/"
                  f"{sub_dev.loc[sub_dev['Batch'].isin(tercios['dev_tercio3']), feat].median():.1f} | "
                  f"lockbox med={s_lb.median():.1f} | %fuera[P1,P99]_dev={pct_fuera:.1f}%", flush=True)

tabla_deriva = pd.DataFrame(filas_deriva)
tabla_deriva.to_csv(RAIZ / "experimentos" / "17_deriva_features.csv", index=False)
print("\nGuardado 17_deriva_features.csv")


# =============================================================================
# 6) Figuras
# =============================================================================
print("\n" + "-" * 100)
print("6) Figuras")
print("-" * 100)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 6a) R2 walk-forward por bloque
bloques_fig = tabla_wf.loc[tabla_wf["tipo"] == "walkforward_bloque"].copy()
if not bloques_fig.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (fase, clave), g in bloques_fig.groupby(["fase", "target"]):
        ax.plot(g["bloque"], g["r2"], marker="o", label=f"{fase} {clave}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Bloque walk-forward (cronologico, TimeSeriesSplit sobre DEV)")
    ax.set_ylabel("R2 en el bloque de test")
    ax.set_title("Walk-forward cronologico dentro de DEV, por bloque")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "r2_walkforward_por_bloque.png", dpi=130)
    plt.close(fig)
    print("  Guardada figs_17/r2_walkforward_por_bloque.png")

# 6b) R2 lockbox vs tamano de ventana de reentrenamiento
vent_fig = tabla_vent.loc[tabla_vent["politica"].isin(["todo_dev", "ultimos_150", "ultimos_100", "ultimos_60"])].copy()
if not vent_fig.empty:
    orden_x = {"ultimos_60": 60, "ultimos_100": 100, "ultimos_150": 150, "todo_dev": len(orden_dev)}
    vent_fig["n_batches_ventana"] = vent_fig["politica"].map(orden_x)
    vent_fig = vent_fig.sort_values("n_batches_ventana")
    fig, ax = plt.subplots(figsize=(9, 5))
    for (fase, clave), g in vent_fig.groupby(["fase", "target"]):
        ax.plot(g["n_batches_ventana"], g["r2_lockbox"], marker="o", label=f"{fase} {clave}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("N de batches DEV usados para entrenar (los mas recientes)")
    ax.set_ylabel("R2 en el lockbox")
    ax.set_title("R2 lockbox vs tamano de la ventana de reentrenamiento")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "r2_lockbox_vs_ventana.png", dpi=130)
    plt.close(fig)
    print("  Guardada figs_17/r2_lockbox_vs_ventana.png")

# 6c) theta por periodo con IC, por palanca (una figura por combo fase/target)
if not tabla_efectos.empty:
    orden_periodos = ["dev_primera_mitad", "dev_segunda_mitad", "lockbox"]
    for (fase, clave), g in tabla_efectos.groupby(["fase", "clave_target"]):
        palancas = sorted(g["palanca"].unique())
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(palancas) + 1.5))
        y0 = 0
        yticks, ylabels = [], []
        colores = {"dev_primera_mitad": "#4C72B0", "dev_segunda_mitad": "#55A868", "lockbox": "#C44E52"}
        for palanca in palancas:
            for j, periodo in enumerate(orden_periodos):
                fila = g.loc[(g["palanca"] == palanca) & (g["periodo"] == periodo)]
                if fila.empty:
                    continue
                fila = fila.iloc[0]
                y = y0 + j * 0.25
                ax.errorbar(fila["efecto_por_1sd"], y,
                            xerr=[[fila["efecto_por_1sd"] - fila["efecto_1sd_ci95_lo"]],
                                  [fila["efecto_1sd_ci95_hi"] - fila["efecto_por_1sd"]]],
                            fmt="o", color=colores[periodo], capsize=3,
                            label=periodo if palanca == palancas[0] else None)
            yticks.append(y0 + 0.25)
            ylabels.append(palanca)
            y0 += 1.2
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Efecto por +1 sd de la palanca (kg), IC95% bootstrap")
        ax.set_title(f"{fase} {clave}: efecto DML conjunto por periodo cronologico")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        nombre = f"theta_por_periodo_{fase}_{clave}.png".replace("ó", "o").replace("í", "i")
        fig.savefig(FIGS / nombre, dpi=130)
        plt.close(fig)
        print(f"  Guardada figs_17/{nombre}")

print("\nExperimento 17 terminado.")
