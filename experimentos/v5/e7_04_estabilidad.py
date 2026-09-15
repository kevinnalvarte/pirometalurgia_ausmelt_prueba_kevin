"""E7-04 -- Parsimonia del estado, estabilidad temporal de los efectos de CONTROL y
reentrenamiento por campana (iteracion 7, disenio en ITERACION_7_diseno.md).

Preguntas: (1) cuantas columnas del estado S hacen falta para el R2 OOF del PLM (curva de
parsimonia por eliminacion hacia atras guiada por importancia de permutacion); (2) que tan
estables son los theta de las palancas de CONTROL entre tercios cronologicos de DEV, DEV
completo y el lockbox (entrenado solo con el lockbox: prueba de fragilidad); (3) cuanto se
degrada el R2 al pasar de OOF aleatorio (GroupKFold) a walk-forward cronologico; (4) que
politica de reentrenamiento en el lockbox (fijo/progresivo/ventana movil) conviene; (5) que
features de estado derivan (KS DEV vs lockbox, fraccion fuera de [P1,P99] DEV) y si conviene
excluirlas.

Referencia de enfoque (NO reutilizar targets): experimentos/17_estabilidad_temporal.py.
Libreria obligatoria: experimentos/v5/v5_lib.py (targets/palancas v5, `evaluar_plm`, anti-fuga).

Convenciones: .venv/Scripts/python.exe, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=4; sin
joblib/multiprocessing (n_jobs=1); OOF GroupKFold por batch en DEV = criterio; lockbox solo
reporte. Script pesado -> se ejecuta por etapas via argumento CLI:
    python e7_04_estabilidad.py <etapa>
    etapas: parsimonia | periodos | walkforward | reentrenamiento | deriva | todas
Salidas en experimentos/v5/: e7_04_parsimonia.csv, e7_04_set_parsimonioso.json,
e7_04_theta_periodos.csv, e7_04_walkforward.csv, e7_04_reentrenamiento.csv, e7_04_deriva.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent          # experimentos/v5
RAIZ = HERE.parent.parent                        # raiz del proyecto
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RAIZ))
import v5_lib as v5                              # noqa: E402
import modelo_predictivo_v4 as mp4               # noqa: E402
import modelo_prescriptivo as mp                 # noqa: E402

SEED = 42
BLOQUE_WF = 50           # walk-forward: tamanio de bloque cronologico en DEV
BLOQUE_LB = 10            # reentrenamiento progresivo: tamanio de bloque en el lockbox
VENTANA_MOVIL = 150       # reentrenamiento: ventana movil (batches)
N_BOOT_PERIODOS = 200
K_MIN = 5
PASO_ELIMINACION = 3
UMBRAL_R2_PARSIMONIA = 0.98   # k* = menor k con r2_oof >= 0.98 * max(r2_oof)

# --------------------------------------------------------------------------- parametrizaciones
PALANCAS_P2_SN = ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
PALANCAS_P2_FEO = ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
PALANCAS_P3_SN = ["v5_dosis_C_sn", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
PALANCAS_P3_FEO = ["v5_exceso_C_pos", "v5_C_x_avance", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
PALANCAS_FUSION = ["relacion_C_Sn_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]

# (fase, target) -> (P2, P3); Fusion solo tiene una parametrizacion (se usa como P2 y P3)
PARAMETRIZACIONES: dict[tuple[str, str], dict[str, list[str]]] = {
    ("Reducción", "v5_ln_sn_dep"): {"P2": PALANCAS_P2_SN, "P3": PALANCAS_P3_SN},
    ("Reducción", "v5_ln_feo_ret"): {"P2": PALANCAS_P2_FEO, "P3": PALANCAS_P3_FEO},
    ("Reducción", "v5_sdi_w10"): {"P2": PALANCAS_P2_FEO, "P3": PALANCAS_P3_FEO},
    ("Fusión", "v5_d_lnirf"): {"P2": PALANCAS_FUSION, "P3": PALANCAS_FUSION},
    ("Fusión", "sn_extraido_est_kg"): {"P2": PALANCAS_FUSION, "P3": PALANCAS_FUSION},
}
# combos usados en el paso 2 (parsimonia): exactamente los del disenio
COMBOS_PARSIMONIA = [
    ("Reducción", "v5_ln_sn_dep", "P3"),
    ("Reducción", "v5_ln_feo_ret", "P3"),
    ("Fusión", "v5_d_lnirf", "P3"),
    ("Fusión", "sn_extraido_est_kg", "P3"),
]
# combos usados en pasos 3-6 (estabilidad/walkforward/reentrenamiento/deriva): R con P2 y P3,
# F con una sola parametrizacion (evaluada una vez, reportada como P2==P3)
TARGETS_R = ["v5_ln_sn_dep", "v5_ln_feo_ret", "v5_sdi_w10"]
TARGETS_F = ["v5_d_lnirf", "sn_extraido_est_kg"]


def combos_estabilidad():
    combos = []
    for t in TARGETS_R:
        for p in ("P2", "P3"):
            combos.append(("Reducción", t, p))
    for t in TARGETS_F:
        combos.append(("Fusión", t, "P2P3"))   # unica parametrizacion para Fusion
    return combos


def palancas_de(fase: str, target: str, param: str) -> list[str]:
    if param == "P2P3":
        return PARAMETRIZACIONES[(fase, target)]["P2"]
    return PARAMETRIZACIONES[(fase, target)][param]


def estado_full(fase: str) -> list[str]:
    return list(v5.ESTADO_R_FULL if fase == "Reducción" else v5.ESTADO_F_FULL)


def estado_curado(fase: str) -> list[str]:
    return list(v5.ESTADO_R_CURADO if fase == "Reducción" else v5.ESTADO_F_CURADO)


def signo_de(target: str) -> dict:
    return v5.SIGNO_TEORICO_V5.get(target, {})


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 2) Curva de parsimonia del estado
# =============================================================================

def _hgb(seed=SEED):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=seed)


def importancia_permutacion(df_fase: pd.DataFrame, dev: set, estado: list[str], target: str, seed=SEED) -> pd.Series:
    """Hold-out por batch (75/25): HGB(S->target) entrenado en train, permutation_importance
    (5 repeticiones, n_jobs=1) sobre el hold-out. Devuelve importancia media ascendente."""
    need = estado + [target, "Batch"]
    d = df_fase.loc[df_fase["Batch"].isin(dev), need].dropna().reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(d[estado], d[target], d["Batch"]))
    modelo = _hgb(seed).fit(d[estado].iloc[tr], d[target].iloc[tr])
    r = permutation_importance(modelo, d[estado].iloc[te], d[target].iloc[te], n_repeats=5,
                                random_state=seed, n_jobs=1)
    return pd.Series(r.importances_mean, index=estado).sort_values()


def curva_parsimonia(df_fase: pd.DataFrame, estado_inicial: list[str], palancas: list[str], target: str) -> list[dict]:
    dev, _ = mp.split_dev_lockbox(df_fase)
    signo = signo_de(target)
    estado = [c for c in estado_inicial if v5.es_segura(c)]
    filas = []
    while True:
        imp = importancia_permutacion(df_fase, dev, estado, target)
        t0 = time.time()
        met, tabla, _ = v5.evaluar_plm(df_fase, estado, palancas, target, n_boot=0, signo_teorico=signo)
        tabla = tabla.set_index("palanca")
        fila = {"target": target, "tipo": "eliminacion", "k": len(estado), "features": ";".join(estado),
                "n_dev": met["n_dev"], "n_lockbox": met["n_lockbox"],
                "r2_oof": met["r2_oof"], "mae_oof": met["mae_oof"],
                "r2_lockbox": met["r2_lockbox"], "mae_lockbox": met["mae_lockbox"]}
        for p in palancas:
            fila[f"theta_{p}"] = float(tabla.loc[p, "theta_1sd"])
            fila[f"theta_lo_{p}"] = float(tabla.loc[p, "ci_lo_1sd"])
            fila[f"theta_hi_{p}"] = float(tabla.loc[p, "ci_hi_1sd"])
        filas.append(fila)
        log(f"  parsimonia {target} k={len(estado):2d} r2_oof={met['r2_oof']:.3f} r2_lockbox={met['r2_lockbox']:.3f} "
            f"({time.time()-t0:.1f}s)")
        if len(estado) <= K_MIN:
            break
        n_quitar = min(PASO_ELIMINACION, len(estado) - K_MIN)
        a_quitar = set(imp.index[:n_quitar].tolist())
        estado = [c for c in estado if c not in a_quitar]
    return filas


def etapa_parsimonia():
    df = v5.cargar_df()
    dR, dF = v5.base_fase(df, "Reducción"), v5.base_fase(df, "Fusión")
    d_por_fase = {"Reducción": dR, "Fusión": dF}
    todas_filas = []
    set_final: dict[str, dict] = {}
    for fase, target, param in COMBOS_PARSIMONIA:
        log(f"=== parsimonia {fase} / {target} (paramétrizacion {param}) ===")
        palancas = palancas_de(fase, target, param)
        df_fase = d_por_fase[fase]
        filas = curva_parsimonia(df_fase, estado_full(fase), palancas, target)
        # comparacion con estado curado
        signo = signo_de(target)
        met_c, tabla_c, _ = v5.evaluar_plm(df_fase, estado_curado(fase), palancas, target, n_boot=0, signo_teorico=signo)
        tabla_c = tabla_c.set_index("palanca")
        fila_c = {"target": target, "tipo": "curado", "k": len(estado_curado(fase)),
                  "features": ";".join(estado_curado(fase)), "n_dev": met_c["n_dev"], "n_lockbox": met_c["n_lockbox"],
                  "r2_oof": met_c["r2_oof"], "mae_oof": met_c["mae_oof"],
                  "r2_lockbox": met_c["r2_lockbox"], "mae_lockbox": met_c["mae_lockbox"]}
        for p in palancas:
            fila_c[f"theta_{p}"] = float(tabla_c.loc[p, "theta_1sd"])
            fila_c[f"theta_lo_{p}"] = float(tabla_c.loc[p, "ci_lo_1sd"])
            fila_c[f"theta_hi_{p}"] = float(tabla_c.loc[p, "ci_hi_1sd"])
        log(f"  parsimonia {target} CURADO k={fila_c['k']} r2_oof={met_c['r2_oof']:.3f} r2_lockbox={met_c['r2_lockbox']:.3f}")
        filas.append(fila_c)
        for f in filas:
            f["fase"] = fase
            f["parametrizacion"] = param
        todas_filas.extend(filas)
        # k* sobre la curva de eliminacion (no sobre curado)
        elim = [f for f in filas if f["tipo"] == "eliminacion"]
        r2max = max(f["r2_oof"] for f in elim)
        elegibles = sorted([f for f in elim if f["r2_oof"] >= UMBRAL_R2_PARSIMONIA * r2max], key=lambda f: f["k"])
        ganador = elegibles[0]
        set_final[target] = {"fase": fase, "k_estrella": ganador["k"], "features": ganador["features"].split(";"),
                              "r2_oof": ganador["r2_oof"], "r2_lockbox": ganador["r2_lockbox"], "r2max_curva": r2max,
                              "palancas": palancas, "parametrizacion": param}
        log(f"  -> k*={ganador['k']} (r2_oof={ganador['r2_oof']:.3f}, {UMBRAL_R2_PARSIMONIA*100:.0f}% del maximo {r2max:.3f})")
    # sdi_w10 hereda la union de features de sn_dep y feo_ret (no se corre curva propia; ver memo)
    if "v5_ln_sn_dep" in set_final and "v5_ln_feo_ret" in set_final:
        union = list(dict.fromkeys(set_final["v5_ln_sn_dep"]["features"] + set_final["v5_ln_feo_ret"]["features"]))
        set_final["v5_sdi_w10"] = {"fase": "Reducción", "k_estrella": len(union), "features": union,
                                    "r2_oof": np.nan, "r2_lockbox": np.nan, "r2max_curva": np.nan,
                                    "palancas": PALANCAS_P3_FEO, "parametrizacion": "P3",
                                    "nota": "union de v5_ln_sn_dep y v5_ln_feo_ret (sin curva propia)"}
    tabla = pd.DataFrame(todas_filas)
    tabla.to_csv(HERE / "e7_04_parsimonia.csv", index=False)
    with open(HERE / "e7_04_set_parsimonioso.json", "w", encoding="utf-8") as fh:
        json.dump(set_final, fh, ensure_ascii=False, indent=2)
    log("Guardado e7_04_parsimonia.csv y e7_04_set_parsimonioso.json")


# =============================================================================
# 3) Estabilidad temporal de theta por periodo (estado curado)
# =============================================================================

def periodos_de(df_fase: pd.DataFrame) -> dict[str, list]:
    dev, lockbox = mp.split_dev_lockbox(df_fase)
    orden = mp.orden_cronologico_batches(df_fase)
    orden_dev = [b for b in orden if b in dev]
    orden_lb = [b for b in orden if b in lockbox]
    n = len(orden_dev)
    return {"dev_tercio1": orden_dev[: n // 3], "dev_tercio2": orden_dev[n // 3: 2 * n // 3],
            "dev_tercio3": orden_dev[2 * n // 3:], "dev_completo": orden_dev, "lockbox": orden_lb}


def theta_en_periodo(df_fase, batches, estado, palancas, target, signo, seed=SEED):
    need = list(dict.fromkeys(estado + palancas + [target, "Batch"]))
    d = df_fase.loc[df_fase["Batch"].isin(set(batches)), need].dropna().reset_index(drop=True)
    if len(d) < 30 or d["Batch"].nunique() < 5:
        return None, len(d), d["Batch"].nunique() if len(d) else 0
    m = mp4.ModeloPLM(estado, palancas, target).fit(d, n_boot=N_BOOT_PERIODOS, seed=seed)
    tabla = m.tabla_theta(signo).reset_index()
    return tabla, len(d), d["Batch"].nunique()


def etapa_periodos():
    df = v5.cargar_df()
    d_por_fase = {"Reducción": v5.base_fase(df, "Reducción"), "Fusión": v5.base_fase(df, "Fusión")}
    filas = []
    for fase, target, param in combos_estabilidad():
        df_fase = d_por_fase[fase]
        estado = estado_curado(fase)
        palancas = palancas_de(fase, target, param)
        signo = signo_de(target)
        periodos = periodos_de(df_fase)
        log(f"=== periodos {fase}/{target}/{param} ===")
        for nombre_p, batches_p in periodos.items():
            t0 = time.time()
            tabla, n, nb = theta_en_periodo(df_fase, batches_p, estado, palancas, target, signo)
            if tabla is None:
                log(f"  {nombre_p}: insuficiente (n={n}, batches={nb})")
                continue
            for _, r in tabla.iterrows():
                filas.append({"fase": fase, "target": target, "parametrizacion": param, "periodo": nombre_p,
                              "palanca": r["palanca"], "theta_1sd": r["theta_1sd"], "ci_lo_1sd": r["ci_lo_1sd"],
                              "ci_hi_1sd": r["ci_hi_1sd"], "significativo": bool(r["significativo"]),
                              "n": n, "n_batches": nb})
            log(f"  {nombre_p}: n={n} batches={nb} ({time.time()-t0:.1f}s)")
    tabla_larga = pd.DataFrame(filas)

    # tabla de estabilidad: signo por periodo, mismo_signo_en_todos, significativo_en_k_periodos,
    # cociente theta_lockbox / theta_DEV_completo
    resumen = []
    for (fase, target, param, palanca), g in tabla_larga.groupby(["fase", "target", "parametrizacion", "palanca"]):
        g = g.set_index("periodo")
        periodos_presentes = [p for p in ["dev_tercio1", "dev_tercio2", "dev_tercio3", "dev_completo", "lockbox"] if p in g.index]
        signos = {p: float(np.sign(g.loc[p, "theta_1sd"])) for p in periodos_presentes}
        tercios = [p for p in ["dev_tercio1", "dev_tercio2", "dev_tercio3"] if p in signos]
        mismo_signo = len(set(signos[p] for p in tercios)) == 1 if len(tercios) == 3 else np.nan
        n_sig = int(sum(bool(g.loc[p, "significativo"]) for p in periodos_presentes))
        theta_dev = g.loc["dev_completo", "theta_1sd"] if "dev_completo" in g.index else np.nan
        theta_lb = g.loc["lockbox", "theta_1sd"] if "lockbox" in g.index else np.nan
        cociente = theta_lb / theta_dev if theta_dev not in (0, np.nan) and not pd.isna(theta_dev) else np.nan
        fila = {"fase": fase, "target": target, "parametrizacion": param, "palanca": palanca,
                "mismo_signo_en_todos": mismo_signo, "significativo_en_k_periodos": n_sig,
                "theta_dev_completo": theta_dev, "theta_lockbox": theta_lb, "ratio_lockbox_dev": cociente}
        for p in periodos_presentes:
            fila[f"signo_{p}"] = signos[p]
            fila[f"theta_{p}"] = float(g.loc[p, "theta_1sd"])
        resumen.append(fila)
    tabla_resumen = pd.DataFrame(resumen)
    out = tabla_larga.merge(tabla_resumen[["fase", "target", "parametrizacion", "palanca", "mismo_signo_en_todos",
                                            "significativo_en_k_periodos", "ratio_lockbox_dev"]],
                             on=["fase", "target", "parametrizacion", "palanca"], how="left")
    out.to_csv(HERE / "e7_04_theta_periodos.csv", index=False)
    tabla_resumen.to_csv(HERE / "e7_04_theta_periodos_resumen.csv", index=False)
    log("Guardado e7_04_theta_periodos.csv y e7_04_theta_periodos_resumen.csv")


# =============================================================================
# 4) Walk-forward cronologico vs OOF aleatorio
# =============================================================================

def etapa_walkforward():
    df = v5.cargar_df()
    d_por_fase = {"Reducción": v5.base_fase(df, "Reducción"), "Fusión": v5.base_fase(df, "Fusión")}
    filas = []
    for fase, target, param in combos_estabilidad():
        df_fase = d_por_fase[fase]
        estado = estado_curado(fase)
        palancas = palancas_de(fase, target, param)
        signo = signo_de(target)
        dev, _ = mp.split_dev_lockbox(df_fase)
        orden = mp.orden_cronologico_batches(df_fase)
        orden_dev = [b for b in orden if b in dev]
        need = list(dict.fromkeys(estado + palancas + [target, "Batch"]))
        log(f"=== walkforward {fase}/{target}/{param} ===")
        bloques = [orden_dev[i:i + BLOQUE_WF] for i in range(0, len(orden_dev), BLOQUE_WF)]
        reales, preds = [], []
        for i in range(1, len(bloques)):
            train_b = set(b for bl in bloques[:i] for b in bl)
            test_b = set(bloques[i])
            d_tr = df_fase.loc[df_fase["Batch"].isin(train_b), need].dropna().reset_index(drop=True)
            d_te = df_fase.loc[df_fase["Batch"].isin(test_b), need].dropna().reset_index(drop=True)
            if len(d_tr) < 50 or len(d_te) < 5:
                continue
            t0 = time.time()
            m = mp4.ModeloPLM(estado, palancas, target).fit(d_tr, n_boot=0, seed=SEED)
            p = m.predict(d_te)
            r2_b = r2_score(d_te[target], p) if d_te[target].std() > 0 else np.nan
            mae_b = mean_absolute_error(d_te[target], p)
            filas.append({"fase": fase, "target": target, "parametrizacion": param, "bloque": i,
                          "tipo": "walkforward_bloque", "n_train": len(d_tr), "n_test": len(d_te),
                          "r2": r2_b, "mae": mae_b})
            log(f"  bloque {i}: n_train={len(d_tr):4d} n_test={len(d_te):4d} r2={r2_b:.3f} ({time.time()-t0:.1f}s)")
            reales.append(d_te[target].to_numpy()); preds.append(p)
        if reales:
            reales_all, preds_all = np.concatenate(reales), np.concatenate(preds)
            r2_wf = r2_score(reales_all, preds_all)
            mae_wf = mean_absolute_error(reales_all, preds_all)
        else:
            r2_wf = mae_wf = np.nan
        filas.append({"fase": fase, "target": target, "parametrizacion": param, "bloque": "global_walkforward",
                      "tipo": "walkforward_global", "n_train": np.nan, "n_test": len(reales_all) if reales else 0,
                      "r2": r2_wf, "mae": mae_wf})
        # OOF aleatorio (GroupKFold) de referencia via v5_lib
        t0 = time.time()
        met_r, _, _ = v5.evaluar_plm(df_fase, estado, palancas, target, n_boot=0, signo_teorico=signo)
        filas.append({"fase": fase, "target": target, "parametrizacion": param, "bloque": "global_oof_aleatorio",
                      "tipo": "oof_aleatorio", "n_train": np.nan, "n_test": met_r["n_dev"],
                      "r2": met_r["r2_oof"], "mae": met_r["mae_oof"]})
        filas.append({"fase": fase, "target": target, "parametrizacion": param, "bloque": "brecha",
                      "tipo": "brecha_aleatorio_menos_walkforward", "n_train": np.nan, "n_test": np.nan,
                      "r2": met_r["r2_oof"] - r2_wf, "mae": mae_wf - met_r["mae_oof"]})
        log(f"  GLOBAL {target}/{param}: oof_aleatorio r2={met_r['r2_oof']:.3f} | walkforward r2={r2_wf:.3f} "
            f"| brecha={met_r['r2_oof']-r2_wf:+.3f} ({time.time()-t0:.1f}s)")
    pd.DataFrame(filas).to_csv(HERE / "e7_04_walkforward.csv", index=False)
    log("Guardado e7_04_walkforward.csv")


# =============================================================================
# 5) Reentrenamiento por campana sobre el lockbox
# =============================================================================

def etapa_reentrenamiento():
    df = v5.cargar_df()
    d_por_fase = {"Reducción": v5.base_fase(df, "Reducción"), "Fusión": v5.base_fase(df, "Fusión")}
    filas = []
    for fase, target, param in combos_estabilidad():
        df_fase = d_por_fase[fase]
        estado = estado_curado(fase)
        palancas = palancas_de(fase, target, param)
        dev, lockbox = mp.split_dev_lockbox(df_fase)
        orden = mp.orden_cronologico_batches(df_fase)
        orden_dev = [b for b in orden if b in dev]
        orden_lb = [b for b in orden if b in lockbox]
        need = list(dict.fromkeys(estado + palancas + [target, "Batch"]))
        log(f"=== reentrenamiento {fase}/{target}/{param} ===")

        def _fila(politica, d_tr, d_te, m):
            p = m.predict(d_te)
            fila = {"fase": fase, "target": target, "parametrizacion": param, "politica": politica,
                    "n_train": len(d_tr), "n_test": len(d_te),
                    "r2_lockbox": r2_score(d_te[target], p) if d_te[target].std() > 0 else np.nan,
                    "mae_lockbox": mean_absolute_error(d_te[target], p)}
            for j, pal in enumerate(palancas):
                fila[f"theta_{pal}"] = float(m.theta[j])
            return fila

        d_dev_full = df_fase.loc[df_fase["Batch"].isin(set(orden_dev)), need].dropna().reset_index(drop=True)
        d_lb_full = df_fase.loc[df_fase["Batch"].isin(set(orden_lb)), need].dropna().reset_index(drop=True)
        if len(d_dev_full) < 50 or len(d_lb_full) < 20:
            log("  insuficiente, se omite"); continue

        # (a) fijo: entrenado solo en DEV
        t0 = time.time()
        m_fijo = mp4.ModeloPLM(estado, palancas, target).fit(d_dev_full, n_boot=0, seed=SEED)
        filas.append(_fila("fijo_DEV", d_dev_full, d_lb_full, m_fijo))
        log(f"  fijo_DEV: r2_lockbox={filas[-1]['r2_lockbox']:.3f} ({time.time()-t0:.1f}s)")

        bloques_lb = [orden_lb[i:i + BLOQUE_LB] for i in range(0, len(orden_lb), BLOQUE_LB)]

        # (b) progresivo: ventana expansiva DEV + lockbox visto
        vistos: list = []
        reales_b, preds_b, m_prog = [], [], None
        for i, bl in enumerate(bloques_lb):
            d_vistos = df_fase.loc[df_fase["Batch"].isin(set(vistos)), need].dropna() if vistos else d_dev_full.iloc[:0]
            d_tr = pd.concat([d_dev_full, d_vistos], ignore_index=True) if len(d_vistos) else d_dev_full
            d_te = df_fase.loc[df_fase["Batch"].isin(set(bl)), need].dropna().reset_index(drop=True)
            if len(d_te) == 0:
                continue
            m_prog = mp4.ModeloPLM(estado, palancas, target).fit(d_tr.reset_index(drop=True), n_boot=0, seed=SEED)
            preds_b.append(m_prog.predict(d_te)); reales_b.append(d_te[target].to_numpy())
            vistos.extend(bl)
        if m_prog is not None:
            reales_all, preds_all = np.concatenate(reales_b), np.concatenate(preds_b)
            fila = {"fase": fase, "target": target, "parametrizacion": param, "politica": "progresivo_expandido_10",
                    "n_train": len(d_dev_full) + len(vistos), "n_test": len(reales_all),
                    "r2_lockbox": r2_score(reales_all, preds_all) if np.std(reales_all) > 0 else np.nan,
                    "mae_lockbox": mean_absolute_error(reales_all, preds_all)}
            for j, pal in enumerate(palancas):
                fila[f"theta_{pal}"] = float(m_prog.theta[j])
            filas.append(fila)
            log(f"  progresivo_expandido_10: r2_lockbox={fila['r2_lockbox']:.3f}")

        # (c) ventana movil: ultimos VENTANA_MOVIL batches de (DEV + lockbox visto)
        vistos = []
        reales_c, preds_c, m_mov = [], [], None
        for i, bl in enumerate(bloques_lb):
            historial = (orden_dev + vistos)[-VENTANA_MOVIL:]
            d_tr = df_fase.loc[df_fase["Batch"].isin(set(historial)), need].dropna().reset_index(drop=True)
            d_te = df_fase.loc[df_fase["Batch"].isin(set(bl)), need].dropna().reset_index(drop=True)
            if len(d_te) == 0 or len(d_tr) < 30:
                continue
            m_mov = mp4.ModeloPLM(estado, palancas, target).fit(d_tr, n_boot=0, seed=SEED)
            preds_c.append(m_mov.predict(d_te)); reales_c.append(d_te[target].to_numpy())
            vistos.extend(bl)
        if m_mov is not None:
            reales_all, preds_all = np.concatenate(reales_c), np.concatenate(preds_c)
            fila = {"fase": fase, "target": target, "parametrizacion": param, "politica": f"ventana_movil_{VENTANA_MOVIL}",
                    "n_train": VENTANA_MOVIL, "n_test": len(reales_all),
                    "r2_lockbox": r2_score(reales_all, preds_all) if np.std(reales_all) > 0 else np.nan,
                    "mae_lockbox": mean_absolute_error(reales_all, preds_all)}
            for j, pal in enumerate(palancas):
                fila[f"theta_{pal}"] = float(m_mov.theta[j])
            filas.append(fila)
            log(f"  ventana_movil_{VENTANA_MOVIL}: r2_lockbox={fila['r2_lockbox']:.3f}")

    pd.DataFrame(filas).to_csv(HERE / "e7_04_reentrenamiento.csv", index=False)
    log("Guardado e7_04_reentrenamiento.csv")


# =============================================================================
# 6) Deriva de features (KS DEV vs lockbox, fraccion fuera de [P1,P99])
# =============================================================================

def etapa_deriva():
    ruta_json = HERE / "e7_04_set_parsimonioso.json"
    if not ruta_json.exists():
        raise SystemExit("Falta e7_04_set_parsimonioso.json: correr la etapa 'parsimonia' primero")
    with open(ruta_json, encoding="utf-8") as fh:
        set_parsimonioso = json.load(fh)

    df = v5.cargar_df()
    d_por_fase = {"Reducción": v5.base_fase(df, "Reducción"), "Fusión": v5.base_fase(df, "Fusión")}

    filas = []
    for target, info in set_parsimonioso.items():
        fase = info["fase"]
        df_fase = d_por_fase[fase]
        dev, lockbox = mp.split_dev_lockbox(df_fase)
        cols = list(dict.fromkeys(info["features"] + info["palancas"]))
        for c in cols:
            if c not in df_fase.columns:
                continue
            x_dev = df_fase.loc[df_fase["Batch"].isin(dev), c].dropna()
            x_lb = df_fase.loc[df_fase["Batch"].isin(lockbox), c].dropna()
            if len(x_dev) < 10 or len(x_lb) < 10:
                continue
            ks_stat, ks_p = stats.ks_2samp(x_dev, x_lb)
            p1, p99 = x_dev.quantile([0.01, 0.99])
            frac_fuera = float(((x_lb < p1) | (x_lb > p99)).mean())
            filas.append({"target": target, "fase": fase, "feature": c, "rol": "estado" if c in info["features"] else "palanca",
                          "ks_stat": float(ks_stat), "ks_p": float(ks_p), "frac_fuera_p1p99_lockbox": frac_fuera,
                          "p1_dev": float(p1), "p99_dev": float(p99), "media_dev": float(x_dev.mean()),
                          "media_lockbox": float(x_lb.mean()), "n_dev": len(x_dev), "n_lockbox": len(x_lb)})
        log(f"deriva calculada para {target} ({len(cols)} columnas)")

    tabla_deriva = pd.DataFrame(filas)

    # PLM con vs sin las features 100% fuera de rango, dentro del set parsimonioso de cada target
    filas_comp = []
    for target, info in set_parsimonioso.items():
        fase = info["fase"]
        df_fase = d_por_fase[fase]
        estado = info["features"]
        palancas = info["palancas"]
        cien_pct_fuera = tabla_deriva.loc[(tabla_deriva["target"] == target) & (tabla_deriva["rol"] == "estado") &
                                           (tabla_deriva["frac_fuera_p1p99_lockbox"] >= 0.999), "feature"].tolist()
        signo = signo_de(target)
        t0 = time.time()
        met_con, _, _ = v5.evaluar_plm(df_fase, estado, palancas, target, n_boot=0, signo_teorico=signo)
        fila = {"target": target, "fase": fase, "features_100pct_fuera": ";".join(cien_pct_fuera) or "(ninguna)",
                "r2_oof_con": met_con["r2_oof"], "r2_lockbox_con": met_con["r2_lockbox"]}
        if cien_pct_fuera:
            estado_sin = [c for c in estado if c not in cien_pct_fuera]
            met_sin, _, _ = v5.evaluar_plm(df_fase, estado_sin, palancas, target, n_boot=0, signo_teorico=signo)
            fila.update({"r2_oof_sin": met_sin["r2_oof"], "r2_lockbox_sin": met_sin["r2_lockbox"]})
        else:
            fila.update({"r2_oof_sin": np.nan, "r2_lockbox_sin": np.nan})
        filas_comp.append(fila)
        log(f"  {target}: 100% fuera={cien_pct_fuera or 'ninguna'} r2_oof con={fila['r2_oof_con']:.3f} "
            f"sin={fila.get('r2_oof_sin', np.nan)} r2_lockbox con={fila['r2_lockbox_con']:.3f} "
            f"sin={fila.get('r2_lockbox_sin', np.nan)} ({time.time()-t0:.1f}s)")

    tabla_deriva = tabla_deriva.merge(pd.DataFrame(filas_comp), on=["target", "fase"], how="left")
    tabla_deriva.to_csv(HERE / "e7_04_deriva.csv", index=False)
    log("Guardado e7_04_deriva.csv")


# =============================================================================
# main
# =============================================================================

ETAPAS = {"parsimonia": etapa_parsimonia, "periodos": etapa_periodos, "walkforward": etapa_walkforward,
          "reentrenamiento": etapa_reentrenamiento, "deriva": etapa_deriva}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=list(ETAPAS) + ["todas"])
    args = ap.parse_args()
    t0 = time.time()
    if args.etapa == "todas":
        for nombre, fn in ETAPAS.items():
            log(f"##### etapa {nombre} #####")
            fn()
    else:
        ETAPAS[args.etapa]()
    log(f"listo en {time.time()-t0:.1f}s")
