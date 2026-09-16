"""E9-06 (iteracion 9): estado parsimonioso de Reduccion que conserve el R2 e identifique el carbon.

Uso:
    .venv/Scripts/python.exe experimentos/v7/e9_06_parsimonia.py <etapa>
    etapas: importancia | eliminacion | sets_teoria | estabilidad | varianza_theta | resumen

Contexto: PLM v6 (ModeloPLMSignos) sobre el estado FULL de Reduccion (38 columnas, v7_lib.ESTADO_R_FULL + m6_resto_frac_prev)
identifica el carbon (Cx_sn_v6 +0.16 sd [0.03, 0.29]) en el target m6_ln_sn_dep; con el estado CURADO (22) el IC cruza 0.
Hipotesis: la identificacion exige la historia de dosificacion (cum_feed_Carbon_kg_prev, relacion_C_cum_Sn_cum_prev,
cum_gn_nm3_prev, ...) porque el operador dosifica el carbon segun el estado y su historia (confusion).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7_lib as L  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402  (mismo objeto de modulo que usa mp5.ModeloPLMSignos)

from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

pd.set_option("display.width", 220)
SEED = L.SEED
OUT = HERE
FASE = "Reducción"

# --------------------------------------------------------------------------- estado / targets / palancas / signos
S_FULL = list(L.ESTADO[(FASE, "sn_dep")])
assert S_FULL == L.ESTADO[(FASE, "feo_ret")], "el estado FULL deberia ser el mismo para ambos targets"
CLAVES = ["sn_dep", "feo_ret"]
TARGETS = {k: L.TARGETS[FASE][k] for k in CLAVES}
PALANCAS = {k: L.PALANCAS[(FASE, k)] for k in CLAVES}
SIGNOS = {k: L.SIGNOS[(FASE, k)] for k in CLAVES}
CURADO = list(L.ESTADO_R_CURADO)

# nucleo fisico (nunca se elimina en la ruta "fisica")
NUCLEO = ["m6_sn_inv_kg_prev", "m6_feo_inv_kg_prev", "m6_masa_kg_prev", "m6_avance_prev",
          "temperatura_horno_celsius_prev", "orden_escalon_fase", "cum_feed_Carbon_kg_prev",
          "relacion_C_cum_Sn_cum_prev", "basicidad_B2_prev", "indice_irf_prev"]
assert all(c in S_FULL for c in NUCLEO), [c for c in NUCLEO if c not in S_FULL]

# sets por teoria (documentados en sets_teoria())
LEYES_PREV = ["ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "ley_sio2_escoria_pct_prev", "ley_cao_escoria_pct_prev"]
TERMICOS_LANZA = ["termocupla_media_celsius_prev", "grad_temperatura_horno_celsius_prev", "posicion_vertical_lanza_mm_prev"]
GASES_HIST = ["cum_gn_nm3_prev", "cum_o2_nm3_prev", "cum_aire_nm3_prev"]
CARGA_LEYES = ["ley_sn_carga_batch_pct", "ley_sn_dross_batch_pct", "ley_sn_pellets_batch_pct"]
HIST_DOSIF = ["cum_feed_Carbon_kg_prev", "relacion_C_cum_Sn_cum_prev"] + GASES_HIST  # historia de dosificacion (H6)

T1 = list(NUCLEO)
T2 = list(dict.fromkeys(T1 + LEYES_PREV + ["m6_resto_frac_prev"]))
T3 = list(dict.fromkeys(T2 + TERMICOS_LANZA))
T4 = list(dict.fromkeys(T3 + GASES_HIST + ["tiempo_fase", "espesor_ladrillo_norm_mm"] + CARGA_LEYES))
for _s in (T2, T3, T4):
    assert all(c in S_FULL for c in _s), [c for c in _s if c not in S_FULL]


def sin_dosificacion(S: list[str]) -> list[str]:
    return [c for c in S if c not in HIST_DOSIF]


SETS_TEORIA = {
    "T1_nucleo_fisico": (T1, "10 variables de balance/estado terminal: inventarios Sn/FeO/masa, avance, T horno, orden del "
                              "escalon, historia de carbon (acumulado y C/Sn acumulado), basicidad B2, IRF"),
    "T2_leyes": (T2, "T1 + leyes de ensayo previas (%Sn,%FeO,%SiO2,%CaO) + fraccion no reducible (m6_resto_frac_prev): "
                     "completa la composicion de la escoria que T1 solo resume via ratios"),
    "T3_termico_lanza": (T3, "T2 + termicos (termocupla, gradiente T horno) + posicion de lanza previa: geometria/energia "
                             "del bano que modula la cinetica de reduccion"),
    "T4_historia_gases": (T4, "T3 + historia acumulada de gases (GN,O2,aire) + tiempo de fase + espesor de ladrillo + leyes "
                              "de carga del batch (Sn en carga/dross/pellets): historia completa de dosificacion y contexto "
                              "de batch"),
    "T1_sin_dosif": (sin_dosificacion(T1), "T1 sin cum_feed_Carbon_kg_prev ni relacion_C_cum_Sn_cum_prev (prueba H6)"),
    "T2_sin_dosif": (sin_dosificacion(T2), "T2 sin la historia de dosificacion de carbon"),
    "T3_sin_dosif": (sin_dosificacion(T3), "T3 sin la historia de dosificacion de carbon"),
    "T4_sin_dosif": (sin_dosificacion(T4), "T4 sin toda la historia de dosificacion (carbon acum., C/Sn acum., GN/O2/aire acum.)"),
    "FULL": (S_FULL, "estado FULL v6 (38, referencia)"),
    "CURADO": (CURADO, "estado curado v5/v6 (22, referencia; sin historia de dosificacion salvo m6_resto_frac_prev)"),
}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- utilidades de datos
def cargar():
    df = L.cargar_df()
    return L.base_fase(df, FASE)


def datos_para(dR: pd.DataFrame, clave: str, S: list[str]):
    target, A = TARGETS[clave], PALANCAS[clave]
    d_dev, d_lb = L.preparar(dR, S + A, target)
    return d_dev, d_lb, target, A


def hgb():
    return L.hgb_nuisance(SEED)


# --------------------------------------------------------------------------- OOF generico (g(S) o m(S) de una columna)
def oof_fold_models(d: pd.DataFrame, S: list[str], col: str, n_splits: int = 5, seed: int = SEED):
    folds = []
    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits).split(d[S], d[col], d["Batch"]):
        mdl = hgb().fit(d[S].iloc[tr], d[col].iloc[tr])
        oof[te] = mdl.predict(d[S].iloc[te])
        folds.append((te, mdl))
    return folds, oof


def permutacion_importancia(d: pd.DataFrame, S: list[str], target: str, n_rep: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Importancia por permutacion OOF (GroupKFold 5), n_rep repeticiones, agrupada por batch (se permuta dentro
    de cada fold de prueba, sin mezclar informacion entre folds)."""
    folds, oof = oof_fold_models(d, S, target, seed=seed)
    r2_base = r2_score(d[target], oof)
    rng = np.random.default_rng(seed)
    filas = []
    for feat in S:
        drops = []
        for _ in range(n_rep):
            oof_perm = oof.copy()
            for te, mdl in folds:
                Xte = d[S].iloc[te].copy()
                Xte[feat] = rng.permutation(Xte[feat].to_numpy())
                oof_perm[te] = mdl.predict(Xte)
            drops.append(r2_base - r2_score(d[target], oof_perm))
        filas.append({"target": target, "feature": feat, "importancia_r2": float(np.mean(drops)),
                      "sd_importancia": float(np.std(drops)), "r2_base": r2_base})
    return pd.DataFrame(filas)


def nested_oof_r2(d_dev: pd.DataFrame, S: list[str], A: list[str], target: str, signos: dict, n_splits: int = 5,
                   seed: int = SEED):
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev[S], d_dev[target], d_dev["Batch"]):
        m = L.ModeloPLMSignos(S, A, target, signos).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
        oof[te] = m.predict(d_dev.iloc[te])
    return float(r2_score(d_dev[target], oof)), oof


def evaluar_set(d_dev, d_lb, S, A, target, signos, n_boot=200, seed=SEED, n_splits=5):
    r2_oof, _ = nested_oof_r2(d_dev, S, A, target, signos, n_splits=n_splits, seed=seed)
    m = L.ModeloPLMSignos(S, A, target, signos).fit(d_dev, n_boot=n_boot, seed=seed, n_splits=n_splits)
    r2_lb = float(r2_score(d_lb[target], m.predict(d_lb)))
    t = m.tabla_theta(signos)
    return dict(r2_oof=r2_oof, r2_lockbox=r2_lb, n_dev=len(d_dev), n_lockbox=len(d_lb)), t, m


def _fila_palanca(fila: dict, t: pd.DataFrame, clave: str, palanca: str, prefijo: str):
    if palanca in t.index:
        fila[f"{prefijo}_theta1sd"] = t.loc[palanca, "theta_1sd"]
        fila[f"{prefijo}_lo"] = t.loc[palanca, "ci_lo_1sd"]
        fila[f"{prefijo}_hi"] = t.loc[palanca, "ci_hi_1sd"]
        fila[f"{prefijo}_en_cota"] = bool(t.loc[palanca, "en_cota"])


def evaluar_checkpoint(dR, S, nombre_extra: dict, seed=SEED) -> dict:
    fila = dict(nombre_extra)
    fila["k"] = len(S)
    fila["features"] = ";".join(S)
    for clave in CLAVES:
        d_dev, d_lb, target, A = datos_para(dR, clave, S)
        signos = SIGNOS[clave]
        met, t, _m = evaluar_set(d_dev, d_lb, S, A, target, signos, seed=seed)
        fila[f"{clave}_r2_oof"] = met["r2_oof"]
        fila[f"{clave}_r2_lockbox"] = met["r2_lockbox"]
        fila[f"{clave}_n_dev"] = met["n_dev"]
        if clave == "sn_dep":
            _fila_palanca(fila, t, clave, "Cx_sn_v6", "Cx_sn")
        if clave == "feo_ret":
            _fila_palanca(fila, t, clave, "tasa_feed_Carbon_kg_min", "C_feo")
        _fila_palanca(fila, t, clave, "tasa_gn_nm3_min", f"GN_{clave}")
    log("checkpoint", {k: v for k, v in fila.items() if k != "features"})
    return fila


# --------------------------------------------------------------------------- 1) importancia
def etapa_importancia():
    dR = cargar()
    filas_pred = []
    for clave in CLAVES:
        d_dev, _d_lb, target, _A = datos_para(dR, clave, S_FULL)
        t0 = time.time()
        imp = permutacion_importancia(d_dev, S_FULL, target, n_rep=5, seed=SEED)
        imp.insert(0, "tipo", "predictiva")
        imp.insert(1, "clave", clave)
        filas_pred.append(imp)
        log(f"[importancia] predictiva {clave} lista en {time.time()-t0:.1f}s")

    # identificacion: leave-one-out sobre el estado, n_boot=100, siguiendo theta+IC de la palanca foco
    focos = [("sn_dep", "Cx_sn_v6"), ("feo_ret", "tasa_feed_Carbon_kg_min")]
    filas_id = []
    for clave, foco in focos:
        d_dev, _d_lb, target, A = datos_para(dR, clave, S_FULL)
        signos = SIGNOS[clave]
        t0 = time.time()
        m_full = L.ModeloPLMSignos(S_FULL, A, target, signos).fit(d_dev, n_boot=100, seed=SEED)
        t_full = m_full.tabla_theta(signos)
        base_lo, base_hi = t_full.loc[foco, "ci_lo_1sd"], t_full.loc[foco, "ci_hi_1sd"]
        base_theta = t_full.loc[foco, "theta_1sd"]
        filas_id.append({"tipo": "identificacion", "clave": clave, "target": target, "palanca_foco": foco,
                          "feature_quitada": "(ninguna, FULL)", "theta_foco_1sd": base_theta,
                          "ci_lo": base_lo, "ci_hi": base_hi, "ancho_ic": base_hi - base_lo,
                          "excluye_0_signo_correcto": bool((base_lo > 0) if signos.get(foco, 0) > 0 else (base_hi < 0)),
                          "delta_theta_vs_full": 0.0, "delta_ancho_vs_full": 0.0})
        for feat in S_FULL:
            S_loo = [c for c in S_FULL if c != feat]
            m = L.ModeloPLMSignos(S_loo, A, target, signos).fit(d_dev, n_boot=100, seed=SEED)
            t = m.tabla_theta(signos)
            th, lo, hi = t.loc[foco, ["theta_1sd", "ci_lo_1sd", "ci_hi_1sd"]]
            filas_id.append({"tipo": "identificacion", "clave": clave, "target": target, "palanca_foco": foco,
                              "feature_quitada": feat, "theta_foco_1sd": th, "ci_lo": lo, "ci_hi": hi,
                              "ancho_ic": hi - lo,
                              "excluye_0_signo_correcto": bool((lo > 0) if signos.get(foco, 0) > 0 else (hi < 0)),
                              "delta_theta_vs_full": th - base_theta, "delta_ancho_vs_full": (hi - lo) - (base_hi - base_lo)})
        log(f"[importancia] identificacion {clave}/{foco} lista en {time.time()-t0:.1f}s")

    out = pd.concat(filas_pred + [pd.DataFrame(filas_id)], ignore_index=True)
    out.to_csv(OUT / "e9_06_importancia.csv", index=False)
    log(out.head(10).round(4).to_string())
    log("FIN importancia")


# --------------------------------------------------------------------------- 2) eliminacion hacia atras
def combinar_importancia(dR, S, n_rep=5, seed=SEED) -> pd.Series:
    d_dev_sn, _, target_sn, _ = datos_para(dR, "sn_dep", S)
    d_dev_feo, _, target_feo, _ = datos_para(dR, "feo_ret", S)
    imp_sn = permutacion_importancia(d_dev_sn, S, target_sn, n_rep=n_rep, seed=seed).set_index("feature")["importancia_r2"]
    imp_feo = permutacion_importancia(d_dev_feo, S, target_feo, n_rep=n_rep, seed=seed).set_index("feature")["importancia_r2"]

    def norm(s):
        mx = s.abs().max()
        return s / mx if mx and mx > 0 else s * 0.0

    comb = (norm(imp_sn).reindex(S).fillna(0) + norm(imp_feo).reindex(S).fillna(0)) / 2
    return comb.sort_values()


CHECKPOINTS = [38, 30, 24, 20, 16, 12, 10, 8, 6]


def backward_elim(dR, ruta: str, seed=SEED) -> pd.DataFrame:
    S = list(S_FULL)
    protegido = set(NUCLEO) if ruta == "fisica" else set()
    piso = max(len(protegido), min(CHECKPOINTS)) if ruta == "fisica" else min(CHECKPOINTS)
    filas = []
    vistos_k = set()

    def registrar():
        k = len(S)
        if k in CHECKPOINTS and k not in vistos_k:
            filas.append(evaluar_checkpoint(dR, S, {"ruta": ruta}, seed=seed))
            vistos_k.add(k)

    registrar()
    ranking = None
    quitadas_desde_recalculo = 4  # fuerza recalculo inicial
    while len(S) > piso:
        if ranking is None or quitadas_desde_recalculo >= 4:
            t0 = time.time()
            ranking = combinar_importancia(dR, S, n_rep=5, seed=seed)
            log(f"[eliminacion:{ruta}] importancia recalculada (k={len(S)}) en {time.time()-t0:.1f}s")
            quitadas_desde_recalculo = 0
        cands = [f for f in ranking.index if f in S and f not in protegido]
        if not cands:
            break
        quitar = cands[0]
        S.remove(quitar)
        quitadas_desde_recalculo += 1
        log(f"[eliminacion:{ruta}] quita '{quitar}' -> k={len(S)}")
        registrar()

    # checkpoints no alcanzables (ruta fisica, k < 10): documentar explicitamente
    for k in CHECKPOINTS:
        if k not in vistos_k and k < piso:
            filas.append({"ruta": ruta, "k": k, "features": None,
                          "nota": f"no aplica: nucleo fisico protegido tiene {len(protegido)} features, no se puede bajar de {piso}"})
    return pd.DataFrame(filas)


def etapa_eliminacion():
    dR = cargar()
    out_libre = backward_elim(dR, "libre")
    out_fisica = backward_elim(dR, "fisica")
    out = pd.concat([out_libre, out_fisica], ignore_index=True)
    out.to_csv(OUT / "e9_06_eliminacion.csv", index=False)
    cols_show = [c for c in ["ruta", "k", "sn_dep_r2_oof", "sn_dep_r2_lockbox", "feo_ret_r2_oof", "feo_ret_r2_lockbox",
                              "Cx_sn_theta1sd", "Cx_sn_lo", "Cx_sn_hi", "Cx_sn_en_cota"] if c in out.columns]
    log(out[cols_show].round(4).to_string())
    log("FIN eliminacion")


# --------------------------------------------------------------------------- 3) sets por teoria
def etapa_sets_teoria():
    dR = cargar()
    filas = []
    for nombre, (S, doc) in SETS_TEORIA.items():
        t0 = time.time()
        fila = evaluar_checkpoint(dR, S, {"set": nombre, "doc": doc})
        # prueba de balance: cuanto explica el estado de Cx_sn_v6 (palanca de sn_dep)
        d_dev_sn, _d_lb, _t, _A = datos_para(dR, "sn_dep", S)
        folds_b, oof_b = oof_fold_models(d_dev_sn, S, "Cx_sn_v6", seed=SEED)
        r2_bal = float(r2_score(d_dev_sn["Cx_sn_v6"], oof_b))
        resid_sd = float(np.std(d_dev_sn["Cx_sn_v6"].to_numpy() - oof_b))
        raw_sd = float(d_dev_sn["Cx_sn_v6"].std())
        fila["balance_Cx_sn_r2_oof"] = r2_bal
        fila["balance_Cx_sn_resid_sd"] = resid_sd
        fila["balance_Cx_sn_raw_sd"] = raw_sd
        fila["balance_Cx_sn_frac_sd_residual"] = resid_sd / raw_sd if raw_sd else np.nan
        filas.append(fila)
        log(f"[sets_teoria] {nombre} (k={len(S)}) listo en {time.time()-t0:.1f}s: "
            f"sn_dep r2_oof={fila['sn_dep_r2_oof']:.3f} feo_ret r2_oof={fila['feo_ret_r2_oof']:.3f} "
            f"balance_r2={r2_bal:.3f}")
    out = pd.DataFrame(filas)
    out.to_csv(OUT / "e9_06_sets.csv", index=False)
    cols_show = [c for c in ["set", "k", "sn_dep_r2_oof", "sn_dep_r2_lockbox", "feo_ret_r2_oof", "feo_ret_r2_lockbox",
                              "Cx_sn_theta1sd", "Cx_sn_lo", "Cx_sn_hi", "balance_Cx_sn_r2_oof"] if c in out.columns]
    log(out[cols_show].round(4).to_string())
    log("FIN sets_teoria")


# --------------------------------------------------------------------------- 4) estabilidad
def theta_por_tercio(d_dev, S, A, target, signos, seed=SEED) -> pd.DataFrame:
    terc = L.tercios_cronologicos(d_dev, n=3)
    filas = []
    for tnum in sorted(terc.dropna().unique()):
        sub = d_dev[terc.values == tnum].reset_index(drop=True)
        if sub["Batch"].nunique() < 10 or len(sub) < 60:
            continue
        m = L.ModeloPLMSignos(S, A, target, signos).fit(sub, n_boot=100, seed=seed)
        t = m.tabla_theta(signos)
        for a in A:
            filas.append({"tercio": int(tnum), "palanca": a, "theta_1sd": t.loc[a, "theta_1sd"],
                          "ci_lo": t.loc[a, "ci_lo_1sd"], "ci_hi": t.loc[a, "ci_hi_1sd"], "n": len(sub)})
    return pd.DataFrame(filas)


def walkforward(d_dev, S, A, target, signos, bloque=50, seed=SEED):
    orden = d_dev.groupby("Batch")["fecha_inicio"].min().sort_values()
    batches_ordenados = orden.index.to_numpy()
    bloques = [batches_ordenados[i:i + bloque] for i in range(0, len(batches_ordenados), bloque)]
    filas, preds_all, ys_all = [], [], []
    for i in range(1, len(bloques)):
        train_batches = np.concatenate(bloques[:i])
        test_batches = bloques[i]
        train = d_dev[d_dev.Batch.isin(train_batches)].reset_index(drop=True)
        test = d_dev[d_dev.Batch.isin(test_batches)].reset_index(drop=True)
        if len(train) < 60 or len(test) < 10:
            continue
        m = L.ModeloPLMSignos(S, A, target, signos).fit(train, n_boot=0, seed=seed)
        pred = m.predict(test)
        filas.append({"bloque": i, "n_train": len(train), "n_test": len(test),
                      "r2_bloque": float(r2_score(test[target], pred)) if len(test) > 5 else np.nan})
        preds_all.append(pred)
        ys_all.append(test[target].to_numpy())
    r2_global = float(r2_score(np.concatenate(ys_all), np.concatenate(preds_all))) if preds_all else np.nan
    return pd.DataFrame(filas), r2_global


def elegir_mejores_sets(k_max=20, k_min=6):
    cand = []
    try:
        elim = pd.read_csv(OUT / "e9_06_eliminacion.csv")
        for _, r in elim.dropna(subset=["features"]).iterrows():
            k = len(str(r["features"]).split(";"))
            if not (k_min <= k <= k_max):
                continue
            score_id = 1.0 if (pd.notna(r.get("Cx_sn_lo")) and r.get("Cx_sn_lo") > 0) else 0.0
            score_r2 = np.nanmean([r.get("sn_dep_r2_oof", np.nan), r.get("feo_ret_r2_oof", np.nan)])
            cand.append({"origen": "eliminacion", "nombre": f"{r['ruta']}_k{k}", "k": k,
                        "features": r["features"], "score_id": score_id, "score_r2": score_r2})
    except FileNotFoundError:
        log("[estabilidad] e9_06_eliminacion.csv no encontrado, se omite como fuente de candidatos")
    try:
        sets_ = pd.read_csv(OUT / "e9_06_sets.csv")
        for _, r in sets_.dropna(subset=["features"]).iterrows():
            if r["set"] in ("FULL", "CURADO"):
                continue
            k = len(str(r["features"]).split(";"))
            if not (k_min <= k <= k_max):
                continue
            score_id = 1.0 if (pd.notna(r.get("Cx_sn_lo")) and r.get("Cx_sn_lo") > 0) else 0.0
            score_r2 = np.nanmean([r.get("sn_dep_r2_oof", np.nan), r.get("feo_ret_r2_oof", np.nan)])
            cand.append({"origen": "sets_teoria", "nombre": r["set"], "k": k,
                        "features": r["features"], "score_id": score_id, "score_r2": score_r2})
    except FileNotFoundError:
        log("[estabilidad] e9_06_sets.csv no encontrado, se omite como fuente de candidatos")
    if not cand:
        return []
    cand = pd.DataFrame(cand).sort_values(["score_id", "score_r2"], ascending=[False, False])
    elegidos, vistos = [], set()
    for _, r in cand.iterrows():
        if r["features"] in vistos:
            continue
        vistos.add(r["features"])
        elegidos.append(r)
        if len(elegidos) == 2:
            break
    return elegidos


def etapa_estabilidad():
    dR = cargar()
    configs = [("FULL", S_FULL), ("CURADO", CURADO)]
    for r in elegir_mejores_sets():
        configs.append((r["nombre"], str(r["features"]).split(";")))
    if len(configs) < 4:
        log(f"[estabilidad] AVISO: solo {len(configs)-2} sets parsimoniosos disponibles (se esperaban 2); "
            "correr eliminacion/sets_teoria primero.")
    filas_tercio, filas_wf = [], []
    for nombre, S in configs:
        for clave in CLAVES:
            d_dev, _d_lb, target, A = datos_para(dR, clave, S)
            signos = SIGNOS[clave]
            t0 = time.time()
            tt = theta_por_tercio(d_dev, S, A, target, signos)
            tt.insert(0, "config", nombre)
            tt.insert(1, "clave", clave)
            filas_tercio.append(tt)
            wf, r2g = walkforward(d_dev, S, A, target, signos)
            wf.insert(0, "config", nombre)
            wf.insert(1, "clave", clave)
            wf["r2_global_walkforward"] = r2g
            filas_wf.append(wf)
            log(f"[estabilidad] {nombre}/{clave} listo en {time.time()-t0:.1f}s (r2 walkforward global={r2g:.3f})")
    out_tercio = pd.concat(filas_tercio, ignore_index=True) if filas_tercio else pd.DataFrame()
    out_wf = pd.concat(filas_wf, ignore_index=True) if filas_wf else pd.DataFrame()
    out_tercio["tabla"] = "tercio"
    out_wf["tabla"] = "walkforward"
    out = pd.concat([out_tercio, out_wf], ignore_index=True, sort=False)
    out.to_csv(OUT / "e9_06_estabilidad.csv", index=False)
    log(out.round(4).to_string())
    log("FIN estabilidad")


# --------------------------------------------------------------------------- 5) varianza de theta
_HGB_ORIG = mp4._hgb_nuisance


def _factory_hgb300(seed):
    return _HGB_ORIG(seed, max_iter=300)


def _factory_ridge(seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))


def _fit_con_nuisance(S, A, target, signos, d_dev, n_splits=5, n_boot=200, seed=SEED, nuisance=None):
    if nuisance is None:
        return L.ModeloPLMSignos(S, A, target, signos).fit(d_dev, n_splits=n_splits, n_boot=n_boot, seed=seed)
    mp4._hgb_nuisance = nuisance
    try:
        return L.ModeloPLMSignos(S, A, target, signos).fit(d_dev, n_splits=n_splits, n_boot=n_boot, seed=seed)
    finally:
        mp4._hgb_nuisance = _HGB_ORIG


def etapa_varianza_theta():
    dR = cargar()
    mejores = elegir_mejores_sets()
    if not mejores:
        log("[varianza_theta] AVISO: no hay set parsimonioso elegido (correr eliminacion/sets_teoria primero); "
            "se usa CURADO como sustituto")
        mejor_nombre, mejor_S = "CURADO", CURADO
    else:
        mejor_nombre, mejor_S = mejores[0]["nombre"], str(mejores[0]["features"]).split(";")

    clave, foco = "sn_dep", "Cx_sn_v6"
    signos = SIGNOS[clave]
    filas = []
    for nombre, S in [("FULL", S_FULL), (mejor_nombre, mejor_S)]:
        d_dev, _d_lb, target, A = datos_para(dR, clave, S)
        variantes = [
            ("baseline_hgb150_5fold_boot200", dict(n_splits=5, n_boot=200, nuisance=None)),
            ("boot500", dict(n_splits=5, n_boot=500, nuisance=None)),
            ("nuisance_hgb300", dict(n_splits=5, n_boot=200, nuisance=_factory_hgb300)),
            ("nuisance_ridge", dict(n_splits=5, n_boot=200, nuisance=_factory_ridge)),
            ("crossfit_10fold", dict(n_splits=10, n_boot=200, nuisance=None)),
        ]
        for var_nombre, kw in variantes:
            t0 = time.time()
            m = _fit_con_nuisance(S, A, target, signos, d_dev, seed=SEED, **kw)
            t = m.tabla_theta(signos)
            th, lo, hi = t.loc[foco, ["theta_1sd", "ci_lo_1sd", "ci_hi_1sd"]]
            filas.append({"config": nombre, "k": len(S), "variante": var_nombre, "theta_1sd": th,
                          "ci_lo": lo, "ci_hi": hi, "ancho_ic": hi - lo,
                          "excluye_0_signo_correcto": bool(lo > 0), "segundos": time.time() - t0})
            log(f"[varianza_theta] {nombre}/{var_nombre}: theta={th:.4f} IC=[{lo:.4f},{hi:.4f}] ({time.time()-t0:.1f}s)")
    out = pd.DataFrame(filas)
    out.to_csv(OUT / "e9_06_varianza_theta.csv", index=False)
    log(out.round(4).to_string())
    log("FIN varianza_theta")


# --------------------------------------------------------------------------- 6) resumen
def etapa_resumen():
    filas = []
    try:
        elim = pd.read_csv(OUT / "e9_06_eliminacion.csv")
        for _, r in elim.iterrows():
            filas.append({"origen": "eliminacion", "nombre": f"{r['ruta']}_k{r['k']}", "k": r["k"],
                          "sn_dep_r2_oof": r.get("sn_dep_r2_oof"), "sn_dep_r2_lockbox": r.get("sn_dep_r2_lockbox"),
                          "feo_ret_r2_oof": r.get("feo_ret_r2_oof"), "feo_ret_r2_lockbox": r.get("feo_ret_r2_lockbox"),
                          "Cx_sn_theta1sd": r.get("Cx_sn_theta1sd"), "Cx_sn_lo": r.get("Cx_sn_lo"), "Cx_sn_hi": r.get("Cx_sn_hi"),
                          "Cx_sn_en_cota": r.get("Cx_sn_en_cota")})
    except FileNotFoundError:
        pass
    try:
        sets_ = pd.read_csv(OUT / "e9_06_sets.csv")
        for _, r in sets_.iterrows():
            filas.append({"origen": "sets_teoria", "nombre": r["set"], "k": r["k"],
                          "sn_dep_r2_oof": r.get("sn_dep_r2_oof"), "sn_dep_r2_lockbox": r.get("sn_dep_r2_lockbox"),
                          "feo_ret_r2_oof": r.get("feo_ret_r2_oof"), "feo_ret_r2_lockbox": r.get("feo_ret_r2_lockbox"),
                          "Cx_sn_theta1sd": r.get("Cx_sn_theta1sd"), "Cx_sn_lo": r.get("Cx_sn_lo"), "Cx_sn_hi": r.get("Cx_sn_hi"),
                          "Cx_sn_en_cota": r.get("Cx_sn_en_cota"), "balance_Cx_sn_r2_oof": r.get("balance_Cx_sn_r2_oof")})
    except FileNotFoundError:
        pass
    out = pd.DataFrame(filas)
    r2_full = out.loc[out["nombre"] == "FULL", "sn_dep_r2_oof"]
    r2_full_sn = float(r2_full.iloc[0]) if len(r2_full) else np.nan
    r2_full_feo = out.loc[out["nombre"] == "FULL", "feo_ret_r2_oof"]
    r2_full_feo = float(r2_full_feo.iloc[0]) if len(r2_full_feo) else np.nan
    out["delta_r2_sn_dep_vs_full"] = out["sn_dep_r2_oof"] - r2_full_sn
    out["delta_r2_feo_ret_vs_full"] = out["feo_ret_r2_oof"] - r2_full_feo
    out["cumple_R2"] = (out["delta_r2_sn_dep_vs_full"] >= -0.01) & (out["delta_r2_feo_ret_vs_full"] >= -0.01)
    out["cumple_identificacion"] = (out["Cx_sn_lo"] > 0)
    out["cumple_diseno"] = out["cumple_R2"] & out["cumple_identificacion"]
    out = out.sort_values(["cumple_diseno", "k"], ascending=[False, True])
    out.to_csv(OUT / "e9_06_resumen.csv", index=False)
    log(out.round(4).to_string())
    ganadores = out[out["cumple_diseno"]]
    if len(ganadores):
        log(f"\nVEREDICTO: {len(ganadores)} set(s) cumplen el criterio (Delta R2 >= -0.01 en ambos targets, IC Cx_sn > 0). "
            f"Recomendado (menor k): {ganadores.iloc[0]['nombre']} (k={ganadores.iloc[0]['k']}).")
    else:
        log("\nVEREDICTO: ningun set <= 20 cumple ambos criterios simultaneamente; ver e9_06_resumen.csv para el mejor compromiso.")
    log("FIN resumen")


ETAPAS = {
    "importancia": etapa_importancia,
    "eliminacion": etapa_eliminacion,
    "sets_teoria": etapa_sets_teoria,
    "estabilidad": etapa_estabilidad,
    "varianza_theta": etapa_varianza_theta,
    "resumen": etapa_resumen,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ETAPAS:
        print(f"uso: python e9_06_parsimonia.py <{'|'.join(ETAPAS)}>")
        sys.exit(1)
    t0 = time.time()
    ETAPAS[sys.argv[1]]()
    log(f"tiempo total: {time.time()-t0:.1f}s")
