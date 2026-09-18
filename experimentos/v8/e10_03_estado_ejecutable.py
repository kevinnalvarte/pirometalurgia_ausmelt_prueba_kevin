"""E10-03 (iteración 10, 2026-09-16): estado ejecutable / latencia del ensayo (H3, `ITERACION_10_diseno.md`).

Pregunta: ¿cuánto R² e identificación del carbón se pierde en Reducción si el estado NO usa el ensayo de
laboratorio de t-1? Tres estados sobre las MISMAS filas (por target):

  A = k15 (`L.ESTADO_R_SIN_ESPESOR`)       -- referencia v7: usa el ensayo de t-1 (ley_*_prev, m6_*_inv_kg_prev)
  B = online (`L.ESTADO_R_ONLINE`)         -- leyes/inventarios de t-2 (`*_prev2`) + variables en línea de t-1
  C = plan   (construido aquí, sufijo `_F6`) -- leyes/inventarios congelados al cierre de F6 (fila R0 del batch,
              conocido antes de R1 en cualquier caso) + orden_escalon_fase + variables en línea de t-1

Las palancas cinéticas del carbón (`Cx_sn_v6`, `Cx_av_v6`) también dependen del inventario de estado: se
reconstruyen para B (`Cx_sn_online`, `Cx_av_online`, con `m6_sn_inv_kg_prev2`/`avance_prev2`) y para C
(`Cx_sn_F6`, `Cx_av_F6`, con `m6_sn_inv_kg_F6`/`avance_F6`). `avance_prev2`/`avance_F6` usan
`fe.division_segura` con umbral 500 kg (evita ratios inestables con `cum_feed_Sn_kg_prev` pequeño).

Uso (desde la raíz del proyecto):
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v8/e10_03_estado_ejecutable.py <etapa>
    <etapa> in {"global", "orden", "simulador_r0", "all"}
    global       -> R2 por conjuntos (OOF/cronológico/WF/lockbox) + referencia HGB directa + theta_dual (libre/acotado)
                    de A/B/C x {sn_dep, feo_ret} sobre las mismas filas    -> e10_03_estados.csv, e10_03_theta.csv
    orden        -> R2 OOF aleatorio (pooled) y sesgo medio por orden R0-R3, para A/B/C x {sn_dep, feo_ret}
                                                                            -> e10_03_por_orden.csv
    simulador_r0 -> en R0: A (ley de F6 real) vs B (ley de F5, *_prev2 en R0)  -> e10_03_simulador_r0.csv

`global` guarda incrementalmente y se salta combinaciones ya calculadas (columna `listo`) si se reinvoca tras un
timeout: reejecutar `global` las veces que haga falta hasta que la última línea de log diga "etapa global terminada".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "experimentos" / "v8"))
import v8_lib as L  # noqa: E402
import feature_engineering as fe  # noqa: E402

OUTDIR = RAIZ / "experimentos" / "v8"
SEED = L.SEED
TARGETS = {"sn_dep": "m6_ln_sn_dep", "feo_ret": "m6_ln_feo_ret"}
UMBRAL_SN_FED_KG = 500.0  # avance_prev2/avance_F6: denominador (cum_feed_Sn_kg_prev) minimo, mas conservador que el 20 kg de v6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- 1) columnas online (Cx_*_online) sobre el estado PLAN nativo
# NOTA (2026-09-16, corrida "simulador_r0"): `v8_lib.agregar_columnas_v8` fue actualizada en paralelo para incluir de forma
# NATIVA el estado PLAN (`L.ESTADO_R_PLAN`, columnas `*_F6`, `Cx_sn_F6`, `Cx_av_F6`) -- construcción idéntica a la que este
# script hacía a mano (mismo `.where(orden==0).groupby(Batch)`). Se adopta la versión de la librería (evita duplicar/])
# colisionar nombres) y aquí sólo se agregan las columnas ONLINE (`*_prev2`), que la librería no trae.
def construir_columnas_e10_03(r: pd.DataFrame) -> pd.DataFrame:
    r = r.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    assert "Cx_sn_F6" in r.columns and "m6_resto_frac_F6" in r.columns, "v8_lib sin estado PLAN nativo (ESTADO_R_PLAN); reejecutar tras actualizar v8_lib"

    tasa_c = r["tasa_feed_Carbon_kg_min"]
    r["avance_prev2"] = 1.0 - fe.division_segura(r["m6_sn_inv_kg_prev2"], r["cum_feed_Sn_kg_prev"], umbral=UMBRAL_SN_FED_KG)
    r["Cx_sn_online"] = tasa_c * r["m6_sn_inv_kg_prev2"] / 1000.0
    r["Cx_av_online"] = tasa_c * r["avance_prev2"]

    L.CLASIFICACION_M8.update({c: "DERIVED (A_t x S_prev)" for c in ("avance_prev2", "Cx_sn_online", "Cx_av_online")})
    return r


ESTADO_A = list(L.ESTADO_R_SIN_ESPESOR)   # k15: usa el ensayo de t-1
ESTADO_B = list(L.ESTADO_R_ONLINE)        # online: leyes/inventarios de t-2 + variables en línea de t-1
ESTADO_C = list(L.ESTADO_R_PLAN)          # plan: leyes/inventarios congelados al cierre de F6 (nativo de v8_lib) + en línea de t-1
ESTADOS = {"A_k15": ESTADO_A, "B_online": ESTADO_B, "C_plan": ESTADO_C}

FIJAS_SN = {"tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": 0}
FIJAS_FEO = {"tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1}
PALANCAS = {
    "A_k15": {"sn_dep": ["Cx_sn_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
              "feo_ret": ["tasa_feed_Carbon_kg_min", "Cx_av_v6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]},
    "B_online": {"sn_dep": ["Cx_sn_online", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
                 "feo_ret": ["tasa_feed_Carbon_kg_min", "Cx_av_online", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]},
    "C_plan": {"sn_dep": ["Cx_sn_F6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
               "feo_ret": ["tasa_feed_Carbon_kg_min", "Cx_av_F6", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]},
}


def signos_para(estado_key: str, target_key: str) -> dict:
    base = dict(FIJAS_SN if target_key == "sn_dep" else FIJAS_FEO)
    palancas = PALANCAS[estado_key][target_key]
    cx_col = palancas[0] if target_key == "sn_dep" else palancas[1]   # Cx_sn_* (sn_dep) / Cx_av_* (feo_ret)
    signo_cx = L.SIGNOS_R_SN_V7["Cx_sn_v6"] if target_key == "sn_dep" else L.SIGNOS_R_FEO_V7["Cx_av_v6"]
    base[cx_col] = signo_cx
    return base


# --------------------------------------------------------------------------- carga y filas comunes
def cargar():
    df = L.cargar_df()
    r = L.base_fase(df, "Reducción")
    r = construir_columnas_e10_03(r)
    dev_b, lb_b = L.split(df)
    return df, r, dev_b, lb_b


def filas_comunes_target(r: pd.DataFrame, target_key: str) -> pd.DataFrame:
    """Filas con estado + palancas COMPLETOS para A, B y C simultáneamente (mismo target): comparación como iguales."""
    target_col = TARGETS[target_key]
    cols = set()
    for estado_key, S in ESTADOS.items():
        cols |= set(S) | set(PALANCAS[estado_key][target_key])
    cols.add(target_col)
    L.asegurar_seguras(sorted(cols - {target_col}))
    return r.dropna(subset=sorted(cols)).reset_index(drop=True)


# --------------------------------------------------------------------------- etapa "global"
def etapa_global(r: pd.DataFrame, dev_b: set, lb_b: set) -> None:
    ruta_r2, ruta_theta = OUTDIR / "e10_03_estados.csv", OUTDIR / "e10_03_theta.csv"
    filas_r2 = pd.read_csv(ruta_r2).to_dict("records") if ruta_r2.exists() else []
    tablas_theta = [pd.read_csv(ruta_theta)] if ruta_theta.exists() else []
    ya_listo = {(f["estado"], f["target"]) for f in filas_r2}

    for target_key, target_col in TARGETS.items():
        d = filas_comunes_target(r, target_key)
        d_dev_all = d[d["Batch"].isin(dev_b)]
        log(f"[global {target_key}] filas comunes A/B/C: n={len(d)} (dev={len(d_dev_all)}, batches_dev={d_dev_all['Batch'].nunique()})")
        for estado_key, S in ESTADOS.items():
            if (estado_key, target_key) in ya_listo:
                log(f"[global {estado_key}/{target_key}] ya calculado, se salta")
                continue
            A = PALANCAS[estado_key][target_key]
            signos = signos_para(estado_key, target_key)
            d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
            d_lb = d[d["Batch"].isin(lb_b)].reset_index(drop=True)

            t0 = time.time()
            met = L.r2_por_conjuntos(d_dev, d_lb, S, A, target_col, signos, con_wf=True)
            # referencia HGB directa (S + A -> target), mismo OOF GroupKFold
            oof_hgb = L.oof_groupkfold(d_dev, L.fit_predict_hgb(S + A, target_col))
            met["r2_oof_hgb_ref"] = L.metricas(d_dev[target_col], oof_hgb)["r2"]
            met.update({"estado": estado_key, "target": target_key, "A": ",".join(A)})
            filas_r2.append(met)
            pd.DataFrame(filas_r2).to_csv(ruta_r2, index=False)

            tabla, extras = L.theta_dual(d_dev, S, A, target_col, signos, n_boot=500, seed=SEED)
            tabla["estado"] = estado_key
            tabla["target"] = target_key
            tablas_theta.append(tabla)
            pd.concat(tablas_theta, ignore_index=True).to_csv(ruta_theta, index=False)

            log(f"[global {estado_key}/{target_key}] n_dev={met['n_dev']} r2_oof={met['r2_oof']:.3f} "
                f"cron={met['r2_oof_cronologico']:.3f} wf={met.get('r2_walk_forward', float('nan')):.3f} "
                f"lb={met.get('r2_lockbox', float('nan')):.3f} hgb_ref={met['r2_oof_hgb_ref']:.3f} ({time.time()-t0:.1f}s)")
    log("etapa global terminada -> e10_03_estados.csv, e10_03_theta.csv")


# --------------------------------------------------------------------------- etapa "orden"
def etapa_orden(r: pd.DataFrame, dev_b: set, lb_b: set) -> None:
    filas = []
    for target_key, target_col in TARGETS.items():
        d = filas_comunes_target(r, target_key)
        d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
        for estado_key, S in ESTADOS.items():
            A = PALANCAS[estado_key][target_key]
            signos = signos_para(estado_key, target_key)
            t0 = time.time()
            fp = L.fit_predict_plm(S, A, target_col, signos)
            oof = L.oof_groupkfold(d_dev, fp)
            y = d_dev[target_col].to_numpy()
            for orden in sorted(d_dev["orden_escalon_fase"].unique()):
                m = (d_dev["orden_escalon_fase"] == orden).to_numpy()
                met = L.metricas(y[m], oof[m])
                sesgo = float(np.nanmean(y[m] - oof[m]))
                filas.append({"target": target_key, "estado": estado_key, "orden": int(orden), "n": met["n"],
                              "r2_oof": met["r2"], "mae_oof": met["mae"], "sesgo_medio": sesgo})
            log(f"[orden {estado_key}/{target_key}] n_dev={len(d_dev)} ({time.time()-t0:.1f}s)")
            pd.DataFrame(filas).to_csv(OUTDIR / "e10_03_por_orden.csv", index=False)

    tabla = pd.DataFrame(filas)
    ref = tabla[tabla["estado"] == "A_k15"].set_index(["target", "orden"])["r2_oof"]
    tabla["r2_ref_A"] = tabla.set_index(["target", "orden"]).index.map(ref)
    tabla["pct_r2_de_A"] = tabla["r2_oof"] / tabla["r2_ref_A"]
    tabla.to_csv(OUTDIR / "e10_03_por_orden.csv", index=False)
    log("Resumen pct_r2_de_A (B_online, C_plan) por orden:")
    log(tabla[tabla["estado"] != "A_k15"][["target", "estado", "orden", "n", "r2_oof", "r2_ref_A", "pct_r2_de_A", "sesgo_medio"]]
        .to_string(index=False))
    log("etapa orden terminada -> e10_03_por_orden.csv")


# --------------------------------------------------------------------------- etapa "simulador_r0"
def etapa_simulador_r0(r: pd.DataFrame, dev_b: set, lb_b: set) -> None:
    filas = []
    for target_key, target_col in TARGETS.items():
        cols = set(ESTADO_A) | set(ESTADO_B) | set(PALANCAS["A_k15"][target_key]) | set(PALANCAS["B_online"][target_key])
        cols.add(target_col)
        d = r[r["orden_escalon_fase"] == 0].dropna(subset=sorted(cols)).reset_index(drop=True)
        d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
        d_lb = d[d["Batch"].isin(lb_b)].reset_index(drop=True)
        log(f"[simulador_r0 {target_key}] filas comunes A/B en R0: n={len(d)} (dev={len(d_dev)}, lockbox={len(d_lb)})")
        for estado_key in ("A_k15", "B_online"):
            S, A = ESTADOS[estado_key], PALANCAS[estado_key][target_key]
            signos = signos_para(estado_key, target_key)
            t0 = time.time()
            met = L.r2_por_conjuntos(d_dev, d_lb, S, A, target_col, signos, con_wf=False)
            met.update({"estado": estado_key, "target": target_key, "A": ",".join(A)})
            filas.append(met)
            log(f"[simulador_r0 {estado_key}/{target_key}] n_dev={met['n_dev']} r2_oof={met['r2_oof']:.3f} "
                f"cron={met['r2_oof_cronologico']:.3f} lb={met.get('r2_lockbox', float('nan')):.3f} ({time.time()-t0:.1f}s)")
            pd.DataFrame(filas).to_csv(OUTDIR / "e10_03_simulador_r0.csv", index=False)
    log("etapa simulador_r0 terminada -> e10_03_simulador_r0.csv")


# --------------------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    etapa = sys.argv[1]
    log(f"cargando datos (etapa {etapa}) ...")
    df, r, dev_b, lb_b = cargar()
    log(f"df {df.shape}, r(Reducción) {r.shape}, dev_batches={len(dev_b)}, lockbox_batches={len(lb_b)}")
    if etapa in ("global", "all"):
        etapa_global(r, dev_b, lb_b)
    if etapa in ("orden", "all"):
        etapa_orden(r, dev_b, lb_b)
    if etapa in ("simulador_r0", "all"):
        etapa_simulador_r0(r, dev_b, lb_b)
    log("listo")


if __name__ == "__main__":
    main()
