"""E10-01 (iteración 10, 2026-09-16): forma funcional del carbón en Reducción y óptimo interior.

Compara 8 formas de la palanca "carbón" sobre `m6_ln_sn_dep` (F0 = Cx_sn_v6 de v7; F1-F7 alternativas: dosis
específica, saturante, cuadrática, heterogénea) y 5 formas sobre `m6_ln_feo_ret` (G0 = referencia v7; G1-G4:
exceso de carbón, saturante, heterogénea), con las demás palancas de la lanza fijas en su signo de teoría.

Uso (desde la raíz del proyecto):
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v8/e10_01_formas_carbon.py <etapa>
    <etapa> in {"1", "2", "3", "4", "all"}
    1 -> R² por conjuntos + theta_dual (libre/acotado) de cada forma       -> e10_01_formas.csv, e10_01_theta.csv
    2 -> estabilidad por tercios cronológicos del término principal        -> e10_01_tercios.csv
         (F0, F2, F3, F4 sobre Sn)
    3 -> dosis-respuesta no paramétrica (residuos cross-fitted, F5 y F2)   -> e10_01_dosis_respuesta.csv
         + partial dependence del HGB directo sobre tasa_feed_Carbon_kg_min por orden de escalón -> e10_01_pd_carbon.csv
    4 -> óptimo interior de J_R(C) en 12 estados representativos           -> e10_01_optimo_interior.csv, e10_01_curvas_J.csv

Todas las etapas son deterministas (SEED=42) y usan sólo DEV para ajustar/decidir; el lockbox aparece únicamente en
las columnas *_lockbox de e10_01_formas.csv (reporte, no selección).
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

OUTDIR = RAIZ / "experimentos" / "v8"
SEED = L.SEED


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- definición de formas (diseño E10-01)
S = L.ESTADO_R_SIN_ESPESOR  # 15 features
TARGET_SN = "m6_ln_sn_dep"
TARGET_FEO = "m6_ln_feo_ret"

FIJAS_SN = {"tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": 0}
FIJAS_FEO = {"tasa_gn_nm3_min": -1, "exceso_o2_combustion_pct": +1, "tasa_aire_nm3_min": +1}

# forma -> (columnas del término de carbón, signos de esas columnas)
FORMAS_SN: dict[str, tuple[list[str], dict[str, int]]] = {
    "F0": (["Cx_sn_v6"], {"Cx_sn_v6": +1}),
    "F1": (["m8_dosis"], {"m8_dosis": +1}),
    "F2": (["m8_ln1p_dosis"], {"m8_ln1p_dosis": +1}),
    "F3": (["m8_dosis", "m8_dosis_sq"], {"m8_dosis": +1, "m8_dosis_sq": -1}),
    "F4": (["m8_ln1p_dosis", "m8_ln1p_dosis_x_av"], {"m8_ln1p_dosis": +1, "m8_ln1p_dosis_x_av": -1}),
    "F5": (["tasa_feed_Carbon_kg_min"], {"tasa_feed_Carbon_kg_min": +1}),
    "F6": (["m8_ln1p_dosis", "m8_C_x_zT", "m8_C_x_zB2"], {"m8_ln1p_dosis": +1, "m8_C_x_zT": 0, "m8_C_x_zB2": 0}),
    "F7": (["m8_ln1p_dosis", "m8_gn_sq"], {"m8_ln1p_dosis": +1, "m8_gn_sq": -1}),
}
FORMAS_FEO: dict[str, tuple[list[str], dict[str, int]]] = {
    "G0": (["tasa_feed_Carbon_kg_min", "Cx_av_v6"], {"tasa_feed_Carbon_kg_min": -1, "Cx_av_v6": -1}),
    "G1": (["m8_exceso_C_pos"], {"m8_exceso_C_pos": -1}),
    "G2": (["m8_exceso_C_pos", "Cx_av_v6"], {"m8_exceso_C_pos": -1, "Cx_av_v6": -1}),
    "G3": (["m8_ln1p_dosis"], {"m8_ln1p_dosis": -1}),
    "G4": (["m8_exceso_C_pos", "m8_C_x_zT", "m8_C_x_zB2"], {"m8_exceso_C_pos": -1, "m8_C_x_zT": 0, "m8_C_x_zB2": 0}),
}
PRINCIPAL_SN = {"F0": "Cx_sn_v6", "F2": "m8_ln1p_dosis", "F3": "m8_dosis", "F4": "m8_ln1p_dosis"}

# objetivo (auditoría A §4, CONFIG_V7_SIN_LEGADO)
K_UNIDAD = 2.893 * 512.0   # kg Sn por unidad de ln_sn_dep
W_FEO = 6.09
COSTO_C = 0.05             # kg Sn equivalente por kg de carbón consumido


def armar_ab(carbon_cols: list[str], carbon_signos: dict[str, int], fijas: dict[str, int]) -> tuple[list[str], dict[str, int]]:
    A = list(carbon_cols) + list(fijas.keys())
    signos = dict(carbon_signos)
    signos.update(fijas)
    return A, signos


def preparar_forma(r: pd.DataFrame, dev_b: set, lb_b: set, S: list[str], A: list[str], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Réplica textual del diseño: `d = r.dropna(subset=S + A + [target])`, luego separar DEV/lockbox."""
    d = r.dropna(subset=S + A + [target]).copy()
    d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
    d_lb = d[d["Batch"].isin(lb_b)].reset_index(drop=True)
    return d_dev, d_lb


def cargar():
    df = L.cargar_df()
    r = L.base_fase(df, "Reducción")
    dev_b, lb_b = L.split(df)
    return df, r, dev_b, lb_b


# --------------------------------------------------------------------------- etapa 1
def etapa1(r, dev_b, lb_b):
    filas_r2, filas_theta = [], []
    for nombre, (ccols, csign) in FORMAS_SN.items():
        A, signos = armar_ab(ccols, csign, FIJAS_SN)
        L.asegurar_seguras(S + A)
        d_dev, d_lb = preparar_forma(r, dev_b, lb_b, S, A, TARGET_SN)
        t0 = time.time()
        met = L.r2_por_conjuntos(d_dev, d_lb, S, A, TARGET_SN, signos)
        met.update({"forma": nombre, "target": "sn_dep", "A": ",".join(A)})
        filas_r2.append(met)
        tabla, extras = L.theta_dual(d_dev, S, A, TARGET_SN, signos, n_boot=500, seed=SEED)
        tabla["forma"] = nombre; tabla["target"] = "sn_dep"
        filas_theta.append(tabla)
        log(f"[SN {nombre}] n_dev={met['n_dev']} r2_oof={met['r2_oof']:.3f} cron={met['r2_oof_cronologico']:.3f} "
            f"wf={met.get('r2_walk_forward', float('nan')):.3f} lb={met.get('r2_lockbox', float('nan')):.3f} "
            f"({time.time()-t0:.1f}s)")
        pd.DataFrame(filas_r2).to_csv(OUTDIR / "e10_01_formas.csv", index=False)
        pd.concat(filas_theta, ignore_index=True).to_csv(OUTDIR / "e10_01_theta.csv", index=False)
    for nombre, (ccols, csign) in FORMAS_FEO.items():
        A, signos = armar_ab(ccols, csign, FIJAS_FEO)
        L.asegurar_seguras(S + A)
        d_dev, d_lb = preparar_forma(r, dev_b, lb_b, S, A, TARGET_FEO)
        t0 = time.time()
        met = L.r2_por_conjuntos(d_dev, d_lb, S, A, TARGET_FEO, signos)
        met.update({"forma": nombre, "target": "feo_ret", "A": ",".join(A)})
        filas_r2.append(met)
        tabla, extras = L.theta_dual(d_dev, S, A, TARGET_FEO, signos, n_boot=500, seed=SEED)
        tabla["forma"] = nombre; tabla["target"] = "feo_ret"
        filas_theta.append(tabla)
        log(f"[FEO {nombre}] n_dev={met['n_dev']} r2_oof={met['r2_oof']:.3f} cron={met['r2_oof_cronologico']:.3f} "
            f"wf={met.get('r2_walk_forward', float('nan')):.3f} lb={met.get('r2_lockbox', float('nan')):.3f} "
            f"({time.time()-t0:.1f}s)")
        pd.DataFrame(filas_r2).to_csv(OUTDIR / "e10_01_formas.csv", index=False)
        pd.concat(filas_theta, ignore_index=True).to_csv(OUTDIR / "e10_01_theta.csv", index=False)
    log("etapa1 terminada -> e10_01_formas.csv, e10_01_theta.csv")


# --------------------------------------------------------------------------- etapa 2
def etapa2(r, dev_b, lb_b):
    filas = []
    for nombre in ["F0", "F2", "F3", "F4"]:
        ccols, csign = FORMAS_SN[nombre]
        A, signos = armar_ab(ccols, csign, FIJAS_SN)
        d_dev, _ = preparar_forma(r, dev_b, lb_b, S, A, TARGET_SN)
        terc = L.tercios_cronologicos(d_dev, n=3)
        for k in sorted(terc.dropna().unique()):
            sub = d_dev[terc.to_numpy() == k].reset_index(drop=True)
            t0 = time.time()
            tabla, extras = L.theta_dual(sub, S, A, TARGET_SN, signos, n_boot=300, seed=SEED)
            row = tabla[tabla["palanca"] == PRINCIPAL_SN[nombre]].iloc[0].to_dict()
            row.update({"forma": nombre, "tercio": int(k), "n": len(sub)})
            filas.append(row)
            log(f"[tercios {nombre} t{int(k)}] n={len(sub)} theta_libre_1sd={row['theta_libre_1sd']:.4f} "
                f"[{row['ci_lo_libre_1sd']:.4f},{row['ci_hi_libre_1sd']:.4f}] ({time.time()-t0:.1f}s)")
            pd.DataFrame(filas).to_csv(OUTDIR / "e10_01_tercios.csv", index=False)
    log("etapa2 terminada -> e10_01_tercios.csv")


# --------------------------------------------------------------------------- etapa 3
def bootstrap_ci_por_batch(vals: np.ndarray, batches: np.ndarray, n_boot: int = 500, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(batches)
    idx_by = {b: np.where(batches == b)[0] for b in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
        boots.append(np.nanmean(vals[idx]))
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def etapa3(r, dev_b, lb_b):
    filas_dr = []
    for nombre in ["F5", "F2"]:
        ccols, csign = FORMAS_SN[nombre]
        A, signos = armar_ab(ccols, csign, FIJAS_SN)
        d_dev, _ = preparar_forma(r, dev_b, lb_b, S, A, TARGET_SN)
        t0 = time.time()
        tabla, extras = L.theta_dual(d_dev, S, A, TARGET_SN, signos, n_boot=10, seed=SEED)  # sólo se usan los residuos
        y_res, A_res = extras["y_res"], extras["A_res"]
        carbon_res = A_res[:, 0]  # primer término de A = carbón
        batches = d_dev["Batch"].to_numpy()
        dec = pd.qcut(carbon_res, 10, labels=False, duplicates="drop")
        for k in sorted(pd.unique(dec)):
            m = dec == k
            lo, hi = bootstrap_ci_por_batch(y_res[m], batches[m])
            filas_dr.append({"forma": nombre, "decil": int(k), "n": int(m.sum()),
                              "carbon_res_media": float(np.mean(carbon_res[m])),
                              "target_res_media": float(np.mean(y_res[m])),
                              "ci_lo": lo, "ci_hi": hi})
        from scipy.stats import spearmanr
        rho, p = spearmanr(carbon_res, y_res)
        log(f"[dosis-respuesta {nombre}] n={len(d_dev)} spearman(rho)={rho:.3f} p={p:.4f} ({time.time()-t0:.1f}s)")
        pd.DataFrame(filas_dr).to_csv(OUTDIR / "e10_01_dosis_respuesta.csv", index=False)

    # partial dependence del HGB directo (S + carbon + GN + O2 + aire -> target), por orden de escalón
    prim = ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"]
    Xcols = S + prim
    filas_pd = []
    for target in [TARGET_SN, TARGET_FEO]:
        d = r.dropna(subset=S + prim + [target]).copy()
        d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
        t0 = time.time()
        mdl = L.hgb_nuisance(SEED, 300).fit(d_dev[Xcols], d_dev[target])
        grid = np.linspace(d_dev["tasa_feed_Carbon_kg_min"].quantile(.05), d_dev["tasa_feed_Carbon_kg_min"].quantile(.95), 15)
        for orden in sorted(d_dev["orden_escalon_fase"].unique()):
            sub = d_dev[d_dev["orden_escalon_fase"] == orden]
            if len(sub) < 10:
                continue
            for c in grid:
                Xg = sub[Xcols].copy()
                Xg["tasa_feed_Carbon_kg_min"] = c
                pred = mdl.predict(Xg)
                filas_pd.append({"target": target, "orden": int(orden), "carbon": float(c),
                                  "pd_media": float(pred.mean()), "n": int(len(sub))})
        log(f"[PD {target}] n_dev={len(d_dev)} ({time.time()-t0:.1f}s)")
        pd.DataFrame(filas_pd).to_csv(OUTDIR / "e10_01_pd_carbon.csv", index=False)
    log("etapa3 terminada -> e10_01_dosis_respuesta.csv, e10_01_pd_carbon.csv")


# --------------------------------------------------------------------------- etapa 4
def zscore(val: pd.Series, col: str) -> pd.Series:
    mu, sd = L.CENTROS_V8[col]
    return (val - mu) / sd


def features_carbon(C: np.ndarray, dt: float, sn_disp: float, sn_inv_prev: float, avance_prev: float,
                     t_horno_prev: float, b2_prev: float, gn: float) -> dict[str, np.ndarray]:
    c_kg_total = C * dt
    dosis = c_kg_total / sn_disp
    ln1p = np.log1p(dosis)
    mu_t, sd_t = L.CENTROS_V8["temperatura_horno_celsius_prev"]
    mu_b, sd_b = L.CENTROS_V8["basicidad_B2_prev"]
    mu_gn, sd_gn = L.CENTROS_V8["tasa_gn_nm3_min"]
    return {
        "tasa_feed_Carbon_kg_min": C,
        "Cx_sn_v6": C * sn_inv_prev / 1000.0,
        "Cx_av_v6": C * avance_prev,
        "m8_dosis": dosis,
        "m8_ln1p_dosis": ln1p,
        "m8_dosis_sq": dosis ** 2,
        "m8_ln1p_dosis_x_av": ln1p * avance_prev,
        "m8_exceso_C_pos": np.clip(c_kg_total - L.C_ESTEQ_POR_SN * sn_disp, 0, None),
        "m8_C_x_zT": C * (t_horno_prev - mu_t) / sd_t,
        "m8_C_x_zB2": C * (b2_prev - mu_b) / sd_b,
        "m8_gn_sq": np.full_like(C, (gn - mu_gn) ** 2),
    }


def etapa4(r, dev_b, lb_b):
    sn_formas = ["F0", "F2", "F3", "F4"]
    feo_formas = ["G0", "G1", "G2"]
    # columnas necesarias para TODAS las formas + el objeto de estado/derivadas
    todas_A = set()
    for n in sn_formas:
        todas_A |= set(armar_ab(*FORMAS_SN[n], FIJAS_SN)[0])
    for n in feo_formas:
        todas_A |= set(armar_ab(*FORMAS_FEO[n], FIJAS_FEO)[0])
    extra_cols = ["m6_sn_disp", "m6_avance_prev", "m6_sn_inv_kg_prev", "temperatura_horno_celsius_prev",
                  "basicidad_B2_prev", "duracion_plan_min", "orden_escalon_fase", "Batch"]
    cols_comunes = list(dict.fromkeys(S + list(todas_A) + extra_cols + [TARGET_SN, TARGET_FEO]))
    d_all = r.dropna(subset=cols_comunes).copy()
    d_dev_all = d_all[d_all["Batch"].isin(dev_b)].reset_index(drop=True)
    log(f"[etapa4] filas comunes a todas las formas: dev={len(d_dev_all)}")

    # 1) ajustar los 7 modelos PLM (theta ACOTADO = el que ya produce ModeloPLMSignos) en su propio DEV completo
    modelos_sn, r2_sn = {}, {}
    for nombre in sn_formas:
        A, signos = armar_ab(*FORMAS_SN[nombre], FIJAS_SN)
        d_dev, _ = preparar_forma(r, dev_b, lb_b, S, A, TARGET_SN)
        modelos_sn[nombre] = L.ModeloPLMSignos(S, A, TARGET_SN, signos).fit(d_dev, n_boot=0, seed=SEED)
        log(f"[etapa4 fit SN {nombre}] n_dev={len(d_dev)} r2_oof_interno={modelos_sn[nombre].r2_oof_interno:.3f}")
    modelos_feo = {}
    for nombre in feo_formas:
        A, signos = armar_ab(*FORMAS_FEO[nombre], FIJAS_FEO)
        d_dev, _ = preparar_forma(r, dev_b, lb_b, S, A, TARGET_FEO)
        modelos_feo[nombre] = L.ModeloPLMSignos(S, A, TARGET_FEO, signos).fit(d_dev, n_boot=0, seed=SEED)
        log(f"[etapa4 fit FEO {nombre}] n_dev={len(d_dev)} r2_oof_interno={modelos_feo[nombre].r2_oof_interno:.3f}")

    # 2) 12 estados representativos: 3 por orden (cuantiles 25/50/75 de m6_avance_prev dentro del orden)
    estados = []
    for orden in sorted(d_dev_all["orden_escalon_fase"].unique()):
        sub = d_dev_all[d_dev_all["orden_escalon_fase"] == orden].reset_index(drop=True)
        for q in (0.25, 0.50, 0.75):
            target_av = sub["m6_avance_prev"].quantile(q)
            idx = (sub["m6_avance_prev"] - target_av).abs().idxmin()
            fila = sub.loc[idx].copy()
            estados.append({"orden": int(orden), "q_avance": q, "fila": fila})
    log(f"[etapa4] {len(estados)} estados representativos seleccionados")

    # rango P5-P95 de carbón por orden (histórico DEV)
    grilla_por_orden = {}
    for orden in sorted(d_dev_all["orden_escalon_fase"].unique()):
        sub = d_dev_all[d_dev_all["orden_escalon_fase"] == orden]
        p5, p95 = sub["tasa_feed_Carbon_kg_min"].quantile([.05, .95])
        grilla_por_orden[int(orden)] = np.linspace(p5, p95, 40)

    filas_curvas, filas_resumen = [], []
    for i, e in enumerate(estados):
        fila, orden, q_av = e["fila"], e["orden"], e["q_avance"]
        C_grid = grilla_por_orden[orden]
        dt = float(fila["duracion_plan_min"])
        sn_disp, sn_inv_prev, avance_prev = float(fila["m6_sn_disp"]), float(fila["m6_sn_inv_kg_prev"]), float(fila["m6_avance_prev"])
        t_prev, b2_prev = float(fila["temperatura_horno_celsius_prev"]), float(fila["basicidad_B2_prev"])
        gn_hist = float(fila["tasa_gn_nm3_min"])
        feats = features_carbon(C_grid, dt, sn_disp, sn_inv_prev, avance_prev, t_prev, b2_prev, gn_hist)

        pred_sn, pred_feo = {}, {}
        for nombre in sn_formas:
            A, _ = armar_ab(*FORMAS_SN[nombre], FIJAS_SN)
            X = pd.DataFrame({c: (feats[c] if c in feats else np.full(len(C_grid), fila[c])) for c in S + A})
            pred_sn[nombre] = modelos_sn[nombre].predict(X)
        for nombre in feo_formas:
            A, _ = armar_ab(*FORMAS_FEO[nombre], FIJAS_FEO)
            X = pd.DataFrame({c: (feats[c] if c in feats else np.full(len(C_grid), fila[c])) for c in S + A})
            pred_feo[nombre] = modelos_feo[nombre].predict(X)

        costo = COSTO_C * C_grid * dt
        C_hist = float(fila["tasa_feed_Carbon_kg_min"])
        for sn_n in sn_formas:
            for feo_n in feo_formas:
                J = K_UNIDAD * (pred_sn[sn_n] + W_FEO * pred_feo[feo_n]) - costo
                j_arg = int(np.argmax(J))
                interior = 0 < j_arg < len(C_grid) - 1
                for k in range(len(C_grid)):
                    filas_curvas.append({"combo": f"{sn_n}+{feo_n}", "estado_id": i, "orden": orden, "q_avance": q_av,
                                          "C": float(C_grid[k]), "J": float(J[k])})
                filas_resumen.append({"combo": f"{sn_n}+{feo_n}", "sn_forma": sn_n, "feo_forma": feo_n,
                                       "estado_id": i, "orden": orden, "q_avance": q_av,
                                       "C_hist": C_hist, "C_p5": float(C_grid[0]), "C_p95": float(C_grid[-1]),
                                       "C_opt": float(C_grid[j_arg]), "interior": interior,
                                       "J_hist_aprox": float(J[np.abs(C_grid - C_hist).argmin()]), "J_opt": float(J[j_arg])})
        if i % 3 == 0:
            log(f"[etapa4] estado {i+1}/{len(estados)} (orden {orden}, q_avance {q_av}) evaluado")

    df_curvas = pd.DataFrame(filas_curvas)
    df_resumen = pd.DataFrame(filas_resumen)
    df_curvas.to_csv(OUTDIR / "e10_01_curvas_J.csv", index=False)
    df_resumen.to_csv(OUTDIR / "e10_01_optimo_interior.csv", index=False)
    resumen_combo = df_resumen.groupby("combo").agg(
        pct_interior=("interior", "mean"), pct_interior_R0_R1=("interior", lambda s: s[df_resumen.loc[s.index, "orden"] <= 1].mean()),
        C_opt_mediana=("C_opt", "median"), C_hist_mediana=("C_hist", "median"),
    ).reset_index()
    log("Resumen por combo (%% interior, %% interior R0-R1, mediana C_opt vs C_hist):")
    log(resumen_combo.to_string(index=False))
    log("etapa4 terminada -> e10_01_optimo_interior.csv, e10_01_curvas_J.csv")


# --------------------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    etapa = sys.argv[1]
    log(f"cargando datos (etapa {etapa}) ...")
    df, r, dev_b, lb_b = cargar()
    log(f"df {df.shape}, r(Reducción) {r.shape}, dev_batches={len(dev_b)}, lockbox_batches={len(lb_b)}")
    if etapa in ("1", "all"):
        etapa1(r, dev_b, lb_b)
    if etapa in ("2", "all"):
        etapa2(r, dev_b, lb_b)
    if etapa in ("3", "all"):
        etapa3(r, dev_b, lb_b)
    if etapa in ("4", "all"):
        etapa4(r, dev_b, lb_b)
    log("listo")


if __name__ == "__main__":
    main()
