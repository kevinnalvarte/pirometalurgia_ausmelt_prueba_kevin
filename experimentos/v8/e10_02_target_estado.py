"""E10-02 (iteración 10, 2026-09-16): target de FeO con descuento del Fe metálico alimentado (dross) y estado
parsimonioso de Reducción (k16 v7 / k15 sin espesor / k17 +T horno +Al2O3).

Pregunta (`ITERACION_10_diseno.md` H2, criterios de decisión 1/2/4; auditoría A §1-2, auditoría B §1/§5):
(a) ¿`m8_ln_feo_ret_dross{40,50,60}` generaliza mejor (OOF cronológico / walk-forward) que `m6_ln_feo_ret` y mejora
la identificación de GN/carbón sobre FeO? (b) ¿`m8_ln_sn_dep_cstar` cambia algo frente a `m6_ln_sn_dep`? (c) ¿Retirar
`espesor_ladrillo_norm_mm` (k15) cuesta R²? ¿k17 conserva la identificación del carbón? (d) ¿Qué fracción del R² de
FeO es reversión del ruido del ensayo de t-1?

Uso (desde la raíz del proyecto):
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v8/e10_02_target_estado.py <etapa> [...]

    targets [estado] [familia] [targets_csv]
        estado      in {k16_v7, k15_sin_espesor, k17, all}   (default: all)
        familia     in {feo, sn, all}                        (default: all)
        targets_csv lista opcional separada por comas para acotar aún más una familia grande
                    (p.ej. "m6_ln_feo_ret,m8_ln_feo_ret_dross40") y mantener la corrida < 10 min
        -> e10_02_targets_estados.csv (R² OOF/cronológico/walk-forward/lockbox), e10_02_theta.csv (theta_dual libre/acotado),
           e10_02_por_orden.csv (R² OOF pooled por orden de escalón), e10_02_corr_targets.csv (Spearman/Pearson entre variantes
           de target FeO, por orden y global). Cada celda (familia x estado x target) cuesta ~60-90 s (con_wf=True); por eso
           el CLI permite acotar a un estado/familia por invocación y así mantener cada corrida < 10 min. Los CSV se
           actualizan por fusión (se reemplazan sólo las filas de la celda (estado, familia) recién calculada).

    reversion
        -> e10_02_reversion.csv: R² OOF aleatorio/cronológico con el estado k15 completo vs sin las 4 features que
           comparten el ensayo de FeO de t-1, para m6_ln_feo_ret y la mejor variante dross (leída de
           e10_02_targets_estados.csv si existe; si no, dross50 por defecto). Fracción de R² atribuible a reversión.

    tercios
        -> e10_02_tercios.csv: theta_dual (libre y acotado) por tercio cronológico de DEV de Cx_sn_v6 (sn_dep,
           m6_ln_sn_dep) y tasa_gn_nm3_min (feo_ret, m6_ln_feo_ret), para k15_sin_espesor y k17.

Todas las etapas usan sólo DEV para ajustar/decidir (SEED=42); el lockbox aparece únicamente en las columnas
*_lockbox de e10_02_targets_estados.csv (reporte, no selección). Anti-fuga: `L.asegurar_seguras` en cada estado+A.
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

ESTADOS = {k: L.ESTADOS_R[k] for k in ("k16_v7", "k15_sin_espesor", "k17")}
FEO_TARGETS = ["m6_ln_feo_ret", "m8_ln_feo_ret_dross40", "m8_ln_feo_ret_dross50", "m8_ln_feo_ret_dross60"]
SN_TARGETS = ["m6_ln_sn_dep", "m8_ln_sn_dep_cstar"]
A_FEO = list(L.PALANCAS_R_FEO_V7)
SIGNOS_FEO = dict(L.SIGNOS_R_FEO_V7)
A_SN = list(L.PALANCAS_R_SN_V7)
SIGNOS_SN = dict(L.SIGNOS_R_SN_V7)
FEATS_FEO_PREV = ["ley_feo_escoria_pct_prev", "m6_feo_inv_kg_prev", "ratio_sn_feo_prev", "interaccion_Sn_x_FeO_prev"]

N_BOOT_MAIN = 500
N_BOOT_TERCIOS = 500


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cargar():
    df = L.cargar_df()
    r = L.base_fase(df, "Reducción")
    dev_b, lb_b = L.split(df)
    return df, r, dev_b, lb_b


# --------------------------------------------------------------------------- utilidades
def r2_conjuntos_con_oof(d_dev: pd.DataFrame, d_lb: pd.DataFrame, S: list[str], A: list[str], target: str,
                          signos: dict, con_wf: bool = True, seed: int = SEED) -> tuple[dict, np.ndarray]:
    """Réplica de `L.r2_por_conjuntos` que además devuelve el OOF pooled (aleatorio) para el desglose por orden
    (evita recomputar el fit_predict una segunda vez sólo para obtener el array)."""
    fp = L.fit_predict_plm(S, A, target, signos, seed)
    oof = L.oof_groupkfold(d_dev, fp)
    ooc = L.oof_cronologico(d_dev, fp)
    out = {"n_dev": len(d_dev), "n_lockbox": len(d_lb),
           "r2_oof": L.metricas(d_dev[target], oof)["r2"], "mae_oof": L.metricas(d_dev[target], oof)["mae"],
           "r2_oof_cronologico": L.metricas(d_dev[target], ooc)["r2"]}
    if con_wf:
        pw, _ = L.walk_forward(d_dev, fp, target)
        m = np.isfinite(pw)
        out["r2_walk_forward"] = L.metricas(d_dev[target][m], pw[m])["r2"]
        out["n_walk_forward"] = int(m.sum())
    else:
        out["r2_walk_forward"] = np.nan
        out["n_walk_forward"] = 0
    if len(d_lb) > 10:
        p_lb = L.ModeloPLMSignos(S, A, target, signos).fit(d_dev, n_boot=0, seed=seed).predict(d_lb)
        out["r2_lockbox"] = L.metricas(d_lb[target], p_lb)["r2"]
        out["mae_lockbox"] = L.metricas(d_lb[target], p_lb)["mae"]
    else:
        out["r2_lockbox"] = np.nan
        out["mae_lockbox"] = np.nan
    return out, oof


def fusionar_csv(path: Path, filas_nuevas: pd.DataFrame, claves: list[str]) -> None:
    """Escribe `filas_nuevas` en `path`, reemplazando sólo las filas que coincidan en `claves` (para poder correr
    la grilla `targets` por partes -- estado x familia -- sin perder lo ya calculado)."""
    if path.exists():
        previo = pd.read_csv(path)
        mask_repetida = pd.Series(True, index=previo.index)
        for c in claves:
            mask_repetida &= previo[c].astype(str).isin(filas_nuevas[c].astype(str).unique())
        previo = previo[~mask_repetida]
        out = pd.concat([previo, filas_nuevas], ignore_index=True)
    else:
        out = filas_nuevas
    out.to_csv(path, index=False)


# --------------------------------------------------------------------------- etapa targets
def etapa_corr(r: pd.DataFrame) -> None:
    from scipy.stats import pearsonr, spearmanr
    d = r.dropna(subset=FEO_TARGETS + ["orden_escalon_fase"]).copy()
    filas = []
    grupos = [("global", d)] + [(f"orden_{int(o)}", d[d["orden_escalon_fase"] == o])
                                 for o in sorted(d["orden_escalon_fase"].unique())]
    for dross in FEO_TARGETS[1:]:
        for nombre_grupo, sub in grupos:
            if len(sub) < 10:
                continue
            rho, p_s = spearmanr(sub["m6_ln_feo_ret"], sub[dross])
            r_p, p_p = pearsonr(sub["m6_ln_feo_ret"], sub[dross])
            filas.append({"comparacion": f"m6_ln_feo_ret vs {dross}", "variante": dross, "grupo": nombre_grupo, "n": len(sub),
                          "spearman": rho, "p_spearman": p_s, "pearson": r_p, "p_pearson": p_p})
    # sensibilidad de ranking ENTRE variantes dross (fFe 0.4/0.5/0.6): ¿importa la elección de fFe dentro del rango físico?
    dross_cols = FEO_TARGETS[1:]
    for i in range(len(dross_cols)):
        for j in range(i + 1, len(dross_cols)):
            a, b = dross_cols[i], dross_cols[j]
            for nombre_grupo, sub in grupos:
                if len(sub) < 10:
                    continue
                rho, p_s = spearmanr(sub[a], sub[b])
                r_p, p_p = pearsonr(sub[a], sub[b])
                filas.append({"comparacion": f"{a} vs {b}", "variante": b, "grupo": nombre_grupo, "n": len(sub),
                              "spearman": rho, "p_spearman": p_s, "pearson": r_p, "p_pearson": p_p})
    pd.DataFrame(filas).to_csv(OUTDIR / "e10_02_corr_targets.csv", index=False)
    log("corr_targets -> e10_02_corr_targets.csv")


def procesar_familia(r: pd.DataFrame, dev_b: set, lb_b: set, targets_calc: list[str], targets_familia: list[str],
                      S: list[str], A: list[str], signos: dict, estado_nombre: str,
                      familia: str) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    """`targets_familia` fija la intersección de filas válidas (SIEMPRE los targets completos de la familia, aunque
    esta llamada sólo calcule métricas para `targets_calc`, un subconjunto) -- así las celdas son comparables entre
    sí sin importar en qué invocación del CLI se calcularon."""
    cols = list(dict.fromkeys(S + A + targets_familia))
    d = r.dropna(subset=cols).copy()
    d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
    d_lb = d[d["Batch"].isin(lb_b)].reset_index(drop=True)
    L.asegurar_seguras(S + A)
    filas_r2, filas_theta, filas_orden = [], [], []
    for target in targets_calc:
        t0 = time.time()
        met, oof = r2_conjuntos_con_oof(d_dev, d_lb, S, A, target, signos, con_wf=True)
        met.update({"estado": estado_nombre, "familia": familia, "target": target, "k_estado": len(S)})
        filas_r2.append(met)
        for orden in sorted(d_dev["orden_escalon_fase"].dropna().unique()):
            msk = (d_dev["orden_escalon_fase"] == orden).to_numpy()
            if msk.sum() < 10:
                continue
            mo = L.metricas(d_dev.loc[msk, target], oof[msk])
            filas_orden.append({"estado": estado_nombre, "familia": familia, "target": target,
                                "orden": int(orden), "n": int(msk.sum()), "r2_oof": mo["r2"], "mae_oof": mo["mae"]})
        tabla, _extras = L.theta_dual(d_dev, S, A, target, signos, n_boot=N_BOOT_MAIN, seed=SEED)
        tabla["estado"] = estado_nombre
        tabla["familia"] = familia
        tabla["target"] = target
        filas_theta.append(tabla)
        log(f"[{familia} {estado_nombre} {target}] n_dev={met['n_dev']} r2_oof={met['r2_oof']:.3f} "
            f"cron={met['r2_oof_cronologico']:.3f} wf={met['r2_walk_forward']:.3f} lb={met['r2_lockbox']:.3f} "
            f"({time.time()-t0:.1f}s)")
    return filas_r2, filas_theta, filas_orden


def etapa_targets(r: pd.DataFrame, dev_b: set, lb_b: set, estado_sel: str = "all", familia_sel: str = "all",
                   targets_sel: str | None = None) -> None:
    etapa_corr(r)
    estados = ESTADOS if estado_sel == "all" else {estado_sel: ESTADOS[estado_sel]}
    familias = ["feo", "sn"] if familia_sel == "all" else [familia_sel]
    lista_targets = targets_sel.split(",") if targets_sel else None
    filas_r2_all, filas_theta_all, filas_orden_all = [], [], []
    for estado_nombre, S in estados.items():
        if "feo" in familias:
            tg = [t for t in FEO_TARGETS if lista_targets is None or t in lista_targets]
            fr2, fth, forden = procesar_familia(r, dev_b, lb_b, tg, FEO_TARGETS, S, A_FEO, SIGNOS_FEO, estado_nombre, "feo_ret")
            filas_r2_all += fr2; filas_theta_all += fth; filas_orden_all += forden
        if "sn" in familias:
            tg = [t for t in SN_TARGETS if lista_targets is None or t in lista_targets]
            fr2, fth, forden = procesar_familia(r, dev_b, lb_b, tg, SN_TARGETS, S, A_SN, SIGNOS_SN, estado_nombre, "sn_dep")
            filas_r2_all += fr2; filas_theta_all += fth; filas_orden_all += forden
    fusionar_csv(OUTDIR / "e10_02_targets_estados.csv", pd.DataFrame(filas_r2_all), ["estado", "familia", "target"])
    fusionar_csv(OUTDIR / "e10_02_theta.csv", pd.concat(filas_theta_all, ignore_index=True), ["estado", "familia", "target"])
    fusionar_csv(OUTDIR / "e10_02_por_orden.csv", pd.DataFrame(filas_orden_all), ["estado", "familia", "target"])
    log("etapa targets terminada -> e10_02_targets_estados.csv, e10_02_theta.csv, e10_02_por_orden.csv, e10_02_corr_targets.csv")


# --------------------------------------------------------------------------- etapa reversion
def elegir_mejor_dross() -> str:
    f = OUTDIR / "e10_02_targets_estados.csv"
    if not f.exists():
        return "m8_ln_feo_ret_dross50"
    df = pd.read_csv(f)
    sub = df[(df["familia"] == "feo_ret") & (df["target"] != "m6_ln_feo_ret")]
    if sub.empty:
        return "m8_ln_feo_ret_dross50"
    score = sub.groupby("target")[["r2_oof_cronologico", "r2_walk_forward"]].mean().mean(axis=1).sort_values(ascending=False)
    return str(score.index[0])


def etapa_reversion(r: pd.DataFrame, dev_b: set, lb_b: set) -> None:
    S = ESTADOS["k15_sin_espesor"]
    S_sin = [c for c in S if c not in FEATS_FEO_PREV]
    mejor_dross = elegir_mejor_dross()
    log(f"mejor variante dross (por r2_oof_cronologico/walk_forward promedio) = {mejor_dross}")
    filas = []
    for target in ["m6_ln_feo_ret", mejor_dross]:
        resultados_estado = {}
        for nombre_estado, S_use in [("k15_full", S), ("k15_sin_feo_prev", S_sin)]:
            L.asegurar_seguras(S_use + A_FEO)
            cols = list(dict.fromkeys(S_use + A_FEO + [target]))
            d = r.dropna(subset=cols).copy()
            d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
            d_lb = d[d["Batch"].isin(lb_b)].reset_index(drop=True)
            t0 = time.time()
            met, _oof = r2_conjuntos_con_oof(d_dev, d_lb, S_use, A_FEO, target, SIGNOS_FEO, con_wf=False)
            met.update({"target": target, "estado": nombre_estado, "k": len(S_use)})
            resultados_estado[nombre_estado] = met
            log(f"[reversion {target} {nombre_estado}] n_dev={met['n_dev']} r2_oof={met['r2_oof']:.3f} "
                f"cron={met['r2_oof_cronologico']:.3f} ({time.time()-t0:.1f}s)")
        full, sin = resultados_estado["k15_full"], resultados_estado["k15_sin_feo_prev"]
        if full["r2_oof"] not in (0, np.nan) and np.isfinite(full["r2_oof"]) and full["r2_oof"] != 0:
            full["frac_reversion_oof"] = (full["r2_oof"] - sin["r2_oof"]) / full["r2_oof"]
        else:
            full["frac_reversion_oof"] = np.nan
        if np.isfinite(full["r2_oof_cronologico"]) and full["r2_oof_cronologico"] != 0:
            full["frac_reversion_cron"] = (full["r2_oof_cronologico"] - sin["r2_oof_cronologico"]) / full["r2_oof_cronologico"]
        else:
            full["frac_reversion_cron"] = np.nan
        sin["frac_reversion_oof"] = np.nan
        sin["frac_reversion_cron"] = np.nan
        filas.append(full); filas.append(sin)
        log(f"[reversion {target}] frac_reversion_oof={full['frac_reversion_oof']:.3f} "
            f"frac_reversion_cron={full['frac_reversion_cron']:.3f}")
    pd.DataFrame(filas).to_csv(OUTDIR / "e10_02_reversion.csv", index=False)
    log("etapa reversion terminada -> e10_02_reversion.csv")


# --------------------------------------------------------------------------- etapa tercios
def etapa_tercios(r: pd.DataFrame, dev_b: set, lb_b: set) -> None:
    specs = [
        ("sn_dep", "m6_ln_sn_dep", A_SN, SIGNOS_SN, "Cx_sn_v6"),
        ("feo_ret", "m6_ln_feo_ret", A_FEO, SIGNOS_FEO, "tasa_gn_nm3_min"),
    ]
    estados_tercios = {"k15_sin_espesor": ESTADOS["k15_sin_espesor"], "k17": ESTADOS["k17"]}
    filas = []
    for nombre_estado, S in estados_tercios.items():
        for label, target, A, signos, palanca in specs:
            L.asegurar_seguras(S + A)
            cols = list(dict.fromkeys(S + A + [target]))
            d = r.dropna(subset=cols).copy()
            d_dev = d[d["Batch"].isin(dev_b)].reset_index(drop=True)
            terc = L.tercios_cronologicos(d_dev, n=3)
            for k in sorted(terc.dropna().unique()):
                sub = d_dev[terc.to_numpy() == k].reset_index(drop=True)
                t0 = time.time()
                tabla, _extras = L.theta_dual(sub, S, A, target, signos, n_boot=N_BOOT_TERCIOS, seed=SEED)
                row = tabla[tabla["palanca"] == palanca].iloc[0].to_dict()
                row.update({"estado": nombre_estado, "target": label, "tercio": int(k), "n": len(sub)})
                filas.append(row)
                log(f"[tercios {nombre_estado} {label} t{int(k)}] n={len(sub)} "
                    f"theta_libre_1sd={row['theta_libre_1sd']:.4f} "
                    f"[{row['ci_lo_libre_1sd']:.4f},{row['ci_hi_libre_1sd']:.4f}] "
                    f"theta_acot_1sd={row['theta_acot_1sd']:.4f} ({time.time()-t0:.1f}s)")
                pd.DataFrame(filas).to_csv(OUTDIR / "e10_02_tercios.csv", index=False)
    log("etapa tercios terminada -> e10_02_tercios.csv")


# --------------------------------------------------------------------------- main
def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    etapa = sys.argv[1]
    log(f"cargando datos (etapa {etapa}) ...")
    df, r, dev_b, lb_b = cargar()
    log(f"df {df.shape}, r(Reducción) {r.shape}, dev_batches={len(dev_b)}, lockbox_batches={len(lb_b)}")
    if etapa == "targets":
        estado_sel = sys.argv[2] if len(sys.argv) > 2 else "all"
        familia_sel = sys.argv[3] if len(sys.argv) > 3 else "all"
        targets_sel = sys.argv[4] if len(sys.argv) > 4 else None
        etapa_targets(r, dev_b, lb_b, estado_sel, familia_sel, targets_sel)
    elif etapa == "corr":
        etapa_corr(r)
    elif etapa == "reversion":
        etapa_reversion(r, dev_b, lb_b)
    elif etapa == "tercios":
        etapa_tercios(r, dev_b, lb_b)
    else:
        print(__doc__)
        sys.exit(1)
    log("listo")


if __name__ == "__main__":
    main()
