"""E8-01 -- variantes, calibracion y robustez del estimador de masa de escoria v6.

Ver `experimentos/v6/ITERACION_8_diseno.md` y el docstring de `masa_v6.py`. Este script NO modifica
`masa_v6.py`; si se detecta un bug se documenta en `e8_01_resultados.md` con el parche propuesto.

Etapas (--stage): grid, calib, sensib, ruido, bordes, anclaje, qc, all (default).
Salidas en `experimentos/v6/`: e8_01_grilla.csv, e8_01_calibracion.csv, e8_01_sensibilidad.csv,
e8_01_ruido.csv, e8_01_anclaje.csv, e8_01_qc.csv (bordes solo imprime a stdout/log, sin CSV propio
segun el encargo).

Uso (background, log a e8_01_log.txt):
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/v6/e8_01_masa_variantes.py --stage all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import masa_v6 as m6  # noqa: E402

OUT = HERE
pd.set_option("display.width", 160)


# =============================================================================
# Carga comun
# =============================================================================

def cargar_todo():
    df = m6.cargar_df_base()
    dev, lb = mp.split_dev_lockbox(df)
    bal = m6.balance_batch(df)
    return df, dev, lb, bal


def calib_sets(df: pd.DataFrame, dev: set, bal: pd.DataFrame) -> dict:
    """Coeficientes efectivos con y sin carbon, trazadores G y CaO (solo DEV)."""
    sets = {}
    for con_c in (False, True):
        cg = m6.calibrar_entradas(df, dev, bal, "G", con_carbon=con_c)
        cc = m6.calibrar_entradas(df, dev, bal, "CaO", con_carbon=con_c)
        sets[con_c] = dict(pG=cg["cal"], gG=cg["carga"], aC=cg.get("carbon", 0.0),
                            pCaO=cc["cal"], cCaO=cc["carga"], aC_cao=cc.get("carbon", 0.0),
                            cal_g=cg, cal_cao=cc)
    return sets


def build_params(fusion: str, reduccion: str, delta_R: float, w_obs_R: float, con_carbon: bool, coefs: dict) -> dict:
    c = coefs[con_carbon]
    p = dict(m6.PARAMS_DEFAULT)
    p.update(fusion=fusion, reduccion=reduccion, delta_R=delta_R, w_obs_R=w_obs_R,
              pG=c["pG"], gG=c["gG"], aC=(c["aC"] if con_carbon else 0.0),
              pCaO=c["pCaO"], cCaO=c["cCaO"])
    return p


# =============================================================================
# 1) Grilla fusion x reduccion x (delta_R, w_obs_R) x aC
# =============================================================================

FUSION_OPTS = ["G", "CaO", "cierre"]
REDUCCION_OPTS = ["cierre", "G", "CaO", "cierre+obs"]
DELTAS = [0.0, 0.01, 0.02]
WOBS = [0.15, 0.30]


def variantes() -> list[dict]:
    vs = []
    for fusion in FUSION_OPTS:
        for reduccion in REDUCCION_OPTS:
            deltas = DELTAS if reduccion in ("cierre", "cierre+obs") else [0.0]
            wobs = WOBS if reduccion == "cierre+obs" else [0.0]
            for con_c in (False, True):
                for d in deltas:
                    for w in wobs:
                        vs.append(dict(fusion=fusion, reduccion=reduccion, delta_R=d, w_obs_R=w, con_carbon=con_c))
    return vs


def signature(v: dict) -> tuple:
    usa_aC = v["fusion"] in ("G", "cierre") or v["reduccion"] in ("G", "cierre", "cierre+obs")
    key_c = v["con_carbon"] if usa_aC else False
    d = round(v["delta_R"], 4) if v["reduccion"] in ("cierre", "cierre+obs") else 0.0
    w = round(v["w_obs_R"], 4) if v["reduccion"] == "cierre+obs" else 0.0
    return (v["fusion"], v["reduccion"], d, w, key_c)


CACHE_SIG_CSV = OUT / "e8_01_cache_sig.csv"


def _cargar_cache_sig() -> dict:
    """Cache en disco por firma de variante (fusion,reduccion,delta,w_obs,aC-flag): permite reanudar el
    script si el proceso muere a mitad de la grilla sin recalcular las firmas ya resueltas."""
    if not CACHE_SIG_CSV.exists():
        return {}
    d = pd.read_csv(CACHE_SIG_CSV)
    return {row["signature"]: row.drop(labels=["signature"]).to_dict() for _, row in d.iterrows()}


def _guardar_cache_sig_row(sig_str: str, row: dict) -> None:
    fila = {"signature": sig_str, **row}
    header = not CACHE_SIG_CSV.exists()
    pd.DataFrame([fila]).to_csv(CACHE_SIG_CSV, mode="a", header=header, index=False)


def etapa_grid(df, dev, lb, bal, coefs) -> pd.DataFrame:
    print("== grid ==", flush=True)
    cache = _cargar_cache_sig()
    if cache:
        print(f"  cache en disco: {len(cache)} firmas ya resueltas ({CACHE_SIG_CSV.name})", flush=True)
    vs = variantes()
    t0 = time.time()
    for i, v in enumerate(vs):
        sig = signature(v)
        sig_str = str(sig)
        if sig_str not in cache:
            p = build_params(v["fusion"], v["reduccion"], v["delta_R"], v["w_obs_R"], v["con_carbon"], coefs)
            df6 = m6.agregar_masa_v6(df, p, bal)
            res = m6.evaluar_masa(df6, "m6_masa_kg", bal, dev, lb)
            row = {k: v2 for k, v2 in res.items() if k != "col"}
            row.update(aC=p["aC"], pG=p["pG"], gG=p["gG"], pCaO=p["pCaO"], cCaO=p["cCaO"])
            cache[sig_str] = row
            _guardar_cache_sig_row(sig_str, row)
            print(f"  [{i+1}/{len(vs)}] nuevo {sig} -> R2_feo={res['R2_feo']:.3f} sd_dlnM_R={res['sd_dlnM_R']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"  [{i+1}/{len(vs)}] cache hit {sig}", flush=True)

    filas = []
    for v in vs:
        sig_str = str(signature(v))
        fila = dict(fusion=v["fusion"], reduccion=v["reduccion"], delta_R=v["delta_R"], w_obs_R=v["w_obs_R"],
                    con_carbon=v["con_carbon"], signature=sig_str)
        fila.update(cache[sig_str])
        filas.append(fila)
    # referencia v3
    ref = m6.evaluar_masa(df.assign(masa_v3=df["masa_escoria_est_kg"]), "masa_v3", bal, dev, lb)
    fila = dict(fusion="v3_ref", reduccion="v3_ref", delta_R=np.nan, w_obs_R=np.nan, con_carbon=np.nan,
                aC=np.nan, pG=np.nan, gG=np.nan, pCaO=np.nan, cCaO=np.nan, signature="v3_ref")
    fila.update({k: v2 for k, v2 in ref.items() if k != "col"})
    filas.append(fila)
    out = pd.DataFrame(filas)
    out.to_csv(OUT / "e8_01_grilla.csv", index=False)
    print(f"grid: {len(vs)} variantes, {len(cache)} firmas unicas -> e8_01_grilla.csv ({time.time()-t0:.0f}s)", flush=True)
    return out


# =============================================================================
# 2) Calibracion: estabilidad temporal (tercios) y por fold (GroupKFold por batch)
# =============================================================================

def etapa_calib(df, dev, bal) -> pd.DataFrame:
    print("== calib ==", flush=True)
    filas = []
    for trazador in ("G", "CaO"):
        for con_c in (False, True):
            r = m6.calibrar_entradas(df, dev, bal, trazador, con_carbon=con_c)
            filas.append(dict(split="full_dev", trazador=trazador, con_carbon=con_c, **{k: v for k, v in r.items() if k != "trazador"}))

    # tercios cronologicos de DEV
    orden = mp.orden_cronologico_batches(df)
    dev_orden = [b for b in orden if b in dev]
    n = len(dev_orden)
    cortes = [dev_orden[:n // 3], dev_orden[n // 3:2 * n // 3], dev_orden[2 * n // 3:]]
    for i, tercio in enumerate(cortes, start=1):
        tset = set(tercio)
        for trazador in ("G", "CaO"):
            try:
                r = m6.calibrar_entradas(df, tset, bal, trazador, con_carbon=False)
                filas.append(dict(split=f"tercio{i}", trazador=trazador, con_carbon=False, **{k: v for k, v in r.items() if k != "trazador"}))
            except Exception as e:
                print(f"  tercio{i} {trazador} fallo: {e}", flush=True)

    # GroupKFold(5) por batch (aqui "grupo" = batch, ya agregado a nivel batch en calibrar_entradas)
    from sklearn.model_selection import KFold
    dev_list = np.array(sorted(dev))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for trazador in ("G", "CaO"):
        for con_c in (False, True):
            for k, (tr_idx, _) in enumerate(kf.split(dev_list)):
                train_b = set(dev_list[tr_idx])
                try:
                    r = m6.calibrar_entradas(df, train_b, bal, trazador, con_carbon=con_c)
                    filas.append(dict(split=f"kfold{k}", trazador=trazador, con_carbon=con_c, **{kk: v for kk, v in r.items() if kk != "trazador"}))
                except Exception as e:
                    print(f"  kfold{k} {trazador} con_c={con_c} fallo: {e}", flush=True)

    out = pd.DataFrame(filas)
    out.to_csv(OUT / "e8_01_calibracion.csv", index=False)
    # resumen de estabilidad por fold (sd de cal/carga a traves de los 5 folds)
    for trazador in ("G", "CaO"):
        for con_c in (False, True):
            sub = out[(out["split"].str.startswith("kfold")) & (out["trazador"] == trazador) & (out["con_carbon"] == con_c)]
            if len(sub):
                print(f"  kfold {trazador} con_carbon={con_c}: cal mean={sub['cal'].mean():.4f} sd={sub['cal'].std():.4f} | "
                      f"carga mean={sub['carga'].mean():.4f} sd={sub['carga'].std():.4f}", flush=True)
    print("calib -> e8_01_calibracion.csv", flush=True)
    return out


# =============================================================================
# 3) Sensibilidad +-30% pG, gG (targets log deberian ser casi invariantes)
# =============================================================================

def etapa_sensib(df, dev, lb, bal, coefs) -> pd.DataFrame:
    print("== sensib ==", flush=True)
    base = coefs[False]
    variantes_s = [("base", 1.0, 1.0), ("pG*0.7", 0.7, 1.0), ("pG*1.3", 1.3, 1.0),
                   ("gG*0.7", 1.0, 0.7), ("gG*1.3", 1.0, 1.3), ("ambos*0.7", 0.7, 0.7), ("ambos*1.3", 1.3, 1.3)]
    filas = []
    for nombre, fp, fg in variantes_s:
        p = dict(m6.PARAMS_DEFAULT)
        p.update(fusion="G", reduccion="cierre", pG=base["pG"] * fp, gG=base["gG"] * fg,
                 pCaO=base["pCaO"] * fp, cCaO=base["cCaO"] * fg, aC=0.0)
        df6 = m6.agregar_masa_v6(df, p, bal)
        res = m6.evaluar_masa(df6, "m6_masa_kg", bal, dev, lb)
        fila = dict(variante=nombre, factor_pG=fp, factor_gG=fg, pG=p["pG"], gG=p["gG"])
        fila.update({k: v for k, v in res.items() if k != "col"})
        filas.append(fila)
        print(f"  {nombre}: R2_feo={res['R2_feo']:.3f} R2_sn={res['R2_sn']:.3f} ratio_bal_dev={res['ratio_bal_dev']:.3f} "
              f"sd_dlnM_R={res['sd_dlnM_R']:.4f}", flush=True)
    out = pd.DataFrame(filas)
    out.to_csv(OUT / "e8_01_sensibilidad.csv", index=False)
    print("sensib -> e8_01_sensibilidad.csv", flush=True)
    return out


# =============================================================================
# 4) Propagacion de error en Reduccion (bootstrap de ruido de ensayo)
# =============================================================================

def etapa_ruido(df, coefs, n_rep: int = 200, seed: int = 0) -> pd.DataFrame:
    print("== ruido ==", flush=True)
    t0 = time.time()
    p_cierre = build_params("G", "cierre", 0.0, 0.0, False, coefs)
    p_cao = build_params("G", "CaO", 0.0, 0.0, False, coefs)
    base_cierre = m6.estimar_masa(df, p_cierre)
    base_cao = m6.estimar_masa(df, p_cao)

    B = df["Batch"]
    R = df["fase_proceso"].eq("Reducción")
    orden = df["orden_escalon_fase"]
    nofeed = R & df["feed_total_kgh"].fillna(0).le(1) & df["feed_CaO_kgh"].fillna(0).le(1)

    base_lnM_cierre = np.log(base_cierre["m6_masa_kg"].replace([np.inf, -np.inf], np.nan))
    base_feo_inv = base_cierre["m6_masa_kg"] * (df["ley_feo_escoria_pct"] / 100)
    base_ln_feo = np.log(base_feo_inv / base_feo_inv.groupby(B).shift(1)).where(R)
    base_lnM_cao = np.log(base_cao["m6_masa_kg"].replace([np.inf, -np.inf], np.nan))

    rng = np.random.default_rng(seed)
    acc_dlnM_cierre, acc_dlnfeo, acc_dlnM_cao = [], [], []
    np.seterr(divide="ignore", invalid="ignore")
    for i in range(n_rep):
        eps_sn = rng.normal(0, 0.02, len(df))
        eps_feo = rng.normal(0, 0.028, len(df))
        eps_cao = rng.normal(0, 0.028, len(df))
        dfx = df.copy()
        dfx["ley_sn_escoria_pct"] = df["ley_sn_escoria_pct"] * (1 + eps_sn)
        dfx["ley_feo_escoria_pct"] = df["ley_feo_escoria_pct"] * (1 + eps_feo)
        est = m6.estimar_masa(dfx, p_cierre)
        lnM = np.log(est["m6_masa_kg"].replace([np.inf, -np.inf], np.nan))
        d_lnM = lnM - base_lnM_cierre
        feo_inv = est["m6_masa_kg"] * (dfx["ley_feo_escoria_pct"] / 100)
        ln_feo = np.log(feo_inv / feo_inv.groupby(B).shift(1)).where(R)
        d_ln_feo = ln_feo - base_ln_feo

        dfx2 = df.copy()
        dfx2["ley_cao_escoria_pct"] = df["ley_cao_escoria_pct"] * (1 + eps_cao)
        est2 = m6.estimar_masa(dfx2, p_cao)
        lnM_cao = np.log(est2["m6_masa_kg"].replace([np.inf, -np.inf], np.nan))
        d_lnM_cao = lnM_cao - base_lnM_cao

        acc_dlnM_cierre.append(pd.DataFrame({"orden": orden, "nofeed": nofeed, "R": R, "d": d_lnM}))
        acc_dlnfeo.append(pd.DataFrame({"orden": orden, "R": R, "d": d_ln_feo}))
        acc_dlnM_cao.append(pd.DataFrame({"orden": orden, "R": R, "d": d_lnM_cao}))
        if (i + 1) % 50 == 0:
            print(f"  replica {i+1}/{n_rep} ({time.time()-t0:.0f}s)", flush=True)

    dd_cierre = pd.concat(acc_dlnM_cierre, ignore_index=True)
    dd_feo = pd.concat(acc_dlnfeo, ignore_index=True)
    dd_cao = pd.concat(acc_dlnM_cao, ignore_index=True)

    filas = []
    for o in sorted(df.loc[R, "orden_escalon_fase"].dropna().unique()):
        sub_c = dd_cierre[(dd_cierre["orden"] == o) & dd_cierre["nofeed"]]["d"].dropna()
        sub_c_all = dd_cierre[(dd_cierre["orden"] == o) & dd_cierre["R"]]["d"].dropna()
        sub_f = dd_feo[dd_feo["orden"] == o]["d"].dropna()
        sub_cao = dd_cao[(dd_cao["orden"] == o) & dd_cao["R"]]["d"].dropna()
        senal_lnM = base_lnM_cierre[nofeed & (orden == o)].groupby(B[nofeed & (orden == o)]).apply(lambda x: x).pipe(
            lambda s: np.log(base_cierre["m6_masa_kg"][nofeed & (orden == o)] / base_cierre["m6_masa_kg"].groupby(B).shift(1)[nofeed & (orden == o)]))
        senal_lnM = senal_lnM.dropna()
        senal_feo = base_ln_feo[R & (orden == o)].dropna()
        ruido_lnM = float(sub_c.std()) if len(sub_c) > 5 else np.nan
        ruido_lnM_all = float(sub_c_all.std()) if len(sub_c_all) > 5 else np.nan
        ruido_feo = float(sub_f.std()) if len(sub_f) > 5 else np.nan
        ruido_cao = float(sub_cao.std()) if len(sub_cao) > 5 else np.nan
        s_lnM = float(senal_lnM.std()) if len(senal_lnM) > 5 else np.nan
        s_feo = float(senal_feo.std()) if len(senal_feo) > 5 else np.nan
        filas.append(dict(orden_R=int(o), n_nofeed=int(len(sub_c) / n_rep) if n_rep else 0,
                           senal_sd_dlnM=s_lnM, ruido_sd_dlnM_cierre=ruido_lnM, snr_dlnM_cierre=s_lnM / ruido_lnM if ruido_lnM else np.nan,
                           ruido_sd_dlnM_cierre_allR=ruido_lnM_all,
                           senal_sd_ln_feo=s_feo, ruido_sd_ln_feo=ruido_feo, snr_ln_feo=s_feo / ruido_feo if ruido_feo else np.nan,
                           ruido_sd_dlnM_CaO=ruido_cao, snr_dlnM_CaO=s_lnM / ruido_cao if ruido_cao else np.nan))
    out = pd.DataFrame(filas)
    out.to_csv(OUT / "e8_01_ruido.csv", index=False)
    print(out.to_string(index=False), flush=True)
    print(f"ruido -> e8_01_ruido.csv ({time.time()-t0:.0f}s)", flush=True)
    return out


# =============================================================================
# 5) Casos borde (sin CSV propio; se documentan en el memo)
# =============================================================================

def etapa_bordes(df, dev, bal, coefs):
    print("== bordes ==", flush=True)
    p = build_params("G", "cierre", 0.0, 0.0, False, coefs)
    df6 = m6.agregar_masa_v6(df, p, bal)
    B = df6["Batch"]
    Fpref = df6["fase_proceso"].eq("Fusión")
    F0 = Fpref & df6["orden_escalon_fase"].eq(0)
    R = df6["fase_proceso"].eq("Reducción")
    g = df6.groupby("Batch")
    entrada_acum_F0 = df6["feed_CaO_kgh"].fillna(0) + df6["feed_total_kgh"].fillna(0)
    ratio_f0_v6 = (df6.loc[F0, "m6_masa_kg"] / entrada_acum_F0[F0]).replace([np.inf, -np.inf], np.nan)
    ratio_f0_v3 = (df6.loc[F0, "masa_escoria_est_kg"] / entrada_acum_F0[F0]).replace([np.inf, -np.inf], np.nan)
    print("F0 m6_masa_kg/(cal+carga) descr:\n", ratio_f0_v6.describe(), flush=True)
    print("F0 masa_v3/(cal+carga) descr:\n", ratio_f0_v3.describe(), flush=True)

    for col, nombre in [("ley_sn_escoria_pct", "Sn"), ("ley_feo_escoria_pct", "FeO"), ("ley_cao_escoria_pct", "CaO")]:
        falt = df6[col].isna()
        print(f"ensayos faltantes {nombre}: {int(falt.sum())} ({100*falt.mean():.2f}%) -- por fase: "
              f"F={int((falt & Fpref).sum())} R={int((falt & R).sum())}", flush=True)
    falt_sf = df6["ley_sn_escoria_pct"].isna() | df6["ley_feo_escoria_pct"].isna()
    masa_nan_tras_falt = df6["m6_masa_kg"].isna() & falt_sf
    print(f"filas con Sn/FeO faltante Y m6_masa_kg NaN: {int(masa_nan_tras_falt.sum())} de {int(falt_sf.sum())} faltantes", flush=True)
    sig_falt = falt_sf.groupby(B).cumsum() > 0
    idx_falt = df6.index[falt_sf]
    reanclados = 0
    for idx in idx_falt:
        pos = df6.index.get_loc(idx)
        if pos + 1 < len(df6) and df6["Batch"].iloc[pos + 1] == df6["Batch"].iloc[pos]:
            reanclados += 1
    print(f"escalones inmediatamente siguientes a un faltante (mismo batch): {reanclados}", flush=True)

    cal_batch = g["feed_CaO_kgh"].sum()
    batches_sin_cal = cal_batch[cal_batch < 100].index
    print(f"batches con cal ~0: {len(batches_sin_cal)} de {cal_batch.shape[0]}", flush=True)
    sub = df6[df6["Batch"].isin(batches_sin_cal)]
    print("  ratio_bal (mfin/mbal) para esos batches -- ver e8_01 grid col ratio_bal; masa mediana escalon:",
          float(sub["m6_masa_kg"].median()), "vs global:", float(df6["m6_masa_kg"].median()), flush=True)

    resto_bajo = df6["m6_resto_frac"] < 0.4
    print(f"escalones con resto_frac<0.4: {int(resto_bajo.sum())} ({100*resto_bajo.mean():.2f}%) -- "
          f"por fase F={int((resto_bajo & Fpref).sum())} R={int((resto_bajo & R).sum())}", flush=True)

    cero_nan = df6["m6_masa_kg"].isna() | df6["m6_masa_kg"].eq(0)
    print(f"m6_masa_kg NaN/0: {int(cero_nan.sum())} -- por fase F={int((cero_nan & Fpref).sum())} R={int((cero_nan & R).sum())} "
          f"otras={int((cero_nan & ~Fpref & ~R).sum())}", flush=True)
    print("bordes: sin CSV, ver log", flush=True)


# =============================================================================
# 6) Diagnostico de m6_k_anclaje
# =============================================================================

def etapa_anclaje(df, bal) -> pd.DataFrame:
    print("== anclaje ==", flush=True)
    df6 = m6.agregar_masa_v6(df, bal=bal)  # PARAMS_DEFAULT
    R = df6["fase_proceso"].eq("Reducción")
    g = df6.groupby("Batch")
    k = g["m6_k_anclaje"].first()
    cal = g["feed_CaO_kgh"].sum()
    carga = g["feed_total_kgh"].sum()
    cc = cal / carga
    frac_sec = g.apply(lambda x: np.average(x["frac_carga_secundaria"].fillna(0), weights=x["feed_total_kgh"].fillna(0).clip(lower=1e-6)))
    ley_sn_carga = g["ley_sn_carga_batch_pct"].first()
    dross_fe = g["feed_dross_Fe_kgh"].sum()
    t_media = g["temperatura_horno_celsius"].mean()
    fecha = g["fecha_inicio"].min()
    espesor = g["espesor_ladrillo_norm_mm"].mean()
    orden_crono = fecha.rank(method="first")
    out = pd.DataFrame({"k_anclaje": k, "cal_kg": cal, "carga_kg": carga, "cal_sobre_carga": cc,
                         "frac_carga_secundaria": frac_sec, "ley_sn_carga_batch_pct": ley_sn_carga,
                         "feed_dross_Fe_kg": dross_fe, "T_media_C": t_media, "fecha_inicio": fecha,
                         "espesor_ladrillo_mm": espesor, "orden_cronologico": orden_crono}).dropna(subset=["k_anclaje"])
    out.to_csv(OUT / "e8_01_anclaje.csv")

    from scipy.stats import spearmanr
    print(f"m6_k_anclaje: n={len(out)} media={out['k_anclaje'].mean():.3f} sd={out['k_anclaje'].std():.3f} "
          f"mediana={out['k_anclaje'].median():.3f} IQR=[{out['k_anclaje'].quantile(.25):.3f},{out['k_anclaje'].quantile(.75):.3f}]", flush=True)
    for c in ["cal_sobre_carga", "frac_carga_secundaria", "ley_sn_carga_batch_pct", "feed_dross_Fe_kg",
              "T_media_C", "espesor_ladrillo_mm", "orden_cronologico"]:
        rho, p = spearmanr(out["k_anclaje"], out[c], nan_policy="omit")
        print(f"  spearman(k_anclaje, {c}) = {rho:.3f} (p={p:.3g}, n={out[c].notna().sum()})", flush=True)
    print("anclaje -> e8_01_anclaje.csv", flush=True)
    return out


# =============================================================================
# 7) Bandera QC de residuo de masa en Reduccion
# =============================================================================

def etapa_qc(df, bal, coefs) -> pd.DataFrame:
    print("== qc ==", flush=True)
    p_cierre = build_params("G", "cierre", 0.0, 0.0, False, coefs)
    p_g = build_params("G", "G", 0.0, 0.0, False, coefs)
    m_cierre = m6.estimar_masa(df, p_cierre)["m6_masa_kg"]
    m_g = m6.estimar_masa(df, p_g)["m6_masa_kg"]
    residuo = np.log(m_cierre.replace(0, np.nan)) - np.log(m_g.replace(0, np.nan))
    R = df["fase_proceso"].eq("Reducción")
    out = pd.DataFrame({"Batch": df["Batch"], "orden_escalon_fase": df["orden_escalon_fase"],
                         "residuo_ln_masa": residuo, "R": R})[R].dropna(subset=["residuo_ln_masa"])
    sd = out["residuo_ln_masa"].std()
    # z-score DEMEANADO por orden de escalon: el residuo tiene un sesgo sistematico (M_cierre < M_G en
    # promedio, ~metodos distintos) que no es una anomalia; la bandera busca outliers dentro de ese nivel.
    mu_o = out.groupby("orden_escalon_fase")["residuo_ln_masa"].transform("mean")
    sd_o = out.groupby("orden_escalon_fase")["residuo_ln_masa"].transform("std")
    out["z_residuo"] = (out["residuo_ln_masa"] - mu_o) / sd_o
    out["flag_qc"] = out["z_residuo"].abs() > 2

    polvo = df.drop_duplicates("Batch").set_index("Batch")["sn_en_polvo_fundicion_batch_t"]
    kpi = bal["recuperacion_refinada_pct"]
    batch_res = out.groupby("Batch").agg(residuo_medio=("residuo_ln_masa", "mean"),
                                          residuo_abs_medio=("residuo_ln_masa", lambda x: x.abs().mean()),
                                          frac_flag=("flag_qc", "mean"))
    batch_res["polvo_sn_t"] = batch_res.index.map(polvo)
    batch_res["kpi_recuperacion_pct"] = batch_res.index.map(kpi)
    batch_res = batch_res.dropna()

    from scipy.stats import spearmanr
    r1, p1 = spearmanr(batch_res["residuo_abs_medio"], batch_res["polvo_sn_t"])
    r2, p2 = spearmanr(batch_res["residuo_abs_medio"], batch_res["kpi_recuperacion_pct"])
    r3, p3 = spearmanr(batch_res["frac_flag"], batch_res["kpi_recuperacion_pct"])
    print(f"residuo ln(M_cierre/M_G) en Reduccion: n={len(out)} media={out['residuo_ln_masa'].mean():.4f} sd={sd:.4f} "
          f"frac_flag(|.|>2sd)={out['flag_qc'].mean():.3f}", flush=True)
    print(out.groupby("orden_escalon_fase")["residuo_ln_masa"].describe().to_string(), flush=True)
    print(f"spearman(residuo_abs_medio_batch, polvo_Sn_batch_t) = {r1:.3f} (p={p1:.3g}, n={len(batch_res)})", flush=True)
    print(f"spearman(residuo_abs_medio_batch, KPI recuperacion_refinada_pct) = {r2:.3f} (p={p2:.3g})", flush=True)
    print(f"spearman(frac_flag_batch, KPI recuperacion_refinada_pct) = {r3:.3f} (p={p3:.3g})", flush=True)

    out.to_csv(OUT / "e8_01_qc.csv", index=False)
    batch_res.to_csv(OUT / "e8_01_qc_batch.csv")
    print("qc -> e8_01_qc.csv (+ e8_01_qc_batch.csv auxiliar)", flush=True)
    return out


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                     choices=["grid", "calib", "sensib", "ruido", "bordes", "anclaje", "qc", "all"])
    ap.add_argument("--n_rep", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    print("cargando datos...", flush=True)
    df, dev, lb, bal = cargar_todo()
    print(f"df={df.shape}, dev={len(dev)}, lockbox={len(lb)} ({time.time()-t0:.0f}s)", flush=True)
    coefs = calib_sets(df, dev, bal)
    print("coefs sin carbon:", {k: round(v, 4) for k, v in coefs[False].items() if isinstance(v, float)}, flush=True)
    print("coefs con carbon:", {k: round(v, 4) for k, v in coefs[True].items() if isinstance(v, float)}, flush=True)

    if args.stage in ("grid", "all"):
        etapa_grid(df, dev, lb, bal, coefs)
    if args.stage in ("calib", "all"):
        etapa_calib(df, dev, bal)
    if args.stage in ("sensib", "all"):
        etapa_sensib(df, dev, lb, bal, coefs)
    if args.stage in ("ruido", "all"):
        etapa_ruido(df, coefs, n_rep=args.n_rep)
    if args.stage in ("bordes", "all"):
        etapa_bordes(df, dev, bal, coefs)
    if args.stage in ("anclaje", "all"):
        etapa_anclaje(df, bal)
    if args.stage in ("qc", "all"):
        etapa_qc(df, bal, coefs)

    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
