"""E8-06: exporta datos (JSON) para el artefacto HTML/Plotly de la iteracion 8 (masa v6 y prescriptor v6).

NO genera imagenes: escribe/actualiza `experimentos/v6/figs/datos_v6.js` (`window.DATOS_V6 = {...}`), numeros
redondeados a 4 cifras significativas, NaN -> null. Tres partes (cada una reemplaza solo sus claves, cargando
el JS existente si lo hay):
  A -- masa v6 (masa_v6.py), calibracion/grilla/ruido (E8-01), anclaje/theta/bateria (E8-03), SHAP+PDP de un
       HGB por (fase, clave) de mp6.TARGETS_V6. No depende de nada mas: se corre primero.
  B -- validacion/efectos/predicciones/curvas de respuesta v6 (E8-05, etapa "validacion"). Espera a que
       aparezcan `e8_05_tabla_validacion_v6.csv`, `e8_05_pred_Reducción_feo_ret.csv` y
       `e8_05_curvas_respuesta_v6.csv`.
  C -- evidencia off-policy y recomendaciones por escalon (E8-05, etapas "evidencia"/folds). Espera a que
       aparezcan `e8_05_evidencia_politica_v6.csv` y `e8_05_por_batch_v6.csv`.

Uso: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe experimentos/v6/e8_06_datos_figuras.py <a|b|c|all>
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from math import floor, log10
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent.parent
V6DIR = RAIZ / "experimentos" / "v6"
sys.path[:0] = [str(RAIZ), str(V6DIR)]
import feature_engineering as fe  # noqa: E402
import masa_v6  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v5 as mp5  # noqa: E402
import modelo_predictivo_v6 as mp6  # noqa: E402

FIGS = V6DIR / "figs"
FIGS.mkdir(exist_ok=True, parents=True)
JS_PATH = FIGS / "datos_v6.js"
PREFIX = "window.DATOS_V6 = "
SEED = 42

T0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# =============================================================================
# Utilidades de JSON (redondeo a 4 cifras significativas, NaN -> null, carga/actualiza por claves)
# =============================================================================

def _round_sig(x, sig: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    if x == 0:
        return 0.0
    d = sig - int(floor(log10(abs(x)))) - 1
    try:
        v = round(x, d)
    except (ValueError, OverflowError):
        return None
    return v


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, pd.Series):
        return _clean(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict(orient="records"))
    if isinstance(o, pd.Timestamp):
        return None if pd.isna(o) else o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, bool):
        return o
    if isinstance(o, (np.floating, float)):
        return _round_sig(o)
    if o is None or isinstance(o, (str, int)):
        return o
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return o


def cargar_js() -> dict:
    if JS_PATH.exists():
        txt = JS_PATH.read_text(encoding="utf-8").strip()
        if txt.startswith(PREFIX):
            txt = txt[len(PREFIX):]
        if txt.endswith(";"):
            txt = txt[:-1]
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            log("ADVERTENCIA: no se pudo parsear datos_v6.js existente; se reescribe desde cero")
            return {}
    return {}


def guardar_js(nuevas: dict) -> dict:
    datos = cargar_js()
    datos.update(nuevas)
    limpio = _clean(datos)
    cuerpo = json.dumps(limpio, ensure_ascii=False, separators=(",", ":"))
    JS_PATH.write_text(PREFIX + cuerpo + ";\n", encoding="utf-8")
    releido = cargar_js()
    assert set(releido) == set(limpio), "claves no coinciden tras guardar/releer datos_v6.js"
    size_mb = JS_PATH.stat().st_size / 1e6
    log(f"datos_v6.js OK -> {len(limpio)} claves totales ({size_mb:.2f} MB). Actualizadas ahora: {sorted(nuevas)}")
    return limpio


def _int_or_none(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return int(v)


def muestra(d: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    return d if len(d) <= n else d.sample(n, random_state=seed)


# =============================================================================
# Carga de datos comun
# =============================================================================

def cargar_datos():
    df = mp6.agregar_columnas_v6(pd.read_pickle(RAIZ / "experimentos" / "cache" / "df_v3.pkl"))
    bal = masa_v6.balance_batch(df)
    dev, lockbox = mp.split_dev_lockbox(df)
    return df, bal, dev, lockbox


# =============================================================================
# PARTE A -- masa v6, E8-01, E8-03, SHAP/PDP
# =============================================================================

def _elegir_batches(df: pd.DataFrame, dev: set, lockbox: set, k_por_batch: pd.Series) -> dict:
    g = df.groupby("Batch")
    n_esc = g.size()
    cal_batch = g["feed_CaO_kgh"].sum()
    dev_info = pd.DataFrame({"n": n_esc, "cal": cal_batch}).loc[sorted(dev)]
    dev_ok = dev_info[dev_info["n"] >= n_esc.median()]
    cal_med = dev_ok["cal"].median()
    score = (dev_ok["cal"] - cal_med).abs() / max(cal_med, 1.0)
    tipicos = score.sort_values().index[:2].tolist()

    cal_casi_0 = cal_batch[(cal_batch < 100) & cal_batch.index.isin(dev)]
    b_cal0 = cal_casi_0.index[0] if len(cal_casi_0) else cal_batch.idxmin()

    ka_dev = k_por_batch.reindex(sorted(dev)).dropna()
    b_alto = ka_dev.idxmax()
    b_bajo = ka_dev.idxmin()

    lb = sorted(lockbox)
    b_lb = lb[len(lb) // 2]

    return {
        "dev_tipico_1": tipicos[0], "dev_tipico_2": tipicos[1], "cal_casi_0": b_cal0,
        "k_anclaje_alto": b_alto, "k_anclaje_bajo": b_bajo, "lockbox": b_lb,
    }


def _trayectoria_batch(df: pd.DataFrame, bal: pd.DataFrame, batch: str) -> list[dict]:
    d = df[df["Batch"] == batch].sort_values("fecha_inicio").copy()
    cal_acum = d["feed_CaO_kgh"].fillna(0).cumsum()
    carga_acum = d["feed_total_kgh"].fillna(0).cumsum()
    mbal = float(bal.loc[batch, "masa_escoria_balance_kg"]) if batch in bal.index else None
    filas = []
    for (_, r), ca, cg in zip(d.iterrows(), cal_acum, carga_acum):
        filas.append({
            "fase": r["fase_proceso"], "orden": _int_or_none(r.get("orden_escalon_fase")),
            "masa_v3": r.get("masa_escoria_est_kg"), "masa_v6": r.get("m6_masa_kg"), "masa_balance": mbal,
            "sn_pct": r.get("ley_sn_escoria_pct"), "feo_pct": r.get("ley_feo_escoria_pct"),
            "cao_pct": r.get("ley_cao_escoria_pct"), "G_pct": r.get("m6_G_pct"),
            "resto_frac": r.get("m6_resto_frac"), "cal_acum": ca, "carga_acum": cg,
        })
    return filas


def calc_masa_trayectorias(df: pd.DataFrame, bal: pd.DataFrame, dev: set, lockbox: set, k_por_batch: pd.Series) -> dict:
    seleccion = _elegir_batches(df, dev, lockbox, k_por_batch)
    log("masa_trayectorias: batches elegidos", seleccion)
    return {etq: {"batch": b, "k_anclaje": _round_sig(k_por_batch.get(b)), "pasos": _trayectoria_batch(df, bal, b)}
            for etq, b in seleccion.items()}


def calc_masa_medianas(df: pd.DataFrame) -> list[dict]:
    d = df.sort_values(["Batch", "fecha_inicio"]).copy()
    B = d["Batch"]
    cal_acum = d["feed_CaO_kgh"].fillna(0).groupby(B).cumsum()
    carga_acum = d["feed_total_kgh"].fillna(0).groupby(B).cumsum()
    ratio = (d["m6_masa_kg"] / (cal_acum + carga_acum)).replace([np.inf, -np.inf], np.nan)
    base = pd.DataFrame({
        "fase": d["fase_proceso"].values, "orden": d["orden_escalon_fase"].values,
        "masa_v3": d["masa_escoria_est_kg"].values, "masa_v6": d["m6_masa_kg"].values,
        "ratio_m6_carga": ratio.values, "cao_pct": d["ley_cao_escoria_pct"].values, "G_pct": d["m6_G_pct"].values,
    })
    filas = []
    for (fase, orden), g in base.groupby(["fase", "orden"]):
        fila = {"fase": fase, "orden": _int_or_none(orden), "n": int(len(g))}
        for col in ["masa_v3", "masa_v6", "ratio_m6_carga", "cao_pct", "G_pct"]:
            s = g[col].dropna()
            if len(s) == 0:
                continue
            fila[f"{col}_p10"] = float(s.quantile(0.10))
            fila[f"{col}_mediana"] = float(s.median())
            fila[f"{col}_p90"] = float(s.quantile(0.90))
        filas.append(fila)
    return filas


def calc_dm_reduccion(df: pd.DataFrame) -> dict:
    d = df.sort_values(["Batch", "fecha_inicio"])
    B = d["Batch"]
    R = d["fase_proceso"].eq("Reducción")
    nofeed = (R & d["feed_total_kgh"].fillna(0).le(1) & d["feed_CaO_kgh"].fillna(0).le(1)).values
    out = {}
    bins = np.linspace(-0.3, 0.3, 31)
    for nombre, col in [("v3", "masa_escoria_est_kg"), ("v6", "m6_masa_kg")]:
        m = d[col].replace([np.inf, -np.inf], np.nan)
        mprev = m.groupby(B).shift(1)
        dln = np.log(m / mprev)
        x = dln[nofeed].dropna()
        x = x[x.abs() < 0.5]
        hist, edges = np.histogram(x, bins=bins)
        out[nombre] = {"bin_edges": edges.tolist(), "counts": hist.tolist(), "frac_pos": float((x > 0).mean()),
                       "n": int(len(x)), "media": float(x.mean()), "sd": float(x.std())}
    return out


def calc_inertes_concordancia(df: pd.DataFrame) -> dict:
    from scipy.stats import pearsonr, spearmanr
    d = df.sort_values(["Batch", "fecha_inicio"])
    B = d["Batch"]
    R = d["fase_proceso"].eq("Reducción")
    nofeed = (R & d["feed_total_kgh"].fillna(0).le(1) & d["feed_CaO_kgh"].fillna(0).le(1)).values

    def lnrat(col):
        v = d[col]
        vprev = v.groupby(B).shift(1)
        return np.log(vprev / v)

    base = pd.DataFrame({
        "cao": lnrat("ley_cao_escoria_pct").values, "al2o3": lnrat("ley_al2o3_escoria_pct").values,
        "sio2": lnrat("ley_sio2_escoria_pct").values,
    })[nofeed].replace([np.inf, -np.inf], np.nan).dropna()
    base = base[(base.abs() < 1.0).all(axis=1)]
    out = {"n": int(len(base))}
    for otro in ("al2o3", "sio2"):
        pr = pearsonr(base["cao"], base[otro])
        sr = spearmanr(base["cao"], base[otro])
        out[f"pearson_{otro}"], out[f"pearson_p_{otro}"] = float(pr[0]), float(pr[1])
        out[f"spearman_{otro}"], out[f"spearman_p_{otro}"] = float(sr.correlation), float(sr.pvalue)
    s = muestra(base, 800)
    out["puntos"] = {"ln_cao": s["cao"].round(4).tolist(), "ln_al2o3": s["al2o3"].round(4).tolist(),
                      "ln_sio2": s["sio2"].round(4).tolist()}
    return out


def calc_f0_composicion(df: pd.DataFrame) -> list[dict]:
    d = df.sort_values(["Batch", "fecha_inicio"]).copy()
    B = d["Batch"]
    cal_acum = d["feed_CaO_kgh"].fillna(0).groupby(B).cumsum()
    carga_acum = d["feed_total_kgh"].fillna(0).groupby(B).cumsum()
    esperado = 100 * cal_acum / (cal_acum + carga_acum)
    F = d["fase_proceso"].eq("Fusión")
    base = pd.DataFrame({
        "orden": d.loc[F, "orden_escalon_fase"].values, "cao_obs": d.loc[F, "ley_cao_escoria_pct"].values,
        "cao_esperado": esperado[F].values, "sio2_obs": d.loc[F, "ley_sio2_escoria_pct"].values,
        "al2o3_obs": d.loc[F, "ley_al2o3_escoria_pct"].values, "sn_obs": d.loc[F, "ley_sn_escoria_pct"].values,
    })
    filas = []
    for orden, g in base.groupby("orden"):
        fila = {"orden": _int_or_none(orden), "n": int(len(g))}
        for col in ["cao_obs", "cao_esperado", "sio2_obs", "al2o3_obs", "sn_obs"]:
            s = g[col].dropna()
            fila[col] = float(s.median()) if len(s) else None
        filas.append(fila)
    return filas


def calc_calibracion(df: pd.DataFrame, bal: pd.DataFrame) -> dict:
    tabla = pd.read_csv(V6DIR / "e8_01_calibracion.csv")
    d = df.sort_values(["Batch", "fecha_inicio"])
    R = d["fase_proceso"].eq("Reducción")
    cal_b = d.groupby("Batch")["feed_CaO_kgh"].sum()
    carga_b = d.groupby("Batch")["feed_total_kgh"].sum()
    cc = (cal_b / carga_b).replace([np.inf, -np.inf], np.nan)
    out_cuartiles = {}
    for nombre, col in [("v3", "masa_escoria_est_kg"), ("v6", "m6_masa_kg")]:
        mfin = d.loc[R].groupby("Batch")[col].last()
        ratio = (bal["masa_escoria_balance_kg"] / mfin).replace([np.inf, -np.inf], np.nan)
        base = pd.DataFrame({"ratio": ratio, "cc": cc}).dropna()
        base = base[(base["ratio"] > 0.3) & (base["ratio"] < 3)]
        base["cuartil"] = pd.qcut(base["cc"], 4, labels=[1, 2, 3, 4], duplicates="drop")
        tab = base.groupby("cuartil", observed=True)["ratio"].median()
        out_cuartiles[nombre] = {str(k): float(v) for k, v in tab.items()}
    return {"filas": tabla.to_dict(orient="records"), "ratio_por_cuartil_calcarga": out_cuartiles}


def calc_grilla() -> list[dict]:
    g = pd.read_csv(V6DIR / "e8_01_grilla.csv")
    cols = ["fusion", "reduccion", "delta_R", "w_obs_R", "frac_dM_pos_R", "sd_dlnM_R", "rho_ratio_calcarga",
            "sd_ln_ratio_bal", "ratio_bal_dev", "ratio_bal_lb", "R2_feo", "R2_feo_sinCaOprev", "R2_sn",
            "F_sd_ln_masa_carga"]
    sel = g[(g["aC"] == 0) | (g["fusion"] == "v3_ref")]
    return sel[cols].to_dict(orient="records")


def calc_ruido() -> list[dict]:
    return pd.read_csv(V6DIR / "e8_01_ruido.csv").to_dict(orient="records")


def calc_k_anclaje(df: pd.DataFrame, k_por_batch: pd.Series) -> dict:
    from scipy.stats import spearmanr
    anc = pd.read_csv(V6DIR / "e8_01_anclaje.csv").set_index("Batch")
    base = anc.join(k_por_batch.rename("k_anclaje_actual"), how="inner")
    vars_corr = ["cal_sobre_carga", "frac_carga_secundaria", "ley_sn_carga_batch_pct", "feed_dross_Fe_kg",
                 "T_media_C", "espesor_ladrillo_mm", "orden_cronologico"]
    corrs = {}
    for v in vars_corr:
        s = base[[v, "k_anclaje_actual"]].dropna()
        if len(s) > 10:
            r, p = spearmanr(s[v], s["k_anclaje_actual"])
            corrs[v] = {"rho": float(r), "p": float(p), "n": int(len(s))}
    x = base["k_anclaje_actual"].dropna()
    hist, edges = np.histogram(x, bins=25)
    return {"hist": {"bin_edges": edges.tolist(), "counts": hist.tolist()}, "media": float(x.mean()),
            "mediana": float(x.median()), "sd": float(x.std()), "n": int(len(x)), "spearman": corrs}


def calc_cierre_sn(df: pd.DataFrame, bal: pd.DataFrame, dev: set, lockbox: set) -> list[dict]:
    d = df.sort_values(["Batch", "fecha_inicio"])
    R = d["fase_proceso"].eq("Reducción")
    inv_v3 = d.loc[R].groupby("Batch")["sn_inventario_escoria_est_kg"].last()
    inv_v6 = d.loc[R].groupby("Batch")["m6_sn_inv_kg"].last()
    b = bal.copy()
    b["mdp"] = b["sn_metal_kg"] + b["sn_dross_kg"] + b["sn_polvo_kg"]
    b["inv_v3"], b["inv_v6"] = inv_v3, inv_v6
    b["cierre_v3"] = (b["feed_sn_total_kg"] - b["inv_v3"]) / b["mdp"]
    b["cierre_v6"] = (b["feed_sn_total_kg"] - b["inv_v6"]) / b["mdp"]
    b = b.reset_index()
    b["conjunto"] = np.where(b["Batch"].isin(dev), "dev", np.where(b["Batch"].isin(lockbox), "lockbox", "otro"))
    cols = ["Batch", "conjunto", "feed_sn_total_kg", "mdp", "inv_v3", "inv_v6", "sn_escoria_final_balance_kg",
            "cierre_v3", "cierre_v6"]
    return b[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["mdp"]).to_dict(orient="records")


def calc_e803() -> dict:
    w_anclaje = pd.read_csv(V6DIR / "e8_03_w_anclaje.csv").to_dict(orient="records")
    w_sweep = pd.read_csv(V6DIR / "e8_03_w_sweep.csv").to_dict(orient="records")
    reversion = pd.read_csv(V6DIR / "e8_03_reversion.csv").to_dict(orient="records")
    theta = pd.read_csv(V6DIR / "e8_03_theta.csv")
    theta = theta[(theta["con_signos"] == True) & (theta["parametrizacion"] == "cinetica")]  # noqa: E712
    theta_cols = ["fase", "grupo", "targets", "target", "palanca", "theta_1sd", "ci_lo_1sd", "ci_hi_1sd",
                  "signo_teorico", "concuerda", "significativo"]
    bateria = pd.read_csv(V6DIR / "e8_03_bateria.csv").to_dict(orient="records")
    fusion_beta = pd.read_csv(V6DIR / "e8_03_fusion.csv").to_dict(orient="records")
    return {
        "anclaje_w": {"w_anclaje": w_anclaje, "w_sweep": w_sweep},
        "reversion": reversion,
        "theta_e803": theta[theta_cols].to_dict(orient="records"),
        "bateria": bateria,
        "fusion_beta": fusion_beta,
    }


def calc_shap_pdp(df: pd.DataFrame, dev: set) -> dict:
    import shap
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import partial_dependence
    from sklearn.model_selection import GroupKFold

    df_base = mp6._preparar_base(df)
    estados_clave = ["m6_avance_prev", "temperatura_horno_celsius_prev", "ley_feo_escoria_pct_prev", "basicidad_B2_prev"]
    resultados = {}
    for fase, claves in mp6.TARGETS_V6.items():
        for clave, target in claves.items():
            X_cols = list(dict.fromkeys(mp6.ESTADO_V6[(fase, clave)] + mp6.PALANCAS_V6[(fase, clave)]))
            sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev)]
            need = list(dict.fromkeys(X_cols + [target, "Batch"]))
            d = sub[need].dropna().reset_index(drop=True)
            if len(d) < 50:
                log(f"SHAP {fase}/{clave}: solo {len(d)} filas, se omite")
                continue
            X, y, Bser = d[X_cols], d[target], d["Batch"]

            def _hgb():
                return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,
                                                      min_samples_leaf=20, l2_regularization=1.0, random_state=42)

            oof = np.full(len(d), np.nan)
            for tr, te in GroupKFold(5).split(X, y, Bser):
                h_fold = _hgb().fit(X.iloc[tr], y.iloc[tr])
                oof[te] = h_fold.predict(X.iloc[te])
            r2 = float(1 - np.nanmean((y - oof) ** 2) / np.nanvar(y))

            h = _hgb().fit(X, y)
            n_shap = min(600, len(X))
            Xs = X.sample(n_shap, random_state=SEED).reset_index(drop=True)
            expl = shap.TreeExplainer(h)
            sv = expl.shap_values(Xs)
            imp = pd.Series(np.abs(sv).mean(axis=0), index=X_cols).sort_values(ascending=False)
            top15 = imp.head(15)
            top10 = set(top15.head(10).index)

            n_bee = min(300, len(Xs))
            idx_bee = np.random.RandomState(SEED).choice(len(Xs), n_bee, replace=False)
            beeswarm = {}
            for j, feat in enumerate(X_cols):
                if feat not in top10:
                    continue
                vals = Xs[feat].to_numpy()[idx_bee]
                p1, p99 = np.nanpercentile(X[feat], [1, 99])
                rango = (p99 - p1) if p99 > p1 else 1.0
                norm = np.clip((vals - p1) / rango, 0, 1)
                beeswarm[feat] = {"shap": np.round(sv[idx_bee, j], 4).tolist(), "valor_norm": np.round(norm, 4).tolist(),
                                  "valor_bruto": np.round(vals, 4).tolist()}

            pdp_feats = list(dict.fromkeys(mp6.PALANCAS_V6[(fase, clave)] + [f for f in estados_clave if f in X_cols]))
            pdp_out = {}
            for feat in pdp_feats:
                try:
                    pdres = partial_dependence(h, X, [feat], grid_resolution=20, kind="average")
                except Exception as e:  # noqa: BLE001
                    log(f"PDP fallo {fase}/{clave}/{feat}: {type(e).__name__}: {e}")
                    continue
                grid, avg = pdres["grid_values"][0], pdres["average"][0]
                deciles = np.nanpercentile(X[feat], np.arange(0, 101, 10))
                pdp_out[feat] = {"grid": np.round(grid, 4).tolist(), "pred": np.round(avg, 4).tolist(),
                                  "deciles": np.round(deciles, 4).tolist()}

            resultados[f"{fase}|{clave}"] = {
                "target": target, "r2_oof": r2, "n": len(d),
                "importancia_top15": {"feature": list(top15.index), "abs_shap_medio": np.round(top15.values, 4).tolist()},
                "beeswarm": beeswarm, "pdp": pdp_out,
            }
            log(f"SHAP/PDP {fase}/{clave}: n={len(d)} R2_oof={r2:.3f} features={len(X_cols)}")
    return resultados


def parte_a() -> None:
    log("=== PARTE A ===")
    df, bal, dev, lockbox = cargar_datos()
    log(f"df {df.shape}, bal {bal.shape}, dev={len(dev)} lockbox={len(lockbox)}")

    out_k = masa_v6.agregar_masa_v6(df, params=dict(masa_v6.PARAMS_DEFAULT), bal=bal)
    k_por_batch = out_k.groupby("Batch")["m6_k_anclaje"].first()
    log("k_anclaje recomputado con PARAMS_DEFAULT: media", float(k_por_batch.mean()))

    masa_trayectorias = calc_masa_trayectorias(df, bal, dev, lockbox, k_por_batch)
    masa_medianas = calc_masa_medianas(df)
    dm_red = calc_dm_reduccion(df)
    inertes = calc_inertes_concordancia(df)
    f0 = calc_f0_composicion(df)
    calib = calc_calibracion(df, bal)
    grilla = calc_grilla()
    ruido = calc_ruido()
    k_anc = calc_k_anclaje(df, k_por_batch)
    cierre_sn = calc_cierre_sn(df, bal, dev, lockbox)
    e803 = calc_e803()
    log("bloques tabulares listos, entrenando HGB + SHAP/PDP...")
    shap_pdp = calc_shap_pdp(df, dev)

    guardar_js({
        "masa_trayectorias": masa_trayectorias,
        "masa_medianas_por_escalon": masa_medianas,
        "dm_reduccion": dm_red,
        "inertes_concordancia": inertes,
        "f0_composicion": f0,
        "calibracion": calib,
        "grilla": grilla,
        "ruido": ruido,
        "k_anclaje": k_anc,
        "cierre_sn": cierre_sn,
        "anclaje_w": e803["anclaje_w"],
        "reversion": e803["reversion"],
        "theta_e803": e803["theta_e803"],
        "bateria": e803["bateria"],
        "fusion_beta": e803["fusion_beta"],
        "shap_pdp": shap_pdp,
    })
    log("=== PARTE A terminada ===")


# =============================================================================
# PARTE B -- validacion, efectos, predicciones y curvas de respuesta (E8-05 "validacion")
# =============================================================================

ARCHIVOS_B = ["e8_05_tabla_validacion_v6.csv", "e8_05_pred_Reducción_feo_ret.csv", "e8_05_curvas_respuesta_v6.csv"]
ARCHIVOS_C = ["e8_05_evidencia_politica_v6.csv", "e8_05_por_batch_v6.csv"]


def calc_pred(fase: str, clave: str) -> dict | None:
    path = V6DIR / f"e8_05_pred_{fase}_{clave}.csv"
    if not path.exists():
        log(f"pred: no existe {path.name}, se omite")
        return None
    tiene_fecha = "fecha_inicio" in pd.read_csv(path, nrows=0).columns
    d = pd.read_csv(path, parse_dates=["fecha_inicio"] if tiene_fecha else None)
    # muestra real/pred
    s = muestra(d, 1500)
    puntos = {"real": s["real"].round(4).tolist(), "pred": s["pred"].round(4).tolist(), "conjunto": s["conjunto"].tolist()}
    # evolutivo por batch cronologico: suma para targets log (agregado telescopico), media para dT
    fagg = "sum" if clave in ("sn_dep", "feo_ret") else "mean"
    d2 = d.copy()
    d2["es_lockbox"] = d2["conjunto"].eq("lockbox")
    agg_kwargs = dict(real=("real", fagg), pred=("pred", fagg), es_lockbox=("es_lockbox", "first"))
    if tiene_fecha:
        agg_kwargs["fecha"] = ("fecha_inicio", "min")
    agg = d2.groupby(["Batch"], as_index=False).agg(**agg_kwargs)
    agg = agg.sort_values("fecha") if "fecha" in agg else agg
    evolutivo = {"batch": agg["Batch"].tolist(),
                 "fecha": agg["fecha"].astype(str).tolist() if "fecha" in agg else None,
                 "real": agg["real"].round(4).tolist(), "pred": agg["pred"].round(4).tolist(),
                 "es_lockbox": agg["es_lockbox"].tolist()}
    # R2/MAE por conjunto y por orden de escalon
    from sklearn.metrics import mean_absolute_error, r2_score
    resumen = []
    for conjunto, g in d.groupby("conjunto"):
        resumen.append({"conjunto": conjunto, "orden": None, "n": int(len(g)),
                         "r2": float(r2_score(g["real"], g["pred"])), "mae": float(mean_absolute_error(g["real"], g["pred"]))})
        if "orden_escalon_fase" in g:
            for orden, go in g.groupby("orden_escalon_fase"):
                if len(go) < 5:
                    continue
                resumen.append({"conjunto": conjunto, "orden": _int_or_none(orden), "n": int(len(go)),
                                 "r2": float(r2_score(go["real"], go["pred"])), "mae": float(mean_absolute_error(go["real"], go["pred"]))})
    return {"muestra": puntos, "evolutivo": evolutivo, "resumen": resumen}


def parte_b() -> None:
    log("=== PARTE B ===")
    validacion = pd.read_csv(V6DIR / "e8_05_tabla_validacion_v6.csv").to_dict(orient="records")
    efectos = pd.read_csv(V6DIR / "e8_05_efectos_control_v6.csv").to_dict(orient="records")
    pred = {}
    for fase, claves in mp6.TARGETS_V6.items():
        for clave in claves:
            r = calc_pred(fase, clave)
            if r is not None:
                pred[f"{fase}|{clave}"] = r
    curvas_df = pd.read_csv(V6DIR / "e8_05_curvas_respuesta_v6.csv")
    cols_curvas = ["fase", "Batch", "avance_prev", "palanca", "valor", "J", "J_sn", "J_feo", "J_costo",
                   "J_temperatura", "J_soporte", "pred_sn_dep", "pred_feo_ret", "pred_temperatura"]
    cols_curvas = [c for c in cols_curvas if c in curvas_df.columns]
    curvas_df = curvas_df[cols_curvas]
    curvas = muestra(curvas_df, 6000).to_dict(orient="records") if len(curvas_df) > 6000 else curvas_df.to_dict(orient="records")
    guardar_js({"validacion": validacion, "efectos": efectos, "pred": pred, "curvas": curvas})
    log("=== PARTE B terminada ===")


# =============================================================================
# PARTE C -- evidencia off-policy y recomendaciones por escalon (E8-05 "evidencia")
# =============================================================================

def _bins_avance(x):
    if pd.isna(x):
        return None
    if x < 0.84:
        return "<0.84"
    if x < 0.96:
        return "0.84-0.96"
    if x < 0.975:
        return "0.96-0.975"
    return ">0.975"


def calc_recomendaciones() -> dict:
    """`e8_05_recomendaciones_v6_escalones.csv` viene en formato ANCHO (una fila por Batch/fase/escalon, con
    columnas `hist__<accion>` / `rec__<accion>` por cada palanca primitiva: ver `reporte_recomendaciones_batch_v6`
    en modelo_predictivo_v6.py). Se pasa a formato largo (palanca, delta=rec-hist) para agregar por tramo.
    """
    d = pd.read_csv(V6DIR / "e8_05_recomendaciones_v6_escalones.csv")
    hist_cols = [c for c in d.columns if c.startswith("hist__")]
    palancas = [c[len("hist__"):] for c in hist_cols if f"rec__{c[len('hist__'):]}" in d.columns]
    filas = []
    for pal in palancas:
        delta = d[f"rec__{pal}"] - d[f"hist__{pal}"]
        base = pd.DataFrame({"fase": d["fase"], "delta": delta,
                              "m6_avance_prev": d.get("m6_avance_prev"), "orden": d.get("orden_escalon_fase")})
        if "m6_avance_prev" in d.columns:
            dr = base[base["fase"] == "Reducción"].copy()
            dr["tramo"] = dr["m6_avance_prev"].apply(_bins_avance)
            for tramo, g in dr.groupby("tramo"):
                gg = g["delta"].dropna()
                if len(gg) == 0:
                    continue
                filas.append({"fase": "Reducción", "palanca": pal, "tramo": tramo, "n": int(len(gg)),
                              "media": float(gg.mean()), "p25": float(gg.quantile(0.25)), "p75": float(gg.quantile(0.75))})
        if "orden_escalon_fase" in d.columns:
            df_ = base[base["fase"] == "Fusión"].copy()
            for orden, g in df_.groupby("orden"):
                gg = g["delta"].dropna()
                if len(gg) == 0:
                    continue
                filas.append({"fase": "Fusión", "palanca": pal, "tramo": _int_or_none(orden), "n": int(len(gg)),
                              "media": float(gg.mean()), "p25": float(gg.quantile(0.25)), "p75": float(gg.quantile(0.75))})
    hist_uplift = {}
    if "uplift_J_kgSn" in d.columns:
        for fase, g in d.groupby("fase"):
            x = g["uplift_J_kgSn"].dropna()
            hist, edges = np.histogram(x, bins=30)
            hist_uplift[fase] = {"bin_edges": edges.tolist(), "counts": hist.tolist(), "n": int(len(x)), "media": float(x.mean())}
    return {"resumen_por_tramo": filas, "hist_uplift_J": hist_uplift}


def parte_c() -> None:
    log("=== PARTE C ===")
    evidencia = pd.read_csv(V6DIR / "e8_05_evidencia_politica_v6.csv").to_dict(orient="records")
    por_batch = pd.read_csv(V6DIR / "e8_05_por_batch_v6.csv").to_dict(orient="records")
    recos_path = V6DIR / "e8_05_recomendaciones_v6_escalones.csv"
    recomendaciones = calc_recomendaciones() if recos_path.exists() else {}
    guardar_js({"evidencia": evidencia, "por_batch": por_batch, "recomendaciones": recomendaciones})
    log("=== PARTE C terminada ===")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else "all"
    if etapa in ("a", "all"):
        parte_a()
    if etapa in ("b", "all"):
        parte_b()
    if etapa in ("c", "all"):
        parte_c()
