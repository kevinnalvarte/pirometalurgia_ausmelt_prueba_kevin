"""ST-02 -- predictibilidad por escalon de los 36 candidatos `st_<id>` (search_targets).

Para cada candidato (fila de `st_targets.REGISTRO`), sobre las filas de SU FASE donde el target
no es NaN (excluyendo F0 = primer escalon del batch, que ya viene NaN por construccion):

  1. HGB (HistGradientBoostingRegressor) OOF GroupKFold(5) por Batch en DEV, y ajuste en todo DEV
     evaluado en lockbox, con tres sets de features:
       A = STATE_V3[fase] + CONTEXT_V3[fase]                        (solo lo que el operador NO decide)
       B = A + CONTROL (4 palancas primitivas + exceso_o2 + Cx_sn_v4 + Cx_av_v4 [+ relacion_C_Sn_carga en Fusion])
       C = solo CONTROL
     -> st_02_predictibilidad.csv
  2. Techo de ruido por Monte Carlo (100 replicas): perturbar %CaO (+-2% relativo), %Sn (+-0.05 pts) y
     %FeO (+-0.3 pts) de forma independiente por escalon y por replica, recalcular la cadena de
     inventario (masa/Sn/FeO por trazador CaO) y volver a construir los targets con
     `st_targets.construir_df_targets`. sd_ruido = RMS de la sd entre replicas por escalon (DEV);
     R2_max = 1 - sd_ruido^2/sd_target^2. Solo aplica a candidatos que dependen de la cadena de
     trazador (se excluyen termico/polvo/B2/IRF, que no se recalculan con esta perturbacion).
     -> st_02_techo_ruido.csv
  3. Importancia por permutacion (OOF, 5 repeticiones por fold, 5 folds) del set B para los 6
     mejores candidatos por R2_oof_B de cada fase (top 8 features cada uno). -> st_02_importancia.csv
  4. Figuras: barras R2 OOF A vs B por candidato/fase; scatter real vs predicho (OOF, set B) de los
     3 mejores candidatos por fase. -> figs/st_02_*.png

Entorno: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4`. Sin joblib.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(HERE))

import st_targets as st  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import feature_engineering as fe  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import r2_score, mean_absolute_error  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CACHE = HERE / "cache_df_st.pkl"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
LOGF = HERE / "st_02_log.txt"

HGB_KW = dict(max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
              l2_regularization=1.0, random_state=42)
FASES = ["Fusión", "Reducción"]
N_REPLICAS_RUIDO = 100
SEED = 42

# Candidatos cuya formula NO se ve afectada por perturbar %CaO/%Sn/%FeO de la escoria (termicos,
# polvo, o que usan basicidad_B2/indice_irf ya calculados aguas arriba y no recalculados aqui):
# el techo de ruido por este metodo no es aplicable (no se puede medir con esta perturbacion).
EXCLUIDOS_RUIDO = {"R17_dT", "R18_T_ventana", "R19_T_gas_bhf",
                   "F06_B2_ventana", "F07_T_media", "F08_T_ventana", "F09_T_gas_bhf",
                   "F10_irf_next", "F15_dT"}

_log_lines: list[str] = []


def log(msg: object = "") -> None:
    s = str(msg)
    print(s)
    _log_lines.append(s)


def flush_log() -> None:
    LOGF.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**HGB_KW)


def es_segura(nombre: str) -> bool:
    """Anti-fuga: acepta CONTEXT/STATE/ACTION/DERIVED seguras; `_v4` se aceptan explicitamente
    (Cx_sn_v4, Cx_av_v4 no estan en el registro de `feature_engineering` porque viven en modelo_predictivo_v4,
    pero son derivadas de STATE_prev x ACTION, seguras por construccion -- mismo criterio que usa v4 internamente)."""
    return nombre.endswith("_v4") or fe.es_feature_segura_para_prescripcion(nombre)


# =============================================================================
# 0) Carga y sets de features
# =============================================================================

def cargar() -> tuple[pd.DataFrame, set, set]:
    df = pd.read_pickle(CACHE)
    df = mp4.agregar_columnas_v4(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    return df, batches_dev, batches_lockbox


def construir_feature_sets() -> dict[str, dict[str, list[str]]]:
    out = {}
    for fase in FASES:
        state_context = list(dict.fromkeys(mp3.STATE_V3[fase] + mp3.CONTEXT_V3[fase]))
        control = list(mp4.ACCIONES_OPTIMIZABLES_V4[fase]) + ["exceso_o2_combustion_pct", "Cx_sn_v4", "Cx_av_v4"]
        if fase == "Fusión":
            control = control + ["relacion_C_Sn_carga"]
        control = list(dict.fromkeys(control))
        setA = state_context
        setB = list(dict.fromkeys(state_context + control))
        setC = control
        out[fase] = dict(A=setA, B=setB, C=setC)
    return out


def validar_anti_fuga(feature_sets: dict) -> None:
    for fase, sets in feature_sets.items():
        for nombre_set, feats in sets.items():
            inseguras = [f for f in feats if not es_segura(f)]
            if inseguras:
                raise AssertionError(f"{fase} set {nombre_set}: features inseguras {inseguras}")
    log("Anti-fuga: OK, todas las features de A/B/C pasan es_feature_segura_para_prescripcion (o son _v4).")


# =============================================================================
# 1) Predictibilidad por escalon (OOF GroupKFold5 DEV + lockbox), sets A/B/C
# =============================================================================

def procesar_candidatos(df: pd.DataFrame, batches_dev: set, batches_lockbox: set,
                         feature_sets: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    cache: dict[str, dict] = {}
    for id_, r in st.REGISTRO.items():
        fase = r["fase"]
        col = "st_" + id_
        try:
            base = df[(df["fase_proceso"] == fase) & df[col].notna() & (~df["es_primer_escalon_batch"])]
            dev = base[base["Batch"].isin(batches_dev)].copy()
            lockbox = base[base["Batch"].isin(batches_lockbox)].copy()
            n_batches_dev = dev["Batch"].nunique()
            if len(dev) < 30 or n_batches_dev < 5:
                log(f"{id_:26s} ({fase:10s}): DEV insuficiente (n={len(dev)}, batches={n_batches_dev}) -> omitido")
                continue

            lo, hi = dev[col].quantile([0.005, 0.995])
            n_winsor = int(((dev[col] < lo) | (dev[col] > hi)).sum())
            dev["y"] = dev[col].clip(lo, hi)
            lockbox["y"] = lockbox[col].clip(lo, hi)
            sd_target_dev = float(dev["y"].std())

            sets = feature_sets[fase]
            resultados = {}
            n_splits = min(5, n_batches_dev)
            for nombre_set in ["A", "B", "C"]:
                fl = sets[nombre_set]
                gkf = GroupKFold(n_splits=n_splits)
                oof = np.full(len(dev), np.nan)
                fold_models = []
                for tr, te in gkf.split(dev[fl], dev["y"], dev["Batch"]):
                    m = make_model()
                    m.fit(dev[fl].iloc[tr], dev["y"].iloc[tr])
                    oof[te] = m.predict(dev[fl].iloc[te])
                    fold_models.append((tr, te, m))
                r2 = r2_score(dev["y"], oof)
                mae = mean_absolute_error(dev["y"], oof)
                m_full = make_model().fit(dev[fl], dev["y"])
                if len(lockbox):
                    pred_lb = m_full.predict(lockbox[fl])
                    r2_lb = r2_score(lockbox["y"], pred_lb)
                else:
                    r2_lb = np.nan
                resultados[nombre_set] = dict(r2=r2, mae=mae, oof=oof, fold_models=fold_models,
                                              r2_lockbox=r2_lb, model_full=m_full)

            row = dict(id=id_, fase=fase, familia=r["familia"], n_dev=len(dev), n_lockbox=len(lockbox),
                       sd_target_dev=sd_target_dev,
                       r2_oof_A=resultados["A"]["r2"], r2_oof_B=resultados["B"]["r2"], r2_oof_C=resultados["C"]["r2"],
                       mae_oof_B=resultados["B"]["mae"],
                       r2_lockbox_A=resultados["A"]["r2_lockbox"], r2_lockbox_B=resultados["B"]["r2_lockbox"],
                       ganancia_control=resultados["B"]["r2"] - resultados["A"]["r2"], n_winsor=n_winsor)
            rows.append(row)
            cache[id_] = dict(dev=dev, lockbox=lockbox, resultados=resultados, fase=fase)
            log(f"{id_:26s} ({fase:10s},{r['familia']:16s}): n_dev={len(dev):5d} n_lb={len(lockbox):4d} "
                f"r2A={row['r2_oof_A']:+.3f} r2B={row['r2_oof_B']:+.3f} r2C={row['r2_oof_C']:+.3f} "
                f"gan={row['ganancia_control']:+.3f} r2lbA={row['r2_lockbox_A']:+.3f} r2lbB={row['r2_lockbox_B']:+.3f} "
                f"n_winsor={n_winsor}")
        except Exception as exc:  # noqa: BLE001
            log(f"{id_}: ERROR -- {exc}")
            log(traceback.format_exc())
        flush_log()
    return pd.DataFrame(rows), cache


# =============================================================================
# 2) Techo de ruido (Monte Carlo, perturbacion del trazador CaO/Sn/FeO)
# =============================================================================

def _replica_perturbada(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # base ya trae columnas st_<id> (viene de cache_df_st.pkl, ya pasado una vez por
    # construir_df_targets): hay que soltarlas antes de volver a llamar construir_df_targets,
    # si no pd.concat duplica el nombre de columna y dft["st_<id>"] deja de ser una Series.
    d = base.drop(columns=[c for c in base.columns if c.startswith("st_")]).copy()
    cao = base["ley_cao_escoria_pct"]
    sn = base["ley_sn_escoria_pct"]
    feo = base["ley_feo_escoria_pct"]
    cao_p = cao * (1.0 + rng.normal(0.0, 0.02, size=len(base)))
    sn_p = sn + rng.normal(0.0, 0.05, size=len(base))
    feo_p = feo + rng.normal(0.0, 0.3, size=len(base))
    d["ley_cao_escoria_pct"] = cao_p
    d["ley_sn_escoria_pct"] = sn_p
    d["ley_feo_escoria_pct"] = feo_p

    cum_cao_incl = d.groupby("Batch")["feed_CaO_kgh"].cumsum()
    masa = cum_cao_incl / (cao_p.where(cao_p > 0.5) / 100.0)
    sn_inv = masa * sn_p / 100.0
    feo_inv = masa * feo_p / 100.0
    masa_prev = masa.groupby(d["Batch"]).shift(1)
    sn_inv_prev = sn_inv.groupby(d["Batch"]).shift(1)
    feo_inv_prev = feo_inv.groupby(d["Batch"]).shift(1)

    d["masa_escoria_est_kg"] = masa
    d["masa_escoria_est_kg_prev"] = masa_prev
    d["sn_inventario_escoria_est_kg"] = sn_inv
    d["sn_inventario_escoria_est_kg_prev"] = sn_inv_prev
    d["feo_inventario_escoria_est_kg"] = feo_inv
    d["feo_inventario_escoria_est_kg_prev"] = feo_inv_prev
    d["d_masa_escoria_est_kg"] = masa - masa_prev

    sn_ext = d["feed_Sn_kgf"] - (sn_inv - sn_inv_prev)
    feo_ext = -(feo_inv - feo_inv_prev)
    d["sn_extraido_est_kg"] = sn_ext
    d["feo_extraido_est_kg"] = feo_ext
    disp = sn_inv_prev + d["feed_Sn_kgf"].fillna(0.0)
    frac = sn_ext / disp
    d["frac_sn_extraido_escalon"] = frac.where(disp > 20.0)

    return st.construir_df_targets(d)


def techo_ruido(df: pd.DataFrame, batches_dev: set, df_pred: pd.DataFrame) -> pd.DataFrame:
    ids_ruido = [id_ for id_ in st.REGISTRO if id_ not in EXCLUIDOS_RUIDO]
    n_rows = len(df)
    acc = {id_: np.full((N_REPLICAS_RUIDO, n_rows), np.nan, dtype=np.float32) for id_ in ids_ruido}

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    n_hechas = N_REPLICAS_RUIDO
    for k in range(N_REPLICAS_RUIDO):
        try:
            dft = _replica_perturbada(df, rng)
        except Exception as exc:  # noqa: BLE001
            log(f"Monte Carlo replica {k}: ERROR -- {exc}")
            n_hechas = k
            break
        for id_ in ids_ruido:
            acc[id_][k, :] = dft["st_" + id_].to_numpy(dtype=np.float32)
        if (k + 1) % 20 == 0:
            log(f"  Monte Carlo: {k + 1}/{N_REPLICAS_RUIDO} replicas ({time.time() - t0:.1f}s)")
    log(f"Monte Carlo terminado: {n_hechas} replicas en {time.time() - t0:.1f}s")
    flush_log()

    rows = []
    for id_, r in st.REGISTRO.items():
        fase = r["fase"]
        col = "st_" + id_
        aplica = id_ not in EXCLUIDOS_RUIDO
        if not aplica:
            rows.append(dict(id=id_, fase=fase, familia=r["familia"], aplica_mc=False, n=0,
                              sd_ruido=np.nan, sd_target=np.nan, r2_max=np.nan, n_replicas=0,
                              nota="no depende de la cadena de trazador CaO/Sn/FeO recalculada aqui "
                                   "(termico/polvo, o usa basicidad_B2/indice_irf calculados aguas arriba): "
                                   "techo de ruido no medible con este metodo"))
            continue
        mask = (df["fase_proceso"] == fase) & df[col].notna() & (~df["es_primer_escalon_batch"]) & df["Batch"].isin(batches_dev)
        idxs = np.where(mask.to_numpy())[0]
        n = len(idxs)
        if n == 0:
            rows.append(dict(id=id_, fase=fase, familia=r["familia"], aplica_mc=True, n=0,
                              sd_ruido=np.nan, sd_target=np.nan, r2_max=np.nan, n_replicas=n_hechas, nota="sin filas DEV"))
            continue
        vals = acc[id_][:n_hechas, idxs]
        n_valid_rep = np.sum(~np.isnan(vals), axis=0)
        ok = n_valid_rep >= max(10, n_hechas // 3)
        if ok.sum() == 0:
            rows.append(dict(id=id_, fase=fase, familia=r["familia"], aplica_mc=True, n=0,
                              sd_ruido=np.nan, sd_target=np.nan, r2_max=np.nan, n_replicas=n_hechas,
                              nota="muy pocas replicas validas por fila"))
            continue
        per_row_sd = np.nanstd(vals[:, ok], axis=0, ddof=1)
        sd_ruido = float(np.sqrt(np.nanmean(per_row_sd ** 2)))
        fila_pred = df_pred.loc[df_pred["id"] == id_, "sd_target_dev"]
        sd_target = float(fila_pred.iloc[0]) if len(fila_pred) else float(df.loc[mask, col].std())
        r2_max = 1.0 - (sd_ruido ** 2) / (sd_target ** 2) if sd_target > 0 else np.nan
        rows.append(dict(id=id_, fase=fase, familia=r["familia"], aplica_mc=True, n=int(ok.sum()),
                          sd_ruido=sd_ruido, sd_target=sd_target, r2_max=r2_max, n_replicas=n_hechas, nota=""))
        log(f"Techo ruido {id_:26s}: n={int(ok.sum()):5d} sd_ruido={sd_ruido:10.4g} sd_target={sd_target:10.4g} r2_max={r2_max:+.3f}")
        flush_log()
    return pd.DataFrame(rows)


# =============================================================================
# 3) Importancia por permutacion (OOF) -- top 6 por fase (R2 oof B), top 8 features
# =============================================================================

def importancia_permutacion(cache: dict, df_pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    top_by_fase = {}
    for fase in FASES:
        d = df_pred[df_pred["fase"] == fase].sort_values("r2_oof_B", ascending=False)
        top_by_fase[fase] = d["id"].head(6).tolist()
    log(f"Top 6 por fase (R2 oof B): {top_by_fase}")

    rows = []
    for fase, ids in top_by_fase.items():
        for id_ in ids:
            try:
                info = cache[id_]
                dev = info["dev"]
                fl_dict = info["resultados"]["B"]
                fold_models = fl_dict["fold_models"]
                feats = list(dev.columns)  # placeholder, se corrige abajo
                # features reales del set B usadas para entrenar cada fold_model:
                feats = fold_models[0][2].feature_names_in_.tolist() if hasattr(fold_models[0][2], "feature_names_in_") else None
                if feats is None:
                    raise RuntimeError("no se pudieron recuperar los nombres de features del modelo B")
                imp_accum = {f: [] for f in feats}
                for tr, te, m in fold_models:
                    Xte, yte = dev[feats].iloc[te], dev["y"].iloc[te]
                    if len(te) < 10:
                        continue
                    pi = permutation_importance(m, Xte, yte, n_repeats=5, random_state=SEED, scoring="r2")
                    for j, f in enumerate(feats):
                        imp_accum[f].extend(pi.importances[j].tolist())
                imp_mean = {f: float(np.mean(v)) if v else np.nan for f, v in imp_accum.items()}
                imp_std = {f: float(np.std(v)) if v else np.nan for f, v in imp_accum.items()}
                top8 = sorted(imp_mean, key=lambda f: (imp_mean[f] if pd.notna(imp_mean[f]) else -np.inf), reverse=True)[:8]
                for f in top8:
                    sub = dev[[f, "y"]].dropna()
                    corr = float(sub[f].corr(sub["y"])) if len(sub) > 5 else np.nan
                    sospechosa = bool(pd.notna(corr) and abs(corr) > 0.9)
                    rows.append(dict(id=id_, fase=fase, feature=f, importancia_mean=imp_mean[f],
                                      importancia_std=imp_std[f], corr_con_target=corr, sospechosa_fuga=sospechosa))
                log(f"Importancia {id_:26s} ({fase}): top8 = " +
                    ", ".join(f"{f}({imp_mean[f]:.3f})" for f in top8))
            except Exception as exc:  # noqa: BLE001
                log(f"Importancia {id_}: ERROR -- {exc}")
                log(traceback.format_exc())
            flush_log()
    return pd.DataFrame(rows), top_by_fase


# =============================================================================
# 4) Figuras
# =============================================================================

def figuras(df_pred: pd.DataFrame, cache: dict, top_by_fase: dict) -> None:
    for fase in FASES:
        d = df_pred[df_pred["fase"] == fase].sort_values("r2_oof_B", ascending=False)
        if not len(d):
            continue
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(d)), 5.5))
        x = np.arange(len(d))
        width = 0.35
        ax.bar(x - width / 2, d["r2_oof_A"], width, label="Set A (STATE+CONTEXT)", color="#4c72b0")
        ax.bar(x + width / 2, d["r2_oof_B"], width, label="Set B (+CONTROL)", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(d["id"], rotation=75, ha="right", fontsize=7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("R2 OOF (DEV, GroupKFold5)")
        ax.set_title(f"{fase}: R2 OOF set A vs set B por candidato")
        ax.legend()
        fig.tight_layout()
        suf = "fusion" if fase == "Fusión" else "reduccion"
        fname = FIGS / f"st_02_r2_AB_{suf}.png"
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        log(f"Figura guardada: {fname}")

    for fase in FASES:
        ids3 = top_by_fase.get(fase, [])[:3]
        if not ids3:
            continue
        fig, axes = plt.subplots(1, len(ids3), figsize=(5 * len(ids3), 5))
        if len(ids3) == 1:
            axes = [axes]
        for ax, id_ in zip(axes, ids3):
            dev = cache[id_]["dev"]
            oof = cache[id_]["resultados"]["B"]["oof"]
            y = dev["y"].to_numpy()
            ax.scatter(y, oof, s=8, alpha=0.35, color="#4c72b0")
            ok = np.isfinite(y) & np.isfinite(oof)
            if ok.any():
                lo = min(y[ok].min(), oof[ok].min())
                hi = max(y[ok].max(), oof[ok].max())
                ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            r2 = df_pred.loc[df_pred["id"] == id_, "r2_oof_B"].iloc[0]
            ax.set_title(f"{id_}\nR2 OOF B = {r2:.3f}")
            ax.set_xlabel("real (winsorizado, DEV)")
            ax.set_ylabel("predicho OOF")
        fig.suptitle(f"{fase}: top 3 candidatos por R2 OOF (set B)")
        fig.tight_layout()
        suf = "fusion" if fase == "Fusión" else "reduccion"
        fname = FIGS / f"st_02_scatter_top3_{suf}.png"
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        log(f"Figura guardada: {fname}")


# =============================================================================
# 5) Resumen interpretativo
# =============================================================================

def resumen_interpretativo(df_pred: pd.DataFrame, df_ruido: pd.DataFrame) -> pd.DataFrame:
    res = df_pred.merge(df_ruido[["id", "sd_ruido", "sd_target", "r2_max", "aplica_mc"]], on="id", how="left")
    res["predecible"] = res["r2_oof_B"] > 0.20
    res["trivial"] = res["r2_oof_A"] > 0.85
    res["sensible_palancas"] = (res["ganancia_control"] > 0.03) & (
        res["ganancia_control"] > 0.15 * res["r2_oof_B"].clip(lower=0.01))
    res["ruido_puro"] = res["aplica_mc"] & (
        (res["r2_max"] < 0.15) | (res["r2_oof_B"] <= res["r2_max"] * 1.05))
    res = res.sort_values(["fase", "r2_oof_B"], ascending=[True, False])

    log("\n=== RESUMEN POR FASE (id, r2_oof_A, r2_oof_B, ganancia_control, r2_lockbox_B, R2_max) ===")
    cols = ["id", "familia", "r2_oof_A", "r2_oof_B", "ganancia_control", "r2_lockbox_B", "r2_max",
            "predecible", "sensible_palancas", "trivial", "ruido_puro"]
    for fase in FASES:
        log(f"\n-- {fase} --")
        sub = res[res["fase"] == fase][cols].round(3)
        log(sub.to_string(index=False))
    return res


# =============================================================================
# main
# =============================================================================

def main() -> None:
    t0 = time.time()
    log(f"ST-02 predictibilidad por escalon -- inicio {time.strftime('%Y-%m-%d %H:%M:%S')}")
    df, batches_dev, batches_lockbox = cargar()
    log(f"cache_df_st.pkl: {df.shape}, batches DEV={len(batches_dev)}, batches lockbox={len(batches_lockbox)}")

    feature_sets = construir_feature_sets()
    for fase in FASES:
        for nombre in ["A", "B", "C"]:
            log(f"Set {nombre} [{fase}] ({len(feature_sets[fase][nombre])} features): {feature_sets[fase][nombre]}")
    validar_anti_fuga(feature_sets)
    flush_log()

    log("\n--- 1) Predictibilidad OOF (GroupKFold5 DEV) + lockbox, sets A/B/C ---")
    df_pred, cache = procesar_candidatos(df, batches_dev, batches_lockbox, feature_sets)
    df_pred.to_csv(HERE / "st_02_predictibilidad.csv", index=False)
    log(f"\nGuardado st_02_predictibilidad.csv ({len(df_pred)} filas)")
    flush_log()

    log("\n--- 2) Techo de ruido (Monte Carlo, perturbacion trazador CaO/Sn/FeO) ---")
    df_ruido = techo_ruido(df, batches_dev, df_pred)
    df_ruido.to_csv(HERE / "st_02_techo_ruido.csv", index=False)
    log(f"\nGuardado st_02_techo_ruido.csv ({len(df_ruido)} filas)")
    flush_log()

    log("\n--- 3) Importancia por permutacion (OOF, top 6 por fase, top 8 features) ---")
    df_imp, top_by_fase = importancia_permutacion(cache, df_pred)
    df_imp.to_csv(HERE / "st_02_importancia.csv", index=False)
    log(f"\nGuardado st_02_importancia.csv ({len(df_imp)} filas)")
    n_sospechosas = int(df_imp["sospechosa_fuga"].sum()) if len(df_imp) else 0
    if n_sospechosas:
        log(f"ATENCION: {n_sospechosas} features con |corr|>0.9 con el target (posible fuga trivial, revisar tabla):")
        log(df_imp[df_imp["sospechosa_fuga"]][["id", "feature", "corr_con_target"]].to_string(index=False))
    flush_log()

    log("\n--- 4) Figuras ---")
    figuras(df_pred, cache, top_by_fase)
    flush_log()

    log("\n--- 5) Resumen interpretativo ---")
    resumen = resumen_interpretativo(df_pred, df_ruido)
    resumen.to_csv(HERE / "st_02_resumen.csv", index=False)

    log(f"\nTotal: {time.time() - t0:.1f}s")
    flush_log()


if __name__ == "__main__":
    main()
