"""ST-01 -- correlacion batch: rho/r de cada candidato (36) vs KPIs de batch (5), en DEV/lockbox/total,
OLS HC3 con controles, dosis-respuesta por quintiles, barrido de lambda (metal-equivalente), combinacion
F+R y redundancia entre candidatos. Ver experimentos/search_targets/SEARCH_TARGETS_diseno.md.

Ejecutar: PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 .venv/Scripts/python.exe experimentos/search_targets/st_01_correlacion_batch.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

SEED = 42
N_BOOT = 1000
RNG = np.random.default_rng(SEED)

KPIS = ["rendimiento_proxy_batch", "f_metal", "f_dross", "f_polvo", "recuperacion_real_pct"]
CONTROLES = ["feed_sn_total_t", "ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm",
             "frac_carga_secundaria_F", "feed_dross_Fe_total_t"]
NEG_KPIS = {"f_dross", "f_polvo"}
FASES = ["Fusión", "Reducción"]

LOG: list[str] = []


def log(msg: object = "") -> None:
    s = str(msg)
    print(s)
    LOG.append(s)


# =============================================================================
# utilidades estadisticas
# =============================================================================

def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """IC95% bootstrap de rho de Spearman. Aproximacion vectorizada: rankea x,y UNA vez
    y bootstrapea Pearson sobre esos rangos (equivalente a Spearman salvo el recalculo de
    empates dentro de cada replica, efecto menor con n~100-300 y pocos empates)."""
    n = len(x)
    if n < 8:
        return np.nan, np.nan
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    idx = RNG.integers(0, n, size=(n_boot, n))
    rxb, ryb = rx[idx], ry[idx]
    mx, my = rxb.mean(axis=1, keepdims=True), ryb.mean(axis=1, keepdims=True)
    dx, dy = rxb - mx, ryb - my
    num = (dx * dy).sum(axis=1)
    den = np.sqrt((dx ** 2).sum(axis=1) * (dy ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r = r[np.isfinite(r)]
    if len(r) < n_boot * 0.5:
        return np.nan, np.nan
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def corr_pair(x: pd.Series, y: pd.Series) -> dict:
    sub = pd.concat([x, y], axis=1).dropna()
    n = len(sub)
    out = dict(n=n, rho=np.nan, p_rho=np.nan, r=np.nan, p_r=np.nan, ci_low=np.nan, ci_high=np.nan)
    if n < 8:
        return out
    xv, yv = sub.iloc[:, 0].to_numpy(float), sub.iloc[:, 1].to_numpy(float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return out
    rho, p_rho = stats.spearmanr(xv, yv)
    r, p_r = stats.pearsonr(xv, yv)
    ci_low, ci_high = bootstrap_ci_spearman(xv, yv)
    out.update(rho=rho, p_rho=p_rho, r=r, p_r=p_r, ci_low=ci_low, ci_high=ci_high)
    return out


def fit_ols(sub: pd.DataFrame, regressors: list[str], kpi: str) -> tuple[object | None, list[str]]:
    """OLS HC3 de kpi ~ const + regressors, excluyendo columnas (casi) constantes (sd<1e-8) en `sub`."""
    used = [c for c in regressors if sub[c].std(ddof=1) >= 1e-8]
    if not used:
        return None, used
    X = sm.add_constant(sub[used], has_constant="add")
    y = sub[kpi]
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception:
        return None, used
    return model, used


# =============================================================================
# main
# =============================================================================

def main() -> None:
    t0 = time.time()
    batch = pd.read_csv(HERE / "st_batch.csv")
    batch = batch.set_index("Batch")
    reg = pd.read_csv(HERE / "st_registro.csv").set_index("id")
    cand_ids = list(reg.index)
    st_col = {c: f"st_{c}" for c in cand_ids}
    n_col = {c: f"n_{c}" for c in cand_ids}

    dev = batch[~batch["es_lockbox"]]
    lockbox = batch[batch["es_lockbox"]]
    total = batch
    muestras = {"dev": dev, "lockbox": lockbox, "total": total}

    log(f"ST-01 correlacion batch -- carga {time.time()-t0:.1f}s | batch {batch.shape}, "
        f"dev n={len(dev)}, lockbox n={len(lockbox)}, candidatos={len(cand_ids)}")

    # -------------------------------------------------------------------
    # 1) correlaciones candidato x KPI x muestra
    # -------------------------------------------------------------------
    rows = []
    for cid in cand_ids:
        col = st_col[cid]
        for kpi in KPIS:
            for mname, mdf in muestras.items():
                r = corr_pair(mdf[col], mdf[kpi])
                row = dict(id=cid, fase=reg.loc[cid, "fase"], familia=reg.loc[cid, "familia"],
                           agg=reg.loc[cid, "agg"], kpi=kpi, muestra=mname,
                           n=r["n"], rho_spearman=r["rho"], p_spearman=r["p_rho"],
                           r_pearson=r["r"], p_pearson=r["p_r"], ci_low=r["ci_low"], ci_high=r["ci_high"])
                if kpi in NEG_KPIS:
                    row["signo_esperado"] = -1
                    row["signo_ok"] = (r["rho"] < 0) if pd.notna(r["rho"]) else np.nan
                else:
                    row["signo_esperado"] = np.nan
                    row["signo_ok"] = np.nan
                rows.append(row)
    corr = pd.DataFrame(rows)

    # dir_ok por candidato: rho(rendimiento) > 0 en dev
    dir_ok_map = {}
    for cid in cand_ids:
        m = (corr["id"] == cid) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")
        rho_dev = corr.loc[m, "rho_spearman"]
        dir_ok_map[cid] = bool(rho_dev.iloc[0] > 0) if len(rho_dev) and pd.notna(rho_dev.iloc[0]) else False
    corr["dir_ok"] = corr["id"].map(dir_ok_map)

    # -------------------------------------------------------------------
    # 2) BH / Holm dentro de fase, en DEV, vs rendimiento_proxy_batch
    # -------------------------------------------------------------------
    corr["p_bh"] = np.nan
    corr["significativo_bh"] = pd.Series([np.nan] * len(corr), dtype=object)
    corr["p_holm"] = np.nan
    corr["significativo_holm"] = pd.Series([np.nan] * len(corr), dtype=object)
    mask_base = (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")
    for fase in FASES:
        m = mask_base & (corr["fase"] == fase)
        sub = corr.loc[m]
        pvals = sub["p_spearman"].to_numpy()
        valid = ~np.isnan(pvals)
        p_bh = np.full(len(pvals), np.nan)
        sig_bh = np.full(len(pvals), np.nan, dtype=object)
        p_holm = np.full(len(pvals), np.nan)
        sig_holm = np.full(len(pvals), np.nan, dtype=object)
        if valid.sum() > 0:
            rej_bh, pb, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
            rej_h, ph, _, _ = multipletests(pvals[valid], alpha=0.05, method="holm")
            p_bh[valid] = pb
            sig_bh[valid] = rej_bh
            p_holm[valid] = ph
            sig_holm[valid] = rej_h
        corr.loc[m, "p_bh"] = p_bh
        corr.loc[m, "significativo_bh"] = sig_bh
        corr.loc[m, "p_holm"] = p_holm
        corr.loc[m, "significativo_holm"] = sig_holm

    corr.to_csv(HERE / "st_01_correlaciones.csv", index=False, encoding="utf-8")
    log(f"\n[1-2] st_01_correlaciones.csv escrito ({len(corr)} filas).")

    # -------------------------------------------------------------------
    # 3) OLS HC3 con controles (estandarizacion con media/sd de TOTAL)
    # -------------------------------------------------------------------
    z_vars = [st_col[c] for c in cand_ids] + CONTROLES + ["idx_cronologico"]
    means_total = {v: total[v].mean() for v in z_vars}
    stds_total = {v: total[v].std(ddof=1) for v in z_vars}
    batch_z = batch.copy()
    for v in z_vars:
        s = stds_total[v]
        batch_z["z_" + v] = (batch[v] - means_total[v]) / s if s and s > 1e-12 else np.nan

    ctrl_z = ["z_" + c for c in CONTROLES]
    idx_z = "z_idx_cronologico"

    ols_rows = []
    for cid in cand_ids:
        cz = "z_" + st_col[cid]
        variantes = {"A_controles": ctrl_z, "B_controles_idx": ctrl_z + [idx_z], "C_solo": []}
        for vname, extra in variantes.items():
            for mname, mdf in muestras.items():
                sub = batch_z.loc[mdf.index, [cz] + extra + ["rendimiento_proxy_batch"]].dropna()
                n = len(sub)
                if n < 15 or sub[cz].std(ddof=1) < 1e-8:
                    ols_rows.append(dict(id=cid, fase=reg.loc[cid, "fase"], variante=vname, muestra=mname,
                                          n=n, coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan,
                                          r2_adj=np.nan, controles_usados="", controles_excluidos=",".join(extra)))
                    continue
                model, used = fit_ols(sub, [cz] + extra, "rendimiento_proxy_batch")
                if model is None or cz not in model.params.index:
                    ols_rows.append(dict(id=cid, fase=reg.loc[cid, "fase"], variante=vname, muestra=mname,
                                          n=n, coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan,
                                          r2_adj=np.nan, controles_usados="", controles_excluidos=",".join(extra)))
                    continue
                ci = model.conf_int().loc[cz]
                excl = [c for c in extra if c not in used]
                ols_rows.append(dict(id=cid, fase=reg.loc[cid, "fase"], variante=vname, muestra=mname, n=n,
                                      coef=model.params[cz], ci_low=ci[0], ci_high=ci[1], p=model.pvalues[cz],
                                      r2_adj=model.rsquared_adj, controles_usados=",".join(u for u in used if u != cz),
                                      controles_excluidos=",".join(excl)))
    ols_df = pd.DataFrame(ols_rows)
    ols_df.to_csv(HERE / "st_01_ols_controles.csv", index=False, encoding="utf-8")
    log(f"[3] st_01_ols_controles.csv escrito ({len(ols_df)} filas).")

    # -------------------------------------------------------------------
    # 4) dosis-respuesta por quintiles en DEV
    # -------------------------------------------------------------------
    dr_rows = []
    for cid in cand_ids:
        col = st_col[cid]
        sub = pd.concat([dev[col], dev["rendimiento_proxy_batch"]], axis=1).dropna()
        sub.columns = ["x", "y"]
        n = len(sub)
        if n < 25 or sub["x"].std(ddof=1) < 1e-8:
            log(f"  [4] {cid}: n={n} insuficiente o x casi constante -- sin quintiles.")
            continue
        try:
            sub["q"] = pd.qcut(sub["x"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        except Exception as e:
            log(f"  [4] {cid}: qcut fallo ({e}) -- sin quintiles.")
            continue
        grp = sub.groupby("q")["y"].agg(["mean", "sem", "count"])
        grp = grp.reindex([1, 2, 3, 4, 5])
        monotonia = int(sum(1 for i in range(1, 5)
                             if pd.notna(grp["mean"].iloc[i]) and pd.notna(grp["mean"].iloc[i - 1])
                             and grp["mean"].iloc[i] > grp["mean"].iloc[i - 1]))
        rho_trend, p_trend = stats.spearmanr(sub["q"].astype(float), sub["y"])
        for q in [1, 2, 3, 4, 5]:
            dr_rows.append(dict(id=cid, fase=reg.loc[cid, "fase"], familia=reg.loc[cid, "familia"], quintil=q,
                                 n=int(grp.loc[q, "count"]) if pd.notna(grp.loc[q, "count"]) else 0,
                                 mean_rendimiento=grp.loc[q, "mean"], sem_rendimiento=grp.loc[q, "sem"],
                                 rho_trend=rho_trend, p_trend=p_trend, monotonia=monotonia))
    dr_df = pd.DataFrame(dr_rows)
    dr_df.to_csv(HERE / "st_01_dosis_respuesta.csv", index=False, encoding="utf-8")
    log(f"[4] st_01_dosis_respuesta.csv escrito ({len(dr_df)} filas, {dr_df['id'].nunique()} candidatos).")

    # -------------------------------------------------------------------
    # 5) barrido de lambda / w -- familia metal-equivalente
    # -------------------------------------------------------------------
    grid = np.round(np.arange(0.0, 2.01, 0.1), 2)
    lam_rows = []
    for lam in grid:
        combo_total = batch["st_R01_sn_ext"] + lam * batch["st_R02_feo_ret"]
        for mname in ["dev", "lockbox"]:
            idx = muestras[mname].index
            r = corr_pair(combo_total.loc[idx], batch.loc[idx, "rendimiento_proxy_batch"])
            lam_rows.append(dict(fase="Reducción", parametro="lambda", valor=lam, muestra=mname,
                                  n=r["n"], rho=r["rho"], p=r["p_rho"]))
    for w in grid:
        combo_total = batch_z["z_" + st_col["F16_sn_ret_frac"]] + w * batch_z["z_" + st_col["F07_T_media"]]
        for mname in ["dev", "lockbox"]:
            idx = muestras[mname].index
            r = corr_pair(combo_total.loc[idx], batch.loc[idx, "rendimiento_proxy_batch"])
            lam_rows.append(dict(fase="Fusión", parametro="w", valor=w, muestra=mname,
                                  n=r["n"], rho=r["rho"], p=r["p_rho"]))
    lam_df = pd.DataFrame(lam_rows)

    lam_df["es_optimo_dev"] = False
    for fase_ in lam_df["fase"].unique():
        m_fase = lam_df["fase"] == fase_
        dev_g = lam_df[m_fase & (lam_df["muestra"] == "dev")]
        if dev_g["rho"].notna().any():
            v_opt = dev_g.loc[dev_g["rho"].idxmax(), "valor"]
            lam_df.loc[m_fase & (lam_df["valor"] == v_opt), "es_optimo_dev"] = True
    lam_df.to_csv(HERE / "st_01_lambda_sweep.csv", index=False, encoding="utf-8")

    lam_star_row = lam_df[(lam_df["fase"] == "Reducción") & (lam_df["muestra"] == "dev") & (lam_df["es_optimo_dev"])]
    lam_star = float(lam_star_row["valor"].iloc[0]) if len(lam_star_row) else np.nan
    rho_lam_star_dev = float(lam_star_row["rho"].iloc[0]) if len(lam_star_row) else np.nan
    lam_star_lb = lam_df[(lam_df["fase"] == "Reducción") & (lam_df["muestra"] == "lockbox") & (lam_df["valor"] == lam_star)]
    rho_lam_star_lockbox = float(lam_star_lb["rho"].iloc[0]) if len(lam_star_lb) else np.nan

    w_star_row = lam_df[(lam_df["fase"] == "Fusión") & (lam_df["muestra"] == "dev") & (lam_df["es_optimo_dev"])]
    w_star = float(w_star_row["valor"].iloc[0]) if len(w_star_row) else np.nan
    rho_w_star_dev = float(w_star_row["rho"].iloc[0]) if len(w_star_row) else np.nan
    w_star_lb = lam_df[(lam_df["fase"] == "Fusión") & (lam_df["muestra"] == "lockbox") & (lam_df["valor"] == w_star)]
    rho_w_star_lockbox = float(w_star_lb["rho"].iloc[0]) if len(w_star_lb) else np.nan

    log(f"[5] st_01_lambda_sweep.csv escrito. lambda* (R, Sn-lambda*FeO) = {lam_star} -> "
        f"rho_dev={rho_lam_star_dev:.4f}, rho_lockbox={rho_lam_star_lockbox:.4f}")
    log(f"    w* (F, z(F16)+w*z(F07)) = {w_star} -> rho_dev={rho_w_star_dev:.4f}, rho_lockbox={rho_w_star_lockbox:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fase, xlabel in zip(axes, FASES, ["lambda (R01 - lambda*R02_feo_ret_raw)", "w (z F16 + w*z F07)"]):
        g = lam_df[lam_df["fase"] == fase]
        for mname, style in [("dev", "-o"), ("lockbox", "--s")]:
            gg = g[g["muestra"] == mname].sort_values("valor")
            ax.plot(gg["valor"], gg["rho"], style, label=mname, markersize=4)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("rho Spearman vs rendimiento_proxy_batch")
        ax.set_title(fase)
        ax.legend()
    fig.suptitle("Barrido de la combinacion metal-equivalente")
    fig.tight_layout()
    fig.savefig(FIGS / "st_01_lambda_sweep.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 6) combinacion F + R
    # -------------------------------------------------------------------
    def mejor_y_top3(fase: str) -> tuple[str, list[str]]:
        sub = corr[(corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev") & (corr["fase"] == fase)]
        sig = sub[sub["significativo_bh"] == True]  # noqa: E712
        base = sig if len(sig) else sub
        base = base.dropna(subset=["rho_spearman"])
        mejor = base.loc[base["rho_spearman"].idxmax(), "id"]
        top3 = sub.dropna(subset=["rho_spearman"]).sort_values("rho_spearman", ascending=False)["id"].head(3).tolist()
        return mejor, top3

    mejor_R, top3_R = mejor_y_top3("Reducción")
    mejor_F, top3_F = mejor_y_top3("Fusión")
    log(f"\n[6] mejor R = {mejor_R} (top3 R = {top3_R}); mejor F = {mejor_F} (top3 F = {top3_F})")

    def fit_full(sub_idx: pd.Index, regressors: list[str], kpi: str = "rendimiento_proxy_batch"):
        sub = batch_z.loc[sub_idx, regressors + [kpi]].dropna()
        n = len(sub)
        used = [c for c in regressors if sub[c].std(ddof=1) >= 1e-8]
        if len(used) == 0 or n < 15:
            return None, used, n
        X = sm.add_constant(sub[used], has_constant="add")
        try:
            model = sm.OLS(sub[kpi], X).fit(cov_type="HC3")
        except Exception:
            return None, used, n
        return model, used, n

    comb_rows = []
    reg_m1 = ["z_" + st_col[mejor_R], "z_" + st_col[mejor_F]] + ctrl_z
    reg_m2 = ["z_" + st_col[c] for c in (top3_R + top3_F)] + ctrl_z

    for modelo_nombre, regressors in [("mejor_R_mejor_F", reg_m1), ("top3_R_top3_F", reg_m2)]:
        for mname, mdf in muestras.items():
            model, used, n = fit_full(mdf.index, regressors)
            vif_map = {}
            if model is not None and len(used) > 1:
                Xc = batch_z.loc[mdf.index, used].dropna()
                if len(Xc) > len(used) + 2:
                    Xc_const = sm.add_constant(Xc, has_constant="add")
                    for i, c in enumerate(used):
                        try:
                            vif_map[c] = variance_inflation_factor(Xc_const.values, i + 1)
                        except Exception:
                            vif_map[c] = np.nan
            if model is None:
                comb_rows.append(dict(modelo=modelo_nombre, muestra=mname, regressor=None, n=n,
                                       coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, r2_adj=np.nan, vif=np.nan))
                continue
            for c in used:
                ci = model.conf_int().loc[c]
                comb_rows.append(dict(modelo=modelo_nombre, muestra=mname, regressor=c.replace("z_st_", "").replace("z_", ""),
                                       n=n, coef=model.params[c], ci_low=ci[0], ci_high=ci[1], p=model.pvalues[c],
                                       r2_adj=model.rsquared_adj, vif=vif_map.get(c, np.nan)))
            if mname == "dev" and modelo_nombre == "top3_R_top3_F":
                altos = {k: v for k, v in vif_map.items() if pd.notna(v) and v >= 10}
                if altos:
                    log(f"    VIF>=10 en dev (top3_R_top3_F): {altos}")
                else:
                    log(f"    VIF dev (top3_R_top3_F): todos < 10 -> {vif_map}")

    comb_df = pd.DataFrame(comb_rows)
    comb_df.to_csv(HERE / "st_01_combinacion_FR.csv", index=False, encoding="utf-8")
    log(f"[6] st_01_combinacion_FR.csv escrito ({len(comb_df)} filas).")

    # -------------------------------------------------------------------
    # 7) redundancia entre candidatos (total)
    # -------------------------------------------------------------------
    cols_all = [st_col[c] for c in cand_ids]
    red = total[cols_all].corr(method="spearman")
    red.columns = cand_ids
    red.index = cand_ids
    red.to_csv(HERE / "st_01_redundancia.csv", encoding="utf-8")
    pares = []
    for i in range(len(cand_ids)):
        for j in range(i + 1, len(cand_ids)):
            v = red.iloc[i, j]
            if pd.notna(v) and abs(v) > 0.8:
                pares.append((cand_ids[i], cand_ids[j], v))
    pares.sort(key=lambda t: -abs(t[2]))
    log(f"\n[7] st_01_redundancia.csv escrito. Pares |rho|>0.8 (total, n={len(pares)}):")
    for a, b, v in pares:
        log(f"    {a} <-> {b}: rho={v:.3f}")

    # -------------------------------------------------------------------
    # 8a) figura barras rho(rendimiento) dev/lockbox por fase
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharex=True)
    for ax, fase in zip(axes, FASES):
        sub_dev = corr[(corr["fase"] == fase) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")]
        sub_dev = sub_dev.sort_values("rho_spearman", ascending=True)
        ids_order = sub_dev["id"].tolist()
        sub_lb = corr[(corr["fase"] == fase) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "lockbox")]
        sub_lb = sub_lb.set_index("id").loc[ids_order]
        y = np.arange(len(ids_order))
        colors = ["tab:blue" if s is True else "lightsteelblue" for s in sub_dev["significativo_bh"]]
        ax.barh(y + 0.2, sub_dev["rho_spearman"], height=0.35, color=colors, label="dev",
                xerr=[sub_dev["rho_spearman"] - sub_dev["ci_low"], sub_dev["ci_high"] - sub_dev["rho_spearman"]],
                error_kw=dict(lw=0.6, ecolor="black"))
        ax.barh(y - 0.2, sub_lb["rho_spearman"], height=0.35, color="tab:orange", alpha=0.6, label="lockbox")
        ax.set_yticks(y)
        ax.set_yticklabels(ids_order, fontsize=6.5)
        ax.axvline(0, color="grey", lw=0.7)
        ax.set_title(f"{fase} (azul oscuro = sig. BH en dev)")
        ax.set_xlabel("rho Spearman vs rendimiento_proxy_batch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "st_01_rho_barras.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 8b) figura dosis-respuesta top4 por fase
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fase in zip(axes, FASES):
        sub_dev = corr[(corr["fase"] == fase) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")]
        top4 = sub_dev.dropna(subset=["rho_spearman"]).sort_values("rho_spearman", ascending=False)["id"].head(4).tolist()
        for cid in top4:
            g = dr_df[dr_df["id"] == cid]
            if not len(g):
                continue
            ax.errorbar(g["quintil"], g["mean_rendimiento"], yerr=g["sem_rendimiento"], marker="o",
                        capsize=3, label=cid)
        ax.set_xlabel("quintil del candidato (DEV)")
        ax.set_ylabel("rendimiento_proxy_batch (media +/- SEM)")
        ax.set_title(f"{fase} -- top 4 por rho dev")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "st_01_dosis_respuesta_top4.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 9) ranking final por fase
    # -------------------------------------------------------------------
    rank_rows = []
    for fase in FASES:
        ids_fase = [c for c in cand_ids if reg.loc[c, "fase"] == fase]
        for cid in ids_fase:
            row = dict(id=cid, fase=fase, familia=reg.loc[cid, "familia"], agg=reg.loc[cid, "agg"])
            dev_r = corr[(corr["id"] == cid) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "dev")].iloc[0]
            lb_r = corr[(corr["id"] == cid) & (corr["kpi"] == "rendimiento_proxy_batch") & (corr["muestra"] == "lockbox")].iloc[0]
            fdross_r = corr[(corr["id"] == cid) & (corr["kpi"] == "f_dross") & (corr["muestra"] == "dev")].iloc[0]
            fpolvo_r = corr[(corr["id"] == cid) & (corr["kpi"] == "f_polvo") & (corr["muestra"] == "dev")].iloc[0]
            row.update(n_dev=dev_r["n"], rho_dev=dev_r["rho_spearman"], p_bh_dev=dev_r["p_bh"],
                       ci_dev=f"[{dev_r['ci_low']:.3f}, {dev_r['ci_high']:.3f}]" if pd.notna(dev_r["ci_low"]) else "",
                       rho_lockbox=lb_r["rho_spearman"], p_lockbox=lb_r["p_spearman"],
                       rho_f_dross_dev=fdross_r["rho_spearman"], rho_f_polvo_dev=fpolvo_r["rho_spearman"])
            ols_dev = ols_df[(ols_df["id"] == cid) & (ols_df["variante"] == "A_controles") & (ols_df["muestra"] == "dev")]
            ols_lb = ols_df[(ols_df["id"] == cid) & (ols_df["variante"] == "A_controles") & (ols_df["muestra"] == "lockbox")]
            row["coef_ols_dev"] = ols_dev["coef"].iloc[0] if len(ols_dev) else np.nan
            row["p_ols_dev"] = ols_dev["p"].iloc[0] if len(ols_dev) else np.nan
            row["coef_ols_lockbox"] = ols_lb["coef"].iloc[0] if len(ols_lb) else np.nan
            mono = dr_df[dr_df["id"] == cid]["monotonia"]
            row["monotonia"] = int(mono.iloc[0]) if len(mono) else np.nan
            row["dir_ok"] = dir_ok_map[cid]
            rank_rows.append(row)
    rank_df = pd.DataFrame(rank_rows)
    rank_df = rank_df.sort_values(["fase", "rho_dev"], ascending=[True, False])
    rank_df.to_csv(HERE / "st_01_ranking.csv", index=False, encoding="utf-8")
    log(f"\n[9] st_01_ranking.csv escrito ({len(rank_df)} filas).")

    # =====================================================================
    # resumen final del log
    # =====================================================================
    log("\n" + "=" * 78)
    log("RESUMEN FINAL")
    log("=" * 78)

    log("\n(i) top-5 por fase (rho dev, Spearman vs rendimiento_proxy_batch):")
    for fase in FASES:
        log(f"  -- {fase} --")
        top5 = rank_df[rank_df["fase"] == fase].head(5)
        for _, rr in top5.iterrows():
            log(f"    {rr['id']:<24} rho_dev={rr['rho_dev']:.4f}  p_bh_dev={rr['p_bh_dev']:.4g}  "
                f"rho_lockbox={rr['rho_lockbox']:.4f}  coef_ols_dev={rr['coef_ols_dev']:.4f}  "
                f"monotonia={rr['monotonia']}  familia={rr['familia']}")

    log("\n(ii) significativos tras BH en DEV que conservan signo en lockbox:")
    sig_confirm = rank_df[(rank_df["p_bh_dev"] < 0.05) & (np.sign(rank_df["rho_dev"]) == np.sign(rank_df["rho_lockbox"]))]
    if len(sig_confirm):
        for _, rr in sig_confirm.sort_values(["fase", "rho_dev"], ascending=[True, False]).iterrows():
            log(f"    {rr['fase']:<10} {rr['id']:<24} rho_dev={rr['rho_dev']:.4f} p_bh_dev={rr['p_bh_dev']:.4g} "
                f"rho_lockbox={rr['rho_lockbox']:.4f}")
    else:
        log("    (ninguno)")

    log("\n(iii) familias dominantes (promedio |rho_dev| entre sus candidatos, por fase):")
    fam_summary = rank_df.groupby(["fase", "familia"])["rho_dev"].agg(["mean", "count"]).round(4)
    log(fam_summary.sort_values(["fase", "mean"], ascending=[True, False]).to_string())

    log("\n(iv) candidatos con signo contrario a la teoria (rho_dev < 0, dir_ok=False) e hipotesis:")
    sorpresas = rank_df[rank_df["rho_dev"] < 0].sort_values("rho_dev")
    if len(sorpresas):
        for _, rr in sorpresas.iterrows():
            log(f"    {rr['fase']:<10} {rr['id']:<24} rho_dev={rr['rho_dev']:.4f} (n={rr['n_dev']}) familia={rr['familia']}")
        log("    Hipotesis general: candidatos exploratorios (direccion=0, ej. dT) sin signo a priori pueden salir "
            "en cualquier sentido; candidatos de escala pequena o con pocos escalones validos (n bajo) son ruidosos; "
            "y algunos 'estado terminal' (ultimo valor) dependen de cuanto avanzo la fase en ESE batch, lo que puede "
            "confundirse con el tamano/duracion del batch mas que con la eficiencia.")
    else:
        log("    (ninguno; todos los candidatos con rho_dev>=0 -- revisar si es plausible o artefacto de orientacion)")

    log(f"\n(v) barrido lambda/w -- familia metal-equivalente:")
    log(f"    lambda* (Reduccion, Sn_ext - lambda*FeO_ext) = {lam_star} -> rho_dev={rho_lam_star_dev:.4f}, "
        f"rho_lockbox={rho_lam_star_lockbox:.4f}  [rho_dev en lambda=0 (=R01 puro): "
        f"{lam_df[(lam_df.fase=='Reducción')&(lam_df.muestra=='dev')&(lam_df.valor==0)]['rho'].iloc[0]:.4f}]")
    log(f"    w* (Fusion, z(F16_sn_ret_frac) + w*z(F07_T_media)) = {w_star} -> rho_dev={rho_w_star_dev:.4f}, "
        f"rho_lockbox={rho_w_star_lockbox:.4f}  [rho_dev en w=0 (=F16 puro): "
        f"{lam_df[(lam_df.fase=='Fusión')&(lam_df.muestra=='dev')&(lam_df.valor==0)]['rho'].iloc[0]:.4f}]")

    debiles = rank_df[rank_df["rho_dev"].abs() < 0.2]
    log(f"\nHonestidad: {len(debiles)}/{len(rank_df)} candidatos tienen |rho_dev| < 0.2 (correlacion debil). "
        f"Maximo |rho_dev| observado: {rank_df['rho_dev'].abs().max():.4f} "
        f"({rank_df.loc[rank_df['rho_dev'].abs().idxmax(), 'id']}).")

    log(f"\nTiempo total: {time.time()-t0:.1f}s")

    with open(HERE / "st_01_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
