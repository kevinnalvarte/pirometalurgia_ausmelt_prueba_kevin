"""E10-05 (iteracion 10): validacion del sistema v8, politica cross-fitted con salvaguardas, sensibilidad del objetivo, evidencia por
escalon (dosis-respuesta residual) y por batch (residuo de la accion -> KPI; adherencia con signo como monitoreo), uplift con IC
propagado y evaluacion cruzada (anti-maldicion del optimizador). Espejo de E9-07 sobre `modelo_predictivo_v8`.

Etapas (CLI): validacion | 0..4 | lockbox | evidencia | cruzada | sensibilidad | dosis_respuesta
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 .venv/Scripts/python.exe experimentos/v8/e10_05_validacion_v8.py <etapa>
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "experimentos" / "v8"))
import modelo_prescriptivo as mp
import modelo_predictivo_v4 as mp4
import modelo_predictivo_v8 as mp8
import v8_lib as L8
SALIDA = RAIZ / "experimentos" / "v8"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 300)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

df = mp8.cargar_df()
etapa = sys.argv[1] if len(sys.argv) > 1 else "validacion"
log("CONFIG:", {k: v for k, v in mp8.CONFIG_V8.items() if k not in ("ordenes_sin_recomendacion",)}, "OBJETIVO:", mp8.OBJETIVO_V8,
    "ESTADO_R:", len(mp8.ESTADO_R_V8), "PALANCAS:", mp8.PALANCAS_V8[("Reducción", "sn_dep")], mp8.PALANCAS_V8[("Reducción", "feo_ret")])


def muestras_Kw(n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap por batch del anclaje base (KPI ~ Σ ln_sn_dep + Σ ln_feo_ret + controles, DEV): muestras de (K_sn, K_feo)."""
    import statsmodels.api as sm
    b = L8.construir_batch(df).reset_index()
    cols = ["R_sum_m6_ln_sn_dep", "R_sum_m6_ln_feo_ret"] + L8.CONTROLES_BATCH
    d = b[b["es_dev"]][[L8.KPI_PRINCIPAL] + cols].dropna().reset_index(drop=True)
    X = sm.add_constant(d[cols]); y = d[L8.KPI_PRINCIPAL]
    rng = np.random.default_rng(seed); out = []
    base = sm.OLS(y, X).fit().params
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        p = sm.OLS(y.iloc[idx], X.iloc[idx]).fit().params
        out.append({"K_sn": p["R_sum_m6_ln_sn_dep"], "K_feo": p["R_sum_m6_ln_feo_ret"]})
    m = pd.DataFrame(out); m.attrs["puntual"] = (float(base["R_sum_m6_ln_sn_dep"]), float(base["R_sum_m6_ln_feo_ret"]))
    return m


if etapa == "validacion":
    tabla = mp8.tabla_validacion_v8(df); tabla.to_csv(SALIDA / "e10_05_tabla_validacion_v8.csv", index=False)
    log("\n", tabla.round(3).to_string())
    sistema = mp8.entrenar_sistema_v8(df, n_boot=500)
    ef = mp8.tabla_efectos_control_v8(sistema); ef.to_csv(SALIDA / "e10_05_efectos_control_v8.csv", index=False)
    log("\n", ef.round(4).to_string())
    for fase in mp8.FASES:
        for clave in mp8.TARGETS_V8[fase]:
            p = mp8.predicciones_oof_y_lockbox_v8(df, fase, clave)
            p.to_csv(SALIDA / f"e10_05_pred_{fase}_{clave}.csv", index=False)
    log("predicciones guardadas")
    df_base = mp8._preparar_base(df); dev = sistema.batches_dev
    curvas = []
    red = df_base[(df_base.fase_proceso == "Reducción") & df_base.Batch.isin(dev)]
    rng = np.random.default_rng(1)
    for orden in (0, 1, 2, 3):
        sub = red[red.orden_escalon_fase == orden].dropna(subset=["m6_sn_inv_kg_prev", "ley_sn_escoria_pct_prev"])
        for q in (0.25, 0.5, 0.75):
            r = sub.iloc[(sub["m6_avance_prev"] - sub["m6_avance_prev"].quantile(q)).abs().argsort()[:1]].iloc[0]
            st, ac, cx = mp8.fila_a_state_action_context("Reducción", r)
            for pal in mp8.ACCIONES_OPTIMIZABLES_V8["Reducción"]:
                for caja in (False, True):
                    try:
                        c = mp8.curva_respuesta_v8(sistema, "Reducción", orden, st, ac, cx, pal, caja=caja)
                    except Exception as e:  # noqa: BLE001
                        log("curva", orden, q, pal, caja, type(e).__name__, e); continue
                    c.insert(0, "caja", caja); c.insert(0, "q_avance", q); c.insert(0, "avance_prev", float(r["m6_avance_prev"]))
                    c.insert(0, "orden", orden); c.insert(0, "Batch", r["Batch"]); c.insert(0, "fase", "Reducción")
                    c["hist"] = float(ac[pal]); curvas.append(c)
    cur = pd.concat(curvas, ignore_index=True); cur.to_csv(SALIDA / "e10_05_curvas_respuesta_v8.csv", index=False)
    # optimo interior: fraccion de curvas (sin caja) cuyo argmax de J esta estrictamente dentro de [P5, P95]
    res = []
    for (b, o, q, pal), g in cur[~cur.caja].groupby(["Batch", "orden", "q_avance", "palanca"]):
        j = g["J"].to_numpy(); i = int(np.nanargmax(j)); res.append({"Batch": b, "orden": o, "q_avance": q, "palanca": pal,
                                                                     "interior": 0 < i < len(j) - 1, "valor_opt": float(g["valor"].iloc[i]), "hist": float(g["hist"].iloc[0])})
    res = pd.DataFrame(res); res.to_csv(SALIDA / "e10_05_optimo_interior.csv", index=False)
    log("optimo interior por palanca:\n", res.groupby("palanca")["interior"].mean().round(2).to_string())

elif etapa == "evidencia":
    partes = [pd.read_csv(SALIDA / f"e10_05_parcial_{k}.csv") for k in ["0", "1", "2", "3", "4", "lockbox"] if (SALIDA / f"e10_05_parcial_{k}.csv").exists()]
    esc = pd.concat(partes, ignore_index=True); esc.to_csv(SALIDA / "e10_05_recomendaciones_v8_escalones_sin_filtro.csv", index=False)
    esc = mp8.filtrar_solo_costo(esc); esc.to_csv(SALIDA / "e10_05_recomendaciones_v8_escalones.csv", index=False)
    log("escalones", esc.shape, "batches", esc.Batch.nunique())
    log("estado de recomendacion por fase/orden:\n", pd.crosstab([esc.fase, esc.orden_escalon_fase], esc.estado_recomendacion, normalize="index").round(3).to_string())
    # --- salvaguardas del comite: magnitud de los cambios por orden (Δ/sd), % en cota de la caja, % fuera de P5-P95, R0 <= P90
    filas = []
    ok = esc[esc.estado_recomendacion.isin(["recomendar", "mantener", "mantener:solo_costo"])]
    sistema = mp8.entrenar_sistema_v8(df, n_boot=0)
    for (fase, orden), g in ok.groupby(["fase", "orden_escalon_fase"]):
        for a in mp8.ACCIONES_OPTIMIZABLES_V8[fase]:
            if f"rec__{a}" not in g or f"sd_orden__{a}" not in g:
                continue
            lim = sistema.limites_orden.get((fase, int(orden)), {}).get(a)
            d = (g[f"rec__{a}"] - g[f"hist__{a}"]); dsd = d / g[f"sd_orden__{a}"]
            fila = {"fase": fase, "orden": orden, "palanca": a, "n": len(g), "hist_P50": float(g[f"hist__{a}"].median()),
                    "delta_P5": float(d.quantile(.05)), "delta_P50": float(d.median()), "delta_P95": float(d.quantile(.95)),
                    "delta_medio": float(d.mean()), "pct_cambia": float((d.abs() > 1e-6).mean() * 100), "pct_sube": float((d > 1e-6).mean() * 100),
                    "pct_baja": float((d < -1e-6).mean() * 100), "pct_abs_delta_mayor_1sd": float((dsd.abs() > 1.0 + 1e-6).mean() * 100)}
            if lim is not None:
                p5, p95, sd, p90 = lim
                fila.update({"pct_rec_mayor_P95": float((g[f"rec__{a}"] > p95 + 1e-6).mean() * 100), "pct_rec_menor_P5": float((g[f"rec__{a}"] < p5 - 1e-6).mean() * 100),
                             "pct_rec_en_borde_caja": float(((g[f"rec__{a}"] - g[f"hist__{a}"]).abs().sub(sd * mp8.CONFIG_V8["delta_max_sd"]).abs() < 1e-6).mean() * 100)})
                if fase == "Reducción" and orden == 0 and a == "tasa_feed_Carbon_kg_min":
                    fila["pct_rec_mayor_P90"] = float((g[f"rec__{a}"] > p90 + 1e-6).mean() * 100)
            filas.append(fila)
    mag = pd.DataFrame(filas); mag.to_csv(SALIDA / "e10_05_magnitud_cambios.csv", index=False); log("\n", mag.round(2).to_string())
    perfil = esc.groupby(["fase", "orden_escalon_fase"])[[c for c in esc.columns if c.startswith("rec__") or c.startswith("hist__")]].mean()
    perfil.to_csv(SALIDA / "e10_05_perfil_politica_v8.csv"); log("\n", perfil.round(2).to_string())
    # --- por batch, agregacion, uplift propagado
    pb = mp8.metricas_por_batch_v8(df, esc); pb.to_csv(SALIDA / "e10_05_por_batch_v8.csv", index=False)
    mKw = muestras_Kw(); mKw.to_csv(SALIDA / "e10_05_muestras_Kw.csv", index=False)
    up = {}
    for nombre, sub in [("DEV", pb[~pb.es_lockbox]), ("LOCKBOX", pb[pb.es_lockbox]), ("TOTAL", pb)]:
        for ft, et in [((1.0, 1.0), "solo_Kw"), ((0.6, 1.4), "Kw_y_theta")]:
            u = mp8.uplift_propagado(sub, mKw, factor_theta=ft); u.update({"subset": nombre, "propaga": et}); up[(nombre, et)] = u
    up = pd.DataFrame(up.values()); up.to_csv(SALIDA / "e10_05_uplift_propagado.csv", index=False); log("\n", up.round(4).to_string())
    log("uplift por batch (pp): mediana", pb["uplift_batch_pp"].median(), "media", pb["uplift_batch_pp"].mean(), "% > 0", (pb["uplift_batch_pp"] > 1e-6).mean())
    log("\n", pb.groupby("es_lockbox")[["uplift_batch_pp", "J_R_temprano_hist_pp", "J_R_temprano_rec_pp", "J_R_tardio_hist_pp", "J_R_tardio_rec_pp"]].mean().round(3).to_string())
    # --- adherencia (monitoreo, dictamen: no es prueba) con distancia y con direccion
    vars_ev = [c for c in pb.columns if c.startswith("dist_") or c.startswith("dir_")] + ["uplift_batch_pp"]
    ev = mp8.evidencia_politica_v8(pb, variables=vars_ev, n_perm=500)
    ev.to_csv(SALIDA / "e10_05_evidencia_politica_v8.csv", index=False)
    log("\n", ev[ev.kpi == "recuperacion_refinada_pct"].round(4).to_string())
    rng = np.random.default_rng(0); pbp = pb.copy()
    for c in vars_ev:
        pbp[c] = rng.permutation(pb[c].to_numpy())
    evp = mp8.evidencia_politica_v8(pbp, variables=vars_ev, n_perm=0); evp.to_csv(SALIDA / "e10_05_evidencia_placebo_v8.csv", index=False)

elif etapa == "cruzada":
    esc = pd.read_csv(SALIDA / "e10_05_recomendaciones_v8_escalones.csv")
    cr = mp8.evaluacion_cruzada(df, esc, log=log); cr.to_csv(SALIDA / "e10_05_evaluacion_cruzada.csv", index=False)
    K, w = mp8.CONFIG_V8["pp_kpi_por_unidad_sdi"], mp8.CONFIG_V8["w_sdi"]
    for nm in ("opt", "cruzado", "hgb"):
        cr[f"dJ_{nm}_pp"] = K * (cr[f"d_sn_dep_{nm}"] + w * cr[f"d_feo_ret_{nm}"])
    res = {"n_escalones": len(cr)}
    for nm in ("opt", "cruzado", "hgb"):
        res[f"dJ_{nm}_media_pp"] = float(cr[f"dJ_{nm}_pp"].mean()); res[f"dJ_{nm}_frac_pos"] = float((cr[f"dJ_{nm}_pp"] > 0).mean())
    res["corr_opt_cruzado"] = float(cr["dJ_opt_pp"].corr(cr["dJ_cruzado_pp"])); res["corr_opt_hgb"] = float(cr["dJ_opt_pp"].corr(cr["dJ_hgb_pp"]))
    res["ratio_cruzado_opt"] = res["dJ_cruzado_media_pp"] / res["dJ_opt_media_pp"] if res["dJ_opt_media_pp"] else np.nan
    res["ratio_hgb_opt"] = res["dJ_hgb_media_pp"] / res["dJ_opt_media_pp"] if res["dJ_opt_media_pp"] else np.nan
    # IC bootstrap por batch de la media del dJ cruzado y hgb
    rng = np.random.default_rng(0); B = cr["Batch"].unique()
    for nm in ("cruzado", "hgb"):
        med = []
        for _ in range(1000):
            bb = rng.choice(B, len(B), replace=True); med.append(cr[cr.Batch.isin(bb)].groupby("Batch")[f"dJ_{nm}_pp"].sum().mean())
        res[f"dJ_{nm}_batch_medio_pp"] = float(cr.groupby("Batch")[f"dJ_{nm}_pp"].sum().mean()); res[f"dJ_{nm}_batch_ic_lo"] = float(np.percentile(med, 2.5)); res[f"dJ_{nm}_batch_ic_hi"] = float(np.percentile(med, 97.5))
    pd.DataFrame([res]).to_csv(SALIDA / "e10_05_evaluacion_cruzada_resumen.csv", index=False); log(res)
    log("\n", cr.groupby("orden_escalon_fase")[["dJ_opt_pp", "dJ_cruzado_pp", "dJ_hgb_pp"]].mean().round(4).to_string())

elif etapa == "sensibilidad":
    dev, lockbox, folds = mp4._folds_dev(df, 5)
    k = 0; batches = folds[k][:70]
    base_cfg = dict(mp8.CONFIG_V8)
    variantes = {"base": (dict(), "log"),
                 "w2": (dict(w_sdi=2.0), "log"), "w4": (dict(w_sdi=4.0), "log"), "w_eiv_7.4": (dict(w_sdi=7.4), "log"),
                 "legado_K3.52_w7.86": (dict(w_sdi=7.86, pp_kpi_por_unidad_sdi=3.522), "log"), "w11": (dict(w_sdi=11.0), "log"),
                 "lineal_kg_l1": (dict(lambda_kg=1.0), "lineal_kg"), "lineal_kg_l3": (dict(lambda_kg=3.0), "lineal_kg"), "lineal_kg_l4.7": (dict(lambda_kg=4.7), "lineal_kg")}
    sistema = mp8.entrenar_sistema_v8(df[df["Batch"].isin(dev - set(folds[k]))], pct_lockbox=0.0, n_boot=100)
    reps = []
    for nombre, (cfg, obj) in variantes.items():
        for b in batches:
            rep = mp8.reporte_recomendaciones_batch_v8(sistema, df, b, objetivo=obj, config=dict(base_cfg, **cfg))
            rep["variante"] = nombre; reps.append(rep)
        log("variante", nombre, "lista")
    S = pd.concat(reps, ignore_index=True); S = mp8.filtrar_solo_costo(S); S.to_csv(SALIDA / "e10_05_sensibilidad_escalones.csv", index=False)
    S = S[S.fase == "Reducción"]
    base = S[S.variante == "base"].set_index(["Batch", "orden_escalon_fase"])
    filas = []
    for nombre in variantes:
        v = S[S.variante == nombre].set_index(["Batch", "orden_escalon_fase"])
        idx = base.index.intersection(v.index)
        for a in mp8.ACCIONES_OPTIMIZABLES_V8["Reducción"]:
            db = np.sign((base.loc[idx, f"rec__{a}"] - base.loc[idx, f"hist__{a}"]).round(6)); dv = np.sign((v.loc[idx, f"rec__{a}"] - v.loc[idx, f"hist__{a}"]).round(6))
            for orden in (0, 1, 2, 3):
                m = idx.get_level_values(1) == orden
                cambia = (db[m] != 0) | (dv[m] != 0)
                filas.append({"variante": nombre, "palanca": a, "orden": orden, "n": int(m.sum()), "n_cambia": int(cambia.sum()),
                              "pct_mismo_signo_entre_los_que_cambian": float((db[m][cambia] == dv[m][cambia]).mean() * 100) if cambia.any() else np.nan,
                              "pct_signo_opuesto": float(((db[m] * dv[m]) < 0).mean() * 100), "delta_medio_base": float((base.loc[idx, f"rec__{a}"] - base.loc[idx, f"hist__{a}"])[m].mean()),
                              "delta_medio_variante": float((v.loc[idx, f"rec__{a}"] - v.loc[idx, f"hist__{a}"])[m].mean())})
    R = pd.DataFrame(filas); R.to_csv(SALIDA / "e10_05_sensibilidad_resumen.csv", index=False); log("\n", R.round(2).to_string())

elif etapa == "dosis_respuesta":
    from scipy import stats
    import statsmodels.api as sm
    df_base = mp8._preparar_base(df); dev, lockbox = mp.split_dev_lockbox(df)
    fase = "Reducción"; filas = []; bins_out = []
    for clave in ("sn_dep", "feo_ret"):
        S, A, tg = mp8.ESTADO_V8[(fase, clave)], mp8.PALANCAS_V8[(fase, clave)], mp8.TARGETS_V8[fase][clave]
        need = list(dict.fromkeys(S + A + [tg, "Batch", "fecha_inicio", "orden_escalon_fase"]))
        d = df_base.loc[(df_base.fase_proceso == fase) & df_base.Batch.isin(dev), need].dropna(subset=S + A + [tg]).reset_index(drop=True)
        t, ex = L8.theta_dual(d, S, A, tg, mp8.SIGNO_TEORICO_V8[(fase, clave)], n_boot=300)
        y_res, A_res = ex["y_res"], ex["A_res"]
        for j, a in enumerate(A):
            x = A_res[:, j]
            # residuo parcial: y_res menos el efecto de las otras palancas (theta libre)
            otros = A_res @ ex["theta_libre"] - x * ex["theta_libre"][j]
            yp = y_res - otros
            rho, p = stats.spearmanr(x, yp)
            q = pd.qcut(x, 5, labels=False, duplicates="drop")
            rng = np.random.default_rng(0); B = d["Batch"].to_numpy(); ub = np.unique(B)
            for b in range(int(q.max()) + 1):
                m = q == b; vals = yp[m]
                boot = []
                for _ in range(300):
                    bb = rng.choice(ub, len(ub), replace=True); sel = np.isin(B, bb) & m
                    boot.append(yp[sel].mean() if sel.any() else np.nan)
                bins_out.append({"clave": clave, "palanca": a, "quintil_residuo": b, "x_medio": float(x[m].mean()), "y_res_medio": float(vals.mean()),
                                 "ic_lo": float(np.nanpercentile(boot, 2.5)), "ic_hi": float(np.nanpercentile(boot, 97.5)), "n": int(m.sum())})
            # por tercio cronologico: signo de la pendiente
            terc = L8.tercios_cronologicos(d).to_numpy(); pend = []
            for tt in range(3):
                m = terc == tt; r_, p_ = stats.spearmanr(x[m], yp[m]); pend.append((float(r_), float(p_)))
            filas.append({"clave": clave, "palanca": a, "signo_teorico": mp8.SIGNO_TEORICO_V8[(fase, clave)].get(a, 0), "n": len(d), "spearman_residual": float(rho), "p": float(p),
                          "theta_libre_1sd": float(t.set_index("palanca").loc[a, "theta_libre_1sd"]), "ci_lo_libre": float(t.set_index("palanca").loc[a, "ci_lo_libre_1sd"]),
                          "ci_hi_libre": float(t.set_index("palanca").loc[a, "ci_hi_libre_1sd"]), "rho_tercio0": pend[0][0], "p_tercio0": pend[0][1], "rho_tercio1": pend[1][0],
                          "p_tercio1": pend[1][1], "rho_tercio2": pend[2][0], "p_tercio2": pend[2][1]})
        # residuo de la accion agregado por batch -> KPI (vinculo directo palanca -> rendimiento, sin pasar por el modelo)
        b = L8.construir_batch(df).reset_index()
        for j, a in enumerate(A):
            for g, ords in (("R_temprano", [0, 1]), ("R_tardio", [2, 3]), ("R_todo", [0, 1, 2, 3])):
                m = d["orden_escalon_fase"].isin(ords).to_numpy()
                agg = pd.Series(A_res[m, j], index=d["Batch"][m]).groupby(level=0).mean().rename("res")
                dd = b.set_index("Batch").join(agg, how="inner")[[L8.KPI_PRINCIPAL, "res"] + L8.CONTROLES_BATCH].dropna()
                X = sm.add_constant((dd[["res"] + L8.CONTROLES_BATCH] - dd[["res"] + L8.CONTROLES_BATCH].mean()) / dd[["res"] + L8.CONTROLES_BATCH].std())
                r = sm.OLS(dd[L8.KPI_PRINCIPAL], X).fit(cov_type="HC3"); rho, p = stats.spearmanr(dd[L8.KPI_PRINCIPAL], dd["res"])
                filas.append({"clave": f"batch_{clave}", "palanca": a, "grupo": g, "n": len(dd), "spearman_residual": float(rho), "p": float(p),
                              "ols_pp_por_sd": float(r.params["res"]), "ols_p": float(r.pvalues["res"])})
    pd.DataFrame(filas).to_csv(SALIDA / "e10_05_dosis_respuesta.csv", index=False); pd.DataFrame(bins_out).to_csv(SALIDA / "e10_05_dosis_respuesta_bins.csv", index=False)
    log("\n", pd.DataFrame(filas).round(4).to_string()); log("\n", pd.DataFrame(bins_out).round(4).to_string())

else:
    esc = mp8.recomendaciones_cross_fitted_v8(df, etapa, n_folds=5, log=log, n_boot=100)
    esc.to_csv(SALIDA / f"e10_05_parcial_{etapa}.csv", index=False)
    log("guardado", etapa, esc.shape)
log("FIN")
