"""Iteracion 8 -- inferencia de masa de escoria por escalon (v6) y targets/estado/palancas derivados.

Motivacion (auditoria del trazador CaO, 2026-09-15/16): la formula v3 `masa = cum cal / %CaO` supone que la
tolva 6 es CaO puro y la unica fuente de CaO. Los datos lo desmienten (CaO en escoria final ~ 0.3-0.5 x cal
+ 0.05-0.08 x carga; %CaO en Fusion 17-20 % frente a 9 % si la cal fuera pura) y el ruido del ensayo de
%CaO (~2.8 % relativo) supera la senal de masa por escalon en Reduccion (-1.3 %): el 31 % de los escalones
sin carga "ganan masa" y dos tercios del R2 de `v5_ln_feo_ret` era reversion del ruido de CaO.

Principios del estimador v6 (todo inferible en linea, escalon a escalon):
  1) Entradas en base humeda (kgh) con leyes desconocidas de CaO, ganga y ceniza de carbon -> se calibran
     coeficientes EFECTIVOS poblacionales (pG, gG, aC) contra la masa final del balance global de batch
     (`feature_engineering.construir_resumen_batch_balance_global`, que usa metal/dross/polvo) SOLO con
     batches DEV. Son constantes de campana, no dependen del batch en inferencia.
  2) Fusion: trazador de matriz G = CaO + SiO2 + Al2O3 + MgO (cuatro ensayos promedian el ruido):
         G_in(t) = pG * sum cal_h + gG * sum carga_h + aC * sum carbon_h
         M_t = G_in(t) / G_t
     o, alternativamente, cierre incremental del "resto no reducible" (ver 3).
  3) Reduccion: CIERRE FISICO sin %CaO. De la escoria solo salen SnO2 (Sn a metal/polvo) y FeO (Fe metalico);
     el resto (SiO2, CaO, Al2O3, MgO, otros) se conserva:
         resto_t = 1 - K * s_t - f_t         K = M_SnO2/M_Sn = 1.270 ; s, f = fracciones de Sn y FeO
         M_t * resto_t = M_{t-1} * resto_{t-1} * exp(-delta_R) + resto_in_t
         resto_in_t = pG * cal_t + gG * carga_t + aC * carbon_t   (carga en Reduccion, pequena)
     delta_R = perdida residual por escalon (polvo, volatiles) calibrada en DEV (~0).
     Opcional: mezcla en log con la observacion del trazador G (peso w_obs_R).
  4) Inventarios: Sn_inv = M s, FeO_inv = M f. Targets log (v5): ln_sn_dep, ln_feo_ret. Estado `_prev`
     (cierre de t-1, causal). Palancas cineticas (misma escala que v4): Cx_sn_v6 = tasa_C[kg/min] x Sn_inv_prev/1000, Cx_av_v6 = tasa_C x avance.
  5) Diagnostico OFFLINE (leakage): factor de anclaje por batch k_b = M_balance / M_fin -> `m6_masa_kg_anclada`.

Columnas creadas (prefijo m6_): ver CLASIFICACION_M6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
import feature_engineering as fe  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402

CACHE = RAIZ / "experimentos" / "cache" / "df_v3.pkl"
K_SNO2 = 150.71 / 118.71
OXIDOS_G = ["ley_cao_escoria_pct", "ley_sio2_escoria_pct", "ley_al2o3_escoria_pct", "ley_mgo_escoria_pct"]
UMBRAL_SN_KG, UMBRAL_FEO_KG = 20.0, 100.0
C_ESTEQ_POR_SN = 2 * 12.011 / 118.71

PARAMS_DEFAULT: dict = dict(
    # Eleccion E8-01 (grilla de 66 variantes, aC<0 excluido por no fisico): cierre fisico del resto no reducible en AMBAS fases,
    # delta_R = 0.01, sin mezcla con el trazador G. Menor sesgo con la dosis de cal (rho 0.03 vs 0.11-0.16), menor dispersion
    # frente al balance (sd ln 0.113), escala correcta (ratio balance/masa 1.04 DEV / 1.02 lockbox), 3.5 % de escalones sin
    # carga que "ganan masa" (v3: 31 %), R2 OOF ln_feo_ret 0.30 (v3: 0.15). pG/gG: solo su suma esta identificada (colineales);
    # los targets log son invariantes a su escala (sensibilidad +-30 %).
    fusion="cierre",       # "G" (trazador matriz), "CaO" (trazador CaO calibrado), "cierre" (resto incremental)
    reduccion="cierre",    # "cierre", "G", "CaO", "cierre+obs"
    pG=0.703, gG=0.3013, aC=0.0,      # coeficientes efectivos de resto_in / G_in (calibrados en DEV contra el balance global)
    pCaO=0.256, cCaO=0.0794,           # idem para el trazador CaO calibrado
    delta_R=0.01,          # perdida residual por escalon en Reduccion (fraccion): polvo, volatiles
    w_obs_R=0.0,           # peso de la observacion del trazador G en Reduccion (0 = cierre puro)
    resto_min=0.30,        # cota inferior de resto_t para evitar divisiones inestables
    ley_min_G=5.0, ley_min_cao=2.0,
)

CLASIFICACION_M6: dict[str, str] = {
    "m6_G_pct": "LEAKAGE (contemporaneo: usa leyes de t; como estado solo m6_G_pct_prev)", "m6_resto_frac": "LEAKAGE (usa leyes de t)",
    "m6_masa_kg": "LEAKAGE (usa leyes de t)", "m6_sn_inv_kg": "LEAKAGE (usa leyes de t)", "m6_feo_inv_kg": "LEAKAGE (usa leyes de t)",
    "m6_masa_kg_prev": "STATE", "m6_sn_inv_kg_prev": "STATE", "m6_feo_inv_kg_prev": "STATE", "m6_avance_prev": "DERIVED-STATE",
    "m6_resto_frac_prev": "STATE", "m6_G_pct_prev": "STATE",
    "m6_d_masa_kg": "TARGET", "m6_sn_extraido_kg": "TARGET", "m6_feo_extraido_kg": "TARGET",
    "m6_sn_disp": "DERIVED (A_t x S_prev)", "m6_ln_sn_dep": "TARGET", "m6_ln_feo_ret": "TARGET", "m6_valido": "DIAG",
    "Cx_sn_v6": "DERIVED (A_t x S_prev)", "Cx_av_v6": "DERIVED (A_t x S_prev)", "m6_dosis_C_sn": "DERIVED (A_t x S_prev)",
    "m6_exceso_C_pos": "DERIVED (A_t x S_prev)", "m6_C_x_avance": "DERIVED (A_t x S_prev)", "m6_gn_esp_nm3_t": "DERIVED (A_t x S_prev)",
    "m6_masa_kg_anclada": "LEAKAGE (usa salidas del batch)", "m6_k_anclaje": "LEAKAGE (usa salidas del batch)",
}
fe.CLASIFICACION_FEATURES.update({k: v for k, v in CLASIFICACION_M6.items() if k not in fe.CLASIFICACION_FEATURES})


def es_segura_v6(nombre: str) -> bool:
    rol = CLASIFICACION_M6.get(nombre)
    if rol is None:
        return fe.es_feature_segura_para_prescripcion(nombre) or nombre in ("Cx_sn_v4", "Cx_av_v4")
    return not (rol.startswith("LEAKAGE") or rol == "TARGET" or rol == "DIAG" or "LEAKAGE" in rol)


# =============================================================================
# 1) Calibracion poblacional de las entradas (solo DEV)
# =============================================================================

def balance_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Balance global por batch (feature_engineering) indexado por Batch."""
    return fe.construir_resumen_batch_balance_global(df).set_index("Batch")


def calibrar_entradas(df: pd.DataFrame, batches_dev: set, bal: pd.DataFrame | None = None, trazador: str = "G",
                      con_carbon: bool = False, cal_min: float = 1000.0) -> dict:
    """OLS sin intercepto (HC3) de X_final_kg = %X_fin * M_balance sobre [cal, carga, (carbon)] por batch DEV.

    Devuelve coeficientes efectivos (fraccion del kg humedo de cada corriente que acaba como X en la escoria).
    trazador: "G" (CaO+SiO2+Al2O3+MgO) o "CaO".
    """
    import statsmodels.api as sm
    bal = balance_batch(df) if bal is None else bal
    d = df.sort_values(["Batch", "fecha_inicio"])
    R = d["fase_proceso"].eq("Reducción")
    col = "m6_G_pct" if trazador == "G" else "ley_cao_escoria_pct"
    if col not in d.columns:
        d = d.assign(m6_G_pct=d[OXIDOS_G].sum(axis=1, min_count=4))
    fin = d[R].dropna(subset=[col]).groupby("Batch")[col].last() / 100
    g = d.groupby("Batch")
    X = pd.DataFrame({"cal": g["feed_CaO_kgh"].sum(), "carga": g["feed_total_kgh"].sum(), "carbon": g["feed_Carbon_kgh"].sum()})
    y = (bal["masa_escoria_balance_kg"] * fin).rename("y")
    dd = pd.concat([y, X], axis=1).dropna()
    dd = dd[dd.index.isin(batches_dev) & (dd["y"] > 0) & (dd["cal"] > cal_min)]
    cols = ["cal", "carga"] + (["carbon"] if con_carbon else [])
    m = sm.OLS(dd["y"], dd[cols]).fit(cov_type="HC3")
    ci = m.conf_int()
    out = {"trazador": trazador, "n": int(len(dd)), "r2": float(m.rsquared)}
    for c in cols:
        out[c] = float(m.params[c]); out[f"{c}_ic"] = (float(ci.loc[c, 0]), float(ci.loc[c, 1])); out[f"{c}_p"] = float(m.pvalues[c])
    return out


# =============================================================================
# 2) Estimador de masa
# =============================================================================

def _entrada_resto(df: pd.DataFrame, p: dict) -> pd.Series:
    return (p["pG"] * df["feed_CaO_kgh"].fillna(0) + p["gG"] * df["feed_total_kgh"].fillna(0)
            + p["aC"] * df["feed_Carbon_kgh"].fillna(0))


def estimar_masa(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Devuelve DataFrame con m6_masa_kg, m6_G_pct, m6_resto_frac (mismo indice que df, ordenado por Batch/fecha)."""
    p = dict(PARAMS_DEFAULT, **(params or {}))
    d = df.sort_values(["Batch", "fecha_inicio"]).copy()
    B = d["Batch"].values
    R = d["fase_proceso"].eq("Reducción").values
    G = d[OXIDOS_G].sum(axis=1, min_count=4)
    Gf = (G.where(G > p["ley_min_G"]) / 100).values
    cao = (d["ley_cao_escoria_pct"].where(d["ley_cao_escoria_pct"] > p["ley_min_cao"]) / 100).values
    s = (d["ley_sn_escoria_pct"] / 100).values
    f = (d["ley_feo_escoria_pct"] / 100).values
    resto = 1 - K_SNO2 * s - f
    ent = _entrada_resto(d, p).values
    g = d.groupby("Batch")
    cum_ent = g.apply(lambda x: _entrada_resto(x, p).cumsum()).reset_index(level=0, drop=True).reindex(d.index).values
    cum_cal = g["feed_CaO_kgh"].cumsum().values; cum_carga = g["feed_total_kgh"].cumsum().values
    M = np.full(len(d), np.nan)
    obs_G = cum_ent / Gf
    obs_CaO = (p["pCaO"] * cum_cal + p["cCaO"] * cum_carga) / cao
    for i in range(len(d)):
        primero = i == 0 or B[i - 1] != B[i]
        fase_R = R[i]
        metodo = p["reduccion"] if fase_R else p["fusion"]
        if metodo == "G":
            M[i] = obs_G[i]
        elif metodo == "CaO":
            M[i] = obs_CaO[i]
        else:  # cierre / cierre+obs
            if primero:
                M[i] = ent[i] / resto[i] if np.isfinite(resto[i]) and resto[i] > p["resto_min"] else obs_G[i]
                continue
            Mp = M[i - 1]
            if not np.isfinite(Mp):
                # re-anclar con la observacion del trazador si el encadenamiento se rompio
                M[i] = obs_G[i]
                continue
            if not (np.isfinite(resto[i]) and np.isfinite(resto[i - 1]) and resto[i] > p["resto_min"]):
                M[i] = obs_G[i] if metodo == "cierre+obs" or not fase_R else np.nan
                continue
            pred = (Mp * resto[i - 1] * np.exp(-p["delta_R"] if fase_R else 0.0) + ent[i]) / resto[i]
            if metodo == "cierre+obs" and np.isfinite(obs_G[i]) and obs_G[i] > 0 and pred > 0:
                M[i] = np.exp(p["w_obs_R"] * np.log(obs_G[i]) + (1 - p["w_obs_R"]) * np.log(pred))
            else:
                M[i] = pred
    out = pd.DataFrame({"m6_masa_kg": M, "m6_G_pct": G.values, "m6_resto_frac": resto}, index=d.index)
    return out.reindex(df.index)


def agregar_masa_v6(df: pd.DataFrame, params: dict | None = None, bal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Anade al df las columnas m6_* (masa, inventarios, estado prev, targets log, palancas cineticas v6)."""
    df = df.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True).copy()
    est = estimar_masa(df, params)
    m = est["m6_masa_kg"].replace([np.inf, -np.inf], np.nan)
    B = df["Batch"]
    s = df["ley_sn_escoria_pct"] / 100; f = df["ley_feo_escoria_pct"] / 100
    sn_inv, feo_inv = m * s, m * f
    n = {"m6_G_pct": est["m6_G_pct"], "m6_resto_frac": est["m6_resto_frac"], "m6_masa_kg": m, "m6_sn_inv_kg": sn_inv, "m6_feo_inv_kg": feo_inv}
    for c in ["m6_masa_kg", "m6_sn_inv_kg", "m6_feo_inv_kg", "m6_resto_frac", "m6_G_pct"]:
        n[c + "_prev"] = n[c].groupby(B).shift(1)
    primero = df["es_primer_escalon_batch"] if "es_primer_escalon_batch" in df else B.ne(B.shift(1))
    for c in list(n):
        if c.endswith("_prev"):
            n[c] = n[c].where(~primero)
    n["m6_avance_prev"] = 1 - fe.division_segura(n["m6_sn_inv_kg_prev"], df["cum_feed_Sn_kg_prev"], umbral=fe.UMBRAL_SN_FED_KG)
    n["m6_d_masa_kg"] = m - n["m6_masa_kg_prev"]
    n["m6_sn_extraido_kg"] = df["feed_Sn_kgf"].fillna(0) - (sn_inv - n["m6_sn_inv_kg_prev"])
    n["m6_feo_extraido_kg"] = n["m6_feo_inv_kg_prev"] - feo_inv
    sn_disp = n["m6_sn_inv_kg_prev"] + df["feed_Sn_kgf"].fillna(0)
    ok = n["m6_sn_inv_kg_prev"].gt(UMBRAL_SN_KG) & sn_inv.gt(UMBRAL_SN_KG) & n["m6_feo_inv_kg_prev"].gt(UMBRAL_FEO_KG) & feo_inv.gt(UMBRAL_FEO_KG) & ~primero
    red = df["fase_proceso"].eq("Reducción")
    n["m6_sn_disp"] = sn_disp; n["m6_valido"] = ok
    n["m6_ln_sn_dep"] = np.log(sn_disp / sn_inv).where(ok & sn_disp.gt(UMBRAL_SN_KG))
    n["m6_ln_feo_ret"] = np.log(feo_inv / n["m6_feo_inv_kg_prev"]).where(ok & red)
    c_kg = df["feed_Carbon_kgh"].fillna(0)
    tasa_c = df["tasa_feed_Carbon_kg_min"] if "tasa_feed_Carbon_kg_min" in df else c_kg / df["delta_tiempo"]
    n["Cx_sn_v6"] = tasa_c * n["m6_sn_inv_kg_prev"] / 1000.0   # misma escala que Cx_sn_v4 (kg/min x t Sn)
    n["Cx_av_v6"] = tasa_c * n["m6_avance_prev"]
    dosis = (c_kg / sn_disp).where(sn_disp.gt(200.0))
    n["m6_dosis_C_sn"] = dosis
    n["m6_exceso_C_pos"] = (c_kg - C_ESTEQ_POR_SN * sn_disp).clip(lower=0)
    n["m6_C_x_avance"] = c_kg * n["m6_avance_prev"]
    mp_ = n["m6_masa_kg_prev"]
    if "volumen_gas_natural_inyectado_lanza_escalon_nm3" in df:
        n["m6_gn_esp_nm3_t"] = (df["volumen_gas_natural_inyectado_lanza_escalon_nm3"].fillna(0) / (mp_ / 1000)).where(mp_.gt(1000))
    out = pd.concat([df, pd.DataFrame(n, index=df.index)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    if bal is not None:  # diagnostico offline (LEAKAGE): anclaje por batch a la masa final del balance
        mfin = out.loc[red].groupby("Batch")["m6_masa_kg"].last()
        k = (bal["masa_escoria_balance_kg"] / mfin).replace([np.inf, -np.inf], np.nan)
        out["m6_k_anclaje"] = out["Batch"].map(k)
        out["m6_masa_kg_anclada"] = out["m6_masa_kg"] * out["m6_k_anclaje"]
    return out


# =============================================================================
# 3) Bateria de evaluacion comun (fisica, cierre, targets)
# =============================================================================

def evaluar_masa(df6: pd.DataFrame, col_masa: str, bal: pd.DataFrame, batches_dev: set, batches_lockbox: set,
                 estado_R: list[str] | None = None, con_r2: bool = True, n_splits: int = 5) -> dict:
    """Metricas independientes de la pureza de la cal. df6 debe traer leyes, feed_*, cum_feed_Sn_kg_prev, salidas de batch.

    fisica:  frac_dM_pos_R (escalones R sin carga con masa creciente), sd_dlnM_R, autocorr_dlnM_R, frac_feo_ret_pos
    escala:  ratio_bal_{dev,lb} = mediana(M_balance / M_fin), sd_ln_ratio_bal, rho_ratio_calcarga (sesgo con dosis de cal)
    cierre:  cierre_sn = (Sn cargado - Sn_inv fin R) / (M+D+P); corr_invR_bal (Sn residual vs balance)
    Fusion:  F_sd_ln_masa_carga (suavidad de M / carga acumulada dentro del batch), F_frac_ln_sn_dep_neg
    targets: R2 OOF (HGB, GroupKFold DEV) de ln_feo_ret y ln_sn_dep en R con estado + inventarios del estimador,
             y R2 de ln_feo_ret sin %CaO_prev/B2/IRF_prev (control de reversion del ruido de ensayo)
    """
    from scipy.stats import spearmanr
    d = df6.sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
    B = d["Batch"]; R = d["fase_proceso"].eq("Reducción").values; F = ~R
    m = d[col_masa].replace([np.inf, -np.inf], np.nan); m_prev = m.groupby(B).shift(1)
    s = d["ley_sn_escoria_pct"] / 100; f = d["ley_feo_escoria_pct"] / 100
    sn_inv, feo_inv = m * s, m * f; sn_prev, feo_prev = sn_inv.groupby(B).shift(1), feo_inv.groupby(B).shift(1)
    nofeed = R & d["feed_total_kgh"].fillna(0).le(1).values & d["feed_CaO_kgh"].fillna(0).le(1).values
    dln = np.log(m / m_prev); x = dln[nofeed].dropna(); x = x[x.abs() < 0.5]
    xl = pd.DataFrame({"a": dln, "b": dln.groupby(B).shift(1)})[nofeed].dropna(); xl = xl[(xl.abs() < 0.5).all(axis=1)]
    ln_sn = np.log((sn_prev + d["feed_Sn_kgf"].fillna(0)) / sn_inv).where(sn_prev + d["feed_Sn_kgf"].fillna(0) > 500).replace([np.inf, -np.inf], np.nan)
    ln_feo = np.log(feo_inv / feo_prev).where(R).replace([np.inf, -np.inf], np.nan)
    g = d.groupby("Batch")
    mfin = m[R].groupby(B[R]).last(); invR = sn_inv[R].groupby(B[R]).last(); invF = sn_inv[F].groupby(B[F]).last()
    feed_sn = g["feed_Sn_kgf"].sum(); cc = g["feed_CaO_kgh"].sum() / g["feed_total_kgh"].sum()
    out_sn = (g[["sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"]].first() * 1000).sum(axis=1)
    cb = pd.DataFrame({"mfin": mfin, "mbal": bal["masa_escoria_balance_kg"], "invR": invR, "invF": invF, "esc_bal": bal["sn_escoria_final_balance_kg"],
                       "feed": feed_sn, "out": out_sn, "cc": cc}).replace([np.inf, -np.inf], np.nan).dropna()
    cb["ratio"] = cb["mbal"] / cb["mfin"]; cb = cb[(cb["ratio"] > 0.3) & (cb["ratio"] < 3)]
    cb["cierre"] = (cb["feed"] - cb["invR"]) / cb["out"]
    res = dict(col=col_masa, masa_finF_med=float(m[F].groupby(B[F]).last().median()), masa_finR_med=float(cb["mfin"].median()),
               frac_dM_pos_R=float((x > 0).mean()), media_dlnM_R=float(x.mean()), sd_dlnM_R=float(x.std()),
               autocorr_dlnM_R=float(xl.corr().iloc[0, 1]) if len(xl) > 10 else np.nan,
               frac_feo_ret_pos=float((ln_feo[nofeed] > 0).mean()), sd_ln_ratio_bal=float(np.log(cb["ratio"]).std()),
               rho_ratio_calcarga=float(spearmanr(cb["ratio"], cb["cc"])[0]),
               sn_finF_pct_feed=float(100 * (cb["invF"] / cb["feed"]).median()))
    for nm, sel in [("dev", cb.index.isin(batches_dev)), ("lb", cb.index.isin(batches_lockbox))]:
        q = cb[sel]
        res[f"ratio_bal_{nm}"] = float(q["ratio"].median()); res[f"cierre_sn_{nm}"] = float(q["cierre"].median())
        res[f"cierre_sn_IQR_{nm}"] = float(q["cierre"].quantile(.75) - q["cierre"].quantile(.25))
        res[f"corr_invR_bal_{nm}"] = float(np.corrcoef(q["invR"], q["esc_bal"])[0, 1]) if len(q) > 5 else np.nan
    rf = (m / (g["feed_CaO_kgh"].cumsum() + g["feed_total_kgh"].cumsum()))[F]
    res["F_sd_ln_masa_carga"] = float(rf.groupby(B[F]).apply(lambda q: np.log(q.replace(0, np.nan)).std()).median())
    res["F_frac_ln_sn_dep_neg"] = float((ln_sn[F] < 0).mean())
    if con_r2:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import GroupKFold
        if estado_R is None:
            import modelo_predictivo_v5 as mp5
            estado_R = [c for c in mp5.ESTADO_R_CURADO if c not in ("masa_escoria_est_kg_prev", "sn_inventario_escoria_est_kg_prev",
                                                                     "feo_inventario_escoria_est_kg_prev", "avance_reduccion_sn_prev")]
        Xs = d[estado_R].copy(); Xs["masa_prev"] = m_prev; Xs["sn_inv_prev"] = sn_prev; Xs["feo_inv_prev"] = feo_prev
        Xs["avance_prev"] = 1 - sn_prev / d["cum_feed_Sn_kg_prev"].where(d["cum_feed_Sn_kg_prev"] > 500)
        es_dev = B.isin(batches_dev).values

        def r2oof(y, X):
            dd = pd.concat([y.rename("y"), X, B.rename("Batch")], axis=1)[R & es_dev].dropna(subset=["y"]).reset_index(drop=True)
            dd = dd[dd["y"].abs() < 2]; oof = np.full(len(dd), np.nan)
            for tr, te in GroupKFold(n_splits).split(dd, dd["y"], dd["Batch"]):
                h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
                h.fit(dd.iloc[tr][X.columns], dd.iloc[tr]["y"]); oof[te] = h.predict(dd.iloc[te][X.columns])
            return float(1 - np.nanmean((dd["y"] - oof) ** 2) / np.nanvar(dd["y"])), int(len(dd))
        res["R2_feo"], res["n_feo"] = r2oof(ln_feo, Xs)
        drop = [c for c in ("ley_cao_escoria_pct_prev", "basicidad_B2_prev", "indice_irf_prev") if c in Xs]
        res["R2_feo_sinCaOprev"], _ = r2oof(ln_feo, Xs.drop(columns=drop))
        res["R2_sn"], res["n_sn"] = r2oof(ln_sn.where(R), Xs)
    return res


def cargar_df_base() -> pd.DataFrame:
    """Cache v3 (features v3 completas), ordenado."""
    return pd.read_pickle(CACHE).sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = cargar_df_base()
    dev, lb = mp.split_dev_lockbox(df)
    bal = balance_batch(df)
    for tr in ("G", "CaO"):
        print(tr, calibrar_entradas(df, dev, bal, tr), calibrar_entradas(df, dev, bal, tr, con_carbon=True))
    df6 = agregar_masa_v6(df, bal=bal)
    print(df6[[c for c in df6.columns if c.startswith("m6_")]].describe().T.round(3).to_string())
    print(pd.Series(evaluar_masa(df6, "m6_masa_kg", bal, dev, lb)).to_string())
    print(pd.Series(evaluar_masa(df6.assign(masa_v3=df6["masa_escoria_est_kg"]), "masa_v3", bal, dev, lb)).to_string())
    print(f"{time.time()-t0:.0f}s")
