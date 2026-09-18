"""modelo_predictivo_v9 -- asesor prescriptivo por escalón v9 (iteración 11, 2026-09-17).

Qué cambia respecto de v8 (`modelo_predictivo_v8.py`, que se conserva para los modelos de escalón, cajas y soporte):

1. El VALOR de cada CONTROL ya no sale de un objetivo J = K·(ln_sn_dep + w·ln_feo_ret) anclado en targets agregados (su ventaja no predice
   el KPI de batches no vistos: E10-05), sino del vínculo DIRECTO y fuera de muestra entre la desviación de la acción respecto de la práctica
   habitual dado el estado, z = (a − E[a|S])/sd, y el KPI del batch (E11-01: GN_R −1.5 pp/sd, q-BH 0.008, RV 17 %, placebos nulos; canal dross).
2. Ese valor DERIVA con la campaña (E11-06c: β_GN de −3 a +1, β_C de −1 a +4 pp por unidad de z medio; el β de DEV aplicado al lockbox falla) →
   RE-ANCLAJE ADAPTATIVO: β se estima con los últimos W batches (ventana móvil) y sólo se usa de forma prospectiva.
3. La recomendación es un setpoint relativo a la PRÁCTICA HABITUAL E[a|S] (no a la acción histórica del operador): a* = E[a|S] + δ·dir·sd_res,
   con dir = signo(β̂) si |t| ≥ t_min (si no, "mantener la práctica habitual"), δ = δ_max·min(1, |t|/2), dentro de la caja por orden de v8
   ([P5, P95] del orden, carbón R0 ≤ P90) y con salvaguarda térmica (no recortar GN con el horno por debajo del P10 de T).
4. Los modelos de escalón de v8 (PLM: Cx_sn → agotamiento, GN → agotamiento y retención de FeO) se usan para EXPLICAR la recomendación (efecto
   físico esperado por escalón, agregable Σ_t a fase y batch), no para valorarla.

STATE = `ESTADO_R` (k15 de v8 + orden + T horno previa); CONTROL optimizable = carbón (kg/min) y GN (Nm³/min) en Reducción; O₂ y aire
siguen al GN con la estequiometría de lanza habitual (λ constante); Fusión = receta (E10-06, E11-05).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent
for _p in (_RAIZ, _RAIZ / "experimentos" / "v8", _RAIZ / "experimentos" / "v9"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v8_lib as L8  # noqa: E402
import modelo_predictivo_v8 as mp8  # noqa: E402

SEED = 42
KPI = L8.KPI_PRINCIPAL
KG_SN_POR_PP_KPI = mp8.KG_SN_POR_PP_KPI
PALANCAS_V9 = {"gn": "tasa_gn_nm3_min", "carbon": "tasa_feed_Carbon_kg_min"}
ESTADO_R_V9 = list(mp8.ESTADO_R_V8) + ["orden_escalon_fase", "temperatura_horno_celsius_prev"]
CONTROLES_V9 = [c for c in L8.CONTROLES_BATCH if c != "espesor_ladrillo_norm_mm"]   # espesor ≡ idx cronológico (reloj)
CONFIG_V9 = dict(W=100, min_hist=100, paso=10, delta_max_sd=1.0, t_min=1.0, t_pleno=2.0, pct_caja=(0.05, 0.95), techo_carbon_R0_pct=0.90,
                 pct_T_min_recorte_gn=0.10, n_splits=5,
                 contraccion="js", sesgo_reciente_batches=0)   # >0: E[a|S] se corrige con el residuo medio (por orden) de los últimos k batches (deriva de receta)   # E13-01: beta~ = beta^·max(0, 1 − 1/t²) (James-Stein, parte positiva): corrige la dilución → puntaje calibrado


@dataclass
class AsesorV9:
    politica: dict = field(default_factory=dict)      # palanca -> HGB E[a|S]
    sd_res: dict = field(default_factory=dict)        # (palanca, orden) -> sd del residuo fuera de muestra
    beta: dict = field(default_factory=dict)          # palanca -> (beta pp KPI por unidad de z medio, se, t)
    caja: dict = field(default_factory=dict)          # (palanca, orden) -> (p5, p95, p90)
    T_min: float = np.nan
    sesgo: dict = field(default_factory=dict)         # (palanca, orden) -> residuo medio reciente de la práctica habitual (deriva de receta)
    theta: dict = field(default_factory=dict)         # efectos físicos de escalón (por unidad de palanca) para explicar
    n_train: int = 0
    n_ventana: int = 0
    config: dict = field(default_factory=lambda: dict(CONFIG_V9))


def filas_reduccion(df_base: pd.DataFrame) -> pd.DataFrame:
    d = df_base[(df_base["fase_proceso"] == "Reducción") & (df_base["orden_escalon_fase"] <= 3)]
    return d.sort_values(["fecha_inicio"]).reset_index(drop=True)


def _z_crossfit(d: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Residuo fuera de muestra (GroupKFold por batch) de cada palanca respecto de E[a|S], y su sd por orden."""
    from sklearn.model_selection import GroupKFold
    z = pd.DataFrame(index=d.index); sd = {}
    for p, col in PALANCAS_V9.items():
        ok = d[col].notna().to_numpy(); idx = np.where(ok)[0]; ah = np.full(len(d), np.nan)
        for tr, te in GroupKFold(n_splits).split(idx, groups=d["Batch"].to_numpy()[idx]):
            ah[idx[te]] = L8.hgb_nuisance(seed).fit(d[ESTADO_R_V9].iloc[idx[tr]], d[col].iloc[idx[tr]]).predict(d[ESTADO_R_V9].iloc[idx[te]])
        r = d[col].to_numpy() - ah
        s = pd.Series(r).groupby(d["orden_escalon_fase"].to_numpy()).std()
        for o, v in s.items():
            sd[(p, int(o))] = float(v)
        z[p] = r / d["orden_escalon_fase"].map(s).to_numpy()
    return z, sd


def _beta_ventana(bt: pd.DataFrame, X: list[str]) -> dict:
    import statsmodels.api as sm
    d = bt[[KPI] + X + CONTROLES_V9].dropna()
    f = sm.OLS(d[KPI], sm.add_constant(d[X + CONTROLES_V9])).fit(cov_type="HC3")
    return {x: (float(f.params[x]), float(f.bse[x]), float(f.tvalues[x])) for x in X}, len(d)


def entrenar_asesor_v9(df_base: pd.DataFrame, batch_tab: pd.DataFrame, batches_train: list, config: dict | None = None, seed: int = SEED) -> AsesorV9:
    """`batches_train` en orden cronológico (el pasado). E[a|S] y cajas con todo el pasado; β con los últimos W batches."""
    cfg = dict(CONFIG_V9, **(config or {}))
    d = filas_reduccion(df_base[df_base["Batch"].isin(batches_train)])
    ase = AsesorV9(config=cfg, n_train=len(batches_train))
    z, ase.sd_res = _z_crossfit(d, cfg["n_splits"], seed)
    for p, col in PALANCAS_V9.items():
        ok = d[col].notna()
        ase.politica[p] = L8.hgb_nuisance(seed).fit(d.loc[ok, ESTADO_R_V9], d.loc[ok, col])
        for o, g in d.groupby("orden_escalon_fase"):
            x = g[col].dropna()
            ase.caja[(p, int(o))] = (float(x.quantile(cfg["pct_caja"][0])), float(x.quantile(cfg["pct_caja"][1])), float(x.quantile(cfg["techo_carbon_R0_pct"])))
    k = int(cfg.get("sesgo_reciente_batches") or 0)
    if k > 0:
        rec_b = set(list(batches_train)[-k:]); m = d["Batch"].isin(rec_b).to_numpy()
        for p, col in PALANCAS_V9.items():
            for o in sorted(d["orden_escalon_fase"].unique()):
                mm = m & (d["orden_escalon_fase"].to_numpy() == o)
                ase.sesgo[(p, int(o))] = float(np.nanmean(z[p].to_numpy()[mm]) * ase.sd_res[(p, int(o))]) if mm.any() else 0.0
        # las exposiciones del pasado también se centran con su propio sesgo móvil (media de los k batches ANTERIORES, por orden)
        for p in PALANCAS_V9:
            zb = z[p].groupby([d["orden_escalon_fase"].to_numpy(), d["Batch"].to_numpy()], sort=False).mean().unstack(0)
            zb = zb.reindex([b for b in batches_train if b in zb.index])
            mov = zb.shift(1).rolling(k, min_periods=5).mean().fillna(0.0)
            corr = mov.stack().rename("c").reset_index(); corr.columns = ["Batch", "orden", "c"]
            mp_ = {(r.Batch, r.orden): r.c for r in corr.itertuples()}
            z[p] = z[p] - np.array([mp_.get((b, o), 0.0) for b, o in zip(d["Batch"].to_numpy(), d["orden_escalon_fase"].to_numpy())])
    ase.T_min = float(d["temperatura_horno_celsius_prev"].quantile(cfg["pct_T_min_recorte_gn"]))
    expo = z.groupby(d["Batch"].to_numpy()).mean().rename(columns={p: f"x_{p}" for p in PALANCAS_V9})
    ventana = list(batches_train)[-cfg["W"]:] if cfg["W"] else list(batches_train)
    bt = batch_tab.loc[batch_tab.index.intersection(ventana)].join(expo, how="inner")
    betas, ase.n_ventana = _beta_ventana(bt, [f"x_{p}" for p in PALANCAS_V9])
    ase.beta = {p: betas[f"x_{p}"] for p in PALANCAS_V9}
    # efectos físicos de escalón (theta libre, residuo sobre residuo) para explicar la recomendación
    for clave, tg, A in (("sn_dep", "m6_ln_sn_dep", ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min"]), ("feo_ret", "m8_ln_feo_ret_dross50", ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min"])):
        dd = d.dropna(subset=mp8.ESTADO_R_V8 + A + [tg]).reset_index(drop=True)
        if len(dd) > 200:
            t, _ = L8.theta_dual(dd, mp8.ESTADO_R_V8, A, tg, None, n_boot=2, seed=seed)
            ase.theta[clave] = dict(zip(A, (t["theta_libre_1sd"] / t["sd_palanca"]).to_numpy()))
    return ase


def recomendar_escalon(ase: AsesorV9, fila: pd.Series) -> dict:
    """Recomendación para un escalón de Reducción a partir SÓLO del estado al inicio del escalón."""
    cfg = ase.config; orden = int(fila["orden_escalon_fase"]); S = pd.DataFrame([fila[ESTADO_R_V9].astype(float)])
    out = {"Batch": fila["Batch"], "orden_escalon_fase": orden, "T_prev": fila.get("temperatura_horno_celsius_prev")}
    d_sn = d_feo = 0.0
    for p, col in PALANCAS_V9.items():
        hab = float(ase.politica[p].predict(S)[0]) + ase.sesgo.get((p, orden), 0.0); b, se, t = ase.beta[p]; sd = ase.sd_res.get((p, orden), np.nan)
        if cfg.get("contraccion") == "js":
            b = b * max(0.0, 1.0 - 1.0 / t ** 2) if abs(t) > 1e-9 else 0.0
        direccion = float(np.sign(b)) if abs(t) >= cfg["t_min"] else 0.0
        motivo = "recomendar" if direccion else "mantener: beta no distinguible de 0"
        if p == "gn" and direccion < 0 and np.isfinite(fila.get("temperatura_horno_celsius_prev", np.nan)) and fila["temperatura_horno_celsius_prev"] < ase.T_min:
            direccion, motivo = 0.0, "mantener: horno frio (T_prev < P10), no recortar GN"
        delta = cfg["delta_max_sd"] * min(1.0, abs(t) / cfg["t_pleno"]) * direccion * (sd if np.isfinite(sd) else 0.0)
        lo, hi, p90 = ase.caja.get((p, orden), (-np.inf, np.inf, np.inf))
        if p == "carbon" and orden == 0:
            hi = min(hi, p90)
        rec = float(np.clip(hab + delta, min(lo, hab), max(hi, hab)))
        hist = float(fila[col]) if pd.notna(fila[col]) else np.nan
        out.update({f"hist__{p}": hist, f"hab__{p}": hab, f"rec__{p}": rec, f"delta__{p}": rec - hab, f"z_hist__{p}": (hist - hab) / sd if sd else np.nan,
                    f"beta__{p}": b, f"t__{p}": t, f"motivo__{p}": motivo,
                    f"valor_hist_pp__{p}": b * ((hist - hab) / sd) if sd else np.nan, f"valor_rec_pp__{p}": b * ((rec - hab) / sd) if sd else np.nan})
        d_sn += ase.theta.get("sn_dep", {}).get(col, 0.0) * (rec - hab); d_feo += ase.theta.get("feo_ret", {}).get(col, 0.0) * (rec - hab)
    out["efecto_ln_sn_dep"] = d_sn; out["efecto_ln_feo_ret"] = d_feo   # efecto físico esperado de pasar de la práctica habitual a la recomendada
    return out


def replay_prospectivo(df: pd.DataFrame, config: dict | None = None, log=print, seed: int = SEED, base: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula el paso del tiempo: cada bloque de `paso` batches se recomienda, escalón a escalón, con un asesor entrenado SÓLO con los
    batches anteriores (≥ min_hist). Devuelve (escalones, por_batch). El valor por batch (`valor_hist_pp`) = Σ_palanca β̂·media_t z: lo que
    el asesor atribuye a lo que el operador hizo; `valor_rec_pp` = ídem para la recomendación (ganancia esperada vs práctica habitual)."""
    cfg = dict(CONFIG_V9, **(config or {}))
    base = mp8._preparar_base(df) if base is None else base; bt = L8.construir_batch(df); bt = bt.set_index("Batch") if "Batch" in bt.columns else bt
    bt = bt.sort_values("idx_cronologico"); orden = list(bt.index)
    esc = []
    for s in range(cfg["min_hist"], len(orden), cfg["paso"]):
        ase = entrenar_asesor_v9(base, bt, orden[:s], cfg, seed)
        bloque = orden[s:s + cfg["paso"]]
        d = filas_reduccion(base[base["Batch"].isin(bloque)])
        for _, fila in d.iterrows():
            if fila[[PALANCAS_V9["gn"], PALANCAS_V9["carbon"]]].isna().any():
                continue
            r = recomendar_escalon(ase, fila); r["inicio_bloque"] = s; esc.append(r)
        log(f"bloque {s}/{len(orden)}  beta_gn {ase.beta['gn'][0]:+.2f} (t {ase.beta['gn'][2]:+.1f})  beta_C {ase.beta['carbon'][0]:+.2f} (t {ase.beta['carbon'][2]:+.1f})  n_ventana {ase.n_ventana}")
    esc = pd.DataFrame(esc)
    agg = {f"{k}__{p}": "mean" for p in PALANCAS_V9 for k in ("valor_hist_pp", "valor_rec_pp", "z_hist", "delta", "beta", "t")}
    agg.update({"efecto_ln_sn_dep": "sum", "efecto_ln_feo_ret": "sum"})
    pb = esc.groupby("Batch").agg(agg)
    pb["valor_hist_pp"] = pb[[f"valor_hist_pp__{p}" for p in PALANCAS_V9]].sum(axis=1)
    pb["valor_rec_pp"] = pb[[f"valor_rec_pp__{p}" for p in PALANCAS_V9]].sum(axis=1)
    pb = bt.join(pb, how="inner")
    return esc, pb


def evidencia_prospectiva(pb: pd.DataFrame, col: str = "valor_hist_pp", n_perm: int = 5000, seed: int = 0) -> pd.DataFrame:
    """KPI ~ valor atribuido a lo que el operador hizo (prospectivo) + controles. HC3, Spearman parcial, permutación dentro de bloques de 20."""
    import statsmodels.api as sm
    from scipy import stats
    filas = []
    for nm, d in (("todo", pb), ("DEV", pb[pb.es_dev]), ("lockbox", pb[~pb.es_dev])):
        d = d.dropna(subset=[col, KPI] + CONTROLES_V9)
        if len(d) < 30 or d[col].std() < 1e-9:
            continue
        f = sm.OLS(d[KPI], sm.add_constant(d[[col] + CONTROLES_V9])).fit(cov_type="HC3")
        ry = sm.OLS(d[KPI], sm.add_constant(d[CONTROLES_V9])).fit().resid; rs = sm.OLS(d[col], sm.add_constant(d[CONTROLES_V9])).fit().resid
        rho, p_rho = stats.spearmanr(ry, rs); r_obs = np.corrcoef(ry, rs)[0, 1]
        rng = np.random.default_rng(seed); blo = np.arange(len(d)) // 20; nul = []
        for _ in range(n_perm):
            perm = np.concatenate([rng.permutation(np.where(blo == b)[0]) for b in np.unique(blo)])
            nul.append(np.corrcoef(ry.to_numpy(), rs.to_numpy()[perm])[0, 1])
        q = pd.qcut(d[col].rank(method="first"), 3, labels=False)
        filas.append({"muestra": nm, "n": len(d), "pendiente": f.params[col], "se": f.bse[col], "t": f.tvalues[col], "p_unilateral": float(stats.norm.sf(f.tvalues[col])),
                      "spearman_parcial": rho, "p_spearman": p_rho, "p_perm_bloques": float((np.sum(np.array(nul) >= r_obs) + 1) / (n_perm + 1)),
                      "kpi_res_tercio_bajo": ry[q == 0].mean(), "kpi_res_tercio_medio": ry[q == 1].mean(), "kpi_res_tercio_alto": ry[q == 2].mean(),
                      "valor_rec_medio_pp": d["valor_rec_pp"].mean() if "valor_rec_pp" in d else np.nan})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    import time
    t0 = time.time(); SAL = _RAIZ / "experimentos" / "v9"
    df = L8.cargar_df()
    esc, pb = replay_prospectivo(df, log=lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True))
    esc.to_csv(SAL / "e11_09_replay_escalones.csv", index=False); pb.to_csv(SAL / "e11_09_replay_por_batch.csv")
    ev = evidencia_prospectiva(pb); ev.to_csv(SAL / "e11_09_evidencia_prospectiva.csv", index=False)
    pd.set_option("display.width", 250); print(ev.round(4).to_string())
