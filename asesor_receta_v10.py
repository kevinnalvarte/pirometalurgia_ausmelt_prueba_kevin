"""asesor_receta_v10 -- asesor prescriptivo por escalón anclado en la RECETA de planta (iteración 15, 2026-09-17).

Hallazgo que lo motiva (E15-01): la hoja "Alimentación" del Excel trae la receta por escalón de Reducción (GN, O₂, aire, carbón; R01-R04). La
exposición deja de necesitar un modelo de "práctica habitual": es el desvío EXACTO entre lo ejecutado y la receta,

    d_jt = (a_jt − receta_jt) / sd_pasado(j, orden)         x_bj = media_{t ∈ R0..R3} d_jt

conocido y ejecutable al inicio del escalón (la receta existe antes del batch). STATE = receta del escalón + orden (+ T horno previa para el
resguardo térmico); CONTROL = GN y carbón ejecutados; O₂ acompaña al GN (ΔO₂ = 2·ΔGN, piso 0) y el aire no se toca.

Valor: KPI_b = controles + β_G·x_bG + β_C·x_bC, OLS sobre los últimos W = 100 batches, re-anclado cada 10, con contracción James-Stein
β̃ = β̂·max(0, 1 − 1/t²) (E13-01). Recomendación: a* = receta + min(1, |t|/2)·signo(β̃)·sd_desvío, dentro de [P5, P95] del orden de lo ejecutado en
el pasado, carbón R0 ≤ P90, sin recorte de GN con T_prev < P10. TODO se estima sólo con batches anteriores (media/sd del desvío incluidas).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent
for _p in (_RAIZ, _RAIZ / "experimentos" / "v8", _RAIZ / "experimentos" / "v9"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v8_lib as L8  # noqa: E402

EXCEL = _RAIZ / "Datos Lingo smelter fase II.xlsx"
CACHE_RECETA = _RAIZ / "experimentos" / "cache" / "receta_reduccion.pkl"
KPI = L8.KPI_PRINCIPAL
CONTROLES = [c for c in L8.CONTROLES_BATCH if c != "espesor_ladrillo_norm_mm"]
PALANCAS = {"gn": ("GN (Nm3/h)-R0{}", "tasa_gn_nm3_min"), "carbon": ("Carbón (kg/h)-R0{}", "tasa_feed_Carbon_kg_min"),
            "o2": ("O2 (Nm3/h)-R0{}", "tasa_o2_nm3_min"), "aire": ("Aire (Nm3/h)-R0{}", "tasa_aire_nm3_min")}
OPTIMIZABLES = ("gn", "carbon")
CONFIG = dict(W=100, paso=10, min_hist=40, unidades="sd",   # "sd": desvío / sd del PASADO por orden (calibración 0.64 con JS); "fisicas" (min_hist=0): Nm³/min, kg/min, sin
              # constantes estimadas (E15-02: evidencia igual de fuerte, calibración 0.49). La sd del pasado fija además el tamaño del paso
              delta_max_sd=1.0, t_min=1.0, t_pleno=2.0, contraccion="js", pct_caja=(0.05, 0.95), techo_carbon_R0_pct=0.90,
              pct_T_min_recorte_gn=0.10)


def cargar_receta() -> pd.DataFrame:
    """Receta por (Batch, orden de Reducción 0-3) en unidades por minuto: plan__gn, plan__carbon, plan__o2, plan__aire."""
    if CACHE_RECETA.exists():
        return pd.read_pickle(CACHE_RECETA)
    a = pd.read_excel(EXCEL, sheet_name="Alimentación", header=1); a["Batch"] = a["Batch"].astype(str); a = a.set_index("Batch")
    filas = []
    for k in range(4):
        f = pd.DataFrame({f"plan__{p}": a[pat.format(k + 1)] / 60.0 for p, (pat, _) in PALANCAS.items()}); f["orden_escalon_fase"] = k
        filas.append(f.reset_index())
    rec = pd.concat(filas, ignore_index=True); CACHE_RECETA.parent.mkdir(exist_ok=True); rec.to_pickle(CACHE_RECETA)
    return rec


def escalones_con_receta(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = L8.cargar_df() if df is None else df
    r = df[(df["fase_proceso"] == "Reducción") & (df["orden_escalon_fase"] <= 3)].merge(cargar_receta(), on=["Batch", "orden_escalon_fase"], how="left")
    for p, (_, col) in PALANCAS.items():
        r[f"dev__{p}"] = r[col] - r[f"plan__{p}"]
    bt = L8.construir_batch(df); bt = bt.set_index("Batch") if "Batch" in bt.columns else bt
    return r, bt.sort_values("idx_cronologico")


def _beta(bt: pd.DataFrame, X: list[str], contraccion: str | None) -> dict:
    import statsmodels.api as sm
    d = bt[[KPI] + X + CONTROLES].dropna(); f = sm.OLS(d[KPI], sm.add_constant(d[X + CONTROLES])).fit(cov_type="HC3"); out = {}
    for x in X:
        b, se, t = float(f.params[x]), float(f.bse[x]), float(f.tvalues[x])
        out[x] = (b * max(0.0, 1 - 1 / t ** 2) if contraccion == "js" and abs(t) > 1e-9 else b, se, t, b)
    return out


def replay_prospectivo(df: pd.DataFrame | None = None, config: dict | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cada bloque de `paso` batches se recomienda con parámetros (sd del desvío por orden, cajas, β) estimados SÓLO con batches anteriores."""
    cfg = dict(CONFIG, **(config or {})); r, bt = escalones_con_receta(df); orden = list(bt.index); pos = r["Batch"].map({b: i for i, b in enumerate(orden)})
    filas = []; z_hist = {p: pd.Series(np.nan, index=r.index) for p in OPTIMIZABLES}
    # historia previa al primer bloque puntuado: exposiciones en crudo o con la sd de los primeros batches (no intervienen en ningún puntaje previo)
    for s in range(cfg["min_hist"] if cfg["unidades"] != "fisicas" else 0, len(orden), cfg["paso"]):
        pas, blo = r[pos < s], r[(pos >= s) & (pos < s + cfg["paso"])]
        sd = {p: (pas if len(pas) else blo).groupby("orden_escalon_fase")[f"dev__{p}"].std() for p in OPTIMIZABLES}
        for p in OPTIMIZABLES:                                   # exposición del bloque con la sd del pasado (sin centrar: 0 = receta)
            z_hist[p].loc[blo.index] = blo[f"dev__{p}"] / (1.0 if cfg["unidades"] == "fisicas" else blo["orden_escalon_fase"].map(sd[p]))
        if s < cfg["min_hist"] + cfg["W"]:   # aún sin ventana completa
            continue
        ventana = orden[s - cfg["W"]:s]
        expo = pd.DataFrame({f"x_{p}": z_hist[p].groupby(r["Batch"]).mean() for p in OPTIMIZABLES})
        beta = _beta(bt.loc[ventana].join(expo), [f"x_{p}" for p in OPTIMIZABLES], cfg["contraccion"])
        T_min = pas["temperatura_horno_celsius_prev"].quantile(cfg["pct_T_min_recorte_gn"])
        for _, f in blo.iterrows():
            o = int(f["orden_escalon_fase"]); out = {"Batch": f["Batch"], "orden_escalon_fase": o, "inicio_bloque": s}
            for p in OPTIMIZABLES:
                col = PALANCAS[p][1]; b, se, t, b_crudo = beta[f"x_{p}"]; sdp = float(sd[p].get(o, np.nan)); g = pas.loc[pas["orden_escalon_fase"] == o, col]
                lo, hi = g.quantile(cfg["pct_caja"][0]), g.quantile(cfg["pct_caja"][1]); hi = min(hi, g.quantile(cfg["techo_carbon_R0_pct"])) if (p == "carbon" and o == 0) else hi
                direccion = np.sign(b_crudo) if abs(t) >= cfg["t_min"] else 0.0; motivo = "recomendar" if direccion else "ejecutar la receta"
                if p == "gn" and direccion < 0 and f["temperatura_horno_celsius_prev"] < T_min:
                    direccion, motivo = 0.0, "ejecutar la receta: horno frio"
                rec = float(np.clip(f[f"plan__{p}"] + cfg["delta_max_sd"] * min(1, abs(t) / cfg["t_pleno"]) * direccion * sdp, min(lo, f[f"plan__{p}"]), max(hi, f[f"plan__{p}"])))
                out.update({f"receta__{p}": f[f"plan__{p}"], f"hist__{p}": f[col], f"rec__{p}": rec, f"z_hist__{p}": f[f"dev__{p}"] / sdp, f"z_rec__{p}": (rec - f[f"plan__{p}"]) / sdp, f"dev_hist__{p}": f[f"dev__{p}"], f"dev_rec__{p}": rec - f[f"plan__{p}"],
                            f"beta__{p}": b, f"t__{p}": t, f"motivo__{p}": motivo, f"valor_hist_pp__{p}": b * f[f"dev__{p}"] / (1.0 if cfg["unidades"] == "fisicas" else sdp), f"valor_rec_pp__{p}": b * (rec - f[f"plan__{p}"]) / (1.0 if cfg["unidades"] == "fisicas" else sdp)})
            filas.append(out)
        log(f"bloque {s}: " + "  ".join(f"beta_{p} {beta[f'x_{p}'][3]:+.2f} (t {beta[f'x_{p}'][2]:+.1f})" for p in OPTIMIZABLES))
    esc = pd.DataFrame(filas)
    pb = esc.groupby("Batch")[[c for c in esc.columns if c.startswith(("valor_", "z_", "dev_", "beta__", "t__"))]].mean()
    pb["valor_hist_pp"] = pb[[f"valor_hist_pp__{p}" for p in OPTIMIZABLES]].sum(axis=1); pb["valor_rec_pp"] = pb[[f"valor_rec_pp__{p}" for p in OPTIMIZABLES]].sum(axis=1)
    pb["valor_receta_pp"] = -pb["valor_hist_pp"]      # ganancia atribuida a haber ejecutado exactamente la receta (dev = 0) en vez de lo ejecutado
    return esc, bt.join(pb, how="inner")


if __name__ == "__main__":
    import time
    import modelo_predictivo_v9 as mp9
    import statsmodels.api as sm
    t0 = time.time(); SAL = _RAIZ / "experimentos" / "v13"
    for tag, cfg in (("js", {}), ("ols", {"contraccion": None})):
        esc, pb = replay_prospectivo(config=cfg, log=lambda *a: None); esc.to_csv(SAL / f"e15_03_escalones_{tag}.csv", index=False); pb.to_csv(SAL / f"e15_03_por_batch_{tag}.csv")
        ev = mp9.evidencia_prospectiva(pb); ev.insert(0, "contraccion", tag); ev.to_csv(SAL / f"e15_03_evidencia_{tag}.csv", index=False)
        pd.set_option("display.width", 250); print(ev.round(4).to_string(), f"[{time.time()-t0:.0f}s]")
        d = pb.dropna(subset=["valor_hist_pp", "valor_rec_pp", KPI] + CONTROLES).reset_index(drop=True); rng = np.random.default_rng(0); nb = len(d) // 10; G = []
        for _ in range(2000):
            idx = np.concatenate([np.arange(b * 10, b * 10 + 10) for b in rng.integers(0, nb, nb)]); dd = d.iloc[idx]
            sl = sm.OLS(dd[KPI], sm.add_constant(dd[["valor_hist_pp"] + CONTROLES])).fit().params["valor_hist_pp"]; G.append((sl, sl * dd.valor_rec_pp.mean(), sl * dd.valor_receta_pp.mean()))
        G = np.array(G)
        for j, nm in ((1, "seguir la recomendacion"), (2, "ejecutar exactamente la receta")):
            print(f"{tag} | {nm}: ganancia calibrada {np.median(G[:, j]):+.2f} pp/batch IC95 [{np.percentile(G[:, j], 2.5):+.2f}, {np.percentile(G[:, j], 97.5):+.2f}] P(>0) {np.mean(G[:, j] > 0):.3f}")
        print(f"{tag} | pendiente IC95 [{np.percentile(G[:, 0], 2.5):.2f}, {np.percentile(G[:, 0], 97.5):.2f}]")


# =============================================================================
# Regla de disciplina (iteracion 18): "GN nunca por encima de la receta" en regimen normal. No necesita modelo ni re-anclaje.
# =============================================================================
PISO_O2_NM3_MIN = {0: 22.0, 1: 16.2, 2: 5.5, 3: 3.1}      # P5 historico por orden (E18-03 C4): con piso 0 el 23.6 % de los casos quedaba bajo el P1
RECETA_GN_R0_HORNO_FRIO = 28.4                             # Nm3/min: por encima, planta ya opera la "receta de horno frio" (fin de campana) -> regla suspendida
T_PREV_MIN_REGLA = 960.0                                   # grados C ≈ P10 histórico de T previa (resguardo de horno frío de v9/v10). E18-03 A1: el efecto persiste por debajo de 1000 °C, así que el conmutador de régimen es la receta de horno frío que declara planta


def regla_disciplina_gn(orden: int, receta_gn: float, receta_o2: float, gn_previsto: float, receta_gn_r0: float, t_horno_prev: float) -> dict:
    """Setpoint de GN para un escalon de Reduccion segun la regla de disciplina. `gn_previsto` = lo que el operador piensa ejecutar.
    Regimen normal (receta de GN de R0 <= 28.3 y T previa >= 1000 C): GN = min(gn_previsto, receta). Fuera de regimen: sin intervencion.
    El O2 acompana el recorte (dO2 = 2 dGN) con piso P5 historico del orden; el aire no se toca."""
    normal = (receta_gn_r0 <= RECETA_GN_R0_HORNO_FRIO) and (not np.isfinite(t_horno_prev) or t_horno_prev >= T_PREV_MIN_REGLA)
    if not normal:
        return {"gn": gn_previsto, "o2": receta_o2, "motivo": "fuera de regimen (receta de horno frio o T previa < 960 C): sin intervencion"}
    gn = min(gn_previsto, receta_gn); o2 = max(PISO_O2_NM3_MIN.get(int(orden), 0.0), receta_o2 + 2.0 * (gn - gn_previsto)) if gn < gn_previsto else receta_o2
    return {"gn": gn, "o2": o2, "motivo": "GN limitado a la receta" if gn < gn_previsto else "dentro de la receta"}
