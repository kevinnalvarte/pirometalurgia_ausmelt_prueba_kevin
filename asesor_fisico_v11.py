"""asesor_fisico_v11 -- asesor por escalón basado en el MODELO FÍSICO intra-batch (iteración 19, 2026-09-17).

Pieza predictiva (E18-02, E19-01): con efectos fijos de batch y de orden, y el desvío EXACTO respecto de la receta,

    ln_feo_ret_bt = α_b + γ_o + θ_o·(GN_bt − receta_bt) + δ·(C_bt − receta_C_bt) + f(estado_bt) + ε        (θ_o por orden R0..R3)

θ_o es preciso (p < 1e-5), estable en toda la campaña (θ_R3 entre −0.006 y −0.008 en las 14 re-estimaciones expansivas) y en el fin de campaña.
Agregación exacta escalón → batch:  FeO extra reducido por el desvío de GN  F_b = Σ_t −θ_o(t)·(GN − receta)_t·FeO_inv_{t−1}  [kg].
Validación prospectiva estricta (θ sólo con batches anteriores): F_b predice el FeO total reducido medido por ensayo con pendiente 0.76 [0.45, 1.07]
(t 4.8, p permutación 0.0002; calibración compatible con 1), el dross (+0.19 pp por 100 kg, t 3.0) y, en régimen normal, el KPI (−0.28 pp por 100 kg, t −3.7).
El Sn "extra" que agota el GN es adelanto: no cambia el Sn de la escoria final (E18-06) → el GN por encima de la receta sólo deja el costo del Fe metalizado.

Recomendación: GN* = min(GN previsto, receta) − k·sd_desvío(orden) (k = 0 por defecto: disciplina de receta; k ≤ 1 en piloto), O₂ acompaña (ΔO₂ = 2·ΔGN,
piso P5 del orden), aire intacto; suspendida fuera de régimen (receta de horno frío, GN de R0 > 28.3 Nm³/min, o T previa < 1000 °C). Cada recomendación informa
los kg de FeO que deja de reducir, los kg de Sn que deja de ir a dross (λ = 0.43 [0.18, 0.67] kg Sn/kg FeO, E18-04) y el efecto esperado sobre el KPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent
for _p in (_RAIZ, _RAIZ / "experimentos" / "v16", _RAIZ / "experimentos" / "v9"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import asesor_receta_v10 as AR  # noqa: E402
import e18_02_lib as E  # noqa: E402

TARGET_FEO = "m8_ln_feo_ret_dross50"
LAMBDA_SN_DROSS = (0.43, 0.18, 0.67)          # kg de Sn a dross por kg de FeO reducido (E18-04, DEV; IC95)
PP_KPI_POR_KG_FEO = (-0.0028, -0.0043, -0.0013)  # régimen normal, prospectivo (E19-01)


def _diseno(r: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    r = r.copy()
    for o in range(4):
        r[f"g{o}"] = r["dev__gn"].where(r[E.ORDEN] == o, 0.0)
    return r, [f"g{o}" for o in range(4)] + ["dev__carbon"] + E.STATE


def estimar_theta(r: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """θ_o (efecto de +1 Nm³/min de GN sobre ln de la retención de FeO) por orden, within batch + orden, con IC bootstrap por batch."""
    r, X = _diseno(r); d = r.dropna(subset=[TARGET_FEO] + X).reset_index(drop=True); til = E.demean_two_way(d, [TARGET_FEO] + X, iters=25)
    Xt = til[[f"{x}__tilde" for x in X]].to_numpy(); yt = til[f"{TARGET_FEO}__tilde"].to_numpy(); th = np.linalg.lstsq(Xt, yt, rcond=None)[0][:4]
    rng = np.random.default_rng(seed); b = d["Batch"].to_numpy(); ub = np.unique(b); idx = {k: np.where(b == k)[0] for k in ub}; B = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[k] for k in rng.choice(ub, len(ub))]); B.append(np.linalg.lstsq(Xt[ii], yt[ii], rcond=None)[0][:4])
    B = np.array(B)
    return pd.DataFrame({"orden": range(4), "theta": th, "ci_lo": np.percentile(B, 2.5, 0), "ci_hi": np.percentile(B, 97.5, 0), "t": th / B.std(0)})


def recomendar(theta: pd.DataFrame, orden: int, receta_gn: float, receta_o2: float, gn_previsto: float, feo_inv_prev_kg: float, receta_gn_r0: float,
               t_horno_prev: float, k_sd: float = 0.0, sd_desvio: float = 0.0) -> dict:
    base = AR.regla_disciplina_gn(orden, receta_gn, receta_o2, gn_previsto, receta_gn_r0, t_horno_prev)
    if base["motivo"].startswith("fuera"):
        return {**base, "feo_evitado_kg": 0.0, "sn_no_a_dross_kg": 0.0, "kpi_pp": 0.0}
    gn = base["gn"] - k_sd * sd_desvio; d_gn = gn - gn_previsto
    o2 = max(AR.PISO_O2_NM3_MIN.get(int(orden), 0.0), receta_o2 + 2.0 * d_gn) if d_gn < 0 else receta_o2
    th = theta.set_index("orden").loc[int(orden)]
    feo = [v * d_gn * feo_inv_prev_kg for v in (th.theta, th.ci_hi, th.ci_lo)]          # kg de FeO que se dejan de reducir (θ < 0, d_gn < 0 → > 0)
    return {"gn": gn, "o2": o2, "motivo": base["motivo"] if k_sd == 0 else f"receta − {k_sd:g} sd", "feo_evitado_kg": feo[0], "feo_evitado_ic": (min(feo), max(feo)),
            "sn_no_a_dross_kg": LAMBDA_SN_DROSS[0] * feo[0], "kpi_pp": -PP_KPI_POR_KG_FEO[0] * feo[0]}


def replay_regla(k_sd: float = 0.0) -> pd.DataFrame:
    """Valor físico de la regla en la historia: por batch, FeO que se habría dejado de reducir y su traducción a Sn y KPI (θ de toda la muestra:
    sólo para dimensionar; la validez prospectiva de θ está en E19-01)."""
    r, bt = E.cargar(); th = estimar_theta(r, n_boot=100).set_index("orden").theta
    sd = r.groupby(E.ORDEN)["dev__gn"].std(); r0 = r[r[E.ORDEN] == 0].set_index("Batch")["plan__gn"]
    normal = (r["Batch"].map(r0) <= AR.RECETA_GN_R0_HORNO_FRIO) & ~(r["temperatura_horno_celsius_prev"] < AR.T_PREV_MIN_REGLA)
    gn_rec = np.minimum(r["tasa_gn_nm3_min"], r["plan__gn"]) - k_sd * r[E.ORDEN].map(sd); d_gn = (gn_rec - r["tasa_gn_nm3_min"]).where(normal, 0.0)
    r["feo_evitado_kg"] = r[E.ORDEN].map(th) * d_gn * r["m6_feo_inv_kg_prev"]
    pb = r.groupby("Batch").agg(feo_evitado_kg=("feo_evitado_kg", "sum"), escalones_en_regimen=("feo_evitado_kg", lambda s: int((normal[s.index]).sum())))
    pb["sn_no_a_dross_kg"] = LAMBDA_SN_DROSS[0] * pb.feo_evitado_kg; pb["kpi_pp"] = -PP_KPI_POR_KG_FEO[0] * pb.feo_evitado_kg
    return bt[["idx_cronologico", "es_dev"]].join(pb).sort_values("idx_cronologico")


if __name__ == "__main__":
    r, bt = E.cargar(); th = estimar_theta(r); pd.set_option("display.width", 200); print(th.round(5).to_string(index=False))
    for k in (0.0, 0.5, 1.0):
        pb = replay_regla(k); g = pb[pb.escalones_en_regimen > 0]
        print(f"regla k={k:g} sd: batches en régimen {len(g)}/{len(pb)} | FeO evitado {g.feo_evitado_kg.mean():.0f} kg/batch (P10 {g.feo_evitado_kg.quantile(.1):.0f}, P90 {g.feo_evitado_kg.quantile(.9):.0f})"
              f" | Sn que no va a dross {g.sn_no_a_dross_kg.mean():.0f} kg [{LAMBDA_SN_DROSS[1]*g.feo_evitado_kg.mean():.0f}, {LAMBDA_SN_DROSS[2]*g.feo_evitado_kg.mean():.0f}]"
              f" | KPI {g.kpi_pp.mean():+.2f} pp [{-PP_KPI_POR_KG_FEO[2]*g.feo_evitado_kg.mean():+.2f}, {-PP_KPI_POR_KG_FEO[1]*g.feo_evitado_kg.mean():+.2f}]")
    ej = r[(r.Batch == r.Batch.iloc[len(r) // 2])].sort_values(E.ORDEN)
    for _, f in ej.iterrows():
        print(int(f[E.ORDEN]), recomendar(th, int(f[E.ORDEN]), f.plan__gn, f.plan__o2, f.tasa_gn_nm3_min, f.m6_feo_inv_kg_prev, ej.plan__gn.iloc[0], f.temperatura_horno_celsius_prev))
