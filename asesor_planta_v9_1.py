"""asesor_planta_v9_1 -- versión de PLANTA del asesor v9 (iteración 12, 2026-09-17).

Sobre `modelo_predictivo_v9.py` (valor directo control → KPI con re-anclaje adaptativo W=100) añade lo que faltaba para usarlo en el horno:

1. **Estado ejecutable al INICIO del escalón** (E12-04, S3'): cierre de F6 para R1-R3 y cierre de F5 para R0 + variables en línea de t−1 +
   acumulados + orden. No usa el ensayo del cierre de t−1 (que llega con el escalón ya en marcha). Conserva 87 % de la evidencia prospectiva
   de v9, con 0 % de inversiones de dirección frente al estado con ensayo.
2. **Lanza coherente**: el aire (protección de lanza) no se toca; el O₂ acompaña al GN (ΔO₂ = 2·ΔGN, piso 0) para conservar el oxígeno libre
   de la lanza (E11-06: el vínculo GN → KPI es por intensidad de fuego, no por exceso de combustible).
3. **Modo exploración (piloto aleatorizado embebido)**: asignación por batch en bloques permutados de 4 a los brazos −δ / +δ (sd del residuo)
   alrededor de la práctica habitual E[a|S], ambos dentro de la caja histórica [P5, P95] del orden. Con efecto lineal, la perturbación
   simétrica tiene costo esperado 0 y convierte al β en un estimador CAUSAL (E12-07: 80 batches → 80 % de potencia para β = −1.1 pp).
4. **Análisis del piloto** (`analizar_piloto`): ITT por brazo con controles, canales (dross, Sn en escoria) y límites secuenciales tipo
   O'Brien-Fleming en los cortes de 40 / 60 / 80 batches.
5. `recomendar_en_vivo`: interfaz de un escalón (dict de estado → setpoints + motivo + efecto físico esperado).

Uso típico:
    ase = entrenar_asesor_planta(df)                       # con todo el histórico disponible
    brazos = asignar_brazos(["B001", "B002", ...], seed=7) # sólo en modo exploración
    rec = recomendar_en_vivo(ase, estado_dict, plan={"tasa_gn_nm3_min": 18.5, "tasa_o2_nm3_min": 40.0, "tasa_aire_nm3_min": 95.0}, brazo=brazos["B001"])
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parent
for _p in (_RAIZ, _RAIZ / "experimentos" / "v8", _RAIZ / "experimentos" / "v9", _RAIZ / "experimentos" / "v10"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v8_lib as L8  # noqa: E402
import modelo_predictivo_v8 as mp8  # noqa: E402
import modelo_predictivo_v9 as mp9  # noqa: E402
import e12_04_lib as E4  # noqa: E402  (estado ejecutable S3')

KPI = mp9.KPI
ESTADO_EJECUTABLE = list(E4.ESTADO_S3P_PLAN)
L8.asegurar_seguras(ESTADO_EJECUTABLE)
CONFIG_PLANTA = dict(mp9.CONFIG_V9, sesgo_reciente_batches=20, delta_exploracion_sd=1.0, bloque_aleatorizacion=4, cortes_secuenciales=(40, 60, 80), alfa=0.05)


def preparar_base_planta(df: pd.DataFrame) -> pd.DataFrame:
    return E4.con_estado_planp(mp8._preparar_base(df))


def _con_estado_ejecutable(f):
    def envoltura(*a, **k):
        previo = mp9.ESTADO_R_V9
        mp9.ESTADO_R_V9 = ESTADO_EJECUTABLE
        try:
            return f(*a, **k)
        finally:
            mp9.ESTADO_R_V9 = previo
    return envoltura


@_con_estado_ejecutable
def entrenar_asesor_planta(df: pd.DataFrame, batches: list | None = None, config: dict | None = None) -> mp9.AsesorV9:
    base = preparar_base_planta(df); bt = L8.construir_batch(df); bt = bt.set_index("Batch") if "Batch" in bt.columns else bt
    bt = bt.sort_values("idx_cronologico"); batches = list(bt.index) if batches is None else batches
    return mp9.entrenar_asesor_v9(base, bt, batches, dict(CONFIG_PLANTA, **(config or {})))


@_con_estado_ejecutable
def replay_planta(df: pd.DataFrame, config: dict | None = None, log=print):
    """Replay prospectivo con el estado EJECUTABLE (lo que el asesor de planta habría recomendado, escalón a escalón, sólo con el pasado)."""
    return mp9.replay_prospectivo(df, dict(CONFIG_PLANTA, **(config or {})), log=log, base=preparar_base_planta(df))


def asignar_brazos(batches: list, seed: int, bloque: int = 4) -> dict:
    """Aleatorización por bloques permutados (mitad −1, mitad +1 en cada bloque). Reproducible: guardar `seed` en el protocolo."""
    rng = np.random.default_rng(seed); out = {}
    for i in range(0, len(batches), bloque):
        b = batches[i:i + bloque]; arms = np.resize([-1, +1], len(b)); rng.shuffle(arms)
        out.update(dict(zip(b, arms.tolist())))
    return out


@_con_estado_ejecutable
def recomendar_en_vivo(ase: mp9.AsesorV9, estado: dict, plan: dict | None = None, brazo: int | None = None, palancas_piloto: tuple = ("gn",)) -> dict:
    """`estado`: dict con ESTADO_EJECUTABLE + 'Batch', 'orden_escalon_fase' (+ acciones históricas opcionales). `plan`: tasas de la receta del
    escalón (GN, O₂, aire) para que O₂ y aire acompañen al GN. `brazo` ∈ {−1, +1, None}: si no es None, modo exploración para `palancas_piloto`."""
    fila = pd.Series({**{c: np.nan for c in ESTADO_EJECUTABLE + list(mp9.PALANCAS_V9.values())}, **estado})
    r = mp9.recomendar_escalon(ase, fila); orden = int(fila["orden_escalon_fase"]); cfg = dict(CONFIG_PLANTA, **ase.config)
    if brazo is not None:
        for p in palancas_piloto:
            sd = ase.sd_res.get((p, orden), 0.0); lo, hi, p90 = ase.caja.get((p, orden), (-np.inf, np.inf, np.inf))
            if p == "carbon" and orden == 0:
                hi = min(hi, p90)
            rec = float(np.clip(r[f"hab__{p}"] + brazo * cfg["delta_exploracion_sd"] * sd, lo, hi))
            frio = p == "gn" and brazo < 0 and np.isfinite(fila.get("temperatura_horno_celsius_prev", np.nan)) and fila["temperatura_horno_celsius_prev"] < ase.T_min
            if frio:
                rec = r[f"hab__{p}"]
            r.update({f"rec__{p}": rec, f"delta__{p}": rec - r[f"hab__{p}"], f"motivo__{p}": "exploracion: horno frio, sin recorte" if frio else f"exploracion: brazo {brazo:+d}"})
    if plan and plan.get("tasa_gn_nm3_min", 0) > 0:
        # aire = setpoint de protección de lanza (dictamen §3): NO se toca; el O2 acompaña al GN para conservar el oxígeno libre de la
        # lanza (CH4 + 2 O2): ΔO2 = 2·ΔGN, con piso 0 (si el piso recorta, la lanza queda algo más oxidante: se reporta).
        d_gn = r["rec__gn"] - plan["tasa_gn_nm3_min"]; o2 = plan.get("tasa_o2_nm3_min", np.nan) + 2.0 * d_gn
        r["rec__o2"] = max(0.0, o2); r["rec__aire"] = plan.get("tasa_aire_nm3_min", np.nan); r["o2_en_piso"] = bool(o2 < 0)
    return r


def limites_obf(cortes: tuple, alfa: float = 0.05) -> dict:
    """Límites z unilaterales tipo O'Brien-Fleming (aprox. z_final·sqrt(N_final/N_k)); z_final corregido ≈ z_{alfa}·1.02 para 3 miradas."""
    from scipy import stats
    zf = stats.norm.isf(alfa) * 1.02
    return {n: zf * np.sqrt(cortes[-1] / n) for n in cortes}


def analizar_piloto(tabla: pd.DataFrame, controles: list | None = None, outcomes: tuple = (KPI, "f_dross", "sn_perdido_escoria_frac")) -> pd.DataFrame:
    """`tabla`: una fila por batch del piloto con 'brazo' (−1/+1), outcomes y controles pre-tratamiento. ITT: outcome ~ brazo + controles (HC3).
    El efecto reportado es por unidad de brazo (= δ sd de GN); para el KPI se espera < 0 (más fuego → menos KPI) y para dross > 0."""
    import statsmodels.api as sm
    controles = [c for c in (controles or mp9.CONTROLES_V9 + ["F_lnirf_F0"]) if c in tabla.columns]
    lim = limites_obf(CONFIG_PLANTA["cortes_secuenciales"], CONFIG_PLANTA["alfa"]); filas = []
    for y in outcomes:
        d = tabla.dropna(subset=["brazo", y] + controles)
        if len(d) < 12:
            continue
        esc = 100.0 if y != KPI else 1.0
        f = sm.OLS(d[y] * esc, sm.add_constant(d[["brazo"] + controles])).fit(cov_type="HC3")
        n_corte = max([n for n in lim if n <= len(d)], default=None)
        filas.append({"outcome": y, "n": len(d), "efecto_por_brazo_pp": f.params["brazo"], "se": f.bse["brazo"], "z": f.tvalues["brazo"],
                      "ic_lo": f.conf_int().loc["brazo", 0], "ic_hi": f.conf_int().loc["brazo", 1],
                      "limite_secuencial_|z|": lim.get(n_corte, np.nan), "cruza_limite": bool(n_corte and abs(f.tvalues["brazo"]) >= lim[n_corte])})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = L8.cargar_df(); ase = entrenar_asesor_planta(df)
    print("beta (pp KPI por unidad de z medio, se, t):", {k: tuple(round(x, 2) for x in v) for k, v in ase.beta.items()}, "| T_min", round(ase.T_min))
    base = preparar_base_planta(df); ult = base[(base.fase_proceso == "Reducción") & (base.orden_escalon_fase <= 3)].sort_values("fecha_inicio").tail(4)
    brazos = asignar_brazos(list(ult.Batch.unique()), seed=7)
    for _, f in ult.iterrows():
        plan = {k: f[k] for k in ("tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min")}
        for brazo in (None, brazos[f.Batch]):
            r = recomendar_en_vivo(ase, f.to_dict(), plan=plan, brazo=brazo)
            print(f"R{r['orden_escalon_fase']} brazo {brazo}: GN hab {r['hab__gn']:.1f} -> rec {r['rec__gn']:.1f} ({r['motivo__gn']}); O2 {r.get('rec__o2', np.nan):.1f}; aire {r.get('rec__aire', np.nan):.1f}; "
                  f"C hab {r['hab__carbon']:.1f} -> rec {r['rec__carbon']:.1f} ({r['motivo__carbon']})")
    print(limites_obf(CONFIG_PLANTA["cortes_secuenciales"]))
