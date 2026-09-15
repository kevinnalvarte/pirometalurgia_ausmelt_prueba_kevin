"""Modelo prescriptivo v4 de Lingo Smelter (iteracion 4, 2026-09-14): modelo PARCIALMENTE LINEAL
por fase (estado no lineal + palancas de CONTROL lineales con estructura cinetica) y objetivo
anclado en el KPI de batch.

Extiende v1/v2/v3 (siguen siendo validos y se reutilizan: dataset, features, simulador de
derivadas, soporte historico). Que cambia y por que (evidencia en experimentos/13-20 y en
candidate_win_model.md):

1. **Forma del modelo: parcialmente lineal (Robinson 1988, cross-fitted).**
       y = g(S) + sum_j theta_j * (a_j - E[a_j | S]) + e
   g(S) y E[a_j|S] son HistGradientBoosting sobre el ESTADO (S, no lineal, sin restricciones);
   theta_j son los efectos de las palancas de CONTROL, con IC95% por bootstrap cluster por
   batch. Es el mismo modelo que el DML conjunto de v3 pero usado tambien como PREDICTOR: en
   Reduccion iguala al HGB en Sn (R2 OOF 0.973-0.975 / lockbox 0.982-0.985) y lo supera en FeO
   (0.36 / 0.38 vs 0.28 / 0.28; experimentos/14 y 20). Ventajas para prescribir: el efecto de
   cada CONTROL es explicito, con signo, magnitud e IC (criterio "distinguir robusta y
   significativamente el efecto de las variables de CONTROL"), y las palancas entran con la
   estructura que dicta la cinetica (carbon x Sn disponible; carbon x avance para el FeO).

2. **Que es palanca y que no (hallazgos de esta iteracion):**
   - `duracion_plan_min` en Reduccion es FIJA por protocolo (30/25/20/13 min, std 0 en los
     362 batches): NO es palanca. En Fusion es plan del operador: tampoco se optimiza.
   - La cal en Fusion se retira como palanca: su efecto sobre el Sn extraido (kg) esta
     contaminado por un artefacto contable del trazador CaO (+2.2 t/sd de masa de escoria no
     explicada, experimentos/18) y a nivel batch mas cal/carga NO mejora f_metal (dross +,
     p=0.06). Queda como plan del operador (basicidad).
   - La mezcla de carga (tolvas) es plan del batch.
   - Palancas de CONTROL por escalon: carbon (tasa), gas natural, O2 y aire de lanza (estos
     tres fijan el exceso de O2 = potencial redox de la lanza y el aporte termico).

3. **Objetivo en kg de Sn a metal (equivalente) anclado en el KPI:**
   - REDUCCION: J_R = Sn_extraido - lambda_Fe * FeO_reducido - costos - w_T*(T fuera de ventana)
     - w_U*ancho_intervalo - w_S*(1 - soporte). lambda_Fe = 0.30 kg Sn a dross por kg FeO
     (0.30 solo dross; 0.60 dross + polvo sobre f_metal, experimentos/15 y 19; ver LAMBDA_FE_RANGO). Coherencia con el KPI
     (experimentos/19): a nivel batch el carbon total de Reduccion BAJA el dross (-0.007/sd,
     p=0.03) y el GN lo SUBE (+0.007/sd) y baja el metal (-0.010/sd, p=0.04): el modelo de
     escalon reproduce el mecanismo (GN: +Sn pero +FeO, no selectivo; carbon: +Sn, ~0 FeO).
     El optimo interior de carbon/GN surge del compromiso Sn vs FeO (la selectividad marginal
     FeO/Sn pasa de 0.11 a 1.3-1.5 cuando el avance supera 0.96, experimentos/15), no de una
     monotonia impuesta.
   - FUSION: el Sn extraido en Fusion no es un objetivo (su relacion con el KPI es ambigua y
     sus palancas quimicas no se identifican en el escalon: experimentos/14, 17, 18). Se usa un
     objetivo ANCLADO EN EL KPI de batch (regresiones con controles, experimentos/19):
       J_F = beta_C * z(C/Sn cargado) - beta_T * z(T predicha) - costos - w_U*ancho - w_S*(1-soporte)
     con beta_C = +0.0087 de f_dross por sd de C/Sn de la fase (p=0.011; mas carbon por Sn cargado
     durante la Fusion reduce el Sn a dross: reduce SnO2 cuando aun esta en exceso, es decir
     selectivamente) y beta_T = 0.0129 de f_polvo por sd de la temperatura media de Fusion
     (p=0.027; mas temperatura, mas volatilizacion), ambos convertidos a kg de Sn por escalon.
     El modelo de escalon de Fusion (Sn extraido, DeltaT) provee la prediccion de T y el
     what-if del Sn extraido; la palanca es el carbon (via C/Sn) y los gases (via T).

4. **Monotonia y optimos interiores (teoria, no supuesto):** las palancas entran linealmente en
   el target de escalon (monotonas: carbon +, GN +, exceso O2 - sobre el Sn extraido; exceso O2
   - sobre el FeO reducido), y los optimos interiores del OBJETIVO provienen de (a) la
   selectividad Sn/Fe decreciente con el avance, (b) la ventana termica (volatilizacion y
   refractario) y (c) el soporte historico. El estado (basicidad, lanza, temperatura) entra
   sin restricciones en g(S).

5. **Evidencia de valor:** `recomendaciones_cross_fitted_v4` genera la recomendacion de cada
   batch con un sistema que NUNCA lo vio (5 folds sobre DEV + sistema DEV para el lockbox) y
   `evidencia_politica_v4` mide la asociacion entre la adherencia historica a la politica y el
   rendimiento proxy del batch (OLS HC3 con controles, dev/lockbox, placebo). Ver
   experimentos/21 y candidate_win_model.md.

Uso tipico:
    import modelo_predictivo_v4 as mp4
    df = mp4.construir_dataset_modelo_v4("Datos Lingo smelter fase II.xlsx")
    sistema = mp4.entrenar_sistema_v4(df)
    print(mp4.tabla_validacion_v4(df))
    print(mp4.tabla_efectos_control_v4(sistema))
    rep = mp4.reporte_recomendaciones_batch_v4(sistema, df, "AP0350")
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import feature_engineering as fe
import modelo_prescriptivo as mp
import modelo_predictivo_v3 as mp3

FASES = mp.FASES
SEED = 42

# =============================================================================
# 0) Dataset y columnas auxiliares v4
# =============================================================================

def construir_dataset_modelo_v4(ruta_excel: str, hoja: str = "Datos") -> pd.DataFrame:
    """Dataset v3 (features + targets en masa) + columnas de estructura cinetica v4."""
    return agregar_columnas_v4(mp3.construir_dataset_modelo_v3(ruta_excel, hoja))


def agregar_columnas_v4(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas de estructura cinetica (accion del escalon x estado previo, seguras):
        Cx_sn_v4  = tasa_feed_Carbon_kg_min * sn_inventario_escoria_est_kg_prev / 1000
                    (cinetica bimolecular SnO2 + 2C: el efecto del carbon es proporcional al Sn disponible)
        Cx_av_v4  = tasa_feed_Carbon_kg_min * avance_reduccion_sn_prev
                    (la sobre-reduccion del Fe crece cuando el SnOx se agota: carbon x avance)
    """
    df = df.copy()
    df["Cx_sn_v4"] = df["tasa_feed_Carbon_kg_min"] * df["sn_inventario_escoria_est_kg_prev"] / 1000.0
    df["Cx_av_v4"] = df["tasa_feed_Carbon_kg_min"] * df["avance_reduccion_sn_prev"]
    return df


def _agregar_columnas_v4_fila(fila: dict) -> dict:
    fila["Cx_sn_v4"] = fila.get("tasa_feed_Carbon_kg_min", np.nan) * fila.get("sn_inventario_escoria_est_kg_prev", np.nan) / 1000.0
    fila["Cx_av_v4"] = fila.get("tasa_feed_Carbon_kg_min", np.nan) * fila.get("avance_reduccion_sn_prev", np.nan)
    return fila


# =============================================================================
# 1) Roles v4: STATE / CONTEXT / CONTROL por (fase, target)
# =============================================================================

CONTEXT_V4 = mp3.CONTEXT_V3
STATE_V4 = mp3.STATE_V3
ACCIONES_PRIMITIVAS_V4 = mp3.ACCIONES_PRIMITIVAS_V3

# Palancas de CONTROL que el optimizador mueve. Duracion (protocolo fijo en Reduccion; plan en
# Fusion), cal (artefacto de trazador, experimentos/18) y mezcla de carga (plan) NO se mueven.
ACCIONES_OPTIMIZABLES_V4: dict[str, list[str]] = {
    "Fusión": ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
    "Reducción": ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
}

TARGETS_V4: dict[str, dict[str, str]] = {
    "Fusión": {"sn_kg": "sn_extraido_est_kg", "dT": "d_temperatura_horno_celsius"},
    "Reducción": {"sn_kg": "sn_extraido_est_kg", "feo_kg": "feo_extraido_est_kg", "dT": "d_temperatura_horno_celsius"},
}

_S_RED = list(dict.fromkeys(mp3.STATE_V3["Reducción"] + mp3.CONTEXT_V3["Reducción"]))
_S_FUS_CURADO = ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev",
                 "ley_feo_escoria_pct_prev", "basicidad_B2_prev", "temperatura_horno_celsius_prev", "orden_escalon_fase",
                 "feed_Sn_kgf", "tasa_feed_total_kg_min"]  # las dos ultimas: plan de carga (control no optimizable)
_S_DT_RED = ["temperatura_horno_celsius_prev", "termocupla_media_celsius_prev", "masa_escoria_est_kg_prev",
             "orden_escalon_fase", "grad_temperatura_horno_celsius_prev", "espesor_ladrillo_norm_mm"]
_S_DT_FUS = _S_DT_RED + ["tasa_feed_total_kg_min", "tasa_feed_CaO_kg_min"]

# Estado (no lineal, g(S)) por (fase, clave)
ESTADO_V4: dict[tuple[str, str], list[str]] = {
    ("Reducción", "sn_kg"): _S_RED,
    ("Reducción", "feo_kg"): _S_RED,
    ("Reducción", "dT"): _S_DT_RED,
    ("Fusión", "sn_kg"): _S_FUS_CURADO,
    ("Fusión", "dT"): _S_DT_FUS,
}
# Palancas lineales (theta) por (fase, clave). Las derivadas (exceso de O2, C/Sn, Cx) se
# reconstruyen desde las primitivas en el simulador.
PALANCAS_V4: dict[tuple[str, str], list[str]] = {
    ("Reducción", "sn_kg"): ["Cx_sn_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
    ("Reducción", "feo_kg"): ["tasa_feed_Carbon_kg_min", "Cx_av_v4", "tasa_gn_nm3_min", "exceso_o2_combustion_pct", "tasa_aire_nm3_min"],
    ("Reducción", "dT"): ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"],
    ("Fusión", "sn_kg"): ["relacion_C_Sn_carga", "relacion_CaO_carga", "tasa_gn_nm3_min", "exceso_o2_combustion_pct"],
    ("Fusión", "dT"): ["tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min", "tasa_feed_Carbon_kg_min"],
}
# Signo esperado por teoria (ITERACION_4_diseno.md) para auditar theta
SIGNO_TEORICO_V4: dict[tuple[str, str], dict[str, int]] = {
    ("Reducción", "sn_kg"): {"Cx_sn_v4": +1, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": -1},
    ("Reducción", "feo_kg"): {"tasa_feed_Carbon_kg_min": +1, "Cx_av_v4": +1, "tasa_gn_nm3_min": +1,
                              "exceso_o2_combustion_pct": -1, "tasa_aire_nm3_min": -1},
    ("Reducción", "dT"): {"tasa_gn_nm3_min": +1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": -1, "tasa_feed_Carbon_kg_min": 0},
    ("Fusión", "sn_kg"): {"relacion_C_Sn_carga": +1, "relacion_CaO_carga": 0, "tasa_gn_nm3_min": +1, "exceso_o2_combustion_pct": 0},
    ("Fusión", "dT"): {"tasa_gn_nm3_min": +1, "tasa_o2_nm3_min": 0, "tasa_aire_nm3_min": -1, "tasa_feed_Carbon_kg_min": 0},
}

for _k, _feats in list(ESTADO_V4.items()) + list(PALANCAS_V4.items()):
    _inseg = [f for f in _feats if f not in fe.filtrar_features_seguras(_feats) and not f.endswith("_v4")]
    if _inseg:
        raise AssertionError(f"v4 {_k}: features inseguras {_inseg}")
del _k, _feats, _inseg


# =============================================================================
# 2) Modelo parcialmente lineal (PLM) con cross-fitting
# =============================================================================

def _hgb_nuisance(seed: int = SEED, max_iter: int = 150):
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=max_iter, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=seed)


@dataclass
class ModeloPLM:
    """y = g(S) + theta . (a - m(S)). `g`/`m` son HGB entrenados en todo el train; theta se estima
    con residuos OUT-OF-FOLD (cross-fitting GroupKFold por batch) para evitar el sesgo de
    sobreajuste de Double ML. `q_abs_res` = cuantiles de |residuo OOF| para intervalos conformales."""
    estado: list[str]
    palancas: list[str]
    target: str
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ci_lo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ci_hi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sd_palancas: np.ndarray = field(default_factory=lambda: np.zeros(0))
    g: object = None
    m: dict = field(default_factory=dict)
    res_oof: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_train: int = 0
    r2_oof_interno: float = np.nan

    def fit(self, d: pd.DataFrame, n_splits: int = 5, n_boot: int = 200, seed: int = SEED) -> "ModeloPLM":
        S, A, y, grupos = self.estado, self.palancas, self.target, d["Batch"].to_numpy()
        cols = [y] + A
        oof = pd.DataFrame(np.nan, index=d.index, columns=cols)
        for tr, te in GroupKFold(n_splits).split(d[S], d[y], grupos):
            for c in cols:
                mdl = _hgb_nuisance(seed).fit(d[S].iloc[tr], d[c].iloc[tr])
                oof.iloc[te, oof.columns.get_loc(c)] = mdl.predict(d[S].iloc[te])
        y_res = d[y].to_numpy() - oof[y].to_numpy()
        A_res = np.column_stack([d[a].to_numpy() - oof[a].to_numpy() for a in A])
        self.theta = np.linalg.lstsq(A_res, y_res, rcond=None)[0]
        self.res_oof = y_res - A_res @ self.theta
        self.r2_oof_interno = float(r2_score(d[y], d[y].to_numpy() - self.res_oof))
        # bootstrap cluster por batch para IC95% de theta
        rng = np.random.default_rng(seed)
        uniq = np.unique(grupos)
        idx_by = {b: np.where(grupos == b)[0] for b in uniq}
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([idx_by[b] for b in rng.choice(uniq, size=len(uniq), replace=True)])
            boots.append(np.linalg.lstsq(A_res[idx], y_res[idx], rcond=None)[0])
        if boots:
            boots = np.array(boots)
            self.ci_lo, self.ci_hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        else:
            self.ci_lo, self.ci_hi = np.full(len(A), np.nan), np.full(len(A), np.nan)
        self.sd_palancas = d[A].std().to_numpy()
        self.g = _hgb_nuisance(seed).fit(d[S], d[y])
        self.m = {a: _hgb_nuisance(seed).fit(d[S], d[a]) for a in A}
        self.n_train = len(d)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.g.predict(X[self.estado]).astype(float)
        for j, a in enumerate(self.palancas):
            pred += self.theta[j] * (X[a].to_numpy(dtype=float) - self.m[a].predict(X[self.estado]))
        return pred

    def predict_intervalo(self, X: pd.DataFrame, alpha: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intervalo conformal (split, residuos OOF del cross-fitting): pred +/- q_{1-alpha}(|res|)."""
        pred = self.predict(X)
        q = float(np.quantile(np.abs(self.res_oof), 1 - alpha)) if len(self.res_oof) else np.nan
        return pred, pred - q, pred + q

    def tabla_theta(self, signo_teorico: dict | None = None) -> pd.DataFrame:
        t = pd.DataFrame({"palanca": self.palancas, "theta_unidad": self.theta, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
                          "sd_palanca": self.sd_palancas, "theta_1sd": self.theta * self.sd_palancas,
                          "ci_lo_1sd": self.ci_lo * self.sd_palancas, "ci_hi_1sd": self.ci_hi * self.sd_palancas})
        t["significativo"] = (t["ci_lo"] > 0) | (t["ci_hi"] < 0)
        if signo_teorico:
            t["signo_teorico"] = [signo_teorico.get(a, 0) for a in self.palancas]
            t["concuerda"] = [(np.sign(th) == s) if s != 0 else np.nan for th, s in zip(self.theta, t["signo_teorico"])]
        return t.set_index("palanca")


# =============================================================================
# 3) Sistema v4: entrenamiento (solo DEV), soporte, limites, ventana termica
# =============================================================================

# Coeficientes KPI-anclados de Fusion (experimentos/19, OLS HC3 con controles, DEV n=298):
#   f_dross ~ -0.0087 por +1 sd de C_por_Sn_F (IC95% [-0.0154, -0.0019], p=0.011)
#   f_polvo ~ +0.0129 por +1 sd de T_media_F  (IC95% [0.0015, 0.0243], p=0.027)
# f_* son fracciones del Sn de salida del batch (metal+dross+polvo ~ 51.2 t); por escalon de
# Fusion (7) se reparte el efecto de batch a partes iguales. Se convierten a kg de Sn.
BETA_KPI_FUSION = {
    "C_por_Sn": {"coef_por_sd": 0.0087, "ci": (0.0019, 0.0154), "sd_batch": 0.018, "media_batch": 0.272},
    "T_media": {"coef_por_sd": 0.0129, "ci": (0.0015, 0.0243), "sd_batch": 75.1, "media_batch": 1068.0},
}
SN_SALIDA_BATCH_KG = 51250.0
N_ESCALONES_FUSION = 7

# kg de Sn a METAL perdidos por kg de FeO reducido en Reduccion. v3 usaba 0.30 (solo el canal de
# dross: Sn_dross ~ 0.30-0.32 * FeO_R, experimentos/15). El KPI (f_metal) pierde ademas por polvo:
# f_metal ~ -0.0139/sd de FeO_R con controles (experimentos/19 M4, p=0.013; sd = 1.16 t, salida
# 51.25 t) => 0.61 kg metal por kg FeO [0.13, 1.10]; rendimiento ~ -0.64 pp/t (experimentos/15 e,
# p=0.009) => 0.33 [0.08, 0.58]. Se adopta 0.60 (canales dross + polvo); con 0.30 el GN parece
# neutro y el optimizador lo sube, en contra de la evidencia de batch (GN sube dross y baja metal).
LAMBDA_FE_KG_SN_POR_KG_FEO = 0.60
LAMBDA_FE_RANGO = (0.30, 0.60)

PESOS_V4_DEFAULT = dict(
    lambda_fe=LAMBDA_FE_KG_SN_POR_KG_FEO,
    costo_carbon_kgSn_por_kg=0.05, costo_gn_kgSn_por_nm3=0.03, costo_o2_kgSn_por_nm3=0.01,
    w_temp=2.0, w_incertidumbre=0.25, w_soporte=2000.0,
)
PERCENTIL_SOPORTE_MINIMO = 0.10


@dataclass
class SistemaPrescriptivoV4:
    modelos: dict = field(default_factory=dict)          # (fase, clave) -> ModeloPLM
    soporte: dict = field(default_factory=dict)          # fase -> kNN sobre estado curado + palancas primitivas
    soporte_minimo: dict = field(default_factory=dict)
    limites_accion: dict = field(default_factory=dict)   # fase -> {primitiva: (p1, p99)}
    ventana_temperatura: dict = field(default_factory=dict)
    batches_dev: set = field(default_factory=set)
    batches_lockbox: set = field(default_factory=set)
    beta_fusion: dict = field(default_factory=lambda: dict(BETA_KPI_FUSION))


FEATURES_SOPORTE_V4 = {
    "Fusión": _S_FUS_CURADO + ["tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
    "Reducción": ["sn_inventario_escoria_est_kg_prev", "masa_escoria_est_kg_prev", "ley_sn_escoria_pct_prev",
                  "ley_feo_escoria_pct_prev", "avance_reduccion_sn_prev", "temperatura_horno_celsius_prev", "orden_escalon_fase",
                  "tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min", "tasa_o2_nm3_min", "tasa_aire_nm3_min"],
}


def _preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df_base = mp.dataset_base_modelo(df)
    if "Cx_sn_v4" not in df_base.columns:
        df_base = agregar_columnas_v4(df_base)
    return df_base


def entrenar_sistema_v4(df: pd.DataFrame, pct_lockbox: float = mp.N_LOCKBOX_PCT, n_boot: int = 200,
                        seed: int = SEED) -> SistemaPrescriptivoV4:
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df, pct_lockbox)
    sistema = SistemaPrescriptivoV4(batches_dev=batches_dev, batches_lockbox=batches_lockbox)
    for fase in FASES:
        sub = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(batches_dev)]
        for clave, target in TARGETS_V4[fase].items():
            S, A = ESTADO_V4[(fase, clave)], PALANCAS_V4[(fase, clave)]
            d = sub.dropna(subset=list(dict.fromkeys(S + A + [target]))).reset_index(drop=True)
            sistema.modelos[(fase, clave)] = ModeloPLM(S, A, target).fit(d, n_boot=n_boot, seed=seed)
        feats_sop = FEATURES_SOPORTE_V4[fase]
        sub_sop = sub.dropna(subset=feats_sop)
        sistema.soporte[fase] = mp._construir_soporte_historico(sub_sop, feats_sop)
        scores = mp3._support_scores_lote(sistema.soporte[fase], sub_sop)
        sistema.soporte_minimo[fase] = float(np.quantile(scores, PERCENTIL_SOPORTE_MINIMO))
        acciones = sub[ACCIONES_PRIMITIVAS_V4[fase]].dropna()
        sistema.limites_accion[fase] = {a: (float(acciones[a].quantile(0.01)), float(acciones[a].quantile(0.99)))
                                        for a in ACCIONES_PRIMITIVAS_V4[fase]}
        t = sub["temperatura_horno_celsius"].dropna()
        sistema.ventana_temperatura[fase] = (float(t.quantile(0.05)), float(t.quantile(0.95)))
    return sistema


def tabla_efectos_control_v4(sistema: SistemaPrescriptivoV4) -> pd.DataFrame:
    """theta de cada palanca por (fase, target): por unidad y por +1 sd, IC95% bootstrap cluster,
    significancia y concordancia con el signo teorico."""
    filas = []
    for (fase, clave), m in sistema.modelos.items():
        t = m.tabla_theta(SIGNO_TEORICO_V4.get((fase, clave))).reset_index()
        t.insert(0, "target", TARGETS_V4[fase][clave]); t.insert(0, "clave", clave); t.insert(0, "fase", fase)
        filas.append(t)
    return pd.concat(filas, ignore_index=True)


# =============================================================================
# 4) Validacion OOF (GroupKFold por batch, DEV) y lockbox por (fase, target)
# =============================================================================

def tabla_validacion_v4(df: pd.DataFrame, n_splits: int = mp.N_SPLITS_OOF, con_baseline_v3: bool = True,
                        seed: int = SEED) -> pd.DataFrame:
    """R2/MAE OOF (DEV) y lockbox del PLM v4 por (fase, target); opcionalmente el HGB v3 como
    referencia sobre las MISMAS filas. El lockbox solo se reporta."""
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    filas = []
    for fase in FASES:
        sub = df_base.loc[df_base["fase_proceso"] == fase]
        for clave, target in TARGETS_V4[fase].items():
            S, A = ESTADO_V4[(fase, clave)], PALANCAS_V4[(fase, clave)]
            feats_v3 = mp3.FEATURES_V3_POR_DEFECTO[fase].get(clave, [])
            need = list(dict.fromkeys(S + A + feats_v3 + [target, "Batch"]))
            d_dev = sub.loc[sub["Batch"].isin(batches_dev), need].dropna().reset_index(drop=True)
            d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), need].dropna().reset_index(drop=True)
            oof = np.full(len(d_dev), np.nan)
            for tr, te in GroupKFold(n_splits).split(d_dev[S], d_dev[target], d_dev["Batch"]):
                m = ModeloPLM(S, A, target).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
                oof[te] = m.predict(d_dev.iloc[te])
            m = ModeloPLM(S, A, target).fit(d_dev, n_boot=0, seed=seed)
            p_lb = m.predict(d_lb)
            filas.append({"fase": fase, "target": clave, "columna": target, "forma": "PLM_v4", "n_estado": len(S), "n_palancas": len(A),
                          "n_dev": len(d_dev), "n_lockbox": len(d_lb),
                          "r2_oof": r2_score(d_dev[target], oof), "mae_oof": mean_absolute_error(d_dev[target], oof),
                          "r2_lockbox": r2_score(d_lb[target], p_lb), "mae_lockbox": mean_absolute_error(d_lb[target], p_lb),
                          "std_target_lockbox": float(d_lb[target].std())})
            if con_baseline_v3 and feats_v3:
                oof3 = np.full(len(d_dev), np.nan)
                for tr, te in GroupKFold(n_splits).split(d_dev[feats_v3], d_dev[target], d_dev["Batch"]):
                    m3 = mp3.construir_modelo_v3(feats_v3, target).fit(d_dev[feats_v3].iloc[tr], d_dev[target].iloc[tr])
                    oof3[te] = m3.predict(d_dev[feats_v3].iloc[te])
                m3 = mp3.construir_modelo_v3(feats_v3, target).fit(d_dev[feats_v3], d_dev[target])
                p3 = m3.predict(d_lb[feats_v3])
                filas.append({"fase": fase, "target": clave, "columna": target, "forma": "HGB_v3_ref", "n_estado": len(feats_v3), "n_palancas": 0,
                              "n_dev": len(d_dev), "n_lockbox": len(d_lb),
                              "r2_oof": r2_score(d_dev[target], oof3), "mae_oof": mean_absolute_error(d_dev[target], oof3),
                              "r2_lockbox": r2_score(d_lb[target], p3), "mae_lockbox": mean_absolute_error(d_lb[target], p3),
                              "std_target_lockbox": float(d_lb[target].std())})
    return pd.DataFrame(filas)


def predicciones_oof_y_lockbox_v4(df: pd.DataFrame, fase: str, clave: str, n_splits: int = mp.N_SPLITS_OOF,
                                  seed: int = SEED) -> pd.DataFrame:
    """Devuelve las filas (Batch, fecha_inicio, orden, real, pred, conjunto) OOF-dev y lockbox para graficas real vs pred."""
    df_base = _preparar_base(df)
    batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
    target = TARGETS_V4[fase][clave]
    S, A = ESTADO_V4[(fase, clave)], PALANCAS_V4[(fase, clave)]
    sub = df_base.loc[df_base["fase_proceso"] == fase]
    need = list(dict.fromkeys(S + A + [target, "Batch", "fecha_inicio", "orden_escalon_fase"]))
    d_dev = sub.loc[sub["Batch"].isin(batches_dev), need].dropna().reset_index(drop=True)
    d_lb = sub.loc[sub["Batch"].isin(batches_lockbox), need].dropna().reset_index(drop=True)
    oof = np.full(len(d_dev), np.nan)
    for tr, te in GroupKFold(n_splits).split(d_dev[S], d_dev[target], d_dev["Batch"]):
        m = ModeloPLM(S, A, target).fit(d_dev.iloc[tr].reset_index(drop=True), n_boot=0, seed=seed)
        oof[te] = m.predict(d_dev.iloc[te])
    m = ModeloPLM(S, A, target).fit(d_dev, n_boot=0, seed=seed)
    out_dev = d_dev[["Batch", "fecha_inicio", "orden_escalon_fase"]].assign(real=d_dev[target], pred=oof, conjunto="OOF_dev")
    out_lb = d_lb[["Batch", "fecha_inicio", "orden_escalon_fase"]].assign(real=d_lb[target], pred=m.predict(d_lb), conjunto="lockbox")
    return pd.concat([out_dev, out_lb], ignore_index=True)


# =============================================================================
# 5) Simulador y objetivo
# =============================================================================

def fila_a_state_action_context(fase: str, row: pd.Series) -> tuple[dict, dict, dict]:
    return mp3.fila_a_state_action_context(fase, row)


def _todas_las_columnas(sistema: SistemaPrescriptivoV4, fase: str) -> list[str]:
    cols = set(sistema.soporte[fase]["features"])
    for (f, k), m in sistema.modelos.items():
        if f == fase:
            cols |= set(m.estado) | set(m.palancas)
    return sorted(cols)


def simular_acciones_lote_v4(sistema: SistemaPrescriptivoV4, fase: str, state: dict, acciones: list[dict],
                             context: dict | None = None, alpha: float = 0.20) -> pd.DataFrame:
    """state + lote de acciones -> Sn extraido [kg], FeO reducido [kg] (Reduccion), DeltaT, T predicha,
    intervalo conformal, soporte y derivadas (C/Sn, exceso O2)."""
    filas = [_agregar_columnas_v4_fila(mp3.recalcular_derivadas_v3(fase, state, a, context)) for a in acciones]
    cols = _todas_las_columnas(sistema, fase)
    X = pd.DataFrame([[f.get(c, np.nan) for c in cols] for f in filas], columns=cols)
    out = pd.DataFrame(index=range(len(acciones)))
    sn, lo, hi = sistema.modelos[(fase, "sn_kg")].predict_intervalo(X, alpha)
    out["pred_sn_extraido_kg"], out["sn_lo"], out["sn_hi"] = sn, lo, hi
    if fase == "Reducción":
        feo, flo, fhi = sistema.modelos[(fase, "feo_kg")].predict_intervalo(X, alpha)
        out["pred_feo_extraido_kg"], out["feo_lo"], out["feo_hi"] = feo, flo, fhi
        out["pred_feo_reducido_kg"] = np.maximum(0.0, feo)
    else:
        out["pred_feo_extraido_kg"] = np.nan; out["feo_lo"] = np.nan; out["feo_hi"] = np.nan; out["pred_feo_reducido_kg"] = np.nan
    dT, dlo, dhi = sistema.modelos[(fase, "dT")].predict_intervalo(X, alpha)
    t_prev = float(state.get("temperatura_horno_celsius_prev", np.nan))
    out["pred_dT"], out["dT_lo"], out["dT_hi"] = dT, dlo, dhi
    out["pred_temperatura"] = t_prev + dT if pd.notna(t_prev) else np.nan
    out["support_score"] = mp3._support_scores_lote(sistema.soporte[fase], X)
    for k in ("exceso_o2_combustion_pct", "oxygen_enrichment_pct", "relacion_C_Sn_carga", "relacion_CaO_carga"):
        out[f"derivada__{k}"] = [f.get(k, np.nan) for f in filas]
    return out


def _z_fusion(sistema: SistemaPrescriptivoV4, nombre: str, valor: np.ndarray) -> np.ndarray:
    b = sistema.beta_fusion[nombre]
    return (np.asarray(valor, dtype=float) - b["media_batch"]) / b["sd_batch"]


def _kg_por_sd_fusion(sistema: SistemaPrescriptivoV4, nombre: str) -> float:
    return sistema.beta_fusion[nombre]["coef_por_sd"] * SN_SALIDA_BATCH_KG / N_ESCALONES_FUSION


def objetivo_lote_v4(sistema: SistemaPrescriptivoV4, fase: str, state: dict, acciones: list[dict], context: dict | None = None,
                     pesos: dict | None = None) -> pd.DataFrame:
    """J [kg Sn a metal, equivalente] por accion candidata.
    Reduccion: J = Sn_extraido - lambda_fe*FeO_reducido - costos - w_T*fuera - w_U*ancho - w_S*(1-soporte)
    Fusion:    J = beta_C*z(C/Sn) - beta_T*z(T_pred) - costos - w_U*ancho_T - w_S*(1-soporte)
               (beta en kg de Sn por escalon, anclados en las regresiones de KPI de batch)."""
    p = dict(PESOS_V4_DEFAULT, **(pesos or {}))
    sim = simular_acciones_lote_v4(sistema, fase, state, acciones, context)
    A = pd.DataFrame(acciones)
    dt = A["duracion_plan_min"].to_numpy(dtype=float)
    costo = (p["costo_carbon_kgSn_por_kg"] * A["tasa_feed_Carbon_kg_min"].to_numpy(dtype=float) * dt
             + p["costo_gn_kgSn_por_nm3"] * A["tasa_gn_nm3_min"].to_numpy(dtype=float) * dt
             + p["costo_o2_kgSn_por_nm3"] * A["tasa_o2_nm3_min"].to_numpy(dtype=float) * dt)
    t_min, t_max = sistema.ventana_temperatura[fase]
    t_pred = sim["pred_temperatura"].to_numpy(dtype=float)
    fuera = np.where(np.isnan(t_pred), 0.0, np.maximum(0.0, t_min - t_pred) + np.maximum(0.0, t_pred - t_max))
    sim["J_costo"] = -costo
    sim["J_temperatura"] = -p["w_temp"] * fuera
    sim["J_soporte"] = -p["w_soporte"] * (1 - sim["support_score"])
    if fase == "Reducción":
        sim["J_sn"] = sim["pred_sn_extraido_kg"]
        sim["J_feo"] = -p["lambda_fe"] * np.nan_to_num(sim["pred_feo_reducido_kg"].to_numpy(dtype=float), nan=0.0)
        sim["J_incertidumbre"] = -p["w_incertidumbre"] * (sim["sn_hi"] - sim["sn_lo"])
    else:
        z_c = _z_fusion(sistema, "C_por_Sn", sim["derivada__relacion_C_Sn_carga"].to_numpy())
        z_t = _z_fusion(sistema, "T_media", t_pred)
        sim["J_sn"] = _kg_por_sd_fusion(sistema, "C_por_Sn") * np.nan_to_num(z_c, nan=0.0)   # menos Sn a dross
        sim["J_feo"] = -_kg_por_sd_fusion(sistema, "T_media") * np.nan_to_num(z_t, nan=0.0)  # menos Sn a polvo
        sim["J_incertidumbre"] = -p["w_incertidumbre"] * (sim["dT_hi"] - sim["dT_lo"]) * _kg_por_sd_fusion(sistema, "T_media") / sistema.beta_fusion["T_media"]["sd_batch"]
    sim["J"] = sim[["J_sn", "J_feo", "J_costo", "J_temperatura", "J_incertidumbre", "J_soporte"]].sum(axis=1)
    return sim


def optimizar_accion_v4(sistema: SistemaPrescriptivoV4, fase: str, state: dict, accion_base: dict, context: dict | None = None,
                        acciones_a_optimizar: list[str] | None = None, n_candidatos: int = 300, n_refinamiento: int = 60,
                        semilla: int = 0, pesos: dict | None = None) -> tuple[dict, float, pd.Series]:
    """Busqueda aleatoria en P1-P99 de cada palanca optimizable + refinamiento local, con soporte
    historico minimo DURO (P10 DEV); la accion base siempre es admisible."""
    optimizables = acciones_a_optimizar or ACCIONES_OPTIMIZABLES_V4[fase]
    limites = {a: sistema.limites_accion[fase][a] for a in optimizables}
    rng = np.random.default_rng(semilla)
    candidatos = [dict(accion_base)]
    for _ in range(n_candidatos):
        cand = dict(accion_base)
        for a, (lo, hi) in limites.items():
            cand[a] = float(rng.uniform(lo, hi))
        candidatos.append(cand)
    sim = objetivo_lote_v4(sistema, fase, state, candidatos, context, pesos)
    umbral = sistema.soporte_minimo.get(fase, 0.0)
    J_adm = sim["J"].where(sim["support_score"] >= umbral)
    J_adm.iloc[0] = sim["J"].iloc[0]
    i_best = int(J_adm.idxmax())
    mejor_accion, mejor_J, mejor_fila = candidatos[i_best], float(sim.loc[i_best, "J"]), sim.loc[i_best]
    if n_refinamiento > 0:
        refin = []
        for _ in range(n_refinamiento):
            cand = dict(mejor_accion)
            for a, (lo, hi) in limites.items():
                cand[a] = float(np.clip(cand[a] + rng.normal(0, 0.08 * (hi - lo)), lo, hi))
            refin.append(cand)
        sim_r = objetivo_lote_v4(sistema, fase, state, refin, context, pesos)
        J_r = sim_r["J"].where(sim_r["support_score"] >= umbral)
        if J_r.notna().any():
            j = int(J_r.idxmax())
            if float(sim_r.loc[j, "J"]) > mejor_J:
                mejor_accion, mejor_J, mejor_fila = refin[j], float(sim_r.loc[j, "J"]), sim_r.loc[j]
    return mejor_accion, mejor_J, mejor_fila


def curva_respuesta_v4(sistema: SistemaPrescriptivoV4, fase: str, state: dict, action: dict, context: dict | None,
                       palanca: str, n: int = 25, pesos: dict | None = None) -> pd.DataFrame:
    """Predicciones y J a lo largo de una rejilla 1-D de `palanca` (resto fijo en `action`): es la
    "dependencia parcial" del prescriptor en el estado concreto del escalon."""
    lo, hi = sistema.limites_accion[fase][palanca]
    grid = np.linspace(lo, hi, n)
    acciones = []
    for v in grid:
        a = dict(action); a[palanca] = float(v); acciones.append(a)
    sim = objetivo_lote_v4(sistema, fase, state, acciones, context, pesos)
    sim.insert(0, "valor", grid); sim.insert(0, "palanca", palanca)
    return sim


# =============================================================================
# 6) Reportes por batch y politica cross-fitted (evidencia de valor)
# =============================================================================

def reporte_recomendaciones_batch_v4(sistema: SistemaPrescriptivoV4, df: pd.DataFrame, batch_id: str,
                                     n_candidatos: int = 200, semilla: int = 0, pesos: dict | None = None) -> pd.DataFrame:
    """Recomendacion escalon a escalon para UN batch (cada escalon sobre su estado REAL, sin encadenar)."""
    df_base = _preparar_base(df)
    sub = df_base.loc[df_base["Batch"] == batch_id].sort_values("fecha_inicio")
    filas = []
    for _, row in sub.iterrows():
        fase = row["fase_proceso"]
        state, accion_hist, context = fila_a_state_action_context(fase, row)
        if any(pd.isna(v) for v in accion_hist.values()) or pd.isna(state.get("ley_sn_escoria_pct_prev")):
            continue
        try:
            sim_h = objetivo_lote_v4(sistema, fase, state, [accion_hist], context, pesos).iloc[0]
            accion_opt, J_o, sim_o = optimizar_accion_v4(sistema, fase, state, accion_hist, context, None, n_candidatos, semilla=semilla, pesos=pesos)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{batch_id} {fase} {row['orden_escalon_fase']}: {type(e).__name__}: {e}")
            continue
        fila = {"Batch": batch_id, "fase": fase, "orden_escalon_fase": row["orden_escalon_fase"],
                "avance_reduccion_sn_prev": state.get("avance_reduccion_sn_prev"), "ley_sn_escoria_pct_prev": state["ley_sn_escoria_pct_prev"],
                "sn_extraido_real_kg": row.get("sn_extraido_est_kg"), "feo_extraido_real_kg": row.get("feo_extraido_est_kg"),
                "temperatura_real": row.get("temperatura_horno_celsius"),
                "pred_sn_kg_historico": sim_h["pred_sn_extraido_kg"], "pred_sn_kg_recomendado": sim_o["pred_sn_extraido_kg"],
                "pred_feo_reducido_kg_historico": sim_h["pred_feo_reducido_kg"], "pred_feo_reducido_kg_recomendado": sim_o["pred_feo_reducido_kg"],
                "pred_T_historico": sim_h["pred_temperatura"], "pred_T_recomendado": sim_o["pred_temperatura"],
                "support_historico": sim_h["support_score"], "support_recomendado": sim_o["support_score"],
                "J_historico": float(sim_h["J"]), "J_recomendado": J_o, "uplift_J_kgSn": J_o - float(sim_h["J"])}
        for k in ("J_sn", "J_feo", "J_costo", "J_temperatura", "J_soporte"):
            fila[f"{k}_hist"] = float(sim_h[k]); fila[f"{k}_rec"] = float(sim_o[k])
        for a in ACCIONES_PRIMITIVAS_V4[fase]:
            fila[f"hist__{a}"] = accion_hist[a]; fila[f"rec__{a}"] = accion_opt[a]
        for k in ("exceso_o2_combustion_pct", "relacion_C_Sn_carga"):
            fila[f"hist_derivada__{k}"] = float(sim_h[f"derivada__{k}"]); fila[f"rec_derivada__{k}"] = float(sim_o[f"derivada__{k}"])
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_reporte_batch_v4(rep: pd.DataFrame) -> dict:
    """Uplift ESTIMADO por el modelo (kg Sn-metal equivalente) y adherencia de la operacion historica."""
    if rep.empty:
        return {}
    fus, red = rep["fase"] == "Fusión", rep["fase"] == "Reducción"
    fis = sum(rep[f"{k}_rec"] - rep[f"{k}_hist"] for k in ("J_sn", "J_feo", "J_costo", "J_temperatura"))
    return {"n_escalones": int(len(rep)),
            "uplift_fisico_fusion_kg": float(fis[fus].sum()), "uplift_fisico_reduccion_kg": float(fis[red].sum()),
            "uplift_fisico_total_kg": float(fis.sum()),   # sin el termino de soporte (regularizador, no valor fisico)
            "uplift_J_fusion_kg": float(rep.loc[fus, "uplift_J_kgSn"].sum()),
            "uplift_J_reduccion_kg": float(rep.loc[red, "uplift_J_kgSn"].sum()),
            "uplift_J_total_kg": float(rep["uplift_J_kgSn"].sum()),
            "reduccion_sn_extraido_adicional_kg": float((rep.loc[red, "pred_sn_kg_recomendado"] - rep.loc[red, "pred_sn_kg_historico"]).sum()),
            "reduccion_feo_reducido_adicional_kg": float((rep.loc[red, "pred_feo_reducido_kg_recomendado"] - rep.loc[red, "pred_feo_reducido_kg_historico"]).sum()),
            "pct_escalones_con_mejora": float((rep["uplift_J_kgSn"] > 1.0).mean() * 100),
            "support_medio_historico": float(rep["support_historico"].mean()),
            "support_medio_recomendado": float(rep["support_recomendado"].mean())}


def _folds_dev(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> tuple[set, set, dict]:
    dev, lockbox = mp.split_dev_lockbox(df)
    dev_list = sorted(dev)
    perm = np.random.default_rng(seed).permutation(len(dev_list))
    fold_de = {dev_list[i]: int(pos % n_folds) for pos, i in enumerate(perm)}
    folds = {k: sorted(b for b, f in fold_de.items() if f == k) for k in range(n_folds)}
    return dev, lockbox, folds


def recomendaciones_cross_fitted_v4(df: pd.DataFrame, etiqueta: str, n_folds: int = 5, n_candidatos: int = 150,
                                    semilla: int = 0, log=print, pesos: dict | None = None) -> pd.DataFrame:
    """Etapa por proceso: `etiqueta` = '0'..'n_folds-1' (fold de DEV; el sistema se entrena SIN esos
    batches) o 'lockbox' (sistema entrenado con todo DEV). Devuelve el detalle por escalon."""
    dev, lockbox, folds = _folds_dev(df, n_folds)
    if etiqueta == "lockbox":
        sistema = entrenar_sistema_v4(df, n_boot=0)
        batches, fold, es_lb = sorted(lockbox), -1, True
    else:
        k = int(etiqueta)
        sistema = entrenar_sistema_v4(df[df["Batch"].isin(dev - set(folds[k]))], pct_lockbox=0.0, n_boot=0)
        batches, fold, es_lb = folds[k], k, False
    reps = []
    for i, b in enumerate(batches):
        rep = reporte_recomendaciones_batch_v4(sistema, df, b, n_candidatos=n_candidatos, semilla=semilla, pesos=pesos)
        if not rep.empty:
            rep["fold"], rep["es_lockbox"] = fold, es_lb
            reps.append(rep)
        if (i + 1) % 10 == 0:
            log(f"{etiqueta}: {i+1}/{len(batches)}")
    return pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()


def metricas_por_batch_v4(df: pd.DataFrame, escalones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por batch: uplift estimado, adherencia (|hist-rec|/sd DEV por palanca y fase, y
    direccion (rec-hist)/sd), KPIs y controles de contexto."""
    dev, _ = mp.split_dev_lockbox(df)
    df_base = _preparar_base(df)
    sd = {}
    for fase in FASES:
        s = df_base.loc[(df_base["fase_proceso"] == fase) & df_base["Batch"].isin(dev)]
        for a in ACCIONES_OPTIMIZABLES_V4[fase]:
            sd[(fase, a)] = float(s[a].std())
    filas = []
    for b, rep in escalones.groupby("Batch"):
        fila = resumen_reporte_batch_v4(rep); fila["Batch"] = b
        fila["es_lockbox"] = bool(rep["es_lockbox"].iloc[0]); fila["fold"] = int(rep["fold"].iloc[0])
        dists = []
        for fase in FASES:
            r = rep.loc[rep["fase"] == fase]
            if r.empty:
                continue
            d_f = []
            for a in ACCIONES_OPTIMIZABLES_V4[fase]:
                d = ((r[f"hist__{a}"] - r[f"rec__{a}"]).abs() / sd[(fase, a)]).mean()
                fila[f"dist_{fase}_{a}"] = float(d); fila[f"dir_{fase}_{a}"] = float(((r[f"rec__{a}"] - r[f"hist__{a}"]) / sd[(fase, a)]).mean())
                d_f.append(d)
            fila[f"dist_{'F' if fase == 'Fusión' else 'R'}"] = float(np.mean(d_f)); dists += d_f
        fila["dist_total"] = float(np.mean(dists))
        g = df.loc[df["Batch"] == b]; gF = g.loc[g["fase_proceso"] == "Fusión"]
        m_, d_, p_ = (float(g[c].iloc[0]) for c in ("sn_en_metal_crudo_batch_t", "sn_en_dross_fe_batch_t", "sn_en_polvo_fundicion_batch_t"))
        tot = m_ + d_ + p_
        fila.update({"rendimiento_proxy_batch": float(g["rendimiento_proxy_batch"].iloc[0]), "f_metal": m_ / tot, "f_dross": d_ / tot, "f_polvo": p_ / tot,
                     "ley_sn_conc_batch_pct": float(g["ley_sn_conc_batch_pct"].iloc[0]), "espesor_ladrillo_norm_mm": float(g["espesor_ladrillo_norm_mm"].mean()),
                     "sn_cargado_batch_kg": float(g["feed_Sn_kgf"].sum()), "temperatura_media_fusion": float(gF["temperatura_horno_celsius"].mean()),
                     "frac_carga_secundaria_media_fusion": float(gF["frac_carga_secundaria"].mean()), "fecha_inicio_batch": g["fecha_inicio"].min()})
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("Batch").reset_index(drop=True)


CONTROLES_KPI = ["ley_sn_conc_batch_pct", "espesor_ladrillo_norm_mm", "sn_cargado_batch_kg", "temperatura_media_fusion", "frac_carga_secundaria_media_fusion"]


def evidencia_politica_v4(por_batch: pd.DataFrame, variables: list[str] | None = None, kpis: list[str] | None = None,
                          n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Asociacion entre adherencia/direccion de la politica y el KPI del batch: OLS HC3 con controles
    (coef por +1 sd de la variable), Spearman, y p de permutacion; por subconjunto DEV/LOCKBOX/TOTAL.
    Hipotesis: dist_* con coeficiente NEGATIVO sobre rendimiento (mas lejos de la politica, peor)."""
    import statsmodels.api as sm
    from scipy import stats
    variables = variables or ["dist_total", "dist_F", "dist_R", "uplift_fisico_total_kg"]
    kpis = kpis or ["rendimiento_proxy_batch", "f_dross", "f_polvo"]
    rng = np.random.default_rng(seed)
    filas = []
    for nombre, sub in [("DEV", por_batch[~por_batch["es_lockbox"]]), ("LOCKBOX", por_batch[por_batch["es_lockbox"]]), ("TOTAL", por_batch)]:
        for kpi in kpis:
            for v in variables:
                d = sub[[kpi, v] + CONTROLES_KPI].dropna()
                if len(d) < 20:
                    continue
                X = d[[v] + CONTROLES_KPI].copy()
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)
                res = sm.OLS(d[kpi], X).fit(cov_type="HC3")
                rho, p_rho = stats.spearmanr(d[kpi], d[v])
                coef = float(res.params[v])
                if n_perm and nombre != "LOCKBOX":
                    nulos = []
                    for _ in range(n_perm):
                        Xp = X.copy(); Xp[v] = rng.permutation(Xp[v].to_numpy())
                        nulos.append(float(sm.OLS(d[kpi], Xp).fit().params[v]))
                    p_perm = float(np.mean(np.abs(nulos) >= abs(coef)))
                else:
                    p_perm = np.nan
                filas.append({"subset": nombre, "kpi": kpi, "variable": v, "n": len(d), "coef_por_sd": coef,
                              "ci_lo": float(res.conf_int().loc[v, 0]), "ci_hi": float(res.conf_int().loc[v, 1]),
                              "p_hc3": float(res.pvalues[v]), "p_perm": p_perm, "spearman": float(rho), "p_spearman": float(p_rho),
                              "r2_adj": float(res.rsquared_adj)})
    return pd.DataFrame(filas)
