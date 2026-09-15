"""Seguimiento del hallazgo mas fuerte del audit Boruta/MI/permutacion (script 02):
en Reduccion, 5 de las 7 features YA EN PRODUCCION (exceso_o2_combustion_pct,
basicidad_B2_prev, grad_temperatura_horno_celsius_prev, posicion_vertical_lanza_mm,
cum_feed_Carbon_kg_prev) no superan el test de shadow-features (Boruta) sobre el pool
completo de 79 candidatas -- 3 de ellas con 0/25 hits (indistinguibles de ruido en ESE
contexto). Esto es coherente con (no contradice) el hallazgo previo de causalidad
inestable de `posicion_vertical_lanza_mm`/`exceso_o2` (Paso 42, hallazgos.md 11.6): dos
metodologias independientes (residualizacion causal y shadow-feature test) convergen.

Aqui se arma un set ALTERNATIVO ("Reduccion v2") a partir de las CONFIRMADAS+TENTATIVAS
del Boruta (script 02), se decorrelaciona (mismo criterio que Pasos 20-27: eliminar
miembros de pares |rho_Spearman|>0.85) y se compara, con la MISMA validacion de 3
niveles ya usada en produccion, contra el set actual -- para decidir con evidencia (no
suposicion) si conviene reemplazar el set de Reduccion.
"""
import os
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

sys.path.insert(0, r"C:\Users\user\Desktop\Linea de Sn\Lingo Smelter")
import modelo_prescriptivo as mp

df = mp.construir_dataset_modelo(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datos Lingo smelter fase II.xlsx"))
df_base = mp.dataset_base_modelo(df)
batches_dev, batches_lockbox = mp.split_dev_lockbox(df)
orden_batches = mp.orden_cronologico_batches(df)

FASE = "Reducción"
TARGET = mp.TARGET_PRESCRIPTIVO_GANADOR[FASE]

# CONFIRMADA (>=0.6 frac) + TENTATIVA (>=0.4) del Boruta sobre el pool completo (script 02).
# EXCLUIDAS a proposito pese a aparecer en el Boruta crudo:
#   - oxidante_sobre_carga_total: documentada (hallazgos.md seccion 10.1) como NO valida en
#     Reduccion (denominador feed_total_kgh~=0 en 27.3% de escalones de Reduccion, ratio
#     inestable) -- el pool generico del script 02 no filtro esto por fase, hay que hacerlo aqui.
#   - interaccion_lanza_x_gn: usa posicion_vertical_lanza_mm CONTEMPORANEA. Si esa variable
#     deja de ofrecerse como ACCION optimizable (ver mas abajo, motivado por la triple
#     evidencia de inestabilidad: SHAP Pasos 7/13b/27/42/45, no-monotonia Pasos 46-47, y ahora
#     Boruta ~0% aqui), esta interaccion ya no es reconstruible de forma coherente dentro de
#     `recalcular_derivadas` sin asumir un valor de accion que ya no se optimiza.
CANDIDATAS_V2 = [
    "ley_sn_escoria_pct_prev", "log_ratio_sn_sio2_prev", "interaccion_Sn_x_FeO_prev",
    "interaccion_C_x_Sn_prev", "tasa_gn_nm3_min",
    "ratio_sn_sio2_prev", "log_ratio_sn_feo_prev", "ratio_sn_feo_prev",
    "interaccion_C_x_exceso_o2", "posicion_vertical_lanza_mm_prev",
    "tasa_feed_Carbon_kg_min", "interaccion_C_x_FeO_prev",
]
CANDIDATAS_V2 = [c for c in CANDIDATAS_V2 if c in df_base.columns]

sub_dev = df_base.loc[(df_base["fase_proceso"] == FASE) & df_base["Batch"].isin(batches_dev)]


def evaluar_set(features, nombre):
    cols = list(dict.fromkeys(features + [TARGET, "d_ley_sn_escoria_pct", "ley_sn_escoria_pct_prev", "Batch"]))
    d_dev = df_base.loc[(df_base["fase_proceso"] == FASE) & df_base["Batch"].isin(batches_dev), cols].dropna(subset=features + [TARGET])
    d_lb = df_base.loc[(df_base["fase_proceso"] == FASE) & df_base["Batch"].isin(batches_lockbox), cols].dropna(subset=features + [TARGET])

    oof = mp.oof_groupkfold(d_dev, features, TARGET)
    pred_oof_fisico = mp.convertir_a_delta_sn_fisico(FASE, oof, d_dev["ley_sn_escoria_pct_prev"].to_numpy())

    res_oot = mp.oot_walkforward(d_dev, features, TARGET, orden_batches)
    valido = res_oot["pred_oot"].notna()
    pred_oot_fisico = mp.convertir_a_delta_sn_fisico(FASE, res_oot.loc[valido, "pred_oot"].to_numpy(), res_oot.loc[valido, "ley_sn_escoria_pct_prev"].to_numpy())

    m_final = xgb.XGBRegressor(**mp.PARAMS_XGB)
    m_final.fit(d_dev[features], d_dev[TARGET])
    pred_lb_fisico = mp.convertir_a_delta_sn_fisico(FASE, m_final.predict(d_lb[features]), d_lb["ley_sn_escoria_pct_prev"].to_numpy())

    print(f"\n[{nombre}] k={len(features)} features: {features}")
    print(f"  n_dev={len(d_dev)}  n_lockbox={len(d_lb)}")
    print(f"  R2 OOF (fisico)     = {r2_score(d_dev['d_ley_sn_escoria_pct'], pred_oof_fisico):.4f}   MAE={mean_absolute_error(d_dev['d_ley_sn_escoria_pct'], pred_oof_fisico):.4f}")
    print(f"  R2 OOT walk-forward = {r2_score(res_oot.loc[valido, 'd_ley_sn_escoria_pct'], pred_oot_fisico):.4f}")
    print(f"  R2 LOCKBOX (fisico) = {r2_score(d_lb['d_ley_sn_escoria_pct'], pred_lb_fisico):.4f}   MAE={mean_absolute_error(d_lb['d_ley_sn_escoria_pct'], pred_lb_fisico):.4f}")
    return d_dev, features


print("=" * 80)
evaluar_set(mp.FEATURES_STATE_ACTION[FASE], "PRODUCCION ACTUAL")
d_dev_v2, _ = evaluar_set(CANDIDATAS_V2, "V2 CANDIDATO (Boruta confirmada+tentativa, sin decorrelacionar)")

# --- Decorrelacionar V2: pares |rho|>0.85, eliminar el miembro menos fundamental ---
print("\nPares con |Spearman|>0.85 dentro de CANDIDATAS_V2:")
corr_pares = []
for i, a in enumerate(CANDIDATAS_V2):
    for b in CANDIDATAS_V2[i + 1:]:
        rho, _ = spearmanr(d_dev_v2[a], d_dev_v2[b], nan_policy="omit")
        if abs(rho) > 0.85:
            corr_pares.append((a, b, rho))
            print(f"  {a} <-> {b}: rho={rho:.3f}")

# Preferencia manual (igual criterio que Pasos 20-27: preferir niveles/interacciones
# interpretables sobre log-ratios/ratios cuando sean redundantes)
PREFERENCIA_DROP = ["log_ratio_sn_sio2_prev", "log_ratio_sn_feo_prev", "ratio_sn_feo_prev", "ratio_sn_sio2_prev"]
a_quitar = set()
for a, b, rho in corr_pares:
    for candidato in PREFERENCIA_DROP:
        if candidato in (a, b):
            a_quitar.add(candidato)
            break
CANDIDATAS_V2_DECORR = [c for c in CANDIDATAS_V2 if c not in a_quitar]
print(f"\nEliminadas por redundancia: {sorted(a_quitar)}")
evaluar_set(CANDIDATAS_V2_DECORR, "V2 DECORRELACIONADO")

print("\nListo.")
