"""E12-01 tarea 0 (construccion): duracion real de R3 y F6 (fecha_final - fecha_inicio, minutos), residuo respecto de
E[duracion | estado al inicio del escalon] (HGB cross-fitted por batch: DEV GroupKFold(5), lockbox = modelo entrenado con
todo DEV), estandarizado -> zD_R3, zD_F6. Guarda tabla por batch con exposiciones + variables auxiliares (hora, turno,
ley_sn al inicio de R3/fin de R2, desvios, targets de escalon de R3 para el mecanismo) en `e12_01_exposiciones.csv`.
"""
import sys, time
sys.path.insert(0, "experimentos/v9")
sys.path.insert(0, "experimentos/v8")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import v9_lib as L, v8_lib as L8
import modelo_predictivo_v8 as mp8

SALIDA = L.RAIZ / "experimentos" / "v10"
SEED = 42


def residuo_duracion(base: pd.DataFrame, fase: str, orden: int, estado: list[str], n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    d = base[(base["fase_proceso"] == fase) & (base["orden_escalon_fase"] == orden)].copy().reset_index(drop=True)
    d["dur_min"] = (pd.to_datetime(d["fecha_final"]) - pd.to_datetime(d["fecha_inicio"])).dt.total_seconds() / 60.0
    dev, lockbox = L8.mp.split_dev_lockbox(base)
    es_dev = d["Batch"].isin(dev).to_numpy()
    ok = d[estado].notna().all(axis=1).to_numpy() & d["dur_min"].notna().to_numpy()
    ahat = np.full(len(d), np.nan)
    idx_dev = np.where(es_dev & ok)[0]
    for tr, te in GroupKFold(n_splits).split(idx_dev, groups=d["Batch"].to_numpy()[idx_dev]):
        m = L8.hgb_nuisance(seed).fit(d[estado].iloc[idx_dev[tr]], d["dur_min"].iloc[idx_dev[tr]])
        ahat[idx_dev[te]] = m.predict(d[estado].iloc[idx_dev[te]])
    idx_lb = np.where(~es_dev & ok)[0]
    r2_lb = np.nan
    if len(idx_lb):
        m = L8.hgb_nuisance(seed).fit(d[estado].iloc[idx_dev], d["dur_min"].iloc[idx_dev])
        ahat[idx_lb] = m.predict(d[estado].iloc[idx_lb])
        r2_lb = r2_score(d["dur_min"].iloc[idx_lb], ahat[idx_lb])
    r2_dev = r2_score(d["dur_min"].iloc[idx_dev], ahat[idx_dev])
    d["dur_hat"] = ahat
    d["res_dur"] = d["dur_min"] - d["dur_hat"]
    sd = float(d.loc[es_dev, "res_dur"].std())
    d["zD"] = d["res_dur"] / sd
    print(f"{fase} orden {orden}: n={len(d)} (dev={es_dev.sum()}, lb={(~es_dev).sum()}) R2_OOF_dev={r2_dev:.3f} R2_lockbox={r2_lb:.3f} sd_res={sd:.2f} min")
    return d, r2_dev, r2_lb, sd


if __name__ == "__main__":
    t0 = time.time()
    df = L8.cargar_df()
    base = mp8._preparar_base(df)
    b = L8.construir_batch(df)
    if "Batch" in b.columns:
        b = b.set_index("Batch")

    r3, r2_dev_r3, r2_lb_r3, sd_r3 = residuo_duracion(base, "Reducción", 3, L.ESTADO_R)
    f6, r2_dev_f6, r2_lb_f6, sd_f6 = residuo_duracion(base, "Fusión", 6, L.ESTADO_F)

    # --- targets de escalon de R3 (mecanismo, tarea 2) y estado de F6->R residual duracion x tasas
    cols_r3 = ["Batch", "fecha_inicio", "fecha_final", "dur_min", "dur_hat", "res_dur", "zD", "desvio_duracion_min",
               "duracion_plan_min", "ley_sn_escoria_pct_prev", "ley_feo_escoria_pct_prev", "m6_ln_sn_dep",
               "m8_ln_feo_ret_dross50", "d_temperatura_horno_celsius", "m6_sn_inv_kg_prev", "m6_sn_inv_kg",
               "m6_feo_inv_kg_prev", "m6_feo_inv_kg", "tasa_feed_Carbon_kg_min", "tasa_gn_nm3_min"]
    R3 = r3[[c for c in cols_r3 if c in r3.columns]].rename(columns={c: f"R3_{c}" for c in cols_r3 if c not in ("Batch",)})
    R3 = R3.set_index("Batch")
    R3["zD_R3"] = r3.set_index("Batch")["zD"]

    cols_f6 = ["Batch", "fecha_inicio", "fecha_final", "dur_min", "dur_hat", "res_dur", "zD", "desvio_duracion_min",
               "duracion_plan_min"]
    F6 = f6[[c for c in cols_f6 if c in f6.columns]].rename(columns={c: f"F6_{c}" for c in cols_f6 if c not in ("Batch",)})
    F6 = F6.set_index("Batch")
    F6["zD_F6"] = f6.set_index("Batch")["zD"]

    t = b.join(R3, how="left").join(F6, how="left")
    t["R3_hora"] = pd.to_datetime(t["R3_fecha_inicio"]).dt.hour
    t["F6_hora"] = pd.to_datetime(t["F6_fecha_inicio"]).dt.hour
    t["R3_turno"] = pd.cut(t["R3_hora"], [-1, 7, 15, 23], labels=["00-08", "08-16", "16-24"])
    t["F6_turno"] = pd.cut(t["F6_hora"], [-1, 7, 15, 23], labels=["00-08", "08-16", "16-24"])
    t = t.sort_values("idx_cronologico")
    t.to_csv(SALIDA / "e12_01_exposiciones.csv")

    resumen = pd.DataFrame([
        {"escalon": "R3", "r2_oof_dev": r2_dev_r3, "r2_lockbox": r2_lb_r3, "sd_res_min": sd_r3},
        {"escalon": "F6", "r2_oof_dev": r2_dev_f6, "r2_lockbox": r2_lb_f6, "sd_res_min": sd_f6},
    ])
    resumen.to_csv(SALIDA / "e12_01_r2_duracion.csv", index=False)
    print(resumen)
    print(f"filas tabla batch: {t.shape}  {time.time()-t0:.0f}s")
