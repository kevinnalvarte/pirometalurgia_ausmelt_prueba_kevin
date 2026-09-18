"""E12-01 tarea 1: descriptivo de la duracion de R3 y F6, dependencia de hora/turno/idx_cronologico, outliers."""
import sys
sys.path.insert(0, "experimentos/v9")
import numpy as np, pandas as pd
from scipy import stats
import e11_07_lib as E

t = pd.read_csv("experimentos/v10/e12_01_exposiciones.csv", index_col=0)
pd.set_option("display.width", 200)

filas = []
for esc in ("R3", "F6"):
    dur = t[f"{esc}_dur_min"]
    dev = dur[t["es_dev"]]
    filas.append({"escalon": esc, "muestra": "DEV", "n": dev.notna().sum(), "media": dev.mean(), "sd": dev.std(),
                  "min": dev.min(), "p25": dev.quantile(.25), "mediana": dev.median(), "p75": dev.quantile(.75),
                  "p95": dev.quantile(.95), "max": dev.max()})
desc = pd.DataFrame(filas)
print("--- distribucion duracion real (DEV) ---")
print(desc.round(2).to_string())
desc.to_csv("experimentos/v10/e12_01_descriptivo_duracion.csv", index=False)

# --- outliers: desvio_duracion_min > 30
for esc in ("R3", "F6"):
    desv = t[f"{esc}_desvio_duracion_min"]
    out = desv > 30
    print(f"\n{esc}: outliers (desvio_duracion_min > 30) = {out.sum()} / {desv.notna().sum()} ({100*out.mean():.1f} %); "
          f"desvio max = {desv.max():.0f}, p99 = {desv.quantile(.99):.1f}")

# --- version winsorizada (clip al P1-P99 de DEV) y version filtrada (excluye outliers > 30 min de desvio)
res_rows = []
for esc in ("R3", "F6"):
    dur = t[f"{esc}_dur_min"]; desv = t[f"{esc}_desvio_duracion_min"]
    dev_mask = t["es_dev"]
    lo, hi = dur[dev_mask].quantile(.01), dur[dev_mask].quantile(.99)
    dur_wz = dur.clip(lo, hi)
    dur_filt = dur.where(desv <= 30)
    res_rows.append({"escalon": esc, "n_total": dur.notna().sum(), "n_filtrado": dur_filt.notna().sum(),
                      "media_raw": dur.mean(), "media_winsor": dur_wz.mean(), "media_filtrado": dur_filt.mean(),
                      "sd_raw": dur.std(), "sd_winsor": dur_wz.std(), "sd_filtrado": dur_filt.std()})
    t[f"{esc}_dur_winsor"] = dur_wz
comp = pd.DataFrame(res_rows)
print("\n--- comparacion raw vs winsorizada (P1-P99 DEV) vs filtrada (excluye desvio>30) ---")
print(comp.round(2).to_string())
comp.to_csv("experimentos/v10/e12_01_outliers.csv", index=False)

# --- relacion con hora del dia / turno / idx_cronologico
filas2 = []
for esc in ("R3", "F6"):
    dur = t[f"{esc}_dur_min"]; hora = t[f"{esc}_hora"]; idxc = t["idx_cronologico"]
    m = dur.notna() & hora.notna()
    rho_h, p_h = stats.spearmanr(hora[m], dur[m])
    m2 = dur.notna() & idxc.notna()
    rho_i, p_i = stats.spearmanr(idxc[m2], dur[m2])
    filas2.append({"escalon": esc, "spearman_hora": rho_h, "p_hora": p_h, "spearman_idx_cronologico": rho_i, "p_idx": p_i})
    turno_medias = t.groupby(f"{esc}_turno", observed=True)[f"{esc}_dur_min"].agg(["mean", "std", "count"])
    print(f"\n{esc}: duracion por turno (hora de fecha_inicio):")
    print(turno_medias.round(2).to_string())
rel = pd.DataFrame(filas2)
print("\n--- correlacion duracion ~ hora / idx_cronologico ---")
print(rel.round(4).to_string())
rel.to_csv("experimentos/v10/e12_01_relacion_hora_idx.csv", index=False)
