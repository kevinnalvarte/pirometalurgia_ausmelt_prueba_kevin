"""E18-03 C -- expediente de seguridad del asesor v10 (replay_prospectivo real, JS, W=100/paso10)."""
import sys, time
sys.path.insert(0, "experimentos/v16")
import numpy as np
import pandas as pd
from scipy import stats as sst
import e18_03_lib as E
import asesor_receta_v10 as A

t0 = time.time()
SAL = "experimentos/v16"
r, bt = E.cargar()
print("corriendo replay_prospectivo()...")
esc, pb = A.replay_prospectivo(log=lambda *a: None)
print(f"escalones recomendados: {len(esc)}  [{time.time()-t0:.0f}s]")

extra = r[["Batch", "orden_escalon_fase", "presion_punta_lanza_kpa", "tiro_horno_pct", "d_temperatura_horno_celsius",
           "desvio_duracion_min", "duracion_plan_min", "temperatura_horno_celsius", "plan__o2", "tasa_o2_nm3_min",
           "tasa_gn_nm3_min", "tasa_feed_Carbon_kg_min"]]
esc = esc.merge(extra, on=["Batch", "orden_escalon_fase"], how="left")

# ---------------------------------------------------------------- C1: rango historico y tamano del paso
hist_q = r.groupby("orden_escalon_fase").agg(
    gn_P1=("tasa_gn_nm3_min", lambda s: s.quantile(0.01)), gn_P5=("tasa_gn_nm3_min", lambda s: s.quantile(0.05)),
    gn_P95=("tasa_gn_nm3_min", lambda s: s.quantile(0.95)), gn_P99=("tasa_gn_nm3_min", lambda s: s.quantile(0.99)),
    carbon_P1=("tasa_feed_Carbon_kg_min", lambda s: s.quantile(0.01)), carbon_P5=("tasa_feed_Carbon_kg_min", lambda s: s.quantile(0.05)),
    carbon_P95=("tasa_feed_Carbon_kg_min", lambda s: s.quantile(0.95)), carbon_P99=("tasa_feed_Carbon_kg_min", lambda s: s.quantile(0.99)),
    o2_P1=("tasa_o2_nm3_min", lambda s: s.quantile(0.01)), o2_P5=("tasa_o2_nm3_min", lambda s: s.quantile(0.05)))
hist_q.to_csv(f"{SAL}/e18_03_C1_rango_historico.csv")

filas_c1 = []
for p in ("gn", "carbon"):
    o1, o99 = esc["orden_escalon_fase"].map(hist_q[f"{p}_P1"]), esc["orden_escalon_fase"].map(hist_q[f"{p}_P99"])
    o5, o95 = esc["orden_escalon_fase"].map(hist_q[f"{p}_P5"]), esc["orden_escalon_fase"].map(hist_q[f"{p}_P95"])
    rec = esc[f"rec__{p}"]
    fuera_1_99 = 100 * ((rec < o1) | (rec > o99)).mean()
    fuera_5_95 = 100 * ((rec < o5) | (rec > o95)).mean()
    paso = (esc[f"dev_rec__{p}"]).abs()
    paso_pct = 100 * paso / esc[f"receta__{p}"]
    filas_c1.append({"palanca": p, "pct_fuera_P1_P99": fuera_1_99, "pct_fuera_P5_P95": fuera_5_95,
                      "paso_max_fisico": paso.max(), "paso_max_pct_receta": paso_pct.max(), "paso_P95_pct_receta": paso_pct.quantile(0.95)})
c1 = pd.DataFrame(filas_c1); c1.to_csv(f"{SAL}/e18_03_C1_resumen.csv", index=False)
print("\n== C1: rango historico y tamano del paso ==")
print(c1.round(3).to_string())

# ---------------------------------------------------------------- C2: gemelos historicos (GN)
esc["twin_gn"] = (esc["z_hist__gn"] - esc["z_rec__gn"]).abs() <= 0.25
twin_batch = esc.groupby("Batch")["twin_gn"].mean()
es_twin_batch = twin_batch >= 0.5
print(f"\nescalones 'gemelos' (|ejecutado - recomendado| <= 0.25 sd): {100*esc['twin_gn'].mean():.1f}% ; batches gemelos (>=2/4 escalones): {es_twin_batch.sum()} de {len(es_twin_batch)}")

indic_esc = {"dT_escalon": "d_temperatura_horno_celsius", "presion_punta_lanza_kpa": "presion_punta_lanza_kpa",
             "tiro_horno_pct": "tiro_horno_pct", "desvio_duracion_min": "desvio_duracion_min"}
filas_c2 = []
for nombre, col in indic_esc.items():
    a = esc.loc[esc["twin_gn"], col].dropna(); b = esc.loc[~esc["twin_gn"], col].dropna()
    if len(a) > 5 and len(b) > 5:
        st, p = sst.mannwhitneyu(a, b, alternative="two-sided")
        filas_c2.append({"indicador": nombre, "nivel": "escalon", "n_twin": len(a), "n_resto": len(b),
                          "media_twin": a.mean(), "media_resto": b.mean(), "p": p})

indic_b = {"dross_f": "f_dross", "polvo_f": "f_polvo", "sn_escoria_frac": "sn_perdido_escoria_frac", "KPI": E.KPI}
bt2 = bt.join(es_twin_batch.rename("es_twin"))
for nombre, col in indic_b.items():
    a = bt2.loc[bt2["es_twin"] == True, col].dropna(); b = bt2.loc[bt2["es_twin"] == False, col].dropna()
    if len(a) > 5 and len(b) > 5:
        st, p = sst.mannwhitneyu(a, b, alternative="two-sided")
        filas_c2.append({"indicador": nombre, "nivel": "batch", "n_twin": len(a), "n_resto": len(b),
                          "media_twin": a.mean(), "media_resto": b.mean(), "p": p})
c2 = pd.DataFrame(filas_c2)
c2["q_BH"] = E.bh(c2["p"].tolist())
c2.to_csv(f"{SAL}/e18_03_C2_gemelos.csv", index=False)
print("\n== C2: gemelos historicos (GN) vs resto -- indicadores de seguridad/proceso y de batch ==")
print(c2.round(4).to_string())

# ---------------------------------------------------------------- C3: peor caso (GN <=-1sd en los 4 escalones)
piv = esc.pivot(index="Batch", columns="orden_escalon_fase", values="z_hist__gn")
peor_4 = (piv <= -1.0).all(axis=1)
peor_3 = (piv <= -1.0).sum(axis=1) >= 3
print(f"\nbatches con GN ejecutado <=-1sd en los 4 escalones: {int(peor_4.sum())} de {len(piv)}  (>=3 de 4: {int(peor_3.sum())})")
filas_c3 = []
for nombre, mask in (("4_de_4", peor_4), ("al_menos_3_de_4", peor_3)):
    bset = mask[mask].index
    if len(bset) >= 3:
        sub = bt.loc[bt.index.isin(bset)]; resto = bt.loc[~bt.index.isin(bset)]
        filas_c3.append({"criterio": nombre, "n": len(bset), "T_media_R_sub": sub["T_media_R"].mean(), "T_media_R_resto": resto["T_media_R"].mean(),
                          "dross_sub": sub["f_dross"].mean(), "dross_resto": resto["f_dross"].mean(),
                          "KPI_sub": sub[E.KPI].mean(), "KPI_resto": resto[E.KPI].mean()})
    else:
        filas_c3.append({"criterio": nombre, "n": len(bset), "T_media_R_sub": np.nan, "T_media_R_resto": np.nan,
                          "dross_sub": np.nan, "dross_resto": np.nan, "KPI_sub": np.nan, "KPI_resto": np.nan})
c3 = pd.DataFrame(filas_c3); c3.to_csv(f"{SAL}/e18_03_C3_peor_caso.csv", index=False)
print("\n== C3: peor caso ==")
print(c3.round(3).to_string())

# ---------------------------------------------------------------- C4: piso de O2
esc["rec_o2_acompanado"] = (esc["plan__o2"] + 2 * esc["dev_rec__gn"]).clip(lower=0)
esc["rec_o2_sin_piso"] = esc["plan__o2"] + 2 * esc["dev_rec__gn"]
o1 = esc["orden_escalon_fase"].map(hist_q["o2_P1"]); o5 = esc["orden_escalon_fase"].map(hist_q["o2_P5"])
pct_neg = 100 * (esc["rec_o2_sin_piso"] < 0).mean()
pct_bajo_p1 = 100 * (esc["rec_o2_acompanado"] < o1).mean()
pct_bajo_p5 = 100 * (esc["rec_o2_acompanado"] < o5).mean()
print(f"\n== C4: O2 acompanando GN (ΔO2=2·ΔGN) ==")
print(f"% escalones con O2 recomendado <0 (sin piso): {pct_neg:.1f}%")
print(f"% escalones con O2 recomendado (con piso 0) < P1 historico del orden: {pct_bajo_p1:.1f}%")
print(f"% escalones con O2 recomendado (con piso 0) < P5 historico del orden: {pct_bajo_p5:.1f}%")
piso_propuesto = hist_q["o2_P5"]
print("Piso propuesto (P5 historico del orden, Nm3/min):"); print(piso_propuesto.round(2).to_string())
pd.DataFrame([{"pct_o2_negativo_sin_piso": pct_neg, "pct_bajo_P1_con_piso0": pct_bajo_p1, "pct_bajo_P5_con_piso0": pct_bajo_p5}]).to_csv(f"{SAL}/e18_03_C4_o2.csv", index=False)
piso_propuesto.to_csv(f"{SAL}/e18_03_C4_piso_o2_propuesto.csv")

esc.to_csv(f"{SAL}/e18_03_C_escalones_completo.csv", index=False)
print(f"\n[{time.time()-t0:.0f}s] listo")
