"""E18-02 tarea 6: traduccion de theta (log) a kg usando los inventarios tipicos. Derivada local:
Sn_extraido = Sn_disp*(1-e^-ln_sn_dep) => d(Sn_extraido)/d(ln_sn_dep) = Sn_inv[t] (inventario de Sn AL CIERRE del
escalon). FeO_extraido = FeO_inv_prev-FeO_inv[t] => d(FeO_extraido)/d(ln_feo_ret) = -FeO_inv[t] (inventario de FeO
al cierre). Se usa la mediana del inventario de CIERRE por tramo de orden como escala (aproximacion local, valida
para los +-1 Nm3/min o +-1 kg/min tipicos del desvio). Selectividad = kg FeO reducido / kg Sn agotado. Balance a
metal: kg_Sn_neto = kg_Sn_agotado - 0.6*kg_FeO_reducido (0.6 kg Sn atrapado en dross de Fe por kg de FeO reducido a
Fe metalico; dross ~39% Sn)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import e18_02_lib as L

SAL = Path(__file__).parent
KG_SN_ATRAPADO_POR_KG_FEO = 0.6


def main():
    r, bt = L.cargar()
    r = r.assign(tramo=r[L.ORDEN].map({0: "R0_R1_temprano", 1: "R0_R1_temprano", 2: "R2_R3_tardio", 3: "R2_R3_tardio"}))
    inv = r.groupby("tramo")[["m6_sn_inv_kg", "m6_feo_inv_kg"]].median()
    inv_all = r[["m6_sn_inv_kg", "m6_feo_inv_kg"]].median()
    inv.loc["global"] = inv_all
    inv.to_csv(SAL / "e18_02_inventarios_tipicos.csv")
    print("=== inventario tipico de cierre (mediana, kg) ===")
    print(inv.round(0).to_string())

    het_o = pd.read_csv(SAL / "e18_02_heterogeneidad_orden.csv")
    het_o = het_o[het_o["target"].isin(["sn_dep", "feo_ret"])]

    filas = []
    for _, row in het_o.iterrows():
        tramo = row["tramo"]
        signo_kg = 1.0 if row["target"] == "sn_dep" else -1.0   # feo_ret: kg reducido = -Delta(ln_feo_ret)*inv
        col = "m6_sn_inv_kg" if row["target"] == "sn_dep" else "m6_feo_inv_kg"
        escala = inv.loc[tramo, col]
        filas.append(dict(target=row["target"], palanca=row["palanca"], tramo=tramo,
                           theta=row["theta"], ci_lo=row["ci_lo"], ci_hi=row["ci_hi"],
                           inventario_tipico_kg=escala,
                           kg_por_unidad=signo_kg * row["theta"] * escala,
                           kg_ci_lo=signo_kg * row["ci_lo"] * escala if signo_kg > 0 else signo_kg * row["ci_hi"] * escala,
                           kg_ci_hi=signo_kg * row["ci_hi"] * escala if signo_kg > 0 else signo_kg * row["ci_lo"] * escala))
    kg = pd.DataFrame(filas)
    kg.to_csv(SAL / "e18_02_kg_por_unidad.csv", index=False)
    print("\n=== kg adicionales por +1 Nm3/min (GN) o +1 kg/min (carbon), por tramo ===")
    print(kg.round(2).to_string(index=False))

    # selectividad marginal y balance a metal
    piv = kg.pivot_table(index=["palanca", "tramo"], columns="target", values="kg_por_unidad")
    piv = piv.rename(columns={"sn_dep": "kg_Sn_agotado", "feo_ret": "kg_FeO_reducido"})
    piv["selectividad_FeO_por_Sn"] = piv["kg_FeO_reducido"] / piv["kg_Sn_agotado"]
    piv["kg_Sn_atrapado_en_dross"] = KG_SN_ATRAPADO_POR_KG_FEO * piv["kg_FeO_reducido"]
    piv["kg_Sn_neto_a_metal"] = piv["kg_Sn_agotado"] - piv["kg_Sn_atrapado_en_dross"]
    piv = piv.reset_index()
    piv.to_csv(SAL / "e18_02_balance_sn_metal.csv", index=False)
    print("\n=== selectividad y balance de Sn a metal por palanca x tramo ===")
    print(piv.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
