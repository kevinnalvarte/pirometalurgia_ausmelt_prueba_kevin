"""B-05b (Fable) -- feature de ESTADO derivada del monitor de Fe (H2 de b05): avance de la perdida de FeO
respecto al inicio de Reduccion, conocido al cierre de t-1:
    b6_avance_feo_prev = 1 - feo_inv_prev[t] / feo_inv_prev[R0]     (trazador CaO v3; STATE)
    b6_feo_perdida_iniciada_prev = 1[b6_avance_feo_prev > 0.03]       (STATE, bandera)
Se anaden al estado del PLM v4 de Reduccion (feo_kg, sn_kg y SDI) y se compara R2 OOF DEV / lockbox y theta.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bal_features as bal  # noqa: E402
sys.path.insert(0, str(bal.RAIZ / "experimentos" / "search_targets"))
import st_targets as st  # noqa: E402
import modelo_predictivo_v4 as mp4  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402


def main():
    df = st.construir_df_targets(bal.cargar_df())
    df = mp4.agregar_columnas_v4(df).sort_values(["Batch", "fecha_inicio"]).reset_index(drop=True)
    red = df["fase_proceso"] == "Reducción"
    feo_prev = df["feo_inventario_escoria_est_kg_prev"]
    # inventario al inicio de Reduccion = feo_prev del primer escalon de Reduccion (cierre de F6)
    ini = feo_prev.where(red & df["es_primer_escalon_fase"].astype(bool)).groupby(df["Batch"]).transform("max")
    df["b6_avance_feo_prev"] = (1 - feo_prev / ini).where(red)
    df["b6_feo_perdida_iniciada_prev"] = (df["b6_avance_feo_prev"] > 0.03).astype(float).where(red)
    dev, lock = mp.split_dev_lockbox(df)
    filas = []
    for clave, target in [("feo_kg", "feo_extraido_est_kg"), ("sn_kg", "sn_extraido_est_kg"), ("feo_kg", "st_R22_sdi")]:
        est0 = mp4.ESTADO_V4[("Reducción", clave)]
        pal = mp4.PALANCAS_V4[("Reducción", clave)]
        for nombre, est in [("v4", est0), ("v4+avance_feo", est0 + ["b6_avance_feo_prev"]),
                            ("v4+avance_feo+bandera", est0 + ["b6_avance_feo_prev", "b6_feo_perdida_iniciada_prev"])]:
            d = df[red & ~df["es_primer_escalon_batch"].astype(bool)].dropna(subset=est + pal + [target])
            dd, dl = d[d["Batch"].isin(dev)], d[d["Batch"].isin(lock)]
            m = mp4.ModeloPLM(est, pal, target).fit(dd, n_splits=5, n_boot=100)
            pred_l = m.predict(dl)
            r2_l = 1 - np.sum((dl[target] - pred_l) ** 2) / np.sum((dl[target] - dl[target].mean()) ** 2)
            t = m.tabla_theta()
            th_c = t.loc[[p for p in pal if "Carbon" in p or "Cx" in p]]
            filas.append(dict(target=target, estado=nombre, n_dev=len(dd), n_lock=len(dl), r2_oof=m.r2_oof_interno, r2_lockbox=r2_l,
                              theta_carbon=";".join(f"{i}:{r.theta_1sd:+.2f}[{r.ci_lo_1sd:+.2f},{r.ci_hi_1sd:+.2f}]" for i, r in th_c.iterrows())))
            print(filas[-1])
    out = pd.DataFrame(filas)
    out.to_csv(HERE / "b05b_avance_feo_estado.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
