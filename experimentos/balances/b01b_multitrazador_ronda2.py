"""B-01b (Fable) -- ronda 2 del multi-trazador: consensos alternativos (sin CaO; pesos empiricos; solo SiO2+Al2O3)
frente al trazador CaO unico. Metricas: autocorrelacion lag-1 de FeO_ext (huella de ruido), Spearman por batch de
sum ln(FeO_fin/FeO_ini) y del SDI (w=10) con el rendimiento proxy en DEV, y R2 OOF HGB del FeO extraido.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bal_features as bal  # noqa: E402
import modelo_prescriptivo as mp  # noqa: E402
import modelo_predictivo_v3 as mp3  # noqa: E402
import feature_engineering as fe  # noqa: E402

VARIANTES = {
    "single_cao": (("cao",), {"cao": 1.0}),
    "multi4_asumido": (("cao", "sio2", "al2o3", "mgo"), None),
    "multi4_empirico": (("cao", "sio2", "al2o3", "mgo"), {"cao": 0.245, "sio2": 0.353, "al2o3": 0.320, "mgo": 0.082}),
    "multi3_sin_cao": (("sio2", "al2o3", "mgo"), {"sio2": 0.47, "al2o3": 0.42, "mgo": 0.11}),
    "multi2_sio2_al2o3": (("sio2", "al2o3"), {"sio2": 0.52, "al2o3": 0.48}),
}


def main():
    base = pd.read_pickle(bal.CACHE)
    dev, lock = mp.split_dev_lockbox(base)
    S = [c for c in dict.fromkeys(mp3.STATE_V3["Reducción"] + mp3.CONTEXT_V3["Reducción"]) if c in base.columns]
    S = fe.filtrar_features_seguras(S)
    filas = []
    for nombre, (inertes, pesos) in VARIANTES.items():
        sd_bak = dict(bal.SD_REL_ENSAYO)
        if pesos:
            for k, w in pesos.items():
                bal.SD_REL_ENSAYO[k] = 1 / np.sqrt(w)
        d = bal.agregar_multitrazador(base, inertes=inertes)
        bal.SD_REL_ENSAYO.update(sd_bak)
        if nombre == "single_cao":
            # referencia v3: inventario re-anclado por CaO acumulado (no encadenado)
            d["feo_ext_var"] = d["feo_extraido_est_kg"].where(d["b6_feo_extraido_multi_kg"].notna())
            fp = d["feo_inventario_escoria_est_kg_prev"]; fi = d["feo_inventario_escoria_est_kg"]
            sp = d["sn_inventario_escoria_est_kg_prev"]; si = d["sn_inventario_escoria_est_kg"]
            ok = fp.gt(100) & fi.gt(100) & sp.gt(20) & si.gt(20) & d["b6_feo_extraido_multi_kg"].notna()
            d["ln_feo_ret_var"] = np.log(fi / fp).where(ok)
            d["sdi_var"] = (np.log(sp / si) + 10 * np.log(fi / fp)).where(ok)
        else:
            d["feo_ext_var"] = d["b6_feo_extraido_multi_kg"]
            d["ln_feo_ret_var"] = d["b6_ln_feo_ret_multi"]
            d["sdi_var"] = d["b6_sdi_multi"]
        red = d[(d["fase_proceso"] == "Reducción")].sort_values(["Batch", "fecha_inicio"])
        # lag-1 dentro del batch
        x = red["feo_ext_var"]; xl = x.groupby(red["Batch"]).shift(1)
        m = x.notna() & xl.notna()
        ac1 = stats.pearsonr(x[m], xl[m])[0]
        # batch
        g = red.groupby("Batch")
        b = pd.DataFrame({"lnfeo": g["ln_feo_ret_var"].sum(min_count=1), "sdi": g["sdi_var"].sum(min_count=1),
                          "kpi": g["rendimiento_proxy_batch"].first()})
        bd = b[b.index.isin(dev)].dropna()
        rho_feo, p_feo = stats.spearmanr(bd["lnfeo"], bd["kpi"])
        rho_sdi, p_sdi = stats.spearmanr(bd["sdi"], bd["kpi"])
        bl = b[b.index.isin(lock)].dropna()
        rho_sdi_l = stats.spearmanr(bl["sdi"], bl["kpi"])[0]
        # HGB OOF DEV del FeO_ext
        dd = red[red["Batch"].isin(dev)].dropna(subset=S + ["feo_ext_var"])
        oof = np.full(len(dd), np.nan)
        for tr, te in GroupKFold(5).split(dd, groups=dd["Batch"]):
            mdl = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                                l2_regularization=1.0, random_state=42).fit(dd[S].iloc[tr], dd["feo_ext_var"].iloc[tr])
            oof[te] = mdl.predict(dd[S].iloc[te])
        r2 = 1 - np.sum((dd["feo_ext_var"] - oof) ** 2) / np.sum((dd["feo_ext_var"] - dd["feo_ext_var"].mean()) ** 2)
        filas.append(dict(variante=nombre, n_escalones=int(m.sum()), autocorr_lag1=ac1, sd_feo_ext=float(x.std()),
                          rho_lnfeo_dev=rho_feo, p=p_feo, rho_sdi_dev=rho_sdi, p_sdi=p_sdi, rho_sdi_lockbox=rho_sdi_l,
                          n_batch_dev=len(bd), r2_oof_hgb_feo=r2, n_hgb=len(dd)))
        print(filas[-1])
    out = pd.DataFrame(filas)
    out.to_csv(HERE / "b01b_variantes_consenso.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
