import sys; sys.path.insert(0,'experimentos/v9')
import v9_lib as L, pandas as pd, numpy as np
import v8_lib as L8, modelo_predictivo_v8 as mp8
df=L8.cargar_df(); res=L.residuos_escalon(df); base=mp8._preparar_base(df)
key=["Batch","fase_proceso","orden_escalon_fase"]
cols=["temperatura_horno_celsius_prev","termocupla_media_celsius_prev","ley_feo_escoria_pct_prev","ley_sn_escoria_pct_prev","m6_sn_inv_kg_prev","m6_feo_inv_kg_prev"]
res=res.merge(base[key+cols].drop_duplicates(key),on=key,how="left")
b=L8.construir_batch(df); b=b.set_index("Batch") if "Batch" in b.columns else b
R=res[res.fase_proceso=="Reducción"]
print(R.groupby("es_dev")[cols].median().round(1).to_string())
for mod in cols:
    med=R.loc[R.es_dev,mod].groupby(R.orden_escalon_fase).median()
    hi=(R[mod]>R.orden_escalon_fase.map(med))
    ex={}
    for p in ("gn","carbon"):
        ex[f"x_{p}_hi"]=(R[f"z__{p}"].where(hi,0)).groupby(R.Batch).mean(); ex[f"x_{p}_lo"]=(R[f"z__{p}"].where(~hi,0)).groupby(R.Batch).mean()
    t=b.join(pd.DataFrame(ex)); X=list(ex)
    for nm,d in (("DEV",t[t.es_dev]),("LB",t[~t.es_dev])):
        ctr=[c for c in L.CONTROLES if nm=="DEV" or c!="espesor_ladrillo_norm_mm"]
        dd=d.dropna(subset=X+ctr+[L.KPI]); f=L.ols_hc3(dd[L.KPI],dd[X+ctr])
        print(mod[:22],nm,len(dd)," ".join(f"{c[2:]}:{f.params[c]:+.2f}({f.pvalues[c]:.2f})" for c in X))
