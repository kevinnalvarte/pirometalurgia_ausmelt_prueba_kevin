import sys; sys.path.insert(0,'experimentos/v9')
import v9_lib as L, pandas as pd, numpy as np
import v8_lib as L8, modelo_predictivo_v8 as mp8
from sklearn.model_selection import GroupKFold
df=L8.cargar_df(); base=mp8._preparar_base(df); dev,lb=L8.mp.split_dev_lockbox(df)
d=base[(base.fase_proceso=="Reducción")&(base.orden_escalon_fase<=3)].copy().reset_index(drop=True)
o2t=d.tasa_o2_nm3_min+0.21*d.tasa_aire_nm3_min
d["gn_quemado"]=np.minimum(d.tasa_gn_nm3_min,o2t/2); d["gn_exceso"]=d.tasa_gn_nm3_min-o2t/2   # >0: combustible sin quemar (reductor); <0: O2 libre
d["o2_libre"]=(-d["gn_exceso"]).clip(lower=0); d["gn_noq"]=d["gn_exceso"].clip(lower=0)
print(d.groupby([d.Batch.isin(dev),"orden_escalon_fase"])[["tasa_gn_nm3_min","gn_quemado","gn_exceso","tasa_feed_Carbon_kg_min"]].median().round(1).to_string())
print("frac gn_exceso>0:",(d.gn_exceso>0).groupby(d.Batch.isin(dev)).mean().round(2).to_dict())
S=L.ESTADO_R; es_dev=d.Batch.isin(dev).to_numpy()
for c in ["gn_quemado","gn_exceso","tasa_gn_nm3_min","tasa_feed_Carbon_kg_min"]:
    ok=d[c].notna().to_numpy(); ah=np.full(len(d),np.nan); i=np.where(es_dev&ok)[0]
    for tr,te in GroupKFold(5).split(i,groups=d.Batch.to_numpy()[i]):
        ah[i[te]]=L8.hgb_nuisance(42).fit(d[S].iloc[i[tr]],d[c].iloc[i[tr]]).predict(d[S].iloc[i[te]])
    j=np.where(~es_dev&ok)[0]; ah[j]=L8.hgb_nuisance(42).fit(d[S].iloc[i],d[c].iloc[i]).predict(d[S].iloc[j])
    r=d[c]-ah; sd=r[es_dev].groupby(d.orden_escalon_fase[es_dev]).std(); d["z_"+c]=r/d.orden_escalon_fase.map(sd)
b=L8.construir_batch(df); b=b.set_index("Batch") if "Batch" in b.columns else b
for grp,ords in (("R",[0,1,2,3]),("Rt",[0,1]),("Rl",[2,3])):
    m=d.orden_escalon_fase.isin(ords)
    ex=pd.DataFrame({k:d.loc[m,"z_"+c].groupby(d.Batch[m]).mean() for k,c in (("heat","gn_quemado"),("exc","gn_exceso"),("C","tasa_feed_Carbon_kg_min"))})
    t=b.join(ex)
    for nm,tt in (("DEV",t[t.es_dev]),("LB",t[~t.es_dev]),("TOT",t)):
        ctr=[c for c in L.CONTROLES if nm!="LB" or c!="espesor_ladrillo_norm_mm"]
        for y in (L.KPI,"f_dross","f_polvo"):
            dd=tt.dropna(subset=list(ex)+ctr+[y]); f=L.ols_hc3(dd[y]*(100 if y[0]=="f" else 1),dd[list(ex)+ctr])
            print(grp,nm,y[:8],len(dd)," ".join(f"{c}:{f.params[c]:+.2f}({f.pvalues[c]:.3f})" for c in ex))
print(d[["z_gn_quemado","z_gn_exceso","z_tasa_gn_nm3_min","z_tasa_feed_Carbon_kg_min"]].corr().round(2).to_string())
print("=== moderacion termica")
for Tcol in ("temperatura_horno_celsius_prev","termocupla_media_celsius_prev"):
    mu,sdv=d.loc[es_dev,Tcol].mean(),d.loc[es_dev,Tcol].std(); d["zT"]=(d[Tcol]-mu)/sdv
    print(Tcol,round(mu),round(sdv), "LB zT mediana",round(d.loc[~es_dev,"zT"].median(),2))
    ex=pd.DataFrame({"heat":d.z_gn_quemado.groupby(d.Batch).mean(),"heatxT":(d.z_gn_quemado*d.zT).groupby(d.Batch).mean(),
                     "C":d.z_tasa_feed_Carbon_kg_min.groupby(d.Batch).mean(),"Tm":d.zT.groupby(d.Batch).mean()})
    ex["Tm2"]=ex.Tm**2
    t=b.join(ex)
    for nm,tt in (("DEV",t[t.es_dev]),("LB",t[~t.es_dev]),("TOT",t)):
        ctr=[c for c in L.CONTROLES if nm!="LB" or c!="espesor_ladrillo_norm_mm"]
        for X in (["heat","heatxT","C"],["heat","heatxT","C","Tm","Tm2"]):
            for y in (L.KPI,"f_dross"):
                dd=tt.dropna(subset=X+ctr+[y]); f=L.ols_hc3(dd[y]*(100 if y[0]=="f" else 1),dd[X+ctr])
                print(" ",nm,y[:8],len(dd)," ".join(f"{c}:{f.params[c]:+.2f}({f.pvalues[c]:.3f})" for c in X))
    # terciles de T (globales sobre 362): efecto del calor por tercil
    q=pd.qcut(d[Tcol],4,labels=False)
    ex2=pd.DataFrame({f"heat_q{k}":d.z_gn_quemado.where(q==k,0).groupby(d.Batch).mean() for k in range(4)}); ex2["C"]=ex["C"]
    tt=b.join(ex2); dd=tt.dropna(subset=list(ex2)+L.CONTROLES+[L.KPI]); f=L.ols_hc3(dd[L.KPI],dd[list(ex2)+L.CONTROLES])
    print("  cuartiles T (TOTAL):",[round(v) for v in d[Tcol].quantile([.25,.5,.75])]," ".join(f"{c}:{f.params[c]:+.2f}({f.pvalues[c]:.3f})" for c in ex2))
