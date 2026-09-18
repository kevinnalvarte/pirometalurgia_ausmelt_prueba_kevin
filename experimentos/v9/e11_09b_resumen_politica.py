import pandas as pd, numpy as np
e=pd.read_csv("experimentos/v9/e11_09_replay_escalones.csv"); pb=pd.read_csv("experimentos/v9/e11_09_replay_por_batch.csv",index_col=0)
e["periodo"]=pd.cut(e.inicio_bloque,[0,200,300,400],labels=["b100-199","b200-299","b300-361(lockbox)"])
f=[]
for (per,o),g in e.groupby(["periodo","orden_escalon_fase"],observed=True):
    r={"periodo":per,"orden":o,"n":len(g)}
    for p in ("gn","carbon"):
        r[f"{p}_hab"]=g[f"hab__{p}"].median(); r[f"{p}_delta_med"]=g[f"delta__{p}"].mean(); r[f"{p}_%rec"]=(g[f"motivo__{p}"]=="recomendar").mean()*100
        r[f"{p}_%baja"]=(g[f"delta__{p}"]<-1e-6).mean()*100; r[f"{p}_%sube"]=(g[f"delta__{p}"]>1e-6).mean()*100
    f.append(r)
t=pd.DataFrame(f).round(2); t.to_csv("experimentos/v9/e11_09b_resumen_politica.csv",index=False); pd.set_option("display.width",250); print(t.to_string())
print("motivos gn:",e.motivo__gn.value_counts(normalize=True).round(2).to_dict())
print("efecto fisico esperado por batch (suma R): ln_sn_dep %.4f  ln_feo_ret %.4f"%(pb.efecto_ln_sn_dep.mean(),pb.efecto_ln_feo_ret.mean()))
print("valor_rec pp/batch: media %.2f  P10 %.2f P90 %.2f ; por periodo:"%(pb.valor_rec_pp.mean(),pb.valor_rec_pp.quantile(.1),pb.valor_rec_pp.quantile(.9)), pb.groupby("es_dev").valor_rec_pp.mean().round(2).to_dict())
print("sd_res tipico:", {p:e[f'delta__{p}'].abs().max().round(2) for p in ('gn','carbon')})
