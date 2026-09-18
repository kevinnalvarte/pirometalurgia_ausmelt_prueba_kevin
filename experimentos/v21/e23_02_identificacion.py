"""Identificación exploratoria within batch/orden y auditoría algebraica del balance."""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from e23_01_acoplamiento import ROOT, OUT, STATE, TARGETS

def load():
    df=pd.read_pickle(ROOT/'experimentos/cache/df_v8.pkl')
    rec=pd.read_pickle(ROOT/'experimentos/cache/receta_reduccion.pkl')
    rec=rec[rec.Batch.isin(df.Batch.unique())]
    r=df[df.fase_proceso.eq('Reducción')].merge(rec,on=['Batch','orden_escalon_fase'],validate='one_to_one')
    for p,col in {'gn':'tasa_gn_nm3_min','carbon':'tasa_feed_Carbon_kg_min','o2':'tasa_o2_nm3_min','aire':'tasa_aire_nm3_min'}.items():
        r['d_'+p]=r[col]-r['plan__'+p]
    r['d_ot']=r.d_o2+.21*r.d_aire
    r['d_exceso']=r.d_ot-2*r.d_gn
    r['exceso']=r.tasa_o2_nm3_min+.21*r.tasa_aire_nm3_min-2*r.tasa_gn_nm3_min
    r['C_x_exceso']=r.d_carbon*r.exceso
    r['C_x_GN']=r.d_carbon*r.d_gn
    r['C_x_Sn']=r.d_carbon*r.m6_sn_inv_kg_prev/1000
    for o in range(4):r['gn_'+str(o)]=r.d_gn.where(r.orden_escalon_fase.eq(o),0)
    batches=df.groupby('Batch').fecha_inicio.min().sort_values().index
    r['pos']=r.Batch.map(dict(zip(batches,range(len(batches)))))
    return r.replace([np.inf,-np.inf],np.nan)

def within(d,cols):
    a=d[cols].astype(float).copy()
    for _ in range(50):
        prev=a.to_numpy().copy()
        a-=a.groupby(d.Batch).transform('mean')
        a-=a.groupby(d.orden_escalon_fase).transform('mean')
        if np.max(np.abs(a.to_numpy()-prev))<1e-9:break
    return a

def estimate(d,y,x):
    z=within(d,[y]+x)
    fit=sm.OLS(z[y],z[x]).fit(cov_type='cluster',cov_kwds={'groups':d.Batch})
    # Account for absorbed fixed-effect degrees of freedom in small-sample covariance.
    rank=np.linalg.matrix_rank(z[x]); fe_rank=d.Batch.nunique()+d.orden_escalon_fase.nunique()-1
    factor=np.sqrt((len(d)-rank)/(len(d)-rank-fe_rank))
    se=fit.bse*factor
    from scipy.stats import t
    tv=fit.params/se; p=2*t.sf(np.abs(tv),df=d.Batch.nunique()-1)
    return pd.DataFrame({'variable':x,'theta':fit.params.to_numpy(),'se':se.to_numpy(),
        'ci_lo':fit.params.to_numpy()-t.ppf(.975,d.Batch.nunique()-1)*se.to_numpy(),
        'ci_hi':fit.params.to_numpy()+t.ppf(.975,d.Batch.nunique()-1)*se.to_numpy(),
        'p':p,'n':len(d),'batches':d.Batch.nunique(),'rank':rank})

def run():
    r=load()
    configs={'v11_estructura':['gn_'+str(o) for o in range(4)]+['d_carbon'],
             'conjunto':['gn_'+str(o) for o in range(4)]+['d_carbon','d_exceso','d_aire'],
             'acoplado':['gn_'+str(o) for o in range(4)]+['d_carbon','d_exceso','d_aire','C_x_exceso','C_x_GN','C_x_Sn']}
    cols=list(dict.fromkeys(STATE+sum(configs.values(),[])))
    allresults=[]; coverage=[]
    periods={'DEV':r.pos<299,'LB':r.pos>=299,'T1':r.pos<100,'T2':r.pos.between(100,199),'T3':r.pos.between(200,298)}
    for y in TARGETS:
        d=r.dropna(subset=cols+[y])
        for order in range(4):
            coverage.append(dict(target=y,orden=order,n_total=int(r.orden_escalon_fase.eq(order).sum()),n_valid=int(d.orden_escalon_fase.eq(order).sum())))
        for period,mask in periods.items():
            sub=d.loc[mask.reindex(d.index)]
            for name,actions in configs.items():
                tab=estimate(sub,y,actions+STATE)
                tab=tab[tab.variable.isin(actions)].copy();tab['target']=y;tab['periodo']=period;tab['modelo']=name
                allresults.append(tab)
        print('terminado',y,flush=True)
    results=pd.concat(allresults,ignore_index=True)
    results['q_bh']=np.nan
    for period,ix in results.groupby('periodo').groups.items():
        results.loc[ix,'q_bh']=multipletests(results.loc[ix,'p'],method='fdr_bh')[1]
    results.to_csv(OUT/'e23_02_efectos.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'e23_02_cobertura.csv',index=False)
    # Conditional independent linear variation; descriptive, not overlap certificate.
    support=[]
    d=r[r.pos<299].dropna(subset=cols)
    z=within(d,cols)
    for name in configs['acoplado']:
        other=[x for x in cols if x!=name]
        raw=z[name].to_numpy(); residual=raw-z[other].to_numpy()@np.linalg.lstsq(z[other],raw,rcond=None)[0]
        independent=np.var(residual)/np.var(raw) if np.var(raw)>0 else np.nan
        support.append(dict(variable=name,sd_within=np.std(raw),sd_independiente=np.std(residual),fraccion_var_independiente=independent,vif=1/independent if independent>0 else np.inf))
    pd.DataFrame(support).to_csv(OUT/'e23_02_contraste.csv',index=False)
    # Constant carbon composition and oxidation endpoint produce linear reparameterizations.
    a=r[['d_gn','d_ot','d_carbon']].dropna(); rows=[]
    for fixed in [.6,.8,1.0]:
        for oxygen_per_c in [22.414/12.011/2,22.414/12.011]:
            deficit=a.d_ot-2*a.d_gn-oxygen_per_c*fixed*a.d_carbon
            reconstructed=np.column_stack([a.d_gn,a.d_ot,a.d_carbon])@np.array([-2,1,-oxygen_per_c*fixed])
            rows.append(dict(C_fijo=fixed,O2_por_C=oxygen_per_c,max_error=np.max(np.abs(deficit-reconstructed)),
                rank_base=np.linalg.matrix_rank(a),rank_con_proxy=np.linalg.matrix_rank(np.column_stack([a,deficit]))))
    pd.DataFrame(rows).to_csv(OUT/'e23_02_escenarios_rango.csv',index=False)
    # Exact log-to-inventory transformation vs v11 linearization, conditional on observed endpoint.
    d=r.dropna(subset=['m8_ln_feo_ret_dross50','m6_feo_inv_kg','m6_feo_inv_kg_prev','d_gn'])
    theta={0:-.0035,1:-.0022,2:-.0027,3:-.0058} # archived v11, sensitivity only, not prospective estimates
    delta_gn=-d.d_gn.clip(lower=0); delta_y=d.orden_escalon_fase.map(theta)*delta_gn
    calc=d[['Batch','orden_escalon_fase','pos']].copy()
    calc['delta_gn']=delta_gn;calc['delta_log_fe']=delta_y
    calc['kg_lineal_v11']=delta_y*d.m6_feo_inv_kg_prev
    calc['kg_exacto_condicional']=d.m6_feo_inv_kg*np.expm1(delta_y)
    calc['ratio_exacto_lineal']=calc.kg_exacto_condicional/calc.kg_lineal_v11.replace(0,np.nan)
    calc.to_csv(OUT/'e23_02_log_a_kg.csv',index=False)
    print(results[(results.periodo=='DEV')&results.modelo.eq('acoplado')][['target','variable','theta','p','q_bh']].to_string(index=False))

if __name__=='__main__':run()
