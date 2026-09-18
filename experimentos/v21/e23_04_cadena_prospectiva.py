"""Validación temporal del pipeline completo: efecto within -> score físico -> endpoint batch.

No confundir score de contribución con predicción absoluta ni asociación con efecto causal.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,r2_score
from e23_02_identificacion import ROOT, OUT, STATE, load, within

CONFIGS={'v11_gn':['gn_'+str(o) for o in range(4)]+['d_carbon'],
         'conjunto':['gn_'+str(o) for o in range(4)]+['d_carbon','d_exceso','d_aire'],
         'acoplado':['gn_'+str(o) for o in range(4)]+['d_carbon','d_exceso','d_aire','C_x_exceso','C_x_GN','C_x_Sn']}
CONTROLS=['feo_ini','sn_ini','T_ini','feed_sn_total_t','ley_sn_conc_batch_pct','frac_carga_secundaria_F','feed_dross_Fe_total_t','pos']
ENDPOINTS=['feo_neto_kg','dross_pp','polvo_pp','recuperacion_refinada_pct']

def score_batch(r,coefs,name):
    # Exact v11-style GN-only scoring for reference, despite carbon in adjustment.
    actions=['gn_'+str(o) for o in range(4)] if name=='v11_gn' else list(coefs.index)
    values=-(r[actions]@coefs[actions])*r.m6_feo_inv_kg_prev
    values[r[actions+['m6_feo_inv_kg_prev']].isna().any(axis=1)]=np.nan
    return values.groupby(r.Batch).sum(min_count=4)

def moving_boot(diff,rng,n=3000,L=10):
    d=np.asarray(diff); size=len(d); out=[]
    for _ in range(n):
        starts=rng.integers(0,size,size=int(np.ceil(size/L)))
        ii=np.concatenate([(s+np.arange(L))%size for s in starts])[:size]
        out.append(d[ii].mean())
    return np.quantile(out,[.025,.975])

def run():
    r=load();bt=pd.read_csv(ROOT/'experimentos/v20/e22_01_tabla.csv').set_index('Batch')
    first=r[r.orden_escalon_fase.eq(0)].set_index('Batch')
    last=r[r.orden_escalon_fase.eq(3)].set_index('Batch')
    bt['pos']=first.pos;bt['feo_ini']=first.m6_feo_inv_kg_prev;bt['sn_ini']=first.m6_sn_inv_kg_prev
    bt['T_ini']=first.temperatura_horno_celsius_prev
    bt['feo_neto_kg']=bt.feo_ini+1.2865*.5*r.groupby('Batch').feed_dross_Fe_kgh.sum(min_count=4)-last.m6_feo_inv_kg
    # No .last() fallback: final missing means endpoint missing, not R2 as final.
    bt['R3_observado']=last.m6_feo_inv_kg.notna()
    allcols=list(dict.fromkeys(sum(CONFIGS.values(),[])+STATE))
    valid=r.dropna(subset=allcols+['m8_ln_feo_ret_dross50'])
    scores=[];predictions=[];theta=[]
    for start in range(100,362,20):
        tr=valid[valid.pos<start];future=r[(r.pos>=start)&(r.pos<start+20)]
        assert tr.pos.max()<future.pos.min()
        period_scores={}
        for name,actions in CONFIGS.items():
            x=actions+STATE;z=within(tr,['m8_ln_feo_ret_dross50']+x)
            coef=pd.Series(np.linalg.lstsq(z[x],z.m8_ln_feo_ret_dross50,rcond=None)[0],index=x)
            for a in actions:theta.append(dict(train_end=start-1,modelo=name,variable=a,theta=coef[a]))
            # Only train-era outcomes inform either stage. Historical scores are in-sample at stage 1.
            period_scores[name]=score_batch(r[r.pos<start+20],coef[actions],name)
            out=future[['Batch','pos']].drop_duplicates().set_index('Batch')
            out['score']=period_scores[name];out['modelo']=name;out['train_end']=start-1;scores.append(out.reset_index())
        local=bt.copy()
        for name,s in period_scores.items():local[name]=s
        for endpoint in ENDPOINTS:
            # Matched batch sets for every model, including controls-only comparator.
            d=local.dropna(subset=CONTROLS+[endpoint]+list(CONFIGS))
            train=d[d.pos<start];test=d[(d.pos>=start)&(d.pos<start+20)]
            if len(test)==0:continue
            for name in ['solo_contexto']+list(CONFIGS):
                x=CONTROLS+([] if name=='solo_contexto' else [name])
                model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(train[x],train[endpoint])
                out=test[['pos',endpoint,'R3_observado']].rename(columns={endpoint:'y'}).copy()
                out['pred']=model.predict(test[x]);out['modelo']=name;out['target']=endpoint;out['train_end']=start-1
                predictions.append(out.reset_index())
        print('cadena bloque',start,flush=True)
    p=pd.concat(predictions,ignore_index=True);p.to_csv(OUT/'e23_04_predicciones.csv',index=False)
    s=pd.concat(scores,ignore_index=True);s.to_csv(OUT/'e23_04_scores.csv',index=False)
    pd.DataFrame(theta).to_csv(OUT/'e23_04_theta.csv',index=False)
    metrics=[];contrasts=[];rng=np.random.default_rng(42)
    for period,mask in [('TOTAL',p.pos>=0),('DEV',p.pos<299),('LB',p.pos>=299),('R3_completo',p.R3_observado)]:
        d=p[mask]
        for (target,name),q in d.groupby(['target','modelo']):
            metrics.append(dict(periodo=period,target=target,modelo=name,n=len(q),mae=mean_absolute_error(q.y,q.pred),r2=r2_score(q.y,q.pred)))
        for target,q in d.groupby('target'):
            q=q.assign(loss=(q.y-q.pred).abs()); wide=q.pivot(index='pos',columns='modelo',values='loss').sort_index()
            for ref in ['solo_contexto','v11_gn']:
                for name in CONFIGS:
                    if name==ref:continue
                    diff=wide[name]-wide[ref];ci=moving_boot(diff,rng)
                    contrasts.append(dict(periodo=period,target=target,modelo=name,referencia=ref,delta_mae=diff.mean(),ci_lo=ci[0],ci_hi=ci[1],n=len(diff)))
    pd.DataFrame(metrics).to_csv(OUT/'e23_04_metricas.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'e23_04_comparaciones.csv',index=False)
    metadata={'r3_missing':int((~bt.R3_observado).sum()),'first_training_batches':100,'block':20,
              'note':'Full pipeline forecast; no future outcomes used for fitting. Scores in kg use linear approximation as common comparator. Nested selection absent; fixed hyperparameters. Circular moving-block bootstrap L=10 exploratory; no multiplicity correction.'}
    (OUT/'e23_04_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    assert (p.pos>p.train_end).all()
    print(pd.DataFrame(metrics).query("periodo == 'TOTAL'").to_string(index=False))

if __name__=='__main__':run()
