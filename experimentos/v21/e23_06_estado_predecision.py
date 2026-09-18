"""Comparación temporal de estados previos, targets observables e inventarios.
No evalúa causalidad ni optimiza acciones. Receta supuesta disponible al decidir.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from e23_02_identificacion import load,STATE,OUT
from e23_01_acoplamiento import TARGETS
from e23_04_cadena_prospectiva import moving_boot

EXTRA=['ley_sio2_escoria_pct_prev','ley_cao_escoria_pct_prev',
       'basicidad_B2_prev','ratio_sn_feo_prev','posicion_vertical_lanza_mm_prev',
       'grad_temperatura_horno_celsius_prev','termocupla_media_celsius_prev',
       'cum_feed_Carbon_kg_prev','cum_gn_nm3_prev','cum_o2_nm3_prev','cum_aire_nm3_prev']
DIRECT=['ley_sn_escoria_pct','ley_feo_escoria_pct','temperatura_horno_celsius']

def run():
    r=load()
    base=STATE+['plan__'+p for p in ['gn','carbon','o2','aire']]+['orden_escalon_fase']
    sets={'base':base,'fenomenologico':base+EXTRA,'reciente_150':base+EXTRA}
    results=[]
    for y in TARGETS+DIRECT:
        # Extra features can be missing: HGB handles missingness using training only.
        d=r.dropna(subset=base+[y]).copy()
        history=[]
        for start in range(100,362,20):
            tr=d[d.pos<start];te=d[d.pos.between(start,start+19)]
            if te.empty:continue
            assert tr.pos.max()<te.pos.min()
            # Online choice uses only prediction errors from batches before this block.
            chosen='base'
            if history:
                past=pd.concat(history)
                past=past[past.pos>=start-100]
                loss=past.assign(loss=(past.y-past.pred).abs()).groupby(['modelo','Batch']).loss.mean().groupby('modelo').mean()
                chosen=loss.idxmin()
            outputs={}
            for name,cols in sets.items():
                train=tr[tr.pos>=start-150] if name=='reciente_150' else tr
                model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=7,min_samples_leaf=20,
                    l2_regularization=5,learning_rate=.05,random_state=42)
                model.fit(train[cols],train[y])
                out=te[['Batch','pos','orden_escalon_fase',y]].rename(columns={y:'y'}).copy()
                out['pred']=model.predict(te[cols]);out['target']=y;out['modelo']=name;out['train_end']=start-1
                out['seleccion']=name;outputs[name]=out
            adaptive=outputs[chosen].copy();adaptive['modelo']='selector_pasado'
            results.extend(list(outputs.values())+[adaptive]);history.extend(outputs.values())
            if y in DIRECT:
                persistence=outputs['base'].copy();persistence['pred']=te[y+'_prev'].to_numpy()
                persistence['modelo']='persistencia';persistence['seleccion']='persistencia';results.append(persistence)
        print('predecision',y,flush=True)
    p=pd.concat(results,ignore_index=True)
    assert (p.pos>p.train_end).all()
    assert not p.duplicated(['Batch','orden_escalon_fase','target','modelo']).any()
    p.to_csv(OUT/'e23_06_predicciones.csv',index=False)
    metrics=[];contrasts=[];rng=np.random.default_rng(42)
    for period,mask in [('TOTAL',p.pos>=0),('DEV',p.pos<299),('LB',p.pos>=299)]:
        for (y,name),q in p[mask].groupby(['target','modelo']):
            metrics.append(dict(periodo=period,target=y,modelo=name,n=len(q),mae=mean_absolute_error(q.y,q.pred),r2=r2_score(q.y,q.pred)))
        for y,q in p[mask].groupby('target'):
            losses=q.assign(loss=(q.y-q.pred).abs()).groupby(['pos','modelo']).loss.mean().unstack().sort_index()
            for name in losses.columns:
                if name=='base':continue
                delta=losses[name]-losses.base;ci=moving_boot(delta,rng)
                contrasts.append(dict(periodo=period,target=y,modelo=name,delta_mae=delta.mean(),ci_lo=ci[0],ci_hi=ci[1],n_batches=len(delta)))
    pd.DataFrame(metrics).to_csv(OUT/'e23_06_metricas.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'e23_06_comparaciones.csv',index=False)
    (OUT/'e23_06_metadata.json').write_text(json.dumps({'features':sets,
        'availability':'Previous-step labels and recipe: actual laboratory delivery timestamps not verified.',
        'selection':'Last 100 positions with past out-of-sample predictions; initial choice base.',
        'limitations':'Exploratory reused data; concentrations are not recovery; no causal policy value.'},indent=2),encoding='utf-8')

if __name__=='__main__':run()
