"""Reexpresión exacta de predicciones externas de deltaT como T final."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score
from e23_02_identificacion import load,OUT
from e23_04_cadena_prospectiva import moving_boot

def run():
    p=pd.read_csv(OUT/'e23_06_predicciones.csv');r=load()
    keys=['Batch','orden_escalon_fase']
    delta=p[p.target.eq('d_temperatura_horno_celsius')].merge(
        r[keys+['temperatura_horno_celsius_prev','temperatura_horno_celsius']],on=keys,validate='many_to_one')
    assert np.allclose(delta.y+delta.temperatura_horno_celsius_prev,delta.temperatura_horno_celsius)
    delta['pred']+=delta.temperatura_horno_celsius_prev
    delta['y']=delta.temperatura_horno_celsius;delta['modelo']='incremental_'+delta.modelo
    direct=p[p.target.eq('temperatura_horno_celsius')]
    d=pd.concat([delta[direct.columns],direct],ignore_index=True)
    d['target']='temperatura_horno_celsius'
    assert d.groupby(keys).modelo.nunique().eq(9).all()
    d.to_csv(OUT/'e23_07_predicciones.csv',index=False)
    metrics=[];comparisons=[];rng=np.random.default_rng(42)
    for period,mask in [('TOTAL',d.pos>=0),('DEV',d.pos<299),('LB',d.pos>=299)]:
        for name,q in d[mask].groupby('modelo'):
            metrics.append(dict(periodo=period,modelo=name,n=len(q),mae=mean_absolute_error(q.y,q.pred),r2=r2_score(q.y,q.pred)))
        loss=d[mask].assign(loss=lambda x:(x.y-x.pred).abs()).groupby(['pos','modelo']).loss.mean().unstack().sort_index()
        for name in ['incremental_fenomenologico','incremental_reciente_150','incremental_selector_pasado']:
            for ref in ['persistencia','reciente_150','incremental_base']:
                diff=loss[name]-loss[ref];ci=moving_boot(diff,rng)
                comparisons.append(dict(periodo=period,modelo=name,referencia=ref,delta_mae=diff.mean(),ci_lo=ci[0],ci_hi=ci[1]))
    pd.DataFrame(metrics).to_csv(OUT/'e23_07_metricas.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'e23_07_comparaciones.csv',index=False)

if __name__=='__main__':run()
