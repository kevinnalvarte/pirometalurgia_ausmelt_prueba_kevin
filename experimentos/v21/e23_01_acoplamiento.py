"""Batería predictiva temporal; no estima beneficio causal de una política."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STATE = ['ley_sn_escoria_pct_prev', 'ley_feo_escoria_pct_prev',
         'm6_sn_inv_kg_prev', 'm6_feo_inv_kg_prev', 'temperatura_horno_celsius_prev']
TARGETS = ['m6_ln_sn_dep', 'm8_ln_feo_ret_dross50', 'd_temperatura_horno_celsius']

def run():
    cache = ROOT/'experimentos/cache/df_v8.pkl'
    df = pd.read_pickle(cache)
    receta = pd.read_pickle(ROOT/'experimentos/cache/receta_reduccion.pkl')
    receta = receta[receta.Batch.isin(df.Batch.unique())].copy()
    r = df[df.fase_proceso.eq('Reducción')].merge(receta, on=['Batch','orden_escalon_fase'], validate='one_to_one')
    names = {'gn':'tasa_gn_nm3_min','carbon':'tasa_feed_Carbon_kg_min',
             'o2':'tasa_o2_nm3_min','aire':'tasa_aire_nm3_min'}
    for p,c in names.items():
        r['d_'+p] = r[c]-r['plan__'+p]
    r['d_ot'] = r.d_o2 + .21*r.d_aire
    for o in range(4):
        r['orden_'+str(o)] = (r.orden_escalon_fase==o).astype(float)
        r['gn_'+str(o)] = r.d_gn*r['orden_'+str(o)]
    base = STATE + ['plan__'+p for p in names] + ['orden_'+str(o) for o in range(4)]
    actions = ['d_carbon','d_gn','d_ot','d_aire']
    # Algebraic differences of executed vs planned products; no future centering.
    for a,b in [('carbon','gn'),('carbon','ot'),('gn','ot')]:
        pa = r['plan__'+a] if a!='ot' else r.plan__o2+.21*r.plan__aire
        pb = r['plan__'+b] if b!='ot' else r.plan__o2+.21*r.plan__aire
        r[a+'_x_'+b] = (pa+r['d_'+a])*(pb+r['d_'+b])-pa*pb
    interaction = ['carbon_x_gn','carbon_x_ot','gn_x_ot']
    sets = {'estado':base, 'estado_hgb':base, 'gn_por_orden':base+['gn_'+str(o) for o in range(4)]+['d_carbon'],
            'conjunto_lineal':base+actions, 'conjunto_interacciones':base+actions+interaction,
            'conjunto_hgb':base+actions}
    batches = df.groupby('Batch').fecha_inicio.min().sort_values().index.tolist()
    r['pos'] = r.Batch.map({b:i for i,b in enumerate(batches)})
    required = list(dict.fromkeys(sum(sets.values(),[])+TARGETS))
    r = r.replace([np.inf,-np.inf],np.nan).dropna(subset=required).copy()
    preds = []
    for start in range(100,len(batches),20):
        train = r[r.pos<start]; test = r[(r.pos>=start)&(r.pos<start+20)]
        assert train.pos.max()<test.pos.min()
        for target in TARGETS:
            for name,cols in sets.items():
                model = (HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=7,
                    min_samples_leaf=20,l2_regularization=5,learning_rate=.05,random_state=42)
                    if name.endswith('hgb') else make_pipeline(StandardScaler(),Ridge(alpha=10)))
                model.fit(train[cols],train[target])
                p = test[['Batch','orden_escalon_fase','pos',target]].rename(columns={target:'y'}).copy()
                p['pred']=model.predict(test[cols]); p['modelo']=name; p['target']=target;p['train_end']=start-1
                preds.append(p)
        print('bloque',start,'train',len(train),'test',len(test),flush=True)
    pred=pd.concat(preds,ignore_index=True); pred.to_csv(OUT/'e23_01_predicciones.csv',index=False)
    metrics=[]; comparisons=[]; rng=np.random.default_rng(42)
    for period,sel in [('TOTAL',pred.pos>=0),('DEV',pred.pos<299),('LB',pred.pos>=299)]:
        for (target,name),d in pred[sel].groupby(['target','modelo']):
            metrics.append(dict(periodo=period,target=target,modelo=name,n=len(d),batches=d.Batch.nunique(),
                                mae=mean_absolute_error(d.y,d.pred),r2=r2_score(d.y,d.pred)))
        for target,d in pred[sel].groupby('target'):
            d=d.assign(loss=(d.y-d.pred).abs())
            losses=d.groupby(['Batch','modelo']).loss.mean().unstack()
            for ref in ['gn_por_orden','estado_hgb']:
                for name in sets:
                    if name==ref:continue
                    diff=(losses[name]-losses[ref]).dropna().to_numpy()
                    boot=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)])
                    comparisons.append(dict(periodo=period,target=target,modelo=name,referencia=ref,delta_mae_batch=diff.mean(),
                        ci_lo=np.quantile(boot,.025),ci_hi=np.quantile(boot,.975),n_batches=len(diff)))
    pd.DataFrame(metrics).to_csv(OUT/'e23_01_metricas.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'e23_01_comparaciones.csv',index=False)
    byorder=[]
    for (target,name,order),d in pred.groupby(['target','modelo','orden_escalon_fase']):
        byorder.append(dict(target=target,modelo=name,orden=order,n=len(d),mae=mean_absolute_error(d.y,d.pred),r2=r2_score(d.y,d.pred)))
    pd.DataFrame(byorder).to_csv(OUT/'e23_01_por_orden.csv',index=False)
    metadata={'python':platform.python_version(),'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
              'n_common_rows':len(r),'n_batches':len(batches),'sets':sets,
              'note':'Comparador GN por orden predictivo; no replica los efectos fijos within de v11. IC exploratorios por batch, sin corregir multiplicidad ni dependencia temporal entre batches.'}
    (OUT/'e23_01_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)

if __name__=='__main__':run()
