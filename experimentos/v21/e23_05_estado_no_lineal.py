"""Diagnóstico exploratorio PLM: signos libres, cross-fitting por batch.

No es validación temporal ni estimador within: no elimina confusión de batch.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests
from e23_02_identificacion import load, STATE, OUT
from e23_01_acoplamiento import TARGETS

GN=['gn_'+str(o) for o in range(4)]
CONFIGS={'carbon_simple':GN+['d_carbon','d_exceso','d_aire'],
         'carbon_por_inventario':GN+['C_x_Sn','d_exceso','d_aire'],
         'carbon_acoplado':GN+['d_carbon','C_x_Sn','C_x_exceso','d_exceso','d_aire']}

def run():
    r=load()
    states=STATE+['plan__'+p for p in ['gn','carbon','o2','aire']]+['orden_escalon_fase']
    actions=list(dict.fromkeys(sum(CONFIGS.values(),[])))
    # Cohorte idéntica entre targets y especificaciones, sin imputación de acciones.
    d=r.dropna(subset=states+actions+TARGETS).copy()
    effects=[];diagnostics=[];audits=[]
    for period,sub in [('DEV',d[d.pos<299]),('LB',d[d.pos>=299])]:
        sub=sub.reset_index(drop=True)
        cols=actions+TARGETS
        residual=pd.DataFrame(index=sub.index,columns=cols,dtype=float)
        for fold,(tr,te) in enumerate(GroupKFold(5).split(sub,groups=sub.Batch)):
            assert set(sub.iloc[tr].Batch).isdisjoint(sub.iloc[te].Batch)
            for col in cols:
                model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=7,
                    min_samples_leaf=20,l2_regularization=5,learning_rate=.05,random_state=42)
                model.fit(sub.iloc[tr][states],sub.iloc[tr][col])
                residual.loc[te,col]=sub.iloc[te][col].to_numpy()-model.predict(sub.iloc[te][states])
            audits.append(dict(periodo=period,fold=fold,n_train=len(tr),n_test=len(te),
                               batches_train=sub.iloc[tr].Batch.nunique(),batches_test=sub.iloc[te].Batch.nunique(),overlap=0))
        assert residual.notna().all().all()
        for name,x in CONFIGS.items():
            a=residual[x].to_numpy();scaled=a/np.std(a,axis=0)
            diagnostics.append(dict(periodo=period,modelo=name,n=len(sub),batches=sub.Batch.nunique(),
                                    rank=np.linalg.matrix_rank(a),p=len(x),condition=np.linalg.cond(scaled)))
            for y in TARGETS:
                fit=sm.OLS(residual[y],residual[x]).fit(cov_type='cluster',cov_kwds={'groups':sub.Batch})
                df=sub.Batch.nunique()-1;critical=t.ppf(.975,df)
                for variable in x:
                    theta=fit.params[variable];se=fit.bse[variable]
                    effects.append(dict(periodo=period,modelo=name,target=y,variable=variable,
                        theta=theta,se=se,ci_lo=theta-critical*se,ci_hi=theta+critical*se,
                        p=2*t.sf(abs(theta/se),df),n=len(sub),batches=sub.Batch.nunique()))
        print('PLM completado',period,len(sub),flush=True)
    result=pd.DataFrame(effects)
    for _,ix in result.groupby('periodo').groups.items():
        result.loc[ix,'q_bh']=multipletests(result.loc[ix,'p'],method='fdr_bh')[1]
    result.to_csv(OUT/'e23_05_efectos.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'e23_05_diagnostico.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'e23_05_folds.csv',index=False)

if __name__=='__main__':run()
