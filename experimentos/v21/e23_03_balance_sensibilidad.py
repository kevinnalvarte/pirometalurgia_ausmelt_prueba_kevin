"""Sensibilidad del cierre y efectos: escenarios, nunca especiación medida."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS','3')
from pathlib import Path
import numpy as np
import pandas as pd
from e23_02_identificacion import ROOT, OUT, STATE, estimate, load
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experimentos/v6'))
import masa_v6 as M

def run():
    df=pd.read_pickle(ROOT/'experimentos/cache/df_v8.pkl').sort_values(['Batch','fecha_inicio']).reset_index(drop=True)
    original=M.K_SNO2
    baseline=M.estimar_masa(df).m6_masa_kg
    assert baseline.isna().equals(df.m6_masa_kg.isna())
    error=float((baseline-df.m6_masa_kg).abs().max())
    assert error<1e-6,error
    scenarios=[('base',original,0,.01),('Sn_II',134.71/118.71,0,.01),
        ('FeIII_30_F',original,.30,.01),('SnII_FeIII30F',134.71/118.71,.30,.01),
        ('delta0',original,0,0),('delta02',original,0,.02)]
    r=load(); tables=[]; masses=[]; common_keys=None; variants={}
    for name,k,fe3,delta in scenarios:
        a=df.copy();M.K_SNO2=k
        # Oxygen mass of ferric versus ferrous iron changes the closure only.
        # Keep reported total iron expressed as FeO-equivalent for the inventory.
        a.loc[a.fase_proceso.eq('Fusión'),'ley_feo_escoria_pct']*=1+fe3*8/71.844
        mass=M.estimar_masa(a,dict(delta_R=delta)).m6_masa_kg
        sn=mass*df.ley_sn_escoria_pct/100; fe=mass*df.ley_feo_escoria_pct/100
        prevsn=sn.groupby(df.Batch).shift(); prevfe=fe.groupby(df.Batch).shift()
        for frac in [.3,.5,.6]:
            t=df[['Batch','orden_escalon_fase','fase_proceso']].copy()
            t['m6_sn_inv_kg_prev']=prevsn;t['m6_feo_inv_kg_prev']=prevfe
            t['ysn']=np.log((prevsn+df.feed_Sn_kgf.fillna(0))/sn)
            t['yfe']=np.log(fe/(prevfe+71.844/55.845*frac*df.feed_dross_Fe_kgh.fillna(0)))
            t=t[t.fase_proceso.eq('Reducción')].drop(columns='fase_proceso')
            d=r.drop(columns=['m6_sn_inv_kg_prev','m6_feo_inv_kg_prev']).merge(t,on=['Batch','orden_escalon_fase'],validate='one_to_one')
            d=d.replace([np.inf,-np.inf],np.nan)
            variants[(name,frac)]=d
        end=df.assign(mass=mass).groupby('Batch').mass.last()
        bend=df.groupby('Batch').m6_masa_kg.last()
        masses.append(dict(escenario=name,ratio_masa_mediana=float((end/bend).median()),n_mass=int(mass.notna().sum())))
    M.K_SNO2=original
    X=['gn_'+str(o) for o in range(4)]+['d_carbon','d_exceso','d_aire']+STATE
    # Identical cohort across physical scenarios for each endpoint.
    for y in ['ysn','yfe']:
        keys=None
        for d in variants.values():
            valid=d.dropna(subset=X+[y]); k=set(zip(valid.Batch,valid.orden_escalon_fase))
            keys=k if keys is None else keys&k
        for (name,frac),d in variants.items():
            d=d[[key in keys for key in zip(d.Batch,d.orden_escalon_fase)]]
            for period,sub in [('DEV',d[d.pos<299]),('LB',d[d.pos>=299])]:
                tab=estimate(sub,y,X);tab=tab[tab.variable.str.startswith('gn_')].copy()
                tab['target']=y;tab['escenario']=name;tab['fraccion_fe_dross']=frac;tab['periodo']=period;tables.append(tab)
        print('completo',y,flush=True)
    pd.concat(tables,ignore_index=True).to_csv(OUT/'e23_03_efectos_sensibilidad.csv',index=False)
    pd.DataFrame(masses).to_csv(OUT/'e23_03_masas.csv',index=False)
    (OUT/'e23_03_verificacion.json').write_text(json.dumps({'max_diff_masa_base':error,'mismos_nan':True,
        'nota':'Escenarios extremos asumidos, no estimados. No constituyen IC ni balances validados con masa medida.'},indent=2),encoding='utf-8')
    print(pd.DataFrame(masses).to_string(index=False))

if __name__=='__main__':run()
