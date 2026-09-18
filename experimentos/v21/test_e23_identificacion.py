"""Contrasta el estimador absorbido con una especificación explícita independiente."""
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from e23_02_identificacion import within, estimate

class WithinTests(unittest.TestCase):
    def test_against_explicit_dummies_unbalanced(self):
        rng=np.random.default_rng(18)
        d=pd.DataFrame({'Batch':np.repeat(np.arange(35),4),'orden_escalon_fase':np.tile(np.arange(4),35)})
        d['x']=rng.normal(size=len(d));d['z']=rng.normal(size=len(d))
        d['y']=2*d.x-.5*d.z+d.Batch*.7+d.orden_escalon_fase*3+rng.normal(size=len(d))
        d=d.drop(index=[1,9,17,31,54]).reset_index(drop=True)
        dummy=pd.get_dummies(d[['Batch','orden_escalon_fase']].astype(str),drop_first=True,dtype=float)
        full=sm.OLS(d.y,sm.add_constant(pd.concat([d[['x','z']],dummy],axis=1))).fit(cov_type='cluster',cov_kwds={'groups':d.Batch})
        absorbed=estimate(d,'y',['x','z']).set_index('variable')
        np.testing.assert_allclose(absorbed.theta,full.params[['x','z']],rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(absorbed.se,full.bse[['x','z']],rtol=1e-7,atol=1e-7)

    def test_absorption_removes_batch_and_order_means(self):
        d=pd.DataFrame({'Batch':np.repeat(np.arange(10),4),'orden_escalon_fase':np.tile(np.arange(4),10)})
        d['y']=d.Batch*2+d.orden_escalon_fase*3
        d=d.drop(index=[0,7])
        a=within(d,['y'])
        self.assertLess(np.max(np.abs(a.y)),1e-7)

if __name__=='__main__':unittest.main()
