import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import Future
import numpy as np
from .check_environment import check
from .async_localizer import AsyncLocalizer


class BackendReadinessTests(unittest.TestCase):
    def test_cpu_torch_is_explicitly_rejected_for_gpu_only(self):
        torch=SimpleNamespace(__version__='test-cpu',version=SimpleNamespace(cuda=None))
        with patch('importlib.util.find_spec',return_value=True), patch.dict(sys.modules,{'torch':torch}):
            gpu=check('xfeat-cuda')
            self.assertFalse(gpu['ok'])
            self.assertTrue(any('CPU 版' in e for e in gpu['errors']))
            self.assertTrue(check('xfeat-cpu')['ok'])

    def test_missing_torch_explained_without_import_crash(self):
        with patch('importlib.util.find_spec',side_effect=lambda name:None if name=='torch' else True):
            report=check('xfeat-cuda')
            self.assertTrue(any('缺少依赖 torch' in e for e in report['errors']))

    def model(self, stamp, reproject):
        model=AsyncLocalizer.__new__(AsyncLocalizer)
        model.previous=np.zeros((20,20),np.uint8)
        model.last_timestamp=10.8
        model.camera=model.track_points=None
        model.track_residual=0
        model.generation=1
        model.last_anchor=0
        model.max_result_age=2.
        model.max_coast=2.5
        model.last_attempt=10.9
        model.anchor_interval=.35
        model._flow=lambda gray,mask:reproject
        model.pending=Future()
        model.pending.set_result((model.previous,stamp,1,
            (np.array([10.,20.]),100,5,np.array([[i,i] for i in range(12)],np.float32),.1),'verified',900.))
        return model

    def test_slow_cpu_anchor_still_requires_successful_reprojection(self):
        frame=np.zeros((20,20,3),np.uint8)
        self.assertEqual(self.model(10.,True).update(frame,10.9).status,'LOCKED')
        self.assertIsNone(self.model(10.,False).update(frame,10.9).camera)
        self.assertIsNone(self.model(8.,True).update(frame,10.9).camera)


if __name__=='__main__':
    unittest.main()
