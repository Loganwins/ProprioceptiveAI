import threading
import unittest
from types import SimpleNamespace as NS
import torch
from lens.hf import DeepModel
from lens.server import LensApp, DEEP_PREFIX

class RequestIsolationTests(unittest.TestCase):
    def test_projection_override_and_exception_cleanup(self):
        m = DeepModel.__new__(DeepModel)
        m.torch, m.device, m.lock = torch, 'cpu', threading.Lock()
        layer = torch.nn.Identity(); m.layers = [layer]
        original = {'0': torch.tensor([[1., 0.]])}; m.projections = original
        m.tokenizer = lambda *a, **kw: {'input_ids': torch.tensor([[1, 2]])}
        entered, release = threading.Event(), threading.Event()
        def forward(**kw):
            entered.set()
            if not release.wait(5): raise RuntimeError('barrier timeout')
            return layer(torch.tensor([[[3., 7.], [5., 9.]]]))
        m.model = forward
        results, errors = [], []
        def run():
            try: results.append(m.read('claim', projections={'p:0': [[0., 1.]]}))
            except BaseException as exc: errors.append(exc)
        t = threading.Thread(target=run); t.start()
        try:
            self.assertTrue(entered.wait(5)); self.assertIs(m.projections, original)
        finally: release.set(); t.join(5)
        self.assertFalse(t.is_alive()); self.assertEqual(errors, [])
        self.assertEqual(results[0]['sigma'], {'p:0': [8.]})
        self.assertEqual(len(layer._forward_hooks), 0)
        m.model = lambda **kw: (_ for _ in ()).throw(ValueError('forced failure'))
        with self.assertRaises(ValueError): m.read('bad', projections={'p:0': [[0., 1.]]})
        self.assertEqual(len(layer._forward_hooks), 0)
        self.assertTrue(m.lock.acquire(blocking=False)); m.lock.release()

    def test_claim_app_never_swaps_shared_projection(self):
        original = {'0': object()}; reader = NS(projections=original); calls = []
        reader.set_projections = lambda values: setattr(reader, 'projections', values)
        def read(text, **kw):
            self.assertIs(reader.projections, original); calls.append(kw)
            return {'sigma': {}, 'norm': {}}
        reader.read = read
        cloud = NS(enabled=True, check_claims=lambda cfg, steps: {'results': [{} for _ in steps]})
        app = NS(cloud=cloud, reader=reader, reader_config={'premium_layers': [1]}, reader_projections={'p:0': [[0., 1.]]})
        LensApp.check_claims(app, ['one', 'two'])
        self.assertEqual(calls, [{'projections': app.reader_projections}] * 2)

    def test_interleaved_fingerprints_and_cancel_cleanup(self):
        closed = []
        def stream(send, **kw):
            identity = send[0]['content']
            try:
                yield {'type':'token', 'text':identity, 'sigma':{'0':[identity]}, 'norm':{'0':1}}
                yield {'type':'done'}
            finally: closed.append(identity)
        cloud = NS(config={'model_id':'model','key_version':1}, creds={'device_id':'device'}, score=lambda rows: {'scores':{}})
        app = NS(ensure_cloud=lambda:True, cloud=cloud, cloud_error=None, deep=NS(chat_stream=stream))
        ca, cb = {}, {}
        a = LensApp.chat_events(app, DEEP_PREFIX+'model', [{'content':'A'}], [], state_context=ca)
        b = LensApp.chat_events(app, DEEP_PREFIX+'model', [{'content':'B'}], [], state_context=cb)
        try:
            while next(a)['type'] != 'token': pass
            list(b); cloud.config['key_version']=2; list(a)
        finally: a.close(); b.close()
        self.assertEqual(ca['state']['sigma'], {'0':['A']}); self.assertEqual(cb['state']['sigma'], {'0':['B']})
        self.assertEqual(ca['space'], 'model|device|1'); self.assertEqual(cb['space'], 'model|device|1')
        self.assertCountEqual(closed, ['A','B']); self.assertFalse(hasattr(app,'last_state'))
        c = LensApp.chat_events(app, DEEP_PREFIX+'model', [{'content':'C'}], [])
        while next(c)['type'] != 'token': pass
        c.close(); self.assertIn('C',closed)

    def test_partial_hook_install_is_cleaned_up(self):
        m = DeepModel.__new__(DeepModel)
        m.torch, m.device, m.lock = torch, 'cpu', threading.Lock()
        good = torch.nn.Identity()
        def fail(*a, **kw): raise ValueError('forced hook installation failure')
        m.layers = [good, NS(register_forward_hook=fail)]; m.n_layers = 2
        m.projections = {'0': torch.eye(2), '1': torch.eye(2)}; m.steering = {}
        m.tokenizer = lambda *a, **kw: {'input_ids': torch.tensor([[1]])}
        with self.assertRaisesRegex(ValueError, 'hook installation'):
            m.read('claim')
        self.assertEqual(len(good._forward_hooks), 0)
        with self.assertRaisesRegex(ValueError, 'hook installation'):
            next(m.chat_stream([]))
        self.assertEqual(len(good._forward_hooks), 0)
        self.assertTrue(m.lock.acquire(blocking=False)); m.lock.release()
