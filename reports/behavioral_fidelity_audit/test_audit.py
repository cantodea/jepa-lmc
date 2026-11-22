import unittest

from audit_core import error_auc, observations, project
from jepa_lmc.evaluation.trace_sanity import Trace
from witnesses import feasible


class AuditTests(unittest.TestCase):
    def test_only_positive_keys_change(self):
        p={(("a",),0):("a",),(("b",),0):("a",)}
        o=observations([{"states":[["a"],["b"]],"actions":[0]}])
        q=project(p,o)
        self.assertEqual(p[(("a",),0)],("a",))
        self.assertEqual(q[(("a",),0)],("b",))
        self.assertEqual(q[(("b",),0)],p[(("b",),0)])
        self.assertEqual(project(p,{}),p)

    def test_conflict_rejected(self):
        with self.assertRaises(ValueError):
            observations([{"states":[[0],[0],[1]],"actions":[0,0]}])

    def test_catalogue_key_required(self):
        with self.assertRaises(ValueError): project({}, {("s","a"):"t"})

    def test_auc_tie_and_direction(self):
        self.assertEqual(error_auc([(0,1),(1,0)]),1)
        self.assertEqual(error_auc([(2,1),(1,0)]),0)
        self.assertEqual(error_auc([(1,1),(1,0)]),0.5)
        self.assertIsNone(error_auc([(1,1)]))

    def test_feasible_rejects_whole_negative_only(self):
        p={(0,0):1,(1,0):0}
        good=Trace((0,1,0),(0,0))
        bad=Trace((0,1,1),(0,0))
        self.assertTrue(feasible(p,[good],[bad]))
        self.assertFalse(feasible(p,[good],[good]))


if __name__=="__main__": unittest.main()
