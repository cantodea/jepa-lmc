"""Small algorithm unit fixtures only; NOT additional experimental maps/data."""
import unittest
import copy
from pathlib import Path
import sys

from graph_core import dependency_dag, exact_repairs
from analyze import check_dag
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem as TS
from jepa_lmc.verification.behavioral_relations import greatest_relation, audit_greatest_relation, bisimulation_by_partition


def system(edges):
    return TS(states=(0,1,2), initial_states=(0,), labels={0:['safe'],1:['safe'],2:['goal']},transitions=edges)


def make(g,t):
    return TS(states=g.states,initial_states=g.initial_states,labels={s:g.propositions(s) for s in g.states},
        transitions={s:[(a,v) for (u,a),v in t.items() if u==s] for s in g.states})


class AnalysisTests(unittest.TestCase):
    def test_single_error_exact_repair_and_full_dag(self):
        real=system({0:[(0,1),(1,0)],1:[(0,2),(1,1)],2:[(0,2),(1,2)]})
        learned=system({0:[(0,1),(1,0)],1:[(0,1),(1,1)],2:[(0,2),(1,2)]})
        for actions in (False,True):
            rel=greatest_relation(real,learned,kind='bisimulation',action_sensitive=actions)
            audit_greatest_relation(real,learned,rel)
            dag=dependency_dag(real,learned,rel)
            check_dag(real,learned,dag,actions)
            self.assertEqual(dag['roots'],[(0,0)])
            nodes={(n['left'],n['right']):n for n in dag['nodes']}
            self.assertTrue(any(n['side']=='label' for n in nodes.values()))
            for n in nodes.values():
                for reply in n['replies']:
                    self.assertLess(nodes[tuple(reply['child'])]['round'],n['round'])
        result=exact_repairs(real,learned,make,greatest_relation,audit_greatest_relation,bisimulation_by_partition)
        self.assertEqual(len(result['trials']),2)
        self.assertEqual(result['minimum_repairs']['insensitive']['minimum'],1)

    def test_two_errors_all_subsets(self):
        real=system({0:[(0,1)],1:[(0,2)],2:[(0,2)]})
        learned=system({0:[(0,0)],1:[(0,1)],2:[(0,2)]})
        result=exact_repairs(real,learned,make,greatest_relation,audit_greatest_relation,bisimulation_by_partition)
        self.assertEqual(len(result['trials']),4)
        self.assertFalse(any(t['insensitive'] for t in result['trials'] if t['size']==1))
        self.assertEqual(result['minimum_repairs']['insensitive']['minimum'],2)

    def test_exported_certificate_rejects_missing_reply_and_root(self):
        real=system({0:[(0,1),(1,0)],1:[(0,2),(1,1)],2:[(0,2),(1,2)]})
        learned=system({0:[(0,1),(1,0)],1:[(0,1),(1,1)],2:[(0,2),(1,2)]})
        rel=greatest_relation(real,learned,kind='bisimulation')
        dag=dependency_dag(real,learned,rel)
        broken=copy.deepcopy(dag)
        node=next(n for n in broken['nodes'] if n['replies'])
        node['replies']=[]
        with self.assertRaises(AssertionError):check_dag(real,learned,broken,False)
        broken=copy.deepcopy(dag);broken['roots']=[]
        with self.assertRaises(AssertionError):check_dag(real,learned,broken,False)

    def test_exported_certificate_rejects_non_decreasing_rank(self):
        real=system({0:[(0,1)],1:[(0,2)],2:[(0,2)]})
        learned=system({0:[(0,1)],1:[(0,1)],2:[(0,2)]})
        rel=greatest_relation(real,learned,kind='bisimulation')
        dag=dependency_dag(real,learned,rel)
        root=next(n for n in dag['nodes'] if (n['left'],n['right'])==(0,0))
        child=root['replies'][0]['child']
        node=next(n for n in dag['nodes'] if (n['left'],n['right'])==tuple(child))
        node['round']=root['round']
        with self.assertRaises(AssertionError):check_dag(real,learned,dag,False)


if __name__=='__main__': unittest.main()
