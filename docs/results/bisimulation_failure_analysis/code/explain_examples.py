"""Verify an additional local successor-class certificate for the report."""
import argparse,json,sys
from dataclasses import asdict
from pathlib import Path
from graph_core import dependency_dag
from analyze import check_dag,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();sys.dont_write_bytecode=True;sys.path[:0]=[str(a.repo.resolve()/'src'),str(a.repo.resolve())]
    from experiments.evaluate_top1_ltl import load_system
    from jepa_lmc.verification.transition_system import ExplicitTransitionSystem as TS
    from jepa_lmc.verification.behavioral_relations import greatest_relation,audit_greatest_relation,bisimulation_by_partition
    rows=[json.loads(x) for x in (a.data/'outputs/top1_quality/round2/behavioral/best/relations.jsonl').read_text().splitlines()]
    r=next(r for r in rows if r['seed']==20260804 and r['case']=='safe_detour_layout1_rot180')
    real=load_system(r,'real')
    def at(s):return TS(states=real.states,initial_states=(s,),labels={u:real.propositions(u) for u in real.states},transitions={u:[(e.action,e.target) for e in real.action_successors(u)] for u in real.states})
    left,right=at(23),at(28);rel=greatest_relation(left,right,kind='bisimulation')
    audit_greatest_relation(left,right,rel);assert rel.pairs==bisimulation_by_partition(left,right)
    assert not rel.relates_initials(left,right)
    dag=dependency_dag(left,right,rel);check_dag(left,right,dag,False)
    write(a.results/'local_successor_certificate.json',dict(seed=r['seed'],case=r['case'],semantics='action_insensitive',
        interpretation='two states in the SAME true graph; diagnostic, not a replacement for the cross-system initial certificate',
        left_state=23,left_coord=r['states'][23],right_state=28,right_coord=r['states'][28],
        pairs=sorted(rel.pairs),eliminations=[asdict(e) for e in rel.eliminations],rounds=rel.rounds,dag=dag))
    print('Local successor-class certificate verified:',len(dag['nodes']),'DAG nodes')


if __name__=='__main__':main()
