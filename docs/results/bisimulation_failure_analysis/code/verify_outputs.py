"""Read back deliverables: verify serialized certificates and subset coverage."""
import argparse,csv,itertools,json,sys
from pathlib import Path
from analyze import check_dag


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();sys.dont_write_bytecode=True;sys.path[:0]=[str(a.repo.resolve()/'src'),str(a.repo.resolve())]
    from experiments.evaluate_top1_ltl import load_system
    from jepa_lmc.verification.behavioral_relations import BehavioralRelation,Elimination,audit_greatest_relation
    records=[json.loads(x) for x in (a.data/'outputs/top1_quality/round2/behavioral/best/relations.jsonl').read_text().splitlines()]
    cases=json.loads((a.results/'cases_72.json').read_text());es=json.loads((a.results/'errors_96.json').read_text())
    assert len(cases)==72 and len(es)==96
    dag_count=0
    for r in records:
        real,pred=(load_system(r,n) for n in ('real','top1'))
        cert=json.loads((a.results/'certificates'/f"{r['seed']}_{r['case']}.json").read_text())
        assert cert['states']==r['states'] and cert['labels']==r['labels'] and cert['initial']==r['initial']
        for mode,data in cert['semantics'].items():
            sensitive=mode=='action_sensitive'
            rel=BehavioralRelation('bisimulation',sensitive,frozenset(map(tuple,data['pairs'])),tuple(Elimination(**x) for x in data['eliminations']),data['rounds'])
            audit_greatest_relation(real,pred,rel)
            if data['initial_failure_dag']:
                assert not rel.relates_initials(real,pred)
                check_dag(real,pred,data['initial_failure_dag'],sensitive);dag_count+=1
            else:assert rel.relates_initials(real,pred)
    assert dag_count==64
    trials=list(csv.DictReader((a.results/'repair_trials_all.csv').open(encoding='utf-8-sig')))
    assert len(trials)==794
    for c in cases:
        if not c['errors']:continue
        edges={e['edge_id'] for e in es if (e['seed'],e['case'])==(c['seed'],c['case'])}
        rows=[t for t in trials if (int(t['seed']),t['case'])==(c['seed'],c['case'])]
        seen={frozenset(json.loads(t['repaired_edges'])) for t in rows}
        expected={frozenset(s) for size in range(len(edges)+1) for s in itertools.combinations(edges,size)}
        assert seen==expected and len(rows)==len(expected)
        for t in rows:
            repaired=json.loads(t['repaired_edges'])
            assert int(t['size'])==len(repaired)
            assert t['insensitive']==t['sensitive']
            assert (t['insensitive']=='True')==(c['initial_bisimulation'] or len(repaired)==len(edges))
    for e in es:
        assert e['reachable_real']==e['reachable_pred']
        assert e['initial_bisimulation']==(not e['reachable_real'] or e['targets_bisimilar_in_real'])
        assert e['only_this_error_retained_passes']==e['initial_bisimulation']
    print('PASS: 144 serialized full certificates, 64 rooted DAGs, all 794 distinct subsets, both semantics and structural cross-tabs.')


if __name__=='__main__':main()
