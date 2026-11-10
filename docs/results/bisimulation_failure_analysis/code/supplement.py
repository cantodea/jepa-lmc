"""Readable supporting tables derived solely from verified graphs/results."""
import argparse,csv,json
from pathlib import Path
from analyze import csvwrite,write


def distances(record,name,prop):
    d={s:0 for s,labels in enumerate(record['labels']) if prop in labels}
    for _ in record['states']:
        new={s:1+min(d[t] for _,t in edges if t in d) for s,edges in enumerate(record['graphs'][name]) if s not in d and any(t in d for _,t in edges)}
        if not new:break
        d.update(new)
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args(); out=a.results
    recs=[json.loads(x) for x in (a.data/'outputs/top1_quality/round2/behavioral/best/relations.jsonl').read_text().splitlines()]
    records={(r['seed'],r['case']):r for r in recs}
    errors=json.loads((out/'errors_96.json').read_text()); cases=json.loads((out/'cases_72.json').read_text())
    all_dist={k:{(name,prop):distances(r,name,prop) for name in ('real','top1') for prop in ('goal','danger')} for k,r in records.items()}
    for e in errors:
        ds=all_dist[e['seed'],e['case']]
        for prop in ('goal','danger'):
            for end in ('true','pred'):
                e[f'{end}_target_distance_to_{prop}_in_real']=ds['real',prop].get(e[f'{end}_successor'])
        e['target_goal_or_danger_distance_differs_in_real']=any(ds['real',p].get(e['true_successor'])!=ds['real',p].get(e['pred_successor']) for p in ('goal','danger'))
    edge_bycase={k:[e for e in errors if (e['seed'],e['case'])==k] for k in records}
    for c in cases:
        key=c['seed'],c['case'];r=records[key];ds=all_dist[key]
        for name in ('real','top1'):
            for prop in ('goal','danger'):c[f'initial_distance_{name}_{prop}']=ds[name,prop].get(r['initial'])
        c['error_successor_equivalence_changes']=sum(not e['targets_bisimilar_in_real'] for e in edge_bycase[key])
        if c['initial_bisimulation']:
            c['mechanism']='exact_graph' if c['errors']==0 else 'unreachable_errors_or_safe_behavior_class_preserved'
        elif c['family']=='sealed_region':c['mechanism']='reachable_spurious_cross_wall_transition_opens_labelled_region'
        else:c['mechanism']='reachable_transition_to_different_future_behavior_class'
    csvwrite(out/'errors_96_detailed.csv',errors);csvwrite(out/'cases_72_detailed.csv',cases)
    csvwrite(out/'single_edge_repairs_13.csv',[e for e in errors if not e['initial_bisimulation'] and e['single_repair_restores']])
    csvwrite(out/'passing_cases_7.csv',[c for c in cases if c['initial_bisimulation'] and c['errors']])
    csvwrite(out/'structural_groups.csv',[dict(initial_pass=b,reachable=reach,errors=len([e for e in errors if e['initial_bisimulation']==b and e['reachable_real']==reach]),
        label_mismatches=sum(e['successor_label_mismatch'] for e in errors if e['initial_bisimulation']==b and e['reachable_real']==reach),
        behavioral_class_changes=sum(not e['targets_bisimilar_in_real'] for e in errors if e['initial_bisimulation']==b and e['reachable_real']==reach)) for b,reach in ((False,True),(True,True),(True,False))])
    same_distance=[e for e in errors if not e['initial_bisimulation'] and not e['target_goal_or_danger_distance_differs_in_real']]
    write(out/'distance_equal_but_behavior_differs.json',same_distance)
    print('Distance vectors do not explain these edges:',json.dumps(same_distance,ensure_ascii=False))


if __name__=='__main__':main()
