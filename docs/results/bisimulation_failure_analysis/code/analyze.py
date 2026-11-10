"""Frozen-graph post-hoc analysis. No training or inference; no repository writes.

python analyze.py --repo REPO --data ROOT_CONTAINING_outputs --out EMPTY_DIR
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from graph_core import dependency_dag, exact_repairs, reachable, table

REV='539410dec1d5c4e4433bd9a57b10a309cd9a850d'
GRAPH_HASH='8d51e45e76bf1f5a0c70732fdcfa90f72b08780c0558e6d8bea39793b2eb5d98'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)


def check_dag(left,right,dag,sensitive):
    """Check exported rooted DAG independently of stored full relation.
    Each node supplies an existing challenge and ALL legal replies; ranks fall.
    """
    assert {tuple(p) for p in dag['roots']}=={(s,t) for s in left.initial_states for t in right.initial_states}
    nodes={(n['left'],n['right']):n for n in dag['nodes']}
    assert len(nodes)==len(dag['nodes'])
    reached=set()
    def visit(pair):
        if pair in reached: return
        reached.add(pair); n=nodes[pair]; s,t=pair
        if n['side']=='label':
            assert n['round']==0 and left.propositions(s)!=right.propositions(t)
            assert not n['replies']; return
        assert n['round']>0 and left.propositions(s)==right.propositions(t)
        reverse=n['side']=='right'; assert n['side'] in ('left','right')
        g,h=(right,left) if reverse else (left,right); p,q=(t,s) if reverse else (s,t)
        assert any(e.target==n['target'] and (not sensitive or e.action==n['action']) for e in g.action_successors(p))
        expected=set()
        for edge in h.action_successors(q):
            if sensitive and edge.action!=n['action']: continue
            child=(edge.target,n['target']) if reverse else (n['target'],edge.target)
            expected.add((edge.action,edge.target,child))
        actual={(r['action'],r['target'],tuple(r['child'])) for r in n['replies']}
        assert actual==expected
        for _,_,child in expected:
            assert child in nodes and nodes[child]['round']<n['round']
            visit(child)
    for root in dag['roots']: visit(tuple(root))
    assert reached==set(nodes)


def all_safe_product_check(real,pred):
    """Explicit closure check of reachable product, also respecting actions."""
    rr,pr=reachable(real),reachable(pred)
    if any(g.propositions(s)!=frozenset(['safe']) for g,rs in ((real,rr),(pred,pr)) for s in rs): return False
    for s in rr:
        for t in pr:
            for g,u,h,v,other in ((real,s,pred,t,pr),(pred,t,real,s,rr)):
                for e in g.action_successors(u):
                    assert any(f.action==e.action and f.target in other for f in h.action_successors(v))
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args(); started=time.perf_counter(); sys.dont_write_bytecode=True
    repo=a.repo.resolve(); root=a.data.resolve(); out=a.out.resolve()
    git=['git','-c',f'safe.directory={repo.as_posix()}','-C',str(repo)]
    assert subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()==REV
    subprocess.run(git+['diff','--exit-code'],check=True)
    if out.exists() and any(out.iterdir()): parser.error('Output directory must be empty.')
    sys.path[:0]=[str(repo/'src'),str(repo)]
    from experiments.evaluate_top1_ltl import read_run,load_system
    from experiments.evaluate_structural import read_linked_ltl
    from jepa_lmc.verification.behavioral_relations import greatest_relation,audit_greatest_relation,bisimulation_by_partition,BehavioralRelation,Elimination
    from jepa_lmc.verification.transition_system import ExplicitTransitionSystem as TS
    from jepa_lmc.verification.ctl import CTLModelChecker
    from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
    from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
    from jepa_lmc.verification.gridworld import gridworld_to_transition_system
    prefix=Path('outputs/top1_quality/round2'); run=root/prefix/'behavioral/best'; ltl_dir=root/prefix/'temporal/best_ltl'
    published=json.loads((repo/'docs/results/structural_evaluation/ranking/report.json').read_text(encoding='utf-8'))
    inputs={}
    for directory,hashes in ((run,published['provenance']['source_sha256']),(ltl_dir,published['provenance']['ltl_source_sha256'])):
        for name,h in hashes.items():
            path=directory/name; actual=hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual==h,(path,actual,h); inputs[path.relative_to(root).as_posix()]=actual
    assert inputs[(prefix/'behavioral/best/relations.jsonl').as_posix()]==GRAPH_HASH
    _,records=read_run(run)
    reference={(r['seed'],r['case']):r for r in published['cases']}
    assert len(records)==72 and len({(r['seed'],r['case']) for r in records})==72
    assert {(r['seed'],r['case']) for r in records}==set(reference)
    ltl,_=read_linked_ltl(ltl_dir,GRAPH_HASH,records)
    original_envs={c.name:c.make_env() for c in radius_stress_cases()}
    out.mkdir(parents=True,exist_ok=True)
    audit_counts=Counter();cases=[];errors=[];failures=[];trials=[];repair_results=[];pass_repairs=[];cert_summary=[]
    properties=[];ctl_saved={}
    with (run/'ctl_queries.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            key=(int(row['seed']),row['case'],int(row['row']),int(row['column']),row['property'])
            assert key not in ctl_saved; ctl_saved[key]=row
    transition_rows={}
    for seed in (20260804,20260805,20260806):
        path=root/prefix/f'ranking/seed_{seed}/evaluation/transitions.csv'
        inputs[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                if row['split']!='stress': continue
                key=(seed,row['map'],int(row['state_row']),int(row['state_col']),int(row['action']))
                assert key not in transition_rows; transition_rows[key]=row
    assert len(transition_rows)==8928
    def make(g,t):
        return TS(states=g.states,initial_states=g.initial_states,labels={s:g.propositions(s) for s in g.states},
            transitions={s:[(a,v) for (u,a),v in sorted(t.items()) if u==s] for s in g.states})
    def audited(left,right,rel):
        audit_greatest_relation(left,right,rel); audit_counts['certificates']+=1
    def partition(left,right,**kw):
        audit_counts['partition_checks']+=1;return bisimulation_by_partition(left,right,**kw)
    for r in records:
        seed,name=r['seed'],r['case']; key=(seed,name); tag=f'{seed}_{name}'; pub=reference[key]
        real,pred=(load_system(r,n) for n in ('real','top1'))
        rt,pt=table(real),table(pred); assert rt.keys()==pt.keys()
        assert all({e.action for e in real.action_successors(s)}=={0,1,2,3} for s in real.states)
        # Verify exact known maps, not generated new maps; coordinates, labels, initial and R.
        exact=gridworld_to_transition_system(original_envs[name])
        assert set(map(tuple,r['states']))==set(exact.states)
        assert {tuple(r['states'][s]) for s in real.initial_states}==set(exact.initial_states)
        for s in real.states:
            xy=tuple(r['states'][s]);assert real.propositions(s)==exact.propositions(xy)
            expected={(e.action,e.target) for e in exact.action_successors(xy)}
            assert {(e.action,tuple(r['states'][e.target])) for e in real.action_successors(s)}==expected
        for (s,act),target in rt.items():
            row=transition_rows[(seed,name,*r['states'][s],act)]
            assert [int(row['true_row']),int(row['true_col'])]==r['states'][target]
            assert [int(row['pred_row']),int(row['pred_col'])]==r['states'][pt[s,act]]
            assert (row['correct']=='True')==(target==pt[s,act])
            audit_counts['prediction_rows_checked']+=1
        rels={}
        for saved in r['relations']:
            g,h=(pred,real) if saved['name']=='top1_to_real' else (real,pred)
            rel=greatest_relation(g,h,kind=saved['kind'],action_sensitive=saved['action_sensitive']);audited(g,h,rel)
            assert rel.pairs==frozenset(map(tuple,saved['pairs'])) and rel.rounds==saved['rounds']
            old={(e['left'],e['right']):e for e in saved['eliminations']}
            assert {(e.left,e.right):e.round for e in rel.eliminations}=={k:e['round'] for k,e in old.items()}
            # A valid certificate may choose a different first failing edge.
            audit_counts['alternative_valid_challenge_choices']+=sum(asdict(e)!=old[e.left,e.right] for e in rel.eliminations)
            historical=BehavioralRelation(saved['kind'],saved['action_sensitive'],frozenset(map(tuple,saved['pairs'])),tuple(Elimination(**e) for e in saved['eliminations']),saved['rounds'])
            audited(g,h,historical);audit_counts['historical_certificates_checked']+=1
            audit_counts['saved_relations_reproduced']+=1
            if saved['name']=='bisimulation':
                assert rel.pairs==partition(g,h,action_sensitive=saved['action_sensitive'])
                rels[saved['action_sensitive']]=rel
        assert len(r['relations'])==6 and set(rels)=={False,True}
        passed=rels[False].relates_initials(real,pred); identity=rels[False].relates_identity(real,pred)
        for sensitive,rel in rels.items():
            mode='action_sensitive' if sensitive else 'action_insensitive'
            assert rel.relates_initials(real,pred)==pub['behavioral'][mode]['initial_bisimulation']
            assert rel.relates_identity(real,pred)==pub['behavioral'][mode]['identity_bisimulation']
        realrel=greatest_relation(real,real,kind='bisimulation');audited(real,real,realrel)
        assert realrel.pairs==partition(real,real)
        rr,pr=reachable(real),reachable(pred);errs=[]
        for (s,act),target in sorted(rt.items()):
            predicted=pt[s,act]
            if target==predicted:continue
            source_label=sorted(real.propositions(s));true_label=sorted(real.propositions(target));pred_label=sorted(pred.propositions(predicted))
            raw=transition_rows[(seed,name,*r['states'][s],act)]
            err=dict(edge_id=f'{tag}:s{s}:a{act}',seed=seed,case=name,family=pub['family'],source=s,source_coord=r['states'][s],action=act,
                action_name=('up','down','left','right')[act],true_successor=target,true_coord=r['states'][target],pred_successor=predicted,pred_coord=r['states'][predicted],
                source_labels=source_label,true_labels=true_label,pred_labels=pred_label,
                reachable_real=s in rr,reachable_pred=s in pr,
                motion='move_to_self_loop' if predicted==s else ('self_loop_to_move' if target==s else 'move_to_wrong_move'),
                successor_label_mismatch=true_label!=pred_label,involves_goal='goal' in source_label+true_label+pred_label,
                involves_danger='danger' in source_label+true_label+pred_label,involves_safe='safe' in source_label+true_label+pred_label,
                targets_bisimilar_in_real=(target,predicted) in realrel.pairs,
                initial_bisimulation=passed,identity_bisimulation=identity,
                true_motion_saved=raw['motion'],true_kind_saved=raw['kind'])
            errs.append(err)
        assert len(rt)-len(errs)==pub['diagnostics']['top1']['correct'] and len(rt)==pub['diagnostics']['top1']['total']
        checkers=[CTLModelChecker(g) for g in (real,pred)];ctl_agreements=[];initial_ctl=[]
        for prop in default_ctl_suite():
            verdicts=[c.satisfying_states(prop.formula) for c in checkers]
            for s in sorted(real.states):
                row=ctl_saved[(seed,name,*r['states'][s],prop.name)]
                actual=[s in v for v in verdicts]
                assert all(row[f] in ('True','False') for f in ('real','top1','initial','agreement'))
                assert [row['real']=='True',row['top1']=='True']==actual
                assert (row['initial']=='True')==(s==r['initial'])
                assert (row['agreement']=='True')==(actual[0]==actual[1])
                audit_counts['native_ctl_verdicts_reproduced']+=2
                ctl_agreements.append(actual[0]==actual[1])
                if s==r['initial']:
                    initial_ctl.append(actual[0]==actual[1]);properties.append(dict(seed=seed,case=name,logic='CTL',property=prop.name,real=actual[0],pred=actual[1],agreement=actual[0]==actual[1]))
        ld=ltl[key];assert all(initial_ctl)==pub['diagnostics']['initial_all_six_ctl_agreement']
        assert ld['initial_all_six_agreement']==pub['diagnostics']['ltl']['initial_all_six_agreement']
        assert ld['matched']==pub['diagnostics']['ltl']['matched']
        case=dict(seed=seed,case=name,family=pub['family'],states=len(real.states),transitions=len(rt),errors=len(errs),error_rate=len(errs)/len(rt),
            reachable_states_real=len(rr),reachable_states_pred=len(pr),same_reachable_sets=rr==pr,
            reachable_errors_real=sum(e['reachable_real'] for e in errs),reachable_errors_pred=sum(e['reachable_pred'] for e in errs),
            move_to_self_loop=sum(e['motion']=='move_to_self_loop' for e in errs),self_loop_to_move=sum(e['motion']=='self_loop_to_move' for e in errs),
            successor_label_mismatches=sum(e['successor_label_mismatch'] for e in errs),
            involves_goal=sum(e['involves_goal'] for e in errs),involves_danger=sum(e['involves_danger'] for e in errs),involves_safe=sum(e['involves_safe'] for e in errs),
            initial_bisimulation=passed,identity_bisimulation=identity,action_sensitive_initial=rels[True].relates_initials(real,pred),
            action_sensitive_identity=rels[True].relates_identity(real,pred),full_bisimulation_relations_equal=rels[False].pairs==rels[True].pairs,
            ctl_all_six=all(initial_ctl),ltl_all_six=ld['initial_all_six_agreement'],all_state_ctl=all(ctl_agreements),all_state_ltl=ld['matched']==ld['comparisons'],
            reachable_product_safe_bisimulation=all_safe_product_check(real,pred) if passed else False,
            failure_reason='none' if passed else 'successor_matching_failure',initial_elimination_round=None,initial_challenge_side=None,
            initial_challenge_target=None,initial_challenge_actions=None,action_sensitive_initial_elimination_round=None)
        certificate=dict(seed=seed,case=name,states=r['states'],labels=r['labels'],initial=r['initial'],semantics={})
        for sensitive,rel in rels.items():
            mode='action_sensitive' if sensitive else 'action_insensitive'
            dag=dependency_dag(real,pred,rel) if not passed else None
            if dag:
                check_dag(real,pred,dag,sensitive);audit_counts['rooted_dags_checked']+=1
                root_node=next(n for n in dag['nodes'] if n['left']==r['initial'] and n['right']==r['initial'])
                challenge_system=real if root_node['side']=='left' else pred
                actions=[e.action for e in challenge_system.action_successors(r['initial']) if e.target==root_node['target'] and (not sensitive or e.action==root_node['action'])]
                cert_summary.append(dict(seed=seed,case=name,semantics=mode,initial=r['initial'],initial_coord=r['states'][r['initial']],round=root_node['round'],side=root_node['side'],target=root_node['target'],target_coord=r['states'][root_node['target']],actual_actions=actions,dag_nodes=len(dag['nodes']),label_leaves=sum(n['side']=='label' for n in dag['nodes']),root_replies=root_node['replies']))
                if not sensitive:
                    case.update(initial_elimination_round=root_node['round'],initial_challenge_side=root_node['side'],initial_challenge_target=root_node['target'],initial_challenge_actions=actions)
                else:case['action_sensitive_initial_elimination_round']=root_node['round']
            certificate['semantics'][mode]=dict(pairs=sorted(rel.pairs),rounds=rel.rounds,eliminations=[asdict(e) for e in rel.eliminations],initial_failure_dag=dag)
        write(out/'certificates'/f'{tag}.json',certificate)
        if errs:
            repair=exact_repairs(real,pred,make,greatest_relation,audited,partition)
            error_key={(e['source'],e['action']):e for e in errs}
            for t in repair['trials']:
                trials.append(dict(seed=seed,case=name,original_initial_bisimulation=passed,size=t['size'],repaired_edges=[error_key[tuple(k)]['edge_id'] for k in t['repairs']],insensitive=t['insensitive'],sensitive=t['sensitive']))
            for k,e in error_key.items():
                alone=next(t for t in repair['trials'] if len(t['repairs'])==len(errs)-1 and list(k) not in t['repairs'])
                single=next(t for t in repair['trials'] if t['repairs']==[list(k)])
                e.update(single_repair_restores=single['insensitive'],only_this_error_retained_passes=alone['insensitive'],
                    can_remain_in_any_successful_subset=any(t['insensitive'] and list(k) not in t['repairs'] for t in repair['trials']))
            item=dict(seed=seed,case=name,errors=len(errs),subsets_evaluated=len(repair['trials']),
                minimum_repairs=repair['minimum_repairs']['insensitive']['minimum'],minimum_repairs_sensitive=repair['minimum_repairs']['sensitive']['minimum'],
                all_minimum_sets=[[error_key[tuple(k)]['edge_id'] for k in ss] for ss in repair['minimum_repairs']['insensitive']['all_minimum_sets']],
                successful_subsets=sum(t['insensitive'] for t in repair['trials']),
                single_edge_successes=[e['edge_id'] for e in errs if e['single_repair_restores']],
                edges_that_can_remain=[e['edge_id'] for e in errs if e['can_remain_in_any_successful_subset']],
                proof='exhaustive_all_error_subsets_both_semantics_audited_partition_checked')
            (pass_repairs if passed else repair_results).append(item)
            if not passed:failures.append(case)
        errors.extend(errs);cases.append(case)
        print(f'{len(cases)}/72 {tag}: errors={len(errs)}, initial={passed}',flush=True)
    # Preserve original LTL verdicts, linking them to the verified graph, not rerunning nuXmv.
    with (ltl_dir/'queries.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['initial']=='True':properties.append(dict(seed=int(row['seed']),case=row['case'],logic='LTL',property=row['property'],real=row['real']=='True',pred=row['top1']=='True',agreement=row['agreement']=='True'))
    assert len(cases)==72 and len(errors)==96 and len(repair_results)==32 and len(pass_repairs)==7
    assert len(ctl_saved)==13392 and audit_counts['native_ctl_verdicts_reproduced']==26784
    assert sum(c['transitions'] for c in cases)==8928 and sum(c['initial_bisimulation'] for c in cases)==40
    assert audit_counts['saved_relations_reproduced']==432
    def group(cs):
        keys={(c['seed'],c['case']) for c in cs};es=[e for e in errors if (e['seed'],e['case']) in keys]
        return dict(cases=len(cs),transitions=sum(c['transitions'] for c in cs),errors=len(es),
            initial_pass=sum(c['initial_bisimulation'] for c in cs),identity_pass=sum(c['identity_bisimulation'] for c in cs),
            reachable_errors=sum(e['reachable_real'] for e in es),unreachable_errors=sum(not e['reachable_real'] for e in es),
            motion=dict(Counter(e['motion'] for e in es)),successor_label_mismatches=sum(e['successor_label_mismatch'] for e in es),
            successor_pairs_bisimilar_in_real=sum(e['targets_bisimilar_in_real'] for e in es),
            involves_goal=sum(e['involves_goal'] for e in es),involves_danger=sum(e['involves_danger'] for e in es),
            ctl_all_six=sum(c['ctl_all_six'] for c in cs),ltl_all_six=sum(c['ltl_all_six'] for c in cs),
            both_all_state=sum(c['all_state_ctl'] and c['all_state_ltl'] for c in cs))
    summary=dict(overall=group(cases),by_initial={str(b):group([c for c in cases if c['initial_bisimulation']==b]) for b in (True,False)},
        by_family={f:group([c for c in cases if c['family']==f]) for f in sorted({c['family'] for c in cases})},
        minimum_repairs_histogram=dict(Counter(x['minimum_repairs'] for x in repair_results)),
        failed_subsets=sum(r['subsets_evaluated'] for r in repair_results),passing_subsets=sum(r['subsets_evaluated'] for r in pass_repairs),
        fail_single_repair_trials=sum(len(x['single_edge_successes']) for x in repair_results),
        counts=dict(audit_counts),same_reachability_for_error_sources=all(e['reachable_real']==e['reachable_pred'] for e in errors),
        cases_same_reachable_sets=sum(c['same_reachable_sets'] for c in cases),full_bisimulation_relations_equal=sum(c['full_bisimulation_relations_equal'] for c in cases),
        all_failed_require_all_errors=all(r['minimum_repairs']==r['errors'] for r in repair_results),
        all_passing_subsets_pass=all(r['successful_subsets']==r['subsets_evaluated'] for r in pass_repairs),
        initial_round_histogram=dict(Counter(c['initial_elimination_round'] for c in failures)))
    csvwrite(out/'cases_72.csv',cases);csvwrite(out/'errors_96.csv',errors);csvwrite(out/'repairs_32.csv',repair_results)
    csvwrite(out/'passing_repairs_7.csv',pass_repairs);csvwrite(out/'repair_trials_all.csv',trials)
    csvwrite(out/'certificate_summary_64.csv',cert_summary);csvwrite(out/'initial_properties_864.csv',properties)
    csvwrite(out/'ctl_agree_bisim_fail_28.csv',[c for c in cases if c['ctl_all_six'] and not c['initial_bisimulation']])
    write(out/'summary.json',summary);write(out/'errors_96.json',errors);write(out/'cases_72.json',cases)
    write(out/'provenance.json',dict(revision=REV,python=platform.python_version(),inputs_sha256=inputs,
        duration_seconds=time.perf_counter()-started,training=False,inference=False,
        ctl='recomputed_all_states_both_graphs',ltl='original_backend_verdicts_graph_and_query_hash_verified_not_rerun',
        repair='exact_enumeration_of_error_corrections_only',source_status=subprocess.check_output(git+['status','--porcelain'],text=True),
        implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
