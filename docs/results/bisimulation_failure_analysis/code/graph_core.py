"""Post-hoc graph replay; no training/inference. Requires hash-verified original exports.
All repairs use oracle R exclusively after prediction. Exhaustive subset search,
not a learned or online repair method. See README.md for evidence limitations.
"""
import argparse
import csv
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def reachable(g):
    seen=set(g.initial_states); todo=list(seen)
    while todo:
        for t in g.successors(todo.pop()):
            if t not in seen:
                seen.add(t); todo.append(t)
    return seen


def table(g):
    result={}
    for s in g.states:
        for edge in g.action_successors(s):
            key=(s,edge.action)
            if key in result:
                raise ValueError('Expected exactly one edge for each state-action.')
            result[key]=edge.target
    return result


def dependency_dag(left, right, relation):
    """Rooted elimination certificate, branching over EVERY permitted reply.

    This is not a trace or CTL/LTL counterexample. Full certificate is audited
    by upstream audit_greatest_relation before this function is called.
    """
    eliminated={(e.left,e.right):e for e in relation.eliminations}
    roots=[(s,t) for s in left.initial_states for t in right.initial_states if (s,t) not in relation.pairs]
    visited={}
    def visit(pair):
        if pair in visited: return
        e=eliminated[pair]; node=asdict(e); node['replies']=[]
        if e.side != 'label':
            opposite=right if e.side=='left' else left
            state=e.right if e.side=='left' else e.left
            for edge in opposite.action_successors(state):
                if relation.action_sensitive and edge.action != e.action: continue
                child=(e.target,edge.target) if e.side=='left' else (edge.target,e.target)
                assert child in eliminated and eliminated[child].round < e.round
                node['replies'].append(dict(action=edge.action,target=edge.target,child=child))
                visit(child)
        else:
            node['left_labels']=sorted(left.propositions(e.left))
            node['right_labels']=sorted(right.propositions(e.right))
        visited[pair]=node
    for pair in roots: visit(pair)
    return dict(roots=roots,nodes=[visited[p] for p in sorted(visited)],
                type='verified_fixed_point_elimination_DAG_not_linear_trace')


def exact_repairs(real, learned, make, greatest, audit, partition, max_errors=12):
    rt,pt=table(real),table(learned)
    keys=sorted(k for k in rt if rt[k]!=pt[k])
    if len(keys)>max_errors:
        raise ValueError('Exact search bound exceeded; no minimum repair claim generated.')
    trials=[]
    # Exhaust all subsets: no monotonicity assumption about bisimulation.
    for size in range(len(keys)+1):
        for subset in itertools.combinations(keys,size):
            updated=dict(pt)
            for k in subset: updated[k]=rt[k]
            g=make(learned,updated)
            trial=dict(repairs=[list(k) for k in subset],size=size)
            for sensitive in (False,True):
                rel=greatest(real,g,kind='bisimulation',action_sensitive=sensitive)
                audit(real,g,rel)
                assert rel.pairs==partition(real,g,action_sensitive=sensitive)
                trial['sensitive' if sensitive else 'insensitive']=rel.relates_initials(real,g)
            trials.append(trial)
    assert trials[-1]['insensitive'] and trials[-1]['sensitive']
    minima={}
    for mode in ('insensitive','sensitive'):
        n=min(t['size'] for t in trials if t[mode])
        minima[mode]=dict(minimum=n,all_minimum_sets=[t['repairs'] for t in trials if t[mode] and t['size']==n])
    return dict(error_keys=keys,trials=trials,minimum_repairs=minima,proof='all_error_subsets_exhausted')

