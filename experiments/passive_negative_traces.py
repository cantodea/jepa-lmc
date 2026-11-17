"""Predeclared, nonadaptive oracle-labelled offline negative-trace pilot.

Run prepare before train; learner arguments contain no environment or truth.
All source/binary historical artifacts are untouched. Outputs refuse overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.trace_sanity import (
    KnownProblem, Trace, sample_good_traces, make_hard_negatives,
)
from jepa_lmc.evaluation.finite_trace import (
    LTLBackendCache, prediction_system, evaluate_predictor,
)
from jepa_lmc.learning.finite_trace import (
    FiniteTraceData, fit_finite_model, predict_successors,
)
from jepa_lmc.learning.passive_negative import (
    analyze_version_space, fit_with_negative_traces,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def encode(t):
    return {'states': t.states, 'actions': t.actions}


def decode(t):
    return Trace(tuple(map(tuple, t['states'])), tuple(t['actions']))


def known(env):
    states = tuple(sorted(env.all_states()))
    return KnownProblem(states, frozenset([env.start]), tuple(sorted(env.ACTIONS)), {
        s: frozenset(({'danger'} if env.is_danger(s) else {'safe'}) |
                     ({'goal'} if env.is_goal(s) else set())) for s in states
    })


def candidates(problem, goods, *, seed, per_good):
    """No oracle, model, topology rule, error location, or unknown-pair filter."""
    rng = random.Random(seed)
    result = []
    for good in goods:
        for _ in range(per_good):
            length = rng.randint(1, len(good.actions))
            states, actions = list(good.states[:length+1]), list(good.actions[:length])
            if rng.randrange(2):
                actions[rng.randrange(length)] = rng.choice(problem.actions)
            else:
                states[rng.randint(1, length)] = rng.choice(problem.states)
            result.append(Trace(tuple(states), tuple(actions)))
    return tuple(result)


class ValidityOracle:
    """Returns one bit; accounting stays on acquisition side."""
    def __init__(self, env):
        self._env = env
        self.queries = self.edge_checks = 0

    def accepts(self, trace):
        self.queries += 1
        # Evaluate all edges for reproducible acquisition-cost accounting.
        valid = trace.states[0] == self._env.start
        for s, a, t in trace.edges:
            self.edge_checks += 1
            valid = (self._env.transition(s, a) == t) and valid
        return bool(valid)


def prepare(config, out):
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'protocol.lock.json', config)
    selected = [c for c in radius_stress_cases() if c.layout == config['layout']
                and c.rotation == config['rotation'] and c.family in config['families']]
    datasets, information = [], []
    for case in selected:
        env, problem = case.make_env(), known(case.make_env())
        for seed in config['seeds']:
            pool = sample_good_traces(env, seed=seed, count=max(config['budgets']), length=config['length'])
            raw = candidates(problem, pool, seed=seed+100, per_good=config['candidates_per_good'])
            oracle = ValidityOracle(env)
            labels = [oracle.accepts(t) for t in raw]
            held_pool = sample_good_traces(env, seed=seed+200, count=config['heldout_goods'], length=config['length'])
            held_raw = candidates(problem, held_pool, seed=seed+300,
                                  per_good=config['heldout_candidates']//len(held_pool))
            # Remove exact train candidates/traces, never filter informativeness.
            forbidden = set(pool) | set(raw)
            held_good = tuple(dict.fromkeys(t for t in held_pool if t not in forbidden))
            held_candidates = tuple(dict.fromkeys(t for t in held_raw if t not in forbidden))
            held_oracle = ValidityOracle(env)
            held_bad = tuple(t for t in held_candidates if not held_oracle.accepts(t))
            for budget in config['budgets']:
                size = budget * config['candidates_per_good']
                natural = pool[:budget]
                gs = natural + tuple(t for t, y in zip(raw[:size], labels[:size]) if y)
                bs = tuple(t for t, y in zip(raw[:size], labels[:size]) if not y)
                # Legacy replayed suffix is diagnostic oracle-assisted acquisition.
                legacy = make_hard_negatives(env, tuple(natural[i % budget] for i in range(len(bs))),
                                             problem.labels, seed=seed+400+budget)
                old = tuple(n.trace for n in legacy)
                ident = f'{case.name}_s{seed}_b{budget}'
                record = {'id': ident, 'case': case.name, 'family': case.family,
                          'seed': seed, 'budget': budget,
                          'goods': list(map(encode, gs)), 'natural_goods': list(map(encode, natural)),
                          'bads': list(map(encode, bs)), 'legacy_bads': list(map(encode, old)),
                          'candidates': [{'trace': encode(t), 'valid': y} for t,y in zip(raw[:size], labels[:size])],
                          'heldout_good': list(map(encode, held_good)), 'heldout_bad': list(map(encode, held_bad)),
                          'cost': {'natural_execution_steps': budget*config['length'],
                                   'validity_queries': size,
                                   'validity_edge_checks': sum(len(t.actions) for t in raw[:size]),
                                   'accepted_candidates': sum(labels[:size]), 'rejected_candidates': len(bs),
                                   'legacy_validation_steps': len(old)*config['length'],
                                   'legacy_suffix_replay_steps': sum(config['length']-n.corrupted_step-1 for n in legacy),
                                   'heldout_execution_steps': len(held_pool)*config['length'],
                                   'heldout_validity_queries': held_oracle.queries,
                                   'heldout_edge_checks': held_oracle.edge_checks,
                                   'nested_budgets_reuse_acquisition': True}}
                datasets.append(record)
                analyses = {arm: analyze_version_space(problem, gs, negatives, max_nodes=250000)
                            for arm, negatives in [('A', ()), ('B', old), ('C', bs)]}
                information.append({'id': ident, **analyses})
                print('FORMAL', ident, {k: v.get('status') for k,v in analyses.items()}, flush=True)
    dump(out / 'datasets.json', datasets)
    dump(out / 'information.json', information)
    dump(out / 'pretraining_manifest.json', {
        'created_unix': time.time(), 'neural_training_started': False,
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in out.iterdir() if p.is_file()},
        'python': sys.version, 'torch': torch.__version__,
    })


def accepts(pred, t):
    return all(pred.get((s,a)) == target for s,a,target in t.edges)


def train(config, out, nuxmv):
    if config != json.loads((out/'protocol.lock.json').read_text(encoding='utf-8')):
        raise ValueError('Training protocol differs from frozen acquisition protocol')
    manifest=json.loads((out/'pretraining_manifest.json').read_text(encoding='utf-8'))
    for filename, expected in manifest['files_sha256'].items():
        if hashlib.sha256((out/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen input was modified: '+filename)
    datasets = json.loads((out/'datasets.json').read_text(encoding='utf-8'))
    info = {r['id']:r for r in json.loads((out/'information.json').read_text(encoding='utf-8'))}
    baseline = json.loads(Path(config['baseline_config']).read_text())
    dump(out/'training_config.json', baseline)
    cases = {c.name:c for c in radius_stress_cases()}
    backend = LTLBackendCache(executable=nuxmv, log_path=out/'ltl_backend.jsonl.gz')
    (out/'checkpoints').mkdir()
    with (out/'results.jsonl').open('x', encoding='utf-8') as handle:
        for d in datasets:
            env = cases[d['case']].make_env()
            problem = known(env)
            gs, bs, old = [tuple(map(decode, d[k])) for k in ('goods','bads','legacy_bads')]
            data = FiniteTraceData.from_traces(problem, gs, height=env.height, width=env.width)
            heldgood, heldbad = [tuple(map(decode,d[k])) for k in ('heldout_good','heldout_bad')]
            # No model selection follows scoring; each fixed arm is retained.
            for kind, arm in [('symbolic','A'),('symbolic','C')] + [(k,a) for k in ('mlp','jepa') for a in ('A','B','C')]:
                started = time.perf_counter()
                if kind == 'symbolic':
                    analysis = info[d['id']][arm]
                    if analysis.get('status') != 'exact':
                        raise RuntimeError('Symbolic counting did not finish: '+d['id'])
                    pred = {(tuple(s),a):tuple(t) for s,a,t in analysis['completion_edges']}
                    meta = {'kind':'symbolic','training_seconds':0,'completion':'lexicographic',
                            'training_sha256':data.training_sha256, 'analysis_seconds':analysis.get('analysis_seconds')}
                else:
                    if arm == 'A':
                        model, meta = fit_finite_model(data,kind=kind,seed=d['seed'],config=baseline)
                    else:
                        model, meta = fit_with_negative_traces(data, old if arm=='B' else bs,
                            kind=kind,seed=d['seed'],config=baseline,negative_weight=config['negative_weight'])
                    pred = predict_successors(model,data,kind=kind)
                    checkpoint = out/'checkpoints'/f"{d['id']}_{kind}_{arm}.pt"
                    torch.save({'state_dict':model.state_dict(),'metadata':meta},checkpoint)
                    meta['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                real = gridworld_to_transition_system(env)
                evaluation = evaluate_predictor(problem,data.observed_pairs,pred,real=real,
                    formal_system=prediction_system(problem,pred),heldout=heldgood,backend=backend)
                moving = [(s,a) for s in problem.states for a in problem.actions
                          if (s,a) not in data.observed_pairs and env.transition(s,a)!=s]
                moving_correct = sum(pred[s,a]==env.transition(s,a) for s,a in moving)
                evaluation['unseen_moving'] = {'correct':moving_correct,'total':len(moving),
                                              'accuracy':moving_correct/len(moving) if moving else None}
                evaluation['heldout_rejection'] = {'correct':sum(not accepts(pred,t) for t in heldbad),
                                                   'total':len(heldbad)}
                evaluation['sample_consistency'] = {'good_accepted':sum(accepts(pred,t) for t in gs),
                    'good_count':len(gs),'natural_bad_rejected':sum(not accepts(pred,t) for t in bs),
                    'natural_bad_count':len(bs),'legacy_bad_rejected':sum(not accepts(pred,t) for t in old),
                    'legacy_bad_count':len(old)}
                row = {k:d[k] for k in ('id','case','family','seed','budget')}
                row.update(kind=kind,arm=arm,training=meta,evaluation=evaluation,
                           predictions=[(s,a,t) for (s,a),t in pred.items()],
                           seconds=time.perf_counter()-started)
                handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush()
                print('RESULT',d['id'],kind,arm,'unseen',evaluation['transitions']['unseen']['accuracy'],flush=True)
    dump(out/'completed.json',{'models':len(datasets)*8,'backend_calls':backend.calls,
         'backend_cache_hits':backend.hits,'completed_unix':time.time()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','train'],required=True)
    parser.add_argument('--config',default='configs/passive_negative_traces_20261117.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--nuxmv',default=os.environ.get('NUXMV_BINARY'))
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.phase=='prepare': prepare(cfg,Path(args.output))
    else: train(cfg,Path(args.output),args.nuxmv)
