"""Descriptive, paired summaries; no edges-as-independent significance tests."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from jepa_lmc.benchmarks.radius_stress import radius_stress_cases


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out):
    records = [json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    assert len(records) == json.loads((out/'completed.json').read_text())['models']
    info = {r['id']:r for r in json.loads((out/'information.json').read_text())}
    environments = {c.name: c.make_env() for c in radius_stress_cases()}
    rows=[]
    for r in records:
        e=r['evaluation']; f=e['formal']; d=f['diagnostics']; rej=e['heldout_rejection']
        row={k:r[k] for k in ('id','case','family','seed','budget','kind','arm')}
        row.update(unseen=e['transitions']['unseen']['accuracy'],
                   unseen_moving=e['unseen_moving']['accuracy'],
                   observed=e['transitions']['observed']['accuracy'],
                   heldout_accept=e['heldout']['acceptance']['accuracy'],
                   heldout_reject=rej['correct']/rej['total'] if rej['total'] else None,
                   ctl_all=d['all_state_ctl']['fraction'],
                   ctl_initial=d['initial_all_six_ctl_agreement'],
                   ltl_all=d['ltl']['matched']/d['ltl']['comparisons'],
                   ltl_initial=d['ltl']['initial_all_six_agreement'],
                   train_seconds=r['training']['training_seconds'],
                   natural_information_bits=info[r['id']]['C']['information_bits'])
        prediction={(tuple(s),a):tuple(t) for s,a,t in r['predictions']}
        supports=info[r['id']]['C']['candidate_successors']
        constrained=[(tuple(x['source']),x['action']) for x in supports
                     if not x['observed'] and x['candidate_count']<len(environments[r['case']].all_states())]
        row['reduced_support_pairs']=len(constrained)
        row['reduced_support_accuracy']=(sum(prediction[s,a]==environments[r['case']].transition(s,a)
                                                  for s,a in constrained)/len(constrained)) if constrained else None
        row['marginal_support_violations']=sum(prediction[tuple(x['source']),x['action']] not in
                                              list(map(tuple,x['successors'])) for x in supports)
        consistency=e['sample_consistency']
        row['train_good_accept']=consistency['good_accepted']/consistency['good_count']
        row['train_bad_reject']=consistency['natural_bad_rejected']/consistency['natural_bad_count'] if consistency['natural_bad_count'] else None
        for s, short in [('action_sensitive','as'),('action_insensitive','ai')]:
            for name in ['simulation_real_to_learned','simulation_learned_to_real',
                         'initial_bisimulation','identity_bisimulation','strict_identity_bisimulation']:
                row[f'{short}_{name}']=f['behavioral'][s][name]
        rows.append(row)
    write_csv(out/'per_case.csv',rows)
    metric_names=[k for k in rows[0] if k not in ('id','case','family','seed','budget','kind','arm')]
    summaries=[]
    for grouping in [('kind','arm'),('family','kind','arm'),('budget','kind','arm'),('seed','kind','arm')]:
        groups=defaultdict(list)
        for row in rows: groups[tuple(row[k] for k in grouping)].append(row)
        for key, group in groups.items():
            summary={'grouping':'+'.join(grouping),'group': '|'.join(map(str,key)),'cases':len(group)}
            for metric in metric_names:
                values=[float(row[metric]) for row in group if row[metric] is not None]
                summary[metric]=statistics.mean(values) if values else None
            summaries.append(summary)
    write_csv(out/'aggregates.csv',summaries)
    lookup={(r['id'],r['kind'],r['arm']):r for r in rows}
    pairs=[]
    for row in rows:
        if row['arm']=='A': continue
        base=lookup[row['id'],row['kind'],'A']
        contrast={k:row[k] for k in ('id','family','seed','budget','kind','arm','natural_information_bits')}
        for metric in metric_names:
            if metric=='natural_information_bits': continue
            contrast['delta_'+metric]=float(row[metric])-float(base[metric]) if row[metric] is not None and base[metric] is not None else None
        pairs.append(contrast)
    write_csv(out/'paired_contrasts.csv',pairs)
    block=[]
    for kind in ('symbolic','mlp','jepa'):
        for arm in ('B','C'):
            for family in sorted({r['family'] for r in rows}):
                group=[r for r in pairs if (r['kind'],r['arm'],r['family'])==(kind,arm,family)]
                if group:
                    block.append({'kind':kind,'arm':arm,'family':family,'repeated_cases':len(group),
                                  'delta_unseen':statistics.mean(r['delta_unseen'] for r in group),
                                  'delta_moving':statistics.mean(r['delta_unseen_moving'] for r in group)})
    write_csv(out/'structural_block_contrasts.csv',block)
    info_rows=[]
    for ident, record in info.items():
        c=record['C']
        info_rows.append({'id':ident,'unknown_pairs':c['unknown_pairs'],'bad_count':c['negative_count'],
                          'nonredundant_given_G':sum(not d['redundant_given_good'] for d in c['negative_diagnostics']),
                          'sequential_nonredundant':sum(not d['sequential_redundant'] for d in c['negative_diagnostics']),
                          'reduced_support_pairs':c['unknown_pairs_with_reduced_support'],
                          'information_bits':c['information_bits'],
                          'count_G':c['model_count_good'],'count_GB':c['model_count_good_bad'],
                          'legacy_information_bits':record['B']['information_bits']})
    write_csv(out/'information_summary.csv',info_rows)
    changes=[]
    raw_lookup={(r['id'],r['kind'],r['arm']):r for r in records}
    for r in records:
        if r['arm']!='C':continue
        a=raw_lookup[r['id'],r['kind'],'A']
        before={(tuple(s),act):tuple(t) for s,act,t in a['predictions']}
        after={(tuple(s),act):tuple(t) for s,act,t in r['predictions']}
        constrained={(tuple(x['source']),x['action']) for x in info[r['id']]['C']['candidate_successors']
                     if not x['observed'] and x['candidate_count']<len(environments[r['case']].all_states())}
        for pair in before:
            if before[pair]!=after[pair]:
                truth=environments[r['case']].transition(*pair)
                changes.append({'id':r['id'],'kind':r['kind'],'source':str(pair[0]),'action':pair[1],
                                'support_reduced':pair in constrained,'before':str(before[pair]),'after':str(after[pair]),
                                'truth':str(truth),'before_correct':before[pair]==truth,'after_correct':after[pair]==truth})
    if changes:write_csv(out/'changed_predictions_C_vs_A.csv',changes)
    with (out/'summary.json').open('x') as f:
        json.dump({'aggregates':summaries,'structural_block_contrasts':block,
                   'information':info_rows,
                   'statistical_scope':'Descriptive paired pilot; only three related fixed structures. No independence or significance claim.'},f,indent=2)
    # Plain SVG: reproducible data figure, no plotting-library dependency.
    width,height=900,420
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial" font-size="13" fill="#172b4d">',
           '<text x="40" y="30" font-size="20">Negative information and paired change in unseen accuracy</text>',
           '<text x="40" y="53">C minus A; fixed maps, repeated seeds/budgets; descriptive pilot</text>']
    vals=[100*r['delta_unseen'] for r in pairs if r['arm']=='C']
    lim=max(1,max(abs(v) for v in vals)*1.2)
    for v in (-lim,0,lim):
        y=220-v/lim*120
        parts.extend([f'<line x1="80" y1="{y}" x2="820" y2="{y}" stroke="#ccd5df"/>',f'<text x="15" y="{y+4}">{v:.2f} pp</text>'])
    colors={'mlp':'#1565c0','jepa':'#c62828','symbolic':'#546e7a'}
    for r in pairs:
        if r['arm']!='C':continue
        x=80+700*r['natural_information_bits']/0.105
        y=220-100*r['delta_unseen']/lim*120
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{colors[r["kind"]]}" fill-opacity="0.65"><title>{r["id"]} {r["kind"]}: {100*r["delta_unseen"]:.4f} pp</title></circle>')
    for x in (0,0.025,0.05,0.075,0.1):
        parts.append(f'<text x="{80+700*x/0.105}" y="365">{x:.3f}</text>')
    parts.append('<text x="310" y="393">log2 |H(G)| - log2 |H(G,B)| (bits)</text>')
    for i,(k,c) in enumerate(colors.items()):parts.append(f'<text x="{610+i*85}" y="85" fill="{c}">{k}</text>')
    parts.append('</g></svg>')
    with (out/'information_vs_accuracy.svg').open('x') as f:f.write('\n'.join(parts))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output)
