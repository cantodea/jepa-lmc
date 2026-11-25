"""Public archival integrity tests: no private data and no new experiments."""
import csv
import gzip
import hashlib
import itertools
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def read_csv(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class ArchiveIntegrityTests(unittest.TestCase):
    def test_01_original_artifact_bytes_and_manifest(self):
        for name, metadata in read_json('source_files.json').items():
            raw = (ROOT / name).read_bytes()
            if metadata['encoding'] == 'gzip':
                raw = gzip.decompress(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), metadata['original_sha256'], name)
        for name, expected in read_json('SHA256SUMS.json').items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_02_required_totals_and_case_identity(self):
        cases = read_json('cases_72.json')
        self.assertEqual(len(cases), 72)
        self.assertEqual(len({(c['seed'], c['case']) for c in cases}), 72)
        self.assertEqual(sum(c['transitions'] for c in cases), 8928)
        self.assertEqual(sum(c['errors'] for c in cases), 96)
        self.assertEqual(sum(c['initial_bisimulation'] for c in cases), 40)
        self.assertEqual(sum(c['identity_bisimulation'] for c in cases), 33)
        self.assertEqual(sum(c['ctl_all_six'] and not c['initial_bisimulation'] for c in cases), 28)
        detailed = read_csv('cases_72_detailed.csv')
        self.assertEqual({(int(c['seed']), c['case']) for c in detailed}, {(c['seed'], c['case']) for c in cases})
        self.assertTrue(all(c['initial_bisimulation'] == c['action_sensitive_initial'] for c in cases))

    def test_03_edge_partition_and_case_counts(self):
        cases = read_json('cases_72.json')
        errors = read_json('errors_96.json')
        self.assertEqual(len(errors), 96)
        self.assertEqual(len({e['edge_id'] for e in errors}), 96)
        self.assertEqual(sum(e['can_remain_in_any_successful_subset'] for e in errors), 28)
        self.assertEqual(sum(not e['only_this_error_retained_passes'] for e in errors), 68)
        self.assertEqual(sum(e['reachable_real'] for e in errors), 78)
        self.assertEqual(sum(e['successor_label_mismatch'] for e in errors), 10)
        for c in cases:
            group = [e for e in errors if (e['seed'], e['case']) == (c['seed'], c['case'])]
            self.assertEqual(len(group), c['errors'])
            self.assertTrue(all(e['initial_bisimulation'] == c['initial_bisimulation'] for e in group))
        self.assertEqual({e['edge_id'] for e in errors}, {e['edge_id'] for e in read_csv('errors_96_detailed.csv')})

    def test_04_exhaustive_subsets_and_exact_minima(self):
        cases = read_json('cases_72.json')
        errors = read_json('errors_96.json')
        trials = read_csv('repair_trials_all.csv')
        repairs = {(int(r['seed']), r['case']): r for r in read_csv('repairs_32.csv')}
        self.assertEqual(len(trials), 794)
        self.assertEqual(len(repairs), 32)
        failing_subsets = 0
        histogram = {}
        for c in cases:
            if not c['errors']:
                continue
            key = c['seed'], c['case']
            edges = {e['edge_id'] for e in errors if (e['seed'], e['case']) == key}
            group = [t for t in trials if (int(t['seed']), t['case']) == key]
            actual = [frozenset(json.loads(t['repaired_edges'])) for t in group]
            expected = {frozenset(s) for n in range(len(edges)+1) for s in itertools.combinations(edges, n)}
            self.assertEqual(set(actual), expected)
            self.assertEqual(len(actual), len(expected))
            for t in group:
                self.assertEqual(int(t['size']), len(json.loads(t['repaired_edges'])))
                self.assertEqual(t['insensitive'], t['sensitive'])
                self.assertEqual(t['insensitive'] == 'True', c['initial_bisimulation'] or int(t['size']) == len(edges))
            if not c['initial_bisimulation']:
                failing_subsets += len(group)
                minimum = min(int(t['size']) for t in group if t['insensitive'] == 'True')
                self.assertEqual(minimum, c['errors'])
                self.assertEqual(minimum, int(repairs[key]['minimum_repairs']))
                self.assertEqual([set(s) for s in json.loads(repairs[key]['all_minimum_sets'])], [edges])
                histogram[minimum] = histogram.get(minimum, 0) + 1
        self.assertEqual(failing_subsets, 222)
        self.assertEqual(histogram, {1: 13, 2: 9, 3: 6, 4: 1, 5: 3})

    def test_05_certificate_partition_and_internal_dags(self):
        paths = sorted((ROOT / 'certificates').glob('*.json.gz'))
        self.assertEqual(len(paths), 72)
        dag_count = 0
        for path in paths:
            cert = json.loads(gzip.decompress(path.read_bytes()))
            states = range(len(cert['states']))
            universe = set(itertools.product(states, states))
            self.assertEqual(set(cert['semantics']), {'action_sensitive', 'action_insensitive'})
            for rel in cert['semantics'].values():
                pairs = set(map(tuple, rel['pairs']))
                eliminated = {(e['left'], e['right']): e for e in rel['eliminations']}
                self.assertFalse(pairs & eliminated.keys())
                self.assertEqual(pairs | eliminated.keys(), universe)
                dag = rel['initial_failure_dag']
                if dag is None:
                    self.assertIn((cert['initial'], cert['initial']), pairs)
                    continue
                dag_count += 1
                self.assertEqual(dag['roots'], [[cert['initial'], cert['initial']]])
                nodes = {(n['left'], n['right']): n for n in dag['nodes']}
                for key, n in nodes.items():
                    self.assertEqual(n['round'], eliminated[key]['round'])
                    if n['side'] == 'label':
                        self.assertNotEqual(cert['labels'][n['left']], cert['labels'][n['right']])
                        self.assertEqual(n['round'], 0)
                    else:
                        for response in n['replies']:
                            self.assertLess(nodes[tuple(response['child'])]['round'], n['round'])
        self.assertEqual(dag_count, 64)

    def test_06_property_comparisons(self):
        rows = read_csv('initial_properties_864.csv')
        self.assertEqual(len(rows), 864)
        self.assertEqual(len({(r['seed'], r['case'], r['logic'], r['property']) for r in rows}), 864)
        for row in rows:
            self.assertEqual(row['agreement'] == 'True', row['real'] == row['pred'])
        for logic, count in [('CTL', 68), ('LTL', 70)]:
            catalogue = {(r['seed'], r['case']) for r in rows if r['logic'] == logic}
            self.assertEqual(sum(all(r['agreement'] == 'True' for r in rows if r['logic'] == logic and (r['seed'], r['case']) == k) for k in catalogue), count)

    def test_07_input_provenance(self):
        p = read_json('provenance.json')
        self.assertEqual(p['source_revision'], '539410dec1d5c4e4433bd9a57b10a309cd9a850d')
        self.assertEqual(p['input_sha256']['outputs/top1_quality/round2/behavioral/best/relations.jsonl'], '8d51e45e76bf1f5a0c70732fdcfa90f72b08780c0558e6d8bea39793b2eb5d98')
        self.assertEqual(p['input_sha256'], read_json('original_provenance.json')['inputs_sha256'])
        self.assertFalse(p['data_availability']['original_graph_committed'])

    def test_08_no_private_or_large_payload(self):
        for name in read_json('SHA256SUMS.json'):
            p = ROOT / name
            self.assertNotEqual(p.name, 'relations.jsonl')
            self.assertNotIn(p.suffix.lower(), {'.pt', '.pth', '.zip', '.log', '.pyc'})
            self.assertNotIn('input', p.relative_to(ROOT).parts)
            self.assertLess(p.stat().st_size, 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
