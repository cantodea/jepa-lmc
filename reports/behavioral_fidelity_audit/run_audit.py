"""Replay the two fixed cohorts, then score them. Run without python -O."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import platform
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import torch

from audit_core import error_auc, load_weights, observations, project, scores, sha
from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.ltl_suite import default_ltl_suite
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.finite_trace import prediction_system
from jepa_lmc.evaluation.structural import evaluate_model_pair
from jepa_lmc.evaluation.trace_sanity import KnownProblem
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.nuxmv import LTLQuery, export_nusmv_ltl_queries, run_nusmv_model

METRICS = ("simulation_real_to_learned", "simulation_learned_to_real", "initial_bisimulation",
           "identity_bisimulation", "strict_identity_bisimulation")
SAFETY = {p.name for p in (*default_ltl_suite(), *default_ctl_suite()) if p.safety_claim}


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def save_csv(path, rows):
    fields = dict.fromkeys(k for row in rows for k in row)
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_edges(records):
    return {(tuple(s), a): tuple(t) for s, a, t in records}


def label_catalogue(env):
    states = tuple(sorted(env.all_states()))
    labels = {s: frozenset(({"danger"} if env.is_danger(s) else {"safe"}) |
                           ({"goal"} if env.is_goal(s) else set())) for s in states}
    return KnownProblem(states, frozenset([env.start]), tuple(sorted(env.ACTIONS)), labels)


class Audit:
    def __init__(self, out, backend):
        self.out, self.backend = out, backend
        self.edges, self.behavior, self.comparison, self.properties = [], [], [], []
        self.replays, self.info, self.bins = [], [], []
        self.cases = {c.name: c.make_env() for c in radius_stress_cases()}
        self.formal_cache, self.logic_cache = {}, {}
        self.logs = []
        self.inputs = {}
        self.real = {}
        self.problems = {k: label_catalogue(env) for k, env in self.cases.items()}

    def ltl(self, system):
        queries = [LTLQuery(next(iter(system.initial_states)), p.formula, p.name)
                   for p in default_ltl_suite()]
        smv = export_nusmv_ltl_queries(system, queries)
        key = sha(smv.encode())
        if key not in self.logic_cache:
            run = run_nusmv_model(smv, self.backend)
            assert len(run.verdicts) == len(queries)
            self.logic_cache[key] = list(run.verdicts)
            self.logs.append(dict(sha256=key, smv=smv, stdout=run.output))
        return self.logic_cache[key]

    def formal(self, case, pred):
        key = case, tuple(sorted(pred.items()))
        if key not in self.formal_cache:
            problem = self.problems[case]
            if case not in self.real:
                self.real[case] = gridworld_to_transition_system(self.cases[case])
            real = self.real[case]
            learned = prediction_system(problem, pred)
            result = evaluate_model_pair(real, learned)
            behavioral = {prefix + "_" + m: result["behavioral"][sem][m]
                          for prefix, sem in (("ai", "action_insensitive"), ("as", "action_sensitive"))
                          for m in METRICS}
            props = [("CTL", r["property"], r["real"], r["learned"])
                     for r in result["diagnostics"]["initial_ctl_verdicts"]]
            props.extend(("LTL", p.name, a, b) for p, a, b in zip(
                default_ltl_suite(), self.ltl(real), self.ltl(learned), strict=True))
            for logic in ("CTL", "LTL"):
                selected = [(r, l) for g, _, r, l in props if g == logic]
                behavioral[logic + "_all6"] = all(r == l for r, l in selected)
            behavioral["learned_true_real_false"] = sum(l and not r for _, _, r, l in props)
            behavioral["false_safe"] = sum(p in SAFETY and l and not r for _, p, r, l in props)
            behavioral["unsafe_opportunities"] = sum(p in SAFETY and not r for _, p, r, _ in props)
            self.formal_cache[key] = behavioral, props
        return self.formal_cache[key]

    def record(self, ident, raw, obs, score_data=None, support=None, archived=None,
               rank_rows=None, certificate=None):
        case = ident["case"]
        problem = self.problems[case]
        states, actions = problem.states, problem.actions
        keys = [(s, a) for s in states for a in actions]
        assert set(raw) == set(keys)
        corrected = project(raw, obs)
        loop = {(s, a): s for s, a in keys}
        # Everything upstream of this line uses only catalogue, observations and model outputs.
        truth = {k: self.cases[case].transition(*k) for k in keys}
        assert all(truth[k] == t for k, t in obs.items())
        sid = {s: i for i, s in enumerate(states)}
        uncertainty = []
        if score_data is not None:
            values = score_data
            order = values.argsort(dim=1, descending=True, stable=True)
            replay = {k: states[int(values[i].argmax())] for i, k in enumerate(keys)}
            mismatch = sum(replay[k] != raw[k] for k in keys)
            self.replays.append({**ident, "queries": len(keys), "mismatches": mismatch})
            if mismatch:
                score_data = None  # retain and report any failed parity, omit incompatible scores
        unseen = [k for k in keys if k not in obs]
        move = [k for k in unseen if truth[k] != k[0]]
        for i, (s, a) in enumerate(keys):
            k = s, a
            rec = dict(**ident, source=sid[s], action=a, seen=int(k in obs), true=sid[truth[k]],
                       raw=sid[raw[k]], E2=sid[corrected[k]], true_move=int(truth[k] != s),
                       pred_move=int(raw[k] != s), correct=int(raw[k] == truth[k]),
                       E2_correct=int(corrected[k] == truth[k]))
            if support:
                options = support[k]
                rec.update(support_size=len(options), raw_feasible=int(raw[k] in options))
                assert truth[k] in options
            if rank_rows:
                rr = rank_rows[k]
                assert raw[k] == (int(rr["pred_row"]), int(rr["pred_col"]))
                assert truth[k] == (int(rr["true_row"]), int(rr["true_col"]))
                rec.update(archive_rank=rr["rank"], oracle_signed_gap_sq=rr["squared_true_vs_nearest_wrong_margin"],
                           position_only_correct=rr["position_only_correct"], learned_only_correct=rr["learned_only_correct"],
                           direct64_disagree=rr["float32_vs_direct64_disagree"])
            if score_data is not None:
                best, second = order[i, :2].tolist()
                gap = float(values[i, best] - values[i, second])
                rec.update(raw_score1=float(values[i, best]), runner_up=sid[states[second]],
                           gap=gap, true_rank=1 + int((order[i] == sid[truth[k]]).nonzero().item()))
                if k not in obs:
                    uncertainty.append((gap, int(raw[k] != truth[k]), k, truth[k] != s))
            self.edges.append(rec)
        title = "E1"
        variants = {title: raw, "E2": corrected}
        # Baseline once per setting (not once per neural comparator).
        if ident["cohort"] == "R" or (ident["kind"], ident["arm"]) == ("symbolic", "A"):
            variants["loop"] = loop
            if obs:
                variants["loop+obs"] = project(loop, obs)
        all_gaps = [r[0] for r in uncertainty]
        for variant, pred in variants.items():
            tag = {**ident, "variant": variant}
            formal, props = self.formal(case, pred)
            self.behavior.append({**tag, **formal, "certificate": certificate if variant in ("E1","E2") else None})
            self.properties.extend({**tag, "logic": g, "property": p, "real": int(r),
                                    "learned": int(l), "safety": int(p in SAFETY)} for g, p, r, l in props)
            if variant == "E1" and archived:
                for key, val in archived.items():
                    assert formal[key] == val, (tag, key, val)
            groups = {"all": keys, "seen": list(obs), "unseen": unseen, "unseen_move": move,
                      "unseen_loop": [k for k in unseen if k[0] == truth[k]]}
            row = dict(**tag, changed_from_E1=sum(raw[k] != pred[k] for k in keys))
            for name, members in groups.items():
                row[name + "_n"] = len(members)
                row[name + "_ok"] = sum(pred[k] == truth[k] for k in members)
            row.update(pred_loops=sum(pred[k] == k[0] for k in keys),
                       missed_moves=sum(pred[k] == k[0] for k in move),
                       action_vary_states=sum(len({pred[s, a] for a in actions}) > 1 for s in states),
                       n_states=len(states), shifted_ok=sum(pred[s, actions[(j+1)%len(actions)]] == truth[s,a]
                                                           for s in states for j,a in enumerate(actions)))
            if variant == "E1" and all_gaps:
                row.update(unseen_median_gap=statistics.median(all_gaps), gap_tie_zero=sum(v==0 for v in all_gaps),
                           distinct_gaps=len(set(all_gaps)), auc_error=error_auc([(g,e) for g,e,_,_ in uncertainty]),
                           moving_auc_error=error_auc([(g,e) for g,e,_,m in uncertainty if m]))
                q = sorted(uncertainty, key=lambda r: (r[0], r[2]))
                for quartile in range(4):
                    group = [r for idx,r in enumerate(q) if 4*idx//len(q)==quartile]
                    self.bins.append({**tag, "quartile":quartile+1, "n":len(group),
                        "errors":sum(r[1] for r in group),"moving":sum(r[3] for r in group),
                        "moving_errors":sum(r[1]*r[3] for r in group), "min_gap":group[0][0],"max_gap":group[-1][0]})
            self.comparison.append(row)
        return truth

    def finite(self, run, weights):
        data = (run/"raw_records.zip").read_bytes()
        import hashlib
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == "2f7d5e80216d64642bdce5c0a62e90192942d5ff"
        self.inputs["raw_records.zip"] = sha(data)
        z = zipfile.ZipFile(io.BytesIO(data))
        ds = {d["id"]:d for d in json.loads(z.read("datasets.json"))}
        info = {d["id"]:d for d in json.loads(z.read("information.json"))}
        rows = [json.loads(line) for line in z.read("results.jsonl").splitlines()]
        assert len(rows)==96 and len(ds)==12
        assert len({(r["id"],r["kind"],r["arm"]) for r in rows})==96
        for ident, pack in info.items():
            for arm in ("A","B","C"):
                f = pack[arm]
                assert f["status"] == "exact"
                self.info.append(dict(id=ident, arm=arm, observed=f["known_pairs"], unknown=f["unknown_pairs"],
                    positives_count=f["model_count_good"], feasible_count=f["model_count_good_bad"],
                    bits=f["information_bits"], reduced=f["unknown_pairs_with_reduced_support"],
                    unseen_singletons=sum(not r["observed"] and r["candidate_count"]==1 for r in f["candidate_successors"])))
        for row in rows:
            case, kind, arm = (row[k] for k in ("case","kind","arm"))
            d = ds[row["id"]]
            ident = dict(cohort="F",case=case,seed=row["seed"],budget=row["budget"],kind=kind,arm=arm)
            positive = observations(d["goods"])
            support = {(tuple(x["source"]),x["action"]):set(map(tuple,x["successors"]))
                       for x in info[row["id"]][arm]["candidate_successors"]}
            raw = as_edges(row["predictions"])
            mat = None
            if kind != "symbolic" and weights:
                name = f"{row['id']}_{kind}_{arm}.pt"
                b = (weights/name).read_bytes()
                model, kw = load_weights(b,row["training"]["checkpoint_sha256"],kind)
                self.inputs[name] = sha(b)
                mat = scores(model,kind,self.problems[case],kw["height"],kw["width"])
            ev = row["evaluation"]
            f = ev["formal"]
            old = {p+"_"+m:f["behavioral"][s][m] for p,s in (("ai","action_insensitive"),("as","action_sensitive")) for m in METRICS}
            old.update(CTL_all6=f["diagnostics"]["initial_all_six_ctl_agreement"],
                       LTL_all6=f["diagnostics"]["ltl"]["initial_all_six_agreement"])
            truth = self.record(ident,raw,positive,mat,support,old)
            for key, members in (("observed",positive),("unseen",truth.keys()-positive.keys()),("overall",truth)):
                assert (sum(raw[k]==truth[k] for k in members),len(members)) == tuple(ev["transitions"][key][v] for v in ("correct","total"))
            original = [("CTL",x) for x in f["diagnostics"]["initial_ctl_verdicts"]]
            original += [("LTL",x) for x in f["diagnostics"]["ltl"]["initial_verdicts"]]
            assert {(g,x["property"],x["real"],x["learned"]) for g,x in original}==set(self.formal(case,raw)[1])

    def ranking(self, bundle, archive):
        z = zipfile.ZipFile(bundle)
        base = "outputs/top1_quality/round2/"
        prov = json.loads((archive/"provenance.json").read_bytes())
        for path, expected in prov["input_sha256"].items():
            assert sha(z.read(path)) == expected, path
            self.inputs[path] = expected
        rreport = json.loads(z.read(base+"behavioral/best/report.json"))
        csvs = defaultdict(dict)
        models = {}
        for seed, expected in rreport["checkpoint_sha256"].items():
            prefix = base+"ranking/seed_"+seed+"/"
            weights = z.read(prefix+"model.pt")
            models[int(seed)] = load_weights(weights,expected,"jepa")
            self.inputs[prefix+"model.pt"] = expected
            all_rows = read_csv(z.read(prefix+"evaluation/transitions.csv"))
            assert {r["map"] for r in all_rows if r["split"]=="train"}.isdisjoint(
                r["map"] for r in all_rows if r["split"]=="stress")
            for r in all_rows:
                if r["split"] == "stress":
                    k = (int(r["state_row"]),int(r["state_col"])),int(r["action"])
                    csvs[int(seed),r["map"]][k]=r
        history = defaultdict(list)
        for r in read_csv((archive/"initial_properties_864.csv").read_bytes()):
            history[int(r["seed"]),r["case"]].append((r["logic"],r["property"],r["real"]=="True",r["pred"]=="True"))
        graph_lines = z.read(base+"behavioral/best/relations.jsonl").splitlines()
        assert len(graph_lines)==72 and len(csvs)==72
        for line in graph_lines:
            graph = json.loads(line)
            case, seed = graph["case"], graph["seed"]
            states = list(map(tuple,graph["states"]))
            problem = self.problems[case]
            assert tuple(states)==problem.states and states[graph["initial"]] in problem.initial_states
            assert all(set(label)==problem.labels[s] for s,label in zip(states,graph["labels"],strict=True))
            def table(which):
                return {(s,a):states[t] for s,edges in zip(states,graph["graphs"][which],strict=True) for a,t in edges}
            raw=table("top1")
            mat=scores(models[seed][0],"jepa",problem,models[seed][1]["height"],models[seed][1]["width"])
            old={}
            for rel in graph["relations"]:
                prefix="as_" if rel["action_sensitive"] else "ai_"
                m={"real_to_top1":"simulation_real_to_learned","top1_to_real":"simulation_learned_to_real","bisimulation":"initial_bisimulation"}[rel["name"]]
                pairs=set(map(tuple,rel["pairs"]))
                old[prefix+m]=(graph["initial"],graph["initial"]) in pairs
                if rel["name"]=="bisimulation":
                    old[prefix+"identity_bisimulation"]=all((i,i) in pairs for i in range(len(states)))
            ident=dict(cohort="R",case=case,seed=seed,budget=None,kind="jepa",arm="ranking")
            cert=f"certificates/{seed}_{case}.json.gz"
            assert (archive/cert).is_file()
            truth=self.record(ident,raw,{},mat,archived=old,rank_rows=csvs[seed,case],certificate=cert)
            assert truth==table("real") and set(history[seed,case])==set(self.formal(case,raw)[1])

    def finish(self, elapsed):
        for name, rows in (("transition_analysis",self.edges),("behavioral_analysis",self.behavior),
                           ("baseline_comparison",self.comparison),("initial_verdicts",self.properties),
                           ("replay_checks",self.replays),("information",self.info),("uncertainty_quartiles",self.bins)):
            if rows:
                save_csv(self.out/(name+".csv"),rows)
        content="".join(json.dumps(row)+"\n" for row in self.logs).encode()
        (self.out/"nuxmv_records.jsonl.gz").write_bytes(gzip.compress(content,mtime=0))
        stats = dict(seconds=elapsed,python=sys.version,torch=torch.__version__,platform=platform.platform(),
                    threads=1,training_runs=0,formal_graph_pairs=len(self.formal_cache),
                    distinct_nuxmv_inputs=len(self.logic_cache),inputs=self.inputs,
                    nuxmv_sha256=sha(self.backend.read_bytes()), transitions=len(self.edges),
                    score_parity_failures=sum(r["mismatches"]>0 for r in self.replays))
        (self.out/"execution.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
        print({k:v for k,v in stats.items() if k!="inputs"},flush=True)


if __name__=="__main__":
    if not __debug__: raise ValueError("Assertions required")
    ap=argparse.ArgumentParser()
    for arg in ("out","backend","finite","weights","ranking","archive"):
        ap.add_argument("--"+arg,type=Path,required=arg in ("out","backend"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    task=Audit(args.out,args.backend)
    if args.finite: task.finite(args.finite,args.weights)
    if args.ranking: task.ranking(args.ranking,args.archive)
    task.finish(time.perf_counter()-started)
