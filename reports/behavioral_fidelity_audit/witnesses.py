"""Bound incomplete identifiability by exhibiting feasible deterministic completions."""
import argparse
import csv
import itertools
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from audit_core import observations
from run_audit import save_csv, label_catalogue
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.trace_sanity import Trace
from jepa_lmc.learning.passive_negative import analyze_version_space


def traces(rows):
    return tuple(Trace(tuple(map(tuple, r["states"])),tuple(r["actions"])) for r in rows)


def feasible(pred, goods, bads):
    def accepts(t):
        return all(pred[s,a]==v for s,a,v in t.edges)
    return all(accepts(t) for t in goods) and not any(accepts(t) for t in bads)


def main(run, out):
    z = zipfile.ZipFile(run/"raw_records.zip")
    datasets = json.loads(z.read("datasets.json"))
    frozen = {r["id"]:r for r in json.loads(z.read("information.json"))}
    cases = {c.name:label_catalogue(c.make_env()) for c in radius_stress_cases()}
    props = defaultdict(dict)
    real_verdicts = defaultdict(list)
    tables = defaultdict(dict)
    def key(r):
        return r["case"], int(r["seed"]), int(r["budget"])
    def name(r, variant):
        return "/".join((r["kind"],r["arm"],variant))
    with (out/"initial_verdicts.csv").open() as f:
        for r in csv.DictReader(f):
            if r["cohort"]=="F":
                props[key(r),name(r,r["variant"])][r["logic"],r["property"]] = bool(int(r["learned"]))
                if (r["kind"],r["arm"],r["variant"])==("symbolic","A","E1"):
                    real_verdicts[key(r)].append(r)
    with (out/"transition_analysis.csv").open() as f:
        for r in csv.DictReader(f):
            if r["cohort"]!="F": continue
            states = cases[r["case"]].states
            s, a = states[int(r["source"])], int(r["action"])
            variants = {"E1":states[int(r["raw"])],"E2":states[int(r["E2"])]}
            if r["kind"]=="symbolic" and r["arm"]=="A":
                variants.update(loop=s)
                variants["loop+obs"] = states[int(r["E2"])] if r["seen"]=="1" else s
            for variant,t in variants.items():
                tables[key(r),name(r,variant)][s,a] = t
    witnesses, counts, offline = [], [], []
    for d in datasets:
        k = d["case"],d["seed"],d["budget"]
        tables_for_k = {model:tab for (setting,model),tab in tables.items() if setting==k}
        distinct = {}
        for model in sorted(tables_for_k):
            edges = tuple(sorted(tables_for_k[model].items()))
            if edges in distinct:
                assert props[k,model] == props[k,distinct[edges]]
            else:
                distinct[edges] = model
        gs = traces(d["goods"])
        for arm, neg in (("A",()),("B",traces(d["legacy_bads"])),("C",traces(d["bads"]))):
            f = analyze_version_space(cases[k[0]],gs,neg,max_nodes=250000)
            old = frozen[d["id"]][arm]
            for field in ("status","model_count_good_bad","model_count_good","candidate_successors",
                          "negative_diagnostics","completion_edges","information_bits"):
                assert json.loads(json.dumps(f[field]))==old[field],(d["id"],arm,field)
            ok = sorted(m for m in distinct.values() if feasible(tables_for_k[m],gs,neg))
            assert ok
            # Offline proof of non-identifiability, separate from the candidate-only scan.
            # The data-generating system is itself one admissible completion.
            env = next(c for c in radius_stress_cases() if c.name==k[0]).make_env()
            real = {key:env.transition(*key) for key in tables_for_k[ok[0]]}
            assert feasible(real,gs,neg)
            reference = {}
            for row in real_verdicts[k]:
                reference[row["logic"],row["property"]]=bool(int(row["real"]))
            for (logic,prop),value in reference.items():
                other = next((m for m in ok if props[k,m][logic,prop]!=value),None)
                if other:
                    offline.append(dict(id=d["id"],arm=arm,logic=logic,property=prop,
                        feasible_candidate=other,real_verdict=int(value),candidate_verdict=int(not value)))
            total = 0
            for (logic,prop) in sorted(props[k,ok[0]]):
                pair = next(((a,b) for a,b in itertools.combinations(ok,2)
                             if props[k,a][logic,prop] != props[k,b][logic,prop]), None)
                if pair:
                    a,b = pair
                    witnesses.append(dict(id=d["id"],arm=arm,logic=logic,property=prop,
                                          model_true=a if props[k,a][logic,prop] else b,
                                          model_false=b if props[k,a][logic,prop] else a))
                    total += 1
            counts.append(dict(id=d["id"],arm=arm,distinct_graphs=len(distinct),
                               feasible_examined=len(ok),certified_ambiguous=total,formulas=12))
    with (out/"witnesses.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("id","arm","logic","property","model_true","model_false"),lineterminator="\n")
        writer.writeheader()
        writer.writerows(witnesses)
    save_csv(out/"offline_ambiguities.csv",offline)
    save_csv(out/"witness_counts.csv",counts)
    print("Verified 36 exact counts, marginal supports, negative diagnostics and symbolic completions.")
    print("Opposite-verdict pairs:",len(witnesses))
    print("Offline disagreements with feasible real system:",len(offline))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("run",type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    main(args.run,args.out)
