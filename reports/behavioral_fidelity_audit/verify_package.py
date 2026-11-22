"""Read-only integrity and accounting checks. Run from any directory."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def verify(root, repo):
    for path, expected in json.loads((root/"SHA256SUMS.json").read_bytes()).items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
    if repo:
        for path, expected in json.loads((root/"pinned_sources.json").read_bytes()).items():
            raw=(repo/path).read_bytes()
            git_hash=lambda b:hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
            assert expected in (git_hash(raw),git_hash(raw.replace(b"\r\n",b"\n"))),path
    def rows(name):
        with (root/(name+".csv")).open(encoding="utf-8",newline="") as f:
            return list(csv.DictReader(f))
    edge=rows("transition_analysis")
    assert len(edge)==20832
    keys=("cohort","case","seed","budget","kind","arm")
    assert len({tuple(r[k] for k in (*keys,"source","action")) for r in edge})==len(edge)
    for r in edge:
        assert r["E2"]==(r["true"] if r["seen"]=="1" else r["raw"])
        assert int(r["correct"])==int(r["true"]==r["raw"])
        if r["gap"]: assert float(r["gap"])>=0
    rec=rows("replay_checks")
    assert len(rec)==144 and sum(int(r["queries"]) for r in rec)==17856
    assert all(r["mismatches"]=="0" for r in rec)
    beh=rows("behavioral_analysis")
    props=rows("initial_verdicts")
    assert len(beh)==432 and len(props)==len(beh)*12
    assert len(rows("information"))==36
    assert len(rows("witness_counts"))==36 and len(rows("witnesses"))==0
    assert len(rows("offline_ambiguities"))==156
    with gzip.open(root/"nuxmv_records.jsonl.gz","rt",encoding="utf-8") as f:
        logs=[json.loads(line) for line in f]
    assert len(logs)==153
    for r in logs:
        assert hashlib.sha256(r["smv"].encode()).hexdigest()==r["sha256"]
    print("OK: manifest, 20,832 unique edge rows, E2 identities, 17,856 neural queries, "
          "432 models/variants, 5,184 property rows, 36 version spaces and 153 backend inputs.")


if __name__=="__main__":
    if not __debug__: raise RuntimeError("Do not disable assertions")
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args()
    verify(a.root,a.repo)
