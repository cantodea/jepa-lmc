"""Read measured raw CSVs; aggregate repeated queries descriptively."""
import argparse
import itertools
import statistics
from pathlib import Path

import pandas as pd


def summarize(out):
    def read(name):
        return pd.read_csv(out/(name+".csv"),keep_default_na=False)
    labels = ["cohort","case","seed","budget","kind","arm","variant"]
    c = read("baseline_comparison")
    b = read("behavioral_analysis")
    joined = c.merge(b,on=labels,validate="one_to_one")
    keys = ["cohort","kind","arm","variant"]
    counts = [k for k in c if k.endswith(("_ok","_n"))] + ["pred_loops","missed_moves","n_states","action_vary_states","changed_from_E1"]
    counts += [k for k in b if k.startswith(("as_","ai_")) or k.endswith("_all6")]
    counts += ["false_safe","unsafe_opportunities","learned_true_real_false"]
    summaries=[]
    for level in (keys, keys+["budget"],keys+["case"]):
        for k,g in joined.groupby(level,sort=True):
            if "budget" in level and g.iloc[0]["cohort"]=="R":continue
            sums={v:int(g[v].sum()) for v in counts}
            # Distinguish model-specific medians from edge pooling.
            for field in ("auc_error","moving_auc_error","unseen_median_gap"):
                vals=pd.to_numeric(g[field],errors="coerce").dropna()
                sums[field+"_n"]=len(vals)
                sums[field+"_median"]=vals.median() if len(vals) else None
            summaries.append({**dict(zip(level,k)), "grouping":level[-1],"cases":len(g),**sums})
    pd.DataFrame(summaries).to_csv(out/"grouped_results.csv",index=False)
    e=read("transition_analysis")
    rows=[]
    for k,g in e.groupby(["cohort","kind","arm","action","seen"]):
        rows.append(dict(zip(["cohort","kind","arm","action","seen"],k),
                         n=len(g),correct=int(g.correct.sum()),moves=int(g.true_move.sum()),
                         correct_moves=int((g.correct*g.true_move).sum()),
                         missed_moves=int((g.true_move*(1-g.pred_move)).sum())))
    pd.DataFrame(rows).to_csv(out/"action_results.csv",index=False)
    q=read("uncertainty_quartiles")
    q.groupby(["cohort","kind","arm","quartile"])[["n","errors","moving","moving_errors"]].sum().to_csv(out/"quartile_summary.csv")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("out",type=Path)
    summarize(ap.parse_args().out)
