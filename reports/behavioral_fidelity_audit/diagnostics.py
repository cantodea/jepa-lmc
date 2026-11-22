"""Produce the compact report counts from published rows."""
import json
import sys
from pathlib import Path

import pandas as pd


def diagnostics(out):
    def read(name):
        return pd.read_csv(out/(name+".csv"))
    e=read("transition_analysis")
    b=read("baseline_comparison").merge(read("behavioral_analysis"),on=["cohort","case","seed","budget","kind","arm","variant"])
    info=read("information")
    c=info[info.arm=="C"]
    result={"info":{
        "bits_range":[c.bits.min(),c.bits.max()],"bits_mean":c.bits.mean(),"positive_settings":int((c.bits>0).sum()),
        "unknown_range":[int(c.unknown.min()),int(c.unknown.max())],"unseen_singletons":int(c.unseen_singletons.sum()),
        "reduced":int(c.reduced.sum()),"legacy_positive":int((info[info.arm=="B"].bits>0).sum())}}
    r=e[e.cohort=="R"]
    err=r[r.correct==0]
    result["rank"]={"true_ranks_wrong":err.true_rank.value_counts().sort_index().to_dict(),
        "position_only":int(r.position_only_correct.sum()),"learned_only":int(r.learned_only_correct.sum()),
        "direct64_disagree":int(r.direct64_disagree.sum()),
        "wrong_position_only_right":int(err.position_only_correct.sum()),"wrong_learned_only_right":int(err.learned_only_correct.sum())}
    for label,cohort,kind,arm in (("rank","R","jepa","ranking"),("finite","F","jepa","A")):
        g=b[(b.cohort==cohort)&(b.kind==kind)&(b.arm==arm)&(b.variant=="E1")]
        result.setdefault(label,{})["gap_by_ai_init"]={str(k):{
            "cases":len(df),"median_of_case_medians":df.unseen_median_gap.median(),
            "min":df.unseen_median_gap.min(),"max":df.unseen_median_gap.max()} for k,df in g.groupby("ai_initial_bisimulation")}
        result[label]["ties"]=int(g.gap_tie_zero.sum())
        subset=e[(e.cohort==cohort)&(e.kind==kind)&(e.arm==arm)&(e.seen==0)]
        result[label]["top2_unseen_moving"]=int((subset[subset.true_move==1].true_rank<=2).sum())
        result[label]["top2_unseen"]=int((subset.true_rank<=2).sum())
    f=e[(e.cohort=="F")&(e.kind=="jepa")&(e.seen==0)]
    a=f[f.arm=="A"].merge(f[f.arm=="C"],on=["case","seed","budget","source","action"],suffixes=("_A","_C"))
    result["finite"].update(changed_C=int((a.raw_A!=a.raw_C).sum()),
        corrections=int(((a.correct_A==0)&(a.correct_C==1)).sum()),
        regressions=int(((a.correct_A==1)&(a.correct_C==0)).sum()),
        infeasible_C=int((f[f.arm=="C"].raw_feasible==0).sum()))
    q=read("witness_counts")
    off=read("offline_ambiguities")
    p=read("initial_verdicts")
    result["witness"]={"candidate_pairs":int(q.certified_ambiguous.sum()),"offline_by_arm":off.groupby("arm").size().to_dict(),
        "offline_C_by_property":off[off.arm=="C"].groupby(["logic","property"]).size().reset_index(name="n").to_dict("records"),
        "feasible_graphs_range":[int(q.feasible_examined.min()),int(q.feasible_examined.max())]}
    for key,cohort,arm in (("rank","R","ranking"),("finite","F","A")):
        sub=p[(p.cohort==cohort)&(p.kind=="jepa")&(p.arm==arm)&(p.variant=="E1")]
        result[key]["mismatch"]=sub[sub.real!=sub.learned].groupby(["logic","property","real","learned"]).size().reset_index(name="n").to_dict("records")
    return result


if __name__=="__main__":
    out=Path(sys.argv[1])
    result=diagnostics(out)
    s=json.dumps(result,indent=2)
    (out/"diagnostics.json").write_text(s+"\n",encoding="utf-8")
    print(s)
