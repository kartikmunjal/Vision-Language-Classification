#!/usr/bin/env python3
"""Run verified-negative retrieval and multi-objective active acquisition."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from vision_language_classification.research.acquisition import minmax,diversity_scores
from vision_language_classification.research.retrieval import retrieval_metrics
from vision_language_classification.research.statistics import paired_trial_summary
from vision_language_classification.research.provenance import run_provenance

SEEDS=[11,22,33,44,55];TASKS=["human_present","animal_present","multiple_subjects"]
def rows(path):
    with open(path,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def ece(y,p):
    out=0
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        m=(p>=a)&(p<(b if b<1 else b+1e-9))
        if m.any():out+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)
def fit(x,y,seed):return LogisticRegression(C=1,solver="saga",max_iter=3000,class_weight="balanced",random_state=seed).fit(x,y)

def retrieval_followup(z,manifest):
    split=z["splits"].astype(str);tr=np.where(split=="train")[0];te=np.where(split=="test")[0];img=z["clean_image"];txt=z["clean_text"]
    cats=[set(manifest[i].get("coco_object_counts",{})) for i in range(len(manifest))];sim=img[tr]@txt[tr].T
    eligible=[]
    for i,gi in enumerate(tr):
        order=[j for j in np.argsort(-sim[i]) if j!=i and cats[gi].isdisjoint(cats[tr[j]])]
        eligible.append(np.array(order[:max(1,int(np.ceil(.1*len(tr))))],int))
    trials=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);random=rng.integers(0,len(tr)-1,len(tr));random+=random>=np.arange(len(tr))
        semi=np.array([rng.choice(e) for e in eligible]);mixed=random.copy();mask=rng.permutation(len(tr))[:len(tr)//2];mixed[mask]=semi[mask]
        trial={"seed":seed}
        for name,neg in [("random",random),("verified_mixed",mixed)]:
            pos=img[tr]*txt[tr];bad=img[tr]*txt[tr[neg]];x=np.vstack([pos,bad]);y=np.r_[np.ones(len(pos)),np.zeros(len(bad))]
            model=LogisticRegression(C=.1,max_iter=2000,random_state=seed).fit(x,y);w=model.coef_[0]
            score=(img[te]*w)@txt[te].T;trial[name]={"text_to_image":retrieval_metrics(score.T),"image_to_text":retrieval_metrics(score)}
        trials.append(trial)
    delta={}
    for direction in ["text_to_image","image_to_text"]:
        delta[direction]={}
        for metric in ["recall_at_1","recall_at_5","median_rank"]:
            delta[direction][metric]=paired_trial_summary([x["verified_mixed"][direction][metric] for x in trials],[x["random"][direction][metric] for x in trials])
    return {"negative_policy":"50% random + 50% object-disjoint top-decile semi-hard","n_train":len(tr),"n_test":len(te),"trials":trials,"paired_deltas_verified_mixed_minus_random":delta,"n_trials":5}

def acquisition_followup(z,manifest,ensemble,silver):
    ids=z["ids"].astype(str);split=z["splits"].astype(str);tr=np.where(split=="train")[0];te=np.where(split=="test")[0]
    tr=np.array([i for i in tr if ids[i] in ensemble and ids[i] in silver]);te=np.array([i for i in te if ids[i] in silver]);blur=np.array([r["blur_score"] for r in manifest]);cut=float(np.quantile(blur[tr],.25));low=te[blur[te]<=cut]
    all_results={}
    for budget in [.1,.2]:
        task_results={};policy_macro={p:[] for p in ["random","disagreement","composite"]}
        for task in TASKS:
            weak=np.array([ensemble.get(i,{"labels":{task:{"label":0}}})["labels"][task]["label"] for i in ids]);truth=np.array([silver.get(i,{"labels":{task:0}})["labels"][task] for i in ids]);dis=np.array([ensemble.get(i,{"labels":{task:{"vote_entropy":0}}})["labels"][task].get("vote_entropy",0) for i in ids])
            n=max(1,round(len(tr)*budget));div=diversity_scores(z["clean_image"][tr]);trials=[]
            for seed in SEEDS:
                base=fit(z["clean_image"][tr],weak[tr],seed);pbase=base.predict_proba(z["clean_image"][tr])[:,1];unc=1-2*np.abs(pbase-.5);lowq=(blur[tr]<=cut).astype(float)
                score=(minmax(dis[tr])+minmax(unc)+minmax(div)+minmax(lowq))/4
                chosen={"random":np.random.default_rng(seed).choice(tr,n,replace=False),"disagreement":tr[np.argsort(-dis[tr])[:n]],"composite":tr[np.argsort(-score)[:n]]}
                result={"seed":seed}
                for policy,sel in chosen.items():
                    y=weak.copy();y[sel]=truth[sel];m=fit(z["clean_image"][tr],y[tr],seed);p=m.predict_proba(z["clean_image"][te])[:,1]
                    result[policy]={"accuracy":float(accuracy_score(truth[te],p>=.5)),"ece":ece(truth[te],p),"low_blur_accuracy":float(accuracy_score(truth[low],p[np.isin(te,low)]>=.5))}
                trials.append(result)
            comp={metric:paired_trial_summary([x["composite"][metric] for x in trials],[x["disagreement"][metric] for x in trials]) for metric in ["accuracy","ece","low_blur_accuracy"]}
            task_results[task]={"n_corrected":n,"trials":trials,"paired_deltas_composite_minus_disagreement":comp}
            for policy in policy_macro:policy_macro[policy].append([x[policy]["accuracy"] for x in trials])
        macro={p:np.mean(np.array(v),axis=0).tolist() for p,v in policy_macro.items()}
        task_results["macro_accuracy"]={p:{"mean":float(np.mean(v)),"trial_values":v} for p,v in macro.items()}
        task_results["macro_accuracy"]["composite_minus_disagreement"]=paired_trial_summary(macro["composite"],macro["disagreement"])
        all_results[str(budget)]=task_results
    return {"blur_train_quartile_cut":cut,"budgets":all_results,"tasks":TASKS,"n_trials":5}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--embeddings",type=Path,required=True);ap.add_argument("--processed-dir",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args();z=np.load(a.embeddings)
    mr=rows(a.processed_dir/"manifest.jsonl");ens={x["example_id"]:x for x in rows(a.processed_dir/"ensemble.jsonl")};silver={x["example_id"]:x for x in rows(a.processed_dir/"coco_silver.jsonl")}
    common={"schema_version":1,"plan":"FOLLOWUP_RESEARCH_PLAN.md","provenance":run_provenance({"embeddings":a.embeddings,"manifest":a.processed_dir/"manifest.jsonl","ensemble":a.processed_dir/"ensemble.jsonl","silver":a.processed_dir/"coco_silver.jsonl"},args=vars(a))}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/"verified_negative_retrieval.json").write_text(json.dumps({**common,"result":retrieval_followup(z,mr)},indent=2,default=str)+"\n")
    (a.output_dir/"active_acquisition.json").write_text(json.dumps({**common,"result":acquisition_followup(z,mr,ens,silver)},indent=2,default=str)+"\n")
if __name__=="__main__":main()
