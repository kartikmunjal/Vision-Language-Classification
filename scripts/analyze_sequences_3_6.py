#!/usr/bin/env python3
"""Analyze frozen embeddings for Sequences 3–6 and emit auditable JSON."""

from __future__ import annotations

import argparse, hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEEDS = [11, 22, 33, 44, 55]
TASKS = ["human_present", "animal_present", "multiple_subjects"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def ci_trials(x, seed=20260903):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    z=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return [float(v) for v in np.quantile(z,[.025,.975])]


def ece(y,p,bins=10):
    cuts=np.linspace(0,1,bins+1); total=0
    for a,b in zip(cuts[:-1],cuts[1:]):
        m=(p>=a)&(p<(b if b<1 else b+1e-9))
        if m.any(): total += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(total)


def retrieval_metrics(scores):
    ranks=np.argsort(np.argsort(-scores,axis=1),axis=1)[np.arange(len(scores)),np.arange(len(scores))]+1
    return {"recall_at_1":float(np.mean(ranks<=1)),"recall_at_5":float(np.mean(ranks<=5)),"median_rank":float(np.median(ranks))}


def sequence3(z):
    t=z["temporal_image"]; txt=z["clean_text"][z["temporal_idx"]]
    names=z["trajectories"].astype(str); out={}
    for k,name in enumerate(names):
        frames=[]
        for s in range(8):
            scores=t[:,k,s]@txt.T
            frames.append(retrieval_metrics(scores))
        drift=1-np.sum(t[:,k,0]*t[:,k,-1],axis=1)
        out[name]={"frames":frames,"rank1_retention_frame0_minus_frame7":float(frames[0]["recall_at_1"]-frames[-1]["recall_at_1"]),
                   "mean_embedding_drift":float(drift.mean()),"n_sequences":len(t),"n_trials":1}
    return out


def load_labels(root):
    manifest={r["example_id"]:r for r in read_jsonl(root/"manifest.jsonl")}
    ens={r["example_id"]:r for r in read_jsonl(root/"ensemble.jsonl")}
    silver={r["example_id"]:r for r in read_jsonl(root/"coco_silver.jsonl")}
    return manifest,ens,silver


def fit_predict(xtr,ytr,xte,seed):
    m=LogisticRegression(C=1,max_iter=2000,random_state=seed,class_weight="balanced").fit(xtr,ytr)
    return m.predict_proba(xte)[:,1]


def sequence4(z,root):
    manifest,ens,silver=load_labels(root)
    ids=z["ids"].astype(str); splits=z["splits"].astype(str)
    train=np.where(splits=="train")[0]; test=np.where(splits=="test")[0]
    train=np.array([i for i in train if ids[i] in ens and ids[i] in silver]); test=np.array([i for i in test if ids[i] in silver])
    blur=np.array([manifest[i]["blur_score"] for i in ids]); cut=float(np.quantile(blur[train],.25)); low=test[blur[test]<=cut]
    out={"quality_slice_definition":{"blur_train_quartile_cut":cut,"n_test":len(low)},"budgets":{}}
    for budget in [.1,.2]:
        bout={}
        for task in TASKS:
            weak=np.array([ens[i]["labels"][task]["label"] for i in ids],int)
            truth=np.array([silver[i]["labels"][task] if i in silver else 0 for i in ids],int)
            entropy=np.array([ens[i]["labels"][task].get("vote_entropy",0) if i in ens else 0 for i in ids])
            n=max(1,int(round(len(train)*budget))); targeted=train[np.argsort(-entropy[train])[:n]]
            trials=[]
            for seed in SEEDS:
                random=np.random.default_rng(seed).choice(train,n,replace=False)
                arm={}
                for name,chosen in [("targeted",targeted),("random",random)]:
                    y=weak.copy(); y[chosen]=truth[chosen]
                    p=fit_predict(z["clean_image"][train],y[train],z["clean_image"][test],seed)
                    arm[name]={"accuracy":float(accuracy_score(truth[test],p>=.5)),"ece":ece(truth[test],p),
                               "worst_slice_accuracy":float(accuracy_score(truth[low],p[np.isin(test,low)]>=.5)) if len(low) else None}
                trials.append({"seed":seed,**arm})
            metrics={}
            for metric in ["accuracy","ece","worst_slice_accuracy"]:
                d=[x["targeted"][metric]-x["random"][metric] for x in trials]
                metrics[metric]={"estimate":float(np.mean(d)),"ci95":ci_trials(d),"trial_deltas":d,"n_trials":5}
            bout[task]={"n_corrected":n,"metrics":metrics,"trials":trials}
        out["budgets"][str(budget)]=bout
    return out


def train_retrieval_weights(img,txt,indices,mode,seed):
    rng=np.random.default_rng(seed); sim=img[indices]@txt[indices].T
    if mode=="hard":
        np.fill_diagonal(sim,-np.inf); neg_local=np.argmax(sim,axis=1)
    else:
        neg_local=rng.integers(0,len(indices)-1,len(indices)); neg_local += neg_local>=np.arange(len(indices))
    pos=img[indices]*txt[indices]; neg=img[indices]*txt[indices[neg_local]]
    x=np.vstack([pos,neg]); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    m=LogisticRegression(C=.1,max_iter=2000,random_state=seed).fit(x,y)
    return m.coef_[0]


def sequence5(z):
    split=z["splits"].astype(str); train=np.where(split=="train")[0]; test=np.where(split=="test")[0]
    trials=[]
    for seed in SEEDS:
        row={"seed":seed}
        for mode in ["random","hard"]:
            w=train_retrieval_weights(z["clean_image"],z["clean_text"],train,mode,seed)
            score=(z["clean_image"][test]*w)@z["clean_text"][test].T
            row[mode]={"text_to_image":retrieval_metrics(score.T),"image_to_text":retrieval_metrics(score)}
        trials.append(row)
    out={"n_train":len(train),"n_test":len(test),"n_trials":5,"trials":trials,"paired_deltas_hard_minus_random":{}}
    for direction in ["text_to_image","image_to_text"]:
        out["paired_deltas_hard_minus_random"][direction]={}
        for metric in ["recall_at_1","recall_at_5","median_rank"]:
            d=[r["hard"][direction][metric]-r["random"][direction][metric] for r in trials]
            out["paired_deltas_hard_minus_random"][direction][metric]={"estimate":float(np.mean(d)),"ci95":ci_trials(d),"trial_deltas":d}
    return out


def sequence6(z,root):
    manifest,ens,silver=load_labels(root); ids=z["ids"].astype(str); t=z["temporal_image"]; idx=z["temporal_idx"]
    names=z["trajectories"].astype(str); clean_txt=z["clean_text"][idx]
    retrieval={}
    for k,name in enumerate(names):
        retrieval[name]={"frame7":retrieval_metrics(t[:,k,-1]@clean_txt.T),"flip_rate_from_frame0":float(np.mean(np.argmax(t[:,k,0]@clean_txt.T,1)!=np.argmax(t[:,k,-1]@clean_txt.T,1)))}
    base_img=t[:,0,0]
    original=np.sum(base_img*clean_txt,1); neg=np.sum(base_img*z["negative_text"],1); count=np.sum(base_img*z["count_text"],1)
    captions={"negation_preferred_over_original_rate":float(np.mean(neg>=original)),"count_swap_preferred_over_original_rate":float(np.mean(count>=original)),"n":len(idx)}
    classifier={}
    split=z["splits"].astype(str); train=np.where(split=="train")[0]
    for task in TASKS:
        tr=np.array([i for i in train if ids[i] in silver]); ytr=np.array([silver[ids[i]]["labels"][task] for i in tr])
        y=np.array([silver[ids[i]]["labels"][task] for i in idx]); model=LogisticRegression(C=1,max_iter=2000,class_weight="balanced",random_state=11).fit(z["clean_image"][tr],ytr)
        cleanp=model.predict_proba(base_img)[:,1]; taskout={"clean_accuracy":float(accuracy_score(y,cleanp>=.5)),"corruptions":{}}
        for k,name in enumerate(names[1:],1):
            p=model.predict_proba(t[:,k,-1])[:,1]
            taskout["corruptions"][name]={"accuracy":float(accuracy_score(y,p>=.5)),"accuracy_delta_from_clean":float(accuracy_score(y,p>=.5)-accuracy_score(y,cleanp>=.5)),"prediction_flip_rate":float(np.mean((p>=.5)!=(cleanp>=.5))),"mean_confidence_shift":float(np.mean(np.abs(p-.5)-np.abs(cleanp-.5)))}
        classifier[task]=taskout
    return {"retrieval":retrieval,"caption_attacks":captions,"classifier":classifier,"n_trials":1,"scope":"controlled corruption audit, not safety certification"}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--embeddings",type=Path,required=True); ap.add_argument("--processed-dir",type=Path,required=True); ap.add_argument("--output-dir",type=Path,default=Path("results/research_sequence")); a=ap.parse_args()
    z=np.load(a.embeddings); a.output_dir.mkdir(parents=True,exist_ok=True)
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit=None
    common={"schema_version":1,"plan":"RESEARCH_SEQUENCE_PLAN.md","git_commit":commit,"embedding_sha256":sha(a.embeddings),
            "embedding_metadata":json.loads(str(z["metadata"])),"python":platform.python_version(),"bootstrap_replicates":10000,
            "seeds":SEEDS}
    for number,fn in [(3,lambda:sequence3(z)),(4,lambda:sequence4(z,a.processed_dir)),(5,lambda:sequence5(z)),(6,lambda:sequence6(z,a.processed_dir))]:
        payload={**common,"sequence":number,"result":fn()}; (a.output_dir/f"sequence{number}_results.json").write_text(json.dumps(payload,indent=2)+"\n")

if __name__=="__main__": main()
