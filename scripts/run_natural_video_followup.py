#!/usr/bin/env python3
"""Run registered UCF101 subset robustness and temporal-adapter experiments."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
import open_clip

CLASSES=["Basketball","Biking","Diving","Drumming","HorseRiding","PlayingGuitar","RockClimbingIndoor","Rowing","Skiing","TaiChi"]
CONDITIONS=["clean","reverse","frame_drop","blur","darkening"]
SEEDS=[11,22,33,44,55]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def ci(x):
    x=np.asarray(x,float); rng=np.random.default_rng(20260903); z=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return [float(v) for v in np.quantile(z,[.025,.975])]
def paired_example_delta(a,b):
    d=np.asarray(a,float)-np.asarray(b,float)
    return {"estimate":float(d.mean()),"ci95":ci(d),"n":len(d),"n_trials":1}

def manifest(split_root,video_root):
    rows=[]
    for split,file,cap in [("train","trainlist01.txt",70),("test","testlist01.txt",30)]:
        paths=[]
        for line in (split_root/file).read_text().splitlines():
            rel=line.split()[0]; cls=rel.split('/')[0]
            if cls in CLASSES: paths.append((cls,rel))
        by=defaultdict(list)
        for cls,rel in paths: by[cls].append(rel)
        for cls in CLASSES:
            for rel in sorted(by[cls])[:cap]: rows.append({"split":split,"class":cls,"label":CLASSES.index(cls),"relative_path":rel,"path":str(video_root/rel)})
    return rows

def decode(path,n=16):
    cap=cv2.VideoCapture(str(path)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames=[]
    if total<=0:return None
    for pos in np.linspace(0,max(total-1,0),n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(pos)); ok,frame=cap.read()
        if not ok:cap.release();return None
        frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    cap.release();return frames

def corrupt(frames,name):
    if name=="clean":return frames
    if name=="reverse":return list(reversed(frames))
    if name=="frame_drop":return [frames[i if i%2==0 else i-1] for i in range(len(frames))]
    if name=="blur":return [x.filter(ImageFilter.GaussianBlur(4)) for x in frames]
    if name=="darkening":return [ImageEnhance.Brightness(x).enhance(.25) for x in frames]
    raise ValueError(name)

def encode_images(model,preprocess,images,device):
    out=[]
    for i in range(0,len(images),128):
        b=torch.stack([preprocess(x) for x in images[i:i+128]]).to(device)
        with torch.inference_mode(),torch.autocast(device_type="cuda",enabled=device.startswith("cuda")):
            z=model.encode_image(b);z=z/z.norm(dim=-1,keepdim=True)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)

def extract(rows,cache,device):
    model,_,pre=open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k");model=model.eval().to(device)
    tok=open_clip.get_tokenizer("ViT-B-32"); texts=tok(["a video of "+c for c in CLASSES]).to(device)
    with torch.inference_mode(): p=model.encode_text(texts);p=(p/p.norm(dim=-1,keepdim=True)).float().cpu().numpy()
    good=[]; vectors=[]; failures=[]
    for start in range(0,len(rows),16):
        chunk=[]; meta=[]
        for r in rows[start:start+16]:
            frames=decode(r["path"])
            if frames is None: failures.append(r);continue
            good.append(r);meta.append(r)
            for name in CONDITIONS:chunk.extend(corrupt(frames,name))
        if chunk:
            z=encode_images(model,pre,chunk,device).reshape(len(meta),len(CONDITIONS),16,-1);vectors.append(z)
    arr=np.concatenate(vectors) if vectors else np.empty((0,5,16,512))
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,frames=arr,labels=np.array([r["label"] for r in good]),splits=np.array([r["split"] for r in good]),prompts=p)
    return good,failures,arr,np.array([r["label"] for r in good]),np.array([r["split"] for r in good]),p

class Temporal(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv1d(512,512,3,padding=1);self.head=torch.nn.Linear(512,len(CLASSES))
    def forward(self,x):return self.head((x+self.conv(x.transpose(1,2)).relu().transpose(1,2)).mean(1))

class Mean(torch.nn.Module):
    def __init__(self):super().__init__();self.head=torch.nn.Linear(512,len(CLASSES))
    def forward(self,x):return self.head(x.mean(1))

def accuracy(logits,y):return float((logits.argmax(1)==y).float().mean())
def worst_recall(logits,y):
    pred=logits.argmax(1);return float(min(((pred[y==c]==c).float().mean()).item() for c in range(len(CLASSES))))

def train_model(cls,frames,y,split,seed,device):
    torch.manual_seed(seed);np.random.seed(seed);m=cls().to(device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    tr=np.where(split=="train")[0];te=np.where(split=="test")[0];rng=np.random.default_rng(seed)
    for _ in range(30):
        for st in range(0,len(tr),32):
            ids=rng.permutation(tr)[st:st+32];x=torch.from_numpy(frames[ids,0]).to(device);yy=torch.from_numpy(y[ids]).long().to(device)
            loss=torch.nn.functional.cross_entropy(m(x),yy);opt.zero_grad();loss.backward();opt.step()
    out={}
    m.eval()
    with torch.inference_mode():
        for k,name in enumerate(CONDITIONS):
            logits=[]
            for st in range(0,len(te),128):logits.append(m(torch.from_numpy(frames[te[st:st+128],k]).to(device)).cpu())
            q=torch.cat(logits);yy=torch.from_numpy(y[te]);out[name]={"accuracy":accuracy(q,yy),"worst_class_recall":worst_recall(q,yy)}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--video-root",type=Path,required=True);ap.add_argument("--split-root",type=Path,required=True);ap.add_argument("--cache",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--archive",type=Path,required=True);ap.add_argument("--split-archive",type=Path,required=True);ap.add_argument("--implementation-commit");ap.add_argument("--device",default="cuda");a=ap.parse_args()
    rows=manifest(a.split_root,a.video_root)
    if a.cache.exists():
        z=np.load(a.cache);frames,y,split,p=z["frames"],z["labels"],z["splits"].astype(str),z["prompts"];good=rows;fail=[]
    else:good,fail,frames,y,split,p=extract(rows,a.cache,a.device)
    te=np.where(split=="test")[0]; zero={};zero_correct={}
    for k,name in enumerate(CONDITIONS):
        logits=frames[te,k].mean(1)@p.T;pred=logits.argmax(1);zero_correct[name]=(pred==y[te]);zero[name]={"accuracy":float(np.mean(zero_correct[name])),"n":len(te)}
    for name in CONDITIONS[1:]:zero[name]["accuracy_delta_from_clean"]=paired_example_delta(zero_correct[name],zero_correct["clean"])
    trials=[]
    for seed in SEEDS:trials.append({"seed":seed,"mean":train_model(Mean,frames,y,split,seed,a.device),"temporal":train_model(Temporal,frames,y,split,seed,a.device)})
    deltas={}
    for cond in CONDITIONS:
        d=[r["temporal"][cond]["accuracy"]-r["mean"][cond]["accuracy"] for r in trials]
        deltas[cond]={"estimate":float(np.mean(d)),"ci95":ci(d),"trial_deltas":d,"n_trials":5}
    by=defaultdict(lambda:defaultdict(int))
    for r in fail:by[r["split"]][r["class"]]+=1
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:commit=None
    result={"schema_version":1,"plan":"FOLLOWUP_RESEARCH_PLAN.md","git_commit":a.implementation_commit or commit,"execution_checkout":commit,"classes":CLASSES,"conditions":CONDITIONS,"caps":{"train":70,"test":30},"sampled_frames":16,"model_id":"open_clip:ViT-B-32:laion2b_s34b_b79k","archive_sha256":sha(a.archive),"split_archive_sha256":sha(a.split_archive),"n_requested":len(rows),"n_decoded":len(good),"decode_failures":fail,"decode_failures_by_slice":by,"zero_shot":zero,"trials":trials,"paired_deltas_temporal_minus_mean":deltas,"n_trials":5,"seeds":SEEDS,"torch":torch.__version__,"device":torch.cuda.get_device_name(0) if a.device.startswith('cuda') else a.device,"python":platform.python_version()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
