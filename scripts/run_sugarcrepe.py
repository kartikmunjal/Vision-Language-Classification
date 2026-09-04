#!/usr/bin/env python3
"""Evaluate frozen OpenCLIP on the official SugarCrepe release."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
import numpy as np,torch
from PIL import Image
import open_clip

FILES=["add_att","add_obj","replace_att","replace_obj","replace_rel","swap_att","swap_obj"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ci(correct):
    x=np.asarray(correct,float);rng=np.random.default_rng(20260903);m=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return [float(v) for v in np.quantile(m,[.025,.975])]
def enc_image(model,pre,paths,device):
    out=[]
    for i in range(0,len(paths),128):
        b=torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i+128]]).to(device)
        with torch.inference_mode(),torch.autocast(device_type="cuda",enabled=device.startswith("cuda")):
            z=model.encode_image(b);z=z/z.norm(dim=-1,keepdim=True)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)
def enc_text(model,tok,texts,device):
    out=[]
    for i in range(0,len(texts),256):
        b=tok(texts[i:i+256]).to(device)
        with torch.inference_mode(),torch.autocast(device_type="cuda",enabled=device.startswith("cuda")):
            z=model.encode_text(b);z=z/z.norm(dim=-1,keepdim=True)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-root",type=Path,required=True);ap.add_argument("--image-root",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--source-commit",required=True);ap.add_argument("--implementation-commit");ap.add_argument("--device",default="cuda");a=ap.parse_args()
    model,_,pre=open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k");model=model.eval().to(a.device);tok=open_clip.get_tokenizer("ViT-B-32")
    result={};all_correct=[];predictions=[]
    for family in FILES:
        path=a.data_root/f"{family}.json";data=json.loads(path.read_text());items=list(data.values());names=sorted(set(x["filename"] for x in items));imap={x:i for i,x in enumerate(names)}
        images=enc_image(model,pre,[a.image_root/x for x in names],a.device);pos=enc_text(model,tok,[x["caption"] for x in items],a.device);neg=enc_text(model,tok,[x["negative_caption"] for x in items],a.device);iv=np.array([images[imap[x["filename"]]] for x in items]);margin=np.sum(iv*pos,1)-np.sum(iv*neg,1);correct=margin>0
        result[family]={"accuracy":float(correct.mean()),"ci95":ci(correct),"n":len(items),"sha256":sha(path)};all_correct.extend(correct.tolist())
        predictions.extend({"family":family,"filename":x["filename"],"correct":bool(c),"margin":float(m)} for x,c,m in zip(items,correct,margin))
    for group,prefix in [("add","add_"),("replace","replace_"),("swap","swap_")]:
        vals=[p["correct"] for p in predictions if p["family"].startswith(prefix)];result[group]={"accuracy":float(np.mean(vals)),"ci95":ci(vals),"n":len(vals)}
    result["macro_family_accuracy"]={"estimate":float(np.mean([result[x]["accuracy"] for x in FILES])),"n_families":len(FILES)}
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:commit=None
    payload={"schema_version":1,"plan":"FOLLOWUP_RESEARCH_PLAN.md","git_commit":a.implementation_commit or commit,"execution_checkout":commit,"source_repository":"https://github.com/RAIVNLab/sugar-crepe","source_commit":a.source_commit,"model_id":"open_clip:ViT-B-32:laion2b_s34b_b79k","result":result,"predictions":predictions,"n_trials":1,"bootstrap_replicates":10000,"device":torch.cuda.get_device_name(0) if a.device.startswith('cuda') else a.device,"torch":torch.__version__,"python":platform.python_version()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2)+"\n")
if __name__=="__main__":main()
