import os, gc, glob, re, json, argparse
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.stats import spearmanr, mannwhitneyu
from tqdm import tqdm
IMG_SIZE=448; SP_SIZE=56; ZERO_EPS=1e-8; FOVR=0.49
def make_fov(size):
    y,x=np.ogrid[:size,:size]; c,r=size/2,size*FOVR
    return (y-c)**2+(x-c)**2<=r**2
FOV=make_fov(IMG_SIZE); FOV_SMALL=make_fov(SP_SIZE); FOV_T=torch.from_numpy(FOV).bool()
def lineage_of(name):
    n=name.lower()
    if "clip" in n or "laion" in n: return "CLIP"
    if "dinov3" in n: return "DINOv3"
    if "dinov2" in n: return "DINOv2"
    if "mae" in n: return "MAE"
    return None
def fov_sum_norm(cam):
    cam=cam*FOV_T; s=cam.flatten(1).sum(1); valid=s>ZERO_EPS
    out=torch.zeros_like(cam); out[valid]=100.0*cam[valid]/s[valid].view(-1,1,1)
    return out,valid
def minmax_fov(r):
    ins=r[FOV]; mn,mx=float(ins.min()),float(ins.max())
    if mx<=mn+1e-12: return None
    return np.clip((r-mn)/(mx-mn+1e-12),0,1)*FOV
def otsu(vals,bins=256):
    hist,edges=np.histogram(vals,bins=bins,range=(0.0,1.0)); hist=hist.astype(float)
    w0=np.cumsum(hist); total=w0[-1]
    if total==0: return 0.5
    centers=(edges[:-1]+edges[1:])/2; csum=np.cumsum(hist*centers); w1=total-w0
    with np.errstate(divide="ignore",invalid="ignore"):
        between=w0*w1*(csum/w0-(csum[-1]-csum)/w1)**2
    between[~np.isfinite(between)]=-1
    return float(centers[np.argmax(between)])
def iou(a,b):
    u=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/u) if u else np.nan
def safe_spearman(a,b):
    if np.ptp(a)<1e-8 or np.ptp(b)<1e-8: return np.nan
    return float(spearmanr(a,b).correlation)
def downsample(cams,chunk=128):
    out=[]
    for s in range(0,len(cams),chunk):
        b=cams[s:s+chunk].float().unsqueeze(1)
        out.append(F.interpolate(b,(SP_SIZE,SP_SIZE),mode="bilinear",align_corners=False).squeeze(1))
    return torch.cat(out).numpy()
def _clean(name): return re.sub(r"_\d{8}_\d{6}$","",name)
def read_auc(name,roots):
    for root in roots:
        for d in glob.glob(os.path.join(root,"*")):
            if _clean(os.path.basename(d))==name:
                mj=os.path.join(d,"test_metrics.json")
                if os.path.exists(mj): return float(json.load(open(mj)).get("0_auc",np.nan))
    return np.nan
def discover(sal_root,roots,auc_min):
    models={}
    for d in sorted(glob.glob(os.path.join(sal_root,"*"))):
        if not os.path.exists(os.path.join(d,"saliency_map.pth")): continue
        name=os.path.basename(d); lin=lineage_of(name); auc=read_auc(name,roots)
        if lin is None: print(f"  [skip] unknown lineage {name}"); continue
        if not (auc>=auc_min): print(f"  [skip] AUC<{auc_min}: {name} ({auc})"); continue
        models[name]=dict(dir=d,lineage=lin,auc=auc)
    return models
def load_one(d):
    cam=torch.load(os.path.join(d,"saliency_map.pth"),map_location="cpu").float()
    keys=pd.read_csv(os.path.join(d,"used.csv"))["unique_name"].astype(str).tolist()
    return cam,keys
def align(keys,common):
    pos={k:i for i,k in enumerate(keys)}; return [pos[k] for k in common]
def common_images(models):
    common=None
    for m,info in models.items():
        s=set(pd.read_csv(os.path.join(info["dir"],"used.csv"))["unique_name"].astype(str))
        common=s if common is None else (common&s)
    common=sorted(common); keep=np.ones(len(common),dtype=bool)
    for m,info in models.items():
        cam,keys=load_one(info["dir"]); cam=cam[align(keys,common)]
        _,v=fov_sum_norm(cam); keep&=v.numpy(); del cam
    kept=[common[i] for i in range(len(common)) if keep[i]]
    print(f"common TP {len(common)} -> valid {len(kept)}"); return kept
def build_lineage_agreement(models,common,out_dir,n_img=200,seed=42):
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(common),size=min(n_img,len(common)),replace=False)
    sel=[common[i] for i in idx]; small={}
    for m,info in models.items():
        cam,keys=load_one(info["dir"]); cam=cam[align(keys,sel)]; small[m]=downsample(cam); del cam
    names=list(models.keys()); intra,inter=[],[]
    for a in range(len(names)):
        for b in range(a+1,len(names)):
            ma,mb=names[a],names[b]; same=models[ma]["lineage"]==models[mb]["lineage"]
            rhos=[safe_spearman(small[ma][n][FOV_SMALL],small[mb][n][FOV_SMALL]) for n in range(len(sel))]
            (intra if same else inter).append(np.nanmean(rhos))
    intra,inter=np.array(intra),np.array(inter)
    try: p=float(mannwhitneyu(intra,inter,alternative="greater").pvalue)
    except: p=np.nan
    res=dict(intra_mean=float(np.nanmean(intra)),inter_mean=float(np.nanmean(inter)),intra_n=len(intra),inter_n=len(inter),mannwhitney_p=p)
    json.dump(res,open(os.path.join(out_dir,"lineage_agreement.json"),"w"),indent=2)
    print(f"[meeting#1] intra-lineage={res['intra_mean']:.3f} > inter-lineage={res['inter_mean']:.3f} (p={p:.2e})")
def build_consensus(models,common,mode="lineage"):
    LINS=sorted(set(v["lineage"] for v in models.values())); Nc=len(common)
    if mode=="best":
        chosen={}
        for g in LINS:
            cand={m:v for m,v in models.items() if v["lineage"]==g}
            chosen[g]=[max(cand,key=lambda m:cand[m]["auc"])]
        print("[meeting#2] best-per-family: "+", ".join(f"{g}={chosen[g][0]}" for g in LINS))
    else:
        chosen={g:[m for m,v in models.items() if v["lineage"]==g] for g in LINS}
    lin_mean={g:torch.zeros(Nc,IMG_SIZE,IMG_SIZE) for g in LINS}
    for g in LINS:
        for m in chosen[g]:
            cam,keys=load_one(models[m]["dir"]); cam=cam[align(keys,common)]
            cam,_=fov_sum_norm(cam); lin_mean[g]+=cam; del cam
        lin_mean[g].div_(float(len(chosen[g])))
    return LINS,chosen,lin_mean
def make_ref(lin_mean,g,LINS,sc):
    o=[h for h in LINS if h!=g]
    return sum(lin_mean[h] for h in o)/len(o), sum(sc[h] for h in o)/len(o)
def evaluate(models,common,LINS,chosen,lin_mean,out_dir):
    Nc=len(common); sc={g:downsample(lin_mean[g]) for g in LINS}; rows=[]
    for g in tqdm(LINS,desc="leave-one-lineage"):
        rf,rfs=make_ref(lin_mean,g,LINS,sc); rb=[None]*Nc
        for n in range(Nc):
            r=minmax_fov(rf[n].numpy()); rb[n]=None if r is None else ((r>otsu(r[FOV]))&FOV)
        for m in chosen[g]:
            cam,keys=load_one(models[m]["dir"]); cam=cam[align(keys,common)]; cs=downsample(cam)
            for n in range(Nc):
                if rb[n] is None: rows.append(dict(img=common[n],model=m,lineage=g,spearman=np.nan,iou=np.nan)); continue
                rho=safe_spearman(cs[n][FOV_SMALL],rfs[n][FOV_SMALL])
                c=minmax_fov(cam[n].numpy()); cb=None if c is None else ((c>otsu(c[FOV]))&FOV)
                rows.append(dict(img=common[n],model=m,lineage=g,spearman=rho,iou=(np.nan if cb is None else iou(cb,rb[n]))))
            del cam,cs
        del rf,rfs,rb; gc.collect()
    detail=pd.DataFrame(rows); detail.to_csv(os.path.join(out_dir,"overlap_detail.csv"),index=False)
    recs=[]
    for m in detail["model"].unique():
        sub=detail[detail["model"]==m]
        recs.append(dict(model=m,lineage=models[m]["lineage"],auc=models[m]["auc"],spearman=sub["spearman"].mean(),iou=sub["iou"].mean(),n=len(sub)))
    agg=pd.DataFrame(recs).sort_values("iou",ascending=False)
    agg["iou_rank"]=agg["iou"].rank(ascending=False,method="min").astype("Int64")
    agg.to_csv(os.path.join(out_dir,"overlap_rank.csv"),index=False)
    print("\n===== proximity to consensus (sorted by iou) =====\n"+agg.to_string(index=False))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--sal_root",required=True); ap.add_argument("--model_roots",nargs="+",required=True)
    ap.add_argument("--out_dir",required=True); ap.add_argument("--auc_min",type=float,default=0.90)
    ap.add_argument("--consensus_mode",choices=["lineage","best"],default="lineage")
    a=ap.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    models=discover(a.sal_root,a.model_roots,a.auc_min)
    print(f"selected {len(models)} models (AUC>={a.auc_min}):")
    for m,v in models.items(): print(f"  {v['lineage']:7s} {m} (auc={v['auc']:.3f})")
    common=common_images(models)
    if len(common)<50: print(f"[WARN] only {len(common)} common images")
    build_lineage_agreement(models,common,a.out_dir)
    LINS,chosen,lin_mean=build_consensus(models,common,a.consensus_mode)
    pd.DataFrame({"unique_name":common}).to_csv(os.path.join(a.out_dir,"common_keys.csv"),index=False)
    evaluate(models,common,LINS,chosen,lin_mean,a.out_dir)
    print("Done:",a.out_dir)
