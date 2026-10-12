"""Full-row incremental screen of candidate pool members against a finished stack (10-12).

    python src/screen_pool.py --stack pr34b_bagstack_fwdlegs pm1012 cz-v12

Candidates = pool members whose name contains any positional substring. For each: 5-fold LR on
(stack logit, candidate z-logit) over all 700k rows minus the stack alone. Controls printed
first: a noise column (null ~ -0.000001) and the in-pool TabPFN-3.5 (+0.000002).
"""
import argparse
import numpy as np, polars as pl, sys, glob
sys.path.insert(0,'src')
from features import build_features, TARGET
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
y=build_features(pl.read_csv('data/train.csv'))[TARGET].to_numpy()
ap=argparse.ArgumentParser(); ap.add_argument('--stack',default='pr34b_bagstack_fwdlegs'); ap.add_argument('subs',nargs='+'); A=ap.parse_args()
s=np.load(f'results/oof_{A.stack}.npy')
def lg(p):
    p=np.asarray(p,float).ravel()
    if p.min()<0 or p.max()>1: return p
    q=np.clip(p,1e-7,1-1e-7); return np.log(q/(1-q))
def cv(X):
    o=np.zeros(len(y))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=7).split(X,y):
        o[va]=LogisticRegression(max_iter=500).fit(X[tr],y[tr]).decision_function(X[va])
    return roc_auc_score(y,o)
base=cv(s[:,None]); print('base',round(base,6))
rng=np.random.default_rng(0)
cands={'noise':rng.normal(size=len(y)),'inpool_tabpfn35':np.load('data/public_oof/_pool/oof_tfm-oof-and-te__tabpfn35.npy')}
for f in sorted(glob.glob('data/public_oof/_pool/oof_*.npy')):
    if any(k in f for k in A.subs): cands[f.split('oof_')[-1][:-4]]=np.load(f)
for k,v in cands.items():
    v=lg(v); v=(v-v.mean())/v.std()
    print(f'{k:40s} {cv(np.c_[s,v])-base:+.6f}',flush=True)
