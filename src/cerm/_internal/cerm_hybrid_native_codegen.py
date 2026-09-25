from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import numpy as np


def _arr(values, typ="double"):
    vals=np.asarray(values).ravel()
    if len(vals)==0:return "{0}"
    if typ=="double":return "{"+",".join(f"{float(v):.17g}" for v in vals)+"}"
    return "{"+",".join(str(int(v)) for v in vals)+"}"


def _compile_lookup(oh, coef):
    coef=np.asarray(coef,dtype=float); out=[]; pos=0
    drop=oh.drop_idx_
    for col,cats0 in enumerate(oh.categories_):
        cats=np.asarray(cats0,dtype=np.int64)
        size=int(cats.max())+1 if len(cats) else 1
        tab=np.zeros(size,dtype=float)
        dropped=None if drop is None else drop[col]
        for cat_pos,cat in enumerate(cats):
            if dropped is not None and cat_pos==int(dropped):continue
            tab[int(cat)]=coef[pos];pos+=1
        out.append(tab)
    if pos!=len(coef):raise RuntimeError(f"coef mismatch {pos} != {len(coef)}")
    return out


def compile_hybrid_native(model,prefix,*,native_arch=True):
    prefix=Path(prefix);cpp=prefix.with_suffix('.cpp');so=prefix.with_suffix('.so')
    base=model.base_; idx=np.asarray(base.feature_idx_,dtype=np.int32); nf=len(idx)
    levels=tuple(int(level) for level in base.encoder_.levels)
    block_level=int(getattr(model,"block_level",levels[-1]))
    maxth=max([len(base.encoder_.thresholds_[int(j)]) for j in idx]+[1])
    maxfine=max([len(base.encoder_.maps_[levels[-1]][int(j)]) for j in idx]+[1])
    nth=np.zeros(nf,dtype=np.int32); th=np.zeros((nf,maxth))
    state_maps={level:np.zeros((nf,maxfine),dtype=np.int16) for level in (4,8,16)}
    direct=np.zeros(nf,dtype=np.int32); direct_card=np.zeros(nf,dtype=np.int32)
    for qidx,r in enumerate(idx):
        r=int(r);t=base.encoder_.thresholds_[r];nth[qidx]=len(t);th[qidx,:len(t)]=t
        direct[qidx]=int(base.encoder_.direct_state_mask_[r])
        direct_card[qidx]=int(base.encoder_.direct_state_cardinalities_[r])
        for requested in (4,8,16):
            source=max(level for level in levels if level <= min(requested,levels[-1]))
            mp=base.encoder_.maps_[source][r]
            state_maps[requested][qidx,:len(mp)]=mp
    luts=_compile_lookup(base.oh_,model.base_coef_)
    lines=['#include <cmath>','#include <cstdint>','#include <cstddef>','extern "C" void cerm_predict(const double* X,int n,int d,double* out){']
    lines += [f' static const int NF={nf},MAXTH={maxth},MAXF={maxfine};',f' static const int feature_idx[NF]={_arr(idx,"int")};',f' static const int nth[NF]={_arr(nth,"int")};',f' static const int direct[NF]={_arr(direct,"int")};',f' static const int direct_card[NF]={_arr(direct_card,"int")};']
    lines += [' static const double thresholds[NF][MAXTH]={' + ','.join(_arr(r) for r in th) + '};']
    for level in (4,8,16):
        lines.append(f' static const int16_t map{level}[NF][MAXF]={{' + ','.join(_arr(r,'int') for r in state_maps[level]) + '};')
    for i,t in enumerate(luts):lines.append(f' static const double lut_{i}[{max(len(t),1)}]={_arr(t)};')
    for gi,g in enumerate(model.execution_groups_):
        if g['kind'] in ('block','fused_pair'):
            tab=np.asarray(g['table'],dtype=float)
            lines.append(f' static const double gtab_{gi}[{tab.size}]={_arr(tab)};')
        else:
            for ti,(_,tab0) in enumerate(g['targets']):
                tab=np.asarray(tab0,dtype=float);lines.append(f' static const double gtab_{gi}_{ti}[{tab.size}]={_arr(tab)};')
    lines += [f' const double intercept={model.intercept_:.17g};',' for(int i=0;i<n;++i){','  const double* row=X+(size_t)i*d;','  int16_t s4[NF],s8[NF],s16[NF];','  for(int j=0;j<NF;++j){ double v=row[feature_idx[j]]; int f=0; if(direct[j]){ f=(int)std::llround(v); if(f<0)f=0; if(f>=direct_card[j])f=direct_card[j]-1; } else { while(f<nth[j] && v>=thresholds[j][f]) ++f; } s4[j]=map4[j][f]; s8[j]=map8[j][f]; s16[j]=map16[j][f]; }','  double score=intercept;']
    state_name={4:'s4',8:'s8',16:'s16'}
    op=0
    main_levels=[level for level in levels if level <= int(base.config_.max_main_level)]
    for j,raw_j in enumerate(idx):
        for pos,level in enumerate(main_levels):
            state=state_name[level]
            if pos==0:
                lines.append(f'  {{ int c={state}[{j}]; if(c<{len(luts[op])}) score+=lut_{op}[c]; }}')
            else:
                arr=np.asarray(base.main_residuals_[level][j],dtype=np.int32)
                lines.insert(4,f' static const int mainres{level}map_{j}[{len(arr)}]={_arr(arr,"int")};')
                lines.append(f'  {{ int c=mainres{level}map_{j}[{state}[{j}]]; if(c<{len(luts[op])}) score+=lut_{op}[c]; }}')
            op+=1
    fine_pairs=set(base.fine_pairs_)
    for pi,(j,k) in enumerate(base.pairs_):
        rawk=int(idx[k])
        for pos,level in enumerate(levels):
            if level>8 and (j,k) not in fine_pairs:
                continue
            state=state_name[level]
            cardk=int(base.encoder_.cardinalities_[level][rawk])
            if pos==0:
                lines.append(f'  {{ int c={state}[{j}]*{cardk}+{state}[{k}]; if(c<{len(luts[op])}) score+=lut_{op}[c]; }}')
            else:
                arr=np.asarray(base.pair_residuals_[level][(j,k)],dtype=np.int32)
                lines.insert(4,f' static const int pairres{level}map_{pi}[{len(arr)}]={_arr(arr,"int")};')
                lines.append(f'  {{ int z={state}[{j}]*{cardk}+{state}[{k}]; int c=pairres{level}map_{pi}[z]; if(c<{len(luts[op])}) score+=lut_{op}[c]; }}')
            op+=1
    if op!=len(luts):raise RuntimeError(f'op mismatch {op} {len(luts)}')
    block_state=state_name[block_level]
    for gi,g in enumerate(model.execution_groups_):
        if g['kind']=='block':
            lines.append(f'  if(s4[{g["gate_j"]}]=={g["gate_state"]}) score+=gtab_{gi}[{block_state}[{g["target_k"]}]];')
        elif g['kind']=='fused_pair':
            tab=np.asarray(g['table']);card=tab.shape[1]
            lines.append(f'  score+=gtab_{gi}[s4[{g["gate_j"]}]*{card}+{block_state}[{g["target_k"]}]];')
        else:
            lines.append(f'  if(s4[{g["gate_j"]}]=={g["gate_state"]}){{')
            for ti,(tk,tab) in enumerate(g['targets']):lines.append(f'   score+=gtab_{gi}_{ti}[{block_state}[{tk}]];')
            lines.append('  }')
    lines += ['  out[i]=1.0/(1.0+std::exp(-score));',' }','}']
    cpp.write_text('\n'.join(lines))
    flags=['g++','-O3','-std=c++17','-shared','-fPIC']
    if native_arch: flags.append('-march=native')
    flags += [str(cpp),'-o',str(so)]
    subprocess.run(flags,check=True,capture_output=True)
    lib=ctypes.CDLL(str(so));fn=lib.cerm_predict;fn.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)];fn.restype=None
    def predict(X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2: raise ValueError('X must be two-dimensional')
        if X.shape[1] <= int(idx.max(initial=-1)): raise ValueError('input width mismatch')
        X=np.ascontiguousarray(X);out=np.empty(len(X),dtype=np.float64);fn(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(X),X.shape[1],out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)));return out
    return predict,cpp,so
