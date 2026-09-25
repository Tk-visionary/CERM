from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .cerm_native_build import compile_shared_library
from .cerm_regression_native_codegen import compile_regression_native


def _arr(values, kind="double"):
    a = np.asarray(values).ravel()
    if not len(a):
        return "{0}"
    if kind == "double":
        return "{" + ",".join(f"{float(x):.17g}" for x in a) + "}"
    return "{" + ",".join(str(int(x)) for x in a) + "}"


def _decl(dst, name, values, kind="double"):
    a = np.asarray(values).ravel()
    dst.append(f"static const {kind} {name}[{max(1,len(a))}]={_arr(a,kind)};")


def _head_tail(body, head, latent, scale=1.0):
    basis = np.asarray(head._basis(head.thresholds_, head.lower_, head.upper_), float)
    for k in range(len(head.thresholds_)):
        terms = "+".join(f"({basis[k,r]:.17g})*{latent[r]}" for r in range(len(latent))) or "0.0"
        body += [
            f"  double z_{k}=({float(head.intercept_):.17g})+{terms};",
            f"  if(z_{k}>40.0)z_{k}=40.0;else if(z_{k}<-40.0)z_{k}=-40.0;",
            f"  const double p_{k}=1.0/(1.0+std::exp(-z_{k}));",
        ]
    t = np.asarray(head.thresholds_, float)
    lower, upper = float(head.lower_), float(head.upper_)
    body.append(f"  double residual={lower:.17g};")
    body.append(f"  residual+=0.5*({float(t[0]-lower):.17g})*(1.0+p_0);")
    for k in range(1, len(t)):
        body.append(f"  residual+=0.5*({float(t[k]-t[k-1]):.17g})*(p_{k-1}+p_{k});")
    body.append(f"  residual+=0.5*({float(upper-t[-1]):.17g})*p_{len(t)-1};")
    body.append("  out[i]=residual;" if scale == 1.0 else f"  out[i]=({float(scale):.17g})*residual;")


def _shared_source(model):
    rep, head = model.representation_, model.head_
    if rep is None or head is None:
        raise RuntimeError("shared fused model is missing fitted representation/head")
    enc, maps, se = rep.encoder, rep.maps, rep.state_encoder
    fi = np.asarray(rep.feature_idx, np.int64)
    levels = tuple(map(int, rep.levels)); coarse = levels[0]
    main_levels = [x for x in levels if x <= int(rep.max_main_level)]
    pair_levels = [x for x in levels if x <= min(int(rep.max_bins), 8)]
    pairs = tuple(tuple(map(int,p)) for p in rep.pairs)
    fine = set(tuple(map(int,p)) for p in rep.fine_pairs)
    coef = np.asarray(head.state_coef_, float); tc = np.asarray(head.threshold_coef_, float)
    rank = coef.shape[1]
    if int(se.n_features_out_) != coef.shape[0]:
        raise RuntimeError("shared fused state/head dimension mismatch")
    dec = ["#include <algorithm>","#include <cmath>","#include <cstdint>","#include <cstddef>",f"static constexpr int R={rank};"]
    body = ['extern "C" __attribute__((visibility("default"))) void cerm_fused_residual_predict(const double* X,int n,int d,double* out){'," for(int i=0;i<n;++i){","  const double* row=X+(size_t)i*d;"]
    states = {level: [] for level in levels}
    for j, raw0 in enumerate(fi):
        raw = int(raw0); fine_name = f"fine_{j}"
        if bool(enc.direct_state_mask_[raw]):
            card = int(enc.direct_state_cardinalities_[raw])
            body += [f"  long {fine_name}=std::lround(row[{raw}]);",f"  if(!std::isfinite(row[{raw}])||{fine_name}<0||{fine_name}>={card}){fine_name}=0;"]
        else:
            th = np.asarray(enc.thresholds_[raw], float); name=f"input_threshold_{j}"; _decl(dec,name,th)
            body.append(f"  const int {fine_name}=int(std::upper_bound({name},{name}+{len(th)},row[{raw}])-{name});")
        for level in levels:
            m=np.asarray(enc.maps_[level][raw],np.int32); name=f"state_map_{level}_{j}"; _decl(dec,name,m,"int32_t")
            s=f"state_{level}_{j}"; body.append(f"  const int32_t {s}={name}[{fine_name}];"); states[level].append(s)
    cards=np.asarray(se.cardinalities_,np.int64); offsets=np.asarray(se.offsets_,np.int64)
    latent=[f"latent_{r}" for r in range(rank)]
    for r,name in enumerate(latent): body.append(f"  double {name}={tc[r]:.17g};")
    col=0
    def emit(expr):
        nonlocal col
        card=int(cards[col]); table=np.zeros((card,rank)); start=int(offsets[col])
        for state in range(1,card): table[state]=coef[start+state-1]
        name=f"code_lookup_{col}"; _decl(dec,name,table.ravel()); code=f"code_{col}"
        body.extend([f"  const int {code}=int({expr});",f"  if({code}>0&&{code}<{card}){{"])
        for r,l in enumerate(latent): body.append(f"   {l}+={name}[{code}*R+{r}];")
        body.append("  }"); col += 1
    for j in range(len(fi)):
        emit(states[coarse][j])
        for level in main_levels[1:]:
            m=np.asarray(maps.main[level][j],np.int32); name=f"main_residual_{level}_{j}"; _decl(dec,name,m,"int32_t"); emit(f"{name}[{states[level][j]}]")
    for pi,(left,right) in enumerate(pairs):
        raw_right=int(fi[right])
        for level in pair_levels:
            joint=f"({states[level][left]}*{int(enc.cardinalities_[level][raw_right])}+{states[level][right]})"
            if level==coarse: emit(joint)
            else:
                m=np.asarray(maps.pair[level][(left,right)],np.int32); name=f"pair_residual_{level}_{pi}"; _decl(dec,name,m,"int32_t"); emit(f"{name}[{joint}]")
        if 16 in levels and (left,right) in fine:
            joint=f"({states[16][left]}*{int(enc.cardinalities_[16][raw_right])}+{states[16][right]})"
            m=np.asarray(maps.pair[16][(left,right)],np.int32); name=f"pair_residual_16_{pi}"; _decl(dec,name,m,"int32_t"); emit(f"{name}[{joint}]")
    if col != len(cards):
        raise RuntimeError(f"shared fused code layout mismatch: emitted {col}, expected {len(cards)}")
    _head_tail(body,head,latent,float(model.residual_scale_)); body += [" }","}"]
    return "\n".join(dec+body)


def _raw_source(model):
    corr=model.raw_correction_
    if corr is None: raise RuntimeError("raw fused model is missing fitted correction")
    head=corr.head_; coef=np.asarray(head.state_coef_,float); tc=np.asarray(head.threshold_coef_,float); rank=coef.shape[1]; d=int(model.n_features_in_)
    dec=["#include <algorithm>","#include <cmath>","#include <cstdint>","#include <cstddef>",f"static constexpr int R={rank};"]
    body=['extern "C" __attribute__((visibility("default"))) void cerm_fused_residual_predict(const double* X,int n,int d,double* out){'," for(int i=0;i<n;++i){","  const double* row=X+(size_t)i*d;"]
    latent=[f"latent_{r}" for r in range(rank)]
    for r,name in enumerate(latent): body.append(f"  double {name}={tc[r]:.17g};")
    states={}; offset=0
    for res in map(int,corr.resolutions):
        edges,_,cards=corr.state_sets_[res]; cards=np.asarray(cards,np.int64); states[res]=[]; fo=0
        for j in range(d):
            edge=np.asarray(edges[j],float); en=f"raw_edge_{res}_{j}"; _decl(dec,en,edge); s=f"raw_state_{res}_{j}"; body.append(f"  const int {s}=int(std::upper_bound({en},{en}+{len(edge)},row[{j}])-{en});"); states[res].append(s)
            card=int(cards[j]); table=coef[offset+fo:offset+fo+card]; ln=f"raw_lookup_{res}_{j}"; _decl(dec,ln,table.ravel())
            for r,l in enumerate(latent): body.append(f"  {l}+={ln}[{s}*R+{r}];")
            fo += card
        offset += int(cards.sum())
    pres=int(corr.pair_resolution); edges,_,cards=corr.state_sets_[pres]; cards=np.asarray(cards,np.int64)
    if pres not in states:
        states[pres]=[]
        for j in range(d):
            edge=np.asarray(edges[j],float); en=f"pair_edge_{j}"; _decl(dec,en,edge); s=f"pair_state_{j}"; body.append(f"  const int {s}=int(std::upper_bound({en},{en}+{len(edge)},row[{j}])-{en});"); states[pres].append(s)
    for pi,(left,right) in enumerate(corr.pairs_):
        left,right=int(left),int(right); card=int(cards[left])*int(cards[right]); table=coef[offset:offset+card]; ln=f"raw_pair_lookup_{pi}"; _decl(dec,ln,table.ravel()); code=f"({states[pres][left]}*{int(cards[right])}+{states[pres][right]})"
        for r,l in enumerate(latent): body.append(f"  {l}+={ln}[{code}*R+{r}];")
        offset += card
    if offset != coef.shape[0]: raise RuntimeError(f"raw fused state/head dimension mismatch: emitted {offset}, expected {coef.shape[0]}")
    _head_tail(body,head,latent); body += [" }","}"]
    return "\n".join(dec+body)


def render_fused_residual_native_source(model):
    """Render deterministic residual C++; semantic IR can later replace this extraction boundary."""
    kind=str(model.selected_kind_)
    if kind=="fused": return _shared_source(model)
    if kind=="raw_fused": return _raw_source(model)
    if kind=="current_mean": raise ValueError("current_mean has no residual native source")
    raise ValueError(f"unsupported fused regression selected_kind: {kind!r}")


@dataclass
class FusedRegressionNativeArtifact:
    predict_raw: Any
    selected_kind: str
    n_features_in: int
    source_path: Path
    library_path: Path
    baseline_source_path: Path
    baseline_library_path: Path
    correction_source_path: Path | None = None
    correction_library_path: Path | None = None
    compile_seconds: float = 0.0

    def _matrix(self,X):
        a=np.ascontiguousarray(X,dtype=np.float64)
        if a.ndim!=2: raise ValueError("X must be two-dimensional")
        if a.shape[1]!=int(self.n_features_in): raise ValueError(f"X has {a.shape[1]} features, expected {int(self.n_features_in)}")
        return a
    def predict(self,X): return np.asarray(self.predict_raw(self._matrix(X)),dtype=np.float64)
    decision_function=predict
    @property
    def artifact_bytes(self):
        return int(sum(p.stat().st_size for p in (self.baseline_library_path,self.correction_library_path) if p is not None and p.is_file()))
    @property
    def source_bytes(self):
        return int(sum(p.stat().st_size for p in (self.baseline_source_path,self.correction_source_path) if p is not None and p.is_file()))


def compile_fused_regression_native(model,prefix):
    """Compile fitted fused regression through CERM's existing native build/runtime contract."""
    from ..native_runtime import load_native_predictor
    if not hasattr(model,"baseline_") or not hasattr(model,"selected_kind_"): raise RuntimeError("FusedResidualCERMRegressor must be fitted before native compilation")
    kind=str(model.selected_kind_)
    if kind not in {"current_mean","fused","raw_fused"}: raise ValueError(f"unsupported fused regression selected_kind: {kind!r}")
    prefix=Path(prefix); prefix.parent.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); bp=prefix.parent/f"{prefix.name}_baseline"
    baseline_predict,bs,bl=compile_regression_native(model.baseline_,bp)
    cs=cl=None
    if kind=="current_mean": raw=baseline_predict; source,library=Path(bs),Path(bl)
    else:
        cs=prefix.parent/f"{prefix.name}_residual.cpp"; cl=prefix.parent/f"{prefix.name}_residual.so"; cs.write_text(render_fused_residual_native_source(model),encoding="utf-8"); compile_shared_library(cs,cl); correction=load_native_predictor(cl,symbol="cerm_fused_residual_predict")
        def raw(X):
            a=np.ascontiguousarray(X,dtype=np.float64); return baseline_predict(a)+correction(a)
        raw._cerm_baseline_predictor=baseline_predict; raw._cerm_correction_predictor=correction; source,library=cs,cl
    artifact=FusedRegressionNativeArtifact(raw,kind,int(model.n_features_in_),Path(source),Path(library),Path(bs),Path(bl),None if cs is None else Path(cs),None if cl is None else Path(cl),float(time.perf_counter()-start))
    composite=artifact.predict_raw
    def checked(X): return composite(artifact._matrix(X))
    checked._cerm_native_artifact=artifact; artifact.predict_raw=checked
    return artifact
