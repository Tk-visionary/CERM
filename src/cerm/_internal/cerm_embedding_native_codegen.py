
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess

import numpy as np


def _c_array(values, typ="double"):
    arr = np.asarray(values).ravel()
    if typ == "double":
        return "{" + ",".join(f"{float(v):.17g}" for v in arr) + "}"
    return "{" + ",".join(str(int(v)) for v in arr) + "}"


def compile_embedding_prototype_adapter(adapter, prefix):
    prefix = Path(prefix)
    cpp = prefix.with_suffix(".cpp")
    so = prefix.with_suffix(".so")

    mean = np.asarray(adapter.mean_, dtype=np.float64)
    scale = np.asarray(adapter.scale_, dtype=np.float64)
    pca_mean = np.asarray(adapter.pca_.mean_, dtype=np.float64)
    components = np.asarray(adapter.pca_.components_, dtype=np.float64)
    lda_coef = np.asarray(adapter.lda_.coef_[0], dtype=np.float64)
    lda_intercept = float(adapter.lda_.intercept_[0])
    proto0 = np.asarray(adapter.prototypes_[0], dtype=np.float64)
    proto1 = np.asarray(adapter.prototypes_[1], dtype=np.float64)
    thresholds = [np.asarray(t, dtype=np.float64) for t in adapter.thresholds_]

    d = len(mean)
    npc = components.shape[0]
    n0, n1 = len(proto0), len(proto1)
    nf = len(thresholds)
    max_th = max(1, max(len(t) for t in thresholds))
    nth = np.asarray([len(t) for t in thresholds], dtype=np.int32)
    th_rect = np.zeros((nf, max_th), dtype=np.float64)
    for j, th in enumerate(thresholds):
        th_rect[j, : len(th)] = th

    lines = [
        "#include <cmath>",
        "#include <cstdint>",
        "#include <cstddef>",
        f"static const int D={d}, NPC={npc}, N0={n0}, N1={n1}, NF={nf}, MAXTH={max_th};",
        f"static const double mean_[D]={_c_array(mean)};",
        f"static const double scale_[D]={_c_array(scale)};",
        f"static const double pca_mean_[D]={_c_array(pca_mean)};",
        "static const double pca_[NPC][D]={" + ",".join(_c_array(r) for r in components) + "};",
        f"static const double lda_coef_[D]={_c_array(lda_coef)};",
        f"static const double lda_intercept_={lda_intercept:.17g};",
        "static const double proto0_[N0][D]={" + ",".join(_c_array(r) for r in proto0) + "};",
        "static const double proto1_[N1][D]={" + ",".join(_c_array(r) for r in proto1) + "};",
        f"static const int nth_[NF]={_c_array(nth, 'int')};",
        "static const double thresholds_[NF][MAXTH]={" + ",".join(_c_array(r) for r in th_rect) + "};",
        f"static const double soft_scale_={float(adapter.soft_scale_):.17g};",
        'extern "C" void embedding_states(const double* X,int n,int input_d,double* out){',
        " for(int i=0;i<n;++i){",
        "  const double* row=X+(size_t)i*input_d;",
        "  double z[D];",
        "  for(int j=0;j<D;++j){ double v=row[j]; if(!std::isfinite(v)) v=mean_[j]; z[j]=(v-mean_[j])/scale_[j]; }",
        "  double feat[NF];",
        "  for(int q=0;q<NPC;++q){ double s=0.0; for(int j=0;j<D;++j) s+=(z[j]-pca_mean_[j])*pca_[q][j]; feat[q]=s; }",
        "  double lda=lda_intercept_; for(int j=0;j<D;++j) lda+=z[j]*lda_coef_[j]; feat[NPC]=lda;",
        "  double d0=1.0e300,d1=1.0e300;",
        "  for(int q=0;q<N0;++q){ double s=0.0; for(int j=0;j<D;++j){ double e=z[j]-proto0_[q][j]; s+=e*e; } if(s<d0)d0=s; }",
        "  for(int q=0;q<N1;++q){ double s=0.0; for(int j=0;j<D;++j){ double e=z[j]-proto1_[q][j]; s+=e*e; } if(s<d1)d1=s; }",
        "  feat[NPC+1]=d0; feat[NPC+2]=d1; feat[NPC+3]=d0-d1;",
        "  double a=(d1-d0)/soft_scale_; if(a>35.0)a=35.0; if(a<-35.0)a=-35.0; feat[NPC+4]=1.0/(1.0+std::exp(a));",
        "  for(int q=0;q<NF;++q){ int s=0; while(s<nth_[q] && feat[q]>=thresholds_[q][s]) ++s; out[(size_t)i*NF+q]=(double)s; }",
        " }",
        "}",
    ]
    cpp.write_text("\n".join(lines), encoding="utf-8")
    subprocess.run(
        ["g++", "-O3", "-march=native", "-shared", "-fPIC", str(cpp), "-o", str(so)],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(so))
    fn = lib.embedding_states
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    fn.restype = None

    def transform(X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        out = np.empty((len(X), nf), dtype=np.float64)
        fn(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(X),
            X.shape[1],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out

    return transform, cpp, so
