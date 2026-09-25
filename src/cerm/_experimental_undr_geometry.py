from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from scipy.linalg import qr
from scipy.stats import chi2

@dataclass
class StateQuotientBasis:
    name: str
    codes: np.ndarray          # row -> state id
    nstates: int
    A: np.ndarray              # state -> quotient function coordinates; H-orthonormal on sample
    t: np.ndarray              # Q^T z = - A^T G_state
    df: int
    gain: float
    p_ref: float
    family: str = ''
    meta: Optional[dict] = None


def logistic_gh(y, score):
    s=np.asarray(score,float); yy=np.asarray(y,float)
    p=1/(1+np.exp(-np.clip(s,-50,50)))
    g=p-yy; h=p*(1-p)
    if np.any(h<=0): raise ValueError('nonpositive Hessian')
    return g,h


def _rank_from_R(R, shape):
    d=np.abs(np.diag(R))
    if d.size==0 or d[0]==0:return 0
    tol=max(shape)*np.finfo(float).eps*d[0]
    return int(np.count_nonzero(d>tol))


def _state_quotient(name, codes, nstates, nuisance_design, g, h, *, family='', meta=None, eligible=None):
    codes=np.asarray(codes,int); g=np.asarray(g,float); h=np.asarray(h,float)
    if len(codes)!=len(g) or len(g)!=len(h): raise ValueError('length mismatch')
    H=np.bincount(codes,weights=h,minlength=nstates).astype(float)
    G=np.bincount(codes,weights=g,minlength=nstates).astype(float)
    N=np.bincount(codes,minlength=nstates)
    active=(H>0)
    if eligible is not None:
        ee=np.asarray(eligible,bool)
        if ee.shape!=(nstates,): raise ValueError('eligible shape')
        active &= ee
    ids=np.flatnonzero(active)
    if len(ids)==0:
        return StateQuotientBasis(name,codes,nstates,np.zeros((nstates,0)),np.zeros(0),0,0.,1.,family,meta)
    D=np.asarray(nuisance_design,float)
    if D.shape[0]!=nstates: raise ValueError('nuisance rows != nstates')
    sh=np.sqrt(H[ids])
    C=D[ids]*sh[:,None]
    # Full Q gives an orthonormal basis of state-support coordinates; trailing columns span C^perp.
    Q,R,piv=qr(C,mode='full',pivoting=True,check_finite=False)
    r=_rank_from_R(R,C.shape)
    Qc=Q[:,r:]
    df=Qc.shape[1]
    A=np.zeros((nstates,df),float)
    if df:
        A[ids]=Qc/sh[:,None]
        t=-(A.T@G)
        gain=.5*float(t@t)
        p=float(chi2.sf(2*gain,df))
    else:
        t=np.zeros(0);gain=0.;p=1.
    return StateQuotientBasis(name,codes,nstates,A,t,df,gain,p,family,meta)



def _state_quotient_from_stats(
    name,
    codes,
    nstates,
    nuisance_design,
    G,
    H,
    *,
    family="",
    meta=None,
    eligible=None,
):
    """Build an exact quotient basis from retained state G/H statistics.

    Row codes are still retained for later cross-Gram calculations, but the
    gradient/Hessian aggregation is reused from candidate search instead of
    rescanning all training rows.
    """
    codes = np.asarray(codes)
    if not np.issubdtype(codes.dtype, np.integer):
        codes = np.asarray(codes, int)
    G = np.asarray(G, float).reshape(-1)
    H = np.asarray(H, float).reshape(-1)
    if G.shape != (int(nstates),) or H.shape != (int(nstates),):
        raise ValueError("state-stat shape mismatch")
    if len(codes) and (
        np.min(codes, initial=0) < 0
        or np.max(codes, initial=0) >= int(nstates)
    ):
        raise ValueError("state code outside domain")
    active = H > 0
    if eligible is not None:
        ee = np.asarray(eligible, bool)
        if ee.shape != (int(nstates),):
            raise ValueError("eligible shape")
        active &= ee
    ids = np.flatnonzero(active)
    if len(ids) == 0:
        return StateQuotientBasis(
            name, codes, int(nstates), np.zeros((int(nstates), 0)),
            np.zeros(0), 0, 0.0, 1.0, family, meta
        )
    D = np.asarray(nuisance_design, float)
    if D.shape[0] != int(nstates):
        raise ValueError("nuisance rows != nstates")
    sh = np.sqrt(H[ids])
    C = D[ids] * sh[:, None]
    Q, R, piv = qr(C, mode="full", pivoting=True, check_finite=False)
    r = _rank_from_R(R, C.shape)
    Qc = Q[:, r:]
    df = int(Qc.shape[1])
    A = np.zeros((int(nstates), df), float)
    if df:
        A[ids] = Qc / sh[:, None]
        t = -(A.T @ G)
        gain = 0.5 * float(t @ t)
        p_ref = float(chi2.sf(2 * gain, df))
    else:
        t = np.zeros(0)
        gain = 0.0
        p_ref = 1.0
    return StateQuotientBasis(
        name, codes, int(nstates), A, t, df, gain, p_ref, family, meta
    )


def resolution_basis_from_stats(
    states_fine,
    feature,
    source_q,
    target_q,
    G,
    H,
    *,
    eligible=None,
    name=None,
):
    # Preserve the existing compact state-bank column as a zero-copy view.
    # The retained-stat path validates integer state labels below.
    z = np.asarray(states_fine[:, feature])
    if np.min(z, initial=0) < 0 or np.max(z, initial=0) >= int(target_q):
        raise ValueError("state outside domain")
    return _state_quotient_from_stats(
        name or f"R{source_q}->{target_q}:x{feature}",
        z,
        int(target_q),
        parent_design(source_q, target_q),
        G,
        H,
        family="resolution",
        meta={
            "feature": int(feature),
            "source_q": int(source_q),
            "target_q": int(target_q),
        },
        eligible=eligible,
    )


def triad_basis_from_stats(
    S,
    triad,
    G,
    H,
    q=4,
    *,
    eligible=None,
    name=None,
):
    a, b, c = triad
    code = (
        (S[:, a].astype(np.int64) * int(q) + S[:, b]) * int(q)
        + S[:, c]
    )
    nstates = int(q) ** 3
    if nstates <= np.iinfo(np.uint8).max + 1:
        code = np.ascontiguousarray(code, dtype=np.uint8)
    elif nstates <= np.iinfo(np.uint16).max + 1:
        code = np.ascontiguousarray(code, dtype=np.uint16)
    return _state_quotient_from_stats(
        name or f"T{tuple(triad)}",
        code,
        int(q) ** 3,
        canonical_pair_closure_design(q),
        G,
        H,
        family="interaction",
        meta={"triad": tuple(triad), "q": int(q)},
        eligible=eligible,
    )

def triad_score_from_stats(
    G,
    H,
    q=4,
    *,
    eligible=None,
):
    """Return exact (df, gain, p_ref) from triad screening sufficient stats.

    Stage-2 candidate search needs only these three values.  It does not need
    row codes or a retained quotient basis; the latter is materialized later
    only if budget selection actually keeps the candidate.
    """
    nstates = int(q) ** 3
    G = np.asarray(G, float).reshape(-1)
    H = np.asarray(H, float).reshape(-1)
    if G.shape != (nstates,) or H.shape != (nstates,):
        raise ValueError("state-stat shape mismatch")
    active = H > 0
    if eligible is not None:
        ee = np.asarray(eligible, bool)
        if ee.shape != (nstates,):
            raise ValueError("eligible shape")
        active &= ee
    ids = np.flatnonzero(active)
    if len(ids) == 0:
        return 0, 0.0, 1.0

    D = canonical_pair_closure_design(q)
    sh = np.sqrt(H[ids])
    C = D[ids] * sh[:, None]
    Q, R, piv = qr(C, mode="full", pivoting=True, check_finite=False)
    r = _rank_from_R(R, C.shape)
    Qc = Q[:, r:]
    df = int(Qc.shape[1])
    if not df:
        return 0, 0.0, 1.0

    # Preserve the same arithmetic as StateQuotientBasis construction so the
    # score path remains bitwise identical to the historical exact solve.
    A = np.zeros((nstates, df), float)
    A[ids] = Qc / sh[:, None]
    t = -(A.T @ G)
    gain = 0.5 * float(t @ t)
    p_ref = float(chi2.sf(2 * gain, df))
    return df, gain, p_ref


def parent_design(source_q,target_q):
    if target_q%source_q: raise ValueError('source_q must divide target_q')
    ratio=target_q//source_q
    parent=np.arange(target_q)//ratio
    D=np.zeros((target_q,source_q),float);D[np.arange(target_q),parent]=1.
    return D


def resolution_basis(states_fine,y,score,feature,source_q,target_q,*,eligible=None,name=None):
    z=np.asarray(states_fine[:,feature],int)
    if np.min(z,initial=0)<0 or np.max(z,initial=0)>=target_q: raise ValueError('state outside domain')
    g,h=logistic_gh(y,score)
    return _state_quotient(name or f'R{source_q}->{target_q}:x{feature}',z,target_q,parent_design(source_q,target_q),g,h,
                           family='resolution',meta={'feature':feature,'source_q':source_q,'target_q':target_q},eligible=eligible)


def canonical_pair_closure_design(q):
    q=int(q);n=q**3;cols=1+3*(q-1)+3*(q-1)**2
    D=np.zeros((n,cols));D[:,0]=1.; col=1
    s=np.arange(n);a=s//(q*q);b=(s//q)%q;c=s%q
    for v in (a,b,c):
        for lev in range(q-1):D[:,col]=(v==lev);col+=1
    for l,r in ((a,b),(a,c),(b,c)):
        for lv in range(q-1):
            lm=(l==lv)
            for rv in range(q-1):D[:,col]=lm&(r==rv);col+=1
    return D


def triad_basis(S,y,score,triad,q=4,*,eligible=None,name=None):
    a,b,c=triad;code=((S[:,a].astype(np.int64)*q+S[:,b])*q+S[:,c]);g,h=logistic_gh(y,score)
    return _state_quotient(name or f'T{tuple(triad)}',code,q**3,canonical_pair_closure_design(q),g,h,
                           family='interaction',meta={'triad':tuple(triad),'q':q},eligible=eligible)


def cross_gram(a:StateQuotientBasis,b:StateQuotientBasis,h):
    if len(a.codes)!=len(b.codes) or len(h)!=len(a.codes): raise ValueError('row mismatch')
    if a.df==0 or b.df==0:return np.zeros((a.df,b.df))
    joint=np.bincount(a.codes.astype(np.int64)*b.nstates+b.codes.astype(np.int64),weights=np.asarray(h,float),minlength=a.nstates*b.nstates).reshape(a.nstates,b.nstates)
    return a.A.T@joint@b.A


def gram_and_target(bases:Sequence[StateQuotientBasis], h):
    dims=[b.df for b in bases];tot=sum(dims)
    K=np.zeros((tot,tot));t=np.zeros(tot); offs=np.cumsum([0]+dims)
    for i,b in enumerate(bases):
        sl=slice(offs[i],offs[i+1]);K[sl,sl]=np.eye(b.df);t[sl]=b.t
        for j in range(i):
            c=bases[j];sj=slice(offs[j],offs[j+1]);X=cross_gram(c,b,h);K[sj,sl]=X;K[sl,sj]=X.T
    return K,t


def _psd_eigensystem(K):
    if K.size==0:return np.zeros(0),np.zeros((0,0)),np.zeros(0,dtype=bool)
    Ks=(K+K.T)*.5
    w,V=np.linalg.eigh(Ks)
    mx=float(np.max(np.abs(w),initial=0))
    if mx==0:return w,V,np.zeros(len(w),dtype=bool)
    # One tolerance governs both df and inversion. This avoids rank/pinv disagreement
    # for nearly redundant candidate spaces.
    tol=max(K.shape)*np.finfo(float).eps*mx*64
    keep=w>tol
    return w,V,keep


def span_gain_from_gram(K, t):
    K = np.asarray(K, float)
    t = np.asarray(t, float).reshape(-1)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or len(t) != K.shape[0]:
        raise ValueError("Gram/target shape mismatch")
    if K.size == 0:
        return 0.0, 0, 1.0
    w, V, keep = _psd_eigensystem(K)
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return 0.0, 0, 1.0
    a = V[:, keep].T @ t
    gain = 0.5 * float(np.sum((a * a) / w[keep]))
    gain = max(0.0, gain)
    p = float(chi2.sf(2 * gain, rank))
    return gain, rank, p


def span_gain(bases:Sequence[StateQuotientBasis], h):
    if not bases:return 0.,0,1.
    K,t=gram_and_target(bases,h)
    return span_gain_from_gram(K,t)


def conditional_gain(selected:Sequence[StateQuotientBasis],candidate:StateQuotientBasis,h):
    g0,r0,_=span_gain(selected,h);g1,r1,_=span_gain(list(selected)+[candidate],h)
    dg=max(0.,g1-g0);dr=max(0,r1-r0);p=float(chi2.sf(2*dg,dr)) if dr else 1.
    return dg,dr,p


def overlap_metrics(a,b,h):
    if a.df==0 or b.df==0:return {'max_canonical':0.,'mean_sq_canonical':0.,'rank_union':a.df+b.df}
    X=cross_gram(a,b,h);s=np.linalg.svd(X,compute_uv=False)
    _,r,_=span_gain([a,b],h)
    return {'max_canonical':float(s[0]) if len(s) else 0.,'mean_sq_canonical':float(np.mean(s*s)) if len(s) else 0.,'rank_union':r}


def fit_quotient_lookup(basis:StateQuotientBasis, l2=20., lr=.5):
    """One regularized Newton correction constrained to the selected quotient space.

    Penalty is l2/2 * ||f_state||_2^2, where f_state=A beta.  Since A stays in the
    H-orthogonal quotient, regularization and scalar shrinkage do not re-introduce nuisance directions.
    """
    if basis.df==0:return np.zeros(basis.nstates)
    M=np.eye(basis.df)+float(l2)*(basis.A.T@basis.A)
    beta=np.linalg.solve(M,basis.t)
    return float(lr)*(basis.A@beta)


def nuisance_fraction(lookup,basis:StateQuotientBasis,y,score,nuisance_design):
    # diagnostic in H metric on represented states
    _,h=logistic_gh(y,score);H=np.bincount(basis.codes,weights=h,minlength=basis.nstates)
    v=H>0; f=np.asarray(lookup,float)[v];D=np.asarray(nuisance_design,float)[v];sh=np.sqrt(H[v]); yy=f*sh
    C=D*sh[:,None]
    if C.size==0:return 0.
    coef=np.linalg.lstsq(C,yy,rcond=None)[0];fit=C@coef;den=float(yy@yy)
    return float((fit@fit)/den) if den>1e-30 else 0.

def fit_joint_quotient_lookups_from_gram(
    bases: Sequence[StateQuotientBasis],
    K,
    t,
    l2=20.0,
    lr=0.5,
):
    """Fit selected quotient lookups from an already-exact union Gram."""
    if not bases:
        return []
    dims = [int(b.df) for b in bases]
    total = int(sum(dims))
    K = np.asarray(K, float)
    t = np.asarray(t, float).reshape(-1)
    if K.shape != (total, total) or t.shape != (total,):
        raise ValueError("selected Gram/target shape mismatch")
    offs = np.cumsum([0] + dims)
    P = np.zeros_like(K)
    for i, b in enumerate(bases):
        sl = slice(offs[i], offs[i + 1])
        P[sl, sl] = b.A.T @ b.A
    M = (K + K.T) * 0.5 + float(l2) * P
    # l2>0 and each A full-column-rank makes M positive definite unless no df.
    try:
        beta = np.linalg.solve(M, t)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(M, hermitian=True, rcond=1e-12) @ t
    out = []
    for i, b in enumerate(bases):
        sl = slice(offs[i], offs[i + 1])
        out.append(float(lr) * (b.A @ beta[sl]))
    return out


def fit_joint_quotient_lookups(bases:Sequence[StateQuotientBasis], h, l2=20., lr=.5):
    """Order-invariant regularized Newton fit in the union of selected quotient spaces."""
    if not bases:return []
    K,t=gram_and_target(bases,h)
    return fit_joint_quotient_lookups_from_gram(
        bases,
        K,
        t,
        l2=l2,
        lr=lr,
    )


def apply_lookup(score,basis:StateQuotientBasis,lookup):
    return np.asarray(score,float)+np.asarray(lookup,float)[basis.codes]


def apply_joint(score,bases,lookups,codes_override=None):
    out=np.asarray(score,float).copy()
    if codes_override is None:
        for b,lu in zip(bases,lookups): out += np.asarray(lu,float)[b.codes]
    else:
        for b,lu,co in zip(bases,lookups,codes_override): out += np.asarray(lu,float)[np.asarray(co,int)]
    return out
