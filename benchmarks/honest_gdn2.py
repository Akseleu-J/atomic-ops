"""Equal-work GDN2 benchmark; every published ratio requires numerical gates.

No historical target speedup is assumed. TPU measurements are not CPU interpretation.
All wrappers use BLHD inputs, FP32 storage, common nonzero h0 and output scale.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, importlib.metadata, json, math, os, platform
import statistics, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from atomic_gdn2 import make_cfg, make_blr_trainable
from atomic_ops.reference import gdn2_token_serial_reference, gdn2_chunked_wy_reference

PREC=jax.lax.Precision.HIGHEST
ARGNAMES=('q','k','v','w','b','g','h0')
TOL=.005  # fixed before execution; never relaxed after seeing a result

def associative(q,k,v,w,b,g,h0,scale):
    """Pure affine composition; form matrices inside each chunk, not for all L."""
    B,L,H,D=q.shape; C=128; N=L//C
    def chunk(x):return jnp.moveaxis(x.reshape(B,N,C,H,D),1,0)
    def combine(left,right):
        ml,cl=left;mr,cr=right
        return jnp.matmul(mr,ml,precision=PREC),jnp.matmul(mr,cl,precision=PREC)+cr
    def step(h,xs):
        qq,kk,vv,ww,bb,gg=xs
        alpha=jnp.exp(gg)
        m=jnp.eye(D)*alpha[...,None,:]-kk[..., :,None]*((bb*kk)*alpha)[...,None,:]
        c=kk[..., :,None]*(ww*vv)[...,None,:]
        pm,pc=jax.lax.associative_scan(combine,(m,c),axis=1)
        states=jnp.matmul(pm,h[:,None],precision=PREC)+pc
        out=scale*jnp.einsum('bchdv,bchd->bchv',states,qq,precision=PREC)
        return states[:,-1],out
    hf,out=jax.lax.scan(jax.checkpoint(step),h0,tuple(map(chunk,(q,k,v,w,b,g))))
    return jnp.moveaxis(out,0,1).reshape(B,L,H,D),hf

def make_paths(D,profile):
    scale=1/math.sqrt(D)
    strict=profile=='strict'
    cfg=make_cfg(bt=128,bc=64,score_bs=128,bs2=64,mb=8,wy_eps=0.0 if strict else .001,
        dot_mode='highest',solve_dot_mode='highest' if strict else 'bf16x3',
        b3_dot_mode='highest' if strict else 'bf16x3',interpret=False)
    kernel=make_blr_trainable(cfg,scale)
    def pallas(q,k,v,w,b,g,h0):
        o,h=kernel(*(jnp.swapaxes(x,1,2) for x in (q,k,v,w,b,g)),h0)
        return jnp.swapaxes(o,1,2),h
    def scan(*xs):return associative(*xs,scale)
    def serial(q,k,v,w,b,g,h0):return gdn2_token_serial_reference(q,k,v,g,b,w,scale,h0)
    def wy(q,k,v,w,b,g,h0):
        return gdn2_chunked_wy_reference(q,k,v,g,b,w,scale,128,h0,wy_eps=0.0)
    return dict(associative_scan=scan,jax_wy=wy,pallas=pallas,serial=serial),dataclasses.asdict(cfg)

def inputs(shape,seed):
    B,L,H,D=shape;r=np.random.default_rng(seed)
    mk=lambda scale:r.normal(size=shape).astype(np.float32)*scale
    q,k,v=mk(.3),mk(.3),mk(.3)
    q=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+1e-6)
    k=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+1e-6)
    w=r.uniform(.2,1,shape).astype(np.float32)
    b=r.uniform(.2,1,shape).astype(np.float32)
    g=-np.abs(mk(.05))
    h0=r.normal(size=(B,H,D,D)).astype(np.float32)*.1
    arrays=(q,k,v,w,b,g,h0)
    digest=hashlib.sha256(b''.join(x.tobytes() for x in arrays)).hexdigest()
    return tuple(jax.device_put(x) for x in arrays),digest

def loss_for(f):
    def loss(*xs):
        o,h=f(*xs)
        return jnp.sum(o*o)+jnp.sum(h*h)
    return loss

def compare(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    finite=bool(np.isfinite(aa).all() and np.isfinite(bb).all())
    if not finite:return dict(finite=False,rel_l2=None,max_abs=None,reference_norm=None)
    diff=aa-bb
    return dict(finite=finite,rel_l2=float(np.linalg.norm(diff)/(np.linalg.norm(bb)+1e-12)),
                max_abs=float(np.max(np.abs(diff))),reference_norm=float(np.linalg.norm(bb)))

def tree_comparison(a,b,names):
    result={name:compare(x,y) for name,x,y in zip(names,jax.tree.leaves(a),jax.tree.leaves(b))}
    passed=all(v['finite'] and v['rel_l2']<=TOL for v in result.values())
    return dict(passed=passed,tensors=result)

def compile_fn(f,args):
    start=time.perf_counter(); exe=jax.jit(f).lower(*args).compile()
    duration=time.perf_counter()-start
    jax.block_until_ready(exe(*args))
    return exe,duration

def timings(exes,args,repeats,warmup,seed):
    for f in exes.values():
        for _ in range(warmup):jax.block_until_ready(f(*args))
    raw={name:[] for name in exes};order=[];rng=np.random.default_rng(seed)
    jax.block_until_ready(args)
    for i in range(repeats):
        names=list(exes);rng.shuffle(names);order.append(names)
        for name in names:
            begin=time.perf_counter();out=exes[name](*args);jax.block_until_ready(out)
            raw[name].append((time.perf_counter()-begin)*1000)
    return dict(samples={name:dict(samples_ms=s,median_ms=statistics.median(s),
        mean_ms=statistics.mean(s),sample_sd_ms=statistics.stdev(s) if len(s)>1 else 0.,
        min_ms=min(s),max_ms=max(s)) for name,s in raw.items()},order=order)

def environment():
    versions={}
    for p in ('jax','jaxlib','libtpu','numpy'):
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]=None
    return dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
        packages=versions,backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],
        device_kinds=[x.device_kind for x in jax.devices()],local_devices=jax.local_device_count(),
        matmul_precision=str(jax.config.jax_default_matmul_precision),
        environment={k:os.environ.get(k) for k in ('XLA_FLAGS','JAX_PLATFORMS','TPU_NAME')})

def run_micro(a):
    shape=tuple(map(int,a.shape.split(',')));D=shape[-1]
    assert len(shape)==4 and shape[1]%128==0
    paths,cfg=make_paths(D,a.profile)
    report=dict(kind='kernel',shape=shape,profile=a.profile,config=cfg,
        objective='sum(o**2)+sum(h_final**2)',argnums=list(ARGNAMES),storage_dtype='float32',
        output_scale=1/math.sqrt(D),initial_state='nonzero, identical for every path',
        tolerance_rel_l2=TOL,environment=environment(),repeats=a.repeats,warmup=a.warmup,
        compile_excluded=True,transfer_excluded=True,seeds=[],ratios=[])
    small,_=inputs((1,128,1,D),12345)
    serial,serial_sec=compile_fn(paths['serial'],small)
    serial_grad,_=compile_fn(jax.value_and_grad(loss_for(paths['serial']),argnums=tuple(range(7))),small)
    oracle=serial(*small);oracle_grad=serial_grad(*small)
    report['small_oracle']={}
    for name in ('associative_scan','jax_wy','pallas'):
        f,_=compile_fn(paths[name],small)
        gf,_=compile_fn(jax.value_and_grad(loss_for(paths[name]),argnums=tuple(range(7))),small)
        forward=tree_comparison(f(*small),oracle,('o','h_final'))
        grad=tree_comparison(gf(*small),oracle_grad,('loss',)+ARGNAMES)
        report['small_oracle'][name]=dict(forward=forward,gradient=grad)
    if not all(r['forward']['passed'] and r['gradient']['passed'] for r in report['small_oracle'].values()):
        report['status']='oracle_gate_failed';return report
    x,_=inputs(shape,a.seeds[0]);exes={};compile_s={};fwds={}
    for name in ('associative_scan','jax_wy','pallas'):
        print('COMPILE',name,shape,a.profile,flush=True)
        fwds[name],_=compile_fn(paths[name],x)
        exes[name],compile_s[name]=compile_fn(jax.value_and_grad(loss_for(paths[name]),argnums=tuple(range(7))),x)
    report['compile_seconds']=compile_s
    for seed in a.seeds:
        xs,digest=inputs(shape,seed)
        ref=fwds['associative_scan'](*xs);ref_grad=exes['associative_scan'](*xs)
        checks={}
        for name in ('jax_wy','pallas'):
            checks[name]=dict(forward=tree_comparison(fwds[name](*xs),ref,('o','h_final')),
                gradient=tree_comparison(exes[name](*xs),ref_grad,('loss',)+ARGNAMES))
        row=dict(seed=seed,input_sha256=digest,checks=checks)
        row['passed']=all(v['forward']['passed'] and v['gradient']['passed'] for v in checks.values())
        if row['passed']:
            row['timing']=timings(exes,xs,a.repeats,a.warmup,seed)
            times=row['timing']['samples'];row['speedups']={
                name:times[name]['median_ms']/times['pallas']['median_ms'] for name in ('associative_scan','jax_wy')}
        report['seeds'].append(row)
        print('SEED_RESULT',json.dumps(row),flush=True)
    report['status']='passed' if all(r['passed'] for r in report['seeds']) else 'gate_failed'
    if report['status']=='passed':
        report['speedups']={name:dict(per_seed=[r['speedups'][name] for r in report['seeds']],
            mean=statistics.mean(r['speedups'][name] for r in report['seeds']),
            sample_sd=statistics.stdev(r['speedups'][name] for r in report['seeds']) if len(a.seeds)>1 else 0.)
            for name in ('associative_scan','jax_wy')}
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shape',default='4,2048,6,128');p.add_argument('--profile',choices=('strict','production'),default='strict')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--repeats',type=int,default=15)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--output',default='honest-benchmark.json')
    p.add_argument('--allow-cpu-oracle',action='store_true')
    a=p.parse_args()
    assert a.repeats>=3 and a.warmup>=1
    jax.config.update('jax_default_matmul_precision','highest')
    if a.allow_cpu_oracle:
        paths,_=make_paths(8,'strict');xs,_=inputs((1,128,1,8),0)
        # CPU checks only pure JAX wrappers, never report Pallas interpretation speed.
        for key in ('associative_scan','jax_wy'):
            check=tree_comparison(jax.jit(paths[key])(*xs),jax.jit(paths['serial'])(*xs),('o','hf'))
            assert check['passed'],check
        print('CPU reference oracle PASS; no TPU speed measured');return
    assert jax.default_backend()=='tpu' and jax.local_device_count()==1,'Requires exactly one TPU device'
    try:report=run_micro(a)
    except Exception as error:
        report=dict(status='error',error=type(error).__name__,detail=str(error),traceback=traceback.format_exc(),environment=environment())
    report['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['kernel_source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (ROOT/'atomic_gdn2').rglob('*.py')}
    report['source_revision']='11616acd8a061768a38feb12c58e5e329952f5d5'
    Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False))
    print('FINAL',json.dumps(report.get('speedups',{})),report['status'],flush=True)
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
