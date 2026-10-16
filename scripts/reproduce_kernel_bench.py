"""Parity-gated kernel benchmark. Historical speedups are not assumed."""
import argparse,json,math,subprocess,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from atomic_gdn2 import make_cfg,make_blr_trainable

def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',default='4,2048,6,128');p.add_argument('--output',default='benchmark.json');p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args();B,L,H,D=map(int,a.shape.split(','));assert D==128 and L%128==0
    assert jax.default_backend()=='tpu','Explicit TPU required; no accidental CPU benchmark'
    cfg=make_cfg(bt=128,bc=64,score_bs=128,bs2=64,mb=8,wy_eps=0.0,dot_mode='highest',solve_dot_mode='highest',b3_dot_mode='highest',interpret=False)
    scale=1/math.sqrt(D);fn=make_blr_trainable(cfg,scale)
    ks=jax.random.split(jax.random.PRNGKey(0),7)
    shape=(B,L,H,D)
    q,k,v=[jax.random.normal(ks[i],shape)*.3 for i in range(3)]
    q=q*jax.lax.rsqrt(jnp.sum(q*q,-1,keepdims=True)+1e-6)
    k=k*jax.lax.rsqrt(jnp.sum(k*k,-1,keepdims=True)+1e-6)
    w=jax.random.uniform(ks[3],shape,minval=.2,maxval=1)
    b=jax.random.uniform(ks[4],shape,minval=.2,maxval=1)
    g=-jnp.abs(jax.random.normal(ks[5],shape))*.05
    h0=jax.random.normal(ks[6],(B,H,D,D))*.1
    inputs=(q,k,v,w,b,g)
    def candidate(q,k,v,w,b,g,h0):
        hm=lambda x:jnp.swapaxes(x,1,2)
        o,h=fn(*map(hm,(q,k,v,w,b,g)),h0)
        return hm(o),h
    def reference(q,k,v,w,b,g,h0):
        alpha=jnp.exp(g)
        eye=jnp.eye(D,dtype=jnp.float32)
        M=eye*alpha[...,None,:]-k[..., :,None]*((b*k)*alpha)[...,None,:]
        C=k[..., :,None]*(w*v)[...,None,:]
        nch=L//128
        chunk=lambda x:jnp.moveaxis(x.reshape(B,nch,128,H,D,D),1,0)
        def combine(left,right):
            ml,cl=left;mr,cr=right
            return mr@ml,mr@cl+cr
        def step(h,x):
            m,c,qq=x
            prefix_m,prefix_c=jax.lax.associative_scan(combine,(m,c),axis=1)
            states=prefix_m@h[:,None]+prefix_c
            o=scale*jnp.einsum('blhdv,blhd->blhv',states,qq)
            return states[:,-1],o
        qs=jnp.moveaxis(q.reshape(B,nch,128,H,D),1,0)
        hf,o=jax.lax.scan(jax.checkpoint(step),h0,(chunk(M),chunk(C),qs))
        return jnp.moveaxis(o,0,1).reshape(B,L,H,D),hf
    def objective(f):
        def loss(*xs):
            o,h=f(*xs,h0)
            return jnp.sum(o.astype(jnp.float32)**2)+jnp.sum(h.astype(jnp.float32)**2)
        return loss
    rel=lambda x,y:float(np.linalg.norm(np.asarray(x,np.float64)-np.asarray(y,np.float64))/(np.linalg.norm(np.asarray(y,np.float64))+1e-12))
    data=dict(shape=[B,L,H,D],jax=jax.__version__,backend=jax.default_backend(),devices=jax.device_count(),repeats=a.repeats,warmup=1,
        config=vars(cfg),objective='sum(o^2)+sum(h_final^2)',initial_state='same nonzero h0',output_scale=scale,
        tolerances=dict(forward_rel_l2=.005,gradient_rel_l2=.005))
    try:
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip();data['source_commit']=sha
    except Exception:data['source_commit']='unavailable'
    try:
        r=jax.jit(reference)(*inputs,h0);c=jax.jit(candidate)(*inputs,h0);jax.block_until_ready((r,c))
        data['forward_rel_l2']=[rel(x,y) for x,y in zip(c,r)]
        gradf={name:jax.jit(jax.grad(objective(fn),argnums=tuple(range(6)))) for name,fn in [('scan',reference),('pallas',candidate)]}
        grads={name:f(*inputs) for name,f in gradf.items()};jax.block_until_ready(grads)
        data['gradient_rel_l2']=[rel(x,y) for x,y in zip(grads['pallas'],grads['scan'])]
        finite=all(np.isfinite(np.asarray(x)).all() for tree in (r,c,*grads.values()) for x in jax.tree.leaves(tree))
        data['parity_passed']=finite and max(data['forward_rel_l2'])<.005 and max(data['gradient_rel_l2'])<.005
        if data['parity_passed']:
            times={}
            for name,f in gradf.items():
                jax.block_until_ready(f(*inputs));samples=[]
                for _ in range(a.repeats):
                    t=time.perf_counter();jax.block_until_ready(f(*inputs));samples.append((time.perf_counter()-t)*1000)
                times[name]=dict(median_ms=float(np.median(samples)),samples_ms=samples)
            data['timings']=times;data['speedup']=times['scan']['median_ms']/times['pallas']['median_ms']
        else:data['timing_skipped']='parity gate failed'
    except Exception as e:data.update(parity_passed=False,error=type(e).__name__,detail=str(e)[:500])
    Path(a.output).write_text(json.dumps(data,indent=2,default=str));print(json.dumps(data,indent=2,default=str))
    return 0 if data.get('parity_passed') else 1
if __name__=='__main__':raise SystemExit(main())
