"""Device-executed exact binary32 dot audit of archived real-LM captures."""
import argparse,csv,hashlib,io,json,math,os,platform,sys,traceback,zipfile
from fractions import Fraction
from decimal import Decimal,localcontext
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision','highest')
MIN_EXP=-320
N_BINS=80

@jax.jit
def exact_device_bins(x,d):
 """No floating multiplication: base-256 limbs, signed int32 scatter accumulation."""
 xb=jax.lax.bitcast_convert_type(x.reshape(-1),jnp.uint32)
 db=jax.lax.bitcast_convert_type(d.reshape(-1),jnp.uint32)
 ex=((xb>>23)&255).astype(jnp.int32);ed=((db>>23)&255).astype(jnp.int32)
 mx=(xb&0x7fffff)|jnp.where(ex==0,jnp.uint32(0),jnp.uint32(1<<23))
 md=(db&0x7fffff)|jnp.where(ed==0,jnp.uint32(0),jnp.uint32(1<<23))
 ex=jnp.where(ex==0,-149,ex-150);ed=jnp.where(ed==0,-149,ed-150)
 sign=1-2*((xb^db)>>31).astype(jnp.int32)
 result=jnp.zeros(N_BINS,jnp.int32)
 for i in range(3):
  for j in range(3):
   a=(mx>>(8*i))&255;b=(md>>(8*j))&255
   shift=ex+ed-MIN_EXP+8*(i+j);index=shift//8
   # Each uint32 product is at most 255^2; shifting by <=7 stays below 2^23.
   p=(a*b)<<(shift%8).astype(jnp.uint32)
   for k in range(3):
    byte=((p>>(8*k))&255).astype(jnp.int32)*sign
    result=result.at[index+k].add(byte)
 return result

def decode_device_bins(bins):
 integer=sum(int(v)<<(8*i) for i,v in enumerate(np.asarray(bins)))
 return Fraction(integer,1<<(-MIN_EXP))

def fraction_oracle(x,d):
 return sum((Fraction.from_float(float(a))*Fraction.from_float(float(b)) for a,b in zip(x.ravel(),d.ravel())),Fraction())

def decimal(f):
 with localcontext() as ctx:
  ctx.prec=100;return str(Decimal(f.numerator)/Decimal(f.denominator))

def check_device(x,d,allow_cpu=False):
 assert x.shape==d.shape and x.dtype==d.dtype==np.float32
 assert np.isfinite(x).all() and np.isfinite(d).all()
 # Conservative worst case counts all 27 byte contributions in one bin.
 assert x.size*27*255<2**31,'int32 overflow bound exceeded'
 xx=jax.device_put(x);dd=jax.device_put(d)
 for original,device in ((x,xx),(d,dd)):
  bits=jax.lax.bitcast_convert_type(device,jnp.uint32)
  assert np.array_equal(np.asarray(bits),original.view(np.uint32)),'Input bits changed on device transfer'
 bins=exact_device_bins(xx,dd);bins.block_until_ready()
 if not allow_cpu:assert all(device.platform=='tpu' for device in bins.devices())
 exact=decode_device_bins(bins);ref=fraction_oracle(x,d)
 assert exact==ref,'Exact device sum differs from independent Fraction oracle'
 return bins,exact

@jax.jit
def local_conv_gradient(u,d):
 """Isolated selected-channel convolution, not a fresh full-LM backward."""
 def loss(w):
  out=jax.lax.conv_general_dilated(u[...,None],w[:,None,None],(1,),[(3,0)],
   dimension_numbers=('NHC','HIO','NHC'),precision=jax.lax.Precision.HIGHEST)
  return jnp.sum(out[...,0]*d)
 return jax.grad(loss)(jnp.zeros(4,jnp.float32))

def dump(out,report):
 p=out/'confirmation.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(p)

def run(a,r):
 backend=jax.default_backend()
 if not a.allow_cpu:assert backend=='tpu',f'TPU required, got {backend}'
 r.update(backend=backend,devices=[str(d) for d in jax.devices()],jax=jax.__version__,numpy=np.__version__,
  python=sys.version,platform=platform.platform(),exact_algorithm='IEEE binary32 bit decomposition; uint32 byte products; signed int32 base-256 scatter bins; host decoding only',
  safety={'minimum_exponent':MIN_EXP,'bins':N_BINS,'max_terms':16384,'conservative_int32_bin_bound':16384*27*255,'limit_exclusive':2**31},
  scope='Fresh device arithmetic and isolated convolution replay on frozen full-LM captures. No new full-LM forward/backward or speed benchmark.')
 dump(a.out,r)
 tiny=np.nextafter(np.float32(0),np.float32(1));big=np.finfo(np.float32).max
 controls=[('zero',np.zeros(16,np.float32),np.ones(16,np.float32)),
  ('cancellation_to_one',np.array([2**25,1,-2**25],np.float32),np.ones(3,np.float32)),
  ('subnormal_product',np.array([tiny],np.float32),np.array([tiny],np.float32)),
  ('overflowing_float32_products_cancel',np.array([big,big,1],np.float32),np.array([big,-big,1],np.float32))]
 rng=np.random.default_rng(20261017)
 for i in range(4):
  bits=rng.integers(0,2**32,size=(2,256),dtype=np.uint32)
  bits=np.where(((bits>>23)&255)==255,bits^np.uint32(1<<23),bits).astype(np.uint32)
  controls.append((f'finite_binary32_bits_{i}',bits[0].view(np.float32),bits[1].view(np.float32)))
 for label,x,d in controls:
  _,v=check_device(x,d,a.allow_cpu);r['controls'].append({'case':label,'exact_equal':True,'exact_value':decimal(v)})
  print('EXACT_CONTROL_PASS',label,flush=True);dump(a.out,r)
 expected_hash='dd6854993c17d8f8a7e3fffdd3804d4c1481636c84768bb62ec2641090b615b6'
 assert hashlib.sha256(a.capture.read_bytes()).hexdigest()==expected_hash
 data={};rows={}
 with zipfile.ZipFile(a.capture) as z:
  manifest=json.loads(z.read('manifest.json'))['files']
  assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
  for path in ('OLD','JAX_REF','NEW'):
   record=json.loads(z.read(f'results/{path}-probe.json'));assert record['status']=='completed'
   for row in record['seeds']:
    key=path,row['seed'];rows[key]=row
    blob=z.read('results/'+row['arrays']);assert hashlib.sha256(blob).hexdigest()==row['arrays_sha256']
    with np.load(io.BytesIO(blob),allow_pickle=False) as n:data[key]={k:n[k].copy() for k in n.files}
 r['capture_archive_sha256']=expected_hash
 saved_bins={};exact={};csvrows=[]
 for seed in (0,1,2):
  for x_path in ('OLD','JAX_REF','NEW'):
   for d_path in ('OLD','JAX_REF','NEW'):
    x=data[x_path,seed]['aligned_input'];d=data[d_path,seed]['cotangent']
    bins,value=check_device(x,d,a.allow_cpu);key=seed,x_path,d_path;saved_bins[key]=bins;exact[key]=value
    row={'seed':seed,'input_path':x_path,'cotangent_path':d_path,'exact_equal':True,'exact_contraction':decimal(value)}
    r['cross_contractions'].append(row);csvrows.append(row)
    if x_path==d_path:
     permutation=rng.permutation(x.size)
     shuffled,other=check_device(x.ravel()[permutation],d.ravel()[permutation],a.allow_cpu)
     assert np.array_equal(np.asarray(shuffled),np.asarray(bins)) and other==value
     u=data[x_path,seed]['conv_input'];got=local_conv_gradient(jax.device_put(u),jax.device_put(d));got.block_until_ready()
     assert np.isfinite(np.asarray(got)).all()
     r['local_float_replays'].append({'seed':seed,'path':x_path,'archived_full_LM_gradient':rows[x_path,seed]['raw_coordinate_gradient'],
      'isolated_TPU_conv_gradient':float(got[2]),'exact_frozen_gradient':decimal(value),
      'isolated_gradient_absolute_error':float(abs(Fraction.from_float(float(got[2]))-value)),
      'order_permutation_exact_equal':True,'note':'Different compiled graph from full LM; floating results need not be bitwise identical.'})
    print('CAPTURE_EXACT_PASS',key,flush=True);dump(a.out,r)
  for aa,bb in [('OLD','NEW'),('OLD','JAX_REF'),('JAX_REF','NEW')]:
   A=saved_bins[seed,aa,aa];BA=saved_bins[seed,bb,aa];B=saved_bins[seed,bb,bb]
   # Bin subtractions and equality are executed on the selected device; bound stays <2^31.
   dx=jax.jit(lambda a,b:a-b)(A,BA);dd=jax.jit(lambda a,b:a-b)(BA,B)
   total=jax.jit(lambda a,b:a-b)(A,B)
   assert bool(jax.jit(lambda a,b,c:jnp.all(a+b==c))(dx,dd,total))
   assert decode_device_bins(total)==exact[seed,aa,aa]-exact[seed,bb,bb]
   r['decomposition'].append({'seed':seed,'path_a':aa,'path_b':bb,'exact_device_identity':True,
    'input_term':decimal(decode_device_bins(dx)),'cotangent_term':decimal(decode_device_bins(dd)),
    'total':decimal(decode_device_bins(total)),'warning':'Algebraic split depends on chosen ordering; not unique causal percentages.'})
 # Confirm the previously hand-calculated retained-difference percentage on TPU FP32.
 ga=rows['OLD',0]['raw_coordinate_gradient'];gb=rows['NEW',0]['raw_coordinate_gradient']
 va=exact[0,'OLD','OLD'];vb=exact[0,'NEW','NEW']
 retained=float(jax.jit(lambda a,b,c,d:100*(a-b)/(c-d))(*[jnp.asarray(v,jnp.float32) for v in (float(va),float(vb),ga,gb)]))
 host=100*float(va-vb)/(ga-gb)
 assert abs(retained-host)<1e-3
 r['retained_difference_percent']={'device_FP32':retained,'host_high_precision_reference':host,
  'absolute_tolerance_percentage_points':.001,'passed':True,'scope':'This archived seed-0 OLD/NEW probe pair only.'}
 r['status']='passed';r['all_exact_checks_passed']=True
 r['new_full_model_execution']=False;r['kernel_universal_correctness_proven']=False
 r['interpretation']='Confirms the exact local calculations for the archived tensors using fresh device integer operations. Does not establish the exact unrounded full-LM gradient or the exclusive upstream cause.'
 for name,items in [('exact-cross-contractions.csv',csvrows),('local-float-replays.csv',r['local_float_replays']),('decomposition.csv',r['decomposition'])]:
  with (a.out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
 np.savez_compressed(a.out/'exact-device-bins.npz',**{f's{s}_{a}_{b}':np.asarray(v) for (s,a,b),v in saved_bins.items()})
 dump(a.out,r);print('ALL_EXACT_DEVICE_CHECKS_PASSED',backend,'retained_percent',retained,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-cpu',action='store_true');a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
 r={'status':'running','controls':[],'cross_contractions':[],'decomposition':[],'local_float_replays':[]}
 try:run(a,r)
 except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());dump(a.out,r);print(r['traceback'],flush=True);raise
