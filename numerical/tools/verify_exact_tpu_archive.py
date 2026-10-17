"""Read actual device integer bins and verify with independent CPU integer arithmetic, stdlib only."""
import argparse,ast,hashlib,io,json,struct,zipfile
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def npy(blob):
 f=io.BytesIO(blob);assert f.read(6)==b'\x93NUMPY';major,minor=f.read(2)
 size=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
 h=ast.literal_eval(f.read(size).decode('latin1'));assert h['fortran_order'] is False
 return h,f.read()
def decode(bits):
 e=(bits>>23)&255;mant=bits&0x7fffff;assert e!=255
 if e:mant|=1<<23;e-=150
 else:e=-149
 return (-mant if bits>>31 else mant),e
def exact(a,b):
 terms=[]
 for (x,),(y,) in zip(struct.iter_unpack('<I',a),struct.iter_unpack('<I',b)):
  m,e=decode(x);n,d=decode(y)
  if m and n:terms.append((m*n,e+d))
 if not terms:return Fraction(0)
 emin=min(e for m,e in terms);total=sum(m<<(e-emin) for m,e in terms)
 return Fraction(total,1<<-emin) if emin<0 else Fraction(total<<emin)
def text(f):
 with localcontext() as ctx:ctx.prec=100;return str(Decimal(f.numerator)/Decimal(f.denominator))
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--archive',type=Path,required=True)
 parser.add_argument('--expected-source',type=Path,required=True)
 parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args();p=args.archive;out=args.out;out.mkdir(exist_ok=True,parents=True)
 with zipfile.ZipFile(p) as z:
  m=json.loads(z.read('manifest.json'))['files'];assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in m.items())
  assert z.read('tpu_exact_confirmation.py')==args.expected_source.read_bytes()
  r=json.loads(z.read('confirmation.json'));assert r['status']=='passed' and r['backend']=='tpu' and r['all_exact_checks_passed']
  assert len(r['controls'])==8 and all(v['exact_equal'] for v in r['controls'])
  assert len(r['local_float_replays'])==9 and all(v['order_permutation_exact_equal'] for v in r['local_float_replays'])
  original=z.read('original-full-LM-capture.zip');assert hashlib.sha256(original).hexdigest()=='dd6854993c17d8f8a7e3fffdd3804d4c1481636c84768bb62ec2641090b615b6'
  inputs={};cot={};values={}
  with zipfile.ZipFile(io.BytesIO(original)) as capture:
   for seed in (0,1,2):
    for path in ('OLD','JAX_REF','NEW'):
     with zipfile.ZipFile(io.BytesIO(capture.read(f'results/{path}-seed{seed}-coordinate.npz'))) as a:
      for name,store in [('aligned_input',inputs),('cotangent',cot)]:
       h,data=npy(a.read(name+'.npy'));assert h['shape']==(8,2048) and h['descr']=='<f4';store[seed,path]=data
  with zipfile.ZipFile(io.BytesIO(z.read('exact-device-bins.npz'))) as bins:
   assert len(bins.namelist())==27 and len(r['cross_contractions'])==27
   for row in r['cross_contractions']:
    seed=row['seed'];aa=row['input_path'];bb=row['cotangent_path'];name=f's{seed}_{aa}_{bb}.npy'
    h,data=npy(bins.read(name));assert h['shape']==(80,) and h['descr']=='<i4'
    limbs=struct.unpack('<80i',data);device=Fraction(sum(v<<(8*i) for i,v in enumerate(limbs)),1<<320)
    reference=exact(inputs[seed,aa],cot[seed,bb]);assert device==reference
    assert text(reference)==row['exact_contraction'];values[seed,aa,bb]=reference
  assert len(r['decomposition'])==9
  for row in r['decomposition']:
   seed=row['seed'];aa=row['path_a'];bb=row['path_b']
   a,b,c=values[seed,aa,aa],values[seed,bb,aa],values[seed,bb,bb]
   assert row['exact_device_identity'] and text(a-b)==row['input_term'] and text(b-c)==row['cotangent_term'] and text(a-c)==row['total']
  percentage=r['retained_difference_percent'];assert percentage['passed'] and abs(percentage['device_FP32']-percentage['host_high_precision_reference'])<.001
  for n in ('confirmation.json','execution-status.json','worker.log','exact-cross-contractions.csv','local-float-replays.csv','decomposition.csv'):(out/n).write_bytes(z.read(n))
  receipt={'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'manifest_files':len(m),'all_hashes_match':True,
   'executed_source_matches_prepared_notebook':True,'reported_backend':'tpu','completed_controls':8,
   'exact_device_bins_independently_verified':27,'exact_decompositions_independently_verified':9,'permutation_checks_reported_pass':9,
   'independent_verification':'stdlib-only IEEE binary32 decoding and unbounded integer multiplication/addition, checked against saved raw device bins',
   'retained_difference_percent':percentage,'status':'verified for the stated frozen-capture scope',
   'new_full_model_execution':False,'universal_kernel_correctness_proven':False}
  (out/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
