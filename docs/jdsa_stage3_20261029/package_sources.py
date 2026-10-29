#!/usr/bin/env python3
"""Create a flat, deterministic journal source ZIP from verified build outputs."""
from pathlib import Path
import argparse,zipfile,hashlib,json
ROOT=Path(__file__).resolve().parents[2];N=ROOT/'manuscript/jdsa-stage3';D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);a=p.parse_args()
files={x.name:x.read_bytes() for x in N.iterdir() if x.suffix in ['.tex','.bib','.cls','.bst','.sty','.pdf'] or x.name in ['build.sh','SUBMISSION_README.txt']}
for name in ['main.aux','supplement.aux','main.bbl','supplement.bbl']:files[name]=(a.build_dir/name).read_bytes()
files['MANIFEST.sha256']=''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name,data in sorted(files.items())).encode()
out=N/'JDSA_Submission_Source_20261029.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for name,data in sorted(files.items()):
  assert '/' not in name
  i=zipfile.ZipInfo(name,(2026,10,29,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=(0o100755 if name=='build.sh' else 0o100644)<<16;z.writestr(i,data)
report={'status':'PASS','archive_path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'flat_layout':True,'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in sorted(files.items())]}
(D/'SUBMISSION_PACKAGE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n');print('PACKAGED',len(files),'FILES',report['bytes'],'BYTES')
