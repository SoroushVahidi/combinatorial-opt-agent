"""Write a new correction overlay; never alter historical crosswalks or claims."""
from pathlib import Path
import hashlib,json,re,sys
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;S=R/'manuscript/jdsa-stage3';build=Path(sys.argv[1]);locations={};files={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def visit(name,doc):
 p=S/(name+'.tex');s=p.read_text();files[p]=doc
 for m in re.finditer(r'\\label\{([^}]+)\}',s):locations[m[1]]={'document':doc,'source':str(p.relative_to(R)),'line':s[:m.start()].count('\n')+1,'source_sha256':sha(p)}
 for n in re.findall(r'\\input\{([^}]+)\}',s):visit(n,doc)
for doc in ['main','supplement']:
 visit(doc,doc)
 for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}',(build/(doc+'.aux')).read_text()):
  if m[1] in locations:locations[m[1]].update(number=m[2],pdf_page=m[3])
oldpath=R/'docs/jdsa_manuscript_v1_20261029/CLAIM_LOCATION_CROSSWALK_V1.json';old=json.loads(oldpath.read_text());mapped={}
for kind in ['named_objects','numerical_passages','investigations']:
 mapped[kind]=[]
 for r in old[kind]:
  labels=[x['label'] for x in r['revised_locations']];assert all(x in locations for x in labels)
  mapped[kind].append({'claim_id':r['original_claim_id'],'location_labels':labels})
assert [len(mapped[k]) for k in ['named_objects','numerical_passages','investigations']]==[37,183,47]
citations=[]
for p,doc in files.items():
 s=p.read_text()
 for m in re.finditer(r'\\cite\w*\{([^}]+)\}',s):citations.append({'source':str(p.relative_to(R)),'line':s[:m.start()].count('\n')+1,'keys':m[1].split(','),'document':doc})
notes=[]
for needle,description in [('Oracle~A selects','C01 / E35: A/C configuration attribution, no changed observation'),('Hit@5 averages','C02: historical P@5 is Hit@5'),('Variant & Method & Hit@5','C02 / tab:ranked-retrieval: display heading only')]:
 p=S/'additional_evidence.tex';s=p.read_text();i=s.index(needle);notes.append({'source':str(p.relative_to(R)),'line':s[:i].count('\n')+1,'description':description})
report={'overlay_version':'jdsa-attribution-correction-v1','baseline_commit':'00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc','prior_crosswalk':str(oldpath.relative_to(R)),'prior_crosswalk_sha256':sha(oldpath),'immutable_claim_manifest':'reproducibility/jdsa-v2.0.0/claims.json','immutable_claim_manifest_sha256':sha(R/'reproducibility/jdsa-v2.0.0/claims.json'),'scope':'Current locations for prior mapped IDs, not a new evidence manifest or assertion of textual equivalence. Evidence limitations/classifications remain unchanged. E35 and the historical P@5 display name are qualified by CORRECTION_NOTE_V1.md.','locations':locations,'mapped_claims':mapped,'correction_passages':notes,'citations':citations}
(D/'CLAIM_LOCATION_OVERLAY_V1.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Correction v1: location overlay','','The JSON companion maps all 37 original numbered objects, 183 numerical passages and 47 investigation IDs through current source labels and compiled PDF pages. It also records current citation locations. Historical crosswalks and the immutable evidence manifest are unchanged.','','| Correction | Source location | PDF destination |','|---|---|---|','| Oracle A/C (E35) | `additional_evidence.tex:73` | Supplement S9.2, p.29 |','| Hit@5 definition and heading | `additional_evidence.tex:54,59` | Supplement S9.2, pp.28–29, Table S25 |','| Contributions and positioning | `main.tex`, introduction / Related Work / Discussion | Main pp.2–3,11 |','','This overlay updates locations, not evidence strength. See CORRECTION_NOTE_V1.md and EXACT_SOURCE_CHANGES.json for substantive corrections.']
(D/'CLAIM_LOCATION_OVERLAY_V1.md').write_text('\n'.join(lines)+'\n')
print('Mapped', {k:len(v) for k,v in mapped.items()},'citations',len(citations))
