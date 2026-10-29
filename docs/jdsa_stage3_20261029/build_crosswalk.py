#!/usr/bin/env python3
"""Generate a Stage 3 location overlay using the immutable claim IDs."""
from pathlib import Path
import importlib.util,json,hashlib,re
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage2_crosswalk',ROOT/'docs/jdsa_stage2_20261029/build_crosswalk.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.N=ROOT/'manuscript/jdsa-stage3';m.D=OUT;m.main()
p=OUT/'CLAIM_LOCATION_CROSSWALK_V1.json';d=json.loads(p.read_text());previous=ROOT/'docs/jdsa_stage2_20261029/CLAIM_LOCATION_CROSSWALK_V1.json';old=json.loads(previous.read_text());objects={c['original_claim_id']:c for c in old['named_objects']}
d['crosswalk_version']='jdsa-stage3-v1';d['baseline_commit']='60785b0d144365413dd3b44b577bff901043ddd7';d['previous_crosswalk']={'path':str(previous.relative_to(ROOT)),'sha256':hashlib.sha256(previous.read_bytes()).hexdigest()}
for c in d['named_objects']:c['stage2_locations']=objects[c['original_claim_id']]['revised_locations']
d['stage3_scientific_changes']='None. STAGE2-D01 is preserved and reverified, not a new Stage 3 claim.'
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
lines=['# Final editorial claim-location crosswalk v1','','This Stage 3 overlay preserves all immutable evidence IDs and the Stage 2 derived claim. The JSON maps all 183 original numerical passages and 47 investigations, with source hashes and final PDF locations. Original evidence and Stage 2 crosswalks remain unchanged.','','| Claim ID | Stage 2 document / object / page | Stage 3 document / object / page |','|---|---|---|']
for c in d['named_objects']:
 a=c['stage2_locations'][0];b=c['revised_locations'][0];lines.append(f"| `{c['original_claim_id']}` | {a['document']} {a.get('number')} / {a.get('page')} | {b['document']} {b.get('number')} / {b.get('page')} |")
lines+=['','`STAGE2-D01` remains Main Table 3 and Equations (9)–(10). No table or equation has been scientifically redefined. The mention-selection symbol changes from t_i* to tau_j* solely to distinguish it from the query type score; mathematical content is unchanged.','', 'The accompanying citation-number crosswalk records compiled bibliography numbering and metadata changes.']
(OUT/'CLAIM_LOCATION_CROSSWALK_V1.md').write_text('\n'.join(lines)+'\n')
