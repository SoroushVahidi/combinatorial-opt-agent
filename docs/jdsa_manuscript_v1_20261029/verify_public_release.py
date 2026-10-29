#!/usr/bin/env python3
"""Verify public release metadata, tag and downloads without authentication."""
import argparse, concurrent.futures, datetime, hashlib, json, subprocess
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
REPO='SoroushVahidi/combinatorial-opt-agent'
TAG='jdsa-manuscript-v1.0.0'
BASE='Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization'
URL='https://github.com/'+REPO+'/releases/tag/'+TAG
HF='https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0'
def get(url):
    r=requests.get(url,timeout=45)
    r.raise_for_status()
    return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--assets-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    r=get('https://api.github.com/repos/'+REPO+'/releases/tags/'+TAG);meta=r.json()
    assert meta['tag_name']==TAG and not meta['draft'] and meta['prerelease']
    remote=subprocess.check_output(['git','ls-remote','origin','refs/tags/'+TAG,'refs/tags/'+TAG+'^{}'],cwd=ROOT,text=True).splitlines()
    assert next(line.split()[0] for line in remote if line.endswith('^{}'))==a.commit
    names={BASE+'.pdf',BASE+'_Supplement.pdf',BASE+'_Source.zip','SHA256SUMS.txt'}
    assert {x['name'] for x in meta['assets']}==names
    def download(asset):
        rr=get(asset['browser_download_url']);local=(a.assets_dir/asset['name']).read_bytes();assert rr.content==local,asset['name']
        return {'asset':asset['name'],'url':asset['browser_download_url'],'status':rr.status_code,'bytes':len(rr.content),'sha256':hashlib.sha256(rr.content).hexdigest(),'local_bytes_identical':True,'api_digest':asset.get('digest')}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:downloads=list(pool.map(download,meta['assets']))
    for row in downloads:
        if row['api_digest']:assert row['api_digest']=='sha256:'+row['sha256']
    sums=(a.assets_dir/'SHA256SUMS.txt').read_text().splitlines()
    assert len(sums)==3
    for line in sums:
        digest,name=line.split('  ',1);assert next(x['sha256'] for x in downloads if x['asset']==name)==digest
    endpoints=[URL,HF,'https://github.com/'+REPO,'https://github.com/'+REPO+'/releases/tag/jdsa-evidence-v2.0.0']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:responses=list(pool.map(get,endpoints))
    out={'status':'PASS','verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authentication':'None; independent public HTTPS GETs','manuscript_commit':a.commit,'tag':TAG,'tag_refs':remote,'release_id':meta['id'],'release_url':URL,'draft':meta['draft'],'prerelease':meta['prerelease'],'assets':downloads,'sha256sums_verified':True,'public_destinations':[{'url':u,'http_status':rr.status_code,'final_url':rr.url} for u,rr in zip(endpoints,responses)]}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print('PASS: four public assets are byte-identical, three checksum entries match, tag dereferences to '+a.commit+'; all permanent destinations return HTTP 200.')
if __name__=='__main__':main()
