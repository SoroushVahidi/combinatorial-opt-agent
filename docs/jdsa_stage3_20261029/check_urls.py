#!/usr/bin/env python3
"""Anonymous read-only checks of the hyperlinks in the final PDFs."""
import concurrent.futures,datetime,json,requests
from pathlib import Path
D=Path(__file__).resolve().parent
j=json.loads((D/'FINAL_DOCUMENT_CHECKS.json').read_text());urls=sorted({u for x in j['documents'].values() for u in x['uris']}|{'https://link.springer.com/journal/41060/submission-guidelines'})
def check(u):
 r={'url':u}
 if u.startswith('mailto:'):return {**r,'status':'NOT_PROBED_EMAIL','note':'Preserved author-provided contact; no email sent.'}
 try:
  a=requests.get(u,timeout=15,headers={'User-Agent':'JDSA-submission-link-audit/1.0'});r.update(http_status=a.status_code,final_url=a.url)
  r['status']='HTTP_SUCCESS_ONLY' if a.status_code==200 else 'NOT_INDEPENDENTLY_ACCESSIBLE'
  if a.status_code==200 and any(x in a.text.lower() for x in ['just a moment','enable javascript','verify you are human']):r['status']='JAVASCRIPT_OR_ACCESS_CHALLENGE'
 except Exception as e:r.update(status='NETWORK_UNVERIFIED',error=str(e))
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p:r=list(p.map(check,urls))
(D/'URL_CHECKS.json').write_text(json.dumps({'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Anonymous GET; HTTP success alone does not verify bibliographic content','links':r},indent=2)+'\n')
print('Checked',len(r),'URLs')
for x in r:
 if x['status']!='HTTP_SUCCESS_ONLY':print(x['status'],x.get('http_status'),x['url'])
