#!/usr/bin/env python3
"""Final deterministic source/PDF checks; no models, solvers, or network access."""
import argparse,collections,decimal,hashlib,json,re,subprocess
from pathlib import Path
import pymupdf
B=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;S=B/'manuscript/jdsa-stage3';O=B/'manuscript/jdsa-stage2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def expanded(name):
 s=(S/(name+'.tex')).read_text();return re.sub(r'\\input\{([^}]+)\}',lambda m:expanded(m[1]),s)
def numbers(s):
 s=re.sub(r'(?m)^%.*','',s)
 if '\\begin{document}' in s:s=s.split('\\begin{document}',1)[1]
 # Front matter/layout is separately checked; retain the abstract explicitly.
 if '\\section{' in s and ('\\maketitle' in s):
  abstract=re.search(r'\\abstract\{(.*?)\}\n\n\\keywords',s,re.S)
  s=(abstract[1] if abstract else '')+s[s.index('\\section{'):]
 s=re.sub(r'\\(?:label|(?:eq)?ref|cite\w*|path|url)\{[^}]*\}','',s)
 s=re.sub(r'\\href\{[^}]*\}',r'',s)
 s=re.sub(r'Stage [23]\b','Stage VERSION',s)
 s=s.replace('{,}',',').replace('SHA-256','SHA256')
 s=re.sub(r'\\setlength\{\\bibsep\}\{[^}]+\}', '', s)
 return collections.Counter(str(decimal.Decimal(x.replace(',','')).normalize()) for x in re.findall(r'(?<![\w.])(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)',s))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build-dir',type=Path,required=True);a=ap.parse_args();out={'status':'PASS','build_dir':str(a.build_dir),'source_numeric_checks':[],'documents':{}}
 for p in O.glob('*.tex'):
  old=numbers(p.read_text());new=numbers((S/p.name).read_text())
  removed=dict(old-new);added=dict(new-old)
  # All24 -> All 24 is a spacing correction; old extraction excludes the attached number.
  if p.name=='grounding_diagnostics.tex' and added.get('24')==1:added.pop('24')
  ok=not removed and not added
  out['source_numeric_checks'].append({'file':p.name,'pass':ok,'removed':removed,'added':added})
 texts={d:expanded(d) for d in ['main','supplement']};alltext='\n'.join(texts.values());labels=set(re.findall(r'\\label\{((?:tab|fig|alg):[^}]+)\}',alltext));refs={re.sub('^[MS]-','',x) for x in re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',alltext)}
 out['uncited_numbered_objects']=sorted(labels-refs)
 out['placeholder_matches']=re.findall(r'\b(?:TODO|TBD|FIXME|XXX|PLACEHOLDER)\b',alltext)
 out['cited_keys']=sorted({x for m in re.findall(r'\\cite\w*\{([^}]+)\}',alltext) for x in m.split(',')})
 out['all_bibliography_entries']=re.findall(r'@\w+\{([^,]+)',(S/'references.bib').read_text())
 out['uncited_bib_entries']=sorted(set(out['all_bibliography_entries'])-set(out['cited_keys']))
 abstract=re.search(r'\\abstract\{(.*?)\}\n\n\\keywords',texts['main'],re.S)[1];out['abstract_words']=len(re.findall(r"[A-Za-z0-9]+(?:[-’'][A-Za-z0-9]+)*",re.sub(r'\\[A-Za-z]+|[{}$]',' ',abstract)))
 out['keyword_count']=len(re.search(r'\\keywords\{([^}]+)\}',texts['main'])[1].split(','))
 docs={f:pymupdf.open(S/f) for f in ['manuscript.pdf','ESM_1.pdf']};names={f:d.resolve_names() for f,d in docs.items()}
 for f,d in docs.items():
  doc='main' if f=='manuscript.pdf' else 'supplement';bad=[];remote=[];internal=0;uris=set();bbox=[];pages=[]
  for i,p in enumerate(d):
   words=p.get_text('words');rect=p.rect
   outside=[w[:5] for w in words if w[0]<-0.5 or w[1]<-0.5 or w[2]>rect.width+.5 or w[3]>rect.height+.5]
   bbox.extend({'page':i+1,'word':w} for w in outside)
   footer=[w[4] for w in words if w[1]>rect.height*.72 and w[4]==str(i+1) and abs((w[0]+w[2])/2-rect.width/2)<15]
   pages.append({'page':i+1,'page_number_found':bool(footer),'text_words':len(words),'size_pt':[rect.width,rect.height]})
   for l in p.get_links():
    if l['kind']==2:uris.add(l['uri'])
    elif l['kind']==1:
     internal+=1
     if not 0<=l.get('page',-1)<len(d):bad.append({'page':i+1,'link':str(l)})
    elif l['kind'] in [3,5]:
     target=l.get('file','');targetfile,_,dest=target.partition('#nameddest=');ok=targetfile in names and dest in names[targetfile]
     remote.append({'page':i+1,'target':target,'pass':ok})
  log=(a.build_dir/(doc+'.log')).read_text();blg=(a.build_dir/(doc+'.blg')).read_text()
  diagnostics=re.findall(r'^.*(?:undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull|^!).*$',log,re.M)
  bibwarnings=[line for line in blg.splitlines() if 'Warning--' in line or 'error' in line.lower()]
  fonttext=subprocess.check_output(['pdffonts',str(S/f)],text=True);(D/(doc+'-fonts.txt')).write_text(fonttext)
  unembedded=[l for l in fonttext.splitlines()[2:] if not re.search(r'\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',l)]
  contenttext='\n'.join(p.get_text() for p in d)
  out['documents'][doc]={'file':f,'sha256':sha(S/f),'bytes':(S/f).stat().st_size,'pages':len(d),'metadata':d.metadata,'catalog_language':d.xref_get_key(d.pdf_catalog(),'Lang'),'outline_entries':len(d.get_toc()),'page_checks':pages,'outside_media_box':bbox,'unembedded_fonts':unembedded,'final_latex_diagnostics':diagnostics,'bibtex_warnings':bibwarnings,'internal_link_count':internal,'bad_internal_links':bad,'cross_document_links':remote,'uri_count':len(uris),'uris':sorted(uris),'replacement_characters':contenttext.count('\ufffd'),'log_sha256':sha(a.build_dir/(doc+'.log'))}
  if diagnostics or bibwarnings or unembedded or bbox or bad or not all(x['pass'] for x in remote) or not all(x['page_number_found'] for x in pages):out['status']='FAIL'
 if not all(x['pass'] for x in out['source_numeric_checks']) or out['uncited_numbered_objects'] or out['placeholder_matches'] or not 150<=out['abstract_words']<=250 or not 4<=out['keyword_count']<=6:out['status']='FAIL'
 (D/'FINAL_DOCUMENT_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
 print(out['status'],'abstract',out['abstract_words'],'keywords',out['keyword_count'],'bibliography',len(out['all_bibliography_entries']),'cited',len(out['cited_keys']))
 for v in out['source_numeric_checks']:
  if not v['pass']:print('NUMERIC DIFFERENCE',v)
 for k,v in out['documents'].items():print(k,'pages',v['pages'],'remote links',len(v['cross_document_links']),'issues',{x:v[x] for x in ['outside_media_box','unembedded_fonts','final_latex_diagnostics','bibtex_warnings','bad_internal_links'] if v[x]})
 return out['status']!='PASS'
if __name__=='__main__':raise SystemExit(main())
