"""Document-only preservation and independent package verification; no experiments."""
from pathlib import Path
import difflib, hashlib, json, re, subprocess, tempfile, zipfile
import pymupdf as fitz
R=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
O=R/'manuscript/jdsa-phase4'; N=R/'manuscript/jdsa-final-review'
changes=json.loads((D/'EDITORIAL_CHANGES.json').read_text())
changed=[];diff=[];tables=0;equations=0
for f in sorted(O.rglob('*.tex')):
 rel=str(f.relative_to(O));a=f.read_text();expected=a;b=(N/rel).read_text()
 for c in changes:
  if c['file']==rel:
   assert expected.count(c['before'])==1
   expected=expected.replace(c['before'],c['after'])
 if rel in ['main.tex','supplement.tex']:
  expected=expected.replace('%% JDSA Phase 4 author-review source; no journal submission or author approval implied.','%% JDSA final editorial-polish source.').replace('JDSA_Phase4','JDSA_Final_Review')
 assert expected==b,rel
 if a!=b:
  changed.append(rel)
  diff.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='phase4/'+rel,tofile='final-review/'+rel))
 for env in ['tabular','tabularx','equation','algorithm','algorithmic']:
  pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
  x=re.findall(pat,a,re.S);assert x==re.findall(pat,b,re.S),(rel,env)
  if env in ['tabular','tabularx']:tables+=len(x)
  if env=='equation':equations+=len(x)
 for pat in [r'\$[^$]*\$',r'\\cite\w*\{[^}]+\}',r'\\(?:sub)*section\{[^}]+\}',r'\\label\{[^}]+\}']:
  assert re.findall(pat,a)==re.findall(pat,b),(rel,pat)
for name in ['references.bib','sn-jnl.cls','sn-basic.bst','orcidlink.sty','d1_contrasts.tex']:
 assert (O/name).read_bytes()==(N/name).read_bytes(),name
for f in (O/'figures').glob('*.pdf'):assert f.read_bytes()==(N/'figures'/f.name).read_bytes()
a=(O/'main.tex').read_text();b=(N/'main.tex').read_text()
assert a[a.index('\\abstract'):a.index('\\backmatter')]==b[b.index('\\abstract'):b.index('\\backmatter')]
for heading in ['Funding.','Competing interests.','Ethics approval and consent to participate.','Author contributions.','Declaration of AI assistance in software development.']:
 pattern=r'\\noindent\\textbf\{'+re.escape(heading)+r'\}[^\n]+'
 assert re.search(pattern,a)[0]==re.search(pattern,b)[0]
inv=R/'docs/jdsa_revision_phase4_20261029/INVESTIGATION_DESTINATIONS.json'
assert len(json.loads(inv.read_text()))==47
assert inv.read_bytes()==subprocess.check_output(['git','show','cea99e86d33ad83219b9bf711c310e929ae18409:'+str(inv.relative_to(R))])
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
preservation={'source_commit':'cea99e86d33ad83219b9bf711c310e929ae18409','exact_allowlisted_prose_changes':len(changes),'changed_tex_files':changed,'all_other_tex_bytes_preserved':True,'main_abstract_through_conclusion_source_byte_identical':True,'table_bodies_preserved':tables,'numbered_equations_preserved':equations,'algorithms_math_citations_headings_labels_identical':True,'table_caption_exception':'Table 9 final sentence: editorial timing replaced by archived-response provenance','investigation_destinations_preserved':47,'d1_source_byte_identical':True,'author_information_preserved':True,'bibliography_sha256':sha(N/'references.bib'),'bibliography_byte_identical':True,'figure_pdf_bytes_identical':True,'class_style_bytes_identical':True}
(D/'PRESERVATION_CHECKS.json').write_text(json.dumps(preservation,indent=2)+'\n')
(D/'EXACT_SOURCE_DIFF.patch').write_text(''.join(diff))
qa=Path(tempfile.mkdtemp(prefix='jdsa-final-package-qa-'))
with zipfile.ZipFile(next(N.glob('*.zip'))) as z:
 assert len(z.namelist())==10 and all('/' not in x for x in z.namelist())
 z.extractall(qa)
with (qa/'independent-build.log').open('w') as log:
 p=subprocess.run(['sh','build.sh'],cwd=qa,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert p.returncode==0,qa
report={}
for kind,f in [('main',next(N.glob('Retrieval*.pdf'))),('supplement',next(N.glob('ESM*.pdf')))]:
 a=fitz.open(f);b=fitz.open(qa/f.name);old=fitz.open(O/f.name.replace('Final_Review','Phase4'))
 assert len(a)==len(b)==len(old)
 changed_pages=[];bounds=[];links=0
 for i,(page,rebuilt) in enumerate(zip(a,b)):
  assert page.get_text()==rebuilt.get_text()
  pix=page.get_pixmap();assert pix.samples==rebuilt.get_pixmap().samples
  if pix.samples!=old[i].get_pixmap().samples:changed_pages.append(i+1)
  for block in page.get_text('dict')['blocks']:
   for line in block.get('lines',[]):
    for span in line['spans']:
     x0,y0,x1,y1=span['bbox']
     if x0<0 or y0<0 or x1>page.rect.width or y1>page.rect.height:bounds.append({'page':i+1,'text':span['text']})
  for link in page.get_links():
   if link['kind'] in [fitz.LINK_GOTO,fitz.LINK_NAMED]:
    assert 0<=link['page']<len(a)
    if link.get('nameddest'):assert link['nameddest'] in a.resolve_names()
    links+=1
   if link['kind'] in [fitz.LINK_GOTOR,fitz.LINK_LAUNCH]:
    filename,_,fragment=link['file'].partition('#');target=N/Path(filename).name
    assert target.exists(),link
    targetdoc=fitz.open(target)
    if fragment.startswith('nameddest='):assert fragment[len('nameddest='):] in targetdoc.resolve_names(),link
    elif isinstance(link.get('page'),int):assert 0<=link['page']<len(targetdoc),link
    else:raise AssertionError(link)
    targetdoc.close();links+=1
 assert not bounds
 report[kind]={'file':f.name,'pages':len(a),'independent_build_text_and_72dpi_pixels_equal':True,'changed_rendered_pages_from_phase4':changed_pages,'unchanged_rendered_pages':len(a)-len(changed_pages),'outside_page_text':bounds,'valid_pdf_links':links}
log=Path('/tmp/jdsa-final-polish-modular-build.log').read_text();build=Path(re.search(r'Build directory: (.+)',log)[1]);issues={}
for kind in ['main','supplement']:
 s=(build/(kind+'.log')).read_text();bad=[x for x in s.splitlines() if re.search('undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull',x)];assert not bad
 issues[kind]={'fatal_or_unresolved_diagnostics':bad,'warnings':[x for x in s.splitlines() if 'Warning' in x]}
report['final_logs']=issues
report['inherited_warning']='Main hyperref bookmark-level adjustment also occurs in the Phase 4 independent-build log; no new warning introduced.'
report['artifact_sha256']={f.name:sha(f) for f in list(N.glob('*.pdf'))+list(N.glob('*.zip'))}
report['independent_zip_file_count']=10
(D/'BUILD_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'preservation':preservation,'build':report,'independent_build_directory':str(qa)},indent=2))
