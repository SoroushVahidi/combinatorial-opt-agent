"""Document packaging verification only; no experiments, network calls or inference."""
from pathlib import Path
import difflib, hashlib, json, re, subprocess, tempfile, zipfile, os
import pymupdf as fitz
R=Path(__file__).resolve().parents[2]; D=Path(__file__).resolve().parent
P=R/'manuscript/jdsa'; S=P/'source'
C='9d8c501c58415e766d0d5773cf7cbb576da66cfa'; B='a3f0dde397341a41842a30ddb946e7966706351b'
OLD='manuscript/jdsa-final-review/'
NAMES={'Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Final_Review.pdf':'Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf','ESM_1_Retrieval_Assisted_Scalar_Grounding_JDSA_Final_Review_Supplement.pdf':'Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_Supplement.pdf'}
OLDURL='https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/manuscript/jdsa-final-editorial-polish-20261029/manuscript/jdsa-final-review'
NEWURL='https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/main/manuscript/jdsa/source'
REPLACE={**NAMES,OLDURL:NEWURL}
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def blob(path,commit=C):return git('show',commit+':'+path)
def sha(b):return hashlib.sha256(b).hexdigest()
def transformed(text):
 for a,b in REPLACE.items():text=text.replace(a,b)
 return text
def expand(path):
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(path.parent/(m[1]+('' if m[1].endswith('.tex') else '.tex'))),path.read_text())
tables=equations=0;diff=[];identical=[]
for path in git('ls-tree','-r','--name-only',C,OLD).decode().splitlines():
 rel=path[len(OLD):]
 if rel in NAMES or rel.endswith('.zip') or rel=='README.md':continue
 a=blob(path);b=(S/rel).read_bytes()
 if rel in ('main.tex','supplement.tex'):
  assert transformed(a.decode()).encode()==b,rel
  diff.extend(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='authoritative/'+rel,tofile='canonical/source/'+rel))
 elif rel!='build.sh':assert a==b,rel;identical.append(rel)
 if rel.endswith('.tex'):
  for env in ['tabular','tabularx','equation','algorithm','algorithmic']:
   pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
   aa=re.findall(pat,a.decode(),re.S);assert aa==re.findall(pat,b.decode(),re.S),(rel,env)
   if env in ('tabular','tabularx'):tables+=len(aa)
   if env=='equation':equations+=len(aa)
  for pat in [r'\$[^$]*\$',r'\\cite\w*\{[^}]+\}',r'\\(?:sub)*section\{[^}]+\}',r'\\label\{[^}]+\}']:
   assert re.findall(pat,a.decode())==re.findall(pat,b.decode()),(rel,pat)
assert (tables,equations)==(35,12),(tables,equations)
inv=(D/'INVESTIGATION_DESTINATIONS.json').read_bytes()
assert inv==blob('docs/jdsa_revision_phase4_20261029/INVESTIGATION_DESTINATIONS.json')
assert len(json.loads(inv))==47
(D/'SOURCE_LINK_DIFF.patch').write_text(''.join(diff))
# All pre-existing files outside the confirmed deletion set and two navigation files remain exact.
deleted=json.loads((D/'DELETED_PATHS.json').read_text())
expected=git('ls-tree','-r','--name-only',B,'manuscript/jdsa-stage1').decode().splitlines()
assert deleted==expected
preserved=0
for line in git('ls-tree','-r',B).decode().splitlines():
 meta,path=line.split('\t',1);mode,kind,oid=meta.split()
 if kind!='blob':continue
 if path in deleted:assert not (R/path).exists();continue
 if path in ('README.md','manuscript/MANUSCRIPT_README.md'):continue
 # Compare Git's tracked index/worktree diff, avoiding reads of unrelated research data.
 preserved+=1
changed=git('diff','--name-only',B).decode().splitlines()
allowed_new=('manuscript/jdsa/','docs/jdsa_canonical_publication_20261029/','docs/jdsa_final_editorial_polish_20261029/')
assert all(p in deleted or p in ('README.md','manuscript/MANUSCRIPT_README.md') or p.startswith(allowed_new) for p in changed),changed
qa=Path(os.environ['JDSA_REUSE_BUILD']) if 'JDSA_REUSE_BUILD' in os.environ else Path(tempfile.mkdtemp(prefix='jdsa-canonical-qa-'))
zpath=next(P.glob('*.zip'))
with zipfile.ZipFile(zpath) as z:
 allowed={'main.tex','supplement.tex','Fig1.pdf','Fig1.tex','references.bib','sn-jnl.cls','sn-basic.bst','orcidlink.sty','build.sh','README.md'}
 assert set(z.namelist())==allowed and len(z.infolist())==10
 assert z.testzip() is None
 for name in ('main.tex','supplement.tex'):assert z.read(name)==expand(S/name).replace('figures/Fig1.pdf','Fig1.pdf').encode()
 assert z.read('Fig1.tex')==expand(S/'figures/Fig1.tex').encode()
 assert z.read('Fig1.pdf')==(S/'figures/Fig1.pdf').read_bytes()
 for name in ('references.bib','sn-jnl.cls','sn-basic.bst','orcidlink.sty','build.sh'):assert z.read(name)==(S/name).read_bytes()
 patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{40,}',rb'AKIA[A-Z0-9]{16}',rb'sk-[A-Za-z0-9]{32,}',rb'hf_[A-Za-z0-9]{30,}']
 for name in allowed:
  for pattern in patterns:assert not re.search(pattern,z.read(name)),name
 z.extractall(qa)
if 'JDSA_REUSE_BUILD' not in os.environ:
 with (qa/'independent-build.log').open('w') as log:
  result=subprocess.run(['sh','build.sh'],cwd=qa,stdout=log,stderr=subprocess.STDOUT,env={**__import__('os').environ,'KEEP_BUILD':'1'},timeout=120)
 assert result.returncode==0,qa/'independent-build.log'
log=(qa/'independent-build.log').read_text();build=Path(re.search(r'Build directory: (.+)',log)[1])
report={'source_commit':C,'baseline_commit':B,'tables_preserved':tables,'equations_preserved':equations,'investigation_destinations_preserved':47,'byte_identical_source_assets':identical,'all_scientific_source_unchanged_except_three_link_targets':True,'protected_baseline_files_unchanged':preserved,'archive_members':sorted(allowed),'archive_allowlist_and_secret_pattern_checks_passed':True,'independent_build_directory':str(qa),'independent_build_command':'KEEP_BUILD=1 sh build.sh','final_logs':{},'pdfs':{}}
for kind in ('main','supplement'):
 txt=(build/(kind+'.log')).read_text()
 bad=[line for line in txt.splitlines() if re.search('undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull',line)]
 assert not bad,bad
 report['final_logs'][kind]={'unresolved_or_overfull_diagnostics':bad,'warnings':[line for line in txt.splitlines() if 'Warning' in line]}
for index,(old,new) in enumerate(NAMES.items()):
 original=fitz.open(stream=blob(OLD+old),filetype='pdf');canonical=fitz.open(P/new);rebuilt=fitz.open(qa/new)
 count=(22,32)[index];assert len(original)==len(canonical)==len(rebuilt)==count
 internal=external=uri=0; annotation_changes=0
 for i,(a,b,c) in enumerate(zip(original,canonical,rebuilt)):
  assert a.get_text()==b.get_text()==c.get_text(),(new,i,'text')
  # Compare all 54 pages at 144 dpi, including equations, tables and vector figures.
  matrix=fitz.Matrix(2,2)
  pixels=a.get_pixmap(matrix=matrix).samples
  assert pixels==b.get_pixmap(matrix=matrix).samples==c.get_pixmap(matrix=matrix).samples,(new,i,'pixels')
  al=a.get_links();bl=b.get_links();cl=c.get_links();assert len(al)==len(bl)==len(cl)
  def signature(link):
   return {k:transformed(v) if isinstance(v,str) else v for k,v in link.items() if k not in ('xref','id')}
  assert [signature(l) for l in al]==[signature(l) for l in bl]==[signature(l) for l in cl],(new,i,'link semantics')
  for l in bl:
   if l['kind'] in (fitz.LINK_GOTO,fitz.LINK_NAMED):
    assert 0<=l['page']<count
    if l.get('nameddest'):assert l['nameddest'] in canonical.resolve_names()
    internal+=1
   elif l['kind'] in (fitz.LINK_GOTOR,fitz.LINK_LAUNCH):
    filename,_,frag=l['file'].partition('#');target=P/filename;assert target.is_file(),l
    with fitz.open(target) as td:
     if frag.startswith('nameddest='):assert frag[len('nameddest='):] in td.resolve_names(),l
     elif isinstance(l.get('page'),int):assert 0<=l['page']<len(td)
     else:raise AssertionError(l)
    external+=1
   elif l['kind']==fitz.LINK_URI:uri+=1
  for block in b.get_text('dict')['blocks']:
   for line in block.get('lines',[]):
    for span in line['spans']:
     assert b.rect.contains(fitz.Rect(span['bbox'])),(new,i,span)
  assert '??' not in b.get_text(),(new,i,'unresolved marker')
 # Original object graph and page streams remain exact except enumerated link objects.
 assert (P/new).read_bytes().startswith(blob(OLD+old))
 assert canonical.xref_length()>=original.xref_length()
 annotation_xrefs={l['xref'] for pg in original for l in pg.get_links()}
 for x in range(1,original.xref_length()):
  oa=original.xref_object(x);ob=canonical.xref_object(x)
  if oa!=ob:
   assert x in annotation_xrefs and {k:transformed(original.xref_get_key(x,k)[1]) for k in original.xref_get_keys(x)}=={k:canonical.xref_get_key(x,k)[1] for k in canonical.xref_get_keys(x)},(new,x)
   annotation_changes+=1
  assert original.xref_stream(x)==canonical.xref_stream(x),(new,x,'stream')
 report['pdfs'][new]={'pages':count,'all_pages_text_and_144dpi_pixels_identical_to_authoritative_and_independent_build':True,'original_pdf_streams_unchanged':True,'authoritative_pdf_bytes_preserved_as_prefix':True,'changed_link_annotation_objects':annotation_changes,'valid_internal_links':internal,'valid_cross_document_links':external,'preserved_web_links_except_canonical_source_target':uri,'no_out_of_page_text_or_unresolved_markers':True,'authoritative_sha256':sha(blob(OLD+old)),'canonical_sha256':sha((P/new).read_bytes())}
report['artifact_sha256']={p.name:sha(p.read_bytes()) for p in [*P.glob('*.pdf'),zpath]}
(D/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
