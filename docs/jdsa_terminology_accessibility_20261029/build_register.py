from pathlib import Path
import subprocess,re,json,csv,collections
R=Path(__file__).resolve().parents[2]; S=R/'manuscript/jdsa-stage3';D=R/'docs/jdsa_terminology_accessibility_20261029';BASE='f2b8a94ea01922618e36b667c1d9b5e2c08f893c'
changes=json.loads((D/'CHANGES.json').read_text())
# Curated roles; proper names are deliberately described, not reverse-expanded.
data=r'''TF--IDF|retrieval|Term frequency--inverse document frequency; catalog-fitted lexical weighting with cosine ranking.
BM25|retrieval|Probabilistic lexical document-ranking method; proper algorithm name, no invented expansion.
LSA|retrieval|Latent semantic analysis; lower-dimensional projection of lexical features.
BGE-M3|model|Pretrained multilingual text encoder used for dense query/schema retrieval, not a generative grounder.
BGE|model alias|Historical shorthand for BGE-M3; main running references standardized, paired contrast labels remain traceable.
R@1|metric|Recall at rank one; here each query has one target, so this is top-ranked schema accuracy.
IR|representation|Intermediate representation in Semantic-IR repair; it does not mean information retrieval here.
LLM|abbreviation|Large language model; generative systems are distinct from the rule-based scalar grounder.
AI|abbreviation|Artificial intelligence; assisted author labels are not independent semantic truth.
NLP4LP|benchmark|Natural-language linear and mixed-integer linear programming benchmark, with reference-derived schemas.
NL4Opt|benchmark|Natural Language for Optimization corpus/competition for entity extraction and formulation representations.
Ner4Opt|system|Named-entity recognition framework for optimization, as described by the cited title.
Text2Opt-Bench|benchmark|Solver-verified text-to-optimization benchmark separating modeling from numerical binding.
OptMATH|system/dataset|Bidirectional data-synthesis framework and associated optimization-modeling data/models; extraction check is a distinct use.
PaMOP|system|Partition-based formulation framework with iterative correction; evaluated version is a project reconstruction.
OptiMUS|system|Formulation/code generation system with solver feedback.
ORLM|system|Optimization-modeling language-model training framework; adapted checkpoint context remains separate.
DeepOR|system|Optimization-modeling foundation model; no common-18 execution is claimed.
OPT2CODE|system|Retrieval-assisted solver-code generation; not parameter-inventory retrieval.
StruG|system|Structure-grounded pretraining for column/value grounding and mapping in text-to-SQL.
RAT-SQL|system|Relation-aware schema encoding and linking for text-to-SQL.
SQL|abbreviation|Structured Query Language; database-query context, separate from optimization grounding.
BIND|named intervention|Externalizes numerical data for programmatic binding; proper intervention name, no guessed expansion.
PICARD|system|Incremental parsing method that imposes structural decoding restrictions.
EquivaMap|system|Checks formulation mappings preserving feasibility and optimality.
OPT-Engine|benchmark|Complexity-scaled formulation/solving benchmark, not the present fixed-catalog task.
Autoformulation|system/concept|LLM-guided search over full modeling decisions, outside the supplied-slot assignment controls.
OptLLM|system|Natural-language formulation/code system; prompting and fine-tuning variants distinguished.
Chain-of-Experts|system|Coordinated agents for operations-research modeling; task table supplies functional context.
LLMOPT|system|Trained general optimization-model definition/solution system.
SIRL|system|Solver-informed reinforcement-learning modeling approach; row supplies function, no artificial name expansion.
OR-R1|system|Test-time reinforcement-learning optimization-modeling system; historical checkpoint unavailability retained.
OptimAI|system|Multi-agent formulation, planning and code-revision preprint; no evaluated performance implied.
AlphaOPT|system|Experience-library optimization-program formulation; no weight updates in reported library-learning distinction.
ORThought|system|Structured-reasoning approach to logistics optimization modeling.
Spider|benchmark|Cross-domain text-to-SQL/database-interface benchmark, established in cited title and context.
ReMatch|system|Retrieval-enhanced schema matching; contextual preprint, no new comparison.
MIPLIB-NL|benchmark|Industrial-scale natural-language optimization-modeling benchmark, used only as context.
DistilRoBERTa|model|Compact pretrained text encoder used by the archived mention-slot classifier.
MiniLM|model|Compact pretrained text encoder; frozen cosine feature and separate cross-encoder implementation are distinct uses.
MS-MARCO|benchmark|Passage-ranking benchmark associated with the unverified cross-encoder run, not a completed outcome.
GPT-5.4|model|Recorded generation model identifier; snapshot/settings retained verbatim.
Qwen2.5-7B|model|Released checkpoint family used for adapted OptMATH generation; exact revision remains in provenance.
LLaMA-3-8B|model|Released checkpoint family used for adapted ORLM generation; execution unavailable.
Gemini|model/service|Recorded Google Vertex-served relationship model; validity is not correctness.
DeepSeek|model/service|Model served through io.net; not a direct DeepSeek API claim.
Cohere|service|Relationship-labeling service; Command A model/request identifiers preserved.
Command A|model|Served Cohere relationship-model family; exact requested model path preserved.
CloudRift|service|Historical unexecuted extension; availability gate, not a zero-accuracy result.
Coverage|metric|Per-query assigned eligible-slot fraction, zero on empty sets, then equal-weight query average.
TypeMatch|metric|Per-query compatible-assignment fraction among assigned slots, zero on empty sets, then query average.
Exact20|metric|Conditional mean within-tolerance numerical slot fraction, excluding misses/empty comparable sets, not complete-instance accuracy.
exact20|saved field|Saved per-query fraction x_i; blank on schema misses or no comparable slots.
InstantiationReady|metric|Historical identifier for coarse/ordinary readiness: both coverage and type fraction at least 0.8.
StrictInstantiationReady|metric|Coarse readiness plus correct-schema gate; no numerical-value test.
InstReady|table abbreviation|InstantiationReady/coarse readiness in main metric tables.
Strict|table abbreviation|StrictInstantiationReady in table headers and historical contrast labels.
coarse readiness|concept|Coverage/type checks; allows omissions and incompatible slots within thresholds, not numerical correctness.
ordinary readiness|concept|Same implemented metric as coarse readiness, retained to distinguish ordinary and strict-ready sets.
strict readiness|concept|Coarse readiness plus schema hit, still not numerical correctness.
complete scalar agreement|endpoint|Every required reference scalar is assigned, comparable, type-compatible and within historical tolerance, on correct schema.
conditional denominator|concept|Only the explicitly qualifying query/slot subset contributes; methods may have different subsets.
eligible|concept|Main slots intersect reference numeric scalar keys with retrieved schema keys; factorial metadata-only inventory differs.
schema hit|concept|Top-ranked predicted schema equals the query's target schema.
reference-assisted|concept|Prediction receives benchmark-derived structure; no autonomous formulation score follows.
gold-assisted|concept|Reference scalar identities/count filter the assignment inventory before numerical selection; magnitudes not given to selector.
supplied-slot|concept|Parameter inventory is provided rather than discovered autonomously.
frozen-score|control|Candidate pools and score matrices fixed within decoder comparisons; differs from comparing whole procedures.
one-to-one|assignment|Each mention index can be assigned at most once; equal values at distinct occurrences remain distinct candidates.
scalar grounding|task|Assign textual quantities to single-valued model parameters.
schema|task|Retrieved model template and parameter structure.
mention|task|A textual occurrence of a quantity, not just its magnitude.
slot|task|Model parameter requiring a value; scalar slots require one number.
abstention|decision|Leave a slot unfilled; reduces coverage and may change conditional metric populations.
null option|decision|Unfilled assignment with score zero in the controlled decoder comparisons.
typed greedy|method|Visit slots in run order and select highest-ranked unused mention using coarse compatibility.
ratio-aware|extractor|Recognizes multiplicative wording such as twice/triple in addition to baseline forms.
constrained assignment|method|Complete grounding procedure, distinct candidates/context/scoring/abstention/decoding from typed greedy.
maximum-weight matching|method|Choose assignment maximizing summed weights under the stated matching restriction.
optimization-role repair|method|Archived role-based grounding procedure; distinct from headline typed-greedy protocol.
search-structured grounding|method|Named full grounding procedure with its own extractor/scoring/search, not an isolated search intervention.
semantic|concept|Meaning/fidelity to source problem; scalar equality and model agreement do not establish it.
Oracle|control|Reference-schema selection or explicitly described diagnostic choice, not deployable autonomous inference.
Typed|factorial label|Typed-token candidate constructor.
Rich|factorial label|Richer mention constructor retaining contextual information; not automatically a better candidate set.
Ordinal|factorial scorer|Numerical encoding of ordered type preferences; does not recreate every historical greedy decision.
Context|factorial scorer|Frozen contextual slot-mention score.
No marker bonus|factorial scorer|Same contextual score without percent/currency marker bonuses.
E slots|table notation|Numerically exact scalar assignment counts; distinct from eligibility set E.
T slots|table notation|Tolerance-correct scalar assignment counts; distinct from compatibility set T_i.
E full|table notation|Whole-query exact success with all keys and compatible types.
T full|table notation|Whole-query tolerance success with all keys and compatible types.
D1|internal identifier|Author-reviewed, language-model-assisted semantic contrast set; twelve pairs/twenty-four samples, not provider predictions.
D1P|internal identifier|Common pair-ID prefix; table captions omit it while preserving individual pair identifiers.
G1|internal identifier|Feasibility audit of autonomous inventory discovery; no measured autonomous accuracy.
G2|internal identifier|Fixed-slot gold-filter intervention audit distinguishing slot participation and mention filtering.
G3|internal identifier|Static schema-only versus gold-key-filtered scalar participation audit, without decoding.
M3|internal identifier|Formulation-result reaggregation evaluator/source phase, not a model or metric.
M5|internal identifier|Local evidence-validation audit/bundle including original records; not identical to curated public release.
P0|internal identifier|Archived classifier-plus-assignment study using handcrafted and frozen embedding features.
common-18|internal population|The same eighteen selected NLP4LP problems used across external formulation attempts; not a random population sample.
Stage-A|internal cache|Pre-ratio saved candidate/selection cache; role/quantity separability audit has its own similarly named Stage A.
Stage A|internal alias|Preserve context: pre-ratio cache versus separately identified role/quantity audit, not interchangeable records.
Recovery v2.1|internal version|Final archived offline repair and historical replacement stage of relationship judgments.
numerical_reference_v2|overlay|Separate versioned fractional-reference correction overlay; original references/predictions not overwritten.
JDSA|journal/release|International Journal of Data Science and Analytics; proper evidence-release branding.
v1.2|release version|Historical evidence release with its own exact revision; not the manuscript patch version.
v2|release shorthand|Curated public evidence release, with exact version and revision stated in reproduction section.
RQ1|research question|Schema retrieval question; exact question preserved.
RQ2|research question|Diagnostic validity question; exact question preserved.
RQ3|research question|Grounding mechanisms question; exact question preserved.
TG|table abbreviation|Typed-greedy grounding; Oracle-TG uses reference schema.
extr.|table abbreviation|Extraction configuration.
Orig|table abbreviation|Original full query; noisy is lexical corruption/numeric masking; short is first sentence.
Avg.|table abbreviation|Average over retrieval variants before rounding.
Ret.|table abbreviation|Retrieval or memory.
Sol.|table abbreviation|Solver feedback during generation or training.
FT|table abbreviation|Fine-tuning or reinforcement-learning training, per landscape caption.
RL|abbreviation|Reinforcement learning.
OR|abbreviation|Operations research.
CoT|abbreviation|Chain of thought.
NL|abbreviation|Natural language.
LP|abbreviation|Linear programming/model; (MI)LP additionally permits integer-valued variables.
MILP|abbreviation|Mixed-integer linear programming.
Comp.|table abbreviation|Direct comparability to fixed-catalog scalar metrics, not general quality.
inst.|table abbreviation|Instantiation in landscape; instances in experimental-block counts; each local caption defines scope.
CI|statistics|Confidence interval; grouped/marginal/historical qualifications preserved.
Diff|table abbreviation|First configuration minus second; caption defines direction.
Hit@5|retrieval metric|Target found in top five; not document precision at five.
P@5|historical field|Stored name for Hit@5, not literal precision among five documents.
MRR@10|retrieval metric|Mean reciprocal target rank truncated at ten.
nDCG@10|retrieval metric|Normalized discounted cumulative gain at ten; single target relevance in retained implementation.
Coverage@10|retrieval metric|Fraction of queries whose target appears in first ten; not scalar-slot Coverage.
W/H|table disposition|Withheld for insufficient numerical precision.
N/A|table disposition|Unavailable execution environment in this comparison; not zero success.
Obj.|table abbreviation|Valid numerical objective extraction among successful runs.
Ref.|table abbreviation|Available reference objectives among valid extractions.
Agree./comp.|table abbreviation|Objective agreement over classifiable executed reference comparisons.
P/E/N/Y+/Y--/Y?/U|case labels|Parse failure/process failure/environment absent/match/discrepancy/no reference/unresolved precision, scoped to case table.
CORRECT|judgment|Model assertion or assisted author decision in local context, not independent semantic gold.
INCORRECT|judgment|Negative assertion relative to supplied candidate; not necessarily a complete-model assessment.
AMBIGUOUS|judgment|One of four nominal model assertion categories.
CANNOT_DETERMINE|judgment|One of four nominal model assertion categories.
UNRESOLVED|disposition|Missing verified lexical endpoint; distinguishes raw from determinate unanimity.
C/I/A/CD|table labels|CORRECT/INCORRECT/AMBIGUOUS/CANNOT_DETERMINE in relationship-label table.
U/D/R/I|table labels|Determinate unanimity/disagreement/unresolved/invalid-incomplete in cluster table; local meanings differ from case table.
AMPL|software|Algebraic optimization modeling language; actual execution uses specified solver.
GAMS|software/source|Optimization modeling system/source models for tiny auxiliary study; no invented acronym expansion.
HiGHS|solver|Optimization solver accessed through SciPy compatibility wrapper; proper name.
SciPy|software|Scientific Python library providing optimization interface.
NumPy|software|Numerical Python library; exact version/seed retained for historical statistics.
Gurobi|solver|Optimization solver; successful generated-model execution does not certify source semantics.
gurobipy|software identifier|Python Gurobi interface, named in solver-environment column.
coptpy|software identifier|Missing solver dependency for ORLM; no execution result inferred.
shim|software concept|Compatibility wrapper adapting reference-program calls to supported optimization interface.
API|abbreviation|Application programming interface.
AST|abbreviation|Abstract syntax tree; syntactic code-value extraction does not identify every source-semantic occurrence.
CSV|file format|Comma-separated values, per-query metric ledger format.
JSON|file format|JavaScript Object Notation; native schema controls response structure, not meaning.
HTTP|protocol|Hypertext Transfer Protocol; transport success does not imply valid judgment content.
UTC|time standard|Coordinated Universal Time, historical request timestamp zone.
FP8|format|Recorded eight-bit floating-point serving format; not authenticated weight checksum.
float32|format|Floating-point precision reported for BGE-M3 embeddings; not a complete runtime record.
L2-normalized|notation|Unit Euclidean length of embedding vectors.
RSS|runtime|Resident set size, physical memory use in retained timing summary.
SHA-256|integrity|Cryptographic content checksum for identifying exact retained files.
SHAs|version shorthand|Source-commit hashes; replaced in prose without changing exact identifiers.
ID|identifier|Record identifier, not a numerical measurement; exact IDs retained.
HF|repository shorthand|Hugging Face; one-use shorthand replaced with full name.
QC|abbreviation|Quality checks; replaced in population-table row.
ESWA-era|historical shorthand|Early archived grounding campaign; journal-era label removed, cohort/results preserved.
ZIP|file format|Archive containing editable source/dependencies; universally familiar packaging term retained.
PDF|file format|Portable document format; universally familiar deliverable term retained.
ORCID|author identifier|Persistent researcher identifier; exact author identifier/link unchanged.
NJIT|institution|New Jersey Institute of Technology, spelled out in affiliation; protected acknowledgment unchanged.
NJ|postal abbreviation|New Jersey in author address; standard postal format preserved.
USA|postal abbreviation|United States of America; standard affiliation country label preserved.
CC BY-NC-SA|license identifier|Exact Creative Commons license identifier retained; no license terms changed or interpreted anew.
CC BY-NC|license identifier|Exact upstream license identifier retained; no redistribution claim broadened.
MIT|license identifier|Standard software-license name retained; no forced institutional expansion.
CRediT|authorship taxonomy|Standard contribution taxonomy already explicitly identified; protected declaration retained.
ChatGPT|proper name|Writing-assistance product named in protected declaration.
OpenAI|proper name|Provider/software identity in factual disclosures; not expanded.
GitHub|proper name|Public code/release hosting service; URLs unchanged.
ShellFish|proper name|Secure ShellFish product named in protected acknowledgment.
arXiv|repository|Preprint repository; identifiers/titles left unchanged.
PYTHONHASHSEED|runtime identifier|Python hash-ordering seed environment variable; historical setting preserved.
IndexError|exception|Named Python indexing exception; historical failure report, not reclassified.
generateContent|API identifier|Exact recorded service route, not an abbreviation requiring expansion.
LOW|service setting|Recorded Gemini thinking setting; exact service value preserved.
''' 
# Additional conceptual vocabulary, reviewed in full context.
concepts={
'gold':'Stored reference, not semantic truth; main assistance subsection and supplement introduction define usage.',
'macro':'Equal-weight average of per-query ratios; distinct from pooled micro ratio.',
'micro':'Pooled numerator/denominator across records; extraction paragraph defines both aggregates.',
'delexicalized':'Number expressions masked before text grouping, per archived grouping implementation.',
'Jaccard':'Intersection-over-union text-shingle similarity; threshold/grouping protocol preserved.',
'bootstrap':'Resampling procedure with specified groups, seed and replication count; not independent-slot sampling.',
'Holm':'Sequential multiple-test adjustment over stated retrospective family; no new significance claim.',
'McNemar':'Exact paired query-level test retained alongside groupwise tests.',
'Fleiss':'Chance-corrected inter-rater agreement, not model accuracy.',
'beam':'Bounded candidate search retaining a limited frontier; local retention/pruning counts distinguish stages.',
'decoder':'Procedure choosing assignments from candidate pools and scores.',
'candidate':'Available numerical mention/option; availability does not mean selection or semantic correctness.',
'cardinalization':'Conversion of ordered preferences to numerical scores; replaced with plain-language explanation.',
'edge-score':'Compatibility score for a possible slot-mention assignment, fixed within decoder pairs.',
'factorial':'Crossed candidate/scorer/decoder/reuse configurations; all factors and populations already specified.',
'null':'Unfilled option at fixed score, not verified absence of a source-semantic relationship.',
'calibration':'Agreement between predictive probabilities and outcomes is not established for archived scorer.',
'weak labels':'Equality-derived first-value-match supervision; not occurrence-specific human annotation.',
'quarantine':'Precautionary grouping exclusion; not measured contamination or error rate.',
'checkpoint':'Saved model or execution state identified by context/revision; incomplete histories not silently repaired.',
'smoke test':'Small implementation sanity run; not evidence of completed full evaluation.',
'zero-shot':'Generation without task examples in the supplied prompt; exact protocol remains system-dependent.',
'presolve':'Solver preprocessing; retained AMPL presolve success distinguished from Gurobi runs.',
'incumbent':'Provisional objective printed before terminal optimal summary; source of historical extraction correction.',
'projection':'Sanitized subset of saved record fields; cannot replace missing raw source validation.',
'ledger':'Row-level saved numerical/validity record used for supported aggregation.',
'provenance':'Origins, versions and transformations of records; public and private boundaries remain distinct.',
'reconstruction':'Recover original prediction/protocol from retained observations, distinct from reaggregating saved metrics.',
'reanalysis':'Recompute supported summaries from archived observations; no new inference.',
'reproduction':'Replay full historical experiment; missing predictions/runtime/source data prevent full reproduction.',
'overlay':'Separate correction record; original scientific evidence remains immutable.',
'half-open':'Character span includes start offset and excludes end offset, explicitly explained locally.',
'float':'Numeric scalar type in reference filtering; booleans excluded and coercion rules retained.',
'raw unknown token':'Masked nonnumeric assignment may count as filled but cannot be numerically compared.',
'role':'Meaning of quantity in model, not determined solely by numeric magnitude or coarse type.',
'feasible':'Solver returns solution to generated/reference program, not certification of source-problem fidelity.',
'objective agreement':'Conditional comparison with stored objective under exact preserved rounding/tolerance predicate.',
'equivalence':'Formulation mappings must preserve feasible sets/objectives, beyond scalar agreement.',
'retention':'Correct candidate remains in saved ranked list; differs from final selection.',
'no-op':'Variant activated no supported relations; unchanged result does not reject useful constraints.',
'codegen':'Code generation, already spelled out in landscape system context.',
'stdout':'Program standard output; saved execution record, not newly generated evidence.',
'enums':'Enumerated allowed identifier choices; replaced by full phrase at first use.',
'top-':'Rank cutoff for retrieval or candidate lists; stated locally.',
}
entries=[tuple(line.split('|',2)) for line in data.splitlines() if line];entries += [(k,'specialized concept',v) for k,v in concepts.items()]
# Notation roles cover each scientific symbol family; all actual math occurrences are also indexed below.
notation={
'q':'Natural-language query; i indexes queries in evaluation, but slots locally in typed assignment.',
'\\mathcal{S}':'Fixed schema catalog; M schemas s_1,...,s_M.',
'\\hat{s}':'Predicted top-ranked schema; s_i^* reference schema for query i.',
'K_G':'Reference keys whose values are numeric scalars, excluding booleans.',
'P_{\\mathrm{all}}':'All predicted metadata keys with reference-record fallback; not scalar-shape filtered.',
'E(':'Eligible main assignment set, intersection of reference scalar keys and predicted keys.',
'y_q':'Stored reference map; hat y partial prediction, both locally defined.',
'A_i':'Assigned eligible-slot set in evaluation, distinct from algorithm mapping A.',
'T_i':'Type-compatible assigned-slot set; distinct from sampling temperature T and tolerance table shorthand.',
'C_i':'Numerically comparable assigned-slot set; distinct from algorithm candidate set mathcal C_i.',
'H_i':'Schema-hit indicator; indicator value one when predicted and target schemas match.',
'c_i':'Per-query eligible-slot coverage, zero for empty eligible set.',
't_i':'Per-query assigned-slot type compatibility, zero for empty assigned set; not selected mention t_i^*.',
'e_{i,s}':'Historical bounded numerical error for scalar key s of query i; equation unchanged.',
'x_i':'Within-tolerance comparable-slot fraction on query i, not whole-query correctness.',
'J=':'Queries with schema hit and nonempty comparable assigned-slot set; conditional Exact20 population.',
'F_i^G':'Indicator of complete required-scalar success with schema, all keys, comparability, types and tolerance.',
'F^G':'Set of queries meeting complete scalar criterion; G distinguishes required gold keys from eligible-only F.',
'I_i':'Ordinary/coarse readiness indicator; I is corresponding query set.',
'S_i':'Strict readiness indicator; S is corresponding query set.',
'\\rho_I':'Complete scalar successes divided by ordinary-ready outputs; descriptive conditional agreement.',
'\\rho_S':'Complete scalar successes divided by strict-ready outputs; descriptive conditional agreement.',
'\\mathcal T':'Extracted mentions in assignment; different scope from type-compatible evaluation set.',
'\\mathcal U':'Used mention indices in algorithm; no index reuse.',
'p_i':'Current slot in typed-greedy iteration; here i indexes slots, not queries.',
't_i^*':'Selected mention for current slot; distinct from type-compatibility fraction t_i.',
'\\mathrm{compat}':'Deterministic slot-mention ranking rule, with tie preferences specified.',
'\\mathcal P':'Eligible algorithm inventory, not complete reference key set in all cases.',
'\\mathcal C_i':'Unused-mention candidate set within algorithm, not evaluation comparability C_i.',
'B=':'Number of bootstrap resamples; distinct historical primary and factorial values retained.',
'p=':'Statistical p-value in test contexts; local share p and top-p sampling explicitly scoped.',
'\\kappa':'Fleiss chance-corrected nominal inter-rater agreement, not correctness.',
'\\Delta':'Difference with direction defined separately in each table caption.',
'n_{\\text{differ}}':'Number of queries changing readiness labels between ordinary and strict definitions.',
'P_{\\mathrm{scalar}}':'Predicted-schema metadata keys with declared scalar shape, distinct from all-key fallback inventory.',
'X':'Component quantity in fractional-share reference convention; X and Y defined locally.',
'p_1':'Percentage-point ingredient coefficient in snack-mix correction, not new measured outcome.',
'b_{\\mathrm{cat}}':'Available cat-paw ingredient mass in kilograms, locally defined.',
'\\hat z':'Terminal generated-program objective; z^* is stored reference objective.',
'\\operatorname{round}':'Python float conversion and nearest-integer ties-to-even rounding, unchanged objective proxy.',
'n_{ic}':'Number of judgments in nominal category c on item i; N complete items and n raters locally defined.',
'\\bar P':'Observed nominal agreement in Fleiss statistic.',
'P_e':'Chance agreement from category marginals in Fleiss statistic.',
'm=(':'Mention tuple (a,b,v): start/end offsets and normalized value.',
'r=(':'Proposed relation (q,m,p) linking a query mention to parameter p.',
'O(':'Standard asymptotic comparison count in eligible slots and mentions; familiar notation not overexplained.',
'10^{-9}':'Preserved scaled numerical exactness threshold in pilot; not bitwise equality.',
'0.20':'Preserved inclusive bounded-error tolerance: absolute below unit magnitude, relative otherwise.',
}
entries += [(k,'mathematical notation',v) for k,v in notation.items()]

def read(n,old):
 return subprocess.check_output(['git','show',BASE+':manuscript/jdsa-stage3/'+n],cwd=R).decode() if old else (S/n).read_text()
def flatten(n,old,doc):
 lines=read(n,old).splitlines();started=n not in {'main.tex','supplement.tex'};out=[]
 for i,l in enumerate(lines,1):
  if not started:
   if r'\begin{document}' in l:started=True
   continue
  if l.lstrip().startswith('%'):continue
  inc=re.fullmatch(r'\s*\\input\{([^}]+)\}\s*',l)
  if inc:out.extend(flatten(inc[1]+'.tex',old,doc));continue
  scope='abstract' if l.startswith(r'\abstract{') else doc
  out.append({'file':n,'line':i,'document':scope,'text':l})
  if r'\includegraphics' in l and 'Fig1.pdf' in l:out.extend(flatten('pipeline_figure.tex',old,doc))
 return out
lines={k:flatten('main.tex',k,'main')+flatten('supplement.tex',k,'supplement') for k in (True,False)}
# Match literal terms except listed alternatives; boundaries avoid TF inside unrelated identifiers.
alias={'CANNOT_DETERMINE':r'CANNOT(?:\\_)?DETERMINE','common-18':r'common-18|Common-18','Stage-A':r'Stage-A|Stage A','P/E/N/Y+/Y--/Y?/U':r'P denotes final parse failure','C/I/A/CD':r'C = CORRECT','U/D/R/I':r'U denotes determinate unanimity','L2-normalized':r'L2-normalized','numerical_reference_v2':r'numerical\\_reference\\_v2'}
alias.update({'M3':r'(?<!BGE-)M3','BGE':r'BGE(?!-M3)','LLM':r'LLMs?','CSV':r'CSVs?','ID':r'IDs?','coarse readiness':r'coarse[- ]readiness|coarse-ready','ordinary readiness':r'ordinary[- ]readiness|ordinary-ready','strict readiness':r'[Ss]trict[- ]readiness|strict-ready','complete scalar agreement':r'complete(?: stored-reference)? scalar agreement'})
def matches(term,l):
 pat=alias.get(term,re.escape(term))
 if term[0].isalnum():pat=r'(?<![A-Za-z])(?:'+pat+r')(?![A-Za-z0-9])'
 return list(re.finditer(pat,l,re.I if term in concepts or term=='common-18' else 0))
def occ(term,old):
 return [dict(file=l['file'],line=l['line'],document=l['document'],column=m.start()+1,context=l['text'][max(0,m.start()-100):m.end()+160]) for l in lines[old] for m in matches(term,l['text'])]
def first(xs):
 return {d:next((f"{x['file']}:{x['line']}" for x in xs if x['document']==d),None) for d in ['abstract','main','supplement']}
# Inventory every literal record/field/model/path/version identifier in the read documents.
known={x[0] for x in entries};literals=set()
for ls in lines.values():
 for l in ls:
  literals.update(re.findall(r'\\(?:path|texttt)\{([^{}]+)\}',l['text']))
  literals.update(re.findall(r'\b(?:Min|Max)[A-Z][A-Za-z]+\b',l['text']))
for term in sorted(literals-known):
 entries.append((term,'literal evidence/software identifier','Exact identifier retained for traceability; its local surrounding paragraph/table states the record, model, field, revision, parameter or command role. It is not an acronym to reverse-expand; no identifier or path is silently renamed.'))
rows=[]
for term,cat,definition in entries:
 a,b=occ(term,True),occ(term,False)
 relevant=[i+1 for i,c in enumerate(changes) if matches(term,c['before']) or matches(term,c['after'])]
 if not a and not b:continue
 changed=bool(relevant)
 rows.append({'term':term,'category':cat,'first_occurrence_before':first(a),'first_occurrence_after':first(b),'existing_definition': 'Baseline first-use wording: '+(' / '.join(d+': '+next(x['context'] for x in a if x['document']==d) for d in ['abstract','main','supplement'] if any(x['document']==d for x in a)) if a else 'Not present in baseline; explanatory notation only.'),'problem_identified':('; '.join(dict.fromkeys(changes[i-1]['reason'] for i in relevant)) if changed else 'No additional expansion needed; official names, ordinary notation and protected metadata are not reverse-expanded.'),'recommended_correction':('Local explanation/expansion only; see numbered before/after edits.' if changed else 'Retain; avoid redundant explanations.'),'final_action':('Clarified; see change IDs '+','.join(map(str,relevant)) if changed else 'Reviewed and retained'),'definition_or_scope':definition,'change_ids':relevant,'occurrences_before':a,'occurrences_after':b})
# Complete mathematical spans: retain every occurrence, including simple familiar arithmetic.
math=[]
for old in (True,False):
 for l in lines[old]:
  for m in re.finditer(r'(?<!\\)\$(.*?)(?<!\\)\$',l['text']):
   math.append({'version':'v1.1.0' if old else 'revised','file':l['file'],'line':l['line'],'document':l['document'],'expression':m[1],'disposition':'Reviewed in local context; formal meanings and numerical criteria preserved. See scoped mathematical rows in register.'})
# Full display equations are indexed separately rather than inferred from individual lines.
for old in (True,False):
 for n in ['main.tex','reference_conventions.tex','llm_evidence.tex','relationship_supplement.tex']:
  t=read(n,old)
  for m in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}',t,re.S):
   math.append({'version':'v1.1.0' if old else 'revised','file':n,'line':t[:m.start()].count('\n')+1,'expression':m[0],'disposition':'Protected display equation: byte identity verified against release baseline.'})
(D/'TERMINOLOGY_REGISTER.json').write_text(json.dumps(rows,indent=2)+'\n')
(D/'MATHEMATICAL_OCCURRENCES.json').write_text(json.dumps(math,indent=2)+'\n')
with (D/'TERMINOLOGY_REGISTER.tsv').open('w') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['Term','Category','First occurrence before','First occurrence after','Existing definition / first-use wording','Audited definition / scope','Problem identified','Recommended correction','Final action'])
 for x in rows:w.writerow([x['term'],x['category'],json.dumps(x['first_occurrence_before']),json.dumps(x['first_occurrence_after']),x['existing_definition'],x['definition_or_scope'],x['problem_identified'],x['recommended_correction'],x['final_action']])
# Normalize sequential edits to a complete final diff for each affected source line.
log=[]
import difflib
for n in sorted({x['file'] for x in changes}):
 a=read(n,True).splitlines();b=read(n,False).splitlines()
 for op,i,j,k,z in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
  if op!='equal':log.append({'file':n,'before_lines':[i+1,j],'after_lines':[k+1,z],'before':'\n'.join(a[i:j]),'after':'\n'.join(b[k:z])})
(D/'BEFORE_AFTER.json').write_text(json.dumps(log,indent=2)+'\n')
(D/'AUDIT_SCOPE.json').write_text(json.dumps({'baseline':BASE,'register_terms':len(rows),'term_occurrences_before':sum(len(x['occurrences_before']) for x in rows),'term_occurrences_after':sum(len(x['occurrences_after']) for x in rows),'mathematical_occurrences':len(math),'scientific_files_read':sorted({l['file'] for l in lines[True]}),'method':'Full sequential source reading of article and recursively included supplement, followed by term/notation scan and local-context review. Bibliographic titles and exact identifiers are protected; common arithmetic is indexed without unnecessary expansion.'},indent=2)+'\n')
print(len(rows),'register rows;',len(math),'math occurrence records')
