"""Check Phase 2 prose edits against Phase 1; no scientific code is executed.

Usage: python3 verify_phase2.py BUILD_DIRECTORY INDEPENDENT_ZIP_BUILD_DIRECTORY
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import zipfile

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
SOURCE = ROOT / 'manuscript/jdsa-stage3'
BASE = 'ae9eb9c14c53c687dafe20d9d774a419d198424d'
MAIN = 'Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf'
checks = []


def check(name, condition):
    checks.append({'check': name, 'pass': bool(condition)})
    if not condition:
        raise AssertionError(name)


def baseline(name):
    return subprocess.check_output(['git', 'show', BASE + ':manuscript/jdsa-stage3/' + name], cwd=ROOT)


def numerical_tokens(text):
    text = re.sub(r'\\(?:cite|ref|eqref)\{[^}]+\}', '', text)
    # Exclude names/question identifiers, not quantitative findings.
    text = re.sub(r'\b(?:RQ[123]|BGE-M3|BM25|NLP4LP|Text2Opt-Bench)\b', '', text)
    return Counter(re.findall(r'\d+(?:\.\d+)?', text))


def text_without_pagination(pdf):
    return '\n'.join('\n'.join(x for x in page.get_text().splitlines()
                               if x.strip() != str(page.number + 1)) for page in pdf)


def main():
    build, independent = map(Path, sys.argv[1:3])
    old = baseline('main.tex').decode()
    new = (SOURCE / 'main.tex').read_text()
    edits = json.loads((REPORT / 'EDITORIAL_LOG.json').read_text())
    reconstructed = old.splitlines()
    omissions = {283: Counter(['0.9013', '0.8640', '0.8469', '0.8328']),
                 386: Counter(['7', '40', '37', '35', '46', '9'])}
    for edit in edits:
        n = edit['baseline_line']
        check('logged baseline text ' + edit['id'], reconstructed[n - 1] == edit['before'])
        check('rationale and preservation location ' + edit['id'],
              edit['reason'] and edit['preservation_or_repetition_location'])
        reconstructed[n - 1] = edit['after']
        a, b = numerical_tokens(edit['before']), numerical_tokens(edit['after'])
        check('numerical findings preserved ' + edit['id'],
              a - b == omissions.get(n, Counter()) and not b - a)
    check('every source edit is logged', '\n'.join(reconstructed) + '\n' == new)
    protected = {}
    for env in ['equation', 'algorithm', 'figure', 'table']:
        pattern = r'\\begin\{' + env + r'\*?\}.*?\\end\{' + env + r'\*?\}'
        before, after = re.findall(pattern, old, re.S), re.findall(pattern, new, re.S)
        check('byte-identical complete ' + env + ' environments', before == after)
        protected[env] = len(before)
    # The only removed inline mathematics are literal values already in Tables 1/5.
    before_math = Counter(re.findall(r'(?<!\\)\$.*?(?<!\\)\$', old, re.S))
    after_math = Counter(re.findall(r'(?<!\\)\$.*?(?<!\\)\$', new, re.S))
    removed_math = Counter('$' + n + '$' for vals in omissions.values() for n in vals.elements())
    check('all mathematical notation preserved except logged duplicate table values',
          before_math - after_math == removed_math and not after_math - before_math)
    for pattern, label in [
        (r'\\cite\{[^}]+\}', 'citations'),
        (r'\\(?:ref|eqref)\{[^}]+\}', 'cross-references'),
        (r'\\(?:url|href|path|nolinkurl)\{[^}]+\}', 'public URLs and pinned identifiers'),
        (r'\\label\{[^}]+\}', 'labels'),
        (r'\\item \\textbf\{RQ.*', 'research questions'),
        (r'\\keywords\{[^}]+\}', 'keywords'),
    ]:
        check('exact sequence of ' + label, re.findall(pattern, old) == re.findall(pattern, new))
    check('front matter, author identity and URL configuration preserved', old.split(r'\abstract')[0] == new.split(r'\abstract')[0])
    check('acknowledgments and post-abstract configuration preserved',
          old.split(r'\keywords')[1].split(r'\section{Introduction}')[0] ==
          new.split(r'\keywords')[1].split(r'\section{Introduction}')[0])
    check('all back matter, funding, contributions, disclosures and availability statements exact',
          old.split(r'\backmatter')[1] == new.split(r'\backmatter')[1])
    for path in SOURCE.iterdir():
        if path.suffix in {'.tex', '.bib', '.bbl', '.cls', '.sty', '.bst'} and path.name != 'main.tex':
            check('unchanged supplementary/bibliographic/template source ' + path.name, path.read_bytes() == baseline(path.name))
    abstract_counts = [len(re.search(r'\\abstract\{(.*?)\}\n', s).group(1).split()) for s in [old, new]]
    check('abstract word limit', 150 <= abstract_counts[1] <= 250)
    keyword_count = len(re.search(r'\\keywords\{(.*?)\}', new).group(1).split(','))
    check('keyword count', 4 <= keyword_count <= 6)
    pdfs = {name: pymupdf.open(SOURCE / name) for name in [MAIN, 'ESM_1.pdf']}
    destinations = {name: pdf.resolve_names() for name, pdf in pdfs.items()}
    result = {'baseline': BASE, 'build_directory': str(build), 'abstract_words_before_after': abstract_counts,
              'keywords': keyword_count, 'logged_rewrites': len(edits), 'protected_environments': protected,
              'omitted_duplicate_prose_values': {str(k): dict(v) for k, v in omissions.items()}, 'pdfs': {}}
    for filename, pdf in pdfs.items():
        stem = 'main' if filename == MAIN else 'supplement'
        before = pymupdf.open(stream=baseline(filename), filetype='pdf')
        check('independent source-ZIP build byte identity ' + stem,
              (SOURCE / filename).read_bytes() == (independent / filename).read_bytes())
        log = (build / (stem + '.log')).read_text()
        blg = (build / (stem + '.blg')).read_text()
        check('no unresolved/overfull LaTeX diagnostics ' + stem,
              not re.search(r'undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull|^!', log, re.M))
        check('clean BibTeX log ' + stem, 'Warning--' not in blg and 'error message' not in blg)
        aux = (build / (stem + '.aux')).read_text()
        cited = set(k for group in re.findall(r'\\citation\{([^}]+)\}', aux) for k in group.split(','))
        keys = re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}', (build / (stem + '.bbl')).read_text(), re.S)
        check('all printed bibliography entries cited ' + stem, set(keys) == cited)
        check('all bibliography entries linked ' + stem, all('cite.' + k in destinations[filename] for k in keys))
        counts = Counter()
        visible = defaultdict(list)
        bounds_ok = True
        for page in pdf:
            bounds_ok &= all(w[0] >= 0 and w[1] >= 0 and w[2] <= page.rect.width and w[3] <= page.rect.height for w in page.get_text('words'))
            for link in page.get_links():
                counts[str(link['kind'])] += 1
                if link.get('file'):
                    target, _, dest = link['file'].partition('#nameddest=')
                    check('companion named destination ' + stem + ':' + dest, target in destinations and dest in destinations[target])
                elif link.get('nameddest'):
                    check('internal named destination ' + stem + ':' + link['nameddest'], link['nameddest'] in destinations[filename])
                elif link['kind'] == pymupdf.LINK_GOTO:
                    check('internal page destination ' + stem, 0 <= link['page'] < len(pdf))
                elif link.get('uri'):
                    visible[link['uri']].append(page.get_textbox(link['from']))
        check('all text within page bounds ' + stem, bounds_ok)
        old_uris = {l['uri'] for page in before for l in page.get_links() if l.get('uri')}
        check('exact external destination set ' + stem, old_uris == set(visible))
        if stem == 'main':
            for uri, pieces in visible.items():
                if not uri.startswith('https://') or 'orcid.org' in uri:
                    continue
                printed = re.sub(r'\s+', '', ''.join(pieces)).rstrip('.,;')
                expected = 'f94012872211ab6a828147bb138fbe8ab695b982' if '/tree/f940' in uri else uri
                check('printed URL/hash matches annotation ' + uri, printed == expected)
        else:
            check('complete extracted supplement text unchanged', text_without_pagination(before) == text_without_pagination(pdf))
            check('every supplement page renders identically', len(before) == len(pdf) and
                  all(a.get_pixmap().samples == b.get_pixmap().samples for a, b in zip(before, pdf)))
        result['pdfs'][filename] = {'pages_before_after': [len(before), len(pdf)], 'cited_references': len(keys),
            'links_by_kind': dict(counts), 'sha256': hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest(),
            'warnings': re.findall(r'^.*Warning.*$', log, re.M), 'underfull_notices': log.count('Underfull')}
    with zipfile.ZipFile(SOURCE / 'JDSA_Stage3_Submission_Source.zip') as archive:
        check('ZIP CRC', archive.testzip() is None)
        check('ZIP members match edited source/artifacts', all('/' not in n and archive.read(n) == (SOURCE / n).read_bytes() for n in archive.namelist()))
    for line in (SOURCE / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split('  ', 1)
        check('checksum ' + name, hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest)
    result.update(status='PASS', checks_passed=len(checks), checks=checks)
    (REPORT / 'VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS:', len(checks), 'editorial preservation and build checks')


if __name__ == '__main__':
    main()
