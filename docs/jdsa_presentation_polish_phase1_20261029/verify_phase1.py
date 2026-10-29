"""Verify Phase 1 presentation changes without running scientific code.

Usage: python3 verify_phase1.py BUILD_DIRECTORY INDEPENDENT_PACKAGE_DIRECTORY
Requires PyMuPDF and the existing pdfLaTeX/BibTeX build outputs.
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
BASE = '1e90184923c753ed18652951f2037525fffad383'
MAIN = 'Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf'
checks = []


def check(name, condition):
    checks.append({'check': name, 'pass': bool(condition)})
    if not condition:
        raise AssertionError(name)


def baseline(name):
    return subprocess.check_output(['git', 'show', BASE + ':manuscript/jdsa-stage3/' + name], cwd=ROOT)


def scientific_source(text):
    # The only prose-bearing lines authorized for editing are the postal address
    # and the three location paragraphs; all remaining manuscript text is exact.
    editable = (r'\affil', 'The public ', r'\noindent\textbf{Data availability.}',
                r'\noindent\textbf{Code availability.}', r'\urlstyle{tt}',
                r'\providecommand{\urlprefix}{}')
    return '\n'.join(line for line in text.splitlines()
                     if line.strip() and not line.startswith(('%', *editable)))


def document_text(pdf):
    return '\n'.join('\n'.join(line for line in page.get_text().splitlines()
                               if line.strip() != str(page.number + 1)) for page in pdf)


def normalize(text):
    return re.sub(r'\s+', '', re.sub(r'[-\u00ad]\n', '', text))


def main():
    build, independent = map(Path, sys.argv[1:3])
    result = {'baseline': BASE, 'build_directory': str(build), 'checks': checks, 'pdfs': {}}
    before = baseline('main.tex').decode()
    after = (SOURCE / 'main.tex').read_text()
    check('scientific manuscript source outside the four allowed lines is byte-identical',
          scientific_source(before) == scientific_source(after))
    for env in ['equation', 'align', 'tabular', 'tabularx', 'algorithm', 'algorithmic']:
        pattern = r'\\begin\{' + env + r'\*?\}.*?\\end\{' + env + r'\*?\}'
        check('unchanged ' + env + ' environments', re.findall(pattern, before, re.S) == re.findall(pattern, after, re.S))
    for token in [r'\cite', r'\caption', r'\abstract', r'\keywords', r'\author']:
        check('unchanged source lines containing ' + token,
              [x for x in before.splitlines() if token in x] == [x for x in after.splitlines() if token in x])
    for path in SOURCE.iterdir():
        if path.suffix in {'.tex', '.cls', '.sty', '.bst'} and path.name != 'main.tex':
            check('unchanged supplementary/template source ' + path.name, path.read_bytes() == baseline(path.name))
    for marker in [' package supports', ' publishes']:
        old = next(x for x in before.splitlines() if marker in x and ('The public' in x or 'Data availability' in x))
        new = next(x for x in after.splitlines() if marker in x and ('The public' in x or 'Data availability' in x))
        check('unchanged evidence qualifications following ' + marker, old.split(marker, 1)[1] == new.split(marker, 1)[1])
    abstract = re.search(r'\\abstract\{(.*?)\}\n', after).group(1)
    keywords = re.search(r'\\keywords\{(.*?)\}', after).group(1).split(',')
    result['front_matter'] = {'abstract_words': len(abstract.split()), 'keywords': len(keywords)}
    check('JDSA abstract and keyword limits', 150 <= len(abstract.split()) <= 250 and 4 <= len(keywords) <= 6)
    check('existing two-column Springer class retained', r'\documentclass[pdflatex,iicol,sn-basic,Numbered]{sn-jnl}' in after)

    pdfs = {name: pymupdf.open(SOURCE / name) for name in [MAIN, 'ESM_1.pdf']}
    names = {name: pdf.resolve_names() for name, pdf in pdfs.items()}
    for filename, pdf in pdfs.items():
        stem = 'main' if filename == MAIN else 'supplement'
        old = pymupdf.open(stream=baseline(filename), filetype='pdf')
        check('independent package build matches ' + stem, (SOURCE / filename).read_bytes() == (independent / filename).read_bytes())
        check('page count preserved ' + stem, len(old) == len(pdf))
        log = (build / (stem + '.log')).read_text()
        blg = (build / (stem + '.blg')).read_text()
        check('clean final LaTeX log ' + stem, not re.search(r'undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull|^!', log, re.M))
        check('clean BibTeX log ' + stem, 'Warning--' not in blg and 'error message' not in blg)
        aux = (build / (stem + '.aux')).read_text()
        cited = set(k for line in re.findall(r'\\citation\{([^}]+)\}', aux) for k in line.split(','))
        bbl = (build / (stem + '.bbl')).read_text()
        keys = re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}', bbl, re.S)
        old_keys = re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}', baseline(stem + '.bbl').decode(), re.S)
        check('all printed references cited ' + stem, cited == set(keys))
        check('citation identities and numbering preserved ' + stem, keys == old_keys)
        check('every reference has a PDF destination ' + stem, all('cite.' + k in names[filename] for k in keys))
        links = Counter()
        visible = defaultdict(list)
        for page in pdf:
            for word in page.get_text('words'):
                check('text within page bounds ' + stem + ':' + str(page.number + 1),
                      word[0] >= 0 and word[1] >= 0 and word[2] <= page.rect.width and word[3] <= page.rect.height)
            for link in page.get_links():
                links[str(link['kind'])] += 1
                if link.get('file'):
                    target, _, dest = link['file'].partition('#nameddest=')
                    check('companion link ' + stem + ':' + dest, target in names and dest in names[target])
                elif link.get('nameddest'):
                    check('internal link ' + stem + ':' + link['nameddest'], link['nameddest'] in names[filename])
                elif link['kind'] == pymupdf.LINK_GOTO:
                    check('internal page link ' + stem, 0 <= link['page'] < len(pdf))
                elif link.get('uri'):
                    visible[link['uri']].append(page.get_textbox(link['from']))
        if stem == 'main':
            for uri, pieces in visible.items():
                if not uri.startswith('https://') or 'orcid.org' in uri:
                    continue
                printed = re.sub(r'\s+', '', ''.join(pieces)).rstrip('.,;')
                expected = 'f94012872211ab6a828147bb138fbe8ab695b982' if '/tree/f940' in uri else uri
                check('printed URL/hash agrees with PDF annotation ' + uri, printed == expected)
            old_uris = {link['uri'] for page in old for link in page.get_links() if link.get('uri')}
            check('all original external destinations retained', old_uris <= set(visible))
            a, b = normalize(document_text(old)), normalize(document_text(pdf))
            for start, end in [('7Conclusion', 'Dataavailability.'),
                               ('Reproducibility.', 'References')]:
                check('extracted protected text from ' + start,
                      a[a.index(start):a.index(end, a.index(start))] ==
                      b[b.index(start):b.index(end, b.index(start))])
            a = a[a.index('Acknowledgements.'):a.index('6.4Reanalysis')]
            b = b[b.index('Acknowledgements.'):b.index('6.4Reanalysis')]
            check('extracted scientific text through Section 6.3 unchanged', a == b)
            check('extracted scientific numerical sequence unchanged', re.findall(r'\d+', a) == re.findall(r'\d+', b))
            check('Newark postal address rendered correctly', 'Newark, NJ 07102-1982, USA' in pdf[0].get_text())
        else:
            check('all extracted supplementary scientific text unchanged',
                  normalize(document_text(old).split('\nReferences\n')[0]) == normalize(document_text(pdf).split('\nReferences\n')[0]))
        result['pdfs'][filename] = {
            'pages': len(pdf), 'sha256': hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest(),
            'printed_references_all_cited': len(keys), 'link_annotation_counts_by_kind': dict(links),
            'changed_render_pages': [i + 1 for i in range(len(pdf)) if old[i].get_pixmap().samples != pdf[i].get_pixmap().samples],
            'warnings': re.findall(r'^.*Warning.*$', log, re.M), 'underfull_notices': log.count('Underfull'),
            'external_destinations': sorted(visible),
        }
    with zipfile.ZipFile(SOURCE / 'JDSA_Stage3_Submission_Source.zip') as archive:
        check('source ZIP CRC', archive.testzip() is None)
        for name in archive.namelist():
            check('source ZIP member matches ' + name, '/' not in name and archive.read(name) == (SOURCE / name).read_bytes())
    for line in (SOURCE / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split('  ', 1)
        check('artifact checksum ' + name, hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest)
    # Summarize repeated page-bound checks to keep the committed report readable.
    result['checks_passed'] = len(checks)
    result['checks'] = [c for c in checks if not c['check'].startswith('text within page bounds')]
    result['status'] = 'PASS'
    (REPORT / 'VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS:', result['checks_passed'], 'presentation/preservation checks')


if __name__ == '__main__':
    main()
