"""Verify Phase 3 presentation against Phase 2 without running scientific code.

Usage: python3 verify_phase3.py BUILD_DIRECTORY INDEPENDENT_ZIP_OUTPUT_DIRECTORY
Requires PyMuPDF and the existing Git baseline; writes VALIDATION.json and CHANGES.json.
"""
from collections import Counter, defaultdict
from fractions import Fraction
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
BASE = '70ff74c79628bdf5df2ae7e7375a0a0c3c6be240'
MAIN = 'Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf'
FLOAT = r'\\begin\{(table\*?|figure\*?|algorithm\*?)\}.*?\\end\{\1\}'
checks = []


def check(name, condition):
    checks.append({'check': name, 'pass': bool(condition)})
    if not condition:
        raise AssertionError(name)


def baseline(name):
    return subprocess.check_output(['git', 'show', BASE + ':manuscript/jdsa-stage3/' + name], cwd=ROOT)


def floats(text):
    return {re.search(r'\\label\{([^}]+)\}', m.group())[1]: m.group()
            for m in re.finditer(FLOAT, text, re.S)}


def rows(block):
    body = block.split(r'\begin{tabular}', 1)[1].split(r'\end{tabular}', 1)[0]
    result = []
    for line in body.splitlines():
        if '&' not in line or not re.search(r'&\s*(?:\$|\\textbf\{|\\pairedcounts\{)?\d', line):
            continue
        cells = line.strip().removesuffix(r'\\').split('&')
        numeric_start = 2 if cells[0].strip() in {'Typed', 'Rich'} else 1
        label = ' / '.join(c.strip() for c in cells[:numeric_start])
        label = label.replace('Value inaccurate', 'Stored-value disagreement')
        values, bold = [], []
        for cell in cells[numeric_start:]:
            bold.append(r'\textbf{' in cell)
            cell = re.sub(r'\\textbf\{([^}]+)\}', r'\1', cell.strip())
            cell = re.sub(r'\\pairedcounts\{(\d+)\}\{(\d+)\}', r'\1→\2', cell)
            values.append(cell.replace('$', '').replace(r'\to', '→').strip())
        result.append({'row': label, 'values': values, 'bold': bold})
    return result


def pdf_numeric_rows(page, x0=0, x1=1000):
    groups = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                if not (x0 <= span['bbox'][0] < x1):
                    continue
                text = span['text'].strip()
                if not re.fullmatch(r'[\d. /→]+', text or 'x') or not re.search(r'\d', text):
                    continue
                y = span['origin'][1]
                group = next((g for g in groups if abs(g[0] - y) < 0.8), None)
                if group is None:
                    group = [y, []]
                    groups.append(group)
                group[1].append(span)
    result = []
    for y, spans in groups:
        tokens = []
        for span in sorted(spans, key=lambda s: s['bbox'][0]):
            tokens.extend((v, 'BX' in span['font'] or 'Bold' in span['font'])
                          for v in re.findall(r'\d+(?:\.\d+)?(?:/\d+)?', span['text']))
        result.append((y, tokens))
    return result


def main():
    build, independent = map(Path, sys.argv[1:3])
    old, new = baseline('main.tex').decode(), (SOURCE / 'main.tex').read_text()
    before, after = floats(old), floats(new)
    check('same eight floats and labels', before.keys() == after.keys() and len(after) == 8)
    check('all prose, equations, abstract, declarations and preamble outside floats byte-identical',
          re.sub(FLOAT, '', old, flags=re.S) == re.sub(FLOAT, '', new, flags=re.S))
    check('algorithm Require, Ensure, all 17 lines and indentation byte-identical',
          re.search(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', old, re.S)[0] ==
          re.search(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', new, re.S)[0])
    for pattern, name in [(r'\\(?:cite|ref|eqref)\{[^}]+\}', 'citations and cross-references'),
                          (r'\\(?:url|href|path|nolinkurl)\{[^}]+\}', 'URLs and pinned identifiers'),
                          (r'\\label\{[^}]+\}', 'labels')]:
        check('exact sequence of ' + name, re.findall(pattern, old) == re.findall(pattern, new))
    figure_old = baseline('pipeline_figure.tex').decode()
    figure_new = (SOURCE / 'pipeline_figure.tex').read_text()
    check('seven figure node texts unchanged',
          re.findall(r'\\node.*? at \([^\n]+?\) (\{.*\});', figure_old) ==
          re.findall(r'\\node.*? at \([^\n]+?\) (\{.*\});', figure_new))
    check('six directed figure dependencies and dashed assistance unchanged',
          re.findall(r'\\draw.*;', figure_old) == re.findall(r'\\draw.*;', figure_new))
    for path in SOURCE.iterdir():
        if path.suffix in {'.tex', '.bib', '.bbl', '.cls', '.sty', '.bst'} and path.name not in {'main.tex', 'pipeline_figure.tex'}:
            check('unchanged protected source ' + path.name, path.read_bytes() == baseline(path.name))
    pdfs = {name: pymupdf.open(SOURCE / name) for name in [MAIN, 'ESM_1.pdf']}
    destinations = {name: pdf.resolve_names() for name, pdf in pdfs.items()}
    table_report = []
    pages = [7, 8, 8, 9, 10, 10]  # Zero-based locations checked against the rendered PDF.
    for number, label in enumerate((k for k in after if k.startswith('tab:')), 1):
        a, b = rows(before[label]), rows(after[label])
        check(f'Table {number}: every cell, precision and row order unchanged',
              [(r['row'], r['values']) for r in a] == [(r['row'], r['values']) for r in b])
        expected = set()
        if number in {1, 2, 5}:
            candidates = [(i, row) for i, row in enumerate(b) if number != 2 or not row['row'].startswith('Oracle')]
            for col in range(len(b[0]['values'])):
                best = max(Fraction(row['values'][col]) for _, row in candidates)
                expected.update((i, col) for i, row in candidates if Fraction(row['values'][col]) == best)
            check(f'Table {number}: caption describes descriptive highlighting and significance boundary',
                  'Bold marks' in after[label] and 'not statistical significance' in after[label])
        actual = {(i, j) for i, row in enumerate(b) for j, bold in enumerate(row['bold']) if bold}
        check(f'Table {number}: all and only eligible maxima bold (all ties included)', actual == expected)
        x0, x1 = (300, 1000) if number == 1 else (0, 290) if number == 3 else (0, 1000)
        rendered = pdf_numeric_rows(pdfs[MAIN][pages[number - 1]], x0, x1)
        for row in b:
            expected_tokens = [(token, bold) for value, bold in zip(row['values'], row['bold'])
                               for token in re.findall(r'\d+(?:\.\d+)?(?:/\d+)?', value)]
            check(f'Table {number}: PDF values and font weight for {row["row"]}',
                  any(tokens == expected_tokens for _, tokens in rendered))
        if number == 4:
            check('Table 4: each mutually exclusive partition totals 331',
                  all(sum(int(row['values'][j]) for row in b[:-1]) == int(b[-1]['values'][j]) == 331 for j in range(4)))
        table_report.append({'number': number, 'label': label, 'page': pages[number - 1]+1, 'rows': b})
    changes = [{'label': label, 'before': before[label], 'after': after[label]} for label in after]
    changes.append({'file': 'pipeline_figure.tex', 'before': figure_old, 'after': figure_new})
    (REPORT / 'CHANGES.json').write_text(json.dumps(changes, indent=2) + '\n')
    result = {'baseline': BASE, 'build_directory': str(build), 'tables': table_report, 'pdfs': {}}
    for filename, pdf in pdfs.items():
        stem = 'main' if filename == MAIN else 'supplement'
        old_pdf = pymupdf.open(stream=baseline(filename), filetype='pdf')
        check('independent source-ZIP build byte identity ' + stem,
              (SOURCE / filename).read_bytes() == (independent / filename).read_bytes())
        log = (build / (stem + '.log')).read_text()
        blg = (build / (stem + '.blg')).read_text()
        check('no unresolved/overfull LaTeX diagnostics ' + stem,
              not re.search(r'undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull|^!', log, re.M))
        check('clean BibTeX log ' + stem, 'Warning--' not in blg and 'error message' not in blg)
        aux = (build / (stem + '.aux')).read_text()
        cited = {k for group in re.findall(r'\\citation\{([^}]+)\}', aux) for k in group.split(',')}
        keys = re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}', (build / (stem + '.bbl')).read_text(), re.S)
        check('all bibliography entries cited and linked ' + stem,
              set(keys) == cited and all('cite.' + k in destinations[filename] for k in keys))
        counts, visible = Counter(), defaultdict(list)
        for page in pdf:
            check(f'all text within page bounds {stem}:{page.number+1}',
                  all(w[0] >= 0 and w[1] >= 0 and w[2] <= page.rect.width and w[3] <= page.rect.height for w in page.get_text('words')))
            for link in page.get_links():
                counts[str(link['kind'])] += 1
                if link.get('file'):
                    target, _, dest = link['file'].partition('#nameddest=')
                    check('companion destination ' + stem + ':' + dest, target in destinations and dest in destinations[target])
                elif link.get('nameddest'):
                    check('internal destination ' + stem + ':' + link['nameddest'], link['nameddest'] in destinations[filename])
                elif link['kind'] == pymupdf.LINK_GOTO:
                    check('internal page destination ' + stem, 0 <= link['page'] < len(pdf))
                elif link.get('uri'):
                    visible[link['uri']].append(page.get_textbox(link['from']))
        old_uris = {l['uri'] for page in old_pdf for l in page.get_links() if l.get('uri')}
        check('exact external destination set ' + stem, old_uris == set(visible))
        if stem == 'main':
            for uri, pieces in visible.items():
                if uri.startswith('https://') and 'orcid.org' not in uri:
                    printed = re.sub(r'\s+', '', ''.join(pieces)).rstrip('.,;')
                    expected = 'f94012872211ab6a828147bb138fbe8ab695b982' if '/tree/f940' in uri else uri
                    check('printed URL/hash matches annotation ' + uri, printed == expected)
        else:
            check('complete supplement PDF byte-identical to Phase 2', (SOURCE / filename).read_bytes() == baseline(filename))
        result['pdfs'][filename] = {'pages_before_after': [len(old_pdf), len(pdf)], 'cited_references': len(keys),
            'links_by_kind': dict(counts), 'sha256': hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest(),
            'warnings': re.findall(r'^.*Warning.*$', log, re.M), 'underfull_notices': log.count('Underfull')}
    for filename in ['Fig1.pdf', 'main.bbl', 'supplement.bbl']:
        check('independent build byte identity ' + filename,
              (SOURCE / filename).read_bytes() == (independent / filename).read_bytes())
    with zipfile.ZipFile(SOURCE / 'JDSA_Stage3_Submission_Source.zip') as archive:
        check('ZIP CRC', archive.testzip() is None)
        check('all ZIP members match current source/artifacts', all('/' not in n and archive.read(n) == (SOURCE / n).read_bytes() for n in archive.namelist()))
        result['zip_members'] = len(archive.namelist())
    for line in (SOURCE / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split('  ', 1)
        check('checksum ' + name, hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest)
    result.update(status='PASS', checks_passed=len(checks), checks=checks)
    (REPORT / 'VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS:', len(checks), 'presentation, scientific-preservation and build checks')


if __name__ == '__main__':
    main()
