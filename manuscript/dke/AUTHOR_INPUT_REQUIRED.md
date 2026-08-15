# AUTHOR-INPUT REQUIRED -- DKE Submission (manuscript/dke/)

This file lists every item that requires input or confirmation from the author
before the `manuscript/dke/` elsarticle version can be submitted to **Data &
Knowledge Engineering**. Everything in this file is deliberately **not
fabricated** in `main.tex`; it must be supplied by the author.

## 1. Author biography (required by DKE; currently a TODO in main.tex)

DKE requires a short author biography (vitae) of **at most 100 words**,
accompanied by a **recent passport-style photo** (common formats: JPEG/PNG,
square, high resolution). Neither is present in `main.tex`. The submission
portal expects these as separate uploads; they do **not** need to live in the
LaTeX source.

TODO (author):
- Write a <=100-word biography for Soroush Vahidi.
- Provide a recent photo file (name it e.g. `author_photo.jpg`).
- Confirm whether to embed via `\authorbio` (CAS-style) or upload separately
  in the DKE submission system.

## 2. CRediT author-contribution statement

`main.tex` Declarations currently states the CRediT roles of the sole author
(Conceptualization, Methodology, Software, Validation, Formal analysis,
Investigation, Data curation, Writing -- original draft, Writing -- review &
editing, Visualization).

TODO (author): confirm these roles are accurate, or edit the CRediT line
before submission.

## 3. Data-deposit statement (Elsevier Option C)

`main.tex` currently maps the data-availability policy to **Option C** (data
available via a repository link, with the third-party gated NLP4LP dataset not
redistributed). Confirm this classification is what you intend for DKE, or
select a different Elsevier option before submission.

## 4. Funding statement

`main.tex` states "This research received no specific grant from any funding
agency in the public, commercial, or not-for-profit sectors."

TODO (author): confirm this is still accurate.

## 5. Generative-AI disclosure wording

The manuscript includes a "Note on generative-AI assistance" in the
methodology section and mentions the use of ChatGPT/Gemini (writing) and
Cursor/GitHub Copilot (coding).

TODO (author): confirm the wording matches Elsevier/DKE disclosure guidance at
submission time, since these policies evolve.

## 6. Author metadata to confirm

- Name: Soroush Vahidi (sole author, corresponding author)
- Affiliation: Ying Wu College of Computing, New Jersey Institute of
  Technology, University Heights, Newark, NJ 07102-1982, USA
- Email: sv96@njit.edu
- ORCID: 0000-0003-1934-6282

TODO (author): verify the affiliation string, postal address, email, and ORCID
are correct before submission.

## 7. Highlights file

`highlights.txt` contains 5 bullets (all <=85 characters). DKE requires 3--5
bullets. Confirm the wording; adjust if needed.

## 8. Double-column / submission format

`main.tex` is built on `elsarticle` with the `3p` (double-column) option, per
the DKE guide note that LaTeX submissions should be double-column. Confirm
this is the format DKE expects for LaTeX submissions (some Elsevier journals
prefer a single-column `review` option for the review stage; DKE's guide text
currently points to double-column).

## 9. Abstract / keywords

- Abstract: 227 words (DKE limit: 250). Confirm no references and standalone.
- Keywords: 6 English keywords (limit 1--7): natural language processing,
  optimization modeling, knowledge representation, information retrieval,
  semantic grounding, intelligent information systems.

## 10. External-baseline limitation wording

The Limitations section now mentions a preliminary common-18 external-baseline
assessment (PaMOP and OptMATH complete, a generic LLM completes 16/18, ORLM's
checkpoint blocked on coptpy, DeepOR/OR-R1 without released checkpoints). TODO
(author): confirm this wording is acceptable and consistent with the
`results/external_baseline_comparison/` provenance docs before submission.