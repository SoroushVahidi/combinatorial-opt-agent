"""NLP4LP access for the PaMOP reproduction, scoped to safe subsets only.

Background (docs/PAMOP_REPRODUCTION_PLAN.md section 13): PaMOP (IJCAI 2025)
cites a "54 LP + 13 MILP, 67 instances" NLP4LP release dated 2024-05-13. That
exact release is not identifiable inside the `udell-lab/NLP4LP` Hugging Face
snapshot this repository has access to -- the snapshot has grown far past it
(361 problems as of this investigation) and does not preserve the original
numbering. What IS known:

  * HF problem ids 1-269 existed continuously since the dataset's initial HF
    upload (2024-11-02), predating PaMOP's Aug-2025 publication -- so PaMOP's
    67 problems, whatever they are, can only be drawn from this range.
  * HF problem ids 270-361 were added 2026-02-12 through 2026-02-27, six
    months AFTER PaMOP was published, and provably cannot be part of its
    evaluation set.

This module therefore only exposes one subset, ``pamop_possible_269``: the
269 structurally-possible ids. It is deliberately never called or aliased
``pamop_67`` anywhere -- that name would misrepresent an unverified superset
as PaMOP's confirmed exact evaluation set. See
``data/baselines/pamop/nlp4lp_pamop_subset.jsonl`` and
``tools/pamop_nlp4lp_alignment.py`` for the underlying analysis this module
encodes.

No Hugging Face token value is ever read into a variable that gets printed,
logged, or returned by this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "udell-lab/NLP4LP"

SUBSET_POSSIBLE_269 = "pamop_possible_269"
_VALID_SUBSETS = (SUBSET_POSSIBLE_269,)

# HF ids 1..PRE_PAMOP_HF_ID_MAX existed before PaMOP's IJCAI 2025 publication.
PRE_PAMOP_HF_ID_MAX = 269
# HF ids >= this were added 2026-02-12 or later -- provably post-publication.
POST_PAMOP_HF_ID_MIN = 270


class UnknownSubsetError(ValueError):
    pass


class PostPamopIdError(ValueError):
    """Raised when code tries to treat a post-2026-02 NLP4LP id as PaMOP-relevant."""


class MissingStructuredDataError(FileNotFoundError):
    """Raised when an NLP4LP problem id has no ``problem_info.json`` anywhere.

    As of this investigation, 6 of the 269 pre-PaMOP ids (28, 51, 57, 123,
    126, 135) have a problem directory (description.txt, metadata.json,
    optimus-code.py) but genuinely no ``problem_info.json`` -- these appear
    to be the dataset's own historical "-infeasible"/"-unsolved" entries,
    renamed back to bare numeric ids by the upstream "Final revision
    cleanup" commit (2026-04-20) without ever having had a structured
    formulation generated. This is not a loader bug to route around
    silently: there is no structured record to recover, so this error is
    raised distinctly (not a generic 404) precisely so callers can count and
    report it rather than mistake it for an auth/network failure.
    """


def list_ids_for_subset(subset: str) -> list[int]:
    """Return the HF NLP4LP problem ids belonging to ``subset``."""
    if subset not in _VALID_SUBSETS:
        raise UnknownSubsetError(
            f"unknown NLP4LP subset {subset!r}; valid subsets: {_VALID_SUBSETS}. "
            f"(Note: there is deliberately no 'pamop_67' subset -- PaMOP's exact "
            f"67-problem set is unresolved, see docs/PAMOP_REPRODUCTION_PLAN.md "
            f"section 13.)"
        )
    return list(range(1, PRE_PAMOP_HF_ID_MAX + 1))


def assert_not_post_pamop(problem_id: int) -> None:
    """Raise ``PostPamopIdError`` if ``problem_id`` postdates PaMOP's publication."""
    if problem_id >= POST_PAMOP_HF_ID_MIN:
        raise PostPamopIdError(
            f"NLP4LP problem id {problem_id} was added to the dataset on or "
            f"after 2026-02-12, six months after PaMOP (IJCAI 2025, Aug 2025) "
            f"was published. It must not be used in any attempt to approximate "
            f"PaMOP's evaluation set. See docs/PAMOP_REPRODUCTION_PLAN.md "
            f"section 13."
        )


def _get_hf_token() -> str:
    """Read an HF token from the environment. Never returned to a caller that logs it."""
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return ""


def load_alignment_manifest() -> list[dict[str, Any]]:
    """Load the id/index mapping manifest built by tools/pamop_nlp4lp_alignment.py.

    Contains only ids, indices, and structural provenance -- no gated problem
    text. See data/baselines/pamop/nlp4lp_pamop_subset.jsonl.
    """
    import json

    manifest_path = ROOT / "data" / "baselines" / "pamop" / "nlp4lp_pamop_subset.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run "
            f"`python tools/pamop_nlp4lp_alignment.py` first."
        )
    with manifest_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _resolve_problem_info_path(problem_id: int, token: str | None) -> str:
    """Return a downloaded local path to ``problem_id``'s ``problem_info.json``.

    Tries the bare numeric id first (the current, cleaned-up snapshot layout
    -- verified live as of this investigation to cover all 361 problem
    directories). Falls back to listing ``data/`` for a suffixed variant
    (``{id}-infeasible``, ``{id}-unsolved``, ...) in case an older or future
    snapshot reintroduces that layout -- this is the "handle suffixed ids
    robustly" half of the fix. Raises ``MissingStructuredDataError`` only
    after both the bare id and every suffix variant have been checked and
    none has a ``problem_info.json``.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(
            DATASET_ID,
            f"data/{problem_id}/problem_info.json",
            repo_type="dataset",
            token=token,
        )
    except EntryNotFoundError:
        pass  # bare id has no problem_info.json -- try suffixed variants below

    api = HfApi(token=token)
    all_files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    prefix = f"data/{problem_id}-"
    suffixed_dirs = sorted(
        {f.split("/")[1] for f in all_files if f.startswith(prefix)}
    )
    for dirname in suffixed_dirs:
        try:
            return hf_hub_download(
                DATASET_ID,
                f"data/{dirname}/problem_info.json",
                repo_type="dataset",
                token=token,
            )
        except EntryNotFoundError:
            continue

    raise MissingStructuredDataError(
        f"NLP4LP problem id {problem_id} has no problem_info.json under "
        f"data/{problem_id}/ or any suffixed variant "
        f"({suffixed_dirs or 'none found'}). See MissingStructuredDataError "
        f"docstring for known affected ids."
    )


def _resolve_problem_file_path(problem_id: int, filename: str, token: str | None) -> str:
    """Return a downloaded local path to a problem file.

    Uses the same bare-id then suffixed-directory lookup as
    ``_resolve_problem_info_path``. This is used for raw prompt text such as
    ``description.txt``; callers must not write returned content to committed
    artifacts.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(
            DATASET_ID,
            f"data/{problem_id}/{filename}",
            repo_type="dataset",
            token=token,
        )
    except EntryNotFoundError:
        pass

    api = HfApi(token=token)
    all_files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    prefix = f"data/{problem_id}-"
    suffixed_dirs = sorted(
        {f.split("/")[1] for f in all_files if f.startswith(prefix)}
    )
    for dirname in suffixed_dirs:
        try:
            return hf_hub_download(
                DATASET_ID,
                f"data/{dirname}/{filename}",
                repo_type="dataset",
                token=token,
            )
        except EntryNotFoundError:
            continue

    raise FileNotFoundError(
        f"NLP4LP problem id {problem_id} has no {filename!r} under "
        f"data/{problem_id}/ or any suffixed variant "
        f"({suffixed_dirs or 'none found'})."
    )


def load_problem_record(problem_id: int) -> dict[str, Any]:
    """Fetch one NLP4LP problem's structured fields (gated; requires HF access).

    Returns the ``problem_info.json`` content for ``problem_id``, resolving
    bare-numeric and suffixed (``{id}-infeasible``, ``{id}-unsolved``, ...)
    directory layouts transparently (see ``_resolve_problem_info_path``).
    Raises ``PostPamopIdError`` before attempting any network access if the
    id is known to postdate PaMOP's publication, and
    ``MissingStructuredDataError`` if no ``problem_info.json`` exists for
    this id under any layout.
    """
    assert_not_post_pamop(problem_id)

    token = _get_hf_token() or None  # huggingface_hub reads HF_TOKEN itself if None
    path = _resolve_problem_info_path(problem_id, token)

    import json

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_problem_text(problem_id: int) -> str:
    """Fetch the original NLP4LP ``description.txt`` for prompt input.

    This gated raw problem text is loaded only in memory. It must not be
    written to committed traces, reports, or result artifacts.
    """
    assert_not_post_pamop(problem_id)

    token = _get_hf_token() or None
    path = _resolve_problem_file_path(problem_id, "description.txt", token)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def iter_subset(subset: str = SUBSET_POSSIBLE_269, limit: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(problem_id, record)`` pairs for every id in ``subset``, in order.

    Requires network + HF auth. Each record is a gated NLP4LP problem_info
    payload -- callers must not write it to a committed file. Does NOT catch
    per-problem fetch failures -- callers that need a failure count (e.g. the
    smoke-run script) should wrap each ``load_problem_record`` call
    themselves so failures are visible rather than silently skipped.
    """
    ids = list_ids_for_subset(subset)
    if limit is not None:
        ids = ids[:limit]
    for problem_id in ids:
        yield problem_id, load_problem_record(problem_id)
