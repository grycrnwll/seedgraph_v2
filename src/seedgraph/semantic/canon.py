"""Deterministic guardrails — the disposing half of the canonicalizer.

The LLM only *proposes* (see :mod:`.llm_propose`); these three deterministic
guardrails *dispose*, so the worst failure is benign under-merge, never
over-merge (doc 07 §12; D10):

1. **closure** — a proposed canonical/umbrella must actually be present among the
   observed labels; invented umbrellas are rejected.
2. **token-overlap / acronym precondition** — members must share a token-set
   Jaccard ≥ τ OR be an acronym expansion of one another.
3. **discriminating-token veto** — a frozen stoplist of version/scale/variant
   tokens (:data:`DISCRIMINATING_TOKENS`) that, when they differ between members,
   vetoes the fold. This is what keeps ``BERT-base`` ≠ ``BERT-large`` and
   ``SQuAD v1.1`` ≠ ``SQuAD v2.0`` (the anti-overmerge acceptance, D10).

Plan implementation step 4. Cluster membership is a *transient build artifact*:
no ``ConceptCluster`` table, no ``cluster_id`` column (deliberate doc-07 §9
deviation, decisions 34/56).
"""

from __future__ import annotations

from .merge import is_acronym_expansion, normalize, token_jaccard

#: Default token-overlap threshold the precondition guardrail applies when
#: :func:`passes_guardrails` is called without a clustering τ in scope.
GUARD_TAU: float = 0.6

#: Frozen discriminating-token veto stoplist. Tokens here, when they differ
#: across candidate members, always block a fold (guardrail 3). Extend only with
#: care — this list is the anti-overmerge backbone (D10). The veto ALWAYS wins.
#: Tokens are compared post-``merge.normalize``, which strips dots — so dotted
#: entries like ``1.1``/``2.0`` could NEVER match a normalized token (dead
#: entries; D8). Instead the BARE numerics are listed: ``SQuAD 1.1`` /
#: ``SQuAD 2.0`` normalize to ``squad 1 1`` / ``squad 2 0``, whose differing
#: ``1``/``2``/``0`` tokens veto the fold; ``v``-prefixed forms are covered by
#: the ``v1``…``v4`` entries.
DISCRIMINATING_TOKENS: frozenset[str] = frozenset(
    {
        "v1",
        "v2",
        "v3",
        "v4",
        "base",
        "large",
        "small",
        "xl",
        "xxl",
        "mini",
        "tiny",
        "encoder",
        "decoder",
        "cased",
        "uncased",
        "static",
        "dynamic",
        "1",
        "2",
        "3",
        "4",
        "0",
        "11",
        "i",
        "ii",
        "iii",
    }
)


def discriminating_conflict(members: frozenset[str] | set[str]) -> bool:
    """True if a discriminating veto token is NOT shared by every member.

    A version/scale/variant token (``base``/``large``/``v1``/``v2``…) that appears
    in some members but not all is a hard signal the members are DISTINCT — the
    fold is vetoed (guardrail 3). This is the frozen anti-overmerge backbone.
    """
    members = set(members)
    if len(members) < 2:
        return False
    token_sets = [set(normalize(m).split()) for m in members]
    union: set[str] = set().union(*token_sets)
    intersection: set[str] = set(token_sets[0]).intersection(*token_sets[1:])
    differing = union - intersection
    return any(tok in DISCRIMINATING_TOKENS for tok in differing)


def candidate_clusters(labels: list[str], tau: float) -> list[frozenset[str]]:
    """Union-find candidate clustering of ``labels``.

    Two labels join the same cluster iff their token-set Jaccard ≥ ``tau`` OR one
    is an acronym expansion of the other (:func:`merge.is_acronym_expansion`).
    Returns the connected components as ``frozenset``s (singletons included).
    These are *candidates only* — :func:`passes_guardrails` still disposes each
    proposed fold. Pinned by ``test_concept_canon``.
    """
    uniq = list(dict.fromkeys(labels))  # de-dup, preserve order
    parent: dict[str, str] = {lbl: lbl for lbl in uniq}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # deterministic root: smaller label wins so output is order-stable
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            if (
                token_jaccard(a, b) >= tau
                or is_acronym_expansion(a, b)
                or is_acronym_expansion(b, a)
            ):
                union(a, b)

    groups: dict[str, set[str]] = {}
    for lbl in uniq:
        groups.setdefault(find(lbl), set()).add(lbl)
    return [frozenset(members) for members in groups.values()]


def passes_guardrails(
    canonical: str,
    members: frozenset[str],
    present_labels: set[str],
) -> bool:
    """Return whether folding ``members`` under ``canonical`` is allowed.

    The conjunction of all three deterministic guardrails:
    ``closure ∧ (token-overlap ∨ acronym) ∧ ¬discriminating-token-veto``.
    ``present_labels`` is the set of labels actually observed in the corpus (used
    by the closure check). Pinned by ``test_concept_overmerge_guard``.
    """
    members = set(members)
    if not members:
        return False
    # (1) closure: the canonical AND every folded member must be OBSERVED labels;
    # an invented umbrella never present in the corpus is rejected.
    if canonical not in present_labels:
        return False
    if not members.issubset(present_labels):
        return False
    # (3) discriminating-token veto — evaluated over members + canonical; ALWAYS
    # wins (a wrong LLM proposal degrades to under-merge / split, never over-merge).
    if discriminating_conflict(members | {canonical}):
        return False
    # (2) token-overlap / acronym precondition: every non-canonical member must
    # share τ token overlap with, OR be an acronym expansion of, the canonical.
    for member in members:
        if member == canonical:
            continue
        if not (
            token_jaccard(member, canonical) >= GUARD_TAU
            or is_acronym_expansion(member, canonical)
            or is_acronym_expansion(canonical, member)
        ):
            return False
    return True
