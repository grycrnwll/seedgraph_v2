"""Concept canonicalization orchestrator (plan §5/§6, implementation step 7).

Drives the AUTO-tier canonicalization that :func:`build_semantic_overlay`
delegates to: read distinct type-scoped ``normalized_label``s from CURRENT
``extracted_claims`` (latest note per ``(work_id, schema_id)`` / latest lens
run — superseded note generations contribute nothing; see
:data:`_CURRENT_CLAIMS_CTE`) → apply sticky ``concept_constraints`` (must/cannot-link)
BEFORE clustering → per-``concept_type`` :func:`canon.candidate_clusters` (the
D9 type fence: fuzzy + acronym tiers never span types; a cross-type
``must_link`` still merges — human decision outranks the fence) → optional
:class:`llm_propose.ConceptProposer` → :func:`canon.passes_guardrails` disposes →
relabel-then-one-merge → upsert ``concepts`` (preserving ``user_confirmed`` /
``user_split`` and inheriting ``concept_type`` from ``ClaimType``; ``definition``
sliced from a contributing claim) + ``concept_aliases`` → write
``claim_concepts`` → tier auto/provisional/review and enqueue borderline folds.

Three-tier gate (D10): exact-key + acronym folds **auto-apply** (``status=auto``);
LLM folds that pass the guardrails but are borderline are enqueued as
``concept_merge_candidate`` and the two concepts STAY SPLIT until resolved (plan
§4.3); obvious-safe folds never reach review. The anti-overmerge guard always
wins — a wrong/empty LLM response degrades to under-merge, never over-merge.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..vocab import EpistemicType, normalize_claim_type
from . import audit, canon, merge
from .access import resolve_access_class
from .llm_propose import ConceptProposer, NullProposer

if TYPE_CHECKING:
    from . import SemanticBuildReport


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


from .current import CURRENT_CLAIMS_CTE as _CURRENT_CLAIMS_CTE


@dataclass(frozen=True)
class ClaimLabel:
    """A distinct normalized claim label feeding canonicalization.

    One per distinct merge-key (``normalize(extracted_claims.normalized_label)``),
    carrying the contributing ``claim_id``/``work_id`` pairs used downstream for
    ``claim_concepts``, ``paper_frequency``, and ``access_class`` resolution.
    ``canonical_label`` is the field-typed surface form; ``concept_type`` is the
    majority ``ClaimType`` of the contributing claims; ``definition`` is a borrowed
    contributing-claim sentence (full-text-derived).
    """

    normalized_label: str
    canonical_label: str
    concept_type: str
    claim_ids: tuple[str, ...]
    work_ids: tuple[str, ...]
    claim_work_pairs: tuple[tuple[str, str], ...]
    definition: str | None


def gather_claim_labels(conn: sqlite3.Connection) -> list[ClaimLabel]:
    """Read distinct ``normalized_label``s from CURRENT ``extracted_claims``.

    One :class:`ClaimLabel` per distinct merge-key. Only ``found`` claims with a
    non-empty ``normalized_label`` are included (no concept is ever created with
    zero claim mentions, doc 07 §1), and only claims whose container is current
    (:data:`_CURRENT_CLAIMS_CTE`) — a superseded note generation (e.g. a
    prompt-version re-extraction) contributes nothing. ``canonical_label`` is the
    field-typed surface form (``field-typed-label-wins``); ``concept_type`` is
    the deterministic majority ``ClaimType``.
    """
    rows = conn.execute(
        _CURRENT_CLAIMS_CTE
        + "SELECT claim_id, work_id, claim_type, normalized_label, claim_text "
        "FROM current_claims "
        "WHERE status = 'found' AND normalized_label IS NOT NULL "
        "AND TRIM(normalized_label) <> ''"
    ).fetchall()

    acc: dict[str, dict] = {}
    for claim_id, work_id, claim_type, surface, claim_text in rows:
        key = merge.normalize(surface)
        if not key:
            continue
        bucket = acc.setdefault(
            key,
            {"pairs": [], "works": set(), "surfaces": [], "types": [], "texts": []},
        )
        bucket["pairs"].append((claim_id, work_id))
        bucket["works"].add(work_id)
        bucket["surfaces"].append(surface)
        bucket["types"].append(normalize_claim_type(claim_type))
        if claim_text:
            bucket["texts"].append(claim_text)

    labels: list[ClaimLabel] = []
    for key, bucket in acc.items():
        canonical = merge.field_typed_label(bucket["surfaces"], bucket["types"])
        concept_type = _majority(bucket["types"])
        definition = bucket["texts"][0] if bucket["texts"] else None
        claim_ids = tuple(cid for cid, _ in bucket["pairs"])
        work_ids = tuple(sorted(bucket["works"]))
        labels.append(
            ClaimLabel(
                normalized_label=key,
                canonical_label=canonical,
                concept_type=concept_type,
                claim_ids=claim_ids,
                work_ids=work_ids,
                claim_work_pairs=tuple(bucket["pairs"]),
                definition=definition,
            )
        )
    labels.sort(key=lambda cl: cl.normalized_label)
    return labels


def _majority(types: list[str]) -> str:
    """Deterministic majority claim_type (ties broken lexicographically)."""
    counts = Counter(types)
    top = max(counts.values())
    return sorted(t for t, c in counts.items() if c == top)[0]


def load_constraints(conn: sqlite3.Connection) -> dict[str, set[tuple[str, str]]]:
    """Load sticky ``concept_constraints`` (must/cannot-link) for this build.

    Returns ``{'must_link': {(a, b), ...}, 'cannot_link': {(a, b), ...}}`` over
    sorted normalized-label pairs so clustering honors prior human decisions (no
    cross-run flip; doc 07 §7). Consulted BEFORE clustering.
    """
    must: set[tuple[str, str]] = set()
    cannot: set[tuple[str, str]] = set()
    for kind, label_a, label_b in conn.execute(
        "SELECT kind, label_a, label_b FROM concept_constraints"
    ).fetchall():
        pair = tuple(sorted((label_a, label_b)))  # type: ignore[assignment]
        (must if kind == "must_link" else cannot).add(pair)  # type: ignore[arg-type]
    return {"must_link": must, "cannot_link": cannot}


def _acronym_or_exact(a: str, b: str) -> bool:
    return (
        a == b
        or merge.is_acronym_expansion(a, b)
        or merge.is_acronym_expansion(b, a)
    )


def _pick_canonical(
    keys: list[str],
    by_key: dict[str, ClaimLabel],
    user_status: dict[str, str],
) -> str:
    """Pick the canonical merge-key for a fold group.

    A member that already carries a sticky user status (``user_confirmed`` /
    ``user_split``) keeps its identity (so re-keying never flips a confirmed
    concept's id); otherwise ``field-typed-label-wins`` over the group's surfaces.
    """
    sticky = sorted(k for k in keys if k in user_status)
    if sticky:
        return sticky[0]
    surfaces = [by_key[k].canonical_label for k in keys]
    types = [by_key[k].concept_type for k in keys]
    chosen_surface = merge.field_typed_label(surfaces, types)
    for k in sorted(keys):
        if by_key[k].canonical_label == chosen_surface:
            return k
    return sorted(keys)[0]


def _guardrail_veto_reason(
    canonical: str, members: frozenset[str], present: set[str]
) -> str:
    """Name WHICH deterministic guardrail vetoed a proposed fold (D10 decision
    log; decision 56). Mirrors :func:`canon.passes_guardrails`' check order
    exactly and is only called after that conjunction returned False, so the
    fall-through leg is the token-overlap/acronym precondition.

    ponytail: reaches canon's ``discriminating_conflict`` rather than
    duplicating the veto predicate; ceiling = a reasons-returning canon API if a
    third consumer appears.
    """
    member_set = set(members)
    if canonical not in present or not member_set.issubset(present):
        return "closure"
    if canon.discriminating_conflict(member_set | {canonical}):
        return "discriminating_token_veto"
    return "token_overlap_precondition"


def build_concepts(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    proposer: "ConceptProposer",
    tau: float = 0.6,
) -> "SemanticBuildReport":
    """Full canonicalization → write ``concepts`` / ``concept_aliases`` /
    ``claim_concepts``; return a :class:`SemanticBuildReport` (concept-side counts;
    edges are filled by :mod:`.edges`).

    Idempotent: clears the prior machine-generated overlay, then re-keys by
    deterministic ``concept_id`` so unchanged labels keep identity and
    ``user_confirmed``/``user_split`` status + ``concept_constraints`` survive
    (doc 07 §7). ``concept_mode`` is ``'deterministic'`` under a
    :class:`~llm_propose.NullProposer` (no synonym folds, no ``provisional`` rows),
    ``'llm'`` when an LLM participates, ``'none'`` when there are zero claims.
    Stamps ``run_id`` + ``access_class`` (:func:`access.resolve_access_class`).
    Enqueues borderline folds as ``concept_merge_candidate`` (concepts stay split).

    Canon decision log (D10, decision 56; Build C chunk 9): every non-empty build
    accumulates per-cluster decisions as it loops — members, deterministic folds,
    cannot_link skips, proposer subsets, enqueued review pairs, and guardrail
    vetoes with WHICH guardrail fired — and writes exactly ONE ``audit_records``
    row (``audit_type='canon_decision_log'``, keyed by
    :func:`audit.concept_set_hash`) before returning, so
    :func:`audit.compare_canon_decision_logs` can detect a cross-build flip. A
    zero-claim (``'none'``) build performs no canonicalization and writes no log.
    """
    from . import SemanticBuildReport
    from ..project.review import enqueue_raw

    now = _now()
    labels = gather_claim_labels(conn)
    if not labels:
        return SemanticBuildReport(run_id=run_id, concept_mode="none")

    # Build B chunk 5 (decision 71): anti-stopword IDF denominator — distinct
    # works in the STAGED extraction set, NOT the whole citation graph. Counts
    # over the SAME current-claims view as gather_claim_labels (stale-claims
    # fix): works are counted iff they have CURRENT claims, so the numerator
    # and denominator of log(N_total / paper_frequency) never mix generations.
    # Computed once per build; the concepts table is delete-and-rewritten below
    # so the weight column never goes stale.
    n_total = conn.execute(
        _CURRENT_CLAIMS_CTE + "SELECT COUNT(DISTINCT work_id) FROM current_claims"
    ).fetchone()[0]

    by_key = {cl.normalized_label: cl for cl in labels}
    present = set(by_key)
    keys = sorted(by_key)

    constraints = load_constraints(conn)
    must = constraints["must_link"]
    cannot = constraints["cannot_link"]

    # Preserve sticky user status across the idempotent rebuild.
    user_status: dict[str, str] = {
        nl: st
        for nl, st in conn.execute(
            "SELECT normalized_label, status FROM concepts "
            "WHERE status IN ('user_confirmed', 'user_split')"
        ).fetchall()
    }

    # Idempotent rebuild: clear the machine-generated overlay (concepts cascade
    # their claim_concepts + concept_aliases). Human-validated interpretive edges
    # (user_validated / user_supplied) survive; sticky concept_constraints survive.
    conn.execute("DELETE FROM concepts")
    conn.execute(
        "DELETE FROM project_graph_edges "
        "WHERE epistemic_type NOT IN ('user_validated', 'user_supplied')"
    )

    # Type-scoped fence (D9, Build C chunk 8): partition keys by concept_type
    # BEFORE clustering so the fuzzy tier, the acronym auto-fold, and the LLM
    # proposal candidates below never span concept types — what phase_7 §2 /
    # §10-step-7 and the module docstring already claim. `keys` is sorted, so
    # each per-type list (and thus cluster order) is deterministic.
    keys_by_type: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        keys_by_type[by_key[k].concept_type].append(k)
    typed_clusters: list[tuple[str, frozenset[str]]] = [
        (concept_type, cluster)
        for concept_type in sorted(keys_by_type)
        for cluster in canon.candidate_clusters(keys_by_type[concept_type], tau)
    ]

    # Canon decision log (D10, decision 56): one entry per type-scoped cluster,
    # accumulated as the loops below run and written as ONE audit_records row
    # before returning. Entry order / member order / pair-iteration order are all
    # deterministic, and NO run-varying values (run_id, timestamps) enter the
    # payload — so two builds over the same concept set produce byte-identical
    # payloads unless a decision actually flipped.
    cluster_logs: list[dict] = [
        {
            "concept_type": concept_type,
            "members": sorted(cluster),
            "deterministic_folds": [],
            "cannot_link_skips": [],
            "proposals": [],
        }
        for concept_type, cluster in typed_clusters
    ]
    must_link_applied: list[list[str]] = []
    must_link_skipped: list[list[str]] = []

    # Union-find over fold groups. DETERMINISTIC auto-folds only: exact-key
    # (already merged at gather) + acronym within a (type-fenced) candidate
    # cluster + sticky must_link. Token-overlap-only clusters are NOT auto-folded
    # (they are mere LLM-proposal candidates) — this is the anti-overmerge stance.
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    for (_, cluster), cluster_log in zip(typed_clusters, cluster_logs):
        members = sorted(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if tuple(sorted((a, b))) in cannot:
                    # D10: a cannot_link that blocks a would-be deterministic
                    # fold is a recorded decision, not a silent skip.
                    if _acronym_or_exact(a, b):
                        cluster_log["cannot_link_skips"].append([a, b])
                    continue
                if _acronym_or_exact(a, b):
                    union(a, b)
                    cluster_log["deterministic_folds"].append([a, b])
    # Sticky must_link stays GLOBAL — a human decision outranks the type fence
    # (D9), so a cross-type must_link pair still merges here, as before.
    # (sorted() only stabilizes the decision-log order; unions commute.)
    for a, b in sorted(must):
        if a in parent and b in parent:
            if tuple(sorted((a, b))) in cannot:
                must_link_skipped.append([a, b])
            else:
                union(a, b)
                must_link_applied.append([a, b])

    # LLM proposals (skipped under NullProposer): each borderline fold that passes
    # the guardrails is ENQUEUED as concept_merge_candidate; the concepts stay
    # split (no auto-apply, no provisional rows) — over-merge is impossible.
    review_items = 0
    if not isinstance(proposer, NullProposer):
        seen_pairs: set[tuple[str, str]] = set()
        for (_, cluster), cluster_log in zip(typed_clusters, cluster_logs):
            members = sorted(cluster)
            if len(members) < 2:
                continue
            for subset in proposer.propose(members):
                subset_keys = sorted(
                    {merge.normalize(s) for s in subset} & present
                )
                prop_log: dict = {"subset": subset_keys}
                cluster_log["proposals"].append(prop_log)
                if len(subset_keys) < 2:
                    prop_log["outcome"] = "under_two_after_closure"
                    continue
                canonical_key = _pick_canonical(subset_keys, by_key, user_status)
                prop_log["canonical"] = canonical_key
                if not canon.passes_guardrails(
                    canonical_key, frozenset(subset_keys), present
                ):
                    # D10 veto-reason capture (was a bare `continue`): the
                    # decision log names WHICH guardrail fired.
                    prop_log["outcome"] = "guardrail_veto"
                    prop_log["guardrail"] = _guardrail_veto_reason(
                        canonical_key, frozenset(subset_keys), present
                    )
                    continue
                enqueued_pairs: list[list[str]] = []
                skipped_pairs: list[dict] = []
                for member_key in subset_keys:
                    if member_key == canonical_key:
                        continue
                    pair = tuple(sorted((canonical_key, member_key)))
                    if pair in cannot or pair in must or pair in seen_pairs:
                        reason = (
                            "cannot_link"
                            if pair in cannot
                            else "must_link" if pair in must else "duplicate_pair"
                        )
                        skipped_pairs.append({"pair": list(pair), "reason": reason})
                        continue
                    if find(member_key) == find(canonical_key):
                        # already deterministically folded
                        skipped_pairs.append(
                            {"pair": list(pair), "reason": "already_folded"}
                        )
                        continue
                    seen_pairs.add(pair)
                    enqueue_raw(
                        conn,
                        "concept_merge_candidate",
                        target_type="Concept",
                        target_id=merge.concept_id(canonical_key),
                        payload={
                            "kind": "concept_merge_candidate",
                            "canonical_label": canonical_key,
                            "member_label": member_key,
                            "canonical_concept_id": merge.concept_id(canonical_key),
                            "member_concept_id": merge.concept_id(member_key),
                            "canonical_surface": by_key[canonical_key].canonical_label,
                            "member_surface": by_key[member_key].canonical_label,
                            "run_id": run_id,
                        },
                    )
                    enqueued_pairs.append(list(pair))
                    review_items += 1
                prop_log["outcome"] = "enqueued" if enqueued_pairs else "no_new_pairs"
                prop_log["enqueued_pairs"] = enqueued_pairs
                prop_log["skipped_pairs"] = skipped_pairs

    # Materialize fold groups → concepts / aliases / claim_concepts.
    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)

    concepts_written = 0
    aliases_written = 0
    claim_concepts_written = 0
    access_summary: Counter = Counter()

    for group in groups.values():
        group = sorted(group)
        canonical_key = _pick_canonical(group, by_key, user_status)
        cid = merge.concept_id(canonical_key)
        canonical_surface = by_key[canonical_key].canonical_label
        concept_type = _group_concept_type(group, by_key)

        all_claim_ids = [c for k in group for c in by_key[k].claim_ids]
        access_class = resolve_access_class(conn, all_claim_ids)
        access_summary[access_class] += 1

        distinct_works = {w for k in group for w in by_key[k].work_ids}
        paper_frequency = len(distinct_works)
        # weight = log(N_total / paper_frequency) — anti-stopword IDF. Guard the
        # degenerate cases: paper_frequency 0 or N_total 0 → weight 0.0.
        if paper_frequency > 0 and n_total > 0:
            weight = math.log(n_total / paper_frequency)
        else:
            weight = 0.0

        definition = next(
            (by_key[k].definition for k in [canonical_key, *group] if by_key[k].definition),
            None,
        )

        status = user_status.get(canonical_key, "auto")
        epistemic = (
            EpistemicType.user_validated.value
            if status == "user_confirmed"
            else EpistemicType.deterministic.value
        )

        conn.execute(
            "INSERT INTO concepts "
            "(concept_id, normalized_label, canonical_label, concept_type, definition, "
            "paper_frequency, weight, status, epistemic_type, access_class, run_id, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                canonical_key,
                canonical_surface,
                concept_type,
                definition,
                paper_frequency,
                weight,
                status,
                epistemic,
                access_class,
                run_id,
                now,
                now,
            ),
        )
        concepts_written += 1

        for k in group:
            if k == canonical_key:
                reason = "exact_key"
            elif _acronym_or_exact(k, canonical_key):
                reason = "acronym"
            else:
                reason = "llm_proposed_reviewed"
            ep = (
                EpistemicType.deterministic.value
                if reason in ("exact_key", "acronym")
                else EpistemicType.llm_extracted.value
            )
            conn.execute(
                "INSERT OR IGNORE INTO concept_aliases "
                "(concept_id, alias_label, fold_reason, epistemic_type, run_id, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (cid, by_key[k].canonical_label, reason, ep, run_id, now),
            )
            aliases_written += 1
            for claim_id, work_id in by_key[k].claim_work_pairs:
                conn.execute(
                    "INSERT OR IGNORE INTO claim_concepts "
                    "(claim_id, concept_id, work_id, epistemic_type, confidence, run_id, "
                    "created_at) VALUES (?,?,?,?,?,?,?)",
                    (claim_id, cid, work_id, ep, None, run_id, now),
                )
                claim_concepts_written += 1

    # Canon decision log (D10, decision 56): exactly ONE audit_records row per
    # non-empty build, keyed by the corpus-set hash, naming every cluster —
    # including vetoed proposals with the guardrail that fired.
    set_hash = audit.concept_set_hash(
        (by_key[k].concept_type, k) for k in keys
    )
    audit.record_canon_decision_log(
        conn,
        run_id=run_id,
        set_hash=set_hash,
        decisions={
            "clusters": cluster_logs,
            "must_link_applied": must_link_applied,
            "must_link_skipped": must_link_skipped,
        },
    )

    # mode label fix (Track 1 regression): "llm" ONLY when the LLM proposer
    # actually contributed a fold (each passes guardrails → enqueued review item).
    # An external proposer that degraded to no proposal (offline / non-success
    # run_llm result) leaves review_items == 0, so the build is honestly
    # "deterministic" — never mislabeled "llm" just because a profile was named.
    mode = "llm" if review_items > 0 else "deterministic"
    return SemanticBuildReport(
        run_id=run_id,
        concept_mode=mode,
        concepts_written=concepts_written,
        aliases_written=aliases_written,
        claim_concepts_written=claim_concepts_written,
        review_items_enqueued=review_items,
        access_class_summary=dict(access_summary),
    )


def _group_concept_type(group: list[str], by_key: dict[str, ClaimLabel]) -> str:
    """Majority ``concept_type`` over a fold group (deterministic tie-break)."""
    return _majority([by_key[k].concept_type for k in group])
