"""Injectable LLM seam for synonym proposal (plan §5/§6, step 5).

The LLM **proposes only** — given a small cluster of concept *labels* (never full
text, never evidence spans; doc 07 §8), it returns proposed synonym subsets. The
deterministic guardrails in :mod:`.canon` dispose. A wrong or empty response
degrades to under-merge (split), never over-merge. Content-access gating: labels
are full-text-derived → private-by-default; prefer a local profile.

Import-safe: no SDK import at module load (the real proposer lazy-imports inside
``propose``). The :class:`NullProposer` is the no-LLM path (decision 38) and is
fully implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..llm.parse import parse_json_object

if TYPE_CHECKING:
    from ..config.models import LLMProfile, ProjectConfig
    from ..llm.backend import LLMBackend

TASK_TYPE = "semantic_graph_extraction"

_PROPOSE_SYSTEM = (
    "You canonicalize concept labels. You are given a small cluster of concept "
    "LABELS only (never document text). Group only labels that are truly "
    "interchangeable synonyms. Reply with strict JSON: "
    '{"synonym_groups": [["label a", "label b"], ...]}. Use [] when none apply.'
)


@runtime_checkable
class ConceptProposer(Protocol):
    """Structural type for a synonym proposer."""

    def propose(self, cluster: list[str]) -> list[list[str]]:
        """Given a candidate cluster of labels, return proposed synonym subsets.

        Each returned subset is a group of labels the proposer believes are truly
        interchangeable. Returns ``[]`` when it has no proposal (or no LLM).
        """
        ...


class NullProposer:
    """No-LLM proposer (decision 38): always proposes nothing.

    Used by ``concepts build --no-llm`` and whenever routing yields no usable
    profile, so the build runs exact-key + acronym folds + ``co_occurs_with``
    edges only, with no fuzzy synonym folds and no ``provisional`` concepts.
    """

    def propose(self, cluster: list[str]) -> list[list[str]]:
        """Return ``[]`` unconditionally (no synonym proposal without an LLM)."""
        return []


class _ExternalConceptProposer:
    """Real LLM-backed proposer routed through the Track 1 executor (labels-only).

    Routes the ``semantic_graph_extraction`` task to ``profile`` via
    :func:`seedgraph.llm.executor.run_llm`, feeding concept *labels only* (never
    full text / evidence spans; doc 07 §8). Any non-success executor result —
    no key, provider down, content-policy block, malformed JSON — yields ``[]`` so
    the build degrades to under-merge (split), never over-merge or a crash. The
    executor/provider SDKs are imported lazily inside :meth:`propose`.
    """

    def __init__(
        self,
        profile: "LLMProfile",
        *,
        config: "ProjectConfig | None" = None,
        backend: "LLMBackend | None" = None,
    ) -> None:
        self._profile = profile
        self._config = config
        self._backend = backend

    def propose(self, cluster: list[str]) -> list[list[str]]:
        if len(cluster) < 2:
            return []
        try:
            return self._propose_via_llm(cluster)
        except Exception:  # noqa: BLE001 - degrade to under-merge, never raise
            return []

    def _propose_via_llm(self, cluster: list[str]) -> list[list[str]]:
        from ..config.models import GlobalConfig
        from ..llm.executor import run_llm

        cfg = self._config or GlobalConfig()
        user = (
            "Cluster of concept labels:\n"
            + "\n".join(f"- {label}" for label in cluster)
            + "\n\nReturn the strict JSON synonym_groups object now."
        )
        # Labels are full-text-derived → private-by-default; the content gate (an
        # external profile + restricted class + policy off) keeps them local or
        # degrades. log=False: concept proposal fires per-cluster, no usage spam.
        result = run_llm(
            TASK_TYPE,
            _PROPOSE_SYSTEM,
            user,
            access_class="user_supplied_private",
            config=cfg,
            profile_id_override=getattr(self._profile, "profile_id", None),
            backend=self._backend,
            parse=_parse_groups,
            log=False,
        )
        if not result.ok or not isinstance(result.parsed, dict):
            return []
        groups = result.parsed.get("synonym_groups")
        if not isinstance(groups, list):
            return []
        out: list[list[str]] = []
        for group in groups:
            if isinstance(group, list):
                members = [g for g in group if isinstance(g, str) and g.strip()]
                if len(members) >= 2:
                    out.append(members)
        return out


def _parse_groups(text: str) -> tuple[dict | None, object]:
    """Parse the proposer's strict-JSON reply (tolerating prose / a code fence)
    via the shared brace-depth parser (``llm/parse.py``)."""
    parsed = parse_json_object(text)
    if parsed is None:
        return None, "invalid json"
    return parsed, None


def make_proposer(
    profile: "LLMProfile | None",
    *,
    config: "ProjectConfig | None" = None,
    backend: "LLMBackend | None" = None,
) -> ConceptProposer:
    """Factory: a real proposer for an available external/local LLM profile, else
    :class:`NullProposer`.

    A ``None`` profile, or a no-LLM profile, yields :class:`NullProposer`
    (deterministic mode). The real proposer routes ``semantic_graph_extraction``
    through the executor seam (``config`` carries routes/policy; ``backend`` is the
    offline test injection)."""
    if profile is None:
        return NullProposer()
    provider = (getattr(profile, "provider", None) or "none").lower()
    access_mode = (getattr(profile, "access_mode", None) or "none").lower()
    if provider == "none" or access_mode == "none":
        return NullProposer()
    return _ExternalConceptProposer(profile, config=config, backend=backend)
