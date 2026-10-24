"""Fixed vocabularies for the note's ``*_type`` fields (design §4.1, schema 2.0.0).

An unknown value is never a rejection. :func:`map_label` maps a raw value onto the
group's fixed value set:

1. normalise (NFKC, lower case, ``_``/``-``/``/`` to spaces, collapsed whitespace);
2. exact match on a fixed value (spaces read as ``_``), then the variant table;
3. the same after dropping one trailing filler word ("simulation result" -> "simulation");
4. ordered keyword rules (first match wins; guards come before broad words, so
   "model selection" never becomes ``model``);
5. otherwise ``other`` (``general`` for assumptions) with ``known=False``, which the
   caller turns into a feedback warning listing the allowed values.

The read side (stage 3 facets, stage 4 concept typing) maps old free-text subtypes
through the same function, so old and new notes facet together. A vocabulary change
bumps ``SCHEMA_VERSION`` (:mod:`seedgraph.notes.format`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ASSUMPTION_TYPES: tuple[str, ...] = ("identification", "regularity", "general")
METHOD_TYPES: tuple[str, ...] = (
    "estimator", "regression", "research_design", "identification", "inference", "model",
    "simulation", "asymptotic_theory", "computation", "decomposition", "empirical_application",
    "data_analysis", "other",
)
RESULT_TYPES: tuple[str, ...] = (
    "empirical", "theoretical", "identification", "asymptotic", "simulation", "heterogeneity",
    "mechanism", "descriptive", "methodological", "policy", "other",
)
LIMITATION_TYPES: tuple[str, ...] = (
    "scope", "external_validity", "identification", "measurement", "data", "inference",
    "model_specification", "method", "computational", "interpretation", "other",
)

#: Note field name -> its fixed values (in the order shown to agents).
VOCABULARIES: dict[str, tuple[str, ...]] = {
    "assumption_type": ASSUMPTION_TYPES,
    "method_type": METHOD_TYPES,
    "result_type": RESULT_TYPES,
    "limitation_type": LIMITATION_TYPES,
}

#: The value an unknown raw value becomes.
FALLBACK: dict[str, str] = {
    "assumption_type": "general",
    "method_type": "other",
    "result_type": "other",
    "limitation_type": "other",
}

#: Exact variants (keys normalised as by :func:`_norm`).
_VARIANTS: dict[str, dict[str, str]] = {
    "assumption_type": {
        "identifying": "identification", "identification assumption": "identification",
        "identifying assumption": "identification",
        "regularity condition": "regularity", "technical": "regularity",
        "technical condition": "regularity", "other": "general", "generic": "general",
    },
    "method_type": {
        "ols": "regression", "ordinary least squares": "regression",
        "iv": "estimator", "gmm": "estimator", "mle": "estimator",
        "design": "research_design", "did": "research_design",
        "rdd": "research_design", "regression discontinuity": "research_design",
        "theory": "model", "framework": "model", "extension": "other",
    },
    "result_type": {
        "theorem": "theoretical", "proposition": "theoretical", "lemma": "theoretical",
        "theory": "theoretical", "conceptual": "theoretical",
        "application": "empirical", "estimate": "empirical",
    },
    "limitation_type": {
        "general": "scope", "theoretical": "scope", "theory": "scope",
        "modeling": "model_specification", "modelling": "model_specification",
    },
}

_FILLER = frozenset({
    "result", "results", "finding", "findings", "evidence", "limitation", "limitations",
    "method", "methods", "procedure", "approach", "strategy", "study", "type", "issue",
    "concern", "restriction", "condition", "analysis",
})

_Rule = tuple[re.Pattern[str], str]


def _rules(*pairs: tuple[str, str]) -> tuple[_Rule, ...]:
    return tuple((re.compile(p), v) for p, v in pairs)


#: Ordered keyword rules per group; the first pattern that matches wins.
_KEYWORDS: dict[str, tuple[_Rule, ...]] = {
    "assumption_type": _rules(
        (r"identif|exogene|overlap|parallel trend|unconfound|exclusion|monoton|"
         r"no anticipation|independence|support", "identification"),
        (r"regular|technical|smooth|moment|continuity|differentiab|asymptot|distribution|"
         r"dependence|sampling|mixing|bootstrap|approximation|\brates?\b", "regularity"),
        (r"model|behavio|structur|measure|design|theor|economic|functional form|"
         r"homogene|equilibrium|normali[sz]ation|initial condition|implementation", "general"),
    ),
    "method_type": _rules(
        (r"\bmodel (selection|fit|comparison|averaging|check)", "inference"),
        (r"simulat|monte carlo", "simulation"),
        (r"asymptot", "asymptotic_theory"),
        (r"decompos", "decomposition"),
        (r"identif", "identification"),
        (r"design|experiment|event study|difference in differences|discontinuity|"
         r"synthetic control", "research_design"),
        (r"regression|least squares|fixed effects", "regression"),
        (r"\btests?\b|testing|bootstrap|resampl|inference|confidence|standard error|"
         r"variance estimat|critical value|p value", "inference"),
        (r"estimat|matching|weighting|bias correction|instrumental variable|likelihood",
         "estimator"),
        (r"bandwidth|kernel|tuning|smoothing", "estimator"),
        (r"randomi[sz]ed|case study|trial", "research_design"),
        (r"algorithm|optimi[sz]|comput|numerical|dimension reduction|software", "computation"),
        (r"replicat|reanalysis", "empirical_application"),
        (r"calibrat", "model"),
        (r"application|illustration", "empirical_application"),
        (r"model|framework|theor", "model"),
        (r"data|survey|measure|descriptive|heterogeneity|sensitivity|diagnostic|correlation|"
         r"imputation|transformation|index construction", "data_analysis"),
    ),
    "result_type": _rules(
        (r"simulat|monte carlo", "simulation"),
        (r"asymptot|consisten|convergence|efficien|bootstrap validity|limiting distribution",
         "asymptotic"),
        (r"identif", "identification"),
        (r"heterogen", "heterogeneity"),
        (r"mechanism|channel", "mechanism"),
        (r"descriptive|summary statistic", "descriptive"),
        (r"policy|welfare|recommendation|counterfactual", "policy"),
        (r"methodolog|inference|bias|power|comparison|model fit|computational|equivalence",
         "methodological"),
        (r"theor|proposition|lemma|characteri[sz]ation|conceptual", "theoretical"),
        (r"empiric|estimat|effect|elasticit|coefficient|first stage|magnitude|"
         r"application|illustration|null|regression|decomposition|robustness|tests?|"
         r"negative result|persistence|instrumental variable", "empirical"),
    ),
    "limitation_type": _rules(
        (r"external validity|generali[sz]|applicab|transportab", "external_validity"),
        (r"identif|endogene|confound|causal|bias|omitted variable|selection|testab",
         "identification"),
        (r"misspecif|specification|functional form|model", "model_specification"),
        (r"measure|proxy", "measurement"),
        (r"comput|scalab|tractab|dimensional", "computational"),
        (r"inference|power|precision|finite sample|standard error|sample size|efficien|"
         r"conservativ|variance", "inference"),
        (r"interpret|mechanism", "interpretation"),
        (r"data|sample|coverage|horizon|covariate", "data"),
        (r"method|estimat|implementation|tuning|bandwidth|design|robustness|simulation",
         "method"),
        (r"heterogene", "external_validity"),
        (r"scope|assumption|regularity|theor|support|general equilibrium|practical", "scope"),
    ),
}


@dataclass(frozen=True)
class VocabMatch:
    """The mapped value. ``known`` is False when the raw value fell back to the default."""

    value: str | None
    known: bool
    raw: str | None


def _norm(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).lower()
    text = re.sub(r"[_\-/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _exact(group: str, text: str) -> str | None:
    as_value = text.replace(" ", "_")
    if as_value in VOCABULARIES[group]:
        return as_value
    return _VARIANTS[group].get(text)


def map_label(group: str, raw: str | None) -> VocabMatch:
    """Map ``raw`` onto ``group``'s fixed values. ``KeyError`` for an unknown group."""
    values = VOCABULARIES[group]  # KeyError on an unknown group
    if raw is None or not str(raw).strip():
        return VocabMatch(value=None, known=True, raw=raw)
    text = _norm(str(raw))
    hit = _exact(group, text)
    if hit is None:
        words = text.split(" ")
        if len(words) > 1 and words[-1] in _FILLER:
            hit = _exact(group, " ".join(words[:-1]))
    if hit is None:
        for pattern, value in _KEYWORDS[group]:
            if pattern.search(text):
                hit = value
                break
    if hit is not None:
        assert hit in values
        return VocabMatch(value=hit, known=True, raw=raw)
    return VocabMatch(value=FALLBACK[group], known=False, raw=raw)
