"""Lawful-access resolver links for missing/paywalled works — LINKS ONLY.

Near-verbatim port of v1 ``report.py:58-216`` (build D chunk 11, gap scan §4.3):
public id resolvers (doi.org / arXiv / OpenAlex / S2 / SSRN abstract page) plus
the OPTIONAL institutional OpenURL / EZproxy links. Everything here generates
``(label, url)`` pairs for the user's *browser* — this module never fetches,
never scrapes, and never reaches a proxy from code.

THE SEPARATION INVARIANT: no configured resolver/proxy value can ever reach
``fetch._PREPRINT_HOSTS`` or any fetch URL. This module is pure stdlib
(``urllib.parse`` only) and deliberately import-independent of
``acquisition/fetch.py`` — both facts are asserted by test
(``tests/test_links.py``), so a future edit that couples the link path to the
fetch path fails loudly.

Configuration: the two NON-secret ``GlobalConfig`` fields ``openurl_resolver``
and ``ezproxy_host`` (config/models.py). v1 read the equivalent values from
``SEEDGRAPH_OPENURL_RESOLVER`` / ``SEEDGRAPH_EZPROXY_HOST`` env vars;
env-var-only config was REJECTED for v2 because v2 centralizes non-secret
settings in config.yaml (ADR-0001 keyring is for secrets only — these are
public URLs, not secrets). The helpers take the two values as ARGUMENTS and
stay pure; config threading happens at the call site (chunk 12's frontier
pane, via the effective inheritance-resolved project config).
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Mapping, Optional

__all__ = [
    "RESOLVERS",
    "resolver_links",
]

#: Public OA resolvers, in priority order. Each entry maps a row field to a
#: ``(label, url-template)`` pair; the template's ``{id}`` is filled with the
#: (normalized) id value. Public resolvers ONLY — no institutional proxy, no
#: scraping (this module links, it never downloads). Field names follow the v2
#: ``works`` columns (v1 used ``s2_id``; v2 stores ``semantic_scholar_id``).
RESOLVERS: tuple[tuple[str, str, str], ...] = (
    ("doi", "DOI", "https://doi.org/{id}"),
    ("arxiv_id", "arXiv", "https://arxiv.org/abs/{id}"),
    ("openalex_id", "OpenAlex", "https://openalex.org/{id}"),
    ("semantic_scholar_id", "S2", "https://www.semanticscholar.org/paper/{id}"),
    # SSRN is LINK-ONLY (never in the auto-fetch allowlist, never constructed
    # for download — fetch.py excludes it from _PREPRINT_HOSTS permanently): a
    # stored ``ssrn_id`` resolves to the abstract page so the user can obtain
    # the PDF behind SSRN's own access flow.
    ("ssrn_id", "SSRN", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id={id}"),
)


def _normalize_openalex_id(value: str) -> str:
    """Return a bare OpenAlex ``W…`` id for the resolver URL.

    The stored shape is bare (``W…`` — providers/openalex + identity confirm
    this), but a row re-imported from elsewhere could carry a full
    ``https://openalex.org/W…`` URL. Strip a leading ``https://openalex.org/``
    (or ``http://`` / bare ``openalex.org/``) if present, so BOTH shapes
    resolve to the same canonical ``https://openalex.org/{bare}`` link.
    """

    text = str(value).strip()
    lowered = text.lower()
    for prefix in ("https://openalex.org/", "http://openalex.org/", "openalex.org/"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip()


def _institutional_links(
    row: Mapping[str, Any],
    openurl_resolver: Optional[str] = None,
    ezproxy_host: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Build OPTIONAL OpenURL / EZproxy ``(label, url)`` links for a row.

    Pure LINK generation for the genuine-paywall remainder — *never* a fetch
    path: these point the user's browser at their institution's authenticated
    resolver / proxy. The fetch code never routes through a proxy (the
    separation invariant — no value passed here ever enters
    ``fetch._PREPRINT_HOSTS``).

    Takes the two NON-secret config values as arguments (``GlobalConfig
    .openurl_resolver`` / ``.ezproxy_host``); when neither is set this returns
    ``[]`` so :func:`resolver_links` output is byte-identical to the
    public-resolvers-only shape (back-compat).

    * **OpenURL** (``openurl_resolver`` set): if the row carries a DOI → a
      ``rft_id=info:doi/…`` OpenURL 1.0 query; DOI-less → a ``genre=article``
      key/encoded-value (KEV) query built from the present-only
      ``rft.atitle`` / ``rft.aulast`` / ``rft.date`` / ``rft.jtitle`` fields
      (row ``title`` / first author / ``year`` / venue). Emitted only when at
      least one of these is present, so a wholly bare row yields no OpenURL.
    * **EZproxy** (``ezproxy_host`` set): **DOI-only**. The value is a
      user-pasted template containing ``{url}`` (or ``{doi}``) → substitute
      the ``https://doi.org/{doi}`` target; a bare host with no placeholder /
      scheme → the documented last-resort ``https://{host}/login?url=…``. A
      scheme-carrying value WITHOUT a placeholder is refused (no link) — it is
      almost certainly a real login template the user pasted incompletely, and
      rewriting it would mangle a ``qurl=`` / inline-rewrite config.
    """

    resolver = (openurl_resolver or "").strip()
    ezproxy = (ezproxy_host or "").strip()
    if not resolver and not ezproxy:
        return []

    doi = str(row.get("doi") or "").strip()
    links: list[tuple[str, str]] = []

    # ---- OpenURL: DOI rft_id, else a genre=article KEV from present fields -----
    if resolver:
        if doi:
            query = urllib.parse.urlencode(
                {
                    "url_ver": "Z39.88-2004",
                    "rfr_id": "info:sid/seedgraph",
                    "rft_id": f"info:doi/{doi}",
                }
            )
        else:
            kev: dict[str, str] = {
                "url_ver": "Z39.88-2004",
                "rfr_id": "info:sid/seedgraph",
                "rft_val_fmt": "info:ofi/fmt:kev:mtx:journal",
                "rft.genre": "article",
            }
            title = str(row.get("title") or "").strip()
            if title:
                kev["rft.atitle"] = title
            authors = row.get("authors") or []
            if authors:
                first = str(authors[0]).strip()
                # ``Last, First`` → surname is the head; ``First Last`` → the tail.
                # Duplicates ``display.extract_surname`` on purpose: links.py is
                # tested pure-stdlib with NO seedgraph imports (test_links.py's
                # import-independence guard), so it cannot import display.
                aulast = first.split(",")[0].split()[-1] if first else ""
                if aulast:
                    kev["rft.aulast"] = aulast
            year = row.get("year")
            if year:
                kev["rft.date"] = str(year)
            venue = str(row.get("venue") or row.get("journal") or "").strip()
            if venue:
                kev["rft.jtitle"] = venue
            # Only emit an OpenURL when the KEV carries ≥1 bibliographic field
            # beyond the fixed scaffolding (else the resolver has nothing to match).
            if len(kev) <= 4:
                query = ""
            else:
                query = urllib.parse.urlencode(kev)
        if query:
            sep = "&" if "?" in resolver else "?"
            links.append(("OpenURL", f"{resolver}{sep}{query}"))

    # ---- EZproxy: DOI-only, via the user-pasted {url}/{doi} template -----------
    if ezproxy and doi:
        # Percent-encode ONLY the substituted value — a DOI may carry reserved
        # chars (``&``, space) that would otherwise leak a fake query param
        # into the proxy URL. The template itself stays OPAQUE (never
        # encoded), so the user's ``?url=`` / ``qurl=`` config is preserved.
        target = "https://doi.org/" + urllib.parse.quote(doi, safe="/")
        if "{url}" in ezproxy:
            url = ezproxy.replace("{url}", target)
        elif "{doi}" in ezproxy:
            url = ezproxy.replace("{doi}", urllib.parse.quote(doi, safe=""))
        elif "://" not in ezproxy:
            # Bare host (no placeholder, no scheme): the documented last-resort
            # form. NOT applied to a value that already carries a scheme — that
            # is almost certainly a real login template the user pasted
            # incompletely, and rewriting it would mangle a ``qurl=`` /
            # inline-rewrite config.
            url = f"https://{ezproxy}/login?url={target}"
        else:
            url = ""
        if url:
            links.append(("EZproxy", url))

    return links


def resolver_links(
    row: Mapping[str, Any],
    *,
    openurl_resolver: Optional[str] = None,
    ezproxy_host: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Build public resolver ``(label, url)`` pairs from a row's identifiers.

    Walks :data:`RESOLVERS` in priority order (DOI → arXiv → OpenAlex → S2 →
    SSRN) and emits one link per id the row actually carries. ``openalex_id``
    is normalized to its bare ``W…`` form first so a stored bare id and a full
    ``https://openalex.org/…`` URL both resolve. Finally appends the OPTIONAL
    :func:`_institutional_links` (OpenURL / EZproxy) — empty unless the caller
    threads the two config values, so config-unset output is byte-identical to
    the public-resolvers-only shape and every existing call-site is unaffected.

    Returns ``[]`` ONLY when the row is truly id-less AND no institutional
    links apply — a degenerate row degrades to "no identifier" rather than
    crashing.
    """

    links: list[tuple[str, str]] = []
    for field, label, template in RESOLVERS:
        raw = row.get(field)
        if not raw:
            continue
        value = str(raw).strip()
        if not value:
            continue
        if field == "openalex_id":
            value = _normalize_openalex_id(value)
            if not value:
                continue
        links.append((label, template.format(id=value)))
    links.extend(_institutional_links(row, openurl_resolver, ezproxy_host))
    return links
