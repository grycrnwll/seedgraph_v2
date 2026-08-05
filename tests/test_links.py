"""OFFLINE tests for the lawful-access link helper (``acquisition/links.py``).

Build D chunk 11 (gap scan §4.3) — port of v1 ``tests/test_report.py``'s
link-shape tables. Fully offline (no DB, no network, no providers). Halves:

* :func:`resolver_links` — each id type → the right public URL; priority
  order; bare-``W`` vs full-``https://openalex.org/W…`` normalization; SSRN is
  link-only; a truly id-less row → ``[]``.
* :func:`_institutional_links` — DOI ``rft_id`` vs DOI-less KEV OpenURL
  construction (≥1-bibliographic-field guard); EZproxy ``{url}`` / ``{doi}``
  template substitution with percent-encoding of the substituted value ONLY;
  bare-host ``login?url=`` last resort; scheme-carrying non-template values
  refused; both configs unset → public resolvers only, byte-stable.
* Config: the two NON-secret ``GlobalConfig`` fields default to ``None``.
* SEPARATION INVARIANT — no configured resolver/proxy value can reach
  ``fetch._PREPRINT_HOSTS`` or any fetch URL, including a grep-style (AST)
  assertion that ``links.py`` is import-independent of ``fetch.py`` and pure
  stdlib.
"""

from __future__ import annotations

import ast
import inspect

from seedgraph.acquisition import links
from seedgraph.acquisition.links import _institutional_links, resolver_links


# --------------------------------------------------------------------------- #
# resolver_links — public resolvers
# --------------------------------------------------------------------------- #
def test_resolver_links_each_id_type_resolves():
    row = {
        "doi": "10.1/abc",
        "arxiv_id": "2401.01234",
        "openalex_id": "W123",
        "semantic_scholar_id": "deadbeef",
    }
    got = dict(resolver_links(row))
    assert got["DOI"] == "https://doi.org/10.1/abc"
    assert got["arXiv"] == "https://arxiv.org/abs/2401.01234"
    assert got["OpenAlex"] == "https://openalex.org/W123"
    assert got["S2"] == "https://www.semanticscholar.org/paper/deadbeef"


def test_resolver_links_priority_order():
    """DOI first, then arXiv, OpenAlex, S2, SSRN — the RESOLVERS priority order."""
    row = {
        "ssrn_id": "77",
        "semantic_scholar_id": "z",
        "openalex_id": "W9",
        "arxiv_id": "1.2",
        "doi": "10.x/y",
    }
    labels = [label for label, _ in resolver_links(row)]
    assert labels == ["DOI", "arXiv", "OpenAlex", "S2", "SSRN"]


def test_resolver_links_openalex_bare_w():
    # The STORED shape is bare 'W…' (providers/openalex + identity confirm this).
    assert resolver_links({"openalex_id": "W2741809807"}) == [
        ("OpenAlex", "https://openalex.org/W2741809807")
    ]


def test_resolver_links_openalex_full_url_normalized():
    # A full-URL openalex_id must normalize to the SAME bare-W resolver link.
    for stored in (
        "https://openalex.org/W2741809807",
        "http://openalex.org/W2741809807",
        "openalex.org/W2741809807",
    ):
        assert resolver_links({"openalex_id": stored}) == [
            ("OpenAlex", "https://openalex.org/W2741809807")
        ], stored


def test_resolver_links_ssrn_abstract_page_link_only():
    # SSRN resolves to the ABSTRACT page (link-only — never a fetch constructor).
    assert resolver_links({"ssrn_id": "1234567"}) == [
        ("SSRN", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1234567")
    ]


def test_resolver_links_no_id_returns_empty():
    assert resolver_links({}) == []
    assert resolver_links({"title": "T", "year": 2020}) == []
    # Empty-string ids are not links either.
    assert resolver_links({"doi": "", "openalex_id": "   "}) == []


# --------------------------------------------------------------------------- #
# Institutional links — both configs unset → byte-stable public-only output
# --------------------------------------------------------------------------- #
def test_configs_unset_public_resolvers_only_byte_stable():
    """Unset (None / empty / whitespace) configs → output byte-identical to the
    id-only public links, and stable across calls."""

    row = {"doi": "10.1/abc", "title": "T", "authors": ["Alpha"], "year": 2020}
    expected = [("DOI", "https://doi.org/10.1/abc")]
    assert resolver_links(row) == expected
    assert resolver_links(row, openurl_resolver=None, ezproxy_host=None) == expected
    assert resolver_links(row, openurl_resolver="", ezproxy_host="  ") == expected
    # Byte-stable: repeated calls produce the identical value.
    assert resolver_links(row) == resolver_links(row)
    assert _institutional_links(row) == []
    assert _institutional_links(row, None, None) == []


# --------------------------------------------------------------------------- #
# Institutional links — OpenURL construction (DOI rft_id vs KEV)
# --------------------------------------------------------------------------- #
def test_institutional_doi_openurl_and_ezproxy_url_template():
    """DOI present → OpenURL rft_id=info:doi/… + EZproxy {url}-substituted."""

    row = {"doi": "10.1/abc", "title": "T", "authors": ["Alpha"], "year": 2020}
    got = dict(
        _institutional_links(
            row,
            "https://lib.example.edu/openurl",
            "https://login.proxy.example.edu/login?url={url}",
        )
    )

    openurl = got["OpenURL"]
    assert openurl.startswith("https://lib.example.edu/openurl?")
    assert "rft_id=info%3Adoi%2F10.1%2Fabc" in openurl
    assert "url_ver=Z39.88-2004" in openurl

    assert got["EZproxy"] == (
        "https://login.proxy.example.edu/login?url=https://doi.org/10.1/abc"
    )


def test_institutional_resolver_with_existing_query_joins_with_amp():
    """A resolver base that already carries ``?`` joins with ``&``, not a 2nd ``?``."""

    openurl = dict(
        _institutional_links({"doi": "10.1/abc"}, "https://lib.example.edu/openurl?inst=42", None)
    )["OpenURL"]
    assert openurl.startswith("https://lib.example.edu/openurl?inst=42&")
    assert openurl.count("?") == 1


def test_institutional_doiless_openurl_kev_no_ezproxy():
    """DOI-less → OpenURL genre=article KEV from title/author/year, NO EZproxy."""

    row = {"title": "A Paywalled Study", "authors": ["Smith, Jane"], "year": 2017}
    got = dict(
        _institutional_links(
            row,
            "https://lib.example.edu/openurl",
            "https://login.proxy.example.edu/login?url={url}",
        )
    )

    openurl = got["OpenURL"]
    assert "rft.genre=article" in openurl
    assert "rft.atitle=A+Paywalled+Study" in openurl
    assert "rft.aulast=Smith" in openurl
    assert "rft.date=2017" in openurl
    # EZproxy is DOI-only — no DOI → no EZproxy link.
    assert "EZproxy" not in got


def test_institutional_doiless_venue_reaches_jtitle():
    got = dict(
        _institutional_links(
            {"venue": "J. Paywall Stud."}, "https://lib.example.edu/openurl", None
        )
    )
    assert "rft.jtitle=J.+Paywall+Stud." in got["OpenURL"]


def test_institutional_doiless_no_bib_fields_no_openurl():
    """A wholly bare row (no DOI, no title/author/year/venue) yields NO OpenURL —
    the ≥1-bibliographic-field guard (else the resolver has nothing to match)."""

    assert _institutional_links({"openalex_id": "W1"}, "https://lib.example.edu/openurl", None) == []


# --------------------------------------------------------------------------- #
# Institutional links — EZproxy template variants
# --------------------------------------------------------------------------- #
def test_institutional_ezproxy_bare_host_fallback():
    """A bare-host EZproxy value (no placeholder, no scheme) → the documented
    last-resort ``https://{host}/login?url=…`` form."""

    got = dict(_institutional_links({"doi": "10.1/abc"}, None, "login.proxy.example.edu"))
    assert got["EZproxy"] == (
        "https://login.proxy.example.edu/login?url=https://doi.org/10.1/abc"
    )


def test_institutional_ezproxy_scheme_without_placeholder_refused():
    """A scheme-carrying value WITHOUT a {url}/{doi} placeholder is REFUSED — it is
    almost certainly an incomplete real login template, and synthesizing around it
    would mangle a ``qurl=`` / inline-rewrite config."""

    for bad in (
        "https://login.proxy.example.edu",
        "https://login.proxy.example.edu/login?qurl=",
        "http://proxy.example.edu/some/path",
    ):
        assert _institutional_links({"doi": "10.1/abc"}, None, bad) == [], bad


def test_institutional_ezproxy_url_value_percent_encoded_only():
    """A DOI with reserved chars (space, ``&``) is percent-encoded in the EZproxy
    target so no fake query param leaks past the template; the template itself
    stays opaque (never encoded)."""

    url = dict(
        _institutional_links(
            {"doi": "10.1000/abc def&x=1"},
            None,
            "https://login.proxy.example.edu/login?url={url}",
        )
    )["EZproxy"]
    assert url == (
        "https://login.proxy.example.edu/login?url="
        "https://doi.org/10.1000/abc%20def%26x%3D1"
    )
    # The raw reserved chars must NOT survive (no leaked ``&x=1`` query param).
    assert "abc def" not in url
    assert "&x=1" not in url


def test_institutional_ezproxy_doi_placeholder_fully_encoded():
    """The ``{doi}`` branch encodes the bare DOI with no ``safe`` chars (no ``/``)."""

    url = dict(
        _institutional_links(
            {"doi": "10.1000/abc def"}, None, "https://proxy.example.edu/connect?doi={doi}"
        )
    )["EZproxy"]
    assert url == "https://proxy.example.edu/connect?doi=10.1000%2Fabc%20def"


def test_institutional_links_flow_through_resolver_links():
    """resolver_links APPENDS the institutional links after the id-only resolvers."""

    labels = [
        label
        for label, _ in resolver_links(
            {"doi": "10.1/abc"},
            openurl_resolver="https://lib.example.edu/openurl",
            ezproxy_host="https://login.proxy.example.edu/login?url={url}",
        )
    ]
    # DOI first (public id resolver), then the appended institutional links.
    assert labels == ["DOI", "OpenURL", "EZproxy"]


# --------------------------------------------------------------------------- #
# GlobalConfig fields — non-secret, plain optional strings, default None
# --------------------------------------------------------------------------- #
def test_global_config_link_fields_default_none_and_roundtrip():
    from seedgraph.config.models import GlobalConfig

    cfg = GlobalConfig()
    assert cfg.openurl_resolver is None
    assert cfg.ezproxy_host is None

    cfg2 = GlobalConfig.model_validate(
        {
            "openurl_resolver": "https://lib.example.edu/openurl",
            "ezproxy_host": "https://login.proxy.example.edu/login?url={url}",
        }
    )
    assert cfg2.openurl_resolver == "https://lib.example.edu/openurl"
    assert cfg2.ezproxy_host == "https://login.proxy.example.edu/login?url={url}"


# --------------------------------------------------------------------------- #
# SEPARATION INVARIANT — links never touch the fetch path
# --------------------------------------------------------------------------- #
def test_no_config_value_reaches_fetch_preprint_hosts():
    """A configured proxy/resolver host must NEVER appear in the constructed-fetch
    allowlist (``fetch._PREPRINT_HOSTS``) — generating links must not perturb it."""

    from seedgraph.acquisition import fetch

    before = frozenset(fetch._PREPRINT_HOSTS)

    resolver = "https://lib.example.edu/openurl"
    ezproxy = "login.proxy.example.edu"
    out = resolver_links(
        {"doi": "10.1/abc", "title": "T"}, openurl_resolver=resolver, ezproxy_host=ezproxy
    )
    assert out  # links were actually generated

    # The allowlist is untouched and carries neither configured value.
    assert fetch._PREPRINT_HOSTS == before
    assert "lib.example.edu" not in fetch._PREPRINT_HOSTS
    assert "login.proxy.example.edu" not in fetch._PREPRINT_HOSTS
    # And no allowlist entry looks like a proxy/login host at all.
    assert not any(
        "proxy" in str(h).lower() or "login" in str(h).lower() for h in fetch._PREPRINT_HOSTS
    )
    # No configured value leaked into any fetch URL surface: the only URL bases
    # fetch constructs are module-level constants — none carries either value.
    for name in dir(fetch):
        const = getattr(fetch, name)
        if isinstance(const, str):
            assert "lib.example.edu" not in const
            assert "proxy.example.edu" not in const


def test_links_module_import_independent_of_fetch_and_pure_stdlib():
    """Grep-style guard: ``links.py`` imports neither ``fetch`` nor ``httpx`` —
    no relative imports at all — and stays pure stdlib. A future edit coupling
    the link path to the fetch path fails HERE."""

    tree = ast.parse(inspect.getsource(links))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) could reach a sibling like ``.fetch``
            # — links.py must have NONE.
            assert node.level == 0, "links.py must not use relative imports"
            imported.add(node.module or "")

    joined = " ".join(sorted(imported))
    assert "fetch" not in joined
    assert "httpx" not in joined
    assert "seedgraph" not in joined  # no package-internal coupling at all
    # Pure stdlib: the full import surface is urllib.parse + typing (+ __future__).
    assert imported <= {"__future__", "urllib", "urllib.parse", "typing"}, imported
