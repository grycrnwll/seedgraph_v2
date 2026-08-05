from seedgraph.vocab import (
    CLAIM_TYPE_OTHER,
    CLAIM_TYPES,
    PROVIDER_SHAREABLE_FIELDS,
    AccessClass,
    field_allowed,
    is_shareable,
    normalize,
    project_provider_fields,
)


def test_access_class_values():
    assert {ac.value for ac in AccessClass} == {
        "open_access",
        "user_supplied_private",
        "metadata_only",
        "licensed_future",
        "unknown",
    }


def test_is_shareable_allowlist():
    assert is_shareable("open_access") is True
    assert is_shareable("metadata_only") is True
    assert is_shareable("user_supplied_private") is False
    assert is_shareable("licensed_future") is False
    assert is_shareable("unknown") is False
    # fail-closed on unknown / NULL / garbage
    assert is_shareable(None) is False
    assert is_shareable("not_a_real_class") is False
    assert field_allowed("open_access") is True
    assert field_allowed(None) is False


def test_normalize_unknown_to_catch_all():
    assert normalize("Finding", CLAIM_TYPES, fallback=CLAIM_TYPE_OTHER) == "finding"
    assert normalize("totally_made_up", CLAIM_TYPES, fallback=CLAIM_TYPE_OTHER) == CLAIM_TYPE_OTHER
    assert normalize(None, CLAIM_TYPES, fallback=CLAIM_TYPE_OTHER) == CLAIM_TYPE_OTHER


def test_provider_shareable_fields_allowlist():
    for shareable in ("doi", "title", "authors", "year", "venue", "oa_url", "referenced_works"):
        assert shareable in PROVIDER_SHAREABLE_FIELDS
    for excluded in ("abstract", "snippet", "raw_payload", "tdm_full_text"):
        assert excluded not in PROVIDER_SHAREABLE_FIELDS
    # Build D ch10: works.abstract now EXISTS as a stored column — the allowlist
    # still excludes it (local-only per CONTENT_ACCESS_POLICY.md:42; D-10).
    assert "abstract" not in PROVIDER_SHAREABLE_FIELDS

    row = {
        "doi": "10.1/x",
        "title": "A paper",
        "abstract": "secret abstract",
        "snippet": "secret snippet",
        "raw_payload": {"blob": 1},
    }
    projected = project_provider_fields(row)
    assert projected == {"doi": "10.1/x", "title": "A paper"}
