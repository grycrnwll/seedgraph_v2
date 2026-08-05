import re

from seedgraph.ids import content_id, new_id, sha256_hex


def test_new_id_format_and_uniqueness():
    ids = {new_id("work") for _ in range(1000)}
    assert len(ids) == 1000  # unique
    for value in list(ids)[:20]:
        assert re.fullmatch(r"work_[0-9a-f]{32}", value)


def test_content_id_deterministic():
    a = content_id("md", "alpha", "beta")
    b = content_id("md", "alpha", "beta")
    c = content_id("md", "alpha", "gamma")
    assert a == b
    assert a != c
    assert re.fullmatch(r"md_[0-9a-f]{64}", a)


def test_sha256_hex_known_vectors():
    assert sha256_hex(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_hex(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
