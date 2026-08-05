import pytest

from seedgraph.errors import ValidationError
from seedgraph.paths import project_dir, resolve_home, validate_slug


def test_validate_slug_accepts_safe_segments():
    assert validate_slug("my-proj.v2") == "my-proj.v2"
    assert validate_slug("abc_123") == "abc_123"


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "UpperCase", "..", ".", "a b", "x\\y"])
def test_validate_slug_rejects_unsafe(bad):
    with pytest.raises(ValidationError):
        validate_slug(bad)


def test_project_dir_validates_before_filesystem():
    with pytest.raises(ValidationError):
        project_dir("../evil")


def test_project_dir_resolves_under_home():
    pdir = project_dir("good-slug")
    assert pdir == resolve_home() / "projects" / "good-slug"
