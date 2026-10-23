"""Synthetic v2 homes for the migration tests (design §8.4, §11 S3). Invented text only.

    from fixtures import build_v2_home
    fx = build_v2_home(tmp_path / "v2home")   # -> V2Fixture(home, works, items)
"""

from .v2home import LENS_DEFINITION_HASH, Planted, V2Fixture, WorkRef, build_v2_home

__all__ = ["LENS_DEFINITION_HASH", "Planted", "V2Fixture", "WorkRef", "build_v2_home"]
