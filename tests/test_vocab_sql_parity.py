import re
from pathlib import Path

import seedgraph.db.migrations as migrations
from seedgraph.vocab import AccessClass

_SCHEMA_DIR = Path(migrations.__file__).resolve().parent / "schema"


def test_access_class_check_matches_enum():
    enum_values = {ac.value for ac in AccessClass}
    matched_files = 0
    for sql_path in _SCHEMA_DIR.rglob("*.sql"):
        text = sql_path.read_text(encoding="utf-8")
        match = re.search(r"access_class\s+IN\s*\(([^)]*)\)", text)
        if not match:
            continue
        matched_files += 1
        literal_values = set(re.findall(r"'([^']+)'", match.group(1)))
        assert literal_values == enum_values, (
            f"{sql_path.name}: SQL CHECK {literal_values} != AccessClass {enum_values}"
        )
    assert matched_files >= 1, "expected at least one .sql with an access_class CHECK"
