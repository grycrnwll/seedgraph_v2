"""Stage 3 acceptance harness (docs/V3_UI_DESIGN.md §10, §11 U10).

Default-suite tests here use synthetic fixture libraries only. The live-serve and
browser tests are opt-in (markers ``live_serve`` and ``browser``). The runnable scripts
(``python -m tests.acceptance.<name>`` from the repo root) work on a real v3 home and
write their reports to ``<home>/ui/acceptance/<run>/``; nothing real is ever committed.
"""
