"""The v3 web UI (docs/V3_UI_DESIGN.md §1-§2, §8): adapters and one view-model layer.

Deliberately import-free: ``seedgraph.web.jobs`` (the converter lock that ``convert``
checks) must stay importable without pulling Starlette or Jinja2. Build the app with
:func:`seedgraph.web.app.create_app`.
"""
