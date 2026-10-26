"""Page routes (thin adapters). Each module here that defines ``ROUTES`` (a list of
Starlette routes) is mounted by :func:`seedgraph.web.app.create_app` in module-name
order; modules whose name starts with ``_`` are skipped. See ``web/app.py`` for the
endpoint conventions (sync ``def``, :func:`~seedgraph.web.app.get_web`, POST for every
side effect)."""
