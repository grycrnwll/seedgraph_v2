"""Traversal records (v3 stage 5, docs/V3_RECORDS_DESIGN.md).

Records, events and sessions live once in ``library.db``; a graph lists its records
through ``record_graphs`` (§2).

- :mod:`.sessions`: session identity (named, then connection), name validation, helpers,
  the start-date/time name, the two-connection warning, :func:`.sessions.family`;
- :mod:`.log`: the session log: one event per logged call, the first event opens a record,
  the unlogged-call count, closing a connection.
"""
