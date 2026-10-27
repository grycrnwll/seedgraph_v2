"""Reference parsing and resolution (pure).

:mod:`.bib_parser` parses a references section; :mod:`.resolve` resolves entries through
the provider chain behind the strong-id gate. Guided growth (``library/growth.py``)
writes ``reference_entries`` and ``citation_links`` itself; the v2 ``citation_edges``
layer was deleted in stage 4.
"""
