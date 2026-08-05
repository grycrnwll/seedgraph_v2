"""Localhost FastAPI read-view router(s) for the citation graph (phase_2).

CLI is the complete surface (decision 81); these are thin localhost-only READ
routes. The router lives in ``web.routes`` as an ``APIRouter`` so the wiring stage
can mount it into the FastAPI app (``api/app.py``) without this phase editing that
shared file — see needs_wiring.

This ``__init__`` imports nothing eagerly to stay import-clean during concurrent
phase landing; import ``web.routes`` directly.
"""

from __future__ import annotations
