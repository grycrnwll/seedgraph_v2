"""Real provider HTTP adapters (Track 1, Build Stage B).

Each adapter implements the :class:`~seedgraph.llm.backend.LLMBackend`
``complete()`` seam over raw ``httpx`` (no provider SDKs). ``httpx`` is imported
INSIDE method bodies so importing this package never requires the network dep at
module load. Adapters never read environment variables — secrets/base_url are
resolved centrally in the executor and injected at construction. Transport / HTTP
errors surface as a typed :class:`ProviderError` (never a bare exception), which
the executor maps to an :class:`~seedgraph.llm.executor.LLMResult`.
"""

from ._http import ProviderError, classify_http_error, post_json

__all__ = ["ProviderError", "classify_http_error", "post_json"]
