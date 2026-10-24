"""Local LLM helpers that survive stage 2.

- :mod:`.ollama`: the Ollama client the metadata worker uses (``work ollama``).
- :mod:`.parse`: tolerant JSON-object extraction from model or agent output.
- :mod:`.tokens`: the cheap token estimate used to size work packages.

The hosted layer (profiles, routing, cost and budgets, keyring secrets, the executor
and the Anthropic/OpenAI/Gemini adapters) was deleted in stage 2: agents do the model
work through the protocol. Importing this package imports nothing else.
"""
