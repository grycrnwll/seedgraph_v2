"""Seedgraph MCP server package (design 00, plan 01 chunk 0).

A thin, in-process consumer of the service layer (decision 81) exposing the
read/query surface over the Model Context Protocol. Every `mcp` SDK import is
confined to this package's modules (chiefly ``server.py``); the base package and
``seedgraph.cli`` never import the SDK, so a no-extra install imports and runs
the full existing suite (the ``[mcp]`` extra is optional).

This ``__init__`` imports nothing eagerly — not even ``server`` — so ``import
seedgraph.mcp`` alone does not require the SDK. Import the submodules directly:
``from seedgraph.mcp.server import build_server``.
"""
