# Content Access Policy

Status: draft policy for Seedgraph v2 design.

This document defines the content-access boundary for Seedgraph v2. It is not legal advice. It is a conservative product and engineering policy intended to reduce the risk that Seedgraph becomes a cross-user substitute for lawful access to scholarly articles.

## 1. Core rule

If an artifact required paywalled full text to produce, do not share that artifact across users by default.

## 2. Local-first default

Seedgraph v2 should default to a local-first architecture. A user may process papers they have lawful access to inside their own research environment. The resulting content artifacts should remain local unless the user deliberately exports them.

Local private content artifacts include:

- PDFs;
- HTML full text;
- Marker output;
- converted markdown;
- block-level full-text JSON;
- images, tables, and equation artifacts;
- evidence-span databases;
- full-text embeddings;
- structured notes derived from full text;
- project semantic graph claims derived from full text.

These artifacts should not be uploaded to or shared through a central multi-user service unless there is a clear license, entitlement model, or user-controlled export workflow.

## 3. Permitted global metadata layer

A future shared metadata service may store public or license-compatible metadata such as:

- DOI;
- OpenAlex ID;
- arXiv ID;
- Semantic Scholar ID;
- title;
- authors;
- year;
- venue;
- public abstract where license permits;
- open-access URL;
- citation metadata from public providers;
- schema templates;
- extraction lens templates;
- software configuration templates.

This layer should not contain paywalled full text or derived full-text substitutes.

## 4. Open-access material

Open-access papers may be cached and processed according to their license terms. The system should record source URL, license metadata when available, acquisition method, and artifact provenance.

## 5. Paywalled material

For paywalled papers, Seedgraph should not bypass access controls. The system may allow a user to manually provide a PDF they can lawfully access. Such files and derived artifacts should remain in the user's local private content cache by default.

## 6. Future publisher or institutional integration

A hosted or multi-user version that processes licensed full text should require:

- user authentication;
- document-level entitlements;
- institutional or publisher access checks;
- access-controlled storage;
- audit logs;
- license-aware retention rules;
- no cross-user leakage of full text or derived full-text artifacts.

## 7. Product wording

Seedgraph v2 should not be described as a full-text article repository, article-sharing system, paywall bypass tool, or substitute for journal access.

Recommended wording:

> Seedgraph is a local-first research harness that lets a researcher process papers they already have lawful access to, cache derived artifacts locally, and build project-specific citation and semantic graphs for LLM-grounded research assistance.
