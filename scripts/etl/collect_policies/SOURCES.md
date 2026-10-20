# Policy collection — free ($0) structured-source inventory

This pipeline builds `policy_registry` from **structured, free, no-LLM** sources only.
Strategy = "Global institution data (tier 1) + major-country structured sources (tier 2)".
Server-Collect: GitHub Actions runner → HF dataset `Robeedau/airlens-live` (`insights-data/policy/`). No client external calls.

## Active sources (in pipeline)

| # | Source | Module | Tier | Cost | Auth | Coverage |
|---|--------|--------|------|------|------|----------|
| 1 | Climate Policy Radar / CCLW (HuggingFace `ClimatePolicyRadar/all-document-text-data`, CC-BY-4.0) | `collect_cpr.py` | 1 — global | Free | `HF_TOKEN` (free) | ~200 AQ policies, global |
| 2 | WHO Air Quality Standards DB / national regulations | `collect_who_standards.py` | 1 — global | Free | none | 88 countries' AQ standards |
| 3 | Government official / UNEP / CCAC major policies | `collect_major_policies.py` | 2 — major countries | Free | none | curated major-emitter policies |
| 4 | US Federal Register (federalregister.gov API v1, keyless) | `collect_federal_register.py` | 2 — major country | Free | none | EPA air-quality RULEs, US |

Merged via `merge_policies.py` (dedup by country+normalized-name) →
`Data/6-policy-analysis/registry/policy_registry.json` (~376 policies / 124 countries) →
`export_frontend.py` (public JSON) → `hf_publish.py upload` → `insights-data/policy/`.

## Continuity (the gap this closes)

- **Weekly cron** = `.github/workflows/policy-collect.yml` (Sun 04:00 UTC). Refreshes the
  light sources (WHO + major). CPR corpus refresh is opt-in (`run_cpr` dispatch input)
  since it's a ~3.59GB HF download — it doesn't change daily.
- **Write** = `hf_publish.py upload` of `policy_registry.json` and the frontend exports to
  `insights-data/policy/` (replaces the earlier database upsert). Each run rebuilds the
  registry from the sources that ran and overwrites the published files, so CPR rows appear
  only after a run with `run_cpr`.
- **CPR in Actions fails on purpose** — the runner does not install `datasets` and the write token
  has no gated-read permission (2026-10-17 decision), so `collect_cpr.py` collects 0 policies.
  It then exits 1 and keeps `cpr.json` unchanged, so a `run_cpr` run fails before publishing
  instead of publishing a registry without CPR. Refreshing CPR needs a run with gated access.
- **Secrets** — `HF_TOKEN` (publish; no gated read). `OPENAI_API_KEY` only for the opt-in LLM
  extraction below.

## Candidate structured sources (next, not in this PR)

Free + keyless + machine-readable. Verified reachable; deferred to keep this change focused.

| Source | Endpoint | Format | Notes |
|--------|----------|--------|-------|
| EUR-Lex | REST / RSS (air pollution directory) | XML/RDF | EU directives. Heavier parsing (SPARQL/RDF). |
| ECOLEX | metadata API | JSON | Overlaps CPR; dedup needed. |

Each would land as a `collect_<source>.py` module merged by `merge_policies.py` (existing pattern).

## Excluded — non-structured crawl ($0 / LLM-0 constraint)

Out of scope **by policy**, not oversight:

- **China (MEE) / India (CPCB) free-text policy pages** — no machine-readable feed; extracting
  structured fields (effective_date, pollutants, standards) from HTML prose requires an LLM.
  That violates the $0 / LLM-0 constraint, so these are excluded. China/India are still covered
  for *standards* via WHO DB (source 2) and for major policies via source 3.
- Any source needing paid API keys or paid LLM enrichment.

LLM-based extraction is now available as a separate **opt-in module** (`collect_llm_extract.py`,
gpt-4o-mini), gated behind `RUN_LLM_EXTRACT=1` + `OPENAI_API_KEY`. The default scheduled cron stays
$0 / LLM-0 — the module runs only on manual `workflow_dispatch` with `run_llm_extract=true`. Add
China (MEE) / India (CPCB) targets to `llm_sources.json` (empty by default) to populate it.
PDF targets are parsed locally via `pdf_extract.py` (pdfplumber, $0) before the LLM pass.
