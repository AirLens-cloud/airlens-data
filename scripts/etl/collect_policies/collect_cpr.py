#!/usr/bin/env python3
"""
collect_cpr.py — Collect air quality policies from Climate Policy Radar (HuggingFace).

Uses the ClimatePolicyRadar/all-document-text-data dataset (CC-BY-4.0, 3.59GB).
Filters for air quality / air pollution related policies and extracts metadata.

Usage:
  .venv/bin/python scripts/collect_policies/collect_cpr.py
  .venv/bin/python scripts/collect_policies/collect_cpr.py --skip-download  # use cached
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root (scripts/etl/collect_policies → ../../../..)
sys.path.insert(0, str(ROOT / "scripts" / "etl"))

from collect_policies.config import (
    BY_SOURCE_DIR,
    ISO3_TO_ISO2,
    REGIONS,
)

# Air quality related keywords for filtering
AQ_KEYWORDS: set[str] = {
    "air quality", "air pollution", "clean air", "ambient air",
    "particulate matter", "pm2.5", "pm10", "fine dust",
    "nitrogen dioxide", "sulfur dioxide", "ozone",
    "emissions standard", "emission standard", "exhaust emission",
    "vehicle emission", "industrial emission",
    "smog", "haze", "air pollutant",
    "national ambient air quality",
    "air quality management", "air quality monitoring",
    "atmospheric pollution", "air contamination",
    "respiratory", "lung disease",
    "coal power plant", "coal-fired",
    "diesel emission", "euro 6", "euro vi",
    "catalytic converter", "scrubber",
    "desulfurization", "denitrification",
    "short-lived climate pollutant", "slcp",
    "black carbon", "methane emission",
}

# Broader environmental keywords (lower priority)
ENV_KEYWORDS: set[str] = {
    "climate change", "greenhouse gas", "carbon",
    "renewable energy", "energy efficiency",
    "environmental protection", "pollution control",
    "waste management", "water pollution",
}


def is_air_quality_related(title: str, description: str, sectors: list[str]) -> tuple[bool, float]:
    """Check if a document is air quality related. Returns (is_related, relevance_score)."""
    text = f"{title} {description}".lower()
    score = 0.0

    for kw in AQ_KEYWORDS:
        if kw in text:
            score += 2.0

    for kw in ENV_KEYWORDS:
        if kw in text:
            score += 0.3

    if any(s.lower() in ("energy", "transport", "industry") for s in sectors):
        score += 0.5

    return score >= 2.0, score


def extract_type_from_category(category: str, doc_type: str) -> str:
    """Map CPR document categories to our policy types."""
    cat_lower = category.lower() if category else ""
    type_lower = doc_type.lower() if doc_type else ""

    if "law" in type_lower or "act" in type_lower or "legislative" in cat_lower:
        return "law"
    if "regulation" in type_lower:
        return "regulation"
    if "decree" in type_lower or "order" in type_lower:
        return "decree"
    if "plan" in type_lower or "action plan" in type_lower:
        return "plan"
    if "strategy" in type_lower:
        return "strategy"
    if "standard" in type_lower or "guideline" in type_lower:
        return "standard"
    if "policy" in type_lower or "executive" in cat_lower:
        return "regulation"
    return "regulation"


def extract_year(date_str: str | None) -> int | None:
    """Extract year from date string."""
    if not date_str:
        return None
    match = re.search(r"(\d{4})", str(date_str))
    return int(match.group(1)) if match else None


def collect_from_huggingface(skip_download: bool = False) -> list[dict]:
    """Collect air quality policies from the CPR HuggingFace dataset."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow required. Install: pip install pyarrow")
        return []

    cache_dir = BY_SOURCE_DIR / "cpr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try loading from HuggingFace datasets library
    try:
        import os
        from datasets import load_dataset

        # Load HF token from env or secrets file
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            secrets_path = ROOT / "secrets" / "models.env"
            if secrets_path.exists():
                for line in secrets_path.read_text().splitlines():
                    if line.startswith("HF_collect_data="):
                        hf_token = line.split("=", 1)[1].strip()
                        break
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        logger.info("Loading CPR dataset from HuggingFace (this may take a while)...")

        # Load only metadata columns to save memory
        ds = load_dataset(
            "ClimatePolicyRadar/all-document-text-data",
            split="train",
            columns=[
                "document_id",
                "document_metadata.slug",
                "document_metadata.document_title",
                "document_metadata.publication_ts",
                "document_metadata.source_url",
                "document_metadata.type",
                "document_metadata.source",
                "document_metadata.category",
                "document_metadata.geographies",
                "document_metadata.description",
                "document_metadata.metadata.sector",
                "document_metadata.metadata.response",
                "document_metadata.metadata.keyword",
                "document_metadata.metadata.instrument",
                "document_metadata.metadata.framework",
            ],
        )
        logger.info(f"Loaded {len(ds)} rows from CPR dataset")

    except Exception as e:
        logger.warning(f"HuggingFace datasets load failed: {e}")
        logger.info("Falling back to direct parquet download...")
        return collect_from_api_fallback()

    # Deduplicate by document_id
    seen_ids: set[str] = set()
    policies: list[dict] = []

    for row in ds:
        doc_id = row.get("document_id", "")
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        title = row.get("document_metadata.document_title", "") or ""
        description = row.get("document_metadata.description", "") or ""
        sectors = row.get("document_metadata.metadata.sector") or []
        if isinstance(sectors, str):
            sectors = [sectors]

        is_aq, relevance = is_air_quality_related(title, description, sectors)
        if not is_aq:
            continue

        # Extract country code
        geographies = row.get("document_metadata.geographies") or []
        if isinstance(geographies, str):
            geographies = [geographies]

        for geo in geographies:
            country_code = ISO3_TO_ISO2.get(geo, geo[:2] if len(geo) == 2 else "")
            if not country_code or len(country_code) != 2:
                continue

            pub_date = row.get("document_metadata.publication_ts", "")
            year = extract_year(pub_date)
            category = row.get("document_metadata.category", "") or ""
            doc_type = row.get("document_metadata.type", "") or ""
            keywords = row.get("document_metadata.metadata.keyword") or []
            instruments = row.get("document_metadata.metadata.instrument") or []
            responses = row.get("document_metadata.metadata.response") or []
            framework = row.get("document_metadata.metadata.framework") or []

            policy = {
                "id": f"CPR-{country_code}-{doc_id[:8]}",
                "countryCode": country_code,
                "region": REGIONS.get(country_code, "Unknown"),
                "name": title,
                "nameLocal": "",
                "type": extract_type_from_category(category, doc_type),
                "status": "in_force",
                "adoptedDate": pub_date[:10] if pub_date else "",
                "effectiveDate": "",
                "sector": sectors if isinstance(sectors, list) else [sectors],
                "pollutants": [],
                "scope": "national",
                "targets": [],
                "standards": [],
                "source": "Climate Policy Radar / CCLW",
                "sourceUrl": row.get("document_metadata.source_url", ""),
                "description": description[:500] if description else "",
                "relevanceScore": round(relevance, 2),
                "sdidTreatmentYear": year,
                "cprCategory": category,
                "cprType": doc_type,
                "keywords": keywords if isinstance(keywords, list) else [],
                "instruments": instruments if isinstance(instruments, list) else [],
                "responses": responses if isinstance(responses, list) else [],
                "framework": framework if isinstance(framework, list) else [],
            }

            # Detect pollutants from title/description
            text_lower = f"{title} {description}".lower()
            detected_pollutants = []
            pollutant_patterns = {
                "PM2.5": [r"pm\s*2\.?5", r"fine particulate", r"fine dust"],
                "PM10": [r"pm\s*10", r"coarse particulate"],
                "NO2": [r"no2", r"nitrogen dioxide", r"nox"],
                "SO2": [r"so2", r"sulfur dioxide", r"sulphur dioxide"],
                "O3": [r"ozone", r"o3"],
                "CO": [r"\bco\b", r"carbon monoxide"],
                "VOC": [r"\bvoc\b", r"volatile organic"],
                "BC": [r"black carbon", r"soot"],
                "NH3": [r"ammonia", r"nh3"],
            }
            for pollutant, patterns in pollutant_patterns.items():
                if any(re.search(p, text_lower) for p in patterns):
                    detected_pollutants.append(pollutant)
            policy["pollutants"] = detected_pollutants

            policies.append(policy)

    logger.info(f"Extracted {len(policies)} air quality policies from CPR dataset")
    return policies


def collect_from_api_fallback() -> list[dict]:
    """Fallback: search climate-laws.org via their search page."""
    logger.info("API fallback not yet implemented. Use HuggingFace dataset.")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect policies from Climate Policy Radar")
    parser.add_argument("--skip-download", action="store_true", help="Use cached data")
    args = parser.parse_args()

    output_path = BY_SOURCE_DIR / "cpr.json"

    # Check for cached result
    if args.skip_download and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        logger.info(f"Using cached CPR data: {len(data)} policies")
        return

    policies = collect_from_huggingface(skip_download=args.skip_download)

    # Zero policies means the collection failed (no `datasets` library or no gated access, which is
    # the Actions run_cpr path). Writing [] would let merge publish without CPR and report success.
    if not policies:
        logger.error(f"CPR collection returned 0 policies; leaving {output_path} unchanged")
        sys.exit(1)

    BY_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(policies, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(policies)} policies to {output_path}")

    # Summary
    countries = {p["countryCode"] for p in policies}
    logger.info(f"Countries covered: {len(countries)}")
    by_type = {}
    for p in policies:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    logger.info(f"By type: {by_type}")


if __name__ == "__main__":
    main()
