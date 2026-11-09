from __future__ import annotations

import argparse
import sys

import httpx
from google.genai import Client
from google.genai.errors import APIError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from config import get_settings


def check_gemini(model: str, api_key: str) -> bool:
    client = Client(api_key=api_key)
    try:
        client.models.get(model=model)
    except APIError as exc:
        print(
            f"Gemini check failed ({type(exc).__name__}, "
            f"HTTP {exc.code}): check the API key, model name, and network."
        )
        return False
    except httpx.HTTPError as exc:
        print(f"Gemini check failed ({type(exc).__name__}): check network access.")
        return False
    except RuntimeError as exc:
        print(
            "Gemini check failed: the SDK closed its HTTP client before the request "
            f"completed ({type(exc).__name__})."
        )
        return False
    finally:
        client.close()
    print(f"Gemini is reachable and model '{model}' is available.")
    return True


def check_qdrant(url: str, api_key: str) -> bool:
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
    except (ValueError, httpx.HTTPError) as exc:
        print(f"Qdrant check failed ({type(exc).__name__}): check the URL and network.")
        return False
    try:
        client.get_collections()
    except UnexpectedResponse as exc:
        print(
            f"Qdrant check failed ({type(exc).__name__}, "
            f"HTTP {exc.status_code}): check the URL, API key, and permissions."
        )
        return False
    except (ResponseHandlingException, httpx.HTTPError) as exc:
        print(f"Qdrant check failed ({type(exc).__name__}): check network access.")
        return False
    finally:
        client.close()
    print("Qdrant is reachable; no collections or points were modified.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Gemini and Qdrant connectivity.")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--qdrant-only",
        action="store_true",
        help="Check Qdrant without making a Gemini API request.",
    )
    checks.add_argument(
        "--gemini-only",
        action="store_true",
        help="Check Gemini without contacting Qdrant.",
    )
    args = parser.parse_args()
    settings = get_settings()
    gemini_key = settings.gemini_api_key
    qdrant_url = settings.qdrant_url
    qdrant_key = settings.qdrant_api_key
    checks_run = 0
    checks_failed = 0

    if gemini_key and not args.qdrant_only:
        checks_run += 1
        checks_failed += not check_gemini(settings.gemini_model, gemini_key)
    elif not args.qdrant_only:
        print("Gemini check skipped: GEMINI_API_KEY is not configured.")

    if qdrant_url and qdrant_key and not args.gemini_only:
        checks_run += 1
        checks_failed += not check_qdrant(qdrant_url, qdrant_key)
    elif not args.gemini_only:
        missing_qdrant = [
            name
            for name, value in (
                ("QDRANT_URL", qdrant_url),
                ("QDRANT_API_KEY", qdrant_key),
            )
            if not value
        ]
        print(
            "Qdrant check skipped: missing "
            + ", ".join(missing_qdrant)
            + "."
        )

    if checks_run == 0:
        print("Add credentials locally to .env; never paste keys into chat.")
        return 2
    return 1 if checks_failed else 0


if __name__ == "__main__":
    sys.exit(main())
