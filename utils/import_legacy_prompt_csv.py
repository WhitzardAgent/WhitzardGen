#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aigc.utils.prompt_import import convert_legacy_prompt_csv_to_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert legacy synthesized prompt CSV files into JSONL files that "
            "aigc run can consume."
        )
    )
    parser.add_argument("--input", required=True, help="Source CSV path.")
    parser.add_argument("--output", required=True, help="Target JSONL path.")
    parser.add_argument(
        "--category",
        default="aigc_safety",
        help="Top-level category to store in metadata. Default: aigc_safety",
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        help="Force all imported prompts to use one normalized language value.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = convert_legacy_prompt_csv_to_jsonl(
        csv_path=args.input,
        jsonl_path=args.output,
        category=args.category,
        forced_language=args.language,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
