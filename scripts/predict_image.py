from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.pipeline import infer_image, prediction_summary_for_cli


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MangoResearch image inference on one JPG/PNG.")
    parser.add_argument("image", type=Path, help="Input JPG or PNG image")
    args = parser.parse_args()
    try:
        payload = args.image.read_bytes()
        result = infer_image(payload, args.image.name)
        print(json.dumps(prediction_summary_for_cli(result), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

