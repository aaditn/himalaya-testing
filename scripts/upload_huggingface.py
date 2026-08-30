"""Package and upload a Himalaya climb run to Hugging Face."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from himalaya.utils.huggingface import upload, write_model_card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--curriculum", type=Path, default=Path("configs/curriculum.json"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    curriculum = json.loads(
        args.curriculum.read_text(encoding="utf-8")
    )["stages"]
    write_model_card(args.folder, curriculum)
    upload(args.folder, args.repo_id, args.private)
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
