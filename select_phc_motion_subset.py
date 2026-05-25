#!/usr/bin/env python3
"""Select a small PHC motion subset for fast viewer/eval smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=3, type=int)
    parser.add_argument("--keys", nargs="*", default=None)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    data = joblib.load(args.input)
    keys = list(data.keys())

    if args.list_only:
        for idx, key in enumerate(keys[: args.limit]):
            print(f"{idx}: {key}")
        return

    if args.keys:
        selected_keys = args.keys
    else:
        selected_keys = keys[: args.limit]

    missing = [key for key in selected_keys if key not in data]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    subset = {key: data[key] for key in selected_keys}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(subset, args.output)

    print(f"wrote {len(subset)} motions to {args.output}")
    for idx, key in enumerate(selected_keys):
        print(f"{idx}: {key}")


if __name__ == "__main__":
    main()
