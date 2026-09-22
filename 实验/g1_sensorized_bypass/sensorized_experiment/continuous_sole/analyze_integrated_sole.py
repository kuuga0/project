"""Summarize continuous-sole cell integration and force closure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, object] = {"file": str(path), "rows": len(rows), "feet": {}}
    for foot in ("left", "right"):
        closure = np.asarray([float(r[f"{foot}_force_closure_error_n"]) for r in rows])
        outside = np.asarray([float(r[f"{foot}_outside_fz"]) for r in rows])
        fz_columns = [k for k in rows[0] if k.startswith(f"{foot}_cell_") and k.endswith("_fz")]
        cell_fz = np.asarray([[float(r[k]) for k in fz_columns] for r in rows])
        result["feet"][foot] = {
            "mean_abs_force_closure_error_n": float(np.mean(np.abs(closure))),
            "max_abs_force_closure_error_n": float(np.max(np.abs(closure))),
            "mean_outside_force_n": float(np.mean(outside)),
            "mean_nonzero_normal_cells": float(np.mean(np.count_nonzero(cell_fz > 1e-6, axis=1))),
            "max_nonzero_normal_cells": int(np.max(np.count_nonzero(cell_fz > 1e-6, axis=1))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = {"files": [summarize(path) for path in args.csv_files]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
