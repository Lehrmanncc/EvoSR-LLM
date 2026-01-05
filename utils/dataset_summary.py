#!/usr/bin/env python3
"""Generate dataset summaries for README.md."""
import argparse
import os
import re
from typing import List, Tuple

import pandas as pd

DATASET_START = "<!-- DATASET_SUMMARY_START -->"
DATASET_END = "<!-- DATASET_SUMMARY_END -->"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _summarize_csv(path: str) -> Tuple[int, List[Tuple[str, float, float]]]:
    df = pd.read_csv(path)
    summary = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            summary.append((col, float(series.min()), float(series.max())))
        else:
            summary.append((col, None, None))
    return len(df), summary


def _collect_rows() -> List[Tuple[str, str, str, int, List[Tuple[str, float, float]]]]:
    root = _repo_root()
    data_roots = [
        ("oes", os.path.join(root, "oes_data")),
        ("llm_srbench", os.path.join(root, "llm_srbench_data")),
    ]

    rows = []
    for bench, data_root in data_roots:
        if not os.path.isdir(data_root):
            continue
        for name in sorted(os.listdir(data_root)):
            ds_path = os.path.join(data_root, name)
            if not os.path.isdir(ds_path):
                continue
            for split in ("train", "test_id", "test_ood"):
                csv_path = os.path.join(ds_path, f"{split}.csv")
                if not os.path.exists(csv_path):
                    continue
                nrows, summary = _summarize_csv(csv_path)
                rows.append((bench, name, split, nrows, summary))
    return rows


def _format_summary(rows) -> str:
    lines = []
    lines.append("## Dataset Details")
    lines.append("")
    lines.append("Each dataset folder contains `train.csv`, `test_id.csv` (in-distribution), and `test_ood.csv` (out-of-distribution).")
    lines.append("Ranges are computed per column for each split. All columns are numeric.")

    current = None
    for bench, name, split, nrows, summary in rows:
        if (bench, name) != current:
            lines.append("")
            lines.append(f"### {bench}:{name}")
            current = (bench, name)
        lines.append(f"- {split}: rows={nrows}")
        for col, vmin, vmax in summary:
            if vmin is None:
                lines.append(f"  - {col}: [non-numeric]")
            else:
                lines.append(f"  - {col}: [{vmin:.6g}, {vmax:.6g}]")
    lines.append("")
    return "\n".join(lines)


def _inject_summary(readme_path: str, summary: str) -> None:
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    block = f"{DATASET_START}\n{summary}\n{DATASET_END}"

    if DATASET_START in content and DATASET_END in content:
        content = re.sub(
            rf"{re.escape(DATASET_START)}.*?{re.escape(DATASET_END)}",
            block,
            content,
            flags=re.DOTALL,
        )
    else:
        content = content.rstrip() + "\n\n" + block + "\n"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize datasets for README.md")
    parser.add_argument("--write-readme", action="store_true", help="Update README.md in-place")
    args = parser.parse_args()

    rows = _collect_rows()
    summary = _format_summary(rows)

    if args.write_readme:
        readme_path = os.path.join(_repo_root(), "README.md")
        _inject_summary(readme_path, summary)
    else:
        print(summary)


if __name__ == "__main__":
    main()
