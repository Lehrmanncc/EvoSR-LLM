#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


def sync_srbench_to_run1(
    src_root: Path,
    dest_run1: Path,
    models: tuple[str, ...] = ("gpt-4o-mini", "gpt-3.5-turbo"),
    domains_with_ins: tuple[str, ...] = ("chem", "bio", "phys", "matsci"),
    skip_existing: bool = True,
) -> None:
    """
    Copy results from results_llm_srbench-style layout into logs_tevc_r1/run1 layout.

    Source layout:
      {src_root}/{model}/{domain}/ins{0,1}
    Destination layout:
      {dest_run1}/{domain}{0,1}/{model}
    """
    for model in models:
        for domain in domains_with_ins:
            for idx in (0, 1):
                src = src_root / model / domain / f"ins{idx}"
                dest = dest_run1 / f"{domain}{idx}" / model

                if not src.is_dir():
                    print(f"skip missing: {src}")
                    continue

                if dest.exists():
                    if skip_existing:
                        print(f"skip existing: {dest}")
                        continue
                    shutil.rmtree(dest)

                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, dirs_exist_ok=False)
                print(f"copied: {src} -> {dest}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_root = repo_root / "results_llm_srbench"
    dest_run1 = repo_root / "logs_tevc_r1" / "run1"

    dest_run1.mkdir(parents=True, exist_ok=True)
    sync_srbench_to_run1(src_root, dest_run1, skip_existing=True)


if __name__ == "__main__":
    main()
