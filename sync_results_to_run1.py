#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


def sync_results_to_run1(
    gpt4o_src: Path,
    gpt35_src: Path,
    dest_run1: Path,
    domains: tuple[str, ...] = ("bactgrow", "oscillator1", "oscillator2", "stressstrain"),
    skip_existing: bool = True,
) -> None:
    """
    Copy results into logs_tevc_r1/run1 layout.

    - gpt4o_src provides gpt-4o-mini results with layout: {gpt4o_src}/{domain}/...
    - gpt35_src provides gpt-3.5-turbo results with layout: {gpt35_src}/{domain}/...
    Destination layout:
      {dest_run1}/{domain}/{model}/...
    """
    model_sources = (
        ("gpt-4o-mini", gpt4o_src),
        ("gpt-3.5-turbo", gpt35_src),
    )

    for model, src_root in model_sources:
        for domain in domains:
            src = src_root / domain
            dest = dest_run1 / domain / model

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
    gpt4o_src = repo_root / "results_gpt4o_mini_25_8_6"
    gpt35_src = repo_root / "results_final"
    dest_run1 = repo_root / "logs_tevc_r1" / "run1"

    dest_run1.mkdir(parents=True, exist_ok=True)
    sync_results_to_run1(gpt4o_src, gpt35_src, dest_run1, skip_existing=True)


if __name__ == "__main__":
    main()
