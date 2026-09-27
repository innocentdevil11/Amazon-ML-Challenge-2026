#!/usr/bin/env python3
"""Builds ``pipeline.ipynb`` from ``notebook_cells.py`` -- stdlib ``json``
only, no ``nbformat`` dependency. Splits on the ``# %% [n] SECTION`` markers
so the same source file is both a runnable script and the notebook.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CELL_MARKER = re.compile(r"^# %% \[(\d+)\] (.+)$")


def build_cells(src: str) -> list[dict]:
    lines = src.splitlines()
    cells: list[dict] = []
    current: list[str] = []
    header = None

    def flush():
        if header is None:
            return
        body = "\n".join(current).strip("\n")
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [f"# {header}\n"] + [l + "\n" for l in body.splitlines()],
        })

    for line in lines:
        m = CELL_MARKER.match(line)
        if m:
            flush()
            header = m.group(2)
            current = []
        else:
            current.append(line)
    flush()
    return cells


def main() -> int:
    src_path = Path(__file__).resolve().parent / "notebook_cells.py"
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "pipeline.ipynb"

    notebook = {
        "cells": build_cells(src_path.read_text(encoding="utf-8")),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
