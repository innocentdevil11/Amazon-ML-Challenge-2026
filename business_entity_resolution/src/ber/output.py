"""Writes ``matching_results.tsv`` / ``candidate_pairs.tsv`` in the exact
format the challenge requires, and invokes the organizers' own
``validate_submission.py`` as a subprocess so we check against the real
validator rather than a reimplementation of it that could drift.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .io import write_tsv


def _to_id_list_df(id_map: dict[str, set], required_ids: list[str], id_col: str, list_col: str) -> pd.DataFrame:
    rows = []
    for sid in required_ids:
        ids = sorted(id_map.get(sid, set()))
        rows.append({id_col: sid, list_col: ",".join(ids)})
    return pd.DataFrame(rows, columns=[id_col, list_col])


def write_matching_results(pred_map: dict[str, set], required_s1_ids: list[str], path: str | Path) -> None:
    df = _to_id_list_df(pred_map, required_s1_ids, "source1_entity_id", "matched_entity_ids")
    write_tsv(df, path)


def write_candidate_pairs(cand_map: dict[str, set], required_s1_ids: list[str], path: str | Path) -> None:
    df = _to_id_list_df(cand_map, required_s1_ids, "source1_entity_id", "candidate_entity_ids")
    write_tsv(df, path)


def candidates_df_to_map(candidates: pd.DataFrame) -> dict[str, set]:
    return candidates.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()


def run_official_validator(
    matching_path: str | Path, candidate_path: str | Path, test_dir: str | Path,
    validator_script: str | Path, check_ids: bool = False,
) -> tuple[int, str]:
    cmd = [
        sys.executable, str(validator_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")
    # The validator prints an em-dash; on Windows, a piped (non-console)
    # child process falls back to the system codepage (cp1252) for stdout
    # unless PYTHONIOENCODING forces otherwise, which previously crashed the
    # subprocess's own output-reader thread with a UnicodeDecodeError before
    # this function ever saw a return code. Force UTF-8 in the child and
    # replace anything that still doesn't decode, rather than fail the whole
    # validation run over a display character.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
