"""Smoke tests: normalization examples pulled straight from EDA on the real
training data, the README's own worked F0.5 example, and the Windows
line-ending trap the official validator is sensitive to."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.io import write_tsv  # noqa: E402
from ber.metrics import f05_macro, f05_pair  # noqa: E402
from ber.normalize import normalize_address, normalize_name  # noqa: E402


def test_readme_f05_example():
    # From README.md: pred={S2-047,S2-193,S3-812}, true={S2-047,S3-812} -> 0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert f05_pair(pred, true) == pytest.approx(0.7142857, abs=1e-4)


def test_f05_macro_singleton_credit():
    gt = {"S1-1": set(), "S1-2": {"S2-1"}}
    pred_correct = {"S1-1": set(), "S1-2": {"S2-1"}}
    assert f05_macro(pred_correct, gt) == 1.0
    pred_false_merge = {"S1-1": {"S2-9"}, "S1-2": {"S2-1"}}
    assert f05_macro(pred_false_merge, gt) == 0.5  # singleton scored 0, other scored 1


def test_legal_suffix_canonicalization():
    a = normalize_name("Surgical Care Associates, LLC")
    b = normalize_name("surgicalcareassociates.com")
    assert a.name_core == "surgical care associates"
    assert b.is_domain

    a = normalize_name("LLC Moncada Learning Center")
    assert "llc" in a.legal_forms
    assert "moncada" in a.name_core

    a = normalize_name("Pvt. EFS Print Ventures Ltd.")
    assert a.legal_forms == frozenset({"pvt", "ltd"})


def test_repeated_token_collapse_and_dba():
    a = normalize_name("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com")
    assert "vidyalaya vidyalaya" not in a.name_norm
    b = normalize_name("Ectolumdrex dba X+ Madison Inc")
    assert "madison" in b.name_alt


def test_address_abbreviation_and_pincode():
    addr = normalize_address("9808 10th Avenue, Tacoma, WA", "US")
    assert "ave" in addr.addr_norm
    india = normalize_address("797, Lake Town Block A, Kolkata, Howrah, West Bengal 700089", "India")
    assert india.pincode == "700089"
    assert india.state_code == "WB"


def test_landmark_extraction():
    addr = normalize_address("Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra", "India")
    assert "fortis hospital" in addr.landmark


def test_write_tsv_uses_unix_line_endings(tmp_path):
    df = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1,S3-2"]})
    out = tmp_path / "out.tsv"
    write_tsv(df, out)
    raw = out.read_bytes()
    assert b"\r\n" not in raw
