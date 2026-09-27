"""Normalization module: turns raw (business_name, business_address, country)
into a bag of normalized strings + extracted fields used by every later
stage (blocking, features).

Design notes
------------
* Every function here is pure / stateless so it can run under
  multiprocessing (joblib, "loky" backend) without pickling issues -- all
  worker functions are module-level, not closures or lambdas, as required on
  Windows (spawn start method).
* No external lookups: every dictionary lives in :mod:`ber.lexicons` and was
  authored from patterns observed in the provided training data (fair-play
  requirement -- no external DBs/APIs).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from anyascii import anyascii

from . import lexicons as lx

try:
    import jellyfish

    def _metaphone(s: str) -> str:
        try:
            return jellyfish.metaphone(s) if s else ""
        except Exception:
            return ""
except ImportError:  # pragma: no cover - jellyfish is a pinned dependency
    def _metaphone(s: str) -> str:
        return s[:4]

_WS_RE = re.compile(r"\s+")
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_DOMAIN_RE = re.compile(
    r"\b[\w.-]+\.(com|net|org|in|co\.in|fr|io|biz|info)\b", re.IGNORECASE
)
_DBA_RE = re.compile(r"\bdba\b|\bd/b/a\b|\btrading as\b", re.IGNORECASE)
_BRACKETS_RE = re.compile(r"[\[\](){}]")
_PIPE_TAIL_RE = re.compile(r"\s*\|.*$")
_REPEAT_TOKEN_RE = re.compile(r"\b(\w+)( \1\b)+", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")
_INDIA_PIN_RE = re.compile(r"(?<!\d)([1-9]\d{2}\s?\d{3})(?!\d)")
_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_FRANCE_ZIP_RE = re.compile(r"\b(\d{5})\b")
_LANDMARK_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in lx.LANDMARK_PREPOSITIONS) + r")\b\.?\s+([^,]+)",
    re.IGNORECASE,
)
_KE_PAS_RE = re.compile(r"([^,]+?)\s+ke\s+pas\b", re.IGNORECASE)
_CITY_OF_RE = re.compile(r"\b(?:city|town|village)\s+of\s+", re.IGNORECASE)
# Unicode-aware punctuation stripping for the native-script embedding text:
# `_NONALNUM_RE` (ASCII-only) would silently delete every Devanagari/Odia/etc.
# character since none of them are in [a-z0-9]; `\w` under Python's default
# Unicode `re` matches "letter or digit in any script".
_WORD_STRIP_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _has_non_ascii(s: str) -> bool:
    return any(ord(c) > 127 for c in s)


def _unicode_base(s: str, transliterate: bool = True) -> tuple[str, bool]:
    """NFKC normalize and map known native-script business/legal words, then
    (when ``transliterate``) romanize anything still non-ASCII with anyascii.
    Returns (text, had_non_ascii).

    ``transliterate=False`` is used to build the *embedding-only* native-script
    text (see ``_native_name_text``/``_native_addr_text``): anyascii's output
    is phonetic and, for names, frequently doesn't resemble the English name
    at all (e.g. "White Industries" -> "vhait imdstrij") -- useless for the
    string-similarity features it's built for there, but actively harmful as
    embedding input, since it feeds the multilingual embedding model garbled
    ASCII instead of the native script it was actually trained to align
    across languages. Confirmed on this project's data: native-script S2/S3
    records were 2.7x over-represented among candidates blocking dropped
    entirely, on both blockers that touch this text (phonetic and ANN).
    """
    if not s:
        return "", False
    s = unicodedata.normalize("NFKC", s)
    had_non_ascii = _has_non_ascii(s)
    if had_non_ascii:
        for native, repl in lx.NATIVE_LEGAL_WORDS.items():
            if native in s:
                s = s.replace(native, f" {repl} ")
        for native, code in lx.INDIA_STATE_NATIVE.items():
            if native in s:
                s = s.replace(native, f" {code} ")
        if transliterate and _has_non_ascii(s):
            s = anyascii(s)
    if transliterate:
        # strip remaining accents (anyascii already handles most, this is a
        # cheap safety net for latin-1 supplement leftovers). Skipped on the
        # native path: Indic scripts use combining vowel signs (matras) that
        # `unicodedata.combining()` would incorrectly strip right along with
        # genuine Latin accents.
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
    return s, had_non_ascii


def _strip_junk_edges(s: str) -> str:
    s = _PIPE_TAIL_RE.sub("", s)
    s = _BRACKETS_RE.sub(" ", s)
    return s.strip(lx.NAME_JUNK_CHARS + " ")


def _native_name_text(raw: str) -> str:
    """Lightweight native-script counterpart of :func:`normalize_name`, for
    embedding only: same junk/domain/dba stripping and case-folding, but
    skips anyascii so a translit record's original script reaches the
    embedding model. Returns "" when ``raw`` is already pure ASCII (nothing
    to gain -- the caller falls back to the regular ``name_core``)."""
    s = raw or ""
    if not _has_non_ascii(s):
        return ""
    ascii_s, _ = _unicode_base(s, transliterate=False)
    ascii_s = ascii_s.replace("&", " and ")
    ascii_s = _DOMAIN_RE.sub(lambda m: m.group(0).split(".")[0], ascii_s)
    m = _DBA_RE.search(ascii_s)
    if m:
        ascii_s = ascii_s[: m.start()]
    ascii_s = _strip_junk_edges(ascii_s)
    ascii_s = ascii_s.lower()
    ascii_s = _WORD_STRIP_RE.sub(" ", ascii_s)
    return _WS_RE.sub(" ", ascii_s).strip()


def _native_addr_text(raw: str) -> str:
    """Native-script counterpart of :func:`normalize_address`, for embedding
    only. Deliberately simpler than the ASCII path (no abbreviation
    canonicalization, no landmark/pincode extraction -- those dictionaries
    are English-keyed and wouldn't fire on native script anyway): just
    enough cleanup that the embedding model sees real address text instead
    of noise."""
    s = raw or ""
    if not s.strip() or not _has_non_ascii(s):
        return ""
    ascii_s, _ = _unicode_base(s, transliterate=False)
    ascii_s = _CITY_OF_RE.sub("", ascii_s)
    ascii_s = ascii_s.lower()
    ascii_s = _WORD_STRIP_RE.sub(" ", ascii_s)
    return _WS_RE.sub(" ", ascii_s).strip()


@dataclass
class NameNorm:
    name_norm: str      # full normalized name (legal suffixes canonicalized, kept)
    name_core: str       # name_norm with legal-form tokens removed
    name_alt: str        # dba/trading-as alternate name (may be "")
    legal_forms: frozenset
    is_domain: bool
    translit: bool
    first_token: str
    metaphone_key: str


def normalize_name(raw: str) -> NameNorm:
    s = raw or ""
    ascii_s, translit = _unicode_base(s)
    ascii_s = ascii_s.replace("&", " and ")
    is_domain = bool(_DOMAIN_RE.search(ascii_s))
    ascii_s = _DOMAIN_RE.sub(lambda m: m.group(0).split(".")[0], ascii_s)

    name_alt = ""
    m = _DBA_RE.search(ascii_s)
    if m:
        head, tail = ascii_s[: m.start()], ascii_s[m.end():]
        ascii_s, name_alt = head, tail

    ascii_s = _strip_junk_edges(ascii_s)
    ascii_s = ascii_s.lower()
    ascii_s = _NONALNUM_RE.sub(" ", ascii_s)
    ascii_s = _WS_RE.sub(" ", ascii_s).strip()

    tokens = [t for t in ascii_s.split() if t not in lx.NAME_HONORIFICS]
    canon_tokens = [lx.LEGAL_SUFFIXES.get(t, t) for t in tokens]
    # collapse immediate repeats ("llc llc" -> "llc")
    dedup = []
    for t in canon_tokens:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    canon_tokens = dedup

    legal_forms = frozenset(t for t in canon_tokens if t in lx.LEGAL_FORM_TOKENS)
    core_tokens = [t for t in canon_tokens if t not in lx.LEGAL_FORM_TOKENS]

    name_norm = " ".join(canon_tokens)
    name_core = " ".join(core_tokens) or name_norm
    first_token = core_tokens[0] if core_tokens else (canon_tokens[0] if canon_tokens else "")

    alt_norm = ""
    if name_alt:
        alt_ascii, _ = _unicode_base(name_alt)
        alt_ascii = _strip_junk_edges(alt_ascii).lower()
        alt_ascii = _NONALNUM_RE.sub(" ", alt_ascii)
        alt_norm = _WS_RE.sub(" ", alt_ascii).strip()

    return NameNorm(
        name_norm=name_norm,
        name_core=name_core,
        name_alt=alt_norm,
        legal_forms=legal_forms,
        is_domain=is_domain,
        translit=translit,
        first_token=first_token,
        metaphone_key=_metaphone(first_token),
    )


@dataclass
class AddrNorm:
    addr_norm: str        # cleaned, abbreviation-canonicalized, landmark removed
    addr_core: str         # addr_norm with digits stripped (pure text tokens)
    pincode: str           # extracted postal code, "" if none
    house_number: str      # first numeric token, "" if none
    numeric_tokens: frozenset
    state_code: str        # "" if not detected
    city: str              # best-guess city token(s), canonicalized alias
    landmark: str          # extracted landmark phrase, "" if none


def _extract_pincode(s: str, country: str) -> str:
    if country == "India":
        m = _INDIA_PIN_RE.search(s)
        return m.group(1).replace(" ", "") if m else ""
    if country == "US":
        m = _US_ZIP_RE.search(s)
        return m.group(1) if m else ""
    if country == "France":
        m = _FRANCE_ZIP_RE.search(s)
        return m.group(1) if m else ""
    return ""


def _extract_state(ascii_lower: str, country: str) -> str:
    if country == "US":
        for name, code in lx.US_STATE_NAMES.items():
            if re.search(rf"\b{re.escape(name)}\b", ascii_lower):
                return code
        for code in lx.US_STATE_CODES:
            if re.search(rf"\b{code.lower()}\b", ascii_lower):
                return code
    elif country == "India":
        for name, code in lx.INDIA_STATE_NAMES.items():
            if re.search(rf"\b{re.escape(name)}\b", ascii_lower):
                return code
        for code in lx.INDIA_STATE_CODES:
            if re.search(rf"\b{code.lower()}\b", ascii_lower):
                return code
    elif country == "France":
        for name, code in lx.FRANCE_REGION_NAMES.items():
            if name in ascii_lower:
                return code
    return ""


def normalize_address(raw: str, country: str) -> AddrNorm:
    s = raw or ""
    if not s.strip():
        return AddrNorm("", "", "", "", frozenset(), "", "", "")

    pincode = _extract_pincode(s, country)
    ascii_s, _ = _unicode_base(s)
    ascii_s = _CITY_OF_RE.sub("", ascii_s)

    landmark = ""
    m = _LANDMARK_RE.search(ascii_s)
    if m:
        landmark = m.group(1).strip().lower()
        ascii_s = ascii_s[: m.start()] + " " + ascii_s[m.end():]
    else:
        m2 = _KE_PAS_RE.search(ascii_s)
        if m2:
            landmark = m2.group(1).strip().lower()
            ascii_s = ascii_s[: m2.start()] + " " + ascii_s[m2.end():]

    ascii_lower_for_state = ascii_s.lower()
    state_code = _extract_state(ascii_lower_for_state, country)

    ascii_s = ascii_s.lower()
    ascii_s = _NONALNUM_RE.sub(" ", ascii_s)
    ascii_s = _WS_RE.sub(" ", ascii_s).strip()

    abbr = lx.ADDR_ABBR_BY_COUNTRY.get(country, {})
    tokens = []
    for t in ascii_s.split():
        if t in lx.NULL_TOKENS:
            continue
        t = lx.ORDINAL_WORDS.get(t, t)
        t = abbr.get(t, t)
        t = lx.CITY_ALIASES.get(t, t)
        tokens.append(t)
    # drop leading zeros on standalone numeric tokens (005101 -> 5101) and
    # split simple ranges like "408-410" is already handled by regex above
    # (hyphen becomes space), so tokens are already separate here.
    tokens = [str(int(t)) if t.isdigit() and len(t) > 1 and t[0] == "0" else t for t in tokens]

    addr_norm = " ".join(tokens)
    numeric_tokens = frozenset(t for t in tokens if t.isdigit())
    house_number = next((t for t in tokens if t.isdigit()), "")
    addr_core = " ".join(t for t in tokens if not t.isdigit())

    city = ""
    for alias, canon in lx.CITY_ALIASES.items():
        if canon in tokens:
            city = canon
            break

    return AddrNorm(
        addr_norm=addr_norm,
        addr_core=addr_core,
        pincode=pincode,
        house_number=house_number,
        numeric_tokens=numeric_tokens,
        state_code=state_code,
        city=city,
        landmark=landmark,
    )


_OUT_COLUMNS = [
    "name_norm", "name_core", "name_alt", "legal_forms", "is_domain", "translit",
    "first_token", "metaphone_key",
    "addr_norm", "addr_core", "pincode", "house_number", "numeric_tokens",
    "state_code", "city", "landmark",
    "native_name", "native_addr",
]


def _normalize_row(name: str, addr: str, country: str) -> tuple:
    # frozensets are stored as sorted, space-joined strings so this row is
    # parquet/Arrow-serializable (needed for the on-disk resume cache);
    # features.py parses them back with `_parse_set`.
    n = normalize_name(name)
    a = normalize_address(addr, country)
    return (
        n.name_norm, n.name_core, n.name_alt,
        " ".join(sorted(n.legal_forms)), n.is_domain, n.translit,
        n.first_token, n.metaphone_key,
        a.addr_norm, a.addr_core, a.pincode, a.house_number,
        " ".join(sorted(a.numeric_tokens)),
        a.state_code, a.city, a.landmark,
        _native_name_text(name), _native_addr_text(addr),
    )


def _normalize_chunk(names: list, addrs: list, countries: list) -> list[tuple]:
    return [_normalize_row(n, a, c) for n, a, c in zip(names, addrs, countries)]


def normalize_dataframe(df: pd.DataFrame, n_jobs: int = 1, chunk_size: int = 20_000) -> pd.DataFrame:
    """Add normalized/extracted columns to a copy of ``df``.

    ``df`` must have ``business_name``, ``business_address``, ``country``.
    Runs in parallel across ``n_jobs`` processes when > 1.
    """
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    countries = df["country"].tolist()
    n = len(names)

    if n_jobs <= 1 or n < chunk_size * 2:
        rows = _normalize_chunk(names, addrs, countries)
    else:
        from joblib import Parallel, delayed

        chunks = range(0, n, chunk_size)
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_normalize_chunk)(
                names[i : i + chunk_size], addrs[i : i + chunk_size], countries[i : i + chunk_size]
            )
            for i in chunks
        )
        rows = [r for chunk in results for r in chunk]

    out = pd.DataFrame(rows, columns=_OUT_COLUMNS, index=df.index)
    out["embed_text"] = (out["name_core"] + " | " + out["addr_core"]).str.strip()
    # Native-script embedding text: only meaningful (non-empty) for records
    # that had non-ASCII name and/or address content; falls back to the
    # already-transliterated core piece on whichever side has no native
    # variant, rather than embedding a blank half of the pair.
    has_native = (out["native_name"].str.len() > 0) | (out["native_addr"].str.len() > 0)
    native_name_part = out["native_name"].where(out["native_name"].str.len() > 0, out["name_core"])
    native_addr_part = out["native_addr"].where(out["native_addr"].str.len() > 0, out["addr_core"])
    out["embed_text_native"] = np.where(
        has_native, (native_name_part + " | " + native_addr_part).str.strip(), ""
    )
    sn_key_b = out["name_core"].apply(lambda s: " ".join(sorted(s.split())))
    out["sn_key_a"] = out["name_core"]
    out["sn_key_b"] = sn_key_b
    result = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    result.index = df.index
    return result
