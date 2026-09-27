"""All hand-authored dictionaries used by normalization.

Kept in one module so the "rules" side of the pipeline is auditable in one
place (relevant for the fair-play review: these are static dictionaries, not
lookups against any external database or API).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Legal suffix canonicalization (name side). Keys are lowercase, alphanumeric
# tokens (after punctuation stripping) mapped to one canonical short token.
# Applied on whole tokens only, so it never matches inside another word.
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    # English / generic
    "private": "pvt", "pvt": "pvt", "pvt.": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltd.": "ltd",
    "corporation": "corp", "corp": "corp", "corp.": "corp",
    "incorporated": "inc", "inc": "inc", "inc.": "inc",
    "company": "co", "co": "co", "co.": "co",
    "llc": "llc", "l.l.c": "llc",
    "llp": "llp", "l.l.p": "llp",
    "pllc": "pllc",
    "pc": "pc", "p.c": "pc",
    "plc": "plc",
    "lp": "lp", "l.p": "lp",
    # French
    "sarl": "sarl", "s.a.r.l": "sarl",
    "sas": "sas",
    "sasu": "sasu",
    "eurl": "eurl",
    "sa": "sa",
    "snc": "snc",
}
# Tokens that indicate a legal form even though they map to themselves; used
# to build the `legal_forms` set feature.
LEGAL_FORM_TOKENS = set(LEGAL_SUFFIXES.values())

# Honorifics / filler words dropped from the front of a name.
NAME_HONORIFICS = {"the", "dr", "dr.", "shri", "sri", "smt", "m/s", "messrs", "messrs."}

# Junk wrapper characters/tokens stripped from name edges.
NAME_JUNK_CHARS = "-<>#*~_=+|"

# Native-script -> ASCII legal/business word map (transliteration misses these
# because they're semantic, not phonetic, translations in some records but
# phonetic transliterations in others -- mapping the common exact strings
# catches the systematic cases; anyascii handles the rest as a fallback).
NATIVE_LEGAL_WORDS = {
    # Devanagari (Hindi/Marathi)
    "प्राइवेट": "private", "प्रा": "pvt", "लिमिटेड": "limited", "लि": "ltd",
    "एलएलपी": "llp", "एलएलसी": "llc", "कंपनी": "company", "कं": "co",
    # Bengali
    "প্রাইভেট": "private", "লিমিটেড": "limited", "লিমি": "ltd",
    "এলএলপি": "llp", "কোম্পানি": "company",
    # Kannada
    "ಪ್ರೈವೇಟ್": "private", "ಲಿಮಿಟೆಡ್": "limited",
    # Tamil
    "பிரைவேட்": "private", "லிமிடெட்": "limited",
    # Telugu
    "ప్రైవేట్": "private", "లిమిటెడ్": "limited",
    # Gujarati
    "પ્રાઇવેટ": "private", "લિમિટેડ": "limited",
    # Gurmukhi (Punjabi)
    "ਪ੍ਰਾਈਵੇਟ": "private", "ਲਿਮਟਿਡ": "limited",
}

# Indian state/UT names (native script + common codes) -> canonical 2-letter
# code. English full names are handled by a separate case-insensitive map.
INDIA_STATE_NATIVE = {
    "राजस्थान": "RJ", "महाराष्ट्र": "MH", "गुजरात": "GJ", "दिल्ली": "DL",
    "उत्तर प्रदेश": "UP", "पश्चिम बंगाल": "WB", "কর্ণাটক": "KA",
    "ಕರ್ನಾಟಕ": "KA", "தமிழ்நாடு": "TN", "తెలంగాణ": "TG", "కేరళ": "KL",
    "কেরালা": "KL", "পাঞ্জাব": "PB", "ਪੰਜਾਬ": "PB", "হরিয়ানা": "HR",
    "मध्य प्रदेश": "MP", "बिहार": "BR", "ओडिशा": "OR", "ওড়িশা": "OR",
    "पंजाब": "PB", "हरियाणा": "HR", "केरल": "KL", "कर्नाटक": "KA",
    "आंध्र प्रदेश": "AP",
}
INDIA_STATE_NAMES = {
    "rajasthan": "RJ", "maharashtra": "MH", "gujarat": "GJ", "delhi": "DL",
    "new delhi": "DL", "uttar pradesh": "UP", "west bengal": "WB",
    "karnataka": "KA", "tamil nadu": "TN", "telangana": "TG", "kerala": "KL",
    "punjab": "PB", "haryana": "HR", "madhya pradesh": "MP", "bihar": "BR",
    "odisha": "OR", "orissa": "OR", "andhra pradesh": "AP", "assam": "AS",
    "jharkhand": "JH", "chhattisgarh": "CG", "uttarakhand": "UK",
    "himachal pradesh": "HP", "goa": "GA", "tripura": "TR", "manipur": "MN",
    "meghalaya": "ML", "nagaland": "NL", "sikkim": "SK", "mizoram": "MZ",
    "arunachal pradesh": "AR", "jammu and kashmir": "JK", "ladakh": "LA",
    "chandigarh": "CH", "puducherry": "PY",
}
US_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
US_STATE_CODES = set(US_STATE_NAMES.values())
INDIA_STATE_CODES = set(INDIA_STATE_NAMES.values())
FRANCE_REGION_NAMES = {
    "ile-de-france": "IDF", "ile de france": "IDF",
    "nouvelle-aquitaine": "NAQ", "nouvelle aquitaine": "NAQ",
    "auvergne-rhone-alpes": "ARA", "occitanie": "OCC",
    "hauts-de-france": "HDF", "hauts de france": "HDF",
    "grand est": "GES", "pays de la loire": "PDL", "bretagne": "BRE",
    "normandie": "NOR", "bourgogne-franche-comte": "BFC",
    "centre-val de loire": "CVL", "provence-alpes-cote d'azur": "PAC",
    "corse": "COR",
}

# City spelling aliases -> canonical form (helps token-overlap features).
CITY_ALIASES = {
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "bangalore": "bengaluru", "poona": "pune", "baroda": "vadodara",
    "cawnpore": "kanpur", "trivandrum": "thiruvananthapuram",
    "mysore": "mysuru", "gurgaon": "gurugram", "allahabad": "prayagraj",
}

# Address abbreviation dictionaries, per country. Each maps a variant token
# to ONE canonical short token so both sides of a match collapse to the same
# string regardless of which variant the source used.
US_ADDR_ABBR = {
    "road": "rd", "rd": "rd",
    "street": "st", "st": "st", "saint": "st",  # "St"->"Saint" misexpansion noise
    "avenue": "ave", "ave": "ave", "av": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr",
    "court": "ct", "ct": "ct",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "place": "pl", "pl": "pl",
    "trail": "trl", "trl": "trl",
    "circle": "cir", "cir": "cir",
    "terrace": "ter", "ter": "ter",
    "square": "sq", "sq": "sq",
    "way": "way",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "apartment": "apt", "apt": "apt",
    "unit": "unit", "suite": "ste", "ste": "ste",
    "floor": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg",
    "number": "no", "no": "no", "#": "no",
}
INDIA_ADDR_ABBR = {
    "nagar": "nagar", "marg": "marg", "road": "rd", "rd": "rd",
    "bypass": "byp", "bye": "byp", "pass": "byp",
    "opposite": "opp", "opp": "opp", "near": "nr", "nr": "nr",
    "building": "bldg", "bldg": "bldg",
    "floor": "fl", "fl": "fl", "flr": "fl", "ground": "grd", "grd": "grd",
    "cross": "crs", "crs": "crs",
    "sector": "sec", "sec": "sec", "phase": "ph", "ph": "ph",
    "colony": "col", "col": "col",
    "taluka": "tal", "tal": "tal", "tehsil": "tal",
    "district": "dist", "dist": "dist",
    "village": "vil", "vil": "vil",
    "block": "blk", "blk": "blk",
    "apartment": "apt", "apartments": "apt", "apt": "apt",
    "society": "soc", "soc": "soc",
}
FRANCE_ADDR_ABBR = {
    "avenue": "av", "av": "av",
    "boulevard": "bd", "boul": "bd", "bd": "bd",
    "chemin": "chem", "chem": "chem",
    "impasse": "imp", "imp": "imp",
    "place": "pl", "pl": "pl",
    "route": "rte", "rte": "rte",
    "allee": "all", "all": "all",
    "faubourg": "fbg", "fbg": "fbg",
    "rue": "rue",
    "saint": "st", "sainte": "ste", "st": "st", "ste": "ste",
}
ADDR_ABBR_BY_COUNTRY = {"US": US_ADDR_ABBR, "India": INDIA_ADDR_ABBR, "France": FRANCE_ADDR_ABBR}

# Ordinal-word -> digit-ordinal (helps "Tenth Avenue" == "10th Avenue").
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
}

# Tokens that mean "no value" and must be dropped, not treated as content.
NULL_TOKENS = {"null", "none", "nan", "n/a", "na", "-", ""}

# Landmark preposition words (English + Hindi postposition) marking a phrase
# to extract out of the address into its own field.
LANDMARK_PREPOSITIONS = [
    "near", "nr", "opp", "opposite", "behind", "beside", "next to",
    "adjacent to", "in front of", "close to",
]
