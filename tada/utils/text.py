import re


def normalize_text(text: str) -> str:
    """
    Replace common Unicode punctuation with ASCII equivalents.
    """
    substitutions = {
        # Quotes
        "“": '"',  # LEFT DOUBLE QUOTATION MARK — U+201C — https://www.fileformat.info/info/unicode/char/201c/index.htm
        "”": '"',  # RIGHT DOUBLE QUOTATION MARK — U+201D — https://www.fileformat.info/info/unicode/char/201d/index.htm
        "„": '"',  # DOUBLE LOW-9 QUOTATION MARK — U+201E — https://www.fileformat.info/info/unicode/char/201e/index.htm
        "‟": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK — U+201F — https://www.fileformat.info/info/unicode/char/201f/index.htm
        "‘": "'",  # LEFT SINGLE QUOTATION MARK — U+2018 — https://www.fileformat.info/info/unicode/char/2018/index.htm
        "’": "'",  # RIGHT SINGLE QUOTATION MARK — U+2019 — https://www.fileformat.info/info/unicode/char/2019/index.htm
        "‚": "'",  # SINGLE LOW-9 QUOTATION MARK — U+201A — https://www.fileformat.info/info/unicode/char/201a/index.htm
        "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK — U+201B — https://www.fileformat.info/info/unicode/char/201b/index.htm
        # Dashes
        "–": "-",  # EN DASH — U+2013 — https://www.fileformat.info/info/unicode/char/2013/index.htm
        "—": "-",  # EM DASH — U+2014 — https://www.fileformat.info/info/unicode/char/2014/index.htm
        "―": "-",  # HORIZONTAL BAR — U+2015 — https://www.fileformat.info/info/unicode/char/2015/index.htm
        "‐": "-",  # HYPHEN — U+2010 — https://www.fileformat.info/info/unicode/char/2010/index.htm
        "‑": "-",  # NON-BREAKING HYPHEN — U+2011 — https://www.fileformat.info/info/unicode/char/2011/index.htm
        # Ellipsis
        "…": "...",  # HORIZONTAL ELLIPSIS — U+2026 — https://www.fileformat.info/info/unicode/char/2026/index.htm
        # Misc punctuation
        "‹": "<",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK — U+2039 — https://www.fileformat.info/info/unicode/char/2039/index.htm
        "›": ">",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK — U+203A — https://www.fileformat.info/info/unicode/char/203a/index.htm
        "«": "<<",  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK — U+00AB — https://www.fileformat.info/info/unicode/char/00ab/index.htm
        "»": ">>",  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK — U+00BB — https://www.fileformat.info/info/unicode/char/00bb/index.htm
    }

    pattern = re.compile("|".join(re.escape(char) for char in substitutions))
    text = pattern.sub(lambda m: substitutions[m.group(0)], text)
    text = (
        text.replace("; ", ". ")
        .replace('"', "")
        .replace(":", ",")
        .replace("(", "")
        .replace(")", "")
        .replace("--", "-")
        .replace("-", ", ")
        .replace(",,", ",")
        .replace(" '", " ")
        .replace("' ", " ")
        .replace("  ", " ")
    )

    # Remove spaces before sentence-ending punctuation
    text = re.sub(r"\s+([.,?!])", r"\1", text)

    # Lowercase the text but keep uppercase after sentence-ending punctuation
    text = re.sub(
        r"([.!?]\s*)(\w)",
        lambda m: m.group(1) + m.group(2).upper(),
        text.lower(),
    )

    # Capitalize the very first character
    if text:
        text = text[0].upper() + text[1:]

    return text
