import re

import mlx.core as mx

__all__ = [
    "normalize_text",
    "gray_code_to_int",
    "decode_gray_code_to_time",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_LANGUAGES_SET",
]

SUPPORTED_LANGUAGES: tuple[str, ...] = ("ar", "ch", "de", "es", "fr", "it", "ja", "pl", "pt")
SUPPORTED_LANGUAGES_SET = frozenset(SUPPORTED_LANGUAGES)


def normalize_text(text: str) -> str:
    substitutions = {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2026": "...",
        "\u2039": "<",
        "\u203a": ">",
        "\u00ab": "<<",
        "\u00bb": ">>",
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
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    text = re.sub(
        r"([.!?]\s*)(\w)",
        lambda m: m.group(1) + m.group(2).upper(),
        text.lower(),
    )

    if text:
        text = text[0].upper() + text[1:]

    return text


def gray_code_to_int(gray: mx.array) -> mx.array:
    binary = gray
    shift = 1

    while shift < 32:
        binary = mx.bitwise_xor(binary, mx.right_shift(binary, mx.array(shift)))
        shift <<= 1

    return binary


def decode_gray_code_to_time(gray_bits: mx.array, num_bits: int) -> mx.array:
    bits_binary = mx.round((gray_bits + 1.0) / 2.0).astype(mx.int32)
    gray_int = mx.zeros(bits_binary.shape[:-1], dtype=mx.int32)

    for i in range(num_bits):
        gray_int = gray_int + mx.left_shift(bits_binary[..., num_bits - 1 - i], mx.array(i))

    return gray_code_to_int(gray_int)
