"""Tests for the league-document passphrase encryption.

Three things carry the weight: a round trip is lossless for arbitrary bytes; a wrong
passphrase is *detected* rather than returning garbage that looks like a document; and
tampering with the committed ciphertext is caught.
"""

from __future__ import annotations

import pytest

from projections.draft.doc_cipher import (
    MAGIC,
    DecryptionError,
    decrypt_bytes,
    encrypt_bytes,
    normalize_passphrase,
)

PHRASE = "the proof is in the poof"


def test_round_trip_restores_the_document_exactly() -> None:
    plain = "# Critts 2026 — draft-day plan\n\nRB, RB, RB — then WR.\n".encode()

    assert decrypt_bytes(encrypt_bytes(plain, PHRASE), PHRASE) == plain


def test_round_trip_is_lossless_over_every_possible_byte() -> None:
    """A document is arbitrary bytes; a transform that mangles one value is unusable."""
    plain = bytes(range(256)) * 4

    assert decrypt_bytes(encrypt_bytes(plain, PHRASE), PHRASE) == plain


def test_round_trip_spans_many_keystream_blocks() -> None:
    """The keystream is SHA256 in counter mode. A dropped or repeated counter would corrupt
    everything past the first 32 bytes, which a short fixture would never reach."""
    plain = bytes(range(256)) * 400  # ~100 KB, >3000 blocks

    assert decrypt_bytes(encrypt_bytes(plain, PHRASE), PHRASE) == plain


def test_empty_document_round_trips() -> None:
    assert decrypt_bytes(encrypt_bytes(b"", PHRASE), PHRASE) == b""


def test_the_ciphertext_does_not_contain_the_plaintext() -> None:
    """The whole point: readable strings must not survive into the committed file."""
    plain = b"Silence of the Lamb drafts at slot 8 on now_or_never_targeted"
    blob = encrypt_bytes(plain, PHRASE)

    assert b"Silence of the Lamb" not in blob
    assert b"now_or_never_targeted" not in blob


def test_encrypting_twice_gives_different_bytes() -> None:
    """A fresh salt per encryption. Without it, committing an unchanged file would produce an
    identical blob and leak that the plan had not changed."""
    plain = b"# plan"

    assert encrypt_bytes(plain, PHRASE) != encrypt_bytes(plain, PHRASE)
    # ...and both still decrypt.
    assert decrypt_bytes(encrypt_bytes(plain, PHRASE), PHRASE) == plain


def test_a_wrong_passphrase_is_reported_rather_than_returning_garbage() -> None:
    blob = encrypt_bytes(b"# plan", PHRASE)

    with pytest.raises(DecryptionError, match="wrong passphrase"):
        decrypt_bytes(blob, "the proof is in the pudding")


def test_a_near_miss_passphrase_is_still_rejected() -> None:
    """No normalization: case and spacing are part of the phrase."""
    blob = encrypt_bytes(b"# plan", PHRASE)

    for near in ("The proof is in the poof", "the proof is in the poof ", "theproofisinthepoof"):
        with pytest.raises(DecryptionError):
            decrypt_bytes(blob, near)


def test_tampering_with_the_ciphertext_is_caught() -> None:
    """The blob is committed to a public repo, so anyone can edit it. Encrypt-then-MAC means
    a flipped byte fails the tag instead of silently changing the recovered document."""
    blob = bytearray(encrypt_bytes(b"take the RB at pick 8", PHRASE))
    blob[-1] ^= 0x01

    with pytest.raises(DecryptionError, match="modified"):
        decrypt_bytes(bytes(blob), PHRASE)


def test_tampering_with_the_salt_is_caught() -> None:
    """The salt is stored in the clear; the tag covers it so it cannot be swapped."""
    blob = bytearray(encrypt_bytes(b"# plan", PHRASE))
    blob[len(MAGIC)] ^= 0xFF

    with pytest.raises(DecryptionError, match="modified"):
        decrypt_bytes(bytes(blob), PHRASE)


def test_a_non_ciphertext_is_reported_by_its_missing_header() -> None:
    with pytest.raises(DecryptionError, match="missing header"):
        decrypt_bytes(b"just some bytes that were never encrypted", PHRASE)


def test_a_truncated_file_is_reported() -> None:
    blob = encrypt_bytes(b"# plan", PHRASE)

    with pytest.raises(DecryptionError, match="truncated"):
        decrypt_bytes(blob[: len(MAGIC) + 4], PHRASE)


def test_an_empty_passphrase_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_passphrase("")
    with pytest.raises(ValueError, match="must not be empty"):
        encrypt_bytes(b"x", "")


def test_a_non_string_passphrase_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a str"):
        normalize_passphrase(17)  # type: ignore[arg-type]


def test_a_unicode_passphrase_round_trips() -> None:
    plain = b"# plan"
    phrase = "the pröof is in the pööf 🏈"

    assert decrypt_bytes(encrypt_bytes(plain, phrase), phrase) == plain
