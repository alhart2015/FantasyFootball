"""Passphrase encryption for league documents committed to this public repo.

The repo is public and `data/leagues/` is otherwise gitignored, but a draft plan is worth
carrying between machines. This encrypts one under a passphrase so the ciphertext can be
committed while the plaintext stays local.

**Construction** (standard library only -- no new dependency for one file):

    salt      = 16 random bytes, fresh per encryption, stored in the clear
    keys      = PBKDF2-HMAC-SHA256(passphrase, salt, 200_000) -> 32-byte enc + 32-byte mac
    keystream = SHA256(enc_key || counter) for counter = 0, 1, 2, ...
    ciphertext= plaintext XOR keystream
    tag       = HMAC-SHA256(mac_key, salt || ciphertext)

Encrypt-then-MAC, in that order: the tag covers the ciphertext, so a wrong passphrase or a
corrupted file is *detected* rather than silently yielding garbage that looks like a
document. The fresh salt means encrypting the same file twice produces different bytes, so
the ciphertext never reveals that the content was unchanged.

**What this is and is not.** The passphrase never enters the repo, so unlike a one-byte key
this is not brute-forceable by scanning a small space -- an attacker has to guess the phrase
itself. It is therefore only as strong as the phrase: a short or famous one falls to a
dictionary attack, and 200k PBKDF2 iterations slow that down without fixing it. Adequate for
a fantasy draft plan. Still not the place for credentials, cookies, or another person's data
-- put nothing here you would mind seeing published.
"""

from __future__ import annotations

import hashlib
import hmac
import os

MAGIC = b"FFLDOC2\n"
_SALT_LEN = 16
_TAG_LEN = 32
_KEY_LEN = 32
_ITERATIONS = 200_000
_HEADER_LEN = len(MAGIC) + _SALT_LEN + _TAG_LEN


class DecryptionError(ValueError):
    """Raised when a document will not decrypt: wrong passphrase, or a corrupted file."""


def normalize_passphrase(passphrase: str) -> bytes:
    """Encode a passphrase to bytes, rejecting the empty one.

    Deliberately does NOT lowercase, strip, or collapse whitespace. Doing so would make
    "The Proof" and "the  proof" interchangeable, which sounds friendly right up to the
    moment it silently widens the guessable space; the caller types what they typed.
    """
    if not isinstance(passphrase, str):
        raise ValueError(f"passphrase must be a str, got {type(passphrase).__name__}")
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    return passphrase.encode("utf-8")


def _derive(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    """Stretch the passphrase into an encryption key and a separate MAC key."""
    material = hashlib.pbkdf2_hmac(
        "sha256", normalize_passphrase(passphrase), salt, _ITERATIONS, dklen=_KEY_LEN * 2
    )
    return material[:_KEY_LEN], material[_KEY_LEN:]


def _keystream(enc_key: bytes, length: int) -> bytes:
    """SHA256 in counter mode. The counter is what stops every block being identical."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(enc_key + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:length])


def encrypt_bytes(data: bytes, passphrase: str) -> bytes:
    """Encrypt `data`. Output is `MAGIC || salt || tag || ciphertext`, safe to commit."""
    salt = os.urandom(_SALT_LEN)
    enc_key, mac_key = _derive(passphrase, salt)
    # strict=True: the keystream is generated to exactly len(data), so a mismatch is a bug
    # in _keystream -- and zip's default would hide it by silently truncating the output.
    stream = _keystream(enc_key, len(data))
    ciphertext = bytes(x ^ y for x, y in zip(data, stream, strict=True))
    tag = hmac.new(mac_key, salt + ciphertext, hashlib.sha256).digest()
    return MAGIC + salt + tag + ciphertext


def decrypt_bytes(blob: bytes, passphrase: str) -> bytes:
    """Invert `encrypt_bytes`, verifying the tag first.

    Raises `DecryptionError` on a wrong passphrase, a truncated file, or any tampering. The
    tag is checked *before* decrypting so a bad file can never be returned as content.
    """
    if not blob.startswith(MAGIC):
        raise DecryptionError("not a league-doc ciphertext (missing header)")
    if len(blob) < _HEADER_LEN:
        raise DecryptionError("file is truncated: header is incomplete")

    salt = blob[len(MAGIC) : len(MAGIC) + _SALT_LEN]
    tag = blob[len(MAGIC) + _SALT_LEN : _HEADER_LEN]
    ciphertext = blob[_HEADER_LEN:]

    enc_key, mac_key = _derive(passphrase, salt)
    expected = hmac.new(mac_key, salt + ciphertext, hashlib.sha256).digest()
    # compare_digest, not ==, so the comparison does not leak where the tags diverge.
    if not hmac.compare_digest(tag, expected):
        raise DecryptionError(
            "wrong passphrase, or the file has been modified since it was encrypted"
        )
    stream = _keystream(enc_key, len(ciphertext))
    return bytes(x ^ y for x, y in zip(ciphertext, stream, strict=True))


__all__ = [
    "MAGIC",
    "DecryptionError",
    "decrypt_bytes",
    "encrypt_bytes",
    "normalize_passphrase",
]
