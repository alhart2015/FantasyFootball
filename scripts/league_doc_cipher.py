"""Encrypt / decrypt a league document so it can ride in this public repo.

Passphrase-based; see `projections.draft.doc_cipher` for the construction and its limits.
Put nothing here you would mind seeing published.

Encrypt (writes `<path>.enc` beside the plaintext, which stays gitignored):

    python scripts/league_doc_cipher.py encrypt \\
        data/leagues/critts_2025_2026/draft_plan.md --passphrase "..."

Decrypt on another machine:

    python scripts/league_doc_cipher.py decrypt \\
        data/leagues/critts_2025_2026/draft_plan.md.enc --passphrase "..."

The passphrase may also come from `FF_LEAGUE_DOC_PASSPHRASE`, which keeps it out of shell
history and `ps` output. `--passphrase` wins if both are set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from projections.draft.doc_cipher import DecryptionError, decrypt_bytes, encrypt_bytes

_SUFFIX = ".enc"
_ENV_VAR = "FF_LEAGUE_DOC_PASSPHRASE"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=["encrypt", "decrypt"])
    p.add_argument("path", type=Path, help="File to transform.")
    p.add_argument(
        "--passphrase",
        default=None,
        help=f"Passphrase. Falls back to ${_ENV_VAR}.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to adding/stripping the .enc suffix.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite the output if it exists.")
    return p.parse_args()


def _default_out(mode: str, path: Path) -> Path:
    if mode == "encrypt":
        return path.with_suffix(path.suffix + _SUFFIX)
    if path.suffix != _SUFFIX:
        raise SystemExit(
            f"{path} does not end in {_SUFFIX}, so the plaintext name cannot be derived. "
            f"Pass --out explicitly."
        )
    return path.with_suffix("")


def main() -> int:
    args = _parse_args()
    passphrase = args.passphrase if args.passphrase is not None else os.environ.get(_ENV_VAR)
    if not passphrase:
        print(f"error: pass --passphrase or set ${_ENV_VAR}", file=sys.stderr)
        return 2
    if not args.path.is_file():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2

    out = args.out if args.out is not None else _default_out(args.mode, args.path)
    # Silently clobbering the plaintext on a decrypt would destroy local edits that exist
    # nowhere else -- the plaintext is gitignored, so there is no copy to recover from.
    if out.exists() and not args.force:
        print(f"error: {out} already exists; pass --force to overwrite", file=sys.stderr)
        return 2

    payload = args.path.read_bytes()
    if args.mode == "encrypt":
        out.write_bytes(encrypt_bytes(payload, passphrase))
    else:
        try:
            out.write_bytes(decrypt_bytes(payload, passphrase))
        except DecryptionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(f"{args.mode}ed {args.path} -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
