"""Make a console entrypoint's stdout/stderr able to carry the characters we print.

Python picks its stdout codec from the console's code page. On a Windows console that is
`cp1252`, which cannot encode `Δ`, `★`, `→`, `Σ`, `β`, or any emoji — and the failure mode is
not a mangled glyph, it is `UnicodeEncodeError` raised out of `print`. `scripts/waiver_
recommender.py` printed a `Δ wins` footer and died on it after successfully printing every
recommendation above it.

**Why UTF-8 rather than deleting the characters.** `Δ wins` is the name of the quantity the
waiver tool ranks on; spelling it "Delta wins" everywhere would be a worse report to read, and
it would only hold until the next person types a character outside cp1252. UTF-8 encodes every
codepoint, so once a stream is reconfigured no `print` in the process can raise this again. A
console still in a legacy code page renders the bytes as mojibake rather than the intended
glyph, which is a display problem on that one terminal instead of a crash on every run.

**Call it from `main`, not at import.** Reconfiguring a stream is a process-wide side effect,
and a library that performs one on import would reach into notebooks, test runners, and any
other program that imports `projections`. Only the entrypoint owns the process's stdio.

**A `main` that delegates does not cover its callers.** `scripts/weekly_report.py` calls
`waiver_recommender.report(...)` rather than `waiver_recommender.main(...)`, so the call in the
waiver tool's `main` never runs on the weekly-report path. Every console entrypoint that can
reach a non-ASCII `print` needs its own call; `tests/test_console.py` checks that they have one.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterable
from typing import IO


def force_utf8_stdio(streams: Iterable[IO[str]] | None = None) -> None:
    """Re-encode `sys.stdout` and `sys.stderr` (or `streams`) as UTF-8, in place.

    Streams that cannot be reconfigured are skipped rather than raising: `reconfigure` belongs
    to `io.TextIOWrapper`, and stdout is legitimately something else under `pytest`'s capture,
    inside a subprocess wrapper, or when a caller has replaced it with a `StringIO`. In every
    one of those cases the encoding is already not the console's, so there is nothing to fix.

    Safe to call more than once; the second call is a no-op on an already-UTF-8 stream.
    """
    targets = (sys.stdout, sys.stderr) if streams is None else streams
    for stream in targets:
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
