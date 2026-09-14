"""`force_utf8_stdio`, plus the repo-wide guard that says which files have to call it."""

from __future__ import annotations

import ast
import io
from pathlib import Path

from projections.console import force_utf8_stdio

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Files that hold a non-cp1252 character in live code but never print one to a console, with
#: the reason. Anything not listed here must call `force_utf8_stdio` — see `_console_guard`.
_NOT_CONSOLE_OUTPUT = {
    "scripts/auction_board.py": "Streamlit app; the emoji are rendered in a browser.",
    "scripts/draft_board.py": "Streamlit app; the emoji are rendered in a browser.",
    "src/projections/draft/assistant/auction/live.py": (
        "The star goes into a DataFrame column that only the Streamlit auction board renders."
    ),
}


def _wrapper(encoding: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding)


def test_reconfigures_the_streams_it_is_given() -> None:
    stream = _wrapper("cp1252")
    force_utf8_stdio([stream])
    assert stream.encoding == "utf-8"


def test_reconfigured_stream_accepts_the_characters_that_used_to_crash() -> None:
    """The actual defect: `Δ` raised `UnicodeEncodeError` out of `print` on a cp1252 console."""
    footer = "Δ wins is a simulated difference."

    before = _wrapper("cp1252")
    try:
        before.write(footer)
        before.flush()
        raised = False
    except UnicodeEncodeError:
        raised = True
    assert raised, "cp1252 is expected to reject Δ; without that there is no bug to fix"

    after = _wrapper("cp1252")
    force_utf8_stdio([after])
    after.write(footer)
    after.flush()  # No exception is the assertion.


def test_is_idempotent() -> None:
    stream = _wrapper("cp1252")
    force_utf8_stdio([stream])
    force_utf8_stdio([stream])
    assert stream.encoding == "utf-8"


def test_skips_streams_that_cannot_be_reconfigured() -> None:
    """`pytest`'s capture replaces stdout with an object that has no `reconfigure`."""
    force_utf8_stdio([io.StringIO()])  # No exception is the assertion.


def _docstring_ids(tree: ast.Module) -> set[int]:
    """`id()` of every docstring constant, so the guard ignores prose about `Δ`."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            out.add(id(body[0].value))
    return out


def _has_unencodable_literal(tree: ast.Module) -> bool:
    docstrings = _docstring_ids(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        try:
            node.value.encode("cp1252")
        except UnicodeEncodeError:
            return True
    return False


def test_every_file_that_prints_an_unencodable_character_forces_utf8() -> None:
    """The regression guard, and the reason this is a repo scan rather than one test per file.

    The bug is not "the waiver tool prints `Δ`" — it is that a character outside cp1252 can be
    added to any console script at any time and will only fail on Windows, only on the line
    that prints it, and only once execution reaches that line. A scan makes adding one a test
    failure at the moment it is written, which is the only point where the author knows whether
    their file writes to a console or to a browser.
    """
    offenders: list[str] = []
    for path in sorted(
        (*(_REPO_ROOT / "scripts").rglob("*.py"), *(_REPO_ROOT / "src").rglob("*.py"))
    ):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        if not _has_unencodable_literal(ast.parse(source)):
            continue
        if rel in _NOT_CONSOLE_OUTPUT:
            continue
        if "force_utf8_stdio()" not in source:
            offenders.append(rel)

    assert not offenders, (
        "These files print a character that a Windows console's cp1252 codec cannot encode, "
        "which raises UnicodeEncodeError out of print(). Call "
        "`projections.console.force_utf8_stdio()` at the top of the entrypoint's main(), or -- "
        "if the text is never written to a console -- add the file to _NOT_CONSOLE_OUTPUT "
        f"with the reason: {offenders}"
    )


def test_exemption_list_has_no_stale_entries() -> None:
    """An exemption that no longer needs to exist is a claim nobody rechecks."""
    stale = [
        rel
        for rel in _NOT_CONSOLE_OUTPUT
        if not _has_unencodable_literal(ast.parse((_REPO_ROOT / rel).read_text(encoding="utf-8")))
    ]
    assert not stale, f"_NOT_CONSOLE_OUTPUT entries no longer hold such a character: {stale}"
