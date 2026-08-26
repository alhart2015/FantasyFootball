"""View models: pure functions returning frozen dataclasses shaped for a template.

**No Flask import belongs in this package.** That is what makes the interesting logic --
formatting, colour scales, rankings, empty states -- testable without an app, a request, or a
browser, which is where the test density goes.
"""
