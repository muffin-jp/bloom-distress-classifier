"""Credential loading for the scripts that call the API.

Kept out of every library module on purpose: reading a file and mutating the
process environment is a side effect, and a side effect that fires on import is
one nobody can see. The two scripts that need a key call :func:`load_env`
explicitly in ``main()``; importing ``dc.schema`` or ``dc.splits`` still touches
nothing.
"""

# python-dotenv ships no py.typed marker, so its import and `load_dotenv` read as
# unknown under pyright strict. Narrowly relax the two rules for this module
# only — the same treatment bloom-langgraph gives sentence_transformers. Our own
# logic below stays fully typed.
# pyright: reportMissingImports=false, reportUnknownVariableType=false
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ENV_PATH", "load_env"]

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    """Load ``.env`` if present, then warn if no Anthropic credential is visible.

    An already-exported variable always wins — ``load_dotenv`` does not override,
    and that precedence is deliberate: a key you set for one command should not
    be silently replaced by a stale file.

    A missing ``ANTHROPIC_API_KEY`` is a warning rather than an error, because it
    is not proof there are no credentials: the SDK also accepts
    ``ANTHROPIC_AUTH_TOKEN`` and an ``ant auth login`` profile, and a bare
    ``AsyncAnthropic()`` picks either up on its own.
    """
    if path.exists():
        from dotenv import load_dotenv

        load_dotenv(path, override=False)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(
            "No ANTHROPIC_API_KEY in the environment or .env. If you have run "
            "`ant auth login` this is fine — the SDK will use that profile. "
            "Otherwise: cp .env.example .env and add your key."
        )
