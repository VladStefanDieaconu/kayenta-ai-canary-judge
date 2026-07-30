"""Load the repository's .env into the process environment, once, at import.

Every entry point in tools/ takes its defaults from environment variables --
`EVAL_N`, `EVAL_SEED`, `EVAL_MODELS`, `JUDGE_PROMPT_ID` -- and `.env` is where
this repository keeps them. Docker Compose reads that file automatically; a bare
`python tools/…` does not, and neither does `nohup python tools/…`.

The consequence has bitten three separate entry points: a run launched without
the shell that had sourced `.env` silently used `DEFAULT_N_PER_FAMILY = 12`
instead of the configured 20, produced a 108-scenario dataset with different
per-scenario seeds, and looked entirely healthy while doing it. Patching the
fourth script the same way was not a fix, so the loading moved here and every
entry point imports it.

    import repo_env  # noqa: F401  - loads .env before argparse reads defaults

Import order matters: this has to run before any `os.environ.get` that supplies
an argparse default, which is why it is a module-level side effect rather than a
function callers must remember to call. That is the whole point of the module.

A variable already set in the real environment always wins. An explicit
`EVAL_N=5 python tools/run_experiment.py` must not be silently overridden by the
file, and CI sets its own values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV = Path(os.environ.get("REPO_DOTENV", REPO_ROOT / ".env"))

# Recorded so a runner can print where its configuration came from. A run whose
# configuration cannot be reconstructed is not evidence.
loaded_from: str = ""
loaded_keys: List[str] = []


def parse(text: str) -> Dict[str, str]:
    """Compose-compatible subset of .env: KEY=VALUE, '#' comments, optional quotes.

    Deliberately not a general shell parser. Docker Compose does not perform
    command substitution or variable expansion in this file either, so a parser
    that did would disagree with the container about what the value is.
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load(path: Path = DOTENV, override: bool = False) -> List[str]:
    """Put `path`'s variables into os.environ. Returns the names actually set."""
    global loaded_from, loaded_keys
    if not path.is_file():
        return []
    applied = []
    for key, value in parse(path.read_text()).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    loaded_from = str(path)
    loaded_keys = applied
    return applied


def describe() -> str:
    """One line for a run banner, so the log records where settings came from."""
    if not loaded_from:
        return "no .env loaded (using the ambient environment)"
    return f"loaded {len(loaded_keys)} variable(s) from {loaded_from}"


load()
