"""The rubric as a named, versioned, on-disk artefact.

The prompt used to be a pair of string literals in llm_judge.py. That made it
impossible to answer the only question that matters when reading an old result
row -- *which rubric produced this?* -- and it made comparing two rubrics a code
change rather than a configuration change. Both of those cost real time: the
models.yaml and json_mode incidents were each a run whose configuration could
not be reconstructed after the fact.

So: one file per variant under prompts/, selected by id, and the resolved id and
a hash of the exact text are stamped into every result row and every ai-log
payload. A row whose provenance is ambiguous is not evidence.

File format, deliberately plain so a reviewer can read and add one:

    @id v1-frozen-2026-06
    @description one line, shown in listings

    # Lines beginning with '#' outside a body section are commentary and are
    # not part of the prompt. Use them to explain the variant's intent.

    @system
    ...system prompt text, verbatim...

    @rubric
    ...verdict instruction text, verbatim...

A body runs to the next directive or end of file and is stripped of surrounding
blank lines; nothing else about it is touched. `@` at the start of a line is
only a directive if it names a known one, so prompt text may contain `@` freely.

Unknown ids raise. There is no silent fallback anywhere in this module: a
fallback is what produced 540 calls of a vision model that never saw an image.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

def _default_prompts_dir() -> Path:
    """Where the prompt files live, in the container and on the host.

    The host-side tooling imports this module directly to report which rubric a
    run used, so the path cannot simply be the container's. Explicit env wins;
    otherwise take the container path if it exists, else prompts/ next to the
    repository root two levels up from this file.
    """
    env = os.environ.get("PROMPTS_DIR")
    if env:
        return Path(env)
    container = Path("/app/prompts")
    if container.is_dir():
        return container
    return Path(__file__).resolve().parents[2] / "prompts"


PROMPTS_DIR = _default_prompts_dir()
PROMPT_SUFFIX = ".prompt"

# The id used when nothing selects one. This is the rubric every published
# measurement ran under; changing it silently re-bases the whole study.
DEFAULT_PROMPT_ID = "v1-frozen-2026-06"

_INLINE_DIRECTIVES = ("id", "description")
_BODY_DIRECTIVES = ("system", "rubric")


@dataclass(frozen=True)
class Prompt:
    id: str
    description: str
    system: str
    rubric: str
    path: str

    @property
    def hash(self) -> str:
        """Hash of the text actually sent to the model, and of nothing else.

        Commentary, description and filename are excluded on purpose: two files
        differing only in their comments are the same prompt and must produce
        the same hash, or the hash stops being a statement about the experiment.
        The NUL separator keeps (system, rubric) unambiguous under concatenation.
        """
        payload = self.system + "\x00" + self.rubric
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class UnknownPromptError(KeyError):
    """Raised for an id with no corresponding file. Never caught internally."""


def _parse(text: str, path: Path) -> Prompt:
    meta: Dict[str, str] = {}
    bodies: Dict[str, List[str]] = {}
    current: str | None = None

    for line in text.split("\n"):
        directive = None
        if line.startswith("@"):
            head = line[1:].split(None, 1)
            name = head[0] if head else ""
            if name in _INLINE_DIRECTIVES or name in _BODY_DIRECTIVES:
                directive = (name, head[1] if len(head) > 1 else "")

        if directive is not None:
            name, value = directive
            if name in _INLINE_DIRECTIVES:
                meta[name] = value.strip()
                current = None
            else:
                current = name
                bodies[name] = []
            continue

        if current is None:
            continue  # commentary and blank lines between sections
        bodies[current].append(line)

    missing = [k for k in ("id", *_BODY_DIRECTIVES) if k not in meta and k not in bodies]
    if missing:
        raise ValueError(f"prompt file {path} is missing directive(s): {', '.join('@'+m for m in missing)}")

    return Prompt(
        id=meta["id"],
        description=meta.get("description", ""),
        system="\n".join(bodies["system"]).strip("\n"),
        rubric="\n".join(bodies["rubric"]).strip("\n"),
        path=str(path),
    )


def available() -> List[str]:
    if not PROMPTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROMPTS_DIR.glob(f"*{PROMPT_SUFFIX}"))


def load(prompt_id: str) -> Prompt:
    """Load a prompt by id, or raise. The id must equal the filename stem."""
    if not prompt_id:
        raise UnknownPromptError("empty prompt id")
    path = PROMPTS_DIR / f"{prompt_id}{PROMPT_SUFFIX}"
    if not path.is_file():
        raise UnknownPromptError(
            f"unknown prompt id {prompt_id!r}: no {path}. Available: {available() or 'none'}"
        )
    prompt = _parse(path.read_text(), path)
    if prompt.id != prompt_id:
        # A mismatch means the file was copied and its @id not updated, which
        # would stamp the wrong provenance onto every row of the run.
        raise ValueError(f"prompt file {path} declares @id {prompt.id!r}, expected {prompt_id!r}")
    return prompt
