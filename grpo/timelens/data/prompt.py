"""Prompt handling.

Design (see repo README): the ``query`` stored in a data source is treated as the
*final* user text by default -- write the full instruction (task prompt + format
instruction) directly into the jsonl and the training / rollout code uses it verbatim.

For convenience a source may instead provide a ``prompt_template`` containing a
``{query}`` placeholder; in that case the raw query is normalised and substituted
deterministically (no randomness), so a sample's prompt is reproducible across the
off-policy rollout and the GRPO training stage.
"""

from __future__ import annotations

import re


def parse_query(query: str) -> str:
    """Collapse whitespace and strip trailing punctuation from a raw query."""
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


def apply_prompt(raw_query: str, prompt_template: str | None) -> str:
    """Return the final user text.

    ``prompt_template is None`` -> use ``raw_query`` verbatim (already-full query).
    Otherwise substitute the normalised query into ``{query}``.
    """
    if prompt_template is None:
        return raw_query
    return prompt_template.replace("{query}", parse_query(raw_query))


# Optional, ready-to-use temporal-grounding templates. These are not used anywhere
# in the hot path; they exist so example data configs (and ad-hoc data-prep) can
# point at a sensible default. Pick one and inline it into a source's
# ``prompt_template`` field, or compose your own.

GROUNDING_FORMAT_INSTRUCTION = (
    "Return the result strictly as a JSON array of [start, end] pairs, where each "
    "value is a timestamp in seconds. Example: [[0.0, 3.5]] for one segment, or "
    "[[0.0, 3.5], [10.0, 12.0]] for multiple."
)

DEFAULT_GROUNDING_TEMPLATE = (
    "When does '{query}' happen in the video?\n" + GROUNDING_FORMAT_INSTRUCTION
)
