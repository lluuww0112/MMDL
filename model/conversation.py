"""Fixed conversation presets for project evaluation.

This follows the LLaVA-style pattern: prompt text lives in named, immutable
presets and recipes select a preset by name.
"""

from __future__ import annotations

import ast
import string
from dataclasses import dataclass
from typing import Any, Mapping


def get_conversation(name: str) -> Conversation:
    """Return a named fixed conversation preset."""

    try:
        return CONVERSATION_TEMPLATES[name]
    except KeyError as error:
        supported = ", ".join(sorted(CONVERSATION_TEMPLATES))
        raise ValueError(f"Unknown conversation {name!r}; choose one of: {supported}.") from error


def _has_text(value: Any) -> bool:
    return value is not None and str(value).strip() and str(value).lower() != "nan"


def _format_options(raw_options: Any) -> str:
    if isinstance(raw_options, str):
        try:
            raw_options = ast.literal_eval(raw_options)
        except (SyntaxError, ValueError):
            return ""
    if not isinstance(raw_options, (list, tuple)):
        return ""

    return "\n".join(
        f"{string.ascii_uppercase[index]}. {option}"
        for index, option in enumerate(raw_options)
        if index < len(string.ascii_uppercase)
    )


@dataclass(frozen=True)
class Conversation:
    """Base conversation containing a fixed optional system prompt."""

    name: str
    system_prompt: str | None = None

    def render(self, document: Mapping[str, Any], *, fallback: str) -> str:
        del document
        return fallback


@dataclass(frozen=True)
class MMMUConversation(Conversation):
    """MMMU prompt template used by feature/soogguang's Qwen evaluation."""

    mmmu_template: str = (
        "{hint}Question: {question}\n"
        "Options:\n{options}\n"
        "Please select the correct answer from the options above."
    )

    def render(self, document: Mapping[str, Any], *, fallback: str) -> str:
        question = document.get("question")
        options = _format_options(document.get("options"))
        if not isinstance(question, str) or not question.strip():
            return fallback

        hint = document.get("hint")
        hint_text = f"Hint: {hint}\n" if _has_text(hint) else ""
        if not options:
            return f"{hint_text}Question: {question}".rstrip()
        return self.mmmu_template.format(hint=hint_text, question=question, options=options).rstrip()


# Add new fixed presets here.  YAML only selects one of these names.
CONVERSATION_TEMPLATES: dict[str, Conversation] = {
    "default": Conversation(name="default"),
    "qwen3_vl_mmmu": MMMUConversation(
        name="qwen3_vl_mmmu",
        # The official Qwen3-VL MMMU recipe uses a single user message.
        system_prompt=None,
    ),
}
