"""The exact string a Qwen3 reranker judges, and why it is built by hand.

The checkpoint ships a reranker-specific chat template that emits its own
``<Instruct>``/``<Query>``/``<Document>`` scaffolding and drops the message body, so
``apply_chat_template`` silently produced the same input for every item: one shared
score for all 300 items of a language, and a single predicted class. The model's own
usage builds the turn literally, so this does too.
"""

RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)

RERANKER_PREFIX = f"<|im_start|>system\n{RERANKER_SYSTEM}<|im_end|>\n<|im_start|>user\n"
# The empty thinking block is closed so the very next token is the verdict itself.
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def reranker_input(prompt: str) -> str:
    """Wrap one rendered Instruct/Query/Document body in the turn the model expects."""
    return f"{RERANKER_PREFIX}{prompt.rstrip()}{RERANKER_SUFFIX}"
