"""The system turn a reranker judges under.

Kept in its own module so publishing can quote it without importing a model runtime:
the card has to state exactly what the model was given, and `hf_scorer` pulls in torch.
"""

RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
