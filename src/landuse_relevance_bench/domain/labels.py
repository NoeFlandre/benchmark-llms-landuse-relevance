"""The binary decision the benchmark asks each model to make."""

from enum import StrEnum


class Label(StrEnum):
    """A land-use relevance verdict, serialised as the lowercase token itself."""

    YES = "yes"
    NO = "no"
