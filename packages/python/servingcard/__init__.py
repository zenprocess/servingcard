"""servingcard — Hardware-specific LLM serving configurations."""

__version__ = "0.1.0"

from servingcard.schema import (
    BenchmarkEntry,
    BenchmarkSection,
    CapacitySection,
    HardwareDetails,
    QuantizationSection,
    ServingCard,
    ServingSection,
    SpeculativeDecodingSection,
)
from servingcard.validate import validate_card

__all__ = [
    "BenchmarkEntry",
    "BenchmarkSection",
    "CapacitySection",
    "HardwareDetails",
    "QuantizationSection",
    "ServingCard",
    "ServingSection",
    "SpeculativeDecodingSection",
    "validate_card",
]
