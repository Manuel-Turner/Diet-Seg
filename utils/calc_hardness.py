"""Backward-compatible shim for legacy Dice-error hardness.

The revised Diet-Seg reproducibility release uses teacher-derived entropy
hardness from soft probability maps. This module remains only so older imports
continue to work; new experiments should use utils.entropy_hardness or
scripts/generate_entropy_hardness.py.
"""

from legacy.legacy_calc_error_hardness import HardnessCalculator

__all__ = ["HardnessCalculator"]
