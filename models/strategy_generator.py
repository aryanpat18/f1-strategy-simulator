from itertools import product
from typing import List, Dict
import random
from models.tire_rules import validate_compound_rule

# -----------------------------
# Configuration / Constants
# -----------------------------

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]

MIN_STINT_LAPS = 5

# Compound-specific maximum stint lengths (realistic F1 limits).
# Softs degrade fast, especially on heavy fuel at race start.
# Hards can run very long stints. Inters/wets vary by conditions.
COMPOUND_MAX_STINT = {
    "SOFT": 20,
    "MEDIUM": 30,
    "HARD": 45,
    "INTERMEDIATE": 30,
    "WET": 25,
}

# Global fallback (only used for unknown compounds)
MAX_STINT_LAPS = 45


# -----------------------------
# Core Strategy Generator
# -----------------------------

class StrategyGenerator:
    """
    Generates valid F1 race strategies for a given race and driver.
    Enforces:
      - FIA mandatory two-compound rule (dry races)
      - Compound-specific stint length limits (SOFT ≤ 20, MEDIUM ≤ 30, HARD ≤ 45)
      - Race distance coverage (stints must sum to race_laps)
    """

    def __init__(
        self,
        race_laps: int,
        is_wet_race: bool = False,
        min_stint_laps: int = MIN_STINT_LAPS,
    ):
        self.race_laps = race_laps
        self.is_wet_race = is_wet_race
        self.min_stint_laps = min_stint_laps

        self.allowed_compounds = (
            WET_COMPOUNDS if self.is_wet_race else DRY_COMPOUNDS
        )

    # -----------------------------
    # Public API
    # -----------------------------

    def generate_strategies(
        self,
        max_stops: int = 2,
        max_strategies: int = 50,
    ) -> List[Dict]:
        """
        Generate a list of valid race strategies.
        Compound-aware: SOFT stints capped at 20 laps, MEDIUM at 30, HARD at 45.
        """
        strategies = []

        for num_stops in range(1, max_stops + 1):
            stint_count = num_stops + 1
            compound_sequences = self._generate_compound_sequences(stint_count)

            for compounds in compound_sequences:
                # Skip if dry race rules violated (need 2+ different dry compounds)
                if not self.is_wet_race and not validate_compound_rule(compounds):
                    continue

                # Generate stints respecting per-compound max limits
                stint_combos = self._generate_stint_lengths_for_compounds(compounds)

                for stints in stint_combos:
                    strategy = {
                        "stints": stints,
                        "compounds": compounds,
                        "num_stops": num_stops,
                    }
                    strategies.append(strategy)

                    if len(strategies) >= max_strategies:
                        return strategies

        return strategies

    # -----------------------------
    # Internal Helpers
    # -----------------------------

    def _compound_max(self, compound: str) -> int:
        """Return max stint length for a given compound."""
        return COMPOUND_MAX_STINT.get(compound, MAX_STINT_LAPS)

    def _generate_stint_lengths_for_compounds(
        self,
        compounds: List[str],
    ) -> List[List[int]]:
        """
        Generate realistic stint length combos that:
          - Sum to self.race_laps
          - Each stint respects the compound-specific max
          - Each stint ≥ min_stint_laps
        """
        stint_count = len(compounds)
        max_per_stint = [self._compound_max(c) for c in compounds]
        valid_combinations = []

        # Random sampling approach (efficient for large search spaces)
        for _ in range(400):
            combo = []
            remaining = self.race_laps
            valid = True

            for i in range(stint_count - 1):
                # Upper bound: compound max, but leave room for remaining stints
                min_needed_rest = sum(
                    self.min_stint_laps for _ in range(stint_count - 1 - i)
                )
                upper = min(max_per_stint[i], remaining - min_needed_rest)

                if upper < self.min_stint_laps:
                    valid = False
                    break

                val = random.randint(self.min_stint_laps, upper)
                combo.append(val)
                remaining -= val

            if not valid:
                continue

            # Last stint: check it fits the compound's max
            if (self.min_stint_laps <= remaining <= max_per_stint[stint_count - 1]
                    and combo not in valid_combinations):
                combo.append(remaining)
                # Verify no combo already exists with same values
                if combo not in valid_combinations:
                    valid_combinations.append(combo)

        return valid_combinations

    def _generate_compound_sequences(self, stint_count: int) -> List[List[str]]:
        """
        Generates tire combinations for the given number of stints.
        """
        return [list(seq) for seq in product(self.allowed_compounds, repeat=stint_count)]

    def _is_valid_strategy(self, strategy: Dict) -> bool:
        """
        Full validation (used as a safety check).
        """
        # 1. Mandatory two-compound rule in dry races
        if not self.is_wet_race:
            if not validate_compound_rule(strategy["compounds"]):
                return False

        # 2. Compound-specific stint length check
        for stint_laps, compound in zip(strategy["stints"], strategy["compounds"]):
            if stint_laps < self.min_stint_laps:
                return False
            if stint_laps > self._compound_max(compound):
                return False

        # 3. Must cover exact race distance
        if sum(strategy["stints"]) != self.race_laps:
            return False

        return True
