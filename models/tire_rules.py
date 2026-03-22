from typing import List


DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def is_wet_race(compounds_used: List[str]) -> bool:
    """
    Determine if a race is wet based on compounds used.
    """
    return any(c in WET_COMPOUNDS for c in compounds_used)


def validate_compound_rule(compounds_used: List[str]) -> bool:
    """
    F1 Rule:
    - In dry races, at least two different dry compounds must be used.
    - In wet races, this rule does not apply.
    """
    if is_wet_race(compounds_used):
        return True

    unique_dry = set(compounds_used) & DRY_COMPOUNDS
    return len(unique_dry) >= 2


def validate_strategy(strategy: dict) -> bool:
    """
    Validate a full strategy dictionary.
    Expected keys:
    - stints: List[int]
    - compounds: List[str]
    """
    if "stints" not in strategy or "compounds" not in strategy:
        return False

    if len(strategy["stints"]) != len(strategy["compounds"]):
        return False

    if not validate_compound_rule(strategy["compounds"]):
        return False

    if sum(strategy["stints"]) <= 0:
        return False

    return True
