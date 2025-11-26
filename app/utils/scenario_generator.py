"""
Scenario generator for hypothetical Airbnb listings.

Key principles:
- All inputs start from an AddressData object.
- Static (location-only) features come from build_scenario_base().
- Controllable features (beds, baths, amenities, etc.) are set by permutations.
- ATTOM is completely removed.
- No predictions are computed here — this is pure input generation.
"""

import itertools
import pandas as pd
from typing import List, Dict, Any

from app.schemas.property import AddressData
from app.utils.input_builder import build_scenario_base_from_address

# ============================================================
# CONTROLLABLE FEATURE DOMAIN (SCENARIO SPACE)
# ============================================================

SCENARIO_SPACE = {
    # Independent structural sliders
    "bedrooms": [0, 1, 2, 3, 4],
    "beds": [1, 2, 3, 4, 5, 6],
    "accommodates": [1, 2, 3, 4, 5, 6, 8],
    "bathrooms": [1.0, 1.5, 2.0, 3.0],
    # Amenities
    "air_conditioning": [0, 1],
    "heating": [0, 1],
    "free_parking": [0, 1],
    "paid_parking": [0, 1],
    "gym": [0, 1],
    "housekeeping": [0, 1],
    "pool": [0, 1],
    "pool_private": [0, 1],
    "pool_shared": [0, 1],
    "pool_indoor": [0, 1],
    "pool_outdoor": [0, 1],
    "hot_tub": [0, 1],
    "hot_tub_private": [0, 1],
    "hot_tub_shared": [0, 1],
    # Privacy (mutually exclusive)
    "privacy_private": [0, 1],
    "privacy_room_in": [0, 1],
    "privacy_shared": [0, 1],
    # Room type (mutually exclusive)
    "room_type_entire": [0, 1],
    "room_type_private_room": [0, 1],
    "room_type_shared_room": [0, 1],
    "room_type_hotel_room": [0, 1],
}


# ============================================================
# SCENARIO VALIDATION
# ============================================================


def enforce_constraints(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Apply logical constraints:
    - Bedrooms/beds/accommodates are independent but must be minimally consistent.
    - Privacy and room_type are one-hot exclusive.
    """

    # --- Basic validity ---
    row["bedrooms"] = max(0, row["bedrooms"])
    row["beds"] = max(1, row["beds"])
    row["accommodates"] = max(1, row["accommodates"])

    if row["accommodates"] < row["beds"]:
        row["accommodates"] = row["beds"]

    # --- Privacy exclusivity ---
    priv_cols = ["privacy_private", "privacy_room_in", "privacy_shared"]
    if sum(row[k] for k in priv_cols) > 1:
        return None

    # --- Room type exclusivity ---
    rt_cols = [
        "room_type_entire",
        "room_type_private_room",
        "room_type_shared_room",
        "room_type_hotel_room",
    ]
    if sum(row[k] for k in rt_cols) > 1:
        return None

    return row


# ============================================================
# SCENARIO PERMUTATION GENERATOR
# ============================================================


def generate_scenarios_from_address(
    address: AddressData,
    max_scenarios: int = 300,
) -> pd.DataFrame:
    """
    Full scenario generation pipeline.
    - Build static (location-only) base
    - Enumerate controllable permutations
    - Enforce constraints
    - Output DataFrame for Pops embedding + prediction
    """

    base = build_scenario_base_from_address(address)

    control_keys = list(SCENARIO_SPACE.keys())
    control_values = [SCENARIO_SPACE[k] for k in control_keys]

    scenarios: List[Dict[str, Any]] = []
    count = 0

    for values in itertools.product(*control_values):
        control_map = dict(zip(control_keys, values))
        row_model = base.model_copy(update=control_map)
        row = row_model.model_dump()

        if row is None:
            continue

        scenarios.append(row)
        count += 1

        if max_scenarios and count >= max_scenarios:
            break

    return pd.DataFrame(scenarios)
