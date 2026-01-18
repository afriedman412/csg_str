import pytest
import pandas as pd
import numpy as np
from app.utils.perm_builder import (
    dict_permutations,
    assemble_options,
    get_original_match_mask,
    scenario_df_verify,
    SCENARIO_DEFAULTS,
)
from app.core.config import PRED_OPTIONS


class TestDictPermutations:
    def test_single_key(self):
        options = {"a": [1, 2, 3]}
        result = dict_permutations(options)
        assert result == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_two_keys(self):
        options = {"a": [1, 2], "b": ["x", "y"]}
        result = dict_permutations(options)
        assert len(result) == 4
        assert {"a": 1, "b": "x"} in result
        assert {"a": 1, "b": "y"} in result
        assert {"a": 2, "b": "x"} in result
        assert {"a": 2, "b": "y"} in result

    def test_empty_dict(self):
        result = dict_permutations({})
        assert result == [{}]

    def test_pred_options_count(self):
        """PRED_OPTIONS should generate 2^5 = 32 permutations (5 binary amenities)"""
        result = dict_permutations(PRED_OPTIONS)
        assert len(result) == 32


class TestScenarioDfVerify:
    def test_adds_required_columns(self):
        df = pd.DataFrame({"bedrooms": [1, 2], "bathrooms": [1.5, 2.0]})
        result = scenario_df_verify(df)

        assert "city" in result.columns
        assert "slice" in result.columns
        assert "avg_price" in result.columns
        assert "med_price" in result.columns

    def test_sets_chicago_defaults(self):
        df = pd.DataFrame({"bedrooms": [1]})
        result = scenario_df_verify(df)

        assert result["city"].iloc[0] == "chicago-il"
        assert result["slice"].iloc[0] == "chicago-il"
        assert result["avg_price"].iloc[0] == 577.59
        assert result["med_price"].iloc[0] == 169.0


class TestAssembleOptions:
    def test_generates_permutations(self):
        base = {
            "latitude": 41.9,
            "longitude": -87.6,
            **SCENARIO_DEFAULTS,
        }
        input_values = {
            "bedrooms": 2,
            "bathrooms": 2.0,
            "accommodates": 4,
            "beds": 2,
        }

        result = assemble_options(base, input_values)

        # Should have permutations: 3 accommodates × 32 amenity combos = 96 rows
        assert len(result) == 3 * 32

    def test_accommodates_variations(self):
        base = {
            "latitude": 41.9,
            "longitude": -87.6,
            **SCENARIO_DEFAULTS,
        }
        input_values = {
            "bedrooms": 2,
            "bathrooms": 2.0,
            "accommodates": 4,
            "beds": 2,
        }

        result = assemble_options(base, input_values)

        # Should have accommodates 4, 5, 6
        assert set(result["accommodates"].unique()) == {4, 5, 6}

    def test_preserves_base_columns(self):
        base = {
            "latitude": 41.9,
            "longitude": -87.6,
            "custom_col": "test_value",
            **SCENARIO_DEFAULTS,
        }
        input_values = {
            "bedrooms": 2,
            "bathrooms": 2.0,
            "accommodates": 4,
            "beds": 2,
        }

        result = assemble_options(base, input_values)

        assert "custom_col" in result.columns
        assert (result["custom_col"] == "test_value").all()


class TestGetOriginalMatchMask:
    def test_returns_correct_mask(self):
        df = pd.DataFrame({
            "accommodates": [4, 4, 5, 5],
            "bedrooms": [2, 2, 2, 2],
            "bathrooms": [2.0, 2.0, 2.0, 2.0],
            "beds": [2, 2, 2, 2],
            "pool": [0, 1, 0, 1],
            "hot_tub": [0, 0, 0, 0],
            "gym": [0, 0, 0, 0],
            "housekeeping": [0, 0, 0, 0],
            "free_parking": [0, 0, 0, 0],
        })
        input_values = {
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms": 2.0,
            "beds": 2,
        }

        mask = get_original_match_mask(df, input_values)

        # Should match row 0 (accommodates=4, all amenities=0)
        assert mask.sum() == 1
        assert mask.iloc[0] == True
        assert mask.iloc[1] == False  # pool=1

    def test_does_not_mutate_input_values(self):
        df = pd.DataFrame({
            "accommodates": [4],
            "bedrooms": [2],
            "bathrooms": [2.0],
            "beds": [2],
            "pool": [0],
            "hot_tub": [0],
            "gym": [0],
            "housekeeping": [0],
            "free_parking": [0],
        })
        input_values = {
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms": 2.0,
            "beds": 2,
        }
        original_keys = set(input_values.keys())

        get_original_match_mask(df, input_values)

        # input_values should not have been modified
        assert set(input_values.keys()) == original_keys
        assert "pool" not in input_values

    def test_handles_float_comparison(self):
        df = pd.DataFrame({
            "accommodates": [4],
            "bedrooms": [2],
            "bathrooms": [2.0],
            "beds": [2],
            "pool": [0],
            "hot_tub": [0],
            "gym": [0],
            "housekeeping": [0],
            "free_parking": [0],
        })
        input_values = {
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms": 2.0,  # float
            "beds": 2,
        }

        mask = get_original_match_mask(df, input_values)

        assert mask.sum() == 1

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({
            "accommodates": [4],
            "bedrooms": [2],
            # missing bathrooms
        })
        input_values = {
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms": 2.0,
            "beds": 2,
        }

        with pytest.raises(KeyError):
            get_original_match_mask(df, input_values)


class TestScenarioDefaults:
    def test_has_required_keys(self):
        required = [
            "bedrooms", "beds", "accommodates", "bathrooms",
            "air_conditioning", "heating", "free_parking",
            "pool", "hot_tub", "gym", "housekeeping",
        ]
        for key in required:
            assert key in SCENARIO_DEFAULTS

    def test_amenities_are_integers(self):
        amenity_keys = ["pool", "hot_tub", "gym", "housekeeping", "free_parking"]
        for key in amenity_keys:
            assert isinstance(SCENARIO_DEFAULTS[key], int)
