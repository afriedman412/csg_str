import pytest
from app.core.config import PRED_OPTIONS, CITY_CENTERS, MODEL_DATA
from app.utils.perm_builder import SCENARIO_DEFAULTS


class TestPredOptions:
    def test_all_values_are_integers(self):
        """PRED_OPTIONS should use integers to match SCENARIO_DEFAULTS"""
        for key, values in PRED_OPTIONS.items():
            for v in values:
                assert isinstance(v, int), f"{key} has non-integer value: {v}"

    def test_all_options_are_binary(self):
        """Each option should have exactly [1, 0] or [0, 1]"""
        for key, values in PRED_OPTIONS.items():
            assert set(values) == {0, 1}, f"{key} is not binary: {values}"

    def test_options_exist_in_scenario_defaults(self):
        """All PRED_OPTIONS keys should exist in SCENARIO_DEFAULTS"""
        for key in PRED_OPTIONS:
            assert key in SCENARIO_DEFAULTS, f"{key} not in SCENARIO_DEFAULTS"


class TestScenarioDefaults:
    def test_amenities_match_pred_options(self):
        """Amenities in SCENARIO_DEFAULTS should be integers matching PRED_OPTIONS type"""
        amenity_keys = list(PRED_OPTIONS.keys())
        for key in amenity_keys:
            val = SCENARIO_DEFAULTS[key]
            assert isinstance(val, int), f"{key} in SCENARIO_DEFAULTS is not int: {type(val)}"

    def test_room_types_are_mutually_exclusive(self):
        """Only one room type should be 1, rest should be 0"""
        room_types = [
            "room_type_entire",
            "room_type_private_room",
            "room_type_shared_room",
            "room_type_hotel_room",
        ]
        values = [SCENARIO_DEFAULTS[k] for k in room_types]
        assert sum(values) == 1, "Room types should have exactly one set to 1"

    def test_privacy_options_are_mutually_exclusive(self):
        """Only one privacy option should be 1, rest should be 0"""
        privacy_opts = [
            "privacy_private",
            "privacy_room_in",
            "privacy_shared",
        ]
        values = [SCENARIO_DEFAULTS[k] for k in privacy_opts]
        assert sum(values) == 1, "Privacy options should have exactly one set to 1"


class TestCityCenters:
    def test_chicago_exists(self):
        assert "chicago-il" in CITY_CENTERS

    def test_coordinates_are_valid(self):
        for city, (lat, lon) in CITY_CENTERS.items():
            assert -90 <= lat <= 90, f"{city} has invalid latitude: {lat}"
            assert -180 <= lon <= 180, f"{city} has invalid longitude: {lon}"

    def test_chicago_coordinates(self):
        lat, lon = CITY_CENTERS["chicago-il"]
        # Chicago is roughly at 41.8, -87.6
        assert 41.5 < lat < 42.5
        assert -88.5 < lon < -87.0


class TestModelData:
    def test_has_required_models(self):
        assert "price" in MODEL_DATA
        assert "occupancy" in MODEL_DATA

    def test_price_model_config(self):
        price = MODEL_DATA["price"]
        assert "target" in price
        assert "transform" in price
        assert "params" in price

    def test_occupancy_model_config(self):
        occ = MODEL_DATA["occupancy"]
        assert "target" in occ
        assert "transform" in occ
        assert "params" in occ
