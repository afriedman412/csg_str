# app/core/store.py
from dataclasses import dataclass, field
import geopandas as gpd
import numpy as np
from functools import cached_property
from shapely.strtree import STRtree
from typing import Optional, Dict, Any, Tuple
from cachetools import TTLCache
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from app.core.config import (
    GEOCODER_DOMAIN,
    GEOCODER_SCHEME,
    GEOCODER_TIMEOUT,
    GEOCODER_USER_AGENT,
    GEOCODER_CACHE_SIZE,
    GEOCODER_CACHE_TTL_SEC,
    GEOCODER_MIN_INTERVAL_SEC,
)
import threading
import time


@dataclass(frozen=True)
class DataStore:
    pipeline: Any
    gdf_features: gpd.GeoDataFrame
    zips: gpd.GeoDataFrame
    trees: Optional[gpd.GeoDataFrame]
    feature_index: STRtree
    zip_bounds_minx: np.ndarray
    zip_bounds_miny: np.ndarray
    zip_bounds_maxx: np.ndarray
    zip_bounds_maxy: np.ndarray
    meta: Dict[str, Any]
    geolocator: Nominatim

    # internal mutable helpers (excluded from equality/repr; OK with frozen dataclass)
    _geo_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _rate_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _last_call: float = field(default=0.0, repr=False, compare=False)
    _cache: TTLCache = field(
        default_factory=lambda: TTLCache(
            maxsize=GEOCODER_CACHE_SIZE,
            ttl=GEOCODER_CACHE_TTL_SEC,
        ),
        repr=False,
        compare=False,
    )

    @cached_property
    def geolocator(self) -> Nominatim:
        """Initialize the geocoder once per DataStore instance (per worker)."""
        # cached_property works fine on frozen dataclasses
        with self._geo_lock:
            return Nominatim(
                user_agent=GEOCODER_USER_AGENT,
                timeout=GEOCODER_TIMEOUT,
                domain=GEOCODER_DOMAIN,
                scheme=GEOCODER_SCHEME,
            )

    # ---- Public helper: cached, rate-limited geocode -----------------------
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        # 1) cache
        if address in self._cache:
            return self._cache[address]

        # 2) rate limit
        with self._rate_lock:
            dt = time.time() - self._last_call
            if dt < GEOCODER_MIN_INTERVAL_SEC:
                time.sleep(GEOCODER_MIN_INTERVAL_SEC - dt)

            try:
                loc = self.geolocator.geocode(address)
            except (GeocoderTimedOut, GeocoderUnavailable):
                loc = None

            self._last_call = time.time()

        if loc:
            coords = (loc.latitude, loc.longitude)
            self._cache[address] = coords
            return coords
        return None
