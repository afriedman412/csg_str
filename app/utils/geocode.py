# # app/utils/geocode.py

# from cachetools import TTLCache
# import threading
# import time
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
# from geopy.extra.rate_limiter import RateLimiter
# import logging
# from app.core.config import GEOCODER_DOMAIN, GEOCODER_SCHEME, GEOCODER_TIMEOUT, GEOCODER_USER_AGENT

# log = logging.getLogger(__name__)

# _geocoder = None
# _geocode = None
# _reverse = None


# def build_geolocator() -> Nominatim:
#     """Construct the Nominatim geocoder safely."""
#     return Nominatim(
#         user_agent=GEOCODER_USER_AGENT,
#         timeout=GEOCODER_TIMEOUT,
#         domain=GEOCODER_DOMAIN,
#         scheme=GEOCODER_SCHEME,
#     )


# _cache = TTLCache(maxsize=10_000, ttl=60 * 60 * 24)

# # Simple thread lock + rate limit control
# _lock = threading.Lock()
# _last_call = 0.0
# _MIN_INTERVAL = 1.0  # Nominatim etiquette: 1 request/sec


# def geocode_address(address: str, store) -> tuple[float, float] | None:
#     """
#     Geocode an address using the shared Nominatim instance in store.
#     Includes local caching and polite rate-limiting.
#     """
#     # Check cache first
#     if address in _cache:
#         return _cache[address]

#     global _last_call
#     with _lock:
#         dt = time.time() - _last_call
#         if dt < _MIN_INTERVAL:
#             time.sleep(_MIN_INTERVAL - dt)

#         try:
#             loc = store.geolocator.geocode(address)
#         except (GeocoderTimedOut, GeocoderUnavailable):
#             loc = None
#         _last_call = time.time()

#     if loc:
#         coords = (loc.latitude, loc.longitude)
#         _cache[address] = coords
#         return coords

#     return None
