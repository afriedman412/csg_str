import pytest


# This test requires the full ML pipeline (models loaded)
@pytest.mark.skipif(
    True,  # Skip in test mode where models aren't loaded
    reason="Requires full ML pipeline (models not loaded in test mode)"
)
def test_perms_from_url_smoke(client):
    url = "https://www.zillow.com/homedetails/555-W-Cornelia-Ave-APT-1011-Chicago-IL-60657/3717256_zpid/"
    resp = client.post(
        "/api/preds",  # Fixed: was "api/preds_from_url" which doesn't exist
        data={
            "url": url,
            "bedrooms": 2,
            "bathrooms": 2,
            "accommodates": 4,
        },
    )

    assert resp.status_code == 200
    # later: assert on resp.json()
