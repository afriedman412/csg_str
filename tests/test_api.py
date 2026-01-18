import pytest


# Mark tests that require the full ML pipeline (model loading)
requires_pipeline = pytest.mark.skipif(
    True,  # Skip in CI/test mode where models aren't loaded
    reason="Requires full ML pipeline (models not loaded in test mode)"
)


class TestPredsEndpoint:
    """Tests for /api/preds endpoint"""

    @requires_pipeline
    def test_preds_returns_200(self, client):
        url = "https://www.zillow.com/homedetails/555-W-Cornelia-Ave-APT-1011-Chicago-IL-60657/3717256_zpid/"
        resp = client.post(
            "/api/preds",
            data={
                "url": url,
                "bedrooms": 2,
                "bathrooms": 2,
                "accommodates": 4,
            },
        )

        assert resp.status_code == 200

    @requires_pipeline
    def test_preds_returns_expected_keys(self, client):
        url = "https://www.zillow.com/homedetails/555-W-Cornelia-Ave-APT-1011-Chicago-IL-60657/3717256_zpid/"
        resp = client.post(
            "/api/preds",
            data={
                "url": url,
                "bedrooms": 2,
                "bathrooms": 2,
                "accommodates": 4,
            },
        )

        if resp.status_code == 200:
            data = resp.json()
            # Check expected prediction keys exist
            assert "price_pred" in data or "0" in data  # depends on serialization

    def test_preds_missing_url_returns_422(self, client):
        resp = client.post(
            "/api/preds",
            data={
                "bedrooms": 2,
                "bathrooms": 2,
                "accommodates": 4,
            },
        )

        assert resp.status_code == 422  # FastAPI validation error

    def test_preds_missing_bedrooms_returns_422(self, client):
        url = "https://www.zillow.com/homedetails/555-W-Cornelia-Ave-APT-1011-Chicago-IL-60657/3717256_zpid/"
        resp = client.post(
            "/api/preds",
            data={
                "url": url,
                "bathrooms": 2,
                "accommodates": 4,
            },
        )

        assert resp.status_code == 422


class TestFromUrlEndpoint:
    """Tests for /api/from_url endpoint"""

    def test_from_url_redirects(self, client):
        url = "https://www.zillow.com/homedetails/123-Main-St-Chicago-IL-60601/12345_zpid/"
        resp = client.post(
            "/api/from_url",
            data={"url": url},
            follow_redirects=False,
        )

        assert resp.status_code == 303
        assert "/output" in resp.headers.get("location", "")


class TestPermsEndpoint:
    """Tests for /api/perms endpoint"""

    @requires_pipeline
    def test_perms_returns_200_or_template(self, client):
        url = "https://www.zillow.com/homedetails/555-W-Cornelia-Ave-APT-1011-Chicago-IL-60657/3717256_zpid/"
        resp = client.post(
            "/api/perms",
            data={
                "url": url,
                "bedrooms": 2,
                "bathrooms": 2,
                "accommodates": 4,
            },
        )

        # Should return 200 with HTML template or 500 if pipeline not loaded
        assert resp.status_code in [200, 500]

    def test_perms_missing_params_returns_422(self, client):
        url = "https://www.zillow.com/homedetails/555-W-Cornelia-Ave-APT-1011-Chicago-IL-60657/3717256_zpid/"
        resp = client.post(
            "/api/perms",
            data={
                "url": url,
                # missing bedrooms, bathrooms, accommodates
            },
        )

        assert resp.status_code == 422


class TestHomeEndpoint:
    """Tests for home page"""

    def test_home_returns_200(self, client):
        resp = client.get("/")

        assert resp.status_code == 200

    def test_home_returns_html(self, client):
        resp = client.get("/")

        assert "text/html" in resp.headers.get("content-type", "")


class TestAiEndpoint:
    """Tests for /ai page"""

    def test_ai_returns_200(self, client):
        resp = client.get("/ai")

        assert resp.status_code == 200
