"""Tests for the Python WhatsApp sidecar HTTP client."""
from __future__ import annotations

import json

import httpx
import pytest
import respx
from httpx import Response

from src.core.exceptions import SidecarError, SidecarNotReadyError
from src.whatsapp.sidecar_client import SidecarClient, SidecarStatus

BASE = "http://sidecar:3000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return SidecarClient(BASE)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestSidecarClientConstructor:
    def test_empty_base_url_raises(self):
        with pytest.raises(ValueError, match="base_url"):
            SidecarClient("")

    def test_blank_base_url_raises(self):
        with pytest.raises(ValueError, match="base_url"):
            SidecarClient("   ")

    def test_trailing_slash_stripped(self):
        c = SidecarClient("http://sidecar:3000/")
        assert not c._base.endswith("/")

    def test_default_timeout_is_positive(self):
        c = SidecarClient(BASE)
        assert c._timeout > 0

    def test_custom_timeout_accepted(self):
        c = SidecarClient(BASE, timeout=30.0)
        assert c._timeout == 30.0


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------

class TestHealth:
    @respx.mock
    async def test_returns_json_on_200(self, client):
        respx.get(f"{BASE}/health").mock(
            return_value=Response(200, json={"status": "ok", "whatsapp_ready": True})
        )
        data = await client.health()
        assert data["status"] == "ok"

    @respx.mock
    async def test_raises_sidecar_error_on_500(self, client):
        respx.get(f"{BASE}/health").mock(return_value=Response(500, text="error"))
        with pytest.raises(SidecarError, match="500"):
            await client.health()

    @respx.mock
    async def test_raises_sidecar_error_on_network_failure(self, client):
        respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(SidecarError):
            await client.health()

    @respx.mock
    async def test_raises_sidecar_error_on_invalid_json(self, client):
        respx.get(f"{BASE}/health").mock(
            return_value=Response(200, content=b"not-json", headers={"content-type": "text/plain"})
        )
        with pytest.raises(SidecarError, match="JSON"):
            await client.health()


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------

class TestStatus:
    @respx.mock
    async def test_returns_sidecar_status_when_ready(self, client):
        respx.get(f"{BASE}/status").mock(
            return_value=Response(200, json={
                "whatsapp_ready": True,
                "community_ids": ["abc@g.us", "def@g.us"],
                "pipeline_url": "http://app:8000",
            })
        )
        status = await client.status()
        assert isinstance(status, SidecarStatus)
        assert status.whatsapp_ready is True
        assert status.community_ids == ["abc@g.us", "def@g.us"]
        assert status.pipeline_url == "http://app:8000"

    @respx.mock
    async def test_raises_not_ready_when_whatsapp_false(self, client):
        respx.get(f"{BASE}/status").mock(
            return_value=Response(200, json={
                "whatsapp_ready": False,
                "community_ids": [],
                "pipeline_url": "http://app:8000",
            })
        )
        with pytest.raises(SidecarNotReadyError):
            await client.status()

    @respx.mock
    async def test_not_ready_error_is_subclass_of_sidecar_error(self, client):
        respx.get(f"{BASE}/status").mock(
            return_value=Response(200, json={"whatsapp_ready": False})
        )
        with pytest.raises(SidecarError):
            await client.status()

    @respx.mock
    async def test_raises_sidecar_error_on_http_failure(self, client):
        respx.get(f"{BASE}/status").mock(return_value=Response(503, text="unavailable"))
        with pytest.raises(SidecarError, match="503"):
            await client.status()

    @respx.mock
    async def test_community_ids_defaults_to_empty_list(self, client):
        respx.get(f"{BASE}/status").mock(
            return_value=Response(200, json={"whatsapp_ready": True})
        )
        status = await client.status()
        assert status.community_ids == []

    @respx.mock
    async def test_pipeline_url_defaults_to_empty_string(self, client):
        respx.get(f"{BASE}/status").mock(
            return_value=Response(200, json={"whatsapp_ready": True})
        )
        status = await client.status()
        assert status.pipeline_url == ""


# ---------------------------------------------------------------------------
# is_ready()
# ---------------------------------------------------------------------------

class TestIsReady:
    @respx.mock
    async def test_returns_true_when_health_says_ready(self, client):
        respx.get(f"{BASE}/health").mock(
            return_value=Response(200, json={"whatsapp_ready": True})
        )
        assert await client.is_ready() is True

    @respx.mock
    async def test_returns_false_when_health_says_not_ready(self, client):
        respx.get(f"{BASE}/health").mock(
            return_value=Response(200, json={"whatsapp_ready": False})
        )
        assert await client.is_ready() is False

    @respx.mock
    async def test_returns_false_on_network_failure(self, client):
        respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("unreachable"))
        assert await client.is_ready() is False

    @respx.mock
    async def test_returns_false_on_500(self, client):
        respx.get(f"{BASE}/health").mock(return_value=Response(500, text="crash"))
        assert await client.is_ready() is False

    @respx.mock
    async def test_returns_false_when_missing_ready_field(self, client):
        respx.get(f"{BASE}/health").mock(
            return_value=Response(200, json={"status": "ok"})
        )
        assert await client.is_ready() is False


# ---------------------------------------------------------------------------
# SidecarStatus dataclass
# ---------------------------------------------------------------------------

class TestSidecarStatusDataclass:
    def test_is_frozen(self):
        s = SidecarStatus(whatsapp_ready=True, community_ids=[], pipeline_url="http://x")
        with pytest.raises((AttributeError, TypeError)):
            s.whatsapp_ready = False

    def test_equality(self):
        s1 = SidecarStatus(True, ["a@g.us"], "http://x")
        s2 = SidecarStatus(True, ["a@g.us"], "http://x")
        assert s1 == s2

    def test_inequality(self):
        s1 = SidecarStatus(True, ["a@g.us"], "http://x")
        s2 = SidecarStatus(False, ["a@g.us"], "http://x")
        assert s1 != s2


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_sidecar_not_ready_is_sidecar_error(self):
        exc = SidecarNotReadyError("not ready")
        assert isinstance(exc, SidecarError)

    def test_sidecar_error_message_preserved(self):
        exc = SidecarError("connection refused")
        assert "connection refused" in str(exc)

    def test_sidecar_not_ready_message_preserved(self):
        exc = SidecarNotReadyError("scan QR code")
        assert "scan QR code" in str(exc)
