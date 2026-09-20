"""VLM client behavior: proxy handling, error reporting, preflight probe.

These tests never hit the network; they assert the client's *configuration and
error surfaces*, which is where real failures were observed:

- a system-wide HTTPS_PROXY (Clash Verge) broke TLS to the campus endpoint with
  UNEXPECTED_EOF_WHILE_READING, so the client must default to direct connections;
- a non-200/non-JSON reply must not be misreported as "unreachable".
"""

from __future__ import annotations

import io

import httpx
import pytest
from PIL import Image

from n2lh.recognition import vlm as vlm_mod
from n2lh.recognition.vlm import VLMError, VLMRecognizer, probe_vlm


@pytest.fixture()
def page_image(tmp_path):
    path = tmp_path / "page.png"
    Image.new("L", (64, 64), 255).save(path, "PNG")
    return path


def test_client_ignores_system_proxy_by_default(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    rec = VLMRecognizer("https://example.invalid/api", "m")
    client = rec._client_for()
    # trust_env must be off so httpx does not pick up the proxy env vars.
    assert client._trust_env is False
    rec.close()


def test_client_opt_in_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    rec = VLMRecognizer("https://example.invalid/api", "m", use_proxy=True)
    client = rec._client_for()
    assert client._trust_env is True
    rec.close()


def test_non_json_error_body_is_surfaced(monkeypatch, page_image):
    """A 500 with an HTML/text body must report the status, not 'unreachable'."""

    class FakeResponse:
        status_code = 500
        text = "<html>gateway error</html>"

        def read(self):
            return self.text.encode()

        def json(self):
            raise ValueError("not json")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeClient:
        def stream(self, *a, **k):
            return FakeResponse()

    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    monkeypatch.setattr(rec, "_client_for", lambda: FakeClient())
    with pytest.raises(VLMError) as exc:
        rec._chat("sys", "user", page_image)
    msg = str(exc.value)
    assert "500" in msg and "gateway error" in msg
    assert "unreachable" not in msg


def test_transport_error_mentions_proxy(monkeypatch, page_image):
    class FakeClient:
        def stream(self, *a, **k):
            raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]")

    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    monkeypatch.setattr(rec, "_client_for", lambda: FakeClient())
    with pytest.raises(VLMError) as exc:
        rec._chat("sys", "user", page_image)
    assert "proxy" in str(exc.value).lower()


def test_probe_vlm_reports_status(monkeypatch):
    class FakeResponse:
        status_code = 404
        text = '{"detail":"Model not found"}'

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return FakeResponse()

    monkeypatch.setattr(vlm_mod.httpx, "Client", lambda *a, **k: FakeClient())
    with pytest.raises(VLMError) as exc:
        probe_vlm("https://example.invalid/api", "default")
    assert "404" in str(exc.value)
    assert "Model not found" in str(exc.value)