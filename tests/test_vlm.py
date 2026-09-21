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


def test_doc_hint_reaches_the_system_prompt(monkeypatch, page_image):
    """The uploaded file's name must reach the model: a handwritten author name
    is genuinely ambiguous (周民强 was read as 周兆庆 and 周美玲 on different
    passes), and the file name spells it out."""
    from n2lh.recognition.base import PageImage
    from n2lh.recognition.prompts import TRANSCRIBE_SYSTEM

    captured = {}

    def fake_stream(client, url, payload, headers):
        captured.update(payload)
        return "```latex\nx\n```"

    hint = '\n\nSOURCE FILE\n- These notes were uploaded as "notes.pdf".'
    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0, doc_hint=hint)
    monkeypatch.setattr(rec, "_client_for", lambda: object())
    monkeypatch.setattr(rec, "_stream_once", fake_stream)
    rec.transcribe(PageImage(index=1, path=page_image), "", [])

    system = captured["messages"][0]["content"]
    assert system.startswith(TRANSCRIBE_SYSTEM.splitlines()[0])
    assert "SOURCE FILE" in system and "notes.pdf" in system


def test_doc_hint_also_reaches_repair_passes(monkeypatch, page_image):
    from n2lh.recognition.base import PageImage

    captured = {}

    def fake_stream(client, url, payload, headers):
        captured.update(payload)
        return "```latex\nx\n```"

    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0,
                        doc_hint='\n\nSOURCE FILE\n- "notes.pdf"')
    monkeypatch.setattr(rec, "_client_for", lambda: object())
    monkeypatch.setattr(rec, "_stream_once", fake_stream)
    rec.transcribe(PageImage(index=1, path=page_image), "", [],
                   guidance="fix this page")

    assert "notes.pdf" in captured["messages"][0]["content"]


def test_a_colored_page_gets_the_color_directive(monkeypatch, tmp_path):
    """The page's own detected colors must reach the user prompt: a title in
    magenta and green pen came out black before the directive existed."""
    from n2lh.recognition.base import PageImage

    path = tmp_path / "colored.png"
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    px = img.load()
    for x in range(50, 350):           # a magenta title stroke
        for y in range(50, 62):
            px[x, y] = (255, 78, 225)
    for x in range(50, 350):           # a green one beside it
        for y in range(80, 90):
            px[x, y] = (91, 164, 128)
    img.save(path, "PNG")

    captured = {}

    def fake_stream(client, url, payload, headers):
        captured.update(payload)
        return "```latex\nx\n```"

    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    monkeypatch.setattr(rec, "_client_for", lambda: object())
    monkeypatch.setattr(rec, "_stream_once", fake_stream)
    rec.transcribe(PageImage(index=1, path=path), "", [])

    user = captured["messages"][1]["content"][0]["text"]
    assert "COLORED INK IS PRESENT" in user
    assert "pink" in user and "green" in user


def test_a_grayscale_page_gets_no_color_directive(monkeypatch, page_image):
    from n2lh.recognition.base import PageImage

    captured = {}

    def fake_stream(client, url, payload, headers):
        captured.update(payload)
        return "```latex\nx\n```"

    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    monkeypatch.setattr(rec, "_client_for", lambda: object())
    monkeypatch.setattr(rec, "_stream_once", fake_stream)
    rec.transcribe(PageImage(index=1, path=page_image), "", [])

    assert "COLORED INK" not in captured["messages"][1]["content"][0]["text"]

# ----------------------------------------------------- commentary stripping
def test_a_commentary_line_is_not_part_of_the_page():
    """A real run opened the document with 'Here is the transcription of the
    page.' -- model bookends must not reach the compiled PDF."""
    out = vlm_mod._extract_latex(
        "Here is the transcription of the page.\n\n\\textbf{Title}")
    assert out == "\\textbf{Title}"


def test_a_closing_commentary_line_is_dropped_too():
    out = vlm_mod._extract_latex(
        "\\textbf{Title}\n\nLet me know if you want any changes.")
    assert out == "\\textbf{Title}"


def test_a_plain_first_line_of_notes_is_kept():
    """Notes can legitimately start with plain words; only commentary naming
    what it is handing over is dropped."""
    out = vlm_mod._extract_latex("Here is my proof of theorem 3\n\n\\textbf{Q}")
    assert out.startswith("Here is my proof")


def test_chatter_inside_a_fence_is_stripped():
    out = vlm_mod._extract_latex(
        "```latex\nHere is the transcription of the page.\n\\textbf{T}\n```")
    assert out == "\\textbf{T}"
