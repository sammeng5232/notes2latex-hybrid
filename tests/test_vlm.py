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


# ----------------------------------------------------------------- strips
def _sent_image_size(payload):
    import base64
    url = next(part["image_url"]["url"] for part in payload["messages"][1]["content"]
               if part.get("type") == "image_url")
    data = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def _capturing(monkeypatch, rec, answer="```latex\nx\n```"):
    captured = []

    def fake_stream(client, url, payload, headers):
        captured.append(payload)
        return answer

    monkeypatch.setattr(rec, "_client_for", lambda: object())
    monkeypatch.setattr(rec, "_stream_once", fake_stream)
    return captured


def test_a_strip_is_sent_at_its_own_resolution_not_the_page_limit(monkeypatch, tmp_path):
    """The whole point of a strip: the page limit (1600px) made dense handwriting
    illegible. The limit is per call; the client's own setting is shared by
    concurrent calls and must not change."""
    path = tmp_path / "strip.png"
    Image.new("L", (3508, 600), 255).save(path, "PNG")
    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    captured = _capturing(monkeypatch, rec)
    rec.transcribe_strip(path, 2, 6, max_edge=3600)
    assert _sent_image_size(captured[0]) == (3508, 600)
    assert rec.max_image_edge == 1600

    from n2lh.recognition.base import PageImage
    rec.transcribe(PageImage(index=1, path=path), "", [])
    assert max(_sent_image_size(captured[1])) == 1600


def test_a_strip_is_told_what_it_is(monkeypatch, tmp_path):
    path = tmp_path / "strip.png"
    Image.new("L", (800, 100), 255).save(path, "PNG")
    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0,
                        doc_hint='\n\nSOURCE FILE\n- "notes.pdf"')
    captured = _capturing(monkeypatch, rec)
    rec.transcribe_strip(path, 1, 6, max_edge=3600, context_tail="previous page")
    rec.transcribe_strip(path, 5, 6, max_edge=3600, context_tail="previous page")
    first, fifth = captured
    assert "strip 1 of 6" in first["messages"][1]["content"][0]["text"]
    assert "previous page" in first["messages"][1]["content"][0]["text"]
    user5 = fifth["messages"][1]["content"][0]["text"]
    assert "strip 5 of 6" in user5 and "previous page" not in user5
    assert "not the top of the page" in user5
    # the file-name hint only settles the spelling of the title, which is on top
    assert "notes.pdf" in first["messages"][0]["content"]
    assert "notes.pdf" not in fifth["messages"][0]["content"]


def test_a_strip_is_never_given_the_colored_ink_directive(monkeypatch, tmp_path):
    """With it, the real endpoint answered strips with garbage from the first
    token on 13 of 13 requests; without it the same strips read cleanly."""
    path = tmp_path / "strip.png"
    img = Image.new("RGB", (400, 100), (255, 255, 255))
    px = img.load()
    for x in range(50, 350):
        for y in range(40, 52):
            px[x, y] = (91, 164, 128)          # green
    img.save(path, "PNG")
    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    captured = _capturing(monkeypatch, rec)
    rec.transcribe_strip(path, 2, 3, max_edge=3600, page_colors=["green", "pink"])
    assert "COLORED INK" not in captured[0]["messages"][1]["content"][0]["text"]


def test_a_re_read_of_a_strip_is_sampled_differently(monkeypatch, tmp_path):
    path = tmp_path / "strip.png"
    Image.new("L", (400, 100), 255).save(path, "PNG")
    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    captured = _capturing(monkeypatch, rec)
    rec.transcribe_strip(path, 1, 2, max_edge=3600)
    rec.transcribe_strip(path, 1, 2, max_edge=3600, variant=1)
    assert captured[0]["temperature"] == 0.1
    assert captured[1]["temperature"] > 0.1


def test_a_text_only_repair_attaches_no_image(monkeypatch):
    from n2lh.recognition.prompts import FIX_SYSTEM
    rec = VLMRecognizer("https://example.invalid/api", "m", retries=0)
    captured = _capturing(monkeypatch, rec, "```latex\nfixed\n```")
    out = rec.repair_text("fix this")
    assert out.latex == "fixed"
    content = captured[0]["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text"]
    assert captured[0]["messages"][0]["content"].startswith(FIX_SYSTEM.splitlines()[0])


def test_a_retry_is_reported_to_the_caller(monkeypatch, tmp_path):
    from n2lh.recognition.vlm import VLMTransientError
    path = tmp_path / "strip.png"
    Image.new("L", (400, 100), 255).save(path, "PNG")
    rec = VLMRecognizer("https://example.invalid/api", "m", retries=2)
    answers = [VLMTransientError("looped"), "```latex\nok\n```"]

    def fake_stream(client, url, payload, headers):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(rec, "_client_for", lambda: object())
    monkeypatch.setattr(rec, "_stream_once", fake_stream)
    monkeypatch.setattr(rec, "_sleep", lambda seconds: None)
    seen = []
    out = rec.transcribe_strip(path, 1, 2, max_edge=3600,
                               on_retry=lambda attempt, error: seen.append((attempt, error)))
    assert out == "ok" and seen == [(1, "looped")]

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
