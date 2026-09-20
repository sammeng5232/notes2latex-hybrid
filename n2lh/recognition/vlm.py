"""OpenAI-compatible VLM recognizer.

One client covers cloud providers (OpenRouter, OpenAI, ...) AND fully local
endpoints (Ollama at http://localhost:11434/v1, vLLM, LM Studio at
http://localhost:1234/v1) -- the BYOK / model-agnostic design of notes2latex,
extended so "local" can mean zero cloud exposure.

Reliability model (learned the hard way against a shared campus endpoint that
returns ``200 OK`` headers immediately and then sometimes never sends a body):

* Every request streams, and a per-request *watchdog thread* watches for actual
  model output (``data:`` events -- keep-alive ``: ping`` comments do not count).
  If nothing arrives for ``stall_timeout`` seconds, or the request runs longer
  than ``timeout`` seconds in total, the watchdog shuts the socket down. That
  unblocks the reader at once *and* lets the server see the disconnect so it can
  drop the request instead of holding a slot for a dead client. httpx's own
  read timeout cannot do this: it measures time since the last byte, which
  keep-alive pings reset forever.
* A stalled / aborted / transport-failed request is retried on a fresh
  connection (``retries`` times) before the error reaches the pipeline, so a
  flaky endpoint costs a couple of minutes, not a failed page.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import socket
import threading
import time
import zlib
from typing import List, Optional, Union

import httpx
from PIL import Image

from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult
from n2lh.recognition.prompts import (
    FIX_SYSTEM,
    LOCATE_SYSTEM,
    LOCATE_USER,
    TRANSCRIBE_SYSTEM,
    fix_user_prompt,
    transcribe_user_prompt,
)

log = logging.getLogger("n2lh.vlm")

_FENCE = re.compile(r"```(?:latex|tex)?\s*\n(.*?)```", re.DOTALL)
_MAX_IMAGE_EDGE = 1600
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class VLMError(RuntimeError):
    """A request failed for good (bad key/model/request, cancelled, retries spent)."""


class VLMTransientError(VLMError):
    """A request failed in a way worth retrying on a fresh connection."""

    # True when the watchdog aborted it (stall / total cap) rather than the
    # transport failing -- the signature of a reasoning loop.
    aborted: bool = False
    # True when the repetition guard ended a runaway loop.
    runaway: bool = False


class _RepetitionGuard:
    """Spots a runaway repetition loop in streamed model text within seconds.

    Observed on the CUHK endpoint: after writing part of a page the model can
    fall into ``\\quad \\quad \\quad ...`` and never stop (20-25k characters in 50s,
    on ~35-50% of identical requests for some pages, at any temperature, with
    reasoning on or off). Real transcripts are ~1-5k characters and compress to
    0.33-0.42 with zlib; the loops compress to ~0.03. Checking the ratio of the
    last ``WINDOW`` characters every ``STEP`` new ones ends a loop about a
    thousand characters (~5s) in instead of waiting out the 10-minute cap.
    A plain length ceiling is the backstop for degenerate-but-incompressible
    output.
    """

    WINDOW = 2000
    STEP = 500
    THRESHOLD = 0.10

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.total = 0
        self._since = 0
        self._tail = ""

    def feed(self, piece: str) -> Optional[str]:
        """Return a reason string when the stream has gone degenerate."""
        self.total += len(piece)
        self._since += len(piece)
        self._tail = (self._tail + piece)[-2 * self.WINDOW:]
        if self.total > self.max_chars:
            return f"output exceeded {self.max_chars} characters without finishing"
        if self._since >= self.STEP and len(self._tail) >= self.WINDOW:
            self._since = 0
            window = self._tail[-self.WINDOW:].encode("utf-8")
            ratio = len(zlib.compress(window, 6)) / len(window)
            if ratio < self.THRESHOLD:
                return (f"model is stuck repeating itself (compression ratio {ratio:.2f} "
                        f"over the last {self.WINDOW} characters, {self.total} characters so far)")
        return None


class _Runaway(Exception):
    """Internal: the repetition guard tripped; carries the reason."""


class _DropParam(Exception):
    """Internal: the endpoint rejected an optional request field; resend without it."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


# Optional request fields some providers reject; on a 400 that names one of
# these, it is dropped and the request resent (not counted as a retry).
_DROPPABLE = ("temperature", "chat_template_kwargs")

# vLLM/Qwen-style switch that turns the model's reasoning phase off.
_NO_THINKING = {"enable_thinking": False}


class _StreamWatchdog:
    """Aborts a streaming response that has stopped making progress."""

    def __init__(self, resp, *, stall: float, total: float,
                 cancel: Optional[threading.Event] = None, poll: float = 0.25) -> None:
        self._resp = resp
        self._stall = stall
        self._total = total
        self._cancel = cancel
        self._poll = poll
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._last_data = self._started
        self.n_data = 0
        self.n_ping = 0
        self.reason: Optional[str] = None
        self.cancelled = False
        self._thread = threading.Thread(target=self._run, name="vlm-watchdog", daemon=True)

    def start(self) -> "_StreamWatchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def progress(self) -> None:
        self._last_data = time.monotonic()
        self.n_data += 1

    def ping(self) -> None:
        self.n_ping += 1

    @property
    def fired(self) -> bool:
        return self.reason is not None

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            now = time.monotonic()
            if self._cancel is not None and self._cancel.is_set():
                self.cancelled = True
                self.reason = "cancelled"
            elif now - self._started > self._total:
                self.reason = (f"request exceeded {self._total:.0f}s without finishing "
                               f"({self.n_data} data events, {self.n_ping} keep-alive pings)")
            elif now - self._last_data > self._stall:
                self.reason = (f"no model output for {self._stall:.0f}s "
                               f"({self.n_data} data events, {self.n_ping} keep-alive pings "
                               "received) -- the server kept the connection open but "
                               "sent nothing")
            else:
                continue
            self._abort()
            return

    def _abort(self) -> None:
        # Shut the raw socket down: this wakes a reader blocked in recv() on any
        # platform and sends FIN so the server sees the client is gone.
        try:
            sock = self._resp.extensions["network_stream"].get_extra_info("socket")
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        except Exception:  # noqa: BLE001 - fall through to close()
            pass
        try:
            self._resp.close()
        except Exception:  # noqa: BLE001
            pass


class VLMRecognizer(Recognizer):
    name = "vlm"

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: int = 600, max_image_edge: int = _MAX_IMAGE_EDGE,
                 use_proxy: bool = False, stall_timeout: float = 120,
                 retries: int = 2, retry_backoff: float = 2.0,
                 thinking: str = "off", max_output_chars: int = 24000) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout                # hard cap on ONE request, seconds
        self.stall_timeout = stall_timeout    # max seconds without a model-output event
        self.retries = max(0, retries)        # extra attempts after a transient failure
        self.retry_backoff = retry_backoff
        # Reasoning ("thinking") models can spend minutes -- or never stop, on
        # some pages -- reasoning before writing a page's LaTeX, and for
        # transcription it bought no accuracy in testing (it even misread
        # \varphi/\psi as \psi/\chi where the direct answer got them right).
        #   "off"  : ask the model not to think (chat_template_kwargs.enable_thinking=false)
        #   "auto" : provider default first; a stalled/looping attempt is retried with it off
        #   "on"   : never send the switch (provider default)
        self.thinking = thinking if thinking in ("off", "auto", "on") else "off"
        # Ceiling on one page's answer text (real ones are ~1-5k chars).
        self.max_output_chars = max_output_chars
        self.max_image_edge = max_image_edge
        # A system-wide HTTPS_PROXY (e.g. Clash Verge) makes httpx tunnel the
        # request; campus/proxied endpoints often drop that TLS handshake with
        # UNEXPECTED_EOF_WHILE_READING. Direct-first, proxy as explicit opt-in.
        self.use_proxy = use_proxy
        # Set by the job runner so "Cancel" aborts an in-flight request.
        self.cancel_event: Optional[threading.Event] = None
        self._client: Optional[httpx.Client] = None
        self._client_lock = threading.Lock()

    @property
    def hard_budget(self) -> float:
        """Upper bound (seconds) on one transcribe() call including retries;
        the pipeline's last-resort deadline must not undercut it."""
        return (self.retries + 1) * (self.timeout + self.retry_backoff * (self.retries + 1)) + 60

    # ------------------------------------------------------------------ public
    def transcribe(self, page: PageImage, context_tail: str,
                   open_environments: List[str],
                   guidance: Optional[str] = None) -> TranscribeResult:
        if guidance:
            system = FIX_SYSTEM
            user = guidance
        else:
            system = TRANSCRIBE_SYSTEM
            user = transcribe_user_prompt(context_tail, open_environments)
        content = self._chat(system, user, page.path)
        latex = _extract_latex(content)
        return TranscribeResult(latex=latex, engine=self.name,
                                notes=None if not guidance else "repair pass")

    def locate_figures(self, page: PageImage) -> Optional[List[tuple]]:
        """One dedicated request for accurate figure boxes (0..1000, top-left origin).

        The inline ``\\figbox`` boxes the model writes while transcribing are coarse
        (round numbers, often a label short); asked on its own, the same model
        returns tight boxes. Returns None when the reply has no parsable boxes.
        """
        from n2lh.pipeline.figures import parse_boxes
        text = self._chat(LOCATE_SYSTEM, LOCATE_USER, page.path)
        boxes = parse_boxes(text)
        return [(b.x0, b.y0, b.x1, b.y1) for b in boxes] if boxes else None

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _client_for(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                # trust_env=False ignores HTTP(S)_PROXY/NO_PROXY so the endpoint is
                # reached directly unless the caller explicitly opts into the proxy.
                # (connect, read, write, pool). The watchdog is the real stall
                # detector; httpx's read timeout is only a backstop. Write must be
                # generous: the request body is a ~0.5-1 MB base64 page image.
                timeout = (30.0, max(float(self.stall_timeout), 30.0) + 30.0, 60.0, 30.0)
                self._client = httpx.Client(
                    timeout=timeout,
                    trust_env=self.use_proxy,
                    proxy=(os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
                           or None) if self.use_proxy else None,
                    http2=False)  # stream reading is simpler with HTTP/1.1
            return self._client

    # ----------------------------------------------------------------- private
    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise VLMError("cancelled")

    def _chat(self, system: str, user_text: str, image_path) -> str:
        client = self._client_for()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": _data_uri(image_path, self.max_image_edge)}},
            ]},
        ]
        payload = {"model": self.model, "temperature": 0.1, "messages": messages, "stream": True}
        headers = {"Authorization": f"Bearer {self.api_key or 'none'}"}
        url = f"{self.base_url}/chat/completions"

        if self.thinking == "off":
            payload["chat_template_kwargs"] = dict(_NO_THINKING)

        attempt = 0
        while True:
            attempt += 1
            self._check_cancel()
            try:
                return self._stream_once(client, url, payload, headers)
            except _DropParam as drop:
                # The endpoint rejected an optional field (e.g. reasoning models
                # reject temperature; non-vLLM providers reject chat_template_kwargs):
                # resend without it. Not a retry -- a field can only be dropped once.
                payload.pop(drop.name, None)
                attempt -= 1
            except VLMTransientError as exc:
                if attempt > self.retries:
                    raise VLMError(f"{exc} [gave up after {attempt} attempt(s)]") from exc
                if getattr(exc, "runaway", False) and "temperature" in payload:
                    # A different sample is the only thing that ever ended a loop
                    # (retries succeed ~half the time); nudge the temperature so
                    # the retry isn't a near-greedy replay of the same path.
                    payload["temperature"] = min(1.0, payload["temperature"] + 0.25)
                if (self.thinking == "auto" and getattr(exc, "aborted", False)
                        and "chat_template_kwargs" not in payload):
                    # A stalled or endless attempt is usually a reasoning loop:
                    # don't ask the same question the same way again.
                    payload["chat_template_kwargs"] = dict(_NO_THINKING)
                    log.warning("VLM attempt %d stalled/looped; retrying with thinking off", attempt)
                log.warning("VLM attempt %d/%d failed: %s -- retrying on a fresh connection",
                            attempt, self.retries + 1, exc)
                self._sleep(self.retry_backoff * attempt)

    def _sleep(self, seconds: float) -> None:
        # Sleep in slices so Cancel is honoured promptly.
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check_cancel()
            time.sleep(min(0.25, max(0.0, end - time.monotonic())))

    def _stream_once(self, client: httpx.Client, url: str, payload: dict,
                     headers: dict) -> str:
        started = time.monotonic()
        content: List[str] = []
        reasoning: List[str] = []
        saw_done = False
        finish: Optional[str] = None
        wd: Optional[_StreamWatchdog] = None
        content_guard = _RepetitionGuard(self.max_output_chars)
        reasoning_guard = _RepetitionGuard(self.max_output_chars * 5)
        try:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                wd = _StreamWatchdog(resp, stall=self.stall_timeout, total=self.timeout,
                                     cancel=self.cancel_event).start()
                if resp.status_code != 200:
                    body = resp.read()
                    text = _error_text_bytes(body)
                    if resp.status_code == 400:
                        for name in _DROPPABLE:
                            if name in payload and (
                                    name in text.lower()
                                    or (name == "chat_template_kwargs"
                                        and "enable_thinking" in text.lower())):
                                raise _DropParam(name)
                    msg = f"VLM endpoint returned {resp.status_code}: {text[:300]}"
                    if resp.status_code in _RETRYABLE_STATUS:
                        raise VLMTransientError(msg)
                    raise VLMError(msg)

                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        body = line[5:].strip()
                        if body == "[DONE]":
                            saw_done = True
                            break
                        wd.progress()
                        try:
                            choice = json.loads(body)["choices"][0]
                        except (KeyError, IndexError, TypeError, ValueError):
                            continue
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
                            why = reasoning_guard.feed(delta["reasoning_content"])
                            if why:
                                raise _Runaway("reasoning: " + why)
                        if delta.get("content"):
                            content.append(delta["content"])
                            why = content_guard.feed(delta["content"])
                            if why:
                                raise _Runaway(why)
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                    elif line.startswith(":"):
                        wd.ping()
        except (VLMError, _DropParam):
            raise
        except _Runaway as loop:
            # Leaving the `with` closes the response, so the server sees the
            # disconnect and stops generating. The text produced so far is NOT
            # salvaged: on real samples the loop began mid-page (~65% through),
            # so a truncated head would silently drop the rest of the page.
            err = VLMTransientError(str(loop))
            err.aborted = True
            err.runaway = True
            raise err from loop
        except Exception as exc:  # noqa: BLE001 - classify below
            if wd is not None and wd.cancelled:
                raise VLMError("cancelled") from exc
            if wd is not None and wd.fired:
                err = VLMTransientError(wd.reason)
                err.aborted = True      # a stall/total-cap abort, not a transport error
                raise err from exc
            if isinstance(exc, httpx.HTTPError):
                hint = ""
                if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                    hint = (" If a system proxy is set, it may be breaking TLS to this "
                            "host; the recognizer connects directly by default.")
                    raise VLMTransientError(
                        f"VLM endpoint unreachable at {url} ({type(exc).__name__}: {exc}).{hint}"
                    ) from exc
                raise VLMTransientError(
                    f"VLM request to {url} failed ({type(exc).__name__}: {exc})") from exc
            raise
        finally:
            if wd is not None:
                wd.stop()

        if wd is not None and wd.cancelled:
            raise VLMError("cancelled")
        if wd is not None and wd.fired:
            # Watchdog fired but the reader ended quietly (e.g. EOF after shutdown).
            err = VLMTransientError(wd.reason)
            err.aborted = True
            raise err

        text = "".join(content).strip()
        if not text:
            thinking = "".join(reasoning).strip()
            if thinking and _FENCE.search(thinking):
                # Provider quirk: the answer was routed into the reasoning channel.
                text = thinking
            else:
                raise VLMTransientError(
                    "model returned no answer content "
                    f"({len(thinking)} chars of reasoning only, finish_reason={finish!r})")
        elif not saw_done and finish is None:
            raise VLMTransientError("response stream ended before the model finished")
        if finish == "length":
            log.warning("VLM response hit the length limit; page LaTeX may be truncated")
        log.info("VLM ok in %.1fs (%d chars)", time.monotonic() - started, len(text))
        return text


def probe_vlm(base_url: str, model: str, api_key: str = "",
              timeout: int = 30, use_proxy: bool = False) -> str:
    """Cheap reachability/auth check: ask the endpoint to list its models.

    Returns the raw response text on success; raises VLMError otherwise. Used by
    /api/preflight so a broken proxy/model is reported before a job is queued.
    """
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key or 'none'}"}
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
             or None) if use_proxy else None
    try:
        client = httpx.Client(timeout=timeout, trust_env=use_proxy, proxy=proxy)
    except TypeError:  # httpx < 0.26 has no per-client proxy kwarg
        client = httpx.Client(timeout=timeout, trust_env=use_proxy)
    with client:
        try:
            resp = client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise VLMError(
                f"VLM endpoint unreachable at {url} ({type(exc).__name__}: {exc}). "
                "If a system proxy is set, it may be breaking TLS to this host; "
                "the recognizer connects directly by default.") from exc
        if resp.status_code != 200:
            raise VLMError(f"VLM endpoint returned {resp.status_code}: {_error_text(resp)[:300]}")
        return _error_text(resp)


def _error_text(resp: httpx.Response) -> str:
    """Best-effort human-readable error body, even for non-JSON responses."""
    try:
        return resp.text
    except Exception:  # noqa: BLE001 - never mask the real failure
        return "<unreadable response body>"


def _error_text_bytes(body: Union[str, bytes]) -> str:
    """Best-effort human-readable error from bytes or string."""
    try:
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="replace")
        return body
    except Exception:  # noqa: BLE001 - never mask the real failure
        return "<unreadable response body>"


def _data_uri(image_path, max_edge: int) -> str:
    with Image.open(image_path) as pil:
        pil = pil.convert("RGB")
        if max(pil.size) > max_edge:
            scale = max_edge / max(pil.size)
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                             Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_latex(content: str) -> str:
    if not content:
        return ""
    m = _FENCE.search(content)
    if m:
        return m.group(1).strip()
    # Unfenced reply: strip a leading "Here is..." line if present.
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    return stripped
