"""Watchdog / retry behaviour of VLMRecognizer against a real local socket server.

These reproduce the failure we hit in production: an endpoint that answers
``200 OK`` headers immediately and then never sends a body (sometimes with
``: ping`` keep-alive comments, which defeat httpx's own read timeout).
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time

import pytest
from PIL import Image

from n2lh.recognition.vlm import VLMError, VLMRecognizer

HEAD = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n")


def chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def sse(obj: str) -> bytes:
    return chunk(f"data: {obj}\n\n".encode())


class FakeSSEServer:
    """Serves one scripted behaviour per accepted connection."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.connections = 0
        self.requests = []              # parsed JSON body of each request
        self.disconnected = []          # one Event per connection
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.port = self._srv.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._stop = True
        try:
            self._srv.close()
        except OSError:
            pass

    def _accept(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            behaviour = self.behaviours.pop(0) if self.behaviours else "ok"
            gone = threading.Event()
            self.connections += 1
            self.disconnected.append(gone)
            threading.Thread(target=self._handle, args=(conn, behaviour, gone),
                             daemon=True).start()

    def _handle(self, conn, behaviour, gone) -> None:
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                part = conn.recv(65536)
                if not part:
                    gone.set()
                    return
                data += part
            head, _, rest = data.partition(b"\r\n\r\n")
            need = int(re.search(rb"content-length: (\d+)", head, re.I).group(1))
            while len(rest) < need:
                rest += conn.recv(65536)
            self.requests.append(json.loads(rest[:need]))
            if behaviour == "reject_kwargs":
                msg = b'{"error":{"message":"Unrecognized request argument: chat_template_kwargs"}}'
                conn.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n"
                             b"Content-Length: " + str(len(msg)).encode() + b"\r\nConnection: close\r\n\r\n" + msg)
                return
            conn.sendall(HEAD)
            if behaviour == "ok":
                conn.sendall(sse('{"choices":[{"delta":{"content":"hello"}}]}'))
                conn.sendall(sse('{"choices":[{"delta":{"content":" world"}}]}'))
                conn.sendall(sse('{"choices":[{"delta":{},"finish_reason":"stop"}]}'))
                conn.sendall(chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")
                conn.recv(1)
            elif behaviour.startswith("say:"):
                conn.sendall(sse(json.dumps({"choices": [{"delta": {"content": behaviour[4:]}}]})))
                conn.sendall(sse('{"choices":[{"delta":{},"finish_reason":"stop"}]}'))
                conn.sendall(chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")
                conn.recv(1)
            elif behaviour == "reasoning_only":
                conn.sendall(sse('{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}'))
                conn.sendall(sse('{"choices":[{"delta":{},"finish_reason":"stop"}]}'))
                conn.sendall(chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")
                conn.recv(1)
            elif behaviour == "quad_loop":
                # The real failure: some page text, then `\quad \quad ...` forever.
                conn.sendall(sse(json.dumps({"choices": [{"delta": {
                    "content": "Def. Some real page text before the loop starts. $x^2 + y^2$ "}}]})))
                conn.settimeout(0.02)
                while True:
                    try:
                        conn.sendall(sse(json.dumps({"choices": [{"delta": {
                            "content": " \\quad" * 12}}]})))
                    except OSError:
                        break
                    try:
                        if conn.recv(1) == b"":
                            break
                    except socket.timeout:
                        continue
            else:
                # silent | pings | endless: keep going until the client hangs up.
                conn.settimeout(0.1)
                while True:
                    if behaviour == "pings":
                        conn.sendall(chunk(b": ping\n\n"))
                    elif behaviour == "endless":
                        conn.sendall(sse('{"choices":[{"delta":{}}]}'))
                    try:
                        if conn.recv(1) == b"":
                            break
                    except socket.timeout:
                        continue
        except OSError:
            pass
        finally:
            gone.set()
            try:
                conn.close()
            except OSError:
                pass


@pytest.fixture()
def page(tmp_path):
    p = tmp_path / "p.png"
    Image.new("L", (32, 32), 255).save(p, "PNG")
    return p


@pytest.fixture()
def serve():
    servers = []

    def make(*behaviours):
        s = FakeSSEServer(behaviours)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def rec_for(server, **kw):
    defaults = dict(timeout=30, stall_timeout=1, retries=0, retry_backoff=0)
    defaults.update(kw)
    return VLMRecognizer(server.url, "m", "key", **defaults)


def test_healthy_stream_returns_content(serve, page):
    srv = serve("ok")
    assert rec_for(srv)._chat("sys", "user", page) == "hello world"


def test_silent_server_is_aborted_and_server_sees_the_disconnect(serve, page):
    srv = serve("silent")
    t0 = time.time()
    with pytest.raises(VLMError) as exc:
        rec_for(srv)._chat("sys", "user", page)
    assert time.time() - t0 < 6
    assert "no model output" in str(exc.value)
    # The socket must really be closed, so the server can free its slot.
    assert srv.disconnected[0].wait(3), "server never saw the client disconnect"


def test_keepalive_pings_do_not_defeat_the_watchdog(serve, page):
    srv = serve("pings")
    t0 = time.time()
    with pytest.raises(VLMError) as exc:
        rec_for(srv)._chat("sys", "user", page)
    assert time.time() - t0 < 6
    msg = str(exc.value)
    assert "no model output" in msg and "keep-alive pings" in msg
    assert not msg.split("(")[1].startswith("0 data events, 0 keep-alive")  # pings were seen


def test_total_timeout_caps_a_stream_that_never_finishes(serve, page):
    srv = serve("endless")
    t0 = time.time()
    with pytest.raises(VLMError) as exc:
        rec_for(srv, timeout=1.5, stall_timeout=30)._chat("sys", "user", page)
    assert time.time() - t0 < 6
    assert "exceeded" in str(exc.value)


def test_stalled_request_is_retried_on_a_fresh_connection(serve, page):
    srv = serve("silent", "ok")
    out = rec_for(srv, retries=1)._chat("sys", "user", page)
    assert out == "hello world"
    assert srv.connections == 2
    assert srv.disconnected[0].wait(3)        # the hung one was really dropped


def test_retries_exhausted_reports_attempt_count(serve, page):
    srv = serve("silent", "silent")
    with pytest.raises(VLMError) as exc:
        rec_for(srv, retries=1)._chat("sys", "user", page)
    assert "gave up after 2 attempt(s)" in str(exc.value)
    assert srv.connections == 2


def test_cancel_event_aborts_an_in_flight_request(serve, page):
    srv = serve("silent")
    rec = rec_for(srv, stall_timeout=60)
    rec.cancel_event = threading.Event()
    threading.Timer(0.5, rec.cancel_event.set).start()
    t0 = time.time()
    with pytest.raises(VLMError) as exc:
        rec._chat("sys", "user", page)
    assert time.time() - t0 < 4
    assert "cancelled" in str(exc.value)
    assert srv.disconnected[0].wait(3)


def test_reasoning_only_reply_is_an_error_not_page_latex(serve, page):
    """A reply that never produced answer content must not be spliced into the
    document as if the model's chain-of-thought were LaTeX."""
    srv = serve("reasoning_only")
    with pytest.raises(VLMError) as exc:
        rec_for(srv)._chat("sys", "user", page)
    assert "no answer content" in str(exc.value)


# ------------------------------------------------------------ thinking switch
def test_thinking_is_off_by_default_and_the_switch_is_sent(serve, page):
    srv = serve("ok")
    assert rec_for(srv)._chat("sys", "user", page) == "hello world"
    assert srv.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_on_never_sends_the_switch(serve, page):
    srv = serve("ok")
    rec_for(srv, thinking="on")._chat("sys", "user", page)
    assert "chat_template_kwargs" not in srv.requests[0]


def test_provider_that_rejects_the_switch_gets_the_request_without_it(serve, page):
    srv = serve("reject_kwargs", "ok")
    out = rec_for(srv)._chat("sys", "user", page)         # retries=0: must not need one
    assert out == "hello world"
    assert "chat_template_kwargs" in srv.requests[0]
    assert "chat_template_kwargs" not in srv.requests[1]


def test_auto_mode_retries_a_stalled_attempt_with_thinking_off(serve, page):
    srv = serve("silent", "ok")
    out = rec_for(srv, thinking="auto", retries=1)._chat("sys", "user", page)
    assert out == "hello world"
    assert "chat_template_kwargs" not in srv.requests[0]                  # provider default first
    assert srv.requests[1]["chat_template_kwargs"] == {"enable_thinking": False}

# --------------------------------------------------------------- runaway loops
from n2lh.recognition.vlm import _RepetitionGuard  # noqa: E402

# A real (abridged) transcript of a dense notes page, incl. the repetitive-looking
# matrix / fraction / partial-derivative markup that must NOT be flagged.
REAL_PAGE = r"""Def. Identity orientations $\mathcal{A}, \mathcal{B}$ if $\mathcal{A} \cup \mathcal{B}$ is an orientation. Then any orientable manifold has 2 orientations.

Def. A nowhere vanishing $n$-form is called a volume form.

Remark. $\omega, \eta$ volume forms, then $\omega = f\eta$, $f > 0$ if same orientation, $f < 0$ if diff orientations.

Eg. $\omega = dx_1 \wedge \dots \wedge dx_n$, then $(\frac{\partial}{\partial x_1}, \dots, \frac{\partial}{\partial x_n})$ ordered basis.

Def. Hausdorff & 2nd countable T.S. $M$ is an $n$-dim mfd w/ boundary if $\forall p \in M$, $\exists p$ nbhd $U$ homeo $\mathbb{R}^n$ or $\mathbb{R}^n_+ = \{(x_1, \dots, x_n) \mid x_n \ge 0\}$. $\partial M := \{p \in M \mid \exists p$ nbhd $U$ homeo $\mathbb{R}^n_+\}$.

Pf. Sps $p \in \partial M \cap U_\alpha \cap U_\beta$. On $M$, $\mathbb{R}^n_+ \cong V_\alpha \xrightarrow{\varphi_\alpha} V_\beta \cong \mathbb{R}^n_+$, $(x_1, \dots, x_n) \mapsto (x_1, \dots, x_n)$.
$$
d\varphi_\alpha = \begin{pmatrix}
\frac{\partial x_\beta}{\partial x_\alpha} & * \\
\frac{\partial x_n}{\partial x_\alpha} & \frac{\partial x_n}{\partial x_n}
\end{pmatrix}
$$
$$
\begin{array}{c} \uparrow \\ 0 \end{array} \quad \begin{array}{c} \uparrow \\ >0 \end{array}
$$
since $\varphi_n(x_1, \dots, x_n) > 0$ if $x_n > 0$.

Sps $dx_1 \wedge \dots \wedge dx_n$ orientation of $U \subset M$, then $(\frac{\partial}{\partial x_1}, \dots, \frac{\partial}{\partial x_n})$ ordered basis. The induced orientation on $\partial M \cap U$ is $(-1)^n dx_1 \wedge \dots \wedge dx_{n-1}$. Ordered basis:
$$
\left\{ \left(\frac{\partial}{\partial x_1}, \dots, \frac{\partial}{\partial x_{n-1}}\right), \text{ } 2 \mid n \right.
$$
$$
\left. \left(\frac{\partial}{\partial x_2}, \frac{\partial}{\partial x_1}, \dots, \frac{\partial}{\partial x_{n-1}}\right), \text{ } 2 \nmid n \right.
$$

Eg. $\partial \mathbb{R}^2_+ = \mathbb{R}$, $\partial \mathbb{R}^3_+ = -\mathbb{R}^2$.
"""


def feed_in_chunks(guard, text, size=7):
    for i in range(0, len(text), size):
        why = guard.feed(text[i:i + size])
        if why:
            return i + size, why
    return None, None


def test_guard_does_not_flag_a_real_dense_transcript():
    at, why = feed_in_chunks(_RepetitionGuard(24000), REAL_PAGE)
    assert why is None, f"false positive at char {at}: {why}"
    # also with the page tripled (a long, repetitive-looking document)
    at, why = feed_in_chunks(_RepetitionGuard(24000), REAL_PAGE * 3)
    assert why is None, f"false positive at char {at}: {why}"


def test_guard_trips_within_a_couple_thousand_chars_of_a_quad_loop():
    guard = _RepetitionGuard(24000)
    at, why = feed_in_chunks(guard, REAL_PAGE[:1750] + " \\quad" * 3000)
    assert why and "repeating itself" in why
    assert at < 1750 + 2500, f"took {at - 1750} loop chars to notice"


def test_guard_length_ceiling_is_a_backstop_for_varied_output():
    import random
    rnd = random.Random(1)
    varied = "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789 \n$\\{}") for _ in range(3000))
    at, why = feed_in_chunks(_RepetitionGuard(1000), varied)
    assert why and "exceeded 1000 characters" in why


def test_runaway_loop_is_aborted_fast_and_retried_with_a_higher_temperature(serve, page):
    srv = serve("quad_loop", "ok")
    t0 = time.time()
    out = rec_for(srv, retries=1, stall_timeout=30, timeout=60)._chat("sys", "user", page)
    assert out == "hello world"
    assert time.time() - t0 < 10
    assert srv.connections == 2
    assert srv.disconnected[0].wait(3), "loop connection was not dropped"
    assert srv.requests[1]["temperature"] > srv.requests[0]["temperature"]


def test_runaway_head_is_not_salvaged_as_the_answer(serve, page):
    """The loop starts mid-page in practice; a truncated head would silently
    drop the rest of the page, so the request must fail instead."""
    srv = serve("quad_loop")
    with pytest.raises(VLMError) as exc:
        rec_for(srv, stall_timeout=30, timeout=60)._chat("sys", "user", page)
    assert "repeating itself" in str(exc.value)

# ------------------------------------------------------------- figure locating
def test_locate_figures_sends_the_locate_prompt_and_parses_the_boxes(serve, page):
    from n2lh.recognition.base import PageImage
    from n2lh.recognition.prompts import LOCATE_SYSTEM
    srv = serve('say:```json\n[{"bbox": [200, 400, 700, 600]}, {"bbox": [10, 20, 30, 40]}]\n```')
    boxes = rec_for(srv).locate_figures(PageImage(index=1, path=page))
    assert boxes == [(200, 400, 700, 600), (10, 20, 30, 40)]
    sent = srv.requests[0]["messages"]
    assert sent[0]["content"] == LOCATE_SYSTEM
    assert sent[1]["content"][1]["type"] == "image_url"


def test_locate_figures_returns_none_when_the_page_has_no_figure(serve, page):
    from n2lh.recognition.base import PageImage
    srv = serve("say:[]")
    assert rec_for(srv).locate_figures(PageImage(index=1, path=page)) is None