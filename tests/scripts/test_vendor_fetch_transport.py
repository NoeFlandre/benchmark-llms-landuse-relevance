"""Transport behaviour of the vendoring script: throttling, retries and error wrapping."""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from landuse_relevance_bench.adapters.translations import TranslationDataError
from scripts import vendor_multilingual_dataset as vendor


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _Transport:
    """A scripted ``urlopen``: each call returns the next payload or raises the next error."""

    def __init__(self, *outcomes: bytes | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.urls: list[str] = []

    def __call__(self, request: Request, timeout: float | None = None) -> _Response:
        self.urls.append(request.full_url)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome)


class _Clock:
    """A monotonic clock that only moves when the code under test sleeps."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(vendor, "time", fake)
    monkeypatch.setitem(vendor._request_state, "last_request_at", 0.0)
    return fake


def _transport(monkeypatch: pytest.MonkeyPatch, *outcomes: bytes | BaseException) -> _Transport:
    transport = _Transport(*outcomes)
    monkeypatch.setattr(vendor, "urlopen", transport)
    return transport


def _http_error(code: int, retry_after: str | None = None) -> HTTPError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return HTTPError("https://example.org/data", code, "status", headers, None)


URL = "https://example.org/data.json"


def test_plain_http_sources_are_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    transport = _transport(monkeypatch)

    with pytest.raises(TranslationDataError, match="only HTTPS sources are allowed"):
        vendor.fetch_bytes("http://example.org/data.json")
    assert transport.urls == []


def test_successful_fetch_returns_the_payload_without_waiting(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    transport = _transport(monkeypatch, b"payload")

    assert vendor.fetch_bytes(URL) == b"payload"
    assert transport.urls == [URL]
    assert clock.sleeps == []
    assert vendor._request_state["last_request_at"] == clock.now


def test_requests_wait_out_the_remainder_of_the_request_delay(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    _transport(monkeypatch, b"payload")
    vendor._request_state["last_request_at"] = clock.now - 0.5

    assert vendor.fetch_bytes(URL) == b"payload"
    assert clock.sleeps == [pytest.approx(vendor.REQUEST_DELAY_SECONDS - 0.5)]


def test_rate_limited_response_is_retried_after_the_advertised_delay(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    transport = _transport(monkeypatch, _http_error(429, retry_after="7"), b"payload")

    assert vendor.fetch_bytes(URL) == b"payload"
    assert len(transport.urls) == 2
    assert clock.sleeps == [7.0]


@pytest.mark.parametrize(
    ("retry_after", "delay"),
    [
        ("120", 30.0),  # capped at 30 seconds
        ("0.5", 2.0),  # raised to the 2 second floor
        ("soon", 2.0),  # not a number: falls back to the first backoff
        (None, 2.0),  # no header: falls back to the first backoff
    ],
)
def test_retry_after_is_bounded_and_invalid_values_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    retry_after: str | None,
    delay: float,
) -> None:
    _transport(monkeypatch, _http_error(429, retry_after=retry_after), b"payload")

    assert vendor.fetch_bytes(URL) == b"payload"
    assert clock.sleeps == [delay]


def test_transient_failures_back_off_exponentially_then_succeed(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    outcomes = [_http_error(503), _http_error(502), _http_error(503), _http_error(504)]
    _transport(monkeypatch, *outcomes, b"payload")

    assert vendor.fetch_bytes(URL) == b"payload"
    assert clock.sleeps == [2.0, 2.0, 4.0, 8.0]


def test_retries_stop_after_the_final_attempt(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    transport = _transport(monkeypatch, *[_http_error(503)] * vendor.MAX_FETCH_ATTEMPTS)

    with pytest.raises(TranslationDataError, match="could not fetch JSON from"):
        vendor.fetch_bytes(URL)
    assert len(transport.urls) == vendor.MAX_FETCH_ATTEMPTS
    # One pause between each pair of attempts; the capped exponential schedule.
    assert clock.sleeps == [2.0, 2.0, 4.0, 8.0, 16.0, 30.0]


def test_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    transport = _transport(monkeypatch, _http_error(404))

    with pytest.raises(TranslationDataError, match="could not fetch JSON from"):
        vendor.fetch_bytes(URL)
    assert len(transport.urls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("failure", [URLError("no route to host"), TimeoutError("timed out")])
def test_network_failures_are_wrapped_as_translation_errors(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, failure: BaseException
) -> None:
    _transport(monkeypatch, failure)

    with pytest.raises(TranslationDataError, match="could not fetch resource from"):
        vendor.fetch_bytes(URL)


def test_back_to_back_requests_are_spaced_by_the_request_delay(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    _transport(monkeypatch, b"first", b"second")

    assert vendor.fetch_bytes(URL) == b"first"
    assert vendor.fetch_bytes(URL) == b"second"
    assert clock.sleeps == [pytest.approx(vendor.REQUEST_DELAY_SECONDS)]


def test_json_endpoints_are_decoded_and_malformed_bodies_are_refused(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    _transport(monkeypatch, b'{"sha": "abc"}', b"\xff\xfe", b"{not json")

    assert vendor.fetch_json(URL) == {"sha": "abc"}
    with pytest.raises(TranslationDataError, match="could not decode JSON from"):
        vendor.fetch_json(URL)
    with pytest.raises(TranslationDataError, match="could not decode JSON from"):
        vendor.fetch_json(URL)


def test_vendor_command_line_passes_its_options_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: dict[str, object] = {}

    def fake_vendor_dataset(**kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(vendor, "vendor_dataset", fake_vendor_dataset)
    arguments = [
        "--dataset",
        "owner/dataset",
        "--split",
        "validation",
        "--output",
        str(tmp_path / "out"),
        "--expected-rows",
        "5",
        "--expected-languages",
        "2",
        "--source",
        "viewer",
        "--force",
    ]

    assert vendor.main(arguments) == 0
    assert received == {
        "dataset": "owner/dataset",
        "split": "validation",
        "output": tmp_path / "out",
        "expected_row_count": 5,
        "expected_language_count": 2,
        "source": "viewer",
        "force": True,
    }


def test_vendor_command_line_reports_refused_datasets_as_usage_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refusing(**_kwargs: object) -> None:
        raise TranslationDataError("the Hub translation source only provides the train split")

    monkeypatch.setattr(vendor, "vendor_dataset", refusing)

    with pytest.raises(SystemExit) as exit_info:
        vendor.main([])

    assert exit_info.value.code == 2
    assert "only provides the train split" in capsys.readouterr().err
