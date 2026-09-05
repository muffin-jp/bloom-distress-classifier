"""Which failures get retried.

Retrying a permanent error is the expensive mistake: it turns a typo into four
backoffs per call across a whole run, while the output still looks like progress.
"""

from __future__ import annotations

import anthropic
import httpx2
import pytest

from dc.api import is_retryable


def status_error(code: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        f"http {code}", response=httpx2.Response(code, request=request), body=None
    )


@pytest.mark.parametrize("code", [429, 500, 502, 503, 529])
def test_transient_statuses_are_retried(code: int) -> None:
    assert is_retryable(status_error(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(code: int) -> None:
    # The 400 that motivated this ("For 'number' type, properties maximum,
    # minimum are not supported") was a property of every request in the run.
    # Retrying it four times per call only delayed the report.
    assert not is_retryable(status_error(code))


def test_connection_failures_are_retried() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    assert is_retryable(anthropic.APIConnectionError(request=request))


def test_a_parse_failure_is_not_retried() -> None:
    # An unparseable reply is a bug in the schema or the parser, not the network.
    assert not is_retryable(ValueError("teacher reply contained no JSON object"))
