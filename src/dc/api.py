"""Which API failures are worth retrying.

Retrying everything is the default mistake, and it is expensive in the way that
matters least visibly: a malformed request fails identically every time, so the
backoff just delays the error while the run looks like it is working. Only
transient failures should be retried; a 4xx means *this request is wrong* and
the only useful response is to stop and say so.
"""

from __future__ import annotations

__all__ = ["is_retryable"]


def is_retryable(exc: BaseException) -> bool:
    """True for transient failures: timeouts, connection drops, 429, and 5xx.

    Everything else — 400 malformed request, 401 bad key, 403, 404 — is a fact
    about the request rather than the network, and will fail the same way on
    every attempt.
    """
    from anthropic import APIConnectionError, APIStatusError

    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False
