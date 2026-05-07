"""Regression tests for native-Gemini support on Google Cloud Vertex AI.

Background
----------
Hermes ships ``GeminiNativeClient`` (``agent/gemini_native_adapter.py``) for the
Google AI Studio surface (``generativelanguage.googleapis.com``). When users
point Hermes at the Vertex AI publisher endpoint instead — e.g.

    https://aiplatform.googleapis.com/v1/projects/<proj>/locations/global/
        publishers/google/models/gemini-3-flash-preview:generateContent

— three things must happen for the request to actually leave the box:

1. ``is_native_gemini_base_url()`` must recognise the Vertex publisher URL so
   the auxiliary client + main agent route the call through
   ``GeminiNativeClient`` instead of the OpenAI SDK (which would happily
   append ``/chat/completions`` to a URL ending in ``:generateContent``,
   producing a silent 404).
2. ``GeminiNativeClient.__init__`` must strip the model + verb suffix
   (``/models/<name>:generateContent``) and remember the per-instance Vertex
   model name — the Gemini REST contract puts the model in the URL, not the
   payload.
3. ``GeminiNativeClient._headers()`` must inject
   ``Authorization: Bearer <gce_metadata_token>`` and **drop** any stale
   ``Authorization`` from ``_default_headers`` (otherwise the SDK's empty
   key bleeds through and the request is rejected with 401).

These tests pin all three behaviours so future refactors of the auxiliary
routing or native adapter don't silently regress Vertex Gemini support.

This is the symmetric companion to ``test_anthropic_vertex_rewriter.py``
which guards the same thing for Claude on Vertex (``:rawPredict``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


VERTEX_GEMINI_BASE_NO_VERB = (
    "https://aiplatform.googleapis.com/v1/projects/test-project/locations/"
    "global/publishers/google/models/gemini-3-flash-preview"
)
VERTEX_GEMINI_BASE_WITH_VERB = VERTEX_GEMINI_BASE_NO_VERB + ":generateContent"
GENLANG_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ---------------------------------------------------------------------------
# 1. URL recognition
# ---------------------------------------------------------------------------


def test_is_native_gemini_base_url_accepts_vertex_publisher_url():
    """Vertex publisher URLs must be routed through GeminiNativeClient."""
    from agent.gemini_native_adapter import is_native_gemini_base_url

    assert is_native_gemini_base_url(VERTEX_GEMINI_BASE_WITH_VERB)
    assert is_native_gemini_base_url(VERTEX_GEMINI_BASE_NO_VERB)


def test_is_native_gemini_base_url_still_accepts_genlang():
    """Backwards compatibility: AI Studio surface must keep matching."""
    from agent.gemini_native_adapter import is_native_gemini_base_url

    assert is_native_gemini_base_url(GENLANG_BASE)
    assert is_native_gemini_base_url(GENLANG_BASE + "/")


def test_is_native_gemini_base_url_rejects_unrelated_endpoints():
    from agent.gemini_native_adapter import is_native_gemini_base_url

    assert not is_native_gemini_base_url("https://api.openai.com/v1")
    assert not is_native_gemini_base_url(
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/"
        "publishers/anthropic/models/claude-opus-4-7:rawPredict"
    )
    assert not is_native_gemini_base_url("")


def test_is_vertex_gemini_base_url_only_matches_vertex():
    from agent.gemini_native_adapter import is_vertex_gemini_base_url

    assert is_vertex_gemini_base_url(VERTEX_GEMINI_BASE_WITH_VERB)
    assert is_vertex_gemini_base_url(VERTEX_GEMINI_BASE_NO_VERB)
    assert not is_vertex_gemini_base_url(GENLANG_BASE)
    assert not is_vertex_gemini_base_url("")


# ---------------------------------------------------------------------------
# 2. Suffix stripping (the OpenAI SDK appends paths to base_url)
# ---------------------------------------------------------------------------


def test_strip_vertex_model_suffix_removes_models_and_verb():
    """``/models/<name>:generateContent`` is part of the per-call path,
    not the base URL — must be stripped at adapter init time."""
    from agent.gemini_native_adapter import _strip_vertex_model_suffix

    out = _strip_vertex_model_suffix(VERTEX_GEMINI_BASE_WITH_VERB)
    assert out.endswith("/publishers/google")
    assert ":generateContent" not in out
    assert "/models/" not in out


def test_strip_vertex_model_suffix_idempotent_on_clean_base():
    """Calling on an already-stripped URL must be a no-op."""
    from agent.gemini_native_adapter import _strip_vertex_model_suffix

    base = (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/"
        "publishers/google"
    )
    assert _strip_vertex_model_suffix(base) == base


def test_strip_vertex_model_suffix_handles_streamGenerateContent():
    """Some auxiliary calls use ``:streamGenerateContent`` for SSE."""
    from agent.gemini_native_adapter import _strip_vertex_model_suffix

    url = VERTEX_GEMINI_BASE_NO_VERB + ":streamGenerateContent"
    out = _strip_vertex_model_suffix(url)
    assert ":streamGenerateContent" not in out
    assert "/models/" not in out


# ---------------------------------------------------------------------------
# 3. Authorization header rewriting
# ---------------------------------------------------------------------------


def test_vertex_client_injects_bearer_token_and_drops_stale_auth():
    """On Vertex, GCE metadata token is the only valid auth.

    Regression: the OpenAI SDK's ``_default_headers`` carry an empty
    ``Authorization`` derived from ``api_key`` — when we route through
    GeminiNativeClient it must overwrite that, not append.
    """
    from agent.gemini_native_adapter import GeminiNativeClient

    with patch(
        "agent.gemini_native_adapter._fetch_gce_metadata_token",
        return_value="ya29.fake-gce-token",
    ):
        client = GeminiNativeClient(
            api_key="ignored-on-vertex",
            base_url=VERTEX_GEMINI_BASE_WITH_VERB,
        )
        headers = client._headers()

    assert headers.get("Authorization") == "Bearer ya29.fake-gce-token"
    # The genlang surface uses x-goog-api-key — Vertex must NOT carry it
    # because GCP rejects "Multiple authentication credentials received".
    assert "x-goog-api-key" not in {k.lower() for k in headers}


def test_genlang_client_uses_x_goog_api_key_not_bearer():
    """AI Studio surface keeps the original behaviour."""
    from agent.gemini_native_adapter import GeminiNativeClient

    client = GeminiNativeClient(
        api_key="AIza-fake-studio-key", base_url=GENLANG_BASE,
    )
    headers = client._headers()

    # Use case-insensitive lookup since header case varies by SDK version
    lower = {k.lower(): v for k, v in headers.items()}
    assert lower.get("x-goog-api-key") == "AIza-fake-studio-key"
    assert "authorization" not in lower or not lower["authorization"]


def test_vertex_client_records_is_vertex_flag():
    """Downstream code branches on ``_is_vertex`` for URL building."""
    from agent.gemini_native_adapter import GeminiNativeClient

    with patch(
        "agent.gemini_native_adapter._fetch_gce_metadata_token",
        return_value="ya29.fake",
    ):
        vertex = GeminiNativeClient(
            api_key="ignored", base_url=VERTEX_GEMINI_BASE_WITH_VERB,
        )
        genlang = GeminiNativeClient(
            api_key="AIza-fake", base_url=GENLANG_BASE,
        )

    assert vertex._is_vertex is True
    assert genlang._is_vertex is False


# ---------------------------------------------------------------------------
# 4. Auxiliary routing — the gate that prevents the silent 404
# ---------------------------------------------------------------------------


def test_wrap_if_needed_rewraps_vertex_gemini_into_native_client():
    """Without this gate, the OpenAI SDK appends ``/chat/completions`` to a
    URL ending in ``:generateContent`` and the request 404s silently. The
    ``_wrap_if_needed`` closure inside ``resolve_provider_client`` must
    detect the URL shape and rewrap the plain OpenAI client into a
    GeminiNativeClient before any other adapter check.

    We can't easily call the closure directly (it captures locals), so we
    drive the public ``resolve_provider_client`` entry point with the
    ``custom`` provider — which is the path users hit via
    ``custom_providers`` in ``config.yaml``.
    """
    from agent.auxiliary_client import resolve_provider_client
    from agent.gemini_native_adapter import GeminiNativeClient

    # Stub out GCE token + the OpenAI SDK constructor so the test stays
    # offline and doesn't need real GCP credentials.
    with patch(
        "agent.gemini_native_adapter._fetch_gce_metadata_token",
        return_value="ya29.fake",
    ):
        client, model = resolve_provider_client(
            provider="custom",
            model="gemini-3-flash-preview",
            explicit_base_url=VERTEX_GEMINI_BASE_WITH_VERB,
            explicit_api_key="ignored-on-vertex",
        )

    assert client is not None, "resolve_provider_client returned None"
    assert isinstance(client, GeminiNativeClient), (
        f"Expected GeminiNativeClient, got {type(client).__name__}. "
        "Without the Vertex Gemini gate the OpenAI SDK takes over and "
        "every aux call (compress / title / vision / curator / web_extract) "
        "silently 404s."
    )
    # Vertex routes the model name in the URL path, so the resolver should
    # surface the requested model unchanged.
    assert "gemini-3-flash-preview" in (model or "")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
