"""Trusted runtime configuration only. No provisioning, seeding, or model calls."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re

from backend.app.domain import Customer
from backend.app.gemini import GeminiProvider
from backend.app.groq import GroqProvider
from backend.app.main import create_app
from backend.app.security import LocalAuthenticationProvider, LocalCredential, Role


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate configuration key")
        result[key] = value
    return result


def _authentication_provider():
    location = os.environ.get("NOVAMIND_AUTH_FILE", "")
    if not location:
        return None, ()
    try:
        path = Path(location)
        if not path.is_absolute():
            raise ValueError("Configuration path must be absolute")
        with path.open("rb") as source:
            raw = source.read(65537)
        if len(raw) > 65536:
            raise ValueError("Configuration is too large")
        records = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(records, list) or len(records) > 1000:
            raise ValueError("Invalid credential list")
        credentials = tuple(LocalCredential.model_validate(item) for item in records)
        provider = LocalAuthenticationProvider(credentials)
        # For every CUSTOMER credential, seed a deterministic Customer domain
        # record using the exact customer_id declared in the identity.
        # The name is a generic demo label; customer names are not part of
        # authentication configuration and must not be added to credentials.
        # A seeded record does NOT grant authentication — the bearer token is
        # still required for every request.
        startup_customers = tuple(
            Customer(id=credential.identity.customer_id, name="Demo Customer")
            for credential in credentials
            if credential.identity.role == Role.CUSTOMER
        )
    except Exception:
        # Never put credentials, file paths, or Pydantic input excerpts in logs.
        raise RuntimeError("Container authentication configuration is invalid") from None
    return provider, startup_customers


def _gemini_provider():
    # Explicit runtime injection only; no .env loading or alternative key lookup.
    key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL")
    if key is None and model is None:
        return None
    try:
        if (not key or len(key) > 4096 or not key.isascii()
                or any(character.isspace() or not character.isprintable() for character in key)):
            raise ValueError("Invalid key")
        if not model or not re.fullmatch(r"(?:models/)?gemini-[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model):
            raise ValueError("Invalid model")
        return GeminiProvider(model=model)
    except Exception:
        # SDK errors can include configuration values; expose only a fixed message.
        raise RuntimeError("Container Gemini configuration is invalid") from None


def _llm_provider():
    selected = os.environ.get("NOVAMIND_LLM_PROVIDER")
    if selected is None:
        # Preserve existing Gemini opt-in behavior; Groq is never auto-selected.
        return _gemini_provider()
    if selected == "gemini":
        provider = _gemini_provider()
        if provider is None:
            raise RuntimeError("Container Gemini configuration is invalid")
        return provider
    if selected == "groq":
        try:
            return GroqProvider()
        except Exception:
            raise RuntimeError("Container Groq configuration is invalid") from None
    raise RuntimeError("Container LLM provider selection is invalid")


def create_container_app():
    """Absent configuration stays default-deny; invalid configuration stops startup."""
    auth_provider, startup_customers = _authentication_provider()
    provider = _llm_provider()
    try:
        app = create_app(auth_provider=auth_provider,
                         local_provider_factory=(lambda: provider) if provider is not None else None,
                         startup_customers=startup_customers)
    except Exception:
        if provider is not None:
            provider.close()
        raise
    if provider is not None:
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            try:
                async with original_lifespan(app):
                    yield
            finally:
                # The existing lifespan drains run workers before closing their client.
                provider.close()

        app.router.lifespan_context = lifespan
    return app
