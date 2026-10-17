"""Trusted runtime configuration only. No provisioning, seeding, or model calls."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from backend.app.bedrock import BedrockProvider
from backend.app.domain import Customer
from backend.app.llm import FakeLLMProvider, LLMResponse
from backend.app.main import create_app
from backend.app.operations import AuthoritativeDeliveryFact, TrustedDeliverySource
from backend.app.schemas import CaseCreate, OrderCreate
from backend.app.security import LocalAuthenticationProvider, LocalCredential, Role


class _OfflineScriptedProvider:
    """Explicit local-test provider; contains no transport or credential support."""

    def __init__(self):
        self._fixture = FakeLLMProvider(LLMResponse(
            text='{"decision":{"kind":"final"}}', finish_reason="stop",
        ))

    def generate(self, request, *, response_schema):
        return self._fixture.generate(request, response_schema=response_schema)

    def close(self):
        return None


_OFFLINE_DEMO_SOURCE = TrustedDeliverySource(id="offline-demo-fulfillment")
_BEDROCK_DEMO_SOURCE = TrustedDeliverySource(id="bedrock-demo-fulfillment")


def _offline_demo_enabled():
    value = os.environ.get("NOVAMIND_OFFLINE_DEMO_DATA")
    if value is None:
        return False
    if (value != "true" or os.environ.get("NOVAMIND_LLM_PROVIDER") != "local_scripted"
            or os.environ.get("NOVAMIND_OFFLINE_TESTING") != "true"):
        raise RuntimeError("Container offline demo configuration is invalid")
    return True


def _bedrock_demo_enabled():
    value = os.environ.get("NOVAMIND_BEDROCK_DEMO_DATA")
    if value is None:
        return False
    if (value != "true" or os.environ.get("NOVAMIND_LLM_PROVIDER") != "bedrock"
            or os.environ.get("NOVAMIND_OFFLINE_TESTING") is not None):
        raise RuntimeError("Container Bedrock demo configuration is invalid")
    return True


def _seed_demo(app, customers, *, source, sku, product_name, subject):
    """Seed one clearly labelled, eligible test case through trusted-host boundaries."""
    unique = {customer.id: customer for customer in customers}
    if len(unique) != 1:
        raise RuntimeError("Container offline demo requires exactly one customer identity")
    customer = next(iter(unique.values()))
    service = app.state.case_service
    order = service.create_order(OrderCreate(
        customer_id=customer.id,
        items=[{"sku": sku, "name": product_name, "quantity": 1}],
    ))
    case = service.create_case(CaseCreate(
        customer_id=customer.id, order_id=order.id,
        subject=subject,
        description=("Synthetic local-only case for demonstrating verified eligibility and "
                     "human review. No production delivery record or business action."),
    ))
    now = datetime.now(timezone.utc)
    app.state.delivery_fact_ingestion.ingest(
        source.id,
        AuthoritativeDeliveryFact(
            id=uuid4(), case_id=case.id, order_id=order.id, customer_id=customer.id,
            sku=sku, quantity=1, delivered_at=now - timedelta(days=5),
            reason="wrong_item", condition="unused", source_id=source.id,
            source_reference=f"synthetic-{case.id}", verified_at=now,
        ),
    )


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


def _llm_provider():
    # Reject retired configuration under every selector; never ignore a stale key.
    if any(name.startswith("GEMINI_") for name in os.environ):
        raise RuntimeError("Container LLM provider selection is invalid")
    if any(name.startswith("GROQ_") for name in os.environ):
        raise RuntimeError("Container LLM provider selection is invalid")
    selected = os.environ.get("NOVAMIND_LLM_PROVIDER")
    offline = os.environ.get("NOVAMIND_OFFLINE_TESTING")
    if selected is None:
        if offline is not None:
            raise RuntimeError("Container LLM provider selection is invalid")
        return None
    if selected == "local_scripted":
        cloud_configuration = any(
            name.startswith(("AWS_", "BEDROCK_"))
            for name in os.environ
        )
        if offline != "true" or cloud_configuration:
            raise RuntimeError("Container offline provider selection is invalid")
        return _OfflineScriptedProvider()
    if selected == "groq":
        raise RuntimeError("Container LLM provider selection is invalid")
    if selected == "bedrock":
        if offline is not None:
            raise RuntimeError("Container Bedrock configuration is invalid")
        try:
            region = os.environ.get("AWS_REGION", "")
            model_id = os.environ.get("BEDROCK_MODEL_ID", "")
            # Validate syntax locally; model availability and permissions are
            # determined by AWS only when a request is made.
            if not re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+){1,3}-[0-9]{1,2}", region):
                raise ValueError("Invalid Bedrock region")
            identifier = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}"
            arn = (r"arn:aws(?:-us-gov|-cn)?:bedrock:[a-z0-9-]{1,32}:"
                   r"(?:[0-9]{12})?:(?:foundation-model|inference-profile|"
                   r"application-inference-profile)/" + identifier)
            if not ((not model_id.startswith("arn:") and re.fullmatch(identifier, model_id))
                    or re.fullmatch(arn, model_id)):
                raise ValueError("Invalid Bedrock model ID")
            return BedrockProvider(model_id=model_id, region_name=region)
        except Exception:
            raise RuntimeError("Container Bedrock configuration is invalid") from None
    raise RuntimeError("Container LLM provider selection is invalid")


def create_container_app():
    """Absent configuration stays default-deny; invalid configuration stops startup."""
    auth_provider, startup_customers = _authentication_provider()
    offline_demo = _offline_demo_enabled()
    bedrock_demo = _bedrock_demo_enabled()
    provider = _llm_provider()
    demo_sources = tuple(source for enabled, source in (
        (offline_demo, _OFFLINE_DEMO_SOURCE), (bedrock_demo, _BEDROCK_DEMO_SOURCE),
    ) if enabled)
    try:
        app = create_app(auth_provider=auth_provider,
                         local_provider_factory=(lambda: provider) if provider is not None else None,
                         provider_label=os.environ["NOVAMIND_LLM_PROVIDER"] if provider is not None
                                        else "local_scripted",
                         startup_customers=startup_customers,
                         trusted_delivery_sources=demo_sources)
        if offline_demo:
            _seed_demo(app, startup_customers, source=_OFFLINE_DEMO_SOURCE,
                       sku="LAP-1", product_name="ProBook Laptop 15",
                       subject="[SYNTHETIC OFFLINE DEMO] Eligible refund review")
        if bedrock_demo:
            _seed_demo(app, startup_customers, source=_BEDROCK_DEMO_SOURCE,
                       sku="LAP-10", product_name="NovaBook Laptop 10",
                       subject="[SYNTHETIC BEDROCK DEMO] Eligible LAP-10 refund review")
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
