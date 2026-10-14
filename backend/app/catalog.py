"""
Product catalog: deterministic demo data and a read-only in-memory store.

SKUs match the items already used across the existing test fixtures, operations,
and knowledge base so catalog lookups are consistent with existing order data.
Persistence belongs to Phase 20; this module is intentionally process-local.
"""

from decimal import Decimal
from threading import RLock
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .domain import Name, Record
from .service import NotFoundError


# ---------------------------------------------------------------------------
# Domain records
# ---------------------------------------------------------------------------

class Money(Record):
    """Immutable money value; currency is always USD in Phase 17B.1."""
    amount: Annotated[Decimal, Field(ge=Decimal("0.01"), max_digits=12, decimal_places=2)]
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "USD"


class Product(Record):
    """Read-only product that can be referenced when creating a customer order."""
    id: UUID
    sku: Name
    name: Name
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
    ]
    unit_price: Money
    available: bool = True


# ---------------------------------------------------------------------------
# Deterministic demo seed data
# ---------------------------------------------------------------------------
# SKUs deliberately mirror those in operations.py (LAP-1, LAP-2) and
# the knowledge base (ACC-1, WAR-1) so existing tests remain coherent.

_SEED_PRODUCTS: tuple[Product, ...] = (
    Product(
        id=UUID("10000000-0000-4000-8000-000000000001"),
        sku="LAP-1",
        name="ProBook Laptop 15",
        description="15-inch business laptop with Intel Core i7, 16 GB RAM, 512 GB SSD.",
        unit_price=Money(amount=Decimal("1299.99")),
        available=True,
    ),
    Product(
        id=UUID("10000000-0000-4000-8000-000000000002"),
        sku="LAP-2",
        name="ProBook Laptop 13",
        description="13-inch ultrabook with Intel Core i5, 8 GB RAM, 256 GB SSD.",
        unit_price=Money(amount=Decimal("999.99")),
        available=False,  # Out of stock — matches available_quantity=0 in operations
    ),
    Product(
        id=UUID("10000000-0000-4000-8000-000000000003"),
        sku="ACC-1",
        name="USB-C Docking Station",
        description="7-port USB-C dock with HDMI, DisplayPort, Ethernet, and 100 W PD.",
        unit_price=Money(amount=Decimal("149.99")),
        available=True,
    ),
    Product(
        id=UUID("10000000-0000-4000-8000-000000000004"),
        sku="WAR-1",
        name="Extended Warranty (3 Year)",
        description="Three-year extended hardware warranty covering parts and labour.",
        unit_price=Money(amount=Decimal("199.99")),
        available=True,
    ),
    Product(
        id=UUID("10000000-0000-4000-8000-000000000005"),
        sku="BAG-1",
        name="Laptop Sleeve 15\"",
        description="Water-resistant neoprene sleeve for 15-inch laptops.",
        unit_price=Money(amount=Decimal("29.99")),
        available=True,
    ),
)


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class CatalogService:
    """
    Read-only product catalog backed by deterministic seed data.
    Write operations (add/update/remove) belong to Phase 20+.
    Thread-safe: all reads go through an RLock for consistent future extension.
    """

    def __init__(self, seed: tuple[Product, ...] = _SEED_PRODUCTS) -> None:
        self._lock = RLock()
        # Index by id and sku for O(1) lookups in both directions.
        self._by_id: dict[UUID, Product] = {}
        self._by_sku: dict[str, Product] = {}
        for product in seed:
            self._by_id[product.id] = product
            self._by_sku[product.sku] = product

    def list_products(self, *, available_only: bool = False) -> list[Product]:
        """Return all products, optionally filtered to available-only."""
        with self._lock:
            products = list(self._by_id.values())
        if available_only:
            products = [p for p in products if p.available]
        return products

    def get_by_id(self, product_id: UUID) -> Product:
        with self._lock:
            product = self._by_id.get(product_id)
        if product is None:
            raise NotFoundError("Product not found")
        return product

    def get_by_sku(self, sku: str) -> Product:
        with self._lock:
            product = self._by_sku.get(sku)
        if product is None:
            raise NotFoundError(f"Product with SKU '{sku}' not found")
        return product

    def sku_exists(self, sku: str) -> bool:
        with self._lock:
            return sku in self._by_sku
