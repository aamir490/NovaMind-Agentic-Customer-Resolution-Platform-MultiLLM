"""Focused M5C1 durable business-record tests; application wiring remains in-memory."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from backend.app.domain import Customer, Order, OrderItem
from backend.app.persistence import SQLiteCaseRepository, SQLitePersistence
from backend.app.schemas import CaseCreate, CustomerCreate, OrderCreate
from backend.app.service import CaseService, ConflictError, NotFoundError


class SQLiteCaseRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "novamind.sqlite"

    def service(self, *, initialize=False):
        store = SQLitePersistence(self.path, initialize=initialize)
        return CaseService(SQLiteCaseRepository(store))

    @staticmethod
    def order_request(customer_id):
        return OrderCreate(customer_id=customer_id, items=(
            OrderItem(sku="LAP-1", name="Laptop", quantity=1),
            OrderItem(sku="MSE-1", name="Mouse", quantity=2),
        ))

    def test_records_survive_service_recreation_with_relationships(self):
        first = self.service(initialize=True)
        customer = first.create_customer(CustomerCreate(name="Ada Lovelace"))
        order = first.create_order(self.order_request(customer.id))
        case = first.create_case(CaseCreate(
            customer_id=customer.id, order_id=order.id,
            subject="Wrong item", description="The laptop box contains the wrong model.",
        ))

        recreated = self.service()
        self.assertEqual(recreated.get_customer(customer.id), customer)
        self.assertEqual(recreated.get_order(order.id), order)
        self.assertEqual(recreated.get_case(case.id), case)
        self.assertEqual(recreated.list_orders(customer.id), [order])
        self.assertEqual(recreated.list_cases(customer.id), [case])

    def test_seed_is_idempotent_and_never_overwrites_a_durable_customer(self):
        customer = Customer(id=uuid4(), name="Trusted seed")
        first = self.service(initialize=True)
        first.seed_customer(customer)

        recreated = self.service()
        recreated.seed_customer(customer)
        with self.assertRaises(ValueError):
            recreated.seed_customer(Customer(id=customer.id, name="Replacement seed"))
        self.assertEqual(recreated.get_customer(customer.id), customer)

    def test_invalid_writes_preserve_integrity_and_do_not_partially_persist(self):
        service = self.service(initialize=True)
        customer = service.create_customer(CustomerCreate(name="Owner"))
        other_customer = service.create_customer(CustomerCreate(name="Other owner"))
        order = service.create_order(self.order_request(customer.id))

        with self.assertRaises(NotFoundError):
            service.create_order(self.order_request(uuid4()))
        with self.assertRaises(ConflictError):
            service.create_case(CaseCreate(
                customer_id=other_customer.id, order_id=order.id,
                subject="Not mine", description="This customer does not own that order.",
            ))
        with self.assertRaises(ConflictError):
            service.create_order(OrderCreate(customer_id=customer.id, items=(
                OrderItem(sku="LAP-1", name="Laptop", quantity=1),
                OrderItem(sku="LAP-1", name="Laptop duplicate", quantity=1),
            )))

        self.assertEqual(service.list_orders(customer.id), [order])
        self.assertEqual(service.list_cases(customer.id), [])

    def test_default_case_service_is_still_process_local(self):
        service = CaseService()
        customer = service.create_customer(CustomerCreate(name="In-memory default"))
        self.assertEqual(service.get_customer(customer.id), customer)


if __name__ == "__main__":
    unittest.main()
