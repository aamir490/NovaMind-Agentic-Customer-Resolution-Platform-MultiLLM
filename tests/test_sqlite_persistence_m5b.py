"""Focused M5B SQLite schema and transaction tests; no service wiring."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from backend.app.persistence import MIGRATIONS, PersistenceError, SQLitePersistence


class SQLitePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "novamind.sqlite"

    def store(self):
        return SQLitePersistence(self.path, initialize=True)

    def insert_case(self, store):
        with store.transaction() as connection:
            connection.execute("INSERT INTO customers VALUES (?, ?)", ("customer-1", "Customer"))
            connection.execute("INSERT INTO orders VALUES (?, ?, ?)", ("order-1", "customer-1", "2026-01-01T00:00:00Z"))
            connection.execute("INSERT INTO order_items VALUES (?, ?, ?, ?)", ("order-1", "LAP-1", "Laptop", 1))
            connection.execute(
                "INSERT INTO support_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("case-1", "customer-1", "order-1", "Issue", "Description", "open",
                 "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )

    def test_initialization_applies_all_versioned_migrations(self):
        store = self.store()
        with store.transaction() as connection:
            versions = [row[0] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(versions, [migration.version for migration in MIGRATIONS])
        self.assertTrue({"customers", "delivery_facts", "eligibility_evidence", "resolution_proposals",
                         "workflow_review_bindings", "workflow_audit_events", "workflow_runs"} <= tables)

    def test_migrations_are_repeatable_without_schema_changes(self):
        self.store()
        reopened = SQLitePersistence(self.path)
        with reopened.transaction() as connection:
            versions = [row[0] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
        self.assertEqual(versions, [migration.version for migration in MIGRATIONS])

    def test_foreign_key_and_unique_constraints_reject_invalid_evidence(self):
        store = self.store()
        self.insert_case(store)
        with self.assertRaises(PersistenceError):
            with store.transaction() as connection:
                connection.execute(
                    "INSERT INTO delivery_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("fact-1", "case-1", "order-1", "customer-1", "LAP-1", 1,
                     "2026-01-01T00:00:00Z", "wrong_item", "unused", "unknown-source", "ref-1",
                     "2026-01-01T00:00:00Z"),
                )
        with store.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM delivery_facts").fetchone()[0], 0)

    def test_delivery_provenance_reference_is_unique(self):
        store = self.store()
        self.insert_case(store)
        with store.transaction() as connection:
            connection.execute("INSERT INTO trusted_delivery_sources VALUES (?)", ("fulfillment",))
            connection.execute(
                "INSERT INTO delivery_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("fact-1", "case-1", "order-1", "customer-1", "LAP-1", 1,
                 "2026-01-01T00:00:00Z", "wrong_item", "unused", "fulfillment", "source-ref",
                 "2026-01-01T00:00:00Z"),
            )
        with self.assertRaises(PersistenceError):
            with store.transaction() as connection:
                connection.execute(
                    "INSERT INTO support_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("case-2", "customer-1", "order-1", "Second issue", "Description", "open",
                     "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
                )
                connection.execute(
                    "INSERT INTO delivery_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("fact-2", "case-2", "order-1", "customer-1", "LAP-1", 1,
                     "2026-01-01T00:00:00Z", "wrong_item", "unused", "fulfillment", "source-ref",
                     "2026-01-01T00:00:00Z"),
                )
        with store.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM delivery_facts").fetchone()[0], 1)

    def test_failed_transaction_rolls_back_prior_writes(self):
        store = self.store()
        with self.assertRaises(PersistenceError):
            with store.transaction() as connection:
                connection.execute("INSERT INTO customers VALUES (?, ?)", ("customer-rollback", "Rollback"))
                connection.execute("INSERT INTO orders VALUES (?, ?, ?)",
                                   ("order-invalid", "missing-customer", "2026-01-01T00:00:00Z"))
        with store.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM customers").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
