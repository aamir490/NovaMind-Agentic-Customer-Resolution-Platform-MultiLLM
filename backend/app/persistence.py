"""M5B SQLite schema and transaction foundation; not wired into application services."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import ContextManager, Iterator, Protocol


class PersistenceError(Exception):
    """Storage failure safe to expose only as a generic application code."""


class UnitOfWork(Protocol):
    def transaction(self) -> ContextManager[sqlite3.Connection]: ...


class BusinessRepository(Protocol):
    """Future customer/order/case repository boundary."""

    def transaction(self) -> ContextManager[sqlite3.Connection]: ...


class WorkflowRepository(Protocol):
    """Future proposal, review, audit, and workflow metadata boundary."""

    def transaction(self) -> ContextManager[sqlite3.Connection]: ...


@dataclass(frozen=True)
class Migration:
    version: int
    statements: tuple[str, ...]


MIGRATIONS = (
    Migration(1, (
        """CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY NOT NULL,
            applied_at TEXT NOT NULL
        )""",
        """CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL CHECK(length(trim(name)) BETWEEN 1 AND 120)
        )""",
        """CREATE TABLE orders (
            order_id TEXT PRIMARY KEY NOT NULL,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX orders_customer ON orders(customer_id, created_at, order_id)",
        """CREATE TABLE order_items (
            order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
            sku TEXT NOT NULL CHECK(length(trim(sku)) BETWEEN 1 AND 120),
            name TEXT NOT NULL CHECK(length(trim(name)) BETWEEN 1 AND 120),
            quantity INTEGER NOT NULL CHECK(quantity BETWEEN 1 AND 1000),
            PRIMARY KEY(order_id, sku)
        )""",
        """CREATE TABLE support_cases (
            case_id TEXT PRIMARY KEY NOT NULL,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            order_id TEXT NOT NULL REFERENCES orders(order_id),
            subject TEXT NOT NULL CHECK(length(trim(subject)) BETWEEN 1 AND 200),
            description TEXT NOT NULL CHECK(length(trim(description)) BETWEEN 1 AND 4000),
            status TEXT NOT NULL CHECK(status IN ('open', 'in_review', 'escalated')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        "CREATE INDEX support_cases_customer ON support_cases(customer_id, updated_at, case_id)",
        "CREATE INDEX support_cases_order ON support_cases(order_id, case_id)",
        """CREATE TABLE trusted_delivery_sources (
            source_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(source_id)) BETWEEN 1 AND 120)
        )""",
        """CREATE TABLE delivery_facts (
            delivery_fact_id TEXT PRIMARY KEY NOT NULL,
            case_id TEXT NOT NULL REFERENCES support_cases(case_id),
            order_id TEXT NOT NULL,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            sku TEXT NOT NULL,
            quantity INTEGER NOT NULL CHECK(quantity BETWEEN 1 AND 1000),
            delivered_at TEXT NOT NULL,
            reason TEXT NOT NULL CHECK(reason IN ('wrong_item', 'damaged', 'change_of_mind')),
            item_condition TEXT NOT NULL CHECK(item_condition IN ('unused', 'used', 'damaged')),
            source_id TEXT NOT NULL REFERENCES trusted_delivery_sources(source_id),
            source_reference TEXT NOT NULL CHECK(length(trim(source_reference)) BETWEEN 1 AND 120),
            verified_at TEXT NOT NULL,
            FOREIGN KEY(order_id, sku) REFERENCES order_items(order_id, sku),
            UNIQUE(case_id, delivery_fact_id),
            UNIQUE(case_id, sku),
            UNIQUE(source_id, source_reference)
        )""",
        """CREATE TABLE eligibility_evidence (
            evidence_id TEXT PRIMARY KEY NOT NULL,
            case_id TEXT NOT NULL REFERENCES support_cases(case_id),
            order_id TEXT NOT NULL,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            sku TEXT NOT NULL,
            delivery_fact_id TEXT NOT NULL REFERENCES delivery_facts(delivery_fact_id),
            policy_id TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            assessment_json TEXT NOT NULL CHECK(json_valid(assessment_json)),
            created_at TEXT NOT NULL,
            FOREIGN KEY(order_id, sku) REFERENCES order_items(order_id, sku),
            FOREIGN KEY(case_id, delivery_fact_id) REFERENCES delivery_facts(case_id, delivery_fact_id),
            UNIQUE(case_id, evidence_id)
        )""",
        "CREATE INDEX eligibility_evidence_case ON eligibility_evidence(case_id, created_at, evidence_id)",
        """CREATE TABLE resolution_proposals (
            proposal_id TEXT PRIMARY KEY NOT NULL,
            case_id TEXT NOT NULL REFERENCES support_cases(case_id),
            action TEXT NOT NULL CHECK(action IN ('refund', 'replacement', 'return')),
            rationale TEXT NOT NULL CHECK(length(trim(rationale)) BETWEEN 1 AND 4000),
            eligibility_evidence_id TEXT REFERENCES eligibility_evidence(evidence_id),
            status TEXT NOT NULL CHECK(status IN ('PENDING_REVIEW', 'APPROVED', 'REJECTED')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewer_user_id TEXT,
            review_note TEXT,
            CHECK((action = 'refund' AND eligibility_evidence_id IS NOT NULL)
                  OR (action <> 'refund' AND eligibility_evidence_id IS NULL)),
            CHECK((status = 'PENDING_REVIEW' AND reviewed_at IS NULL AND reviewer_user_id IS NULL AND review_note IS NULL)
                  OR (status IN ('APPROVED', 'REJECTED') AND reviewed_at IS NOT NULL
                      AND reviewer_user_id IS NOT NULL AND review_note IS NOT NULL)),
            FOREIGN KEY(case_id, eligibility_evidence_id) REFERENCES eligibility_evidence(case_id, evidence_id),
            UNIQUE(case_id, proposal_id)
        )""",
        "CREATE INDEX resolution_proposals_case ON resolution_proposals(case_id, status, created_at, proposal_id)",
        """CREATE TABLE workflow_runs (
            workflow_id TEXT PRIMARY KEY NOT NULL,
            run_id TEXT UNIQUE NOT NULL,
            case_id TEXT NOT NULL REFERENCES support_cases(case_id),
            owner_user_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            provider TEXT NOT NULL CHECK(provider IN ('local_scripted', 'groq', 'bedrock')),
            status TEXT NOT NULL CHECK(status IN ('RUNNING', 'RESUMING', 'REVIEW_REQUIRED', 'REVIEWED', 'COMPLETED', 'FAILED', 'RECOVERY_REQUIRED')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(owner_user_id, request_id),
            UNIQUE(workflow_id, case_id)
        )""",
        "CREATE INDEX workflow_runs_case ON workflow_runs(case_id, updated_at, workflow_id)",
        """CREATE TABLE workflow_review_bindings (
            proposal_id TEXT PRIMARY KEY NOT NULL REFERENCES resolution_proposals(proposal_id),
            workflow_id TEXT UNIQUE NOT NULL REFERENCES workflow_runs(workflow_id),
            case_id TEXT NOT NULL REFERENCES support_cases(case_id),
            review_id TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(case_id, proposal_id) REFERENCES resolution_proposals(case_id, proposal_id),
            FOREIGN KEY(workflow_id, case_id) REFERENCES workflow_runs(workflow_id, case_id)
        )""",
        """CREATE TABLE workflow_audit_events (
            workflow_id TEXT NOT NULL REFERENCES workflow_runs(workflow_id) ON DELETE CASCADE,
            sequence INTEGER NOT NULL CHECK(sequence >= 1),
            kind TEXT NOT NULL CHECK(kind IN ('AGENT', 'REVIEW_REQUIRED', 'HUMAN_REVIEW', 'FAILURE')),
            proposal_id TEXT REFERENCES resolution_proposals(proposal_id),
            reviewer_user_id TEXT,
            proposal_status TEXT CHECK(proposal_status IN ('PENDING_REVIEW', 'APPROVED', 'REJECTED')),
            event_json TEXT NOT NULL CHECK(json_valid(event_json)),
            created_at TEXT NOT NULL,
            PRIMARY KEY(workflow_id, sequence)
        )""",
    )),
)


class SQLiteUnitOfWork:
    """Short-lived SQLite transactions following the conversation-store convention."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except PersistenceError:
            if connection is not None:
                connection.rollback()
            raise
        except Exception:
            if connection is not None:
                connection.rollback()
            raise PersistenceError("PERSISTENCE_UNAVAILABLE") from None
        finally:
            if connection is not None:
                connection.close()


class SQLitePersistence(SQLiteUnitOfWork):
    """Versioned SQLite foundation. It deliberately owns no application service wiring."""

    def __init__(self, path: str | Path, *, initialize: bool = False) -> None:
        super().__init__(path)
        try:
            if initialize:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.touch(exist_ok=True)
            if not self._path.is_file():
                raise PersistenceError("PERSISTENCE_UNAVAILABLE")
            self._migrate()
            self._validate()
        except PersistenceError:
            raise
        except Exception:
            raise PersistenceError("PERSISTENCE_UNAVAILABLE") from None

    def _migrate(self) -> None:
        connection = None
        try:
            connection = sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True, timeout=5)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY NOT NULL, applied_at TEXT NOT NULL)")
            applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            known = {migration.version for migration in MIGRATIONS}
            if applied - known:
                raise PersistenceError("PERSISTENCE_UNAVAILABLE")
            for migration in MIGRATIONS:
                if migration.version in applied:
                    continue
                for statement in migration.statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                    (migration.version,),
                )
            connection.commit()
        except PersistenceError:
            if connection is not None:
                connection.rollback()
            raise
        except Exception:
            if connection is not None:
                connection.rollback()
            raise PersistenceError("PERSISTENCE_UNAVAILABLE") from None
        finally:
            if connection is not None:
                connection.close()

    def _validate(self) -> None:
        try:
            with self.transaction() as connection:
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise PersistenceError("PERSISTENCE_UNAVAILABLE")
                if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise PersistenceError("PERSISTENCE_UNAVAILABLE")
                versions = [row[0] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
                if versions != [migration.version for migration in MIGRATIONS]:
                    raise PersistenceError("PERSISTENCE_UNAVAILABLE")
        except PersistenceError:
            raise

