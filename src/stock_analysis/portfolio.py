"""Persistent single-user portfolio data stored in SQLite.

Holdings are the current position snapshot; transactions are a separate,
editable activity log and do not automatically change holdings.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from pathlib import Path
import re
import sqlite3
from typing import Iterator, Mapping


class PortfolioValidationError(ValueError):
    """Raised when portfolio input is invalid."""


class PortfolioDatabaseError(RuntimeError):
    """Raised when the portfolio database cannot be accessed or updated."""


@dataclass(frozen=True)
class Holding:
    symbol: str
    quantity: float
    average_cost: float
    sector: str


@dataclass(frozen=True)
class Transaction:
    id: int
    symbol: str
    side: str
    quantity: float
    price: float
    transaction_date: str
    notes: str | None


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_cost: float
    latest_price: float
    market_value: float
    cost_basis: float
    profit_loss: float
    profit_loss_pct: float
    sector: str


@dataclass(frozen=True)
class Allocation:
    name: str
    value: float
    percentage: float


@dataclass(frozen=True)
class ConcentrationWarning:
    symbol: str
    percentage: float
    threshold: float


_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")


def _symbol(value: str) -> str:
    if not isinstance(value, str):
        raise PortfolioValidationError("Ticker symbol must be text.")
    normalized = value.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(normalized):
        raise PortfolioValidationError(
            "Ticker must be 1-32 letters or digits, optionally containing '.', '_' or '-'."
        )
    return normalized


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise PortfolioValidationError(f"{label} must be a finite number greater than zero.")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortfolioValidationError(
            f"{label} must be a finite number greater than zero."
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise PortfolioValidationError(f"{label} must be a finite number greater than zero.")
    return number


def _transaction_date(value: str | date | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise PortfolioValidationError("Transaction date must be an ISO date or datetime.")
    try:
        return datetime.fromisoformat(value).isoformat(timespec="seconds")
    except ValueError:
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise PortfolioValidationError(
                "Transaction date must be an ISO date or datetime."
            ) from exc


class PortfolioStore:
    """Read and write a local portfolio database.

    Args:
        db_path: SQLite file path. Defaults to
            ``~/.stock_analysis/portfolio.sqlite3``; inject a temporary path
            for tests or an alternate user-data location.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).expanduser() if db_path is not None else (
            Path.home() / ".stock_analysis" / "portfolio.sqlite3"
        )
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.db_path, timeout=5.0)
            connection.row_factory = sqlite3.Row
            yield connection
            connection.commit()
        except (sqlite3.Error, OSError) as exc:
            if connection is not None:
                connection.rollback()
            raise PortfolioDatabaseError(
                f"Portfolio database operation failed: {exc}"
            ) from exc
        except Exception:
            if connection is not None:
                connection.rollback()
            raise
        finally:
            if connection is not None:
                connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS holdings (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL CHECK (quantity > 0),
                    average_cost REAL NOT NULL CHECK (average_cost > 0),
                    sector TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
                    quantity REAL NOT NULL CHECK (quantity > 0),
                    price REAL NOT NULL CHECK (price > 0),
                    transaction_date TEXT NOT NULL,
                    notes TEXT
                );
                CREATE INDEX IF NOT EXISTS transactions_symbol_date
                    ON transactions(symbol, transaction_date DESC);
                """
            )

    def upsert_holding(
        self,
        symbol: str,
        quantity: float,
        average_cost: float,
        sector: str | None = None,
    ) -> Holding:
        """Insert or replace a symbol's position; omitted sectors become ``Unknown``."""
        normalized = _symbol(symbol)
        shares = _positive_number(quantity, "Quantity")
        cost = _positive_number(average_cost, "Average cost")
        normalized_sector = "Unknown" if sector is None else sector.strip()
        if not normalized_sector:
            raise PortfolioValidationError("Sector must not be empty.")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO holdings(symbol, quantity, average_cost, sector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity = excluded.quantity,
                    average_cost = excluded.average_cost,
                    sector = excluded.sector
                """,
                (normalized, shares, cost, normalized_sector),
            )
        return Holding(normalized, shares, cost, normalized_sector)

    def update_holding(
        self,
        symbol: str,
        quantity: float,
        average_cost: float,
        sector: str | None = None,
    ) -> Holding:
        """Update a holding, inserting it if the symbol does not yet exist."""
        return self.upsert_holding(symbol, quantity, average_cost, sector)

    def remove_holding(self, symbol: str) -> bool:
        """Remove a holding by symbol; return whether a row was removed."""
        normalized = _symbol(symbol)
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM holdings WHERE symbol = ?", (normalized,))
            return cursor.rowcount > 0

    def get_holdings(self) -> list[Holding]:
        """Return all holdings ordered by ticker."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT symbol, quantity, average_cost, sector FROM holdings ORDER BY symbol"
            ).fetchall()
        return [
            Holding(row["symbol"], row["quantity"], row["average_cost"], row["sector"])
            for row in rows
        ]

    def add_transaction(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        transaction_date: str | date | datetime | None = None,
        notes: str | None = None,
    ) -> Transaction:
        """Create a BUY/SELL activity record without changing the holdings snapshot."""
        normalized = _symbol(symbol)
        normalized_side = side.strip().upper() if isinstance(side, str) else ""
        if normalized_side not in {"BUY", "SELL"}:
            raise PortfolioValidationError("Transaction side must be 'BUY' or 'SELL'.")
        shares = _positive_number(quantity, "Quantity")
        unit_price = _positive_number(price, "Transaction price")
        date_value = _transaction_date(transaction_date)
        if notes is not None and not isinstance(notes, str):
            raise PortfolioValidationError("Transaction notes must be text or None.")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO transactions(symbol, side, quantity, price, transaction_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (normalized, normalized_side, shares, unit_price, date_value, notes),
            )
            transaction_id = cursor.lastrowid
        return Transaction(
            int(transaction_id), normalized, normalized_side, shares, unit_price, date_value, notes
        )

    def update_transaction(
        self,
        transaction_id: int,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        transaction_date: str | date | datetime | None = None,
        notes: str | None = None,
    ) -> Transaction:
        """Replace a transaction's fields; raise ``KeyError`` if its ID is absent."""
        if isinstance(transaction_id, bool) or not isinstance(transaction_id, int) or transaction_id < 1:
            raise PortfolioValidationError("Transaction ID must be a positive integer.")
        normalized = _symbol(symbol)
        normalized_side = side.strip().upper() if isinstance(side, str) else ""
        if normalized_side not in {"BUY", "SELL"}:
            raise PortfolioValidationError("Transaction side must be 'BUY' or 'SELL'.")
        shares = _positive_number(quantity, "Quantity")
        unit_price = _positive_number(price, "Transaction price")
        date_value = _transaction_date(transaction_date)
        if notes is not None and not isinstance(notes, str):
            raise PortfolioValidationError("Transaction notes must be text or None.")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE transactions
                SET symbol = ?, side = ?, quantity = ?, price = ?, transaction_date = ?, notes = ?
                WHERE id = ?
                """,
                (normalized, normalized_side, shares, unit_price, date_value, notes, transaction_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"No transaction found with ID {transaction_id}.")
        return Transaction(
            transaction_id, normalized, normalized_side, shares, unit_price, date_value, notes
        )

    def remove_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction by ID; return whether a row was removed."""
        if isinstance(transaction_id, bool) or not isinstance(transaction_id, int) or transaction_id < 1:
            raise PortfolioValidationError("Transaction ID must be a positive integer.")
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
            return cursor.rowcount > 0

    def get_transactions(self, symbol: str | None = None) -> list[Transaction]:
        """Return the activity log, optionally filtered by ticker, newest first."""
        with self._connection() as connection:
            if symbol is None:
                rows = connection.execute(
                    """
                    SELECT id, symbol, side, quantity, price, transaction_date, notes
                    FROM transactions ORDER BY transaction_date DESC, id DESC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, symbol, side, quantity, price, transaction_date, notes
                    FROM transactions WHERE symbol = ?
                    ORDER BY transaction_date DESC, id DESC
                    """,
                    (_symbol(symbol),),
                ).fetchall()
        return [
            Transaction(
                row["id"],
                row["symbol"],
                row["side"],
                row["quantity"],
                row["price"],
                row["transaction_date"],
                row["notes"],
            )
            for row in rows
        ]

    def get_transaction(self, transaction_id: int) -> Transaction:
        """Return a transaction by ID; raise ``KeyError`` if absent."""
        if isinstance(transaction_id, bool) or not isinstance(transaction_id, int) or transaction_id < 1:
            raise PortfolioValidationError("Transaction ID must be a positive integer.")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, symbol, side, quantity, price, transaction_date, notes
                FROM transactions WHERE id = ?
                """,
                (transaction_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No transaction found with ID {transaction_id}.")
        return Transaction(
            row["id"],
            row["symbol"],
            row["side"],
            row["quantity"],
            row["price"],
            row["transaction_date"],
            row["notes"],
        )

    def positions(self, latest_prices: Mapping[str, float]) -> list[Position]:
        """Calculate market value, cost basis, and unrealized P&L for each holding.

        Prices must be supplied for every holding. Price mapping keys are
        normalized using the same ticker rules as holdings.
        """
        prices: dict[str, float] = {}
        for key, value in latest_prices.items():
            normalized = _symbol(key)
            prices[normalized] = _positive_number(value, f"Latest price for {normalized}")
        holdings = self.get_holdings()
        missing = [holding.symbol for holding in holdings if holding.symbol not in prices]
        if missing:
            raise PortfolioValidationError(
                f"Missing latest prices for: {', '.join(missing)}."
            )
        results = []
        for holding in holdings:
            latest_price = prices[holding.symbol]
            market_value = holding.quantity * latest_price
            cost_basis = holding.quantity * holding.average_cost
            profit_loss = market_value - cost_basis
            results.append(
                Position(
                    holding.symbol,
                    holding.quantity,
                    holding.average_cost,
                    latest_price,
                    market_value,
                    cost_basis,
                    profit_loss,
                    profit_loss / cost_basis * 100,
                    holding.sector,
                )
            )
        return results

    def allocation_by_symbol(self, latest_prices: Mapping[str, float]) -> list[Allocation]:
        """Return current-value allocation per ticker, largest first."""
        return self._allocations(self.positions(latest_prices), key=lambda position: position.symbol)

    def allocation_by_sector(self, latest_prices: Mapping[str, float]) -> list[Allocation]:
        """Return current-value allocation per sector, largest first."""
        return self._allocations(self.positions(latest_prices), key=lambda position: position.sector)

    def concentration_warnings(
        self, latest_prices: Mapping[str, float], threshold: float = 0.25
    ) -> list[ConcentrationWarning]:
        """Warn when a single ticker exceeds ``threshold`` of portfolio value."""
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise PortfolioValidationError("Concentration threshold must be between 0 and 1.")
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise PortfolioValidationError("Concentration threshold must be between 0 and 1.")
        return [
            ConcentrationWarning(item.name, item.percentage, float(threshold) * 100)
            for item in self.allocation_by_symbol(latest_prices)
            if item.percentage / 100 > threshold
        ]

    @staticmethod
    def _allocations(
        positions: list[Position], key: object
    ) -> list[Allocation]:
        if not positions:
            return []
        total_value = sum(position.market_value for position in positions)
        values: dict[str, float] = {}
        for position in positions:
            name = key(position)  # type: ignore[operator]
            values[name] = values.get(name, 0.0) + position.market_value
        return [
            Allocation(name, value, value / total_value * 100)
            for name, value in sorted(values.items(), key=lambda item: (-item[1], item[0]))
        ]
