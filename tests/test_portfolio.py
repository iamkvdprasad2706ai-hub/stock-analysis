import pytest

from stock_analysis.portfolio import (
    PortfolioDatabaseError,
    PortfolioStore,
    PortfolioValidationError,
)


def test_holdings_crud_persists_and_normalizes_symbols(tmp_path):
    database = tmp_path / "data" / "portfolio.sqlite3"
    store = PortfolioStore(database)

    added = store.upsert_holding(" reliance.ns ", 10, 2500, "Energy")
    assert (added.symbol, added.quantity, added.average_cost, added.sector) == (
        "RELIANCE.NS",
        10,
        2500,
        "Energy",
    )

    updated = store.update_holding("RELIANCE.NS", 12, 2600, "Oil & Gas")
    assert updated.quantity == 12
    assert updated.average_cost == 2600
    assert updated.sector == "Oil & Gas"
    assert PortfolioStore(database).get_holdings() == [updated]

    assert store.remove_holding("reliance.ns")
    assert not store.remove_holding("RELIANCE.NS")
    assert store.get_holdings() == []


def test_positions_allocations_and_concentration_use_current_prices(tmp_path):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    store.upsert_holding("AAA", 10, 8, "Technology")
    store.upsert_holding("BBB", 5, 20, "Technology")
    store.upsert_holding("CCC", 10, 5, "Utilities")

    positions = {position.symbol: position for position in store.positions(
        {"aaa": 10, "BBB": 18, "CCC": 10}
    )}
    assert positions["AAA"].market_value == 100
    assert positions["AAA"].cost_basis == 80
    assert positions["AAA"].profit_loss == 20
    assert positions["AAA"].profit_loss_pct == 25
    assert positions["BBB"].profit_loss == -10

    by_symbol = store.allocation_by_symbol({"AAA": 10, "BBB": 18, "CCC": 10})
    assert [item.name for item in by_symbol] == ["AAA", "CCC", "BBB"]
    assert [item.percentage for item in by_symbol] == pytest.approx(
        [100 / 2.9, 100 / 2.9, 90 / 2.9]
    )
    by_sector = store.allocation_by_sector({"AAA": 10, "BBB": 18, "CCC": 10})
    assert [(item.name, item.value) for item in by_sector] == [
        ("Technology", 190),
        ("Utilities", 100),
    ]
    warnings = store.concentration_warnings({"AAA": 10, "BBB": 18, "CCC": 10}, threshold=0.33)
    assert [item.symbol for item in warnings] == ["AAA", "CCC"]
    assert [item.percentage for item in warnings] == pytest.approx([100 / 2.9, 100 / 2.9])
    assert all(item.threshold == 33 for item in warnings)


def test_empty_portfolio_and_missing_or_invalid_prices(tmp_path):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    assert store.positions({}) == []
    assert store.allocation_by_symbol({}) == []
    assert store.allocation_by_sector({}) == []
    assert store.concentration_warnings({}) == []

    store.upsert_holding("ABC", 1, 10)
    with pytest.raises(PortfolioValidationError, match="Missing latest prices"):
        store.positions({})
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(PortfolioValidationError, match="Latest price"):
            store.positions({"ABC": invalid})


@pytest.mark.parametrize(
    ("symbol", "quantity", "cost"),
    [
        ("bad ticker!", 1, 1),
        ("ABC", 0, 1),
        ("ABC", -1, 1),
        ("ABC", float("nan"), 1),
        ("ABC", 1, 0),
        ("ABC", 1, float("inf")),
    ],
)
def test_holding_inputs_are_validated(tmp_path, symbol, quantity, cost):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    with pytest.raises(PortfolioValidationError):
        store.upsert_holding(symbol, quantity, cost)


def test_transactions_crud_and_symbol_filtering(tmp_path):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    transaction = store.add_transaction(
        "tcs", "buy", 2, 3500, transaction_date="2025-01-03", notes="Initial purchase"
    )
    assert transaction.symbol == "TCS"
    assert transaction.side == "BUY"
    assert store.get_transaction(transaction.id) == transaction
    assert store.get_transactions("TCS") == [transaction]

    updated = store.update_transaction(
        transaction.id,
        "TCS.NS",
        "sell",
        1,
        3600,
        transaction_date="2025-02-01",
        notes="Partial sale",
    )
    assert updated.side == "SELL"
    assert updated.quantity == 1
    assert updated.transaction_date.startswith("2025-02-01")
    assert store.get_transactions("TCS.NS") == [updated]
    assert store.get_holdings() == []

    assert store.remove_transaction(transaction.id)
    assert not store.remove_transaction(transaction.id)
    with pytest.raises(KeyError):
        store.get_transaction(transaction.id)
    with pytest.raises(KeyError):
        store.update_transaction(transaction.id, "TCS", "BUY", 1, 10)


@pytest.mark.parametrize(
    ("symbol", "side", "quantity", "price"),
    [
        ("bad ticker!", "BUY", 1, 1),
        ("ABC", "HOLD", 1, 1),
        ("ABC", "BUY", 0, 1),
        ("ABC", "BUY", 1, -1),
    ],
)
def test_transaction_inputs_are_validated(tmp_path, symbol, side, quantity, price):
    store = PortfolioStore(tmp_path / "portfolio.sqlite3")
    with pytest.raises(PortfolioValidationError):
        store.add_transaction(symbol, side, quantity, price)


def test_database_failures_are_reported_as_portfolio_database_errors(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("not a directory")
    with pytest.raises(PortfolioDatabaseError, match="database operation failed"):
        PortfolioStore(blocker / "portfolio.sqlite3")
