import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
from finrl.meta.paper_trading.paper_trader import PaperTrader

@pytest.mark.skip(reason="Disabling this test for now")
@pytest.fixture
def mock_broker():
    """Mock the broker for testing."""
    broker = MagicMock()
    broker.get_account.return_value = MagicMock(cash=1000, last_equity=1000)
    broker.fetch_latest_data.return_value = (1.0, np.zeros((1,)), 0)
    return broker

@pytest.fixture
def paper_trader(mock_broker):
    """Create a PaperTrader instance using the mocked broker."""
    trader = PaperTrader(
        ticker_list=["AAPL"],
        time_interval="1Min",
        drl_lib="elegantrl",
        agent="ppo",
        cwd=".",
        net_dim=10,
        state_dim=10,
        action_dim=5,
        tech_indicator_list=["SMA"],
        broker="alpaca",
        argv= {
            "ALPACA_API_KEY": os.environ.get("ALPACA_API_KEY"),
            "ALPACA_API_SECRET": os.environ.get("ALPACA_API_SECRET"),
            "ALPACA_API_BASE_URL": os.environ.get("ALPACA_API_BASE_URL"),
        }
    )
    trader.broker = mock_broker  # Use the mocked broker
    return trader

@pytest.mark.skip(reason="Disabling this test for now")
def test_initialization(paper_trader):
    """Test the initialization of PaperTrader."""
    assert paper_trader.cash == 1000
    assert paper_trader.stockUniverse == ["AAPL"]
    assert paper_trader.time_interval == 60  # 1Min

@pytest.mark.skip(reason="Disabling this test for now")
def test_get_state(paper_trader):
    """Test the get_state method."""
    state = paper_trader.get_state()
    assert state is not None
    assert len(state) == 17  # Adjust based on your state size

@pytest.mark.skip(reason="Disabling this test for now")
def test_trade(paper_trader):
    """Test the trade method."""
    paper_trader.get_state = MagicMock(return_value=np.zeros((10,)))
    paper_trader.submitOrder = MagicMock()

    paper_trader.trade()

    # Check that submitOrder is called when actions are taken
    assert paper_trader.submitOrder.call_count == 0  # Adjust based on the action logic

@pytest.mark.skip(reason="Disabling this test for now")
def test_latency(paper_trader):
    """Test the test_latency method."""
    latency = paper_trader.test_latency()
    assert latency >= 0  # Ensure latency is non-negative

@pytest.mark.skip(reason="Disabling this test for now")
def test_run(paper_trader):
    """Test the run method."""
    paper_trader.broker.list_orders.return_value = []
    paper_trader.run()  # Should not raise any exceptions
    assert True  # If it runs without exceptions