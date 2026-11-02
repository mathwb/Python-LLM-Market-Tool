import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Import core modules from the script
from llmimt_main import (
    StockAllocation,
    PortfolioResponse,
    execute_rebalance,
    ValidationError,
    is_ticker_valid,
    load_portfolio_state,
    query_model,
    APIAuthenticationError
)
import tempfile
import litellm

class TestLLMIMT(unittest.TestCase):

    def test_pydantic_validation_success(self):
        """Test that a valid portfolio allocation passes Pydantic validation."""
        valid_json = {
            "reasoning": "Investing in high growth tech and keeping some cash.",
            "allocations": [
                {"ticker": "aapl", "percentage": 30.0},
                {"ticker": "msft", "percentage": 40.0},
                {"ticker": "CASH", "percentage": 10.0}
            ]
        }
        
        response = PortfolioResponse(**valid_json)
        self.assertEqual(response.reasoning, "Investing in high growth tech and keeping some cash.")
        
        # Verify ticker normalization to uppercase
        self.assertEqual(response.allocations[0].ticker, "AAPL")
        self.assertEqual(response.allocations[1].ticker, "MSFT")
        self.assertEqual(response.allocations[2].ticker, "CASH")
        
        # Verify percentages are correct
        self.assertEqual(response.allocations[0].percentage, 30.0)
        self.assertEqual(response.allocations[1].percentage, 40.0)

    def test_pydantic_validation_duplicates(self):
        """Test that duplicate tickers are consolidated and summed correctly."""
        dup_json = {
            "reasoning": "Duplicate entries test.",
            "allocations": [
                {"ticker": "AAPL", "percentage": 20.0},
                {"ticker": "aapl", "percentage": 15.0},
                {"ticker": "MSFT", "percentage": 40.0}
            ]
        }
        response = PortfolioResponse(**dup_json)
        
        # Tickers should be combined: AAPL has 35.0%, MSFT has 40.0%
        alloc_dict = {a.ticker: a.percentage for a in response.allocations}
        self.assertEqual(len(response.allocations), 2)
        self.assertEqual(alloc_dict["AAPL"], 35.0)
        self.assertEqual(alloc_dict["MSFT"], 40.0)

    def test_pydantic_validation_failure_exceeds_100(self):
        """Test that validation fails if allocations exceed 100%."""
        invalid_json = {
            "reasoning": "Over-allocated test.",
            "allocations": [
                {"ticker": "AAPL", "percentage": 60.0},
                {"ticker": "MSFT", "percentage": 50.0}
            ]
        }
        with self.assertRaises(ValidationError):
            PortfolioResponse(**invalid_json)

    def test_pydantic_validation_invalid_percentage(self):
        """Test that negative percentages or values > 100 fail validation."""
        invalid_json_neg = {
            "reasoning": "Negative percentage.",
            "allocations": [
                {"ticker": "AAPL", "percentage": -5.0}
            ]
        }
        with self.assertRaises(ValidationError):
            PortfolioResponse(**invalid_json_neg)

        invalid_json_too_high = {
            "reasoning": "Too high percentage.",
            "allocations": [
                {"ticker": "AAPL", "percentage": 105.0}
            ]
        }
        with self.assertRaises(ValidationError):
            PortfolioResponse(**invalid_json_too_high)

    @patch('llmimt_main.is_ticker_valid')
    def test_rebalancing_mathematics(self, mock_valid):
        """Test execution of portfolio rebalancing and fractional share math."""
        # Mock yfinance validation to say AAPL and MSFT are valid, but ignore others
        mock_valid.side_effect = lambda t: t in ["AAPL", "MSFT"]
        
        prev_state = {
            "portfolio_value": 10000.0,
            "cash_balance": 5000.0,
            "holdings": {
                "AAPL": {"shares": 25.0, "price": 200.0}  # $5000 in AAPL
            }
        }
        
        # Today's open prices: AAPL rose to 220.0, MSFT is 400.0
        # Starting value: 25 shares * $220 = $5500 + $5000 cash = $10,500
        open_prices = {
            "AAPL": 220.0,
            "MSFT": 400.0
        }
        
        new_allocations = PortfolioResponse(
            reasoning="Rebalance to 50% AAPL, 30% MSFT, rest cash.",
            allocations=[
                StockAllocation(ticker="AAPL", percentage=50.0),
                StockAllocation(ticker="MSFT", percentage=30.0)
            ]
        )
        
        val, cash, holdings, warnings = execute_rebalance("test-model", prev_state, new_allocations, open_prices)
        
        # 1. Total starting portfolio value = $10,500
        # 2. AAPL target: $10,500 * 50% = $5,250
        #    AAPL shares = 5,250 / 220.0 = 23.863636 shares
        # 3. MSFT target: $10,500 * 30% = $3,150
        #    MSFT shares = 3,150 / 400.0 = 7.875 shares
        # 4. Total stock value = 23.863636 * 220 + 7.875 * 400 = 5249.99992 + 3150 = $8,399.99992
        # 5. Remaining Cash = 10,500 - 8,399.99992 = $2,100.00008
        
        self.assertAlmostEqual(val, 10500.0, places=2)
        self.assertAlmostEqual(holdings["AAPL"]["shares"], 23.863636, places=6)
        self.assertEqual(holdings["MSFT"]["shares"], 7.875)
        self.assertAlmostEqual(cash, 2100.0, places=2)
        self.assertEqual(len(warnings), 0)

    @patch('llmimt_main.is_ticker_valid')
    def test_rebalancing_with_invalid_ticker(self, mock_valid):
        """Test that invalid tickers proposed by the model are skipped and converted to cash."""
        mock_valid.side_effect = lambda t: t in ["AAPL"]
        
        prev_state = {
            "portfolio_value": 10000.0,
            "cash_balance": 10000.0,
            "holdings": {}
        }
        
        open_prices = {
            "AAPL": 200.0,
            "XYZ": 50.0  # mock price, but XYZ is invalid
        }
        
        new_allocations = PortfolioResponse(
            reasoning="Try to buy an invalid ticker.",
            allocations=[
                StockAllocation(ticker="AAPL", percentage=50.0),
                StockAllocation(ticker="XYZ", percentage=30.0)  # Invalid! Should be ignored
            ]
        )
        
        val, cash, holdings, warnings = execute_rebalance("test-model", prev_state, new_allocations, open_prices)
        
        # Starting value: $10,000
        # AAPL target: 50% = $5,000 -> 25 shares
        # XYZ invalid -> Skipped. That 30% ($3000) stays in Cash.
        # Total Cash = $10,000 - $5,000 (AAPL) = $5,000.
        
        self.assertAlmostEqual(val, 10000.0, places=2)
        self.assertEqual(holdings["AAPL"]["shares"], 25.0)
        self.assertNotIn("XYZ", holdings)
        self.assertAlmostEqual(cash, 5000.0, places=2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Invalid/Delisted ticker 'XYZ' skipped.", warnings[0])

    def test_load_state_uses_latest_row_even_if_fallback(self):
        """A fallback row is the latest valuation, so it must be the starting point for returns."""
        csv = (
            "date,model_id,model_name,portfolio_value,daily_return_pct,cash_balance,holdings_json,model_reasoning,execution_status\n"
            "2026-07-24,m,x/m,10000.0,0.0,0.0,\"{\"\"AAPL\"\": {\"\"shares\"\": 10.0, \"\"price\"\": 1000.0}}\",ok,SUCCESS\n"
            "2026-10-30,m,x/m,12000.0,20.0,0.0,\"{\"\"AAPL\"\": {\"\"shares\"\": 10.0, \"\"price\"\": 1200.0}}\",fallback,FAILED_AUTH\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(csv)
            state = load_portfolio_state(path, ["m", "new-model"])

        self.assertEqual(state["m"]["last_date"], "2026-10-30")
        self.assertEqual(state["m"]["portfolio_value"], 12000.0)
        # Unseen models start fresh with $10k cash
        self.assertEqual(state["new-model"]["cash_balance"], 10000.0)

    @patch('llmimt_main.litellm.completion')
    def test_invalid_key_bad_request_is_auth_failure_without_retry(self, mock_completion):
        """xAI reports bad keys as a 400; that should be FAILED_AUTH, not a schema retry loop."""
        mock_completion.side_effect = litellm.exceptions.BadRequestError(
            message='XaiException - {"error":"Incorrect API key provided."}',
            model="grok-4.5",
            llm_provider="xai",
        )
        with self.assertRaises(APIAuthenticationError):
            query_model("xai/grok-4.5", "bad-key", "sys", "user")
        self.assertEqual(mock_completion.call_count, 1)

    @patch('llmimt_main.litellm.completion')
    def test_unknown_model_fails_fast(self, mock_completion):
        """A non-auth bad request (e.g. unknown model) is raised immediately, not retried."""
        mock_completion.side_effect = litellm.exceptions.NotFoundError(
            message="model not found", model="x/nope", llm_provider="openai"
        )
        with self.assertRaises(litellm.exceptions.NotFoundError):
            query_model("openai/nope", "key", "sys", "user")
        self.assertEqual(mock_completion.call_count, 1)

if __name__ == '__main__':
    unittest.main()
