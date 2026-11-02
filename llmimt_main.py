import os
import sys
import json
import argparse
import datetime
import logging
import traceback
import time
from typing import List, Dict, Any, Tuple
from zoneinfo import ZoneInfo

# Load packages (will be imported inside main/helpers to avoid start-up import failures if virtualenv not ready yet, 
# but we can list them here for clarity)
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import litellm
from pydantic import BaseModel, Field, field_validator, ValidationError
from dotenv import load_dotenv

# Define America/New_York timezone as market standard
NY_TZ = ZoneInfo("America/New_York")

# Model slots: (ledger label, model env var, default litellm ID, API key env var)
# Labels are stable so upgrading a model ID continues the same portfolio.
# A blank default means the slot is skipped until the env var is set.
MODEL_SLOTS = [
    ("anthropic-large", "MODEL_ANTHROPIC_LARGE", "anthropic/claude-opus-5-5", "ANTHROPIC_API_KEY"),
    ("anthropic-small", "MODEL_ANTHROPIC_SMALL", "anthropic/claude-haiku-5-5", "ANTHROPIC_API_KEY"),
    ("gemini-large", "MODEL_GEMINI_LARGE", "", "GEMINI_API_KEY"),
    ("gemini-small", "MODEL_GEMINI_SMALL", "gemini/gemini-3.5-flash", "GEMINI_API_KEY"),
    ("openai-large", "MODEL_OPENAI_LARGE", "openai/gpt-5.6", "OPENAI_API_KEY"),
    ("openai-small", "MODEL_OPENAI_SMALL", "", "OPENAI_API_KEY"),
    ("xai-large", "MODEL_XAI_LARGE", "xai/grok-4.5", "XAI_API_KEY"),
    ("xai-small", "MODEL_XAI_SMALL", "", "XAI_API_KEY"),
]

# Custom Exceptions
class MarketClosedError(Exception):
    """Raised when the script is run on a day when the NYSE/NASDAQ is closed."""
    pass

class DataNotYetAvailableError(Exception):
    """Raised when yfinance does not yet have open price data for today."""
    def __init__(self, ticker: str, date_str: str, timestamp: str):
        self.ticker = ticker
        self.date_str = date_str
        self.timestamp = timestamp
        super().__init__(f"[DATA_UNAVAILABLE] Market open data for {ticker} is not yet published by yfinance at {timestamp} on {date_str}. Flagging pending_retry.")

class APIAuthenticationError(Exception):
    """Raised when authentication with an LLM provider fails due to missing or invalid keys."""
    pass

class ModelResponseTimeoutError(Exception):
    """Raised when an LLM call times out after 60 seconds."""
    pass

class InvalidJSONSchemaError(Exception):
    """Raised when an LLM response fails validation or is malformed after all retries."""
    pass

class PortfolioAllocationError(Exception):
    """Raised when a model proposes an invalid portfolio (e.g. >100% or invalid tickers)."""
    pass

class DiskWriteError(Exception):
    """Raised when unable to save ledger or state files."""
    pass

# Pydantic Schemas
class StockAllocation(BaseModel):
    ticker: str = Field(..., description="Ticker symbol in uppercase (e.g., 'AAPL', 'NVDA', 'CASH')")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total portfolio assigned (0 to 100)")

class PortfolioResponse(BaseModel):
    reasoning: str = Field(..., description="Concise explanation (1-3 sentences) for the allocation decision.")
    allocations: List[StockAllocation] = Field(..., max_length=5, description="List of up to 5 stock allocations.")


    @field_validator('allocations', mode='before')
    @classmethod
    def preprocess_allocations(cls, value):
        if not isinstance(value, list):
            return value
        
        # Preprocess each item to make it clean
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                ticker = str(item.get("ticker", "")).upper().strip()
                percentage = item.get("percentage")
                # Normalize CASH tickers to CASH
                if ticker in ["CASH", "USD", "CURRENCY"]:
                    ticker = "CASH"
                cleaned.append({"ticker": ticker, "percentage": percentage})
            else:
                cleaned.append(item)
        return cleaned

    @field_validator('allocations')
    @classmethod
    def validate_total_percentage(cls, allocations):
        # Consolidate duplicate tickers
        seen = {}
        for item in allocations:
            ticker = item.ticker
            seen[ticker] = seen.get(ticker, 0.0) + item.percentage
            
        combined = []
        for ticker, pct in seen.items():
            combined.append(StockAllocation(ticker=ticker, percentage=pct))
            
        total = sum(item.percentage for item in combined)
        if total > 100.0 + 1e-5:
            raise ValueError(f"Total allocation percentage ({total}%) exceeds 100%.")
            
        return combined


# Setup Logging
def setup_logging():
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("llmimt.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Save critical error log if storage fails
def log_critical_failure(message):
    try:
        with open("LLMIMT_CRITICAL.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {message}\n")
    except Exception:
        pass

# Check if ticker is valid by querying past history
def is_ticker_valid(ticker: str) -> bool:
    if ticker == "CASH":
        return True
    try:
        t = yf.Ticker(ticker)
        # Fetching 5 days is fast and checks if yfinance recognizes the ticker
        df = t.history(period="5d")
        return not df.empty
    except Exception:
        return False

# Fetch open price for today (or a specific date)
def get_open_price(ticker: str, run_date_str: str) -> float:
    if ticker == "CASH":
        return 1.0
        
    run_date_dt = datetime.datetime.strptime(run_date_str, "%Y-%m-%d")
    next_date_str = (run_date_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        t = yf.Ticker(ticker)
        # Fetch data for this specific day
        df = t.history(start=run_date_str, end=next_date_str)
        if df.empty:
            raise DataNotYetAvailableError(ticker, run_date_str, datetime.datetime.now(NY_TZ).strftime("%H:%M:%S"))
        
        # Verify index matches the date
        last_idx = df.index[-1]
        if hasattr(last_idx, "tz_convert"):
            last_date_str = last_idx.tz_convert(NY_TZ).strftime("%Y-%m-%d")
        else:
            last_date_str = last_idx.strftime("%Y-%m-%d")
            
        if last_date_str != run_date_str:
            raise DataNotYetAvailableError(ticker, run_date_str, datetime.datetime.now(NY_TZ).strftime("%H:%M:%S"))
            
        open_val = df['Open'].iloc[-1]
        if pd.isna(open_val) or open_val <= 0:
            raise DataNotYetAvailableError(ticker, run_date_str, datetime.datetime.now(NY_TZ).strftime("%H:%M:%S"))
            
        return float(open_val)
    except Exception as e:
        if isinstance(e, DataNotYetAvailableError):
            raise e
        raise DataNotYetAvailableError(ticker, run_date_str, datetime.datetime.now(NY_TZ).strftime("%H:%M:%S"))

# Fetch market headlines
def fetch_market_news() -> str:
    logging.info("Fetching market headlines from yfinance...")
    try:
        # Use SPY as benchmark for news
        spy = yf.Ticker("SPY")
        news_items = spy.news
        if not news_items:
            # Fallback to ^GSPC
            gspc = yf.Ticker("^GSPC")
            news_items = gspc.news
            
        news_str = ""
        if news_items:
            for item in news_items[:10]:  # limit to top 10
                title = item.get("title", "No Title")
                publisher = item.get("publisher", "Unknown Publisher")
                news_str += f"- {title} (Publisher: {publisher})\n"
        if news_str:
            return news_str
    except Exception as e:
        logging.warning(f"Failed to fetch market news: {e}")
    return "No recent market headlines available."

# Load previous state from CSV
def load_portfolio_state(csv_path: str, models: List[str]) -> Dict[str, Dict[str, Any]]:
    # Default initial state
    state = {}
    for m in models:
        state[m] = {
            "portfolio_value": 10000.0,
            "cash_balance": 10000.0,
            "holdings": {},  # Format: {ticker: {"shares": float, "price": float}}
            "last_date": None
        }
        
    if not os.path.exists(csv_path):
        logging.info(f"Ledger file {csv_path} does not exist. Initializing all models with $10,000 CASH.")
        return state
        
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.info(f"Ledger file {csv_path} is empty. Initializing all models with $10,000 CASH.")
            return state
            
        # Use each model's latest row regardless of status.
        # Fallback rows hold the retained holdings valued at that day's open,
        # so skipping them would measure "daily" returns from a stale row.
        for m in models:
            m_rows = df[df['model_id'] == m]
            if not m_rows.empty:
                last_row = m_rows.iloc[-1]
                try:
                    holdings = json.loads(last_row['holdings_json'])
                except Exception:
                    holdings = {}
                state[m] = {
                    "portfolio_value": float(last_row['portfolio_value']),
                    "cash_balance": float(last_row['cash_balance']),
                    "holdings": holdings,
                    "last_date": last_row['date']
                }
                logging.info(f"Loaded portfolio for {m} from {last_row['date']}: Value = ${state[m]['portfolio_value']:.2f}")
    except Exception as e:
        logging.error(f"Error loading state from {csv_path}: {e}")
        
    return state

# Write to last_run.json
def write_last_run(date_str: str, status: str):
    try:
        with open("last_run.json", "w", encoding="utf-8") as f:
            json.dump({"last_run_date": date_str, "status": status}, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to write last_run.json: {e}")
        log_critical_failure(f"Failed to write last_run.json: {e}")

# Write to pending_retry.json
def write_pending_retry(date_str: str):
    retry_count = 0
    if os.path.exists("pending_retry.json"):
        try:
            with open("pending_retry.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == date_str:
                    retry_count = data.get("retry_count", 0)
        except Exception:
            pass
            
    try:
        with open("pending_retry.json", "w", encoding="utf-8") as f:
            json.dump({
                "date": date_str,
                "retry_count": retry_count + 1,
                "last_attempt": datetime.datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to write pending_retry.json: {e}")

def clean_pending_retry():
    if os.path.exists("pending_retry.json"):
        try:
            os.remove("pending_retry.json")
        except Exception as e:
            logging.warning(f"Could not remove pending_retry.json: {e}")

# Call LLM with validation retry
def query_model(model_id: str, api_key: str, system_prompt: str, user_prompt: str, is_dry_run: bool = False) -> Tuple[PortfolioResponse, str]:
    if is_dry_run:
        # Mock responses for dry run
        logging.info(f"[DRY RUN] Simulating response for {model_id}")
        mock_response = PortfolioResponse(
            reasoning="Simulated rebalancing to capitalize on mock market indicators.",
            allocations=[
                StockAllocation(ticker="AAPL", percentage=40.0),
                StockAllocation(ticker="MSFT", percentage=40.0),
                StockAllocation(ticker="CASH", percentage=20.0)
            ]
        )
        return mock_response, "SUCCESS"

    # Define provider based on model prefix
    # litellm expects api keys to be in the environment, which they are.
    # Set model key dynamically if it is model-specific
    provider = model_id.split('/')[0]
    logging.info(f"Querying model {model_id} via litellm...")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    attempts = 3
    for attempt in range(attempts):
        content = ""
        try:
            # We request a json_object response format
            response = litellm.completion(
                model=model_id,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=60
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty content from model.")
                
            logging.info(f"Model {model_id} response content:\n{content}")
            
            # Parse and validate with Pydantic
            data = json.loads(content)
            portfolio = PortfolioResponse(**data)
            return portfolio, "SUCCESS"
            
        except litellm.exceptions.AuthenticationError as auth_err:
            logging.error(f"[AUTH_FAILURE] Failed to authenticate with provider {provider}: Missing or invalid API key for {model_id}. Error: {auth_err}")
            raise APIAuthenticationError(f"Authentication failed for {model_id}: {auth_err}")

        except (litellm.exceptions.BadRequestError, litellm.exceptions.NotFoundError) as req_err:
            # Retrying won't fix a bad request or unknown model, so fail fast.
            # Some providers (e.g. xAI) report invalid keys as a 400, not a 401.
            if "api key" in str(req_err).lower():
                logging.error(f"[AUTH_FAILURE] Failed to authenticate with provider {provider}: Missing or invalid API key for {model_id}. Error: {req_err}")
                raise APIAuthenticationError(f"Authentication failed for {model_id}: {req_err}")
            logging.error(f"[API_ERROR] Request to {model_id} was rejected: {req_err}")
            raise

        except Exception as e:
            err_msg = str(e)
            logging.warning(f"Attempt {attempt + 1} failed for {model_id}: {err_msg}")
            
            if attempt < attempts - 1:
                # Add validation failure context to messages for repair
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Your previous response failed validation. Error: {err_msg}. Please correct the issues, output only valid JSON matching the schema."})
                # Short wait before retry
                time.sleep(2)
            else:
                logging.error(f"[SCHEMA_MISMATCH] Model {model_id} returned invalid JSON structure. Retries exhausted.")
                raise InvalidJSONSchemaError(model_id, content, err_msg)
                
    raise InvalidJSONSchemaError(model_id, "", "Retries exhausted")

# Rebalance portfolio
def execute_rebalance(
    model_id: str,
    prev_state: Dict[str, Any],
    new_response: PortfolioResponse,
    open_prices: Dict[str, float]
) -> Tuple[float, float, Dict[str, Any], List[str]]:
    
    # Yesterday's shares and cash
    prev_holdings = prev_state.get("holdings", {})
    prev_cash = prev_state.get("cash_balance", 10000.0)
    
    # 1. Calculate today's initial portfolio value (valuing previous shares at today's open price)
    shares_value = 0.0
    for ticker, hold_info in prev_holdings.items():
        shares_qty = hold_info["shares"]
        # Get today's open price
        open_price = open_prices.get(ticker, 0.0)
        shares_value += shares_qty * open_price
        
    portfolio_start_value = shares_value + prev_cash
    logging.info(f"[{model_id}] Starting Portfolio Value at today's open: ${portfolio_start_value:.2f} (Shares: ${shares_value:.2f}, Cash: ${prev_cash:.2f})")
    
    # 2. Process model's allocations
    allocations = new_response.allocations
    warnings = []
    
    # Filter out invalid tickers and normalize
    valid_allocations = []
    total_pct_assigned = 0.0
    
    for alloc in allocations:
        ticker = alloc.ticker
        pct = alloc.percentage
        
        if ticker == "CASH":
            # Cash is handled at the end
            continue
            
        # Validate ticker in yfinance
        if not is_ticker_valid(ticker):
            warnings.append(f"Invalid/Delisted ticker '{ticker}' skipped.")
            logging.warning(f"[{model_id}] proposed invalid/delisted ticker '{ticker}'. Skipping and treating as CASH.")
            continue
            
        valid_allocations.append((ticker, pct))
        total_pct_assigned += pct
        
    if total_pct_assigned > 100.0 + 1e-5:
        # Pro-rate percentages down to 100% total
        warnings.append(f"Total percentage {total_pct_assigned}% exceeded 100%. Normalized.")
        logging.warning(f"[{model_id}] proposed total allocations of {total_pct_assigned}%. Normalizing down to 100%.")
        scale = 100.0 / total_pct_assigned
        valid_allocations = [(t, p * scale) for t, p in valid_allocations]
        total_pct_assigned = 100.0
        
    # 3. Execute Buys/Sells
    new_holdings = {}
    total_stock_value = 0.0
    
    for ticker, pct in valid_allocations:
        if pct <= 0:
            continue
        # Dollar amount to invest in this stock
        target_val = portfolio_start_value * (pct / 100.0)
        open_price = open_prices.get(ticker)
        
        if open_price is None or open_price <= 0:
            # Ticker price missing, treat as cash
            warnings.append(f"Price for '{ticker}' unavailable. Allocated to CASH.")
            logging.warning(f"[{model_id}] Price for '{ticker}' was not available. Treating allocation as CASH.")
            continue
            
        shares_qty = round(target_val / open_price, 6)
        actual_val = shares_qty * open_price
        new_holdings[ticker] = {
            "shares": shares_qty,
            "price": open_price
        }
        total_stock_value += actual_val
        logging.info(f"[{model_id}] Allocation: {ticker} ({pct:.2f}%) -> invested ${target_val:.2f} @ ${open_price:.2f}/share -> {shares_qty:.6f} shares (Value: ${actual_val:.2f})")
        
    # 4. Cash is remainder
    new_cash = max(0.0, portfolio_start_value - total_stock_value)
    final_portfolio_value = total_stock_value + new_cash
    
    logging.info(f"[{model_id}] Final Portfolio Value: ${final_portfolio_value:.2f} (Stocks: ${total_stock_value:.2f}, Cash: ${new_cash:.2f})")
    
    return final_portfolio_value, new_cash, new_holdings, warnings

def main():
    setup_logging()
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="LLM-Integrated Market Tool (LLMIMT)")
    parser.add_argument("--date", type=str, help="Run date in YYYY-MM-DD format (defaults to today)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate model calls and avoid real API queries")
    args = parser.parse_args()
    
    # 1. Resolve model slots & API keys (empty strings count as unset)
    model_ids = {}
    model_keys = {}
    for label, model_env, default_model, key_env in MODEL_SLOTS:
        model_name = os.getenv(model_env) or default_model
        if not model_name:
            logging.warning(f"[SKIPPED_NO_MODEL] {model_env} is not set. Skipping slot {label}.")
            continue
        model_ids[label] = model_name
        model_keys[label] = os.getenv(key_env) or None

    # If dry run, we assume mock keys are present
    if args.dry_run:
        logging.info("[DRY RUN] Dry run enabled. Mocking API keys and model completions.")
        for m in model_keys:
            model_keys[m] = "MOCK_KEY"
            
    # Check if we have ANY keys configured
    configured_models = [m for m, key in model_keys.items() if key is not None]
    if not configured_models:
        logging.info("[INFO] No API keys configured (ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, XAI_API_KEY are all missing or empty). Skipping execution cleanly.")
        sys.exit(0)
        
    # Get target date
    if args.date:
        run_date_str = args.date
        logging.info(f"Target run date specified: {run_date_str}")
    else:
        # Get today's date in New York
        run_date_str = datetime.datetime.now(NY_TZ).strftime("%Y-%m-%d")
        logging.info(f"Defaulting to current New York date: {run_date_str}")
        
    # 2. Calendar Check
    logging.info("Checking NYSE market schedule...")
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=run_date_str, end_date=run_date_str)
        if schedule.empty:
            logging.info(f"[MARKET_CLOSED] Today ({run_date_str}) is not an active trading day on NYSE/NASDAQ. Skipping execution.")
            sys.exit(0)
    except Exception as e:
        logging.error(f"Error checking market schedule: {e}")
        # If schedule check fails, we proceed but log a warning
        logging.warning("Market calendar lookup failed. Proceeding with caution.")

    # 3. Idempotency Check
    if not args.date: # Only check idempotency if we are running for "today"
        if os.path.exists("last_run.json"):
            try:
                with open("last_run.json", "r", encoding="utf-8") as f:
                    last_run = json.load(f)
                    if last_run.get("last_run_date") == run_date_str and last_run.get("status") == "SUCCESS":
                        logging.info(f"[INFO] Execution already completed for today ({run_date_str}). Exiting.")
                        sys.exit(0)
            except Exception as e:
                logging.warning(f"Error reading last_run.json: {e}")

    # 4. Load prior state
    csv_path = "llmimt_history.csv"
    prev_portfolios = load_portfolio_state(csv_path, list(model_ids.keys()))

    # 5. Market Data Integrity & Availability Check
    # We must check today's Open price for SPY (benchmark) and all tickers currently in holdings of ALL models.
    tickers_to_check = {"SPY"}
    for m in model_ids:
        holdings = prev_portfolios[m].get("holdings", {})
        for ticker in holdings.keys():
            if ticker != "CASH":
                tickers_to_check.add(ticker)
                
    logging.info(f"Verifying market open data availability for: {sorted(list(tickers_to_check))}")
    
    today_open_prices = {}
    data_missing = []
    
    for ticker in tickers_to_check:
        try:
            price = get_open_price(ticker, run_date_str)
            today_open_prices[ticker] = price
            logging.info(f"Open price for {ticker} on {run_date_str}: ${price:.2f}")
        except DataNotYetAvailableError as dna:
            logging.warning(str(dna))
            data_missing.append(ticker)
        except Exception as e:
            logging.error(f"Unexpected error fetching price for {ticker}: {e}")
            data_missing.append(ticker)
            
    if data_missing:
        # Today's data is not yet fully available
        logging.warning(f"[DATA_UNAVAILABLE] Open price missing for tickers: {data_missing}. Setting retry state.")
        write_pending_retry(run_date_str)
        # Exit code 61 instructs the scheduler to retry later
        sys.exit(61)

    # 6. Gather News Context
    news_headlines = fetch_market_news()

    # 7. Prompt Models & Rebalance
    system_prompt = (
        "You are a professional quantitative financial manager running an automated trading trial.\n"
        "Your goal is to maximize the return of a $10,000 virtual paper portfolio over time.\n"
        "You must respond ONLY with a JSON object that adheres strictly to the following structure:\n"
        "{\n"
        "  \"reasoning\": \"A concise 1-3 sentence description of your decision basis.\",\n"
        "  \"allocations\": [\n"
        "    {\"ticker\": \"TICKER_SYMBOL\", \"percentage\": float},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "Allocations Constraints:\n"
        "1. Up to 5 stock ticker symbols (e.g. 'AAPL', 'MSFT', 'NVDA').\n"
        "2. If you want to keep cash, either add \"CASH\" ticker with a percentage, or make the sum of stock percentages less than 100. Any unallocated percentage will automatically remain as uninvested CASH.\n"
        "3. Ticker symbols must be valid US equities traded on NYSE/NASDAQ.\n"
        "4. Total percentages of stock + cash allocations must equal exactly 100.0% (or less, with remainder going to cash).\n"
        "5. Valid values for percentage are 0.0 to 100.0.\n"
        "6. Do not include any formatting, text, or markdown outside the JSON object itself."
    )
    
    new_records = []
    
    for model_key_id, model_name in model_ids.items():
        logging.info(f"--- Processing {model_key_id} ({model_name}) ---")
        
        # Get prior portfolio state
        p_state = prev_portfolios[model_key_id]
        prev_value = p_state["portfolio_value"]
        prev_holdings = p_state["holdings"]
        prev_cash = p_state["cash_balance"]
        
        # Calculate today's initial portfolio value (valuing old shares at today's open price)
        start_value_shares = 0.0
        for ticker, hold_info in prev_holdings.items():
            shares_qty = hold_info["shares"]
            open_price = today_open_prices.get(ticker, 0.0)
            start_value_shares += shares_qty * open_price
            
        today_starting_value = start_value_shares + prev_cash
        
        # Calculate daily return percentage since last run
        daily_return_pct = 0.0
        if prev_value > 0:
            daily_return_pct = ((today_starting_value - prev_value) / prev_value) * 100.0
            
        # Check if API key is configured
        api_key = model_keys[model_key_id]
        if api_key is None:
            logging.info(f"[SKIPPED_NO_KEY] API key for {model_key_id} is missing or blank. Retaining previous portfolio.")
            # Fallback execution (no change in shares, just valued at today's open)
            # Rebalance with current allocations to update prices and value
            mock_resp = PortfolioResponse(
                reasoning="Bypassed model call due to missing API key. Retained prior holdings.",
                allocations=[StockAllocation(ticker=t, percentage=0.0) for t in prev_holdings.keys()]
            )
            # Set allocations to exactly match prior holdings percentage
            mock_allocs = []
            for t, h in prev_holdings.items():
                pct = (h["shares"] * today_open_prices.get(t, 0.0) / today_starting_value) * 100.0
                mock_allocs.append(StockAllocation(ticker=t, percentage=pct))
            mock_resp.allocations = mock_allocs
            
            # Value at today's open
            # We must populate today_open_prices for these tickers (we already verified they are ready)
            final_val, final_cash, final_holdings, warnings = execute_rebalance(
                model_key_id, p_state, mock_resp, today_open_prices
            )
            
            new_records.append({
                "date": run_date_str,
                "model_id": model_key_id,
                "model_name": model_name,
                "portfolio_value": round(final_val, 2),
                "daily_return_pct": round(daily_return_pct, 4),
                "cash_balance": round(final_cash, 2),
                "holdings_json": json.dumps(final_holdings),
                "model_reasoning": "API key not configured. Prior holdings retained.",
                "execution_status": "SKIPPED_NO_KEY"
            })
            continue
            
        # Construct User Prompt
        holdings_desc = ""
        if prev_holdings:
            holdings_desc = "\n".join([
                f"- {ticker}: {h_info['shares']:.6f} shares (current open value: ${h_info['shares']*today_open_prices.get(ticker, 0.0):.2f})" 
                for ticker, h_info in prev_holdings.items()
            ])
        else:
            holdings_desc = "- (None, currently 100% CASH)"
            
        user_prompt = (
            f"Today's Date: {run_date_str}\n"
            f"Benchmark Index News Headlines:\n{news_headlines}\n\n"
            f"Your Current Portfolio Status:\n"
            f"Total Portfolio Value: ${today_starting_value:.2f}\n"
            f"Cash Balance: ${prev_cash:.2f}\n"
            f"Current Holdings:\n{holdings_desc}\n\n"
            "Given the recent headlines and data, decide whether to rebalance or keep your current allocations.\n"
            "Submit your portfolio distribution using up to 5 tickers. Remember: any unallocated percentage goes to CASH.\n"
            "Format your output strictly as a JSON object with keys 'reasoning' and 'allocations'."
        )
        
        portfolio_resp = None
        status_code = "SUCCESS"
        reasoning_str = ""
        
        try:
            # Run model completion
            portfolio_resp, status_code = query_model(
                model_name, api_key, system_prompt, user_prompt, is_dry_run=args.dry_run
            )
            reasoning_str = portfolio_resp.reasoning
        except APIAuthenticationError as ae:
            logging.error(f"Authentication failure for {model_key_id}: {ae}")
            status_code = "FAILED_AUTH"
            reasoning_str = f"Authentication failed with API provider: {ae}"
        except InvalidJSONSchemaError as je:
            logging.error(f"JSON validation failure for {model_key_id}: {je}")
            status_code = "FAILED_SCHEMA"
            # args = (model_id, raw_content, error_message)
            reasoning_str = f"Model failed to return valid JSON schema after retries: {je.args[-1]}"
        except Exception as e:
            logging.error(f"Unexpected query error for {model_key_id}: {e}")
            status_code = "FAILED_API"
            reasoning_str = f"API query failed: {str(e)}"
            
        # 8. Rebalancing
        if status_code == "SUCCESS" and portfolio_resp is not None:
            # We need today's open prices for any new tickers chosen by the model
            tickers_needed = [alloc.ticker for alloc in portfolio_resp.allocations if alloc.ticker != "CASH"]
            
            # Fetch prices for new tickers if not already fetched
            new_ticker_prices_ready = True
            for t in tickers_needed:
                if t not in today_open_prices:
                    if not is_ticker_valid(t):
                        # Skip validation of this ticker, execute_rebalance will handle it
                        continue
                    try:
                        price = get_open_price(t, run_date_str)
                        today_open_prices[t] = price
                        logging.info(f"Open price for new ticker {t} on {run_date_str}: ${price:.2f}")
                    except Exception as e:
                        logging.error(f"Could not get open price for new ticker {t}: {e}")
                        new_ticker_prices_ready = False
                        
            if not new_ticker_prices_ready:
                logging.warning(f"[{model_key_id}] Chosen ticker prices were not ready in yfinance. Postponing rebalance to retry.")
                write_pending_retry(run_date_str)
                sys.exit(61)
                
            # Perform rebalance
            final_val, final_cash, final_holdings, warnings = execute_rebalance(
                model_key_id, p_state, portfolio_resp, today_open_prices
            )
            
            new_records.append({
                "date": run_date_str,
                "model_id": model_key_id,
                "model_name": model_name,
                "portfolio_value": round(final_val, 2),
                "daily_return_pct": round(daily_return_pct, 4),
                "cash_balance": round(final_cash, 2),
                "holdings_json": json.dumps(final_holdings),
                "model_reasoning": reasoning_str,
                "execution_status": "SUCCESS"
            })
        else:
            # Fallback: keep previous portfolio (shares are valued at today's open)
            logging.warning(f"[{model_key_id}] Running fallback to prior holdings due to status: {status_code}")
            
            mock_resp = PortfolioResponse(
                reasoning=f"Fallback. Prior holdings retained. Status: {status_code}",
                allocations=[]
            )
            # Reconstruct allocations to match previous holdings
            mock_allocs = []
            for t, h in prev_holdings.items():
                pct = (h["shares"] * today_open_prices.get(t, 0.0) / today_starting_value) * 100.0
                mock_allocs.append(StockAllocation(ticker=t, percentage=pct))
            mock_resp.allocations = mock_allocs
            
            final_val, final_cash, final_holdings, warnings = execute_rebalance(
                model_key_id, p_state, mock_resp, today_open_prices
            )
            
            new_records.append({
                "date": run_date_str,
                "model_id": model_key_id,
                "model_name": model_name,
                "portfolio_value": round(final_val, 2),
                "daily_return_pct": round(daily_return_pct, 4),
                "cash_balance": round(final_cash, 2),
                "holdings_json": json.dumps(final_holdings),
                "model_reasoning": f"Fallback applied: {reasoning_str}",
                "execution_status": status_code
            })

    # Dry runs must never pollute the real ledger or idempotency state
    if args.dry_run:
        logging.info("[DRY RUN] Skipping ledger and last_run.json writes. Records that would be written:")
        for rec in new_records:
            logging.info(f"[DRY RUN] {rec['model_id']} ({rec['model_name']}): ${rec['portfolio_value']:.2f}, "
                         f"return {rec['daily_return_pct']:.4f}%, status {rec['execution_status']}")
        return

    # 9. Commit to Ledger CSV
    logging.info("Writing execution records to ledger...")
    try:
        new_df = pd.DataFrame(new_records)
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            # Concatenate existing and new records
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
            
        combined_df.to_csv(csv_path, index=False)
        logging.info(f"Successfully appended {len(new_records)} records to {csv_path}")
    except Exception as e:
        logging.error(f"[STORAGE_FAILURE] Unable to write execution log to {csv_path}: {e}")
        log_critical_failure(f"Unable to write execution log to {csv_path}: {e}")
        sys.exit(1)

    # 10. Update last_run.json
    write_last_run(run_date_str, "SUCCESS")
    clean_pending_retry()
    logging.info("LLMIMT execution finished successfully.")

if __name__ == "__main__":
    main()
