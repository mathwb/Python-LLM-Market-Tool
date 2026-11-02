# LLM-Integrated Market Tool (LLMIMT) — Architecture & Design Document

**Document Version:** 1.0  
**Status:** Approved for Implementation  
**Target Platform:** Windows 10 / 11 (Python 3.10+)  
**Output Target:** System-automated daily execution & CSV history ledger  

---

## 1. Executive Summary & Core Objectives

The **LLM-Integrated Market Tool (LLMIMT)** is an automated quantitative testing framework designed to evaluate and compare the stock-picking and portfolio-management capabilities of state-of-the-art Large Language Models (LLMs) over time.

Each competing LLM manages an independent, paper-traded **$10,000 virtual portfolio**. Models are prompted daily with identical market data, pricing context, and top financial headlines, and must decide whether to retain their existing holdings or rebalance across up to **5 US equity tickers** (including holding uninvested cash).

### Key Competing Models
Each major provider runs a **large** and a **small** model (8 slots). Slot labels (the ledger `model_id`) are stable; the actual litellm model ID is set per slot in `.env` (`MODEL_<PROVIDER>_<LARGE|SMALL>`) and recorded in the ledger's `model_name` column.

| Slot (`model_id`) | Default model | Notes |
| :--- | :--- | :--- |
| `anthropic-large` | `anthropic/claude-opus-5-5` | |
| `anthropic-small` | `anthropic/claude-haiku-5-5` | |
| `gemini-large` | *(set in `.env`)* | Skipped until configured |
| `gemini-small` | `gemini/gemini-3.5-flash` | History since 2026-07-24 |
| `openai-large` | `openai/gpt-5.6` | History since 2026-07-24 (gpt-5.5 until 08-03) |
| `openai-small` | *(set in `.env`)* | Skipped until configured |
| `xai-large` | `xai/grok-4.5` | History since 2026-07-24 |
| `xai-small` | *(set in `.env`)* | Skipped until configured |

---

## 2. Technical Stack & Tooling

| Component / Function | Library / Tool | Purpose & Justification |
| :--- | :--- | :--- |
| **LLM Abstraction** | `litellm` | Unified API calling interface across Anthropic, Google, OpenAI, and xAI with standardized error handling and strict JSON schema support. |
| **Market Data & News** | `yfinance` | Retrieval of real-time/historical daily `Open` prices, ticker verification, and ticker news headlines. |
| **Market Schedule** | `pandas_market_calendars` | Precise checking of NYSE/NASDAQ trading days, early closes, and market holidays to prevent invalid executions. |
| **Data Enforcement** | `pydantic` | Schema definition and strict JSON validation for model portfolio response payloads. |
| **Data Manipulation** | `pandas` | Financial calculations, portfolio valuation, return calculations, and CSV ledger updates. |
| **Environment Management** | `python-dotenv` | Secure loading of local API keys from `.env` files without hardcoding credentials. |
| **Process Orchestration** | GitHub Actions (scheduled workflow) | Cloud cron runs the daily job without a local machine; commits the updated ledger back to the repo. |

---

## 3. High-Level System Architecture & Execution Flow

```text
[ GitHub Actions schedule (weekdays) ]
│
▼
┌───────────────────────────────┐
│ 1. State & Calendar Check     │ ◄── pandas_market_calendars & last_run.json
└───────────────┬───────────────┘
│ (Is today a trading day & not yet executed?)
▼
┌───────────────────────────────┐
│ 2. Market Data Readiness      │ ◄── yfinance (Check today's Open price availability)
└───────────────┬───────────────┘
│ (Data Ready?)
├─── NO ──► [Write state: pending_retry.json & Exit cleanly]
▼ YES
┌───────────────────────────────┐
│ 3. Market News & Context Feed │ ◄── Fetch headlines & current holdings context
└───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│ 4. Model Prompting            │ ◄── litellm (large + small model per provider, 8 slots)
└───────────────┬───────────────┘
│ (Strict JSON Pydantic Validation)
▼
┌───────────────────────────────┐
│ 5. Portfolio Accounting Engine│ ◄── Value shares at Market Open, calc returns & cash
└───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│ 6. CSV Ledger & State Commit  │ ◄── Update llmimt_history.csv & last_run.json
└───────────────────────────────┘
```

---

## 4. Operational Specifications

### 4.1. Trading Rules & Constraints
* **Starting Capital:** Exactly **$10,000.00 USD** per model on Day 1.
* **Allocation Limits:** Maximum of **5 distinct tickers** per portfolio at any time.
* **Cash Rule:** Unallocated percentage is automatically kept as uninvested cash (yields 0% nominal return).
* **Execution Price:** All buys, sells, and portfolio valuations occur at the official **Market Open (`Open`) price** of the execution day.
* **Fractional Shares:** Fully supported. Shares are tracked to 6 decimal places (e.g., `14.285714` shares of AAPL).

### 4.2. Automation (GitHub Actions)
* **Workflow:** `.github/workflows/daily.yml`. API keys are repo **Secrets**; `MODEL_*` IDs are repo **Variables** (unset → code default).
* **Trigger:** Weekdays at **15:00 UTC** (11:00 EDT / 10:00 EST) plus a backup run at **18:00 UTC**; `workflow_dispatch` allows manual runs.
* **Retry Loop:** If open prices aren't published yet, the script exits `61` and the backup run retries later.
* **Idempotency:** Before running full execution, the script checks `last_run.json`. If `last_run_date == today_str` and `status == "SUCCESS"`, the script terminates immediately with log message `[INFO] Execution already completed for today. Exiting.`.
* **Persistence:** The job commits `llmimt_history.csv` and `last_run.json` back to the repo and uploads `llmimt.log` as a run artifact.
* **Local runs:** `python llmimt_main.py --dry-run` is safe for testing. Avoid real local runs, or the local and GitHub ledgers will diverge.

---

## 5. Structured Data Schemas

### 5.1. Pydantic Output Schema (`PortfolioSchema`)

To guarantee clean response parsing across all providers via `litellm`:

```python
from pydantic import BaseModel, Field, field_validator
from typing import List

class StockAllocation(BaseModel):
    ticker: str = Field(..., description="Ticker symbol in uppercase (e.g., 'AAPL', 'NVDA', 'CASH')")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total portfolio assigned (0 to 100)")

class PortfolioResponse(BaseModel):
    reasoning: str = Field(..., description="Concise explanation (1-3 sentences) for the allocation decision.")
    allocations: List[StockAllocation] = Field(..., max_items=5, description="List of up to 5 stock allocations.")

    @field_validator('allocations')
    def validate_total_percentage(cls, allocations):
        total = sum(item.percentage for item in allocations)
        if total > 100.0 + 1e-5:
            raise ValueError(f"Total allocation percentage ({total}%) exceeds 100%.")
        return allocations
```

### 5.2. Daily CSV Ledger Schema (`llmimt_history.csv`)

| Column Name | Data Type | Example Value | Description |
| --- | --- | --- | --- |
| `date` | YYYY-MM-DD | `2026-07-22` | Execution date |
| `model_id` | String | `gemini-small` | Stable slot label (provider + size) |
| `model_name` | String | `gemini/gemini-3.5-flash` | Exact litellm model ID used for this row |
| `portfolio_value` | Float | `10450.25` | Total liquid portfolio value (Holdings + Cash) |
| `daily_return_pct` | Float | `1.25` | Percentage change in portfolio value since prior trading day |
| `cash_balance` | Float | `1500.00` | Uninvested cash reserve |
| `holdings_json` | JSON String | `{"AAPL": {"shares": 12.5, "price": 220.50}, "NVDA": {"shares": 25.0, "price": 125.00}}` | Serialized detailed positions |
| `model_reasoning` | String | `"Reallocated 15% from cash into NVDA due to earnings momentum."` | Explanation provided by LLM |
| `execution_status` | String | `SUCCESS` | Operational status of the run |

---

## 6. Comprehensive Error Handling & Error Messages

The system implements strict, explicit exception hierarchy to ensure visibility into API failures, data latency, and parsing errors.

### 6.1. System Error Catalog

| Exception Class | Trigger Condition | Specific Log & CLI Error Message | Handling Action |
| --- | --- | --- | --- |
| `MarketClosedError` | Script executed on weekend, holiday, or outside trading days. | `[MARKET_CLOSED] Today ({date}) is not an active trading day on NYSE/NASDAQ. Skipping execution.` | Exit code `0` (Graceful skip). |
| `DataNotYetAvailableError` | `yfinance` returns `NaN` or missing `Open` prices for target date post-market open. | `[DATA_UNAVAILABLE] Market open data for {ticker} is not yet published by yfinance at {timestamp}. Flagging pending_retry.` | Write `pending_retry.json`, exit code `61`; the backup scheduled run retries. |
| `APIAuthenticationError` | Invalid or missing API key in environment (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`). Includes providers that report bad keys as HTTP 400. | `[AUTH_FAILURE] Failed to authenticate with provider {provider}: Missing or invalid API key for {model_id}.` | Log error, flag model run as `FAILED_AUTH`, continue with remaining models. |
| `ModelResponseTimeoutError` | `litellm.completion` call exceeds timeout threshold (60s). | `[TIMEOUT_ERROR] Model {model_id} timed out after 60 seconds of unresponsiveness.` | Log timeout, retry up to 2 times, then fallback to retaining prior day's portfolio. |
| `InvalidJSONSchemaError` | Model response fails Pydantic schema validation or outputs malformed JSON. (Rejected requests / unknown models fail fast as `FAILED_API` instead.) | `[SCHEMA_MISMATCH] Model {model_id} returned invalid JSON structure: {raw_response}. Validation Error: {error_details}` | Trigger schema-repair prompt retry. Fallback to prior portfolio if 3 retries fail. |
| `PortfolioAllocationError` | Sum of model allocations exceeds 100% or contains invalid/delisted tickers. | `[ALLOCATION_INVALID] Model {model_id} proposed illegal portfolio: Allocation total {total_pct}% > 100% or invalid ticker '{ticker}'.` | Normalize percentage proportionally or discard invalid ticker; log warnings. |
| `DiskWriteError` | File system error or lock preventing write to `llmimt_history.csv`. | `[STORAGE_FAILURE] Unable to write execution log to {csv_path}: Permission denied or disk full.` | Log critical error to fallback log `LLMIMT_CRITICAL.log`. |

---

## 7. Step-by-Step Implementation Outline

1. **Environment Setup:**
   * Create Python virtual environment (`venv`).
   * Install requirements: `pip install litellm yfinance pandas pandas-market-calendars pydantic python-dotenv`.
   * Copy `.env.template` to `.env` and fill in `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, and any `MODEL_*` slots without defaults.

2. **Core Script (`llmimt_main.py`):**
   * **Module 1 (Scheduler & Calendar):** Verify market status using `pandas_market_calendars.get_calendar('NYSE')`.
   * **Module 2 (Market Data Engine):** Query `yfinance` for S&P 500 news and verify today's `Open` price availability across benchmark tickers.
   * **Module 3 (Prompt Builder & LiteLLM Interface):** Construct uniform prompt containing news, market trends, and current holdings. Issue parallel calls to `litellm.completion()` enforcing `response_format=PortfolioSchema`.
   * **Module 4 (Accounting & Execution):** Calculate share purchases/sales at today's `Open`, compute ending portfolio value and daily returns.
   * **Module 5 (Ledger Logger):** Append daily snapshots to `llmimt_history.csv` and write state update to `last_run.json`.

3. **Automation Deployment (GitHub Actions):**
   * Push the project to a GitHub repo (`.gitignore` excludes `.env` and `.venv/`).
   * Add the 4 API keys as Actions Secrets and the `MODEL_*` IDs as Actions Variables.
   * Trigger `Daily LLMIMT run` once via `workflow_dispatch` to confirm it commits the ledger.
