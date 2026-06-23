"""
This module contains two main functions for stock data preprocessing:
1. `generate_stock_selection()`: Fetches stock info from FinMind, filters by industry and liquidity, and saves the top-k stocks per industry to a JSON file.
2. `download_stock_data_to_db()`: Downloads daily stock data from TWSE and saves it to PostgreSQL. Supports both full download and incremental update modes.
Usage:
- To generate stock selection JSON:
    python data_preprocess.py select --top-k 10 --categories "半導體業" "電子工業"
- To download stock data (incremental mode):
    python data_preprocess.py download
- To download stock data (full mode):
    python data_preprocess.py download --start-date 20200101 --end-date 20201231
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader
from sqlalchemy import create_engine
from tqdm import tqdm

# Add workspace root to import twse stock API module.
workspace_root = Path("/app")
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from trade.stock_api import get_twse_stock_data
load_dotenv()

PG_URI = os.getenv("POSTGRES_URI")

log_path = Path("/app/logs/error.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
_fh = logging.FileHandler(log_path, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_fh)

OUTPUT_JSON_PATH = Path("/app/data/topk_stocks_by_industries.json")
TARGET_CATEGORIES = [
    "ETF", "水泥工業", "食品工業",
    "電器電纜", "農業科技業", "觀光餐旅", "塑膠工業", "建材營造",
    "汽車工業", "電子零組件業", "紡織纖維", "貿易百貨", "運動休閒",
    "電子工業", "電機機械", "生技醫療業", "電腦及週邊設備業", "化學工業",
    "其他電子業", "玻璃陶瓷", "造紙工業", "鋼鐵工業", "居家生活",
    "橡膠工業", "航運業", "半導體業", "通信網路業", "光電業",
    "電子通路業", "資訊服務業", "油電燃氣業", "數位雲端", "金融保險",
    "文化創意業", "綠能環保", "電子商務業",
]
INSPECT_CATEGORIES = [
    "ETF", "建材營造", "電子零組件業", "半導體業", "通信網路業",
]
DEFAULT_START_DATE = "20100101"
# =============================================================================
# Module 1: Generate stock selection JSON
# =============================================================================

def _fetch_finmind_stock_info() -> pd.DataFrame:
    api = DataLoader()
    return api.taiwan_stock_info_with_warrant()


def _filter_liquidity(
    trade_vol_share: int = 100_000,
    trade_value_ntd: int = 5_000_000,
) -> pd.DataFrame:
    try:
        month_stock_vol = pd.read_json(
            "https://openapi.twse.com.tw/v1/exchangeReport/FMSRFK_ALL"
        )
        month_stock_vol.columns = month_stock_vol.columns.str.lower()
        for col in ["tradevolumeb", "tradevaluea"]:
            month_stock_vol[col] = (
                month_stock_vol[col].astype(str).str.replace(",", "").astype(float)
            )
        mask = (
            (month_stock_vol["tradevolumeb"] >= trade_vol_share)
            & (month_stock_vol["tradevaluea"] >= trade_value_ntd)
        )
        return month_stock_vol[mask]
    except Exception as e:
        logger.error(f"Error fetching liquidity data: {e}")
        return pd.DataFrame()


def generate_stock_selection(
    inspect_categories: list[str] | None = None,
    top_k: int = 10,
    trade_vol_share: int = 100_000,
    trade_value_ntd: int = 5_000_000,
    output_path: Path = OUTPUT_JSON_PATH,
) -> dict:
    """
    Module 1: Filter Taiwan stocks by industry and liquidity, then select
    the top-k by trading value per industry. Saves result to JSON.

    Args:
        inspect_categories: Industries to include. Defaults to a preset list.
        top_k: Number of top stocks per industry.
        trade_vol_share: Minimum monthly trading volume (shares).
        trade_value_ntd: Minimum monthly trading value (NTD).
        output_path: Path to write the output JSON.

    Returns:
        Dict mapping industry -> list of stock codes.
    """
    if inspect_categories is None:
        inspect_categories = INSPECT_CATEGORIES

    print("Fetching stock info from FinMind...")
    df = _fetch_finmind_stock_info()

    # Filter by industry and market type
    filter_df = df[
        df["industry_category"].isin(TARGET_CATEGORIES)
        & (df["type"] == "twse")
    ]
    print(f"After industry filter: {filter_df.shape[0]} stocks")

    # Filter by liquidity
    print("Fetching monthly liquidity data from TWSE...")
    filtered_stocks = _filter_liquidity(trade_vol_share, trade_value_ntd)
    if filtered_stocks.empty:
        raise RuntimeError("Failed to fetch liquidity data.")
    print(f"After liquidity filter: {filtered_stocks.shape[0]} stocks")

    # Merge industry info into liquidity data
    mapping_df = df[["stock_id", "industry_category"]]
    merged = filtered_stocks.merge(
        mapping_df, left_on="code", right_on="stock_id", how="left"
    )
    for col in ["tradevolumeb", "tradevaluea"]:
        merged[col] = merged[col].astype(str).str.replace(",", "").astype(float)

    # Top-k per industry by trading value
    top_k_stocks = (
        merged.groupby("industry_category", group_keys=False)
        .apply(lambda x: x.nlargest(top_k, "tradevaluea"), include_groups=False)
    )

    # Build output dict for selected categories
    selected_stocks_dict = {
        category: merged[
            (merged["industry_category"] == category)
            & merged["code"].isin(top_k_stocks["code"])
        ]["code"].tolist()
        for category in inspect_categories
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_stocks_dict, f, ensure_ascii=False, indent=4)

    print(f"Saved stock selection to {output_path}")
    for cat, codes in selected_stocks_dict.items():
        print(f"  {cat}: {codes}")

    return selected_stocks_dict


# =============================================================================
# Module 2: Download stock daily data to PostgreSQL
# =============================================================================

def _get_latest_date(table_name: str, engine) -> pd.Timestamp | None:
    """Return MAX(date) for any table, or None if the table doesn't exist."""
    check_sql = f"""
        SELECT table_name FROM information_schema.tables
        WHERE table_name = '{table_name}'
    """
    with engine.connect() as conn:
        exists = pd.read_sql(check_sql, conn)
    if exists.empty:
        return None
    date_sql = f'SELECT MAX(date) AS max_date FROM "{table_name}"'
    with engine.connect() as conn:
        result = pd.read_sql(date_sql, conn)
    max_date = result["max_date"].iloc[0]
    return pd.Timestamp(max_date) if max_date is not None else None


def _get_latest_date_for_stock(code: str, engine) -> datetime | None:
    """Return the latest date in daily_info_{code} table, or None if not found."""
    return _get_latest_date(f"daily_info_{code}", engine)


def _save_stock_data(code: str, start_date: str, end_date: str, engine, already_exist: str):
    code_df = get_twse_stock_data(code, start_date, end_date)
    if code_df is None or code_df.empty:
        logger.warning(f"No data returned for {code} ({start_date} ~ {end_date})")
        return

    num_cols = ["capacity", "turnover", "high", "low", "close", "change", "transaction_volume", "open"]
    code_df[num_cols] = code_df[num_cols].apply(pd.to_numeric, errors="coerce")
    code_df["date"] = pd.to_datetime(code_df["date"], unit="s")
    code_df["stock_code_id"] = code_df["stock_code_id"].astype(str)
    code_df.to_sql(name=f"daily_info_{code}", con=engine, if_exists=already_exist, index=False)


# =============================================================================
# Module 3: Download institutional investor & margin purchase/short sale data
# =============================================================================

def _to_finmind_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD required by FinMind async API."""
    if len(date_str) == 8 and "-" not in date_str:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def _load_stock_codes(stock_codes: list[str] | None, json_path: Path) -> list[str]:
    """Load and deduplicate stock codes from explicit list or JSON file."""
    if stock_codes is not None:
        return stock_codes
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run generate_stock_selection() first."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        stock_dict = json.load(f)
    codes = [code for codes in stock_dict.values() for code in codes]
    return list(dict.fromkeys(codes))


FINMIND_CHUNK_SIZE = 20  # FinMind free-tier row limit caps out around 30 stocks × full range


def download_institutional_to_db(
    stock_codes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    json_path: Path = OUTPUT_JSON_PATH,
    chunk_size: int = FINMIND_CHUNK_SIZE,
):
    """
    Module 3a: Download institutional investors (三大法人) buy/sell data and
    store in PostgreSQL as pivoted wide-format tables (institutional_{code}).

    Incremental mode (start_date=None): checks MAX(date) per table, appends
    only new rows. Full mode (start_date provided): replaces existing table.

    Stocks are fetched in chunks of chunk_size to stay within FinMind's
    free-tier single-request row limit.
    """
    if end_date is None:
        end_date = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y%m%d")

    stock_codes = _load_stock_codes(stock_codes, json_path)
    incremental_mode = start_date is None
    engine = create_engine(PG_URI)
    fallback_start = DEFAULT_START_DATE

    print(f"[institutional] Mode: {'incremental' if incremental_mode else 'full'} | "
          f"End: {end_date} | Stocks: {len(stock_codes)}")

    if incremental_mode:
        next_starts: dict[str, str] = {}
        for code in stock_codes:
            latest = _get_latest_date(f"institutional_{code}", engine)
            if latest is None:
                next_starts[code] = fallback_start
            else:
                next_day = (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
                next_starts[code] = next_day
        codes_to_fetch = [c for c in stock_codes if next_starts[c] <= end_date]
    else:
        codes_to_fetch = stock_codes

    if not codes_to_fetch:
        print("[institutional] All stocks already up to date.")
        return

    api = DataLoader()
    chunks = [codes_to_fetch[i:i + chunk_size] for i in range(0, len(codes_to_fetch), chunk_size)]
    print(f"[institutional] Fetching {len(codes_to_fetch)} stocks in {len(chunks)} chunks...")

    for chunk in tqdm(chunks, desc="Fetching institutional chunks"):
        if incremental_mode:
            effective_start = min(next_starts[c] for c in chunk)
        else:
            effective_start = start_date

        df = api.taiwan_stock_institutional_investors(
            stock_id_list=chunk,
            start_date=_to_finmind_date(effective_start),
            end_date=_to_finmind_date(end_date),
            use_async=True,
        )
        if df is None or df.empty:
            logger.warning(f"[institutional] No data for chunk {chunk}")
            continue

        df_pivot = df.pivot_table(
            index=["date", "stock_id"],
            columns="name",
            values=["buy", "sell"],
            aggfunc="first",
        )
        df_pivot.columns = [f"{val}_{name}" for val, name in df_pivot.columns]
        df_pivot = df_pivot.reset_index()
        df_pivot["date"] = pd.to_datetime(df_pivot["date"])

        for code in chunk:
            try:
                sub = df_pivot[df_pivot["stock_id"] == code].copy()
                if sub.empty:
                    continue
                if incremental_mode:
                    if_exists = "replace" if next_starts[code] == fallback_start else "append"
                else:
                    if_exists = "replace"
                sub.to_sql(name=f"institutional_{code}", con=engine, if_exists=if_exists, index=False)
                logger.info(f"[institutional] {code}: {if_exists} ({len(sub)} rows)")
            except Exception as e:
                logger.error(f"[institutional] Error saving {code}: {e}")

    print("[institutional] Done.")


def download_margin_to_db(
    stock_codes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    json_path: Path = OUTPUT_JSON_PATH,
    chunk_size: int = FINMIND_CHUNK_SIZE,
):
    """
    Module 3b: Download margin purchase / short sale (融資融券) data and
    store in PostgreSQL as wide-format tables (margin_{code}).

    Incremental mode (start_date=None): checks MAX(date) per table, appends
    only new rows. Full mode (start_date provided): replaces existing table.

    Stocks are fetched in chunks of chunk_size to stay within FinMind's
    free-tier single-request row limit.
    """
    if end_date is None:
        end_date = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y%m%d")

    stock_codes = _load_stock_codes(stock_codes, json_path)
    incremental_mode = start_date is None
    engine = create_engine(PG_URI)
    fallback_start = DEFAULT_START_DATE

    print(f"[margin] Mode: {'incremental' if incremental_mode else 'full'} | "
          f"End: {end_date} | Stocks: {len(stock_codes)}")

    if incremental_mode:
        next_starts: dict[str, str] = {}
        for code in stock_codes:
            latest = _get_latest_date(f"margin_{code}", engine)
            if latest is None:
                next_starts[code] = fallback_start
            else:
                next_day = (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
                next_starts[code] = next_day
        codes_to_fetch = [c for c in stock_codes if next_starts[c] <= end_date]
    else:
        codes_to_fetch = stock_codes

    if not codes_to_fetch:
        print("[margin] All stocks already up to date.")
        return

    api = DataLoader()
    chunks = [codes_to_fetch[i:i + chunk_size] for i in range(0, len(codes_to_fetch), chunk_size)]
    print(f"[margin] Fetching {len(codes_to_fetch)} stocks in {len(chunks)} chunks...")

    for chunk in tqdm(chunks, desc="Fetching margin chunks"):
        if incremental_mode:
            effective_start = min(next_starts[c] for c in chunk)
        else:
            effective_start = start_date

        df = api.taiwan_stock_margin_purchase_short_sale(
            stock_id_list=chunk,
            start_date=_to_finmind_date(effective_start),
            end_date=_to_finmind_date(end_date),
            use_async=True,
        )
        if df is None or df.empty:
            logger.warning(f"[margin] No data for chunk {chunk}")
            continue

        df["date"] = pd.to_datetime(df["date"])

        for code in chunk:
            try:
                sub = df[df["stock_id"] == code].copy()
                if sub.empty:
                    continue
                if incremental_mode:
                    if_exists = "replace" if next_starts[code] == fallback_start else "append"
                else:
                    if_exists = "replace"
                sub.to_sql(name=f"margin_{code}", con=engine, if_exists=if_exists, index=False)
                logger.info(f"[margin] {code}: {if_exists} ({len(sub)} rows)")
            except Exception as e:
                logger.error(f"[margin] Error saving {code}: {e}")

    print("[margin] Done.")


def download_stock_data_to_db(
    stock_codes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    json_path: Path = OUTPUT_JSON_PATH,
):
    """
    Module 2: Download TWSE daily stock data and store in PostgreSQL.

    Supports two modes:
    - Full download: provide start_date explicitly (e.g. "20200101").
      Uses if_exists='replace' to overwrite existing data.
    - Incremental update: leave start_date=None. The function checks the
      latest date in each stock's table and fetches from the next day.
      If the table doesn't exist, falls back to full download from "20200101".
      Uses if_exists='append' to preserve history.

    Args:
        stock_codes: List of stock codes. Defaults to reading from json_path.
        start_date: Start date string "YYYYMMDD". None means incremental mode.
        end_date: End date string "YYYYMMDD". Defaults to today.
        json_path: Path to topk_stocks_by_industries.json (used when stock_codes is None).
    """
    if end_date is None:
        end_date = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y%m%d")

    if stock_codes is None:
        if not json_path.exists():
            raise FileNotFoundError(
                f"{json_path} not found. Run generate_stock_selection() first."
            )
        with open(json_path, "r", encoding="utf-8") as f:
            stock_dict = json.load(f)
        stock_codes = [code for codes in stock_dict.values() for code in codes]
        stock_codes = list(dict.fromkeys(stock_codes))  # deduplicate, preserve order

    incremental_mode = start_date is None
    engine = create_engine(PG_URI)
    fallback_start = DEFAULT_START_DATE

    print(f"Mode: {'incremental update' if incremental_mode else 'full download'}")
    print(f"End date: {end_date} | Stocks: {len(stock_codes)}")

    for code in tqdm(stock_codes, desc="Downloading stock data"):
        try:
            if incremental_mode:
                latest = _get_latest_date_for_stock(code, engine)
                if latest is None:
                    # Table doesn't exist — full download
                    effective_start = fallback_start
                    if_exists = "replace"
                elif latest == pd.to_datetime(end_date):
                    print(f"  {code}: already up to date (latest={latest.date()}), skipping.")
                    continue
                else:
                    next_day = latest + pd.Timedelta(days=1)
                    effective_start = next_day.strftime("%Y%m%d")
                    if effective_start > end_date:
                        print(f"  {code}: already up to date (latest={latest.date()}), skipping.")
                        continue
                    if_exists = "append"
            else:
                effective_start = start_date
                if_exists = "replace"

            _save_stock_data(code, effective_start, end_date, engine, if_exists)
            logger.info(f"Finished {code}: {effective_start} ~ {end_date} ({if_exists})")

        except Exception as e:
            logger.error(f"Error processing {code}: {e}")

    print("Done.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stock data preprocessing pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Module 1
    p1 = subparsers.add_parser("select", help="Generate stock selection JSON")
    p1.add_argument("--top-k", type=int, default=10)
    p1.add_argument("--categories", nargs="*", default=None)

    # Module 2
    p2 = subparsers.add_parser("download", help="Download stock data to DB")
    p2.add_argument("--start-date", default=None, help="YYYYMMDD; omit for incremental mode")
    p2.add_argument("--end-date", default=None, help="YYYYMMDD; defaults to today")
    p2.add_argument("--codes", nargs="*", default=None)

    # Module 3a
    p3a = subparsers.add_parser("institutional", help="Download institutional investors data to DB")
    p3a.add_argument("--start-date", default=None, help="YYYYMMDD; omit for incremental mode")
    p3a.add_argument("--end-date", default=None, help="YYYYMMDD; defaults to today")
    p3a.add_argument("--codes", nargs="*", default=None)

    # Module 3b
    p3b = subparsers.add_parser("margin", help="Download margin purchase/short sale data to DB")
    p3b.add_argument("--start-date", default=None, help="YYYYMMDD; omit for incremental mode")
    p3b.add_argument("--end-date", default=None, help="YYYYMMDD; defaults to today")
    p3b.add_argument("--codes", nargs="*", default=None)

    args = parser.parse_args()

    if args.command == "select":
        generate_stock_selection(
            inspect_categories=args.categories,
            top_k=args.top_k,
        )
    elif args.command == "download":
        download_stock_data_to_db(
            stock_codes=args.codes,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    elif args.command == "institutional":
        download_institutional_to_db(
            stock_codes=args.codes,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    elif args.command == "margin":
        download_margin_to_db(
            stock_codes=args.codes,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        parser.print_help()
