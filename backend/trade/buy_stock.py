"""
買入策略：
1. 從 lstm_prediction 資料表讀取當日所有股票的預測價格與報酬率
2. 過濾 pred_ret_pct > 1.0%，依報酬率降冪排序，取前 25 支
3. 排名第 1~5 買入 2 張，第 6~25 買入 1 張
4. 跳過已在庫存中的股票（避免重複買入）
5. 以 pred_price（預測價格）作為下單價格
"""

import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import pandas as pd
import logging
import sqlalchemy
from pathlib import Path
from stock_api import Buy_Stock, Get_User_Stocks

load_dotenv()
log_path = Path("/app/logs/error.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True,
)

ACCOUNT = os.getenv('STOCK_ACCOUNT_USERNAME')
PASSWORD = os.getenv('STOCK_ACCOUNT_PASSWORD')
PG_URI = os.getenv('POSTGRES_URI')
ENGINE = sqlalchemy.create_engine(PG_URI)

TOP_N = 25          # 最多買入支數
TOP_N_DOUBLE = 5    # 前幾名買 2 張
MIN_RET_PCT = 1.0   # 報酬率門檻（%）
SHARES_PER_ORDER = 5 # 每筆訂單買幾張（前 TOP_N_DOUBLE 名除外）
SHARES_PER_ORDER_DOUBLE = 10 # 前 TOP_N_DOUBLE 名買幾張


def check_pred_table(inference_date: str | None = None) -> pd.DataFrame | None:
    today = inference_date or time.strftime('%Y-%m-%d')
    sql = "SELECT * FROM lstm_prediction WHERE inference_date = %(d)s"
    try:
        df = pd.read_sql(sql, ENGINE, params={'d': today})
        return df
    except Exception as e:
        logging.error(f"Error checking lstm_prediction table: {e}")
        return None


def select_buy_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """篩選買入候選：pred_ret_pct > MIN_RET_PCT，取前 TOP_N，標記張數"""
    df = df.copy()
    df['pred_ret_pct'] = pd.to_numeric(df['pred_ret_pct'], errors='coerce')
    df['pred_price'] = pd.to_numeric(df['pred_price'], errors='coerce')

    candidates = (
        df[df['pred_ret_pct'] > MIN_RET_PCT]
        .sort_values('pred_ret_pct', ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )

    # 前 TOP_N_DOUBLE 名買 2 張，其餘買 1 張
    candidates['shares_to_buy'] = SHARES_PER_ORDER
    candidates.loc[:TOP_N_DOUBLE - 1, 'shares_to_buy'] = SHARES_PER_ORDER_DOUBLE

    return candidates


def get_existing_holdings() -> set:
    """取得目前已持有的股票代碼，避免重複買入"""
    try:
        holdings = Get_User_Stocks(ACCOUNT, PASSWORD)
        return {str(h['stock_code_id']) for h in holdings}
    except Exception as e:
        logging.error(f"Error fetching holdings: {e}")
        return set()


def batch_buy_order(candidates: pd.DataFrame):
    existing = get_existing_holdings()
    order_date = time.strftime('%Y-%m-%d')
    records = []

    for _, row in candidates.iterrows():
        stock_code = str(row['stock_code'])
        stock_price = float(row['pred_price'])
        shares = int(row['shares_to_buy'])

        if stock_code in existing:
            print(f"Skip {stock_code}: already in portfolio")
            continue

        status = 'ok'
        try:
            Buy_Stock(ACCOUNT, PASSWORD, stock_code=stock_code, stock_shares=shares, stock_price=stock_price)
            print(f"Buy {stock_code}  shares={shares}  price={stock_price}  ret={row['pred_ret_pct']:.2f}%")
        except Exception as e:
            logging.error(f"Error buying {stock_code}: {e}")
            status = 'error'

        records.append({
            'order_date':   order_date,
            'stock_code':   stock_code,
            'shares':       shares,
            'price':        stock_price,
            'pred_ret_pct': round(float(row['pred_ret_pct']), 2),
            'status':       status,
        })
        time.sleep(1)

    if records:
        try:
            pd.DataFrame(records).to_sql('trade_buy_log', ENGINE, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Error saving buy log to DB: {e}")


if __name__ == "__main__":
    START_TIME = datetime.now()
    print("=== Starting batch buy process ===")

    pred_table = check_pred_table()
    if pred_table is not None and not pred_table.empty:
        candidates = select_buy_candidates(pred_table)
        print(f"Buy candidates ({len(candidates)} stocks):")
        print(candidates[['stock_code', 'pred_ret_pct', 'pred_price', 'shares_to_buy']].to_string(index=False))
        batch_buy_order(candidates)

    END_TIME = datetime.now()
    print(f"=== Batch buy completed in {(END_TIME - START_TIME).total_seconds():.1f}s ===")
