"""
賣出策略：
1. 透過 Get_User_Stocks() 取得目前所有持倉（含買入成本價與張數）
2. 對每支持股用 twstock.realtime 取得即時成交價
3. 計算損益比例：(現價 - 成本) / 成本 * 100
4. 漲幅 >= 10%：停利賣出全部張數
5. 跌幅 >= 10%：停損賣出全部張數
6. 介於 -10% ~ 10%：持有不動
"""

from datetime import datetime
import time
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import pandas as pd
import sqlalchemy
import twstock
from stock_api.core import Sell_Stock, Get_User_Stocks

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

ACCOUNT  = os.getenv('STOCK_ACCOUNT_USERNAME')
PASSWORD = os.getenv('STOCK_ACCOUNT_PASSWORD')
PG_URI   = os.getenv('POSTGRES_URI')
ENGINE   = sqlalchemy.create_engine(PG_URI)

TAKE_PROFIT_PCT = 10.0  # 停利：漲超過 10%
STOP_LOSS_PCT   = 10.0  # 停損：跌超過 10%


def fetch_current_price(stock_code: str) -> float | None:
    """取得即時成交價"""
    try:
        data = twstock.realtime.get(stock_code)
        if not data.get('success'):
            logging.error(f"twstock realtime failed for {stock_code}")
            return None
        return float(data['realtime']['latest_trade_price'])
    except Exception as e:
        logging.error(f"Error fetching price for {stock_code}: {e}")
        return None


def calc_pnl_pct(current_price: float, cost_price: float) -> float:
    return (current_price - cost_price) / cost_price * 100


def batch_sell_order():
    try:
        holdings = Get_User_Stocks(ACCOUNT, PASSWORD)
    except Exception as e:
        logging.error(f"Error fetching holdings: {e}")
        return

    order_date = time.strftime('%Y-%m-%d')
    records = []

    for holding in holdings:
        stock_code = str(holding['stock_code_id'])
        cost_price = float(holding['beginning_price'])
        shares = int(holding['shares'])

        current_price = fetch_current_price(stock_code)
        if current_price is None:
            print(f"Skip {stock_code}: cannot fetch current price")
            continue

        pnl_pct = calc_pnl_pct(current_price, cost_price)

        if pnl_pct >= TAKE_PROFIT_PCT:
            reason = 'take_profit'
        elif pnl_pct <= -STOP_LOSS_PCT:
            reason = 'stop_loss'
        else:
            print(f"Hold {stock_code}: pnl={pnl_pct:.2f}%  current={current_price}  cost={cost_price}")
            continue

        status = 'ok'
        try:
            Sell_Stock(ACCOUNT, PASSWORD, stock_code=stock_code, stock_shares=shares, stock_price=current_price)
            print(f"Sell {stock_code}  shares={shares}  price={current_price}  reason={reason}  pnl={pnl_pct:.2f}%")
        except Exception as e:
            logging.error(f"Error selling {stock_code}: {e}")
            status = 'error'

        records.append({
            'order_date':   order_date,
            'stock_code':   stock_code,
            'shares':       shares,
            'price':        current_price,
            'cost_price':   cost_price,
            'pnl_pct':      round(pnl_pct, 2),
            'reason':       reason,
            'status':       status,
        })
        time.sleep(1)

    if records:
        try:
            pd.DataFrame(records).to_sql('trade_sell_log', ENGINE, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Error saving sell log to DB: {e}")


if __name__ == "__main__":
    START_TIME = datetime.now()
    print("=== Starting batch sell process ===")
    batch_sell_order()
    END_TIME = datetime.now()
    print(f"=== Batch sell completed in {(END_TIME - START_TIME).total_seconds():.1f}s ===")
