"""
This module contains functions and classes for buying stocks.
"""

import random
import requests
import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import pandas as pd
import logging
import sqlalchemy
from pathlib import Path
from tqdm import tqdm
from stock_api import Buy_Stock, Sell_Stock

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
stock_dict_path = '/app/data/top5_stocks_by_category.json'
with open(stock_dict_path, 'r', encoding='utf-8') as f:
    stock_dict = json.load(f)

ACCOUNT = os.getenv('STOCK_ACCOUNT_USERNAME')
PASSWORD = os.getenv('STOCK_ACCOUNT_PASSWORD')
PG_URI = os.getenv('POSTGRES_URI')
ENGINE = sqlalchemy.create_engine(PG_URI)
STOCK_SHARE = 1

# Check predictions table
def check_pred_table():
    sql = """
     SELECT *
     FROM lstm_prediction
    """
    try:
        df = pd.read_sql(sql, ENGINE)
        return df
    except Exception as e:
        logging.error(f"Error checking lstm_prediction table: {e}")
        return None

# Fetch current stock price
# def fetch_current_price(stock_code):

# Modify the buy price based on the current stock price and the predicted price

# Make a batch_buy order
def batch_buy_order(df: pd.DataFrame, stock_code: dict):
    try:
        for item in stock_code:
            match_col = f"{item}_pred_price"
            stock_price = df[match_col].iloc[0]  # Get the nearest predicted price for the stock
            Buy_Stock(ACCOUNT, PASSWORD, stock_code=item, stock_shares=STOCK_SHARE, stock_price=stock_price)
            print(f"Placed buy order for {item} at price {stock_price}")
            time.sleep(1)  # Sleep to avoid hitting API rate limits
    except Exception as e:
        logging.error(f"Error making buy order for {stock_code}: {e}")

# Check the order status and update the database accordingly

if __name__ == "__main__":
    START_TIME = datetime.now()
    print("=== Starting batch buy process ===")
    pred_table = check_pred_table()
    if pred_table is not None and not pred_table.empty:
        batch_buy_order(pred_table, stock_dict['半導體業'])
    
    END_TIME = datetime.now()
    print(f"=== Batch buy process completed in {(END_TIME - START_TIME).total_seconds()} seconds ===")