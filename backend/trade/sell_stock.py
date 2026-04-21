"""
This module contains functions and classes for selling stocks.
"""

import random
import requests
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import pandas as pd
import logging
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

ACCOUNT = os.getenv('STOCK_ACCOUNT_USERNAME')
PASSWORD = os.getenv('STOCK_ACCOUNT_PASSWORD')
PG_URI = os.getenv('POSTGRES_URI')

# Fetch current stock price

# Fetch current portfolio information

# Calcualte net value and profit/loss for the stock

# Make a sell order

# Check the order status and update the database accordingly

if __name__ == "__main__":
    Sell_Stock(ACCOUNT, PASSWORD, stock_code=2330, stock_shares=1, stock_price=2081)