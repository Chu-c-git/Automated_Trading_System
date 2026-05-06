"""
Stock Return Baseline Models. 評估 2026-01-01 到 2026-05-30 的表現
三種 baseline：
  1. Naive（隨機遊走）：預測值 = 上一期的值
  2. Moving Average：預測值 = 前 N 期的平均
  3. AR(p)（自回歸模型）：用 statsmodels AutoReg 擬合
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from dotenv import load_dotenv
from sqlalchemy import create_engine
from statsmodels.tsa.ar_model import AutoReg
import timeit
from tqdm import tqdm
import os
import json
load_dotenv()

DB_URI = os.getenv("POSTGRES_URI")
TABLE_NAME = "daily_info_"
STOCK_LIST_PATH = '/app/data/top5_stocks_by_category.json'
START_DATE = '2026-01-01'
END_DATE = '2026-05-05'
with open(STOCK_LIST_PATH, 'r') as f:
    data = json.load(f)
    STOCK_CODE_LIST = [ code for codes in data.values() for code in codes]

# 
def fetch_data() -> pd.Series:

    engine = create_engine(DB_URI)
    table_list = []
    print(STOCK_CODE_LIST)
    with engine.connect() as conn:
        for code in tqdm(STOCK_CODE_LIST, desc="Fetching data", total=len(STOCK_CODE_LIST)):
            try:
                table_name = TABLE_NAME + str(code)
                sql = f"""
                SELECT change, close, date
                FROM "{table_name}"
                WHERE date >= '{START_DATE}' AND date <= '{END_DATE}'
                ORDER BY date ASC
                """
                table = pd.read_sql(sql, conn, index_col="date")
                table.index = pd.to_datetime(table.index)
                table = table.rename(columns={"change": f"change_{code}", "close": f"close_{code}"})
                table_list.append(table)
            except Exception as e:
                print(f"Error fetching data for stock code {code}: {e}")
    df = pd.concat(table_list, axis=1)
    print(df.shape)
    print(df.info())
    print(df.head())
    return df

# ── 評估函式（三種 baseline 共用）────────────────────────────────────────────

def evaluate(y_true: pd.Series, y_pred: pd.Series, name: str) -> dict:
    # 計算 R²、RMSE、以及 RMSE 除以 y_true 標準差（正規化 RMSE）
    r_squared = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_norm = rmse / np.std(y_true)

    # 回傳一個 dict，key 為 "model", "r2", "rmse", "rmse_norm"
    result_dict = {
        "model": name,
        "r2": r_squared,
        "rmse": rmse,
        "rmse_norm": rmse_norm
    }
    return result_dict


# ── Baseline 1：Naive（隨機遊走）────────────────────────────────────────────

def naive_baseline(y: pd.Series, stock_code: str) -> dict:
    # Step 1：產生預測值：把 y 向後位移一期（shift(1)）
    y_pred = y.shift(1)
    
    # Step 2：對齊 y_true 與 y_pred（去掉 NaN 的那一列）
    y_true = y[y_pred.notna()]
    y_pred = y_pred[y_pred.notna()]

    # Step 3：呼叫 evaluate，name 設為 "Naive"
    evaluate_result = evaluate(y_true, y_pred, name=f"Naive_{stock_code}")
    return evaluate_result


# ── Baseline 2：Moving Average──────────────────────────────────────────────

def ma_baseline(y: pd.Series, stock_code: str, window: int = 5) -> dict:
    # Step 1：計算 rolling mean（window 期），再 shift(1) 避免 lookahead
    rol_mean = y.rolling(window=window).mean()
    y_pred = rol_mean.shift(1)

    # Step 2：對齊 y_true 與 y_pred（dropna）
    y_true = y[y_pred.notna()]
    y_pred = y_pred[y_pred.notna()]

    # Step 3：呼叫 evaluate，name 設為 f"MA({window})"
    evaluate_result = evaluate(y_true, y_pred, name=f"Mov_Avg_({window}Day)_{stock_code}")
    return evaluate_result


# ── Baseline 3：AR(p) 自回歸模型────────────────────────────────────────────

def ar_baseline(y: pd.Series, stock_code: str, lags: int = 5, test_size: float = 0.2) -> dict:
    # Step 1：依 test_size 切分 train / test（時間序列不可隨機切！）
    split_point = int(len(y) * (1 - test_size))
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]

    # Step 2：用 statsmodels AutoReg(y_train, lags=lags).fit() 訓練
    model = AutoReg(y_train, lags=lags).fit()

    # Step 3：用 model.predict(start, end) 產生 test 區間的預測值
    y_pred = model.predict(start=split_point, end=len(y)-1)

    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_test.index)
    y_pred = y_pred.iloc[:len(y_test)]

    # Step 4：呼叫 evaluate，name 設為 f"AR({lags})"
    evaluate_result = evaluate(y_test, y_pred, name=f"AutoReg_({lags}day)_{stock_code}")
    return evaluate_result


# ── 主程式：跑完三種 baseline 並印出比較表────────────────────────────────────

def run_all_baselines(ma_window: int = 5, ar_lags: int = 5):
    results = []
    raw_df = fetch_data()
    target_col_list = [col for col in raw_df.columns if col.startswith("change_")]
    for col in target_col_list:
        print(f"Evaluating baselines for {col}...")
        y = raw_df[col].dropna()
        stock_code = col.split("_")[1]
        # Step 1：依序呼叫 naive_baseline、ma_baseline、ar_baseline，把回傳的 dict append 進 results
        results.append(naive_baseline(y, stock_code))
        results.append(ma_baseline(y, stock_code, window=ma_window))
        results.append(ar_baseline(y, stock_code, lags=ar_lags, test_size=0.2))

    # Step 2：用 pd.DataFrame(results) 整理成表格後印出
    eval_df = pd.DataFrame(results)
    eval_df['start_date'] = START_DATE
    eval_df['end_date'] = END_DATE

    # Step 3: 存到 PG，table name 可以叫 "baseline_evaluation"
    engine = create_engine(DB_URI)
    with engine.connect() as conn:
        eval_df.to_sql("baseline_evaluation", conn, if_exists="replace", index=False)
        print(">>> Baseline evaluation results saved to PostgreSQL table 'baseline_evaluation'.")
        print(eval_df)
        


if __name__ == "__main__":
    exec_time = timeit.timeit(run_all_baselines, number=1)
    print(f"Total execution time: {exec_time:.2f} seconds")
    pass
