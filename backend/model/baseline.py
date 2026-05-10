"""
Stock Return Baseline Models.
三種 baseline：
  1. Naive（隨機遊走）：預測值 = 上一期的值
  2. Moving Average：預測值 = 前 N 期的平均
  3. AR(p)（自回歸模型）：用 statsmodels AutoReg 擬合

目標變數統一使用 pct_change（報酬率），與 LSTM 對齊。
Test set 切法由外部傳入，確保與 LSTM 使用相同日期區間。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg


def _evaluate(y_true: pd.Series, y_pred: pd.Series, name: str,
              actual_mean_ret: float, next_day_pred_ret: float,
              train_r2: float, train_rmse: float) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "model":              name,
        "train_r2":           train_r2,
        "train_rmse":         train_rmse,
        "test_r2":            r2,
        "test_rmse":          rmse,
        "pred_mean_ret":      float(y_pred.mean()),
        "actual_mean_ret":    actual_mean_ret,
        "next_day_pred_ret":  next_day_pred_ret,
    }


def _split(ret: pd.Series, test_start: str):
    """依固定日期切 train / test。"""
    train = ret[ret.index < test_start]
    test  = ret[ret.index >= test_start]
    return train, test


# ── Baseline 1：Naive ────────────────────────────────────────────────────────

def naive_baseline(ret: pd.Series, stock_code: str, test_start: str) -> dict:
    train, test = _split(ret, test_start)
    shifted     = ret.shift(1)

    tr_pred = shifted.reindex(train.index)
    tr_mask = tr_pred.notna()
    tr_r2   = r2_score(train[tr_mask], tr_pred[tr_mask])
    tr_rmse = np.sqrt(mean_squared_error(train[tr_mask], tr_pred[tr_mask]))

    y_pred = shifted.reindex(test.index)
    mask   = y_pred.notna()
    y_true, y_pred = test[mask], y_pred[mask]

    actual_mean_ret   = float(y_true.mean())
    next_day_pred_ret = float(ret.iloc[-1])

    return _evaluate(y_true, y_pred,
                     name=f"Naive_{stock_code}",
                     actual_mean_ret=actual_mean_ret,
                     next_day_pred_ret=next_day_pred_ret,
                     train_r2=tr_r2, train_rmse=tr_rmse)


# ── Baseline 2：Moving Average ───────────────────────────────────────────────

def ma_baseline(ret: pd.Series, stock_code: str, test_start: str,
                window: int = 5) -> dict:
    train, test = _split(ret, test_start)
    rolled      = ret.rolling(window=window).mean().shift(1)

    tr_pred = rolled.reindex(train.index)
    tr_mask = tr_pred.notna()
    tr_r2   = r2_score(train[tr_mask], tr_pred[tr_mask])
    tr_rmse = np.sqrt(mean_squared_error(train[tr_mask], tr_pred[tr_mask]))

    y_pred = rolled.reindex(test.index)
    mask   = y_pred.notna()
    y_true, y_pred = test[mask], y_pred[mask]

    actual_mean_ret   = float(y_true.mean())
    next_day_pred_ret = float(ret.rolling(window=window).mean().iloc[-1])

    return _evaluate(y_true, y_pred,
                     name=f"MA({window})_{stock_code}",
                     actual_mean_ret=actual_mean_ret,
                     next_day_pred_ret=next_day_pred_ret,
                     train_r2=tr_r2, train_rmse=tr_rmse)


# ── Baseline 3：AR(p) ────────────────────────────────────────────────────────

def ar_baseline(ret: pd.Series, stock_code: str, test_start: str,
                lags: int = 5) -> dict:
    ret = ret.rename(stock_code)
    train, test = _split(ret, test_start)

    # 用純 RangeIndex 傳入 AutoReg，避免 DatetimeIndex 缺頻率的警告
    model  = AutoReg(pd.Series(train.values, dtype=float), lags=lags).fit()
    start  = len(train)
    end    = len(train) + len(test) - 1
    y_pred = pd.Series(model.predict(start=start, end=end).values, index=test.index)
    y_true = test.iloc[:len(y_pred)]

    actual_mean_ret = float(y_true.mean())

    tr_pred_vals = model.fittedvalues.values
    tr_true      = train.iloc[len(train) - len(tr_pred_vals):]
    mask         = ~np.isnan(tr_pred_vals)
    tr_r2        = r2_score(tr_true.values[mask], tr_pred_vals[mask])
    tr_rmse      = np.sqrt(mean_squared_error(tr_true.values[mask], tr_pred_vals[mask]))

    model_full        = AutoReg(pd.Series(ret.values, dtype=float), lags=lags).fit()
    next_day_pred_ret = float(model_full.predict(start=len(ret), end=len(ret)).iloc[0])

    return _evaluate(y_true, y_pred,
                     name=f"AR({lags})_{stock_code}",
                     actual_mean_ret=actual_mean_ret,
                     next_day_pred_ret=next_day_pred_ret,
                     train_r2=tr_r2, train_rmse=tr_rmse)


# ── 對外 API：接受 ret_wide（已算好的報酬率 DataFrame）────────────────────────

def run_baselines(ret_wide: pd.DataFrame, test_start: str,
                  ma_window: int = 5, ar_lags: int = 5) -> pd.DataFrame:
    """
    Parameters
    ----------
    ret_wide   : columns = stock_code（整數或字串），index = date，值為 pct_change
    test_start : '2025-01-01'

    Returns
    -------
    DataFrame with one row per (model, stock_code)
    """
    results = []
    for col in ret_wide.columns:
        ret        = ret_wide[col].dropna()
        stock_code = str(col)
        results.append(naive_baseline(ret, stock_code, test_start))
        results.append(ma_baseline(ret, stock_code, test_start, window=ma_window))
        results.append(ar_baseline(ret, stock_code, test_start, lags=ar_lags))

    return pd.DataFrame(results)
