"""
Model Pipeline Entrypoint

執行順序（每個 category）：
  1. 拉資料
  2. Baseline pipeline（Naive / MA / AR）
  3. LSTM pipeline（feature engineering → train → evaluate → forecast）
  4. 合併比較表存入 PostgreSQL
     - model_comparison_per_stock
     - model_comparison_per_category
  5. 交易執行（先賣後買）
     - sell_stock: 停利 +10% / 停損 -10%
     - buy_stock:  pred_ret_pct > 1%，取前25，前5買2張其餘1張
"""

import json
import logging
import os
import sys
import time
import timeit

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/trade')
from model.baseline import run_baselines
from model.feature_engineering import build_features, create_sequences
from model.LSTM import build_forecast_df, evaluate_metrics, train_lstm
from model.utils import get_stock_data_by_category, init_mlflow
from data_preprocess.data_preprocess import download_stock_data_to_db
from trade.sell_stock import batch_sell_order
from trade.buy_stock import check_pred_table, select_buy_candidates, batch_buy_order

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/pipeline.log', mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

POSTGRES_URI: str = os.getenv("POSTGRES_URI", "")
STOCK_DICT   = json.load(open('/app/data/topk_stocks_by_industries.json', 'r', encoding='utf-8'))

TRAIN_START = '2015-01-01'
RUN_DATE    = time.strftime('%Y-%m-%d')
TEST_START  = (pd.Timestamp(RUN_DATE) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')

SEQ_LEN    = 7
MA_WINDOW  = 5
AR_LAGS    = 5
HIDDEN     = 16
NUM_LAYERS = 1
DROPOUT    = 0.3
LR         = 0.001
BATCH_SIZE = 32
EPOCHS     = 100
EVAL_EVERY = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_meta(df: pd.DataFrame, category: str) -> pd.DataFrame:
    df = df.copy()
    df['run_date']    = RUN_DATE
    df['category']    = category
    df['train_start'] = TRAIN_START
    df['train_end']   = str(pd.Timestamp(TEST_START) - pd.Timedelta(days=1))[:10]
    df['test_start']  = TEST_START
    df['test_end']    = RUN_DATE
    return df


def _baseline_rows(ret_wide: pd.DataFrame, category: str) -> pd.DataFrame:
    bl_df = run_baselines(ret_wide, test_start=TEST_START,
                          ma_window=MA_WINDOW, ar_lags=AR_LAGS)
    # model 欄位格式如 'Naive_2330'，拆出 stock_code
    bl_df[['model', 'stock_code']] = bl_df['model'].str.rsplit('_', n=1, expand=True)
    return _add_meta(bl_df, category)


def _lstm_rows(df_raw: pd.DataFrame, category: str, stock_codes: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    (X_tr, X_te,
     y_tr, y_te,
     df_train, df_test,
     x_sc, y_sc,
     feat_cols, tgt_cols, close_wide) = build_features(df_raw, test_start=TEST_START)

    X_train_np, y_train_np = create_sequences(X_tr, y_tr, seq_len=SEQ_LEN)
    X_test_np,  y_test_np  = create_sequences(X_te, y_te, seq_len=SEQ_LEN)

    model, _, _, test_loader, device, train_rmse, train_mae, train_r2 = train_lstm(
        X_train_np, y_train_np, X_test_np, y_test_np,
        y_sc, stock_codes,
        hidden_size=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT,
        lr=LR, batch_size=BATCH_SIZE, num_epochs=EPOCHS, eval_every=EVAL_EVERY,
    )

    rmse, mae, r2, hit_rate, pred_mean_ret, next_day_pred_ret = evaluate_metrics(
        model, test_loader, y_sc, device
    )

    # test set 實際平均報酬率（benchmark），tgt_cols 對應 stock_codes 順序
    actual_mean_ret = np.array([df_test[col].mean() for col in tgt_cols])

    rows = []
    for i, code in enumerate(stock_codes):
        rows.append({
            'model':             'LSTM',
            'stock_code':        str(code),
            'train_r2':          float(train_r2[i]),
            'train_rmse':        float(train_rmse[i]),
            'test_r2':           float(r2[i]),
            'test_rmse':         float(rmse[i]),
            'test_mae':          float(mae[i]),
            'test_hit_rate':     float(hit_rate[i]),
            'pred_mean_ret':     float(pred_mean_ret[i]),
            'actual_mean_ret':   float(actual_mean_ret[i]),
            'next_day_pred_ret': float(next_day_pred_ret[i]),
        })

    lstm_df = pd.DataFrame(rows)

    forecast_df = build_forecast_df(
        model, X_test_np, df_test, close_wide,
        x_sc, y_sc, feat_cols, stock_codes,
        device=device, category=category,
    )

    return _add_meta(lstm_df, category), forecast_df


def _build_comparison_tables(per_stock: pd.DataFrame):
    num_cols = ['test_r2', 'test_rmse', 'test_mae', 'test_hit_rate',
                'pred_mean_ret', 'actual_mean_ret', 'next_day_pred_ret']
    per_cat = (
        per_stock
        .groupby(['run_date', 'category', 'model',
                  'train_start', 'train_end', 'test_start', 'test_end'])[num_cols]
        .mean()
        .reset_index()
    )
    return per_stock, per_cat


def _save_to_db(per_stock: pd.DataFrame, per_cat: pd.DataFrame,
                forecast: pd.DataFrame, engine):
    with engine.begin() as conn:
        per_stock.to_sql('model_comparison_per_stock',  conn, if_exists='replace', index=False)
        per_cat.to_sql('model_comparison_per_category', conn, if_exists='replace', index=False)
        forecast.to_sql('lstm_prediction',              conn, if_exists='append', index=False)
    print(f"[{RUN_DATE}] Saved comparison tables to PostgreSQL.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    logger.info("Updating stock data (incremental)...")
    try:
        download_stock_data_to_db()
        logger.info("Stock data update complete.")
    except Exception:
        logger.exception("Stock data update failed — continuing with existing data.")

    init_mlflow()
    engine        = create_engine(POSTGRES_URI)
    all_per_stock  = []
    all_forecasts  = []

    with mlflow.start_run(run_name=f"pipeline_{RUN_DATE}"):
        for category, codes in STOCK_DICT.items():
            logger.info("=" * 60)
            logger.info(f"Category: {category}  stocks: {codes}")
            stock_codes = [str(c) for c in codes]

            try:
                df_raw = get_stock_data_by_category(category, TRAIN_START, RUN_DATE)
            except Exception:
                logger.exception(f"[{category}] Failed to fetch data — skipping")
                continue

            if df_raw.empty:
                logger.warning(f"[{category}] No data returned — skipping")
                continue

            # ── Baseline：從 raw df 算 pct_change ──
            try:
                close_pivot = (
                    df_raw.assign(date=pd.to_datetime(df_raw['date']))
                    .pivot_table(index='date', columns='stock_code_id',
                                 values='close', aggfunc='mean')
                    .sort_index().ffill().bfill()
                )
                ret_wide = (
                    close_pivot.pct_change()
                    .replace([float('inf'), float('-inf')], float('nan'))
                    .fillna(0.0)
                )
                ret_wide.columns = [str(c) for c in ret_wide.columns]
                bl_df = _baseline_rows(ret_wide, category)
                logger.info(f"[{category}] Baseline done")
            except Exception:
                logger.exception(f"[{category}] Baseline pipeline failed — skipping category")
                continue

            # ── LSTM ──
            try:
                with mlflow.start_run(run_name=category, nested=True):
                    lstm_df, forecast_df = _lstm_rows(df_raw, category, stock_codes)
                logger.info(f"[{category}] LSTM done")
            except Exception:
                logger.exception(f"[{category}] LSTM pipeline failed — skipping category")
                continue

            all_per_stock.append(pd.concat([bl_df, lstm_df], ignore_index=True))
            all_forecasts.append(forecast_df)

    if not all_per_stock:
        logger.warning("No results to save — pipeline produced no output.")
        return

    per_stock = pd.concat(all_per_stock, ignore_index=True)
    per_stock, per_cat = _build_comparison_tables(per_stock)
    forecast = pd.concat(all_forecasts, ignore_index=True)

    logger.info("\n── Per-stock comparison ──\n" + per_stock.to_string(index=False))
    logger.info("\n── Per-category comparison ──\n" + per_cat.to_string(index=False))

    try:
        _save_to_db(per_stock, per_cat, forecast, engine)
        logger.info("Comparison tables saved to PostgreSQL.")
    except Exception:
        logger.exception("Failed to save comparison tables to PostgreSQL.")

    # ── 交易執行：先賣後買 ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting trade execution (sell → buy)")

    try:
        logger.info("Running sell strategy...")
        batch_sell_order()
        logger.info("Sell strategy completed.")
    except Exception:
        logger.exception("Sell strategy failed.")

    try:
        logger.info("Running buy strategy...")
        pred_table = check_pred_table()
        if pred_table is not None and not pred_table.empty:
            candidates = select_buy_candidates(pred_table)
            logger.info(f"Buy candidates: {len(candidates)} stocks")
            batch_buy_order(candidates)
            logger.info("Buy strategy completed.")
        else:
            logger.warning("No prediction data found — skipping buy.")
    except Exception:
        logger.exception("Buy strategy failed.")


if __name__ == "__main__":
    exec_time = timeit.timeit(run_pipeline, number=1)
    logger.info(f"Total execution time: {exec_time:.2f} seconds")
