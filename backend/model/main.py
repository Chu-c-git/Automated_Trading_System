from utils import init_mlflow, get_stock_data_by_category
import pandas as pd
import time
from feature_engineering import build_features, create_sequences
from LSTM import train_lstm, evaluate_metrics, recursive_forecast

START_DATE = '2020-01-01'
END_DATE = time.now().strftime('%Y-%m-%d')
MLFLOW_URI = init_mlflow()

if __name__ == "__main__":
    df = get_stock_data_by_category('半導體業', START_DATE, END_DATE)
    X_tr, X_te, y_tr, y_te, df_train, df_test, x_sc, y_sc, feat_cols, tgt_cols, close_wide = build_features(df)

    X_train_np, y_train_np = create_sequences(X_tr, y_tr, seq_len=7)
    X_test_np,  y_test_np  = create_sequences(X_te, y_te, seq_len=7)

    model, run, train_loader, test_loader, device = train_lstm(
        X_train_np, y_train_np, X_test_np, y_test_np, y_sc, stock_codes
    )