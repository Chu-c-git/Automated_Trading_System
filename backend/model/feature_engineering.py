import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

EXTRA_COLS = ['open', 'high', 'low', 'capacity', 'turnover', 'transaction_volume', 'change']


def build_features(df: pd.DataFrame, train_ratio: float = 0.9):
    """
    Input : raw df with columns [date, stock_code_id, close, open, high, low, ...]
    Output: X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            df_train, df_test, x_scaler, y_scaler,
            feature_cols, target_cols, close_wide
    """
    selected_stocks = sorted(df['stock_code_id'].unique())

    df_sub = df[df['stock_code_id'].isin(selected_stocks)].copy()
    df_sub['date'] = pd.to_datetime(df_sub['date']).dt.normalize()

    # ── close pivot ──────────────────────────────────────────────────────────
    close_wide = (
        df_sub.groupby(['date', 'stock_code_id'], as_index=False)['close']
        .mean()
        .pivot_table(index='date', columns='stock_code_id', values='close', aggfunc='mean')
        .sort_index().ffill().bfill()
    )
    close_wide.columns = [f"close_{int(c)}" for c in close_wide.columns]

    # ── extra OHLCV + change pivots ──────────────────────────────────────────
    extra_wide_list = []
    for col in EXTRA_COLS:
        w = (
            df_sub.groupby(['date', 'stock_code_id'], as_index=False)[col]
            .mean()
            .pivot_table(index='date', columns='stock_code_id', values=col, aggfunc='mean')
            .sort_index().ffill().bfill()
        )
        w.columns = [f"{col}_{int(c)}" for c in w.columns]
        extra_wide_list.append(w)

    # ── return & cross-stock features ────────────────────────────────────────
    ret_wide = close_wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_wide.columns = [f"ret_{c.split('_')[1]}" for c in close_wide.columns]

    rel_close = close_wide.sub(close_wide.mean(axis=1), axis=0)
    rel_close.columns = [f"rel_{c}" for c in close_wide.columns]

    # column order: close | extra... | ret | market | rel
    feature_df = pd.concat(
        [close_wide] + extra_wide_list + [
            ret_wide,
            close_wide.mean(axis=1).to_frame('market_mean_close'),
            ret_wide.mean(axis=1).to_frame('market_mean_ret'),
            rel_close,
        ],
        axis=1,
    ).dropna()

    target_cols  = ret_wide.columns.tolist()
    feature_cols = feature_df.columns.tolist()

    n_train  = int(len(feature_df) * train_ratio)
    df_train = feature_df.iloc[:n_train]
    df_test  = feature_df.iloc[n_train:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(df_train[feature_cols].values)
    X_test_scaled  = x_scaler.transform(df_test[feature_cols].values)
    y_train_scaled = y_scaler.fit_transform(df_train[target_cols].values)
    y_test_scaled  = y_scaler.transform(df_test[target_cols].values)

    return (
        X_train_scaled, X_test_scaled,
        y_train_scaled, y_test_scaled,
        df_train, df_test,
        x_scaler, y_scaler,
        feature_cols, target_cols, close_wide,
    )


def create_sequences(X_array: np.ndarray, y_array: np.ndarray, seq_len: int):
    X_out, y_out = [], []
    for i in range(len(X_array) - seq_len):
        X_out.append(X_array[i:i + seq_len])
        y_out.append(y_array[i + seq_len])
    return np.array(X_out), np.array(y_out)
