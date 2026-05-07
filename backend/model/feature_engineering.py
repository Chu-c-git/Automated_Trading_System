import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_features(df: pd.DataFrame, train_ratio: float = 0.9):
    """
    Input : raw df with columns [date, stock_code_id, close, ...]
    Output: df_X_train, df_X_test, df_y_train, df_y_test,
            x_scaler, y_scaler, feature_cols, target_cols, close_wide
    """
    selected_stocks = sorted(df['stock_code_id'].unique())[:4]

    df_sub = df[df['stock_code_id'].isin(selected_stocks)].copy()
    df_sub['date'] = pd.to_datetime(df_sub['date']).dt.normalize()

    close_daily = (
        df_sub.groupby(['date', 'stock_code_id'], as_index=False)['close']
        .mean()
        .sort_values(['date', 'stock_code_id'])
    )
    close_wide = (
        close_daily.pivot_table(index='date', columns='stock_code_id',
                                values='close', aggfunc='mean')
        .sort_index().ffill().bfill()
    )
    close_wide.columns = [f"close_{int(c)}" for c in close_wide.columns]

    ret_wide = close_wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_wide.columns = [f"ret_{c.split('_')[1]}" for c in close_wide.columns]

    rel_close = close_wide.sub(close_wide.mean(axis=1), axis=0)
    rel_close.columns = [f"rel_{c}" for c in close_wide.columns]

    feature_df = pd.concat([
        close_wide,
        ret_wide,
        close_wide.mean(axis=1).to_frame('market_mean_close'),
        ret_wide.mean(axis=1).to_frame('market_mean_ret'),
        rel_close,
    ], axis=1).dropna()

    target_cols  = ret_wide.columns.tolist()
    feature_cols = feature_df.columns.tolist()

    n_train  = int(len(feature_df) * train_ratio)
    df_train = feature_df.iloc[:n_train]
    df_test  = feature_df.iloc[n_train:]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

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
