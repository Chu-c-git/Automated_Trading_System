"""
Multi-stock LSTM for next-day return forecasting.

Public API:
  - LSTMModel                       : nn.Module definition
  - train_lstm                      : train + MLflow logging
  - evaluate_metrics                : one-step RMSE / R² over a DataLoader
  - recursive_forecast              : autoregressive n-step price forecast
  - evaluate_autoregressive_metrics : rolling-window backtest
  - save_forecast_to_db             : run 1-day forecast and persist to PostgreSQL
"""

import copy
import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset


# ── Model ────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


# ── Training ─────────────────────────────────────────────────────────────────

def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_test:  np.ndarray, y_test:  np.ndarray,
               y_scaler, stock_codes,
               hidden_size=64, num_layers=3, dropout=0.3,
               lr=0.001, batch_size=32, num_epochs=100, eval_every=5):
    """
    Returns (best_model, mlflow_run, train_loader, test_loader, device).
    X_train / y_train are already-scaled numpy arrays of shape (samples, seq_len, features).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    params = dict(input_size=X_train.shape[2], hidden_size=hidden_size,
                  num_layers=num_layers, output_size=y_train.shape[1], dropout=dropout)
    model     = LSTMModel(**params).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_rmse, best_state = float('inf'), None

    with mlflow.start_run(nested=True) as run:
        mlflow.log_params(params)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                loss = loss_fn(model(X_b), y_b)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if (epoch + 1) % eval_every == 0:
                rmse, _ = evaluate_metrics(model, train_loader, y_scaler, device)
                avg_ret_rmse = float(rmse.mean())

                metrics = {'train_mse_scaled': avg_loss, 'train_rmse_ret_avg': avg_ret_rmse}
                for i, code in enumerate(stock_codes):
                    metrics[f'train_rmse_ret_{code}'] = float(rmse[i])
                mlflow.log_metrics(metrics, step=epoch)

                print(f'Epoch[{epoch+1}/{num_epochs}] MSE={avg_loss:.4f}  '
                      f'RMSE_ret_avg={avg_ret_rmse:.4f}  '
                      + '  '.join(f'{c}:{rmse[i]:.4f}' for i, c in enumerate(stock_codes)))

                if avg_ret_rmse < best_rmse:
                    best_rmse  = avg_ret_rmse
                    best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        mlflow.pytorch.log_model(model, name='best_LSTM')
        print(f'Best train RMSE (ret avg): {best_rmse:.6f}')

    return model, run, train_loader, test_loader, device


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_metrics(model, data_loader, y_scaler, device):
    """One-step-ahead RMSE and R² arrays (one value per stock)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in data_loader:
            preds.append(model(X_b.to(device)).cpu().numpy())
            trues.append(y_b.numpy())
    p = y_scaler.inverse_transform(np.concatenate(preds))
    t = y_scaler.inverse_transform(np.concatenate(trues))
    rmse = np.sqrt(np.mean((p - t) ** 2, axis=0))
    r2   = np.array([r2_score(t[:, i], p[:, i]) for i in range(t.shape[1])])
    return rmse, r2


def evaluate_autoregressive_metrics(model, X_test_np, y_test_np, x_scaler, y_scaler,
                                    n_steps, stock_codes, df_test_closes,
                                    feature_cols, device):
    """Rolling-window backtest over the test set."""
    n_stocks  = y_scaler.n_features_in_
    seq_len   = X_test_np.shape[1]
    n_windows = len(X_test_np) - n_steps
    if n_windows <= 0:
        raise ValueError(f'Test set too short for {n_steps}-step evaluation.')

    all_preds, all_trues = [], []
    for i in range(n_windows):
        last_close = df_test_closes.iloc[i + seq_len - 1].values
        rets, _ = recursive_forecast(
            model, X_test_np[i], n_steps,
            x_scaler, y_scaler, last_close, feature_cols, device,
        )
        all_preds.append(rets)
        all_trues.append(y_scaler.inverse_transform(y_test_np[i:i + n_steps]))

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    rmse_per_step = np.sqrt(np.mean((all_preds - all_trues) ** 2, axis=0))
    r2_per_step   = np.array([
        [r2_score(all_trues[:, s, k], all_preds[:, s, k]) for k in range(n_stocks)]
        for s in range(n_steps)
    ])

    for step in range(n_steps):
        row = '  '.join(f'{stock_codes[k]} RMSE={rmse_per_step[step,k]:.4f}'
                        f' R²={r2_per_step[step,k]:.3f}' for k in range(n_stocks))
        print(f'Day+{step+1:<2}  {row}')

    return rmse_per_step, r2_per_step


# ── Forecasting ──────────────────────────────────────────────────────────────

def recursive_forecast(model, seed_X_scaled, n_steps, x_scaler, y_scaler,
                       last_close, feature_cols, device):
    """
    Autoregressively predict n_steps days of returns, then convert to prices.

    Only close_* and ret_* columns are updated each step; all other columns
    (open, high, low, capacity, turnover, transaction_volume, change, market,
    rel) stay frozen at their last known value — they cannot be derived from
    the predicted return alone.
    """
    model.eval()
    window        = seed_X_scaled.copy()
    current_close = last_close.copy().astype(float)

    close_idx = [feature_cols.index(c) for c in feature_cols if c.startswith('close_')]
    ret_idx   = [feature_cols.index(c) for c in feature_cols if c.startswith('ret_')]

    x_mean_close  = x_scaler.mean_[close_idx]
    x_scale_close = x_scaler.scale_[close_idx]

    forecasts = []
    with torch.no_grad():
        for _ in range(n_steps):
            x           = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = model(x).cpu().numpy()[0]
            forecasts.append(pred_scaled)

            actual_ret    = y_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
            current_close = current_close * (1 + actual_ret)

            window = np.roll(window, -1, axis=0)
            window[-1, close_idx] = (current_close - x_mean_close) / x_scale_close
            window[-1, ret_idx]   = pred_scaled

    rets   = y_scaler.inverse_transform(np.array(forecasts))
    prices = np.cumprod(1 + rets, axis=0) * last_close
    return rets, prices


# ── DB persistence ───────────────────────────────────────────────────────────

def save_forecast_to_db(model, X_test_np, df_test, close_wide,
                        x_scaler, y_scaler, feature_cols, stock_codes,
                        postgres_uri: str, device,
                        table_name: str = 'lstm_prediction'):
    """
    Run a 1-day forecast from the last test window and write to PostgreSQL.

    Returns the forecast_df that was saved.
    """
    seed       = X_test_np[-1]
    last_close = df_test[close_wide.columns].iloc[-1].values

    forecast_rets, forecast_prices = recursive_forecast(
        model, seed, n_steps=1,
        x_scaler=x_scaler, y_scaler=y_scaler,
        last_close=last_close, feature_cols=feature_cols, device=device,
    )

    future_dates = pd.date_range(start=close_wide.index[-1], periods=2)[1:]
    forecast_df  = pd.DataFrame(
        forecast_prices,
        index   = future_dates,
        columns = [f'{code}_pred_price' for code in stock_codes],
    )
    forecast_df = forecast_df.round(1).astype(int)
    forecast_df = forecast_df.reset_index().rename(columns={'index': 'pred_date'})
    forecast_df['inference_date'] = time.strftime('%Y-%m-%d')

    engine = create_engine(postgres_uri)
    forecast_df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f'Saved 1-day forecast to PostgreSQL table: {table_name}')
    return forecast_df
