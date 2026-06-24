# Automated Trading System using LSTM
Production-ready project combining model training, monitoring, and automated trading on Taiwan stock market.

**Course |** Fundamentals of deep learning networks: Theory and industrial applications (NCKU IIMS)

**Data Source |**
- [**TWSE API**](https://openapi.twse.com.tw/) | Open API from the Taiwan Stock Exchange
- [**Finmind**](https://finmind.github.io/) | Open-source platform with 50+ Taiwan financial datasets

## Install
```bash
docker compose up -d
```

Services started:
| Service | Port | Description |
|---------|------|-------------|
| Backend (FastAPI) | 6001 | Main pipeline, REST API, scheduler |
| MLflow | 5000 | Experiment tracking UI |
| PostgreSQL | 5432 | Trade data + MLflow backend store |
| PGAdmin | 5050 | Database admin UI |

## Dev Environment
Attach your IDE to `ats-backend` container to run notebooks and experiments.

## Structure

![infra](src\infra.png)

```
.
├── backend/
│   ├── app/main.py              # FastAPI app + APScheduler
│   ├── model/                   # LSTM, baselines, feature engineering
│   ├── trade/                   # Buy/sell strategy + TWSE API wrappers
│   ├── data_preprocess/         # Stock data download pipeline
│   └── entrypoint.py            # Daily pipeline orchestrator
├── mlflow/                      # MLflow artifact storage
├── docker-compose.yml
└── .env.example
```



<details>
<summary><b><font size="4">Prediction Module (DL)</font></b></summary>

Predict target: next-day stock price (high/low/close)

### Model
| Model | Status |
|-------|--------|
| LSTM | ✅ Implemented |
| Baseline (Naive / MA / AR) | ✅ Implemented |
| Transformer | 🚧 Planned |
| DRL | 🚧 Planned |
| TimesFM | 🚧 Planned |

**LSTM architecture:** 1 layer · 16 hidden units · dropout 0.3 · seq_len 7 · 100 epochs · L1 loss · Adam optimizer

**Baselines:**
- **Naive** — random walk (last price = prediction)
- **MA** — 5-day moving average
- **AR(5)** — autoregression via statsmodels

### Model Performance

Evaluated on last 2 months as hold-out test set. Per-stock RMSE tracked in MLflow. Best model selected by average RMSE across all stocks.

### MLFlow Monitoring

Tracking URI: `http://localhost:5000`

Each daily run logs:
- Hyperparameters (hidden size, dropout, epochs, seq_len)
- Per-epoch train/val loss
- Per-stock test RMSE
- Model comparison table (LSTM vs baselines)

Run naming: `pipeline_YYYY-MM-DD` with nested runs per stock category.

</details>

<details>
<summary><b><font size="4">Trading Module</font></b></summary>

**Buy rule:**
1. Filter `lstm_prediction` table for today's inference date
2. Keep stocks with `pred_ret_pct > 1.0%`
3. Sort descending by predicted return, take top 25
4. Rank 1–5 → buy 2 shares; Rank 6–25 → buy 1 share
5. Skip stocks already held in portfolio
6. Order price = `pred_price`

**Sell rule:**
- **Take-profit:** current price ≥ cost + 5% → sell all shares
- **Stop-loss:** current price ≤ cost − 5% → sell all shares
- **Hold:** if P&L within ±5% → no action

**Schedule:**
Daily at **12:00 (Asia/Taipei)** via APScheduler + FastAPI:
1. Fetch latest TWSE stock data
2. Run baselines + LSTM training
3. Compare models, log to MLflow
4. Execute sell orders (check open positions)
5. Execute buy orders (from today's predictions)

Manual trigger: `POST /pipeline/run`

**Trading account:** NCKU simulated stock exchange

</details>

### Reference
