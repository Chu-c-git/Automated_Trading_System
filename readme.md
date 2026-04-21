# Deep Learning Project - Automated Trading System 
An production-ready project combining model training, monitoring and automated trading.

**Data Source |**

## Install
```
docker compose up -d
```
## Structure


<details>
<summary><b><font size="4">Prediction Module (DL)</font></b></summary>
Predict target: 

### Model
- LSTM
- Transformer based
- DRL 
- Times FM

### Model Performance

### MLFlow Monitoring

</details>

<details>
<summary><b><font size="4">Trading Module (DL)</font></b></summary>

- Buy rule:

- Sell rule:
    If the net profit 

- Schedule: 
Every day after the market close, start fetching latest data, train model and make prediction table of tomorrow. The prediction table contains the high and low. If the prediction shows the stock will rise 

</details>

### Reference