"""
This file uses LSTM to predict the stock price.
Includes the following functions:
- `train_lstm_model`: trains the LSTM model on the given data.
- `predict_lstm_model`: uses the trained LSTM model to make predictions on new data.
- `evaluate_lstm_model`: evaluates the performance of the LSTM model using metrics such as RMSE and MAE.
- `performance_metrics`: calculates performance metrics for the LSTM model predictions.
"""

from dotenv import load_dotenv
from utils import init_mlflow
import os
import json
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import math
import time
import mlflow


START_DATE = "2020-01-01"
END_DATE = time.strftime("%Y-%m-%d")
STOCK_CODE = 2330

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(Colour_Palette))
tqdm.pandas()
mlflow.set_experiment("LSTM_Test")
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)
load_dotenv()
POSTGRES_URI = os.getenv('POSTGRES_URI')
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

map_dict_path = '/app/data/column_mapping.json'
init_mlflow()
with open(map_dict_path, 'r', encoding='utf-8') as f:
    map_dict = json.load(f)
# '6770', '2337', '2344', '2303', '2408', '2515', '5521', '2542', '1316', '2442', '2485', '6285', '2455', '2412', '4977', '2367', '2313', '3037', '4958', '2327'

if __name__ == "__main__":
    print("LSTM model for stock price prediction")