import socket
import os
import mlflow
import pandas as pd
import json
from dotenv import load_dotenv
load_dotenv()
POSTGRES_URI = os.getenv('POSTGRES_URI')

stock_dict_path = '/app/data/topk_stocks_by_industries.json'
map_dict_path = '/app/data/column_mapping.json'
with open(stock_dict_path, 'r') as f:
    stock_dict = json.load(f)
with open(map_dict_path, 'r') as f:
    map_dict = json.load(f)

def init_mlflow(host="mlflow", port=5000):
    try:
        ip = socket.gethostbyname(host)
        uri = f"http://{ip}:{port}"
        os.environ["MLFLOW_HTTP_REQUEST_ALLOW_HOSTS"] = "any"
        mlflow.set_tracking_uri(uri)
        return uri
    except socket.gaierror:
        # 如果在 container 外執行，退回 localhost
        mlflow.set_tracking_uri(f"http://localhost:{port}")
        return f"http://localhost:{port}"
    
def get_stock_data(stock_code, start_date: str, end_date: str, return_all=False):
    # Fetch data from PostgreSQL database
    table_name = f'daily_info_{stock_code}'
    query = f"""
    SELECT *
    FROM "{table_name}"
    WHERE stock_code_id = '{stock_code}'
      AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date ASC;
    """
    df = pd.read_sql(query, POSTGRES_URI)

    # Rename columns using the mapping dictionary
    df.rename(columns=map_dict, inplace=True)
    return df

def get_stock_data_by_category(category, start_date: str, end_date: str):
    """
    category: 'ETF', '建材營造', '電子零組件業', '半導體業', '通信網路業'
    """
    stock_codes = stock_dict.get(category, [])
    stock_codes = [str(code) for code in stock_codes]
    all_data = []
    for code in stock_codes:
        try:
            df = get_stock_data(code, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error fetching data for stock code {code}: {e}")
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)