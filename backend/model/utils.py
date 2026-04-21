import socket
import os
import mlflow

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